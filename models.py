"""Model zoo (graph baselines, tabular baselines, proposed SIG-Net v2) and training engine.

Every neural model: target z-score standardisation, AdamW + one-cycle LR, early stopping on val.
"""
import copy, random, time
from typing import Dict, Any, Optional
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.nn.models import AttentiveFP
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats
from xgboost import XGBRegressor, XGBClassifier

import os
N_JOBS = int(os.environ.get("JCIM_NJOBS", "-1"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def metrics(y, p, task) -> Dict[str, float]:
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        sp = stats.spearmanr(y, p)[0]
        return {"primary": rmse, "RMSE": rmse, "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p)), "Spearman": float(0 if np.isnan(sp) else sp)}
    yi = y.astype(int); pr = np.clip(p, 1e-6, 1 - 1e-6); ok = len(np.unique(yi)) > 1
    auc = float(roc_auc_score(yi, pr)) if ok else 0.5
    return {"primary": auc, "ROC_AUC": auc, "PR_AUC": float(average_precision_score(yi, pr)) if ok else 0.5, "Brier": float(brier_score_loss(yi, pr))}


# ----------------------------------------------------------------------------- graph baselines
def _head(i, h, drop):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))


class GCNNet(nn.Module):
    def __init__(self, in_c, ed=None, h=256, drop=0.1):
        super().__init__(); self.emb = nn.Linear(in_c, h)
        self.convs = nn.ModuleList([GCNConv(h, h) for _ in range(3)]); self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.drop, self.head = drop, _head(2 * h, h, drop)
    def forward(self, d):
        x = self.emb(d.x)
        for c, n in zip(self.convs, self.norms): x = x + F.dropout(F.relu(n(c(x, d.edge_index))), self.drop, self.training)
        g = torch.cat([global_mean_pool(x, d.batch), global_add_pool(x, d.batch)], -1); return self.head(g), g


class GINENet(nn.Module):
    def __init__(self, in_c, ed, h=256, drop=0.1):
        super().__init__(); self.emb = nn.Linear(in_c, h)
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=ed) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)]); self.drop, self.head = drop, _head(2 * h, h, drop)
    def forward(self, d):
        x = self.emb(d.x)
        for c, n in zip(self.convs, self.norms): x = x + F.dropout(F.relu(n(c(x, d.edge_index, d.edge_attr))), self.drop, self.training)
        g = torch.cat([global_mean_pool(x, d.batch), global_add_pool(x, d.batch)], -1); return self.head(g), g


class GATv2Net(nn.Module):
    def __init__(self, in_c, ed, h=256, heads=4, drop=0.1):
        super().__init__(); self.emb = nn.Linear(in_c, h)
        self.convs = nn.ModuleList([GATv2Conv(h, h // heads, heads=heads, edge_dim=ed, dropout=drop) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)]); self.drop, self.head = drop, _head(2 * h, h, drop)
    def forward(self, d):
        x = self.emb(d.x)
        for c, n in zip(self.convs, self.norms): x = x + F.dropout(F.relu(n(c(x, d.edge_index, d.edge_attr))), self.drop, self.training)
        g = torch.cat([global_mean_pool(x, d.batch), global_add_pool(x, d.batch)], -1); return self.head(g), g


class AttentiveFPNet(nn.Module):
    def __init__(self, in_c, ed, h=256, drop=0.1):
        super().__init__(); self.m = AttentiveFP(in_c, h, 1, edge_dim=ed, num_layers=3, num_timesteps=2, dropout=drop)
    def forward(self, d): return self.m(d.x, d.edge_index, d.edge_attr, d.batch), None


class DMPNNEncoder(nn.Module):
    """Chemprop-style directed-bond message passing; returns molecule embedding [B, h]."""
    def __init__(self, in_c, ed, h=300, depth=3, drop=0.1):
        super().__init__(); self.h, self.depth, self.drop = h, depth, drop
        self.W_i = nn.Linear(in_c + ed, h, bias=False); self.W_h = nn.Linear(h, h, bias=False); self.W_o = nn.Linear(in_c + h, h)
    @staticmethod
    def _rev(ei, n):
        src, dst = ei; fwd, rev = src * n + dst, dst * n + src; s = torch.argsort(fwd)
        return s[torch.clamp(torch.searchsorted(fwd[s], rev), 0, len(s) - 1)]
    def forward(self, d):
        x, ei, ea = d.x, d.edge_index, d.edge_attr; n = x.size(0); src, dst = ei; rev = self._rev(ei, n)
        h0 = F.relu(self.W_i(torch.cat([x[src], ea], -1))); h = h0
        for _ in range(self.depth - 1):
            inc = torch.zeros(n, self.h, device=h.device).index_add_(0, dst, h)
            h = F.dropout(F.relu(h0 + self.W_h(inc[src] - h[rev])), self.drop, self.training)
        inc = torch.zeros(n, self.h, device=h.device).index_add_(0, dst, h)
        return global_mean_pool(F.relu(self.W_o(torch.cat([x, inc], -1))), d.batch)


class DMPNNNet(nn.Module):
    def __init__(self, in_c, ed, phys_dim=0, h=300, drop=0.1, use_rdkit=False):
        super().__init__(); self.enc = DMPNNEncoder(in_c, ed, h, 3, drop); self.use_rdkit = use_rdkit
        self.head = _head(h + (phys_dim if use_rdkit else 0), h, drop)
    def forward(self, d):
        g = self.enc(d); z = torch.cat([g, d.phys_scaled], -1) if self.use_rdkit else g
        return self.head(z), g


# ----------------------------------------------------------------------------- proposed model
class SIGNetV2(nn.Module):
    """Dual-stream (D-MPNN + channel-attentive prior) with learned instance gate and OOF anchor residual.

    Ablation flags: use_topo / use_prior / use_gate / use_anchor / use_oof / gate_sees_novelty.
    Unlike v1 the similarity score is NOT hard-wired into the gate output; the gate may learn to use it.
    """
    def __init__(self, in_c, ed, prior_dim, h=300, drop=0.1, use_topo=True, use_prior=True, use_gate=True,
                 use_anchor=True, use_oof=True, gate_sees_novelty=True):
        super().__init__()
        self.use_topo, self.use_prior, self.use_gate, self.use_anchor, self.use_oof, self.gsn = use_topo, use_prior, use_gate, use_anchor, use_oof, gate_sees_novelty
        if use_topo:
            self.enc = DMPNNEncoder(in_c, ed, h, 3, drop); self.topo_proj = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(drop))
        if use_prior:
            r = max(16, prior_dim // 4)
            self.se = nn.Sequential(nn.Linear(prior_dim, r), nn.SiLU(), nn.Linear(r, prior_dim), nn.Sigmoid())
            self.prior_proj = nn.Sequential(nn.Linear(prior_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(drop))
        if use_topo and use_prior and use_gate:
            self.gate = nn.Sequential(nn.Linear(3 * h + (3 if gate_sees_novelty else 0), h // 2), nn.LayerNorm(h // 2), nn.SiLU(), nn.Linear(h // 2, h), nn.Sigmoid())
        zin = 2 * h if (use_topo and use_prior and not use_gate) else h
        self.head = nn.Sequential(nn.Linear(zin, h // 2), nn.LayerNorm(h // 2), nn.SiLU(), nn.Dropout(drop), nn.Linear(h // 2, 1))
        self.res_scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, d):
        ht = hp = g = None
        if self.use_topo: ht = self.topo_proj(self.enc(d))
        if self.use_prior: p = d.prior_vec; hp = self.prior_proj(p * (1 + self.se(p)))
        if ht is not None and hp is not None:
            if self.use_gate:
                inp = [ht, hp, (ht - hp).abs()] + ([d.novelty_vec] if self.gsn else [])
                g = self.gate(torch.cat(inp, -1)); z = g * ht + (1 - g) * hp
            else: z = torch.cat([ht, hp], -1)
        else: z = ht if ht is not None else hp
        delta = self.head(z)
        if self.use_anchor:
            anc = d.y_anchor_oof if (self.training and self.use_oof) else d.y_anchor_ins
            return anc + torch.tanh(self.res_scale) * delta, ht, hp, g
        return delta, ht, hp, g


def scaffold_decorrelation_loss(ht, scaffold_ids, y):
    """Same-scaffold activity-cliff repulsion + cross-scaffold target alignment on normalised h_topo."""
    if ht is None or ht.size(0) < 6: return torch.tensor(0.0, device=y.device)
    hn = F.normalize(ht, dim=-1); sim = hn @ hn.T; yv = y.view(-1, 1); ys = yv.std() + 1e-6
    tsim = torch.exp(-0.5 * torch.cdist(yv / ys, yv / ys) ** 2); same = (scaffold_ids.view(-1, 1) == scaffold_ids.view(1, -1)).float()
    w = (1 - same) * 1.2 + same * (1 - tsim) * 2.0; eye = 1 - torch.eye(ht.size(0), device=y.device)
    return (w * eye * (sim - (2 * tsim - 1)) ** 2).sum() / (eye.sum() + 1e-6)


def coral_loss(h):
    if h is None or h.size(0) < 6: return torch.tensor(0.0, device=DEVICE)
    a, b = h[: h.size(0) // 2], h[h.size(0) // 2:]
    ca, cb = a - a.mean(0, keepdim=True), b - b.mean(0, keepdim=True)
    return ((a.mean(0) - b.mean(0)) ** 2).mean() + ((ca.T @ ca / max(len(a) - 1, 1) - cb.T @ cb / max(len(b) - 1, 1)) ** 2).mean()


# ----------------------------------------------------------------------------- training engine
@torch.no_grad()
def _predict(model, loader, y_mean, y_std, task, signet):
    model.eval(); ys, ps, gs = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        if signet:
            out, _, _, g = model(b); gs.append(g.mean(-1).cpu().numpy() if g is not None else np.full(b.y.size(0), np.nan, np.float32))
        else:
            out, _ = model(b); gs.append(np.full(b.y.size(0), np.nan, np.float32))
        o = out.view(-1).cpu().numpy()
        ps.append(o * y_std + y_mean if task == "regression" else 1 / (1 + np.exp(-np.clip(o, -20, 20)))); ys.append(b.y_raw.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps).astype(np.float32), np.concatenate(gs).astype(np.float32)


def train_neural(model, fold, seed, signet=False, reg="none", lam=0.0, epochs=80, lr=1e-3, patience=None, batch=50) -> Dict[str, Any]:
    patience = patience or epochs  # default: full schedule, keep best-val checkpoint (Chemprop practice)
    set_seed(seed); t0 = time.time(); task = fold["task"]
    tr, va, te = fold["graphs"]; ym, ys = fold["y_mean"], fold["y_std"]
    Ltr = DataLoader(tr, batch, shuffle=True, drop_last=(len(tr) % batch == 1)); Lva = DataLoader(va, 128); Lte = DataLoader(te, 128)
    model = model.to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(Ltr), pct_start=0.1, final_div_factor=100)
    crit = nn.HuberLoss(delta=1.0) if task == "regression" else nn.BCEWithLogitsLoss()
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)
    best, best_state, wait = (np.inf if task == "regression" else -np.inf), copy.deepcopy(model.state_dict()), 0
    ep_run = 0
    for ep_run in range(1, epochs + 1):
        model.train()
        for b in Ltr:
            b = b.to(DEVICE); yt = b.y.view(-1, 1); opt.zero_grad()
            if signet:
                out, ht, _, g = model(b); loss = crit(out, yt)
                if lam > 0 and reg == "scaff": loss = loss + lam * scaffold_decorrelation_loss(ht, b.scaffold_id, yt)
            else:
                out, hg = model(b); loss = crit(out, yt)
                if lam > 0 and reg == "coral": loss = loss + lam * coral_loss(hg)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        v = metrics(*_predict(model, Lva, ym, ys, task, signet)[:2], task)["primary"]
        if better(v, best): best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    _, p_va, _ = _predict(model, Lva, ym, ys, task, signet); y_te, p_te, g_te = _predict(model, Lte, ym, ys, task, signet)
    beta = 1.0
    if signet and getattr(model, "use_anchor", False):  # final val-selected blend with anchor
        _, a_va, a_te = fold["anchor"]; y_va = fold["y"][1]; bs = np.inf if task == "regression" else -np.inf
        for bb in np.linspace(0, 1, 21):
            s = metrics(y_va, (1 - bb) * a_va + bb * p_va, task)["primary"]
            if better(s, bs): bs, beta = s, float(bb)
        p_va = (1 - beta) * a_va + beta * p_va; p_te = ((1 - beta) * a_te + beta * p_te).astype(np.float32)
    return {"metrics": metrics(y_te, p_te, task), "val_preds": p_va, "test_preds": p_te, "gates": g_te, "beta": beta, "time": time.time() - t0, "epochs_run": ep_run, "_model": model}


# ----------------------------------------------------------------------------- tabular baselines
class HighwayMLP(nn.Module):
    def __init__(self, i, h=256, drop=0.25):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(drop))
        self.blk = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(drop), nn.Linear(h, h), nn.LayerNorm(h))
        self.gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid()); self.out = nn.Sequential(nn.SiLU(), nn.Linear(h, h // 2), nn.SiLU(), nn.Dropout(drop), nn.Linear(h // 2, 1))
    def forward(self, x):
        h = self.inp(x); g = self.gate(h); return self.out(g * self.blk(h) + (1 - g) * h)


def run_tabular(fold, seed, epochs=60) -> Dict[str, Dict[str, Any]]:
    task = fold["task"]; y_tr, y_va, y_te = fold["y"]; ym, ys = fold["y_mean"], fold["y_std"]; res = {}
    specs = [("RF (MACCS-167)", "rf", "maccs"), ("RF (ECFP4-2048)", "rf", "ecfp"), ("RF (RDKit-2D)", "rf", "phys"),
             ("XGBoost (ECFP4-2048)", "xgb", "ecfp"), ("XGBoost (RDKit-2D)", "xgb", "phys"), ("XGBoost (ECFP4 + RDKit-2D)", "xgb", "hybrid")]
    for name, fam, view in specs:
        Xtr, Xva, Xte = fold["X"][view]
        if fam == "rf":
            m = (RandomForestRegressor(n_estimators=300, max_depth=16, random_state=seed, n_jobs=N_JOBS) if task == "regression"
                 else RandomForestClassifier(n_estimators=300, max_depth=16, class_weight="balanced", random_state=seed, n_jobs=N_JOBS))
        else:
            kw = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbosity=0, early_stopping_rounds=40)
            m = XGBRegressor(**kw) if task == "regression" else XGBClassifier(eval_metric="logloss", **kw)
        if task == "regression":
            m.fit(Xtr, y_tr, **({"eval_set": [(Xva, y_va)], "verbose": False} if fam == "xgb" else {})); pv, pt = m.predict(Xva), m.predict(Xte)
        else:
            m.fit(Xtr, y_tr.astype(int), **({"eval_set": [(Xva, y_va.astype(int))], "verbose": False} if fam == "xgb" else {})); pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        res[name] = {"metrics": metrics(y_te, pt, task), "val_preds": pv.astype(np.float32), "test_preds": pt.astype(np.float32)}
    # HighwayMLP on hybrid view
    set_seed(seed); Xtr, Xva, Xte = fold["X"]["hybrid"]
    mlp = HighwayMLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.AdamW(mlp.parameters(), 1e-3, weight_decay=1e-4)
    crit = nn.HuberLoss() if task == "regression" else nn.BCEWithLogitsLoss()
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Xt, Xv, Xs = T(Xtr), T(Xva), T(Xte); yt = T((y_tr - ym) / ys if task == "regression" else y_tr).view(-1, 1)
    inv = (lambda o: o * ys + ym) if task == "regression" else (lambda o: 1 / (1 + np.exp(-np.clip(o, -20, 20))))
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b); best, bw = (np.inf if task == "regression" else -np.inf), None
    for _ in range(epochs):
        mlp.train(); perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); crit(mlp(Xt[idx]), yt[idx]).backward(); opt.step()
        mlp.eval()
        with torch.no_grad(): s = metrics(y_va, inv(mlp(Xv).view(-1).cpu().numpy()), task)["primary"]
        if better(s, best): best, bw = s, copy.deepcopy(mlp.state_dict())
    mlp.load_state_dict(bw); mlp.eval()
    with torch.no_grad(): pv, pt = inv(mlp(Xv).view(-1).cpu().numpy()), inv(mlp(Xs).view(-1).cpu().numpy())
    res["HighwayMLP (ECFP4 + RDKit-2D)"] = {"metrics": metrics(y_te, pt, task), "val_preds": pv.astype(np.float32), "test_preds": pt.astype(np.float32)}
    return res


def blend_on_val(fold, pa_va, pa_te, pb_va, pb_te):
    """Single global weight selected on validation (the 'Stacked' baseline)."""
    task = fold["task"]; y_va = fold["y"][1]; better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)
    best, bw = (np.inf if task == "regression" else -np.inf), 0.5
    for w in np.linspace(0, 1, 21):
        s = metrics(y_va, w * pa_va + (1 - w) * pb_va, task)["primary"]
        if better(s, best): best, bw = s, float(w)
    return (bw * pa_te + (1 - bw) * pb_te).astype(np.float32), bw


# ----------------------------------------------------------------------------- proposed: SIG-Stack
def oof_graph_expert(fold, seed, build_fn, epochs=80) -> Dict[str, Any]:
    """Train the graph expert on each of the 3 scaffold-grouped train sub-folds (OOF) plus the full train set.
    Returns OOF train predictions, val/test predictions of the full model, and the 4-model ensemble predictions."""
    task = fold["task"]; tr_g, va_g, te_g = fold["graphs"]; n_tr = len(tr_g)
    oof = np.zeros(n_tr, np.float32); va_list, te_list = [], []
    for k, f_val in enumerate(fold["oof_folds"]):
        hold = set(f_val); sub_tr = [tr_g[i] for i in range(n_tr) if i not in hold]; sub_val = [tr_g[i] for i in f_val]
        if len(f_val) == 0 or len(sub_tr) < 20: continue
        sub_fold = dict(fold); sub_fold["graphs"] = (sub_tr, va_g, sub_val)  # test slot = held-out sub-fold
        r = train_neural(build_fn(), sub_fold, seed + 1000 * (k + 1), epochs=epochs)
        oof[f_val] = r["test_preds"]
        m = r.pop("_model")  # evaluate this sub-model on the real val/test sets for ensembling
        va_list.append(_predict(m, DataLoader(va_g, 128), fold["y_mean"], fold["y_std"], task, False)[1])
        te_list.append(_predict(m, DataLoader(te_g, 128), fold["y_mean"], fold["y_std"], task, False)[1])
    full = train_neural(build_fn(), fold, seed, epochs=epochs); full.pop("_model")
    va_list.append(full["val_preds"]); te_list.append(full["test_preds"])
    return {"oof_train": oof, "val": full["val_preds"], "test": full["test_preds"],
            "ens_val": np.mean(va_list, 0).astype(np.float32), "ens_test": np.mean(te_list, 0).astype(np.float32), "n_ens": len(te_list)}


class _GateMLP(nn.Module):
    def __init__(self, i, h=32):
        super().__init__(); self.net = nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
    def forward(self, x): return torch.sigmoid(self.net(x))


def _gate_features(p_g, p_a, nov):
    """Per-molecule gate inputs: expert predictions, disagreement, novelty (nn_sim, top5_sim, basis_std)."""
    return np.column_stack([p_g, p_a, np.abs(p_g - p_a), nov]).astype(np.float32)


def sig_stack(fold, seed, p_g_oof, p_a_oof, p_g_va, p_a_va, p_g_te, p_a_te, nov_tr, nov_va, nov_te,
              epochs=300, wd=1e-3) -> Dict[str, Any]:
    """Novelty-conditioned instance-wise stacking of graph expert (g) and tabular anchor (a).

    The gate is fitted on OUT-OF-FOLD train predictions only. Model selection between the learned gate and a
    single global weight is done on the validation fold, so the method degrades gracefully to plain stacking.
    """
    task = fold["task"]; y_tr, y_va, y_te = fold["y"]; set_seed(seed)
    ms = (lambda a: (a - fold["y_mean"]) / fold["y_std"]) if task == "regression" else (lambda a: np.log(np.clip(a, 1e-5, 1 - 1e-5) / (1 - np.clip(a, 1e-5, 1 - 1e-5))))
    inv = (lambda a: a * fold["y_std"] + fold["y_mean"]) if task == "regression" else (lambda a: 1 / (1 + np.exp(-a)))
    # standardise gate inputs on train OOF statistics
    F_tr = _gate_features(ms(p_g_oof), ms(p_a_oof), nov_tr); mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    Z = lambda F_: torch.tensor((F_ - mu) / sd, dtype=torch.float32, device=DEVICE)
    Ztr, Zva, Zte = Z(F_tr), Z(_gate_features(ms(p_g_va), ms(p_a_va), nov_va)), Z(_gate_features(ms(p_g_te), ms(p_a_te), nov_te))
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    g_tr, a_tr, yt = T(ms(p_g_oof)), T(ms(p_a_oof)), T(ms(y_tr) if task == "regression" else y_tr)
    gate = _GateMLP(Ztr.shape[1]).to(DEVICE); opt = torch.optim.AdamW(gate.parameters(), 3e-3, weight_decay=wd)
    crit = nn.HuberLoss() if task == "regression" else nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        gate.train(); opt.zero_grad(); w = gate(Ztr).view(-1)
        crit(w * g_tr + (1 - w) * a_tr, yt).backward(); opt.step()
    gate.eval()
    with torch.no_grad(): w_va, w_te = gate(Zva).view(-1).cpu().numpy(), gate(Zte).view(-1).cpu().numpy()
    gated_va = inv(w_va * ms(p_g_va) + (1 - w_va) * ms(p_a_va)); gated_te = inv(w_te * ms(p_g_te) + (1 - w_te) * ms(p_a_te))
    # global-weight alternative (plain stacking) fitted on val
    const_te, w_const = blend_on_val(fold, p_g_va, p_g_te, p_a_va, p_a_te)
    const_va = w_const * p_g_va + (1 - w_const) * p_a_va
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)
    s_gate, s_const = metrics(y_va, gated_va, task)["primary"], metrics(y_va, const_va, task)["primary"]
    use_gate = better(s_gate, s_const)
    p_te = gated_te.astype(np.float32) if use_gate else const_te
    return {"metrics": metrics(y_te, p_te, task), "test_preds": p_te, "gates": w_te.astype(np.float32), "used_gate": bool(use_gate),
            "w_const": w_const, "val_gate": s_gate, "val_const": s_const}


# ----------------------------------------------------------------------------- proposed (final): SIG-Net v2
def ensemble_graph_expert(fold, build_fn, seed, n_members=3, epochs=80):
    """Equal-budget multi-seed ensemble of the graph expert on the full train set."""
    va_l, te_l = [], []
    for k in range(n_members):
        r = train_neural(build_fn(), fold, seed + 7919 * k, epochs=epochs); r.pop("_model", None)
        va_l.append(r["val_preds"]); te_l.append(r["test_preds"])
    return {"val": np.mean(va_l, 0).astype(np.float32), "test": np.mean(te_l, 0).astype(np.float32),
            "members_val": va_l, "members_test": te_l}


def _fit_lowcap_gate(F_, g, a, y, task, l2=1e-2, iters=400):
    """3-parameter gate w = sigmoid(theta . [1, sim, |g-a|]); returns theta (numpy)."""
    th = torch.zeros(F_.shape[1], requires_grad=True); opt = torch.optim.Adam([th], lr=0.05)
    F_t, g_t, a_t, y_t = map(lambda z: torch.tensor(z, dtype=torch.float32), (F_, g, a, y))
    crit = nn.HuberLoss() if task == "regression" else nn.BCEWithLogitsLoss()
    for _ in range(iters):
        opt.zero_grad(); w = torch.sigmoid(F_t @ th); loss = crit(w * g_t + (1 - w) * a_t, y_t) + l2 * (th[1:] ** 2).sum(); loss.backward(); opt.step()
    return th.detach().numpy()


def sig_gate_on_val(fold, seed, p_g_va, p_g_te, p_a_va, p_a_te, nov_va, nov_te, k_cv=5) -> Dict[str, Any]:
    """Novelty-conditioned blend of graph expert g and anchor a.
    Gate (3 params) is fitted on the validation fold; the choice gate-vs-constant is made by k-fold CV *within*
    validation, so the test fold is never touched. Degrades to plain stacking when the gate does not help."""
    task = fold["task"]; y_va, y_te = fold["y"][1], fold["y"][2]; set_seed(seed)
    ms = (lambda z: (z - fold["y_mean"]) / fold["y_std"]) if task == "regression" else (lambda z: np.log(np.clip(z, 1e-5, 1 - 1e-5) / (1 - np.clip(z, 1e-5, 1 - 1e-5))))
    inv = (lambda z: z * fold["y_std"] + fold["y_mean"]) if task == "regression" else (lambda z: 1 / (1 + np.exp(-z)))
    g_va, a_va, g_te, a_te = ms(p_g_va), ms(p_a_va), ms(p_g_te), ms(p_a_te); yv = ms(y_va) if task == "regression" else y_va
    feats = lambda g, a, nov: np.column_stack([np.ones(len(g)), nov[:, 0], np.abs(g - a)]).astype(np.float32)
    Fva, Fte = feats(g_va, a_va, nov_va), feats(g_te, a_te, nov_te)
    mu, sd = Fva[:, 1:].mean(0), Fva[:, 1:].std(0) + 1e-6; Fva[:, 1:] = (Fva[:, 1:] - mu) / sd; Fte[:, 1:] = (Fte[:, 1:] - mu) / sd
    # CV within validation: gate vs constant
    n = len(y_va); perm = np.random.RandomState(seed).permutation(n); folds = np.array_split(perm, min(k_cv, max(2, n // 10)))
    pg, pc = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for hold in folds:
        trn = np.setdiff1d(perm, hold)
        th = _fit_lowcap_gate(Fva[trn], g_va[trn], a_va[trn], yv[trn], task); w = 1 / (1 + np.exp(-(Fva[hold] @ th)))
        pg[hold] = w * g_va[hold] + (1 - w) * a_va[hold]
        th0 = _fit_lowcap_gate(Fva[trn][:, :1], g_va[trn], a_va[trn], yv[trn], task); wc = 1 / (1 + np.exp(-th0[0]))
        pc[hold] = wc * g_va[hold] + (1 - wc) * a_va[hold]
    s_gate, s_const = metrics(y_va, inv(pg), task)["primary"], metrics(y_va, inv(pc), task)["primary"]
    better = (lambda x, y: x < y) if task == "regression" else (lambda x, y: x > y)
    use_gate = better(s_gate, s_const)
    th = _fit_lowcap_gate(Fva if use_gate else Fva[:, :1], g_va, a_va, yv, task)
    w_te = 1 / (1 + np.exp(-((Fte if use_gate else Fte[:, :1]) @ th)))
    p_te = inv(w_te * g_te + (1 - w_te) * a_te).astype(np.float32)
    return {"metrics": metrics(y_te, p_te, task), "test_preds": p_te, "gates": w_te.astype(np.float32), "used_gate": bool(use_gate),
            "theta": th, "cv_gate": s_gate, "cv_const": s_const}
