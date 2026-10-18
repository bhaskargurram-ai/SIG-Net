"""SIG-Net v3: local-competence mixture of experts with novelty shrinkage.

For every fold:
  1. Each expert m is cross-fitted on the training set (K scaffold-grouped folds). This yields out-of-fold (OOF)
     predictions o_m(j) for every training molecule j, and val/test predictions from the mean of the K models.
  2. Global weights w_glob (simplex) are fitted on the training OOF predictions.
  3. For a query x with training neighbours N_k(x) (ECFP4 Tanimoto), the local error of expert m is
        e_m(x) = sum_j s_j (y_j - o_m(j))^2 / sum_j s_j
     giving local weights w_loc(x) = softmax_m(-e_m(x) / (tau * median_m e_m)).
  4. Novelty shrinkage: w(x) = lam(x) w_loc(x) + (1 - lam(x)) w_glob,  lam(x) = (mean_j s_j)^gamma.
  5. (k, tau, gamma) are selected per fold on the validation set only.
The test fold is never used for fitting or selection.
"""
import os, time
import numpy as np, torch
from scipy.optimize import minimize
from rdkit import DataStructs
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from core_data import prepare_fold, dataset_fps, load_bundle, _scaffold_kfold, N_JOBS
from models import (DMPNNNet, GINENet, AttentiveFPNet, train_neural, _predict, metrics, set_seed, run_tabular, blend_on_val,
                    ensemble_graph_expert, sig_gate_on_val, DEVICE)

K_CROSSFIT = int(os.environ.get("JCIM_KFIT", 5))
K_GRID, TAU_GRID, GAMMA_GRID = (5, 10, 20, 50), (0.25, 0.5, 1.0, 2.0), (0.0, 1.0, 2.0, 4.0, "global")


# ----------------------------------------------------------------------------- cross-fitting
def crossfit_neural(fold, build_fn, seed, reg="none", lam=0.0, epochs=80):
    tr_g, va_g, te_g = fold["graphs"]; n = len(tr_g); task = fold["task"]
    folds = _scaffold_kfold([g.bm_scaffold for g in tr_g], K_CROSSFIT)
    oof = np.zeros(n, np.float32); va_l, te_l = [], []
    for k, hold in enumerate(folds):
        hs = set(hold); sub = [tr_g[i] for i in range(n) if i not in hs]
        sf = dict(fold); sf["graphs"] = (sub, va_g, [tr_g[i] for i in hold])
        r = train_neural(build_fn(), sf, seed + 101 * (k + 1), reg=reg, lam=lam, epochs=epochs); m = r.pop("_model")
        oof[hold] = r["test_preds"]
        va_l.append(_predict(m, DataLoader(va_g, 256), fold["y_mean"], fold["y_std"], task, False)[1])
        te_l.append(_predict(m, DataLoader(te_g, 256), fold["y_mean"], fold["y_std"], task, False)[1])
        del m; torch.cuda.empty_cache()
    return {"oof": oof, "val": np.mean(va_l, 0).astype(np.float32), "test": np.mean(te_l, 0).astype(np.float32)}


def _tab_model(kind, task, seed):
    if kind == "xgb":
        kw = dict(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbosity=0)
        return XGBRegressor(**kw) if task == "regression" else XGBClassifier(eval_metric="logloss", **kw)
    return (RandomForestRegressor(n_estimators=300, max_depth=16, random_state=seed, n_jobs=N_JOBS) if task == "regression"
            else RandomForestClassifier(n_estimators=300, max_depth=16, class_weight="balanced", random_state=seed, n_jobs=N_JOBS))


def crossfit_tabular(fold, kind, view, seed):
    task = fold["task"]; Xtr, Xva, Xte = fold["X"][view]; y = fold["y"][0]; n = len(y)
    folds = _scaffold_kfold([g.bm_scaffold for g in fold["graphs"][0]], K_CROSSFIT)
    pr = (lambda m, X: m.predict(X)) if task == "regression" else (lambda m, X: m.predict_proba(X)[:, 1])
    oof = np.zeros(n, np.float32); va_l, te_l = [], []
    for k, hold in enumerate(folds):
        trn = np.setdiff1d(np.arange(n), hold)
        if task != "regression" and len(np.unique(y[trn])) < 2:
            continue
        m = _tab_model(kind, task, seed + k); m.fit(Xtr[trn], y[trn] if task == "regression" else y[trn].astype(int))
        oof[hold] = pr(m, Xtr[hold]); va_l.append(pr(m, Xva)); te_l.append(pr(m, Xte))
    return {"oof": oof, "val": np.mean(va_l, 0).astype(np.float32), "test": np.mean(te_l, 0).astype(np.float32)}


# ----------------------------------------------------------------------------- local competence gate
def neighbours(ds, query_idx, train_idx, kmax=50):
    fps = dataset_fps(ds); trf = [fps[i] for i in train_idx]
    idx = np.zeros((len(query_idx), kmax), np.int32); sim = np.zeros((len(query_idx), kmax), np.float32)
    for r, q in enumerate(query_idx):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[q], trf), np.float32)
        top = np.argpartition(-s, kmax - 1)[:kmax]; top = top[np.argsort(-s[top])]
        idx[r], sim[r] = top, s[top]
    return idx, sim


def _simplex(P, y, loss):
    M = P.shape[1]
    r = minimize(lambda w: loss(y, P @ w) + 1e-3 * np.sum((w - 1 / M) ** 2), np.ones(M) / M, bounds=[(0, 1)] * M,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return r.x if r.success else np.ones(M) / M


def combine(P_q, nb_idx, nb_sim, E_tr, w_glob, k, tau, gamma):
    """P_q [n, M] expert predictions for queries; E_tr [n_train, M] squared OOF errors of experts on training set."""
    if gamma == "global":
        return P_q @ w_glob, np.repeat(w_glob[None], len(P_q), 0)
    s = np.clip(nb_sim[:, :k], 1e-3, None); ii = nb_idx[:, :k]
    e = (s[..., None] * E_tr[ii]).sum(1) / s.sum(1, keepdims=True)              # [n, M]
    scale = np.median(e, axis=1, keepdims=True) + 1e-9
    z = -e / (tau * scale); z -= z.max(1, keepdims=True); w_loc = np.exp(z); w_loc /= w_loc.sum(1, keepdims=True)
    lam = (s.mean(1, keepdims=True)) ** gamma if gamma > 0 else np.ones((len(P_q), 1))
    w = lam * w_loc + (1 - lam) * w_glob[None]
    return (P_q * w).sum(1), w


def sig_v3(fold, ds, experts, seed):
    """experts: dict name -> {oof, val, test}. Returns proposed + ablations; selection uses validation only."""
    task = fold["task"]; y_tr, y_va, y_te = fold["y"]; names = list(experts)
    O = np.column_stack([experts[n]["oof"] for n in names]); V = np.column_stack([experts[n]["val"] for n in names])
    T = np.column_stack([experts[n]["test"] for n in names])
    if task == "regression":
        sd = fold["y_std"]; E_tr = ((O - y_tr[:, None]) / sd) ** 2; loss = lambda y, p: np.mean((y - p) ** 2)
    else:
        Oc = np.clip(O, 1e-4, 1 - 1e-4); E_tr = (Oc - y_tr[:, None]) ** 2; loss = lambda y, p: np.mean((y - np.clip(p, 1e-4, 1 - 1e-4)) ** 2)
    w_glob = _simplex(O, y_tr, loss)
    va_i, va_s = neighbours(ds, fold["val_idx"], fold["train_idx"]); te_i, te_s = neighbours(ds, fold["test_idx"], fold["train_idx"])
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)
    grid = []
    for gamma in GAMMA_GRID:
        for k in (K_GRID if gamma != "global" else (K_GRID[0],)):
            for tau in (TAU_GRID if gamma != "global" else (TAU_GRID[0],)):
                p, _ = combine(V, va_i, va_s, E_tr, w_glob, k, tau, gamma)
                grid.append(((k, tau, gamma), metrics(y_va, p, task)["primary"]))
    pick = lambda cands: min(cands, key=lambda c: c[1]) if task == "regression" else max(cands, key=lambda c: c[1])
    best_full = pick(grid)[0]
    best_local = pick([g for g in grid if g[0][2] == 0.0])[0]           # ablation: no novelty shrinkage
    out = {}
    for label, cfg in [("SIG-Net v3 (Proposed)", best_full), ("SIG-Net v3 w/o novelty shrinkage", best_local),
                       ("Global stack (same cross-fitted experts)", (K_GRID[0], TAU_GRID[0], "global"))]:
        p, w = combine(T, te_i, te_s, E_tr, w_glob, *cfg)
        pv, _ = combine(V, va_i, va_s, E_tr, w_glob, *cfg)
        out[label] = {"metrics": metrics(y_te, p, task), "test_preds": p.astype(np.float32), "val_score": metrics(y_va, pv, task)["primary"],
                      "config": cfg, "weights": w.astype(np.float32), "expert_names": names}
    out["_w_glob"] = dict(zip(names, np.round(w_glob, 3)))
    return out


# ----------------------------------------------------------------------------- full fold
def run_fold_v3(ds, proto, seed, epochs=80):
    f = prepare_fold(ds, proto, seed); task = f["task"]; t0 = time.time()
    ic, ed, pdim = f["in_channels"], f["edge_dim"], f["phys_dim"]; y_tr, y_va, y_te = f["y"]
    ex = {
        "D-MPNN + RDKit-2D": crossfit_neural(f, lambda: DMPNNNet(ic, ed, pdim, use_rdkit=True), seed, epochs=epochs),
        "D-MPNN + Deep CORAL": crossfit_neural(f, lambda: DMPNNNet(ic, ed, pdim, use_rdkit=True), seed + 7, reg="coral", lam=0.10, epochs=epochs),
        "GINE": crossfit_neural(f, lambda: GINENet(ic, ed), seed + 13, epochs=epochs),
        "AttentiveFP": crossfit_neural(f, lambda: AttentiveFPNet(ic, ed), seed + 17, epochs=epochs),
        "XGBoost (ECFP4 + RDKit-2D)": crossfit_tabular(f, "xgb", "hybrid", seed),
        "RF (RDKit-2D)": crossfit_tabular(f, "rf", "phys", seed),
    }
    a_tr_oof, (a_tr, a_va, a_te) = f["anchor_oof_train"], f["anchor"]
    ex["Anchor (Multi-View Stack)"] = {"oof": a_tr_oof.astype(np.float32), "val": a_va, "test": a_te}
    R = sig_v3(f, ds, ex, seed); w_glob = R.pop("_w_glob")
    for n, e in ex.items():
        R[f"{n} [{K_CROSSFIT}-fold bagged]"] = {"metrics": metrics(y_te, e["test"], task), "test_preds": e["test"], "val_score": metrics(y_va, e["val"], task)["primary"]}
    meta = {"y_true": y_te, "nn_sim": f["novelty"][2][:, 0], "smiles": f["test_smiles"], "task": task, "w_glob": w_glob, "time": time.time() - t0}
    return {"results": R, "meta": meta}


# ----------------------------------------------------------------------------- v4 support: save all expert matrices
EXPERTS_V3 = ["D-MPNN + RDKit-2D", "D-MPNN + Deep CORAL", "GINE", "AttentiveFP", "XGBoost (ECFP4 + RDKit-2D)", "RF (RDKit-2D)", "Anchor (Multi-View Stack)"]
EXPERTS_EXTRA = ["ChemBERTa-2 + XGBoost", "XGBoost (ECFP4)"]


def run_fold_experts(ds, proto, seed, epochs=80):
    """Cross-fit all 9 experts and save their OOF / validation / test predictions. Routers are applied afterwards."""
    f = prepare_fold(ds, proto, seed); task = f["task"]; t0 = time.time()
    ic, ed, pdim = f["in_channels"], f["edge_dim"], f["phys_dim"]
    ex = {
        "D-MPNN + RDKit-2D": crossfit_neural(f, lambda: DMPNNNet(ic, ed, pdim, use_rdkit=True), seed, epochs=epochs),
        "D-MPNN + Deep CORAL": crossfit_neural(f, lambda: DMPNNNet(ic, ed, pdim, use_rdkit=True), seed + 7, reg="coral", lam=0.10, epochs=epochs),
        "GINE": crossfit_neural(f, lambda: GINENet(ic, ed), seed + 13, epochs=epochs),
        "AttentiveFP": crossfit_neural(f, lambda: AttentiveFPNet(ic, ed), seed + 17, epochs=epochs),
        "XGBoost (ECFP4 + RDKit-2D)": crossfit_tabular(f, "xgb", "hybrid", seed),
        "RF (RDKit-2D)": crossfit_tabular(f, "rf", "phys", seed),
        "ChemBERTa-2 + XGBoost": crossfit_tabular(f, "xgb", "chemberta", seed),
        "XGBoost (ECFP4)": crossfit_tabular(f, "xgb", "ecfp", seed),
    }
    a_tr_oof, (a_tr, a_va, a_te) = f["anchor_oof_train"], f["anchor"]
    ex["Anchor (Multi-View Stack)"] = {"oof": a_tr_oof.astype(np.float32), "val": a_va.astype(np.float32), "test": a_te.astype(np.float32)}
    y_tr, y_va, y_te = f["y"]
    return {"experts": ex, "y": {"train": y_tr, "val": y_va, "test": y_te},
            "idx": {"train": f["train_idx"], "val": f["val_idx"], "test": f["test_idx"]},
            "meta": {"task": task, "y_mean": f["y_mean"], "y_std": f["y_std"], "time": time.time() - t0,
                     "test_nn_sim": f["test_nn_sim"], "oof_folds": _scaffold_kfold([g.bm_scaffold for g in f["graphs"][0]], K_CROSSFIT)}}
