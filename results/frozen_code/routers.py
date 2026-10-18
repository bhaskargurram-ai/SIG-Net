"""Routers applied to saved cross-fitted expert matrices (results/experts/*.pkl).

global : simplex weights minimising OOF loss on the whole training set (== min w' Sigma_glob w for squared loss)
v3     : SIG-Net v3 (local softmax competence on ECFP neighbours + novelty shrinkage), exact replica
v4     : SIG-Net v4 = local error-covariance routing + novelty-damped local residual correction:
           K(x,j)      = a*s_ecfp + (1-a)*(s_maccs + s_phys)/2                     (composite kernel)
           Sigma_loc(x)= sum_j w_j r_j r_j'   over the k nearest training molecules  (OOF residuals r_j, weights w_j ~ K^2)
           Sigma_hat(x)= lam(x) Sigma_loc(x) + (1-lam(x)) Sigma_glob,  lam(x) = sbar(x)^gamma   (shrinkage to global)
           w(x)        = argmin_{w in simplex} w' (Sigma_hat + rho tr(Sigma_hat)/M I) w
           delta(x)    = nu * sbar(x)^gamma_d * sum_j w_j (y_j - o(j)' w(x))          (local residual correction)
           y_hat(x)    = p(x)' w(x) + delta(x)
         with hyperparameters selected on the validation fold. Nothing uses the test fold.
"""
import os, glob, pickle
import numpy as np, torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERTS_V3 = ["D-MPNN + RDKit-2D", "D-MPNN + Deep CORAL", "GINE", "AttentiveFP", "XGBoost (ECFP4 + RDKit-2D)", "RF (RDKit-2D)", "Anchor (Multi-View Stack)"]
EXPERTS_ALL = EXPERTS_V3 + ["ChemBERTa-2 + XGBoost", "XGBoost (ECFP4)"]
_B = None


def bundle():
    global _B
    if _B is None:
        _B = torch.load(os.environ.get("JCIM_BUNDLE", "C:/Users/Bhaskar/Downloads/phase1_outputs/phase1_bundle_v3.pt"), map_location="cpu", weights_only=False)
    return _B


def score(y, p, task):
    if task == "regression":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    return float(roc_auc_score(y.astype(int), np.clip(p, 1e-6, 1 - 1e-6))) if len(np.unique(y)) > 1 else 0.5


def better(a, b, task):
    return a < b if task == "regression" else a > b


def tanimoto(A, B):
    inter = A @ B.T; na = A.sum(1, keepdims=True); nb = B.sum(1, keepdims=True).T
    return inter / np.maximum(na + nb - inter, 1e-9)


def similarities(ds, proto, seed, idx):
    """ECFP, MACCS (Tanimoto) and physchem RBF similarities of val/test molecules to the training set."""
    data = bundle()["datasets"][ds]; sc = bundle()["fold_scalers"][ds][proto][seed]
    ecfp = np.vstack([d.ecfp4.numpy() for d in data]).astype(np.float32); maccs = np.vstack([d.maccs.numpy() for d in data]).astype(np.float32)
    raw = np.vstack([d.phys_raw.numpy() for d in data])
    phys = np.clip(np.nan_to_num(((np.clip(raw, sc["p01"], sc["p99"]) - sc["mean"]) / sc["std"])[:, sc["non_const_mask"]]), -10, 10).astype(np.float32)
    tr = np.array(idx["train"]); out = {}
    for part in ["val", "test"]:
        q = np.array(idx[part])
        d2 = (phys[q] ** 2).sum(1)[:, None] + (phys[tr] ** 2).sum(1)[None] - 2 * phys[q] @ phys[tr].T
        d = np.sqrt(np.maximum(d2, 0)); h = np.median(d) + 1e-9
        out[part] = {"ecfp": tanimoto(ecfp[q], ecfp[tr]), "maccs": tanimoto(maccs[q], maccs[tr]), "phys": np.exp(-0.5 * (d / h) ** 2)}
    return out


def simplex_fit(P, y, task):
    M = P.shape[1]
    loss = (lambda w: np.mean((y - P @ w) ** 2)) if task == "regression" else (lambda w: np.mean((y - np.clip(P @ w, 1e-4, 1 - 1e-4)) ** 2))
    r = minimize(lambda w: loss(w) + 1e-3 * np.sum((w - 1 / M) ** 2), np.ones(M) / M, bounds=[(0, 1)] * M,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
    return r.x if r.success else np.ones(M) / M


def proj_simplex(V):
    """Euclidean projection of each row of V onto the probability simplex."""
    n, M = V.shape; U = -np.sort(-V, axis=1); css = np.cumsum(U, axis=1) - 1
    ind = np.arange(1, M + 1); cond = U - css / ind > 0; rho = cond.cumsum(1).argmax(1)
    theta = css[np.arange(n), rho] / (rho + 1); return np.maximum(V - theta[:, None], 0)


def batched_simplex_qp(S, w0, iters=300):
    """min_w w'S w on the simplex for a batch of PSD matrices S [n,M,M] (projected gradient, step 1/L)."""
    L = np.linalg.eigvalsh(S)[:, -1] * 2 + 1e-12; w = w0.copy()
    for _ in range(iters):
        g = 2 * np.einsum("nij,nj->ni", S, w); w = proj_simplex(w - g / L[:, None])
    return w


class Fold:
    def __init__(self, path, experts):
        o = pickle.load(open(path, "rb")); self.o = o
        base = os.path.basename(path)[:-4]; self.ds, self.proto, s = base.split("__"); self.seed = int(s)
        self.task = o["meta"]["task"]; self.names = experts
        sd = o["meta"]["y_std"] if self.task == "regression" else 1.0; self.sd = sd
        g = lambda part: np.column_stack([o["experts"][n][part] for n in experts]).astype(np.float64)
        self.O, self.V, self.T = g("oof"), g("val"), g("test")
        self.y_tr, self.y_va, self.y_te = (o["y"][k].astype(np.float64) for k in ["train", "val", "test"])
        self.R = (self.y_tr[:, None] - self.O) / sd                  # standardised OOF residuals [N, M]
        self.S = similarities(self.ds, self.proto, self.seed, o["idx"])
        self.w_glob = simplex_fit(self.O, self.y_tr, self.task)
        self.Sig_glob = self.R.T @ self.R / len(self.R)


def neighbours(Sq, k):
    idx = np.argpartition(-Sq, k - 1, axis=1)[:, :k]; s = np.take_along_axis(Sq, idx, 1); return idx, s


def v4_predict(F, part, k, a, gamma, nu, gamma_d, rho=0.05, experts_mask=None):
    P = F.V if part == "val" else F.T; S = F.S[part]
    K = a * S["ecfp"] + (1 - a) * 0.5 * (S["maccs"] + S["phys"])
    idx, kv = neighbours(K, k); kv = np.maximum(kv, 1e-6); wn = kv ** 2 / (kv ** 2).sum(1, keepdims=True)
    sbar = np.take_along_axis(S["ecfp"], idx, 1).mean(1)                       # structural proximity (ECFP)
    M = F.O.shape[1]
    if gamma == "global":
        W = np.repeat(F.w_glob[None], len(P), 0)
    else:
        Rn = F.R[idx]                                                          # [n, k, M]
        Sloc = np.einsum("nk,nki,nkj->nij", wn, Rn, Rn)
        lam = (sbar ** gamma) if gamma > 0 else np.ones_like(sbar)
        Sh = lam[:, None, None] * Sloc + (1 - lam[:, None, None]) * F.Sig_glob[None]
        tr = np.trace(Sh, axis1=1, axis2=2) / M
        Sh = Sh + (rho * tr + 1e-8)[:, None, None] * np.eye(M)[None]
        W = batched_simplex_qp(Sh, np.repeat(F.w_glob[None], len(P), 0))
    pred = (P * W).sum(1)
    if nu > 0:
        loc_res = F.y_tr[idx] - np.einsum("nkm,nm->nk", F.O[idx], W)          # neighbours' OOF residual under w(x)
        pred = pred + nu * (sbar ** gamma_d) * (wn * loc_res).sum(1)
    if F.task != "regression":
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return pred, W


GRID = {"k": [10, 25, 50], "a": [0.5, 0.75, 1.0], "gamma": [0.0, 0.5, 1.0, 2.0, 4.0, "global"], "nu": [0.0, 0.25, 0.5, 0.75], "gamma_d": [1.0, 2.0]}


def v4_select(F, grid=GRID, fixed=None):
    """Validation-only selection over the grid (or evaluation of a fixed configuration)."""
    cands = []
    for k in grid["k"]:
        for a in grid["a"]:
            for g in grid["gamma"]:
                for nu in grid["nu"]:
                    for gd in (grid["gamma_d"] if nu > 0 else [1.0]):
                        if g == "global" and (k != grid["k"][0] or a != grid["a"][0]) and nu == 0:
                            continue
                        cands.append((k, a, g, nu, gd))
    if fixed is not None:
        cands = [fixed]
    best, best_cfg = None, None
    for c in cands:
        s = score(F.y_va, v4_predict(F, "val", *c)[0], F.task)
        if best is None or better(s, best, F.task): best, best_cfg = s, c
    p, W = v4_predict(F, "test", *best_cfg)
    return {"test_preds": p, "weights": W, "config": best_cfg, "val_score": best}


def v3_predict(F, part, k, tau, gamma):
    """Exact replica of the SIG-Net v3 combination (softmax over neighbour-weighted squared OOF errors)."""
    P = F.V if part == "val" else F.T; S = F.S[part]["ecfp"]
    if gamma == "global":
        return P @ F.w_glob
    if F.task == "regression":
        E = ((F.O - F.y_tr[:, None]) / F.sd) ** 2
    else:
        E = (np.clip(F.O, 1e-4, 1 - 1e-4) - F.y_tr[:, None]) ** 2
    idx = np.argsort(-S, axis=1)[:, :k]; s = np.clip(np.take_along_axis(S, idx, 1), 1e-3, None)
    e = (s[..., None] * E[idx]).sum(1) / s.sum(1, keepdims=True); scale = np.median(e, 1, keepdims=True) + 1e-9
    z = -e / (tau * scale); z -= z.max(1, keepdims=True); wl = np.exp(z); wl /= wl.sum(1, keepdims=True)
    lam = s.mean(1, keepdims=True) ** gamma if gamma > 0 else np.ones((len(P), 1))
    W = lam * wl + (1 - lam) * F.w_glob[None]
    return (P * W).sum(1)


def v3_select(F):
    best, cfg = None, None
    for g in [0.0, 1.0, 2.0, 4.0, "global"]:
        for k in ([5, 10, 20, 50] if g != "global" else [5]):
            for tau in ([0.25, 0.5, 1.0, 2.0] if g != "global" else [0.25]):
                s = score(F.y_va, v3_predict(F, "val", k, tau, g), F.task)
                if best is None or better(s, best, F.task): best, cfg = s, (k, tau, g)
    return {"test_preds": v3_predict(F, "test", *cfg), "config": cfg, "val_score": best}


def v4_components(F, part, k, a, gamma, rho=0.05):
    """Weights and the pieces needed for any (nu, gamma_d); the QP is solved once per (k, a, gamma)."""
    P = F.V if part == "val" else F.T; S = F.S[part]
    K = a * S["ecfp"] + (1 - a) * 0.5 * (S["maccs"] + S["phys"])
    idx, kv = neighbours(K, k); kv = np.maximum(kv, 1e-6); wn = kv ** 2 / (kv ** 2).sum(1, keepdims=True)
    sbar = np.take_along_axis(S["ecfp"], idx, 1).mean(1); M = F.O.shape[1]
    if gamma == "global":
        W = np.repeat(F.w_glob[None], len(P), 0)
    else:
        Rn = F.R[idx]; Sloc = np.einsum("nk,nki,nkj->nij", wn, Rn, Rn)
        lam = (sbar ** gamma) if gamma > 0 else np.ones_like(sbar)
        Sh = lam[:, None, None] * Sloc + (1 - lam[:, None, None]) * F.Sig_glob[None]
        Sh = Sh + (rho * np.trace(Sh, axis1=1, axis2=2) / M + 1e-8)[:, None, None] * np.eye(M)[None]
        W = batched_simplex_qp(Sh, np.repeat(F.w_glob[None], len(P), 0))
    base = (P * W).sum(1)
    corr = (wn * (F.y_tr[idx] - np.einsum("nkm,nm->nk", F.O[idx], W))).sum(1)
    return base, corr, sbar, W


def v4_from_components(F, comp, nu, gamma_d):
    base, corr, sbar, W = comp; p = base + (nu * (sbar ** gamma_d) * corr if nu > 0 else 0.0)
    return np.clip(p, 1e-6, 1 - 1e-6) if F.task != "regression" else p


def v4_fast_select(F, grid=GRID, restrict=None):
    """Validation-only grid search. restrict: dict of fixed values (e.g. {'nu': [0.0]}) for ablations."""
    g = dict(grid); g.update(restrict or {})
    best, cfg = None, None
    for k in g["k"]:
        for a in g["a"]:
            for gam in g["gamma"]:
                if gam == "global" and (k, a) != (g["k"][0], g["a"][0]):
                    continue
                comp = v4_components(F, "val", k, a, gam)
                for nu in g["nu"]:
                    for gd in (g["gamma_d"] if nu > 0 else [1.0]):
                        s = score(F.y_va, v4_from_components(F, comp, nu, gd), F.task)
                        if best is None or better(s, best, F.task): best, cfg = s, (k, a, gam, nu, gd)
    k, a, gam, nu, gd = cfg; comp = v4_components(F, "test", k, a, gam)
    return {"test_preds": v4_from_components(F, comp, nu, gd), "weights": comp[3], "config": cfg, "val_score": best}
