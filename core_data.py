"""Data layer: load Phase 1 bundle, apply Chemprop-grade featurization, build
zero-leakage per-fold feature matrices, OOF tabular anchor and novelty features.
"""
import os
from collections import defaultdict
from typing import Dict, Any
import numpy as np, torch
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import brier_score_loss
from xgboost import XGBRegressor, XGBClassifier
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from featurize import refeaturize_dataset

RDLogger.DisableLog("rdApp.*")
BUNDLE_PATH = os.environ.get("JCIM_BUNDLE", "data/phase1_bundle.pt")
N_JOBS = int(os.environ.get("JCIM_NJOBS", "-1"))
SEEDS = [42, 100, 2026, 7, 99]
PROTOCOLS = ["random", "bm_scaffold", "generic_scaffold", "butina_cluster"]
DATASET_CFG = {
    "FreeSolv": "regression", "ESOL": "regression", "Lipophilicity": "regression", "BACE": "classification", "BBBP": "classification",
}
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_BUNDLE, _FPS, _FOLD_CACHE = None, {}, {}


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        b = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        for name, dl in b["datasets"].items():
            refeaturize_dataset(dl)
        _BUNDLE = b
    return _BUNDLE


def dataset_fps(name):
    if name not in _FPS:
        _FPS[name] = [_MORGAN.GetFingerprint(Chem.MolFromSmiles(d.smiles)) for d in load_bundle()["datasets"][name]]
    return _FPS[name]


def _basis_models(task, seed):
    """Five diverse tabular base learners on different views (the 'multi-view anchor')."""
    if task == "regression":
        return [
            ("maccs", RandomForestRegressor(n_estimators=300, max_depth=16, random_state=seed, n_jobs=N_JOBS)),
            ("phys", RandomForestRegressor(n_estimators=300, max_depth=16, random_state=seed, n_jobs=N_JOBS)),
            ("full", Ridge(alpha=100.0)),
            ("phys", XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbosity=0)),
            ("hybrid", XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbosity=0)),
        ]
    return [
        ("ecfp", RandomForestClassifier(n_estimators=300, max_depth=16, class_weight="balanced", random_state=seed, n_jobs=N_JOBS)),
        ("maccs", RandomForestClassifier(n_estimators=300, max_depth=16, class_weight="balanced", random_state=seed, n_jobs=N_JOBS)),
        ("full", LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced")),
        ("ecfp", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbosity=0, eval_metric="logloss")),
        ("hybrid", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbosity=0, eval_metric="logloss")),
    ]


def _fit_basis(views_tr, y_tr, views_ev, task, seed):
    cols = []
    for view, m in _basis_models(task, seed):
        if task == "regression":
            m.fit(views_tr[view], y_tr); cols.append(m.predict(views_ev[view]))
        else:
            m.fit(views_tr[view], y_tr.astype(int)); cols.append(m.predict_proba(views_ev[view])[:, 1])
    return np.column_stack(cols).astype(np.float32)


def _simplex_weights(P_va, y_va, task):
    K = P_va.shape[1]
    if task == "regression":
        f = lambda w: np.mean((y_va - P_va @ w) ** 2) + 0.02 * np.sum((w - 1 / K) ** 2)
    else:
        f = lambda w: brier_score_loss(y_va.astype(int), np.clip(P_va @ w, 1e-5, 1 - 1e-5)) + 0.02 * np.sum((w - 1 / K) ** 2)
    r = minimize(f, np.ones(K) / K, bounds=[(0, 1)] * K, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return r.x if r.success else np.ones(K) / K


def _scaffold_kfold(scaffolds, k=3):
    buckets = defaultdict(list)
    for i, s in enumerate(scaffolds):
        buckets[s].append(i)
    folds = [[] for _ in range(k)]
    for j, grp in enumerate(sorted(buckets.values(), key=len, reverse=True)):
        folds[j % k].extend(grp)
    return folds


def prepare_fold(ds: str, proto: str, seed: int) -> Dict[str, Any]:
    """Zero-leakage fold: scalers, PCA, anchor, OOF anchor and novelty computed on train only."""
    key = (ds, proto, seed)
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    b = load_bundle(); task = DATASET_CFG[ds]
    data = b["datasets"][ds]; sp = b["splits"][ds][proto][seed]
    tr, va, te = list(sp["train"]), list(sp["val"]), list(sp["test"])
    n = len(data); min_fold = max(10, int(0.05 * n))
    if len(va) < min_fold:
        if len(te) >= 2 * min_fold:
            cut = len(te) // 2; va, te = va + te[:cut], te[cut:]
        else:
            need = min_fold - len(va); va, tr = tr[-need:] + va, tr[:-need]
    if len(te) < min_fold:
        need = min_fold - len(te); te, va = va[-need:] + te, va[:-need]

    sc = b["fold_scalers"][ds][proto][seed]
    raw = np.vstack([d.phys_raw.numpy() for d in data])
    phys = np.clip(np.nan_to_num(((np.clip(raw, sc["p01"], sc["p99"]) - sc["mean"]) / sc["std"])[:, sc["non_const_mask"]]), -10, 10).astype(np.float32)
    maccs = np.vstack([d.maccs.numpy() for d in data]).astype(np.float32)
    ecfp = np.vstack([d.ecfp4.numpy() for d in data]).astype(np.float32)
    y = np.array([d.y.item() for d in data], dtype=np.float32)
    pca = PCA(n_components=min(64, len(tr) - 2), random_state=seed).fit(ecfp[tr])
    ecfp_low = pca.transform(ecfp).astype(np.float32)
    views = {"phys": phys, "maccs": maccs, "ecfp": ecfp, "hybrid": np.hstack([ecfp, phys]), "full": np.hstack([phys, maccs, ecfp]),
             "prior": np.hstack([phys, maccs, ecfp_low])}
    emb_path = os.path.join(os.environ.get("JCIM_EMB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "embeddings")), f"{ds}_chemberta77m.npy")
    if os.path.exists(emb_path):  # frozen ChemBERTa-2 embeddings (bundle order), used by the extended expert pool
        views["chemberta"] = np.load(emb_path).astype(np.float32)
    V = lambda idx: {k: v[idx] for k, v in views.items() if k != "chemberta"}
    y_mean = float(y[tr].mean()) if task == "regression" else 0.0
    y_std = float(y[tr].std() + 1e-6) if task == "regression" else 1.0

    # Anchor: basis fitted on full train, simplex weights on val
    P_all = _fit_basis(V(tr), y[tr], views, task, seed)
    w = _simplex_weights(P_all[va], y[va], task)
    anchor = (P_all @ w).astype(np.float32)
    # OOF anchor on train (3-fold scaffold-grouped) so a residual learner never sees in-sample tree fits
    P_oof = np.zeros((len(tr), P_all.shape[1]), np.float32); tr_arr = np.array(tr)
    scaffs = [data[i].bm_scaffold for i in tr]
    oof_folds = _scaffold_kfold(scaffs)
    for f_val in oof_folds:
        f_trn = [i for i in range(len(tr)) if i not in set(f_val)]
        g_trn, g_val = tr_arr[f_trn], tr_arr[f_val]
        if len(f_val) == 0 or len(f_trn) < 10 or (task != "regression" and len(np.unique(y[g_trn])) < 2):
            P_oof[f_val] = P_all[g_val]; continue
        P_oof[f_val] = _fit_basis(V(g_trn), y[g_trn], V(g_val), task, seed)
    anchor_oof = anchor.copy(); anchor_oof[tr] = P_oof @ w
    basis_std = np.std(P_all, axis=1); basis_std[tr] = np.std(P_oof, axis=1)
    bs_norm = ((basis_std - basis_std[tr].mean()) / (basis_std[tr].std() + 1e-6)).astype(np.float32)

    # Novelty: NN Tanimoto to train (leave-own-scaffold-out for train molecules)
    fps = dataset_fps(ds); tr_fps = [fps[i] for i in tr]; tr_sc = np.array(scaffs)
    nn_sim = np.zeros(n, np.float32); top5 = np.zeros(n, np.float32); tr_set = set(tr)
    for i in range(n):
        s = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps), np.float32)
        if i in tr_set:
            m = tr_sc != data[i].bm_scaffold
            s = s[m] if m.any() else s[s < 0.9999]
        if len(s):
            nn_sim[i] = s.max(); top5[i] = np.sort(s)[-5:].mean()

    to_ms = (lambda a: (a - y_mean) / y_std) if task == "regression" else (lambda a: np.log(np.clip(a, 1e-5, 1 - 1e-5) / (1 - np.clip(a, 1e-5, 1 - 1e-5))))
    graphs = []
    for i, d0 in enumerate(data):
        d = d0.clone()
        for attr in ("ecfp4", "maccs", "phys_raw", "gen_scaffold_id", "inchikey", "gen_scaffold"):
            if attr in d: del d[attr]
        d.phys_scaled = torch.tensor(phys[i:i + 1]); d.prior_vec = torch.tensor(views["prior"][i:i + 1])
        d.y_raw = torch.tensor([[y[i]]]); d.y = torch.tensor([[(y[i] - y_mean) / y_std if task == "regression" else y[i]]], dtype=torch.float32)
        d.y_anchor_oof = torch.tensor([[to_ms(anchor_oof[i])]], dtype=torch.float32)
        d.y_anchor_ins = torch.tensor([[to_ms(anchor[i])]], dtype=torch.float32)
        d.novelty_vec = torch.tensor([[nn_sim[i], top5[i], float(np.clip(bs_norm[i], -5, 5))]], dtype=torch.float32)
        graphs.append(d)

    fold = {
        "task": task, "train_idx": tr, "val_idx": va, "test_idx": te,
        "in_channels": int(data[0].x.shape[1]), "edge_dim": int(data[0].edge_attr.shape[1]),
        "phys_dim": phys.shape[1], "prior_dim": views["prior"].shape[1], "y_mean": y_mean, "y_std": y_std,
        "y": (y[tr], y[va], y[te]),
        "X": {k: (v[tr], v[va], v[te]) for k, v in views.items()},
        "anchor": (anchor[tr], anchor[va], anchor[te]), "anchor_oof_train": anchor_oof[tr], "oof_folds": oof_folds,
        "basis": (P_oof, P_all[va], P_all[te]),
        "novelty": (np.column_stack([nn_sim, top5, bs_norm])[tr], np.column_stack([nn_sim, top5, bs_norm])[va], np.column_stack([nn_sim, top5, bs_norm])[te]),
        "graphs": ([graphs[i] for i in tr], [graphs[i] for i in va], [graphs[i] for i in te]),
        "test_nn_sim": nn_sim[te], "test_smiles": [data[i].smiles for i in te], "test_scaffolds": [data[i].bm_scaffold for i in te],
    }
    _FOLD_CACHE[key] = fold
    return fold


if __name__ == "__main__":
    import time
    t = time.time(); f = prepare_fold("FreeSolv", "bm_scaffold", 42)
    print("fold built in %.1fs" % (time.time() - t), {k: len(f[k]) for k in ["train_idx", "val_idx", "test_idx"]}, f["in_channels"], f["edge_dim"], f["prior_dim"])
    y_te = f["y"][2]; a_te = f["anchor"][2]
    print("anchor test RMSE", float(np.sqrt(np.mean((y_te - a_te) ** 2))))
