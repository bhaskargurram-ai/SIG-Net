"""Pretrained foundation-model baselines on frozen ChemBERTa-2 (77M-MTR) embeddings, run on exactly the same folds.
  - ChemBERTa-2 + ridge/logistic (regularisation chosen on the validation fold)
  - ChemBERTa-2 + XGBoost (early stopping on the validation fold)
Writes results/folds_cb/<ds>__<proto>__<seed>.pkl."""
import os, sys, pickle
import numpy as np, torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_data import SEEDS, PROTOCOLS, DATASET_CFG, BUNDLE_PATH
from models import metrics

ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "results", "folds_cb"); os.makedirs(OUT, exist_ok=True)
B = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)


def fold_indices(ds, proto, seed):
    """Identical to the index logic in core_data.prepare_fold (min-fold-size rebalancing), without any model fitting."""
    data = B["datasets"][ds]; sp = B["splits"][ds][proto][seed]
    tr, va, te = list(sp["train"]), list(sp["val"]), list(sp["test"]); n = len(data); mf = max(10, int(0.05 * n))
    if len(va) < mf:
        if len(te) >= 2 * mf:
            cut = len(te) // 2; va, te = va + te[:cut], te[cut:]
        else:
            need = mf - len(va); va, tr = tr[-need:] + va, tr[:-need]
    if len(te) < mf:
        need = mf - len(te); te, va = va[-need:] + te, va[:-need]
    return tr, va, te


def run(ds, proto, seed):
    task = DATASET_CFG[ds]; E = np.load(os.path.join(ROOT, "results", "embeddings", f"{ds}_chemberta77m.npy"))
    y = np.array([d.y.item() for d in B["datasets"][ds]], np.float32); tr, va, te = fold_indices(ds, proto, seed)
    sc = StandardScaler().fit(E[tr]); Xtr, Xva, Xte = sc.transform(E[tr]), sc.transform(E[va]), sc.transform(E[te])
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b); R = {}
    best = None
    for a in [0.1, 1, 10, 100, 1000, 10000]:
        m = Ridge(alpha=a).fit(Xtr, y[tr]) if task == "regression" else LogisticRegression(C=1 / a, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr].astype(int))
        pv = m.predict(Xva) if task == "regression" else m.predict_proba(Xva)[:, 1]
        s = metrics(y[va], pv, task)["primary"]
        if best is None or better(s, best[0]): best = (s, m)
    m = best[1]; pt = m.predict(Xte) if task == "regression" else m.predict_proba(Xte)[:, 1]
    R["ChemBERTa-2 (77M-MTR) + linear"] = {"metrics": metrics(y[te], pt, task), "test_preds": pt.astype(np.float32)}
    kw = dict(n_estimators=1000, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=8, verbosity=0, early_stopping_rounds=50)
    m = XGBRegressor(**kw) if task == "regression" else XGBClassifier(eval_metric="logloss", **kw)
    m.fit(Xtr, y[tr] if task == "regression" else y[tr].astype(int), eval_set=[(Xva, y[va] if task == "regression" else y[va].astype(int))], verbose=False)
    pt = m.predict(Xte) if task == "regression" else m.predict_proba(Xte)[:, 1]
    R["ChemBERTa-2 (77M-MTR) + XGBoost"] = {"metrics": metrics(y[te], pt, task), "test_preds": pt.astype(np.float32)}
    return {"results": R, "meta": {"task": task, "y_true_cb": y[te]}}


if __name__ == "__main__":
    # sanity: fold indices must reproduce the stored test labels of the main benchmark
    chk = 0
    for ds in ["FreeSolv", "ESOL", "Lipophilicity", "BACE"]:
        for proto in PROTOCOLS:
            p = os.path.join(ROOT, "results", "folds", f"{ds}__{proto}__42.pkl")
            o = pickle.load(open(p, "rb")); y = np.array([d.y.item() for d in B["datasets"][ds]], np.float32)
            assert np.allclose(o["meta"]["y_true"], y[fold_indices(ds, proto, 42)[2]]), (ds, proto); chk += 1
    print("fold-index check passed on", chk, "folds", flush=True)
    for ds in DATASET_CFG:
        for proto in PROTOCOLS:
            for seed in SEEDS:
                path = os.path.join(OUT, f"{ds}__{proto}__{seed}.pkl")
                if os.path.exists(path): continue
                out = run(ds, proto, seed); pickle.dump(out, open(path, "wb"))
        print(ds, "done", flush=True)
