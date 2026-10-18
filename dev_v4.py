"""Design evaluation of router variants on DEVELOPMENT folds only (seed 11). Never reads confirmatory seeds."""
import os, sys, glob, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routers import *

files = sorted(set(p for d in ["experts_podD", "experts_podE", "experts"] for p in glob.glob(os.path.join(ROOT, "results", d, "*__11.pkl"))))
uniq = {}
for p in files: uniq[os.path.basename(p)] = p
rows = []
for name, p in sorted(uniq.items()):
    assert name.endswith("__11.pkl"), "development script must only read seed 11"
    t = time.time(); F9 = Fold(p, EXPERTS_ALL); F7 = Fold(p, EXPERTS_V3)
    res = {
        "Global stack (7)": F7.T @ F7.w_glob,
        "Global stack (9)": F9.T @ F9.w_glob,
        "v3 router (7)": v3_select(F7)["test_preds"],
        "v4 full (9)": v4_fast_select(F9)["test_preds"],
        "v4 full (7)": v4_fast_select(F7)["test_preds"],
        "v4 w/o residual corr (9)": v4_fast_select(F9, restrict={"nu": [0.0]})["test_preds"],
        "v4 w/o composite kernel (9)": v4_fast_select(F9, restrict={"a": [1.0]})["test_preds"],
        "v4 w/o novelty shrinkage (9)": v4_fast_select(F9, restrict={"gamma": [0.0, "global"], "gamma_d": [0.0]})["test_preds"],
    }
    for m, pred in res.items():
        rows.append({"Dataset": F9.ds, "Split": F9.proto, "Model": m, "score": score(F9.y_te, pred, F9.task), "task": F9.task})
    print(f"{F9.ds:13s} {F9.proto:16s} ({time.time()-t:.0f}s)", flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(ROOT, "results", "dev_v4_seed11.csv"), index=False)
ref = "Global stack (7)"
def rel(g):
    r = g.set_index("Model").score; lower = g.task.iloc[0] == "regression"; b = r[ref]
    return 100 * ((b - r) / b if lower else (r - b) / b)
R = D.groupby(["Dataset", "Split"]).apply(rel)
print(f"\nDEV (seed 11, {R.shape[0]} folds): relative improvement vs {ref} (%)  [positive = better]")
summ = pd.DataFrame({"mean": R.mean(), "median": R.median(), "wins": (R > 0).sum(), "losses": (R < 0).sum()}).sort_values("mean", ascending=False)
print(summ.round(3).to_string())
print("\nper dataset mean:"); print(R.groupby(level=0).mean().round(2).to_string())
