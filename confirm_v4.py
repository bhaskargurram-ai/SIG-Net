"""Confirmatory evaluation of the frozen SIG-Net v4 on seeds 1, 2, 3 (pre-registered in ANALYSIS_PLAN.md, Addendum 2).
Run only after the design freeze. Writes results/confirm_v4.csv and results/confirm_v4_summary.json."""
import os, sys, glob, json, time
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routers import *

CONF_SEEDS = {1, 2, 3}
files = {}
for d in ["experts_podD", "experts_podE", "experts"]:
    for p in glob.glob(os.path.join(ROOT, "results", d, "*.pkl")):
        if int(os.path.basename(p)[:-4].split("__")[2]) in CONF_SEEDS:
            files[os.path.basename(p)] = p
print(len(files), "confirmatory folds found", flush=True)
rows = []
for name, p in sorted(files.items()):
    t = time.time(); F9 = Fold(p, EXPERTS_ALL); F7 = Fold(p, EXPERTS_V3)
    res = {"SIG-Net v4 (frozen)": v4_fast_select(F9)["test_preds"],
           "v4 w/o residual correction": v4_fast_select(F9, restrict={"nu": [0.0]})["test_preds"],
           "v4 w/o composite kernel": v4_fast_select(F9, restrict={"a": [1.0]})["test_preds"],
           "v4 w/o novelty shrinkage": v4_fast_select(F9, restrict={"gamma": [0.0, "global"], "gamma_d": [0.0]})["test_preds"],
           "SIG-Net v3 router (7)": v3_select(F7)["test_preds"],
           "Global stack (7)": F7.T @ F7.w_glob, "Global stack (9)": F9.T @ F9.w_glob}
    for j, n in enumerate(EXPERTS_ALL):
        res[f"{n} [5-fold bagged]"] = F9.T[:, j]
    for m, pr in res.items():
        rows.append({"Dataset": F9.ds, "Split": F9.proto, "Seed": F9.seed, "Model": m, "Task": F9.task, "Primary": score(F9.y_te, pr, F9.task)})
    print(f"{F9.ds:13s} {F9.proto:16s} seed {F9.seed} ({time.time()-t:.0f}s)", flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(ROOT, "results", "confirm_v4.csv"), index=False)


def h2h(D, a, b, exclude_freesolv_generic=True):
    rel = []
    for (ds, sp, s), g in D.groupby(["Dataset", "Split", "Seed"]):
        if exclude_freesolv_generic and ds == "FreeSolv" and sp == "generic_scaffold":
            continue
        m = g.set_index("Model").Primary; lower = g.Task.iloc[0] == "regression"
        rel.append(100 * ((m[b] - m[a]) / m[b] if lower else (m[a] - m[b]) / m[b]))
    rel = np.array(rel); w = stats.wilcoxon(rel, zero_method="zsplit")
    return {"n": len(rel), "wins": int((rel > 0).sum()), "mean": float(rel.mean()), "median": float(np.median(rel)), "p": float(w.pvalue)}


P = "SIG-Net v4 (frozen)"
summary = {"primary_vs_global7": h2h(D, P, "Global stack (7)")}
for b in ["SIG-Net v3 router (7)", "Global stack (9)", "v4 w/o residual correction", "v4 w/o composite kernel", "v4 w/o novelty shrinkage"] + [f"{n} [5-fold bagged]" for n in EXPERTS_ALL]:
    summary[f"vs {b}"] = h2h(D, P, b)
summary["v3_vs_global7"] = h2h(D, "SIG-Net v3 router (7)", "Global stack (7)")
summary["freesolv_generic_included_primary"] = h2h(D, P, "Global stack (7)", exclude_freesolv_generic=False)
json.dump(summary, open(os.path.join(ROOT, "results", "confirm_v4_summary.json"), "w"), indent=1)
for k, v in summary.items():
    print(f"{k:55s} n={v['n']:3d} wins={v['wins']:3d} mean={v['mean']:+.3f}% median={v['median']:+.3f}% p={v['p']:.2e}")
