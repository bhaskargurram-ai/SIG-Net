"""Aggregate per-fold results into tables, statistics and figures. Works on partial results."""
import os, glob, pickle
from collections import defaultdict
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
PROPOSED = os.environ.get("JCIM_PROPOSED", "SIG-Net v3 (Proposed)")
DS_ORDER = ["FreeSolv", "ESOL", "Lipophilicity", "BACE", "BBBP"]
PROTO_ORDER = ["random", "bm_scaffold", "generic_scaffold", "butina_cluster"]


def _merged_fold_files():
    """Merge v2 (results/folds) and v3 (results/folds_v3 or v3_pod*/folds_v3) pickles per fold."""
    files = defaultdict(list)
    for d in [os.path.join(RES, "folds"), os.path.join(RES, "folds_v3"), os.path.join(RES, "folds_cb")] + glob.glob(os.path.join(RES, "v3_pod*", "folds_v3")):
        for p in glob.glob(os.path.join(d, "*.pkl")):
            files[os.path.basename(p)].append(p)
    return files


def _load_merged(paths):
    out = None
    for p in paths:
        with open(p, "rb") as fh:
            o = pickle.load(fh)
        if out is None:
            out = o
        else:
            out["results"].update(o["results"]); out["meta"].update({k: v for k, v in o["meta"].items() if k not in out["meta"]})
    return out


def load_folds(require_v3=False):
    rows, preds = [], {}
    for name, paths in sorted(_merged_fold_files().items()):
        ds, proto, seed = name[:-4].split("__")
        out = _load_merged(paths)
        if require_v3 and "SIG-Net v3 (Proposed)" not in out["results"]:
            continue
        task = out["meta"]["task"]
        preds[(ds, proto, int(seed))] = out
        for model, r in out["results"].items():
            if model.startswith("_"):
                continue
            m = r["metrics"]
            rows.append({"Dataset": ds, "Split": proto, "Seed": int(seed), "Task": task, "Model": model,
                         "Primary": m["primary"], **{k: m.get(k, np.nan) for k in ["RMSE", "MAE", "R2", "ROC_AUC", "PR_AUC"]},
                         "used_gate": r.get("used_gate", np.nan)})
    return pd.DataFrame(rows), preds


def complete_settings(df):
    """(dataset, split) pairs that have all 5 seeds."""
    c = df[df.Model == PROPOSED].groupby(["Dataset", "Split"]).Seed.nunique()
    return set(c[c == 5].index)


def main_table(df, models=None):
    lower = df.Task.eq("regression")
    g = df.groupby(["Dataset", "Split", "Model"]).Primary.agg(["mean", "std", "count"]).reset_index()
    if models is not None:
        g = g[g.Model.isin(models)]
    return g


def rankings(df, settings=None):
    recs = []
    for (ds, sp), grp in df.groupby(["Dataset", "Split"]):
        if settings is not None and (ds, sp) not in settings:
            continue
        lower = grp.Task.iloc[0] == "regression"
        means = grp.groupby("Model").Primary.mean()
        ranks = means.rank(ascending=lower, method="average")
        for m, r in ranks.items():
            recs.append({"Model": m, "Dataset": ds, "Split": sp, "OOD": sp != "random", "Rank": r})
    R = pd.DataFrame(recs)
    if R.empty:
        return R, R
    out = R.groupby("Model").agg(Overall_Rank=("Rank", "mean"),
                                 First=("Rank", lambda s: int((s == 1).sum())),
                                 Top3=("Rank", lambda s: int((s <= 3).sum())),
                                 N_settings=("Rank", "size")).reset_index()
    ood = R[R.OOD].groupby("Model").Rank.mean().rename("OOD_Rank")
    out = out.merge(ood, on="Model", how="left").sort_values("Overall_Rank").reset_index(drop=True)
    return out, R


def friedman_nemenyi(R):
    """Friedman test over settings x models, Nemenyi critical difference (alpha=0.05)."""
    piv = R.pivot_table(index=["Dataset", "Split"], columns="Model", values="Rank").dropna(axis=1)
    k, n = piv.shape[1], piv.shape[0]
    chi, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
         13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}[min(k, 20)]
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"chi2": chi, "p": p, "k": k, "n": n, "CD": cd, "mean_ranks": piv.mean().sort_values()}


def head_to_head(df, baseline, settings=None):
    """Paired comparison of PROPOSED vs one baseline over every (dataset, split, seed) fold.
    Effect = relative improvement (%) oriented so positive means PROPOSED is better."""
    rows = []
    for (ds, sp, s), grp in df.groupby(["Dataset", "Split", "Seed"]):
        if settings is not None and (ds, sp) not in settings:
            continue
        m = grp.set_index("Model").Primary
        if PROPOSED not in m or baseline not in m:
            continue
        lower = grp.Task.iloc[0] == "regression"
        a, b = m[PROPOSED], m[baseline]
        rel = 100 * ((b - a) / b if lower else (a - b) / b)
        rows.append({"Dataset": ds, "Split": sp, "Seed": s, "rel": rel, "win": rel > 0})
    H = pd.DataFrame(rows)
    if len(H) < 6:
        return None
    w = stats.wilcoxon(H.rel, alternative="two-sided", zero_method="zsplit")
    per_setting = H.groupby(["Dataset", "Split"]).rel.mean()
    w_set = stats.wilcoxon(per_setting, alternative="two-sided", zero_method="zsplit") if len(per_setting) >= 6 else None
    return {"baseline": baseline, "n_folds": len(H), "wins": int(H.win.sum()), "mean_rel": H.rel.mean(), "median_rel": H.rel.median(),
            "p_folds": w.pvalue, "n_settings": len(per_setting), "settings_won": int((per_setting > 0).sum()),
            "p_settings": (w_set.pvalue if w_set is not None else np.nan)}


def per_setting_tests(df, settings=None):
    """Per (dataset, split): proposed vs best other model, seed-paired t-test (5 seeds)."""
    rows = []
    for (ds, sp), grp in df.groupby(["Dataset", "Split"]):
        if settings is not None and (ds, sp) not in settings:
            continue
        lower = grp.Task.iloc[0] == "regression"
        means = grp.groupby("Model").Primary.mean().sort_values(ascending=lower)
        others = means.drop(PROPOSED)
        best = others.index[0]
        a = grp[grp.Model == PROPOSED].sort_values("Seed").Primary.values
        b = grp[grp.Model == best].sort_values("Seed").Primary.values
        t = stats.ttest_rel(a, b) if len(a) == len(b) and len(a) > 1 else None
        rank = int((means.rank(ascending=lower, method="min"))[PROPOSED])
        rows.append({"Dataset": ds, "Split": sp, "Proposed": f"{a.mean():.3f} ± {a.std(ddof=1):.3f}",
                     "Best other": f"{b.mean():.3f} ± {b.std(ddof=1):.3f}", "Best other model": best, "Rank": rank,
                     "Seed wins": int(((a < b) if lower else (a > b)).sum()), "paired t p": (t.pvalue if t else np.nan)})
    return pd.DataFrame(rows)


def novelty_stratified(preds, models, settings=None, bins=(0, 0.35, 0.5, 1.01)):
    """Mean absolute error per Tanimoto-similarity bin for regression datasets, pooled over OOD folds."""
    rows = []
    for (ds, sp, s), out in preds.items():
        if out["meta"]["task"] != "regression" or sp == "random" or (settings is not None and (ds, sp) not in settings):
            continue
        y, sim = out["meta"]["y_true"], out["meta"]["nn_sim"]
        ystd = np.std(y) + 1e-9
        for m in models:
            if m not in out["results"]:
                continue
            err = np.abs(y - out["results"][m]["test_preds"]) / ystd
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (sim >= lo) & (sim < hi)
                if sel.any():
                    rows.append({"Dataset": ds, "Model": m, "Bin": f"[{lo:.2f},{min(hi,1):.2f})", "nMAE": err[sel].mean(), "n": int(sel.sum())})
    return pd.DataFrame(rows)


def gate_analysis(preds, settings=None):
    """Relationship between learned gate weight and novelty for folds where the gate was selected."""
    rows = []
    for (ds, sp, s), out in preds.items():
        if settings is not None and (ds, sp) not in settings:
            continue
        r = out["results"].get("SIG-Net v2 (Proposed)")
        if r is None: continue
        rows.append({"Dataset": ds, "Split": sp, "Seed": s, "used_gate": r["used_gate"],
                     "theta_sim": (r["theta"][1] if r["used_gate"] else np.nan),
                     "theta_disagree": (r["theta"][2] if r["used_gate"] else np.nan),
                     "mean_w": float(np.mean(r["gates"])),
                     "r_w_sim": (stats.spearmanr(out["meta"]["nn_sim"], r["gates"])[0] if r["used_gate"] else np.nan)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, preds = load_folds()
    print(f"{df[['Dataset','Split','Seed']].drop_duplicates().shape[0]} folds loaded")
    S = complete_settings(df); print("complete settings:", len(S))
    rk, R = rankings(df, S or None)
    print(rk.to_string())
    for b in ["Stacked (Ensemble + Anchor, global w)", "D-MPNN + RDKit-2D Ensemble (3x)", "Stacked (XGBoost + D-MPNN)", "Anchor-Only (Multi-View Stack)", "D-MPNN + RDKit-2D"]:
        h = head_to_head(df, b)
        if h: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})
    print(per_setting_tests(df).to_string())
