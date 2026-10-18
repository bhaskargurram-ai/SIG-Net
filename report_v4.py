"""Confirmatory (fresh-seed) tables, numbers and figure for the manuscript. Reads results/confirm_v4.csv and
results/confirm_v4_summary.json (written by confirm_v4.py) and results/dev_v4_seed11_FROZEN.csv."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures import OK

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "results"); FINAL = os.path.join(RES, "final")
os.makedirs(os.path.join(FINAL, "figures"), exist_ok=True)
DS = ["FreeSolv", "ESOL", "Lipophilicity", "BACE", "BBBP"]


def rel_table(D, a, b):
    out = []
    for (ds, sp, s), g in D.groupby(["Dataset", "Split", "Seed"]):
        m = g.set_index("Model").Primary; lower = g.Task.iloc[0] == "regression"
        out.append({"Dataset": ds, "Split": sp, "Seed": s, "rel": 100 * ((m[b] - m[a]) / m[b] if lower else (m[a] - m[b]) / m[b])})
    return pd.DataFrame(out)


def build():
    D = pd.read_csv(os.path.join(RES, "confirm_v4.csv")); S = json.load(open(os.path.join(RES, "confirm_v4_summary.json")))
    Dx = D[~((D.Dataset == "FreeSolv") & (D.Split == "generic_scaffold"))]
    N = {"summary": S, "n_conf": int(S["primary_vs_global7"]["n"])}
    per = {}
    for key, a, b in [("v4_vs_global", "SIG-Net v4 (frozen)", "Global stack (7)"), ("v3_vs_global", "SIG-Net v3 router (7)", "Global stack (7)"),
                      ("v4_vs_nores", "SIG-Net v4 (frozen)", "v4 w/o residual correction")]:
        r = rel_table(Dx, a, b); per[key] = {d: {"mean": float(r[r.Dataset == d].rel.mean()), "wins": int((r[r.Dataset == d].rel > 0).sum()), "n": int((r.Dataset == d).sum())} for d in DS}
    N["per_dataset"] = per
    g = Dx.groupby(["Dataset", "Split", "Model", "Task"]).Primary.mean().reset_index(); ranks = []
    for (ds, sp), q in g.groupby(["Dataset", "Split"]):
        lower = q.Task.iloc[0] == "regression"; q = q.set_index("Model").Primary; ranks.append(q.rank(ascending=lower, method="average"))
    R = pd.concat(ranks, axis=1).mean(1).sort_values(); N["ranks"] = R.round(2).to_dict(); N["n_conf_settings"] = len(ranks)
    dev = pd.read_csv(os.path.join(RES, "dev_v4_seed11_FROZEN.csv")); ref = "Global stack (7)"; dv = []
    for (ds, sp), q in dev.groupby(["Dataset", "Split"]):
        m = q.set_index("Model").score; lower = q.task.iloc[0] == "regression"
        for mm in m.index: dv.append({"Model": mm, "rel": 100 * ((m[ref] - m[mm]) / m[ref] if lower else (m[mm] - m[ref]) / m[ref])})
    dv = pd.DataFrame(dv).groupby("Model").rel.agg(["mean", "median", lambda x: int((x > 0).sum())]); dv.columns = ["mean", "median", "wins"]
    N["dev"] = dv.round(3).to_dict(orient="index")
    json.dump(N, open(os.path.join(FINAL, "numbers_v4.json"), "w"), indent=1, default=float)
    # figure: per-dataset mean relative improvement over the global stack (confirmatory folds)
    fig, ax = plt.subplots(figsize=(6.8, 2.5)); x = np.arange(len(DS)); w = 0.27
    for j, (key, lab, col) in enumerate([("v3_vs_global", "v3 router (local softmax)", OK["grey"]), ("v4_vs_nores", "v4 vs v4 w/o residual corr.", OK["orange"]), ("v4_vs_global", "SIG-Net v4 (frozen)", OK["red"])]):
        r = rel_table(Dx, *{"v4_vs_global": ("SIG-Net v4 (frozen)", "Global stack (7)"), "v3_vs_global": ("SIG-Net v3 router (7)", "Global stack (7)"),
                            "v4_vs_nores": ("SIG-Net v4 (frozen)", "v4 w/o residual correction")}[key])
        m = [r[r.Dataset == d].rel.mean() for d in DS]; se = [r[r.Dataset == d].rel.std(ddof=1) / np.sqrt((r.Dataset == d).sum()) for d in DS]
        ax.bar(x + (j - 1) * w, m, w, yerr=se, color=col, label=lab, capsize=2, error_kw={"lw": 0.7})
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(DS)
    ax.set_ylabel("Mean relative improvement (%)\nover reference (± SE)"); ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    ax.set_title(f"Pre-registered confirmation on fresh seeds 1-3 ({N['n_conf']} folds; FreeSolv generic excluded)", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FINAL, "figures", "fig5_confirmatory.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    return N


if __name__ == "__main__":
    N = build(); print(json.dumps({k: N[k] for k in ["n_conf", "n_conf_settings", "per_dataset"]}, indent=1, default=float)[:2500])
