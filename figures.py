"""Publication figures. Palette: colour-blind-safe (Okabe-Ito)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "yellow": "#F0E442"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 120})
SPLIT_LABEL = {"random": "Random", "bm_scaffold": "BM scaffold", "generic_scaffold": "Generic skeleton", "butina_cluster": "Butina cluster"}


def _num(s):
    return float(str(s).split("±")[0].split("±")[0])


def fig_leakage(t2, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    dss = [d for d in ["FreeSolv", "ESOL", "Lipophilicity", "BACE", "BBBP"] if d in set(t2.Dataset)]; splits = list(SPLIT_LABEL); x = np.arange(len(dss)); w = 0.2
    cols = [OK["grey"], OK["blue"], OK["green"], OK["orange"]]
    for k, sp in enumerate(splits):
        sub = t2[t2.Split == sp].set_index("Dataset").loc[dss]
        axes[0].bar(x + (k - 1.5) * w, [_num(v) for v in sub["Test mols w/ seen generic skeleton (%)"]], w, color=cols[k], label=SPLIT_LABEL[sp])
        axes[1].bar(x + (k - 1.5) * w, [_num(v) for v in sub["NN Tanimoto"]], w, color=cols[k])
    axes[0].set_ylabel("Test molecules whose generic\nskeleton occurs in training (%)"); axes[1].set_ylabel("Mean nearest-neighbour\nTanimoto to training set")
    for a in axes: a.set_xticks(x); a.set_xticklabels(dss)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right"); axes[0].set_ylim(0, 115)
    axes[0].set_title("(a) Carbon-skeleton leakage", loc="left", fontsize=8.5); axes[1].set_title("(b) Structural novelty of the test set", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_cd(fn, path, highlight="SIG-Net v2 (Proposed)", top=12):
    mr = fn["mean_ranks"].iloc[:top]; cd = fn["CD"]
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(mr) + 0.8))
    y = np.arange(len(mr))[::-1]
    for yi, (m, r) in zip(y, mr.items()):
        c = OK["red"] if m == highlight else OK["blue"]
        ax.plot([r], [yi], "o", color=c, ms=5); ax.text(r + 0.15, yi, f"{m}  ({r:.2f})", va="center", fontsize=7, color=c)
    best = mr.iloc[0]; ax.axvspan(best, best + cd, color=OK["grey"], alpha=0.15, lw=0)
    ax.text(best + cd, y[-1] - 0.6, f"Nemenyi CD = {cd:.2f} ranks", fontsize=7, color=OK["grey"], ha="right")
    ax.set_yticks([]); ax.set_xlabel(f"Mean rank over {fn['n']} dataset-split settings (lower is better)")
    ax.set_xlim(min(mr) - 0.5, max(mr) + 9); ax.spines["left"].set_visible(False)
    ax.set_title(f"Friedman χ² = {fn['chi2']:.1f}, p = {fn['p']:.1e}  (k = {fn['k']} models, n = {fn['n']} settings)", loc="left", fontsize=8, pad=8)
    ax.set_ylim(y[-1] - 1.0, y[0] + 0.6)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_novelty(nov, path, models, labels=None):
    if nov.empty:
        print("fig_novelty: no OOD regression folds yet, skipped"); return
    dss = [d for d in ["FreeSolv", "ESOL", "Lipophilicity"] if d in set(nov.Dataset)]
    fig, axes = plt.subplots(1, len(dss), figsize=(2.4 * len(dss), 2.5), sharey=False)
    axes = np.atleast_1d(axes); cols = [OK["grey"], OK["orange"], OK["blue"], OK["red"]]
    for a, ds in zip(axes, dss):
        sub = nov[nov.Dataset == ds]; bins = sorted(sub.Bin.unique()); x = np.arange(len(bins)); w = 0.8 / len(models)
        for k, m in enumerate(models):
            v = sub[sub.Model == m].groupby("Bin").nMAE.mean().reindex(bins)
            a.bar(x + (k - (len(models) - 1) / 2) * w, v.values, w, color=cols[k % len(cols)], label=(labels or models)[k])
        n = sub[sub.Model == models[0]].groupby("Bin").n.sum().reindex(bins)
        a.set_xticks(x); a.set_xticklabels([f"{b}\nn={int(c)}" for b, c in zip(bins, n)], fontsize=6.5)
        a.set_title(ds, fontsize=8.5); a.set_xlabel("NN Tanimoto to training set")
    axes[0].set_ylabel("MAE / SD(y)  (OOD splits pooled)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(models), frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_gate(preds, path, proposed="SIG-Net v2 (Proposed)"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    sims, ws, dsl = [], [], []
    for (ds, sp, s), out in preds.items():
        r = out["results"][proposed]
        if r.get("used_gate"):
            sims.append(out["meta"]["nn_sim"]); ws.append(r["gates"]); dsl += [ds] * len(r["gates"])
    if sims:
        sims, ws, dsl = np.concatenate(sims), np.concatenate(ws), np.array(dsl)
        cmap = {"FreeSolv": OK["blue"], "ESOL": OK["orange"], "Lipophilicity": OK["green"], "BACE": OK["purple"]}
        for ds, c in cmap.items():
            m = dsl == ds
            if m.any(): axes[0].scatter(sims[m], ws[m], s=3, alpha=0.35, color=c, label=ds, rasterized=True)
        axes[0].set_xlabel("NN Tanimoto to training set"); axes[0].set_ylabel("Graph-expert weight w(x)")
        axes[0].legend(frameon=False, fontsize=6.5, markerscale=3)
    axes[0].set_title("(a) Learned gate vs. novelty (gate-selected folds)", loc="left", fontsize=8.5)
    use = pd.Series({k: v["results"][proposed]["used_gate"] for k, v in preds.items()})
    frac = use.groupby(level=[0, 1]).mean().unstack()
    im = axes[1].imshow(frac.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(frac.shape[1])); axes[1].set_xticklabels([SPLIT_LABEL.get(c, c) for c in frac.columns], rotation=20, fontsize=7)
    axes[1].set_yticks(range(frac.shape[0])); axes[1].set_yticklabels(frac.index, fontsize=7)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            axes[1].text(j, i, f"{frac.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    axes[1].set_title("(b) Fraction of seeds where CV selected the gate", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
