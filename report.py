"""Final tables (CSV + LaTeX) and figures from the merged v2+v3 fold results. Every number used in the
manuscript is written to results/final/numbers.json by this script, so nothing in the paper is hand-typed."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis as A
from figures import fig_leakage, fig_cd, fig_novelty, OK, SPLIT_LABEL

FINAL = os.path.join(A.RES, "final"); FIG = os.path.join(FINAL, "figures"); os.makedirs(FIG, exist_ok=True)
P = "SIG-Net v3 (Proposed)"
OWN_ABL = ["SIG-Net v3 w/o novelty shrinkage", "Global stack (same cross-fitted experts)"]
DISPLAY = {  # manuscript names
    "SIG-Net v3 (Proposed)": "SIG-Net", "SIG-Net v3 w/o novelty shrinkage": "SIG-Net w/o novelty shrinkage",
    "Global stack (same cross-fitted experts)": "Global stack (same experts)", "SIG-Net v2 (Proposed)": "Two-expert novelty gate",
    "SIG-Net v2 w/o ensemble": "Two-expert gate (single D-MPNN)", "Dual-Stream Gated Residual (v1 design)": "Dual-stream gated residual",
    "Stacked (Ensemble + Anchor, global w)": "Stack: D-MPNN ens. + anchor", "Stacked (XGBoost + D-MPNN)": "Stack: XGBoost + D-MPNN",
    "Anchor-Only (Multi-View Stack)": "Multi-view anchor", "Anchor (Multi-View Stack) [5-fold bagged]": "Multi-view anchor (5-fold bagged)",
}
dn = lambda m: DISPLAY.get(m, m)
MAIN_COLS = ["XGBoost (ECFP4 + RDKit-2D)", "HighwayMLP (ECFP4 + RDKit-2D)", "AttentiveFP", "D-MPNN (Chemprop)", "D-MPNN + RDKit-2D",
             "D-MPNN + Deep CORAL", "D-MPNN + RDKit-2D [5-fold bagged]", "ChemBERTa-2 (77M-MTR) + linear", "ChemBERTa-2 (77M-MTR) + XGBoost", "Stacked (XGBoost + D-MPNN)", "Global stack (same cross-fitted experts)", P]


def fmt(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def build():
    os.environ["JCIM_PROPOSED"] = P
    df, preds = A.load_folds(require_v3=True); S = A.complete_settings(df); N = {}
    df.to_csv(os.path.join(A.RES, "all_fold_metrics_v2v3.csv"), index=False)
    N["n_folds"], N["n_settings"], N["n_models"] = len(preds), len(S), int(df.Model.nunique())

    # ---- Table 3: main benchmark (selected columns)
    g = df.groupby(["Dataset", "Split", "Model"]).Primary.agg(["mean", "std"]).reset_index()
    rows = []
    for ds in A.DS_ORDER:
        for sp in A.PROTO_ORDER:
            if (ds, sp) not in S:
                continue  # only settings with all five seeds
            sub = g[(g.Dataset == ds) & (g.Split == sp)].set_index("Model"); lower = ds not in ("BACE", "BBBP")
            best = sub.loc[[c for c in MAIN_COLS if c in sub.index], "mean"]; b1 = best.idxmin() if lower else best.idxmax()
            r = {"Dataset": ds, "Split": SPLIT_LABEL[sp]}
            for c in MAIN_COLS:
                cell = fmt(sub.loc[c, "mean"], sub.loc[c, "std"]); r[dn(c)] = f"\\textbf{{{cell}}}" if c == b1 else cell
            rows.append(r)
    T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(FINAL, "table3_main_benchmark.csv"), index=False)

    # ---- rankings (external models + proposed; own ablations reported separately)
    ext = df[~df.Model.isin(OWN_ABL)]
    rk, R = A.rankings(ext, S); rk.insert(0, "Name", rk.Model.map(dn)); rk.to_csv(os.path.join(FINAL, "table4_rankings.csv"), index=False)
    fn = A.friedman_nemenyi(R); N.update(friedman_chi2=fn["chi2"], friedman_p=fn["p"], cd=fn["CD"], cd_k=fn["k"])
    pr = rk.set_index("Model").loc[P]; N.update(rank_overall=pr.Overall_Rank, rank_ood=pr.OOD_Rank, firsts=int(pr.First), top3=int(pr.Top3))
    nxt = rk[rk.Model != P].iloc[0]; N.update(second_model=dn(nxt.Model), second_rank=nxt.Overall_Rank)
    per = R[R.Model == P].set_index(["Dataset", "Split"]).Rank
    N["firsts_by_dataset"] = {ds: int((per.xs(ds) == 1).sum()) for ds in A.DS_ORDER if ds in set(per.index.get_level_values(0))}

    # ---- head-to-head over 80 folds
    H = pd.DataFrame([A.head_to_head(df, b, S) for b in sorted(df.Model.unique()) if b != P])
    H.insert(0, "Name", H.baseline.map(dn)); H = H.sort_values("mean_rel"); H.to_csv(os.path.join(FINAL, "table5_head_to_head.csv"), index=False)
    def h(b):
        r = H.set_index("baseline").loc[b]; return {"wins": int(r.wins), "mean": float(r.mean_rel), "median": float(r.median_rel), "p": float(r.p_folds), "settings": int(r.settings_won)}
    for key, b in [("vs_global", "Global stack (same cross-fitted experts)"), ("vs_noshrink", "SIG-Net v3 w/o novelty shrinkage"),
                   ("vs_v2", "SIG-Net v2 (Proposed)"), ("vs_xgbdmpnn", "Stacked (XGBoost + D-MPNN)"), ("vs_dmpnn", "D-MPNN + RDKit-2D"),
                   ("vs_dmpnn_bag", "D-MPNN + RDKit-2D [5-fold bagged]"), ("vs_coral", "D-MPNN + Deep CORAL"), ("vs_anchor", "Anchor-Only (Multi-View Stack)"),
                   ("vs_chemprop", "D-MPNN (Chemprop)"), ("vs_xgb", "XGBoost (ECFP4 + RDKit-2D)"), ("vs_attfp", "AttentiveFP"),
                   ("vs_cb_lin", "ChemBERTa-2 (77M-MTR) + linear"), ("vs_cb_xgb", "ChemBERTa-2 (77M-MTR) + XGBoost")]:
        N[key] = h(b)
    ext_h = H[~H.baseline.isin(OWN_ABL)]
    # pre-specified primary endpoint: original 4 datasets (80 folds), original comparator set
    orig = df[df.Dataset.isin(["FreeSolv", "ESOL", "Lipophilicity", "BACE"]) & ~df.Model.str.startswith("ChemBERTa")]
    S0 = A.complete_settings(orig); hp = A.head_to_head(orig, "Global stack (same cross-fitted experts)", S0)
    N["prereg"] = {"n_folds": int(hp["n_folds"]), "wins": int(hp["wins"]), "mean": float(hp["mean_rel"]), "median": float(hp["median_rel"]), "p": float(hp["p_folds"]), "settings": int(hp["settings_won"]), "n_settings": int(hp["n_settings"])}
    N["min_wins_external"] = int(ext_h.wins.min()); N["max_p_external"] = float(ext_h.p_folds.max())
    # gain decomposition: global stack of same experts vs strongest external stack
    os.environ["JCIM_PROPOSED"] = "Global stack (same cross-fitted experts)"
    import importlib; importlib.reload(A)
    N["global_vs_xgbdmpnn"] = float(A.head_to_head(df, "Stacked (XGBoost + D-MPNN)", S)["mean_rel"])
    os.environ["JCIM_PROPOSED"] = P; importlib.reload(A)

    # ---- per-setting table
    PS = A.per_setting_tests(ext, S); PS["Best other model"] = PS["Best other model"].map(dn); PS.to_csv(os.path.join(FINAL, "tableS_per_setting.csv"), index=False)

    # ---- ablation table (SIG-Net components), mean primary per setting
    abl_models = ["D-MPNN + RDKit-2D", "D-MPNN + RDKit-2D Ensemble (3x)", "D-MPNN + RDKit-2D [5-fold bagged]", "SIG-Net v2 (Proposed)",
                  "Global stack (same cross-fitted experts)", "SIG-Net v3 w/o novelty shrinkage", P]
    gS = g[[(d, sp) in S for d, sp in zip(g.Dataset, g.Split)]]
    ab = gS[gS.Model.isin(abl_models)].pivot_table(index=["Dataset", "Split"], columns="Model", values="mean")[abl_models]
    ab.columns = [dn(c) for c in ab.columns]; ab.to_csv(os.path.join(FINAL, "table6_ablation.csv"))
    # mean relative improvement of each ablation stage over single D-MPNN+RDKit
    rel = {}
    for m in abl_models[1:]:
        os.environ["JCIM_PROPOSED"] = m; importlib.reload(A)
        rel[dn(m)] = float(A.head_to_head(df, "D-MPNN + RDKit-2D", S)["mean_rel"])
    os.environ["JCIM_PROPOSED"] = P; importlib.reload(A); N["ablation_rel_vs_dmpnn"] = rel

    # ---- splits
    t2 = pd.read_csv(os.path.join(A.RES, "table2_split_diagnostics_5seed.csv"))
    num = lambda s: float(str(s).split("±")[0].split("±")[0])
    bm = t2[t2.Split == "bm_scaffold"]
    N["bm_seen_skeleton_mol_range"] = [min(map(num, bm["Test mols w/ seen generic skeleton (%)"])), max(map(num, bm["Test mols w/ seen generic skeleton (%)"]))]
    N["bm_skeleton_overlap_range"] = [min(map(num, bm["Generic skeleton overlap (%)"])), max(map(num, bm["Generic skeleton overlap (%)"]))]

    # ---- novelty-stratified
    M = ["Stacked (XGBoost + D-MPNN)", "D-MPNN + RDKit-2D [5-fold bagged]", "Global stack (same cross-fitted experts)", P]
    nov = A.novelty_stratified(preds, M, S); nov.to_csv(os.path.join(FINAL, "table_novelty_stratified.csv"), index=False)
    pv = nov.groupby(["Bin", "Model"]).apply(lambda q: np.average(q.nMAE, weights=q.n)).unstack()
    N["novelty_bins"] = {b: {"vs_xgbdmpnn": float(100 * (pv.loc[b, M[0]] - pv.loc[b, P]) / pv.loc[b, M[0]]),
                             "vs_global": float(100 * (pv.loc[b, M[2]] - pv.loc[b, P]) / pv.loc[b, M[2]])} for b in pv.index}

    # ---- expert weights by novelty
    W = []
    for (ds, sp, s), out in preds.items():
        if sp == "random" or (ds, sp) not in S: continue
        r = out["results"][P]; sim = out["meta"]["nn_sim"]
        for j, name in enumerate(r["expert_names"]):
            for lo, hi in [(0, .35), (.35, .5), (.5, 1.01)]:
                sel = (sim >= lo) & (sim < hi)
                if sel.any(): W.append({"Dataset": ds, "Expert": name, "Bin": f"[{lo:.2f},{min(hi,1):.2f})", "w": float(r["weights"][sel, j].mean()), "n": int(sel.sum())})
    W = pd.DataFrame(W); W.to_csv(os.path.join(FINAL, "table_expert_weights.csv"), index=False)
    cfg = pd.Series([str(out["results"][P]["config"][2]) for out in preds.values()]).value_counts(); N["gamma_selected"] = cfg.to_dict()

    # ---- figures
    fig_leakage(t2, os.path.join(FIG, "fig1_leakage.png"))
    fn["mean_ranks"].index = [dn(m) for m in fn["mean_ranks"].index]; fig_cd(fn, os.path.join(FIG, "fig2_ranks.png"), highlight="SIG-Net", top=14)
    fig_novelty(nov, os.path.join(FIG, "fig3_novelty.png"), M, ["Stack: XGB+D-MPNN", "D-MPNN (bagged)", "Global stack", "SIG-Net"])
    fig_weights(W, os.path.join(FIG, "fig4_expert_weights.png"))
    fig_toc(rk, N, os.path.join(FIG, "toc_graphic.png"))
    json.dump(N, open(os.path.join(FINAL, "numbers.json"), "w"), indent=1, default=float)
    return N, T3, rk, H, PS, ab


def fig_toc(rk, N, path):
    """ACS TOC graphic: exactly 3.25 x 1.75 in."""
    top = rk.head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 1.75), dpi=300)
    cols = [OK["red"] if n == "SIG-Net" else OK["blue"] for n in top.Name]
    ax.barh(range(len(top)), top.Overall_Rank, color=cols, height=0.62)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([n.replace(" (5-fold bagged)", "") for n in top.Name], fontsize=5.2)
    ax.set_xlabel(f"Mean rank, {N['n_settings']} settings (lower is better)", fontsize=5.2); ax.tick_params(axis="x", labelsize=5)
    ax.set_title(f"SIG-Net: mixture of 7 cross-fitted experts\n{N['n_folds']} folds, {N['n_models'] - 3} comparison models", fontsize=5.6, pad=3)
    for i, v in enumerate(top.Overall_Rank): ax.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=4.8)
    ax.set_xlim(0, top.Overall_Rank.max() * 1.18)
    fig.subplots_adjust(left=0.42, right=0.96, top=0.80, bottom=0.20); fig.savefig(path, dpi=300); plt.close(fig)


def fig_weights(W, path):
    dss = [d for d in A.DS_ORDER if d in set(W.Dataset)]; fig, axes = plt.subplots(1, len(dss), figsize=(1.8 * len(dss), 2.4), sharey=True)
    experts = list(W.Expert.unique()); cols = [OK["blue"], OK["sky"], OK["green"], OK["purple"], OK["orange"], OK["yellow"], OK["grey"]]
    for a, ds in zip(axes, dss):
        sub = W[W.Dataset == ds]; bins = sorted(sub.Bin.unique())
        mat = sub.groupby(["Bin", "Expert"]).apply(lambda q: np.average(q.w, weights=q.n)).unstack().reindex(bins)[experts]
        bottom = np.zeros(len(bins))
        for k, e in enumerate(experts):
            a.bar(range(len(bins)), mat[e].values, bottom=bottom, color=cols[k % len(cols)], label=e if ds == dss[0] else None, width=0.7); bottom += mat[e].values
        a.set_xticks(range(len(bins))); a.set_xticklabels(bins, rotation=30, fontsize=6.5); a.set_title(ds, fontsize=8.5)
    axes[0].set_ylabel("Mean expert weight (OOD folds)"); fig.legend(loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.supxlabel("NN Tanimoto similarity to training set", fontsize=7.5, y=0.07); fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    N, *_ = build(); print(json.dumps(N, indent=1, default=float)[:3000])
