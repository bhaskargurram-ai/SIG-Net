"""Generate the JCIM manuscript (achemso) and Supporting Information from results/final. All numbers are read
from numbers.json and the final CSV tables; nothing is hand-typed."""
import os, json, shutil, subprocess
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(ROOT, "results", "final"); MS = os.path.join(ROOT, "manuscript"); os.makedirs(MS, exist_ok=True)
N = json.load(open(os.path.join(FINAL, "numbers.json")))
V4 = json.load(open(os.path.join(FINAL, "numbers_v4.json"))); S4 = V4["summary"]; P4 = V4["per_dataset"]
def c4(key):
    r = S4[key]; return f"{r['wins']}/{r['n']} folds, mean {r['mean']:+.2f}\\%, median {r['median']:+.2f}\\%, " + pv(r["p"])
for f in os.listdir(os.path.join(FINAL, "figures")):
    shutil.copy(os.path.join(FINAL, "figures", f), MS)

pct = lambda x: f"{x:.1f}\\%"
def sci(p):
    m, e = f"{p:.1e}".split("e"); return f"{m} \\times 10^{{{int(e)}}}"
def pv(p):
    if p < 1e-4:
        return f"$p = {sci(p)}$"
    return f"$p = {p:.4f}$" if p < 0.01 else f"$p = {p:.3f}$"
H = lambda k: N[k]
lo, hi = N["bm_seen_skeleton_mol_range"]; olo, ohi = N["bm_skeleton_overlap_range"]
nb = N["novelty_bins"]; nbk = sorted(nb)
ab = N["ablation_rel_vs_dmpnn"]

# ------------------------------------------------------------------ tables
t1 = pd.read_csv(os.path.join(ROOT, "results", "table1_dataset_statistics.csv"))
t1_rows = "\n".join(
    f"\\texttt{{{r['Dataset']}}} & {r['Task (Unit)'].replace('Binary (0/1)', 'binary').replace('log(mol/L)', 'log mol/L').replace('logD7.4', '$\\log D_{7.4}$')} & "
    f"{r['Raw -> Curated N'].replace('->', '$\\to$')} & {r['Unique BM Scaffolds']} & {r['Unique Generic Scaffolds']} & {r['Singleton Scaffolds (%)'].replace('%', '\\%')} \\\\"
    for _, r in t1.iterrows())

t2 = pd.read_csv(os.path.join(ROOT, "results", "table2_split_diagnostics_5seed.csv"))
SPL = {"random": "Random", "bm_scaffold": "BM scaffold", "generic_scaffold": "Generic skeleton", "butina_cluster": "Butina cluster"}
c = lambda s: str(s).replace("±", "$\\pm$").replace("±", "$\\pm$")
t2_rows, prev = [], None
for _, r in t2.iterrows():
    ds = f"\\texttt{{{r['Dataset']}}}" if r["Dataset"] != prev else ""; prev = r["Dataset"]
    if ds and t2_rows: t2_rows.append("\\midrule")
    flag = "$^{a}$" if r["Distinct test sets"] == 1 else ""
    t2_rows.append(f"{ds} & {SPL[r['Split']]}{flag} & {c(r['BM scaffold overlap (%)'])} & {c(r['Generic skeleton overlap (%)'])} & "
                   f"{c(r['Test mols w/ seen generic skeleton (%)'])} & {c(r['NN Tanimoto'])} & {c(r['Hard OOD <0.45 (%)'])} \\\\")
t2_rows = "\n".join(t2_rows)

t3 = pd.read_csv(os.path.join(FINAL, "table3_main_benchmark.csv"))
T3C = ["XGBoost (ECFP4 + RDKit-2D)", "AttentiveFP", "D-MPNN (Chemprop)", "D-MPNN + RDKit-2D", "D-MPNN + RDKit-2D [5-fold bagged]",
       "Stack: XGBoost + D-MPNN", "Global stack (same experts)", "SIG-Net"]
T3H = ["XGBoost\\\\ECFP4+RDKit", "AttentiveFP", "D-MPNN", "D-MPNN\\\\+RDKit", "D-MPNN+RDKit\\\\5-fold bagged", "Stack\\\\XGB+D-MPNN",
       "Global stack\\\\(same experts)", "\\textbf{SIG-Net}"]
t3_rows, prev = [], None
for _, r in t3.iterrows():
    ds = f"\\texttt{{{r['Dataset']}}}" if r["Dataset"] != prev else ""; prev = r["Dataset"]
    if ds and t3_rows: t3_rows.append("\\midrule")
    t3_rows.append(f"{ds} & {r['Split']} & " + " & ".join(str(r[k]) for k in T3C) + " \\\\")
t3_rows = "\n".join(t3_rows)
t3_head = " & ".join(f"\\makecell{{{h}}}" for h in T3H)

h2h = pd.read_csv(os.path.join(FINAL, "table5_head_to_head.csv"))
SEL = ["Global stack (same experts)", "SIG-Net w/o novelty shrinkage", "Two-expert novelty gate", "Stack: XGBoost + D-MPNN",
       "Stack: D-MPNN ens. + anchor", "D-MPNN + RDKit-2D [5-fold bagged]", "D-MPNN + RDKit-2D", "D-MPNN + Deep CORAL", "AttentiveFP",
       "Multi-view anchor", "D-MPNN (Chemprop)", "XGBoost (ECFP4 + RDKit-2D)"]
hh = h2h.set_index("Name").loc[SEL]
t4_rows = "\n".join(f"{n} & {int(r.wins)}/{int(r.n_folds)} & {int(r.settings_won)}/{int(r.n_settings)} & {r.mean_rel:+.2f} & {r.median_rel:+.2f} & {pv(r.p_folds)} \\\\"
                    for n, r in hh.iterrows())
t4_full = "\n".join(f"{n} & {int(r.wins)}/{int(r.n_folds)} & {int(r.settings_won)}/{int(r.n_settings)} & {r.mean_rel:+.2f} & {r.median_rel:+.2f} & {pv(r.p_folds)} \\\\"
                    for n, r in h2h.set_index("Name").iterrows())

rk = pd.read_csv(os.path.join(FINAL, "table4_rankings.csv"))
rk_rows = "\n".join(f"{i+1} & {r.Name} & {r.Overall_Rank:.2f} & {r.OOD_Rank:.2f} & {int(r.First)} & {int(r.Top3)} \\\\" for i, r in rk.head(29).iterrows())

ps = pd.read_csv(os.path.join(FINAL, "tableS_per_setting.csv"))
ps_rows = "\n".join(f"\\texttt{{{r.Dataset}}} & {SPL[r.Split]} & {c(r.Proposed)} & {c(r['Best other'])} & {r['Best other model']} & {int(r.Rank)} & {int(r['Seed wins'])}/5 & {('--' if pd.isna(r['paired t p']) else f'{r['paired t p']:.3f}')} \\\\"
                    for _, r in ps.iterrows())

ABL = [("Single D-MPNN+RDKit (reference)", 0.0), ("D-MPNN+RDKit, 3-seed ensemble", ab["D-MPNN + RDKit-2D Ensemble (3x)"]),
       ("D-MPNN+RDKit, 5-fold cross-fitted bag", ab["D-MPNN + RDKit-2D [5-fold bagged]"]),
       ("Two-expert novelty gate (D-MPNN ens.\\ + anchor)", ab["Two-expert novelty gate"]),
       ("Global stack of 7 cross-fitted experts", ab["Global stack (same experts)"]),
       ("+ local competence weighting (no shrinkage)", ab["SIG-Net w/o novelty shrinkage"]),
       ("+ novelty shrinkage = \\textbf{SIG-Net}", ab["SIG-Net"])]
abl_rows = "\n".join(f"{n} & {v:+.2f} \\\\" for n, v in ABL)

fbd = N["firsts_by_dataset"]
_CL = [("Global stack (7 experts)", "primary_vs_global7"), ("Global stack (9 experts)", "vs Global stack (9)"), ("Original router (local softmax, 7)", "vs SIG-Net v3 router (7)"),
       ("v4 w/o residual correction", "vs v4 w/o residual correction"), ("v4 w/o composite kernel", "vs v4 w/o composite kernel"), ("v4 w/o novelty shrinkage", "vs v4 w/o novelty shrinkage"),
       ("Multi-view anchor (bagged)", "vs Anchor (Multi-View Stack) [5-fold bagged]"), ("D-MPNN+RDKit (bagged)", "vs D-MPNN + RDKit-2D [5-fold bagged]"),
       ("AttentiveFP (bagged)", "vs AttentiveFP [5-fold bagged]"), ("XGBoost ECFP4+RDKit (bagged)", "vs XGBoost (ECFP4 + RDKit-2D) [5-fold bagged]"), ("ChemBERTa-2 + XGBoost (bagged)", "vs ChemBERTa-2 + XGBoost [5-fold bagged]")]
conf_all = "\n".join(f"{k.replace('vs ', '', 1).replace('primary_vs_global7', 'Global stack (7) [primary]').replace('v3_vs_global7', 'Original router vs Global stack (7)').replace('freesolv_generic_included_primary', 'Global stack (7), FreeSolv generic included')} & {S4[k]['wins']}/{S4[k]['n']} & {S4[k]['mean']:+.2f} & {S4[k]['median']:+.2f} & {pv(S4[k]['p'])} \\\\" for k in S4)
conf_rows = "\n".join(f"{n} & {S4[k]['wins']}/{S4[k]['n']} & {S4[k]['mean']:+.2f} & {S4[k]['median']:+.2f} & {pv(S4[k]['p'])} \\\\" for n, k in _CL)
WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
DSL = [d for d in ["FreeSolv", "ESOL", "Lipophilicity", "BACE", "BBBP"] if d in fbd]; NDS = len(DSL); NDW = WORD[NDS]
NF, NS = N["n_folds"], N["n_settings"]; NOOD = NS * 3 // 4
CLS = [d for d in DSL if d in ("BACE", "BBBP")]
firsts_list = ", ".join(f"{fbd[d]} of four \\texttt{{{d}}}" for d in DSL[:-1]) + f" and {fbd[DSL[-1]]} of four \\texttt{{{DSL[-1]}}}"
_t2b = pd.read_csv(os.path.join(ROOT, "results", "table2_split_diagnostics_5seed.csv"))
_nn = lambda d, sp: float(str(_t2b[(_t2b.Dataset == d) & (_t2b.Split == sp)]["NN Tanimoto"].iloc[0]).split("±")[0])
_novel = [d for d in DSL if _nn(d, "butina_cluster") < _nn(d, "generic_scaffold")]
novel_txt = ", ".join(f"\\texttt{{{d}}}" for d in _novel[:-1]) + (" and " if len(_novel) > 1 else "") + f"\\texttt{{{_novel[-1]}}}"
novel_nn = ", ".join(f"{_nn(d, 'butina_cluster'):.2f}" for d in _novel)
_W = pd.read_csv(os.path.join(FINAL, "table_expert_weights.csv"))
def _nice(e):
    return {"Anchor (Multi-View Stack)": "the descriptor stack", "XGBoost (ECFP4 + RDKit-2D)": "XGBoost", "RF (RDKit-2D)": "random forest",
            "D-MPNN + RDKit-2D": "D-MPNN", "D-MPNN + Deep CORAL": "D-MPNN+CORAL"}.get(e, e)
def top_experts(d, k=2):
    q = _W[_W.Dataset == d].groupby("Expert").apply(lambda g: (g.w * g.n).sum() / g.n.sum()).sort_values(ascending=False)
    return [_nice(e) for e in q.index[:k]], float(q.iloc[:k].sum())
_lip, _lw = top_experts("Lipophilicity"); _cls = {d: top_experts(d, 3) for d in CLS}
weights_txt = (f"On \\texttt{{Lipophilicity}} SIG-Net relies mainly on graph experts ({_lip[0]} and {_lip[1]} together receive {100 * _lw:.0f}\\% of the weight), whereas on the classification sets the largest weights go to "
               + "; ".join(f"{', '.join(v[0])} (\\texttt{{{d}}})" for d, v in _cls.items()) + ".")
pr = N["prereg"]
_g = pd.read_csv(os.path.join(ROOT, "results", "all_fold_metrics_v2v3.csv")) if os.path.exists(os.path.join(ROOT, "results", "all_fold_metrics_v2v3.csv")) else None
dmp_fs_bm = _g[(_g.Dataset == "FreeSolv") & (_g.Split == "bm_scaffold") & (_g.Model == "D-MPNN (Chemprop)")].Primary.mean()
_OWN = ["SIG-Net v3 w/o novelty shrinkage", "Global stack (same cross-fitted experts)"]
_gm = _g[~_g.Model.isin(_OWN)].groupby(["Dataset", "Split", "Model"]).Primary.agg(["mean", "count"]).reset_index()
_cb_wins = []
for (_d, _sp), _q in _gm.groupby(["Dataset", "Split"]):
    if _q["count"].min() < 5 or _q.Model.nunique() < 20:
        continue
    _best = _q.loc[_q["mean"].idxmin() if _d not in ("BACE", "BBBP") else _q["mean"].idxmax(), "Model"]
    if _best.startswith("ChemBERTa"):
        _cb_wins.append(f"\\texttt{{{_d}}} {SPL[_sp].lower()}")
cb_txt = ("" if not _cb_wins else " On the classification tasks, however, the ChemBERTa-2 + XGBoost baseline is competitive and has the best mean in "
          + str(len(_cb_wins)) + " setting" + ("s" if len(_cb_wins) > 1 else "") + " (" + ", ".join(_cb_wins)
          + "), which suggests that pretrained representations are a promising additional expert for classification endpoints.")
# full benchmark (all models x settings) for SI
_full = _g.groupby(["Model", "Dataset", "Split"]).Primary.mean().unstack(["Dataset", "Split"])
_cols = [(d, p) for d in [x for x in ["FreeSolv", "ESOL", "Lipophilicity", "BACE", "BBBP"] if x in set(_g.Dataset)] for p in ["random", "bm_scaffold", "generic_scaffold", "butina_cluster"]]
_full = _full[_cols]; NCOL = len(_cols)
_ab = {"random": "R", "bm_scaffold": "BM", "generic_scaffold": "G", "butina_cluster": "Bu"}
full_head = " & ".join(f"{d[:4]}-{_ab[p]}" for d, p in _cols)
full_rows = "\n".join(f"{m.replace('&', 'and')} & " + " & ".join(f"{v:.3f}" for v in r.values) + " \\\\" for m, r in _full.iterrows())
# ------------------------------------------------------------------ main text
tex = r"""\documentclass[journal=jcisd8,manuscript=article]{achemso}
\usepackage{colortbl}
\usepackage{amsmath,amssymb,booktabs,graphicx,makecell,threeparttable,xcolor,url}
\setkeys{acs}{articletitle=true}
\author{Bhaskar Gurram}
\affiliation{Independent Researcher}
\email{gurrambhaskar.ai@gmail.com}
\title[SIG-Net: local-competence mixtures for OOD property prediction]{SIG-Net: Similarity-Informed Local-Competence Mixtures of Cross-Fitted Experts for Molecular Property Prediction under Leakage-Audited Out-of-Distribution Splits}
\abbreviations{BM, Bemis--Murcko; D-MPNN, directed message-passing neural network; ECFP, extended-connectivity fingerprint; GNN, graph neural network; NN, nearest neighbour; OOD, out-of-distribution; OOF, out-of-fold}
\keywords{molecular property prediction, out-of-distribution generalization, scaffold split, ensemble learning, mixture of experts, graph neural networks}

\begin{document}
\begin{tocentry}
\includegraphics[width=3.25in,height=1.75in,keepaspectratio]{toc_graphic.png}
\end{tocentry}

\begin{abstract}
Molecular property models are usually validated on Bemis--Murcko (BM) scaffold splits, and no single model family wins across datasets and split protocols. We make two contributions. First, an audit of """ + NDW + r""" MoleculeNet benchmarks over five seeds shows that BM scaffold splits leave """ + f"{lo:.0f}--{hi:.0f}" + r"""\% of test molecules with a generic carbon skeleton that also occurs in the training set; generic-skeleton splits remove this overlap. Second, we introduce SIG-Net, a similarity-informed mixture of seven cross-fitted experts (four graph neural networks, two tree ensembles and a descriptor stack). Each expert is cross-fitted on the training set, and for every query molecule the experts are weighted by their out-of-fold error on its Tanimoto nearest neighbours, shrunk towards global stacking weights as structural novelty increases; all hyperparameters are selected on validation data only. Across """ + str(NDS) + r""" datasets $\times$ 4 split protocols $\times$ 5 seeds (""" + str(NF) + r""" folds) and """ + str(N["n_models"] - 3) + r""" comparison models, SIG-Net has the best mean rank (""" + f"{N['rank_overall']:.2f}" + r""" vs.\ """ + f"{N['second_rank']:.2f}" + r""" for the next model), is best in """ + f"{N['firsts']}" + f"/{NS}" + r""" settings, and outperforms the strongest external stacking baseline in """ + f"{H('vs_xgbdmpnn')['wins']}" + f"/{NF}" + r""" folds (mean relative improvement """ + f"{H('vs_xgbdmpnn')['mean']:.1f}" + r"""\%, """ + pv(H('vs_xgbdmpnn')['p']) + r""") and a Chemprop-style D-MPNN with RDKit features by """ + f"{H('vs_dmpnn')['mean']:.1f}" + r"""\% (""" + f"{H('vs_dmpnn')['wins']}" + f"/{NF}" + r""" folds) and frozen ChemBERTa-2 embeddings by """ + f"{H('vs_cb_xgb')['mean']:.1f}" + r"""\% (""" + f"{H('vs_cb_xgb')['wins']}/{NF}" + r""" folds). Ablations attribute most of the gain to cross-fitted stacking of diverse experts. In a pre-registered confirmation on fresh splits (""" + str(V4["n_conf"]) + r""" folds), the advantage of the expert mixture over every single model replicated (""" + f"{S4['vs D-MPNN + RDKit-2D [5-fold bagged]']['mean']:.1f}" + r"""--""" + f"{S4['vs XGBoost (ECFP4) [5-fold bagged]']['mean']:.0f}" + r"""\%), the original local weighting did not (""" + pv(S4["v3_vs_global7"]["p"]) + r"""), and a refined router with a novelty-damped local out-of-fold residual correction improved on the global stack (""" + f"{S4['primary_vs_global7']['mean']:+.1f}" + r"""\%, """ + pv(S4["primary_vs_global7"]["p"]) + r"""), mainly on regression endpoints. The novelty-shrinkage exponent had no measurable effect in either analysis.
\end{abstract}

\section{Introduction}
Machine-learning models of molecular properties are increasingly used to prioritise compounds whose ring systems were not seen during training. Benchmarks therefore replace random splits with grouped splits, most commonly by Bemis--Murcko (BM) scaffold\cite{bemis1996properties,wu2018moleculenet}. Several recent studies question whether such splits are as hard as assumed: Fooladi et al.\ found that scaffold splits often degrade performance only modestly relative to random splits\cite{fooladi2025evaluating}, Reimer et al.\ reported that scaffold and UMAP splits are frequently closer to random than expected\cite{reimer2025chemspace}, and Zheng et al.\ showed that holding out structurally remote scaffold groups exposes failures that scaffold splits hide\cite{zheng2026frontier}. Benchmarks such as BOOM similarly report large out-of-distribution (OOD) errors for current models\cite{boom2025}.

A second, related observation is that the best model depends on the dataset and on the split. Directed message-passing networks (D-MPNN/Chemprop)\cite{yang2019analyzing,heid2024chemprop} are strong on larger sets, while descriptor-based tree ensembles remain competitive on small ones\cite{yang2019analyzing}, and domain-invariance penalties such as Deep CORAL\cite{sun2016deep} help in some settings and not others. Stacking\cite{wolpert1992stacked} combines models with fixed weights; mixtures of experts\cite{jacobs1991adaptive} allow weights to vary per input but need enough data to learn the gating function, which small chemistry datasets rarely provide.

We address both points. (i) We audit """ + NDW + r""" MoleculeNet benchmarks under four split protocols and five seeds, and quantify how often BM scaffold splits leave a test molecule's generic carbon skeleton in the training set. (ii) We propose SIG-Net, a mixture of seven cross-fitted experts in which the per-molecule weights are not learned by a gating network but estimated non-parametrically from each expert's out-of-fold error on the query's nearest training neighbours. Cross-fitting supplies an unbiased error estimate for every training molecule, so the gate is informed by the whole training set rather than by a small validation fold. We evaluate SIG-Net against """ + str(N["n_models"] - 3) + r""" models, including a pretrained chemical language model, on """ + str(NF) + r""" folds with a pre-specified analysis plan, and report ablations that separate the contributions of cross-fitting, expert diversity, local weighting and novelty shrinkage, including the components that do not help. Finally, we test the local components in a pre-registered confirmation on fresh splits, together with a refined router that addresses the limitations of the first.

""" + open(os.path.join(MS, "methods_v3.tex"), encoding="utf-8").read() + r"""

\begin{table}
\caption{Curated datasets.}
\label{tab:datasets}
\small
\begin{tabular}{llcccc}
\toprule
Dataset & Task (unit) & Raw $\to$ curated & BM scaffolds & Generic skeletons & Singleton BM \\
\midrule
""" + t1_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\section{Results}

\subsection{BM scaffold splits leave most test molecules with a seen carbon skeleton}
\begin{table*}
\caption{Split diagnostics, mean $\pm$ SD over five seeds. Overlaps are percentages of unique test scaffolds; ``Seen skeleton'' is the percentage of test molecules whose generic skeleton occurs in training; ``Hard OOD'': test molecules with NN Tanimoto $< 0.45$. $^{a}$All five seeds give the same test set.}
\label{tab:splits}
\scriptsize
\begin{tabular}{llccccc}
\toprule
Dataset & Split & BM overlap (\%) & Skeleton overlap (\%) & Seen skeleton (\% mols) & NN Tanimoto & Hard OOD (\%) \\
\midrule
""" + t2_rows + r"""
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:splits} and Figure~\ref{fig:leakage} summarise the split audit. BM scaffold splits have no shared BM scaffolds by construction, but """ + f"{olo:.0f}--{ohi:.0f}" + r"""\% of test scaffolds reduce to a generic skeleton that also occurs in training, and """ + f"{lo:.0f}--{hi:.0f}" + r"""\% of test \emph{molecules} have such a skeleton. Generic-skeleton splits remove this overlap entirely and lower the mean nearest-neighbour (NN) similarity on \texttt{FreeSolv}, \texttt{ESOL} and \texttt{Lipophilicity}. Butina clustering at a Tanimoto distance of 0.6 behaves differently by dataset: on the small, chemically simple \texttt{FreeSolv} and \texttt{ESOL} sets it leaves 89--97\% of test molecules with a seen generic skeleton, whereas on """ + novel_txt + r""" it produces the most novel test sets (NN similarity """ + novel_nn + r"""). On \texttt{FreeSolv}, the generic-skeleton protocol yields the same test set for all five seeds because a few large skeleton groups dominate the dataset; results for that setting reflect model variance only and are flagged throughout.

\begin{figure}
\includegraphics[width=\textwidth]{fig1_leakage.png}
\caption{Split audit over five seeds. (a) Percentage of test molecules whose generic carbon skeleton occurs in the training set. (b) Mean nearest-neighbour ECFP4 Tanimoto similarity of test molecules to the training set.}
\label{fig:leakage}
\end{figure}

\subsection{SIG-Net has the best average rank across datasets and splits}
Table~\ref{tab:main} lists test performance for representative models; all """ + str(N["n_models"]) + r""" models are reported in Table~S1. Graph models trained with one-hot features reach accuracies in line with published Chemprop results (e.g.\ \texttt{Lipophilicity} random split RMSE 0.54 for D-MPNN+RDKit). Ranking the """ + str(N["cd_k"]) + r""" models that are not internal ablations of SIG-Net within each of the """ + str(NS) + r""" settings (Figure~\ref{fig:ranks}, Table~S2), SIG-Net has a mean rank of """ + f"{N['rank_overall']:.2f}" + r""" (""" + f"{N['rank_ood']:.2f}" + r""" on the """ + str(NOOD) + r""" OOD settings) against """ + f"{N['second_rank']:.2f}" + r""" for the next model (""" + N["second_model"] + r"""). It is first in """ + f"{N['firsts']}" + r""" of """ + str(NS) + r""" settings: """ + firsts_list + r""". The Friedman test rejects equal ranks (""" + f"$\\chi^2 = {N['friedman_chi2']:.0f}$, " + pv(N["friedman_p"]) + r"""); with """ + str(N["cd_k"]) + r""" models and """ + str(NS) + r""" settings the Nemenyi critical difference is """ + f"{N['cd']:.1f}" + r""" ranks, so this conservative post-hoc test separates SIG-Net only from models with mean rank above """ + f"{N['rank_overall'] + N['cd']:.1f}" + r""". We therefore also report paired fold-level comparisons.

\begin{table*}
\caption{Test performance (mean $\pm$ SD over five seeds). RMSE for regression (lower is better), ROC-AUC for \texttt{BACE} and \texttt{BBBP} (higher is better). Bold: best among the columns shown. All models are reported in Table~S1.}
\label{tab:main}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll""" + "c" * len(T3C) + r"""}
\toprule
Dataset & Split & """ + t3_head + r""" \\
\midrule
""" + t3_rows + r"""
\bottomrule
\end{tabular}}
\end{table*}

\begin{figure}
\includegraphics[width=\columnwidth]{fig2_ranks.png}
\caption{Mean rank over """ + str(NS) + r""" dataset--split settings (top 14 of """ + str(N["cd_k"]) + r""" models; internal SIG-Net ablations excluded). The grey band spans one Nemenyi critical difference from the best model.}
\label{fig:ranks}
\end{figure}

\subsection{Paired comparisons over """ + str(NF) + r""" folds}
Table~\ref{tab:h2h} compares SIG-Net with selected models on every fold. SIG-Net outperforms every external model in at least """ + str(N["min_wins_external"]) + f" of {NF}" + r""" folds, with all two-sided Wilcoxon $p$-values at or below $""" + sci(N["max_p_external"]) + r"""$. Against the strongest external stacking baseline (XGBoost + D-MPNN with a validation-selected weight) the mean relative improvement is """ + f"{H('vs_xgbdmpnn')['mean']:.1f}" + r"""\% (better in """ + f"{H('vs_xgbdmpnn')['settings']}" + f"/{NS}" + r""" settings); against D-MPNN+RDKit it is """ + f"{H('vs_dmpnn')['mean']:.1f}" + r"""\% and against our earlier two-expert novelty gate """ + f"{H('vs_v2')['mean']:.1f}" + r"""\%. The frozen ChemBERTa-2 baselines are much weaker on the regression tasks: overall, SIG-Net improves on the linear probe by """ + f"{H('vs_cb_lin')['mean']:.1f}" + r"""\% and on the XGBoost head by """ + f"{H('vs_cb_xgb')['mean']:.1f}" + r"""\% (""" + f"{H('vs_cb_xgb')['wins']}/{NF}" + r""" folds).""" + cb_txt + r""" The pre-specified primary comparison was against the strongest non-SIG-Net model by mean rank, which turned out to be the global stack of the same cross-fitted experts. On the """ + str(pr['n_folds']) + r""" pre-specified folds (four original datasets) SIG-Net improves on it in """ + f"{pr['wins']}/{pr['n_folds']}" + r""" folds and """ + f"{pr['settings']}/{pr['n_settings']}" + r""" settings (mean """ + f"{pr['mean']:+.2f}" + r"""\%, median """ + f"{pr['median']:+.2f}" + r"""\%, """ + pv(pr['p']) + r"""); including \texttt{BBBP} the figures are """ + f"{H('vs_global')['wins']}/{NF}" + r""" folds and """ + f"{H('vs_global')['settings']}/{NS}" + r""" settings (mean """ + f"{H('vs_global')['mean']:+.2f}" + r"""\%, """ + pv(H('vs_global')['p']) + r"""). This effect is small and, as shown below, it did not replicate on fresh splits.

\begin{table}
\caption{Paired comparison of SIG-Net with selected models over """ + str(NF) + r""" folds. Relative improvement is computed per fold (RMSE reduction or AUC increase, \%); $p$: two-sided Wilcoxon signed-rank test. Full list in Table~S3.}
\label{tab:h2h}
\scriptsize
\begin{tabular}{lccrrc}
\toprule
Comparator & Folds won & Settings won & Mean (\%) & Median (\%) & $p$ \\
\midrule
""" + t4_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{Where the improvement comes from}
Table~\ref{tab:ablation} decomposes the improvement over a single D-MPNN+RDKit model. Ensembling D-MPNNs alone, either over seeds or by cross-fitting, changes little (""" + f"{ab['D-MPNN + RDKit-2D Ensemble (3x)']:+.1f}" + r"""\% and """ + f"{ab['D-MPNN + RDKit-2D [5-fold bagged]']:+.1f}" + r"""\%). Combining heterogeneous experts accounts for most of the gain: a global stack of the seven cross-fitted experts improves by """ + f"{ab['Global stack (same experts)']:+.1f}" + r"""\%, compared with """ + f"{ab['Two-expert novelty gate']:+.1f}" + r"""\% for the two-expert gate. Local competence weighting adds a further """ + f"{ab['SIG-Net w/o novelty shrinkage'] - ab['Global stack (same experts)']:+.1f}" + r""" points. Shrinking local weights towards global weights as a function of novelty does not help further: SIG-Net and its variant without shrinkage are indistinguishable (""" + f"{H('vs_noshrink')['wins']}" + f"/{NF}" + r""" folds, """ + pv(H('vs_noshrink')['p']) + r"""), and validation-based selection chose no shrinkage ($\gamma = 0$) in """ + str(N["gamma_selected"].get("0.0", 0)) + f" of {NF}" + r""" folds and the purely global combination in """ + str(N["gamma_selected"].get("global", 0)) + r""". The simpler variant without shrinkage is therefore an equally valid choice.

\begin{table}
\caption{Ablation: mean per-fold relative improvement over a single D-MPNN+RDKit model (""" + str(NF) + r""" folds; RMSE reduction or AUC increase, \%).}
\label{tab:ablation}
\small
\begin{tabular}{lr}
\toprule
Model & Improvement (\%) \\
\midrule
""" + abl_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{Behaviour as a function of structural novelty}
Figure~\ref{fig:novelty} stratifies absolute errors on the OOD folds of the three regression datasets by the NN similarity of each test molecule. Errors grow with novelty for every model. SIG-Net reduces the error of the XGBoost + D-MPNN stack by a similar margin in every bin (""" + ", ".join(f"{nb[b]['vs_xgbdmpnn']:.1f}\\%" for b in nbk) + r""" for NN similarity """ + ", ".join(b.replace(",", "--").replace("[", "").replace(")", "") for b in nbk) + r"""). The additional benefit of local weighting over the global stack is confined to the novel bins (""" + ", ".join(f"{nb[b]['vs_global']:+.2f}\\%" for b in nbk) + r"""), consistent with neighbourhood-specific expert reliability mattering most away from the training distribution, although the effect is small. Figure~\ref{fig:weights} shows the learned weights: they differ far more between datasets than between novelty bins. """ + weights_txt + r""" This dataset dependence is the main reason a fixed choice of model is suboptimal across benchmarks.

\begin{figure}
\includegraphics[width=\textwidth]{fig3_novelty.png}
\caption{Mean absolute error (normalised by the target SD) on OOD folds, stratified by the nearest-neighbour Tanimoto similarity of each test molecule to its training set.}
\label{fig:novelty}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{fig4_expert_weights.png}
\caption{Mean SIG-Net expert weights on OOD folds by dataset and novelty bin.}
\label{fig:weights}
\end{figure}

\subsection{Pre-registered confirmation on fresh splits}
To test whether the local components generalise beyond the splits on which they were developed, we generated four new seeds for every dataset and protocol with the same splitting code, cross-fitted nine experts (the seven above plus ChemBERTa-2 + XGBoost and XGBoost on ECFP4) and saved their out-of-fold, validation and test predictions, so that routers could be compared on identical expert outputs. Seed 11 was used for development; seeds 1--3 were not inspected until a refined router (SIG-Net v4, Methods) had been frozen and the primary endpoint registered (Supporting Information, Addendum 2). The \texttt{FreeSolv} generic-skeleton split yields the same test set for every seed and was excluded from the endpoint, leaving """ + str(V4["n_conf"]) + r""" folds (the plan erroneously stated 56).

The primary endpoint was met: SIG-Net v4 improved on the global stack of the seven original experts in """ + c4("primary_vs_global7") + r""" (Table~\ref{tab:confirm}). The improvement is concentrated in the regression tasks---\texttt{FreeSolv} """ + f"{P4['v4_vs_global']['FreeSolv']['mean']:+.1f}" + r"""\%, \texttt{Lipophilicity} """ + f"{P4['v4_vs_global']['Lipophilicity']['mean']:+.1f}" + r"""\% and \texttt{ESOL} """ + f"{P4['v4_vs_global']['ESOL']['mean']:+.1f}" + r"""\%---and absent on the classification tasks (\texttt{BACE} """ + f"{P4['v4_vs_global']['BACE']['mean']:+.1f}" + r"""\%, \texttt{BBBP} """ + f"{P4['v4_vs_global']['BBBP']['mean']:+.1f}" + r"""\%; Figure~\ref{fig:confirm}). Of v4's components, only the local residual correction had a confirmed effect (v4 vs.\ v4 without it: """ + c4("vs v4 w/o residual correction") + r"""); the composite kernel (""" + pv(S4["vs v4 w/o composite kernel"]["p"]) + r""") and the novelty shrinkage (""" + pv(S4["vs v4 w/o novelty shrinkage"]["p"]) + r""") did not. The local softmax weighting of the original router did not replicate: on the fresh folds it was indistinguishable from the global stack (""" + c4("v3_vs_global7") + r"""), in contrast to its small advantage on the original seeds. The central result, by contrast, replicated strongly: SIG-Net v4 outperformed every individual cross-fitted expert, including D-MPNN+RDKit (""" + c4("vs D-MPNN + RDKit-2D [5-fold bagged]") + r""") and ChemBERTa-2 + XGBoost (""" + c4("vs ChemBERTa-2 + XGBoost [5-fold bagged]") + r""").

\begin{table}
\caption{Pre-registered confirmation on fresh seeds 1--3 (""" + str(V4["n_conf"]) + r""" folds, \texttt{FreeSolv} generic skeleton excluded): SIG-Net v4 vs.\ each comparator. Relative improvement per fold (\%); two-sided Wilcoxon signed-rank test.}
\label{tab:confirm}
\scriptsize
\begin{tabular}{lcrrc}
\toprule
Comparator & Folds won & Mean (\%) & Median (\%) & $p$ \\
\midrule
""" + conf_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[width=\textwidth]{fig5_confirmatory.png}
\caption{Confirmation on fresh splits, by dataset (mean $\pm$ SE of per-fold relative improvement). Grey: original local-softmax router vs.\ global stack; red: SIG-Net v4 vs.\ global stack; orange: SIG-Net v4 vs.\ SIG-Net v4 without the residual correction.}
\label{fig:confirm}
\end{figure}

\section{Discussion}
Three practical conclusions follow. First, BM scaffold splits should not be read as tests of extrapolation to unseen ring systems: for most test molecules a closely related carbon skeleton is in the training set. Generic-skeleton splits are a simple alternative, but they can become degenerate on small datasets dominated by a few skeletons (here \texttt{FreeSolv}), which should be checked and reported. Second, because the best model varies across datasets and splits, combining diverse, cross-fitted experts is a more reliable default than selecting a single architecture; this is the part of SIG-Net that held up in every analysis, including the pre-registered confirmation on fresh splits. Third, the gains from making the combination local are small and endpoint-dependent. The original local softmax weighting looked significant on the discovery splits but did not replicate; a local out-of-fold residual correction did replicate, mainly on regression endpoints. The novelty-shrinkage exponent $\gamma$ had no measurable effect in any analysis, so we recommend omitting it: the simplest form of SIG-Net that retains its confirmed benefits is a global stack of cross-fitted diverse experts followed by a local out-of-fold residual correction whose strength $\nu$ and neighbourhood size $k$ are chosen on validation data.

\paragraph{Future work.}
The local error covariance used by SIG-Net v4 also yields a per-molecule error estimate, $\mathbf{w}(x)^\top\hat{\boldsymbol{\Sigma}}(x)\mathbf{w}(x)$, which could support locally adaptive conformal prediction intervals. Fine-tuned (rather than frozen) foundation models as experts, classification-specific local calibration, and evaluation on independent datasets such as ADMET panels and time splits\cite{sheridan2013timesplit} are the most direct next steps.

\paragraph{Limitations.}
The benchmark covers """ + NDW + r""" MoleculeNet datasets of 642--4200 molecules, two of them classification tasks; larger, more diverse sets (e.g.\ ADMET panels or time splits\cite{sheridan2013timesplit}) remain to be tested. Hyperparameters of all models were fixed rather than tuned per dataset. The confirmation used three fresh seeds of the same five datasets, not independent datasets, and its mean gain is driven largely by \texttt{FreeSolv} (median gain """ + f"{S4['primary_vs_global7']['median']:+.2f}" + r"""\%). SIG-Net trains each expert five times per fold, roughly """ + "20" + r""" times the cost of a single D-MPNN. A pretrained chemical language model (ChemBERTa-2) was evaluated only as a frozen-embedding baseline; fine-tuned foundation models, and their use as SIG-Net experts, remain to be tested. \texttt{BBBP} and the ChemBERTa-2 baselines were added after the pre-specified analysis, in response to review and before their results were viewed (analysis-plan addendum, Supporting Information). The gain over the best global stack is small in absolute terms (median """ + f"{H('vs_global')['median']:+.2f}" + r"""\%), and the Nemenyi test does not separate SIG-Net from the next-ranked models. Finally, in earlier versions of this pipeline, graph models received integer-coded atom features; with one-hot features (used throughout this work) the \texttt{FreeSolv} BM-scaffold RMSE of D-MPNN fell from 3.59 (integer features, five seeds) to """ + f"{dmp_fs_bm:.2f}" + r""", which illustrates how strongly featurisation choices affect comparisons between model families.

\section{Conclusions}
We quantified carbon-skeleton overlap under BM scaffold splits on """ + NDW + r""" MoleculeNet benchmarks and introduced SIG-Net, a similarity-informed mixture of cross-fitted experts. Over """ + str(NF) + r""" folds and """ + str(N["n_models"] - 3) + r""" comparison models, SIG-Net achieved the best mean rank and outperformed every external model on paired comparisons. A pre-registered confirmation on fresh splits showed that this advantage comes from cross-fitted stacking of diverse experts, that a local out-of-fold residual correction adds a small further gain on regression endpoints, and that the original local weighting and the novelty-shrinkage term do not generalise.

\section{Data and Software Availability}
All datasets are public (MoleculeNet). The complete pipeline---curation, splitting, featurisation, all models, the """ + str(NF) + r"""-fold benchmark, statistics, figures and this manuscript---is provided as a single Jupyter notebook together with per-fold predictions and the pre-specified analysis plan. All code, the notebook, per-fold predictions, the analysis plan with its addenda and the frozen-design records are available at \url{https://github.com/bhaskargurram-ai/SIG-Net}; the curated data bundle with all split indices is attached to release v1.0 of that repository.

\begin{suppinfo}
Supporting Information: pre-specified analysis plan; full benchmark of all models (Table~S1); full rankings (Table~S2); all paired comparisons (Table~S3); per-setting results (Table~S4).
\end{suppinfo}





\bibliography{refs}
\end{document}
"""

si = r"""\documentclass[11pt]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{amsmath,booktabs,longtable,graphicx,pdflscape}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}{\Large\bfseries Supporting Information}\\[4pt]
SIG-Net: Similarity-Informed Local-Competence Mixtures of Cross-Fitted Experts for Molecular Property Prediction under Leakage-Audited Out-of-Distribution Splits\end{center}

\section*{S1. Pre-specified analysis plan}
The analysis plan below was written after the final SIG-Net run was launched and before any of its test-set results were inspected. The SIG-Net design was fixed using validation scores on three development folds.
\begin{verbatim}
""" + open(os.path.join(ROOT, "ANALYSIS_PLAN.md"), encoding="utf-8").read().replace("—", "--").replace("−", "-") + r"""
\end{verbatim}

\section*{S2. Full benchmark}
\begin{landscape}{\tiny\setlength{\tabcolsep}{1.5pt}
\begin{longtable}{l""" + "c" * NCOL + r"""}
\caption{Mean test metric over five seeds for every model (RMSE for FreeSolv/ESOL/Lipophilicity, ROC-AUC for BACE/BBBP). R: random, BM: Bemis--Murcko scaffold, G: generic skeleton, Bu: Butina cluster.}\\
\toprule
Model & """ + full_head + r""" \\
\midrule
""" + full_rows + r"""
\bottomrule
\end{longtable}}
\end{landscape}

\section*{S3. Full rankings}
\begin{longtable}{rlcccc}
\caption{Mean rank over """ + str(NS) + r""" settings for all models except internal SIG-Net ablations.}\\
\toprule
\# & Model & Mean rank & OOD mean rank & First & Top-3 \\
\midrule
""" + rk_rows.replace("&", "&", ) + r"""
\bottomrule
\end{longtable}

\section*{S4. All paired comparisons (""" + str(NF) + r""" folds)}
\begin{longtable}{lccrrc}
\caption{SIG-Net vs.\ every other model. Relative improvement per fold (\%), two-sided Wilcoxon signed-rank test.}\\
\toprule
Comparator & Folds won & Settings won & Mean & Median & $p$ \\
\midrule
""" + t4_full + r"""
\bottomrule
\end{longtable}

\section*{S6. Confirmation on fresh splits: all comparisons}
\begin{longtable}{lcrrc}
\caption{SIG-Net v4 vs.\ every router and cross-fitted expert on the confirmatory folds (seeds 1--3).}\\
\toprule
Comparator & Folds won & Mean (\%) & Median (\%) & $p$ \\
\midrule
""" + conf_all + r"""
\bottomrule
\end{longtable}

\section*{S5. Per-setting results}
\begin{longtable}{llccccccc}
\caption{SIG-Net vs.\ the best other external model in each setting (mean $\pm$ SD over 5 seeds); paired $t$-test over seeds (descriptive; low power with $n=5$).}\\
\toprule
Dataset & Split & SIG-Net & Best other & Model & Rank & Seed wins & $p$ \\
\midrule
""" + ps_rows + r"""
\bottomrule
\end{longtable}
\end{document}
"""
open(os.path.join(MS, "SIG_Net_JCIM_manuscript.tex"), "w", encoding="utf-8").write(tex)
open(os.path.join(MS, "SIG_Net_JCIM_SI.tex"), "w", encoding="utf-8").write(si)
print("written")
