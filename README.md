# SIG-Net

**SIG-Net: Similarity-Informed Local-Competence Mixtures of Cross-Fitted Experts for Molecular Property Prediction under Leakage-Audited Out-of-Distribution Splits**
Bhaskar Gurram (Independent Researcher) · gurrambhaskar.ai@gmail.com

Code, per-fold predictions, the pre-specified analysis plan and the manuscript for the paper. Everything (data curation → splits → all models → statistics → figures → compiled manuscript) is reproduced by one notebook.

## Repository layout
| Path | Contents |
|---|---|
| `SIG_Net_Complete_Pipeline.ipynb` | Single notebook: writes and imports every module, runs the full study (resumable) and compiles the manuscript |
| `*.py` | The modules written by the notebook (featurisation, data layer, models, SIG-Net, routers, statistics, figures, manuscript generator) |
| `manuscript/` | `SIG_Net_JCIM_manuscript.pdf/.tex`, `SIG_Net_JCIM_SI.pdf/.tex`, figures, bibliography |
| `ANALYSIS_PLAN.md` | Pre-specified analysis plan, Addendum 1 (BBBP, ChemBERTa-2 baselines) and Addendum 2 (SIG-Net v4: selection rule, design freeze, confirmatory endpoint) |
| `results/folds`, `results/folds_v3`, `results/folds_cb` | Per-fold predictions/metrics of the discovery benchmark (100 folds each: baselines, SIG-Net, ChemBERTa-2) |
| `results/experts` | 9-expert out-of-fold / validation / test prediction matrices for development seed 11 and confirmatory seeds 1–3 (80 folds) |
| `results/final` | All tables (CSV), figures, `numbers.json`, `numbers_v4.json` |
| `results/v4_freeze_hashes.txt`, `results/frozen_code/`, `results/dev_v4_seed11_FROZEN.csv` | Records showing the v4 design was fixed before the confirmatory folds were evaluated (`frozen_code/` files match the recorded SHA-256; the notebook-written copies differ only in line endings and, for `routers.py`, a relative default data path) |
| `data/` | Not in git (135 MB). Download from the release (below) |

## Data
The curated bundle (5 MoleculeNet datasets: FreeSolv, ESOL, Lipophilicity, BACE, BBBP; featurisation; split indices for seeds 42/100/2026/7/99, development seed 11 and confirmatory seeds 1/2/3) is attached to release **v1.0**:
```
mkdir data
gh release download v1.0 -R bhaskargurram-ai/SIG-Net -p phase1_bundle.pt -D data
# or: https://github.com/bhaskargurram-ai/SIG-Net/releases/download/v1.0/phase1_bundle.pt  ->  data/phase1_bundle.pt
```

## Reproduce
```
pip install torch torch_geometric rdkit xgboost scikit-learn pandas scipy matplotlib transformers nbclient ipykernel
jupyter notebook SIG_Net_Complete_Pipeline.ipynb   # run all cells from the repository root
```
With the shipped per-fold results the notebook rebuilds every table, figure and the PDFs in minutes. To retrain from scratch, delete `results/folds*` and `results/experts` (about 25 GPU-hours on RTX 4090-class GPUs; the published runs used several RunPod RTX 4090s in parallel). A LaTeX distribution (pdflatex + bibtex) is needed to compile the manuscript.

## Main results
**Discovery** (5 datasets × 4 split protocols × 5 seeds = 100 folds; 30 comparison models): SIG-Net has the best mean rank (2.60 vs 5.20) and is first in 12/20 settings; it beats every external model in ≥ 68/100 folds (all p ≤ 1.6×10⁻⁵), e.g. +3.8% vs an XGBoost + D-MPNN stack, +5.2% vs D-MPNN+RDKit and +23% vs frozen ChemBERTa-2. Bemis–Murcko scaffold splits leave 54–62% of test molecules with a generic carbon skeleton seen in training.

**Pre-registered confirmation** (fresh seeds 1–3, 57 folds): the refined router (SIG-Net v4) improves on a global stack of the same experts (35/57 folds, mean +1.27%, median +0.20%, p = 0.020), mainly on regression endpoints; the expert mixture beats every single cross-fitted expert by 5–28%; the local out-of-fold residual correction is confirmed (p = 0.016); the novelty-shrinkage term (p = 0.63) and the original local softmax weighting (p = 0.81) are not.

## Citation
Manuscript in preparation for the *Journal of Chemical Information and Modeling*. Please cite the repository until the article is published.
