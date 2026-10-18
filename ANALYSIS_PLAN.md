# Pre-specified analysis plan for SIG-Net v3

Written 2026-10-17 ~23:35 local time, after the v3 RunPod run was launched and **before any v3 test-set result was viewed**.
The v3 design (experts, cross-fitting K=5, grid for k/tau/gamma) was fixed using validation scores on three development
folds (FreeSolv bm_scaffold s42, ESOL bm_scaffold s42, ESOL butina_cluster s42); no test scores were inspected during design.

## Data
- 4 datasets x 4 split protocols x 5 seeds = 80 folds; identical splits/fold preparation for v2 and v3 runs.
- v2 results (`results/folds/`) and v3 results (`results/folds_v3/`) are merged per fold.

## Primary endpoint
- SIG-Net v3 (Proposed) vs. the strongest non-v3 comparator, defined as the non-v3 model with the best average rank
  across the 16 settings. Test: two-sided Wilcoxon signed-rank test on per-fold relative improvement over all 80 folds.
  Also reported: number of settings where v3 has the better 5-seed mean.

## Secondary endpoints
1. v3 vs. "Global stack (same cross-fitted experts)" — isolates the contribution of local competence + novelty shrinkage.
2. v3 vs. each bagged expert and each v2-run baseline (same Wilcoxon procedure, 80 folds).
3. Average rank over all models (Friedman test, Nemenyi CD).
4. Error stratified by nearest-neighbour Tanimoto similarity (OOD folds, regression datasets).

## Reporting rules
- All 16 settings reported, including those where v3 is not best.
- No post-hoc removal of datasets, splits, seeds or baselines.
- FreeSolv generic_scaffold has a single distinct test set across seeds; it is reported but flagged.
- If v3 does not beat the strongest comparator on the primary endpoint, this is stated as the result.

## Addendum (2026-10-18 ~00:55, written before any BBBP or ChemBERTa-2 result was viewed)
Following external review of the draft, two extensions were added. The SIG-Net design (experts, K, grids) is unchanged.
1. **BBBP** (MoleculeNet blood-brain barrier penetration, classification) is added as a fifth dataset, curated and split
   with the identical Phase 1 code (4 protocols x 5 seeds). All models, including SIG-Net, are run on its 20 folds.
2. **Pretrained foundation-model baselines:** frozen ChemBERTa-2 (DeepChem/ChemBERTa-77M-MTR) mean-pooled embeddings with
   (a) a ridge/logistic head and (b) an XGBoost head, both selected on the validation fold, on all 100 folds.
   ChemBERTa-2 is a comparison model only; it is not added as a SIG-Net expert (that would change the design post hoc).
Reporting: the main text reports all 5 datasets (100 folds). The pre-specified 80-fold primary endpoint (4 original
datasets, original comparator set) is reported unchanged alongside it. No other analysis changes.

## Addendum 2 — SIG-Net v4 study (2026-10-18 ~09:40, before any confirmatory result was computed)
**Motivation:** reviewer request to simplify or justify the novelty exponent and to strengthen the local router.
**Splits:** fresh seeds generated with the identical splitting code. Development seed **11** (all 20 settings) may be used
for design decisions, including its test folds. Confirmatory seeds **1, 2, 3** (60 folds) are not inspected until the design
is frozen. FreeSolv generic_scaffold yields the same test set for every seed, so it is **excluded from the confirmatory
endpoint** (reported separately).
**Experts:** 9 cross-fitted experts (the 7 of v3 + ChemBERTa-2/XGBoost + XGBoost/ECFP4); all OOF/val/test matrices saved.
**Candidate routers (development):** v4 (local error-covariance routing with novelty shrinkage toward the global
covariance, composite ECFP/MACCS/physchem kernel, novelty-damped local residual correction), on 7 or 9 experts; v3 router;
global stacks; and v4 ablations.
**Design-selection rule (fixed now, before the remaining development folds arrived):** the v4 variant with the highest mean
per-fold relative improvement over the 7-expert global stack across the 20 seed-11 folds is chosen (ties: more wins).
**Confirmatory primary endpoint:** chosen v4 vs the 7-expert global stack over the 56 confirmatory folds (60 minus FreeSolv
generic), two-sided Wilcoxon signed-rank test on per-fold relative improvement. **Secondary:** v4 vs v3 router; v4 vs
9-expert global stack; v4 ablations (no residual correction, no composite kernel, no novelty shrinkage).
**Reporting:** whatever the confirmatory result is, it is reported; the v3 paper results are unchanged.

### Design freeze (2026-10-18 09:54, before any confirmatory result was computed)
Applying the rule above to the 20 development folds (file results/dev_v4_seed11_FROZEN.csv): **v4 full (9 experts)** has the
highest mean improvement over the 7-expert global stack (+0.68%, median +0.51%, 14/20 wins) and is frozen as **SIG-Net v4**:
9 experts; grid k in {10,25,50}, a in {0.5,0.75,1.0}, gamma in {0,0.5,1,2,4,global}, nu in {0,0.25,0.5,0.75},
gamma_d in {1,2}, rho = 0.05; all selected per fold on the validation fold. Ablations are those listed above.
