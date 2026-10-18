"""Recompute Table 1 (dataset statistics) and Table 2 (split leakage / shift diagnostics) over all 5 seeds
directly from the stored split indices, including how distinct the 5 seed-splits actually are."""
import os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from rdkit import DataStructs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_data import load_bundle, dataset_fps, SEEDS, PROTOCOLS, DATASET_CFG


def split_diagnostics():
    b = load_bundle(); rows = []
    for ds in DATASET_CFG:
        data = b["datasets"][ds]; fps = dataset_fps(ds)
        y = np.array([d.y.item() for d in data]); bm = [d.bm_scaffold for d in data]; gen = [d.gen_scaffold for d in data]
        Xd = np.clip(np.nan_to_num(RobustScaler().fit_transform(np.vstack([d.phys_raw.numpy() for d in data]))), -10, 10)
        for proto in PROTOCOLS:
            acc = {k: [] for k in ["bm_ov", "gen_ov", "bm_mol", "gen_mol", "sim", "hard", "w1", "shift", "n_te"]}
            test_sets = []
            for s in SEEDS:
                sp = b["splits"][ds][proto][s]; tr, te = sp["train"], sp["test"]; test_sets.append(frozenset(te))
                tr_bm, tr_gen = set(bm[i] for i in tr), set(gen[i] for i in tr)
                te_bm, te_gen = set(bm[i] for i in te), set(gen[i] for i in te)
                acc["bm_ov"].append(100 * len(te_bm & tr_bm) / max(len(te_bm), 1))
                acc["gen_ov"].append(100 * len(te_gen & tr_gen) / max(len(te_gen), 1))
                acc["bm_mol"].append(100 * np.mean([bm[i] in tr_bm for i in te]))
                acc["gen_mol"].append(100 * np.mean([gen[i] in tr_gen for i in te]))
                trf = [fps[i] for i in tr]
                sims = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[j], trf)) for j in te])
                acc["sim"].append(sims.mean()); acc["hard"].append(100 * np.mean(sims < 0.45))
                sd = y[tr].std() + 1e-9
                acc["w1"].append(stats.wasserstein_distance(y[tr] / sd, y[te] / sd))
                acc["shift"].append(np.linalg.norm(Xd[tr].mean(0) - Xd[te].mean(0)) / np.sqrt(Xd.shape[1]))
                acc["n_te"].append(len(te))
            jac = [len(a & b_) / len(a | b_) for a, b_ in itertools.combinations(test_sets, 2)]
            ms = lambda k, f="{:.1f}": f"{f.format(np.mean(acc[k]))} ± {f.format(np.std(acc[k], ddof=1))}"
            rows.append({"Dataset": ds, "Split": proto, "Test N": f"{np.mean(acc['n_te']):.0f}",
                         "Distinct test sets": len(set(test_sets)), "Mean pairwise test Jaccard": f"{np.mean(jac):.2f}",
                         "BM scaffold overlap (%)": ms("bm_ov"), "Generic skeleton overlap (%)": ms("gen_ov"),
                         "Test mols w/ seen generic skeleton (%)": ms("gen_mol"),
                         "NN Tanimoto": ms("sim", "{:.3f}"), "Hard OOD <0.45 (%)": ms("hard"),
                         "Target W1": ms("w1", "{:.3f}"), "Descriptor mean shift": ms("shift", "{:.3f}")})
    return pd.DataFrame(rows)


def dataset_stats():
    b = load_bundle(); rows = []
    for r in b["table1_summary"]:
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    t2 = split_diagnostics(); t2.to_csv(os.path.join(out, "table2_split_diagnostics_5seed.csv"), index=False)
    dataset_stats().to_csv(os.path.join(out, "table1_dataset_statistics.csv"), index=False)
    pd.set_option("display.width", 250); print(t2.to_string())
