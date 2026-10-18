"""Add fresh split seeds to the bundle with the identical Phase 1 splitting code (random / BM scaffold / generic
skeleton / Butina at Tanimoto distance 0.6, 80/10/10, large groups to train, small groups shuffled by seed).
Development seed 11; confirmatory seeds 1, 2, 3. Writes phase1_bundle_v3.pt."""
import sys, random
from collections import defaultdict
import numpy as np, torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog("rdApp.*")

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/phase1_bundle.pt"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/phase1_bundle.pt"
NEW_SEEDS = [11, 1, 2, 3]
PROTOCOLS = ["random", "bm_scaffold", "generic_scaffold", "butina_cluster"]
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def assign_groups(groups, n_total, seed):
    big = [g for g in groups if len(g) > n_total * 0.05]; small = [g for g in groups if len(g) <= n_total * 0.05]
    random.Random(seed).shuffle(small); big.sort(key=len, reverse=True)
    tr, va, te = [], [], []; t_cut, v_cut = int(round(0.8 * n_total)), int(round(0.1 * n_total))
    for g in big + small:
        if len(tr) + len(g) <= t_cut: tr.extend(g)
        elif len(va) + len(g) <= v_cut: va.extend(g)
        else: te.extend(g)
    return tr, va, te


b = torch.load(SRC, map_location="cpu", weights_only=False)
for ds, feats in b["datasets"].items():
    n = len(feats); bm_g, gen_g = defaultdict(list), defaultdict(list)
    for i, d in enumerate(feats):
        bm_g[d.bm_scaffold].append(i); gen_g[d.gen_scaffold].append(i)
    fps = [MORGAN.GetFingerprint(Chem.MolFromSmiles(d.smiles)) for d in feats]
    dists = []
    for i in range(1, n): dists.extend([1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    butina = [list(c) for c in Butina.ClusterData(dists, n, 0.6, isDistData=True)]
    X = np.vstack([d.phys_raw.numpy() for d in feats])
    for proto in PROTOCOLS:
        for s in NEW_SEEDS:
            assert s not in b["splits"][ds][proto], f"seed {s} already exists for {ds} {proto}"
            if proto == "random":
                idx = list(range(n)); random.Random(s).shuffle(idx); nt, nv = int(round(0.8 * n)), int(round(0.1 * n)); tr, va, te = idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
            elif proto == "bm_scaffold": tr, va, te = assign_groups(list(bm_g.values()), n, s)
            elif proto == "generic_scaffold": tr, va, te = assign_groups(list(gen_g.values()), n, s)
            else: tr, va, te = assign_groups(butina, n, s)
            b["splits"][ds][proto][s] = {"train": tr, "val": va, "test": te}
            Xt = X[tr]; mask = np.var(Xt, 0) > 1e-6
            if not mask.any(): mask[:] = True
            p01, p99 = np.percentile(Xt, 1, 0).astype(np.float32), np.percentile(Xt, 99, 0).astype(np.float32)
            Xc = np.clip(Xt, p01, p99); mean, std = Xc.mean(0).astype(np.float32), Xc.std(0).astype(np.float32); std[std < 1e-6] = 1
            b["fold_scalers"][ds][proto][s] = {"mean": mean, "std": std, "non_const_mask": mask, "p01": p01, "p99": p99}
    # overlap of new test sets with the original-seed test sets (how 'fresh' are they?)
    jac = []
    for proto in PROTOCOLS:
        old = [set(b["splits"][ds][proto][s]["test"]) for s in [42, 100, 2026, 7, 99]]
        for s in [1, 2, 3]:
            t = set(b["splits"][ds][proto][s]["test"]); jac.append(max(len(t & o) / len(t | o) for o in old))
    print(f"{ds:13s} new seeds added; max Jaccard of a confirmatory test set with any original test set: mean {np.mean(jac):.2f}, max {np.max(jac):.2f}", flush=True)
b["metadata"]["seeds_dev"] = [11]; b["metadata"]["seeds_confirmatory"] = [1, 2, 3]
torch.save(b, DST); print("wrote", DST)
