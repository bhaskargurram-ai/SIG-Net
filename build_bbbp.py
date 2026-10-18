"""Add BBBP to the Phase 1 bundle using the identical curation / featurisation / splitting code of the original
pipeline (Complete_JCIM_Final_Revision notebook, cell 2). Writes phase1_bundle_v2.pt = original bundle + BBBP."""
import os, sys, random, hashlib
from collections import defaultdict, Counter
import numpy as np, torch
from scipy import stats
from sklearn.preprocessing import RobustScaler
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.datasets import MoleculeNet
RDLogger.DisableLog("rdApp.*")

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/phase1_bundle.pt"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/phase1_bundle.pt"
SEEDS = [42, 100, 2026, 7, 99]; PROTOCOLS = ["random", "bm_scaffold", "generic_scaffold", "butina_cluster"]
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
name, task = "BBBP", "classification"

b = torch.load(SRC, map_location="cpu", weights_only=False)
desc_names = b["metadata"]["rdkit_descriptor_names"]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
frag, unch = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()


def safe_inchikey(mol, smi):
    try:
        k = Chem.MolToInchiKey(mol)
        if k: return k
    except Exception:
        pass
    return "SMIKEY-" + hashlib.sha256(smi.encode()).hexdigest()[:20].upper()


def scaffolds(mol):
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumHeavyAtoms() == 0: return "[Acyclic]", "[Acyclic]"
        bm = Chem.MolToSmiles(core, canonical=True, isomericSmiles=False) or "[Acyclic]"
        gen = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core), canonical=True, isomericSmiles=False) or "[Acyclic]"
        return bm, gen
    except Exception:
        return "[Acyclic]", "[Acyclic]"


def assign_groups(groups, n_total, seed):
    big = [g for g in groups if len(g) > n_total * 0.05]; small = [g for g in groups if len(g) <= n_total * 0.05]
    random.Random(seed).shuffle(small); big.sort(key=len, reverse=True)
    tr, va, te = [], [], []; t_cut, v_cut = int(round(0.8 * n_total)), int(round(0.1 * n_total))
    for g in big + small:
        if len(tr) + len(g) <= t_cut: tr.extend(g)
        elif len(va) + len(g) <= v_cut: va.extend(g)
        else: te.extend(g)
    return tr, va, te


raw = MoleculeNet(root="data/moleculenet", name="BBBP"); raw_count = len(raw)
by_key = defaultdict(list)
for idx in range(raw_count):
    d = raw[idx]
    if d.x is None or d.x.shape[0] < 1 or d.y is None or d.y.numel() == 0: continue
    yv = float(d.y.view(-1)[0].item())
    if np.isnan(yv): continue
    smi = str(d.smiles).strip(); mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    try:
        if "." in smi:
            mf = frag.choose(mol)
            if mf is not None and mf.GetNumAtoms() > 0: mol = mf
        mu = unch.uncharge(mol)
        if mu is not None and mu.GetNumAtoms() > 0: Chem.SanitizeMol(mu); mol = mu
    except Exception:
        mol = Chem.MolFromSmiles(smi)
    cs = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True); mc = Chem.MolFromSmiles(cs) or mol
    by_key[safe_inchikey(mc, cs)].append((d, mc, cs, yv))

curated = []
for k, ent in by_key.items():
    if len(ent) == 1:
        d, mc, cs, yv = ent[0]; curated.append((d, mc, cs, k, yv))
    elif len(set(int(round(e[3])) for e in ent)) == 1:
        d, mc, cs, yv = ent[0]; curated.append((d, mc, cs, k, float(int(round(yv)))))

bm_vocab, gen_vocab, feats, fps = {}, {}, [], []
mws, logps, tpsas, ro5 = [], [], [], []
for d, mc, cs, k, yv in curated:
    bm_s, gen_s = scaffolds(mc); bm_vocab.setdefault(bm_s, len(bm_vocab)); gen_vocab.setdefault(gen_s, len(gen_vocab))
    mw, lp, tp = Descriptors.MolWt(mc), Crippen.MolLogP(mc), rdMolDescriptors.CalcTPSA(mc)
    viol = int(mw > 500) + int(lp > 5) + int(Lipinski.NumHDonors(mc) > 5) + int(Lipinski.NumHAcceptors(mc) > 10)
    mws.append(mw); logps.append(lp); tpsas.append(tp); ro5.append(int(viol <= 1))
    d.x = d.x.float(); ne = d.edge_index.shape[1] if d.edge_index is not None else 0
    d.edge_attr = d.edge_attr.float() if (d.edge_attr is not None and d.edge_attr.numel() > 0) else torch.zeros((ne, 3))
    d.y = torch.tensor([[yv]], dtype=torch.float32)
    d.smiles, d.inchikey, d.bm_scaffold, d.gen_scaffold = cs, k, bm_s, gen_s
    d.scaffold_id = torch.tensor([bm_vocab[bm_s]]); d.gen_scaffold_id = torch.tensor([gen_vocab[gen_s]])
    try:
        rd = np.clip(np.nan_to_num(np.array(desc_calc.CalcDescriptors(mc), dtype=np.float32), 0.0), -1e6, 1e6)
    except Exception:
        rd = np.zeros(len(desc_names), np.float32)
    d.phys_raw = torch.tensor(rd).unsqueeze(0)
    fe = MORGAN.GetFingerprint(mc); fps.append(fe); ea = np.zeros(2048, np.float32); DataStructs.ConvertToNumpyArray(fe, ea); d.ecfp4 = torch.tensor(ea).unsqueeze(0)
    fm = MACCSkeys.GenMACCSKeys(mc); ma = np.zeros(167, np.float32); DataStructs.ConvertToNumpyArray(fm, ma); d.maccs = torch.tensor(ma).unsqueeze(0)
    feats.append(d)

n = len(feats); bmc = Counter(d.bm_scaffold for d in feats); gnc = Counter(d.gen_scaffold for d in feats)
b["table1_summary"].append(dict(zip(list(b["table1_summary"][0].keys()), {
    "Dataset": name, "Task (Unit)": "Classification (Binary (0/1))", "Raw -> Curated N": f"{raw_count} -> {n}",
    "Unique BM": len(bmc), "Unique Generic": len(gnc), "Singleton BM (%)": f"{100 * sum(1 for c in bmc.values() if c == 1) / len(bmc):.1f}%",
    "MolWt (Da)": f"{np.mean(mws):.1f} ± {np.std(mws):.1f}", "LogP": f"{np.mean(logps):.2f} ± {np.std(logps):.2f}",
    "TPSA (A2)": f"{np.mean(tpsas):.1f} ± {np.std(tpsas):.1f}", "Ro5 Pass (%)": f"{100 * np.mean(ro5):.1f}%"}.values())))
bm_g, gen_g = defaultdict(list), defaultdict(list)
for i, d in enumerate(feats): bm_g[d.bm_scaffold].append(i); gen_g[d.gen_scaffold].append(i)
dists = []
for i in range(1, n): dists.extend([1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
butina = [list(c) for c in Butina.ClusterData(dists, n, 0.6, isDistData=True)]
X = np.vstack([d.phys_raw.numpy() for d in feats])
splits, scalers = defaultdict(dict), defaultdict(dict)
for proto in PROTOCOLS:
    for s in SEEDS:
        if proto == "random":
            idx = list(range(n)); random.Random(s).shuffle(idx); nt, nv = int(round(0.8 * n)), int(round(0.1 * n)); tr, va, te = idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
        elif proto == "bm_scaffold": tr, va, te = assign_groups(list(bm_g.values()), n, s)
        elif proto == "generic_scaffold": tr, va, te = assign_groups(list(gen_g.values()), n, s)
        else: tr, va, te = assign_groups(butina, n, s)
        splits[proto][s] = {"train": tr, "val": va, "test": te}
        Xt = X[tr]; mask = np.var(Xt, 0) > 1e-6
        if not mask.any(): mask[:] = True
        p01, p99 = np.percentile(Xt, 1, 0).astype(np.float32), np.percentile(Xt, 99, 0).astype(np.float32)
        Xc = np.clip(Xt, p01, p99); mean, std = Xc.mean(0).astype(np.float32), Xc.std(0).astype(np.float32); std[std < 1e-6] = 1
        scalers[proto][s] = {"mean": mean, "std": std, "non_const_mask": mask, "p01": p01, "p99": p99}
        print(proto, s, len(tr), len(va), len(te), "pos-rate test %.2f" % np.mean([feats[i].y.item() for i in te]), flush=True)
b["datasets"][name] = feats; b["splits"][name] = dict(splits); b["fold_scalers"][name] = dict(scalers)
b["metadata"]["datasets"] = list(b["metadata"]["datasets"]) + [{"name": name, "task": task, "unit": "Binary (0/1)", "desc": "Blood-brain barrier penetration"}]
torch.save(b, DST); print("wrote", DST, "BBBP raw", raw_count, "curated", n, "BM", len(bmc), "generic", len(gnc))
