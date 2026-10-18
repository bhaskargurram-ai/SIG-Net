"""Benchmark worker: runs the full model zoo on every (protocol, seed) fold of the given datasets.
Checkpoints one pickle per fold so interrupted runs resume. Usage: python run_benchmark.py FreeSolv ESOL
"""
import os, sys, time, pickle, traceback
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_data import prepare_fold, SEEDS, PROTOCOLS, DATASET_CFG
from models import *
from v3 import run_fold_v3
RUN_V3 = os.environ.get("JCIM_V3", "0") == "1"

OUT = os.environ.get("JCIM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
FOLD_DIR = os.path.join(OUT, "folds_v3" if os.environ.get("JCIM_V3", "0") == "1" else "folds")
os.makedirs(FOLD_DIR, exist_ok=True)
EPOCHS = int(os.environ.get("JCIM_EPOCHS", 80))
N_ENS = 3


def run_fold(ds, proto, seed):
    f = prepare_fold(ds, proto, seed); task = f["task"]
    ic, ed, pdim, pr = f["in_channels"], f["edge_dim"], f["phys_dim"], f["prior_dim"]
    y_tr, y_va, y_te = f["y"]; a_tr, a_va, a_te = f["anchor"]; nov_tr, nov_va, nov_te = f["novelty"]
    R = {}  # name -> {metrics, test_preds, [val_preds], [gates], [extra]}
    t0 = time.time()
    R.update(run_tabular(f, seed))
    R["Anchor-Only (Multi-View Stack)"] = {"metrics": metrics(y_te, a_te, task), "val_preds": a_va, "test_preds": a_te}
    # graph baselines (all Chemprop-grade features, target-standardised, full schedule, best-val checkpoint)
    for name, build, reg, lam in [
        ("GCN", lambda: GCNNet(ic), "none", 0.0),
        ("GINE", lambda: GINENet(ic, ed), "none", 0.0),
        ("GATv2", lambda: GATv2Net(ic, ed), "none", 0.0),
        ("AttentiveFP", lambda: AttentiveFPNet(ic, ed), "none", 0.0),
        ("D-MPNN (Chemprop)", lambda: DMPNNNet(ic, ed, pdim), "none", 0.0),
        ("D-MPNN + Deep CORAL", lambda: DMPNNNet(ic, ed, pdim, use_rdkit=True), "coral", 0.10),
    ]:
        r = train_neural(build(), f, seed, reg=reg, lam=lam, epochs=EPOCHS); r.pop("_model", None); R[name] = r
    # graph expert: 3-seed D-MPNN+RDKit ensemble; member 0 doubles as the single-model baseline
    ge = ensemble_graph_expert(f, lambda: DMPNNNet(ic, ed, pdim, use_rdkit=True), seed, n_members=N_ENS, epochs=EPOCHS)
    R["D-MPNN + RDKit-2D"] = {"metrics": metrics(y_te, ge["members_test"][0], task), "val_preds": ge["members_val"][0], "test_preds": ge["members_test"][0]}
    R[f"D-MPNN + RDKit-2D Ensemble ({N_ENS}x)"] = {"metrics": metrics(y_te, ge["test"], task), "val_preds": ge["val"], "test_preds": ge["test"]}
    # previous-generation design (dual-stream gated OOF-anchor residual), re-featurised, for the record
    r = train_neural(SIGNetV2(ic, ed, pr), f, seed, signet=True, epochs=EPOCHS); r.pop("_model", None); R["Dual-Stream Gated Residual (v1 design)"] = r
    # stacking baselines
    xgb = R["XGBoost (ECFP4 + RDKit-2D)"]
    p, w = blend_on_val(f, xgb["val_preds"], xgb["test_preds"], ge["members_val"][0], ge["members_test"][0])
    R["Stacked (XGBoost + D-MPNN)"] = {"metrics": metrics(y_te, p, task), "test_preds": p, "w": w}
    p, w = blend_on_val(f, ge["val"], ge["test"], a_va, a_te)
    R["Stacked (Ensemble + Anchor, global w)"] = {"metrics": metrics(y_te, p, task), "test_preds": p, "w": w}
    # proposed
    R["SIG-Net v2 (Proposed)"] = sig_gate_on_val(f, seed, ge["val"], ge["test"], a_va, a_te, nov_va, nov_te)
    # ablation: gate on single expert (no ensemble)
    R["SIG-Net v2 w/o ensemble"] = sig_gate_on_val(f, seed, ge["members_val"][0], ge["members_test"][0], a_va, a_te, nov_va, nov_te)
    meta = {"y_true": y_te, "nn_sim": nov_te[:, 0], "smiles": f["test_smiles"], "scaffolds": f["test_scaffolds"], "task": task,
            "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te), "time": time.time() - t0}
    for v in R.values():
        v.pop("_model", None)
    return {"results": R, "meta": meta}


def claim(path):
    """Atomically claim a fold via a lock file so several workers can share the queue."""
    if os.path.exists(path):
        return False
    try:
        os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL)); return True
    except FileExistsError:
        return False


if __name__ == "__main__":
    args = sys.argv[1:]; reverse = "--reverse" in args; datasets = [a for a in args if not a.startswith("--")] or list(DATASET_CFG)
    jobs = [(ds, proto, seed) for ds in datasets for proto in PROTOCOLS for seed in SEEDS]
    for ds, proto, seed in (jobs[::-1] if reverse else jobs):
                path = os.path.join(FOLD_DIR, f"{ds}__{proto}__{seed}.pkl")
                if not claim(path):
                    continue
                try:
                    out = run_fold_v3(ds, proto, seed, epochs=EPOCHS) if RUN_V3 else run_fold(ds, proto, seed)
                    with open(path, "wb") as fh:
                        pickle.dump(out, fh)
                    R, m = out["results"], out["meta"]
                    if RUN_V3:
                        g = lambda k: R[k]["metrics"]["primary"]
                        print(f"[{time.strftime('%H:%M:%S')}] {ds:13s} {proto:16s} seed {seed:4d} ({m['time']:.0f}s) v3 {g('SIG-Net v3 (Proposed)'):.3f} "
                              f"cfg={R['SIG-Net v3 (Proposed)']['config']} | global-stack {g('Global stack (same cross-fitted experts)'):.3f}", flush=True)
                        continue
                    key = "RMSE" if m["task"] == "regression" else "AUC"
                    print(f"[{time.strftime('%H:%M:%S')}] {ds:13s} {proto:16s} seed {seed:4d} ({m['time']:.0f}s) {key}: "
                          f"DMPNN+RD {R['D-MPNN + RDKit-2D']['metrics']['primary']:.3f} | Ens {R['D-MPNN + RDKit-2D Ensemble (3x)']['metrics']['primary']:.3f} | "
                          f"Anchor {R['Anchor-Only (Multi-View Stack)']['metrics']['primary']:.3f} | Stack {R['Stacked (Ensemble + Anchor, global w)']['metrics']['primary']:.3f} | "
                          f"SIG-v2 {R['SIG-Net v2 (Proposed)']['metrics']['primary']:.3f} gate={R['SIG-Net v2 (Proposed)']['used_gate']}", flush=True)
                except Exception:
                    print(f"FAILED {ds} {proto} {seed}\n{traceback.format_exc()}", flush=True)
                finally:
                    if os.path.exists(path + ".lock"):
                        os.remove(path + ".lock")
    print("WORKER DONE", datasets, flush=True)
