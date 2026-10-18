"""Expert worker for the v4 study: saves cross-fitted expert matrices per fold. Prints timings only (no test metrics),
so the confirmatory test results stay unseen until the v4 router is frozen. Lock files let several workers share the queue.
Usage: python run_experts.py <dataset> [<dataset> ...] [--seeds 11,1,2,3]"""
import os, sys, time, pickle, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_data import PROTOCOLS
from v3 import run_fold_experts

OUT = os.path.join(os.environ.get("JCIM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "experts")
os.makedirs(OUT, exist_ok=True)
EPOCHS = int(os.environ.get("JCIM_EPOCHS", 80))
args = sys.argv[1:]; seeds = [11, 1, 2, 3]
if "--seeds" in args:
    i = args.index("--seeds"); seeds = [int(s) for s in args[i + 1].split(",")]; args = args[:i] + args[i + 2:]
jobs = [(ds, proto, s) for s in seeds for ds in args for proto in PROTOCOLS]
for ds, proto, s in jobs:
    path = os.path.join(OUT, f"{ds}__{proto}__{s}.pkl")
    if os.path.exists(path):
        continue
    try:
        os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        continue
    try:
        out = run_fold_experts(ds, proto, s, epochs=EPOCHS)
        pickle.dump(out, open(path, "wb"))
        print(f"[{time.strftime('%H:%M:%S')}] {ds:13s} {proto:16s} seed {s:3d} done in {out['meta']['time']:.0f}s", flush=True)
    except Exception:
        print(f"FAILED {ds} {proto} {s}\n{traceback.format_exc()}", flush=True)
    finally:
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")
print("WORKER DONE", flush=True)
