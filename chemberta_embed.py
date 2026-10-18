"""Frozen ChemBERTa-2 (DeepChem/ChemBERTa-77M-MTR) embeddings: masked mean-pool of the last hidden state.
Writes results/embeddings/<dataset>_chemberta77m.npy in bundle order. Usage: python chemberta_embed.py [bundle.pt] [datasets...]"""
import os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

MODEL = "DeepChem/ChemBERTa-77M-MTR"
ROOT = os.path.dirname(os.path.abspath(__file__))
bundle_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("JCIM_BUNDLE", "data/phase1_bundle.pt")
want = sys.argv[2:] or None
out_dir = os.path.join(ROOT, "results", "embeddings"); os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL).to(dev).eval()
b = torch.load(bundle_path, map_location="cpu", weights_only=False)
for ds, data in b["datasets"].items():
    if want and ds not in want:
        continue
    path = os.path.join(out_dir, f"{ds}_chemberta77m.npy")
    if os.path.exists(path):
        print("exists", path); continue
    smi = [d.smiles for d in data]; embs = []
    with torch.no_grad():
        for i in range(0, len(smi), 128):
            enc = tok(smi[i:i + 128], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.concatenate(embs).astype(np.float32); np.save(path, E); print(ds, E.shape, flush=True)
