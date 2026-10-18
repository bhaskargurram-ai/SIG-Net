"""Execute the pipeline notebook in place with nbclient. Usage: python execute_notebook.py [out.ipynb]"""
import os, sys
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(ROOT, "SIG_Net_Complete_Pipeline.ipynb")
dst = os.path.join(ROOT, sys.argv[1]) if len(sys.argv) > 1 else src
nb = nbformat.read(src, as_version=4)
os.environ.setdefault("JCIM_WORKDIR", ROOT)
client = NotebookClient(nb, timeout=None, kernel_name="py314", resources={"metadata": {"path": ROOT}})
try:
    client.execute()
    print("EXECUTED OK")
finally:
    nbformat.write(nb, dst)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    print(f"ERROR in cell {i}: {o['ename']}: {o['evalue']}")
