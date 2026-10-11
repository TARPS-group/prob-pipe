# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""Run a notebook's code cells in one process and report each cell's time and compilations.

Usage: uv run python review/jax-performance-bench/nbprofile.py NOTEBOOK [MODE]
"""

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
from compilecount import _secs, _stats

nb_path = os.path.abspath(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
if mode != "baseline":
    import patches

    patches.apply(mode)
with open(nb_path) as handle:
    nb = json.load(handle)
os.chdir(os.path.dirname(nb_path))
ns = {"__name__": "__main__"}
tot = 0
totc = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    lines = [line for line in src.splitlines() if not line.lstrip().startswith(("%", "!"))]
    src = "\n".join(lines)
    _stats.clear()
    _secs.clear()
    t = time.perf_counter()
    try:
        exec(compile(src, f"cell{i}", "exec"), ns)
    except Exception as e:
        print(f"cell {i}: ERROR {type(e).__name__}: {str(e)[:200]}")
    dt = time.perf_counter() - t
    tot += dt
    totc += _secs["compile_s"]
    first = next((ln for ln in lines if ln.strip() and not ln.strip().startswith("#")), "")[:70]
    print(
        f"cell {i:3d} {dt:8.2f}s  compiles={_stats['compiles']:4d} ({_secs['compile_s']:6.2f}s)  traces={_stats['traces']:4d}  | {first}",
        flush=True,
    )
    import matplotlib.pyplot as plt

    plt.close("all")
print(f"TOTAL {tot:.1f}s, compile {totc:.1f}s")
