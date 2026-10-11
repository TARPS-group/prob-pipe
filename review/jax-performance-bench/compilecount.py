"""Count XLA compilations and jaxpr traces with jax.monitoring listeners."""

import time
from collections import Counter

import jax
from jax import monitoring

_stats = Counter()
_secs = Counter()


def _listener(event, duration, **kw):
    if event == "/jax/core/compile/backend_compile_duration":
        _stats["compiles"] += 1
        _secs["compile_s"] += duration
    elif event == "/jax/core/compile/jaxpr_trace_duration":
        _stats["traces"] += 1
        _secs["trace_s"] += duration
    elif event == "/jax/core/compile/jaxpr_to_mlir_module_duration":
        _stats["lowerings"] += 1


monitoring.register_event_duration_secs_listener(_listener)


def measure(fn, n=3):
    rows = []
    for _ in range(n):
        _stats.clear()
        _secs.clear()
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(jax.tree.leaves(out.raw() if hasattr(out, "raw") else out))
        dt = time.perf_counter() - t
        rows.append((dt, _stats["compiles"], _secs["compile_s"], _stats["traces"]))
    return rows


def fmt(name, rows):
    cells = "".join(f"  {dt:7.3f}s c={c:<3d}({cs:5.2f}s) t={tr:<4d}" for dt, c, cs, tr in rows)
    return f"{name:44s}{cells}"
