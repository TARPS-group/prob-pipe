"""Stable content hashing for ProbPipe objects.

Produces a short hex digest that is deterministic across processes for the
same inputs.  Intended for provenance tracking (populating
``ParentInfo.fingerprint``) and as the foundation for a future Prefect
``cache_key_fn``.

Supported types
---------------
- ``jax.Array`` / ``numpy.ndarray`` — shape + dtype + bytes (up to
  ``max_array_bytes``; sampled beyond that)
- Python scalars (``int``, ``float``, ``bool``, ``str``, ``None``)
- ``Record`` — field names + values, hashed recursively
- ``Distribution`` — class name + distribution name + numeric parameters
- ``WorkflowFunction`` — bytecode of the wrapped user function
- Everything else — ``repr()`` truncated to 500 characters

All imports of ProbPipe types are deferred to call time so this module can
be imported early in the package without triggering circular-import issues.

Notes
-----
**Collision semantics:** the digest is a 64-bit prefix of SHA-256.  At
ProbPipe scale a collision is negligible in probability, but when one does
occur it produces a *wrong cached result*, not a cache miss.

**JAX x64 mode:** ``str(dtype)`` is used for array dtype, so digests for
the same array differ between ``jax_enable_x64=False`` (default, float32)
and ``jax_enable_x64=True`` (float64).  Digests are stable within a fixed
JAX configuration.

**Python version portability:** bytecode (``WorkflowFunction``) is not
portable across Python minor versions — a 3.12-compiled digest differs from
a 3.13 one for the same source.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

__all__ = ["fingerprint"]

import numpy as _np

try:
    import jax as _jax

    _JAX_ARRAY_TYPE = _jax.Array
except ImportError:
    _JAX_ARRAY_TYPE = type(None)  # type: ignore[assignment,misc]

_NP_ARRAY_TYPE = _np.ndarray

# Default cap: arrays up to 256 MB are hashed in full (zero-copy via
# memoryview).  Arrays beyond this are sampled at evenly-spaced offsets so
# large posteriors are still discriminated without a full buffer read.
# Pass ``max_array_bytes=None`` to always hash the full buffer.
_DEFAULT_MAX_ARRAY_BYTES: int = 256 << 20  # 256 MB


def fingerprint(obj: Any, *, max_array_bytes: int | None = _DEFAULT_MAX_ARRAY_BYTES) -> str:
    """Return a 16-character hex digest that stably identifies *obj*'s content.

    The digest is deterministic within a Python session and across processes
    for the same inputs.  It is NOT cryptographically secure — use it only
    for cache keying and provenance labelling.

    Parameters
    ----------
    obj:
        Any ProbPipe object or Python primitive.  Unknown types fall back
        to a ``repr()``-based hash.
    max_array_bytes:
        Arrays whose byte size is at or below this threshold are hashed in
        full (zero-copy via ``memoryview``).  Larger arrays are sampled at
        evenly-spaced offsets.  Pass ``None`` to always hash the full buffer.
        Default: 256 MB.

    Returns
    -------
    str
        A 16-character hex string (64-bit prefix of a SHA-256 digest).
    """
    h = hashlib.sha256()
    _update(h, obj, depth=0, max_array_bytes=max_array_bytes)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------


def _update(h: hashlib._Hash, obj: Any, depth: int, max_array_bytes: int | None) -> None:
    if depth > 32:
        h.update(b"[max_depth]")
        return

    if isinstance(obj, (_NP_ARRAY_TYPE, _JAX_ARRAY_TYPE)):
        _update_array(h, obj, max_array_bytes)
    elif _is_record(obj):
        _update_record(h, obj, depth, max_array_bytes)
    elif _is_distribution(obj):
        _update_distribution(h, obj, depth, max_array_bytes)
    elif _is_workflow_function(obj):
        _update_workflow_function(h, obj, max_array_bytes)
    elif isinstance(obj, bool):
        # bool before int — bool is a subclass of int
        h.update(b"bool:")
        h.update(b"1" if obj else b"0")
    elif isinstance(obj, int):
        h.update(b"int:")
        h.update(str(obj).encode())
    elif isinstance(obj, float):
        h.update(b"float:")
        h.update(struct.pack(">d", obj))
    elif isinstance(obj, str):
        h.update(b"str:")
        h.update(obj.encode())
    elif obj is None:
        h.update(b"none")
    elif isinstance(obj, (list, tuple)):
        tag = b"list:" if isinstance(obj, list) else b"tuple:"
        h.update(tag)
        h.update(struct.pack(">I", len(obj)))
        for item in obj:
            _update(h, item, depth + 1, max_array_bytes)
    elif isinstance(obj, dict):
        h.update(b"dict:")
        h.update(struct.pack(">I", len(obj)))
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            _update(h, k, depth + 1, max_array_bytes)
            _update(h, v, depth + 1, max_array_bytes)
    elif _is_tfp_object(obj):
        _update_tfp_object(h, obj, depth, max_array_bytes)
    else:
        h.update(b"repr:")
        h.update(repr(obj)[:500].encode())


# ---------------------------------------------------------------------------
# Array hashing
# ---------------------------------------------------------------------------


def _update_array(h: hashlib._Hash, arr: Any, max_bytes: int | None) -> None:
    """Hash an array by shape, dtype, and content bytes.

    For large JAX arrays, ``np.asarray`` triggers a device→host transfer.
    That is acceptable here — fingerprinting happens at provenance-creation
    time, not inside a JIT-compiled function.
    """
    try:
        buf = _np.ascontiguousarray(_np.asarray(arr))
    except Exception:
        h.update(b"array:unreadable")
        return

    h.update(f"array:{buf.shape}:{buf.dtype}:".encode())

    if max_bytes is None or buf.nbytes <= max_bytes:
        # Full hash, zero-copy: memoryview avoids an extra .tobytes() copy.
        h.update(memoryview(buf).cast("B"))
    else:
        # Sample ~16 evenly-spaced windows directly from the contiguous
        # buffer view — O(sample) slices, no full copy.
        view = memoryview(buf).cast("B")
        n = len(view)
        window = max(1, max_bytes // 16)
        step = max(1, n // 16)
        for i in range(0, n, step):
            h.update(view[i : i + window])
        h.update(struct.pack(">Q", n))


# ---------------------------------------------------------------------------
# TFP-native object hashing (nested distributions, bijectors)
# ---------------------------------------------------------------------------


def _is_tfp_object(obj: Any) -> bool:
    """Return True for TFP-native distributions and bijectors.

    Detected by duck-typing: they carry a ``parameters`` dict and expose
    ``log_prob`` (distributions) or ``forward`` (bijectors).  This avoids
    an explicit TFP import and lets the check degrade gracefully if TFP is
    not installed.
    """
    params = getattr(obj, "parameters", None)
    if not isinstance(params, dict):
        return False
    return hasattr(obj, "log_prob") or hasattr(obj, "forward")


def _update_tfp_object(h: hashlib._Hash, obj: Any, depth: int, max_array_bytes: int | None) -> None:
    """Hash a TFP-native distribution or bijector by type and parameters."""
    h.update(b"tfp:")
    h.update(type(obj).__name__.encode())
    h.update(b":")
    params = getattr(obj, "parameters", {}) or {}
    for k, v in sorted(params.items()):
        h.update(k.encode())
        h.update(b"=")
        _update(h, v, depth + 1, max_array_bytes)
        h.update(b";")


# ---------------------------------------------------------------------------
# Record hashing
# ---------------------------------------------------------------------------


def _is_record(obj: Any) -> bool:
    try:
        from .record import Record

        return isinstance(obj, Record)
    except Exception:
        return False


def _update_record(h: hashlib._Hash, record: Any, depth: int, max_array_bytes: int | None) -> None:
    """Hash a Record field-by-field in insertion order."""
    h.update(b"record:")
    h.update(type(record).__name__.encode())
    h.update(b":")
    for field in record.fields:
        h.update(field.encode())
        h.update(b"=")
        _update(h, record[field], depth + 1, max_array_bytes)
        h.update(b";")


# ---------------------------------------------------------------------------
# Distribution hashing
# ---------------------------------------------------------------------------


def _is_distribution(obj: Any) -> bool:
    try:
        from ._distribution_base import Distribution

        return isinstance(obj, Distribution)
    except Exception:
        return False


def _update_distribution(
    h: hashlib._Hash, dist: Any, depth: int, max_array_bytes: int | None
) -> None:
    """Hash a distribution by class name, distribution name, and parameters.

    For TFP-backed distributions (those with a ``_tfp_dist`` attribute) the
    TFP parameter dict is hashed directly — this covers every concrete
    parametric (Normal, Gamma, Beta, ...) without needing per-class logic.
    For other distributions (EmpiricalDistribution, etc.) we fall back to
    iterating public instance attributes.
    """
    h.update(b"dist:")
    h.update(type(dist).__name__.encode())
    h.update(b":")
    name = getattr(dist, "name", None) or ""
    h.update(name.encode())
    h.update(b":")

    tfp_dist = getattr(dist, "_tfp_dist", None)
    samples = getattr(dist, "_samples", None)

    if tfp_dist is not None:
        params = getattr(tfp_dist, "parameters", {}) or {}
        # Skip construction flags and the name (already hashed above).
        _TFP_SKIP = frozenset({"name", "validate_args", "allow_nan_stats"})
        for k, v in sorted(params.items()):
            if k in _TFP_SKIP:
                continue
            h.update(k.encode())
            h.update(b"=")
            _update(h, v, depth + 1, max_array_bytes)
            h.update(b";")
    elif samples is not None:
        # EmpiricalDistribution / RecordEmpiricalDistribution: hash the
        # sample data and weight array explicitly so reweighted posteriors
        # (IS/SMC) are distinguished from the original.
        h.update(b"samples=")
        _update(h, samples, depth + 1, max_array_bytes)
        w = getattr(dist, "_w", None)
        log_w = getattr(w, "log_normalized", None)
        h.update(b"log_weights=")
        if log_w is not None:
            _update(h, log_w, depth + 1, max_array_bytes)
        else:
            h.update(b"uniform")
    else:
        # Generic fallback for other non-TFP distributions.
        _SKIP = frozenset({"_name", "_source", "_auxiliary", "_sampling_cost"})
        for attr, val in sorted(vars(dist).items()):
            if attr in _SKIP or attr.startswith("__"):
                continue
            h.update(attr.encode())
            h.update(b"=")
            _update(h, val, depth + 1, max_array_bytes)
            h.update(b";")


# ---------------------------------------------------------------------------
# WorkflowFunction hashing
# ---------------------------------------------------------------------------


def _is_workflow_function(obj: Any) -> bool:
    try:
        from .node import WorkflowFunction

        return isinstance(obj, WorkflowFunction)
    except Exception:
        return False


def _hash_code(h: hashlib._Hash, code: Any, max_array_bytes: int | None) -> None:
    """Recursively hash a code object without repr-ing nested code objects.

    ``repr(code.co_consts)`` embeds memory addresses for any nested ``def``,
    ``lambda``, or generator — making the digest process-dependent.  Instead
    we walk ``co_consts`` and recurse into nested code objects directly.
    ``co_filename`` and ``co_firstlineno`` are intentionally excluded: they
    are machine- and position-dependent and defeat cross-machine caching.
    """
    import types

    h.update(code.co_code)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _hash_code(h, c, max_array_bytes)
        else:
            _update(h, c, 0, max_array_bytes)


def _update_workflow_function(h: hashlib._Hash, wf: Any, max_array_bytes: int | None) -> None:
    """Hash a WorkflowFunction by its user function's bytecode.

    Hashes ``_func.__code__`` directly rather than the Prefect task-wrapper
    closure, so changes to the user function body are detected even when the
    wrapper source is unchanged.
    """
    h.update(b"wf:")
    func = getattr(wf, "_func", None)
    if func is None:
        h.update(b"unknown")
        return

    code = getattr(func, "__code__", None)
    if code is None:
        h.update(repr(func)[:200].encode())
        return

    _hash_code(h, code, max_array_bytes)
