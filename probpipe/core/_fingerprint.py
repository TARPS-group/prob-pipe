"""Stable content hashing for ProbPipe objects.

Produces a short hex digest that is deterministic across processes for the
same inputs.  Intended for provenance tracking (populating
``ParentInfo.fingerprint``) and as the foundation for a future Prefect
``cache_key_fn``.

Supported types
---------------
- ``jax.Array`` / ``numpy.ndarray`` — shape + dtype + bytes (capped at 1 MB)
- Python scalars (``int``, ``float``, ``bool``, ``str``, ``None``)
- ``Record`` — field names + values, hashed recursively
- ``Distribution`` — class name + distribution name + numeric parameters
- ``WorkflowFunction`` — bytecode of the wrapped user function
- Everything else — ``repr()`` truncated to 500 characters

All imports of ProbPipe types are deferred to call time so this module can
be imported early in the package without triggering circular-import issues.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

__all__ = ["fingerprint"]

# Arrays larger than this threshold are hashed by sampling bytes at
# evenly-spaced offsets instead of reading the whole buffer.  1 MB is
# large enough that most parameter arrays are hashed completely.
_FULL_HASH_BYTES = 1 << 20  # 1 MB


def fingerprint(obj: Any) -> str:
    """Return a 16-character hex digest that stably identifies *obj*'s content.

    The digest is deterministic within a Python session and across processes
    for the same inputs.  It is NOT cryptographically secure — use it only
    for cache keying and provenance labelling.

    Parameters
    ----------
    obj:
        Any ProbPipe object or Python primitive.  Unknown types fall back
        to a ``repr()``-based hash.

    Returns
    -------
    str
        A 16-character hex string (64-bit prefix of a SHA-256 digest).
    """
    h = hashlib.sha256()
    _update(h, obj, depth=0)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------


def _update(h: "hashlib._Hash", obj: Any, depth: int) -> None:
    if depth > 32:
        h.update(b"[max_depth]")
        return

    import numpy as np

    try:
        import jax

        _jax_array = jax.Array
    except ImportError:
        _jax_array = type(None)

    if isinstance(obj, (np.ndarray, _jax_array)):
        _update_array(h, obj)
    elif _is_record(obj):
        _update_record(h, obj, depth)
    elif _is_distribution(obj):
        _update_distribution(h, obj, depth)
    elif _is_workflow_function(obj):
        _update_workflow_function(h, obj)
    elif isinstance(obj, bool):
        # bool before int — bool is a subclass of int
        h.update(b"bool:")
        h.update(b"1" if obj else b"0")
    elif isinstance(obj, int):
        h.update(b"int:")
        h.update(struct.pack(">q", obj))
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
            _update(h, item, depth + 1)
    elif isinstance(obj, dict):
        h.update(b"dict:")
        h.update(struct.pack(">I", len(obj)))
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            _update(h, k, depth + 1)
            _update(h, v, depth + 1)
    else:
        h.update(b"repr:")
        h.update(repr(obj)[:500].encode())


# ---------------------------------------------------------------------------
# Array hashing
# ---------------------------------------------------------------------------


def _update_array(h: "hashlib._Hash", arr: Any) -> None:
    """Hash an array by shape, dtype, and content bytes."""
    import numpy as np

    # Normalise to a numpy array for a uniform bytes() interface.
    # For large JAX arrays this triggers a device→host transfer; that is
    # acceptable here since fingerprinting happens at provenance-creation
    # time, not inside a JIT-compiled function.
    try:
        arr_np = np.asarray(arr)
    except Exception:
        h.update(b"array:unreadable")
        return

    h.update(b"array:")
    h.update(str(arr_np.shape).encode())
    h.update(b":")
    h.update(str(arr_np.dtype).encode())
    h.update(b":")

    raw = arr_np.tobytes()
    if len(raw) <= _FULL_HASH_BYTES:
        h.update(raw)
    else:
        # Sample 16 evenly-spaced 256-byte windows so large arrays are still
        # discriminated without reading the full buffer.
        step = max(1, len(raw) // 16)
        for i in range(0, len(raw), step):
            h.update(raw[i : i + 256])
        h.update(struct.pack(">Q", len(raw)))


# ---------------------------------------------------------------------------
# Record hashing
# ---------------------------------------------------------------------------


def _is_record(obj: Any) -> bool:
    try:
        from .record import Record

        return isinstance(obj, Record)
    except Exception:
        return False


def _update_record(h: "hashlib._Hash", record: Any, depth: int) -> None:
    """Hash a Record field-by-field in insertion order."""
    h.update(b"record:")
    h.update(type(record).__name__.encode())
    h.update(b":")
    for field in record.fields:
        h.update(field.encode())
        h.update(b"=")
        _update(h, record[field], depth + 1)
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


def _update_distribution(h: "hashlib._Hash", dist: Any, depth: int) -> None:
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
    if tfp_dist is not None:
        params = getattr(tfp_dist, "parameters", {}) or {}
        for k, v in sorted(params.items()):
            h.update(k.encode())
            h.update(b"=")
            _update(h, v, depth + 1)
            h.update(b";")
    else:
        # Non-TFP distribution: hash public attributes that look numeric or
        # array-like, skipping framework internals.
        _SKIP = frozenset({"_name", "_source", "_auxiliary", "_sampling_cost"})
        for attr, val in sorted(vars(dist).items()):
            if attr in _SKIP or attr.startswith("__"):
                continue
            h.update(attr.encode())
            h.update(b"=")
            _update(h, val, depth + 1)
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


def _update_workflow_function(h: "hashlib._Hash", wf: Any) -> None:
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

    h.update(code.co_code)
    # co_consts captures literal values in the function body (strings,
    # numbers, nested code objects).  Stringifying avoids hashing nested
    # code objects recursively, which would require special casing.
    h.update(repr(code.co_consts).encode())
    h.update(code.co_filename.encode())
    h.update(struct.pack(">I", code.co_firstlineno))
