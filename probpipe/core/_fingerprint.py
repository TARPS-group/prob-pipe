"""Stable content hashing for ProbPipe objects.

Produces a short hex digest that is deterministic across processes for the
same inputs.  Intended for provenance tracking (populating
``ParentInfo.fingerprint``) and as the foundation for a future Prefect
``cache_key_fn``.

Supported types
---------------
- ``jax.Array`` / ``numpy.ndarray`` — shape + dtype + bytes (up to
  ``max_array_bytes``; sampled, tail included, beyond that). Object-dtype
  arrays are hashed element-wise (their raw bytes are per-process pointers).
- Python scalars (``int``, ``float``, ``bool``, ``str``, ``None``) and numpy
  scalars; ``float`` ``-0.0``/``0.0`` and all NaN payloads are canonicalized
- ``set`` / ``frozenset`` — order-independent (element sub-digests, sorted)
- ``Record`` — leaf paths + leaf values (leaf-keyed collection)
- ``Distribution`` — class + name + parameters; ``EmpiricalDistribution``
  hashes samples + weights; ``Weights`` are hashed by content
- ``WorkflowFunction`` — user-function bytecode, referenced names, and
  captured/default values
- Numeric containers (``xarray`` / ``pandas`` / registered array backends)
  — concrete type + materialised shape + dtype + bytes **and** the
  container's identity-bearing metadata (coords / index / dims / attrs, via
  the backend's ``metadata`` hook), so equal values under different coords
  hash differently; a lazy leaf materialises when fingerprinted
- Everything else — ``repr()`` (may be process-dependent for opaque types)

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
import math
import struct
from typing import Any

import jax as _jax
import numpy as _np

__all__ = ["fingerprint"]

# JAX is a hard dependency of probpipe, so ``jax.Array`` is always importable.
_JAX_ARRAY_TYPE = _jax.Array
_NP_ARRAY_TYPE = _np.ndarray

# Canonical NaN payload — every NaN hashes identically regardless of the bit
# pattern a computation happened to produce.
_CANONICAL_NAN = struct.pack(">d", float("nan"))

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
    elif _is_weights(obj):
        _update_weights(h, obj, depth, max_array_bytes)
    elif isinstance(obj, _np.generic):
        # numpy scalars: np.int64 is NOT an int subclass, and np.float32's repr
        # is numpy-version-dependent — hash by dtype + raw bytes, never repr.
        h.update(b"npscalar:")
        h.update(str(obj.dtype).encode())
        h.update(b":")
        h.update(obj.tobytes())
    elif isinstance(obj, bool):
        # bool before int — bool is a subclass of int
        h.update(b"bool:")
        h.update(b"1" if obj else b"0")
    elif isinstance(obj, int):
        h.update(b"int:")
        h.update(str(obj).encode())
    elif isinstance(obj, float):
        h.update(b"float:")
        if math.isnan(obj):
            h.update(_CANONICAL_NAN)  # collapse all NaN payloads
        elif obj == 0.0:
            h.update(struct.pack(">d", 0.0))  # collapse -0.0 and 0.0
        else:
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
    elif isinstance(obj, (set, frozenset)):
        # Order-independent: hash each element to a sub-digest and combine the
        # sorted digests, so the result never depends on set iteration order
        # (which is PYTHONHASHSEED-randomized across processes).
        h.update(b"set:")
        h.update(struct.pack(">I", len(obj)))
        for d in sorted(_subdigest(e, depth + 1, max_array_bytes) for e in obj):
            h.update(d)
    elif isinstance(obj, dict):
        h.update(b"dict:")
        h.update(struct.pack(">I", len(obj)))
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            _update(h, k, depth + 1, max_array_bytes)
            _update(h, v, depth + 1, max_array_bytes)
    elif _is_tfp_object(obj):
        _update_tfp_object(h, obj, depth, max_array_bytes)
    elif (content := _numeric_container_to_numpy(obj)) is not None:
        # A numeric container (xarray / pandas / a registered array backend):
        # hash by concrete type, materialised values, AND the container's
        # identity-bearing metadata (coords / index / dims / attrs), so the
        # digest is a complete content identifier — two containers with equal
        # values but different coords fingerprint differently — and is
        # content-stable across processes rather than falling to ``repr``.
        h.update(b"container:")
        h.update(type(obj).__qualname__.encode())
        h.update(b":")
        _update_array(h, content, max_array_bytes)
        from ._array_backend import _metadata_of

        metadata = _metadata_of(obj)
        if metadata is not None:
            h.update(b":meta:")
            _update(h, metadata, depth + 1, max_array_bytes)
    else:
        # Last resort. ``repr`` may embed a memory address for objects with the
        # default ``__repr__`` (making the digest process-dependent); the
        # branches above cover ProbPipe's content-bearing types, so this is
        # reached only for genuinely opaque values. Hash the full repr — no
        # truncation, which would collide objects sharing a 500-char prefix.
        h.update(b"repr:")
        h.update(repr(obj).encode())


def _subdigest(obj: Any, depth: int, max_array_bytes: int | None) -> bytes:
    """Content digest of a single object, for order-independent combination."""
    sub = hashlib.sha256()
    _update(sub, obj, depth, max_array_bytes)
    return sub.digest()


def _numeric_container_to_numpy(obj: Any) -> _np.ndarray | None:
    """Materialise a numeric container's content as ``np.ndarray``, or ``None``.

    The input is a leaf in its native container form (an ``xarray`` /
    ``pandas`` object, a registered array backend); the output is that
    container's numeric content, materialised for hashing. Registry-first: a
    registered :class:`~probpipe.ArrayBackend` supplies the materialisation
    (``to_numpy``, re-wrapped in ``np.asarray`` so the return type holds
    regardless of the hook); a duck-typed container with a numeric numpy
    ``dtype`` / ``shape`` falls to ``np.asarray``. Anything else — the bare
    array types handled earlier, scalars, genuinely opaque objects — reports
    ``None``. Materialisation is the documented cost of fingerprinting a
    lazy / disk-backed leaf.
    """
    from ._array_backend import _is_numeric_dtype, array_backend_for

    backend = array_backend_for(obj)
    if backend is not None:
        return _np.asarray(backend.to_numpy(obj)) if backend.is_numeric(obj) else None
    dtype = getattr(obj, "dtype", None)
    if dtype is not None and hasattr(obj, "shape") and _is_numeric_dtype(dtype):
        return _np.asarray(obj)
    return None


# ---------------------------------------------------------------------------
# Array hashing
# ---------------------------------------------------------------------------


def _update_array(h: hashlib._Hash, arr: Any, max_bytes: int | None) -> None:
    """Hash an array by shape, dtype, and content bytes.

    For large JAX arrays, ``np.asarray`` triggers a device→host transfer.
    That is acceptable here — fingerprinting happens at provenance-creation
    time, not inside a JIT-compiled function.
    """
    # Shape + dtype up front — available even on a JAX tracer — so arrays of
    # different structure never collide, including on the unreadable path below.
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)
    h.update(f"array:{shape}:{dtype}:".encode())

    # Object-dtype arrays hold Python-object *pointers*; their raw buffer bytes
    # are per-process addresses, so hash the elements by content instead.
    if dtype is not None and getattr(dtype, "kind", None) == "O":
        try:
            flat = _np.asarray(arr).ravel()
        except Exception:
            h.update(b"unreadable")
            return
        h.update(b"O")
        h.update(struct.pack(">Q", flat.size))
        for el in flat:
            _update(h, el, 1, max_bytes)
        return

    nbytes = getattr(arr, "nbytes", None)
    try:
        if (
            max_bytes is not None
            and nbytes is not None
            and nbytes > max_bytes
            and dtype is not None
        ):
            # Above the cap: transfer and hash only evenly-spaced windows —
            # sliced device-side so the whole array is never copied host-side —
            # always including the final tail window, plus the total byte count.
            # Lossy (arrays differing only outside the windows can collide) but
            # it never silently ignores the tail.
            flat = arr.reshape(-1)
            total = int(flat.shape[0])
            win = max(1, (max_bytes // 16) // max(1, dtype.itemsize))
            tail = max(0, total - win)
            # 16 evenly-spaced window starts spanning 0..tail *inclusive*, so the
            # final window is anchored at the buffer end (tail) and the last
            # element is always hashed.
            starts = sorted({(i * tail) // 15 for i in range(16)})
            h.update(struct.pack(">Q", int(nbytes)))
            for off in starts:
                window = _np.ascontiguousarray(_np.asarray(flat[off : off + win]))
                h.update(memoryview(window).cast("B"))
        else:
            buf = _np.ascontiguousarray(_np.asarray(arr))
            # Full hash, zero-copy: memoryview avoids an extra .tobytes() copy.
            h.update(memoryview(buf).cast("B"))
    except Exception:
        # No concrete content (e.g. a JAX tracer under jit/vmap). Shape+dtype
        # above still discriminate by structure; two same-shape tracers are
        # indistinguishable at trace time — inherent, not a defect here.
        h.update(b"unreadable")


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
    except ImportError:
        return False


def _update_record(h: hashlib._Hash, record: Any, depth: int, max_array_bytes: int | None) -> None:
    """Hash a Record by its leaf-keyed items (full ``/``-paths → leaf values).

    ``Record`` is a leaf-keyed collection: ``items()`` yields every leaf by its
    canonical ``/``-joined path, so this flat walk captures the full nested
    structure without recursing — and without indexing interior sub-Records,
    which raises under the leaf-keyed API.
    """
    h.update(b"record:")
    h.update(type(record).__name__.encode())
    h.update(b":")
    for path, value in record.items():
        h.update(path.encode())
        h.update(b"=")
        _update(h, value, depth + 1, max_array_bytes)
        h.update(b";")


# ---------------------------------------------------------------------------
# Distribution hashing
# ---------------------------------------------------------------------------


def _is_distribution(obj: Any) -> bool:
    try:
        from ._distribution_base import Distribution

        return isinstance(obj, Distribution)
    except ImportError:
        return False


def _is_empirical(obj: Any) -> bool:
    try:
        from ._empirical import EmpiricalDistribution

        return isinstance(obj, EmpiricalDistribution)
    except ImportError:
        return False


def _is_weights(obj: Any) -> bool:
    try:
        from .._weights import Weights

        return isinstance(obj, Weights)
    except ImportError:
        return False


def _update_weights(h: hashlib._Hash, w: Any, depth: int, max_array_bytes: int | None) -> None:
    """Hash a ``Weights`` object by content: uniformity, count, and log-weights.

    Hashed via the public API so a ``Weights`` reached through a distribution's
    generic ``vars()`` fallback (e.g. a mixture or broadcast distribution) is
    distinguished by its actual weights rather than by ``repr`` (which drops the
    values, collapsing reweighted distributions to one digest).
    """
    h.update(b"weights:")
    h.update(b"1" if w.is_uniform else b"0")
    h.update(struct.pack(">Q", int(w.n)))
    if not w.is_uniform:
        _update(h, w.log_normalized, depth + 1, max_array_bytes)


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
    elif _is_empirical(dist):
        # EmpiricalDistribution / RecordEmpiricalDistribution: hash the sample
        # data and weights via the PUBLIC accessors. The Record-backed subclass
        # stores no ``_samples`` attribute, so keying on it silently dropped
        # into the generic fallback below and hashed weights by repr (losing the
        # values); using ``.samples`` / ``.is_uniform`` / ``.log_weights``
        # distinguishes reweighted posteriors (IS/SMC) from the original.
        h.update(b"samples=")
        _update(h, dist.samples, depth + 1, max_array_bytes)
        h.update(b"uniform=")
        h.update(b"1" if dist.is_uniform else b"0")
        if not dist.is_uniform:
            h.update(b"log_weights=")
            _update(h, dist.log_weights, depth + 1, max_array_bytes)
    else:
        # Generic fallback for other non-TFP distributions.
        _SKIP = frozenset(
            {"_name", "_name_is_auto", "_provenance", "_annotations", "_sampling_cost"}
        )
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
    except ImportError:
        return False


def _hash_code(h: hashlib._Hash, code: Any, depth: int, max_array_bytes: int | None) -> None:
    """Recursively hash a code object without repr-ing nested code objects.

    Hashes the executable bytecode (``co_code``) *and* the symbol names it
    references by index — ``co_names`` (globals / attributes / methods),
    ``co_varnames`` (locals / arguments) and ``co_freevars`` — so a body that
    only changes *which* name it calls (``jnp.sin`` vs ``jnp.cos``, ``r.mean()``
    vs ``r.sum()``) is detected; ``co_code`` alone is identical for those.

    ``repr(code.co_consts)`` embeds memory addresses for any nested ``def``,
    ``lambda``, or generator — making the digest process-dependent — so we walk
    ``co_consts`` and recurse into nested code objects directly (threading
    ``depth`` so the recursion guard bounds deeply nested closures).
    ``co_filename`` and ``co_firstlineno`` are intentionally excluded: they are
    machine- and position-dependent and defeat cross-machine caching.
    """
    import types

    h.update(b"code:")
    h.update(code.co_code)
    for names in (code.co_names, code.co_varnames, code.co_freevars):
        h.update(struct.pack(">I", len(names)))
        for n in names:
            h.update(n.encode())
            h.update(b",")
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _hash_code(h, c, depth + 1, max_array_bytes)
        else:
            _update(h, c, depth + 1, max_array_bytes)


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

    _hash_code(h, code, 0, max_array_bytes)
    # Captured/closed-over state: two functions with identical bytecode but
    # different defaults or closure values (e.g. ``make(1.0)`` vs ``make(2.0)``)
    # must not collide.
    h.update(b"defaults=")
    _update(h, func.__defaults__, 1, max_array_bytes)
    h.update(b"kwdefaults=")
    _update(h, func.__kwdefaults__, 1, max_array_bytes)
    h.update(b"closure=")
    closure = func.__closure__
    if closure is None:
        h.update(b"none")
    else:
        h.update(struct.pack(">I", len(closure)))
        for cell in closure:
            _update(h, cell.cell_contents, 1, max_array_bytes)
