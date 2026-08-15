"""NumericArray — the tracked class of the numeric-array kind.

See design III.1.
"""

from __future__ import annotations

import operator
from typing import Any

import jax
import numpy as np

from ._array_backend import (
    _event_shape_of,
    _is_numeric_leaf,
    _numpy_dtype_of,
    _to_jax_array,
    _to_numpy_array,
)
from .event_template import NumericArraySpec
from .provenance import Provenance
from .tracked import Annotated, TrackedTerm, auto_name

__all__ = ["NumericArray"]


class NumericArray(TrackedTerm, Annotated):
    """One numeric array value, with identity.

    The tracked class of the numeric-array kind, as :class:`~probpipe.Record` is
    of the record kind: what an operation returns when its declared kind is a
    :class:`~probpipe.NumericArraySpec`. It holds a single array and carries no
    batch axes, so :attr:`shape` is the **event** shape; multiplicity lives in
    :class:`~probpipe.NumericArrayBatch`.

    Parameters
    ----------
    value : array-like
        The array this names, stored verbatim in its native form: a bare array,
        an ``xarray`` / ``pandas`` container, or any registered backend, so a
        lazy or disk-backed value stays lazy. A bare Python scalar carries no
        metadata to read and is normalised to a 0-d ``jax.Array``.
    name : str, optional
        The value's name. Defaults to ``"numericarray"``, marked auto-derived.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given. A value left
        unnamed is auto-named regardless.
    spec : NumericArraySpec, optional
        What this value satisfies. Derived from the array's shape and dtype when
        omitted, with an unconstrained support.
    provenance : Provenance, optional
        How this value was produced.

    Raises
    ------
    TypeError
        If *spec* is not a :class:`NumericArraySpec`, or if *value* is not a
        numeric leaf.
    ValueError
        If *value*'s shape or dtype does not satisfy *spec*.

    Notes
    -----
    Construction validates against metadata alone, so the value materialises at
    most once per instance and only at a compute boundary — the storage rule
    :class:`~probpipe.NumericRecord` follows for its leaves.

    It carries the full array surface: arithmetic, comparison, and the
    conversion hooks. With one value and no fields, ``arr + 1`` has a single
    meaning, which is what lets :class:`~probpipe.Record` stay a container.
    The operators forward to the stored value and return what it returns, so
    arithmetic on a numpy-backed one yields ``numpy``, and identity stays with
    the operations that attach it.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> value = NumericArray(jnp.arange(3.0), name="draw")
    >>> value.shape
    (3,)
    >>> value + 1          # a bare array, not a NumericArray
    Array([1., 2., 3.], dtype=float32)
    """

    __slots__ = (
        "_annotations",
        "_jax_cache",
        "_name",
        "_name_is_auto",
        "_provenance",
        "_spec",
        "_value",
    )

    #: Derived from the value rather than transported, as for ``NumericRecord``.
    _transient_state = ("_jax_cache",)

    def __init__(
        self,
        value: Any,
        *,
        name: str | None = None,
        name_is_auto: bool = False,
        spec: NumericArraySpec | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if spec is not None and not isinstance(spec, NumericArraySpec):
            raise TypeError(
                f"NumericArray spec must be a NumericArraySpec, got {type(spec).__name__}"
            )
        if not _is_numeric_leaf(value):
            raise TypeError(
                f"NumericArray holds one numeric array; {type(value).__name__} is not a numeric leaf"
            )
        # A bare Python scalar carries no metadata to read, so it is the one
        # thing normalised at construction — as ``NumericRecord`` normalises it.
        stored = _to_jax_array(value) if isinstance(value, (int, float, complex, bool)) else value
        shape, dtype = _event_shape_of(stored), _numpy_dtype_of(stored)
        if spec is None:
            spec = NumericArraySpec(shape=shape, dtype=dtype)
        elif not spec.is_valid(stored):
            raise ValueError(
                f"the array does not satisfy its declaration: shape {shape} and "
                f"dtype {dtype} against {spec}"
            )
        if name is None:
            name, name_is_auto = auto_name(name, "numericarray")
        object.__setattr__(self, "_value", stored)
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(name, name_is_auto=name_is_auto, provenance=provenance)

    # -- what it holds ------------------------------------------------------

    @property
    def value(self) -> Any:
        """The stored value, in the form it was given. Untracked."""
        return self._value

    def as_jax(self) -> Any:
        """The value as a ``jax.Array`` — the single conversion point.

        A value already stored as one passes through, tracers included; a
        native container converts through its registered backend once and is
        memoised for this instance.
        """
        if isinstance(self._value, jax.Array):
            return self._value
        cached = getattr(self, "_jax_cache", None)
        if cached is None:
            cached = _to_jax_array(self._value)
            object.__setattr__(self, "_jax_cache", cached)
        return cached

    @property
    def spec(self) -> NumericArraySpec:
        """This value's own declaration."""
        return self._spec

    @property
    def shape(self) -> tuple[int, ...]:
        """The event shape, read from the value's metadata. No batch axes."""
        return _event_shape_of(self._value)

    @property
    def dtype(self) -> Any:
        """The stored value's dtype, or ``None`` when it has no single one.

        The value's rather than the declaration's, as for a single-field
        ``NumericRecord``: a caller sizing a buffer or branching on the dtype is
        asking about the data. The two can differ, since ``is_valid`` admits a
        same-kind cast; :attr:`spec` carries the declaration.
        """
        return _numpy_dtype_of(self._value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return f"NumericArray({self._value!r}, name={self.name!r})"

    # -- the array surface --------------------------------------------------

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = _to_numpy_array(self._value)
        arr = np.asarray(arr, dtype=dtype) if dtype is not None else arr
        return arr.copy() if copy else arr

    # JAX reads this for ``jnp.asarray``; the numpy hook above would win
    # otherwise and drop tracing.
    def __jax_array__(self) -> Any:
        return self.as_jax()

    def __float__(self) -> float:
        return float(self.as_jax())

    def __int__(self) -> int:
        return int(self.as_jax())

    def __bool__(self) -> bool:
        return bool(self.as_jax())

    def __index__(self) -> int:
        return operator.index(self.as_jax())

    def __getitem__(self, key: Any) -> Any:
        return self._value[key]

    def __iter__(self):
        return iter(self._value)


def _unwrap(other: Any) -> Any:
    """The array inside a ``NumericArray``, or *other* unchanged."""
    return other._value if isinstance(other, NumericArray) else other


def _install_array_operators() -> None:
    """Forward the array operators to the held array, returning what it returns.

    Installed from a table so the forty of them stay one rule. An in-place
    operator on an immutable term is the out-of-place one.
    """
    binary = [
        "add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod",
        "pow", "lshift", "rshift", "and", "xor", "or",
    ]  # fmt: skip
    comparison = ["lt", "le", "eq", "ne", "gt", "ge"]
    unary = ["neg", "pos", "abs", "invert"]

    def _binary(name: str):
        def method(self: NumericArray, other: Any) -> Any:
            return getattr(self._value, f"__{name}__")(_unwrap(other))

        method.__name__ = f"__{name}__"
        return method

    def _reflected(name: str):
        def method(self: NumericArray, other: Any) -> Any:
            return getattr(self._value, f"__r{name}__")(_unwrap(other))

        method.__name__ = f"__r{name}__"
        return method

    def _unary(name: str):
        def method(self: NumericArray) -> Any:
            return getattr(self._value, f"__{name}__")()

        method.__name__ = f"__{name}__"
        return method

    for op in binary:
        setattr(NumericArray, f"__{op}__", _binary(op))
        setattr(NumericArray, f"__r{op}__", _reflected(op))
        setattr(NumericArray, f"__i{op}__", _binary(op))
    for op in comparison:
        setattr(NumericArray, f"__{op}__", _binary(op))
    for op in unary:
        setattr(NumericArray, f"__{op}__", _unary(op))


_install_array_operators()

# ``__eq__`` is elementwise, so a hash would promise more than it can keep.
NumericArray.__hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _numeric_array_flatten(
    value: NumericArray,
) -> tuple[list, tuple[NumericArraySpec, str, bool]]:
    """Flatten for JAX traversal: the array, keyed by the declaration and identity.

    The aux triple every tracked class flattens to. The declaration rides along
    rather than being re-read off the child, because it is not recoverable from
    one: ``is_valid`` admits a same-kind cast, so a float32 value under a
    float64 declaration would come back declaring float32.
    """
    # The boundary presents a bare array, as a ``NumericRecord``'s does: this
    # is one of the compute boundaries native form converts at.
    return [value.as_jax()], (value._spec, value._name, value._name_is_auto)


def _numeric_array_unflatten(
    aux: tuple[NumericArraySpec, str, bool], children: list
) -> NumericArray:
    """Rebuild without converting or validating the child.

    JAX unflattens with whatever it carries, and a skeleton from
    ``tree_map(lambda x: None, value)`` or an internal sentinel is not an array
    — the reason ``Record`` and ``RecordBatch`` take ``_validate_leaves=False``
    on this path. A transform may also have resized the value, which is why the
    spec a rebuilt value carries is the one it was declared with rather than one
    read off the child: on this path a shape is transform-relative.
    """
    spec, name, name_is_auto = aux
    (array,) = children
    value = object.__new__(NumericArray)
    object.__setattr__(value, "_value", array)
    object.__setattr__(value, "_spec", spec)
    value._init_tracked(name, name_is_auto=name_is_auto)
    return value


jax.tree_util.register_pytree_node(NumericArray, _numeric_array_flatten, _numeric_array_unflatten)
