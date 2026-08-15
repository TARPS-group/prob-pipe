"""NumericArray — the tracked class of the numeric-array kind.

See design III.1.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ._array_backend import _to_jax_array
from ._immutable import Immutable
from .event_template import NumericArraySpec
from .provenance import Provenance
from .tracked import Annotated, TrackedTerm, auto_name

__all__ = ["NumericArray"]


class NumericArray(Immutable, TrackedTerm, Annotated):
    """One numeric array value, with identity.

    The tracked class of the numeric-array kind, as :class:`~probpipe.Record` is
    of the record kind: what an operation returns when its declared kind is a
    :class:`~probpipe.NumericArraySpec`. It holds a single array and carries no
    batch axes, so :attr:`shape` is the **event** shape; multiplicity lives in
    :class:`~probpipe.NumericArrayBatch`.

    Parameters
    ----------
    value : array-like
        The array this names. Converted through the registered backend, so a
        native leaf becomes a ``jax.Array`` at the same boundary every other
        numeric conversion uses.
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
        If *spec* is not a :class:`NumericArraySpec`, or if *value* does not
        convert to an array.
    ValueError
        If *value*'s shape or dtype does not satisfy *spec*.

    Notes
    -----
    **Arithmetic yields a bare array**, not a ``NumericArray``. Identity is
    attached by operations, and arithmetic is not one: a ``NumericArray`` is
    what an operation hands back and what a caller computes *from*. The full
    array surface is here — arithmetic, comparison, and the conversion hooks —
    because with no fields and no field count ``arr + 1`` has exactly one
    meaning, which is what lets :class:`~probpipe.Record` stay a container and
    carry none of it.

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
        "_name",
        "_name_is_auto",
        "_provenance",
        "_spec",
        "_value",
    )

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
        try:
            array = _to_jax_array(value)
        except Exception as exc:
            raise TypeError(
                f"NumericArray holds one numeric array; {type(value).__name__} does not convert"
            ) from exc
        if spec is None:
            spec = NumericArraySpec(shape=tuple(array.shape), dtype=array.dtype)
        elif not spec.is_valid(array):
            raise ValueError(
                f"the array does not satisfy its declaration: shape {tuple(array.shape)} and "
                f"dtype {array.dtype} against {spec}"
            )
        if name is None:
            name, name_is_auto = auto_name(name, "numericarray")
        object.__setattr__(self, "_value", array)
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(name, name_is_auto=name_is_auto, provenance=provenance)

    # -- what it holds ------------------------------------------------------

    @property
    def value(self) -> Any:
        """The array itself, untracked."""
        return self._value

    @property
    def spec(self) -> NumericArraySpec:
        """This value's own declaration."""
        return self._spec

    @property
    def shape(self) -> tuple[int, ...]:
        """The event shape. A ``NumericArray`` carries no batch axes."""
        return tuple(self._value.shape)

    @property
    def dtype(self) -> Any:
        return self._value.dtype

    @property
    def ndim(self) -> int:
        return int(self._value.ndim)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return f"NumericArray({self._value!r}, name={self.name!r})"

    # -- the array surface --------------------------------------------------
    #
    # Every operator returns what the underlying array returns, which is a bare
    # array. ``_unwrap`` lets a NumericArray appear on either side.

    @staticmethod
    def _unwrap(other: Any) -> Any:
        return other._value if isinstance(other, NumericArray) else other

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = np.asarray(self._value, dtype=dtype) if dtype is not None else np.asarray(self._value)
        return arr.copy() if copy else arr

    # JAX reads ``__jax_array__`` for ``jnp.asarray``; without it the numpy
    # hook above would win and tracing support would be lost.
    def __jax_array__(self) -> Any:
        return self._value

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __index__(self) -> int:
        return operator.index(self._value)

    def __getitem__(self, key: Any) -> Any:
        return self._value[key]

    def __iter__(self):
        return iter(self._value)


def _install_array_operators() -> None:
    """Forward the array operators to the held array, returning what it returns.

    Written once rather than spelled out per operator: there are some forty of
    them, they all do the same thing, and a hand-written list is where one gets
    forgotten. The reflected and in-place forms come from the same table — an
    in-place operator on an immutable term is the out-of-place one, and returns
    a bare array like the rest.
    """
    binary = [
        "add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod",
        "pow", "lshift", "rshift", "and", "xor", "or",
    ]  # fmt: skip
    comparison = ["lt", "le", "eq", "ne", "gt", "ge"]
    unary = ["neg", "pos", "abs", "invert"]

    def _binary(name: str):
        def method(self: NumericArray, other: Any) -> Any:
            return getattr(self._value, f"__{name}__")(NumericArray._unwrap(other))

        method.__name__ = f"__{name}__"
        return method

    def _reflected(name: str):
        def method(self: NumericArray, other: Any) -> Any:
            return getattr(self._value, f"__r{name}__")(NumericArray._unwrap(other))

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
        # An in-place operator rebinds the caller's name to the bare result; the
        # term itself is immutable and unchanged.
        setattr(NumericArray, f"__i{op}__", _binary(op))
    for op in comparison:
        setattr(NumericArray, f"__{op}__", _binary(op))
    for op in unary:
        setattr(NumericArray, f"__{op}__", _unary(op))


_install_array_operators()

# ``__eq__`` is elementwise, so the inherited ``__hash__`` would be a false
# promise: two values comparing "equal" produce an array, not a bool.
NumericArray.__hash__ = None  # type: ignore[assignment]
