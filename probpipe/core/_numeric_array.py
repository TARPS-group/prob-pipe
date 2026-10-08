"""NumericArray — the tracked class of the numeric-array kind.

See design III.1.
"""

from __future__ import annotations

import operator
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._array_backend import (
    _event_shape_of,
    _is_numeric_leaf,
    _numpy_dtype_of,
    _read_only,
    _to_jax_array,
    _to_numpy_array,
)
from ._numeric import Numeric
from ._repr import BINARY_SYMBOLS, format_dtype, format_value, grouped_label, term_repr
from ._specs import NumericArraySpec
from .provenance import Provenance
from .tracked import Annotated, TrackedTerm

__all__ = ["NumericArray"]


class NumericArray(TrackedTerm, Annotated, Numeric):
    """One numeric array value, with identity.

    The tracked class of the numeric-array kind, as :class:`~probpipe.Record` is
    of the record kind: what an operation returns when its declared kind is a
    :class:`~probpipe.NumericArraySpec`. It holds a single array and carries no
    batch axes, so :attr:`shape` is the **event** shape; multiplicity lives in
    :class:`~probpipe.NumericArrayBatch`.

    Parameters
    ----------
    label : str
        The value's label, **required**, as a :class:`~probpipe.Record`'s and an
        :class:`~probpipe.Opaque`'s are. A value carries no fields to describe it,
        so the label is what says which one it is; a class-name default would label
        every array in a pipeline alike.
    value : array-like
        The array this term holds, stored verbatim in its native form: a bare array,
        an ``xarray`` / ``pandas`` container, or any registered backend, so a
        lazy or disk-backed value stays lazy. Python numeric scalars, including
        subclasses, are normalised to a 0-d ``jax.Array``; NumPy scalars retain
        their native form and dtype. A NumPy array is marked read-only in place,
        so a write through the caller's handle or through :meth:`raw` raises
        ``ValueError``.
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
    Construction validates native arrays and NumPy scalars against metadata
    alone, deferring JAX conversion to :meth:`as_jax`. NumPy scalars retain their
    original precision in :attr:`value` and ``np.asarray(value)``. JAX conversion
    follows its x64 configuration and can round or overflow; ``float(value)``
    also goes through :meth:`as_jax`.

    It carries the full array surface: arithmetic, comparison, and the
    conversion hooks. With one value and no fields, ``arr + 1`` has a single
    meaning, which is what lets :class:`~probpipe.Record` stay a container.
    It implements :class:`~probpipe.Numeric`: its vector is the array raveled in
    row-major order, and its conversion hooks present the array itself rather
    than that vector, so NumPy and JAX functions see its shape and return bare
    arrays.

    An operator returns a ``NumericArray`` holding the stored value's result,
    labeled by the expression in evaluation order, such as ``draw + 1``, with its
    tracked operands as the provenance's parents. A term presents as its raw
    representation inside a JAX trace, so there an operator returns the bare
    result. Indexing and iteration return the stored value's entries and rows.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> value = NumericArray("draw", jnp.arange(3.0))
    >>> value.shape
    (3,)
    >>> value + 1
    NumericArray('draw + 1', shape=(3,), dtype=float32)
    """

    __slots__ = (
        "_annotations",
        "_jax_cache",
        "_label",
        "_provenance",
        "_spec",
        "_value",
    )

    #: Derived from the value rather than transported, as for ``NumericRecord``.
    _transient_state = ("_jax_cache",)

    def __init__(
        self,
        label: str,
        value: Any,
        /,
        *,
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
        stored = _stored(value)
        if spec is None:
            spec = _inferred_spec(stored)
        elif not spec.is_valid(stored):
            raise ValueError(
                f"the array does not satisfy its declaration: shape {_event_shape_of(stored)} "
                f"and dtype {_numpy_dtype_of(stored)} against {spec}"
            )
        object.__setattr__(self, "_value", _read_only(stored))
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(label, provenance=provenance)

    @classmethod
    def _view(
        cls, label: str, value: Any, spec: NumericArraySpec, provenance: Provenance | None
    ) -> NumericArray:
        """The array *value* under *spec*, as a container's view of it, without validation.

        The container validated the value against *spec* when it was built, and
        under a JAX transform a shape is transform-relative, so checking it again
        would refuse a value the container holds. A Python scalar is normalized as
        the constructor normalizes it.
        """
        view = object.__new__(cls)
        object.__setattr__(view, "_value", _stored(value))
        object.__setattr__(view, "_spec", spec)
        view._init_tracked(label, provenance=provenance)
        return view

    # -- what it holds ------------------------------------------------------

    @property
    def value(self) -> Any:
        """The stored value, in the form it was given. Untracked."""
        return self._value

    def raw(self) -> Any:
        """The stored array, in the form it was given, as :attr:`value` holds it."""
        return self._value

    def as_jax(self) -> Any:
        """The value as a ``jax.Array`` — the single conversion point.

        A value already stored as one passes through, tracers included; a
        native container converts through its registered backend. Concrete
        conversions are memoised; traced conversions stay within their transform.

        Conversion follows JAX's x64 configuration, so a stored float64 can
        become float32 and round or overflow when x64 is disabled. Enable x64
        before the first conversion when float64 is required; an existing
        concrete cache is reused even if that configuration later changes.
        """
        if isinstance(self._value, jax.Array):
            return self._value
        cached = getattr(self, "_jax_cache", None)
        if cached is None:
            cached = _to_jax_array(self._value)
            if not isinstance(cached, jax.core.Tracer):
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

    # -- 1-D vector conversion ----------------------------------------------

    @property
    def vector_size(self) -> int:
        """Length of this array's 1-D vector, the number of its elements."""
        return prod(self.shape)

    def to_vector(self) -> jax.Array:
        """Serialize to the dense 1-D vector of shape ``(vector_size,)``, in row-major order.

        The value converts to ``jax.Array`` at the compute boundary, as
        :meth:`as_jax` does. The inverse is :meth:`from_vector`.
        """
        return jnp.reshape(self.as_jax(), -1)

    @classmethod
    def from_vector(cls, label: str, spec: NumericArraySpec, vec: Any) -> NumericArray:
        """Reconstruct a single array from its dense 1-D vector.

        The value-level inverse of :meth:`to_vector`: reshapes *vec* to the shape
        *spec* declares, casts it to the declared dtype when there is one, and
        returns a ``NumericArray`` carrying *spec* under *label*. The rebuilt
        value is a bare ``jax.Array``, since a flat vector carries no native
        container to restore.

        Parameters
        ----------
        label : str
            The reconstructed array's label.
        spec : NumericArraySpec
            The declaration supplying the shape and dtype, with every dimension
            bound.
        vec : Array
            A vector of shape ``(spec.vector_size,)``, one unbatched value.

        Returns
        -------
        NumericArray
            The reconstructed array, whose ``to_vector()`` equals *vec*.

        Raises
        ------
        TypeError
            If *vec* is not one-dimensional; a batch of vectors belongs to
            :class:`~probpipe.NumericArrayBatch`.
        ValueError
            If *spec* has unbound dimensions, or the vector's length is not
            ``spec.vector_size``.
        """
        vec = jnp.asarray(vec)
        if vec.ndim != 1:
            raise TypeError(
                f"NumericArray.from_vector expects a 1-D vector (one value); "
                f"got shape {tuple(vec.shape)}"
            )
        if vec.shape[0] != spec.vector_size:
            raise ValueError(
                f"NumericArray.from_vector: the vector has length {vec.shape[0]}, "
                f"expected vector_size={spec.vector_size}"
            )
        value = jnp.reshape(vec, spec.shape)
        if spec.dtype is not None:
            value = value.astype(spec.dtype)
        return cls(label, value, spec=spec)

    def __len__(self) -> int:
        return len(self._value)

    def __str__(self) -> str:
        """The stored value's string, as ``print`` and an f-string show the array."""
        return str(self._value)

    def __format__(self, format_spec: str) -> str:
        """The stored value formatted by *format_spec*, as ``f"{x:.2f}"`` formats a scalar."""
        return format(self._value, format_spec)

    def __repr__(self) -> str:
        """The label, then the declared shape and dtype, and the support when one is declared.

        A declaration with an open dtype shows the stored value's dtype. The repr
        reads the spec and the value's metadata alone.
        """
        spec = self._spec
        dtype = spec.dtype if spec.dtype is not None else self.dtype
        fields = [("shape", repr(tuple(spec.shape))), ("dtype", format_dtype(dtype))]
        if spec.support is not None:
            fields.append(("support", repr(spec.support)))
        return term_repr("NumericArray", self.label, fields)

    # -- the array surface --------------------------------------------------

    # The coordinate protocols present the array itself, not Numeric's flat vector.
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


def _stored(value: Any) -> Any:
    """*value* as a ``NumericArray`` stores it.

    A Python numeric scalar, a subclass included, becomes a 0-d ``jax.Array``,
    and any other value is kept as it is. A NumPy scalar keeps its form, since
    ``np.float64`` and ``np.complex128`` also inherit ``float`` and ``complex``.
    """
    if isinstance(value, (int, float, complex, bool)) and not isinstance(value, np.generic):
        return _to_jax_array(value)
    return value


def _inferred_spec(value: Any) -> NumericArraySpec:
    """The spec a ``NumericArray`` of the numeric leaf *value* declares when it is given none.

    The spec states the shape and the dtype of the stored value and leaves the
    support unset.
    """
    stored = _stored(value)
    return NumericArraySpec(shape=_event_shape_of(stored), dtype=_numpy_dtype_of(stored))


def _unwrap(other: Any) -> Any:
    """The array inside a ``NumericArray``, or *other* unchanged."""
    return other._value if isinstance(other, NumericArray) else other


#: The form each unary operator gives the label its result derives.
_UNARY_FORMS = {"neg": "-{}", "pos": "+{}", "abs": "abs({})", "invert": "~{}"}


def _operand_label(operand: Any) -> str:
    """How *operand* reads in a derived label: its grouped label, or its value when untracked."""
    if isinstance(operand, NumericArray):
        return grouped_label(operand.label)
    return format_value(operand)


def _tracked_result(value: Any, label: str, operator_name: str, operands: tuple[Any, ...]) -> Any:
    """The operator's *value* as a ``NumericArray`` labeled *label*.

    Its tracked *operands* are its parents. The result declares its value's
    shape, and its value's dtype when every tracked operand declares a dtype, so
    it declares as much as its operands do. A traced value is returned bare,
    since a term presents as its raw representation inside a JAX trace (II.4),
    and a value that is not numeric, ``NotImplemented`` among them, is returned
    as it is.
    """
    if isinstance(value, jax.core.Tracer) or not _is_numeric_leaf(value):
        return value
    parents = [operand for operand in operands if isinstance(operand, TrackedTerm)]
    declared = all(
        operand.spec.dtype is not None for operand in parents if isinstance(operand, NumericArray)
    )
    dtype = _numpy_dtype_of(value) if declared else None
    return NumericArray(
        label,
        value,
        spec=NumericArraySpec(_event_shape_of(value), dtype),
        provenance=Provenance.create(operator_name, parents=parents),
    )


def _install_array_operators() -> None:
    """Install the array operators, each applying the stored value's operator and tracking its result.

    Installed from a table so the forty of them stay one rule. An in-place
    operator on an immutable term is the out-of-place one. ``divmod`` returns
    the stored value's pair, since a pair is no numeric array.
    """

    def _binary(name: str):
        symbol = BINARY_SYMBOLS.get(name)

        def method(self: NumericArray, other: Any) -> Any:
            value = getattr(self._value, f"__{name}__")(_unwrap(other))
            if symbol is None:
                return value
            label = f"{_operand_label(self)} {symbol} {_operand_label(other)}"
            return _tracked_result(value, label, f"__{name}__", (self, other))

        method.__name__ = f"__{name}__"
        return method

    def _reflected(name: str):
        symbol = BINARY_SYMBOLS.get(name)

        def method(self: NumericArray, other: Any) -> Any:
            value = getattr(self._value, f"__r{name}__")(_unwrap(other))
            if symbol is None:
                return value
            label = f"{_operand_label(other)} {symbol} {_operand_label(self)}"
            return _tracked_result(value, label, f"__r{name}__", (other, self))

        method.__name__ = f"__r{name}__"
        return method

    def _unary(name: str):
        form = _UNARY_FORMS[name]

        def method(self: NumericArray) -> Any:
            value = getattr(self._value, f"__{name}__")()
            # A call form brackets its operand already.
            operand = self.label if form.endswith("({})") else _operand_label(self)
            return _tracked_result(value, form.format(operand), f"__{name}__", (self,))

        method.__name__ = f"__{name}__"
        return method

    binary = [
        "add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod",
        "pow", "lshift", "rshift", "and", "xor", "or",
    ]  # fmt: skip
    for op in binary:
        setattr(NumericArray, f"__{op}__", _binary(op))
        setattr(NumericArray, f"__r{op}__", _reflected(op))
        setattr(NumericArray, f"__i{op}__", _binary(op))
    for op in ("lt", "le", "eq", "ne", "gt", "ge"):
        setattr(NumericArray, f"__{op}__", _binary(op))
    for op in _UNARY_FORMS:
        setattr(NumericArray, f"__{op}__", _unary(op))


_install_array_operators()

# ``__eq__`` is elementwise, so a hash would promise more than it can keep.
NumericArray.__hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _numeric_array_flatten(
    value: NumericArray,
) -> tuple[list, tuple[NumericArraySpec, str]]:
    """Flatten for JAX traversal: the array, keyed by the declaration and identity.

    The aux pair every tracked class flattens to. The declaration rides along
    rather than being re-read off the child, because it is not recoverable from
    one: ``is_valid`` admits a same-kind cast, so a float32 value under a
    float64 declaration would come back declaring float32.
    """
    # The boundary presents a bare array, as a ``NumericRecord``'s does: this
    # is one of the compute boundaries native form converts at.
    return [value.as_jax()], (value._spec, value._label)


def _numeric_array_unflatten(aux: tuple[NumericArraySpec, str], children: list) -> NumericArray:
    """Rebuild without converting or validating the child.

    JAX unflattens with whatever it carries, and a skeleton from
    ``tree_map(lambda x: None, value)`` or an internal sentinel is not an array
    — the reason ``Record`` and ``RecordBatch`` take ``_validate_leaves=False``
    on this path. A transform may also have resized the value, which is why the
    spec a rebuilt value carries is the one it was declared with rather than one
    read off the child: on this path a shape is transform-relative.
    """
    spec, name = aux
    (array,) = children
    value = object.__new__(NumericArray)
    object.__setattr__(value, "_value", array)
    object.__setattr__(value, "_spec", spec)
    value._init_tracked(name)
    return value


jax.tree_util.register_pytree_node(NumericArray, _numeric_array_flatten, _numeric_array_unflatten)
