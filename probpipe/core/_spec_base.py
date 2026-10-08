"""Kind typing and the symbolic-dimension protocol, independent of record structure."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt

from ._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
from ._repr import format_dtype, public_class_name, term_repr
from ._shapes import ShapeLike, _as_shape
from .constraints import Constraint
from .named_tree import NamedTree


def _require_hashable(value: Any, *, context: str) -> None:
    """Fail at construction when a schema component cannot be hashed."""
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{context} must be hashable: {error}") from None


class TermSpec(ABC):
    """The kind and structure of one value, independent of its label.

    Specs are immutable, hashable, and compare by declaration. Symbolic
    dimensions share a scope through nested schemas and declarations.
    """

    __slots__ = ("__weakref__",)

    @property
    def is_concrete(self) -> bool:
        """Whether no symbolic dimensions remain."""
        return not self.free_dims

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Return a new spec with symbolic dimensions replaced by the supplied sizes.

        Parameters
        ----------
        **sizes : int
            Non-negative sizes keyed by dimension name. Unmentioned dimensions
            stay symbolic; names absent from the spec are ignored.

        Returns
        -------
        Self
            The substituted spec, with other metadata preserved.

        Raises
        ------
        TypeError
            If a size is not an integer.
        ValueError
            If a size is negative.
        """
        bindings = {}
        for name, size in sizes.items():
            try:
                size = operator.index(size)
            except TypeError:
                raise TypeError(
                    f"with_dim_sizes(): size for {name!r} must be an integer, got {size!r}"
                ) from None
            if size < 0:
                raise ValueError(
                    f"with_dim_sizes(): size for {name!r} must be non-negative, got {size}"
                )
            bindings[name] = size
        return self._substitute_dims(bindings)

    def with_dim_names(self, **names: str) -> Self:
        """Return a spec with symbolic dimensions renamed simultaneously.

        Parameters
        ----------
        **names : str
            Old dimension names mapped to non-empty new names. Fixed sizes and
            unrelated metadata are preserved; unknown old names are ignored.
            Renaming two symbols to one intentionally joins their scopes.

        Returns
        -------
        Self
            The renamed spec. The original is unchanged.

        Raises
        ------
        TypeError
            If a new name is not a non-empty string.
        """
        for old, new in names.items():
            if not isinstance(new, str) or not new:
                raise TypeError(
                    f"with_dim_names(): the new name for {old!r} must be a non-empty string, "
                    f"got {new!r}"
                )
        return self._substitute_dims(names)

    def bind_dims_from_value(self, value: Any) -> Self:
        """Unify against a value and return the resulting spec.

        Parameters
        ----------
        value : Any
            The value supplying actual dimensions. All occurrences of a symbol
            must agree within this spec. Missing callable declarations leave
            their dimensions symbolic.

        Returns
        -------
        Self
            A spec with the observed dimensions bound; no input is mutated.

        Raises
        ------
        ValueError
            If kinds, structure, or repeated dimension sizes disagree.
        """
        bindings: dict[str, int] = {}
        self._bind_dims_from_value(value, bindings, "value")
        return self._substitute_dims(bindings)

    def bind_dims_from_spec(self, other: TermSpec) -> Self:
        """Unify against another declaration and return the resulting spec.

        Parameters
        ----------
        other : TermSpec
            The spec to unify with, in one scope: a symbolic dimension on either
            side binds to the size the other side gives at its axis, and a name
            both sides declare is one dimension, which may stay free.

        Returns
        -------
        Self
            This declaration with sizes learned from the other spec.

        Raises
        ------
        TypeError
            If the argument is not a spec.
        ValueError
            If kinds, structure, or repeated sizes disagree, or two different
            symbolic dimensions meet at one axis, which ``with_dim_names`` resolves.
        """
        if not isinstance(other, TermSpec):
            raise TypeError(f"bind_dims_from_spec() expects a TermSpec, got {type(other).__name__}")
        bindings: dict[str, int] = {}
        _unify_specs(self, other, bindings, "other")
        return self._substitute_dims(bindings)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound symbolic dimension names this spec declares.

        A spec that declares no shape and holds no schema has none, which is the
        default. A spec carrying a shape reports its symbolic entries; a spec
        carrying another schema reports that schema's, so a name is visible
        wherever it is declared and not only at the outermost level.

        The names are what makes a template *polymorphic*. They live in one scope
        per template: the same name in two places is one dimension, whether the
        two places are sibling fields or one inside a term spec's own schema, and
        binding resolves every occurrence together.
        """
        return frozenset()

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> Self:
        """This spec with the dimensions named in *bindings* replaced by sizes.

        The counterpart of :attr:`free_dims`: whatever a spec reports there, it
        substitutes here. A spec declaring no dimensions returns itself, which is
        the default.

        A name absent from *bindings* is **left symbolic** rather than refused.
        Binding is a refinement, and one caller — the unification pass — resolves
        names across every field before it knows which are bound, so a spec must
        be substitutable while the answer is still partial.
        """
        return self

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind the dimensions this spec declares from *value*'s own structure.

        A spec reports its dimensions in :attr:`free_dims`, substitutes them in
        :meth:`_substitute_dims`, and binds them here. Each spec owns all three, so
        a spec defined outside this module resolves its own dimensions.

        An implementation checks the kind and binds sizes into *bindings*, the
        caller's own mutable scope: a name inside a spec is the same dimension as
        that name beside it, so it binds once and a disagreement raises. Nothing
        is substituted here, since a name may be bound by a field the pass has not
        reached; the caller substitutes once the scope is closed.

        The default raises, so a spec that declares dimensions it cannot bind says
        so rather than passing silently. A spec declaring none validates the value.
        """
        if not self.free_dims:
            if not self.is_valid(value):
                raise _kind_mismatch(path, self, value)
            return
        raise ValueError(
            f"cannot check {path} against a {type(self).__name__} with symbolic dimensions "
            f"{sorted(self.free_dims)}; set their sizes with with_dim_sizes() first"
        )

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind this spec's dimensions from an authoritative *actual* spec.

        The counterpart of :meth:`_bind_dims_from_value` for the path that
        validates a declaration against another declaration rather than against a
        live value. Binding follows the same rules: sizes go into the caller's
        *bindings*, a name binds once, and a disagreement raises.

        Parameters
        ----------
        actual : TermSpec
            The spec to bind from, such as a produced term's spec.
        bindings : dict of str to int
            The caller's dimension scope, which receives each size bound.
        path : str
            The location of this spec, which an error message names.

        Returns
        -------
        bool
            Whether this spec bound from *actual*. ``False`` leaves the caller to
            compare the two specs instead, which is the default.
        """
        return False

    @abstractmethod
    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a valid value for this spec.

        Each concrete spec checks everything it declares; see its own
        ``is_valid`` docstring for the exact conditions.

        Parameters
        ----------
        value : Any
            The concrete value to check against this spec.

        Returns
        -------
        bool
            ``True`` iff *value* matches this spec; a value the spec does not
            describe returns ``False`` rather than raising. A spec swallows
            only the specific conditions that mean "does not match" (each spec
            documents its own); it does not suppress an unexpected error from
            inspecting a malformed value, so a genuine bug still surfaces.
        """


class NumericSpec(TermSpec):
    """A spec whose values have a flat numeric layout; this adds no kind.

    Subclasses implement ``_vector_size`` for concrete specs. The public
    ``vector_size`` property rejects unbound dimensions before calling it.
    """

    __slots__ = ()

    @property
    def vector_size(self) -> int:
        """The number of scalar coordinates in one value's flat numeric vector.

        Raises
        ------
        ValueError
            If the spec still has symbolic dimensions. The message lists
            the dimensions that must first be made concrete.
        """
        if not self.is_concrete:
            free = sorted(self.free_dims)
            noun = "dimension" if len(free) == 1 else "dimensions"
            sizes = ", ".join(f"{name}=..." for name in free)
            raise ValueError(
                f"vector_size needs concrete dimensions, but {type(self).__name__} has symbolic "
                f"{noun} {', '.join(free)}; set the sizes with with_dim_sizes({sizes})"
            )
        return self._vector_size()

    @abstractmethod
    def _vector_size(self) -> int:
        """Return the scalar count; called only for concrete specs."""
        raise NotImplementedError(f"{type(self).__name__}._vector_size is not implemented")


@dataclass(frozen=True, eq=False, init=False)
class NumericArraySpec(NumericSpec):
    """A numeric-array value spec: an event ``shape`` plus optional metadata.

    ``dtype`` and ``support`` are optional (default ``None``); when unset the
    spec describes its shape only. Each dimension is either a fixed
    non-negative integer or a symbolic dimension name, which is a Python
    identifier such as ``n_obs``. Repeated names must have the same size when a
    value is validated. ``dtype`` accepts any ``numpy.dtype``
    spelling (a dtype instance, a scalar type such as ``jnp.float32``, or a
    string such as ``"float32"``) and is normalised to ``numpy.dtype`` at
    construction, so equal dtypes compare and hash equal however they were
    spelled. A spec with ``dtype=None`` is **not** equal to one with a
    concrete dtype. ``support`` must be hashable when set.

    Parameters
    ----------
    shape : int, str, or iterable of int or str
        The event shape, which holds one dimension per axis and is stored as a
        tuple. A single int or str is one dimension: ``3`` means ``(3,)`` and
        ``"n"`` means ``("n",)``. A dimension that is an integer of another type,
        such as ``numpy.int64``, is stored as a Python ``int``.
    dtype : dtype-like, optional
        The dtype of the values. Stored as a ``numpy.dtype``.
    support : Constraint, optional
        The constraint the entries of a value satisfy.

    Raises
    ------
    TypeError
        If *shape* is not an int, a str, or an iterable of them, or a dimension
        is a ``bool`` or is neither an integer nor a string.
    ValueError
        If a dimension is a negative integer or a name that is not a Python
        identifier.
    """

    shape: tuple[int | str, ...]
    dtype: np.dtype | None
    support: Constraint | None

    def __init__(
        self,
        shape: ShapeLike,
        dtype: npt.DTypeLike | None = None,
        support: Constraint | None = None,
    ) -> None:
        dimensions = _as_shape(shape, what="NumericArraySpec shape")
        if support is not None:
            _require_hashable(support, context="NumericArraySpec.support")
        object.__setattr__(self, "shape", dimensions)
        object.__setattr__(self, "dtype", None if dtype is None else np.dtype(dtype))
        object.__setattr__(self, "support", support)

    @property
    def free_dims(self) -> frozenset[str]:
        """The symbolic entries of :attr:`shape`."""
        return frozenset(entry for entry in self.shape if isinstance(entry, str))

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind the symbolic entries of :attr:`shape` from *value*'s own shape.

        Every symbolic entry takes a size, since an actual array has one per axis,
        which is why an array declaration is concrete as soon as it is bound.
        """
        actual_shape = _full_array_shape_or_none(value)
        if actual_shape is None:
            raise _kind_mismatch(path, self, value)
        _unify_array_shape(self.shape, actual_shape, bindings, path)
        if self.dtype is not None:
            actual_dtype = _numpy_dtype_of(value)
            if actual_dtype is None:
                raise _kind_mismatch(path, self, value)
            if not np.can_cast(actual_dtype, self.dtype, casting="same_kind"):
                raise _dtype_mismatch(path, actual_dtype, self.dtype)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Unify the entries of :attr:`shape` with *actual*'s, symbols on both sides."""
        if not isinstance(actual, NumericArraySpec):
            return False
        _unify_array_shape(self.shape, actual.shape, bindings, path)
        if self.dtype is not None and actual.dtype is not None:
            if not np.can_cast(actual.dtype, self.dtype, casting="same_kind"):
                raise _dtype_mismatch(path, actual.dtype, self.dtype)
        return True

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> NumericArraySpec:
        """This spec with each bound entry of :attr:`shape` replaced by its size."""
        return NumericArraySpec(
            tuple(
                bindings.get(entry, entry) if isinstance(entry, str) else entry
                for entry in self.shape
            ),
            dtype=self.dtype,
            support=self.support,
        )

    def _vector_size(self) -> int:
        """The product of the concrete array dimensions."""
        return prod(cast(tuple[int, ...], self.shape))

    def __eq__(self, other: object) -> bool:
        # Mirror the dataclass-generated ``__eq__``: on a class mismatch,
        # defer to the reflected comparison (Python then falls back to
        # ``False`` when both sides decline).
        if other.__class__ is not self.__class__:
            return NotImplemented
        other = cast(NumericArraySpec, other)
        # ``numpy.dtype`` treats ``None`` as an alias for the default dtype
        # (``np.dtype(None)`` is float64), so a plain field comparison would
        # report an unset dtype equal to a concrete one. Compare set-ness
        # explicitly: unset matches only unset.
        if (self.dtype is None) != (other.dtype is None):
            return False
        return (self.shape, self.dtype, self.support) == (other.shape, other.dtype, other.support)

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype, self.support))

    def __repr__(self) -> str:
        """The shape, then the dtype and the support where they are set."""
        fields = [("shape", repr(self.shape))]
        if self.dtype is not None:
            fields.append(("dtype", format_dtype(self.dtype)))
        if self.support is not None:
            fields.append(("support", repr(self.support)))
        return term_repr(public_class_name(type(self)), None, fields)

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a numeric array (or scalar) matching this spec.

        Checks that *value* is a numeric array-like (a numeric Python scalar,
        or an object with a numeric ``dtype`` and a ``shape``) whose shape
        has the declared rank and fixed sizes (a numeric scalar has shape
        ``()``), with repeated symbolic dimensions agreeing, and whose
        dtype is **same-kind castable** to ``dtype`` when set — a widening
        promotion (e.g. ``float32`` for a ``float64`` spec) or a within-kind
        narrowing both pass, while a cross-kind conversion (e.g. a float where
        an integer dtype is declared) does not (a bare Python scalar reports
        the dtype ``np.asarray`` gives it). Strings, mappings, Python
        lists/tuples, and non-numeric arrays are invalid. Never raises on a
        mismatched value — a value the spec does not describe returns
        ``False``.

        ``support`` is **not** checked here. Unlike shape and dtype it is a
        data-dependent, element-wise check that cannot run under ``jax.jit``
        tracing, and ``is_valid`` is the check ``Record`` construction runs
        (which happens inside traces). ``support`` is therefore descriptive
        metadata on the spec; :meth:`is_valid` validates structure only and so
        runs under ``jax.jit`` unchanged.
        """
        try:
            self._bind_dims_from_value(value, {}, "value")
        except ValueError:
            return False
        return True


def _described(value: Any) -> str:
    """*value* as a message names it: an array by its shape, anything else by its type."""
    shape = getattr(value, "shape", None)
    if shape is None or getattr(value, "dtype", None) is None or isinstance(value, TermSpec):
        return type(value).__name__
    return f"an array of shape {tuple(shape)}"


def _kind_mismatch(path: str, spec: TermSpec, value: Any) -> ValueError:
    """The error for the value at *path*, which is not of the kind *spec* declares."""
    return ValueError(f"{path} does not conform to {spec!r}: got {_described(value)}")


def _name_mismatch(actual: Iterable[str], declared: Iterable[str]) -> str:
    """The declared names *actual* lacks and the names in it that are not declared."""
    actual, declared = list(actual), list(declared)
    missing = [name for name in declared if name not in actual]
    unexpected = [name for name in actual if name not in declared]
    parts = [f"missing {missing}"] if missing else []
    if unexpected:
        parts.append(f"unexpected {unexpected}")
    return ", ".join(parts)


def _dtype_mismatch(path: str, actual: np.dtype, declared: np.dtype) -> ValueError:
    """The error for the array at *path*, whose dtype *actual* cannot stand for *declared*."""
    return ValueError(f"{path} has dtype {actual}, which cannot be cast to the declared {declared}")


def _unify_specs(expected: TermSpec, actual: TermSpec, bindings: dict[str, int], path: str) -> None:
    """Check two declarations and collect dimensions in the caller's shared scope."""
    if not expected._bind_dims_from_spec(actual, bindings, path) and expected != actual:
        raise ValueError(f"{path} spec {actual!r} does not conform to {expected!r}")


def _full_array_shape_or_none(val: Any) -> tuple[int, ...] | None:
    """Return the shape of a numeric array-like value, or ``None``.

    A numeric scalar reports shape ``()`` and a numeric array reports its
    ``shape``. Resolution is registry-first: a value whose type has a
    registered :class:`~probpipe.ArrayBackend` answers through its
    ``is_numeric`` / ``event_shape`` hooks (container metadata only — values
    are not touched); everything else falls to the numpy-protocol duck path.
    Strings, object arrays, Python lists/tuples, and any remaining value
    without a numeric ``dtype`` / ``shape`` report ``None``.
    """
    if isinstance(val, (TermSpec, Mapping, NamedTree)):
        return None
    own_spec = getattr(val, "spec", None)
    if isinstance(own_spec, TermSpec) and not isinstance(own_spec, NumericArraySpec):
        return None
    return _event_shape_of(val) if _is_numeric_leaf(val) else None


def _unify_array_shape(
    declared: tuple[int | str, ...],
    actual: tuple[int | str, ...],
    bindings: dict[str, int],
    path: str,
) -> tuple[int | str, ...]:
    """Unify two shapes axis by axis in the caller's shared scope *bindings*.

    A symbolic dimension on either side binds to the size the other side gives
    at its axis, and a symbol already bound in *bindings* stands for its size.
    The same symbol on both sides is one dimension and may stay free, while two
    different symbols at one axis are different quantities until renamed to agree.

    Parameters
    ----------
    declared : tuple of int or str
        The expected shape, which an error message quotes.
    actual : tuple of int or str
        The shape found, of a value or of another declaration.
    bindings : dict of str to int
        The caller's dimension scope, which receives each size bound.
    path : str
        The location of the shape, which an error message names.

    Returns
    -------
    tuple of int or str
        The unified shape, with the dimensions still unbound left symbolic.

    Raises
    ------
    ValueError
        If the ranks differ, two sizes disagree, or two different unbound
        symbols meet at one axis.
    """
    if len(declared) != len(actual):
        raise ValueError(
            f"{path} has rank {len(actual)}, expected rank {len(declared)} from shape {declared!r}"
        )
    unified: list[int | str] = []
    for declared_dimension, actual_dimension in zip(declared, actual, strict=True):
        declared_size = (
            bindings.get(declared_dimension, declared_dimension)
            if isinstance(declared_dimension, str)
            else declared_dimension
        )
        actual_size = (
            bindings.get(actual_dimension, actual_dimension)
            if isinstance(actual_dimension, str)
            else actual_dimension
        )
        if isinstance(declared_size, int) and isinstance(actual_size, int):
            if declared_size != actual_size:
                if isinstance(declared_dimension, str):
                    raise ValueError(
                        f"{path} binds symbolic dimension {declared_dimension!r} to "
                        f"{actual_size}, but it is already bound to {declared_size}"
                    )
                if isinstance(actual_dimension, str):
                    raise ValueError(
                        f"{path} binds symbolic dimension {actual_dimension!r} to "
                        f"{declared_size}, but it is already bound to {actual_size}"
                    )
                raise ValueError(
                    f"{path} has dimension {actual_size}, expected "
                    f"{declared_size} from shape {declared!r}"
                )
            unified.append(declared_size)
        elif isinstance(declared_size, str) and isinstance(actual_size, int):
            bindings[declared_size] = actual_size
            unified.append(actual_size)
        elif isinstance(declared_size, int) and isinstance(actual_size, str):
            bindings[actual_size] = declared_size
            unified.append(declared_size)
        elif declared_size == actual_size:
            unified.append(declared_size)
        else:
            raise ValueError(
                f"{path} has symbolic dimension {actual_size!r} where {declared_size!r} is "
                f"expected; rename one with with_dim_names() so they match"
            )
    return tuple(unified)


@dataclass(frozen=True, repr=False)
class OpaqueSpec(TermSpec):
    """The fallback value spec, for a value no other spec describes.

    An opaque value carries no exposed structure, as a string, a DataFrame, or
    an arbitrary Python object does.

    Parameters
    ----------
    type : type or None
        The Python type of the values the spec admits, which :meth:`is_valid`
        checks. ``None`` admits every value that is not a mapping. Construction
        from a value infers it.
    meta : Hashable
        Free-form metadata, such as units or a tag. It is part of the spec's
        equality and hash, and it is never checked against a value or inferred,
        so ``None`` unifies with a set ``meta``, as an open ``type`` does with a
        set one.

    Raises
    ------
    TypeError
        If *type* is neither a class nor ``None``, or *meta* is not hashable.
    """

    type: type | None = None
    meta: Hashable = None

    def __post_init__(self) -> None:
        if self.type is not None and not isinstance(self.type, type):
            raise TypeError(f"OpaqueSpec type must be a class or None, got {self.type!r}")
        _require_hashable(self.meta, context="OpaqueSpec.meta")

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a valid opaque value: an instance of :attr:`type`, and no mapping.

        A ``Mapping`` denotes tree structure, a subtree rather than a leaf, so no
        opaque spec admits one. With no type every other value is valid,
        including a numeric array or scalar, which a :class:`NumericArraySpec`
        typically describes but an explicitly opaque field still accepts.
        ``meta`` is not checked against the value.

        Notes
        -----
        The record layer honours the same rule: mappings are never leaves, so
        :class:`~probpipe.Record` construction materialises a mapping field
        value into a nested subtree.
        """
        if isinstance(value, Mapping):
            return False
        return self.type is None or isinstance(_opaque_value(value), self.type)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Check another opaque spec: types and ``meta`` each equal or one of them ``None``.

        Parameters
        ----------
        actual : TermSpec
            The spec to check against this one.
        bindings : dict of str to int
            The caller's dimension scope, which an opaque spec leaves unchanged.
        path : str
            The location of this spec, which the error message names.

        Returns
        -------
        bool
            Whether *actual* is an opaque spec; ``False`` leaves the caller to compare
            the two specs instead.

        Raises
        ------
        ValueError
            If the two opaque specs do not unify.
        """
        if not isinstance(actual, OpaqueSpec):
            return False
        if not _agree(self.type, actual.type) or not _agree(self.meta, actual.meta):
            raise ValueError(f"{path} spec {actual!r} does not conform to {self!r}")
        return True

    def __repr__(self) -> str:
        """The type and the metadata where they are set, so an open spec reads ``OpaqueSpec()``."""
        return term_repr(public_class_name(type(self)), None, self._repr_arguments())

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The type and the metadata where they are set, formatted for a repr."""
        fields = []
        if self.type is not None:
            fields.append(("type", self.type.__qualname__))
        if self.meta is not None:
            fields.append(("meta", repr(self.meta)))
        return fields


def _agree(first: Any, second: Any) -> bool:
    """Whether two opaque attributes unify: equal, or one of them ``None``."""
    return first is None or second is None or first == second


def _known_type(first: OpaqueSpec, second: OpaqueSpec) -> OpaqueSpec:
    """The unification of two opaque specs that unify, which takes the known type and ``meta``."""
    unified = OpaqueSpec(
        type=first.type if first.type is not None else second.type,
        meta=first.meta if first.meta is not None else second.meta,
    )
    return first if unified == first else second if unified == second else unified


def _opaque_value(value: Any) -> Any:
    """*value*, or the value it wraps when it is a tracked opaque value, an ``Opaque``."""
    if isinstance(getattr(value, "spec", None), OpaqueSpec) and hasattr(value, "value"):
        return value.value
    return value


def _opaque_spec_of(values: Iterable[Any]) -> OpaqueSpec:
    """The opaque spec of *values*: the type they share exactly, or ``None`` when they differ.

    A tracked opaque value counts by the value it wraps, and no values share no
    type, so their spec admits any value.
    """
    types = {type(_opaque_value(value)) for value in values}
    return OpaqueSpec(type=types.pop() if len(types) == 1 else None)
