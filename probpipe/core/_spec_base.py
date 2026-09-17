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

    @property
    def is_concrete(self) -> bool:
        """Whether no symbolic dimensions remain."""
        return not self.free_dims

    def with_dims(self, **sizes: int) -> Self:
        """Return a spec with the supplied symbolic dimensions replaced.

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
                raise TypeError(f"with_dims: {name}= must be an integer") from None
            if size < 0:
                raise ValueError(f"with_dims: {name}= must be non-negative")
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
        if any(not isinstance(name, str) or not name for name in names.values()):
            raise TypeError("dimension names must be non-empty strings")
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
        self._bind_dims_from_value(value, bindings, type(self).__name__)
        return self._substitute_dims(bindings)

    def bind_dims_from_spec(self, other: TermSpec) -> Self:
        """Unify against another declaration and return the resulting spec.

        Parameters
        ----------
        other : TermSpec
            An authoritative spec. Dimensions used as sizes must be concrete;
            dimension names on the two sides are not unified as aliases.

        Returns
        -------
        Self
            This declaration with sizes learned from the other spec.

        Raises
        ------
        TypeError
            If the argument is not a spec.
        ValueError
            If kinds, structure, or repeated sizes disagree.
        """
        if not isinstance(other, TermSpec):
            raise TypeError("bind_dims_from_spec expects a TermSpec")
        bindings: dict[str, int] = {}
        _unify_specs(self, other, bindings, type(self).__name__)
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
                raise ValueError(f"{path} does not conform to its field spec ({self!r})")
            return
        raise ValueError(
            f"{path} declares {type(self).__name__}, whose dimensions this pass cannot "
            f"bind from a value; bind them with with_dims before validating against one"
        )

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind this spec's dimensions from an authoritative *actual* spec.

        The counterpart of :meth:`_bind_dims_from_value` for the path that
        validates a declaration against another declaration rather than against a
        live value. Binding follows the same rules: sizes go into the caller's
        *bindings*, a name binds once, and a disagreement raises.

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
    """A spec whose values have a flat numeric layout; this adds no kind."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """The number of scalar coordinates, defined only for concrete specs."""


@dataclass(frozen=True, eq=False, init=False)
class NumericArraySpec(NumericSpec):
    """A numeric-array value spec: an event ``shape`` plus optional metadata.

    ``dtype`` and ``support`` are optional (default ``None``); when unset the
    spec describes its shape only. Each dimension is either a fixed
    non-negative integer or a non-empty symbolic name. Repeated names must have
    the same size when a value is validated. ``dtype`` accepts any ``numpy.dtype``
    spelling (a dtype instance, a scalar type such as ``jnp.float32``, or a
    string such as ``"float32"``) and is normalised to ``numpy.dtype`` at
    construction, so equal dtypes compare and hash equal however they were
    spelled. A spec with ``dtype=None`` is **not** equal to one with a
    concrete dtype. ``support`` must be hashable when set.
    """

    shape: tuple[int | str, ...]
    dtype: np.dtype | None
    support: Constraint | None

    def __init__(
        self,
        shape: Iterable[int | str],
        dtype: npt.DTypeLike | None = None,
        support: Constraint | None = None,
    ) -> None:
        """Store the shape and metadata, normalising *shape* and *dtype*.

        The fields are the *stored* types; the wider parameters here are the
        accepted spellings, normalised away before assignment.
        """
        dimensions = tuple(shape)
        if not all(
            (isinstance(d, int) and d >= 0) or (isinstance(d, str) and bool(d)) for d in dimensions
        ):
            raise TypeError(
                "NumericArraySpec.shape must contain only non-negative ints or non-empty "
                f"symbolic dimension names, got {shape!r}"
            )
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
            raise ValueError(
                f"{path} does not conform to its field spec ({self!r}): got {type(value).__name__}"
            )
        _unify_array_shape(self.shape, actual_shape, bindings, path)
        if self.dtype is not None:
            actual_dtype = _numpy_dtype_of(value)
            if actual_dtype is None or not np.can_cast(
                actual_dtype, self.dtype, casting="same_kind"
            ):
                raise ValueError(f"{path} does not conform to its field spec ({self!r})")

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind the symbolic entries of :attr:`shape` from *actual*'s own shape."""
        if not isinstance(actual, NumericArraySpec):
            return False
        if any(isinstance(entry, str) for entry in actual.shape):
            raise ValueError(
                f"{path} has a polymorphic actual template; concrete dimensions are required"
            )
        _unify_array_shape(self.shape, actual.shape, bindings, path)
        if self.dtype is not None and actual.dtype is not None:
            if not np.can_cast(actual.dtype, self.dtype, casting="same_kind"):
                raise ValueError(f"{path} dtype {actual.dtype} does not conform to {self.dtype}")
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

    @property
    def vector_size(self) -> int:
        """The flat array size; raises ValueError while dimensions are symbolic."""
        if self.free_dims:
            raise ValueError(f"vector_size has unbound dimensions: {sorted(self.free_dims)}")
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
            self._bind_dims_from_value(value, {}, type(self).__name__)
        except ValueError:
            return False
        return True


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
) -> tuple[int, ...]:
    """Validate fixed dimensions and bind symbols against a concrete shape."""
    if len(declared) != len(actual):
        raise ValueError(
            f"{path} has rank {len(actual)}, expected rank {len(declared)} from shape {declared!r}"
        )
    concrete: list[int] = []
    for declared_dimension, actual_dimension in zip(declared, actual, strict=True):
        if not isinstance(actual_dimension, int):
            raise ValueError(f"{path} has non-concrete dimension {actual_dimension!r}")
        if isinstance(declared_dimension, int):
            if declared_dimension != actual_dimension:
                raise ValueError(
                    f"{path} has dimension {actual_dimension}, expected "
                    f"{declared_dimension} from shape {declared!r}"
                )
        else:
            previous = bindings.setdefault(declared_dimension, actual_dimension)
            if previous != actual_dimension:
                raise ValueError(
                    f"{path} binds symbolic dimension {declared_dimension!r} to "
                    f"{actual_dimension}, but it is already bound to {previous}"
                )
        concrete.append(actual_dimension)
    return tuple(concrete)


@dataclass(frozen=True)
class OpaqueSpec(TermSpec):
    """The fallback value spec, for a value no other spec describes.

    An opaque value carries no exposed structure (a string, a DataFrame, an
    arbitrary Python object, ...). ``meta`` is optional opaque metadata and
    must be hashable (or ``None``).
    """

    meta: Hashable = None

    def __post_init__(self) -> None:
        _require_hashable(self.meta, context="OpaqueSpec.meta")

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a valid opaque value — anything but a mapping.

        As the fallback spec, ``OpaqueSpec`` accepts any value **except** a
        ``Mapping``: a mapping denotes tree structure (a subtree), never a
        leaf. Every other value is valid, including a numeric array or scalar
        — such a value is *typically* described by an :class:`NumericArraySpec`, but
        an explicitly-opaque field still accepts it. ``meta`` is metadata
        about the spec and is not checked against the value.

        Notes
        -----
        The record layer honours the same rule: mappings are never leaves, so
        :class:`~probpipe.Record` construction materialises a mapping field
        value into a nested subtree.
        """
        return not isinstance(value, Mapping)
