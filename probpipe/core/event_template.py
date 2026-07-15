"""EventTemplate — ProbPipe's structural schema.

An :class:`EventTemplate` describes the **structure** of a value, independent of
the data itself. In particular, an event template describes the structure of a
Python object that takes the form of a named tree: a nested object with unique
string key paths to each leaf. We refer to the leaves of the tree as *fields*.
Thus, an event template can be thought of as the schema for a set of ordered
named fields, where the fields are allowed to be stored in an object with nested
structure.

An event template is designed to be quite general, able to describe the
structure of a single array or a complicated nested object storing arbitrary
Python objects. The restriction is that there must be a sequence of strings
forming a unique path to each field. This follows ProbPipe's convention of
working with *names* in most cases to avoid ambiguity. Even the event template
for a single object with no nested structure describes a *named* field: the tree
is a root node with a single named leaf. That field name is still required,
though ProbPipe's higher-level constructors will often supply one automatically
when it would be inconvenient for the user to (for example, a scalar draw is
named after the distribution that produced it).

Field names are required and unique within a node; ``/`` is reserved as the path
separator, so every leaf has a unique ``/``-delimited string path
(e.g., "a/b/c"). The canonical leaf order is depth-first in insertion order.

In order to define the structure of trees consistently, :class:`EventTemplate`
clearly defines which objects are considered leaves and which are considered
internal nodes in the tree. The rule is intentionally restrictive for clarity:
- **non-leaf node**: an ``EventTemplate``.
- **leaf node**: a :class:`ValueSpec`.

A :class:`ValueSpec` describes the structure of **one value** — it says: "the
object at this path is a leaf of the tree, and it has this structure". A spec
carries no name of its own; it becomes a *field* only once a template gives it
one. Every spec answers :meth:`ValueSpec.is_valid`, which checks whether a
concrete value matches the spec. The specs for certain value types may
contain lots of useful structure (e.g., shape and dtype for arrays), while
others may expose no structure at all (e.g., an opaque Python object).
The concrete specs are as follows:
- :class:`ArraySpec`: describes a numeric array (shape, optional dtype/support)
- :class:`DistributionSpec`: describes a ``Distribution``. Carries a sub-template
  describing the structure of one sample from the distribution.
- :class:`FunctionSpec`: describes a callable. Optionally carries sub-templates
  for the structure of the function's inputs and outputs.
- :class:`OpaqueSpec`: fallback for any other object (no structure exposed).

Numeric vs. Mixed
-----------------

When every field is an ``ArraySpec`` the template is all-numeric, and
``EventTemplate(...)`` auto-promotes to :class:`NumericEventTemplate` — the
specialization describing a value that is a PyTree of arrays. That subclass
adds the flat-vector layout — ``vector_size`` and :meth:`from_vector`
reconstruction; the ``to_vector`` serialization is a value method on
:class:`~probpipe.NumericRecord` / :class:`~probpipe.NumericRecordArray`. See
its docstring for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ._array_backend import array_backend_for
from .constraints import Constraint
from .named_tree import (
    _PATH_SEP,
    NamedTree,
    _check_no_path_sep,
    _unflatten_paths,
)

__all__ = [
    "ArraySpec",
    "DistributionSpec",
    "EventTemplate",
    "FunctionSpec",
    "NumericEventTemplate",
    "OpaqueSpec",
    "ValueSpec",
]


class ValueSpec(ABC):
    """The structure of one leaf value — the base of the concrete specs.

    A ``ValueSpec`` describes what a single value looks like; the concrete
    specs (see the module docstring for the catalog) cover numeric arrays,
    distributions, callables, and opaque Python objects. These are the
    leaves of an :class:`EventTemplate`.

    A spec carries no name: it becomes a *field* only when an
    :class:`EventTemplate` stores it under a key. Every subclass **must** be
    a frozen, hashable dataclass comparing by value — a template hashes its
    specs (e.g. as a jit cache key), so an unhashable or mutable spec would
    break every template that stores it.
    """

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


@dataclass(frozen=True, eq=False)
class ArraySpec(ValueSpec):
    """A numeric-array value spec: a fixed event ``shape`` plus optional metadata.

    ``dtype`` and ``support`` are optional (default ``None``); when unset the
    spec describes its shape only. ``dtype`` accepts any ``numpy.dtype``
    spelling (a dtype instance, a scalar type such as ``jnp.float32``, or a
    string such as ``"float32"``) and is normalised to ``numpy.dtype`` at
    construction, so equal dtypes compare and hash equal however they were
    spelled. A spec with ``dtype=None`` is **not** equal to one with a
    concrete dtype. ``support`` must be hashable when set.
    """

    shape: tuple[int, ...]
    # ``DTypeLike`` types the constructor *input* (a dtype instance, scalar
    # type, or string); ``__post_init__`` normalises it, so the stored value
    # is always ``np.dtype | None``.
    dtype: npt.DTypeLike | None = None
    support: Constraint | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not all(isinstance(d, int) and d >= 0 for d in shape):
            raise TypeError(
                f"ArraySpec.shape must be a tuple of non-negative ints, got {self.shape!r}"
            )
        object.__setattr__(self, "shape", shape)
        if self.dtype is not None:
            object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def __eq__(self, other: object) -> bool:
        # Mirror the dataclass-generated ``__eq__``: on a class mismatch,
        # defer to the reflected comparison (Python then falls back to
        # ``False`` when both sides decline).
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, ArraySpec)  # narrow for the type checker
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
        equals ``shape`` exactly (a numeric scalar has shape ``()``), and whose
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
        shape = _full_array_shape_or_none(value)
        if shape is None or shape != self.shape:
            return False
        if self.dtype is not None:
            backend = array_backend_for(value)
            if backend is not None:
                # A registered container answers from metadata; ``None``
                # means it has no single dtype (a heterogeneous frame), which
                # cannot match a dtype-pinned spec.
                actual = backend.numpy_dtype(value)
                if actual is None:
                    return False
            else:
                actual = getattr(value, "dtype", None)
                if actual is None:
                    actual = np.asarray(value).dtype
            # A same-kind cast (a widening promotion or a within-kind
            # narrowing) matches; a cross-kind cast (e.g. float where an int
            # dtype is declared) does not.
            if not np.can_cast(np.dtype(actual), self.dtype, casting="same_kind"):
                return False
        return True


@dataclass(frozen=True)
class OpaqueSpec(ValueSpec):
    """The fallback value spec, for a value no other spec describes.

    An opaque value carries no exposed structure (a string, a DataFrame, an
    arbitrary Python object, ...). ``meta`` is optional opaque metadata and
    must be hashable (or ``None``).
    """

    meta: Hashable = None

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a valid opaque value — anything but a mapping.

        As the fallback spec, ``OpaqueSpec`` accepts any value **except** a
        ``Mapping``: a mapping denotes tree structure (a subtree), never a
        leaf. Every other value is valid, including a numeric array or scalar
        — such a value is *typically* described by an :class:`ArraySpec`, but
        an explicitly-opaque field still accepts it. ``meta`` is metadata
        about the spec and is not checked against the value.

        Notes
        -----
        The record layer honours the same rule: mappings are never leaves, so
        :class:`~probpipe.Record` construction materialises a mapping field
        value into a nested subtree (as ``from_nested_dict`` does).
        """
        return not isinstance(value, Mapping)


@dataclass(frozen=True)
class DistributionSpec(ValueSpec):
    """A value spec for a ``Distribution``.

    ``event_template`` is the :class:`EventTemplate` of one draw from that
    distribution.

    Raises
    ------
    TypeError
        If ``event_template`` is not an :class:`EventTemplate`.
    """

    event_template: EventTemplate

    def __post_init__(self) -> None:
        if not isinstance(self.event_template, EventTemplate):
            raise TypeError(
                f"DistributionSpec.event_template must be an EventTemplate, "
                f"got {type(self.event_template).__name__}"
            )

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``Distribution`` whose draws match ``event_template``.

        *value* must be a :class:`~probpipe.Distribution` whose own
        ``event_template`` equals this spec's. A distribution that is not one,
        or that legitimately exposes no template — no ``event_template``
        attribute, or a template that cannot yet be derived — does not satisfy
        the spec and returns ``False``. These are the only two "schema
        unavailable" conditions treated as a non-match; any *other* error
        raised while reading ``event_template`` signals a malfunctioning
        distribution and is left to propagate rather than being masked as
        invalid.
        """
        from ._distribution_base import Distribution

        if not isinstance(value, Distribution):
            return False
        try:
            template = value.event_template
        except (AttributeError, TypeError):
            # The two documented "schema unavailable" signals: no
            # ``event_template`` attribute (AttributeError) or a template that
            # cannot be derived (TypeError — e.g. an un-named auto-deriving
            # distribution). Both mean the value can't be certified. A
            # narrower catch than ``Exception`` on purpose: an unexpected
            # error is a bug to surface, not a silent "invalid".
            return False
        return template == self.event_template


@dataclass(frozen=True)
class FunctionSpec(ValueSpec):
    """A value spec for a callable, optionally typed by its input/output structure.

    ``input_template`` / ``output_template`` are the :class:`EventTemplate`\\ s
    of the callable's input and output, and each is optional (default
    ``None``). A template describes a structured signature: its named fields
    are the callable's inputs (or outputs), so
    ``FunctionSpec(EventTemplate(x=(), y=(3,)), EventTemplate(out=()))`` types
    ``f(x, y) -> out``. ``None`` means that side's structure is unspecified,
    so a bare ``FunctionSpec()`` describes *any* callable — the natural spec
    for a function of unknown or variable signature.

    Each specified side must be an :class:`EventTemplate`; a single-field
    signature is written out explicitly, e.g.
    ``FunctionSpec(EventTemplate(x=()), EventTemplate(out=()))``, rather than
    passed as a bare :class:`ValueSpec` — so a function's field names are
    always caller-chosen and meaningful, not a fixed ``"input"`` / ``"output"``
    placeholder. This matches :class:`DistributionSpec`, which likewise takes an
    explicit template.

    Raises
    ------
    TypeError
        If either side is not ``None`` or an :class:`EventTemplate`.
    """

    input_template: EventTemplate | None = None
    output_template: EventTemplate | None = None

    def __post_init__(self) -> None:
        for attr in ("input_template", "output_template"):
            template = getattr(self, attr)
            if template is not None and not isinstance(template, EventTemplate):
                raise TypeError(
                    f"FunctionSpec.{attr} must be None or an EventTemplate, "
                    f"got {type(template).__name__}"
                )

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a callable.

        The input/output structure of a bare callable cannot be inspected, so
        validity is callability alone; ``input_template`` / ``output_template``
        document the intended signature but are not checked against the value.
        """
        return callable(value)


# A stored field spec is a ``ValueSpec`` leaf or a nested ``EventTemplate``;
# ``_FieldSpecInput`` also admits the construction-time sugar the constructor
# normalises (a bare shape tuple or ``None``).
type _FieldSpec = ValueSpec | EventTemplate
type _FieldSpecInput = _FieldSpec | tuple[int, ...] | None


def _to_spec(spec: _FieldSpecInput) -> _FieldSpec:
    """Normalise a constructor input to a stored field spec.

    Construction-time sugar (preserved): a bare shape ``tuple`` becomes an
    :class:`ArraySpec`, ``None`` becomes an :class:`OpaqueSpec`, and a nested
    :class:`EventTemplate` is kept as-is. Already-built specs pass through, so
    new code may supply explicit ``ArraySpec(...)`` / ``OpaqueSpec(...)`` etc.
    """
    if isinstance(spec, (ValueSpec, EventTemplate)):
        return spec
    if spec is None:
        return OpaqueSpec()
    if isinstance(spec, tuple):
        return ArraySpec(shape=spec)
    raise TypeError(
        f"spec must be a shape tuple, None, a ValueSpec, or an "
        f"EventTemplate, got {type(spec).__name__}"
    )


def _is_numeric_spec(spec: Any) -> bool:
    """A numeric leaf: an :class:`ArraySpec` or a (nested) :class:`NumericEventTemplate`.

    A mapping spec (nested structure, not yet materialised into a nested
    template) counts as numeric iff all of its own values are — so
    auto-promotion sees through nesting introduced by ``/``-keys or a nested
    ``dict``.
    """
    if isinstance(spec, Mapping):
        return _all_numeric(spec.values())
    return isinstance(spec, (ArraySpec, NumericEventTemplate))


def _all_numeric(specs: Iterable[Any]) -> bool:
    """True iff every (raw, pre-normalisation) input spec is numeric.

    Drives the base-class auto-promotion hook so ``EventTemplate(x=(), y=(3,))``
    returns a ``NumericEventTemplate`` without opting in explicitly. Raw inputs
    also allow the shape-tuple sugar; ``None``, every non-``ArraySpec`` spec,
    mixed nested templates, and any unsupported type are non-numeric
    (``__init__`` rejects the latter).
    """
    return all(isinstance(s, tuple) or _is_numeric_spec(s) for s in specs)


# dtype.kind codes for numeric arrays: b=bool, i=int, u=uint, f=float, c=complex.
_NUMERIC_KINDS = frozenset("biufc")


def _is_numeric_dtype(dtype: Any) -> bool:
    """Whether *dtype* is a numeric dtype — the shared numeric-gate predicate.

    Covers the standard numpy kinds (bool, int, uint, float, complex) plus
    the ml_dtypes extension types JAX registers (``bfloat16``, the
    ``float8_*`` family, ``int4`` / ``uint4``), which numpy reports as kind
    ``"V"``. A dtype that is **not** ``numpy.dtype``-coercible — a pandas
    extension / masked dtype such as ``Int64Dtype`` — is **not** numeric
    here: it reports a numpy ``kind`` but is not a dense numpy dtype, so a
    container using it is not a plain numeric array (it carries NA / masking
    semantics ``jax`` cannot represent) and must not promote to a
    ``NumericRecord``. Structured (record) dtypes are likewise not numeric.
    Every place that decides "is this array numeric?" — template inference,
    spec validation, and the ``NumericRecord`` / ``NumericRecordArray`` leaf
    gates — routes through this predicate so the sites cannot drift apart.
    """
    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        # ``np.dtype`` raises ``TypeError`` for a pandas extension / masked
        # dtype and ``ValueError`` for an object it cannot interpret at all
        # (e.g. a mock). Either way it is not a dense numpy dtype, so not
        # numeric.
        return False
    kind = np_dtype.kind
    if kind in _NUMERIC_KINDS:
        return True
    return kind == "V" and jnp.issubdtype(np_dtype, jnp.number)


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
    if isinstance(val, (bool, int, float, complex, np.integer, np.floating, np.bool_)):
        return ()
    backend = array_backend_for(val)
    if backend is not None:
        return tuple(backend.event_shape(val)) if backend.is_numeric(val) else None
    if hasattr(val, "shape") and hasattr(val, "dtype") and _is_numeric_dtype(val.dtype):
        return tuple(val.shape)
    return None


# ---------------------------------------------------------------------------
# EventTemplate — structural skeleton
# ---------------------------------------------------------------------------


class EventTemplate(NamedTree[ValueSpec]):
    """Structural description of a value: its named, possibly-nested leaf structure.

    An ``EventTemplate`` describes the **structure** of a value as a **named
    tree** — an insertion-ordered map of named fields whose only internal node
    is a nested ``EventTemplate`` and whose leaves are value specs. It is the
    schema of a :class:`~probpipe.Record` (the value type with the same
    named-tree shape), **not** a description of an arbitrary JAX PyTree (see
    *Terminology* and *JAX pytree contract* below).

    The word *event* follows probabilistic-programming usage and **generalizes**
    the ``event`` / ``event_shape`` notion from other PPLs (TensorFlow
    Probability, distrax, NumPyro). There, ``event_shape`` is the shape of a
    single draw of one array-valued random variable. ProbPipe supports
    distributions over general value types, not just arrays. The *event* in this
    context can thus be a structured Python object, with structure described by
    the ``EventTemplate``.

    Terminology
    -----------
    Used precisely throughout this class:

    - **field** — one named object in the collection (here, a value spec),
      addressed by its full ``/``-delimited **key** (path from the root, e.g.
      ``"physics/mass"``; a single name for a flat template). The mapping
      protocol (:meth:`keys` / :meth:`values` / :meth:`items` / iteration /
      ``len`` / ``in`` / ``[]``) ranges over the fields, keyed by path.
    - **leaf** — a *terminal* node: a :class:`ValueSpec`. A nested
      ``EventTemplate`` is an *internal node*, not a leaf; the fields are the
      leaves.
    - **key vs. path** — a **key** addresses a field (a leaf); a **path** may
      also address an interior node. The mapping operators (``[]`` / ``in`` /
      iteration) are leaf-keyed, so a partial path is *not* a member and
      ``template["physics"]`` (a subtree) raises ``KeyError`` — reach a subtree
      with :meth:`at_path`, and use :attr:`children` for the one-level view. The
      same path strings index a template or the value it describes
      (``template["physics/mass"]`` / ``record["physics/mass"]``); this
      collection protocol is shared with :class:`~probpipe.Record`.
    - **canonical leaf order** — the order in which leaves are traversed:
      depth-first, following each level's insertion order. This is the single
      ordering every leaf-wise operation uses. :meth:`keys` is its canonical
      definition — it returns the key (path) of every leaf in this order;
      the value-level ``to_vector`` and :meth:`from_vector` lay out and read
      leaves in it, and
      :attr:`~NumericEventTemplate.leaf_shapes` is keyed by it.

    JAX pytree contract
    -------------------
    An ``EventTemplate`` is **not** a registered JAX pytree node — its value specs
    are atomic, so ``jax.tree_util.tree_leaves(template) == [template]``. It is
    the *schema* of the value pytrees it describes, not a pytree itself (think of
    it as an enriched ``PyTreeDef`` that also carries each leaf's kind / shape).

    For a value ``v`` it describes (a :class:`~probpipe.Record`): a nested
    ``EventTemplate`` mirrors a nested ``Record`` (both internal nodes), and each
    value spec mirrors one field value. When every leaf is an array (the
    :class:`NumericEventTemplate` / :class:`~probpipe.NumericRecord` case),
    ``jax.tree_util.tree_leaves(v)`` returns the leaves in :meth:`keys`
    order. The one place the template's leaves and JAX's diverge is an
    :class:`OpaqueSpec` leaf whose value is *itself* a JAX container (a ``tuple``
    / ``list``; a ``dict`` is never a leaf — mappings denote tree structure):
    the template counts it as a single leaf while JAX descends into it. See
    :class:`~probpipe.Record` for the full statement.

    Parameters
    ----------
    **field_specs
        Named fields. Each value is one of:

        - ``tuple[int, ...]`` — shape of a numeric array leaf (e.g. ``()`` for a
          scalar, ``(3,)`` for a 3-vector); normalised to :class:`ArraySpec`.
        - ``None`` — opaque (non-array) leaf; normalised to :class:`OpaqueSpec`.
        - a :class:`ValueSpec` — an already-built spec (passed through).
        - ``EventTemplate`` — a nested sub-structure (an internal node).

    Examples
    --------
    ::

        EventTemplate(x=(), y=(3,))                     # -> NumericEventTemplate
        EventTemplate(label=None, x=())                 # -> EventTemplate (mixed)
        EventTemplate(physics=EventTemplate(force=(), mass=()), obs=())

    Notes
    -----
    Inspired by JAX's ``PyTreeDef``: a template can reconstruct a value from its
    leaves and describes the expected structure for type-checking and
    vectorization. Leaves are stored as frozen, hashable spec objects, so a
    template is itself hashable (usable as a jit / treedef cache key).
    ``__getitem__`` returns the stored value spec (and raises on an interior
    node — see *Terminology*); the enumeration of leaves is :meth:`keys`, and
    per-leaf array shapes (on a numeric template) live on
    :attr:`~NumericEventTemplate.leaf_shapes`.

    Calling ``EventTemplate(...)`` directly auto-promotes to a
    :class:`NumericEventTemplate` when every spec is numeric (and every nested
    sub-template is itself all-numeric), so :attr:`vector_size` and
    :attr:`~NumericEventTemplate.leaf_shapes` are reachable in the common all-numeric case
    without naming the subclass. Mixed templates (any opaque / ``None`` spec)
    stay plain ``EventTemplate`` and do not expose :attr:`vector_size` — it is
    not a meaningful quantity once opaque leaves are present.
    """

    __slots__ = ("_tree",)

    def __new__(
        cls,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericEventTemplate(...)`` calls bypass this path
        # and run their own strict validation.
        if cls is EventTemplate:
            specs = _field_specs if _field_specs is not None else field_specs
            if specs and _all_numeric(specs.values()):
                return object.__new__(NumericEventTemplate)
        return object.__new__(cls)

    def __init__(
        self,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        if _field_specs is not None:
            if field_specs:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            nested = _unflatten_paths(_field_specs)
        else:
            for name in field_specs:
                _check_no_path_sep(name)
            nested = dict(field_specs)
        if not nested:
            raise ValueError(f"{type(self).__name__} requires at least one field")
        specs: dict[str, _FieldSpec] = {}
        for name, spec in nested.items():
            if isinstance(spec, Mapping):
                # A mapping spec is nested structure: materialise a subtree.
                specs[name] = EventTemplate(spec)
            else:
                try:
                    converted = _to_spec(spec)
                except TypeError as exc:
                    raise TypeError(f"Field {name!r}: {exc}") from None
                if not isinstance(converted, EventTemplate):
                    self._check_leaf(name, converted)
                specs[name] = converted
        self._post_validate(specs)
        object.__setattr__(self, "_tree", specs)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        """Subclass hook for stricter spec validation. No-op on the base."""
        return

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __reduce__(self):
        return (_unpickle_event_template, (dict(self._tree),))

    # -- Tree structure -----------------------------------------------------
    #
    # The mapping and path-navigation methods (``keys`` / ``values`` /
    # ``items`` / ``[]`` / ``at_path`` / ``children``) are inherited from
    # :class:`~probpipe.core.named_tree.NamedTree`. A leaf here is a
    # :class:`ValueSpec`; an internal node is a nested ``EventTemplate``.

    @classmethod
    def _node_type(cls) -> type:
        return EventTemplate

    @classmethod
    def _leaf_type(cls) -> type:
        # Every leaf of a template is a value spec; construction converts
        # the shorthand forms (shapes, None, ...) via ``_to_spec`` first,
        # so the substrate check validates the converted leaf.
        return ValueSpec

    @classmethod
    def _rebuild_class(cls) -> type:
        # Structural edits rebuild through the base class so ``__new__``
        # re-decides the numeric auto-promotion from the edited specs: an
        # all-numeric result promotes to ``NumericEventTemplate`` and a mixed
        # one stays (or becomes) a plain ``EventTemplate`` — replacing an array
        # spec with an opaque one must not be rejected by the original
        # subclass's validation.
        return EventTemplate

    # -- Numeric queries & projection ---------------------------------------

    @property
    def is_numeric(self) -> bool:
        """Whether every reachable leaf is an :class:`ArraySpec`.

        Recursive: descends into nested :class:`EventTemplate` fields and
        returns ``True`` only if *all* leaves (at every depth) are numeric
        array leaves. Any non-:class:`ArraySpec` leaf — or a nested
        sub-template that is not itself all-numeric — makes the whole
        template non-numeric.

        This is computed as an explicit recursive leaf check rather than
        ``isinstance(self, NumericEventTemplate)``. Under the ``__new__``
        auto-promotion invariant the two agree, but the recursive form is
        also correct for hand-built mixed nestings.

        Returns
        -------
        bool
            ``True`` iff every reachable leaf is an :class:`ArraySpec`.
        """
        for spec in self._tree.values():
            if isinstance(spec, ArraySpec):
                continue
            if isinstance(spec, EventTemplate):
                if not spec.is_numeric:
                    return False
                continue
            # Opaque / distribution / function leaf — not numeric.
            return False
        return True

    def numeric_subset(self) -> NumericEventTemplate:
        """Project to the :class:`ArraySpec`-leaf sub-template.

        Keeps every numeric leaf, recursing into nested
        :class:`EventTemplate` fields (each contributes its own
        ``numeric_subset()``); drops every non-:class:`ArraySpec` leaf; and
        prunes any nested template that becomes empty. Surviving leaves keep their
        ``/``-delimited paths (the projection is path-stable). Inference uses
        this to recover the numeric leaves of a mixed template.

        On an already-all-numeric template the result is an equal
        :class:`NumericEventTemplate` (the projection is idempotent).

        Returns
        -------
        NumericEventTemplate
            The numeric-leaf sub-template, so that :attr:`vector_size` and
            :attr:`~NumericEventTemplate.leaf_shapes` are available.

        Raises
        ------
        ValueError
            If no numeric leaves survive — an :class:`EventTemplate` needs at
            least one field, so an empty numeric subset is meaningless. The
            message names the dropped (non-numeric) fields.
        """
        specs: dict[str, _FieldSpec] = {}
        for name, spec in self._tree.items():
            if isinstance(spec, ArraySpec):
                specs[name] = spec
            elif isinstance(spec, EventTemplate):
                try:
                    specs[name] = spec.numeric_subset()
                except ValueError:
                    # The empty-projection guard below is the *only* ValueError
                    # numeric_subset() raises, so catching it here means the
                    # nested template had no numeric leaves — prune it. If a
                    # future change adds another ValueError path, narrow this
                    # catch so it can't mask an unrelated failure.
                    continue
            # Opaque / distribution / function leaves are dropped.
        if not specs:
            dropped = tuple(
                name
                for name, spec in self._tree.items()
                if not (
                    isinstance(spec, ArraySpec)
                    or (isinstance(spec, EventTemplate) and spec.is_numeric)
                )
            )
            raise ValueError(
                f"numeric_subset() of {type(self).__name__} is empty: no "
                f"ArraySpec leaves survive. Dropped non-numeric fields: {dropped}."
            )
        return NumericEventTemplate(specs)

    # -- Equality and hashing -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventTemplate):
            return NotImplemented
        # Order-sensitive comparison so equality matches the
        # order-sensitive ``__hash__`` (insertion order is part of the
        # template's identity). dict.__eq__ alone would ignore order,
        # breaking the eq/hash contract.
        return tuple(self._tree.items()) == tuple(other._tree.items())

    def __hash__(self) -> int:
        # All field specs (value specs and nested templates) are hashable, so
        # the order-sensitive item tuple hashes directly. Insertion order is
        # part of the template's identity.
        return hash(tuple(self._tree.items()))

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def infer_from(cls, value: Any) -> EventTemplate:
        """Best-effort, **lossy** schema inferred by inspecting a value.

        Two cases:

        - A :class:`~probpipe.Record` already carries its authoritative schema,
          so ``infer_from`` returns its :attr:`~probpipe.Record.event_template`
          unchanged.
        - A **mapping** of named fields (e.g. a ``Record``'s field dict) is
          inferred field by field: a nested ``Record`` field contributes its
          own ``event_template``; a numeric array or scalar becomes an
          :class:`ArraySpec` of its shape; anything else becomes a bare
          :class:`OpaqueSpec`. The result auto-promotes to a
          :class:`NumericEventTemplate` when every field is numeric.

        This is the **fallback** for wrapping a raw value that has no template
        yet (e.g. at a workflow boundary); for a value you already hold, read
        its authoritative ``event_template`` directly. Inference is lossy — it
        cannot recover an :class:`ArraySpec`'s ``dtype`` / ``support``, an
        :class:`OpaqueSpec`'s ``meta``, or a :class:`DistributionSpec` /
        :class:`FunctionSpec`. A Python ``list`` / ``tuple`` leaf (no ``.shape``
        / ``.dtype``) is treated as opaque even if it holds numbers; wrap it in
        ``np.asarray`` / ``jnp.asarray`` first for a numeric leaf.

        Parameters
        ----------
        value : Any
            A :class:`~probpipe.Record`, or a mapping of field name to value
            (arrays / scalars / nested ``Record``\\ s).

        Returns
        -------
        EventTemplate
            The inferred schema (a :class:`NumericEventTemplate` when every
            field is numeric).

        Raises
        ------
        TypeError
            If *value* is neither a ``Record`` nor a mapping.
        ValueError
            If *value* is an empty mapping (a template needs at least one field).
        """
        from .record import Record

        if isinstance(value, Record):
            return value.event_template
        if not isinstance(value, Mapping):
            raise TypeError(
                f"infer_from expects a Record or a mapping of fields, got {type(value).__name__}."
            )

        def _leaf_spec(val: Any) -> _FieldSpecInput:
            if isinstance(val, Record):
                return val.event_template
            # A mapping is never a leaf: it denotes tree structure, so infer a
            # nested template from it rather than an (invalid) opaque-leaf spec
            # — matching ``from_nested_dict`` and the workflow-output wrap.
            if isinstance(val, Mapping):
                return cls.infer_from(val)
            # Any numeric array-like — bare arrays and native containers
            # (xarray / pandas / registered backends) alike — infers an
            # ``ArraySpec``; leaves are stored in native form, so nothing is
            # lost by classing them numeric.
            return _full_array_shape_or_none(val)

        specs: dict[str, _FieldSpecInput] = {name: _leaf_spec(val) for name, val in value.items()}
        return EventTemplate(specs)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, spec in self._tree.items():
            if isinstance(spec, EventTemplate):
                parts.append(f"{name}={spec!r}")
            elif isinstance(spec, ArraySpec) and spec.dtype is None and spec.support is None:
                # Bare specs render as their sugar form (shape tuple / None).
                parts.append(f"{name}={spec.shape}")
            elif isinstance(spec, OpaqueSpec) and spec.meta is None:
                parts.append(f"{name}=None")
            else:
                parts.append(f"{name}={spec!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# NumericEventTemplate — all-numeric specialisation
# ---------------------------------------------------------------------------


class NumericEventTemplate(EventTemplate):
    """EventTemplate where every leaf is numeric.

    Extends :class:`EventTemplate` by requiring each spec to be a shape
    tuple (or a nested :class:`NumericEventTemplate`) — no opaque
    ``None`` leaves are allowed. That restriction is what makes
    :attr:`vector_size` and :attr:`leaf_shapes` meaningful:
    ``vector_size`` is the length of the per-element 1-D vector — the total
    number of scalar elements across every numeric leaf — and
    :meth:`~probpipe.NumericRecord.from_vector` takes a template of this class
    so that every field can be reconstructed from a slice of that vector. A
    *batch* of such values is a matrix of shape ``(*batch_shape, vector_size)``,
    not a single vector.

    Use :meth:`EventTemplate.infer_from` on a :class:`NumericRecord`
    (it auto-promotes) or call this constructor directly when you have
    the shape specs in hand.
    """

    __slots__ = ("_vector_size",)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        for name, spec in field_specs.items():
            if _is_numeric_spec(spec):
                continue
            if isinstance(spec, EventTemplate):
                raise TypeError(
                    f"NumericEventTemplate: nested field {name!r} is a "
                    f"{type(spec).__name__}; nested sub-templates must "
                    f"themselves be NumericEventTemplate."
                )
            # Any non-array leaf — OpaqueSpec, DistributionSpec, or FunctionSpec.
            raise TypeError(
                f"NumericEventTemplate: field {name!r} is a {type(spec).__name__}; "
                f"only ArraySpec leaves (or a nested NumericEventTemplate) are "
                f"allowed — use EventTemplate if you need a mixed template."
            )

    def __init__(
        self,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        super().__init__(_field_specs, **field_specs)
        object.__setattr__(self, "_vector_size", self._compute_vector_size())

    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-leaf array shapes, keyed by :meth:`keys` (canonical leaf order).

        Maps each leaf's ``/``-delimited path to its array ``shape``. Defined
        only on :class:`NumericEventTemplate` — where every leaf is an
        :class:`ArraySpec` and therefore *has* a shape — because a shape is an
        array notion; on a general (mixed) :class:`EventTemplate` the leaves are
        a heterogeneous sum with no uniform shape, so the structural view there
        is :meth:`keys`. A nested sub-template contributes one entry per
        nested leaf.
        """
        result: dict[str, tuple[int, ...]] = {}
        for name, spec in self._tree.items():
            if isinstance(spec, NumericEventTemplate):
                for sub_name, sub_shape in spec.leaf_shapes.items():
                    result[f"{name}{_PATH_SEP}{sub_name}"] = sub_shape
            else:
                # ``_post_validate`` guarantees a non-nested spec is an ArraySpec.
                result[name] = spec.shape
        return result

    def _compute_vector_size(self) -> int:
        """Total scalar count across all numeric leaves."""
        total = 0
        for spec in self._tree.values():
            if isinstance(spec, NumericEventTemplate):
                total += spec.vector_size
            else:
                # spec is an ArraySpec — validated by ``_post_validate``.
                total += prod(spec.shape) if spec.shape else 1
        return total

    @property
    def vector_size(self) -> int:
        """Length of the per-element 1-D vector (``to_vector`` / ``from_vector``).

        The total number of scalar elements across all numeric leaves — the
        trailing-axis length of a value's
        :meth:`~probpipe.NumericRecord.to_vector` output. A single value
        serializes to shape ``(vector_size,)``; a batch serializes to a matrix
        ``(*batch_shape, vector_size)``, not a single vector.
        """
        return self._vector_size

    # 1-D numeric (de)serialization is a value operation and lives on the
    # value types: ``to_vector`` on :class:`~probpipe.NumericRecord` /
    # :class:`~probpipe.NumericRecordArray`, and their ``from_vector``
    # classmethods (which take a template). A template describes structure
    # and does not depend on the value type, so it carries neither.


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------


def _unpickle_event_template(specs: dict) -> EventTemplate:
    return EventTemplate(specs)
