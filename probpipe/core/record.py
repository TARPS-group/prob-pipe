"""Record — ProbPipe's structured value type.

A ``Record`` is a single structured value: an immutable collection of named,
ordered fields. In ProbPipe, it is widely used as a wrapper for deterministic
quantities, and thus can be viewed as the non-random counterpart to
:class:`~probpipe.core._distribution_base.Distribution`. Every ``Record``
carries an :class:`EventTemplate` that describes the structure of the
stored value. The event template is the schema encoding the structure of
the concrete value.

A canonical value wrapper
-------------------------
``Record``\\ s are one of the building blocks of unified, reproducible
probabilistic pipelines in ProbPipe. They are used to wrap concrete values,
attaching metadata (name, provenance) and structural information
(an :class:`EventTemplate`). The typical pattern is for
:class:`WorkflowFunction`\\ s to work with native types; ``Record``\\ s come
into play at the boundaries, wrapping the inputs and outputs of these functions.
For example, by default the return value of the ``sample`` operator is wrapped
as a record.

The ``Record`` family
---------------------
- :class:`Record`: represents a single value, which may contain multiple fields.
- :class:`~probpipe.NumericRecord`: a subclass in which all fields are JAX arrays.
- :class:`~probpipe.RecordArray`: batch of ``Record``s sharing one ``EventTemplate``.
- :class:`~probpipe.NumericRecordArray`: batch of ``NumericRecord``s sharing one ``EventTemplate``.

Notes
-----
**Performance considerations.** ``Record``\\ s carry extra information (structure,
provenance) that supports consistency, validation, and debugging but is not meant
for the hot loops of inference algorithms like MCMC, which rely on plain
array-based frameworks. As noted above, a ``Record``'s role is at the boundaries
of expensive compute nodes rather than inside them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from .event_template import EventTemplate, NumericEventTemplate, _full_array_shape_or_none
from .named_tree import _PATH_SEP, NamedTree, _check_no_path_sep, _PathSubtree, _unflatten_paths
from .tracked import Annotated, Tracked

if TYPE_CHECKING:
    from ._numeric_record import NumericRecord

__all__ = [
    "Record",
]

# A field value: nested ``Record`` or anything else (stored as-is).
type _FieldValue = Any


def _is_numeric_field_value(value: Any) -> bool:
    """Whether a raw field value triggers the ``NumericRecord`` promotion.

    A value counts iff coercing it to a ``jax.Array`` loses nothing: a bare
    numeric array (jax / numpy) or a numeric Python scalar. A backend-typed
    numeric leaf — one the aux registry knows how to capture, e.g. an
    ``xarray.DataArray`` or a ``pandas`` object — does **not** trigger
    promotion, so it stays stored verbatim in a plain ``Record`` and the
    caller opts into the coercion explicitly (``to_numeric()`` /
    ``NumericRecord(...)``). A nested record counts iff it is a
    ``NumericRecord``; a batched child never is.
    """
    if isinstance(value, _PathSubtree):
        # Structural nesting from path-keyed construction: numeric iff every
        # value beneath it is.
        return bool(value) and all(_is_numeric_field_value(v) for v in value.values())
    if isinstance(value, Record):
        from ._numeric_record import NumericRecord

        return isinstance(value, NumericRecord)
    from ._array_backend import aux_for

    if aux_for(value) is not None:
        return False
    return _full_array_shape_or_none(value) is not None


def _derived_record_name(field_keys: Iterable[str]) -> str:
    """The deterministic name an operation derives for a record it produces."""
    return "record(" + ",".join(field_keys) + ")"


def _auto_record(
    fields: Mapping[str, Any],
    *,
    name: str | None = None,
    event_template: EventTemplate | None = None,
) -> Record:
    """Operation-side constructor: build a record under an auto-derived name.

    The ``Record`` constructor requires a user-given name; an *operation*
    that produces a record instead derives a deterministic name from its
    inputs and marks it ``name_is_auto=True``. This helper is that derived
    path: *name* supplies the operation's derived name (e.g. the producing
    distribution's name), defaulting to the field-derived
    ``record(<keys>)``. Promotion applies as in normal construction.
    """
    return Record(
        name if name is not None else _derived_record_name(fields),
        dict(fields),
        event_template=event_template,
        name_is_auto=True,
    )


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Record(NamedTree, Tracked, Annotated):
    """A single structured value with metadata.

    A ``Record`` holds a single concrete value: an ordered, named collection
    of fields. Every record stores an :attr:`event_template`, which encodes
    the structure of the value: the names, how the fields are stored, and
    specs on the structure of the fields themselves. A record is immutable
    and its :attr:`event_template` is fixed at construction.

    A named collection of values
    ----------------------------
    A ``Record`` is a :class:`~probpipe.core.named_tree.NamedTree`, so it inherits a
    mapping/dictionary-like interface over the fields it stores: a field is
    accessed as ``record["a"]``, and a nested field as ``record["a/b/c"]`` or
    ``record["a", "b", "c"]``. In effect a record behaves like an ordered
    ``dict`` keyed by the unique ``/``-path to each field — ``len(record)`` is
    the number of fields, ``"a/b/c" in record`` tests membership, and
    ``record.keys()`` / ``record.values()`` / ``record.items()`` iterate the
    paths / field values / pairs. The canonical field order is depth-first in
    insertion order. A record must have at least one field; an empty record is
    not allowed. ::

        r = Record("r", x=1.5, y=Record("y", a=0.0, b=2.0))
        r["x"]          # 1.5
        r["y/a"]        # 0.0   — a nested leaf, by /-path ...
        r["y", "a"]     # 0.0   — ... or by tuple
        len(r)          # 3
        "y/a" in r      # True
        list(r.keys())  # ['x', 'y/a', 'y/b']

    Tree structure
    --------------
    A ``Record`` can equally be viewed as a tree, with the fields at its
    leaves. The *only* allowed internal node is a nested ``Record``; every
    other value is interpreted as a leaf. Names are unique within each node, so
    every field has a unique ``/``-path. The structure of a field
    ``record[key]`` is described by the matching spec
    ``record.event_template[key]``. Any node — leaf or interior — is reachable
    with ``record.at_path(path)``, which returns a field for a leaf path and a
    sub-``Record`` for an interior one. Plain indexing ``record[key]`` is
    reserved for fields and raises if *key* points to an interior node (use
    :meth:`at_path` for that); :attr:`children` gives the one-level
    ``name -> child`` view. ::

        r.at_path("y")    # Record(a=0.0, b=2.0)  — interior node
        r.at_path("y/a")  # 0.0                   — leaf
        dict(r.children)  # {'x': 1.5, 'y': Record(a=0.0, b=2.0)}
        r["y"]            # KeyError: 'y' is a subtree — use at_path()

    Structure encoded by the event template
    ----------------------------------------
    :attr:`event_template` always reflects the structure of the stored value.
    Because an :class:`EventTemplate` is itself a :class:`~probpipe.core.named_tree.NamedTree`, its tree
    mirrors the record's exactly: each nested ``Record`` corresponds to a
    nested ``EventTemplate``, and each field value corresponds to a value spec
    (an array to an :class:`ArraySpec`, any non-array to an
    :class:`OpaqueSpec`, and so on). ::

        r.event_template
        # NumericEventTemplate(x=(), y=NumericEventTemplate(a=(), b=()))
        r.event_template["y/a"]  # ArraySpec(shape=())  — the spec for r["y/a"]

        # A record's subtree and its template's subtree stay in lock-step:
        r.at_path("y").event_template == r.event_template.at_path("y")  # True

        # Each field value maps to a value spec by type:
        Record("r", vec=jnp.zeros(3), label="fox").event_template
        # EventTemplate(vec=(3,), label=None)   — array -> ArraySpec, str -> OpaqueSpec

    Metadata: identity and annotations
    ----------------------------------
    A record is a tracked term: it is :class:`~probpipe.core.tracked.Tracked`,
    carrying a human-readable :attr:`name` — with :attr:`name_is_auto`
    recording whether the name was auto-derived rather than user-given — and,
    optionally, a :attr:`provenance`, the
    :class:`~probpipe.core.provenance.Provenance` describing how it was
    created, attached write-once via :meth:`with_provenance`. It is also
    :class:`~probpipe.core.tracked.Annotated`, so free-form
    :attr:`annotations` may be attached after construction.

    Construction and validation
    ---------------------------
    A record is built from a flat mapping of key/field pairs — either as keyword
    arguments or as a single positional mapping, but not both. The
    :meth:`from_nested_dict` alternate constructor instead takes a nested
    dictionary. Mappings are never leaves: a mapping value denotes tree
    structure, so passing a plain ``dict`` as a field value raises
    ``TypeError`` — nest the entries as fields (via :meth:`from_nested_dict`
    or a nested ``Record``) instead. When every leaf is a numeric array (so
    the carried template is a :class:`NumericEventTemplate`), ``Record(...)``
    **auto-promotes** to :class:`NumericRecord`, mirroring the
    ``EventTemplate`` promotion; the numeric axis is re-derived whenever a
    transform constructs a new record, so an edit that removes the last
    non-numeric leaf promotes the result and one that introduces a
    non-numeric leaf demotes it. An explicit non-numeric ``event_template``
    keeps the result a plain ``Record``. ::

        # All three build the same record:
        Record("r", x=1.5, y=Record("y", a=0.0, b=2.0))
        Record("r", {"x": 1.5, "y/a": 0.0, "y/b": 2.0})
        Record.from_nested_dict("r", {"x": 1.5, "y": {"a": 0.0, "b": 2.0}})

    When an ``event_template`` is supplied it is validated against the value's
    structure, and any mismatch in tree shape or field/spec kind raises
    ``ValueError``. When omitted, the template is inferred via
    :meth:`EventTemplate.infer_from`; inference recovers the tree structure but
    is lossy on the value specs (e.g. it cannot recover a :class:`FunctionSpec`'s
    input / output structure). ::

        Record("r", a=1.0, event_template=EventTemplate(a=(), b=()))
        # ValueError: event_template fields ['a', 'b'] do not match record fields ['a'] ...

    Equality and hashing
    --------------------
    Two records are equal when they share a concrete class (both ``Record``, or
    both ``NumericRecord``), have equal :attr:`event_template`\\ s, and are
    field-by-field equal. Because equality includes the template, whether an
    identity transform round-trips to an equal record depends on how it treats
    the template. A transform that **threads the template through** — ``replace``
    / ``merge`` / ``without``, or reconstruction via
    :meth:`Record.from_field_values` — preserves it exactly and compares
    equal. A transform that instead **re-infers** the template — ``map`` /
    ``map_with_keys``, which rebuild the result and re-infer its specs — compares
    equal only when inference recovers the original template, for instance when
    that template was itself inferred (the record was built without an explicit
    ``event_template``). Inference is lossy: :meth:`EventTemplate.infer_from`
    cannot recover an ``ArraySpec``'s ``dtype`` / ``support``, an
    ``OpaqueSpec``'s ``meta``, etc., so an identity ``map`` of a record carrying
    a richer explicit template does *not* compare equal::

        r.map(lambda x: x) == r   # True only if r's template is inference-recoverable

    Hashing is **structural**: a record's hash combines its class, its field
    names, and each field's shape and dtype (nested records hash recursively; an
    opaque leaf falls back to its type). It fingerprints the record's structure
    rather than its data, so records that share a structure fall in the same
    hash bucket.

    Parameters
    ----------
    name : str
        The record's name — the required first positional argument on
        explicit construction (a user-given name). An operation that
        produces a record supplies a deterministic name derived from its
        inputs and marks it via ``name_is_auto``.
    _fields : Mapping, optional
        Fields as a positional mapping (any ``collections.abc.Mapping``, copied
        into a ``dict`` at construction) — an alternative to keyword ``**fields``
        (passing both raises). Use it when a field name would collide with the
        ``event_template`` / ``name_is_auto`` keywords. Positional-only (the
        leading underscore keeps it from shadowing a field literally named
        ``fields``).
    **fields
        Named values, stored unchanged: ``jax`` / ``numpy`` arrays, Python
        scalars, strings, ``xarray`` / ``pandas`` objects, nested ``Record``s,
        or any opaque object. At least one field is required. A nested record
        takes its name from the field key it sits under.
    name_is_auto : bool, optional
        ``True`` when *name* was derived by the producing operation rather
        than supplied by the user. Defaults to ``False``.
    event_template : EventTemplate, optional
        The value's authoritative schema. When omitted it is inferred from the
        field data at construction (via :meth:`EventTemplate.infer_from`); when
        supplied — e.g. carried forward from the distribution that produced the
        value — it is validated against the field names and stored. Either way it
        is fixed for the life of the record; read it back via
        :attr:`event_template`.

    Raises
    ------
    ValueError
        If no fields are given, a field name contains ``/``, both ``_fields`` and
        keyword fields are passed, or a supplied ``event_template`` does not
        match the field names.
    TypeError
        If a field value is a ``Mapping`` — mappings denote tree structure
        and are never leaves.

    Notes
    -----
    A ``Record`` is a registered JAX PyTree, but the tree from JAX's perspective
    may differ from the tree from ProbPipe's perspective. ProbPipe only allows
    nested ``Record``s to be internal nodes, while JAX also allows objects like
    tuples and dictionaries (which are viewed as leaves by ``Record``). This
    implies that JAX sees a tree with finer structure, so that methods like
    ``jax.tree_util.tree_flatten`` / ``tree_leaves`` / ``tree_map``
    (and ``jit`` / ``vmap`` / ``grad``) descend into nested
    ``Record``\\s *and* into container leaves like tuples and dictionaries.
    It is thus typically best practice to use the custom `Record` functionality
    for traversing the fields. The above JAX functions will work, but users must
    be aware that they will traverse the finer tree. The two notions of the tree
    coincide when every field is an array (e.g. :class:`NumericRecord`).

    The PyTree registration's children are the field values and its static aux
    data is the ``(event_template, name, name_is_auto)`` triple, so the
    template and name survive a ``tree_flatten`` / ``tree_unflatten``
    round-trip. :attr:`provenance` and :attr:`annotations` do **not** cross a
    JAX transform boundary; re-attach provenance on the reconstructed Record
    if you need to preserve the chain.
    """

    __slots__ = (
        "_annotations",
        "_event_template",
        "_name",
        "_name_is_auto",
        "_provenance",
        "_tree",
    )

    def __new__(cls, *args: Any, **kwargs: Any):
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericRecord(...)`` calls run their own strict
        # validation, and the subclasses (batch types, views) own their
        # construction entirely, so the signature stays generic here.
        # The numeric axis keys on the carried template: an explicit
        # template decides directly, and otherwise the raw values are
        # probed with the same predicate template inference uses, so the
        # decision agrees with the template the instance will carry.
        if cls is Record:
            from ._numeric_record import NumericRecord

            event_template = kwargs.get("event_template")
            if len(args) > 1 and args[1] is not None:
                source: Mapping[str, Any] = args[1]
            else:
                source = {
                    k: v for k, v in kwargs.items() if k not in ("event_template", "name_is_auto")
                }
            # Promote only when every field coerces to a bare array with
            # nothing lost (the value probe) *and* no explicit non-numeric
            # template vetoes it. A backend-typed leaf (xarray / pandas)
            # fails the probe, so a plain ``Record`` carrying one stays plain
            # even when its numeric template is threaded back in — by a
            # structural edit or ``from_field_values`` — rather than silently
            # coercing the leaf and breaking the ``to_native`` round-trip.
            template_allows = event_template is None or isinstance(
                event_template, NumericEventTemplate
            )
            numeric = (
                template_allows
                and bool(source)
                and all(_is_numeric_field_value(value) for value in source.values())
            )
            if numeric:
                return object.__new__(NumericRecord)
        return object.__new__(cls)

    def __init__(
        self,
        name: str,
        _fields: Mapping[str, _FieldValue] | None = None,
        /,
        *,
        event_template: EventTemplate | None = None,
        name_is_auto: bool = False,
        **fields: _FieldValue,
    ):
        if _fields is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            field_inputs = _unflatten_paths(_fields)
        else:
            for field_name in fields:
                _check_no_path_sep(field_name)
            field_inputs = dict(fields)
        if not field_inputs:
            raise ValueError("Record requires at least one named field")

        field_map: dict[str, _FieldValue] = {}
        for field_name, value in field_inputs.items():
            sub_template: EventTemplate | None = None
            if event_template is not None:
                template_child = event_template.children.get(field_name)
                if isinstance(template_child, EventTemplate):
                    sub_template = template_child
            try:
                if isinstance(value, _PathSubtree):
                    field_map[field_name] = type(self)(
                        field_name, value, event_template=sub_template, name_is_auto=True
                    )
                elif (
                    sub_template is not None
                    and isinstance(value, Record)
                    # A batched child (RecordArray) carries its own template and
                    # is stored verbatim; ``batch_shape`` is duck-typed because
                    # importing RecordArray here would be circular.
                    and not hasattr(value, "batch_shape")
                ):
                    if value.event_template is sub_template:
                        # The child already carries this exact template object
                        # (e.g. built by ``from_field_values`` or threaded from
                        # the producing generator) — reuse it verbatim, keeping
                        # its provenance / backend aux. Identity, not ``==``:
                        # the point is to adopt the *supplied* template object,
                        # and an equal-but-distinct child template must still be
                        # rebuilt through it.
                        field_map[field_name] = self._named_by_key(field_name, value)
                    else:
                        field_map[field_name] = type(value)(
                            field_name,
                            dict(value._tree),
                            event_template=sub_template,
                            name_is_auto=True,
                        )
                else:
                    if isinstance(value, Record):
                        value = self._named_by_key(field_name, value)
                    else:
                        self._check_leaf(field_name, value)
                    field_map[field_name] = value
            except ValueError as error:
                raise ValueError(f"at {field_name!r}: {error}") from None

        object.__setattr__(self, "_tree", field_map)
        self._init_tracked(name, name_is_auto=name_is_auto)
        if event_template is None:
            event_template = EventTemplate.infer_from(field_map)
        else:
            self._validate_event_template(event_template)
        object.__setattr__(self, "_event_template", event_template)

    @staticmethod
    def _named_by_key(field_name: str, child: Record) -> Record:
        """Align a nested record's name with the field key it sits under.

        A nested object takes its name from its field key. When the child
        already carries that name it is stored as-is; otherwise a shallow
        identity copy (fields, template, aux shared) is stored under the key
        name, marked auto — the name follows the key, so a later rename of
        the field re-derives it.
        """
        if child._name == field_name:
            return child
        renamed = child._shallow_copy()
        object.__setattr__(renamed, "_name", field_name)
        object.__setattr__(renamed, "_name_is_auto", True)
        return renamed

    def _validate_event_template(self, event_template: EventTemplate) -> None:
        """Check that an explicitly-supplied template matches this record's structure.

        Validates the **whole tree**, recursively: at every level the field-name
        sets must match, a nested ``Record`` must align with a nested
        ``EventTemplate`` (both internal nodes), and a non-``Record`` leaf must
        align with a value spec. Per-leaf shape / dtype / kind conformance is the
        producing generator's responsibility and is *not* checked here.

        Raises ``ValueError`` naming the ``/``-path of the first mismatch.
        """

        def _check(record: Record, template: EventTemplate, prefix: str) -> None:
            record_fields = set(record._tree)
            template_fields = set(template.fields)
            if record_fields != template_fields:
                location = prefix.rstrip(_PATH_SEP) or "the top level"
                raise ValueError(
                    f"event_template fields {sorted(template_fields)} do not match record "
                    f"fields {sorted(record_fields)} at {location}"
                )
            for name, value in record._tree.items():
                spec = template.children[name]
                path = f"{prefix}{name}"
                value_is_node = isinstance(value, Record)
                spec_is_node = isinstance(spec, EventTemplate)
                if value_is_node and spec_is_node:
                    _check(value, spec, f"{path}{_PATH_SEP}")
                elif value_is_node != spec_is_node:
                    raise ValueError(
                        f"event_template / record structure mismatch at {path!r}: "
                        f"template has a {'nested template' if spec_is_node else 'value spec'} "
                        f"but record has a {'nested Record' if value_is_node else 'leaf value'}"
                    )
                # both leaves -> OK (leaf-content conformance is not checked here)

        _check(self, event_template, "")

    # -- Name & provenance --------------------------------------------------
    #
    # ``name`` / ``name_is_auto`` / ``provenance`` / ``with_name`` /
    # ``with_provenance`` are provided by the
    # :class:`~probpipe.core.tracked.Tracked` mixin, and ``annotations`` by
    # :class:`~probpipe.core.tracked.Annotated`. Semantic transformations
    # (``replace``, ``merge``, ``without``, ``map``, ``map_with_keys``)
    # return a *new* Record with no provenance; the caller attaches fresh
    # provenance there if desired.

    @property
    def event_template(self) -> EventTemplate:
        """The authoritative :class:`EventTemplate` describing this value's structure.

        Fixed at construction and always present. When a template was supplied
        (carried forward from the producing generator) that template is
        returned; otherwise one was inferred from the data at construction (via
        :meth:`EventTemplate.infer_from`) and stored.

        Notes
        -----
        Inference is a lossy fallback (it cannot recover an ``ArraySpec``'s
        ``dtype`` / ``support``, an ``OpaqueSpec``'s ``meta``, or a
        ``DistributionSpec`` / ``FunctionSpec``). The template rides in the
        JAX pytree aux data, so a value reconstructed by ``tree_unflatten``
        carries this same template; unpickling instead infers a fresh
        template from the rebuilt data.
        """
        return self._event_template

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Record is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Record is immutable")

    def __reduce__(self):
        return (
            _unpickle_record,
            (dict(self._tree), self._name, self._name_is_auto, self._provenance),
        )

    # -- Tree structure -----------------------------------------------------
    #
    # The mapping and path-navigation methods (``keys`` / ``values`` /
    # ``items`` / ``[]`` / ``at_path`` / ``children``) are inherited from
    # :class:`~probpipe.core.named_tree.NamedTree`. A leaf here is a stored
    # (non-``Record``) value; an internal node is a nested ``Record``.

    @classmethod
    def _node_type(cls) -> type:
        return Record

    @classmethod
    def _rebuild_class(cls) -> type:
        # Structural edits rebuild through the base class so ``__new__``
        # re-decides the numeric auto-promotion from the edited leaves: a
        # replacement that removes the last non-numeric leaf promotes the
        # result to ``NumericRecord``, and one that introduces a non-numeric
        # leaf demotes it — the numeric axis is an invariant of the current
        # leaves, not of the object's history.
        return Record

    def _rebuild_node(self, leaves: Mapping[str, Any], *, node_name: str | None) -> Record:
        # A nested node is named by its field key (auto — the name follows
        # the key). The root follows the transform identity rule: a
        # user-given name is preserved, an auto-derived one is re-derived
        # from the result's field keys.
        if node_name is not None:
            return self._rebuild_class()(node_name, leaves, name_is_auto=True)
        if self._name_is_auto:
            return self._rebuild_class()(_derived_record_name(leaves), leaves, name_is_auto=True)
        return self._rebuild_class()(self._name, leaves)

    # -- Selection ----------------------------------------------------------

    def select(self, *fields: str, **mapping: str) -> dict[str, _FieldValue]:
        """Select fields into a plain ``dict``, for splatting into function calls.

        Positional arguments use the field path as the key (identity mapping);
        keyword arguments remap (``select(x="r")`` → ``{"x": self["r"]}``). Each
        argument is resolved with :meth:`at_path`, so a key reaches a leaf value
        and a partial path reaches a subtree. Returns a value-only ``dict``; it is
        not a ``Record`` and carries no schema.

        Usage::

            predict(**params.select("r", "K"), x=x_grid)
            predict(**params.select(growth_rate="r"), x=x_grid)

        Raises
        ------
        KeyError
            If a requested path is not present.
        """
        result: dict[str, _FieldValue] = {}
        for f in fields:
            result[f] = self.at_path(f)
        for arg_name, field_path in mapping.items():
            result[arg_name] = self.at_path(field_path)
        return result

    def select_all(self) -> dict[str, _FieldValue]:
        """Return every top-level field as a ``dict``, for splatting into a call.

        Sugar for ``select`` over the one-level names (``self.children``). On the
        batch and distribution subclasses, whose ``__getitem__`` returns a
        per-field view, the dict holds those views, so the result can be splatted
        back into a workflow function field-by-field.
        """
        return self.select(*self.children)

    # -- Immutable updates --------------------------------------------------
    #
    # ``without`` / ``merge`` / ``replace`` are the structural edits from
    # ``NamedTree``, overridden here as recursive child surgery: untouched
    # children — leaf values and whole subtrees — are reused **verbatim**,
    # preserving their concrete class (a nested ``NumericRecord`` stays
    # numeric) and their metadata (name, provenance, backend aux); only the
    # children a path actually touches are rebuilt, recursively. The
    # authoritative ``event_template`` is assembled from the surviving
    # children's own templates, so the subtree lock-step invariant holds by
    # construction (and the constructor's identity check stores the reused
    # children without copying).

    def _child_spec(self, name: str, child: Any) -> Any:
        """The template entry describing an untouched child, identity-preserved."""
        if isinstance(child, Record):
            return child.event_template
        return self.event_template.children[name]

    def _rebuild_root(self, children: Mapping[str, Any], event_template: EventTemplate) -> Record:
        """Rebuild the root of a structural transform, threading the template.

        Applies the transform identity rule in one place: a user-given name
        is preserved, while an auto-derived name is re-derived from the
        result's top-level field keys — matching :meth:`_rebuild_node` (used
        by :meth:`map`) so every structural edit agrees. Without this, a
        transform that changes the field set (``without`` / ``merge`` /
        ``with_path_names``) would keep an auto name describing the pre-edit
        fields.
        """
        if self._name_is_auto:
            name = _derived_record_name(event_template.children)
            return self._rebuild_class()(
                name, children, event_template=event_template, name_is_auto=True
            )
        return self._rebuild_class()(self._name, children, event_template=event_template)

    def without(self, *paths: str) -> Record:
        """Return a new Record without the fields/subtrees at *paths*.

        The structural contract is :meth:`NamedTree.without`. Surviving fields
        keep their order and their authoritative specs; an untouched child is
        reused verbatim (class and metadata preserved), and a subtree a path
        reaches into is edited recursively. Dropping every leaf under a child
        prunes that child entirely.
        """
        norms = [self._norm_path(p) for p in paths]
        for p in norms:
            self.at_path(p)  # KeyError if the path does not exist
        full_drops = {p for p in norms if _PATH_SEP not in p}
        sub_drops: dict[str, list[str]] = {}
        for p in norms:
            head, sep, rest = p.partition(_PATH_SEP)
            if sep:
                sub_drops.setdefault(head, []).append(rest)

        new_children: dict[str, Any] = {}
        specs: dict[str, Any] = {}
        for name, child in self._tree.items():
            if name in full_drops:
                continue
            drops = sub_drops.get(name)
            if drops is None:
                new_children[name] = child
                specs[name] = self._child_spec(name, child)
                continue
            # ``at_path`` above guarantees ``child`` is an interior Record here.
            remaining = [
                k
                for k in child
                if not any(k == d or k.startswith(f"{d}{_PATH_SEP}") for d in drops)
            ]
            if not remaining:
                continue  # every leaf dropped -> prune the child entirely
            edited = child.without(*drops)
            new_children[name] = edited
            specs[name] = edited.event_template
        if not new_children:
            raise ValueError("Cannot remove all fields from a collection")
        return self._rebuild_root(new_children, EventTemplate(specs))

    def merge(self, other: Record) -> Record:
        """Return the union of this Record and *other* (see :meth:`NamedTree.merge`).

        The merge is by field key, so subtrees sharing a top-level name merge
        recursively; a child present on one side only is reused verbatim (class
        and metadata preserved). Both records' authoritative specs are carried
        into the merged ``event_template``.
        """
        self._leaves_merged(other)  # validates: overlapping field keys raise
        new_children: dict[str, Any] = {}
        specs: dict[str, Any] = {}
        for name, child in self._tree.items():
            other_child = other._tree.get(name)
            if other_child is None:
                new_children[name] = child
                specs[name] = self._child_spec(name, child)
            elif isinstance(child, Record) and isinstance(other_child, Record):
                merged = child.merge(other_child)
                new_children[name] = merged
                specs[name] = merged.event_template
            else:
                # One side holds a leaf where the other holds a subtree.
                raise ValueError(f"name {name!r} is used both as a field and as a path prefix")
        for name, child in other._tree.items():
            if name in self._tree:
                continue
            new_children[name] = child
            specs[name] = other._child_spec(name, child)
        return self._rebuild_root(new_children, EventTemplate(specs))

    def replace(self, _updates: Mapping[str, Any] | None = None, /, **updates: Any) -> Record:
        """Return a new Record with the values at the given paths replaced.

        The structural contract is :meth:`NamedTree.replace`: every path must
        already exist, and a partial path replaces a whole subtree. Untouched
        fields keep their authoritative specs (untouched children are reused
        verbatim — class and metadata preserved); a replaced field takes its
        new value's inferred spec; a nested update edits the nested child
        recursively. Overlapping update paths (one an ancestor of another)
        raise ``ValueError``.
        """
        resolved = self._resolve_replace_updates(_updates, updates)
        if not resolved:
            return self
        norms = {self._norm_path(p): v for p, v in resolved.items()}
        for p in norms:
            self.at_path(p)  # KeyError if the path does not exist
        full_updates = {p: v for p, v in norms.items() if _PATH_SEP not in p}
        sub_updates: dict[str, dict[str, Any]] = {}
        for p, v in norms.items():
            head, sep, rest = p.partition(_PATH_SEP)
            if sep:
                sub_updates.setdefault(head, {})[rest] = v
        overlap = sorted(set(full_updates) & set(sub_updates))
        if overlap:
            raise ValueError(
                f"replace() update paths overlap: {overlap[0]!r} and a path "
                f"beneath it address the same subtree; replace the enclosing "
                f"path once instead"
            )

        new_children: dict[str, Any] = {}
        specs: dict[str, Any] = {}
        for name, child in self._tree.items():
            if name in full_updates:
                value = full_updates[name]
                new_children[name] = value
                specs[name] = self._spec_of(value)
            elif name in sub_updates:
                # ``at_path`` above guarantees ``child`` is an interior Record.
                edited = child.replace(sub_updates[name])
                new_children[name] = edited
                specs[name] = edited.event_template
            else:
                new_children[name] = child
                specs[name] = self._child_spec(name, child)
        return self._rebuild_class()(
            self._name,
            new_children,
            event_template=EventTemplate(specs),
            name_is_auto=self._name_is_auto,
        )

    def _spec_of(self, value: _FieldValue) -> Any:
        """The value spec describing a new field *value*, for template threading."""
        if isinstance(value, Record):
            return value.event_template
        return EventTemplate.infer_from({"_leaf_": value}).children["_leaf_"]

    # -- Field renaming -------------------------------------------------------

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Record:
        """Return a Record with the given fields renamed, ``old -> new``.

        The structural contract is :meth:`NamedTree.with_path_names`: keys are
        node paths, or bare names when unambiguous; values are the new
        single-segment names; renames apply simultaneously. The authoritative
        :attr:`event_template` renames in lockstep, so the subtree invariant
        holds on the result. The record's own identity follows the
        :attr:`name_is_auto` rule: an auto-derived name is re-derived from the
        new field names, while a user-given name is preserved (this renames
        the fields *within* the record; ``with_name`` renames the record
        itself). The result carries no provenance.
        """
        renames = self._resolve_path_renames(mapping, kwargs)
        renamed = self._renamed_leaf_map(renames)
        if len(renamed) != len(self):
            raise ValueError(
                "with_path_names() produced colliding field keys; a rename "
                "must not collide with an existing sibling name"
            )
        renamed_template = self.event_template.with_path_names(mapping, **kwargs)
        return self._rebuild_root(renamed, renamed_template)

    # -- Backend conversion -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a nested ``dict`` of stored values (leaves verbatim, no coercion).

        Temporary: this is the same nested-tree export as :meth:`to_nested_dict`
        (the canonical name). It is retained during the migration and scheduled
        for removal; new code should call :meth:`to_nested_dict`. (The flat,
        path-keyed view is the builtin ``dict(record)``.)
        """
        return self.to_nested_dict()

    def to_numpy(self) -> dict[str, Any]:
        """Return a dict of numpy arrays (recursive for nested Record).

        Each numeric leaf is converted via ``np.asarray``. Non-numeric
        leaves (strings, opaque objects) are returned as-is. Backend
        metadata (xarray dims / coords, pandas index) is stripped — use
        :meth:`to_numeric` followed by :meth:`NumericRecord.to_native`
        if you need a metadata-preserving round-trip.
        """
        result: dict[str, Any] = {}
        for name, val in self._tree.items():
            if isinstance(val, Record):
                result[name] = val.to_numpy()
            elif hasattr(val, "shape") or isinstance(val, (int, float, complex)):
                result[name] = np.asarray(val)
            else:
                result[name] = val
        return result

    def to_numeric(self) -> NumericRecord:
        """Convert to a :class:`NumericRecord` with every leaf a ``jax.Array``.

        Per-field metadata that ``jnp.asarray`` would drop (xarray
        dims / coords / attrs, pandas index / columns / dtypes) is
        captured via the aux registry in
        :mod:`probpipe.core._array_backend` and stored on the resulting
        ``NumericRecord``. Calling :meth:`NumericRecord.to_native`
        on the result reverses the conversion, restoring each leaf to
        its original backend type. Nested ``Record`` children recurse
        — every level becomes a ``NumericRecord``.

        This is the single entry point for the ``Record`` → ``NumericRecord``
        conversion; constructing ``NumericRecord(**record.select_all())`` runs
        the same coercion and aux-capture path.

        Raises
        ------
        TypeError
            If any leaf is not coercible via ``jnp.asarray`` (e.g.
            strings, opaque Python objects).
        """
        # Lazy import to avoid the module-level circular dep:
        # _numeric_record.py imports Record from this module.
        from ._numeric_record import NumericRecord

        return NumericRecord(
            self._name,
            {
                field_name: val.to_numeric() if isinstance(val, Record) else val
                for field_name, val in self._tree.items()
            },
            name_is_auto=self._name_is_auto,
        )

    # -- Coercion -----------------------------------------------------------

    @classmethod
    def ensure(cls, x: Any, *, name: str | None = None) -> Record:
        """Coerce *x* to Record if it isn't already.

        - ``Record`` → pass through (any *name* is ignored)
        - ``dict`` → its entries become the fields
        - anything else → a single-field record keyed ``data``

        A freshly wrapped value is named *name* when given; otherwise its
        name is derived from the resulting field keys and marked auto.
        """
        if isinstance(x, cls):
            return x
        fields = dict(x) if isinstance(x, dict) else {"data": x}
        if name is None:
            return cls(_derived_record_name(fields), fields, name_is_auto=True)
        return cls(name, fields)

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_dict(cls, name: str, d: dict[str, ArrayLike | Record]) -> Record:
        """Construct a Record named *name* from a dict of arrays."""
        return cls(name, d)

    @classmethod
    def from_nested_dict(
        cls,
        name: str,
        data: Mapping[str, Any],
        /,
        *,
        event_template: EventTemplate | None = None,
    ) -> Record:
        """Build a Record named *name* from a **nested** mapping.

        Every ``Mapping`` level becomes a nested sub-record and every other
        value a field — mappings are never leaves (see
        :meth:`NamedTree.from_nested_dict`, whose signature this extends
        with the record's required name). With *event_template* given, the
        template is validated against that structure and carried onto the
        result exactly as in normal construction.
        """
        flat = cls._flatten_paths(data)
        if event_template is None:
            return cls(name, flat)
        return cls(name, flat, event_template=event_template)

    @classmethod
    def from_field_values(cls, name: str, template: EventTemplate, values: Iterable[Any]) -> Record:
        """Reconstruct a value from an ordered sequence of field values.

        *values* supplies one object per field, in canonical order (the order
        of the template's :meth:`~NamedTree.keys`); the names and tree shape
        come from *template*, which the result carries as its
        **authoritative** :attr:`event_template` (nothing is inferred), so
        the round-trip is faithful:
        ``Record.from_field_values(r.name, r.event_template, r.values()) == r``.
        The export side is just ``list(record.values())``. The result's class
        follows the template's numericness — a :class:`NumericEventTemplate`
        builds a :class:`NumericRecord` via the auto-promotion.

        Parameters
        ----------
        name : str
            Name for the reconstructed record (user-given).
        template : EventTemplate
            The authoritative schema supplying names, nesting, and order.
        values : iterable
            One field value per template key, in canonical order.

        Returns
        -------
        Record
            The reconstructed value (a ``NumericRecord`` for a numeric
            template).

        Raises
        ------
        ValueError
            If the number of *values* is not the number of fields
            (``len(template)``).
        """
        values = list(values)
        if len(values) != len(template):
            raise ValueError(
                f"Record.from_field_values: got {len(values)} values, "
                f"expected {len(template)} (one per field)."
            )
        leaf_iter = iter(values)

        def _build(tpl: EventTemplate, node_name: str, node_is_auto: bool) -> Record:
            fields = {
                field_name: (
                    _build(spec, field_name, True)
                    if isinstance(spec, EventTemplate)
                    else next(leaf_iter)
                )
                for field_name, spec in tpl.children.items()
            }
            # ``Record.__new__`` selects the class from the template.
            return Record(node_name, fields, event_template=tpl, name_is_auto=node_is_auto)

        return _build(template, name, False)

    # -- Leaf-wise operations -----------------------------------------------
    #
    # ``map`` / ``map_with_keys`` are inherited from ``NamedTree`` (they apply a
    # function to each field value and rebuild the same structure, re-inferring
    # the result's per-leaf specs). See ``NamedTree.map``.

    # A record's leaves in canonical order are ``list(record.values())``;
    # reconstruct via ``Record.from_field_values(name, template, values)``.

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, val in self._tree.items():
            if isinstance(val, Record):
                parts.append(f"{name}={val!r}")
            elif hasattr(val, "shape") and val.shape != ():
                parts.append(f"{name}=array(shape={val.shape})")
            else:
                parts.append(f"{name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # -- Call-forwarding shim for single-field Records ----------------------
    #
    # When a WorkflowFunction wraps a callable return as
    # ``Record({fn_name: callable})``, the caller can keep invoking it via
    # ``result(args)``. Multi-field records raise — unwrapping one of many
    # fields would be ambiguous.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(self._tree) != 1:
            raise TypeError(
                f"{type(self).__name__} with {len(self._tree)} fields is not "
                f"callable; access a specific field with record['field_name'] "
                f"first."
            )
        only = next(iter(self._tree.values()))
        if not callable(only):
            raise TypeError(
                f"{type(self).__name__} single field is not callable (got {type(only).__name__})."
            )
        return only(*args, **kwargs)

    # -- Equality / hash ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Two records are equal iff they have equal structure **and** equal data.

        Concretely, equality requires (1) the same concrete type, (2) equal
        :attr:`event_template` (so two values with structurally distinct schemas
        — e.g. differing ``support`` on a numeric leaf — are unequal even with
        identical bytes), and (3) field-by-field equal values (arrays compared
        with ``jnp.array_equal``, nested records recursively, opaque leaves with
        ``==``). Self-identity short-circuits to ``True`` so a record equals
        itself even when a leaf contains ``NaN``.

        Whether two records with equal data compare equal therefore turns on
        their templates. Built from equal data *without* an explicit template,
        both infer the same template and compare equal; but a record given a
        richer explicit template compares equal to a re-inferred rebuild of
        itself (e.g. via :meth:`map`) only when inference recovers that template,
        since :meth:`EventTemplate.infer_from` is lossy (it cannot recover an
        ``ArraySpec``'s ``dtype`` / ``support``, etc.).

        :meth:`__hash__` is consistent with this: it hashes the per-field
        shape / dtype structure, so equal records (equal data ⟹ equal shapes)
        always hash equal.
        """
        # Identity fast-path: self-equality must return True even when
        # leaves contain NaN (``jnp.array_equal`` treats NaN != NaN).
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        if tuple(self._tree) != tuple(other._tree):
            return False
        if self.event_template != other.event_template:
            return False
        for name, a in self._tree.items():
            b = other._tree[name]
            if isinstance(a, Record) and isinstance(b, Record):
                if a != b:
                    return False
            elif isinstance(a, Record) or isinstance(b, Record):
                return False
            elif hasattr(a, "shape") or hasattr(b, "shape"):
                try:
                    if not jnp.array_equal(jnp.asarray(a), jnp.asarray(b)):
                        return False
                except Exception:
                    if a is not b and a != b:
                        return False
            else:
                if a != b:
                    return False
        return True

    def __hash__(self) -> int:
        # Structural hash over class, field names, and per-field shape+dtype.
        # Numeric leaves are coerced via ``jnp.asarray`` so a scalar and its
        # array wrapping hash alike (they compare equal under ``__eq__``);
        # opaque leaves fall back to ``type(val)``.
        parts: list[Any] = [type(self).__name__]
        for name, val in self._tree.items():
            if isinstance(val, Record):
                parts.append((name, hash(val)))
                continue
            try:
                arr = jnp.asarray(val)
            except (TypeError, ValueError):
                parts.append((name, "opaque", type(val).__name__))
                continue
            parts.append((name, tuple(arr.shape), str(arr.dtype)))
        return hash(tuple(parts))


def _pack_fields(
    fields: tuple[str, ...],
    field_kwargs: dict[str, Any],
    *,
    owner: str = "",
) -> Record:
    """Validate that *field_kwargs* names exactly *fields*, then build a Record.

    The general "named values → validated :class:`Record`" operation. Raises
    ``TypeError`` if any field is missing or unexpected; otherwise returns a
    ``Record`` keyed in *fields* order. *owner* (optional) prefixes the error
    message — the calling distribution or template class name.

    :meth:`Distribution._pack_value` is the main caller; it works from the
    field-name tuple alone, since some distributions expose ``fields`` without
    an :class:`EventTemplate` instance.
    """
    given = set(field_kwargs)
    expected = set(fields)
    missing = [f for f in fields if f not in given]
    extra = [k for k in field_kwargs if k not in expected]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unexpected {extra}")
        prefix = f"{owner}: " if owner else ""
        raise TypeError(
            f"{prefix}expected exactly the fields {tuple(fields)} — {'; '.join(parts)}."
        )
    field_map = {f: field_kwargs[f] for f in fields}
    return Record(_derived_record_name(field_map), field_map, name_is_auto=True)


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------


def _unpickle_record(store: dict, name: str, name_is_auto: bool, provenance) -> Record:
    r = Record(name, store)
    return r._restore_identity(name_is_auto=name_is_auto, provenance=provenance)


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_flatten(v: Record) -> tuple[list, tuple[EventTemplate, str, bool]]:
    """Flatten Record for JAX pytree traversal.

    The children are the stored field values exactly as-is; JAX further
    traverses any nested ``Record`` children because ``Record`` is a
    registered pytree type, and non-pytree objects (strings, opaque objects)
    become pytree leaves themselves. The static aux data is the
    ``(event_template, name, name_is_auto)`` triple — the template and name
    survive a ``tree_flatten`` / ``tree_unflatten`` round-trip, while
    provenance and annotations do not cross a JAX transform boundary.
    """
    # Emit children in the template's field order so they realign with the
    # aux template on unflatten. ``_tree`` order normally matches, but a
    # record built with an explicit template ordered differently would
    # otherwise zip each value against the wrong field name.
    children = [v._tree[name] for name in v._event_template.children]
    return children, (v._event_template, v._name, v._name_is_auto)


def _record_unflatten(aux: tuple[EventTemplate, str, bool], children: list) -> Record:
    """Unflatten Record from JAX pytree traversal, threading the aux template.

    Reconstructs a plain ``Record`` unconditionally — JAX requires the
    unflattened tree to reproduce the flattened treedef, so this path must
    not re-run the ``__new__`` promotion (a verbatim backend leaf, e.g. an
    ``xarray.DataArray``, has a numeric template but was flattened as a
    plain ``Record``).
    """
    template, name, name_is_auto = aux
    r = object.__new__(Record)
    r.__init__(name, dict(zip(tuple(template.children), children)), event_template=template)
    return r._restore_identity(name_is_auto=name_is_auto, provenance=None)


jax.tree_util.register_pytree_node(Record, _record_flatten, _record_unflatten)
