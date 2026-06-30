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
``Record``s are one of the building blocks of unified, reproducible
probabilistic pipelines in ProbPipe. TODO: left off here


The ``Record`` family
---------------------



A ``Record`` binds three things:

* **data** — named fields, stored verbatim: ``record[name]`` returns exactly the
  object that was passed in (a ``jax`` / ``numpy`` array, a Python scalar, a
  string, an ``xarray`` / ``pandas`` object, a nested ``Record``, ...).
* **structure** — an authoritative :attr:`~Record.event_template` (an
  :class:`EventTemplate`, the schema living in
  :mod:`probpipe.core.event_template`), fixed at construction.
* **identity** — a :attr:`~Record.name` and write-once provenance
  (:attr:`~Record.source`).

Fields iterate in **insertion order**, and ``/`` is reserved as the nested-path
separator (``record["outer/inner"]``), so it is rejected in field names.

The Record family
-----------------

| Class | Purpose |
|---|---|
| :class:`Record` | single value; fields may be any type (arrays, scalars, strings, nested ``Record``s). |
| :class:`~probpipe.NumericRecord` (subclass) | single value, every leaf coerced to ``jax.Array``; adds 1-D ``to_vector``. |
| :class:`~probpipe.RecordArray` | a batch of ``Record``s sharing one ``EventTemplate``; each field is shaped ``(*batch_shape, *leaf_shape)``. |
| :class:`~probpipe.NumericRecordArray` (subclass) | a batch of ``NumericRecord``s; adds 1-D ``to_vector`` and across-batch reductions (``mean`` / ``var``). |

The structural schema itself — :class:`EventTemplate` /
:class:`~probpipe.NumericEventTemplate` and the leaf specs — lives in
:mod:`probpipe.core.event_template`; reach for it directly when you need to
describe structure *without* an example value.

**When to reach for which**

* :class:`Record` — heterogeneous fields, or when you want to keep the original
  backend objects intact (it coerces nothing).
* :class:`~probpipe.NumericRecord` — every leaf numeric: gives a uniform
  ``jax.Array`` type and a flat 1-D vector (``to_vector`` /
  :meth:`~probpipe.NumericEventTemplate.from_vector`).
* :class:`~probpipe.RecordArray` / :class:`~probpipe.NumericRecordArray` —
  collections (e.g. posterior draws): integer indexing materialises one element,
  string indexing returns the batched field.

Usage::

    from probpipe import Record, NumericRecord

    params = NumericRecord(r=1.8, K=70.0, phi=10.0)
    data = Record(counts=np.array([2, 1, 3, 0, 5]), label="horseshoe")

    params["r"]            # jnp.array(1.8)
    tuple(params.keys())   # ('r', 'K', 'phi')  — field keys, insertion order
    params.event_template  # NumericEventTemplate(r=(), K=(), phi=())
    params.to_vector()     # jnp.array([1.8, 70., 10.])

    data["counts"]         # np.array([2, 1, 3, 0, 5])  — stored verbatim
    data["label"]          # "horseshoe"

Converting to / from JAX-native form
------------------------------------

ProbPipe's native array form is the ``jax.Array``. :meth:`Record.to_numeric`
converts any ``Record`` to a :class:`NumericRecord` (every leaf a ``jax.Array``);
:meth:`NumericRecord.to_native` reverses it, restoring backend-specific metadata
(``xarray`` dims / coords / attrs, ``pandas`` index / columns / dtypes) captured
via the registry in :mod:`probpipe.core._array_backend`. Direct
``NumericRecord(...)`` construction consults the same registry, so the two paths
are identical.

Notes
-----
**Performance considerations.** ``Record``s are value wrappers that carry around
additional information (e.g. structure, provenance) that is useful in probabilistic
workflows (e.g. for consistency, validation, debugging, etc.). They are not designed
for optimized performance in statistical inference algorithms like MCMC. Such
algorithms rely on standard array-based computing frameworks. ``Record``'s role
appears at the boundaries: wrapping the inputs and outputs of expensive compute
nodes, enabling the construction of unified probabilistic workflows.

**No coercion (plain ``Record``).** Leaves are stored as-is, so ``jax.tree.map``
and ``jnp`` operations see exactly the types you provided. A Python ``list`` /
``tuple`` leaf has no ``.shape`` / ``.dtype`` and is therefore treated as opaque
(even if it holds numbers) — wrap it in ``np.asarray`` / ``jnp.asarray`` for a
numeric leaf, or use :class:`NumericRecord`, which coerces every leaf to a
``jax.Array`` at construction.

**Identity and structure across pytree round-trips.** :attr:`~Record.name`,
:attr:`~Record.source`, and :attr:`~Record.event_template` are runtime metadata,
not part of the JAX pytree aux (which holds only the field names). A value
rebuilt by ``tree_unflatten`` — or unpickled — gets a default name, no
provenance, and a freshly inferred template; re-attach identity if you need to
preserve it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from .event_template import (
    _PATH_SEP,
    EventTemplate,
    _check_no_path_sep,
    _NamedTree,
    _PathSubtree,
    _unflatten_paths,
)
from .provenance import Provenance

if TYPE_CHECKING:
    from ._numeric_record import NumericRecord

__all__ = [
    "Record",
]

# A field value: nested ``Record`` or anything else (stored as-is).
type _FieldValue = Any


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Record(_NamedTree):
    """A single structured value with metadata.

    A ``Record`` holds a single concrete value: an ordered, named collection
    of fields. Every record stores an :attr:`event_template`, which encodes
    the structure of the value: the names, how the fields are stored, and
    specs on the structure of the fields themselves. A record is immutable
    and its :attr:`event_template` is fixed at construction.

    A named collection of values
    ----------------------------
    A ``Record`` is a :class:`_NamedTree`, so it inherits a
    mapping/dictionary-like interface over the fields it stores: a field is
    accessed as ``record["a"]``, and a nested field as ``record["a/b/c"]`` or
    ``record["a", "b", "c"]``. In effect a record behaves like an ordered
    ``dict`` keyed by the unique ``/``-path to each field — ``len(record)`` is
    the number of fields, ``"a/b/c" in record`` tests membership, and
    ``record.keys()`` / ``record.values()`` / ``record.items()`` iterate the
    paths / field values / pairs. The canonical field order is depth-first in
    insertion order. A record must have at least one field; an empty record is
    not allowed. ::

        r = Record(x=1.5, y=Record(a=0.0, b=2.0))
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
    Because an :class:`EventTemplate` is itself a :class:`_NamedTree`, its tree
    mirrors the record's exactly: each nested ``Record`` corresponds to a
    nested ``EventTemplate``, and each field value corresponds to a leaf spec
    (an array to an :class:`ArraySpec`, any non-array to an
    :class:`OpaqueSpec`, and so on). ::

        r.event_template
        # NumericEventTemplate(x=(), y=NumericEventTemplate(a=(), b=()))
        r.event_template["y/a"]  # ArraySpec(shape=())  — the spec for r["y/a"]

        # A record's subtree and its template's subtree stay in lock-step:
        r.at_path("y").event_template == r.event_template.at_path("y")  # True

        # Each field value maps to a leaf spec by type:
        Record(vec=jnp.zeros(3), label="fox").event_template
        # EventTemplate(vec=(3,), label=None)   — array -> ArraySpec, str -> OpaqueSpec

    Metadata: name, provenance, and annotations
    -------------------------------------------
    A record is ``Tracked``, meaning it has a ``name`` and ``provenance``. It is
    also ``Annotated``, so that it can carry along additional metadata as
    necessary.

    Construction and validation
    ---------------------------
    A record is built from a flat mapping of key/field pairs — either as keyword
    arguments or as a single positional mapping, but not both. The
    :meth:`from_nested_dict` alternate constructor instead takes a nested
    dictionary. A plain ``dict`` passed to the standard constructor is *not*
    treated as nesting; it is stored as one opaque leaf, since only a ``Record``
    may be an interior node. A plain ``Record`` is **not** auto-promoted when
    every field happens to be numeric — the result is still a ``Record`` (though
    its :attr:`event_template` does promote to a :class:`NumericEventTemplate`);
    construct a :class:`NumericRecord` explicitly, or call :meth:`to_numeric`,
    when you want the numeric value type. ::

        # All three build the same record:
        Record(x=1.5, y=Record(a=0.0, b=2.0))
        Record({"x": 1.5, "y/a": 0.0, "y/b": 2.0})
        Record.from_nested_dict({"x": 1.5, "y": {"a": 0.0, "b": 2.0}})

    When an ``event_template`` is supplied it is validated against the value's
    structure, and any mismatch in tree shape or field/spec kind raises
    ``ValueError``. When omitted, the template is inferred via
    :meth:`EventTemplate.infer_from`; inference recovers the tree structure but
    is lossy on the leaf specs (e.g. it cannot recover a :class:`FunctionSpec`'s
    input / output structure). ::

        Record(a=1.0, event_template=EventTemplate(a=(), b=()))
        # ValueError: event_template fields ['a', 'b'] do not match record fields ['a'] ...

    Equality and hashing
    --------------------
    Two records are equal when they share a concrete class (both ``Record``, or
    both ``NumericRecord``), have equal :attr:`event_template`\\ s, and are
    field-by-field equal. So an identity transform that rebuilds the same data
    compares equal to the original::

        r.map(lambda x: x) == r   # True

    Parameters
    ----------
    _fields : Mapping, optional
        Fields as a positional mapping (any ``collections.abc.Mapping``, copied
        into a ``dict`` at construction) — an alternative to keyword ``**fields``
        (passing both raises). Use it when a field name would collide with the
        ``name`` / ``event_template`` keywords. Positional-only (the leading
        underscore keeps it from shadowing a field literally named ``fields``).
    **fields
        Named values, stored unchanged: ``jax`` / ``numpy`` arrays, Python
        scalars, strings, ``xarray`` / ``pandas`` objects, nested ``Record``s,
        or any opaque object. At least one field is required.
    name : str, optional
        Human-readable label for introspection / provenance. Defaults to a label
        derived from the field names.
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

    :attr:`name`, :attr:`source`, and :attr:`event_template` are runtime metadata
    and are not serialised into the PyTree aux (which holds only the field
    names).
    """

    __slots__ = ("_event_template", "_name", "_source", "_tree")

    def __init__(
        self,
        _fields: Mapping[str, _FieldValue] | None = None,
        /,
        *,
        name: str | None = None,
        event_template: EventTemplate | None = None,
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
                    field_map[field_name] = type(self)(value, event_template=sub_template)
                elif (
                    sub_template is not None
                    and isinstance(value, Record)
                    and not hasattr(value, "batch_shape")
                ):
                    field_map[field_name] = type(value)(
                        dict(value._tree), event_template=sub_template
                    )
                else:
                    field_map[field_name] = value
            except ValueError as error:
                raise ValueError(f"at {field_name!r}: {error}") from None

        object.__setattr__(self, "_tree", field_map)
        if name is None:
            name = "record(" + ",".join(field_map.keys()) + ")"
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_source", None)
        if event_template is None:
            event_template = EventTemplate.infer_from(field_map)
        else:
            self._validate_event_template(event_template)
        object.__setattr__(self, "_event_template", event_template)

    def _validate_event_template(self, event_template: EventTemplate) -> None:
        """Check that an explicitly-supplied template matches this record's structure.

        Validates the **whole tree**, recursively: at every level the field-name
        sets must match, a nested ``Record`` must align with a nested
        ``EventTemplate`` (both internal nodes), and a non-``Record`` leaf must
        align with a leaf spec. Per-leaf shape / dtype / kind conformance is the
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
                        f"template has a {'nested template' if spec_is_node else 'leaf spec'} "
                        f"but record has a {'nested Record' if value_is_node else 'leaf value'}"
                    )
                # both leaves -> OK (leaf-content conformance is not checked here)

        _check(self, event_template, "")

    # -- Name & provenance --------------------------------------------------

    @property
    def name(self) -> str:
        """Name of this Record."""
        return self._name

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
        ``DistributionSpec`` / ``FunctionSpec``). Like :attr:`name` and
        :attr:`source`, the template is runtime metadata: it is not serialised
        into the JAX pytree aux, so a value reconstructed by ``tree_unflatten``
        (or unpickling) infers a fresh template from the rebuilt data.
        """
        return self._event_template

    @property
    def source(self) -> Provenance | None:
        """Provenance describing how this Record was created, or ``None``."""
        return self._source

    def with_source(self, source: Provenance | None) -> Record:
        """Attach provenance to this Record (write-once).

        Passing ``None`` (e.g. the result of ``Provenance.create()`` under
        :attr:`ProvenanceMode.OFF`) is a no-op.

        Mirrors ``Distribution.with_source`` — `_source` is set once and
        subsequent calls raise. Semantic transformations (``replace``,
        ``merge``, ``without``, ``map``, ``map_with_keys``) return a
        *new* Record with an empty source; the caller attaches fresh
        provenance there if desired.

        Notes
        -----
        ``_source`` is runtime-only metadata — it is not serialised into
        the JAX pytree aux (a ``Provenance`` parent is a ``Distribution``
        or ``Record``, neither of which is hashable by structure).
        Round-tripping through ``jax.tree_util.tree_flatten`` /
        ``tree_unflatten`` therefore drops the source; re-attach it on
        the reconstructed Record if you need to preserve the chain.
        """
        if source is None:
            return self
        if self._source is not None:
            raise RuntimeError(
                f"Source already set on {self!r}. "
                "Provenance is write-once; create a new Record instead."
            )
        object.__setattr__(self, "_source", source)
        return self

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Record is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Record is immutable")

    def __reduce__(self):
        return (_unpickle_record, (dict(self._tree), self._name, self._source))

    # -- Tree structure -----------------------------------------------------
    #
    # The mapping / navigation surface is inherited from :class:`_NamedTree`. A
    # leaf here is a stored (non-``Record``) value; an internal node is a nested
    # ``Record``.

    @classmethod
    def _node_type(cls) -> type:
        return Record

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
    # ``_NamedTree``, overridden here to thread the authoritative
    # ``event_template`` through the edit (untouched fields keep their specs;
    # only replaced / added fields are re-inferred). They reuse the base's pure
    # leaf-map helpers and return ``type(self)``.

    def without(self, *paths: str) -> Record:
        """Return a new Record without the fields/subtrees at *paths*.

        The structural contract is :meth:`_NamedTree.without`; here the surviving
        fields keep their authoritative specs by editing ``event_template`` the
        same way.
        """
        return type(self)(
            self._leaves_without(paths),
            event_template=self.event_template.without(*paths),
        )

    def merge(self, other: Record) -> Record:
        """Return the union of this Record and *other* (see :meth:`_NamedTree.merge`).

        Both records' authoritative specs are preserved by merging their
        ``event_template``\\ s.
        """
        return type(self)(
            self._leaves_merged(other),
            event_template=self.event_template.merge(other.event_template),
        )

    def replace(self, _updates: Mapping[str, Any] | None = None, /, **updates: Any) -> Record:
        """Return a new Record with the values at the given paths replaced.

        The structural contract is :meth:`_NamedTree.replace`; here untouched
        fields keep their authoritative specs and each replaced field takes its
        new value's inferred spec (via the threaded ``event_template``).
        """
        resolved = self._resolve_replace_updates(_updates, updates)
        if not resolved:
            return self
        flat = self._leaves_replaced(resolved)
        spec_updates = {self._norm_path(p): self._spec_of(v) for p, v in resolved.items()}
        return type(self)(flat, event_template=self.event_template.replace(spec_updates))

    def _spec_of(self, value: _FieldValue) -> Any:
        """The leaf spec describing a new field *value*, for template threading."""
        if isinstance(value, Record):
            return value.event_template
        return EventTemplate.infer_from({"_leaf_": value}).children["_leaf_"]

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
            {
                name: val.to_numeric() if isinstance(val, Record) else val
                for name, val in self._tree.items()
            }
        )

    # -- Coercion -----------------------------------------------------------

    @classmethod
    def ensure(cls, x: Any) -> Record:
        """Coerce *x* to Record if it isn't already.

        - ``Record`` → pass through
        - ``dict`` → ``Record(**x)``
        - array-like → ``Record(data=x)``
        """
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            return cls(x)
        return cls(data=x)

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, ArrayLike | Record]) -> Record:
        """Construct Record from a dict of arrays."""
        return cls(d)

    @classmethod
    def from_nested_dict(
        cls, data: Mapping[str, Any], *, event_template: EventTemplate | None = None
    ) -> Record:
        """Build a Record from a **nested** ``dict`` (see :meth:`_NamedTree.from_nested_dict`).

        With *event_template* given, the template — not the Python type — decides
        structure at each position: a ``dict`` where the template has an interior
        node is recursed into, while a ``dict`` where the template has a leaf spec
        is kept verbatim as opaque data. The template is then carried onto the
        result exactly as in normal construction.
        """
        if event_template is None:
            return super().from_nested_dict(data)
        flat = cls._flatten_paths(data, recurse_into=lambda path: not event_template.is_leaf(path))
        return cls(flat, event_template=event_template)

    # -- Leaf-wise operations -----------------------------------------------
    #
    # ``map`` / ``map_with_keys`` are inherited from ``_NamedTree`` (they apply a
    # function to each field value and rebuild the same structure, re-inferring
    # the result's per-leaf specs). See ``_NamedTree.map``.

    # A record's leaves in canonical order are ``list(record.values())``;
    # reconstruct via ``record.event_template.from_field_values(...)``.

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

        Because a template absent at construction is *inferred from the data*,
        two records built from equal data without explicit templates always have
        equal templates — the template check only ever distinguishes records that
        were given structurally different explicit schemas.

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
        if self.fields != other.fields:
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
    return Record(**{f: field_kwargs[f] for f in fields})


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------


def _unpickle_record(store: dict, name: str, source) -> Record:
    r = Record(name=name, **store)
    if source is not None:
        object.__setattr__(r, "_source", source)
    return r


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_flatten(v: Record) -> tuple[list, tuple[str, ...]]:
    """Flatten Record for JAX pytree traversal.

    Leaves are the stored values exactly as-is. JAX will further traverse
    any nested ``Record`` children it encounters because ``Record`` is a
    registered pytree type. Non-pytree objects (strings, opaque objects)
    become pytree leaves themselves, and any leaf-wise transformation
    applied by JAX must accept them.
    """
    children = list(v._tree.values())
    return children, v.fields


def _record_unflatten(aux: tuple[str, ...], children: list) -> Record:
    """Unflatten Record from JAX pytree traversal."""
    return Record(dict(zip(aux, children)))


jax.tree_util.register_pytree_node(Record, _record_flatten, _record_unflatten)
