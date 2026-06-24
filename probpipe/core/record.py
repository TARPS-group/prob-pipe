"""Record — ProbPipe's structured value type.

A ``Record`` is a single structured value: a named, immutable,
JAX-pytree-registered container of fields. It is the non-random counterpart to
:class:`~probpipe.core._distribution_base.Distribution` — the value you get back
from a draw, an observation, or a workflow-function result — and it pairs every
value with the schema that describes it.

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
| :class:`~probpipe.NumericRecord` (subclass) | single value, every leaf coerced to ``jax.Array``; adds ``to_vector`` and reductions. |
| :class:`~probpipe.RecordArray` | a batch of ``Record``s sharing one ``EventTemplate``; each field is shaped ``(*batch_shape, *leaf_shape)``. |
| :class:`~probpipe.NumericRecordArray` (subclass) | a batch of ``NumericRecord``s; adds ``to_vector`` / ``mean`` / ``var``. |

The structural schema itself — :class:`EventTemplate` /
:class:`~probpipe.NumericEventTemplate` and the leaf specs — lives in
:mod:`probpipe.core.event_template`; reach for it directly when you need to
describe structure *without* an example value.

**When to reach for which**

* :class:`Record` — heterogeneous fields, or when you want to keep the original
  backend objects intact (it coerces nothing).
* :class:`~probpipe.NumericRecord` — every leaf numeric: gives a uniform
  ``jax.Array`` type, a flat 1-D vector (``to_vector`` /
  :meth:`~probpipe.NumericEventTemplate.from_vector`), and reductions.
* :class:`~probpipe.RecordArray` / :class:`~probpipe.NumericRecordArray` —
  collections (e.g. posterior draws): integer indexing materialises one element,
  string indexing returns the batched field.

Usage::

    from probpipe import Record, NumericRecord

    params = NumericRecord(r=1.8, K=70.0, phi=10.0)
    data = Record(counts=np.array([2, 1, 3, 0, 5]), label="horseshoe")

    params["r"]            # jnp.array(1.8)
    params.fields          # ('r', 'K', 'phi')  — insertion order
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
**Records are boundary objects.** They carry structure, identity, and provenance
at the edges of a computation — the values you inspect, save, or flow into the
next op. Hot numerical inner loops run on raw arrays; only their inputs and
outputs are wrapped, so a ``Record``'s bookkeeping costs nothing where speed
matters.

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

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from .event_template import _PATH_SEP, EventTemplate, _check_no_path_sep
from .provenance import Provenance

if TYPE_CHECKING:
    # Annotation-only back-references: these live in modules that import
    # Record from here, so TYPE_CHECKING avoids the runtime import cycle.
    from ._numeric_record import NumericRecord

__all__ = [
    "Record",
]

# A field value: nested ``Record`` or anything else (stored as-is).
type _FieldValue = Any

# ``_PATH_SEP`` (the ``record["a/b/c"]`` separator, reserved in field names) and
# ``_check_no_path_sep`` are imported from :mod:`probpipe.core.event_template`,
# their single home — the path convention is shared by templates and values.


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Record:
    """A single structured value: named fields bound to a schema and identity.

    A ``Record`` holds **data** (named fields, stored verbatim — no coercion), a
    **structure** (:attr:`event_template`, fixed at construction), and an
    **identity** (:attr:`name` plus write-once :attr:`source` provenance). It is
    immutable and registered as a JAX pytree (the field values are the leaves,
    the field names the aux). Fields iterate in insertion order; ``/`` is
    reserved as the nested-path separator and rejected in field names.

    Use :class:`NumericRecord` when every leaf is numeric and you want a uniform
    ``jax.Array`` type plus 1-D vector (de)serialization.

    Parameters
    ----------
    _dict : dict, optional
        Fields as a positional mapping — an alternative to keyword ``**fields``
        (passing both raises). Use it when a field name would collide with the
        ``name`` / ``event_template`` keywords.
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
        If no fields are given, a field name contains ``/``, both ``_dict`` and
        keyword fields are passed, or a supplied ``event_template`` does not
        match the field names.

    Notes
    -----
    Two records are equal iff they share a type, an equal :attr:`event_template`,
    and field-by-field equal data (see :meth:`__eq__`). :attr:`name`,
    :attr:`source`, and :attr:`event_template` are runtime metadata and are not
    serialised into the pytree aux, so a ``tree_unflatten``'d or unpickled record
    re-derives them.
    """

    __slots__ = ("_event_template", "_name", "_source", "_store")

    def __init__(
        self,
        _dict: dict[str, _FieldValue] | None = None,
        /,
        *,
        name: str | None = None,
        event_template: EventTemplate | None = None,
        **fields: _FieldValue,
    ):
        if _dict is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            fields = _dict
        if not fields:
            raise ValueError("Record requires at least one named field")
        for field_name in fields:
            _check_no_path_sep(field_name)
        store = dict(fields)
        object.__setattr__(self, "_store", store)
        if name is None:
            name = "record(" + ",".join(store.keys()) + ")"
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_source", None)
        # Authoritative structural schema, fixed at construction. When supplied
        # (e.g. carried forward from the generator that produced this value) it
        # is validated against the data; otherwise it is inferred once, now,
        # from the field data. Every Record has a template from birth.
        if event_template is None:
            event_template = EventTemplate.infer_from(store)
        else:
            self._validate_event_template(event_template)
        object.__setattr__(self, "_event_template", event_template)

    def _validate_event_template(self, event_template: EventTemplate) -> None:
        """Check that an explicitly-supplied template matches this value's fields.

        Raises ``ValueError`` if the template's top-level field names differ
        from this record's. (Per-leaf shape / dtype conformance is the
        generator's responsibility; only the structural field set is checked
        here.)
        """
        if set(event_template.fields) != set(self._store):
            raise ValueError(
                f"event_template fields {sorted(event_template.fields)} do not "
                f"match record fields {sorted(self._store)}"
            )

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
        ``merge``, ``without``, ``map``, ``map_with_names``) return a
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
        return (_unpickle_record, (dict(self._store), self._name, self._source))

    # -- Field access -------------------------------------------------------

    def __getitem__(self, key: str | tuple[str, ...]) -> _FieldValue:
        if isinstance(key, str):
            if _PATH_SEP in key:
                return self[tuple(key.split(_PATH_SEP))]
            store = self._store
            if key not in store:
                raise KeyError(key)
            return store[key]
        if isinstance(key, tuple):
            v: Any = self
            for i, k in enumerate(key):
                if i > 0 and not isinstance(v, Record):
                    # Descending past a non-Record leaf is a path
                    # error, not an indexing error — translate so that
                    # ``"a/b" in record`` and ``record["a/b"]`` both
                    # raise the user-friendly KeyError instead of
                    # leaking a numpy/list-side IndexError.
                    raise KeyError(
                        f"path {'/'.join(key)!r} descends through "
                        f"non-Record leaf {type(v).__name__} at "
                        f"{'/'.join(key[:i])!r}"
                    )
                v = v[k]
            return v
        raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names in insertion order."""
        return tuple(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        if _PATH_SEP in name:
            try:
                self[name]
            except (KeyError, TypeError, IndexError):
                return False
            return True
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def items(self) -> Iterator[tuple[str, _FieldValue]]:
        """Iterate over (name, value) pairs."""
        return iter(self._store.items())

    def keys(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._store)

    def values(self) -> Iterator[_FieldValue]:
        """Iterate over values."""
        return iter(self._store.values())

    # -- Selection ----------------------------------------------------------

    def select(self, *fields: str, **mapping: str) -> dict[str, _FieldValue]:
        """Select fields as a dict, for splatting into function calls.

        Positional args use the field name as the key (identity mapping).
        Keyword args remap: ``select(x="field_name")`` → ``{"x": self.field_name}``.

        Usage::

            predict(**params.select("r", "K"), x=x_grid)
            predict(**params.select(growth_rate="r"), x=x_grid)
        """
        result: dict[str, _FieldValue] = {}
        for f in fields:
            if f not in self._store:
                raise KeyError(f"No field {f!r} in Record")
            result[f] = self[f]
        for arg_name, field_name in mapping.items():
            if field_name not in self._store:
                raise KeyError(f"No field {field_name!r} in Record")
            result[arg_name] = self[field_name]
        return result

    def select_all(self) -> dict[str, _FieldValue]:
        """Return every field as a dict, for splatting into function calls.

        Sugar for ``select(*self.fields)``. Subclasses whose
        ``__getitem__`` returns a view (``RecordArray`` →
        ``_RecordArrayView``, ``RecordDistribution`` →
        ``_RecordDistributionView``) inherit this method and return
        per-field views — so ``f(**ra.select_all())`` triggers the
        parent-identity zip sweep in ``WorkflowFunction``, and
        ``f(**dist.select_all())`` similarly preserves cross-field
        correlation.
        """
        return self.select(*self.fields)

    # -- Immutable updates --------------------------------------------------

    def replace(self, **updates: ArrayLike | Record) -> Record:
        """Return a new Record with specified fields replaced.

        Returns an instance of ``type(self)`` so that subclasses
        (``NumericRecord``) preserve their class through the update.
        """
        new = dict(self._store)
        for k, v in updates.items():
            if k not in new:
                raise KeyError(f"Cannot replace non-existent field {k!r}")
            new[k] = v
        return type(self)(new)

    def merge(self, other: Record) -> Record:
        """Return a new Record combining fields from self and other.

        Raises ``ValueError`` if any field names overlap. Returns an
        instance of ``type(self)``.
        """
        overlap = set(self._store) & set(other._store)
        if overlap:
            raise ValueError(f"Overlapping field names: {overlap}")
        combined = dict(self._store)
        combined.update(other._store)
        return type(self)(combined)

    def without(self, *names: str) -> Record:
        """Return a new Record with the specified fields removed.

        Returns an instance of ``type(self)``.
        """
        new = {k: v for k, v in self._store.items() if k not in names}
        if not new:
            raise ValueError("Cannot remove all fields from Record")
        return type(self)(new)

    # -- Backend conversion -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of stored values (recursive for nested Record).

        Leaves are returned verbatim; no coercion to numpy or JAX.
        """
        result: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                result[name] = val.to_dict()
            else:
                result[name] = val
        return result

    def to_numpy(self) -> dict[str, Any]:
        """Return a dict of numpy arrays (recursive for nested Record).

        Each numeric leaf is converted via ``np.asarray``. Non-numeric
        leaves (strings, opaque objects) are returned as-is. Backend
        metadata (xarray dims / coords, pandas index) is stripped — use
        :meth:`to_numeric` followed by :meth:`NumericRecord.to_native`
        if you need a metadata-preserving round-trip.
        """
        result: dict[str, Any] = {}
        for name, val in self._store.items():
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
                for name, val in self._store.items()
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

    # -- Leaf-wise operations -----------------------------------------------

    def map(self, fn: Callable[[Any], Any]) -> Record:
        """Apply *fn* to each leaf, returning a new Record.

        Nested ``Record`` objects are traversed and rebuilt with the same
        class. ``fn`` sees leaves as stored (no coercion).
        """
        fields: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                fields[name] = val.map(fn)
            else:
                fields[name] = fn(val)
        return type(self)(fields)

    def map_with_names(self, fn: Callable[[str, Any], Any]) -> Record:
        """Apply *fn(name, value)* to each leaf, returning a new Record."""
        fields: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                fields[name] = val.map_with_names(fn)
            else:
                fields[name] = fn(name, val)
        return type(self)(fields)

    # -- JAX-pytree (de)serialization ---------------------------------------

    def flatten(self) -> tuple[list[Any], jax.tree_util.PyTreeDef]:
        """Decompose into ``(leaves, aux)``, à la :func:`jax.tree_util.tree_flatten`.

        The general **JAX-pytree** operation, defined for *any* leaf type
        (numeric or opaque). Returns this record's pytree *leaves* — the
        stored values in canonical leaf order (insertion order, descending
        depth-first into nested records) — together with the structural
        *aux* (a :class:`jax.tree_util.PyTreeDef`). :meth:`unflatten` is the
        inverse: ``Record.unflatten(*record.flatten()) == record``.

        This is distinct from :meth:`EventTemplate.to_vector`. ``flatten``
        keeps each leaf whole (any type) and carries the full structure in
        ``aux``; ``to_vector`` applies only to *numeric* values and goes
        further, ravelling and concatenating the numeric leaves into a
        single dense 1-D ``vec``.

        Returns
        -------
        tuple
            ``(leaves, aux)`` — ``leaves`` is the list of pytree leaves in
            canonical leaf order; ``aux`` is the ``jax.tree_util.PyTreeDef``
            describing the structure.

        See Also
        --------
        unflatten : Reconstruct a value from ``(leaves, aux)`` (the inverse).
        EventTemplate.to_vector : Dense 1-D serialization of a numeric value.
        """
        return jax.tree_util.tree_flatten(self)

    @staticmethod
    def unflatten(leaves: list[Any], aux: jax.tree_util.PyTreeDef) -> Any:
        """Reconstruct a value from ``(leaves, aux)``, à la :func:`jax.tree_util.tree_unflatten`.

        The inverse of :meth:`flatten`: rebuilds the original (possibly
        nested) value from its pytree *leaves* and the *aux* structure.
        Defined for any leaf type; the concrete result class is encoded in
        ``aux``, so ``Record.unflatten`` and ``NumericRecord.unflatten``
        behave identically (the static method does not depend on which
        class it is called on).

        Parameters
        ----------
        leaves : list
            Pytree leaves in canonical leaf order, as returned by
            :meth:`flatten`.
        aux : jax.tree_util.PyTreeDef
            The structural definition returned by :meth:`flatten`.

        Returns
        -------
        Any
            The reconstructed value, of the class encoded in ``aux``.

        See Also
        --------
        flatten : Decompose a value into ``(leaves, aux)`` (the inverse).
        EventTemplate.from_vector : Reconstruct a numeric value from a dense 1-D ``vec``.
        """
        return jax.tree_util.tree_unflatten(aux, leaves)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, val in self._store.items():
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
        if len(self._store) != 1:
            raise TypeError(
                f"{type(self).__name__} with {len(self._store)} fields is not "
                f"callable; access a specific field with record['field_name'] "
                f"first."
            )
        only = next(iter(self._store.values()))
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
        for name, a in self._store.items():
            b = other._store[name]
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
        for name, val in self._store.items():
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

    The general "named values → validated :class:`Record`" operation behind
    :meth:`EventTemplate.pack`. Raises ``TypeError`` if any field is missing
    or unexpected; otherwise returns a ``Record`` keyed in *fields* order.
    *owner* (optional) prefixes the error message — the calling distribution or
    template class name.

    :meth:`Distribution._pack_value` calls this directly rather than through
    :meth:`EventTemplate.pack` because some distributions expose ``fields``
    without a ``EventTemplate`` instance.
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
    children = list(v._store.values())
    return children, v.fields


def _record_unflatten(aux: tuple[str, ...], children: list) -> Record:
    """Unflatten Record from JAX pytree traversal."""
    return Record(dict(zip(aux, children)))


jax.tree_util.register_pytree_node(Record, _record_flatten, _record_unflatten)
