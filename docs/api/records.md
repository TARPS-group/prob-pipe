# Records and data

Named, immutable containers for structured non-random data, plus the
batched (`RecordBatch`) and parameter-sweep (`Design`) variants built on
top.

Field access is bracket-only: `record["x"]`, `batch["x"]`. Slash-delimited
strings index nested paths: `record["params/intercept"]`.

## Records

::: probpipe.Record

::: probpipe.NumericRecord

::: probpipe.RecordSpec

::: probpipe.NumericRecordSpec

## Kind and component declarations

All kinds use `TermSpec`. `NumericSpec` additionally exposes the flat numeric
size. `RecordSpec` is both the record-kind declaration and its schema; it
replaces `EventTemplate`, with `NumericRecordSpec` replacing its numeric form.

`InputSpec` maps Python parameter names to term specs. `OutputSpec` distinguishes
one named whole value from exposed record fields:

```python
from probpipe import InputSpec, NumericArraySpec, OutputSpec, RecordSpec

beta = NumericArraySpec(("p",))
sigma = NumericArraySpec(())
inputs = InputSpec(data=NumericArraySpec(("n", "p")))
whole_array = OutputSpec(beta=beta)
pending_type = OutputSpec(beta=None)
one_field_record = OutputSpec(RecordSpec(beta=beta))
record_fields = OutputSpec(beta=beta, sigma=sigma)
whole_record = OutputSpec(parameters=RecordSpec(beta=beta, sigma=sigma))
```

The single-keyword form describes the whole returned value; it inserts no
single-field record. The positional record form exposes immediate children
regardless of field count. `spec` and `components` are read-only derived views;
nested records stay nested. Only a single named whole value can carry a `None`
type hole. Replace that declaration with the same component name and a known
spec when the type becomes available.

Symbolic dimensions share one scope across nested specs and input slots.
`TermSpec`, `InputSpec`, and `OutputSpec` provide `with_dim_sizes` to substitute
supplied sizes and leave the rest symbolic, and `with_dim_names` to rename
symbols simultaneously. `TermSpec` and `InputSpec` also provide
`bind_dims_from_value` and `bind_dims_from_spec`, which return refined specs
and reject conflicting sizes.

Value validation reads the actual fields of a `Record` or mapping, including
array shapes and dtypes. Binding from another spec uses only the information
that declaration supplies. The same spec-binding rules apply directly and
inside records, input slots, or batches.

These shared declarations do not yet replace the legacy live `Function`
input/output-template or distribution event-template constructor APIs.

`DistributionSpec` carries a record draw schema. Concrete value validation
requires an exact schema match; dimension binding can learn sizes from a
distribution's schema or another distribution declaration. `FunctionSpec`
optionally declares the input and output of a callable. Its validity check is
callability alone, while binding reads available declarations without running
the callable. An undeclared callable side leaves its dimensions symbolic.

::: probpipe.TermSpec

::: probpipe.NumericSpec

::: probpipe.NumericArraySpec

::: probpipe.OpaqueSpec

::: probpipe.DistributionSpec

::: probpipe.FunctionSpec

::: probpipe.InputSpec

::: probpipe.OutputSpec

## The tree substrate

`Record` and `RecordSpec` are both built on `NamedTree`, the shared
named, ordered tree that owns the leaf-keyed mapping interface, path
navigation, the structure-preserving edits (`merge` / `without` /
`replace` / `with_path_names`), and nested-dict export (`to_nested_dict`)
that the constructor reads back.

::: probpipe.NamedTree

## The multiplicity axis

`Batch` is the shared substrate for collections: it says *how many* objects
there are, separately from what one object contains, so `len` / `iter` /
`batch_shape` / `batch_size` speak only about the batch axes. Axes are
partitioned into ordered, named **levels**, addressed by name with `at_levels`
or by position with `[]`, and indexing returns a view named by what it selected
— draw 7 of chain 0 is named `"posterior[chain=0, draw=7]"`. A batch's type is a
`BatchSpec`: the element's specification together with that named multiplicity.

::: probpipe.Batch

::: probpipe.BatchSpec

### Batch forms that store objects

Every value spec has one tracked class and one batch form. A numeric array
batches natively, with the batch axes leading, but native storage is not
identity: `NumericArrayBatch` is what carries a level name, a specification, and
provenance, so the array kind takes a class like every other. A callable and an
opaque object have no native stacking at all — there is nothing to stack them
into — so each gets a thin `Batch` that stores its elements and carries the one
specification they all satisfy, adding no other interface.

Both take their elements the same way. Pass a flat sequence, or an object array
of any shape to give the batch more than one axis; a nested sequence is *not*
unpacked, since what nesting means for an arbitrary object is the caller's to
decide. Elements are never looked inside, so a batch of two arrays stays a batch
of two things rather than becoming one 2-d array. The name comes first, as it does for a
`Record` and an `Opaque`. Every level takes a name, with one axis per level unless
`axes_per_level` says how many each holds, and every element is checked against
the shared specification at construction.

Two consequences worth knowing. The store is frozen and a supplied array is
copied — the pointer array only, so the elements themselves stay shared — so a
batch holds the elements it validated even if the caller keeps writing to the
array they passed. And a batch of no elements is a batch: `OpaqueBatch("draws", [], "draw")`
and `batch[0:0]` both give one, since zero is a count the level can carry. What a
batch does need is an *axis* — a single object with none has no level to count
along, and is refused.

::: probpipe.NumericArray

::: probpipe.NumericArrayBatch

::: probpipe.Opaque

::: probpipe.FunctionBatch

::: probpipe.OpaqueBatch

## Record batches

::: probpipe.RecordBatch

::: probpipe.NumericRecordBatch

## Weights

::: probpipe.Weights

## Parameter-sweep designs

`FullFactorialDesign(**marginals)` materialises the Cartesian product of
per-field marginals as a sweep-ready `RecordBatch`.

::: probpipe.Design

::: probpipe.FullFactorialDesign

## Array-backend registry

`NumericRecord` stores each leaf in its native form (an `xarray.DataArray`
keeps its dims / coords / attrs, a `pandas` object its index / columns /
dtypes) and converts to `jax.Array` lazily at the compute boundary.
Containers speaking the numpy protocol need no registration; registering an
`ArrayBackend` makes any other container type a first-class numeric leaf —
recognised by template inference and `NumericArraySpec.is_valid`, promoted,
converted at the boundary, and fingerprinted by content.

::: probpipe.register_array_backend

::: probpipe.array_backend_for

::: probpipe.ArrayBackend
