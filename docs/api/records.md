# Records and data

Named, immutable containers for structured non-random data, plus the
batched (`RecordBatch`) and parameter-sweep (`Design`) variants built on
top.

Field access is bracket-only: `record["x"]`, `batch["x"]`. Slash-delimited
strings index nested paths: `record["params/intercept"]`.

## Records

::: probpipe.Record

::: probpipe.NumericRecord

::: probpipe.EventTemplate

::: probpipe.NumericEventTemplate

## The tree substrate

`Record` and `EventTemplate` are both built on `NamedTree`, the shared
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
of two things rather than becoming one 2-d array. Every level takes a name, with
one axis per level unless `axis_groups` states otherwise, and every element is
checked against the shared specification at construction.

Two consequences worth knowing. The store is frozen and a supplied array is
copied — the pointer array only, so the elements themselves stay shared — so a
batch holds the elements it validated even if the caller keeps writing to the
array they passed. And construction needs at least one element, while *selecting*
none is fine: an empty batch is reached with `batch[0:0]` rather than built from
an empty sequence, whose shape could not be inferred anyway.

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
