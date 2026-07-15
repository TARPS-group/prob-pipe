# Records and data

Named, immutable containers for structured non-random data, plus the
batched (`RecordArray`) and parameter-sweep (`Design`) variants built on
top.

Field access is bracket-only: `record["x"]`, `array["x"]`. Slash-delimited
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
`replace` / `with_path_names`), and nested-dict (de)construction.

::: probpipe.NamedTree

## Record arrays

::: probpipe.RecordArray

::: probpipe.NumericRecordArray

## Weights

::: probpipe.Weights

## Parameter-sweep designs

`FullFactorialDesign(**marginals)` materialises the Cartesian product of
per-field marginals as a sweep-ready `RecordArray`.

::: probpipe.Design

::: probpipe.FullFactorialDesign

## Array-backend registry

`NumericRecord` stores each leaf in its native form (an `xarray.DataArray`
keeps its dims / coords / attrs, a `pandas` object its index / columns /
dtypes) and converts to `jax.Array` lazily at the compute boundary.
Containers speaking the numpy protocol need no registration; registering an
`ArrayBackend` makes any other container type a first-class numeric leaf —
recognised by template inference and `ArraySpec.is_valid`, promoted,
converted at the boundary, and fingerprinted by content.

::: probpipe.register_array_backend

::: probpipe.array_backend_for

::: probpipe.ArrayBackend
