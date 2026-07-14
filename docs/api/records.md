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

## Auxiliary-metadata registry

`Record` → `NumericRecord` conversion drops backend-specific metadata
(xarray `dims` / `coords` / `attrs`, pandas `index` / `columns` /
`dtypes`); the auxiliary registry round-trips it so `to_native()`
reproduces the original container.

::: probpipe.register_aux

::: probpipe.aux_for

::: probpipe.AuxHooks
