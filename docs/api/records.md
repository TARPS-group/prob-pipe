# Records and data

`Record` is the universal container for non-random structured data;
[`Distribution`](distributions.md) is the universal container for random
quantities. Both support named fields, `select()` for workflow-function
splatting, and JAX pytree traversal — see CONTRIBUTING.md for the full
"parallel containers" framing.

Field access is bracket-only: `record["x"]`, `array["x"]`, `dist["x"]`.
Slash-delimited strings (`record["params/intercept"]`) index nested paths.

## Records

`Record` holds named, immutable values with no coercion. `NumericRecord`
adds the invariant that every leaf is a `jax.Array`, plus flatten / unflatten
helpers backed by a `RecordTemplate`.

::: probpipe.Record

::: probpipe.NumericRecord

### Templates

`RecordTemplate` is the structural skeleton (field names + per-field shapes
or `None`); `NumericRecordTemplate` is the numeric specialisation that backs
flatten / unflatten round-trips. Calling `RecordTemplate(...)` directly
auto-promotes to `NumericRecordTemplate` when all shapes are concrete.

::: probpipe.RecordTemplate

::: probpipe.NumericRecordTemplate

## Record arrays

A `RecordArray` is a batch of `Record` elements sharing a `RecordTemplate`:
integer indices select an element; field indices return a batched array.
`NumericRecordArray` is the numeric specialisation with `flatten` / `mean` /
`var`.

::: probpipe.RecordArray

::: probpipe.NumericRecordArray

## Weights

`Weights` carries a 1-D array of (typically normalised) weights for use
with weighted empirical distributions and resampling.

::: probpipe.Weights

## Parameter-sweep designs

`Design` is a `RecordArray` subclass carrying per-field marginals;
`FullFactorialDesign(**marginals)` materialises the Cartesian product as a
sweep-ready `RecordArray`. Pipe the result into a `WorkflowFunction` as a
single `Record`-typed argument to trigger a sweep.

::: probpipe.Design

::: probpipe.FullFactorialDesign

## Auxiliary-metadata registry

`Record` → `NumericRecord` conversion goes through `jnp.asarray`, which drops
backend-specific metadata (xarray `dims` / `coords` / `attrs`, pandas
`index` / `columns` / `dtypes`). The auxiliary registry round-trips that
metadata so `to_native()` reproduces the original container.

::: probpipe.register_aux

::: probpipe.aux_for

::: probpipe.AuxHooks
