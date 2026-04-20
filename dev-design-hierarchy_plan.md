# Plan — Design hierarchy (PR 2 of issue #130)

## Motivation

The preceding PR #135 landed **direct vectorization with the product
rule** at the WorkflowFunction layer: passing a `RecordArray` (or
`DistributionArray`) of batch_shape `B` to a `@workflow_function` slot
whose hint is not that same batched type triggers a cell-by-cell sweep;
multiple array-valued inputs combine by Cartesian product.

What's missing is the **user-facing construction story for sweeps**.
Today, to sweep a workflow function over a Cartesian grid of values,
the user has to build a `NumericRecordArray` by hand — computing field
shapes, materialising the Cartesian product, tracking a
`RecordTemplate`, etc.

This PR introduces `Design`, a `RecordArray` subclass that carries
**marginals** (the candidate values per field) and **materialises them
into a sweep-ready RecordArray** according to the subclass's rule.

**Scope for this PR** is just the Cartesian case: `FullFactorialDesign`.
Random and space-filling designs (`RandomDesign`,
`LatinHypercubeDesign`, `SobolDesign`) land in follow-up PRs.

`Design` is a RecordArray, so everything the WF broadcast path already
knows about RecordArrays (sweep, wrap, provenance) applies. No new
broadcast rule; no new output type.

## Location

New `probpipe/record/` subpackage — sibling to `distributions/`,
`modeling/`, `inference/`. It houses record-adjacent constructions
that build on the `Record` / `RecordArray` family in `core/` (here:
parameter-sweep `Design`s). Re-exported from top-level `probpipe`.

## API sketch

```python
from probpipe import FullFactorialDesign

# Cartesian grid — 3 * 2 = 6 rows
ff = FullFactorialDesign(
    r=[1.5, 1.8, 2.0],
    K=[60.0, 80.0],
)
ff.batch_shape                      # (6,)
ff.fields                           # ('K', 'r')      (sorted)
ff.marginals                        # {'K': [60.0, 80.0], 'r': [1.5, 1.8, 2.0]}

# Mixed numeric + categorical marginals produce a plain RecordArray
ff2 = FullFactorialDesign(
    method=['nutpie', 'pymc'],
    scale=[0.5, 1.0],
)
ff2.batch_shape                     # (4,)

# Usage pattern A: pass the Design as a single Record-typed arg to
# trigger the full sweep (one inner call per row).
@workflow_function
def fit(p): return p["r"] * p["K"]
result = fit(p=ff)                  # NumericRecordArray, batch_shape=(6,)

# Usage pattern B: splat via `select_all()` when the inner body's ops
# broadcast naturally over the column arrays — one WF call, one JAX
# trace; no per-row iteration. Faster than pattern A for pure JAX
# arithmetic but not a general substitute (e.g. `float(x)` / f-strings
# receive arrays, not scalars, so they fail).
@workflow_function
def product(r, K): return r * K
result = product(**ff.select_all())  # NumericRecord(product=array(shape=(6,)))
```

## Class surface

```python
class Design(RecordArray):
    """RecordArray that carries its per-field marginals."""
    @property
    def marginals(self) -> Mapping[str, Sequence]: ...
    def select_all(self) -> dict[str, Any]: ...

class FullFactorialDesign(Design):
    def __init__(self, **marginals: Sequence): ...
```

### Design base

- Inherits `RecordArray`'s field store + template + pytree machinery.
- Adds a `._marginals` slot (frozen dict) and `.marginals` / `.select_all`.
- `select_all()` returns `{f: self[f] for f in self.fields}` for
  splat-based calls into JAX-vectorizable bodies. Does **not** wrap
  the values as single-field RecordArrays — so splatting does not
  trigger the WF sweep (by design; see pattern B above).

### FullFactorialDesign

- Marginals passed as keyword args of Python sequences (lists, tuples,
  numpy / jax arrays).
- Computes `np.meshgrid(..., indexing='ij')` across marginals, flattens
  each to a 1-D column of length `prod(sizes)`.
- Field dtype: if the marginal is numeric, store as `jnp.ndarray`;
  otherwise store as a `numpy.ndarray(dtype=object)` so strings /
  arbitrary objects survive into the sweep (heterogeneous path in
  `_make_stack` already handles object-dtype fields).
- Auto-infers whether to build a `NumericRecordArray` (all numeric) or
  a plain `RecordArray`.

## Tests

`tests/test_design.py`. Coverage:

- `TestFullFactorial`: shape, row order (lexicographic over sorted
  fields), mixed numeric/categorical marginals, single-field edge case,
  empty / empty-marginal raises.
- `TestDesignAsSweep`: sweep a `@workflow_function` over a
  `FullFactorialDesign` via pattern A (single Record arg) and pattern
  B (splat into JAX-vectorizable body).
- `TestMarginalsIntrospection`: `.marginals` returns a dict of the
  original sequences for the fields; ordering matches
  `design.fields`.

## Docs

- Short entry in `CONTRIBUTING.md` key-abstractions table.
- `record/` row in `CONTRIBUTING.md` package-structure tree and in
  the `STYLE_GUIDE.md` subpackage dependency graph.
- One new section in `docs/api/*.md` — pointing at
  `probpipe.record.design`.

## Roadmap after this PR

- PR 3 (follow-up): `RandomDesign` (Distribution-valued + sequence
  marginals).
- PR 4: `LatinHypercubeDesign` and `SobolDesign` (space-filling over
  continuous ranges).
- PR 5 (deferred): `FractionalFactorialDesign` (DoE-specific).

## Out of scope (for this PR)

- `RandomDesign`, `LatinHypercubeDesign`, `SobolDesign` — follow-up PRs.
- Design composition operators (`a * b`, `a | b`) — defer until a user
  asks.

## Estimated size

~150 LOC implementation + ~120 LOC tests = **~270 LOC**. One focused PR.
