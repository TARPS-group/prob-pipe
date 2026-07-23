# Identity & provenance

Every `Distribution` or `Record` returned by a Function carries a
`Provenance` record describing the operation that produced it. Tracked inputs
are lineage `parents`; resolved ordinary values are fingerprinted separately
under `inputs`. The resulting directed acyclic graph traverses only tracked
parents, while ordinary arrays, scalars, defaults, and Module-provided values
still distinguish otherwise identical calls.

`provenance_ancestors(value)` returns the transitive set of ancestors that
went into producing `value`. `provenance_dag(value)` renders the same
information as a Graphviz `Digraph` — useful for debugging or displaying
lineage in a notebook.

## The `Tracked` and `Annotated` mixins

The identity attributes and methods are defined once, by two mixins in
`probpipe.core.tracked`, and shared by every core object:

- **`Tracked`** — a `name`, a `name_is_auto` flag recording whether the name
  was auto-derived by the operation that produced the object (`True`) or
  supplied by the user (`False`), and a write-once `provenance` attached via
  `with_provenance(...)`. `with_name(name)` returns a shallow copy under a
  new user-given name, with provenance recording the rename. `Distribution`,
  `Record`/`NumericRecord`, and the batch types are all `Tracked`.
- **`Annotated`** — a free-form `annotations` mapping for auxiliary
  information attached after construction (diagnostics, validation results).
  An `xarray.DataTree` is a valid value; fitted posteriors use it with
  `arviz/` and `diagnostics/` subtrees. `Distribution` and `Record` are
  `Annotated`.

```python
from probpipe import Normal

n = Normal(loc=0.0, scale=1.0, name="weight")
n.name          # "weight"
n.name_is_auto  # False — user-given
m = n.with_name("prior_weight")
m.provenance.operation  # "with_name"; the parent descriptor points at n
```

## Tracking modes

How much history is retained is controlled by a global `ProvenanceMode`:

| Mode | What is stored | Memory cost |
|------|----------------|-------------|
| `LIGHTWEIGHT` (default) | `ParentInfo` descriptors for tracked parents and plain inputs; tracked parents also carry their own provenance chain | Low — parent and input values can be GC'd |
| `FULL` | `ParentInfo` plus a live reference to each tracked parent or plain input via `.parent` | Higher — full ancestry and call inputs stay in memory |
| `OFF` | Nothing — `dist.provenance` is `None` | Zero |

Set the mode once at startup:

```python
import probpipe
from probpipe import ProvenanceMode

probpipe.provenance_config.mode = ProvenanceMode.FULL   # for debugging
probpipe.provenance_config.mode = ProvenanceMode.OFF    # for production without lineage
```

## Accessing ancestors

`provenance_ancestors` returns `ParentInfo` descriptors in LIGHTWEIGHT and
FULL modes:

```python
from probpipe import Normal, provenance_ancestors, ProvenanceMode, sample
import probpipe

prior = Normal(loc=0.0, scale=1.0, name="prior")
posterior = wf(prior)

ancestors = provenance_ancestors(posterior)
anc = ancestors[0]
print(anc.type_name)   # "Normal"
print(anc.name)        # "prior"
print(anc.parent)         # None in LIGHTWEIGHT — parent may have been GC'd
```

To access the live parent object, switch to FULL mode before running the
workflow:

```python
probpipe.provenance_config.mode = ProvenanceMode.FULL

posterior = wf(prior)
anc = provenance_ancestors(posterior)[0]
anc.parent                                      # the live Normal distribution
sample(anc.parent, key=key, sample_shape=(100,))  # sample from it
```

## Resolved plain inputs

`Provenance.inputs` maps stable parameter labels to `ParentInfo` descriptors.
These values are not returned by `provenance_ancestors()` and do not add DAG
edges. They capture the resolved call before lifting or sweep execution, so a
broadcast records the original Distribution as a tracked parent and retains
ordinary static arguments without substituting per-cell samples.

```python
conditioned = condition_on(joint, x=0.0)
info = conditioned.provenance.inputs["**kwargs['x']"]
print(info.fingerprint)  # differs from the fingerprint for x=5.0
```

Operation controls such as dispatch and sample count remain in provenance
`metadata`; they are not duplicated in `inputs`.

## Migration from the pre-LIGHTWEIGHT API

Before `ProvenanceMode` was introduced, `provenance_ancestors` always
returned live distribution objects.  Code that relied on this needs two
small changes:

```python
# Old — ancestors were live Distribution objects
ancestors = provenance_ancestors(result)
assert prior in ancestors          # identity check
sample(ancestors[0], key=key, sample_shape=(10,))   # sampling from ancestor

# New — ancestors are ParentInfo descriptors by default
ancestors = provenance_ancestors(result)
assert any(a.name == "prior" for a in ancestors)   # name-based check

# Or: opt in to FULL mode for live-object access
probpipe.provenance_config.mode = ProvenanceMode.FULL
ancestors = provenance_ancestors(result)
assert any(a.parent is prior for a in ancestors)
sample(ancestors[0].parent, key=key, sample_shape=(10,))
```

## Content fingerprints

Every tracked-parent or plain-input `ParentInfo` descriptor carries a
`fingerprint` — a 16-character hex string that stably identifies its content
across processes. It is populated automatically by `Provenance.create()` and
visible in `to_dict()` output:

```python
prior = Normal(loc=0.0, scale=1.0, name="prior")
posterior = wf(prior)

anc = provenance_ancestors(posterior)[0]
print(anc.fingerprint)   # e.g. "8d86780c50cea472"

d = posterior.provenance.to_dict()
print(d["parents"][0]["fingerprint"])   # same digest
```

The fingerprint covers the full content of the parent:

| Parent type | What is hashed |
|---|---|
| TFP-backed distribution (`Normal`, `Gamma`, …) | class name + distribution name + all TFP constructor parameters |
| `EmpiricalDistribution` | class name + name + sample arrays + log-normalised weight array |
| `Record` | field names + values, recursively |
| `Function` | frozen signature and input/output templates, plus user-function bytecode/defaults/closure or the private implementation type |
| JAX / NumPy array | shape + dtype + raw bytes (large arrays are sampled) |

The fingerprint is intended as the foundation for a future Prefect
`cache_key_fn` that will enable cross-run task caching and failure recovery.

## API reference

::: probpipe.Tracked

::: probpipe.Annotated

::: probpipe.ProvenanceMode

::: probpipe.ParentInfo

::: probpipe.Provenance

::: probpipe.provenance_ancestors

::: probpipe.provenance_dag
