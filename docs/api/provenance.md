# Provenance

Every `Distribution` or `Record` returned by a workflow function carries
a `Provenance` record linking it to its inputs and the op that produced
it. The result is a directed acyclic graph: each node is a value, each
edge points from a value to one of its inputs.

`provenance_ancestors(value)` returns the transitive set of ancestors that
went into producing `value`. `provenance_dag(value)` renders the same
information as a Graphviz `Digraph` — useful for debugging or displaying
lineage in a notebook.

## Tracking modes

How much history is retained is controlled by a global `ProvenanceMode`:

| Mode | What is stored | Memory cost |
|------|----------------|-------------|
| `LIGHTWEIGHT` (default) | `ParentInfo` descriptors — type name, distribution name, and the parent's own provenance chain | Low — parent data arrays can be GC'd |
| `FULL` | `ParentInfo` plus a live reference to the parent object via `.obj` | Higher — full ancestry stays in memory |
| `OFF` | Nothing — `dist.source` is `None` | Zero |

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
from probpipe import Normal, provenance_ancestors, ProvenanceMode
import probpipe

prior = Normal(loc=0.0, scale=1.0, name="prior")
posterior = wf(prior)

ancestors = provenance_ancestors(posterior)
anc = ancestors[0]
print(anc.type_name)   # "Normal"
print(anc.name)        # "prior"
print(anc.obj)         # None in LIGHTWEIGHT — parent may have been GC'd
```

To access the live parent object, switch to FULL mode before running the
workflow:

```python
probpipe.provenance_config.mode = ProvenanceMode.FULL

posterior = wf(prior)
anc = provenance_ancestors(posterior)[0]
anc.obj                       # the live Normal distribution
anc.obj.sample(key, (100,))   # sample from it
```

## Migration from the pre-LIGHTWEIGHT API

Before `ProvenanceMode` was introduced, `provenance_ancestors` always
returned live distribution objects.  Code that relied on this needs two
small changes:

```python
# Old — ancestors were live Distribution objects
ancestors = provenance_ancestors(result)
assert prior in ancestors          # identity check
ancestors[0].sample(key, (10,))   # sampling from ancestor

# New — ancestors are ParentInfo descriptors by default
ancestors = provenance_ancestors(result)
assert any(a.name == "prior" for a in ancestors)   # name-based check

# Or: opt in to FULL mode for live-object access
probpipe.provenance_config.mode = ProvenanceMode.FULL
ancestors = provenance_ancestors(result)
assert any(a.obj is prior for a in ancestors)
ancestors[0].obj.sample(key, (10,))
```

## API reference

::: probpipe.ProvenanceMode

::: probpipe.ParentInfo

::: probpipe.Provenance

::: probpipe.provenance_ancestors

::: probpipe.provenance_dag
