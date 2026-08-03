# Identity & provenance

Every `Distribution` or `Record` returned by a Function carries a
`Provenance` record describing the operation that produced it. Tracked-term inputs
are lineage `parents`; resolved ordinary values are fingerprinted separately
under `inputs`. The resulting directed acyclic graph traverses only tracked
parents, while ordinary arrays, scalars, defaults, and Module-provided values
still distinguish otherwise identical calls.

`provenance_ancestors(value)` returns the transitive set of ancestors that
went into producing `value`. `provenance_dag(value)` renders the same
information as a Graphviz `Digraph` — useful for debugging or displaying
lineage in a notebook.

## The `TrackedTerm` and `Annotated` mixins

The identity attributes and methods are defined once, by two mixins in
`probpipe.core.tracked`, and shared by every core object:

- **`TrackedTerm`** — a `name`, a `name_is_auto` flag recording whether the name
  was auto-derived by the operation that produced the object (`True`) or
  supplied by the user (`False`), and a write-once `provenance` attached via
  `with_provenance(...)`. `with_name(name)` returns a shallow copy under a
  new user-given name, with provenance recording the rename. `Distribution`,
  `Record`/`NumericRecord`, and the batch types are all `TrackedTerm`.
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
| `LIGHTWEIGHT` (default) | `ParentInfo` descriptors, exact controls, and diagnostics; tracked parents also carry their own provenance chain | Low — parent and input values can be GC'd |
| `FULL` | The same controls and diagnostics, plus a live reference to each tracked parent or plain input via `.parent` | Higher — full ancestry and call inputs stay in memory |
| `OFF` | Nothing — `dist.provenance` is `None`, including no RNG recipe | Zero |

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

Descriptive operation controls such as dispatch and sample count remain in
provenance `metadata`; they are not duplicated in `inputs`. A stochastic result
may separately encode the canonical replay plan in exact `controls`.

## Metadata, controls, and diagnostics

These three mappings have deliberately different authority:

- `metadata` holds descriptive operation information. Serialization preserves
  common JSON values but may stringify unsupported objects, so replay never
  treats metadata as an exact compatibility record.
- `controls` holds exact JSON-native data that can affect reproducibility or
  replay compatibility. Workflow RNG recipes and callable/plan anchors live
  here. Construction and serialization reject non-JSON-native values rather
  than weakening them.
- `diagnostics` holds exact JSON-native observations that must not define the
  mathematical call: the route used, source-artifact locations, and replay
  drift observations, for example. Route changes can therefore be reported
  without changing RNG identity.

`FULL` and `LIGHTWEIGHT` record equivalent `controls` and `diagnostics`.
`OFF` creates no provenance and therefore no replay recipe. Records written by
older ProbPipe versions remain readable; absent fields are restored as empty
mappings.

## Workflow RNG recipes and replay

A successful workflow-owned stochastic call records its structural RNG root,
event identities, stochastic plan, execution capability, and strong callable
anchor in `Provenance.controls`. The recipe records how ProbPipe-owned keys were
derived; it does not store raw derived keys, worker ownership, retry attempts,
or runtime ledgers. Deterministic, wholly exact, and exclusively caller-keyed
operations do not create an RNG recipe.

Recipes survive a JSON serialization round-trip and can drive a validated
re-execution:

```python
import json

from probpipe import Normal, Provenance, replay_run, sample, workflow_run

with workflow_run(seed=42):
    original = sample(Normal(loc=0.0, scale=1.0, name="draw"))

payload = json.loads(json.dumps(original.provenance.to_dict()))
restored = Provenance.from_dict(payload)

with replay_run(restored):
    repeated = sample(Normal(loc=0.0, scale=1.0, name="draw"))
```

The replay scope restores the recorded root and occurrence path, then checks
the current callable, canonical stochastic plan, execution capability, and
every expected random event before key derivation. The caller still supplies
and executes the current code and inputs: `replay_run` neither loads code nor
reconstructs a call from provenance.

One replay scope must contain exactly one top-level `Function.__call__`.
An empty scope, a second top-level call, `Function.apply`, or nesting with
`workflow_run` is rejected. A parent callable containing a nested automatic-key
call is not eligible for standalone replay.

Strong callable anchors are intentionally narrow. Replayable user definitions
must be importable, module-level, closure-free Python `def` functions that
still resolve to the same definition. Lambdas, local functions, closures,
bound methods, partials, callable objects, classes, and builtins may execute
normally, but replay raises `ReplayUnsupportedCallableError` rather than using
a weak identity.

The recorded execution route is diagnostic, not authoritative. Replay may move
between compatible rowwise, thread, JAX, or Prefect routes; any drift is written
to the new result's diagnostics. This validates structural RNG identity and
declared provider contracts, but it is not a semantic replay facility and does
not promise bit-identical floating-point results across execution routes.
Malformed or incompatible recipes, plan/provider drift, and unexpected or
duplicate events raise `ReplayCompatibilityError` before the affected random
event is sampled. Missing expected events are reported when the invocation
completes. Legacy records and `ProvenanceMode.OFF` results have no recipe to
guess from and are rejected explicitly.

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

## Best-effort fingerprints

Every tracked-parent or plain-input `ParentInfo` descriptor carries a
`fingerprint` — a 16-character best-effort digest — plus
`fingerprint_is_weak`, which states whether any part of that digest used
process-local identity. Both are populated automatically by
`Provenance.create()` and visible in `to_dict()` output:

```python
prior = Normal(loc=0.0, scale=1.0, name="prior")
posterior = wf(prior)

anc = provenance_ancestors(posterior)[0]
print(anc.fingerprint)   # e.g. "8d86780c50cea472"
print(anc.fingerprint_is_weak)  # False for this structured distribution

d = posterior.provenance.to_dict()
print(d["parents"][0]["fingerprint"])   # same digest
```

The classification is conservative. Known content-bearing values receive
strong structural fingerprints; unsupported objects and callable forms whose
runtime behavior cannot be recovered safely receive process-local identity
fingerprints marked as weak. Weakness propagates through records, containers,
closures, and other composite values.

| Parent type | What is hashed |
|---|---|
| TFP-backed distribution (`Normal`, `Gamma`, …) | class name + distribution name + all TFP constructor parameters |
| `EmpiricalDistribution` | class name + name + sample arrays + log-normalised weight array |
| `Record` | field names + values, recursively |
| `Function` | frozen signature and input/output templates, plus user-function bytecode/defaults/closure or the private implementation type |
| JAX / NumPy array | shape + dtype + raw bytes (large arrays are sampled) |
| Closure-free Python function | module + qualified name + bytecode + defaults |
| Closure-bearing function, bound method, partial, callable instance, class, builtin, or unsupported object | process-local identity; marked weak |

The fingerprint is intended as the foundation for a future Prefect
`cache_key_fn`. A cross-run cache must fail closed when
`fingerprint_is_weak` is true rather than treating that digest as portable
content identity.

## API reference

::: probpipe.TrackedTerm

::: probpipe.Annotated

::: probpipe.ProvenanceMode

::: probpipe.ParentInfo

::: probpipe.Provenance

::: probpipe.replay_run

::: probpipe.ReplayCompatibilityError

::: probpipe.ReplayUnsupportedCallableError

::: probpipe.provenance_ancestors

::: probpipe.provenance_dag
