> **AI-generated.** An AI assistant drafted this proposal and ran its examples, and no maintainer has reviewed it yet. Please comment on the pull request that adds it.

# A repeated parent in a routed call's provenance

The repr of a routed call's provenance can name the same parent twice:

```
Provenance('workflow.condition_on', parents=[condition_on, model, model])
Provenance('workflow.evaluate', parents=[evaluate, square, Normal, square])
```

The record is correct, and the repr reads as a bug.
In each line the two entries are different objects: the first is an input of the call, and the second is the result the selected route returned.
This note explains why the label repeats, shows what the record holds, and proposes alternatives for the team to choose from.
The examples run on `dev/overhaul` at `ab15a798`.

---

## Background

### What a provenance record holds

Each tracked term carries a `Provenance` record (design II.4).
The record holds the operation that produced the term, a `ParentInfo` descriptor for each parent, the resolved controls, and the plain inputs.
`Provenance.__repr__` prints the operation and each parent's label, or its type name when it has no label (`probpipe/core/provenance.py:141`).

Design II.4 defines the parents as "descriptors of the tracked inputs".
Design V.3 says the same of a function call: each tracked argument is a dependency, and the dependencies are the result's provenance parents.

### The parent a routed call adds

An operation such as `condition_on` or `evaluate` is a `Function` realized by routes.
A call selects one route, the route computes the result, and the return step gives that result the call's label and record.
The result a route returns can carry a record of its own.
An inference method's posterior has a record that names the method and its settings, the sampling lift's pushforward has a record that names the sample count, and a mean by component has a record that names the route of each component.

Two commits decide how the call keeps that record:
1. **04fc33a0** (2026-10-01): the call's record keeps the route's result as a further parent, so the result's lineage keeps the method that produced it.
   Without it, the record would lose the route's record when the return step copies the result.
2. **28735e28** (2026-10-09, merged in [#526](https://github.com/TARPS-group/prob-pipe/pull/526)): that parent is shown under the result's label, so the parent of a posterior of `model` reads `model`.
   Before it, the parent read `posterior`, which is the default label an inference method gives its result.

Design V.9 states the combined rule (`design/05-functions.md:295`):

> A route's result that carries a record of its own [...] is a parent too, under the expression the result boundary gives the result, so the parent of a posterior of `model` reads `model` and keeps the method's record.

---

## Why a label repeats

The route's parent always carries the result's own label.
A result often takes its label from one of the call's inputs (design II.4):
- **an inference posterior:** keeps the label of the model it conditions, so `posterior.label == "model"`;
- **a pushforward:** takes the label of the map, so `evaluate(square, mu_prior).label == "square"`.

When a result is labeled after an input, the repr prints that label twice: once for the input, and once for the route's result.
The route's result is the call's result one step earlier, so the repr lists the result's label among its own parents.

---

## Examples

### Tutorial 2: a posterior from NUTS

The model is the Ricker model of `docs/tutorials/02_fitting.ipynb`.

```python
model = (likelihood * prior).with_label("model")
with workflow_run(seed=1):
    posterior = condition_on(model, {"y": counts})
posterior.provenance
# Provenance('workflow.condition_on', parents=[condition_on, model, model])
```

| Parent | Label | Type | The parent's own record |
|---|---|---|---|
| 0 | `condition_on` | `Operation` | none |
| 1 | `model` | `FactoredDistribution` | `Provenance('with_label', parents=[counts·prior])` |
| 2 | `model` | `EmpiricalDistribution` | `Provenance('blackjax_nuts', parents=[model])` |

Parent 1 is the joint the user passed.
Parent 2 is the posterior NUTS returned, before the return step gave it the call's record.
The two records hold different metadata:
- **the call's record:** `{'func': 'condition_on', 'route': 'inference_methods', 'exact': False, 'method': 'blackjax_nuts'}`;
- **the route's record:** `{'method': 'blackjax_nuts', 'num_results': 1000, 'num_warmup': 500, 'num_chains': 4}`.

The sampler settings are only in the route's record, so parent 2 is the one place the record names them.

### The overhaul tour: a pushforward by the sampling lift

```python
mu_prior = Normal("mu", 0.0, 5.0)

@function
def square(x):
    return x**2

with workflow_run(seed=0):
    pushforward = evaluate(square, mu_prior)
pushforward.provenance
# Provenance('workflow.evaluate', parents=[evaluate, square, Normal, square])
```

| Parent | Label | Type | The parent's own record |
|---|---|---|---|
| 0 | `evaluate` | `Operation` | none |
| 1 | `square` | `Function` | none |
| 2 | `Normal` | `Normal` | none |
| 3 | `square` | `EmpiricalDistribution` | `Provenance('broadcast', parents=[square, Normal])` |

Here the repeated label names a function and a distribution.
Parent 1 is the map the user passed, and parent 3 is the pushforward law the sampling lift returned.
The route's record holds `'n_samples': 256` and the dispatch mode, and the call's record holds neither.

This case predates 28735e28.
At `28735e28^` the repr is the same, because the sampling lift labels its result `square` on its own.

### A mean by component: the result among its own parents

```python
prior = Normal("mu", 2.0, 1.0)
likelihood = conditional_distribution(
    lambda mu: Poisson("y", jnp.exp(mu)), label="lik", given_spec=prior.event_spec.components
)
model = (likelihood * prior).with_label("model")
with workflow_run(seed=0):
    m = mean(model)
m.provenance
# Provenance('workflow.mean', parents=[mean, model, 𝔼[(y, mu) ~ model]])
```

`mean` takes its `by_component` route, which computes each component by its own route.
The result's label is `𝔼[(y, mu) ~ model]`, and no input has that label, so no label repeats.
Parent 2 still carries the result's label, so the repr lists the result among its own parents.
Its record is `Provenance('mean', parents=[model])`, and its metadata holds each component's route:

```python
{'route': 'by_component',
 'blocks': [{'components': ['y'], 'route': 'monte_carlo', 'exact': False},
            {'components': ['mu'], 'route': 'closed_form', 'exact': True}]}
```

The call's record holds `{'func': 'mean', 'route': 'by_component', 'exact': False}`, so the blocks are only in the route's record.

An exact `condition_on` has the same structure.
Its route parent is the slice's law, whose record holds `{'stage': 'exact', 'route': 'slice'}`.

### What the examples share

In each example the route's record has the call's inputs as its parents.
The extra parent therefore adds no ancestor to the lineage.
It adds three things:
1. **the route's operation:** `blackjax_nuts`, `broadcast`, `mean`, or the exact `condition_on` stage;
2. **the route's metadata:** the sampler settings, the sample count, the route and exactness of each component, or the stage;
3. **a node in the graph:** `provenance_ancestors` and `provenance_dag` include the route's result as a separate ancestor.

Thirteen tests read the route's parent, and they fail when the call does not record it:
- **`tests/core/test_provenance.py`:** three tests that list the parents of an exact `condition_on`;
- **`tests/operations/test_condition.py`:** three tests of the posterior's record and of the exact stage's record;
- **`tests/operations/test_moments.py`:** seven tests that read the blocks of a mean by component.

---

## Alternatives

Each alternative is shown on the tutorial 2 example.

### 1. Show the route's parent apart in the repr

`ParentInfo` gains a field that marks a parent as the route's result, and the repr prints that parent by its operation, after the inputs:

```
Provenance('workflow.condition_on', parents=[condition_on, model], route_record=blackjax_nuts)
```

- **Data:** unchanged, apart from the new marker. `provenance.parents` still holds three entries.
- **Keeps:** everything the record holds now.
- **Leaves:** the repeated label in `provenance.parents`, in `provenance_ancestors`, and in `provenance_dag`, where the two `model` nodes differ only by type. Design II.4 and V.9 still disagree on what a parent is.
- **Changes:** `ParentInfo`, `Provenance.__repr__`, serialization, and the repr text of design II.4.

### 2. Move the route's record to a field of its own

`Provenance` gains a field, such as `route_record: Provenance | None`, that holds the route's record.
`parents` holds only the tracked inputs, as design II.4 states.

```python
posterior.provenance
# Provenance('workflow.condition_on', parents=[condition_on, model], route_record=blackjax_nuts)
posterior.provenance.route_record
# Provenance('blackjax_nuts', parents=[model])
posterior.provenance.route_record.metadata
# {'method': 'blackjax_nuts', 'num_results': 1000, 'num_warmup': 500, 'num_chains': 4}
```

- **Data:** the route's record moves from `parents` to the new field.
- **Keeps:** the route's whole record, with its operation, metadata, and parents. A mean by component keeps its blocks at `m.provenance.route_record.metadata["blocks"]`.
- **Resolves:** the repeated label, and the disagreement between design II.4 and V.9. The route's record needs no label, so the relabeled copy that `_recorded_route_result` builds is no longer needed.
- **Changes:**
  - **the record:** `Provenance`, `Provenance.create`, and serialization;
  - **the call:** the route-record code in `probpipe/functions/_function.py`;
  - **the readers:** `provenance_ancestors`, `provenance_dag`, and the thirteen tests that read the route's parent;
  - **the design:** II.4 and V.9.

### 3. Fold the route's metadata into the call's record

The call's record keeps no extra parent, and its metadata gains the route's operation and metadata under one key:

```python
posterior.provenance
# Provenance('workflow.condition_on', parents=[condition_on, model])
posterior.provenance.metadata["route_record"]
# {'operation': 'blackjax_nuts', 'method': 'blackjax_nuts', 'num_results': 1000, 'num_warmup': 500, 'num_chains': 4}
```

- **Data:** one record per call.
- **Keeps:** the route's operation and metadata.
- **Drops:** the route's own parents, controls, diagnostics, and fingerprint. These are empty or repeat the call's in the examples above, and a future route could fill them.
- **Changes:** the route-record code in `probpipe/functions/_function.py`, the same thirteen tests, and design V.9.

### 4. Show the route's parent under its own label

This reverts 28735e28, so the tutorial 2 repr reads `parents=[condition_on, model, posterior]`.

- **Resolves:** the repeated label in the `condition_on` example.
- **Leaves:** the repeated label in the `evaluate` example, since the sampling lift labels its own result `square`.
- **Brings back:** the parent label `posterior`, which the user did not choose and which 28735e28 removed.

### 5. Keep the rule and document it

Tutorial 2 and the overhaul tour explain the repeated parent and print `posterior.provenance.parents[2].provenance`.
The code is unchanged, and the repr still reads as a bug to a reader who has not seen the explanation.

---

## Recommendation

Alternative 2 is the recommendation.
It makes `parents` mean what design II.4 and V.3 say, which is the tracked inputs of the call.
It keeps every piece of information that 04fc33a0 added to the record.
It removes the label question that 28735e28 answered, because the route's record no longer appears as a parent.

Alternative 1 is the smaller change if the team prefers to keep `provenance.parents` as it is.

---

## Questions for the reviewers

1. **What is a parent?** Should `Provenance.parents` hold only the call's tracked inputs, as design II.4 states, or also the route's result, as design V.9 states?
2. **Which alternative?** Is alternative 2 the right change, or does one of the others fit better?
3. **The name of the field.** The call's metadata already uses the key `route` for the selected route, such as `inference_methods` or `by_component`. Is `route_record` a clear name for the new field, or would another name read better?
4. **The graph.** With alternative 2, should `provenance_dag` draw the route's record as a node of its own, or as a note on the result's node?
