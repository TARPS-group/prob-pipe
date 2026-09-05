# Part IV — Functions

A **`Function`** (III.3) wraps an ordinary Python callable. This part describes its **engine**: the call semantics beyond plain evaluation, installed on the base at import, which lift the callable into ProbPipe's world of distributions and values. The user writes a plain function over its "natural" values, and wrapping it makes that callable (i) **lift** automatically over distribution- and batch-valued arguments and (ii) **act** as a tracked node in a computation graph, so its result carries provenance. The operations of Part V are themselves `Function`s, which is why this part comes first: `sample`, `log_prob`, and `condition_on` inherit the lifting, tracking, randomness, dispatch, and orchestration defined here. `Function`s compose into a *workflow*.

Wrapping a callable `f` as a `Function` adds five features, each defined in a section below:

| §     | Concern                  | What it adds to `f`                                                                                                                                                                                     |
| ----- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IV.1  | the wrapper              | `f` becomes a tracked node in a computation graph, and a plain call returns a `TrackedTerm` result with provenance.                                                                                |
| IV.2  | lifting                  | a distribution passed where a value is expected is sampled and `f` applied per draw, a batch is swept, and correlated arguments co-sample. |
| IV.3  | randomness               | every ProbPipe-caused draw takes a key derived structurally from a workflow scope, so results are reproducible, order-independent, and parallel-safe.                                                    |
| IV.4  | controls vs. arguments   | ProbPipe controls (sample count, dispatch, …) are kept in a namespace separate from that of the arguments to `f`.                                                                                            |
| IV.5  | dispatch & orchestration | *how* the per-draw calls run computationally and *whether* they are traced for lineage.                                                                                                                 |

## IV.1 — `Function`

### Contract

A `Function` wraps exactly one callable and presents it as a node in a computation graph. It is created with the `@function` decorator:

```python
@function
def predict(theta, x): ...                    # an ordinary callable over concrete values

@function(n_broadcast_samples=500, dispatch="jax")   # optional construction-time controls
def predict(theta, x): ...
```

Construction may also declare claims. The decorator's `differentiable` argument declares which inputs gradients propagate through: a non-empty `NumericSpec` covering exactly those values — array-native, with no gradient-breaking operations. Omitting the argument makes no claim, and there is no shorthand for claiming every input, so a claim never widens as the function changes. The constructed `Function` then carries `SupportsDifferentiation` with that schema, read through `is_differentiable` wherever gradients are required. Like the declared sides, the declaration is fixed at construction: a claim, not a control.

```python
@runtime_checkable
class SupportsDifferentiation(Protocol):
    @property
    def differentiable_template(self) -> NumericSpec: ...
    # exactly the values gradients propagate through: a sub-schema of the numeric
    # input slots (maps) or of the numeric event schema (distributions)

def is_differentiable(x: Any, values: NamedTree | None = None) -> bool: ...
# True when every value named in `values` lies in x's differentiable template;
# with no `values`, when the template covers every numeric value. False when x
# does not declare the capability.
```

The capability is cross-kind: the `differentiable_template` is a sub-schema of the numeric input slots for a map and of the numeric event schema for a distribution, so a linear operator claims its whole input schema and a distribution family claims the event values its sampling reparameterizes. The claim composes: a field view restricts its parent's schema to the viewed path, a joint assembles its factors' schemas under their field names, and a value is differentiable through a chain of steps exactly when every step claims it. An operation that needs gradients checks `is_differentiable` for the values it differentiates and names the first step that fails, before a backend trace runs; execution dispatch is a separate control, so `jax` vectorizes a call whether or not the object differentiates.

Calling it runs the wrapped function and returns a `TrackedTerm` result: the output is wrapped as a value or distribution carrying `Provenance` that records this `Function` and its tracked inputs. A call whose arguments are all ordinary values is one invocation of `f` followed by that wrap. A distribution- or batch-valued argument triggers lifting instead.

A `Function` is a node in a directed graph: arguments that are themselves tracked terms become graph **dependencies**, and the rest are plain **inputs**. That graph is what provenance and orchestration traverse.

**The engine contract.** The engine is one callable installed into the base's call path (III.3), once, at import. On a call it receives the `Function` and its arguments, reads the controls the object carries (III.3), runs lifting and dispatch as described below, and returns a `TrackedTerm` result — unless the call passes `raw=True`, which the engine reads at this same boundary: the wrap and identity step is skipped and the result returns detached (II.4), never re-wrapped by the enclosing call. Two obligations align it with the operation model of Part V. On concrete values it agrees with plain evaluation, adding only the wrap and the provenance, so installing the engine never changes a result the base gives. And it consults the evaluation-rule registry at each single-distribution and each batched application, so the direct call and `evaluate` take the same route (V.1). The engine is itself no operation — the operations are `Function`s built on it — so the operation model's route resolution applies one level up, with `evaluate` as the call's operation form.

### Rationale

This makes `C1 – Uniform interface to functions, distributions, and values` and `C4 – Function lifting` operational: a user writes mathematics as an ordinary, testable function, and ProbPipe lifts it to act on distributions and values without the function being rewritten. Making every `Function` a graph node delivers `C6 – Traceable and reproducible workflows`: each result records how it was produced, and a whole workflow can later be traced or re-run. Because the wrapper changes only invocation and tracking, the operations can be *defined* as `Function`s and inherit all of it. Differentiability as a declared claim is `D3 – Capability-based operations` applied to gradients: support is promised by the object and checked before a backend trace, never inferred from a value being numeric.

### Open points

- *Differentiability of sampling-based routes.* Whether a Monte Carlo fallback differentiates through its sampler's reparameterization is unsettled. So is the eventual `grad` operation the claims feed, with registered routes: a custom gradient method where an object supplies one, the automatic-differentiation route gated by the declared template, and finite differences as the fallback at approximate fidelity. Both are left to a dedicated pass.

## IV.2 — Lifting over distributions and batches

### Contract

A `Function` compares each argument against the type its function expects, and lifts where they differ. A single-distribution application resolves through the evaluation-rule registry (V.1), so a plain callable lifts by **sampling** and a typed map takes its registered rule; a batched application resolves through the same registry, the elementwise sweep its **floor**, the always-feasible rule that ranks last. Grouped, multi-distribution lifts always take the sampling path, which is what co-sampling requires.

**The trigger.** A raw argument is normalized to its kind on entry — a backend distribution entering through its registered converter — before types are read, so a backend distribution lifts exactly as its converted form does. A parameter that is unannotated, or annotated with a value type, expects a value, so a distribution passed in that position is lifted. A parameter annotated `Distribution`, `Distribution[...]`, or a distribution capability protocol of III.8 declares that the function consumes the distribution itself, which then passes through unlifted. The function capabilities of III.3 and III.15 annotate `Function`-valued parameters, which are values, so the value rule above governs them. Per draw, the function receives what `sample` returns, the draw, at the kind the law's event declaration names.

- **A distribution where a value is expected → broadcast.** The distribution is sampled `n` times and the function is applied to each draw. The result is an **empirical distribution** over the outputs, which approximates the pushforward of the input through `f`.
- **A batch where one element is expected → sweep.** The function is mapped over the batch's elements, returning a batch of outputs. The registry selects how (V.1).
- **Both at once → a nested sweep of broadcasts.** The function is mapped over the batch's elements, with a broadcast performed within each.
- **Neither → a plain call.** A `ConditionalDistribution`-valued argument is an error rather than a plain call, since a kernel has no marginal law to lift over.

**Grouping and correlation.** The lifted arguments are grouped by **root ancestor**, transitively: sibling views of one parent, the same distribution passed twice, and a parent passed alongside its own view all fall in one group. Each group contributes one joint draw per repetition, so dependence between its members flows through `f` rather than being broken by independent sampling. A view lifts by sampling its parent, so its parent must itself sample. Groups with no common ancestor draw independently: the lift samples the **product law**, and, as a corollary, detached marginals of one joint lift independently while its views co-sample. For example, `f(d, d["x"])` forms one group, and each repetition evaluates `f` on a joint draw and its own projection, while `f(d1, d2)` for unrelated `d1` and `d2` samples the product of their laws. The number of lifted arguments changes only the grouping, never the mechanism or the return type.

**The output wrapping.** The output is wrapped by the **kind-directed** boundary, the same boundary every operation's result crosses: a raw value is wrapped into the ProbPipe kind it is, so wrapping depends on what the return already is. A **tracked term** keeps its kind, every kind alike — a `Function`, `Distribution`, `Record`, or `NumericArray` the callable produced is the result, not a field buried inside a fresh `Record` — under the call's fresh, derived identity like any output. A **raw callable** becomes a `Function`. A raw **mapping** becomes a `Record` (III.5). A raw **array** becomes a `NumericArray`, and any other raw return an `Opaque`, tuples included — the rules are ordered rather than disjoint, a callable also being a non-mapping value, so `Opaque` is the fallback the named kinds fall through to. An opaque output samples downstream but carries none of the numeric interface, so a function whose output deserves structure returns a mapping or declares it. The optional `output_spec=` on the decorator declares what the producer knows and inference cannot recover: an `OutputSpec` (II.2), or a bare term spec with the name defaulted to the function's own, captured once. Its spec declares the result's kind — a `FunctionSpec` for a fitted mapping, a `ConditionalDistributionSpec` for a predictive kernel — and a `RecordSpec` declares field structure such as a constrained support or a shared symbolic dimension. The declaration is bound per call by unification, which validates the output, and a tracked record supplied at a record-shaped position is stored with its identity kept (III.5). For a non-record output, the declaration's name is the result's field name wherever one is required — the joint layout below and composition's produced slots (III.12) — so no field name ever derives from the function's renamable label.

```python
@function(output_spec=RecordSpec(rate=NumericArraySpec(("obs",), float32, positive)))
def rate(x):
    return jnp.exp(x)
# inference alone would read the support as real; the declaration carries support=positive
# and binds "obs" to the actual output length on each call
```

**Including the inputs.** By default the result holds only the outputs. With `include_inputs=True` it is instead the **joint** empirical distribution over the sampled inputs and the outputs: one top-level field per lifted parameter, named by the parameter, whose subtree is that argument's event schema (a single-field argument still nests, so the layout never depends on the argument's field count), plus the output fields. A term-drawing law has no fields to include, so a parameter drawn from one contributes its draw as a single term-valued field. A plain-value argument contributes no field, since it is recorded in provenance rather than sampled. Sibling uniqueness applies across the lifted-parameter names and the output names, and a collision, such as an output declared under the name of one of its own lifted parameters, is an error — at construction when the declaration makes it knowable, and at result construction otherwise. Grouping affects only how the draws are taken, never this layout.

```python
# posterior's event schema == RecordSpec(beta=NumericArraySpec(shape=(5,), dtype=float32, support=real))

@function(include_inputs=True, n_broadcast_samples=200)
def predict(theta, x):
    return x @ theta["beta"]      # theta arrives as the Record a posterior draw is

result = predict(theta=posterior, x=X_new)   # X_new: a plain (20, 5) array, not lifted

# result: empirical over 200 atoms, each one joint draw (theta_s, predict(theta_s, X_new)):
#   RecordSpec(
#       theta=RecordSpec(beta=NumericArraySpec(shape=(5,), dtype=float32, support=real)),
#       predict=NumericArraySpec(shape=(20,), dtype=float32, support=real),
#   )
# so the fields are theta/beta and predict; X_new is recorded in provenance, not in the law
```

Each atom is one joint draw, so the result couples every sampled input with its own output, which is what a predictive check or a sensitivity analysis reads off it.

### Rationale

This is `C4 – Function lifting` realized in both of its cases: replacing an argument of `f` with a distribution over that argument's type leaves `f` well-defined and returns the pushforward, and replacing one with a batch over that type leaves it equally well-defined and returns the broadcast — one substitution rule, differing only in whether the multiplicity is a law or a collection. Realizing the first by sampling and the second by the sweep keeps the contract fully general (`D2 – Generality first`): it works for any `f`, any number of lifted arguments, any distribution that samples, and any batch, with closed-form rules and fused batched rules alike registering above those floors, and it leaves the user's function untouched. The annotation trigger makes the lifting boundary explicit in the one place the author already states intent, the signature. Co-sampling by root ancestor is what makes the lift *correct* rather than merely type-correct: it is the same correlation-preserving mechanism the field views rest on, so passing sibling views through a function transports their joint law. Declared output structure is `D5 – Explicit, carried structure` at the lift boundary: inference from a returned value is lossy, so the producer that knows the support or the dimension identities declares them, and they travel with the result.

### Open points

- *Default sample count.* How many draws a broadcast takes by default (a speed-versus-accuracy ceiling, with an explicit per-call override always available) is unsettled. The default should signal "rough estimate," not "tuned."

## IV.3 — Randomness: workflow scopes and structural keys

### Contract

Randomness is scoped, not carried: no `Function` and no operation holds a seed. A **workflow scope** owns every ProbPipe-caused draw inside it, and each such **workflow-owned random event** receives a key derived structurally from the scope's root seed — never from a counter, ambient state, or the order of execution.

```python
with workflow_run(seed=42):              # a seeded scope: same seed, same structure ⇒ same draws
    result = predict(theta=posterior, x=X_new)

with replay_run(result.provenance):      # re-runs one recorded call on its recorded draws
    replayed = predict(theta=posterior, x=X_new)
```

**Scopes.** `workflow_run(seed)` opens a scope whose root the seed fixes; entering it again with the same stochastic structure reproduces its workflow-owned events. `workflow_run()` with no seed opens an anonymous scope rooted in fresh entropy, and a bare omitted-key call outside any scope gets its own equivalent ephemeral scope, so unscoped code is fresh rather than repeatable. Scopes nest, a nested scope extending the enclosing structure rather than restarting it, and a scope fixes the run's provenance mode (II.4) at entry. A scope is thread- and task-local: work crosses into another thread or task only through the engine's managed work items, and entering a copied scope unmanaged raises rather than silently forking the stream.

**The key rule.** An operation called with an explicit PRNG `key` is **caller-owned**: the key is consumed as given and the scope's stream is untouched. An operation with the key omitted is **workflow-owned**: its key is derived from the scope. The two never interact, so code that threads keys explicitly composes with scoped code without perturbing either stream.

**Structural event identity.** A workflow-owned key is a pure function of the scope's root seed and the event's identity, three structural coordinates: the **occurrence path** — where the invocation sits in the workflow, extended by nesting and by managed work items, with repeated or recursive invocations of the same call distinguished by a deterministic logical ordinal; the **stochastic source** — which co-sampling group of IV.2 is drawing; and the **logical unit** — which broadcast repetition or sweep cell consumes the draw. Identity follows the workflow's logical structure, never the physical schedule: ordinals are fixed by program order rather than by thread completion or scheduling order, no key is drawn twice, and the same call produces the same draws under any dispatch mode, thread count, or orchestration.

**Consequences.** Because keys attach to structure, perturbing an input reuses the same keys, preserving common random numbers: comparisons across nearby inputs and reparameterization gradients stay low-variance rather than being swamped by independent sampling noise. A fresh estimate or an independent stream is obtained by changing the seed. The one cost is that the streams are tied to the program's structure, so restructuring the computation reshuffles them.

**The derivation contract.** The derivation from seed and identity to key is a fixed, versioned contract: keyed and domain-separated, so distinct events cannot collide by construction, and fixed by version in every execution route and every replay record, so a reproduced run either reproduces under the same contract or refuses. The backend key type is produced through an adapter whose behavior is certified before first use, and a deviation raises rather than yielding silently different draws. The version string, not the hash primitive, is what the design fixes: the primitive may be anything that honors the contract.

**Replay.** `replay_run(provenance)` re-executes exactly one recorded `Function` call with its recorded workflow-owned events. Replay validates rather than approximates: the recorded plan, the execution capability, the derivation version, and the events actually observed must all match, and an incompatibility raises. Caller-owned keys need no replay support, since they reproduce from their values.

**Caching.** The same structural identity that makes a run reproducible makes it cacheable: a call whose function, operand fingerprints, resolved controls, and random-event identities all match a recorded call must produce the same result, so the result may be served from a cache instead of recomputed. Caching is therefore a workflow-layer option with no semantic effect — off by default, never changing a result, and never a reason for an operation to behave differently. What it can serve is bounded by what provenance records: under the lightweight mode the identity tier is structural, so a cache keyed on it assumes unchanged inputs, while the full mode's content-verifiable fingerprints let a hit be checked rather than assumed.

### Rationale

Scoped, structural randomness is `C6 – Traceable and reproducible workflows` made mechanical: one seed reproduces a workflow, one provenance record replays a call, and identity-derived keys make the result independent of execution order, so reproducibility survives parallelism and orchestration rather than trading off against them. Key management is `C3 – Computational detail hidden by default, available on demand` applied to randomness: with the key omitted, draws are fresh, reproducible on demand, and never silently repeated, while an explicit key hands the caller full manual control. Versioning the derivation makes reproduction exact: a run reproduces under the contract that produced it or refuses, never drifting silently.

## IV.4 — Controls vs. arguments

### Contract

A `Function` keeps two namespaces strictly apart:

- **The wrapped function's arguments.** Every positional and keyword argument of a call binds to the wrapped function.
- **ProbPipe controls.** The controls are the sample count (`n_broadcast_samples`), the `include_inputs` switch, the `raw` opt-out the engine reads at the wrap boundary (IV.1), and the dispatch and orchestration selectors. Each is set on the decorator (construction time) or, for a single call, through a `with_options` view (III.3), which covers every control; the controls are stored on the `Function`, and the engine reads them at call time. Randomness is deliberately not a control: seeding belongs to the workflow scope (IV.3), and no `Function` carries a seed:

```python
predict.with_options(n_broadcast_samples=1000)(theta=prior, x=x_obs)
# controls go to with_options; theta and x bind to the wrapped function
```

### Rationale

A `Function` must wrap an *ordinary* function with no naming restrictions (`C5 – Naming for unambiguous meaning`): a user should never have to rename a `seed` parameter because the framework wanted that word. Holding the control plane in a separate namespace removes the collision entirely — and with seeding scoped rather than carried (IV.3), ProbPipe claims no `seed` argument at all, so the word stays the user's — while keeping the bare decorator and a single call site convenient.

## IV.5 — Dispatch and orchestration

### Contract

Two orthogonal computational concerns sit beneath a lifted call, both with defaults so a user need not touch them:

- **Dispatch — *how* the per-draw / per-element calls run.** `jax` vectorizes them (one `vmap`); `sequential` runs them one at a time; `thread` runs them on a thread pool; `auto` probes whether the call is array-traceable and picks `jax`, falling back to `sequential`. Under `jax`, a lifted call is traced end-to-end, and it differentiates end-to-end when the `Function` claims `SupportsDifferentiation`; dispatch never changes the result beyond floating-point effects of evaluation order. Because keys attach to structure (IV.3), the result is identical across `jax`, `sequential`, and `thread`, and parallel execution contends for no mutable random state. Each route's versioned capability contract (IV.3) is checked before sampling begins, so an unsupported route is refused up front rather than approximated.
- **Orchestration — *whether* the call is traced.** Off by default. A `Function` can instead run as a traced task or flow, recording the computation graph for lineage and scheduling. Tracing never changes the result. Work that crosses a thread, task, or orchestrated-flow boundary travels as a **managed work item** extending the occurrence path (IV.3), so orchestrated and distributed runs draw from the same structural stream as local ones.

### Rationale

Dispatch and orchestration are `C3 – Computational detail hidden by default, available on demand` in action: the algorithm that realizes a lifted call, and whether its graph is recorded, are computational concerns, handled automatically by default and exposed for users who need control. Keeping them orthogonal (how a call runs is independent of whether it is traced) lets the fast vectorized path and full lineage tracking compose rather than trade off.

### Open points

- *Non-array backends.* Lifting and dispatch are array-native, built for a differentiable array backend. First-class support for other tensor frameworks (e.g., a Torch model as the wrapped function, with conversion at the boundary) is not yet settled, though it should be feasible using, e.g., Keras.
