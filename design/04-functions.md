# Part IV — Functions

A **`Function`** (III.2) wraps an ordinary Python callable. This part is its **engine**: the call semantics beyond plain evaluation, installed on the base at import, which lift the callable into ProbPipe's world of distributions and values. The user writes a plain function over its "natural" values, and wrapping it makes that callable (i) **lift** automatically over distribution- and batch-valued arguments and (ii) **act** as a tracked node in a computation graph, so its result carries provenance. The operations of Part V are themselves `Function`s, which is why this part comes first: `sample`, `log_prob`, and `condition_on` inherit the lifting, tracking, dispatch, and orchestration defined here. `Function`s compose into a *workflow*.

## IV.0 — Overview: what a `Function` adds

Wrapping a callable `f` layers four things on top of calling `f`, each defined in a section below:

| §     | Concern                  | What it adds to `f`                                                                                                                                                                                     |
| ----- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IV.1  | the wrapper              | `f` becomes a tracked node in a computation graph, and a plain call returns a `Tracked` result with provenance.                                                                                |
| IV.2  | lifting                  | a distribution passed where a value is expected is sampled and `f` applied per draw, a batch is swept, and correlated arguments co-sample. |
| IV.3  | controls vs. arguments   | ProbPipe controls (sample count, seed, …) are kept in a namespace separate from that of the arguments to `f`.                                                                                            |
| IV.4  | dispatch & orchestration | *how* the per-draw calls run computationally and *whether* they are traced for lineage.                                                                                                                 |

Each concern has one home, and only two things are open for extension — the engine slot and the `evaluate` rule registry:

| Piece | Lives in | Extension status |
| --- | --- | --- |
| identity (`name`, `provenance`) and the templates | the base (III.2) | fixed at construction |
| plain evaluation | the base (III.2) | the call path before the engine installs |
| the engine itself | this part, installed into the base's call path | one slot, filled once at import; cataloged like the converter and bijector factories |
| lifting trigger, planning and co-sampling groups, the lift and the sweep, key splitting, result wrapping, provenance | the engine (IV.1–IV.2) | internal machinery; nothing to register |
| the controls store and `with_options` | the base (III.2) | ride the object; the engine gives them meaning (IV.3) |
| the `@function` decorator and the control semantics | the engine (IV.3–IV.4) | controls, a namespace apart from the arguments |
| execution dispatch and orchestration | the engine (IV.4) | modes selected by a control; neither is a registry |
| invertibility, the Jacobian determinant, differentiation | capabilities the object claims (III.14, III.2) | claims declared at construction, never registrations; `SupportsInverse` and `SupportsLogDetJacobian` feed `evaluate`'s change-of-variables rule and the `inverse` / `log_det_jacobian` operations (V.7), `SupportsDifferentiation` whatever requires gradients (`D6`) |
| evaluation rules, exact and approximate | the rule registry, defined beside the engine (V.6) | registered upward, ranked by fidelity above the sampling lift, the rule at the generic pair; consulted by the engine at each single-distribution application, `evaluate` the operation form |

## IV.1 — `Function`

### Contract

A `Function` wraps exactly one callable and presents it as a node in a computation graph. It is created with the `@function` decorator:

```python
@function
def predict(theta, x): ...                    # an ordinary callable over concrete values

@function(seed=42, dispatch="jax")    # optional construction-time controls
def predict(theta, x): ...
```

Construction may also declare claims. The decorator's `differentiable` argument declares which inputs gradients propagate through: a non-empty numeric event template naming exactly those values — array-native, with no gradient-breaking operations. Omitting the argument makes no claim, and there is no shorthand for claiming every input, so a claim never widens as the function changes. The constructed `Function` then carries `SupportsDifferentiation` with that template (`D6 – Differentiability as a capability`), read through `is_differentiable` wherever gradients are required. Like the templates, the declaration is fixed at construction: a claim, not a control.

Calling it runs the wrapped function and returns a `Tracked` result: the output is wrapped as a value or distribution carrying `Provenance` that records this `Function` and its tracked inputs. A call whose arguments are all ordinary values is one invocation of `f` followed by that wrap. A distribution- or batch-valued argument triggers lifting instead. The engine is installed on the III.2 base at import; imports still point strictly downward.

A `Function` is a node in a directed graph: arguments that are themselves tracked terms become graph **dependencies**, and the rest are plain **inputs**. That graph is what provenance and orchestration traverse. A `Function` may also belong to a *module* that supplies some of its inputs and dependencies, but the unit of execution is always the single wrapped function.

**The engine contract.** The engine is one callable installed into the base's call path (III.2), once, at import. On a call it receives the `Function` and its arguments, reads the controls the object carries (III.2, IV.3), runs the machinery of IV.2 and IV.4, and returns a `Tracked` result. Two obligations align it with the operation model of Part V. On concrete values it agrees with plain evaluation, adding only the wrap and the provenance, so installing the engine never changes an answer the base gives. And it consults the evaluation-rule registry at each single-distribution and each batched application (IV.2, V.6), the sampling lift and the elementwise sweep registered at the generic pairs as its floors, and the selected rule records its name and fidelity in the result's provenance from either entry point, so the direct call and `evaluate` take the same route and agree in value and in record; `evaluate` is the operation form, adding `method=` selection and the operation's provenance record. The registry is defined beside the engine and populated upward by the families, so this layer stays below the operations. The engine is itself no operation — the operations are `Function`s built on it — so the operation model's capability dispatch and fallback structure apply one level up, with `evaluate` as the call's operation form.

### Rationale

This makes `C1 – Uniform interface to distributions and values` and `C4 – Function lifting via pushforward` operational: a user writes mathematics as an ordinary, testable function, and ProbPipe lifts it to act on distributions and values without the function being rewritten. Making every `Function` a graph node delivers `C6 – Traceable and reproducible workflows`: each result records how it was produced, and a whole workflow can later be traced or re-run. Because the wrapper changes only invocation and tracking, the operations can be *defined* as `Function`s and inherit all of it.

## IV.2 — Lifting over distributions and batches

### Contract

A `Function` compares each argument against the type its function expects, and lifts where they differ. A single-distribution application resolves through the evaluation-rule registry (V.6), where the sampling lift is the rule registered at the generic pair: a plain callable therefore lifts by **sampling**, and a typed map takes its registered rule. A batched application resolves through the same registry, with the elementwise sweep the rule at its generic pair, so a fused batched implementation registers like any other rule. Grouped, multi-distribution lifts always take the sampling path, which is what co-sampling requires.

**The trigger.** A parameter that is unannotated, or annotated with a value type, expects a value, so a distribution passed in that position is lifted. A parameter annotated `Distribution`, `Distribution[...]`, or a distribution capability protocol of III.7 declares that the function consumes the distribution itself, which then passes through unlifted. The function capabilities of III.2 and III.14 annotate `Function`-valued parameters, which are values, so the value rule above governs them. Per draw, the function receives what `sample` returns, the draw's `Record`.

- **A distribution where a value is expected → broadcast.** The distribution is sampled `n` times and the function is applied to each draw. The result is an **empirical distribution** over the outputs, which approximates the pushforward of the input through `f`.
- **A batch where one element is expected → sweep.** The function is mapped over the batch's elements, returning a batch of outputs. The registry selects how: the elementwise map at the generic pair, or a fused batched rule where one is registered.
- **Both at once → a nested sweep of broadcasts.** The function is mapped over the batch's elements, with a broadcast performed within each.
- **Neither → a plain call.** A `ConditionalDistribution`-valued argument is an error rather than a plain call, since a kernel has no marginal law to lift over.

**Grouping and correlation.** The lifted arguments are grouped by **root ancestor**, transitively: sibling views of one parent, the same distribution passed twice, and a parent passed alongside its own view all fall in one group. Each group contributes one joint draw per repetition, so dependence between its members flows through `f` rather than being broken by independent sampling. A view lifts by sampling its parent, so its parent must itself sample. Groups with no common ancestor draw independently: the lift samples the **product law**, and, as a corollary, detached marginals of one joint lift independently while its views co-sample. For example, `f(d, d["x"])` forms one group, and each repetition evaluates `f` on a joint draw and its own projection, while `f(d1, d2)` for unrelated `d1` and `d2` samples the product of their laws. The number of lifted arguments changes only the grouping, never the mechanism or the return type.

**The output wrapping.** The result's fields come from the function's return value. A **mapping** return becomes fields keyed by the mapping's keys, since mappings are never leaves. Any other return becomes one field keyed by the `Function`'s name, its spec inferred, an `ArraySpec` for an array and an `OpaqueSpec` otherwise, tuples included. An opaque output samples downstream but carries none of the numeric machinery, so a function whose output deserves structure returns a mapping or declares it. The optional `output_template=` on the decorator declares the output structure the producer knows and inference cannot recover, such as a constrained support or a symbolic dimension shared across fields. The declared template is bound per call by unification, which also validates the function's output against it on every call.

```python
@function(output_template=EventTemplate(rate=ArraySpec(("obs",), float32, positive)))
def rate(x):
    return jnp.exp(x)
# inference alone would read the support as real; the declaration carries support=positive
# and binds "obs" to the actual output length on each call
```

**Including the inputs.** By default the result holds only the outputs. With `include_inputs=True` it is instead the **joint** empirical distribution over the sampled inputs and the outputs: one top-level field per lifted parameter, named by the parameter, whose subtree is that argument's `event_template` (a single-field argument still nests, so the layout never depends on the argument's field count), plus the output fields. A plain-value argument contributes no field, since it is recorded in provenance rather than sampled. Sibling uniqueness applies across the lifted-parameter names and the output keys, and a collision, such as a function named after one of its own lifted parameters, is an error at result construction. Grouping affects only how the draws are taken, never this layout.

```python
# posterior.event_template == EventTemplate(beta=ArraySpec(shape=(5,), dtype=float32, support=real))

@function(include_inputs=True, n_broadcast_samples=200, seed=7)
def predict(theta, x):
    return x @ theta["beta"]      # theta arrives as the Record a posterior draw is

result = predict(theta=posterior, x=X_new)   # X_new: a plain (20, 5) array, not lifted

# result: empirical over 200 atoms, each one joint draw (theta_s, predict(theta_s, X_new)):
#   EventTemplate(
#       theta=EventTemplate(beta=ArraySpec(shape=(5,), dtype=float32, support=real)),
#       predict=ArraySpec(shape=(20,), dtype=float32, support=real),
#   )
# so the fields are theta/beta and predict; X_new lands in provenance, not in the law
```

Each atom is one joint draw, so the result couples every sampled input with its own output, which is what a predictive check or a sensitivity analysis reads off it.

### Rationale

This is `C4 – Function lifting via pushforward` realized: replacing any argument of `f` with a distribution over that argument's type leaves `f` well-defined, and the result is the pushforward. Doing it by sampling keeps the contract fully general (`D2 – Generality first`): it works for any `f`, any number of lifted arguments, and any distribution that samples, with closed-form shortcuts reserved for the operations built on structured maps, and it leaves the user's function untouched. The annotation trigger makes the lifting boundary explicit in the one place the author already states intent, the signature. Co-sampling by root ancestor is what makes the lift *correct* rather than merely type-correct: it is the same correlation-preserving mechanism the field views rest on, so passing sibling views through a function transports their joint law. Declared output structure is `D5 – Explicit, carried structure` at the lift boundary: inference from a returned value is lossy, so the producer that knows the support or the dimension identities declares them, and they travel with the result.

### Open points

- *Default sample count.* How many draws a broadcast takes by default (a speed-versus-accuracy ceiling, with an explicit per-call override always available) is unsettled. The default should signal "rough estimate," not "tuned."
- *Single-value presentation.* Whether a single-field draw presents to the wrapped function as the bare value rather than the single-field `Record` follows the single-value-coercion question left open with `Record`.

## IV.3 — Controls vs. arguments

### Contract

A `Function` keeps two namespaces strictly apart:

- **The wrapped function's arguments.** Every positional and keyword argument of a call binds to the wrapped function.
- **ProbPipe controls.** The controls are the sample count (`n_broadcast_samples`), the PRNG `seed`, the `include_inputs` switch, and the dispatch and orchestration selectors. Each is set on the decorator (construction time) or, for a single call, through a `with_options` view (III.2), which covers every control; the controls ride the `Function`, and the engine reads them at call time. The `seed` deterministically derives the PRNG `Key` that any operation invoked inside the call consumes:

```python
predict.with_options(n_broadcast_samples=1000, seed=7)(theta=prior, x=x_obs)
# controls go to with_options; theta and x bind to the wrapped function
```

The `seed` fixes randomness deterministically and *structurally*. It is split along the computation graph, and again by index along each broadcast or batch axis, so every draw receives a distinct key as a pure function of its position. Perturbing an input therefore reuses the same keys, preserving common random numbers: comparisons across nearby inputs and reparameterization gradients stay low-variance rather than being swamped by independent sampling noise. A fresh estimate or an independent stream is obtained by changing the `seed`. The one cost is that the streams are tied to the program's structure, so reordering the computation reshuffles them.

### Rationale

A `Function` must wrap an *ordinary* function with no naming restrictions (`C5 – Naming for unambiguous meaning`): a user should never have to rename a `seed` parameter because the framework wanted that word. Holding the control plane in a separate namespace removes the collision entirely, while keeping the bare decorator and a single call site ergonomic.

## IV.4 — Dispatch and orchestration

### Contract

Two orthogonal computational concerns sit beneath a lifted call, both with defaults so a user need not touch them:

- **Dispatch — *how* the per-draw / per-element calls run.** `jax` vectorizes them (one `vmap`); `sequential` runs them one at a time; `thread` runs them on a thread pool; `auto` probes whether the call is array-traceable and picks `jax`, falling back to `sequential`. Under `jax`, a lifted call is traced end-to-end, and it differentiates end-to-end when the `Function` claims `SupportsDifferentiation`; dispatch never changes the result beyond floating-point effects of evaluation order. Because each unit's PRNG key is fixed by its index rather than taken from a shared counter, the result is identical across `jax`, `sequential`, and `thread`, and parallel execution contends for no mutable random state: nothing is locked, and no key is drawn twice.
- **Orchestration — *whether* the call is traced.** Off by default. A `Function` can instead run as a traced task or flow, recording the computation graph for lineage and scheduling. Tracing never changes the result.

### Rationale

Dispatch and orchestration are `C3 – Computational detail hidden by default, available on demand` in action: the algorithm that realizes a lifted call, and whether its graph is recorded, are computational concerns, handled automatically by default and exposed for users who need control. Keeping them orthogonal (how a call runs is independent of whether it is traced) lets the fast vectorized path and full lineage tracking compose rather than trade off.

### Open points

- *Non-array backends.* Lifting and dispatch are array-native, built for a differentiable array backend. First-class support for other tensor frameworks (e.g., a Torch model as the wrapped function, with conversion at the boundary) is not yet settled, though it should be feasible using, e.g., Keras.
