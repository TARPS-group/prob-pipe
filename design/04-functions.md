# Part IV — Functions

A **`Function`** (III.3) wraps an ordinary Python callable. This part describes its **engine**: the call semantics beyond plain evaluation, installed on the base at import, which lift the callable ProbPipe-native. The user writes a plain function over its "natural" values, and wrapping it makes that callable (i) **lift** automatically over distribution- and batch-valued arguments and (ii) **act** as a tracked node in a computation graph, so its result carries provenance. `Function`s therefore compose into a workflow.

Wrapping a callable `f` as a `Function` runs one **stack** of steps on every call. IV.1 states the stack and what it reads from the `Function`; the sections that follow describe each step in the order it runs, with what it requires and what happens when that fails; the last states the construction-time claim the operations read.

| §     | Step or concern   | What it covers                                                                                                  |
| ----- | ----------------- | --------------------------------------------------------------------------------------------------------------- |
| IV.1  | the stack         | the `Function`, its engine, the three declarations the engine reads, and the steps in order                     |
| IV.2  | controls          | how the framework's controls are set and resolved apart from the arguments of `f` (step 1)                      |
| IV.3  | binding           | how arguments bind to the signature and become dependencies or inputs (step 2)                                  |
| IV.4  | normalization     | wrapping raw hosts into their kinds, converting distributions, and admitting each argument (step 3)             |
| IV.5  | lifting           | which arguments are lifted, how they group, and the regime that results (step 4)                                |
| IV.6  | planning          | validating the declarations and computing the result declaration before anything runs (step 5)                 |
| IV.7  | resolution        | the evaluation-rule registry and the selection of the route that realizes the call (step 6)             |
| IV.8  | randomness        | workflow scopes and the structural keys the executed points consume                                             |
| IV.9  | execution         | applying the function under a dispatch mode, optionally traced (step 7)                                             |
| IV.10 | return            | assembling, validating, wrapping, and identifying the result, or detaching it under `raw=True` (step 8)          |
| IV.11 | differentiability | the construction-time claim of which inputs gradients propagate through                                          |

## IV.1 — The `Function` and its engine

### Contract

A `Function` wraps exactly one callable and presents it as a node in a computation graph. It is created with the `@function` decorator:

```python
@function
def predict(theta, x): ...                    # an ordinary callable over concrete values

@function(n_broadcast_samples=500, dispatch="jax")   # optional construction-time controls
def predict(theta, x): ...
```

**The engine.** The engine is one callable installed into the base's call path (III.3), once, at import. On concrete values it agrees with plain evaluation, adding only the wrap and the provenance. Every call runs the stack below in order, and a failure ends the call at its step. The stack reads three declarations from the `Function` it runs: what each parameter **accepts**, the **result declaration**, and the **realization**. For a `@function` the three are as follows. A parameter accepts its declared input spec, or any value where it is unannotated. The result declaration is the `output_spec` given at construction, bound per call by unification, or is read from the return when none was given. The realization is the body, or, for a lifted application, the route the evaluation registry selects (IV.7).

1. **Configure** (IV.2). The effective controls are resolved: the framework's defaults, the decorator's values, and a `with_options` view, in that order.
2. **Bind** (IV.3). The arguments bind to the wrapped function's signature, tracked arguments as dependencies and the rest as inputs.
3. **Normalize** (IV.4). Each argument is wrapped into its kind, a distribution is converted where its parameter names another class, and the result is admitted against what the parameter accepts.
4. **Classify the lift** (IV.5). The lifted arguments are found and grouped, resulting in one of the following lifting regimes: a plain call, a broadcast, a sweep, or a nested sweep.
5. **Plan** (IV.6). The declarations are validated and the result declaration is computed from information that is static under compilation.
6. **Resolve** (IV.7). The route that realizes the call is selected among the feasible candidates.
7. **Execute** (IV.9). The selected route runs over the points under the dispatch mode, each point with its key (IV.8).
8. **Return** (IV.10). The points are assembled, validated against the planned declaration, wrapped, and given identity, or detached under `raw=True`.

**Checking feasibility.** The `f.check(...)` method runs steps 1 to 6 without executing and reports which routes are feasible, what each infeasible one is missing, and which would be selected.

### Rationale

This makes `C1 – Uniform interface to functions, distributions, and values` and `C4 – Function lifting` operational: a user writes mathematics as an ordinary, testable function, and ProbPipe lifts it to act on distributions and values without the function being rewritten. Making every `Function` a graph node delivers `C6 – Traceable and reproducible workflows`: each result records how it was produced, and a whole workflow can later be traced or re-run. Because the wrapper changes only invocation and tracking, the operations can be *defined* as `Function`s and inherit all of it. The stack is the boundary principles made mechanical: normalization is `B1 – Either presentation in`, execution over raw types is `B2 – Representations only inside`, and return is `B3 – Tracked forms out by default`; planning and the lift are what the engine promises beyond them.

## IV.2 — Controls (step 1)

### Contract

A `Function` keeps two namespaces strictly apart: the wrapped function's arguments, which are every positional and keyword argument of a call, and the framework's **controls**, which no wrapped function declares. The controls are the sample count (`n_broadcast_samples`), the `include_inputs` switch (IV.6), `method` for route selection and `min_fidelity` as a fidelity floor on it (IV.7), `conversions` for the per-parameter conversion settings of IV.4, the `raw` opt-out read at return (IV.10), and the dispatch and orchestration selectors (IV.9). Randomness is not a control: seeding belongs to the workflow scope, and an explicit `key` is an argument of the operations that draw, caller-owned when supplied (IV.8).

**Setting a control.** Each control's effective value is resolved before any argument is read, from three layers: the framework's default, then the decorator or constructor, then a `with_options` view, which returns a callable with the controls revised for the calls made through it and leaves the `Function` itself unchanged. Every control has a default, so a bare decorator and a bare call are complete.

```python
predict.with_options(n_broadcast_samples=1000)(theta=prior, x=x_obs)
# controls go to with_options; theta and x bind to the wrapped function
```

*Requires:* every control set is one the framework defines, and its value is admissible, for example a positive sample count or a known dispatch mode. *On failure:* a `TypeError` or `ValueError` at the decorator or at `with_options`, before any call.

### Rationale

A `Function` must wrap an *ordinary* function with no naming restrictions (`C5 – Naming for unambiguous meaning`): a user should never have to rename a `seed` parameter because the framework wanted that word. Holding the controls in a separate namespace removes the collision while keeping the bare decorator and a single call site convenient, and with seeding scoped rather than carried (IV.8), ProbPipe claims no `seed` argument at all.

### Open points

- *Default sample count.* How many draws a broadcast takes by default is unsettled; the default is a speed-versus-accuracy ceiling, and an explicit per-call override is always available. The default should signal "rough estimate," not "tuned."

## IV.3 — Binding (step 2)

### Contract

The arguments bind to the wrapped function's signature by Python's own rules (III.3). An argument that is a tracked term becomes a graph **dependency** and any other a plain **input**; the dependencies are the result's provenance parents and the inputs are recorded by parameter name (IV.10). *Requires:* every argument binds to a parameter. *On failure:* Python's binding error, naming the parameter.

### Rationale

Binding by the ordinary signature is what lets the wrapped function stay ordinary (`C1 – Uniform interface to functions, distributions, and values`), and classifying the arguments at binding is what gives every result its lineage (`C6 – Traceable and reproducible workflows`).

## IV.4 — Normalization (step 3)

### Contract

Normalization brings each bound argument to a tracked term its parameter accepts, in three sub-steps that run in order for every argument.

**3a. Wrap.** A raw argument is wrapped into the kind it is by the **kind-directed** table, the same table that wraps a return (IV.10): a tracked term is kept as it is; a raw callable becomes a `Function`; a raw mapping becomes a `Record` (III.5); a raw array becomes a `NumericArray`; a backend distribution passes to 3b; and a value no other kind admits becomes an `Opaque` (III.2), as a list, a tuple, or a set does. A numeric host is recognized through the array-backend registry (II.3), registry first and duck typing second: a registered container supplies its event shape and dtype without being materialized, its named dimensions bind the spec's symbolic dimensions (II.1), and its remaining metadata, such as coordinates and attributes, is carried as annotations (II.4). *Requires:* the host constructs as its kind. *On failure:* that kind's construction error, for example a mapping keyed by a name that is no identifier.

**3b. Convert.** A distribution-shaped argument whose class is not the one its parameter names is converted through the converter registry, which is the `convert` operation (V.10) applied on entry: a backend distribution to its ProbPipe form, and a ProbPipe distribution to the class the parameter names. The pre-conversion law becomes a provenance parent of the converted one. The parameter's entry in the `conversions` control (IV.2) selects the converter by name, floors its fidelity, and passes any converter-specific arguments; where the control is silent the framework's defaults apply. *Requires:* a converter feasible at or above the floor. *On failure:* `ResolutionError` from the registry (II.7), naming the converters tried.

**3c. Admit.** The normalized argument's kind is checked against what the parameter accepts, which is the kind of its declared spec for a `@function`, any kind where the parameter is unannotated, and the kinds its role names for an operation (V.0). A `Distribution` or a `Batch` over an accepted kind is admitted for lifting (IV.5), whereas a `ConditionalDistribution` at a value parameter is refused, since a kernel has no marginal law to lift over. Every kind's term is a `TrackedTerm`, so a bare object, which 3a wrapped as an `Opaque`, passes only a parameter that accepts `Opaque`, whatever methods it carries. *Requires:* the kind is accepted, directly or as the element kind of a lifted argument. *On failure:* `ApplicabilityError`, naming the parameter, what it accepts, and what arrived.

### Rationale

Normalization is `B1 – Either presentation in` made mechanical, in the order the information becomes available: a raw host says what kind it is, a distribution says what class it has, and only then can the parameter's acceptance be judged. Conversion as a step of its own leaves the two entry registries visible as registries (`D2 – Generality first`), and reading a container's named dimensions into the declaration is `D5 – Explicit, carried structure` taken from the data rather than declared by hand.

## IV.5 — Lifting (step 4)

### Contract

A `Function` compares each admitted argument against the kind its parameter expects and lifts where they differ. A `Distribution` where a value is expected is **broadcast**: it is sampled `n` times and the function applied to each draw, so the result approximates the pushforward of the input through `f`. A `Batch` where one element is expected is **swept**: the function is mapped over its elements. Both at once give a **nested** sweep of broadcasts, one broadcast within each element, and neither gives a plain call.

**The trigger.** A parameter that is unannotated, or annotated with a value type, expects a value, so a distribution passed in that position is lifted. A parameter annotated `Distribution`, `Distribution[...]`, or a distribution capability protocol of III.8 declares that the function consumes the distribution itself, which then passes through unlifted. The function capabilities of III.3 and III.15 annotate `Function`-valued parameters, which are values, so the value rule above governs them. Per draw, the function receives the draw as `sample` returns it (V.3), at the kind the law's event declaration names.

**Grouping and correlation.** The lifted arguments are grouped by **root ancestor**, transitively: sibling views of one parent, the same distribution passed twice, and a parent passed alongside its own view all fall in one group. Each group contributes one joint draw per repetition, so dependence between its members is preserved through `f` rather than broken by independent sampling. A view lifts by sampling its parent, so its parent must itself sample. Groups with no common ancestor draw independently: the lift samples the **product law**, and, as a corollary, detached marginals of one joint lift independently while its views co-sample. For example, `f(d, d["x"])` forms one group, and each repetition evaluates `f` on a joint draw and its own projection, while `f(d1, d2)` for unrelated `d1` and `d2` samples the product of their laws. The number of lifted arguments changes only the grouping.

**Alignment.** Swept batches align by level name, as V.11 states for batched operations, and the classification fixes the regime and the groups without reading any value. *Requires:* the aligned levels have broadcast-compatible shapes. *On failure:* `ApplicabilityError`, naming the level.

### Rationale

This is `C4 – Function lifting` realized in both of its cases: replacing an argument of `f` with a distribution over that argument's type leaves `f` well-defined and returns the pushforward, and replacing one with a batch over that type leaves it equally well-defined and returns the broadcast: one substitution rule, differing only in whether the multiplicity is a law or a collection. Realizing the first by sampling and the second by the sweep keeps the contract general (`D2 – Generality first`): it works for any `f`, any number of lifted arguments, and any distribution that samples, with exact routes registering above those floors, and it leaves the user's function unchanged. The annotation trigger makes the lifting boundary explicit in the signature, where the author already states intent. Co-sampling by root ancestor is what makes the lift *correct* rather than merely type-correct: it is the same correlation-preserving mechanism the field views rest on, so passing sibling views through a function transports their joint law.

## IV.6 — Planning (step 5)

### Contract

Planning validates the declarations and computes the **result declaration**, once per call and before anything runs, from what is static under compilation: the specs of the arguments, a lifted argument contributing its element spec, declared capabilities, the paths an argument supplied, and the control values, and never traced array data. The declaration is therefore `jit`-safe and cannot depend on what the computation produces, and the executed result must satisfy it (IV.10).

**The declared output.** The optional `output_spec=` on the decorator declares what the producer knows and inference cannot recover: an `OutputSpec` (II.2), or a bare term spec with the name defaulted to the function's own, captured once. Its spec declares the result's kind, for example a `FunctionSpec` for a fitted mapping, and a `RecordSpec` declares field structure such as a constrained support or a shared symbolic dimension. The declaration is bound per call by unification, which validates the output, and a tracked record supplied at a record-shaped position is stored with its identity kept (III.5). For a non-record output, the declaration's name is the result's field name wherever one is required, as in the joint layout below and composition's produced slots (III.12), so no field name derives from the function's renamable label.

```python
@function(output_spec=RecordSpec(rate=NumericArraySpec(("obs",), float32, positive)))
def rate(x):
    return jnp.exp(x)
# inference alone would read the support as real; the declaration carries support=positive
# and binds "obs" to the actual output length on each call
```

**The result of a lift.** A broadcast's result declaration is an empirical distribution over the output declaration, or the joint below under `include_inputs`; a sweep's is a batch on the swept levels whose `element_spec` is the output declaration bound for the call or, with no declaration, the spec the row results share (II.5); a nested sweep's is a batch of empiricals. An operation's result declaration is its result rule (V.0).

**Including the inputs.** By default the result holds only the outputs. With `include_inputs=True` it is instead the **joint** empirical distribution over the sampled inputs and the outputs: one top-level field per lifted parameter, named by the parameter, whose subtree is that argument's event schema, plus the output fields. A single-field argument still nests, so the layout never depends on the argument's field count. A term-drawing law has no fields to include, so a parameter drawn from one contributes its draw as a single term-valued field. A plain-value argument contributes no field, since it is recorded in provenance rather than sampled. Sibling uniqueness applies across the lifted-parameter names and the output names, and a collision, such as an output declared under the name of one of its own lifted parameters, is an error, raised at construction when the declaration makes it knowable and at result construction otherwise. Grouping affects only how the draws are taken.

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

*Requires:* every argument conforms to what its parameter accepts under one unification (II.1), every condition the `Function` declares holds, and the result declaration is complete, its symbolic dimensions bound. *On failure:* `ApplicabilityError`, naming the condition, or the dimensions in conflict or left free.

### Rationale

Planning the declaration before execution, from what is static under compilation, is `D5 – Explicit, carried structure` made checkable: the result's structure is known before the work starts and verified after it, and where inference from a returned value would be lossy the producer that knows the support or the dimension identities declares them, so they travel with the result.

## IV.7 — Resolution (step 6)

### Contract

Resolution selects the route that realizes the call. Every candidate is checked on the declarations alone, and the feasible candidates rank by fidelity, then specificity, then registration order (II.7); `method=` names one outright and `min_fidelity=` excludes those below a floor (IV.2). A plain call of a plain function has its body as its one candidate, and an operation resolves among its routes (V.0). A lifted application, which is the direct call `f(d)` or `f(batch)`, resolves through the **evaluation-rule registry**: a `BinaryDispatchRegistry` keyed on the map's and the operand's types whose methods are **evaluation rules**, each a route of the lifted application. `evaluate` (V.1) exposes the same registry as an operation, so the direct call and `evaluate` take the same route. The rules, in selection order:
- **Closed-form rules** return an exact parametric result. For example, `A @ d` for a Gaussian `d` is again Gaussian, with mean `A @ mean(d)` and covariance `A Σ Aᵀ` built lazily through the operator algebra.
- **Change of variables** applies when the map is invertible and carries the Jacobian claim (`is_invertible` and `SupportsLogDetJacobian`), returning a transformed distribution whose `log_prob` is exact via the log-determinant of the Jacobian.
- **The sampling lift** is the rule registered at the generic pair for a distribution operand and always applies: draws from `d` are pushed through the map, returning an empirical distribution over the outputs, with the sample count and PRNG key as controls. It is the route every plain callable takes, and grouped, multi-distribution lifts always take it, which is what co-sampling requires (IV.5).
- **The elementwise sweep** is the batch counterpart: the rule at the generic pair for a batch operand. A fused batched implementation, such as an operator's matrix–matrix routine or a single vectorized call over array-backed elements, registers above it.

The **floor** is the always-feasible route, which ranks last: in the evaluation registry, the sampling lift for a distribution operand and the elementwise sweep for a batch, so a lift never fails to resolve while its operand samples. Rules need not be exact: an approximate scheme, for example quadrature or an unscented transform, registers at its recorded fidelity, above the lift. The selected route records its name and fidelity in the result's provenance (IV.10). *Requires:* a feasible candidate, or the named one. *On failure:* `ResolutionError`, naming each candidate and what it was missing.

### Rationale

Dispatching over pairs of map and operand types realizes `C3 – Computational detail hidden by default, available on demand`, since a pair with a known closed form or a fused batched routine gets it automatically, while every other pair still resolves through the floors. Registration grows the exact set without changing call sites (`D2 – Generality first`), and recording the producing route makes the approximation explicit (`D1 – Mathematical fidelity`).

## IV.8 — Randomness: workflow scopes and structural keys

### Contract

Randomness is scoped, not carried: no `Function` and no operation holds a seed. A **workflow scope** owns every ProbPipe-caused draw inside it, and each such **workflow-owned random event** receives a key derived structurally from the scope's root seed, never from a counter, ambient state, or the order of execution.

```python
with workflow_run(seed=42):              # a seeded scope: same seed, same structure ⇒ same draws
    result = predict(theta=posterior, x=X_new)

with replay_run(result.provenance):      # re-runs one recorded call on its recorded draws
    replayed = predict(theta=posterior, x=X_new)
```

**Scopes.** `workflow_run(seed)` opens a scope whose root the seed fixes; entering it again with the same stochastic structure reproduces its workflow-owned events. `workflow_run()` with no seed opens an anonymous scope rooted in fresh entropy, and a bare omitted-key call outside any scope gets its own equivalent ephemeral scope, so unscoped code is fresh rather than repeatable. Scopes nest, a nested scope extending the enclosing structure rather than restarting it, and a scope fixes the run's provenance mode (II.4) at entry. A scope is thread- and task-local: work crosses into another thread or task only through the engine's managed work items, and entering a copied scope unmanaged raises rather than silently forking the stream.

**The key rule.** An operation called with an explicit PRNG `key` is **caller-owned**: the key is consumed as given and the scope's stream is untouched. An operation with the key omitted is **workflow-owned**: its key is derived from the scope. The two never interact, so code that threads keys explicitly composes with scoped code without perturbing either stream.

**Structural event identity.** A workflow-owned key is a pure function of the scope's root seed and the event's identity, three structural coordinates:
1. the **occurrence path**: the invocation's position in the workflow, extended by nesting and by managed work items, with repeated or recursive invocations of the same call distinguished by a deterministic logical ordinal;
2. the **stochastic source**: which co-sampling group of IV.5 is drawing;
3. the **logical unit**: which broadcast repetition or sweep cell consumes the draw.

Identity follows the workflow's logical structure: ordinals are fixed by program order, no key is drawn twice, and the same call produces the same draws under any dispatch mode, thread count, or orchestration.

**Consequences.** Because keys attach to structure, perturbing an input reuses the same keys, preserving common random numbers: comparisons across nearby inputs and reparameterization gradients stay low-variance rather than being swamped by independent sampling noise. A fresh estimate or an independent stream is obtained by changing the seed. The one cost is that the streams are tied to the program's structure, so restructuring the computation reshuffles them.

**The derivation contract.** The derivation from seed and identity to key is a versioned contract: it is keyed and domain-separated, so distinct events cannot collide, and its version is recorded with every execution and every replay record, so a reproduced run either reproduces under the same contract or refuses. The backend key type is produced through an adapter whose behavior is certified before first use, and a deviation raises rather than yielding silently different draws. The version string, not the hash primitive, is what the design fixes: the primitive may be anything that honors the contract.

**Replay.** `replay_run(provenance)` re-executes exactly one recorded `Function` call with its recorded workflow-owned events. Replay validates rather than approximates: the recorded plan, the execution capability, the derivation version, and the events actually observed must all match, and an incompatibility raises. Caller-owned keys need no replay support, since they reproduce from their values.

**Caching.** The same structural identity that makes a run reproducible makes it cacheable: a call whose function, operand fingerprints, resolved controls, and random-event identities all match a recorded call must produce the same result, so the result may be served from a cache instead of recomputed. Caching is therefore a workflow-layer option with no semantic effect, off by default. What it can serve is bounded by what provenance records: under the lightweight mode the identity tier is structural, so a cache keyed on it assumes unchanged inputs, while the full mode's content-verifiable fingerprints let a hit be checked rather than assumed.

### Rationale

Scoped, structural randomness is `C6 – Traceable and reproducible workflows` made mechanical: one seed reproduces a workflow, one provenance record replays a call, and identity-derived keys make the result independent of execution order, so reproducibility survives parallelism and orchestration rather than trading off against them. Key management is `C3 – Computational detail hidden by default, available on demand` applied to randomness: with the key omitted, draws are fresh and reproducible on demand, while an explicit key hands the caller full manual control. Versioning the derivation makes reproduction exact: a run reproduces under the contract that produced it or refuses.

## IV.9 — Execution (step 7)

### Contract

Execution runs the selected route over the points: once for a plain call, per draw for a broadcast, and per element for a sweep, each point with its key (IV.8). This is the only step that reads values. The points run under two orthogonal computational settings, both with defaults so a user need not set them:

- **Dispatch mode: how the points run.** `jax` vectorizes them (one `vmap`); `sequential` runs them one at a time; `thread` runs them on a thread pool; `auto` probes whether the call is array-traceable and picks `jax`, falling back to `sequential`. Under `jax`, a lifted call is traced end-to-end, and it differentiates end-to-end when the `Function` claims `SupportsDifferentiation`. Because keys attach to structure (IV.8), the result is identical across `jax`, `sequential`, and `thread` up to the floating-point effects of evaluation order, and parallel execution contends for no mutable random state. Each dispatch mode's versioned capability contract (IV.8) is checked before sampling begins, so an unsupported mode is refused up front rather than approximated.
- **Orchestration: whether the call is traced.** Off by default. A `Function` can instead run as a traced task or flow, recording the computation graph for lineage and scheduling. Tracing never changes the result. Work that crosses a thread, task, or orchestrated-flow boundary travels as a **managed work item** extending the occurrence path (IV.8), so orchestrated and distributed runs draw from the same structural stream as local ones.

*Requires:* the route completes. *On failure:* the route's own error propagates, as for any numerical method.

### Rationale

Dispatch and orchestration are `C3 – Computational detail hidden by default, available on demand` in action: the algorithm that realizes a lifted call, and whether its graph is recorded, are computational concerns, handled automatically by default and exposed for users who need control. Keeping them orthogonal, so that how a call runs is independent of whether it is traced, lets the fast vectorized path and full lineage tracking compose rather than trade off.

### Open points

- *Non-array backends.* Lifting and dispatch are array-native, built for a differentiable array backend. First-class support for other tensor frameworks, for example a Torch model as the wrapped function with conversion at the boundary, is not yet settled, though it should be feasible through Keras.

## IV.10 — Return (step 8)

### Contract

Return assembles the points into the declared result and returns it tracked. Each point's result is validated against the planned declaration and the points are assembled into the result IV.6 declares. The result then crosses the kind-directed wrap of IV.4: a tracked term keeps its kind under the call's fresh, derived identity, so a `Distribution` or a `NumericArray` the body produced is the result rather than a field inside a fresh `Record`, and a raw return takes the kind it is. An opaque output samples downstream but carries none of the numeric interface, so a function whose output deserves structure returns a mapping or declares it (IV.6). The result receives identity: an auto-derived name, and a `provenance` recording the `Function`, its dependencies as parents and its inputs by name (IV.3), the selected route with its fidelity (IV.7), and the resolved controls. With `raw=True` the identity is not constructed and the result returns detached (II.4); the enclosing call never re-wraps it. *Requires:* the result satisfies the planned declaration. *On failure:* a wrong kind and a schema mismatch raise distinct errors, and either is a defect of the route rather than of the caller.

### Rationale

Return is `B3 – Tracked forms out by default` at the one boundary every call crosses, and attaching provenance there is what makes every result record how it was produced (`C6 – Traceable and reproducible workflows`).

## IV.11 — Differentiability claims

### Contract

The decorator's `differentiable` argument declares which inputs gradients propagate through: a non-empty `NumericSpec` covering exactly those values, which are array-native with no gradient-breaking operations. Omitting the argument makes no claim, and there is no shorthand for claiming every input, so a claim never widens as the function changes. The constructed `Function` then carries `SupportsDifferentiation` with that schema, read through `is_differentiable` wherever gradients are required. Like the declared sides, the declaration is fixed at construction: a claim, not a control.

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

### Rationale

Differentiability as a declared claim is `D3 – Capability-based operations` applied to gradients: support is promised by the object and checked before a backend trace, never inferred from a value being numeric.

### Open points

- *Differentiability of sampling-based routes.* Whether a Monte Carlo fallback differentiates through its sampler's reparameterization is unsettled. So is the eventual `grad` operation the claims feed, with registered routes: a custom gradient method where an object supplies one, the automatic-differentiation route gated by the declared template, and finite differences as the fallback at approximate fidelity. Both are left to a dedicated pass.
