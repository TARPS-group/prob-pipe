# Part V — Operations

Parts II–IV fixed the *shared abstractions*, the *values and distributions*, and the *computations* that act on them. Part V fixes the **operations**. Every operation is a computation, so it inherits lifting, provenance, dispatch, and orchestration from the computation layer. This part adds only what is specific to each operation: its signature, its argument and return types and shapes, the choice between an exact and a default algorithm, its error behavior, and how its result is wrapped and tracked. Every operation is also capability-dispatched, so it applies to any object that implements the matching capability, and closed, so it returns another **tracked term**.

**Conventions.** The user-facing names are the bare operations (`sample`, `log_prob`, `mean`, …). The implementer counterparts are `_`-prefixed (`_sample`, `_log_prob`, …) and, for conditional distributions, the `_conditional_*` family.

## V.0 — The operation model

### Contract

Every operation shares the following mechanics:
- **Capability dispatch.** An operation tests the relevant structural protocol with `isinstance(obj, SupportsX)` and calls the implementer method. An object that does not implement the capability raises a clear error.
- **Two levels.** The implementer's `_x` method acts on the raw value type `T`. The user-facing operation wraps the result at the boundary, attaches identity, metadata, and provenance, and broadcasts, all at no cost to the implementer.
- **The wrap boundary.** A raw value becomes the `Record` its `event_template` describes, and the wrap is unconditional: a bare array becomes a single-field `Record`, a raw value that is already a `Record` is not wrapped again, and any other raw type, including the `Distribution` a random measure yields, becomes the `Record`'s leaf value. A concrete family such as a `Normal` can therefore be written in plain arrays, and a user writes the family's natural constructor rather than supplying a template by hand. A `ConditionalDistribution` adds the `given=` fused paths over its `_conditional_*` methods.
- **The `raw` opt-out.** Every operation accepts `raw=True` and returns the implementer method's result unchanged, skipping the wrap and the identity attachment: the raw event type `T` for an event-typed result, the bare array for a density, and the bare operator for `cov`. The wrapped, tracked result is always the default.
- **Default algorithms.** An operation uses a closed form when the object provides one, and otherwise a sensible default such as Monte Carlo, with the sample count and PRNG key exposed as controls for users who need them.
- **Controls in the signature.** An operation is not a user function, so its signature is its control surface: the PRNG key, sample counts, and `method=` selection appear as ordinary parameters. The separate options namespace exists for wrapped user functions, whose argument names the framework must not constrain.
- **Randomness.** No operation draws from ambient random state. A draw-producing operation (`sample`) requires an explicit PRNG `key` and raises without one, since without ambient state a defaulted key would silently repeat a draw. An operation whose contract is a deterministic quantity but that falls back to Monte Carlo, such as a moment, a marginal, or an inference result, instead takes the key as an optional control defaulting to one derived from the `seed`, so the Monte Carlo route neither enters the signature nor burdens the caller. The resolved key is recorded in `provenance`.
- **Output identity.** Every tracked term an operation mints is fully specified, never left implicit. Its `event_template` is carried or derived from the inputs, its `provenance` records the operation and its parent descriptors, and its `name` is auto-derived and marked `name_is_auto`. Free-form `annotations` do not auto-propagate, since lineage rides on `provenance` rather than on annotations.
- **Tracking scope.** Every result is tracked, including a numeric summary: a density returns as a single-field `Record` whose provenance records the operation, and the bare value is one `raw=True` away.

### Rationale

The operation model is where the core principles become mechanical. Capability dispatch is `D3 – Capability-based operations`: a fixed vocabulary of operations applies to any object that mathematically supports it, independent of the object's concrete encoding. The two-level split serves `C3 – Computational detail hidden by default, available on demand` and `C6 – Traceable and reproducible workflows`, since an implementer writes the mathematics in plain arrays while the user always receives a tracked, named result with its algorithm chosen automatically. Output identity realizes `D5 – Explicit, carried structure` and `C6 – Traceable and reproducible workflows` together, because each result records both its structure and its lineage. Keeping controls in the signature where the signature is the framework's own, and recording the key each result used, is what makes the provenance record complete enough to re-run. That every operation returns another **tracked term** is `D4 – Closed system of objects under operations`.

## V.1 — Moments: `mean`, `variance`, `cov`, `quantile`, `expectation`

### Contract

The moment operations summarize a distribution by a deterministic value.
- `mean(d, raw=False)` and `variance(d, raw=False)` return an event-typed value, that is, a value shaped like a draw. Neither is restricted to numeric draws, but each requires the event type to support it: a random function has a mean function and a pointwise variance function, while a random measure has a mean (the marginalized law) but, in general, no event-typed variance. The result is a `Record` whose event template matches the distribution's, or the raw event-typed value `T` with `raw=True`.
- `cov(d)` requires a numeric draw and returns a covariance operator over the *flattened* draw, a `(vector_size, vector_size)` `LinOp` rather than an event-typed value, since covariance couples distinct coordinates. Its input and output templates are both the distribution's `NumericEventTemplate`, so it applies directly to draws.
- `quantile(d, q)` requires a numeric draw. It takes a level `q ∈ [0, 1]` or array of such levels and returns the quantile for each, computed per coordinate for a multivariate draw. If a single level is provided it returns a `Record`; for multiple levels, it returns a `RecordBatch`.
- `expectation(d, f)` returns `E[f(X)]`, shaped by the output of `f`, for any event type `f` accepts. It returns a `Record`.

Each operation uses a closed form when the distribution implements the matching capability (`SupportsMean`, `SupportsVariance`, `SupportsCovariance`, `SupportsQuantile`, `SupportsExpectation`), whatever the event type. Otherwise, if it implements `SupportsSampling`, it falls back to a Monte Carlo estimate, with the sample count and PRNG key specified by the user. The fallback is defined per event kind. A numeric event averages draws coordinatewise. A function-valued event answers with a lazy function: its mean at a point is the average of the sampled callables there, and its variance the pointwise sample variance. A measure-valued event's mean is the finite mixture of the sampled draws, the Monte Carlo estimate of the mean measure. `cov` and `quantile` keep numeric-only fallbacks, the sample covariance returned as a `DenseLinOp` and the per-coordinate empirical quantiles. `expectation` averages the array outputs of `f` and so applies to any event type that samples. A moment requested of a distribution that supports neither path raises a capability error.

### Rationale

A mean is defined whenever draws can be averaged: coordinate-wise for arrays, pointwise for functions, and set-wise for measures, where `A ↦ E[ξ(A)]` is again a measure. An event-typed variance additionally requires the second moment to be a value of the event type. That holds for arrays and functions, but fails for a general random measure: to be a measure, `A ↦ Var(ξ(A))` would have to be additive, which holds only when disjoint regions are uncorrelated (a completely random measure) and never for a random probability measure, whose fixed total mass forces negative correlation. A random measure's second-moment structure is instead a covariance over pairs of sets, the analog of `cov` rather than of `variance`. Gating `mean` and `variance` by capability rather than by a numeric event is `D1 – Mathematical fidelity`, and `cov` and `quantile` remain numeric-only, since a flat covariance operator and ordered quantile levels have no meaning for a non-numeric draw. `cov` returns a flat operator rather than an event-typed value because it couples coordinates that the event's field structure keeps separate. The closed-form-or-Monte-Carlo split realizes `C3 – Computational detail hidden by default, available on demand`: a distribution that can give an exact moment does, and one that can only sample still answers, approximately.

## V.2 — `sample`

### Contract

`sample(d, key, sample_shape=(), raw=False)` draws from a distribution.
- With `sample_shape=()` it returns a single draw, wrapped as the `Record` the `event_template` describes. A scalar law returns a single-field `Record` keyed by the distribution's name (the key is initialized from the name and thereafter independent of it). The wrap is unconditional: even a draw whose raw type carries identity, such as the `Distribution` a random measure yields, is returned as a `Record` with that draw as its leaf value.
- A non-empty `sample_shape` prepends batch axes and returns a `RecordBatch`, or a `NumericRecordBatch` when the draw is numeric, with those leading dimensions on a level named `draw`.
- With `raw=True`, `sample` skips the wrap and the identity attachment. For `sample_shape=()` it returns the bare draw type `T`; for `T = Record` the wrap is already the identity, so `raw=True` is a no-op and the implementer's record, with its own name, is returned, while the bare types carry no name or provenance. For a non-empty `sample_shape` it returns `T`'s **native batch form**:

| raw type `T` | native batch form |
|---|---|
| `Array` | an array with the `sample_shape` axes leading |
| `Record` / `NumericRecord` | `RecordBatch` / `NumericRecordBatch` |
| `Distribution` | `DistributionBatch` |
| `ConditionalDistribution` | `ConditionalDistributionBatch` |
| function | `FunctionBatch` |
| opaque | `OpaqueBatch` |
- Sampling requires a concrete event template and raises with the free dimensions named; in the fused conditional path, the given value binds them first.
- The PRNG `key` is explicit and supplied by the caller, so a draw is reproducible from its key. Under a non-empty `sample_shape` the key splits by draw index, so the draws are jointly independent and reproducible together; inside a `Computation` the `seed` supplies the key by the same structural split, so draws are never hand-threaded.
- The draw carries `provenance` recording `sample` and the distribution it came from.

For a `ConditionalDistribution`, `sample(K, given=s, key=...)` is the fused conditional path that's equivalent to `sample(condition_on(K, s), key=...)`.

### Rationale

Threading an explicit PRNG `key` makes every draw reproducible from its inputs, which is `C6 – Traceable and reproducible workflows`. Wrapping even a scalar draw as a single-field `Record` lets `sample` return a tracked term of uniform shape whatever the distribution's raw type, serving `C1 – Uniform interface to distributions and values`. The `raw` opt-out serves `C3 – Computational detail hidden by default, available on demand`: the wrapped, tracked draw is the default, but the bare value remains available when the user needs it.

## V.3 — `log_prob` and `unnormalized_log_prob`

### Contract

`log_prob(d, value)` returns the log-density of `value` under `d`. The value may be a `Record` matching the `event_template`, or a bare array for a scalar law.
- `log_prob` requires `SupportsLogProb` and returns the *normalized* log-density.
- `unnormalized_log_prob` requires only `SupportsUnnormalizedLogProb` and returns the log-density up to an additive constant, which is what inference against an unnormalized target needs.
- The result is a tracked term: a single-field `Record` whose provenance records the operation, the distribution, and the scored value, with `raw=True` returning the bare array.
- A scored value binds any symbolic event dimensions for that call only, so one law scores datasets of different sizes.
- A batch of values, or a `DistributionBatch`, maps elementwise to the batched densities, a single-field `NumericRecordBatch`.
- For a `ConditionalDistribution`, `log_prob(K, y, given=s)` is the fused conditional path that's equivalent to `log_prob(condition_on(K, s), y)`.

### Rationale

Splitting `log_prob` from `unnormalized_log_prob` keeps each capability honest (`D1 – Mathematical fidelity`): a distribution that knows its normalizing constant offers the true density, while one that does not still serves inference, which needs the density only up to a constant.

## V.4 — `condition_on` and `predictive`

### Contract

`condition_on(d, given)` fixes some fields of a distribution or conditional distribution and returns the resulting distribution.

**The `given` argument.** `given` is field-keyed: a `Record`, or a mapping from field paths to values, with each value conforming to the spec at its path, checked as one simultaneous unification that binds any symbolic dimensions and raises on jointly inconsistent shapes. Each key must name either a *given* field (a path in the `given_template`) or a *produced* field (a path in the `event_template`), and any other key is an error. A key may also name an interior path, in which case its value is a sub-record checked against the sub-template. Conditioning is stated entirely in terms of fields, and factors never appear in the call: the derived factor graph is read only to decide which case below applies and to carry it out.

**Dispatch.** The operation dispatches on the conditioning capability: a `ConditionalDistribution` always implements it (`_condition_on` is its required primitive), a `Distribution` implements it optionally (`SupportsConditioning`), and the factored classes implement it with one structural rule, dispatched on where each conditioned field sits in the factor graph.
- **Exogenous given, so curry.** Binding a field that the object conditions on but does not produce returns a smaller `ConditionalDistribution`, or an ordinary `Distribution` once all given fields are bound. This is exact and involves no inference. For example, binding a regression model's covariates curries it toward the data-ready likelihood.
- **Upstream or independent produced field, so exact slice.** Conditioning on a produced field that no remaining factor depends on through an unconditioned path returns the exact conditional by slicing the factor graph, again with no inference. The slice is exact only when the field's own factor is single-field or itself implements the conditioning capability; a multi-field factor that cannot condition internally falls to the Bayes' rule case. For example, conditioning a Gaussian prior on one of its own fields slices, while an atomic empirical factor over two fields cannot be sliced on one of them.
- **Downstream data, so Bayes' rule.** Conditioning on a field that downstream factors depend on is an application of Bayes' rule: the conditional is proportional to the joint density at the fixed value, and is generally not available in closed form. It is delegated to an inference algorithm registered for the model, and only the factored classes support it. For example, conditioning on observed responses downstream of the coefficients is the Bayes' rule case. On an atomic distribution, conditioning on produced fields uses its own `SupportsConditioning` when implemented, as a `MultivariateNormal` does exactly, and raises otherwise.

When `given` names several fields, the cases combine: the exact bindings (curry and slice) are applied first, and Bayes' rule runs on what remains. Field classification is computed once, on the graph with every conditioned field marked, so the outcome does not depend on the order the fields are listed. Conditioning on a produced field does not require the given fields to be bound first. The result stays conditional on the unmet givens, with the produced-field conditioning applied within each slice of the given, so the result curries like any other `ConditionalDistribution` and the two orders agree: conditioning on a produced field and then binding the given yields the same distribution as binding the given first. When that produced-field conditioning requires Bayes' rule, the resulting `ConditionalDistribution` may realize the inference lazily, once its given is bound, or through a method that supports amortization.

`condition_on` always binds the supplied value as the field's fixed value, whatever the value's type. The predictive mixture `∫ K(s, ·) μ(ds)` over a mixing distribution `μ` is requested explicitly through the `predictive(K, mixing)` operation. The result carries `provenance` recording the operation and the conditioning fields, and `predictive` additionally records the mixing distribution.

**The inference-method registry.** The Bayes' rule case is dispatched through the **inference-method registry**, a `UnaryDispatchRegistry` keyed on the model's type whose methods are inference algorithms such as MCMC or variational families. Each method registers under a unique `name`, declares the models it applies to, and probes feasibility before it runs. `condition_on(..., method="tfp_nuts")` selects a specific algorithm, and otherwise the registry picks the best applicable one.

**Prioritization.** Methods are ranked by an integer `priority` that tiers them: above 50 is an **exact** method, 1 to 50 an **inexact** one, and 0 is **opt-in-only**, skipped by auto-selection and reachable only by name. A newly registered method is opt-in-only until a contributor classifies it, so adding an algorithm never silently changes what runs. A deployment re-ranks methods at runtime with `set_priorities`, which warns when a method crosses into or out of opt-in-only.

### Rationale

A single operation covers binding, slicing, and Bayes' rule because all three are the same mathematical act of conditioning, and they differ only in whether the conditioned field is exogenous, upstream, or downstream in the factor graph. Collapsing them into one operation keeps the user interface small (`C1 – Uniform interface to distributions and values`), while the derived graph decides the algorithm (`C3 – Computational detail hidden by default, available on demand`).

## V.5 — `joint`

### Contract

Recall that the composition operator `A * B`, exposed on `Distribution` and `ConditionalDistribution` as `__mul__` constructs the joint (conditional) distribution of `A` and `B`. It returns either a `FactoredDistribution` or, when some givens remain, a `FactoredConditionalDistribution`. Its `provenance` records `*` as the operation and the operand factors, and its `name` is auto-derived, as described next.

**Auto-derived names.** A joint is derived, not constructed by hand, so `*` derives its `name` deterministically from its factors. The factors are listed in **canonical order** (the conditional-first topological order of the flattened factor graph, with factors incomparable in the derived graph ordered lexicographically by the fields they produce), and their names are joined by `·`. So `lik * prior` is named `lik·prior`, and because neither association nor the ordering of independent factors changes the canonical list, `A * B * C`, `(A * B) * C`, and `A * (B * C)` produce the same joint distribution. The derived name is marked `name_is_auto`.

Re-composition reads `name_is_auto`. An auto-named operand is **flattened**: its factors enter the new joint directly, its old name is discarded, and a fresh name is derived from the full factor list. An operand whose name the user has pinned with `with_name` is **not** flattened. It enters as a single factor under that name, and that name appears as one token in the parent's derived name. So `(lik * prior).with_name("posterior")` both labels the joint and, in any later composition, keeps it as the single factor `posterior`.

**The realigning `joint` form.** A limitation of `*` is that it requires a producer's field names to match the names its consumer conditions on. This motivates the `joint(A, B, **align)` op, which realigns fields first and then composes exactly as `*`, so it is equivalent to `A * B.with_path_names(**align)`. For example, a likelihood that conditions on `slope` can be combined with a prior where the slope is called `beta` with `joint(lik, prior, beta="slope")`.

### Rationale

Composition is written as an expression so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and `*` returns a first-class joint so the result composes further (`D4 – Closed system of objects under operations`). Deriving the name deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`), while `with_name` lets the user impose grouping where it carries intent. Realignment is an exact rename: `with_path_names` returns the same law under new field names, so `joint` connects mismatched factors without altering their joint law (`D1 – Mathematical fidelity`).

### Open points

- *The `align` contract.* The precise realignment semantics of `joint`, including freshness and injectivity requirements on the new names and which operand's fields may be realigned, is deliberately deferred: the question is subtle, and implementation experience should inform the decision.

## V.6 — `evaluate`

### Contract

`evaluate(f, v)` is the call `f(v)` with the rule registry layered on top. The built-in `Computation` call already covers every operand kind: a plain call for a native value, the elementwise sweep for a batch, and the sampling lift for a distribution. `evaluate` consults the registry first and falls back to exactly that call when no rule applies, so the two differ only where a registered rule improves on sampling. For a distribution operand the result is the law of `f(X)` for `X ~ v`, the pushforward `f♯v`. The map `f` may be a plain function or a `Computation`, a `Bijector`, or a `LinOp`, and `v`'s template must conform to the map's input, unifying any symbolic dimensions, with the result carrying the map's output template under the resulting substitution. For a `LinOp`, `A @ v` is operator sugar for `evaluate(A, v)`. As with `*`, the meanings coexist by operand type: `@` composes two operators and evaluates the map on any other operand.

A map with more than one parameter is evaluated over exactly one of them, with `fixed_args` supplying the rest by name, as in `evaluate(predict, posterior, fixed_args={"x": X_new})`. `fixed_args` must leave exactly one parameter free, the one the operand maps to; leaving two unbound is an error. The registry keys on the map's own type and the fixed arguments pass through to each rule's feasibility check, so binding side arguments neither wraps the map nor loses its registered identity, and the single mapping parameter keeps the operation's controls separate from the map's arguments.

**The evaluation registry.** For a distribution operand the operation dispatches through a `BinaryDispatchRegistry` keyed on the map's and the distribution's types, whose methods are **evaluation rules**. Auto-selection tries the rules in priority order:
- **Closed-form rules** return an exact parametric result. For example, `A @ d` for a Gaussian `d` is again Gaussian, with mean `A @ mean(d)` and covariance `A Σ Aᵀ` built lazily through the operator algebra.
- **Change of variables** applies when the map is an invertible `Bijector`, returning a transformed distribution whose `log_prob` is exact via the log-determinant of the Jacobian.
- **The sampling fallback** always applies, and it *is* the computation lift: draws from `d` are pushed through the map, returning an empirical distribution over the outputs, with the sample count and PRNG key exposed as controls.

The result records which rule produced it, mirroring the converter registry's recorded fidelity, and `method="..."` forces a rule by name. New rules join by registration, so an exact evaluation for a new pair of types is added without touching the operation.

**Linear maps push moments exactly.** Whatever rule realizes `A @ d`, the result's `mean` and `cov` delegate exactly whenever `d` supports them, since `E[A X] = A E[X]` and `Cov(A X) = A Cov(X) Aᵀ`. An approximate linear pushforward therefore still reports exact first and second moments.

Applied to a `ConditionalDistribution`, evaluation acts on the event side, giving the kernel `s ↦ f♯K(s, ·)` with the same given template.

The result is a tracked term: its `provenance` records `evaluate`, the map, and the operand, and its `name` is auto-derived.

### Rationale

`evaluate` is `C4 – Function lifting via pushforward` in operation form: applying a map is one act whatever the operand kind, and replacing a value by a distribution over it leaves that act well-defined, with this operation returning the resulting law directly. Dispatching over pairs of map and distribution types realizes `C3 – Computational detail hidden by default, available on demand`, since a pair with a known closed form gets it automatically and every other pair still answers by sampling. Recording the producing rule keeps the approximation honest (`D1 – Mathematical fidelity`), registration grows the exact set without changing call sites (`D2 – Generality first`), and the result is a tracked term that composes further (`D4 – Closed system of objects under operations`). The operation is named for what it does across every operand kind, with *pushforward* reserved for mathematical statements, as *kernel* is for `ConditionalDistribution`.

### Open points

- *A first-class transport map.* The maps `evaluate` accepts (functions, bijectors, and linear operators) are separate kinds dispatched by type. Whether they should be unified under a first-class transport-map base, carrying evaluation and pushforward as one interface, is left open.

## V.7 — `marginal` and `factor`

### Contract

Two operations read the parts of a structured or factored distribution. (The third access form, the `d[field]` **view**, is not an operation: it returns a correlation-preserving reference into its parent rather than a standalone object, and its contract is fixed with the factored distributions.)

- `marginal(d, field)` returns the **detached** marginal of a field or field group, a standalone `Distribution` with no reference back to `d`. It dispatches on `SupportsMarginals`. When the capability is absent, or the path has no exact route within it, it falls back to a Monte Carlo approximation through `_sample`, projecting draws onto the field and returning an empirical marginal, and it raises a capability error when the distribution cannot sample either. The fallback's provenance names the route (`monte_carlo`) and the sample count, so an approximate marginal is distinguishable from an exact one after the fact.
- `factor(d, name)` returns a building-block **factor** of a joint, keyed by factor name, either a `Distribution` or a `ConditionalDistribution` for a dependent edge. It requires `SupportsFactors`, raising a capability error on a distribution that exposes no factors.

Both return a **tracked term** whose `provenance` records the access and its source distribution.

### Rationale

`marginal` and `factor` are the two query directions of the field and factor interfaces fixed with the factored distributions: `marginal` reads a field's law detached from its joint, and `factor` reads a construction block. Exposing them as named operations rather than as indexing is what separates the detached query from the correlation-preserving `d[field]` view (`D1 – Mathematical fidelity`). Dispatching `marginal` on `SupportsMarginals` opens the detached query to any distribution that knows its marginals, factored or not. Gating `factor` on `SupportsFactors` keeps factor access honest, since only a distribution actually built from named parts can answer it (`D3 – Capability-based operations`).

## V.8 — Batched operations

### Contract

Every operation lifts to a `Batch` by mapping over its elements, which is the computation sweep applied to the operation itself.
- `sample` over a `DistributionBatch` returns a **nested** batch, the outer level ranging over the laws and the inner over each law's draws: `sample(d_batch, key, sample_shape=(S,))` has `axis_groups` `(d_batch.batch_shape, (S,))` and appends an inner draw level, named `draw` or the first free `draw2`, `draw3`, … when `d_batch` already carries one so that level names stay unique, so iterating it visits one law's `RecordBatch` of draws at a time. `log_prob` maps elementwise to the batched densities, with the batch axes preserved.
- A moment over a `DistributionBatch` returns a batch of the corresponding values, such as a `LinOpBatch` for `cov`. A multi-level query nests the same way: `quantile(d_batch, q)` keeps the laws on the outer level and adds an inner level named `quantile` for the levels of `q`.
- **Alignment.** A binary operation matches the operands' levels **by name**: a level in both must have broadcast-compatible shapes, with size-1 broadcasting; a level in only one operand broadcasts across the other; and an outer product is requested by explicit reshaping rather than implied. Because every level is named, there is no positional fallback, and two levels meant to correspond under different names are lined up by renaming one with `with_level_names` first, exactly as `joint` realigns fields for composition. So a flat batch of values on a `laws` level scores against the `laws` level of a nested sampling result. `given=` accepts a `RecordBatch` and yields the `DistributionBatch` of conditioned laws.
- Two operands are exempt from batch lifting: the factors of composition (`*` and `joint`) and the map operand of `evaluate`, which are consumed as objects rather than swept.
- When the elements are array-backed, the map is a single vectorized call rather than a Python loop, which is the batched-backend optimization.
- An operation applied to a batch whose elements lack the required capability raises the same capability error a single element would.

### Rationale

A batched operation is not a new operation but the computation sweep applied to an existing one, so a `Batch` supports exactly the operations its elements do (`D3 – Capability-based operations`). Nesting the laws level over the draws level keeps two genuinely different multiplicities distinct in the result itself (`D1 – Mathematical fidelity`), and matching levels by name rather than by position carries `C5 – Naming for unambiguous meaning` onto the multiplicity axis, making the rename the single tool for lining levels up, as field renaming is for composition. When the elements are array-backed the sweep is a single vectorized call, which preserves differentiability (`D6 – Differentiability where possible`).
