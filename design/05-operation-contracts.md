# Part V — Operation Contracts

Parts II–IV fixed the *infrastructure*, the *objects*, and the *workflow functions* that act on them. Part V fixes the **operations**. Every operation is a workflow function, so it inherits lifting, provenance, dispatch, and orchestration from the workflow-function layer. This part adds only what is specific to each operation: its signature, its argument and return types and shapes, the choice between an exact and a default algorithm, its error behavior, and how its result is wrapped and tracked. Every operation is also capability-dispatched, so it applies to any object that implements the matching capability, and closed, so it returns another **tracked term**.

**Conventions.** The user-facing names are the bare operations (`sample`, `log_prob`, `mean`, …). The implementer counterparts are `_`-prefixed (`_sample`, `_log_prob`, …) and, for conditional distributions, the `_conditional_*` family.

## V.0 — The operation model

### Contract

Every operation shares the following mechanics: 
- **Capability dispatch.** An operation tests the relevant structural protocol with `isinstance(obj, SupportsX)` and calls the implementer method. An object that does not implement the capability raises a clear error. 
- **Two levels.** The implementer's `_x` method acts on the raw value type `T`. The user-facing operation wraps the result at the boundary, attaches identity, metadata, and provenance, and broadcasts, all at no cost to the implementer.
- **The wrap boundary.** A raw value becomes the `Record` its `event_template` describes, and the wrap is unconditional: a bare array becomes a single-field `Record`, a raw value that is already a `Record` is not wrapped again, and any other raw type, including the `Distribution` a random measure yields, becomes the `Record`'s leaf value. A concrete family such as a `Normal` can therefore be written in plain arrays, and a user writes the family's natural constructor rather than supplying a template by hand. A `ConditionalDistribution` adds the `given=` fused paths over its `_conditional_*` methods.
- **The `raw` opt-out.** Any operation whose result is event-typed accepts `raw=True` and returns the raw event type `T` unwrapped, carrying no name or provenance. The wrapped, tracked result is always the default.
- **Default algorithms.** An operation uses a closed form when the object provides one, and otherwise a sensible default such as Monte Carlo, with the sample count and PRNG key exposed as controls for users who need them.
- **Output identity.** Every tracked term an operation mints is fully specified, never left implicit. Its `event_template` is carried or derived from the inputs, its `provenance` records the operation and its parent descriptors, and its `name` is auto-derived and marked `name_is_auto`. Free-form `annotations` do not auto-propagate, since lineage rides on `provenance` rather than on annotations.
- **Tracking scope.** Results that are values or distributions are `Tracked`. Whether a bare numeric summary, such as a `log_prob` density, is itself wrapped is settled per operation.

### Rationale

The operation model is where the core principles become mechanical. Capability dispatch is `D3 – Capability-based operations`: a fixed vocabulary of operations applies to any object that mathematically supports it, independent of the object's concrete encoding. The two-level split serves `C3 – Computational detail hidden by default, available on demand` and `C6 – Traceable and reproducible workflows`, since an implementer writes the mathematics in plain arrays while the user always receives a tracked, named result with its algorithm chosen automatically. Output identity realizes `D5 – Explicit, carried structure` and `C6 – Traceable and reproducible workflows` together, because each result records both its structure and its lineage. That every operation returns another **tracked term** is `D4 – Closed system of objects under operations`.

## V.1 — Moments: `mean`, `variance`, `cov`, `quantile`, `expectation`

### Contract

The moment operations summarize a distribution by a deterministic value.
- `mean(d, raw=False)` and `variance(d, raw=False)` return an event-typed value, that is, a value shaped like a draw. Neither is restricted to numeric draws, but each requires the event type to support it: a random function has a mean function and a pointwise variance function, while a random measure has a mean (the marginalized law) but, in general, no event-typed variance. The result is a `Record` whose event template matches the distribution's, or the raw event-typed value `T` with `raw=True`. 
- `cov(d)` requires a numeric draw and returns a covariance operator over the *flattened* draw, a `(vector_size, vector_size)` `LinOp` rather than an event-typed value, since covariance couples distinct coordinates. Its input and output templates are both the distribution's `NumericEventTemplate`, so it applies directly to draws.
- `quantile(d, q)` requires a numeric draw. It takes a level `q ∈ [0, 1]` or array of such levels and returns the quantile for each, computed per coordinate for a multivariate draw. If a single level is provided it returns a `Record`; for multiple levels, it returns a `RecordBatch`. 
- `expectation(d, f)` returns `E[f(X)]`, shaped by the output of `f`, for any event type `f` accepts. It returns a `Record`. 

Each operation uses a closed form when the distribution implements the matching capability (`SupportsMean`, `SupportsVariance`, `SupportsCovariance`, `SupportsQuantile`, `SupportsExpectation`), whatever the event type. Otherwise, if it implements `SupportsSampling`, it falls back to a Monte Carlo estimate, with the sample count and PRNG key specified by the user. The fallback averages draws, so for `mean` and `variance` it requires a numeric event, while for `expectation` it averages the array outputs of `f` and so applies to any event type that samples. A moment requested of a distribution that supports neither path raises a capability error.

### Rationale

A mean is defined whenever draws can be averaged: coordinate-wise for arrays, pointwise for functions, and set-wise for measures, where `A ↦ E[ξ(A)]` is again a measure. An event-typed variance additionally requires the second moment to be a value of the event type. That holds for arrays and functions, but fails for a general random measure: to be a measure, `A ↦ Var(ξ(A))` would have to be additive, which holds only when disjoint regions are uncorrelated (a completely random measure) and never for a random probability measure, whose fixed total mass forces negative correlation. A random measure's second-moment structure is instead a covariance over pairs of sets, the analog of `cov` rather than of `variance`. Gating `mean` and `variance` by capability rather than by a numeric event is `D1 – Mathematical fidelity`, and `cov` and `quantile` remain numeric-only, since a flat covariance operator and ordered quantile levels have no meaning for a non-numeric draw. `cov` returns a flat operator rather than an event-typed value because it couples coordinates that the event's field structure keeps separate. The closed-form-or-Monte-Carlo split realizes `C3 – Computational detail hidden by default, available on demand`: a distribution that can give an exact moment does, and one that can only sample still answers, approximately.

## V.2 — `sample`

### Contract

`sample(d, key, sample_shape=(), raw=False)` draws from a distribution.
- With `sample_shape=()` it returns a single draw, wrapped as the `Record` the `event_template` describes. A scalar law returns a single-field `Record` keyed by the distribution's name. The wrap is unconditional: even a draw whose raw type carries identity, such as the `Distribution` a random measure yields, is returned as a `Record` with that draw as its leaf value.
- A non-empty `sample_shape` prepends batch axes and returns a `RecordBatch`, or a `NumericRecordBatch` when the draw is numeric, with those leading dimensions.
- With `raw=True`, `sample` skips the wrap. For `sample_shape=()` it returns the bare draw type `T`, carrying no name or provenance, whatever the event type. For a non-empty `sample_shape` it returns `T`'s batch form: an array with the `sample_shape` axes leading for a numeric draw, and an element batch otherwise (a `DistributionBatch` for a distribution-valued draw, and a `FunctionBatch` or `OpaqueBatch` for function-valued or opaque draws).
- The PRNG `key` is explicit and supplied by the caller, so a draw is reproducible from its key.
- The draw carries `provenance` recording `sample` and the distribution it came from.

For a `ConditionalDistribution`, `sample(K, given=s, key=...)` is the fused conditional path that's equivalent to `sample(condition_on(K, s), key=...)`. 

### Rationale

Threading an explicit PRNG `key` makes every draw reproducible from its inputs, which is `C6 – Traceable and reproducible workflows`. Wrapping even a scalar draw as a single-field `Record` lets `sample` return a tracked term of uniform shape whatever the distribution's raw type, serving `C1 – Uniform interface to distributions and values`. The `raw` opt-out serves `C3 – Computational detail hidden by default, available on demand`: the wrapped, tracked draw is the default, but the bare value remains available when the user needs it. 

## V.3 — `log_prob` and `unnormalized_log_prob`

### Contract

`log_prob(d, value)` returns the log-density of `value` under `d`. The value may be a `Record` matching the `event_template`, or a bare array for a scalar law.
- `log_prob` requires `SupportsLogProb` and returns the *normalized* log-density.
- `unnormalized_log_prob` requires only `SupportsUnnormalizedLogProb` and returns the log-density up to an additive constant, which is what inference against an unnormalized target needs.
- A batch of values, or a `DistributionBatch`, maps elementwise to a batched array of densities.
- For a `ConditionalDistribution`, `log_prob(K, y, given=s)` is the fused conditional path that's equivalent to `log_prob(condition_on(K, s), y)`.

### Rationale

Splitting `log_prob` from `unnormalized_log_prob` keeps each capability honest (`D1 – Mathematical fidelity`): a distribution that knows its normalizing constant offers the true density, while one that does not still serves inference, which needs the density only up to a constant.

### Open points

- *Tracking of densities.* A density is a number rather than a value or distribution, so the default is a bare array. Whether a density is instead returned as a `Tracked` scalar is the per-operation instance of the tracking-scope question.

## V.4 — `condition_on` and `predictive`

### Contract

`condition_on(d, given)` fixes some fields of a distribution or conditional distribution and returns the resulting distribution.

**The `given` argument.** `given` is field-keyed: a `Record`, or a mapping from field paths to values, with each value conforming to the spec at its path. Each key must name either a *given* field (a path in the `given_template`) or a *produced* field (a path in the `event_template`), and any other key is an error. Conditioning is stated entirely in terms of fields, and factors never appear in the call: the derived factor graph is read only to decide which case below applies and to carry it out.

**Dispatch.** The operation dispatches on the conditioning capability: a `ConditionalDistribution` always implements it (`_condition_on` is its required primitive), a `Distribution` implements it optionally (`SupportsConditioning`), and the factored classes implement it with one structural rule, dispatched on where each conditioned field sits in the factor graph.
- **Exogenous given, so curry.** Binding a field that the object conditions on but does not produce returns a smaller `ConditionalDistribution`, or an ordinary `Distribution` once all given fields are bound. This is exact and involves no inference.
- **Upstream or independent produced field, so exact slice.** Conditioning on a produced field that no remaining factor depends on through an unconditioned path returns the exact conditional by slicing the factor graph, again with no inference.
- **Downstream data, so Bayesian inversion.** Conditioning on a field that downstream factors depend on requires inference. It is delegated to an inference algorithm registered for the model, and only the factored classes can be inverted.

When `given` names several fields, the cases combine: the exact bindings (curry and slice) are applied first, and inversion runs on what remains. Conditioning on a produced field does not require the given fields to be bound first. The result stays conditional on the unmet givens, with the produced-field conditioning applied within each slice of the given, so the result curries like any other `ConditionalDistribution` and the two orders agree: conditioning on a produced field and then binding the given yields the same distribution as binding the given first. When that produced-field conditioning requires inversion, the resulting `ConditionalDistribution` may realize the inference lazily, once its given is bound, or through a method that supports amortization.

Conditioning on a *distribution* over the given, rather than on a value, returns the predictive mixture `∫ K(s, ·) μ(ds)`, exposed through the `predictive` convenience operation. The result carries `provenance` recording the operation and the conditioning fields.

**The inference-method registry.** Bayesian inversion is dispatched through the **inference-method registry**, a `UnaryDispatchRegistry` keyed on the model's type whose methods are inference algorithms such as MCMC or variational families. Each method registers under a unique `name`, declares the models it applies to, and probes feasibility before it runs. `condition_on(..., method="tfp_nuts")` selects a specific algorithm, and otherwise the registry picks the best applicable one.

**Prioritization.** Methods are ranked by an integer `priority` that tiers them: above 50 is an **exact** method, 1 to 50 an **inexact** one, and 0 is **opt-in-only**, skipped by auto-selection and reachable only by name. A newly registered method is opt-in-only until a contributor classifies it, so adding an algorithm never silently changes what runs. A deployment re-ranks methods at runtime with `set_priorities`, which warns when a method crosses into or out of opt-in-only.

### Rationale

A single operation covers binding, slicing, and inversion because all three are the same mathematical act of conditioning, and they differ only in whether the conditioned field is exogenous, upstream, or downstream in the factor graph. Collapsing them into one operation keeps the user interface small (`C1 – Uniform interface to distributions and values`), while the derived graph decides the algorithm (`C3 – Computational detail hidden by default, available on demand`).

## V.5 — `joint`

### Contract

Recall that the composition operator `A * B`, exposed on `Distribution` and `ConditionalDistribution` as `__mul__` constructs the joint (conditional) distribution of `A` and `B`. It returns either a `FactoredDistribution` or, when some givens remain, a `FactoredConditionalDistribution`. Its `provenance` records `*` as the operation and the operand factors, and its `name` is auto-derived, as described next. 

**Auto-derived names.** A joint is derived, not constructed by hand, so `*` derives its `name` deterministically from its factors. The factors are listed in **canonical order** (the conditional-first topological order of the flattened factor graph, with mutually independent factors ordered by the canonical order of the fields they produce), and their names are joined by `·`. So `lik * prior` is named `lik·prior`, and because neither association nor the ordering of independent factors changes the canonical list, `A * B * C`, `(A * B) * C`, and `A * (B * C)` produce the same joint distribution. The derived name is marked `name_is_auto`.

Re-composition reads `name_is_auto`. An auto-named operand is **flattened**: its factors enter the new joint directly, its old name is discarded, and a fresh name is derived from the full factor list. An operand whose name the user has pinned with `with_name` is **not** flattened. It enters as a single factor under that name, and that name appears as one token in the parent's derived name. So `(lik * prior).with_name("posterior")` both labels the joint and, in any later composition, keeps it as the single factor `posterior`.

**The realigning `joint` form.** A limitation of `*` is that it requires a producer's field names to match the names its consumer conditions on. This motivates the `joint(A, B, **align)` op, which realigns fields first and then composes exactly as `*`, so it is equivalent to `A * B.with_names(**align)`. For example, a likelihood that conditions on `slope` can be combined with a prior where the slope is called `beta` with `joint(lik, prior, beta="slope")`.

### Rationale

Composition is written as an expression so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and `*` returns a first-class joint so the result composes further (`D4 – Closed system of objects under operations`). Deriving the name deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`), while `with_name` lets the user impose grouping where it carries intent. Realignment is an exact rename: `with_names` returns the same law under new field names, so `joint` connects mismatched factors without altering their joint law (`D1 – Mathematical fidelity`).

## V.6 — `marginal` and `factor`

### Contract

Two operations read the parts of a structured or factored distribution. (The third access form, the `d[field]` **view**, is not an operation: it returns a correlation-preserving reference into its parent rather than a standalone object, and its contract is fixed with the factored distributions.)

- `marginal(d, field)` returns the **detached** marginal of a field or field group, a standalone `Distribution` with no reference back to `d`. It dispatches on `SupportsMarginals`. When the capability is absent, or the path has no exact route within it, it falls back to a Monte Carlo approximation through `_sample`, projecting draws onto the field and returning an empirical marginal, and it raises a capability error when the distribution cannot sample either.
- `factor(d, name)` returns a building-block **factor** of a joint, keyed by factor name, either a `Distribution` or a `ConditionalDistribution` for a dependent edge. It requires `SupportsFactors`, raising a capability error on a distribution that exposes no factors.

Both return a **tracked term** whose `provenance` records the access and its source distribution.

### Rationale

`marginal` and `factor` are the two query directions of the field and factor interfaces fixed with the factored distributions: `marginal` reads a field's law detached from its joint, and `factor` reads a construction block. Exposing them as named operations rather than as indexing is what separates the detached query from the correlation-preserving `d[field]` view (`D1 – Mathematical fidelity`). Dispatching `marginal` on `SupportsMarginals` opens the detached query to any distribution that knows its marginals, factored or not. Gating `factor` on `SupportsFactors` keeps factor access honest, since only a distribution actually built from named parts can answer it (`D3 – Capability-based operations`).

## V.7 — Batched operations

### Contract

Every operation lifts to a `Batch` by mapping over its elements, which is the workflow-function sweep applied to the operation itself.
- `sample` and `log_prob` over a `DistributionBatch` return a `RecordBatch` and a batched array of densities, with the batch axes preserved.
- A moment over a `DistributionBatch` returns a batch of the corresponding values.
- When the elements are array-backed, the map is a single vectorized call rather than a Python loop, which is the batched-backend optimization.
- An operation applied to a batch whose elements lack the required capability raises the same capability error a single element would.

### Rationale

A batched operation is not a new operation but the workflow-function sweep applied to an existing one, so a `Batch` supports exactly the operations its elements do (`D3 – Capability-based operations`). When the elements are array-backed the sweep is a single vectorized call, which preserves differentiability (`D6 – Differentiability where possible`).
