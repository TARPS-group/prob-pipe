# Part VI — Operations

Parts II–V fixed the *shared abstractions*, the *term kinds*, the *distributions*, and the *functions* that act on them. Part VI fixes the **operations**: what one is (VI.0), and for each operation its operands, its result, and its routes.

**Conventions.** The user-facing names are the bare operations, such as `sample` and `mean`. The implementer counterparts are `_`-prefixed, such as `_sample`, and for conditional distributions the prefix `_conditional` is added as well.

## VI.0 — The operation model

### Contract

An **operation** is a `Function` with additional capabilities to provide a uniform interface over the function-level implementations: one call, `mean(d)`, whatever the kind of `d` and whichever implementation realizes it. Each capability answers one way in which the implementations differ, and the engine's stack reads each at its step (V.1):

1. **Operand roles.** The implementations differ in the kinds they handle, so a parameter cannot be typed by a concrete spec: `mean` takes any distribution whatever its event. A role names the kind a parameter accepts by spec class, and admission checks it (V.4).
2. **Applicability conditions and a result rule.** The result depends on which operand arrived, so its declaration cannot be fixed at construction: the mean of a law is a value of the law's event kind. A result rule derives the result's declaration, and the checks it must defer to return, per call from the operands' specs and from any parameter that selects rather than supplies, such as `marginal`'s field path, and the applicability conditions state what makes a call well-formed at the level of declarations before the rule runs (V.6).
3. **Routes.** One interface fronts many implementations, so an operation is realized by a set of routes the engine selects among (V.7 and V.9), or, for a *derived* operation, by an identity in terms of other operations.

A result rule declares the result's components independently of its label. It carries the operand's interface through when the operation preserves it, transforms that interface when the mathematics changes it, and otherwise declares a fixed component named for the operation, such as `log_prob` for a log-density. It never copies an object label into a component, and a type hole it leaves is filled before any law is constructed (II.2).

Since the engine runs the stack, `op(...)` is a tracked term under fresh, derived identity whose spec satisfies the completed declaration, `op.with_options(raw=True)(...)` is that result detached (II.4), and `op.check(...)` is the engine's `check`.

**Routes.** The implementations behind one operation come from different places, and a route is the one form they all take. It has the interface of a dispatch method (II.7), that is, `check`, `execute`, and a **fidelity**, but is bound to a call rather than to argument types, and it is registered against the operation it realizes by upward registration, as for any registry. Routes come from four sources, and an operation may carry any combination:

| route source | the implementation comes from | example |
|---|---|---|
| **structural** | the operands' declared structure | `condition_on` currying a given slot; `joint` composing two factors |
| **capability** | a `Supports*` protocol on a named operand | `_sample` on the subject of `sample` |
| **registry** | a registered method selected by dispatch | the inference methods behind Bayes' rule; the evaluation rules behind `evaluate` |
| **fallback** | a generic scheme applicable to a stated domain | Monte Carlo through a sampling operand |

Where a route dispatches on a capability it names the operand it dispatches on. Its check reads the call's specs, the parameters the result rule reads, the controls, and declared representation metadata such as a factorization or a guarded capability (III.8), and it never evaluates the body. A capability route requires protocol membership plus the capability's guard, and membership alone where the capability is total on its domain. A structural route checks declared structure, a registry route delegates its probe to that registry, and a fallback checks its stated domain and assumptions. Missing declarations are reported as unresolved (V.1), and execution never catches a failure and tries another route.

Routes use II.7's fidelity, specificity, and registration order, with no configurable within-tier priority at the operation itself. Registry routes retain their registry's priorities. Their local fidelity is the selected method's; derived routes likewise use the selected chain's guarantee rather than a fixed exact tag on the wrapper.

```python
@dataclass(frozen=True)
class BoundCall:                   # one call, after binding and normalization
    operation: Function
    operands:  Mapping[str, Any]           # wrapped terms, or planned conversions, by role
    controls:  Mapping[str, Any]           # the resolved controls: the sample count, the method, …
    @property
    def specs(self) -> Mapping[str, TermSpec]: ...
    # each operand's spec, or the target spec a planned conversion promises
    @property
    def declarations(self) -> Mapping[str, Any]: ...
    # the parameters the result rule reads and the representation metadata planning admitted (V.6)

class OperationRoute(Protocol):    # one interface; the helpers below are construction shorthand
    name:     str
    source:   RouteSource
    fidelity: Fidelity | None     # None for a delegated route until its plan selects a method
    def check(self, call: BoundCall, result: OutputSpec | None) -> MethodInfo: ...
    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any: ...

mean.capability_route("closed_form", operand="d", protocol=SupportsMean, method="_mean",
                      fidelity=Fidelity.EXACT)
mean.fallback_route("monte_carlo", check=_can_sample, execute=_mc_mean,
                    fidelity=Fidelity.APPROXIMATE)
marginal.capability_route("exact", operand="d", protocol=SupportsMarginals,
                          method="_marginal", check=_can_marginalize_path,
                          fidelity=Fidelity.EXACT)
# An omitted capability guard means membership suffices on the declared domain.
condition_on.structural_route("curry", check=_can_curry, execute=_curry, fidelity=Fidelity.EXACT)
condition_on.registry_route("bayes", registry=inference_method_registry)
```

**Primitive versus derived operations.** Some operations mean something in terms of others, and the interface keeps that definition visible. A **primitive** operation states its own contract and carries its own routes. A **derived** operation is instead defined by an identity over other operations; `mixture`, for example, is the reconstructed kernel output projected from a composed joint (VI.9). That identity is what the operation *means*, so its result rule, feasibility, and failure modes follow from the operations it is defined by. The identity is itself a route on the domain where its constituent operations are available, and a derived operation may carry routes that realize it **directly** besides: a Gaussian mixture computed in closed form need not compose and then marginalize. Direct routes rank above the identity when their fidelity and specificity warrant it; the identity is a fallback on its own domain (V.7), as the sampling lift is under `evaluate`. Either way the operation adds its own outer provenance record, and the selected route records which path ran.

**Declaring an operation.** `@operation` registers an operation and takes its result rule. The authored signature contains the operands and the parameters the result rule reads: `log_prob`'s scored value, `expectation`'s function, `quantile`'s levels, `sample_shape`, and alignment mappings are arguments even when their values do not determine the output shape. The result rule reads their specs and any declared static information it requires. Controls choose realization, numerical budgets, execution, or presentation and are set through `with_options` (V.2). A primitive body is empty; a derived body states its identity. For a conditional operand the framework adds `given=` as the argument of the fused path (III.9).

```python
def _mean_result(d: DistributionSpec) -> OutputSpec: ...
# planning reads the operands' specs and the declaration inputs, and nothing traced (V.6)

@operation(result=_mean_result)
def mean(d: Distribution): ...              # primitive: no body; the registered routes implement it

@operation(result=_mixture_result)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    return evaluate(_kernel_output_projection(K), K * mixing)  # the identity in VI.9
```

**Four raw mechanisms, one meaning.** *Raw* always means the representation layer, detached from the workflow; the mechanisms differ only in where they act. Route implementations are *written* over raw types (`T`); `apply` (III.3) *evaluates* a wrapped callable with no lifting, tracking, or provenance; `with_options(raw=True)` *skips* the identity an operation would mint; and `raw()` (II.4) *detaches* an existing term.

**Randomness.** An operation takes no key: each draw it causes is a workflow-owned random event (V.8), whether the operation's contract is a random draw or a deterministic quantity whose route samples.

**Listing the operations and their routes.** The operations are themselves a registry, so the vocabulary is discoverable the way every other extensible set in the library is (II.7). `operation_registry.list()` returns one summary per operation, covering each operand with the kinds it accepts, whether the operation is primitive or derived, and each route with its source, fidelity, and requirement; `describe()` renders the same content as text, for one operation or for all of them. The registry satisfies `SupportsRegistryCataloging`, so it appears in the catalog beside the other registries, and a user asking what ProbPipe can do has one place to look.

```python
class RouteSource(Enum):        # where a route's implementation comes from
    STRUCTURAL = "structural"
    CAPABILITY = "capability"
    REGISTRY   = "registry"
    FALLBACK   = "fallback"

@dataclass(frozen=True)
class OperandSummary:
    name:     str                          # the role, as the signature spells it
    accepts:  tuple[type[TermSpec], ...]   # the kinds it takes, each named by its spec class;
                                           #   empty when the parameter selects rather than supplies,
                                           #   as a field path or an alignment mapping does
    planning: bool                         # whether the result rule reads it

@dataclass(frozen=True)
class RouteSummary:
    name:     str
    source:   RouteSource
    fidelity: Fidelity | None    # None until a delegated route is resolved (II.7)
    requires: tuple[type, ...]   # the protocols a capability route needs; empty otherwise
    condition: str               # the feasibility condition in words, for the routes that types cannot state

@dataclass(frozen=True)
class OperationSummary(EntrySummary):
    operands:   tuple[OperandSummary, ...]   # in signature order
    is_derived: bool
    identity:   str | None                   # the defining identity, when derived
    routes:     tuple[RouteSummary, ...]     # in selection order

class OperationRegistry(SupportsRegistryCataloging):
    def register(self, op: Function) -> None: ...
    def list(self) -> list[OperationSummary]: ...
    def describe(self, name: str | None = None) -> str: ...
    def __getitem__(self, name: str) -> Function: ...

operation_registry: OperationRegistry     # the global instance
```

### Rationale

Defining an operation by its operands, a result rule, and a set of routes is what keeps the vocabulary closed: adding an operation cannot add a mechanism, and adding an *implementation* is registering a route rather than amending a contract (`D2 – Generality first`). Separating what a call means from how it is realized makes totality a property of the routes available at call time rather than of the operation, so one rule says when a call can fail instead of per-operation exceptions. Capability dispatch remains `D3 – Capability-based operations`, now as one route source among four rather than as the definition of an operation. Deriving the control block rather than authoring it per operation is `C1 – Uniform interface to functions, distributions, and values` at the operation layer: `raw` and `method` mean the same thing everywhere because no author writes them. Defining a derived operation by an identity gives its behavior a single definition, and leaving the stack to the engine (V.1) so that an operation differs only in its declarations gives the call path one as well (`D6 – Single source of truth`): a plain function and an operation fail at the same steps with the same errors. That every operation returns another tracked term is `D4 – Closed system of objects under operations`, and a route written over `T` is `B2 – Representations only inside`, with both boundaries left to the engine.

## VI.1 — `evaluate`

### Contract

`evaluate(f, v)` applies the map `f` to the operand `v`: for a value it returns `f(v)`; for a distribution, the law of `f(X)` for `X ~ v`, the pushforward `f♯v`; for a batch, the elementwise result. It is the operation form of the engine's resolve step (V.7): the direct call `f(v)` takes the same route, and `evaluate` adds its own outer provenance entry. Both forms support the same controls through `with_options` (V.2). The map `f` is a `Function`; `LinOp` is its linear subtype, and invertible maps claim `SupportsInverse`. `v`'s schema must conform to the map's input, unifying any symbolic dimensions, with the result carrying the map's output declaration under the resulting substitution. For a `LinOp`, `A @ v` is operator notation for `evaluate(A, v)`. `@` composes two operators and evaluates the map on any other operand.

Explicit evaluation binds the operand to the selected parameter; its component name is not required to match that parameter (V.5).

A map with more than one parameter is evaluated over exactly one of them, with `fixed_args` supplying the rest by name, as in `evaluate(predict, posterior, fixed_args={"x": X_new})`; leaving two parameters unbound is an error. The registry keys on the map's own type and the fixed arguments pass through to each evaluation rule's feasibility check, so binding side arguments neither wraps the map nor loses its registered identity, and the single mapping parameter keeps the operation's controls separate from the map's arguments.

**Linear maps push moments exactly.** The exact lazy linear-pushforward route (VII.4) delegates `mean` and `cov` whenever `d` supports them, since `E[A X] = A E[X]` and `Cov(A X) = A Cov(X) Aᵀ`. If the caller explicitly selects an empirical approximation, its moments are those of that empirical law; exact target moments cannot be substituted for the moments of the law returned (II.7).

Applied to a `ConditionalDistribution`, evaluation acts on the event side, giving the kernel `s ↦ f♯K(s, ·)` with the same given spec.

### Rationale

`evaluate` is `C4 – Function lifting` in operation form: applying a map is one act whatever the operand kind, and substituting a multiplicity for a value leaves that act well-defined either way, a distribution giving the pushforward law and a batch the elementwise result, and this operation returns it directly. The result is a tracked term that composes further (`D4 – Closed system of objects under operations`).

## VI.2 — `inverse` and `log_det_jacobian`

### Contract

Two operations read a map's inverse structure.
- `inverse(f)` returns the inverse map as a `Function`: `inverse(f)(y)` is the preimage of `y` under `f`. Its capability route is read through `is_invertible`; an unavailable inverse raises `ResolutionError`, while known noninvertibility is a mathematical domain error (II.7). The result is itself invertible, with `f` as its inverse, and it carries the Jacobian claim whenever `f` does.
- `log_det_jacobian(f, x)` returns the log-determinant of the Jacobian of `f` at `x`. Its capability route is `SupportsLogDetJacobian`. The reverse direction needs no second operation, since `log_det_jacobian(inverse(f), y) = −log_det_jacobian(f, inverse(f)(y))`.
- Each carries a capability route and no other today. A numerical route, for example root finding for an inverse, may register later at its recorded fidelity, widening what resolves without changing either contract (VI.0).
- `inverse(f)`'s name is derived from `f`'s, and its result label is independent of any slot name. Its output declaration comes from `f`'s input slots: one slot is returned whole under that slot's name, and several form an exposed record, which every inverse route preserves.

### Rationale

Reparameterization moves in both directions between a constrained and an unconstrained space, so the inverse must be available to user code, and the operation form keeps the capability's implementer methods private, per this part's naming convention. Returning the inverse as a `Function` keeps the system closed (`D4 – Closed system of objects under operations`): the inverse evaluates, composes, and pushes forward like any map.

## VI.3 — `sample`

### Contract

`sample(d, sample_shape=())` draws from a distribution.

- With `sample_shape=()` it returns a single draw, tracked, at the kind the declaration names (III.7); a non-empty `sample_shape` prepends batch axes and returns the tracked batch form of that kind, the leading dimensions on a level named `sample`. `sample.with_options(raw=True)(...)` is the draw detached: the kind's raw value for a single draw, which for a `Distribution`-valued draw is the law itself, and the storage view (II.5) for a batch:

| declared kind | one draw | tracked batch | `raw=True`, batched |
|---|---|---|---|
| `NumericArraySpec` | `NumericArray` | `NumericArrayBatch` | the stacked array, batch axes leading |
| `RecordSpec` | `Record` | `RecordBatch` / `NumericRecordBatch` | the nested mapping of raw columns |
| `DistributionSpec` | `Distribution` | `DistributionBatch` | an object array of the drawn laws |
| `ConditionalDistributionSpec` | `ConditionalDistribution` | `ConditionalDistributionBatch` | an object array of the drawn kernels |
| `FunctionSpec` | `Function` | `FunctionBatch` | an object array of the drawn callables |
| `OpaqueSpec` | `Opaque` | `OpaqueBatch` | an object array of the drawn objects |

- Sampling reads the returned kind from `event_spec.spec`, not the component count or object name (II.2). An array and a one-field record remain distinct under every `sample_shape`.
- Sampling requires a concrete declaration and raises with the free dimensions named; in the fused conditional path, the given value binds them first.
- Under a non-empty `sample_shape` the key (V.8) splits by draw index, so the draws are jointly independent and reproducible together.

### Rationale

Every draw is reproducible from its record (V.8), which is `C6 – Traceable and reproducible workflows`. Returning every draw as the tracked term of its declared kind serves `C1 – Uniform interface to functions, distributions, and values` without making the kinds uniform: what is the same across laws is that a draw is tracked and its type is fixed by the declaration, not that every draw is a `Record`. The `raw` opt-out is `B3 – Tracked forms out by default` at the sampling boundary: the wrapped, tracked draw is the default, and the bare value is an explicit request.

## VI.4 — `log_prob`, `unnormalized_log_prob`, and the density variants

### Contract

`log_prob(d, value)` returns the log-density of `value` under `d`. The value conforms to `d.event_spec.spec`, including its packaging: an array-valued event takes an array and a one-field record event takes a record or matching mapping. A joint reconstructs each factor's event before delegating (IV.2).

- `log_prob` requires `SupportsLogProb` and returns the *normalized* log-density.
- `unnormalized_log_prob` requires only `SupportsUnnormalizedLogProb` and returns the log-density up to an additive constant, which is what inference against an unnormalized target needs.
- A scored value binds any symbolic event dimensions for that call only, so a polymorphic law scores datasets of different sizes.
- `prob` and `unnormalized_prob` are derived operations (VI.0), defined by the identities `prob = exp ∘ log_prob` and `unnormalized_prob = exp ∘ unnormalized_log_prob`; a family with a stable density may register a direct route above the identity.
- `random_log_prob(M)` and `random_unnormalized_log_prob(M)` take a random measure `M` (VII.5) and return the law of `x ↦ log D(x)` for `D ~ M`, a `RandomFunction`, through capability routes on `SupportsRandomLogProb` and `SupportsRandomUnnormalizedLogProb` (III.8). The density at a point is that random function called at the point (VII.5), so neither takes a value.

### Rationale

Splitting `log_prob` from `unnormalized_log_prob` makes each capability claim only what it provides (`D1 – Mathematical fidelity`): a distribution that knows its normalizing constant offers the true density, while one that does not still serves inference, which needs the density only up to a constant. Deriving `prob` from `log_prob` gives the density one definition, which a family overrides only where it has a better one (`D6 – Single source of truth`).

## VI.5 — Distribution functionals: `mean`, `variance`, `cov`, `quantile`, `expectation`

### Contract

The moment operations summarize a distribution by a deterministic value.
- `mean(d)` and `variance(d)` return an event-typed value, that is, a value shaped like a draw. Neither is restricted to numeric draws, but each requires the event type to support it: a random function has a mean function and a pointwise variance function, while a random measure has a mean, which is the marginalized law, but in general no event-typed variance. The result is wrapped at the law's declared event kind: a `Record` whose schema matches the distribution's for a record-drawing law, and a term for a term-drawing one, so a random function's mean is a `Function`.
- `cov(d)` requires a numeric draw and returns a covariance operator over the *flattened* draw, a `(vector_size, vector_size)` `LinOp`, since covariance couples distinct coordinates. Its input term spec and `output_spec.spec` are the distribution's numeric event term spec; application checks these term specs without matching output component names to the input parameter (III.4, V.5).
- `quantile(d, q)` requires a numeric draw. It takes a level `q ∈ [0, 1]` or array of such levels and returns the quantile for each, computed per coordinate for a multivariate draw. Its result declaration is event-kind-directed like any other: a single level returns the event's own kind, for example a `NumericArray` for an array-drawing law, and a plural `q` adds a level over those, giving the matching batch. Whether `q` holds one level or several is known before execution, so planning reads it (V.6).
- `expectation(d, f)` returns `E[f(X)]`, shaped by the output of `f`, for any event type `f` accepts. The result is wrapped at the kind `f`'s output declaration names, a `Record` for the usual record output.

A moment of the event's kind keeps the event's components and packaging and derives only the result term specs, support included, so the mean of a Bernoulli event is supported on the unit interval. `expectation` takes its declaration from the integrand's output declaration, and `cov` takes the operator codomain from the event declaration. A higher-order result, such as a function-valued mean, keeps its inner declaration separate from the operation's own output component.

Each carries a guarded capability route on its matching protocol, `SupportsMean` for `mean` and so on, and a Monte Carlo fallback on the event kinds where the required averaging is defined. Sampling availability alone does not establish moment existence: methods state their assumptions, and known undefinedness is handled as in II.7. The approximation budget is a control (V.2). A distribution that claims neither raises `ResolutionError`. The fallback route is defined per event kind. A numeric event averages draws coordinatewise. For a function-valued event the fallback returns a lazy function: its mean at a point is the average of the sampled callables there, and its variance the pointwise sample variance. A measure-valued event's mean is the finite mixture of the sampled draws, which is the Monte Carlo estimate of the mean measure. `cov` and `quantile` keep numeric-only fallback routes: the sample covariance, returned as a `DenseLinOp`, and the per-coordinate empirical quantiles. `expectation`'s numeric fallback averages the outputs of `f` and reconstructs their declared numeric kind, including record packaging. It requires sampling and integrability of those outputs, not a numeric input event.

### Rationale

A mean is defined whenever draws can be averaged, which is coordinatewise for arrays, pointwise for functions, and setwise for measures, while an event-typed variance requires the second moment to be a value of the event type, which fails for a general random measure. `A ↦ Var(ξ(A))` is additive only when disjoint regions are uncorrelated, and never for a random probability measure, whose fixed total mass forces negative correlation, so a random measure's second-moment structure is a covariance over pairs of sets, which is the analog of `cov`. Gating `mean` and `variance` by capability rather than by a numeric event is therefore `D1 – Mathematical fidelity`, as is keeping `cov` and `quantile` numeric-only, and `cov` returns a flat operator because it couples coordinates the event's field structure keeps separate. The closed-form-or-Monte-Carlo split realizes `C3 – Computational detail hidden by default, available on demand`: a distribution that can give an exact moment does, and a sampling fallback returns an approximate one under its stated assumptions.

## VI.6 — `condition_on`

### Contract

`condition_on(d, given)` fixes some fields of a distribution or conditional distribution and returns the resulting distribution.

**The `given` argument.** `given` is field-keyed: a `Record`, or a mapping from field paths to values, with every value conforming to the spec at its path, all of them unified together (II.1). Each key must name either a *given* slot, that is, a name in the `given_spec` or a path into a structured slot, or a *produced* field, that is, a path in the event schema; any other key is an error. A key may also name an interior path, in which case its value is a sub-record checked against the sub-schema. Binding part of a structured slot is defined as restructure-then-bind (II.6): the bound part is promoted and bound, the residual slot remains, and a group emptied by the binding dissolves. Conditioning is stated entirely in terms of fields, and factors never appear in the call: the derived factor graph is read only to decide which case below applies and to carry it out.

**The routes.** `condition_on` resolves across three of the four route sources (VI.0): structural routes that curry a given slot or slice the factor graph, a capability route on `SupportsConditioning`, which a `ConditionalDistribution` always satisfies since `_condition_on` is its required primitive, and a registry route through the inference methods for Bayes' rule. Binding a given slot applies the kernel, and binding a produced field conditions the law; the factor graph helps select a route for the second, never changes what it means. Feasibility is checked from the declarations and available capabilities:

- **Exogenous given, so curry.** Binding a slot that the object conditions on but does not produce returns a smaller `ConditionalDistribution`, or an ordinary `Distribution` once all given slots are bound. This is exact and involves no inference. For example, binding a regression model's covariates curries it toward the data-ready likelihood.
- **Produced field with an exact slice.** Binding leaves a conditional that can be assembled from available normalized factors and exact local conditioning operations. For example, in `p(y | beta) p(beta)`, fixing `beta` leaves the existing kernel `p(y | beta)`. A multi-field factor must support the required internal conditioning; its fields are never assumed independent. Upstream or independent fields commonly admit this route, but graph position alone does not establish feasibility.
- **Produced field requiring Bayes' rule.** Binding leaves a likelihood contribution involving unconditioned variables, requiring an exact conditioning method or inference. For example, fixing `y` in `p(y | beta) p(beta)` leaves the likelihood `p(y | beta)` over the unknown `beta`, even though `y` has no downstream dependents. An available guarded `SupportsConditioning` route may compute the conditional exactly, as a `MultivariateNormal` does. Otherwise the operation considers inference methods registered for the model's representation (below), including atomic representations, and raises `ResolutionError` when none applies.

When `given` names several fields, the cases combine: the exact bindings, curry and slice, are applied first, and Bayes' rule runs on what remains. Field classification is computed once, on the graph with every conditioned field marked, so the outcome does not depend on the order the fields are listed. Conditioning on a produced field does not require the given fields to be bound first. The result stays conditional on the unmet givens, with the produced-field conditioning applied within each slice of the given, so the result curries like any other `ConditionalDistribution`, and in the exact cases the two orders agree: conditioning on a produced field and then binding the given yields the same distribution as binding the given first. An approximate route records its assumptions and fidelity instead of promising equality in law, and a conditional at a value is the version the selected route defines, since almost-everywhere uniqueness does not fix it pointwise. When the produced-field conditioning requires Bayes' rule, the resulting `ConditionalDistribution` may realize the inference lazily, once its given is bound, or through a method that supports amortization.

`condition_on` always binds the supplied value as the field's fixed value, whatever the value's type; the mixture `∫ K(s, ·) μ(ds)` over a mixing distribution is requested explicitly through the separate `mixture` operation.

**The inference-method registry.** The Bayes' rule case is dispatched through the **inference-method registry**, a `UnaryDispatchRegistry` keyed on the model's representation type, whose methods are inference algorithms such as MCMC or variational families. A method's feasibility check reads the interface it requires, a target density, a backend program handle, or an exposed factorization, so a factorization is required only by the methods that inspect it. Method parameters such as warmup and approximation budgets are controls set through `with_options` (V.2); the plan may read them when they determine a result's batch shape or other static structure.

**Fidelity.** Inference uses the shared contract of II.7. A finite MCMC posterior is approximate; the chain's invariant target and convergence assumptions are method guarantees. Exact conditioning routes return a representation of the conditional law itself. The selected route's local fidelity and target are recorded in provenance.

### Rationale

Applying a kernel to its given and conditioning a law on a produced field are one operation because both fix named quantities (`C1 – Uniform interface to functions, distributions, and values`), and the distinction between them is kept (`D1 – Mathematical fidelity`) while dispatch chooses the available method (`C3 – Computational detail hidden by default, available on demand`, `D3 – Capability-based operations`).

## VI.7 — `joint`

### Contract

`A * B` (IV.2) is the composition operator; its result's `provenance` records `*` as the operation, and its `name` follows the canonical-order rule fixed there.

**The realigning `joint` form.** A limitation of `*` is that it requires a producer's field names to match the names its consumer conditions on. This motivates the `joint(A, B, **align)` operation, which realigns fields first and then composes as `*` does, so it is equivalent to `A * B.with_path_names(**align)`. For example, a likelihood that conditions on `slope` can be combined with a prior where the slope is called `beta` with `joint(lik, prior, beta="slope")`.

### Rationale

Realignment is an exact rename: `with_path_names` returns the same law under new field names, so `joint` connects mismatched factors without altering their joint law (`D1 – Mathematical fidelity`).

### Open points

- *The `align` contract.* `align` pairs are `with_path_names` pairs, path-valued targets included (II.6), so realignment can promote a nested field to a slot-matchable name. The remaining details, such as freshness and injectivity requirements on the new names, are deferred: the question is subtle, and implementation experience should inform the decision.

## VI.8 — `marginal` and `factor`

### Contract

Two operations read the parts of a structured or factored distribution; the view `d[field]` is not an operation (III.7).

- `marginal(d, field)` returns the **detached** marginal of a field or field group, a standalone `Distribution` with no reference back to `d`. It carries a capability route on `SupportsMarginals` guarded at the requested path (IV.1), and a Monte Carlo fallback route through `_sample`, projecting draws onto the field and returning an empirical marginal. When the capability is absent, or the path has no exact route within it, the fallback resolves, and a distribution that cannot sample either raises `ResolutionError`.
- `factor(d, name)` returns a building-block **factor** of a joint, keyed by factor name, either a `Distribution` or a `ConditionalDistribution` for a dependent edge. Its capability route is `SupportsFactors`, so a distribution that exposes no factors raises `ResolutionError`.

### Rationale

Exposing them as named operations rather than as indexing is what separates the detached query from the correlation-preserving `d[field]` view (`D1 – Mathematical fidelity`). Dispatching `marginal` on `SupportsMarginals` opens the detached query to any distribution that knows its marginals, factored or not. Gating `factor` on `SupportsFactors` ensures that only a distribution built from named parts offers it (`D3 – Capability-based operations`).

## VI.9 — `mixture`

### Contract

`mixture(K, mixing)` returns the mixture `μK = ∫ K(s, ·) μ(ds)`: the law of `T` for `S ~ mixing` and `T ~ K(S, ·)`, where the mixing distribution's produced slots meet the kernel's given slots by name.

`mixture` is a **derived** operation (VI.0). Let `J = K * mixing` and let `r_K` extract the kernel's produced components from a joint draw and reconstruct its original event using `K.event_spec` (II.2). Its identity is `mixture(K, mixing) = evaluate(r_K, J)`, returned as a standalone law. The internal projection may first marginalize onto those components when an exact marginal route exists. Reconstruction is essential: an array-valued kernel stays array-valued, a one-field record stays a record, and a record exposed as one named component is unpacked back to that record. The result carries the kernel's event declaration unchanged.

The identity's feasibility and fidelity follow from composition and the selected projection route. A direct route may realize the same law more efficiently. Finite mixing admits an explicit `MixtureDistribution`, closed Gaussian cases stay Gaussian, and the generic sampling route gives an empirical approximation. Unmet givens remain conditional, as composition specifies; the identity applies within each given slice.

### Rationale

Leaving the integral out of `condition_on` keeps conditioning single-valued, since a supplied value always binds and `μK` is always asked for by name, so neither call has a data-dependent meaning (`C1 – Uniform interface to functions, distributions, and values`). The name is the result's mathematical name: `μK` is the mixture of the kernel family with mixing distribution `μ`, which covers predictive, compound, and state-propagation uses without privileging one (`C5 – Naming for unambiguous meaning`, `D1 – Mathematical fidelity`). Defining the operation as the reconstructed kernel output of the composed joint adds no second semantics: one identity ties it to operations already fixed, so every behavior has a single source (`D6 – Single source of truth`).

## VI.10 — `convert`

### Contract

`convert(d, target)` returns a distribution of the requested class or satisfying the requested capability protocol (III.8). Its registry route uses the converter plan and event-preservation contract of IV.3. `with_options(method=..., min_fidelity=...)` controls selection, and provenance records the selected converter's local fidelity. A source already satisfying the target needs no numerical conversion and returns under fresh identity. Entry normalization plans the same conversion; execution constructs it and records it by the same contract (V.4, V.9).

### Rationale

Exposing conversion as an operation gives a change of representation the same record as any other result (`C6 – Traceable and reproducible workflows`), and routing it through the registry keeps the set of convertible pairs open to registration (`D2 – Generality first`).

## VI.11 — Batched operations

### Contract

Every operation lifts to a `Batch` by mapping over its elements, which is the elementwise sweep applied to the operation itself, and an operation that mints a level names it after itself (II.5).
- `sample` over a `DistributionBatch` returns a **nested** batch, the outer level ranging over the laws and the inner over each law's draws: `sample(d_batch, sample_shape=(S,))` has `axis_groups` `(*d_batch.axis_groups, (S,))` and appends an inner draw level named `sample`, so iterating it visits one law's batch of draws at a time, which is a `RecordBatch` for a record-drawing law and the batch form of the declared kind otherwise. `log_prob` maps elementwise to the batched densities, with the batch axes preserved.
- A moment over a `DistributionBatch` returns a batch of the corresponding values, such as a `LinOpBatch` for `cov`. A multi-level query nests the same way: `quantile(d_batch, q)` keeps the laws on the outer level and adds an inner level named `quantile` for the levels of `q`.
- **Alignment.** A binary operation matches the operands' levels **by name**: a level in both must have broadcast-compatible shapes, with size-1 broadcasting; a level in only one operand broadcasts across the other; and an outer product is requested by explicit reshaping rather than implied. Because every level is named, there is no positional fallback, and two levels meant to correspond under different names are aligned by renaming one with `with_level_names` first, as `joint` realigns fields for composition. So a flat batch of values on a `laws` level scores against the `laws` level of a nested sampling result. `given=` accepts a `RecordBatch` and yields the `DistributionBatch` of conditioned laws.
- Two operands are exempt from batch lifting: the factors of composition (`*` and `joint`) and the map operand of `evaluate`, which are consumed as objects rather than swept.
- Batched application resolves through the evaluation-rule registry (V.7).
- An operation applied to a batch whose elements lack the required capability raises the `ResolutionError` a single element would.

### Rationale

A batched operation is not a new operation but the elementwise sweep applied to an existing one, so a `Batch` supports exactly the operations its elements do (`D3 – Capability-based operations`). Nesting the laws level over the draws level keeps two different multiplicities distinct in the result itself (`D1 – Mathematical fidelity`), and matching levels by name rather than by position carries `C5 – Naming for unambiguous meaning` onto the multiplicity axis, making the rename the one way to align levels, as field renaming is for composition. When the elements are array-backed and claim differentiation, the sweep is a single vectorized call that preserves the claim, which is `D3 – Capability-based operations` applied to the batch through its elements.
