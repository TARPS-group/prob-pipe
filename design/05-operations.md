# Part V — Operations

Parts II–IV fixed the *shared abstractions*, the *values and distributions*, and the *functions* that act on them. Part V fixes the **operations**: what one is (V.0), and for each operation its operands, its result, and its routes.

**Conventions.** The user-facing names are the bare operations, such as `sample` and `mean`. The implementer counterparts are `_`-prefixed, such as `_sample`, and for conditional distributions the prefix `_conditional` is added as well.

## V.0 — The operation model

### Contract

An **operation** is a `Function` that declares three things and is realized by a fourth:

1. its **operand roles**: which parameters are the mathematical inputs, and the kinds each accepts, named by spec class because an operation is generic over a kind, so that `mean` takes any distribution whatever its event;
2. its **applicability conditions**: what makes a call well-formed at the level of declarations;
3. a **result rule**: computing the result's complete spec before anything runs, from the operands' actual specs and from any parameter that selects rather than supplies, such as `marginal`'s field path;

and it is realized either by registered **routes** or, for a *derived* operation, by an identity defined in terms of other operations.

**The call sequence.** Every call runs the steps below in order, and a failure ends the call at its step. Two failures are named, since a caller can act on them: `ApplicabilityError`, for a call malformed at the level of declarations, and `ResolutionError`, for a well-formed call that no route can realize. `op.check(...)` runs steps 1 to 6 without executing and reports what it found: which routes are feasible, what each infeasible route is missing, and which would be selected. It is the registry `check` (II.7) at the operation.

1. **Bind.** The arguments bind to the operation's signature. The controls are read from their own namespace (IV.4), and they are the same for every operation, supplied by the framework rather than by its author: `raw`, `method` for route selection, the PRNG `key`, and the sampling controls a Monte Carlo route consumes. *Requires:* every argument binds to a parameter, and every control is one the framework defines. *On failure:* a binding error naming the parameter.
2. **Admit.** Each operand is normalized to a tracked term: a tracked term as it is, a raw host wrapped into its kind (IV.2), and a backend object converted through its registered converter (III.14). Every kind's term is a `TrackedTerm`, so a bare object is refused here whatever methods it carries. *Requires:* the term's spec class is a kind the role accepts, or the term is a `Distribution` or a `Batch` over such a kind, which step 3 lifts. *On failure:* `ApplicabilityError`, naming the role, the kinds it accepts, and what arrived.
3. **Lift.** Where a role holds a `Distribution` or a `Batch` over the kind it accepts, the engine (IV.2) applies the remaining steps per draw or per element, co-sampling by root ancestor, and assembles the results into an empirical distribution or a batch; the rule that realizes the lift is selected through the evaluation registry (V.1). *Requires:* a lifted distribution samples, and a batch's elements support the operation. *On failure:* the `ResolutionError` a single application would raise (V.10).
4. **Validate applicability.** The operation's declared conditions are checked against specs, structure, and control values alone, for example that `evaluate`'s operand conforms to the map's input or that `sample`'s declaration is concrete. *Requires:* each condition holds. *On failure:* `ApplicabilityError`, naming the condition.
5. **Plan.** The result rule computes the result declaration from what is static under compilation: operand specs and structure, declared capabilities, which paths an argument supplied, and the control values, and never traced array data. The declaration is therefore `jit`-safe and cannot depend on what the computation produces. *Requires:* the rule yields a complete spec, its symbolic dimensions bound by unification (II.1). *On failure:* `ApplicabilityError`, naming the dimensions in conflict or left free.
6. **Resolve.** Each route's `check(call, result)` runs on the declarations. A capability route is feasible when its named operand satisfies its protocol, and protocol membership is the whole test, since step 2 settled the kind; a structural route when the declared structure supports it; a registry route when its registry's own check finds a method; a fallback route on its stated domain. Feasible routes rank by fidelity, then specificity, then registration order, with no within-tier priority at the operation, and `method=` names one outright. *Requires:* a feasible route, or the named one. *On failure:* `ResolutionError`, naming each route and what it was missing.
7. **Execute.** The selected route's `execute(call, result)` runs over the operands' raw types and is the only step that reads values. An implementation tests nothing about its operand, since the sequence has established a tracked term of the role's kind that satisfies the protocol the route named. *Requires:* the route completes. *On failure:* the route's own error propagates, as for any numerical method.
8. **Return.** The result crosses the kind-directed wrap (IV.2), where a tracked term keeps its kind under the operation's fresh identity, a raw host wraps into its own kind, and a backend distribution converts through its converter (III.14). It is coerced to the planned declaration and validated against it, and it receives identity: an auto-derived name, and a `provenance` recording the operation, its parent descriptors, the selected route with its fidelity, and the resolved controls, the key included. Every result is tracked, a numeric summary included, so a density returns as a `NumericArray`. With `raw=True` the identity is not constructed and the result returns detached (II.4). *Requires:* the result satisfies the planned declaration. *On failure:* a wrong kind and a schema mismatch raise distinct errors, and either is a route defect rather than a caller error.

**Routes.** A route is one way to realize an operation. It has the interface of a dispatch method (II.7), that is, `check`, `execute`, and a **fidelity**, but is bound to a call rather than to argument types, and it is registered against the operation it realizes by upward registration, as for any registry. Routes come from four sources, and an operation may carry any combination:

| route source | the implementation comes from | example |
|---|---|---|
| **structural** | the operands' declared structure | `condition_on` currying a given slot; `joint` composing two factors |
| **capability** | a `Supports*` protocol on a named operand | `_sample` on the subject of `sample` |
| **registry** | a registered method selected by dispatch | the inference methods behind Bayes' rule; the evaluation rules behind `evaluate` |
| **fallback** | a generic scheme applicable to a stated domain | Monte Carlo through a sampling operand |

Where a route dispatches on a capability it names the operand it dispatches on, so an operation needs no single distinguished subject: `joint` has two peer operands, and `evaluate` resolves on the map and the operand together. The split between a call's `specs` and its `operands` is what keeps feasibility cheap and planning static, since `check` reads the declarations alone.

```python
@dataclass(frozen=True)
class BoundCall:                   # one call, after binding and normalization
    operation: Function
    operands:  Mapping[str, Any]           # by role name, each normalized to its kind (II.1)
    controls:  Mapping[str, Any]           # the resolved controls: the key, the sample count, …
    @property
    def specs(self) -> Mapping[str, TermSpec]: ...
    # each operand's spec: what `check` and the result rule read, never the values

class OperationRoute(Protocol):    # one interface; the helpers below are construction shorthand
    name:     str
    source:   RouteSource
    fidelity: Fidelity
    def check(self, call: BoundCall, result: TermSpec) -> MethodInfo: ...
    def execute(self, call: BoundCall, result: TermSpec) -> Any: ...

mean.capability_route("closed_form", operand="d", protocol=SupportsMean, method="_mean",
                      fidelity=Fidelity.EXACT)
mean.fallback_route("monte_carlo", check=_can_sample, execute=_mc_mean,
                    fidelity=Fidelity.SAMPLE)
condition_on.structural_route("curry", check=_can_curry, execute=_curry, fidelity=Fidelity.EXACT)
condition_on.registry_route("bayes", registry=inference_method_registry)
```

A registry route delegates selection to a registry of II.7, so the inference methods and the evaluation rules keep their own priorities and feasibility probes rather than having them restated here.

**Primitive versus derived operations.** A **primitive** operation states its own contract and carries its own routes. A **derived** operation is instead defined by an identity over other operations; `mixture`, for example, is the detached `marginal` of a composed joint. That identity is what the operation *means*, so its result rule, feasibility, and failure modes follow from the operations it is defined by. The identity is itself a route, the one always available, and a derived operation may carry routes that realize it **directly** besides: a Gaussian mixture computed in closed form need not compose and then marginalize. Direct routes rank above the identity by fidelity and specificity in the usual way, and the identity is the floor (IV.2), as the sampling lift is under `evaluate`. Either way the operation adds its own outer provenance record, and the selected route records which path ran.

**Declaring an operation.** An operation is created with the `@operation` decorator, which takes the result rule and registers the operation. The decorated function's parameters are its operands together with anything the result rule reads, for example `sample_shape` or an alignment mapping; the controls are never authored (step 1), so **a parameter is authored exactly when the result rule reads it, and everything else is a control**. A primitive operation's body is empty, and a derived operation's body is its identity. For a `ConditionalDistribution` operand the framework also adds `given=` as the fused conditional path of III.9, which planning reads.

```python
def _mean_result(d: DistributionSpec) -> TermSpec: ...
# planning reads the operands' specs and the declaration inputs, and nothing traced (step 5)

@operation(result=_mean_result)
def mean(d: Distribution): ...              # primitive: no body; the registered routes implement it

@operation(result=_mixture_result)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    return marginal(K * mixing, ...)        # derived: the body is the identity, its floor route
```

**Four raw mechanisms, one meaning.** *Raw* always means the representation layer, detached from the workflow; the mechanisms differ only in where they act. Route implementations are *written* over raw types (`T`); `apply` (III.3) *evaluates* a wrapped callable with no lifting, tracking, or provenance; `raw=True` *skips* the identity an operation would mint; and `raw()` (II.4) *detaches* an existing term.

**Randomness.** An operation's PRNG `key` is a control under the key rule of IV.3, caller-owned when supplied and a workflow-owned random event when omitted, whether the operation's contract is a random draw or a deterministic quantity whose resolved route samples.

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
    fidelity: Fidelity           # the shared scale of II.7
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

Defining an operation by its operands, a result rule, and a set of routes is what keeps the vocabulary closed: adding an operation cannot add a mechanism, and adding an *implementation* is registering a route rather than amending a contract (`D2 – Generality first`). Separating what a call means from how it is realized makes totality a property of the routes available at call time rather than of the operation, so one rule says when a call can fail instead of per-operation exceptions. Planning the declaration before execution, from what is static under compilation, is `D5 – Explicit, carried structure` made checkable: the result's structure is known before the work starts and verified after it. Capability dispatch remains `D3 – Capability-based operations`, now as one route source among four rather than as the definition of an operation. Deriving the control block rather than authoring it per operation is `C1 – Uniform interface to functions, distributions, and values` at the operation layer: `raw` and `method` mean the same thing everywhere because no author writes them. Defining a derived operation by an identity gives its behavior a single definition (`D6 – Single source of truth`), and that every operation returns another tracked term is `D4 – Closed system of objects under operations`. The sequence restates the boundary principles at the operation layer: admission is `B1 – Either presentation in`, execution over `T` is `B2 – Representations only inside`, and the return step is `B3 – Tracked forms out by default`; planning and lifting are what an operation promises beyond them.

## V.1 — `evaluate`

### Contract

`evaluate(f, v)` applies the map `f` to the operand `v`: for a value it returns `f(v)`; for a distribution, the law of `f(X)` for `X ~ v`, the pushforward `f♯v`; for a batch, the elementwise result. It is computed by the rule the registry selects: a plain call for a value, the selected rule for a distribution, the sweep for a batch. The direct call `f(v)` resolves through the same registry, so it and `evaluate(f, v)` take the same route and agree in value and in selected rule; `evaluate` is the operation form, adding `method=` selection and its own outer record. The map `f` is a `Function`; `LinOp` is its linear subtype, and invertible maps claim `SupportsInverse`. `v`'s schema must conform to the map's input, unifying any symbolic dimensions, with the result carrying the map's output declaration under the resulting substitution. For a `LinOp`, `A @ v` is operator notation for `evaluate(A, v)`. `@` composes two operators and evaluates the map on any other operand.

A map with more than one parameter is evaluated over exactly one of them, with `fixed_args` supplying the rest by name, as in `evaluate(predict, posterior, fixed_args={"x": X_new})`. `fixed_args` must leave exactly one parameter free, the one the operand maps to; leaving two unbound is an error. The registry keys on the map's own type and the fixed arguments pass through to each rule's feasibility check, so binding side arguments neither wraps the map nor loses its registered identity, and the single mapping parameter keeps the operation's controls separate from the map's arguments.

**The evaluation registry.** For a distribution or batch operand the call dispatches through a `BinaryDispatchRegistry` keyed on the map's and the operand's types, whose methods are **evaluation rules**. The registry is defined beside the engine (Part IV) and consulted by every single-distribution and batched application, the operation and the direct call alike. The rules, in selection order (II.7):
- **Closed-form rules** return an exact parametric result. For example, `A @ d` for a Gaussian `d` is again Gaussian, with mean `A @ mean(d)` and covariance `A Σ Aᵀ` built lazily through the operator algebra.
- **Change of variables** applies when the map is invertible and carries the Jacobian claim (`is_invertible` and `SupportsLogDetJacobian`), returning a transformed distribution whose `log_prob` is exact via the log-determinant of the Jacobian.
- **The sampling lift** is the rule registered at the generic pair and always applies: draws from `d` are pushed through the map, returning an empirical distribution over the outputs, with the sample count and PRNG key exposed as controls. It is the registry's floor, and the route every plain callable takes.
- **The elementwise sweep** is the batch counterpart: the rule at the generic pair for a batch operand. A fused batched implementation, such as an operator's matrix–matrix routine or a single vectorized call over array-backed elements, registers above it.

The selected rule records its name and fidelity in the result's provenance from either entry point; `evaluate`'s record adds its own outer operation entry, the one respect in which the two records differ. Rules need not be exact: an approximate scheme, for example quadrature or an unscented transform, registers at its recorded fidelity, above the lift.

**Linear maps push moments exactly.** Whatever rule realizes `A @ d`, the result's `mean` and `cov` delegate exactly whenever `d` supports them, since `E[A X] = A E[X]` and `Cov(A X) = A Cov(X) Aᵀ`. An approximate linear pushforward therefore still reports exact first and second moments.

Applied to a `ConditionalDistribution`, evaluation acts on the event side, giving the kernel `s ↦ f♯K(s, ·)` with the same given spec.

### Rationale

`evaluate` is `C4 – Function lifting` in operation form: applying a map is one act whatever the operand kind, and substituting a multiplicity for a value leaves that act well-defined either way, a distribution giving the pushforward law and a batch the elementwise result, and this operation returns it directly. Dispatching over pairs of map and operand types realizes `C3 – Computational detail hidden by default, available on demand`, since a pair with a known closed form or a fused batched routine gets it automatically, while every other pair still resolves through the floors, sampling for a distribution and the sweep for a batch. Recording the producing rule makes the approximation explicit (`D1 – Mathematical fidelity`), registration grows the exact set without changing call sites (`D2 – Generality first`), and the result is a tracked term that composes further (`D4 – Closed system of objects under operations`).

## V.2 — `inverse` and `log_det_jacobian`

### Contract

Two operations read a map's inverse structure.
- `inverse(f)` returns the inverse map as a `Function`: `inverse(f)(y)` is the preimage of `y` under `f`. Its capability route is read through `is_invertible`, and a map that fails that check resolves to nothing and raises. The result is itself invertible, with `f` as its inverse, and it carries the Jacobian claim whenever `f` does.
- `log_det_jacobian(f, x)` returns the log-determinant of the Jacobian of `f` at `x`. Its capability route is `SupportsLogDetJacobian`. The reverse direction needs no second operation, since `log_det_jacobian(inverse(f), y) = −log_det_jacobian(f, inverse(f)(y))`.
- Each carries a capability route and no other today. A numerical route, for example root finding for an inverse, may register later at its recorded fidelity, widening what resolves without changing either contract (V.0).
- `inverse(f)`'s name is derived from `f`'s.

### Rationale

Reparameterization moves in both directions between a constrained and an unconstrained space, so the inverse must be reachable from user code, and the operation form keeps the capability's implementer methods private, per the V.0 convention. Returning the inverse as a `Function` keeps the system closed (`D4 – Closed system of objects under operations`): the inverse evaluates, composes, and pushes forward like any map.

## V.3 — `sample`

### Contract

`sample(d, key=None, sample_shape=(), raw=False)` draws from a distribution.
- With `sample_shape=()` it returns a single draw, tracked, at the kind the declaration names (III.7); a non-empty `sample_shape` prepends batch axes and returns the tracked batch form of that kind, the leading dimensions on a level named `sample`. `sample(..., raw=True)` is the draw detached: the kind's raw value for a single draw, which for a `Distribution`-valued draw is the law itself, and the storage view (II.5) for a batch:

| declared kind | one draw | tracked batch | `raw=True`, batched |
|---|---|---|---|
| `NumericArraySpec` | `NumericArray` | `NumericArrayBatch` | the stacked array, batch axes leading |
| `RecordSpec` | `Record` | `RecordBatch` / `NumericRecordBatch` | the nested mapping of raw columns |
| `DistributionSpec` | `Distribution` | `DistributionBatch` | an object array of the drawn laws |
| `ConditionalDistributionSpec` | `ConditionalDistribution` | `ConditionalDistributionBatch` | an object array of the drawn kernels |
| `FunctionSpec` | `Function` | `FunctionBatch` | an object array of the drawn callables |
| `OpaqueSpec` | `Opaque` | `OpaqueBatch` | an object array of the drawn objects |
- Sampling requires a concrete declaration and raises with the free dimensions named; in the fused conditional path, the given value binds them first.
- Under a non-empty `sample_shape` the key (IV.3) splits by draw index, so the draws are jointly independent and reproducible together.

### Rationale

Every draw is reproducible from its record (IV.3), which is `C6 – Traceable and reproducible workflows`. Returning every draw as the tracked term of its declared kind serves `C1 – Uniform interface to functions, distributions, and values` without making the kinds uniform: what is the same across laws is that a draw is tracked and its type is fixed by the declaration, not that every draw is a `Record`. The `raw` opt-out is `B3 – Tracked forms out by default` at the sampling boundary: the wrapped, tracked draw is the default, and the bare value is an explicit ask.

## V.4 — `log_prob` and `unnormalized_log_prob`

### Contract

`log_prob(d, value)` returns the log-density of `value` under `d`. The value may be a `Record` matching the event schema, or a bare array for a scalar law.
- `log_prob` requires `SupportsLogProb` and returns the *normalized* log-density.
- `unnormalized_log_prob` requires only `SupportsUnnormalizedLogProb` and returns the log-density up to an additive constant, which is what inference against an unnormalized target needs.
- A scored value binds any symbolic event dimensions for that call only, so one law scores datasets of different sizes.

### Rationale

Splitting `log_prob` from `unnormalized_log_prob` makes each capability claim only what it provides (`D1 – Mathematical fidelity`): a distribution that knows its normalizing constant offers the true density, while one that does not still serves inference, which needs the density only up to a constant.

## V.5 — Distribution functionals: `mean`, `variance`, `cov`, `quantile`, `expectation`

### Contract

The moment operations summarize a distribution by a deterministic value.
- `mean(d)` and `variance(d)` return an event-typed value, that is, a value shaped like a draw. Neither is restricted to numeric draws, but each requires the event type to support it: a random function has a mean function and a pointwise variance function, while a random measure has a mean, which is the marginalized law, but in general no event-typed variance. The result is wrapped at the law's declared event kind — a `Record` whose schema matches the distribution's for a record-drawing law, and a term for a term-drawing one, so a random function's mean is a `Function`.
- `cov(d)` requires a numeric draw and returns a covariance operator over the *flattened* draw, a `(vector_size, vector_size)` `LinOp`, since covariance couples distinct coordinates. Its input and output schemas are both the distribution's numeric event spec, so it applies directly to draws.
- `quantile(d, q)` requires a numeric draw. It takes a level `q ∈ [0, 1]` or array of such levels and returns the quantile for each, computed per coordinate for a multivariate draw. Its result declaration is event-kind-directed like any other: a single level returns the event's own kind, for example a `NumericArray` for an array-drawing law, and a plural `q` adds a level over those, giving the matching batch. Whether `q` holds one level or several is known before execution, so planning reads it (V.0).
- `expectation(d, f)` returns `E[f(X)]`, shaped by the output of `f`, for any event type `f` accepts. The result is wrapped at the kind `f`'s output declaration names, a `Record` for the usual record output.

Each carries a capability route on its matching protocol, `SupportsMean` for `mean` and so on, feasible whatever the event type, and a Monte Carlo fallback route feasible through `SupportsSampling`, with the sample count and PRNG key as controls (V.0). A distribution that claims neither resolves to nothing and raises. The fallback route is defined per event kind. A numeric event averages draws coordinatewise. For a function-valued event the fallback returns a lazy function: its mean at a point is the average of the sampled callables there, and its variance the pointwise sample variance. A measure-valued event's mean is the finite mixture of the sampled draws, the Monte Carlo estimate of the mean measure. `cov` and `quantile` keep numeric-only fallback routes, the sample covariance returned as a `DenseLinOp` and the per-coordinate empirical quantiles. `expectation` averages the array outputs of `f` and so applies to any event type that samples.

### Rationale

A mean is defined whenever draws can be averaged, which is coordinate-wise for arrays, pointwise for functions, and set-wise for measures, while an event-typed variance requires the second moment to be a value of the event type, which fails for a general random measure: `A ↦ Var(ξ(A))` is additive only when disjoint regions are uncorrelated, never for a random probability measure, whose fixed total mass forces negative correlation, so a random measure's second-moment structure is a covariance over pairs of sets, the analog of `cov`. Gating `mean` and `variance` by capability rather than by a numeric event is therefore `D1 – Mathematical fidelity`, as is keeping `cov` and `quantile` numeric-only, and `cov` returns a flat operator because it couples coordinates the event's field structure keeps separate. The closed-form-or-Monte-Carlo split realizes `C3 – Computational detail hidden by default, available on demand`: a distribution that can give an exact moment does, and one that can only sample still returns an approximate one.

## V.6 — `condition_on`

### Contract

`condition_on(d, given)` fixes some fields of a distribution or conditional distribution and returns the resulting distribution.

**The `given` argument.** `given` is field-keyed: a `Record`, or a mapping from field paths to values, with every value conforming to the spec at its path, all of them unified together (II.1). Each key must name either a *given* slot, that is, a name in the `given_spec` or a path into a structured slot, or a *produced* field, that is, a path in the event schema; any other key is an error. A key may also name an interior path, in which case its value is a sub-record checked against the sub-schema. Binding part of a structured slot is defined as restructure-then-bind (II.6): the bound part is promoted and bound, the residual slot remains, and a group emptied by the binding dissolves. Conditioning is stated entirely in terms of fields, and factors never appear in the call: the derived factor graph is read only to decide which case below applies and to carry it out.

**The routes.** `condition_on` resolves across three of the four route sources (V.0): structural routes that curry a given slot or slice the factor graph, a capability route on `SupportsConditioning`, which a `ConditionalDistribution` always satisfies since `_condition_on` is its required primitive, and a registry route through the inference methods for Bayes' rule. Which is feasible follows from each conditioned field's position in the factor graph, read from declarations:
- **Exogenous given, so curry.** Binding a slot that the object conditions on but does not produce returns a smaller `ConditionalDistribution`, or an ordinary `Distribution` once all given slots are bound. This is exact and involves no inference. For example, binding a regression model's covariates curries it toward the data-ready likelihood.
- **Upstream or independent produced field, so exact slice.** Conditioning on a produced field that no remaining factor depends on through an unconditioned path returns the exact conditional by slicing the factor graph, again with no inference. The slice is exact only when the field's own factor is single-field or itself implements the conditioning capability; a multi-field factor that cannot condition internally falls to the Bayes' rule case. For example, conditioning a Gaussian prior on one of its own fields slices, while an atomic empirical factor over two fields cannot be sliced on one of them.
- **Downstream data, so Bayes' rule.** Conditioning on a field that downstream factors depend on is an application of Bayes' rule: the conditional is proportional to the joint density at the fixed value, and is generally not available in closed form. It is delegated to an inference algorithm registered for the model, and only the factored classes support it. For example, conditioning on observed responses downstream of the coefficients is the Bayes' rule case. On an atomic distribution, conditioning on produced fields uses its own `SupportsConditioning` when implemented, as a `MultivariateNormal` does exactly, and raises otherwise.

When `given` names several fields, the cases combine: the exact bindings (curry and slice) are applied first, and Bayes' rule runs on what remains. Field classification is computed once, on the graph with every conditioned field marked, so the outcome does not depend on the order the fields are listed. Conditioning on a produced field does not require the given fields to be bound first. The result stays conditional on the unmet givens, with the produced-field conditioning applied within each slice of the given, so the result curries like any other `ConditionalDistribution` and the two orders agree: conditioning on a produced field and then binding the given yields the same distribution as binding the given first. When that produced-field conditioning requires Bayes' rule, the resulting `ConditionalDistribution` may realize the inference lazily, once its given is bound, or through a method that supports amortization.

`condition_on` always binds the supplied value as the field's fixed value, whatever the value's type; the mixture `∫ K(s, ·) μ(ds)` over a mixing distribution is requested explicitly through the separate `mixture` operation.

**The inference-method registry.** The Bayes' rule case is dispatched through the **inference-method registry**, a `UnaryDispatchRegistry` keyed on the model's type whose methods are inference algorithms such as MCMC or variational families.

**Fidelity.** An inference method's declared fidelity (II.7) is what its result targets: `exact` for a method that targets the posterior itself, as MCMC does in the limit of its run, and `approximate` for one that targets a surrogate, as a variational family does.

### Rationale

A single operation covers binding, slicing, and Bayes' rule because all three are the same mathematical act of conditioning, and they differ only in whether the conditioned field is exogenous, upstream, or downstream in the factor graph. Collapsing them into one operation keeps the user interface small (`C1 – Uniform interface to functions, distributions, and values`), while the derived graph decides the algorithm (`C3 – Computational detail hidden by default, available on demand`).

## V.7 — `joint`

### Contract

`A * B` (III.12) is the composition operator; its result's `provenance` records `*` as the operation, and its `name` follows the canonical-order rule fixed there.

**The realigning `joint` form.** A limitation of `*` is that it requires a producer's field names to match the names its consumer conditions on. This motivates the `joint(A, B, **align)` op, which realigns fields first and then composes exactly as `*`, so it is equivalent to `A * B.with_path_names(**align)`. For example, a likelihood that conditions on `slope` can be combined with a prior where the slope is called `beta` with `joint(lik, prior, beta="slope")`.

### Rationale

Realignment is an exact rename: `with_path_names` returns the same law under new field names, so `joint` connects mismatched factors without altering their joint law (`D1 – Mathematical fidelity`).

### Open points

- *The `align` contract.* `align` pairs are `with_path_names` pairs, path-valued targets included (II.6), so realignment can promote a nested field to a slot-matchable name. The remaining details, such as freshness and injectivity requirements on the new names, are deferred: the question is subtle, and implementation experience should inform the decision.

## V.8 — `marginal` and `factor`

### Contract

Two operations read the parts of a structured or factored distribution; the view `d[field]` is not an operation (III.7).

- `marginal(d, field)` returns the **detached** marginal of a field or field group, a standalone `Distribution` with no reference back to `d`. It carries a capability route on `SupportsMarginals` and a Monte Carlo fallback route through `_sample`, projecting draws onto the field and returning an empirical marginal. When the capability is absent, or the path has no exact route within it, the fallback resolves, and a distribution that cannot sample either resolves to nothing and raises.
- `factor(d, name)` returns a building-block **factor** of a joint, keyed by factor name, either a `Distribution` or a `ConditionalDistribution` for a dependent edge. Its capability route is `SupportsFactors`, so a distribution that exposes no factors resolves to nothing and raises.

### Rationale

Exposing them as named operations rather than as indexing is what separates the detached query from the correlation-preserving `d[field]` view (`D1 – Mathematical fidelity`). Dispatching `marginal` on `SupportsMarginals` opens the detached query to any distribution that knows its marginals, factored or not. Gating `factor` on `SupportsFactors` ensures that only a distribution actually built from named parts offers it (`D3 – Capability-based operations`).

## V.9 — `mixture`

### Contract

`mixture(K, mixing)` returns the mixture `μK = ∫ K(s, ·) μ(ds)`: the law of `T` for `S ~ mixing` and `T ~ K(S, ·)`, where the mixing distribution's produced slots meet the kernel's given slots by name.

`mixture` is a **derived** operation (V.0), defined by the identity `mixture(K, mixing) = marginal(K * mixing, ...)` onto the kernel's produced slots, detached. That identity is its floor route, so its feasibility and its failure modes are those of `*` and `marginal` — a call resolves at worst when the composition is well-formed and the marginal has a route, and a mixing distribution that cannot be sampled and admits no exact marginal raises there rather than here. A direct route may realize a case in one step instead, as the Gaussian algebra does, ranking above the identity by fidelity and specificity. Slot matching, spec unification, and unmet givens therefore behave as composition (III.12) and `marginal` fix them, with nothing separately defined: a given slot the mixing distribution does not meet stays unmet, and the result is then a `ConditionalDistribution` over the unmet givens. Exactness follows the same route: a finite mixing distribution yields the explicit `MixtureDistribution`, a closed family such as the Gaussian algebra stays closed, and otherwise the result is the Monte Carlo empirical marginal through ancestral sampling.

### Rationale

Keeping the integral out of `condition_on` keeps conditioning single-valued, since a supplied value always binds and `μK` is always asked for by name, so neither call has a data-dependent meaning (`C1 – Uniform interface to functions, distributions, and values`). The name is the result's mathematical name: `μK` is the mixture of the kernel family with mixing distribution `μ`, which covers predictive, compound, and state-propagation uses without privileging one (`C5 – Naming for unambiguous meaning`, `D1 – Mathematical fidelity`). Defining the operation as the marginal of the composed joint adds no second semantics: one identity ties it to operations already fixed, so every behavior has a single source (`D6 – Single source of truth`).

## V.10 — Batched operations

### Contract

Every operation lifts to a `Batch` by mapping over its elements, which is the elementwise sweep applied to the operation itself, and an operation that mints a level names it after itself (II.5).
- `sample` over a `DistributionBatch` returns a **nested** batch, the outer level ranging over the laws and the inner over each law's draws: `sample(d_batch, key, sample_shape=(S,))` has `axis_groups` `(*d_batch.axis_groups, (S,))` and appends an inner draw level named `sample`, so iterating it visits one law's batch of draws at a time, which is a `RecordBatch` for a record-drawing law and the batch form of the declared kind otherwise. `log_prob` maps elementwise to the batched densities, with the batch axes preserved.
- A moment over a `DistributionBatch` returns a batch of the corresponding values, such as a `LinOpBatch` for `cov`. A multi-level query nests the same way: `quantile(d_batch, q)` keeps the laws on the outer level and adds an inner level named `quantile` for the levels of `q`.
- **Alignment.** A binary operation matches the operands' levels **by name**: a level in both must have broadcast-compatible shapes, with size-1 broadcasting; a level in only one operand broadcasts across the other; and an outer product is requested by explicit reshaping rather than implied. Because every level is named, there is no positional fallback, and two levels meant to correspond under different names are lined up by renaming one with `with_level_names` first, exactly as `joint` realigns fields for composition. So a flat batch of values on a `laws` level scores against the `laws` level of a nested sampling result. `given=` accepts a `RecordBatch` and yields the `DistributionBatch` of conditioned laws.
- Two operands are exempt from batch lifting: the factors of composition (`*` and `joint`) and the map operand of `evaluate`, which are consumed as objects rather than swept.
- Batched application resolves through the evaluation-rule registry (V.1).
- An operation applied to a batch whose elements lack the required capability raises the `ResolutionError` a single element would.

### Rationale

A batched operation is not a new operation but the elementwise sweep applied to an existing one, so a `Batch` supports exactly the operations its elements do (`D3 – Capability-based operations`). Nesting the laws level over the draws level keeps two different multiplicities distinct in the result itself (`D1 – Mathematical fidelity`), and matching levels by name rather than by position carries `C5 – Naming for unambiguous meaning` onto the multiplicity axis, making the rename the single tool for lining levels up, as field renaming is for composition. When the elements are array-backed and claim differentiation, the sweep is a single vectorized call that preserves the claim, which is `D3 – Capability-based operations` applied to the batch through its elements.
