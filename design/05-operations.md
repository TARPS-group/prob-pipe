# Part V — Operations

Parts II–IV fixed the *shared abstractions*, the *values and distributions*, and the *functions* that act on them. Part V fixes the **operations**. Every operation is a `Function`, so it inherits lifting, provenance, dispatch, and orchestration from the function layer. This part adds only what is specific to each operation: its signature, its argument and return types and shapes, the choice between an exact and a default algorithm, its error behavior, and how its result is wrapped and tracked. Every operation is also capability-dispatched, so it applies to any object that implements the matching capability, and closed, so it returns another **tracked term**.

**Conventions.** The user-facing names are the bare operations (`sample`, `log_prob`, `mean`, …). The implementer counterparts are `_`-prefixed (`_sample`, `_log_prob`, …) and, for conditional distributions, the prefix `_conditional` is added as well.

## V.0 — The operation model

### Contract

An **operation** is a `Function` that declares three things and is realized by a fourth:

1. its **operand roles** — which arguments are the mathematical inputs, and the kind each expects;
2. its **applicability conditions** — what makes a call well-formed at the level of declarations;
3. a **result rule** — computing the result's complete spec before anything runs;

and it is realized either by registered **routes** or, for a *derived* operation, by an identity defined in terms of those other operations. An operation adds operands, a result rule, and routes to the dispatch machinery.

**The four laws.**

- **Planning.** The result declaration is computed before any route runs. Its inputs are exactly what is static under compilation — operand specs and structure, declared capabilities, which paths an argument supplied, and the control values — never traced array data, so a result declaration is `jit`-safe by construction and cannot depend on what the computation produces. The executed result must satisfy the declaration it was planned against.
- **Resolution.** After applicability validation, a call succeeds when at least one route is **feasible**. Routes rank by fidelity, then specificity, then registration order, and the selected route records its name and fidelity in the result's provenance.
- **Boundary.** `op(...)` returns a tracked term under fresh, derived identity. `op(..., raw=True)` **is** that result detached (II.4), computed without constructing the identity it would discard.
- **Lifting.** Substituting a `Distribution` or a `Batch` for a value operand lifts through the Part IV engine — the pushforward for a distribution, the elementwise sweep for a batch, co-sampling by root ancestor for several at once.

**Routes.** A route is one way to realize an operation. Each declares a `check`, which reports feasibility for a bound call without computing the result, and an `execute`; each carries a **fidelity**, and an approximate route records the fidelity it achieved. Routes come from four sources, and an operation may carry any combination:

| route source | the implementation comes from | example |
|---|---|---|
| **structural** | the operands' declared structure | `condition_on` currying a given slot; `joint` composing two factors |
| **capability** | a `Supports*` protocol on a named operand | `_sample` on the subject of `sample` |
| **registry** | a registered method selected by dispatch | the inference methods behind Bayes' rule; the evaluation rules behind `evaluate` |
| **fallback** | a generic scheme applicable to a stated domain | Monte Carlo through a sampling operand |

Where a route dispatches on a capability it names the operand it dispatches on, so an operation needs no single distinguished subject: `joint` has two peer operands, and `evaluate` resolves on the map and the operand together.

**Failure modes.** A malformed call fails **applicability**, before planning. A call that is well-formed and mathematically meaningful but that no registered route can realize fails **resolution**, naming the requirements each route was missing. A resolved call may still fail in **execution**, as any numerical method may.

**Checking feasible routes.** `op.check(...)` runs applicability and route feasibility for a bound call and reports what would happen: which routes are available, which are not and what each is missing, and which would be selected. It is similar to the kind of `check` that the converters and inference methods expose.

**Primitive versus derived operations.** A **primitive** operation states its own contract and carries its own routes. A **derived** operation is instead defined by an identity over other operations — `mixture` is the detached `marginal` of a composed joint — and that identity is what the operation *means*, so its result rule, its feasibility, and its failure modes follow from the operations it is defined by. The identity is itself a route, the one always available, and a derived operation may carry routes that realize it **directly** besides: a Gaussian mixture computed in closed form need not compose and then marginalize. Direct routes rank above the identity by fidelity and specificity in the usual way, and the identity is the floor that answers when none of them applies — the same shape the sampling lift has under `evaluate`. Either way the operation adds its own outer provenance record, and the selected route records which path ran.

**Operand roles are declared by kind, not by spec.** An operation's operands are named and typed, but not by an `InputSpec` (II.2), which maps names to *concrete* term specs — the slots of one particular map-like term. An operation is generic over a kind: `mean` takes any distribution whatever its event, and `DistributionSpec` cannot express that, since it carries an event declaration of its own. Some authored parameters are not terms at all — `marginal`'s field path, `factor`'s factor name, `joint`'s alignment mapping — and a declaration that types terms has nothing to say about them. So an operand role declares the kind it accepts — named by the spec *class*, since a spec's class is what fixes a term's kind, and a class needs no event declaration of its own — together with what a route will require of it, and the concrete typing happens per call, when the result rule reads the operands' actual specs. That is why planning is a rule rather than a stored declaration.

**Declaring an operation.** An operation is created using the `@operation` decorator, which takes the result rule and registers the operation. The decorated function's parameters are its operands and declaration inputs; the controls are implemented universally by the framework. For a **primitive** operation the body is empty because all that is required is a declaration, with the routes providing implementations. For a **derived** operation, the function body implements its identity in terms of other operations, and direct routes may be registered besides where realizing it in one step is more efficient than composing.

```python
def _mean_result(d: DistributionSpec) -> TermSpec: ...
# planning reads the operands' specs and the declaration inputs, and nothing traced (above)

@operation(result=_mean_result)
def mean(d: Distribution): ...              # primitive: no body; the registered routes implement it

@operation(result=_mixture_result)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    return marginal(K * mixing, ...)        # derived: the body is the identity, its floor route
```

**Registering routes.** A route is registered against the operation it realizes, at import, by whichever layer owns the implementation — the same upward registration the evaluation rules and converters use, so a route reaches an operation without the operation importing its provider. Every route has one shape, and the four helpers differ only in how one is built:

```python
class OperationRoute(Protocol):    # one shape; the helpers below are construction sugar
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

A capability route names the operand it dispatches on, which is why no operation needs a single distinguished subject. A registry route delegates selection to a registry of II.7, so the inference methods and the evaluation rules keep their own priorities and feasibility probes rather than having them restated here.

**How a call resolves.** Every call runs the same steps, and each is a contract stated above: bind and normalize the arguments, either presentation accepted (II.1); validate applicability; plan the result declaration; collect the feasible routes by calling each `check`; select one; execute it; and cross the wrap boundary. Selection ranks by **fidelity** first, then **specificity** — the closest-in-method-resolution-order tie-break the dispatch registries already use — and then registration order, so exactness is never silently traded for specificity, while `method=` names a route outright and skips the ranking.

**What a caller writes, and what the framework adds.** An operation's authored parameters are its operands together with anything the result rule reads — `sample_shape`, an alignment mapping, the plurality of a level. Its **controls** are the same for every operation and are supplied by the framework rather than by each author: `raw`, `method` for route selection, the PRNG `key`, and the sampling controls a Monte Carlo route consumes. The rule is one line: **a parameter is authored exactly when the result rule reads it; everything else is a control.** So an operation's signature is its operand list plus one uniform control block, and no operation may spell a control differently or omit one.

**The wrap boundary.** The boundary is kind-directed and wraps only the untracked.

- *A tracked term keeps its kind.* A `Record`, `Distribution`, `ConditionalDistribution`, `Function`, `NumericArray`, or `Opaque` an operation returns is never re-wrapped and never buried inside another kind. Keeping the kind is not keeping the identity: by the boundary law the result carries the operation's fresh name and provenance, while the term the route produced is left untouched.
- *A raw host wraps into its own kind*, by the kind-directed wrap of the call boundary (IV.2): a bare array becomes a `NumericArray`, a raw mapping a structured `Record`, a raw callable a `Function`, and any other raw value an `Opaque`, the ordered fallback. A backend distribution enters through its converter, so a concrete family such as a `Normal` can be written in plain arrays with no hand-written schema.
- *The planned declaration is enforced.* The result is coerced to the declaration planning computed and validated against it; a wrong kind and a schema mismatch raise distinct errors.
- A `ConditionalDistribution` adds the `given=` fused paths over its `_conditional_*` methods.

**Two levels.** A route's implementation acts on the implementer draw type `T` (III.7); the user-facing operation plans, resolves, wraps, and attaches identity, all at no cost to the implementer.

**Four raw mechanisms, one meaning.** *Raw* always means the representation layer, detached from the workflow; the mechanisms differ only in where they act. Route implementations are *written* over raw types (`T`); `apply` (III.3) *evaluates* a wrapped callable with no lifting, tracking, or provenance; `raw=True` *elides* the identity an operation would mint; and `raw()` (II.4) *detaches* an existing term.

**Randomness.** No operation draws from ambient random state, and no operation carries a seed. An explicit PRNG `key` is caller-owned: the draw is consumed as given and reproducible from the key alone. With the key omitted, the draw is a **workflow-owned random event** (IV.3): its key derives structurally from the enclosing workflow scope, and a bare call outside any scope gets its own ephemeral scope. An operation whose contract is a deterministic quantity but whose resolved route samples takes its key the same way. The resolved key is recorded in `provenance`.

**Output identity.** Every tracked term an operation mints is fully specified, never left implicit: its spec is the planned declaration, its `provenance` records the operation, its parent descriptors, and the selected route, and its `name` is auto-derived and marked `name_is_auto`. Every result is tracked, a numeric summary included — a density returns as a `NumericArray`, and the detached value is one `raw=True` away.

**Listing available routes.** The operations are themselves a registry, so the vocabulary is discoverable the way every other extensible set in the library is (II.7). `operation_registry.list()` returns one summary per operation — each operand with the kinds it accepts, whether the operation is primitive or derived, and each route with its source, fidelity, and requirement — and `describe()` renders the same content as text, for one operation or for all of them. The registry satisfies `SupportsRegistryCataloging`, so it appears in the catalog beside the converters, evaluation rules, inference methods, and bijector factories, and a user asking what ProbPipe can do has one place to look.

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

Defining an operation by its operands, a result rule, and a set of routes is what keeps the vocabulary closed: adding an operation cannot add a mechanism, and adding an *implementation* is registering a route rather than amending a contract (`D2 – Generality first`). Separating what a call means from how it is realized is why the old question "is this operation total?" was unanswerable — totality is a property of the routes available at call time, not of the operation — and stating resolution once gives the user one rule for when a call can fail instead of per-operation fine print. Planning the declaration before execution, from what is static under compilation, is `D5 – Explicit, carried structure` made checkable: the result's structure is known before the work starts and verified after it. Capability dispatch remains `D3 – Capability-based operations`, now as one route source among four rather than as the definition of an operation. Deriving the control block rather than authoring it per operation is `C1 – Uniform interface to functions, distributions, and values` at the operation layer: `raw` and `method` mean the same thing everywhere because no author writes them. Defining a derived operation by an identity keeps its behavior in one place (`D6 – Single source of truth`), and that every operation returns another tracked term is `D4 – Closed system of objects under operations`.

## V.1 — `evaluate`

### Contract

`evaluate(f, v)` applies the map `f` to the operand `v`: for a value it returns `f(v)`; for a distribution, the law of `f(X)` for `X ~ v`, the pushforward `f♯v`; for a batch, the elementwise result. It is computed by the rule the registry selects, the sampling lift being the rule at the generic pair: a plain call for a value, the selected rule for a distribution, the sweep for a batch. The direct call `f(v)` resolves through the same registry, so it and `evaluate(f, v)` take the same route and agree in value; `evaluate` is the operation form, adding `method=` selection and its own outer record, as stated below. The map `f` may be a plain callable or a `Function`; `LinOp` is its linear subtype, and invertible maps claim `SupportsInverse`. `v`'s schema must conform to the map's input, unifying any symbolic dimensions, with the result carrying the map's output declaration under the resulting substitution. For a `LinOp`, `A @ v` is operator sugar for `evaluate(A, v)`. As with `*`, the meanings coexist by operand type: `@` composes two operators and evaluates the map on any other operand.

A map with more than one parameter is evaluated over exactly one of them, with `fixed_args` supplying the rest by name, as in `evaluate(predict, posterior, fixed_args={"x": X_new})`. `fixed_args` must leave exactly one parameter free, the one the operand maps to; leaving two unbound is an error. The registry keys on the map's own type and the fixed arguments pass through to each rule's feasibility check, so binding side arguments neither wraps the map nor loses its registered identity, and the single mapping parameter keeps the operation's controls separate from the map's arguments.

**The evaluation registry.** For a distribution or batch operand the call dispatches through a `BinaryDispatchRegistry` keyed on the map's and the operand's types, whose methods are **evaluation rules**. The registry is defined beside the engine (Part IV) and consulted by every single-distribution and batched application, the operation and the direct call alike. Auto-selection tries the rules in priority order:
- **Closed-form rules** return an exact parametric result. For example, `A @ d` for a Gaussian `d` is again Gaussian, with mean `A @ mean(d)` and covariance `A Σ Aᵀ` built lazily through the operator algebra.
- **Change of variables** applies when the map is invertible and carries the Jacobian claim (`is_invertible` and `SupportsLogDetJacobian`), returning a transformed distribution whose `log_prob` is exact via the log-determinant of the Jacobian.
- **The sampling lift**, the rule registered at the generic pair, always applies: draws from `d` are pushed through the map, returning an empirical distribution over the outputs, with the sample count and PRNG key exposed as controls. It is the registry's floor, and the route every plain callable takes.
- **The elementwise sweep** is the batch counterpart: the rule at the generic pair for a batch operand. A fused batched implementation, such as an operator's matrix–matrix routine or a single vectorized call over array-backed elements, registers above it.

The selected rule records its name and fidelity in the result's provenance wherever it is invoked, the direct call and the operation alike, mirroring the converter registry's recorded fidelity; `evaluate` adds its own outer operation record on top, so the two agree in value and in selected rule while their complete records differ by that one entry. `method="..."` forces a rule by name. Rules need not be exact: an approximate scheme — quadrature, an unscented transform, quasi-Monte Carlo — registers at its recorded fidelity. Selection is fidelity-first: priorities are assigned from fidelity tiers, exact rules above approximate ones with the sampling lift the floor at the generic pair, and within a tier the more specific type pair wins, by the same closest-in-method-resolution-order tie-break the dispatch registries use, then registration order. Exactness is therefore never traded for type specificity silently: a rule that is more specific but approximate is selected only when no exact rule applies, or by `method=`. New rules join by registration, so an evaluation route for a new pair of types, exact or approximate, is added without touching the operation.

**Linear maps push moments exactly.** Whatever rule realizes `A @ d`, the result's `mean` and `cov` delegate exactly whenever `d` supports them, since `E[A X] = A E[X]` and `Cov(A X) = A Cov(X) Aᵀ`. An approximate linear pushforward therefore still reports exact first and second moments.

Applied to a `ConditionalDistribution`, evaluation acts on the event side, giving the kernel `s ↦ f♯K(s, ·)` with the same given spec.

The result is a tracked term: its `provenance` records `evaluate`, the map, and the operand, and its `name` is auto-derived.

### Rationale

`evaluate` is `C4 – Function lifting` in operation form: applying a map is one act whatever the operand kind, and substituting a multiplicity for a value leaves that act well-defined either way — a distribution gives the pushforward law, a batch the elementwise result — with this operation returning it directly. Dispatching over pairs of map and operand types realizes `C3 – Computational detail hidden by default, available on demand`, since a pair with a known closed form or a fused batched routine gets it automatically, while every other pair still answers through the floors, sampling for a distribution and the sweep for a batch. Recording the producing rule keeps the approximation honest (`D1 – Mathematical fidelity`), registration grows the exact set without changing call sites (`D2 – Generality first`), and the result is a tracked term that composes further (`D4 – Closed system of objects under operations`). The operation is named for what it does across every operand kind, with *pushforward* reserved for mathematical statements, as *kernel* is for `ConditionalDistribution`.

## V.2 — `inverse` and `log_det_jacobian`

### Contract

Two operations read a map's inverse structure.
- `inverse(f)` returns the inverse map as a `Function`: `inverse(f)(y)` is the preimage of `y` under `f`. Its capability route is read through `is_invertible`, and a map that fails that check resolves to nothing and raises. The result is itself invertible, with `f` as its inverse, and it carries the Jacobian claim whenever `f` does.
- `log_det_jacobian(f, x)` returns the log-determinant of the Jacobian of `f` at `x`. Its capability route is `SupportsLogDetJacobian`. The result is a tracked scalar, with `raw=True` returning the bare array. The reverse direction needs no second operation, since `log_det_jacobian(inverse(f), y) = −log_det_jacobian(f, inverse(f)(y))`.
- Each carries a capability route and no other today. A numerical route — root finding for an inverse, automatic differentiation for a Jacobian — may register later at its recorded fidelity, widening what resolves without changing either contract (V.0).
- Both results are tracked terms whose `provenance` records the operation and the map, with `inverse(f)`'s name auto-derived from `f`'s.

### Rationale

Reparameterization moves in both directions between a constrained and an unconstrained space, so the inverse must be reachable from user code, and the operation form keeps the capability's implementer methods private, per the V.0 convention. Returning the inverse as a `Function` keeps the system closed (`D4 – Closed system of objects under operations`): the inverse evaluates, composes, and pushes forward like any map. Splitting the Jacobian claim from invertibility is `D3 – Capability-based operations`: a map can be invertible without a tractable Jacobian determinant, and the determinant exists only for a differentiable map, so each claim stays honest on its own and change of variables requires exactly the pair.

## V.3 — `sample`

### Contract

`sample(d, key=None, sample_shape=(), raw=False)` draws from a distribution.
- With `sample_shape=()` it returns a single draw **at the declared kind**, tracked. The kind is the event declaration's class, fixed by the declaration and never by the runtime value: a `RecordSpec` draws a `Record`, a `DistributionSpec` a `Distribution`, a `FunctionSpec` a `Function`, a `NumericArraySpec` a `NumericArray`, and an `OpaqueSpec` an `Opaque`. Every spec names a class, so no kind is presented by wrapping it in another: a scalar law draws a `NumericArray`, not a single-field `Record`.
- A non-empty `sample_shape` prepends batch axes and returns the tracked batch type of the draw's kind, with those leading dimensions on a level named for the operation that mints them, `sample`: a `RecordBatch` (`NumericRecordBatch` when numeric) for a record event, and the tracked batch form of the table below — a `DistributionBatch`, `FunctionBatch`, and so on — for a term-drawing law.
- `sample(..., raw=True)` is the draw detached (V.0): the kind's raw value for `sample_shape=()`, and for a `Distribution`- or `ConditionalDistribution`-valued draw the law itself, without provenance or annotations. For a non-empty `sample_shape` it is the **storage view** (II.5), the draws' raw values in their native stacked layout:

| declared kind | tracked batch | `raw=True` result |
|---|---|---|
| `NumericArraySpec` | `NumericArrayBatch` | the stacked array, batch axes leading |
| `RecordSpec` | `RecordBatch` / `NumericRecordBatch` | the nested mapping of raw columns |
| `DistributionSpec` | `DistributionBatch` | an object array of the drawn laws |
| `ConditionalDistributionSpec` | `ConditionalDistributionBatch` | an object array of the drawn kernels |
| `FunctionSpec` | `FunctionBatch` | an object array of the drawn callables |
| `OpaqueSpec` | `OpaqueBatch` | an object array of the drawn objects |
- Sampling requires a concrete declaration and raises with the free dimensions named; in the fused conditional path, the given value binds them first.
- A caller-supplied PRNG `key` makes the draw reproducible from the key alone; with the key omitted, the draw is a workflow-owned random event (IV.3). Under a non-empty `sample_shape` the key splits by draw index, so the draws are jointly independent and reproducible together; inside a `Function`, the workflow scope supplies each draw's key by structural identity, so draws are never hand-threaded.
- The draw carries `provenance` recording `sample` and the distribution it came from.

### Rationale

Every draw is reproducible from its inputs — from the key alone when the caller supplies one, and from the workflow scope's seed and the draw's structural identity otherwise (IV.3) — which is `C6 – Traceable and reproducible workflows`. Returning every draw as the tracked term of its declared kind serves `C1 – Uniform interface to functions, distributions, and values` without making the kinds uniform: what is the same across laws is that a draw is tracked and its type is fixed by the declaration, not that every draw is a `Record`. The `raw` opt-out serves `C3 – Computational detail hidden by default, available on demand`: the wrapped, tracked draw is the default, but the bare value remains available when the user needs it.

## V.4 — `log_prob` and `unnormalized_log_prob`

### Contract

`log_prob(d, value)` returns the log-density of `value` under `d`. The value may be a `Record` matching the event schema, or a bare array for a scalar law.
- `log_prob` requires `SupportsLogProb` and returns the *normalized* log-density.
- `unnormalized_log_prob` requires only `SupportsUnnormalizedLogProb` and returns the log-density up to an additive constant, which is what inference against an unnormalized target needs.
- The result is a tracked term: a `NumericArray` whose provenance records the operation, the distribution, and the scored value, with `raw=True` returning the bare array.
- A scored value binds any symbolic event dimensions for that call only, so one law scores datasets of different sizes.
- A batch of values, or a `DistributionBatch`, maps elementwise to the batched densities, a `NumericArrayBatch`.

### Rationale

Splitting `log_prob` from `unnormalized_log_prob` keeps each capability honest (`D1 – Mathematical fidelity`): a distribution that knows its normalizing constant offers the true density, while one that does not still serves inference, which needs the density only up to a constant.

## V.5 — Distribution functionals: `mean`, `variance`, `cov`, `quantile`, `expectation`

### Contract

The moment operations summarize a distribution by a deterministic value.
- `mean(d, raw=False)` and `variance(d, raw=False)` return an event-typed value, that is, a value shaped like a draw. Neither is restricted to numeric draws, but each requires the event type to support it: a random function has a mean function and a pointwise variance function, while a random measure has a mean (the marginalized law) but, in general, no event-typed variance. The result is wrapped at the law's declared event kind — a `Record` whose schema matches the distribution's for a record-drawing law, and a term for a term-drawing one, so a random function's mean is a `Function` — or the raw event-typed value `T` with `raw=True`.
- `cov(d)` requires a numeric draw and returns a covariance operator over the *flattened* draw, a `(vector_size, vector_size)` `LinOp`, since covariance couples distinct coordinates. Its input and output schemas are both the distribution's numeric event spec, so it applies directly to draws.
- `quantile(d, q)` requires a numeric draw. It takes a level `q ∈ [0, 1]` or array of such levels and returns the quantile for each, computed per coordinate for a multivariate draw. Its result declaration is event-kind-directed like any other: a single level returns the event's own kind — a `NumericArray` for an array-drawing law, a `NumericRecord` for a record-drawing one — and a plural `q` adds a level over those, giving the matching batch. The plurality of `q` is a shape, so planning reads it (V.0).
- `expectation(d, f)` returns `E[f(X)]`, shaped by the output of `f`, for any event type `f` accepts. The result is wrapped at the kind `f`'s output declaration names, a `Record` for the usual record output.

Each carries a capability route on the matching protocol (`SupportsMean`, `SupportsVariance`, `SupportsCovariance`, `SupportsQuantile`, `SupportsExpectation`), feasible whatever the event type, and a Monte Carlo fallback route feasible through `SupportsSampling`, with the sample count and PRNG key as controls (V.0). A distribution that claims neither resolves to nothing and raises. The fallback route is defined per event kind. A numeric event averages draws coordinatewise. A function-valued event answers with a lazy function: its mean at a point is the average of the sampled callables there, and its variance the pointwise sample variance. A measure-valued event's mean is the finite mixture of the sampled draws, the Monte Carlo estimate of the mean measure. `cov` and `quantile` keep numeric-only fallback routes, the sample covariance returned as a `DenseLinOp` and the per-coordinate empirical quantiles. `expectation` averages the array outputs of `f` and so applies to any event type that samples.

### Rationale

A mean is defined whenever draws can be averaged: coordinate-wise for arrays, pointwise for functions, and set-wise for measures, where `A ↦ E[ξ(A)]` is again a measure. An event-typed variance additionally requires the second moment to be a value of the event type. That holds for arrays and functions, but fails for a general random measure: to be a measure, `A ↦ Var(ξ(A))` would have to be additive, which holds only when disjoint regions are uncorrelated (a completely random measure) and never for a random probability measure, whose fixed total mass forces negative correlation. A random measure's second-moment structure is instead a covariance over pairs of sets, the analog of `cov` rather than of `variance`. Gating `mean` and `variance` by capability rather than by a numeric event is `D1 – Mathematical fidelity`, and `cov` and `quantile` remain numeric-only, since a flat covariance operator and ordered quantile levels have no meaning for a non-numeric draw. `cov` returns a flat operator rather than an event-typed value because it couples coordinates that the event's field structure keeps separate. The closed-form-or-Monte-Carlo split realizes `C3 – Computational detail hidden by default, available on demand`: a distribution that can give an exact moment does, and one that can only sample still answers, approximately.

## V.6 — `condition_on`

### Contract

`condition_on(d, given)` fixes some fields of a distribution or conditional distribution and returns the resulting distribution.

**The `given` argument.** `given` is field-keyed: a `Record`, or a mapping from field paths to values, with each value, in either presentation (II.1), conforming to the spec at its path, checked as one simultaneous unification that binds any symbolic dimensions and raises on jointly inconsistent shapes. Each key must name either a *given* slot (a name in the `given_spec`, or a path into a structured slot) or a *produced* field (a path in the event schema), and any other key is an error. A key may also name an interior path, in which case its value is a sub-record checked against the sub-schema. Binding part of a structured slot is defined as restructure-then-bind (II.6): the bound part is promoted and bound exactly, the residual slot remains, and a group emptied by the binding dissolves. Conditioning is stated entirely in terms of fields, and factors never appear in the call: the derived factor graph is read only to decide which case below applies and to carry it out.

**The routes.** `condition_on` resolves across three of the four route sources (V.0): structural routes that curry a given slot or slice the factor graph, a capability route on `SupportsConditioning` — which a `ConditionalDistribution` always satisfies, `_condition_on` being its required primitive — and a registry route through the inference methods for Bayes' rule. Which is feasible follows from where each conditioned field sits in the factor graph, read from declarations:
- **Exogenous given, so curry.** Binding a slot that the object conditions on but does not produce returns a smaller `ConditionalDistribution`, or an ordinary `Distribution` once all given slots are bound. This is exact and involves no inference. For example, binding a regression model's covariates curries it toward the data-ready likelihood.
- **Upstream or independent produced field, so exact slice.** Conditioning on a produced field that no remaining factor depends on through an unconditioned path returns the exact conditional by slicing the factor graph, again with no inference. The slice is exact only when the field's own factor is single-field or itself implements the conditioning capability; a multi-field factor that cannot condition internally falls to the Bayes' rule case. For example, conditioning a Gaussian prior on one of its own fields slices, while an atomic empirical factor over two fields cannot be sliced on one of them.
- **Downstream data, so Bayes' rule.** Conditioning on a field that downstream factors depend on is an application of Bayes' rule: the conditional is proportional to the joint density at the fixed value, and is generally not available in closed form. It is delegated to an inference algorithm registered for the model, and only the factored classes support it. For example, conditioning on observed responses downstream of the coefficients is the Bayes' rule case. On an atomic distribution, conditioning on produced fields uses its own `SupportsConditioning` when implemented, as a `MultivariateNormal` does exactly, and raises otherwise.

When `given` names several fields, the cases combine: the exact bindings (curry and slice) are applied first, and Bayes' rule runs on what remains. Field classification is computed once, on the graph with every conditioned field marked, so the outcome does not depend on the order the fields are listed. Conditioning on a produced field does not require the given fields to be bound first. The result stays conditional on the unmet givens, with the produced-field conditioning applied within each slice of the given, so the result curries like any other `ConditionalDistribution` and the two orders agree: conditioning on a produced field and then binding the given yields the same distribution as binding the given first. When that produced-field conditioning requires Bayes' rule, the resulting `ConditionalDistribution` may realize the inference lazily, once its given is bound, or through a method that supports amortization.

`condition_on` always binds the supplied value as the field's fixed value, whatever the value's type; the mixture `∫ K(s, ·) μ(ds)` over a mixing distribution is requested explicitly through the separate `mixture` operation. The result carries `provenance` recording the operation and the conditioning fields.

**The inference-method registry.** The Bayes' rule case is dispatched through the **inference-method registry**, a `UnaryDispatchRegistry` keyed on the model's type whose methods are inference algorithms such as MCMC or variational families. Each method registers under a unique `name`, declares the models it applies to, and probes feasibility before it runs. `condition_on(..., method="tfp_nuts")` selects a specific algorithm, and otherwise the registry picks the best applicable one.

**Prioritization.** Methods are ranked by an integer `priority` that tiers them: above 50 is an **exact** method, 1 to 50 an **inexact** one, and 0 is **opt-in-only**, skipped by auto-selection and reachable only by name. A newly registered method is opt-in-only until a contributor classifies it, so adding an algorithm never silently changes what runs. A deployment re-ranks methods at runtime with `set_priorities`, which warns when a method crosses into or out of opt-in-only.

### Rationale

A single operation covers binding, slicing, and Bayes' rule because all three are the same mathematical act of conditioning, and they differ only in whether the conditioned field is exogenous, upstream, or downstream in the factor graph. Collapsing them into one operation keeps the user interface small (`C1 – Uniform interface to functions, distributions, and values`), while the derived graph decides the algorithm (`C3 – Computational detail hidden by default, available on demand`).

## V.7 — `joint`

### Contract

Recall that the composition operator `A * B`, exposed on `Distribution` and `ConditionalDistribution` as `__mul__` constructs the joint (conditional) distribution of `A` and `B`. It returns either a `FactoredDistribution` or, when some givens remain, a `FactoredConditionalDistribution`. Its `provenance` records `*` as the operation and the operand factors, and its `name` is auto-derived by the canonical-order rule fixed with `*` (III.12).

**The realigning `joint` form.** A limitation of `*` is that it requires a producer's field names to match the names its consumer conditions on. This motivates the `joint(A, B, **align)` op, which realigns fields first and then composes exactly as `*`, so it is equivalent to `A * B.with_path_names(**align)`. For example, a likelihood that conditions on `slope` can be combined with a prior where the slope is called `beta` with `joint(lik, prior, beta="slope")`.

### Rationale

Composition is written as an expression so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and `*` returns a first-class joint so the result composes further (`D4 – Closed system of objects under operations`). Realignment is an exact rename: `with_path_names` returns the same law under new field names, so `joint` connects mismatched factors without altering their joint law (`D1 – Mathematical fidelity`).

### Open points

- *The `align` contract.* `align` pairs are `with_path_names` pairs, path-valued targets included (II.6), so realignment can promote a nested field to a slot-matchable name. The remaining fine print — freshness and injectivity requirements on the new names, and which operand's fields may be realigned — stays deliberately deferred: the question is subtle, and implementation experience should inform the decision.

## V.8 — `marginal` and `factor`

### Contract

Two operations read the parts of a structured or factored distribution. (The third access form, the `d[field]` **view**, is not an operation: it returns a correlation-preserving reference into its parent rather than a standalone object, and its contract is fixed with the factored distributions.)

- `marginal(d, field)` returns the **detached** marginal of a field or field group, a standalone `Distribution` with no reference back to `d`. It carries a capability route on `SupportsMarginals` and a Monte Carlo fallback route through `_sample`, projecting draws onto the field and returning an empirical marginal; when the capability is absent, or the path has no exact route within it, the fallback is what resolves, and a distribution that cannot sample either resolves to nothing and raises. The fallback's provenance names the route (`monte_carlo`) and the sample count, so an approximate marginal is distinguishable from an exact one after the fact.
- `factor(d, name)` returns a building-block **factor** of a joint, keyed by factor name, either a `Distribution` or a `ConditionalDistribution` for a dependent edge. Its capability route is `SupportsFactors`, so a distribution that exposes no factors resolves to nothing and raises.

Both return a **tracked term** whose `provenance` records the access and its source distribution.

### Rationale

`marginal` and `factor` are the two query directions of the field and factor interfaces fixed with the factored distributions: `marginal` reads a field's law detached from its joint, and `factor` reads a construction block. Exposing them as named operations rather than as indexing is what separates the detached query from the correlation-preserving `d[field]` view (`D1 – Mathematical fidelity`). Dispatching `marginal` on `SupportsMarginals` opens the detached query to any distribution that knows its marginals, factored or not. Gating `factor` on `SupportsFactors` keeps factor access honest, since only a distribution actually built from named parts can answer it (`D3 – Capability-based operations`).

## V.9 — `mixture`

### Contract

`mixture(K, mixing)` returns the mixture `μK = ∫ K(s, ·) μ(ds)`: the law of `T` for `S ~ mixing` and `T ~ K(S, ·)`, the mixing distribution's produced slots meeting the kernel's given slots by name.

`mixture` is a **derived** operation (V.0), defined by the identity `mixture(K, mixing) = marginal(K * mixing, ...)` onto the kernel's produced slots, detached. That identity is its floor route, so its feasibility and its failure modes are those of `*` and `marginal` — a call resolves at worst when the composition is well-formed and the marginal has a route, and a mixing distribution that cannot be sampled and admits no exact marginal raises there rather than here. A direct route may realize a case in one step instead, as the Gaussian algebra does, ranking above the identity by fidelity and specificity. Slot matching, spec unification, and unmet givens therefore behave exactly as composition (III.12) and `marginal` fix them, with nothing separately defined: a given slot the mixing distribution does not meet stays unmet, and the result is then a `ConditionalDistribution` over the unmet givens. Exactness follows the same route — a finite mixing distribution yields the explicit `MixtureDistribution`, a closed family stays closed, as in the Gaussian algebra, and otherwise the result is the Monte Carlo empirical marginal through ancestral sampling, with the route recorded in provenance.

Mixing is always an explicit request: `condition_on` binds a supplied value as a value whatever its type (V.6), so no conditioning call ever mixes implicitly. The result is a tracked term whose `provenance` records the operation, the kernel, and the mixing distribution.

### Rationale

Keeping the integral out of `condition_on` keeps conditioning single-valued — a supplied value always binds, and `μK` is always asked for by name — so neither call has a data-dependent meaning (`C1 – Uniform interface to functions, distributions, and values`). The name is the result's mathematical name: `μK` is the mixture of the kernel family with mixing distribution `μ`, which covers predictive, compound, and state-propagation uses without privileging one (`C5 – Naming for unambiguous meaning`, `D1 – Mathematical fidelity`). Defining the operation as the marginal of the composed joint adds no second semantics: one identity ties it to machinery already fixed, so every behavior has a single source (`D6 – Single source of truth`).

## V.10 — Batched operations

### Contract

Every operation lifts to a `Batch` by mapping over its elements, which is the elementwise sweep applied to the operation itself.
- `sample` over a `DistributionBatch` returns a **nested** batch, the outer level ranging over the laws and the inner over each law's draws: `sample(d_batch, key, sample_shape=(S,))` has `axis_groups` `(*d_batch.axis_groups, (S,))` and appends an inner draw level named `sample`, so iterating it visits one law's batch of draws at a time — a `RecordBatch` for a record-drawing law, and the batch form of the declared kind otherwise. An operation names the level it mints after itself, which is why `quantile` adds a `quantile` level (V.5). Level names are unique, so a `d_batch` already carrying a `sample` level is renamed with `with_level_names` before the draw, rather than the new level being silently altered to `sample2`; giving `sample` its own level-name argument is the alternative, and either way the choice is the caller's. `log_prob` maps elementwise to the batched densities, with the batch axes preserved.
- A moment over a `DistributionBatch` returns a batch of the corresponding values, such as a `LinOpBatch` for `cov`. A multi-level query nests the same way: `quantile(d_batch, q)` keeps the laws on the outer level and adds an inner level named `quantile` for the levels of `q`.
- **Alignment.** A binary operation matches the operands' levels **by name**: a level in both must have broadcast-compatible shapes, with size-1 broadcasting; a level in only one operand broadcasts across the other; and an outer product is requested by explicit reshaping rather than implied. Because every level is named, there is no positional fallback, and two levels meant to correspond under different names are lined up by renaming one with `with_level_names` first, exactly as `joint` realigns fields for composition. So a flat batch of values on a `laws` level scores against the `laws` level of a nested sampling result. `given=` accepts a `RecordBatch` and yields the `DistributionBatch` of conditioned laws.
- Two operands are exempt from batch lifting: the factors of composition (`*` and `joint`) and the map operand of `evaluate`, which are consumed as objects rather than swept.
- Batched application resolves through the evaluation-rule registry: the elementwise map is the rule at the generic pair, and a fused implementation, a single vectorized call over array-backed elements or an operator's matrix–matrix routine, registers above it and is selected like any other rule.
- An operation applied to a batch whose elements lack the required capability raises the same capability error a single element would.

### Rationale

A batched operation is not a new operation but the elementwise sweep applied to an existing one, so a `Batch` supports exactly the operations its elements do (`D3 – Capability-based operations`). Nesting the laws level over the draws level keeps two genuinely different multiplicities distinct in the result itself (`D1 – Mathematical fidelity`), and matching levels by name rather than by position carries `C5 – Naming for unambiguous meaning` onto the multiplicity axis, making the rename the single tool for lining levels up, as field renaming is for composition. When the elements are array-backed and claim differentiation, the sweep is a single vectorized call that preserves the claim, which is `D3 – Capability-based operations` reaching the batch through its elements.
