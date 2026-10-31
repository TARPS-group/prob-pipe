# Part VI — Operations

Parts II–V fixed the *shared abstractions*, the *term kinds*, the *distributions*, and the *functions* that act on them. Part VI fixes the **operations**: what one is (VI.0), and for each operation its operands, its result, and its routes.

**Conventions.** The user-facing names are the bare operations, such as `sample` and `mean`. The implementer counterparts are `_`-prefixed, such as `_sample`, and for conditional distributions the prefix `_conditional` is added as well.

## VI.0 — The operation model

### Contract

An **operation** is a `Function` with three additions that provide a uniform interface over the function-level implementations: one call, `mean(d)`, whatever the kind of `d` and whichever implementation realizes it. Each addition answers one way in which the implementations differ, and the engine's stack reads each at its step (V.1):

1. **Operand roles.** The implementations differ in the kinds they handle, so a parameter cannot be typed by a concrete spec: `mean` takes any distribution whatever its event. A role names the kind a parameter accepts by spec class, and admission checks it (V.4).
2. **Applicability checks and a result rule.** The result depends on which operand arrived, so its output spec cannot be fixed at construction: the mean of a law is a value of the law's event kind. A result rule derives the result's output spec, and the checks it must defer to return, per call from the operands' specs and from any parameter that selects rather than supplies, such as `marginal`'s paths, and the applicability checks state what makes a call's declarations well-formed before the rule runs (V.6).
3. **Routes.** One interface fronts many implementations, so an operation is realized by a set of routes the engine selects among (V.7 and V.9), or, for a *derived* operation, by a formula in terms of other operations.

A result rule declares the result's outputs independently of its label. An output names what other terms receive from the result, and the label states what the result is:

1. **A value indexed by the law's event** takes the event's outputs and paths. Such a value is a draw, a batch of draws, or a field-by-field summary such as a mean or a quantile. The mean of a law over `mu` and `tau` is therefore a record whose fields are `mu` and `tau`, and the mean of `Normal("x", 0.0, 1.0)` is an array, a base value standing alone. A summary defined through a derived law takes that law's outputs, so `expectation(d, f)`, which is `mean(pushforward(d, f))`, takes the outputs of `f`'s return value.
2. **A law**, such as a conditional, a marginal, or a converted law, has its event's outputs (II.2).
3. **Any other value**, which no single output of the event indexes, takes the operation's name, as a function's output takes the function's name (III.3). A score is therefore declared under `log_prob`, and a covariance under `cov`.

Indexing commutes with a field-by-field summary. For a law over a record, the part of `op(d)` at a path `p` and `op(d[p])` have one estimand, and they are equal where both take exact routes, so `mean(d)["mu"]` is the mean of `d["mu"]`. For a law over a base value, the summary is the summary of that value. A summary also binds where a draw binds whenever its term spec conforms to the event's (II.1), so a plug-in call such as `log_prob(d, mean(d))` or `condition_on(model, mean(posterior))` takes the summary of a real-valued event as it takes a draw. The mean of an integer event is floating (VI.5), so an input that requires integers refuses it.

The result's label is independent of its outputs: it is the rendering of the expression the operation's rule builds from its operands' expressions (II.4), as `𝔼[(y, mu) ~ model]` for `mean(model)`, and an operation without a rule gives its result its primary operand's expression. A public `label=` rule gives the result its label alone.

A rule that cannot know the output spec before a route runs, as for the factor that `factor` returns, declares none, and the returned term carries its own. An output spec whose spec a rule leaves unspecified is completed before any law is constructed (II.2).

Since the engine runs the stack, `op(...)` is a tracked term under fresh, derived identity whose spec conforms to the completed output spec, `op.using(raw=True)(...)` is that result detached (II.4), and `op.check(...)` is the engine's `check`.

**Routes.** The implementations behind one operation come from different places, and a route is the one form they all take. It has the interface of a dispatch method (II.7), that is, `check`, `execute`, and an **exact** flag, but is bound to a call rather than to argument types, and it is registered against the operation it realizes by upward registration, as for any registry. Routes come from four sources, and an operation may carry any combination:

| route source | the implementation comes from | example |
|---|---|---|
| **structural** | the operands' declared structure | `condition_on` binding an input; `joint` merging two factors |
| **capability** | a `Supports*` protocol on a named operand | `_sample` on the subject of `sample` |
| **registry** | a registered method selected by dispatch | the inference methods behind Bayes' rule; the evaluation rules behind `pushforward` |
| **fallback** | a generic scheme applicable to a stated domain | Monte Carlo through a sampling operand |

Where a route dispatches on a capability it names the operand it dispatches on. Its check reads the call's specs, the parameters the result rule reads, the controls, and declared representation metadata such as a factorization or a guarded capability (III.8), and it never evaluates the body. A capability route requires protocol membership plus the capability's guard, and membership alone where the capability is total on its domain. A structural route checks declared structure, a registry route delegates its probe to that registry, and a fallback checks its stated domain and assumptions. Missing declarations are reported as unresolved (V.1), and execution never catches a failure and tries another route.

Routes use II.7's exactness, specificity, and registration order, with no configurable priority at the operation itself. A fallback route ranks below every other route on its stated domain, whatever their exactness, as a floor does (V.7). Registry routes retain their registry's priorities. Their local fidelity is the selected method's; derived routes likewise use the selected chain's guarantee rather than a fixed exact tag on the wrapper. A route that delegates its exactness and can restrict itself to exact implementations, as a registry route or a structural route that resolves the operation on parts of its operands can, is ranked twice: restricted to the implementations that are exact, among the exact routes, and otherwise among the approximate routes, so `exact_only` keeps only the first. A derived operation's formula is realized by its derived route, which ranks below the operation's direct routes of the same exactness. By default that route runs the body, and an operation may instead register a composite route in its place, which runs a route of each constituent operation in turn. A composite route passes the `method` control to the constituent whose routes it names, and the call's provenance records the exactness of the routes that ran, which the route learns when it runs. A derived route learns its exactness only from the constituents its formula calls, so it is ranked once. Without `exact_only` it ranks among the approximate routes. Under `exact_only` every call within it selects an exact route, as **Controls inside a derived route** states, and its check reports under the same restriction whether the constituents it probes have one, so a composite route and a body route with a `derived_route_check` rank among the exact routes. A body route without a `derived_route_check` has no such report, and `exact_only` excludes it.

**Naming a route or a method.** The `method` control names a route of the operation or a method inside one of its registry routes. A plain name resolves when it matches a single route or registry method of the operation, which then runs, subject to its check (II.7). A name that matches several raises `ResolutionError`, naming each candidate in the qualified form `route/method`. That form selects a method within the named registry route, and a route is named by its plain name alone. A name that matches nothing raises `ResolutionError` (II.7). For example, `convert` carries the registry route `converters` over the converter registry and the registry route `normalize` over the inference-method registry (VI.10), so when both registries hold a method named `nuts`, `method="nuts"` raises and `method="normalize/nuts"` selects the inference method.

```python
@dataclass(frozen=True)
class BoundCall:                   # one call, after binding and normalization
    operation: Function
    operands:  Mapping[str, Any]           # wrapped terms, or planned conversions, by role
    controls:  Mapping[str, Any]           # the resolved controls: the two counts, the method, …
    @property
    def specs(self) -> Mapping[str, TermSpec]: ...
    # each operand's spec, or the target spec a planned conversion promises
    @property
    def declarations(self) -> Mapping[str, Any]: ...
    # the parameters the result rule reads and the representation metadata planning admitted (V.6)

class OperationRoute(Protocol):    # one interface; the helpers below are construction shorthand
    name:     str
    source:   RouteSource
    exact: bool | None            # None for a delegated route until its plan selects a method
    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility: ...
    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any: ...

mean.capability_route("closed_form", operand="d", protocol=SupportsMean, method="_mean",
                      exact=True)
mean.fallback_route("monte_carlo", check=_can_sample, execute=_mc_mean, exact=False)
marginal.capability_route("exact", operand="d", protocol=SupportsMarginals,
                          method="_marginal", check=_can_marginalize_path, exact=True)
# An omitted capability guard means membership suffices on the declared domain.
condition_on.structural_route("bind_inputs", check=_can_bind_inputs,
                              execute=_bind_inputs, exact=True)
condition_on.registry_route("inference_methods", registry=inference_method_registry)
```

**Primitive versus derived operations.** Some operations mean something in terms of others, and the interface keeps that definition visible. A **primitive** operation states its own contract and carries its own routes. A **derived** operation is instead defined by a formula over other operations; `mixture`, for example, is the rebuilt kernel output projected from a merged joint (VI.9), and `expectation` is the mean of a pushforward (VI.5). That formula is what the operation *means*, so its result rule, feasibility, and failure modes follow from the operations it is defined by. The formula is itself realized by the derived route on the domain where its constituent operations are available, and a derived operation may carry routes that realize it **directly** besides: a Gaussian mixture computed in closed form need not merge and then marginalize. Direct routes rank above the derived route when their fidelity and specificity warrant it; the generic derived route is a fallback on its own domain (V.7), as the sampling lift is under `pushforward`, and a composite derived route ranks as the paragraph on ranking states. Either way the operation adds its own outer provenance record, and that record names the route that ran.

**Controls inside a derived route.** A derived route passes three controls of the derived call to the `Function` calls made within it, operations and plain functions alike. The engine runs the derived route check and the formula in the scope that passes them:
1. `exact_only` restricts the route selection of every call in the scope, at any depth, whatever the call sets;
2. `max_evaluations`, when the derived call sets it, applies to each call the scope makes outside any other call, unless the called `Function`'s own controls, from its decorator or its view, set their own;
3. `num_samples` applies to the same calls under the same condition.

Each count passes as itself, so `standard_deviation.using(num_samples=n)` sets the draws of the `variance` its formula calls. A formula that estimates a summary through a lift instead spends the summary's resolved `num_samples` as the lift's `max_evaluations`, whether or not the derived call sets it and whatever the map sets, since each draw is one evaluation of the map: `expectation(d, f)` evaluates `f` at most `num_samples` times (VI.5). The call's provenance records the `num_samples` of the Monte Carlo draws the constituents of its formula take.

The call context carries the passed controls, so the scope covers the calls made in the calling thread and in the workers the engine dispatches to. The other controls stay each call's own, since route names and budgets belong to the function that declares them. A `method` that names no route of the derived operation, but names a route or method of an operation its formula calls, raises `ResolutionError` naming the call on that operation that selects it. A planned conversion follows its parameter's `conversions` control (V.4), as in a direct call. Under `exact_only` every route selected within the formula is therefore exact. A `derived_route_check` reports as deferred the check of each constituent whose arguments are known only when the formula runs. The call raises `ResolutionError` when such a constituent then has no route. For example, `prob.using(exact_only=True)` restricts the `log_prob` its formula calls to exact routes.

**Execution controls.** The executor and its options, named on an operation's call, apply to everything the call does, while the counts apply only to the calls a derived operation's formula makes directly. They govern every set of evaluation points of an opaque callable made during the call, at any depth, by the rules of V.9. `condition_on.using(executor="slurm")`, which is not a derived operation, therefore sends the evaluations of a model's opaque simulator to `slurm`, while the compiled parts of the model stay compiled.

**Declaring an operation.** `@operation` registers an operation and takes its result rule. The authored signature contains the operands and the parameters the result rule reads: `log_prob`'s scored value, `expectation`'s function, `quantile`'s probabilities, `sample_shape`, and alignment mappings are arguments even when their values do not determine the output shape. The result rule reads their specs and any declared static information it requires. Controls choose realization, execution, or presentation and are set through `using`, which also sets a method's numerical budgets as `method_options` (V.2). A primitive body is empty; a derived body states its formula. Two kinds of check say where an operation applies. Its `applicability_checks` apply to every call, whatever route serves it: each reads the operands' declarations before any route is chosen, and a check that fails raises `ApplicabilityError`, so no route runs. A derived operation's `derived_route_check` is the derived route's own feasibility check: it reads the call's argument values, as the formula does, and when it reports the call infeasible the engine tries the operation's other routes. For a conditional operand the framework adds `given=` as the argument of the fused conditional call (III.9): each operation whose subject is a distribution, `condition_on` aside, appends a keyword-only `given=None`. A call on a kernel `K` with a given is served by the conditional twin of the route's capability where `K` claims it, and otherwise by the operation applied to `condition_on(K, given)`.

```python
def _mean_result(d: DistributionSpec) -> OutputSpec: ...
# planning reads the operands' specs and the declaration inputs, and nothing traced (V.6)

@operation(result=_mean_result)
def mean(d: Distribution): ...              # primitive: no body; the registered routes implement it

@operation(result=_mixture_result)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    return pushforward(K & mixing, _kernel_output_projection(K))  # the formula in VI.9
```

**Four raw mechanisms, one meaning.** *Raw* always means the representation layer, detached from the workflow; the mechanisms differ only in where they act. Route implementations are *written* over raw forms (II.4); `apply` (III.3) *evaluates* a wrapped callable with no lifting, tracking, or provenance; `using(raw=True)` *skips* the identity an operation would mint; and `raw` (II.4) *detaches* an existing term.

**Randomness.** Each draw an operation causes is a workflow-owned random event (V.8), whether the operation's contract is a random draw or a deterministic quantity whose route samples.

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
    exact:    bool | None        # None until a delegated route is resolved (II.7)
    requires: tuple[type, ...]   # the protocols a capability route needs; empty otherwise
    condition: str               # the feasibility condition in words, for the routes that types cannot state

@dataclass(frozen=True)
class OperationSummary(EntrySummary):
    operands:   tuple[OperandSummary, ...]   # in signature order
    is_derived: bool
    formula:    str | None                   # the formula, when derived
    routes:     tuple[RouteSummary, ...]     # in selection order

class OperationRegistry(SupportsRegistryCataloging):
    def register(self, op: Function) -> None: ...
    def list(self) -> list[OperationSummary]: ...
    def describe(self, name: str | None = None) -> str: ...
    def __getitem__(self, name: str) -> Function: ...

operation_registry: OperationRegistry     # the global instance
```

### Rationale

Defining an operation by its operands, a result rule, and a set of routes keeps the vocabulary closed: adding an operation cannot add a mechanism, and adding an *implementation* is registering a route rather than amending a contract (`D2 – Generality first`). Separating what a call means from how it is realized makes totality a property of the routes available at call time rather than of the operation, so one rule says when a call can fail instead of per-operation exceptions. Capability dispatch is `D3 – Capability-based operations` as one route source among four. Deriving the control block rather than authoring it per operation is `C1 – Uniform interface to functions, distributions, and values` at the operation layer: `raw` and `method` mean the same thing everywhere because no author writes them. Passing `exact_only` and the two counts into a derived route gives a derived operation the controls a primitive one has, so a control set on the derived call takes effect where the computation happens (`C1 – Uniform interface to functions, distributions, and values`). Restricting the route selection of every call within the formula under `exact_only` makes the exactness of its routes a property of the whole body, rather than of the constituents its check could probe (`D1 – Mathematical fidelity`). Defining a derived operation by a formula gives its behavior a single definition, and leaving the stack to the engine (V.1), so that an operation differs only in its declarations, gives the execution of a call one as well (`D6 – Single source of truth`): a plain function and an operation fail at the same steps with the same errors. That every operation returns another tracked term is `D4 – Closed system of objects under operations`, and a route written over raw forms is `B2 – Representations only inside`, with both boundaries left to the engine.

## VI.1 — `pushforward`

### Contract

`pushforward(d, f, fixed_args)` returns the pushforward `f♯d`, which is the law of `f(X)` for `X ~ d`. The map receives one draw of `d` at a time at its open parameter, whatever that parameter's annotation is, so a parameter annotated `Any` or `Distribution` receives draws. Each draw is presented by V.5's rule for one draw, batch-valued draws included. The direct call `f(d)` instead passes a law whole to a parameter annotated with a law class or `Any` (V.5), so `pushforward` and the direct call agree for every map that takes draws. The law comes first, as for every operation on a law (VI.5), and `pushforward(K, f, given=s)` is therefore `pushforward(condition_on(K, s), f)` (VI.0).

The operand is a law. A `DistributionBatch` gives the batch of the pushforwards of its laws (VI.11). A value or a batch of values is refused at admission, and its map is applied by the direct call `f(v)`. Applied to a `ConditionalDistribution` without `given`, the pushforward acts on the event side, giving the kernel `s ↦ f♯K(s, ·)` with the same given spec. Construction retains the source and the map. Partial binding retains a kernel, and complete binding computes the ordinary pushforward through its evaluation rules. The generic sampling route builds its empirical atoms at that binding. A conditional sampler maps source draws directly, with the supplied key and sample shape, whenever the source supplies that capability.

The map `f` is a `Function` or a plain callable, which is wrapped under its name. `LinOp` is its linear subtype, and invertible maps claim `SupportsInverse`. Exactly one parameter of `f` has no default and is absent from `fixed_args`, and each draw binds it, whatever the draw's output names (V.5). The result carries the expression of the map's call on the law (II.4), so it displays as the map applied to a draw, as `f(mu ~ prior)`, and takes the map's `output_label`.

**Two applicability checks** run before any route is chosen (VI.0), and `expectation` shares them (VI.5). A call that fails one raises `ApplicabilityError` from the call and from `check`, so no route runs and nothing is drawn.
1. **The open parameter admits a draw.** Let `E` be the spec of one draw, `d.event_spec.term_spec`. Each declaration the open parameter carries must admit a value of spec `E`:
   - an input of the map's `input_spec`, to which `E` and the fixed arguments' specs conform in one dimension scope (II.1);
   - an operation's role, which must admit `E`'s kind;
   - otherwise its annotation, of which some arm must admit `E`. An arm that names a law class or capability admits a law, an arm that names a kernel class or capability admits a kernel, and an arm that names a term class admits a value of its kind. `None` admits nothing, and any other arm admits every draw, as `Any`, a value class such as `float`, and a raw class such as `jax.Array` do. No annotation admits every draw.
2. **The fixed arguments are constants.** Each is passed to the map as the direct call `f(x, **fixed_args)` presents it, on every route. A law the direct call would sample, or a batch it would sweep, is refused, while a law at a parameter whose annotation or role takes it as it is is a constant. The joint law of the map over `d` and a law `A` is the direct call `f(x=d, a=A)`, which draws the two as V.5 groups them.

**The route.** `pushforward` is primitive, and its one route, `evaluation_rules`, is the registry route over the evaluation-rule registry (V.7), whose rules the `method` control names. Where the direct call samples `d`, the rules run on `f` itself, so a rule keyed on the map's type, such as a `LinOp`'s, applies. Where the direct call would pass `d` whole, they run on a new `Function` whose one unannotated parameter receives each draw and calls `f` on it, keeping `f`'s label, output spec, and `max_evaluations`. A control set on `pushforward` applies to its call of the map, and a control left unset there keeps the map's own value, so `pushforward(d, f)` evaluates `f` at most the `max_evaluations` that `f` was constructed with. `include_inputs` is not passed on, since the law of `(X, f(X))` is the pushforward of a different map, and the direct call gives it.

**Linear maps push moments exactly.** The exact lazy linear-pushforward route (VII.4) delegates `mean` and `cov` whenever `d` supports them, since `𝔼[A X] = A 𝔼[X]` and `Cov(A X) = A Cov(X) Aᵀ`. If the caller explicitly selects an empirical approximation, its moments are those of that empirical law (II.7).

**Operator notation.** For a `LinOp`, `A @ B` composes two operators, and `A @ v` is the direct call `A(v)` for any other operand.

### Rationale

Applying a map to a value, a batch, or a law is one act in the direct call, which is `C4 – Function lifting`, but the pushforward is one mathematical operation, the law of the map's value at a draw. Giving it an operation of its own lets `expectation`, `mixture`, and the bootstrap distribution of a statistic name the law they mean, whatever the map's annotation, where an operation that dispatched as the direct call does would apply a map that takes laws to the law itself (`D1 – Mathematical fidelity`). Taking the law first, as every operation on a law does, gives the conditional form `given=` with no rule of its own (`C1 – Uniform interface to functions, distributions, and values`). The result is a tracked term that composes further (`D4 – Closed system of objects under operations`).

## VI.2 — `inverse` and `log_det_jacobian`

### Contract

Two operations read a map's inverse structure.
- `inverse(f)` returns the inverse map as a `Function`: `inverse(f)(y)` is the preimage of `y` under `f`. Its capability route is read through `is_invertible`; an unavailable inverse raises `ResolutionError`, while known noninvertibility is a mathematical domain error (II.7). The result is itself invertible, with `f` as its inverse, and it carries the Jacobian claim whenever `f` does.
- `log_det_jacobian(f, x)` returns the log-determinant of the Jacobian of `f` at `x`. Its capability route is `SupportsLogDetJacobian`. The reverse direction needs no second operation, since `log_det_jacobian(inverse(f), y) = −log_det_jacobian(f, inverse(f)(y))`.
- Each carries a capability route, and a numerical route, for example root finding for an inverse, may register beside it at its recorded fidelity, widening what resolves without changing either contract (VI.0).
- `inverse(f)`'s name is derived from `f`'s. Its output spec comes from `f`'s inputs: one input whose spec is a base spec gives an output at the root under that input's name, and any other inputs give the record of them, whose top-level names are the outputs, which every inverse route preserves.

### Rationale

Reparameterization moves in both directions between a constrained and an unconstrained space, so the inverse must be available to user code, and the operation form keeps the capability's implementer methods private, per this part's naming convention. Returning the inverse as a `Function` keeps the system closed (`D4 – Closed system of objects under operations`): the inverse evaluates, composes, and pushes forward like any map.

## VI.3 — `sample`

### Contract

`sample(d, sample_shape=())` draws from a distribution. `sample_shape` is a shape argument of integers (II.1), so `sample_shape=100` is `sample_shape=(100,)`.

- With `sample_shape=()` it returns a single draw, tracked, at the kind the event spec names (III.7); a non-empty `sample_shape` prepends batch axes and returns the tracked batch form of that kind, the leading dimensions on a level named `sample`. A draw and a batch of draws take the event's outputs (VI.0), so a batch of draws of a law over `mu` and `tau` is the batch of records whose fields are `mu` and `tau`, and its element spec is the law's event spec (II.5). A draw is labeled `outputs ~ label` (II.4), as `(mu, tau) ~ d`, and a batch of draws has the label of one draw. `sample.using(raw=True)(...)` is the draw detached: the kind's raw value for a single draw, which for a `Distribution`-valued draw is the law itself, and the storage view (II.5) for a batch:

| declared kind | one draw | tracked batch | `raw=True`, batched |
|---|---|---|---|
| `NumericArraySpec` | `NumericArray` | `NumericArrayBatch` | the stacked array, batch axes leading |
| `RecordSpec` | `Record` | `RecordBatch` / `NumericRecordBatch` | the nested mapping of raw columns |
| `DistributionSpec` | `Distribution` | `DistributionBatch` | an object array of the drawn laws |
| `ConditionalDistributionSpec` | `ConditionalDistribution` | `ConditionalDistributionBatch` | an object array of the drawn kernels |
| `FunctionSpec` | `Function` | `FunctionBatch` | an object array of the drawn callables |
| `OpaqueSpec` | `Opaque` | `OpaqueBatch` | an object array of the drawn objects |

- Sampling reads the returned kind from `event_spec.term_spec` (II.2), so an array and a one-field record remain distinct under every `sample_shape`.
- Sampling requires a concrete event spec and raises with the free dimensions named; in a fused conditional call, the given value binds them first.
- Under a non-empty `sample_shape` the key (V.8) splits by draw index, so the draws are jointly independent and reproducible together.
- An unnormalized law (III.8) samples through an approximate route that normalizes it with a method of the inference-method registry (VI.6), and provenance records the method.

### Rationale

Every draw is reproducible from its record (V.8), which is `C6 – Traceable and reproducible workflows`. Returning every draw as the tracked term of its declared kind serves `C1 – Uniform interface to functions, distributions, and values`: across laws, a draw is tracked and its type is fixed by the declaration, whatever kind that is. The `raw` opt-out is `B3 – Tracked forms out by default` at the sampling boundary: the wrapped, tracked draw is the default, and the bare value is an explicit request.

## VI.4 — `log_prob`, `unnormalized_log_prob`, and the density variants

### Contract

`log_prob(d, value)` returns the log-density of `value` under `d`. The value conforms to `d.event_spec.term_spec`, which fixes its kind: an array-valued event takes an array, as `log_prob(Normal("x", 0.0, 1.0), 0.3)`, and a one-field record event takes a record or a mapping, which is laid out in the declared order (III.5). The value is read by the law's paths. A joint rebuilds each factor's draw before delegating (IV.2).

- `log_prob` requires `SupportsLogProb` and returns the *normalized* log-density.
- `unnormalized_log_prob` requires only `SupportsUnnormalizedLogProb` and returns the log-density up to an additive constant, as inference against an unnormalized target needs.
- A scored value binds any symbolic event dimensions for that call only, so a polymorphic law scores datasets of different sizes.
- `prob` and `unnormalized_prob` are derived operations (VI.0), defined by the formulas `prob = exp ∘ log_prob` and `unnormalized_prob = exp ∘ unnormalized_log_prob`; a family with a stable density may register a direct route above the derived route.
- `random_log_prob(M)` and `random_unnormalized_log_prob(M)` take a random measure `M` (VII.5) and return the law of `x ↦ log D(x)` for `D ~ M`, a `RandomFunction`, through capability routes on `SupportsRandomLogProb` and `SupportsRandomUnnormalizedLogProb` (III.8). The density at a point is that random function called at the point (VII.5), so neither takes a value. The returned law carries its own declaration.
- A score of `d` is labeled `log` followed by `d`'s notation, as `log prior(mu)`, and a density by `d`'s notation, as `prior(mu)` (II.4). The score's output is named by the operation, as `log_prob` (VI.0). A law at the value lifts the call, and the law of the scores displays as the operation applied to its arguments, with the scored law by its notation, as `log_prob(prior(mu), mu ~ q)`. That law declares the output one score declares, `log_prob`.

### Rationale

Splitting `log_prob` from `unnormalized_log_prob` lets each law claim only the density it provides (`D1 – Mathematical fidelity`): a distribution that knows its normalizing constant offers the true density, while one that does not still serves inference, which needs the density only up to a constant. Deriving `prob` from `log_prob` gives the density one definition, which a family overrides only where it has a better one (`D6 – Single source of truth`).

## VI.5 — Distribution functionals: `mean`, `variance`, `standard_deviation`, `cov`, `quantile`, `expectation`

### Contract

The distribution functionals are operations that summarize a distribution by a deterministic value.
- `mean(d)` and `variance(d)` return an event-typed value, that is, a value shaped like a draw. Each applies to any event type that supports it: a random function has a mean function and a pointwise variance function, while a random measure has a mean, which is the marginalized law, but in general no event-typed variance. The result is wrapped at the law's declared event kind: a `Record` whose fields are the distribution's outputs for a record-drawing law, and a term of the drawn kind otherwise, so a random function's mean is a `Function`.
- `standard_deviation(d)` returns the square root of `variance(d)`, coordinate by coordinate, as a value of the same kind: a random function's is its pointwise standard deviation function. It applies wherever `variance` does.
- `cov(d)` requires a numeric draw and returns a covariance operator over the *flattened* draw, a `(vector_size, vector_size)` `LinOp`, since covariance couples distinct coordinates. Its input term spec and `output_spec.term_spec` are the distribution's numeric event term spec; application checks these term specs, whatever the output names (III.4, V.5).
- `quantile(d, q)` requires a numeric draw. It takes a probability `q ∈ [0, 1]` or an array of probabilities and returns the quantile at each, computed per coordinate for a multivariate draw. Its result's output spec is event-kind-directed like any other: a single probability returns the event's own kind, for example a `NumericArray` for an array-drawing law, and a plural `q` adds a level named `quantile` over those, giving the matching batch, whose outputs are the event's, as the single probability's are. Whether `q` holds one probability or several is known before execution, so planning reads it (V.6).
- A batch-valued event is numeric when its elements are. Its mean, variance, and quantiles are elementwise: each is a batch on the event's levels whose element at each position summarizes the draw's element there, and a plural `q` puts the level `quantile` before the event's levels. `cov` flattens such a draw over the event's batch axes and then the element's own axes.
- `expectation(d, f)` returns `𝔼[f(X)]`, shaped by the output of `f`, for any event type `f` accepts. The result is wrapped at the kind `f`'s output spec names, a `Record` for the usual record output.

A moment of the event's kind keeps the event's outputs, paths, and kind (VI.0), so the mean of a law over the record `population` holds `population/mu`. A moment is labeled by the moment of a draw (II.4), as `𝔼[(mu, tau) ~ d]`, `𝕍[...]`, `√𝕍[...]` for a standard deviation, `ℂ[...]`, or `ℚ[...]` for a quantile, and `expectation(d, f)` is labeled `𝔼[f(mu ~ d)]`. It derives the result's term specs, support included, so the mean of a Bernoulli event is supported on the unit interval. A moment that diverges returns `inf` (VII.1), and the support derived for a moment that can diverge admits it, so the half-Cauchy mean's support is (0, ∞] and that of a variance or a standard deviation is [0, ∞]. `expectation` declares the mean of the integrand's output spec, so it takes the integrand's outputs, and an integrand that declares no output spec leaves it to the returned value. `cov` is declared under the output name `cov` and takes the operator codomain from the event spec. A measure-valued event's mean is a law, so it has its event's outputs. A higher-order result, such as a function-valued mean, keeps its inner output spec separate from the operation's own output name.

`mean`, `variance`, `cov`, and `quantile` each carry a guarded capability route on the matching protocol, `SupportsMean` for `mean` and so on, and a Monte Carlo fallback on the event kinds where the required averaging is defined. Sampling availability alone does not establish moment existence: methods state their assumptions, and known undefinedness is handled as in II.7. The number of draws of a fallback is the `num_samples` control (V.2), which the result's provenance records. A distribution that claims neither the capability nor sampling raises `ResolutionError`. The fallback route is defined per event kind. A numeric event averages draws coordinatewise, and its estimate of a moment is the moment of the draws' empirical law (VII.2). For a function-valued event the fallback returns a lazy function: its mean at a point is the average of the sampled callables there, and its variance the pointwise sample variance. A measure-valued event's mean is the finite mixture of the sampled draws, which is the Monte Carlo estimate of the mean measure. `cov` and `quantile` have numeric-only fallback routes: the sample covariance, returned as a `DenseLinOp`, and the per-coordinate empirical quantiles.

**Output by output.** `mean`, `variance`, and `quantile` are computed coordinate by coordinate, so for a law `d` over a record, the summary of `d` is the record of the summaries of the coupled marginals `d[B]` (III.7) over any partition of its outputs into blocks `B`. Each of the three carries the structural route `by_output`, which realizes that equation, ranked after the capability route and before the fallback. The route forms its blocks by resolving the operation on each output's coupled marginal, so the coupled marginal decides each output's exactness as III.8 derives it. An output for which the operation selects an exact route is a block of its own, and the other outputs form one block, so they share one set of draws. Each block is resolved and run as the operation would run a call on its coupled marginal, under the call's controls with `method` unset and `exact_only` that of the ranked candidate, and inside the call itself, so its draws are the call's own and the call replays as one (V.8). The route applies when the event is a record of at least two outputs, at least one of which has an exact route, and the remaining block has a route. Its exactness is delegated (VI.0): the result is exact when every block's route is, and approximate otherwise, so a dependent joint's mean is approximate while the means of its root factors in it are exact. The result's provenance names, for each block, its outputs, the route that computed it, and that route's exactness (II.7). Under `exact_only`, a law with an output that has no exact route raises `ResolutionError`, naming the outputs that have one and the call on a coupled marginal that computes one exactly. The route splits the outputs only, so a group is one block, summarized whole by the route its own coupled marginal selects. `cov` carries no such route, since its blocks across outputs need joint draws, and `expectation` carries none, since its integrand couples outputs. `standard_deviation` carries no route of its own, and the `variance` its formula calls selects `by_output`, so its outputs keep the exactness of their variances.

**`expectation` as a derived operation.** `expectation` is a derived operation (VI.0), defined by the formula `expectation(d, f) = mean(pushforward(d, f))`, as `prob = exp ∘ log_prob` is in VI.4. By the change of variables this is the integral `∫ f(x) d(dx)`, and the integrand receives draws of `d` on every route, whatever its parameter's annotation. `expectation` declares the two applicability checks of `pushforward` (VI.1), so its domain is the pushforward's on every route: an integrand whose declared input cannot take a draw of `d`, or a law or a batch in `fixed_args`, raises `ApplicabilityError` before any route is chosen. On every route a draw is passed to the integrand as V.5 presents one, so a record atom of a law with a closed form arrives as a `Record`, as a record draw does, and the fixed arguments arrive as the direct call presents them.

`expectation` carries two routes:
1. `closed_form`, its capability route on `SupportsExpectation`, which is the exact case (III.8);
2. `pushforward_mean`, which is the composite derived route (VI.0): an evaluation rule of `pushforward` builds `f♯d`, and a route of `mean` averages it.

`pushforward_mean` ranks once, as a derived route: among the approximate routes, so `closed_form` precedes it whenever it applies, and under `exact_only` among the exact routes, with both the rule and the mean restricted to exact routes. A call that needs Monte Carlo in either step under `exact_only` therefore raises `ResolutionError`. The mean's route is chosen only once the pushforward exists, so without `exact_only` the call's report states the call approximate, as for every route whose exactness is open, and defers the mean's check. The run records the exactness of both steps, which is exact only when both are. `expectation` shares with `pushforward`:
- controls: the controls `pushforward` passes to the map, such as `executor`, except that the map's `max_evaluations` is the call's `num_samples` (VI.0), which the mean's Monte Carlo route also reads;
- method names: the names of the evaluation rules, which select the rule as for `pushforward`, as `method="sampling_lift"` or `method="pushforward_mean/sampling_lift"`;
- `fixed_args`: the other parameters of a map with several, which are constants, as for `pushforward`;
- randomness: the workflow-owned draws of a sampling route (V.8).

The Monte Carlo expectation is therefore the mean of the sampling lift's empirical law, which requires sampling from `d` and integrability of the outputs of `f`, whatever the event type of `d`. An integration rule registers once, as an evaluation rule (V.7), and then serves `pushforward` and `expectation` alike. The arguments of `expectation` are `(d, f)`, law first, as for `mean(d)` and `quantile(d, q)`. The value and the output spec of an undeclared integrand's expectation do not depend on the integrand's name or label, which change only the result's label. A law in `fixed_args` has two meanings, which other calls state: `mean(function(f)(x=d, a=A))` integrates over both laws, and the direct call of a function that calls `expectation` at a draw of `A` gives the law of the expectation at that draw.

**`standard_deviation` as a derived operation.** `standard_deviation` is a derived operation (VI.0), defined by the formula `standard_deviation(d) = sqrt(variance(d))`. The square root is taken coordinate by coordinate: of each leaf of a numeric value, and of a function's values at each point. Its one route is the body route of its formula (VI.0), whose `derived_route_check` is `variance`'s check, so every call takes the routes of `variance`, and the call records the exactness of the route `variance` selects. The derived route passes the call's `exact_only`, `max_evaluations`, and `num_samples` to `variance`, as **Controls inside a derived route** (VI.0) states, so `standard_deviation.using(exact_only=True)(d)` returns an exact result whenever `variance` has an exact route for `d`, and the call's provenance records the `num_samples` of a Monte Carlo variance. A body route does not pass `method` on, so a route of `variance` is chosen on `variance.using(method=...)`, as the error of a `method` naming one says. Its outputs and paths are the event's (VI.0).

**A table of summaries.** `summarize(d, *statistics, interval=0.8, mcse=False)` computes several functionals of one law and returns them in one `NumericRecord` nested by statistic. It is a function over the operations rather than an operation, since it composes their results and has no implementation to select among. Each top-level field is the term that its statistic returns for `d`, with that term's provenance, and a law over a base value holds the term under the name of its output at the root. Each path of the record is a statistic followed by an event path, so `summarize(d)["sd/mu"]` is the standard deviation of `mu`. The record is labeled `summary of` followed by the law's notation, and it displays as a table (II.4). Its columns are:
1. **the defaults:** `mean`, `sd` from `standard_deviation`, `median` from `quantile(d, 0.5)`, and the two ends of the central interval of probability `interval`, named by their percentages, as `q10` and `q90` for `0.8`;
2. **given statistics:** the callables of `statistics` replace the defaults. A callable is named by its label, as an operation is, or by its `__name__`, and the name must be an identifier, and a mapping names each of its callables by its key;
3. **Monte Carlo standard errors:** a column `mcse_<name>` for each statistic, when `mcse=True`. A statistic of an MCMC posterior estimates the posterior the sampler targets, so its error is ArviZ's for the statistic's kind. For another law the error is `0` at a path that an exact route computed, `sd / sqrt(n)` for a mean from `n` draws, and `nan` where no estimator applies without the draws;
4. **MCMC diagnostics:** for a posterior whose draws lie on the levels `chain` and `draw`, `r_hat`, `ess_bulk`, and `ess_tail` over the same paths, computed as `add_mcmc_diagnostics` computes them.

The record's provenance names `summarize` and records, under `computations`, how each statistic column was computed, which the line below its table states. A statistic of an MCMC posterior reads `MCMC` with the posterior's chains and draws, as `MCMC, 4 chains of 1,000 draws`, since it is exact for the draws and a Monte Carlo estimate of the posterior they target. Another statistic reads as its provenance records it (II.4).

A `DistributionBatch` gives a `NumericRecordBatch` of the summaries of its laws, on the batch's levels, whose provenance records each law's descriptions. A column that only some laws give, as `r_hat` of a batch that mixes MCMC posteriors with other laws, holds `nan` for the others.

### Rationale

A mean is defined whenever draws can be averaged, which is coordinatewise for arrays, pointwise for functions, and setwise for measures, while an event-typed variance requires the second moment to be a value of the event type, which fails for a general random measure. `A ↦ Var(ξ(A))` is additive only when disjoint regions are uncorrelated, and never for a random probability measure, whose fixed total mass forces negative correlation, so a random measure's second-moment structure is a covariance over pairs of sets, which is the analog of `cov`. Gating `mean` and `variance` by capability rather than by a numeric event is therefore `D1 – Mathematical fidelity`, as is keeping `cov` and `quantile` numeric-only, and `cov` returns a flat operator because it couples coordinates the event's field structure keeps separate. The closed-form-or-Monte-Carlo split realizes `C3 – Computational detail hidden by default, available on demand`: a distribution that can give an exact moment does, and a sampling fallback returns an approximate one under its stated assumptions. Computing a record's summary output by output gives each output its best route, so an exact output stays exact inside an approximate result (`D1 – Mathematical fidelity`), and leaving each output's exactness to its coupled marginal keeps that decision in one place, the coupled marginal's capability derivation (`D6 – Single source of truth`). Defining `expectation` by `mean(pushforward(d, f))` gives the integral one definition over the operation it means, so a rule that integrates registers once and serves both operations (`D6 – Single source of truth`), and an integrand receives draws whatever its annotation (`D1 – Mathematical fidelity`). Defining `standard_deviation` by `sqrt(variance(d))` likewise leaves `variance` the one place a second moment is computed, so a variance route added later serves both (`D6 – Single source of truth`). Giving a summary the event's outputs and paths makes a path the address of a coordinate, and the label states the value's meaning, as `𝔼[mu ~ d]` (`C5 – Naming for unambiguous meaning`). A summary computed coordinate by coordinate under the event's names is read and bound as a draw is, so a plug-in estimate composes with the operations a draw composes with (`C1 – Uniform interface to functions, distributions, and values`). `summarize` returns the terms the operations return, so each summary is computed once and its provenance states its route and its exactness (`D6 – Single source of truth`), and its table is the display of any record of that form, so a record of summaries built by hand reads the same way (`C1 – Uniform interface to functions, distributions, and values`).

## VI.6 — `condition_on`

### Contract

`condition_on(d, given)` fixes some fields of a distribution or conditional distribution and returns the resulting distribution, normalized (III.8).

**The `given` argument.** `given` is keyed by path: a `Record` read by its paths, or a mapping from paths to values, with every value conforming to the spec at its path, all of them in one dimension scope (II.1). Each path must name either an *input*, that is, a name in the `given_spec` or a path into a record input, or a path of the *event*, nested paths included; any other path is an error. A path may name a group, and its value is then a record checked against the group's spec: to condition on the whole group `g`, name it, as `{"g": value}`, since a bare `{b, c}` would be read as the top-level names `b` and `c`. Binding a node of a record input supplies that node alone: for an input `g` of spec `{b, c}`, binding `g/b` leaves the input `g` of spec `{c}`, now required, since a default is one whole value. Conditioning at a path inside a group fixes that node, and the rest of the group stays in place. Conditioning is stated in terms of paths alone: the derived factor graph decides which case below applies and carries it out.

**The routes.** `condition_on` resolves across three of the four route sources (VI.0): structural routes that bind an input or slice the factor graph, a capability route for each of the two conditioning capabilities, which a `ConditionalDistribution` claims since `_condition_on` is its required primitive, and a registry route through the inference methods, which normalizes a conditional known only up to its normalizing constant. Binding an input applies the kernel, and fixing a path of the event conditions the law; the factor graph selects among the routes for the second, which all compute one conditional. The controls are resolved before a route is selected, so `exact_only` excludes the approximate conditioning capability and the registry's approximate methods alike, and it stays a control, apart from the selected route's arguments. Exactness is compared across route sources, as VI.0 requires: an exact registered method outranks the approximate conditioning capability, so no approximate route runs while an exact one applies. Feasibility is checked from the declarations and available capabilities:

- **An input, so bind it.** Binding an input that the object conditions on and does not output returns a smaller `ConditionalDistribution`, or an ordinary `Distribution` once all required inputs are bound. Binding evaluates the kernel and involves no inference. It is exact unless the kernel claims `SupportsApproximateConditioning` (III.9), as an amortized posterior does, so `exact_only=True` excludes binding the inputs of such a kernel. The normalization stage below applies to the result of either. For example, binding a regression model's covariates gives the data-ready likelihood, and binding a Stan program's data gives the program's unnormalized posterior over its parameters.
- **An event path with an exact slice.** Fixing it leaves a conditional that can be assembled from available normalized factors and exact local conditioning operations. For example, in `p(y | beta) p(beta)`, fixing `beta` leaves the existing kernel `p(y | beta)`. A multi-field factor must support the required internal conditioning, since its fields may be dependent. Upstream or independent fields commonly admit this route, and the factors' conditioning capabilities decide its feasibility.
- **An event path requiring Bayes' rule.** Fixing it leaves a likelihood contribution involving unconditioned variables, requiring an exact conditioning method or inference. For example, fixing `y` in `p(y | beta) p(beta)` leaves the likelihood `p(y | beta)` over the unknown `beta`, even though `y` has no downstream dependents. An available `SupportsExactConditioning` route may compute the conditional exactly, as a `MultivariateNormal` does, and a `SupportsApproximateConditioning` route may answer it with a stand-in, as a pre-trained amortized posterior does. Otherwise the exact stage forms the unnormalized conditional, and the normalization stage selects an inference method for it (below), raising `ResolutionError` when none applies.

When `given` names several paths, the cases combine: the exact steps, binding the inputs and slicing, are applied first, and Bayes' rule runs on what remains. Path classification is computed once, on the graph with every conditioned path marked, so the outcome does not depend on the order the paths are listed. A path of the event can be conditioned on before the inputs are bound. The result stays conditional on the inputs that no factor feeds, with the conditioning on event paths applied within each value of them, so the result binds like any other `ConditionalDistribution`, and in the exact cases the two orders agree: conditioning on an event path and then binding the inputs yields the same distribution as binding the inputs first. An inference algorithm that cannot condition at a nested path refuses, with a message naming the algorithm rather than the path. An approximate route records its assumptions and fidelity instead of promising equality in law, and a conditional at a value is the version the selected route defines, since almost-everywhere uniqueness does not fix it pointwise. A `ConditionalDistribution` result is normalized per value: the normalization stage runs for each law it yields once its given is bound, lazily or through a method that supports amortization. The kernel that normalizes per value claims `SupportsConditionalSampling`, since every inference result samples, and it claims `SupportsApproximateConditioning` unless only exact methods normalize it, because evaluating it otherwise runs an approximate method (III.9).

**The result's event.** On every route, the result's event spec is `d.event_spec.without(*fixed)`, where `fixed` are the event paths the given fixes (II.2): conditioning removes the nodes it fixes and keeps every other node at its path, while binding removes inputs. So conditioning a law over `{mu, sigma}` on `mu` gives a law over `{sigma}`, and conditioning `d` on a draw of `d[["mu"]]` fixes `mu` and nothing else. The result is one factor in a later merge, whatever route computed it, with any optional input that remains fixed at its default (IV.2).

**The result's label and fixed paths.** The result keeps the label of the law it conditions and records the paths it fixes (II.4):

1. **The label.** The result keeps the label of `d`, so `condition_on(model, {"y": data})` is labeled `model`, and a kernel applied at its inputs keeps the kernel's label. A joint whose given fixes the whole events of the factors upstream of the rest leaves the other factors at the given values, and the result takes their labels joined with `·`, so `condition_on(lik & prior, {"mu": 0.5})` is labeled `lik`.
2. **The fixed paths.** The result holds the paths the given fixes, after any paths `d` holds, and its signature lists them after `;`. A law used as the given fixes its outputs, and a kernel applied at some of its inputs holds those inputs fixed while its other inputs stay inputs. An input with a default that the given leaves out stays an input with its default, as `n0=50.0`, also once the result is a law, since the law is the kernel's at that default; a given that binds it moves it to the fixed paths. A given that names no path fixes none. The result of every route holds them, an inference method's included.

For `model = (lik & prior).with_label("model")`, with `lik` over `y` given `mu` and `prior` over `mu`, a kernel `glm` over `y`, and a kernel `counts` over `y` given `K` and `r` and the optional input `n0` whose default is `50.0`:

| Call | Label | Notation |
|---|---|---|
| `condition_on(model, {"y": data})` | `model` | `model(mu; y)` |
| `condition_on(model, {"mu": 0.5})` | `lik` | `lik(y; mu)` |
| `condition_on(glm, {"beta": b})` for `glm` given `beta` | `glm` | `glm(y; beta)` |
| `condition_on(glm, {"beta": b})` for `glm` given `beta` and `sigma` | `glm` | `glm(y \| sigma; beta)` |
| `condition_on(counts, {"K": k, "r": q})` | `counts` | `counts(y \| n0=50.0; K, r)` |
| `condition_on(counts, {"K": k, "r": q, "n0": n})` | `counts` | `counts(y; K, r, n0)` |

A batch of givens labels its batch of laws by the same rule, read at the paths each element fixes, and each law of the batch holds those paths fixed. The result has its event's outputs (VI.0), which are the conditional law's.

**Normalization.** Bayes' rule multiplies a likelihood by a prior and normalizes the product, and `condition_on` completes both steps, so its result is normalized (III.8). It resolves in two stages:

1. **The exact stage** computes the conditional by the cases above. Where no exact route conditions on an event path, it forms the **unnormalized conditional**: the law of the unconditioned fields whose unnormalized log-density is the joint's at the given values. That law also carries the joint and the given values, for methods that simulate rather than evaluate a density.
2. **The normalization stage** returns an exact-stage result that is normalized. Otherwise it passes the result to the inference-method registry as the **target**, and the selected method returns a normalized law.

Binding a normal kernel's given yields a normal law, which the normalization stage returns as it is. Binding a Stan program's data yields the program's unnormalized posterior, which a method such as Stan's NUTS normalizes. `method="unnormalized"` selects the exact stage alone and returns its result, normalized or not, for a caller that runs its own algorithm on the target, and a caller selects it only by name. `exact_only=True` admits an exact method that normalizes the exact stage's result, and when the result is unnormalized and no exact method applies, the call raises `ResolutionError`, naming `method="unnormalized"`. `check` reports both stages, the exact route and the normalization method, and provenance records both.

`condition_on` always binds the supplied value as the field's fixed value, whatever the value's type; the mixture `∫ K(s, ·) μ(ds)` over a mixing distribution is requested explicitly through the separate `mixture` operation.

**The inference-method registry.** The normalization stage is dispatched through the **inference-method registry**, a `UnaryDispatchRegistry` whose methods are inference algorithms such as MCMC or variational families. A method is exact when its result is the conditional law itself, and the registry tries the exact methods first (II.7). For example, `empirical_reweighting` conditions a joint whose prior is an empirical law: the posterior keeps the prior's atoms, each weighted by its prior weight times the likelihood of the given values there. A method takes the target alone, whose data are already bound. The target is an **inference target** (`InferenceTarget`), and a method reads the model through its interface: the model as `joint`; the observed values as `given`, which are the values of the conditioned fields, the data a program takes, or nothing for a law that is itself unnormalized and passed as it is; and the model's prior, likelihood, and observed value as `factors()` when the model is a merge of them. The target is itself a law, so its unnormalized density is read as any law's is, and a method that works on vectors reads a value of the parameters through the `to_vector` layout of the event's values (II.3). A method's `execute` receives the run's random key as its keyword `key`, which the registry derives for each run from the enclosing workflow scope (V.8), so a seed reproduces the run; the method draws all of its randomness from that key, and one that draws nothing ignores it. `make_posterior` builds a method's result from chains of `to_vector` coordinates of draws: the result declares the target's event, and its provenance names the target, or the law itself for the target of conditioning a law on nothing, since conditioning on nothing is the identity. A method's feasibility check reads the interface it requires, such as a target density, a backend program handle in the target's joint, or a prior and likelihood to simulate from, so each method requires only the interface it uses. A method that runs its target inside a JAX program, such as a gradient-based MCMC method or elliptical slice sampling, reads the target's traceability (V.13) instead of tracing the target itself, so one `condition_on` traces nothing to select its method. A law's or a kernel's traceability is that of its density, or of its sampler when it has none, and it derives it from its parts and keeps it: a family is traceable, a law built from functions takes the outcome of the probe of its density, or of its sampler when it has none, the kernel of a function takes the traceability of the law its function returned when the kernel was built, a joint takes the least of its factors', a view or a rename takes its parent's, and any other law or kernel probes its density once, at a stand-in of one value, or its sampler at one key when it has no density. A method that also runs its target in Python, such as random-walk Metropolis, runs a traceable target in a compiled loop and any other in a Python loop. A method whose chain state is the target's flat vector starts its chains inside the target's support: from draws of the factors of the target's joint that sample, restricted to the unconditioned fields, and from a `Uniform(-2, 2)` box only for the fields that no sampling factor draws. For example, in the joint of a learned likelihood ratio and a prior, only the prior samples (VII.7). Method parameters such as warmup and approximation budgets are set through the `method_options` control (V.2), which the selected method validates and a conditioning capability receives as the keyword options of `_condition_on` (III.9); the plan may read them when they determine a result's batch shape or other static structure.

**Fidelity.** Inference uses the shared contract of II.7. A finite MCMC posterior is approximate; the chain's invariant target and convergence assumptions are method guarantees. Exact conditioning routes return a representation of the conditional law itself. The selected route's local fidelity and target are recorded in provenance.

### Rationale

Applying a kernel to its given and conditioning a law on a path of its event are one operation because both fix named quantities (`C1 – Uniform interface to functions, distributions, and values`), and the distinction between them is kept (`D1 – Mathematical fidelity`) while dispatch chooses the available method (`C3 – Computational detail hidden by default, available on demand`, `D3 – Capability-based operations`). A normalized result completes Bayes' rule, whose normalization the inference methods carry out. A user therefore receives the posterior from one call, whether the conditional is exact or approximated (`C3 – Computational detail hidden by default, available on demand`). Separating the exact stage from normalization records each step's fidelity (`C6 – Traceable and reproducible workflows`). Recording the fixed paths gives a conditional a notation of its own, as `model(mu; y)`, because the result keeps its law's label and the label alone would name the posterior as it names the model (`C5 – Naming for unambiguous meaning`).

## VI.7 — `joint`

### Contract

`A & B` (IV.2) is the merge operator. Its result's `provenance` records `&` as the operation, with `A` and `B` as its parents in the order written, so a law computed from the joint traces back through the merge to each operand's lineage. Its label follows the labeling rule of IV.2.

**The realigning `joint` form.** `&` feeds an input from the output of the same name. The `joint(A, B, **align)` operation renames first and then merges as `&` does, so it is equivalent to `A & B.rename(**align)`. For example, a likelihood that conditions on `slope` can be combined with a prior where the slope is called `beta` with `joint(lik, prior, beta="slope")`.

### Rationale

Realignment is an exact rename: `rename` returns the same law under new names, so `joint` connects mismatched factors without altering their joint law (`D1 – Mathematical fidelity`).

### Open points

- *The `align` contract.* `align` pairs are `rename` pairs, path-valued targets included (II.6), so realignment can promote a nested field to a name that an input matches. The remaining details, such as freshness and injectivity requirements on the new names, are deferred: the question is subtle, and implementation experience should inform the decision.

## VI.8 — `marginal` and `factor`

### Contract

Two operations read the parts of a structured or factored distribution, and the coupled marginal `d[paths]` is indexing (III.7).

- `marginal(d, paths)` returns the **standalone marginal** at a path or a list of paths: a new `Distribution`, drawn independently of `d` and of its coupled marginals (III.7). It is the law of the coupled marginal `d[paths]` when computed exactly, and an empirical approximation of that law on the Monte Carlo fallback route below. It may hold `d` only to compute from it, and it neither conditions nor scores through `d`. A path and a list of paths are read by form (II.6), so `marginal(d, "g")` has the group's children as outputs, `marginal(d, ["g"])` has the one output `g`, and a list keeps each node at its own path, in the law's order. Its event spec is `d.event_spec[paths]` (II.2), and a group's marginal is the marginal of the factor that holds the group, at the group. It carries a capability route on `SupportsMarginals` guarded at the requested paths (IV.1), and a Monte Carlo fallback route through `_sample`, selecting the draws at the paths and returning an empirical marginal. When the capability is absent, or the path has no exact route within it, the fallback resolves, and a distribution that cannot sample either raises `ResolutionError`. A marginal over the whole events of some factors of a joint, none of which conditions on a field outside them, is the law of those factors. One factor keeps its own label, so `marginal(location & scale, "tau")` displays as `scale(tau)`. Several display as a product without a label (IV.2), in the law's order whatever the order of the list, so `marginal(model, ["a", "b"])` and `marginal(model, ["b", "a"])` both display as `a(a)·b(b)`. Any other marginal integrates a factor out, as the prior predictive does, and keeps the law's label, so `marginal(model, "y")` displays as `model(y)`, as the view `model["y"]` does (III.7). The marginal holds the paths the law holds fixed (VI.6). Whatever route computes it, the marginal is one factor in a later merge, with any optional input that remains fixed at its default (IV.2), so a view and a marginal of one joint merge alike.
- `factor(d, output_name)` returns the complete building-block **factor** that gives the named output of a joint (IV.1), under the factor's own name, either a `Distribution` or a `ConditionalDistribution` for a dependent edge. Its capability route is `SupportsFactors`, so a distribution that exposes no factors raises `ResolutionError`.

With `given=`, `marginal` tries the kernel's exact conditional marginal first, then the exact marginal of the law obtained by exact conditioning, then Monte Carlo through the conditional sampler. Conditional delegation with no exactness restriction remains the last route. The exact conditional candidate evaluates the kernel only during execution and retains that law for the selected marginal. A check reports this choice unresolved when it needs the bound law. Execution uses feasibility reports to choose a route; an execution error propagates. An explicit `method="monte_carlo"` selects sampling directly.

The empirical marginal's atom axis has the level name `sample`. If the selected event is itself a batch, underscores are appended to that name until it differs from every level of the event. The event's own levels and event spec are preserved.

### Rationale

Exposing them as named operations rather than as indexing separates the standalone marginal from the coupled marginal `d[paths]`, which keeps the dependence between the parts of one law (`D1 – Mathematical fidelity`). Dispatching `marginal` on `SupportsMarginals` opens the standalone query to any distribution that knows its marginals, factored or not. Gating `factor` on `SupportsFactors` ensures that only a distribution with an explicit factorization offers it (`D3 – Capability-based operations`).

## VI.9 — `mixture`

### Contract

`mixture(K, mixing)` returns the mixture `μK = ∫ K(s, ·) μ(ds)`: the law of `T` for `S ~ mixing` and `T ~ K(S, ·)`, where the mixing distribution's outputs feed the kernel's inputs by name (IV.2).

`mixture` is a **derived** operation (VI.0). Let `J = K & mixing` and let `r_K` select the kernel's outputs from a joint draw and rebuild the kernel's draw using `K.event_spec` (II.2). Its formula is `mixture(K, mixing) = pushforward(J, r_K)`, returned as a standalone law. The internal projection may first marginalize onto those outputs when an exact marginal route exists. Rebuilding the draw is essential: an array-valued kernel stays array-valued, and a one-field record stays a record. The result carries the kernel's event spec unchanged.

The formula's feasibility and fidelity follow from the merge and the selected projection route. A direct route may realize the same law more efficiently. Finite mixing admits an explicit `MixtureDistribution`, closed Gaussian cases stay Gaussian, and the generic sampling route gives an empirical approximation. Inputs that no factor feeds remain inputs, as the merge specifies (IV.2); the formula applies within each value of them. Complete binding retains the finite mixture route when mixing is empirical. Conditional sampling projects draws of the merged kernel directly, so the returned kernel can be a dependent factor of another joint.

### Rationale

Leaving the integral out of `condition_on` keeps conditioning single-valued, since a supplied value always binds and `μK` is always asked for by name, so neither call has a data-dependent meaning (`C1 – Uniform interface to functions, distributions, and values`). The name is the result's mathematical name: `μK` is the mixture of the kernel family with mixing distribution `μ`, which covers predictive, compound, and state-propagation uses without privileging one (`C5 – Naming for unambiguous meaning`, `D1 – Mathematical fidelity`). Defining the operation as the rebuilt kernel output of the merged joint adds no second semantics: one formula ties it to operations already fixed, so every behavior has a single source (`D6 – Single source of truth`).

## VI.10 — `convert`

### Contract

`convert(d, target)` returns a distribution of the requested class or satisfying the requested capability protocol (III.8). The result keeps `d`'s label and the paths `d` holds fixed, unless `d` carries its constructor's default label alone, which names only `d`'s class, and then the result takes its own default label (II.4). Its registry route uses the converter plan and event-preservation contract of IV.3. The result's event spec is the source's, support included. Under the converter option `check_support=False`, which `method_options` passes, planning leaves each support open, so the result has the converted law's support (IV.3). `using(method=..., exact_only=...)` controls selection, and provenance records the selected converter's local fidelity. A converter that draws takes the call's `num_samples` (V.2) unless its own option of that name is given, and provenance records the number of draws. A source already satisfying the target needs no numerical conversion and returns under fresh identity. Argument normalization plans the same conversion; execution constructs it and records it by the same contract (V.4, V.9). Converting an unnormalized law (III.8) to a normalized target, such as `EmpiricalDistribution`, selects a method of the inference-method registry (VI.6).

### Rationale

Exposing conversion as an operation gives a change of representation the same record as any other result (`C6 – Traceable and reproducible workflows`), and routing it through the registry keeps the set of convertible pairs open to registration (`D2 – Generality first`).

## VI.11 — Batched operations

### Contract

Every operation lifts to a `Batch` by mapping over its elements, which is the elementwise sweep applied to the operation itself, and an operation that mints a level names it after itself (II.5).
- `sample` over a `DistributionBatch` returns a **nested** batch, the outer level ranging over the laws and the inner over each law's draws: `sample(d_batch, sample_shape=(S,))` has `axis_groups` `(*d_batch.axis_groups, (S,))` and appends an inner draw level named `sample`, so iterating it visits one law's batch of draws at a time, which is a `RecordBatch` for a record-drawing law and the batch form of the declared kind otherwise. `log_prob` maps elementwise to the batched densities, with the batch axes preserved.
- A moment over a `DistributionBatch` returns a batch of the corresponding values, such as a `LinOpBatch` for `cov`. A multi-level query nests the same way: `quantile(d_batch, q)` keeps the laws on the outer level and adds an inner level named `quantile` for the probabilities in `q`.
- **Alignment.** A binary operation matches the operands' levels **by name**: a level in both must have broadcast-compatible shapes, with size-1 broadcasting; a level in only one operand broadcasts across the other; and an outer product is requested by explicit reshaping rather than implied. Because every level is named, alignment is by name alone, and two levels meant to correspond under different names are aligned by renaming one with `with_level_names` first, as `joint` realigns names for a merge. So a flat batch of values on a `laws` level scores against the `laws` level of a nested sampling result. `given=` accepts a `RecordBatch` and yields the `DistributionBatch` of conditioned laws.
- Two operands are exempt from batch lifting: the operands of a merge of laws (`&` and `joint`) and the map operand of `pushforward` and of `expectation`, which are consumed as objects rather than swept.
- Two batches merge element by element with `&` when their levels are equal (II.5), so a batch of values named `x` and one named `y` over the level `sample` give a batch of records `{x, y}`; batches over unequal levels are refused with `ValueError`.
- Batched application resolves through the evaluation-rule registry (V.7).
- An operation applied to a batch whose elements lack the required capability raises the `ResolutionError` a single element would.

### Rationale

A batched operation is the elementwise sweep applied to an existing operation, so a `Batch` supports exactly the operations its elements do (`D3 – Capability-based operations`). Nesting the laws level over the draws level keeps two different multiplicities distinct in the result itself (`D1 – Mathematical fidelity`), and matching levels by name rather than by position carries `C5 – Naming for unambiguous meaning` onto the multiplicity axis, making the rename the one way to align levels, as renaming is for a merge. When the elements are array-backed and claim differentiation, the sweep is a single vectorized call that preserves the differentiability claim, which is `D3 – Capability-based operations` applied to the batch through its elements.
