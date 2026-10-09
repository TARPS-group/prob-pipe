# Part IV — Constructing distributions

Part IV covers how distributions are constructed, from other distributions and from functions that a user writes.

| §    | Category           | Contents                                                                                              | Role                                                                                                   |
| ---- | ------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| IV.1 | From distributions | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from sub-distributions, with the factor and field access interfaces.              |
| IV.2 | From distributions | the `*` operator                                                                                      | Constructs a joint (conditional) distribution from two (conditional) distributions.                    |
| IV.3 | From distributions | cross-type conversion (`converter_registry`)                                                          | Moving a distribution between representations, at a recorded fidelity.                                 |
| IV.4 | From functions     | `distribution` and `conditional_distribution`                                                         | A law built from a sampler, a density, or both, and a kernel built from a function that returns a law. |

## IV.1 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from sub-distributions**: it carries an explicit factorization into its parts, marked by the capability `SupportsFactors`, which `FactoredDistribution` and `FactoredConditionalDistribution` implement generically.

Both carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's given slots against the components that the factors to its right produce in the conditional-first order (IV.2). The joint's event declaration is an exposed `RecordSpec` over the disjoint union of the factors' declared output components (II.2, IV.2), and a packaged joint declares that record as one whole term. Extraction and reconstruction preserve each factor's event packaging. Each produced component belongs to exactly one factor; factor labels may repeat. Conditioning a `FactoredConditionalDistribution` on all of its given slots yields a `FactoredDistribution`. Sampling and the log-prob capabilities are the intersection of the factors'. A `FactoredConditionalDistribution`'s conditional capabilities pass its unmet givens to each factor's own conditional capability, and the joint's claim to each conditional capability is read from its factors' claims; a given that omits a slot or names one the joint lacks raises `KeyError`. The moment capabilities are decided at construction and are present exactly when the joint's structure makes the moment derivable. An edge-free joint derives its moments componentwise, so it has a moment exactly when every factor does; a joint of jointly Gaussian factors has its moments in closed form (VII.6); any other dependent joint carries no moment capability, since factor-wise conditional moments do not compose into a closed form. For example, with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `E[y] = e^{1/2}` is not computable from the factors' means.

```python
@runtime_checkable
class SupportsFactors(Protocol):
    @property
    def factors(self) -> tuple[Distribution | ConditionalDistribution, ...]: ...   # ordered

class FactoredDistribution(Distribution, SupportsFactors): ...
class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors): ...

# numeric markers for which of the (given, event) sides is numeric. For conditional distributions, "Numeric" before "Distribution" marks the event as numeric and before "Conditional" marks the given as numeric.
class FactoredNumericDistribution(FactoredDistribution): ...                                # unconditional joint, event numeric
class FactoredConditionalNumericDistribution(FactoredConditionalDistribution): ...          # conditional joint, event numeric
class FactoredNumericConditionalDistribution(FactoredConditionalDistribution): ...          # conditional joint, given numeric
class FactoredFullyNumericConditionalDistribution(
        FactoredNumericConditionalDistribution, FactoredConditionalNumericDistribution): ... # conditional joint, both numeric
```

**Field versus factor.** A **field** is a named leaf of a draw, addressed by a key of the event schema (II.6). A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint exposes up to two interfaces, each through operators of its own.
- The **field interface** is available on every distribution: `d["intercept"]` is the field view of III.7, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, component_name="intercept")` selects the complete factor by one of `d`'s component names, returning a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. Several components may select the same factor; factors with no produced components remain accessible through `factors`.

**Renaming a joint.** `with_path_names` renames a joint through its factors: the factor that produces a renamed component renames it, each factor that consumes the component renames the matching given slot, and the result is the factored joint of the renamed factors over the same graph. A renamed given slot of a conditional joint is renamed in every factor that names it. A rename that gathers components under a new node regroups the factors that produce them into a **packaged sub-joint**: the result has one factor per gathering node, whose event is that node as one component holding the record of the gathered components. A sub-joint is labeled as `*` labels the joint of its factors (IV.2). A factor that consumes a gathered component conditions on the sub-joint through its renamed given slot, which holds the node's whole record, so the dependencies carry over and the result stays factored. Conditioning on a node then takes the factored routes, as do the factor, the view, and the marginal at a node. A gathering whose groups condition on one another in a cycle has no factored form, as gathering `mu` and `theta` when `tau` conditions on `mu` and `theta` on `tau` does. Such a rename, like any rename the factors cannot carry, returns a law that renames the joint's values at its boundary. That law is the joint's law under the new names, and it claims no `SupportsFactors`. For example, renaming the eight-schools prior `Normal("mu", 0, 5) * HalfCauchy("tau", 0, 5) * Normal("theta_tilde", jnp.zeros(8), 1)` by `{"mu": "population/mu", "tau": "population/tau", "theta_tilde": "groups/theta_tilde"}` gives a joint of two factors: the sub-joint `population` of the `mu` and `tau` factors, whose record is `{mu, tau}`, and the factor `groups`, which packages the `theta_tilde` factor as `{theta_tilde}`. In the centered model, where `theta` conditions on `mu` and `tau`, `groups` is a conditional sub-joint whose given slot `population` holds `mu` and `tau`, and conditioning the prior on `population` binds that slot. The rename is also how a nested joint is built, one group at a time or all at once.

**Marginals of a joint.** Whether a marginal is exact depends on the factors' own marginal support and on the target's position in the dependence graph, so the factored classes resolve `_marginal` per path. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out without computation. What remains is integrating the extra ancestor fields back out, which is exact in three cases:
1. there are none, because the target is ancestrally closed, as for a root factor or an edge-free group;
2. no factor of the closure consumes an extra field, and each factor that holds extra fields is exact at its fields in the target, through its own `SupportsMarginals` or, for a `ConditionalDistribution` factor, `SupportsConditionalMarginals`;
3. the affected factors admit closed-form integration, as when they are jointly Gaussian.

In the second case the marginal is the joint of the reduced factors. At any other path the exact route's guard rejects the call, leaving the fallback available when sampling is supported and the fidelity controls permit it (VI.8).

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, within the one class hierarchy of distributions. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two differ in the mathematics. Deciding moment presence at construction follows `D3 – Capability-based operations`: a capability is advertised exactly when the object can compute it.

### Notes

- *Group views.* The field interface also accepts an interior path, which names a group of fields. For example, when the event declaration nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns the field view of the whole group.

## IV.2 — Composition

### Contract

Composition builds a factored distribution from parts, written with an operator: a single binary operator `*` combines `Distribution`s and `ConditionalDistribution`s into a single joint (conditional) distribution. The kind of the result is derived from the operands. The base objects expose `*` via `__mul__`, which delegates to the operator.

**The `*` operator.** `A * B` composes two operands into a joint. It is **conditional-first**: the left operand may condition on the right, so `lik * prior` reads as the density p(y | β) · p(β), while the reverse, with the producer on the left of its consumer, is an error. Characterize each operand by its **produced slots** `F`, which are exactly the component names in its `event_spec.components` (II.2), and its **unmet given slots** `G`, which are empty for a `Distribution` and the given slots for a `ConditionalDistribution`. The dependency topology is fixed by the name sets; matched specs must also unify:

```
bound  = G_A ∩ F_B            # dependency edges: left A conditions on a name that right B produces
unmet  = (G_A − F_B) ∪ G_B    # residual exogenous givens — met by no factor
require  F_A ∩ F_B = ∅         # each name is produced exactly once
     and G_B ∩ F_A = ∅         # B must not consume a name A produces — else reorder (producer on the right)
law:  p(F_A, F_B | unmet) = p_A(F_A | bound ∪ (G_A − F_B)) · p_B(F_B | G_B)      # reads left → right
```

An optional slot (II.2) is in `G` and is matched as a required one is. Unmet, it takes its default (IV.4), so the result is a conditional distribution exactly when a required slot is unmet, and a plain distribution otherwise. A slot of `unmet` is optional when every factor that names it holds it optional:

| required slots of `unmet` | result |
|---|---|
| `∅` | `FactoredDistribution` — a joint `Distribution` |
| `≠ ∅` | `FactoredConditionalDistribution` — a joint `ConditionalDistribution`, its `given_spec` exactly `unmet` |

A composition that breaks a composition rule raises `ValueError`, for example a producer on the left of its consumer or matched specs that do not unify. A factor that is neither a `Distribution` nor a `ConditionalDistribution` raises `TypeError` at construction, and `__mul__` returns `NotImplemented` for such an operand, so a scalar operand can scale.

`*` returns the **most specific** class, recomputed from the *flattened* factor graph at each step. A family supplies a more specific factored class, such as `FactoredMultivariateGaussian` (VII.6), by registering it at import with a predicate over the flattened factors, and `*` constructs the most specific registered class whose predicate holds. The refinement after binding (VII.6) reads the same registrations. `A * B * C` builds one flat N-factor joint, with independent factors commuting and dependent ones kept in conditional-first order. Flattening replaces a joint operand by its factors, and a packaged joint (IV.1) stays one factor, since its packaging is part of its declaration. Same-named unmet givens unify into one slot of the joint:
1. their specs must unify, and a disagreement raises at composition;
2. the joint's slot carries their unification, which admits only the values every consuming factor admits;
3. binding the slot feeds every factor that names it.

Two givens that are different quantities are renamed apart first, as fields and levels require. Symbolic dimensions follow the same rule: the operands share one dimension scope, so a name two factors both use is one dimension, bound once and required to agree (II.1), and two dimensions that are different quantities are renamed apart first with `with_dim_names`. The joint stores the factors so unified. The unification is canonical, so derived names and fingerprints are deterministic.

**Packing the joint.** The joint's event is an exposed record of the components in canonical factor order, preserving each declaration's component order. Assembly extracts components from each factor's one draw; scoring reconstructs that factor's original event value before calling its density method. Extraction reads the declaration: a whole-term component is the draw itself, and an exposed record's components are its immediate children, so reconstructing all of a factor's components restores the draw's kind, structure, and coordinates. In particular, an array and a one-field record may both export `beta`, but their density implementations receive different declared event kinds. Multiple connections from one factor share the same draw. The joint retains the declarations needed for reconstruction and reads packaging from them alone.

**Labeling the result.** The joint's label joins the operands' current labels with `·`, each grouped by the rules of II.4 so that it reads as one unit. So `lik * prior` is labeled `lik·prior`, `(lik * prior).with_label("posterior") * d` is labeled `posterior·d`, and `lik * d` for a law labeled `my model` is labeled `lik·[my model]`. Labels join associatively: an operand whose label is a product joins as it is, so `(lik * prior) * d` and `lik * (prior * d)` are both labeled `lik·prior·d`. Exchanging independent operands may change the label while preserving the joint law.

**The notation of a product.** A product that `*` or `joint` builds is unlabeled: its expression (II.4) is the product of its operands' expressions, so its notation joins their notations with `·`, as `lik(y | mu)·prior(mu)`. A product given a label, by `with_label` or by the `FactoredDistribution` constructor, displays by that label, as `model(y, mu)`. A law built from a product, such as a copy or a packaged sub-joint, keeps its expression. A product without a label that holds fixed paths (VI.6) displays by its grouped label and its signature, as `(lik·prior)(mu; y)`, since its factors' notations do not show the fixed paths. A marginal over the whole events of several factors is a product without a label whether or not the joint is labeled, as `marginal(model, ("a", "b"))` displays as `a(a)·b(b)` (VI.8). A product that a function returns takes the function's `output_label` as its label, so it displays as `predict(y, mu)` (III.3). A product without a label enters a further composition as its factors, so `(lik * prior) * d` displays as `lik(y | mu)·prior(mu)·d(z)`, and a labeled product enters as one operand, so `model * d` displays as `model(y, mu)·d(z)` and is labeled `model·d`.

Composition determines structure from the names of components and given slots alone. Every operand enters as its flattened factors, so the joint's factors, its produced components, and its connections are the same whatever the operands are called: `AB * C` and `AB.with_label(AB.label) * C` are one joint under one label, and `(lik * prior).with_label("posterior") * d` is the three-factor joint of `lik`, `prior`, and `d`. The only effect of `with_label` on a later composition is the text it contributes to the derived label.

```python
def __mul__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product, where `bound = ∅`, is an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. Deriving the label deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`). An unlabeled product displays factor by factor because a single joined label followed by the joint's signature, as `lik·prior(y, mu)`, would read as one law over `y` and `mu`. The conditional-first order is a valid topological listing, so acyclicity is automatic and needs no separate graph inference. Associativity rests on the `G_B ∩ F_A = ∅` requirement, under which the validity of `(A * B) * C` and `A * (B * C)` coincide. Composition is written with an operator so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and every result is a first-class joint that composes further (`D4 – Closed system of objects under operations`).

### Notes

- *Operator coexistence.* `*` also denotes scalar scaling on some objects, such as a random function or a linear operator. The two coexist by operand-type dispatch: `Distribution` and `ConditionalDistribution` operands compose, while scalar operands scale.

## IV.3 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or a backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current class to a requested target while preserving its law and its event declaration. A **converter** is a binary dispatch method (II.7) that declares the source types it converts *from* and the target types it converts *to*. Conversions differ in fidelity, so each converter declares whether it is **exact** (II.7): exact for an equivalent representation, approximate for a stand-in such as moment matching or a Monte Carlo representation. A caller may set `exact_only`, which the registry's `check` enforces (II.7).

**The registry.** The **converter registry** is the binary dispatch registry keyed on `(type(source), target)`. A target is a distribution class or a capability protocol (III.8). Specificity on the target side is the position of the requested target in the declared target's method-resolution order, so a converter that declares the requested target itself is closest, and a structural match counts as least specific, as on the source side (II.7). For a protocol target the registry considers the converters that promise the required guarded capability. The registry's `check` and `execute` test the source first: a source that already satisfies the target, which for a protocol target means claiming the protocol and passing its guard, is reported feasible with no converter selected and is returned as it is, and `convert` gives it fresh identity (VI.10). A protocol target is feasible when both the claim of the protocol and its guard can be established, on the source or on the converter's promised result. A converter that promises a guarded protocol whose guard can be decided only once the converted law exists is reported unresolved, with that guard as the pending requirement (II.7).

**What a conversion preserves.** A conversion changes the representation alone. The result carries the source's event declaration, its component names, kind, and packaging included, and realizes the same law up to the recorded fidelity. It keeps the source's declared support, so a converter refuses a result on another support, as a moment-matched `Normal` fit to a law on the positive half-line would be. The converter option `check_support=False` overrides the refusal, and the result then has the family's support. Flattening a record event to an array changes the event space, so it is the declared isomorphism of III.7 rather than a conversion.

**Check and execute.** A converter has an `execute` method to carry out the conversion and a `check` method to determine what `execute` would do if called. `check` returns a `ConversionInfo`: the promised target spec, the representation class when known, the capabilities guaranteed on the result, and, as pending requirements, whatever it cannot settle without the source's values. It never samples, fits, or evaluates a density. The registry's `check` returns a `ConversionInfo` in every state, so a planner reads one report type, and an infeasible report leaves `target_spec`, `target_class`, and `capabilities` at their empty defaults. A converter whose result cannot carry the source's event declaration reports itself infeasible at `check`, as a moment-matched fit does whose draws do not cast to the source's dtype or whose family has a fixed support other than the source's, and the registry then tries the next converter. A support that depends on the fitted parameters is checked when the fit runs. The registry raises `TypeError` for a converter whose `check` returns another type, and `ValueError` for a feasible promise that does not carry the source's declaration. The function stack plans through `check` at argument normalization and constructs through `execute` at execution (V.4, V.9), and `convert` exposes the same two steps to the user (VI.10).

```python
@dataclass(frozen=True)
class ConversionInfo(Feasibility):
    target_spec: DistributionSpec | None = None   # None only while unresolved
    target_class: type | None = None
    capabilities: tuple[type, ...] = ()           # capabilities guaranteed on the result
    # pending requirements are inherited from Feasibility; exactness is the converter's declaration

class Converter(BinaryDispatchMethod):
    name: str
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (source, target) types
    def check(self, source, target_type: type) -> ConversionInfo: ...
    def execute(self, source, target_type: type) -> Distribution: ...             # the conversion itself

class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    # keyed on (type(source), target): the target is a distribution class, or a capability protocol (III.8)
    def convert(self, source, target_type: type,
                method: str | None = None, exact_only: bool = False) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity makes the approximation explicit, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while an `approximate` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`). Realizing the registry as a subclass of the shared dispatch registry gives conversion registration, feasibility probing, prioritized selection, and cataloging without duplicating any of them (`D6 – Single source of truth`).

## IV.4 — Distributions from functions

### Contract

**A law from functions.** `distribution` builds a `Distribution` from a sampling function, a log-density, or both, as `function` builds a `Function` from a callable (V.1). It takes each function under the name of the operation it realizes, the event declaration, and the optional keywords `component` and `label`: `sample(key)` returns one draw at a PRNG key, at the kind the event declaration names, and `log_prob` or `unnormalized_log_prob` scores one value and returns a real scalar. A density receives an array for an array event, a `Record` for a record event, and the value itself for any other event. A call gives at least one function, at most one density, and an `event_spec` of any kind a `Distribution` declares, and a missing or non-callable function raises `TypeError`, which names it. An `OutputSpec` names its components and a bare `RecordSpec` exposes its fields, while any other bare spec is a whole term under `component`, which such a spec requires and the other two refuse, both with `TypeError`. The label defaults to `p`. The law claims the capability that each function given realizes:

- `sample`: `SupportsSampling`;
- `log_prob`: `SupportsLogProb`, which provides the unnormalized density as a normalized family's does;
- `unnormalized_log_prob`: `SupportsUnnormalizedLogProb`.

A law of a density alone is therefore unnormalized (III.8). A non-empty sample shape maps `sample` over keys split from the draw's key with `jax.vmap`. A sampler that does not trace in JAX, such as one that calls NumPy or an external program, and a sampler of an event that is not numeric are called at one key at a time instead, and their draws stack into the batch form of the event's kind. A density scores a batch of values along its leading axes, with `jax.vmap` when it traces.

Construction draws nothing and scores no value. It evaluates each function that traces abstractly, with `jax.eval_shape`: the abstract draw of `sample` must unify with `event_spec`, and each density must return a real scalar at a stand-in of one value of the event, or `ValueError` names the failed check. The abstract draw completes the declaration, as a family's parameters complete its own (III.7). A function whose `jax.eval_shape` fails does not trace, and construction reads nothing from it, so a declaration that no traced draw completes is complete as given, and a pending type raises `TypeError`.

```python
def distribution(
    *,
    sample: Callable[[Key], Any] | None = None,
    log_prob: Callable[[Any], Array] | None = None,
    unnormalized_log_prob: Callable[[Any], Array] | None = None,
    event_spec: OutputSpec | TermSpec,
    component: str | None = None,
    label: str | None = None,
) -> Distribution: ...
    # at least one function and at most one density; component for a bare whole-term spec
```

**A kernel from a function.** `conditional_distribution` builds a `ConditionalDistribution` (III.9) from a function of its given values that returns a law. Its call form takes the function and an optional label, as in `conditional_distribution(lambda mu, tau: Normal("y", mu, tau), label="lik", given_spec=...)`, and its decorator form on a `def` takes the same keywords. The kernel's label defaults to the function's `__name__`, and to `p` for a lambda, so the decorator form labels the kernel after the function. Each parameter of the function is a given slot, and its spec is its entry in `given_spec` or else its annotation, which must then be a term spec. A parameter with a default is an optional slot (II.2), which holds a constant of the model, and its default's value declares it when neither does. A parameter with no default and no declaration raises `TypeError`, whose message gives the `given_spec` entry that declares it. Construction evaluates the function once, abstractly, at a stand-in of each required slot's type and at the default of each optional slot, and reads three things from the law it returns:

1. the event declaration: the kernel declares the law's, or an explicit `event_spec` that names the law's components and unifies with its type, where a bare spec other than a record names the law's one component, and a support that a given value sets is left undeclared;
2. the claims: the kernel claims `SupportsConditionalSampling`, `SupportsConditionalLogProb`, and `SupportsConditionalUnnormalizedLogProb` exactly when the law claims the capability each one twins;
3. the guards: each twin's guard reports what the law's guard of the capability reported.

Binding every required slot calls the function with each given value as the argument of that name, at the kind its slot declares, as a function's body receives a draw (V.5), and with its default for each optional slot left unbound. It returns the law the call returns, which must agree with the kernel's event declaration, so a law that departs from it raises `ValueError`. Binding fewer slots curries the kernel over the rest. In a joint, a factor that produces a component named as an optional slot meets it (IV.2), so one kernel serves a model with the constant and a model with a prior on it.

```python
def conditional_distribution(
    fn: Callable[..., Distribution] | None = None,
    /,
    *,
    label: str | None = None,
    given_spec: InputSpec | Mapping[str, TermSpec] | None = None,
    event_spec: OutputSpec | TermSpec | None = None,
) -> ConditionalDistribution | Callable[[Callable[..., Distribution]], ConditionalDistribution]: ...
    # conditional_distribution(fn) is the kernel of fn; without fn it is a decorator, and
    # @conditional_distribution on a def labels the kernel after the function
```

A kernel whose function returns `distribution(...)` reads the law's claims as for any law, so the kernel of a simulator, `conditional_distribution(lambda rate: distribution(sample=lambda key: jax.random.poisson(key, rate, (10,)), event_spec=counts, component="y"), given_spec={"rate": positive_scalar})`, claims conditional sampling and no density.

### Rationale

A law built from functions claims only what its functions realize (`D3 – Capability-based operations`). Its construction checks each function that traces against the declaration, which every consumer of the law reads (`D5 – Explicit, carried structure`), and it draws nothing, so a kernel that builds a law at every binding calls the law's functions only when an operation asks for a draw or a density (`C3 – Computational detail hidden by default, available on demand`). A kernel built from a function of its given values is written as that function, as a `Function` is (`C1 – Uniform interface to functions, distributions, and values`), and reading its claims from the law the function returns makes it advertise only what that law supports (`D3 – Capability-based operations`).
