# Part IV — Constructing distributions

Part IV covers how distributions are constructed, from other distributions and from functions that a user writes.

| §    | Category           | Contents                                                                                              | Role                                                                                                   |
| ---- | ------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| IV.1 | From distributions | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from sub-distributions, with the factor and field access interfaces.              |
| IV.2 | From distributions | the merge operator `&`                                                                                | Merges two (conditional) distributions by name into a joint (conditional) distribution.                |
| IV.3 | From distributions | cross-type conversion (`converter_registry`)                                                          | Moving a distribution between representations, at a recorded fidelity.                                 |
| IV.4 | From functions     | `distribution` and `conditional_distribution`                                                         | A law built from a sampler, a density, or both, and a kernel built from a function that returns a law. |

## IV.1 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from sub-distributions**: it carries an explicit factorization into its parts, marked by the capability `SupportsFactors`, which `FactoredDistribution` and `FactoredConditionalDistribution` implement generically.

Both carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's inputs against the outputs of the factors to its right, in the conditional-first order (IV.2). The joint's event spec is the positional form over the record of the factors' outputs (II.2, IV.2), and assembly and scoring keep each factor's own draw. Each output belongs to exactly one factor; factor labels may repeat. Conditioning a `FactoredConditionalDistribution` on all of its inputs yields a `FactoredDistribution`. Sampling and the log-prob capabilities are the intersection of the factors'. A `FactoredConditionalDistribution`'s conditional capabilities pass the values of the inputs that no factor feeds to each factor's own conditional capability, and the joint's claim to each conditional capability is read from its factors' claims; a given that omits an input or names one the joint lacks raises `KeyError`. The moment capabilities are decided at construction and are present exactly when the joint's structure makes the moment derivable. An edge-free joint derives its moments output by output, so it has a moment exactly when every factor does; a joint of jointly Gaussian factors has its moments in closed form (VII.6); any other dependent joint carries no moment capability, since factor-wise conditional moments do not compose into a closed form. For example, with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `𝔼[y] = e^{1/2}` is not computable from the factors' means. The `by_component` route of VI.5 still computes exactly each output of such a joint's mean, variance, or quantile whose view has an exact route, as a root factor's view does.

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

**Field versus factor.** A **field** is an entry of a draw that is not a record, addressed by its path (II.6). A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint offers up to two interfaces, each through operators of its own.
- The **path interface** is available on every distribution: `d["intercept"]` is the marginal view of III.7, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, output_name="intercept")` selects the complete factor by one of `d`'s outputs, returning a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. Several outputs may select the same factor; factors with no outputs remain accessible through `factors`.

**Renaming a joint.** `rename` renames a joint through its factors: the factor that gives a renamed output renames it, each factor that takes the output renames the matching input, and the result is the factored joint of the renamed factors over the same graph. A renamed input of a conditional joint is renamed in every factor that names it. A rename keeps each factor it does not gather. A rename that gathers outputs of several factors under a new group makes the factors that give them one factor, a **grouped sub-joint**: the result has one factor per gathering group, which wraps its group, so its event spec is the positional form over the record `{g: {...}}` of that group, its draws carry the group's level, and it has the one output `g`. A grouped sub-joint stays one factor of the joint and of any later merge, so the optional inputs its factors left at their defaults stay there (IV.2). A sub-joint is labeled as `&` labels the joint of its factors (IV.2). A factor that takes a gathered output conditions on the sub-joint through its renamed input, which holds the group's whole record, so the dependencies carry over and the result stays factored. Conditioning on a group then takes the factored routes, as do the factor, the view, and the marginal at a group. A gathering whose groups condition on one another in a cycle has no factored form, as gathering `mu` and `theta` when `tau` conditions on `mu` and `theta` on `tau` does. Such a rename, like any rename the factors cannot carry, returns a law that renames the joint's values at its boundary. That law is the joint's law under the new names, and it claims no `SupportsFactors`. For example, renaming the eight-schools prior `Normal("mu", 0, 5) & HalfCauchy("tau", 0, 5) & Normal("theta_tilde", jnp.zeros(8), 1)` by `{"mu": "population/mu", "tau": "population/tau", "theta_tilde": "groups/theta_tilde"}` gives a joint of two factors: the sub-joint `population` of the `mu` and `tau` factors, a law over `{population: {mu, tau}}`, and the factor `groups`, which wraps the `theta_tilde` factor as a law over `{groups: {theta_tilde}}`. In the centered model, where `theta` conditions on `mu` and `tau`, `groups` is a conditional sub-joint whose input `population` holds `mu` and `tau`, and conditioning the prior on `population` binds that input. The rename is also how a nested joint is built, one group at a time or all at once.

**Marginals of a joint.** Whether a marginal is exact depends on the factors' own marginal support and on the target's position in the dependence graph, so the factored classes resolve `_marginal` per path. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out without computation. What remains is integrating the extra ancestor fields back out, which is exact in three cases:
1. there are none, because the target is ancestrally closed, as for a root factor or an edge-free group;
2. no factor of the closure consumes an extra field, and each factor that holds extra fields is exact at its fields in the target, through its own `SupportsMarginals` or, for a `ConditionalDistribution` factor, `SupportsConditionalMarginals`;
3. the affected factors admit closed-form integration, as when they are jointly Gaussian.

In the second case the marginal is the joint of the reduced factors. At any other path the exact route's guard rejects the call, leaving the fallback available when sampling is supported and the fidelity controls permit it (VI.8).

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, within the one class hierarchy of distributions. Keeping the path interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two differ in the mathematics. Deciding moment presence at construction follows `D3 – Capability-based operations`: a capability is advertised exactly when the object can compute it.

### Notes

- *Group views.* The path interface also accepts the path of a group. For example, when the event spec nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns the marginal view of the group, whose outputs are `intercept` and `slope`, and `d[["coeffs"]]` the view that keeps the group under its name.

## IV.2 — Merging distributions

### Contract

A model is often a product of densities joined by shared names, as `p(y, μ, τ) = p(y | μ) · p(μ, τ)`. The **merge** `a & b` builds that joint from two operands, each a `Distribution` or a `ConditionalDistribution`, by matching names: each output of `b` **feeds** the input of `a` of the same name, which means that `a` receives at that input what `b` gives at that output (II.2). A merge reads right to left, as the density does, so `b` may supply inputs of `a` and never the reverse: `lik & prior` is the density `p(y | μ) · p(μ)`. The kind of the result is derived from the operands, and the base classes expose `&` through `__and__`.

**What the operands take and give.** A joint keeps the laws and kernels it was built from, its factors (IV.1), and a later merge reads them, so that `(a & c) & e` and `a & (c & e)` give the same law. For a kernel `a(y | mu, sigma)` whose `sigma` is optional and a law `c(mu)`, the law `a & c` takes no inputs, yet its factor `a` still takes `sigma`, which a law `e(sigma)` merged later supplies. Each operand is therefore characterized by two sets of names:
1. `O`, its outputs, which are those of its event spec (II.2);
2. `I`, the inputs that its factors take and that no factor of it feeds: for a kernel that is its own factor, its inputs; for a law that is its own factor, none, since the optional inputs of what it was computed from are fixed at their defaults; and for a joint, the inputs its factors take that no factor feeds, which for a law are the optional inputs its factors left at their defaults.

Only a merge and a rename keep factors (IV.1). Every other law or kernel is its own one factor, including a view, a marginal, a conditional, and the result of any other operation that computes a new law, whatever algorithm computed it. The topology is fixed by the name sets:

```
fed     = I_a ∩ O_b              # inputs of a's factors that b's outputs feed
inputs  = (I_a − O_b) ∪ I_b      # the result's inputs, in order of first appearance
require   O_a ∩ O_b = ∅           # each output has one source
     and  I_b ∩ O_a = ∅           # b takes nothing a outputs; otherwise write b & a
     and  I_a ∩ O_a = I_b ∩ O_b = ∅   # within an operand, no name is both an input and an output
law:  p(O_a, O_b | inputs) = p_a(O_a | fed, I_a − O_b) · p_b(O_b | I_b)   # reads left to right
```

An optional input is fed like a required one. One that no operand feeds stays an input of the result, and an input both operands take, which neither feeds, is one input of the result, required unless it is optional in both. The kind of the result follows the required inputs that remain, so a merge that leaves only optional ones is a law, whose optional inputs take their defaults when it is sampled or scored, and a later merge can still feed them:

| required names in `inputs` | result |
|---|---|
| `∅` | `FactoredDistribution`, a joint `Distribution` |
| `≠ ∅` | `FactoredConditionalDistribution`, a joint `ConditionalDistribution`, whose `given_spec` is `inputs` |

A kernel standing alone stays a kernel even when all its inputs are optional (III.9).

**Refused merges.** `a & b` is refused unless every condition below holds:

| Condition | Error | Why |
|---|---|---|
| 1. both operands are laws or kernels | `TypeError`, since `__and__` returns `NotImplemented` | each kind has its own merge: records merge by name (II.6), and batches element by element (VI.11) |
| 2. each operand's event spec has a base or record `term_spec` | `TypeError`, naming the operand and saying to wrap the values it is over under a name | each operand is rebuilt from fields of the result, and a law over laws, functions, or batches has no outputs (II.2) |
| 3. `O_a ∩ O_b = ∅` | `ValueError`, naming the output and both operands | one name cannot hold two values |
| 4. `I_b ∩ O_a = ∅` | `ValueError`: "write b & a: b takes x, which a outputs, and in a & b only b's outputs feed a's inputs" | each operand is fed only by operands to its right, so no name of the result is both an input and an output |
| 5. within each operand, no name is both an input and an output | `ValueError`, saying to rename one | otherwise it is unclear whether the other operand's output of that name feeds it or is fed by it |
| 6. each output's spec conforms to the spec of the input it feeds (II.1) | `ValueError`, naming both specs | every value fed to an input must conform to its spec |
| 7. the two specs of an input both take unify (II.1) | `ValueError`, naming both specs | one value supplies both |
| 8. every spec of both operands, with the dimension sizes earlier merges bound in each, forms one dimension scope (II.1) | `ValueError`, naming the dimension and both sizes | a dimension name stands for one size |

A merge never lifts: an output that is a law does not feed an input that expects a value, since lifting the receiving kernel over that law would ignore the law's dependence on the kernel's other inputs. Two views of one law merge as independent laws, since each view is one factor: for a law `d` over `{g1: {theta}, g2: {theta}}`, `d[["g1"]] & d[["g2"]]` is the product of the two marginals, and the joint law of both groups is `d["g1", "g2"]`. `d["g1"] & d["g2"]` is refused by condition 3, both having the output `theta`, and its message points to `d["g1", "g2"]`.

**The value.** Sampling `a & b` draws from `b` at its inputs, then from `a` at its own inputs and what `b`'s draw feeds, and its density is the product above. Binding or conditioning a merged law in closed form follows the same definition, using what later operands feed.

**The joint's event.** The joint's event spec is the positional form over the record of both operands' outputs, `a`'s then `b`'s, each in its operand's order, which is `a.event_spec & b.event_spec` (II.2): an output at the root `y` becomes the field `y`, and each child of a record event keeps its name. Every path of an operand is therefore a path of the joint, at the same spec. Assembly takes each factor's outputs from its one draw, and scoring rebuilds that factor's draw before calling its density: an output at the root is the draw itself, and the outputs of a record event are its children, so rebuilding all of a factor's outputs restores the draw's kind, structure, and coordinates. An array and a one-field record may both give the output `beta`, but their densities receive different kinds. Several feeds from one factor share the same draw.

**Flattening.** `&` returns the **most specific** class, recomputed from the *flattened* factor graph at each step. A family supplies a more specific factored class, such as `FactoredMultivariateGaussian` (VII.6), by registering it at import with a predicate over the flattened factors, and `&` constructs the most specific registered class whose predicate holds. The refinement after binding (VII.6) reads the same registrations. `a & b & c` builds one flat joint of three factors, with independent factors commuting and dependent ones kept in conditional-first order. Flattening replaces a joint operand by its factors, and a grouped sub-joint (IV.1) stays one factor. An input that several factors take is one input of the joint:
1. its specs must unify, and a disagreement raises at the merge;
2. the joint's input carries their unification, which admits only the values every factor that takes it admits;
3. binding the input supplies every factor that takes it, and a binding that omits an optional one leaves each factor at its own default.

Two inputs that are different quantities are renamed apart first, as fields and levels require. Symbolic dimensions follow the same rule: the operands share one dimension scope, so a name two factors both use is one dimension, bound once and required to agree (II.1), and two dimensions that are different quantities are renamed apart first with `with_dim_names`. The joint stores the factors so unified and keeps the sizes its merge bound, so a later merge starts from both operands' kept sizes. The unification is canonical, so derived names and fingerprints are deterministic.

**`*` and truth values.** `*` scales and `&` merges. `*` between two laws, two kernels, or a law and a kernel raises `TypeError: * does not merge laws; write lik & prior`, and returns `NotImplemented` for any other operand, so a scalar keeps Python's own handling and a family that scales, such as a random function, scales. A law or a kernel has no truth value: `bool()` raises `TypeError: a law has no truth value; to merge laws write a & b`, so `lik and prior` raises instead of returning `prior`, while a record keeps its length-based truth. Both refusals are permanent, so a later arithmetic meaning of `*` cannot give old code a new one silently.

**Labeling the result.** The joint's label joins the operands' current labels with `·`, each grouped by the rules of II.4 so that it reads as one unit. So `lik & prior` is labeled `lik·prior`, `(lik & prior).with_label("posterior") & d` is labeled `posterior·d`, and `lik & d` for a law labeled `my model` is labeled `lik·[my model]`. Labels join associatively: an operand whose label is a product joins as it is, so `(lik & prior) & d` and `lik & (prior & d)` are both labeled `lik·prior·d`. Exchanging independent operands may change the label while preserving the joint law.

**The notation of a product.** A product that `&` or `joint` builds is unlabeled: its expression (II.4) is the product of its operands' expressions, so its notation joins their notations with `·`, as `lik(y | mu)·prior(mu)`. A product given a label, by `with_label` or by the `FactoredDistribution` constructor, displays by that label, as `model(y, mu)`. A law built from a product, such as a copy or a grouped sub-joint, keeps its expression. A product without a label that holds fixed paths (VI.6) displays by its grouped label and its signature, as `(lik·prior)(mu; y)`, since its factors' notations do not show the fixed paths. A marginal over the whole events of several factors is a product without a label whether or not the joint is labeled, as `marginal(model, ["a", "b"])` displays as `a(a)·b(b)` (VI.8). A product that a function returns takes the function's `output_label` as its label, so it displays as `predict(y, mu)` (III.3). A product without a label enters a further merge as its factors, so `(lik & prior) & d` displays as `lik(y | mu)·prior(mu)·d(z)`, and a labeled product enters as one operand, so `model & d` displays as `model(y, mu)·d(z)` and is labeled `model·d`.

A merge determines structure from the names of outputs and inputs alone. Every operand enters as its flattened factors, so the joint's factors, its outputs, and its feeds are the same whatever the operands are called: `ab & c` and `ab.with_label(ab.label) & c` are one joint under one label, and `(lik & prior).with_label("posterior") & d` is the three-factor joint of `lik`, `prior`, and `d`. The only effect of `with_label` on a later merge is the text it contributes to the derived label.

**What a merge is not.** A merge is not composition, which passes `g`'s result to `f` by position in `f ∘ g` and returns `f`'s result alone, while a merge matches names and keeps both operands' outputs. It is not binding or conditioning: binding `lik`'s input `mu` removes the input, and conditioning `lik & prior` on `mu` removes the output, while in `lik & prior` the output `mu` feeds `lik` and stays an output. It is not renaming either, so an operand is renamed first (II.6) and then merged.

```python
def __and__(self, other: Distribution | ConditionalDistribution, /) -> FactoredDistribution | FactoredConditionalDistribution: ...
def __mul__(self, other: object, /) -> NoReturn: ...   # raises TypeError naming & for a law or a kernel; NotImplemented otherwise
def __bool__(self) -> NoReturn: ...   # raises TypeError: a law has no truth value
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent joint, where `fed = ∅`, is an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. Matching by name rather than position lets a model be assembled from parts written separately, and each name has one source, so the joint is well defined (`C5 – Naming for unambiguous meaning`). The conditional-first order is a valid topological listing, so acyclicity is automatic and needs no separate graph inference; associativity holds with or without the order rule, and the rule keeps the joint well formed, since each density in the product is conditioned only on factors to its right. Keeping factors only through a merge or a rename makes a result independent of the algorithm that computed it, so a view and a marginal of one joint merge alike (`C3 – Computational detail hidden by default, available on demand`). A law as a value has no outputs, so a merge refuses one and asks for a wrap, which keeps a law's parts from being received apart from their dependence (`D1 – Mathematical fidelity`). The operator is `&` because it reads as "both" while `*` is arithmetic on arrays, records, and linear operators, and the refusals of `*` and of a truth value keep the old spelling and `lik and prior` from silently meaning something else (`C5 – Naming for unambiguous meaning`). Deriving the label deterministically keeps a joint's meaning clear without forcing the user to label every intermediate result (`C5 – Naming for unambiguous meaning`). An unlabeled product displays factor by factor because a single joined label followed by the joint's signature, as `lik·prior(y, mu)`, would read as one law over `y` and `mu`. A merge is written with an operator so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and every result is a first-class joint that merges further (`D4 – Closed system of objects under operations`).

### Notes

- *`&` on other kinds.* `&` merges records by top-level name (II.6), batches element by element over equal levels (VI.11), and input specs and output specs (II.2); on two `NumericArray`s it is NumPy's elementwise `&`, which no merge replaces, since a base value carries no output name.
- *Objects of two kinds.* A record could be read as a law that gives itself with probability one, and a function as a kernel of point masses, but a law with a point-mass factor has no joint density, so a law and a record, or a kernel and a function, do not merge.

## IV.3 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or a backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current class to a requested target while preserving its law and its event spec. A **converter** is a binary dispatch method (II.7) that declares the source types it converts *from* and the target types it converts *to*. Conversions differ in fidelity, so each converter declares whether it is **exact** (II.7): exact for an equivalent representation, approximate for a stand-in such as moment matching or a Monte Carlo representation. A caller may set `exact_only`, which the registry's `check` enforces (II.7).

**The registry.** The **converter registry** is the binary dispatch registry keyed on `(type(source), target)`. A target is a distribution class or a capability protocol (III.8). Specificity on the target side is the position of the requested target in the declared target's method-resolution order, so a converter that declares the requested target itself is closest, and a structural match counts as least specific, as on the source side (II.7). For a protocol target the registry considers the converters that promise the required guarded capability. The registry's `check` and `execute` test the source first: a source that already satisfies the target, which for a protocol target means claiming the protocol and passing its guard, is reported feasible with no converter selected and is returned as it is, and `convert` gives it fresh identity (VI.10). A protocol target is feasible when both the claim of the protocol and its guard can be established, on the source or on the converter's promised result. A converter that promises a guarded protocol whose guard can be decided only once the converted law exists is reported unresolved, with that guard as the pending requirement (II.7).

**What a conversion preserves.** A conversion changes the representation alone. The result carries the source's event spec, its outputs, paths, and kind included, and realizes the same law up to the recorded fidelity. It keeps the source's declared support, so a converter refuses a result on another support, as a moment-matched `Normal` fit to a law on the positive half-line would be. The converter option `check_support=False` overrides the refusal, and the result then has the family's support. Flattening a record event to an array changes the event space, so it is the declared isomorphism of III.7 rather than a conversion.

**Check and execute.** A converter has an `execute` method to carry out the conversion and a `check` method to determine what `execute` would do if called. `check` returns a `ConversionInfo`: the promised target spec, the representation class when known, the capabilities guaranteed on the result, and, as pending requirements, whatever it cannot settle without the source's values. It never samples, fits, or evaluates a density. A converter's report states only its promise: it leaves `method_name` and `exact` unset, and the registry sets both from the registration of the converter it selected, so the reported exactness is always the converter's declaration (II.7). `ConversionInfo` therefore relaxes the rule of `MethodInfo` that a feasible or unresolved report names its method: the rule holds for the registry's report of a selected converter, while a converter's own report leaves `method_name` and `exact` unset, and the report of a source that needs no conversion carries `exact=True` with no `method_name`. The registry's `check` returns a `ConversionInfo` in every state, so a planner reads one report type, and an infeasible report leaves `target_spec`, `target_class`, and `capabilities` at their empty defaults. A converter whose result cannot carry the source's event spec reports itself infeasible at `check`, as a moment-matched fit does whose draws do not cast to the source's dtype or whose family has a fixed support other than the source's, and the registry then tries the next converter. A support that depends on the fitted parameters is checked when the fit runs. The registry raises `TypeError` for a converter whose `check` returns another type, and `ValueError` for a feasible promise that does not carry the source's event spec. The function stack plans through `check` at argument normalization and constructs through `execute` at execution (V.4, V.9), and `convert` exposes the same two steps to the user (VI.10).

**Converter options.** The registry passes a call's converter options as keywords to the `check` of each converter it tries and to the `execute` of the one it selects. Examples are `num_samples` for a converter that draws, `check_support`, and `event_spec`, which declares the event of a backend object, since a backend object carries no event spec of its own. A converter reads the options it names, and its `execute` raises `TypeError` for any other, so a misspelled option fails rather than being ignored, and so does an option that only another converter reads. A call that passes one converter's options names that converter with `method` when automatic selection could select another. Since the registry offers the same options to every converter it tries, `check` accepts an option it does not read, so a misspelled option passes planning and fails when the conversion executes. The parameter's `conversions` control supplies the options at normalization (V.4).

**Random events.** A report's `claims_random_events` is `True` when executing the conversion claims at least one workflow-owned random event (V.8), so its result depends on the workflow scope's seed, as when a converter draws from the source. It is `False` when the conversion claims none, so its result is a deterministic function of the source and the converter options. `check` decides it without drawing, and a report that selects no converter states `False`. Exactness is a separate declaration: the empirical law of a law's draws is approximate and draws, while a moment fit from closed-form moments is approximate and does not draw.

```python
@dataclass(frozen=True)
class ConversionInfo(MethodInfo):
    target_spec: DistributionSpec | None = None   # None unless feasible
    target_class: type | None = None
    capabilities: tuple[type, ...] = ()           # capabilities guaranteed on the result
    claims_random_events: bool = False            # executing it claims a workflow-owned random event (V.8)
    # pending requirements are inherited from Feasibility; a converter leaves method_name and exact
    # unset, and the registry sets both from the registration

class Converter(BinaryDispatchMethod):
    name: str
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (source, target) types
    def check(self, source, target_type: type, **options) -> ConversionInfo: ...
    def execute(self, source, target_type: type, **options) -> Distribution: ...   # the conversion itself

class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    # keyed on (type(source), target): the target is a distribution class, or a capability protocol (III.8)
    def convert(self, source, target_type: type, *,
                method: str | None = None, exact_only: bool = False, **options) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity makes the approximation explicit, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while an `approximate` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`). Realizing the registry as a subclass of the shared dispatch registry gives conversion registration, feasibility probing, prioritized selection, and cataloging without duplicating any of them (`D6 – Single source of truth`).

## IV.4 — Distributions from functions

### Contract

**A law from functions.** `distribution` builds a `Distribution` from a sampling function, a log-density, or both, as `function` builds a `Function` from a callable (V.1). It takes each function under the name of the operation it realizes, the event spec, and the optional keywords `output_name` and `label`: `sample(key)` returns one draw at a PRNG key, at the kind the event spec names, and `log_prob` or `unnormalized_log_prob` scores one value and returns a real scalar. A density receives an array for an array event, a `Record` for a record event, and the value itself for any other event. A call gives at least one function, at most one density, and an `event_spec` of any kind a `Distribution` declares, and a missing or non-callable function raises `TypeError`, which names it. An `OutputSpec` states the outputs, and a bare spec is read by III.7's rule: a bare `RecordSpec` is the positional form, a bare base spec is the keyword form under `output_name`, which such a spec requires and the others refuse, both with `TypeError`, and any other bare spec is the positional form (II.2), so a law over laws has no outputs unless its sampler returns a record that wraps them under a name. The label defaults to `ℙ`. The law claims the capability that each function given realizes:

- `sample`: `SupportsSampling`;
- `log_prob`: `SupportsLogProb`, which provides the unnormalized density as a normalized family's does;
- `unnormalized_log_prob`: `SupportsUnnormalizedLogProb`.

A law of a density alone is therefore unnormalized (III.8). A non-empty sample shape maps `sample` over keys split from the draw's key with `jax.vmap`. A sampler that does not trace in JAX, such as one that calls NumPy or an external program, and a sampler of an event that is not numeric are called at one key at a time instead, and their draws stack into the batch form of the event's kind. A density scores a batch of values along its leading axes, with `jax.vmap` when it traces.

Construction draws nothing and scores no value. It evaluates each function that traces abstractly, with `jax.eval_shape`: the abstract draw of `sample` must conform to `event_spec`, and each density must return a real scalar at a stand-in of one value of the event, or `ValueError` names the failed check. The abstract draw completes the event spec, as a family's parameters complete its own (III.7). A function whose `jax.eval_shape` fails does not trace, and construction reads nothing from it, so an event spec that no traced draw completes is taken as given, and an unspecified spec raises `TypeError`.

```python
def distribution(
    *,
    sample: Callable[[Key], Any] | None = None,
    log_prob: Callable[[Any], Array] | None = None,
    unnormalized_log_prob: Callable[[Any], Array] | None = None,
    event_spec: OutputSpec | TermSpec,
    output_name: str | None = None,
    label: str | None = None,
) -> Distribution: ...
    # at least one function and at most one density; output_name for a bare base spec
```

**A kernel from a function.** `conditional_distribution` builds a `ConditionalDistribution` (III.9) from a function of its given values that returns a law. Its call form takes the function and an optional label, as in `conditional_distribution(lambda mu, tau: Normal("y", mu, tau), label="lik", given_spec=...)`, and its decorator form on a `def` takes the same keywords. The kernel's label defaults to the function's `__name__`, and to `ℙ` for a lambda, so the decorator form labels the kernel after the function. Each named parameter of the function is an input, and its spec is its entry in `given_spec` or else its annotation, which must then be a term spec. `given_spec` is read through the `InputSpec` constructor from any of its four sources (II.2), and the idiom is `given_spec=d.event_spec`, cut down with `[...]`, `without`, and `&`, which declares the inputs that the outputs of a law `d` supply. A parameter with a default is an optional input (II.2), which holds a constant of the model, and its default's value declares it when neither does. Construction refuses a mismatch between `given_spec` and the function, each with a message that names the fix:
1. an input the function does not take raises `TypeError`, as "given_spec of 'fn' declares 'sigma', which the function does not take (it takes mu); leave it out with event_spec.without('sigma'), or pick the inputs with event_spec[['mu']]";
2. the fields of a group where the function takes the group raise `TypeError`, as "given_spec of 'fn' declares a and b, the fields of the group g; to declare the input g, index with a list: event_spec[['g']]";
3. a law or its spec given as the source raises `TypeError`, saying to pass the law's `event_spec`;
4. an input marked optional whose parameter has no default raises `ValueError`;
5. a parameter with no default and no spec raises `TypeError`, whose message gives the `given_spec` entry that declares it.

Construction evaluates the function once, abstractly, at a stand-in of each required input's spec and at the default of each optional input, and reads three things from the law it returns:

1. the event spec: the kernel takes the law's, or an explicit `event_spec` that states the law's outputs and to which the law's spec conforms, where a bare base spec takes the law's output name, and a support that a given value sets is left undeclared;
2. the claims: the kernel claims `SupportsConditionalSampling`, `SupportsConditionalLogProb`, and `SupportsConditionalUnnormalizedLogProb` exactly when the law claims the capability each one twins;
3. the guards: each twin's guard reports what the law's guard of the capability reported.

Binding every required input calls the function with each given value as the argument of that name, at the kind its input declares, as a function's body receives a draw (V.5), and with its default for each optional input left unbound. A record given for an input is laid out in the input's declared order (III.5). It returns the law the call returns, which must conform to the kernel's event spec, so a law that departs from it raises `ValueError`. Binding fewer inputs returns the kernel over the rest. In a merge, an output of the same name as an optional input feeds it (IV.2), so one kernel serves a model with the constant and a model with a prior on it. The function must be pure, so the law it returns is determined by its arguments. A joint that maps the kernel over many given values compiles the function and keeps the compiled program, and a call that reuses the program computes with the values that the globals the function reads had when it was compiled. A value that changes between calls is therefore an input.

```python
def conditional_distribution(
    fn: Callable[..., Distribution] | None = None,
    /,
    *,
    label: str | None = None,
    given_spec: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None,
    event_spec: OutputSpec | TermSpec | None = None,
) -> ConditionalDistribution | Callable[[Callable[..., Distribution]], ConditionalDistribution]: ...
    # conditional_distribution(fn) is the kernel of fn; without fn it is a decorator, and
    # @conditional_distribution on a def labels the kernel after the function
```

A kernel whose function returns `distribution(...)` reads the law's claims as for any law, so the kernel of a simulator, `conditional_distribution(lambda rate: distribution(sample=lambda key: jax.random.poisson(key, rate, (10,)), event_spec=counts, output_name="y"), given_spec={"rate": positive_scalar})`, claims conditional sampling and no density.

### Rationale

A law built from functions claims only what its functions realize (`D3 – Capability-based operations`). Its construction checks each function that traces against the event spec, which every consumer of the law reads (`D5 – Explicit, carried structure`), and it draws nothing, so a kernel that builds a law at every binding calls the law's functions only when an operation asks for a draw or a density (`C3 – Computational detail hidden by default, available on demand`). A kernel built from a function of its given values is written as that function, as a `Function` is (`C1 – Uniform interface to functions, distributions, and values`), and reading its claims from the law the function returns makes it advertise only what that law supports (`D3 – Capability-based operations`).
