# Part IV — Distributions

Part IV covers how new distributions can be constructed from other distributions, such as by composing them into joint distributions and converting between representations.

| §    | Category   | Contents                                                                                              | Role                                                                                                         |
| ---- | ---------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| IV.1 | Structure  | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from named sub-distributions, with the factor and field access interfaces.              |
| IV.2 | Structure  | the `*` operator                                                                                      | Constructs a joint (conditional) distribution from two (conditional) distributions.                          |
| IV.3 | Registries | cross-type conversion (`converter_registry`)                                                          | Moving a distribution between representations, at a recorded fidelity.                                       |

## IV.1 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from named sub-distributions**: beyond being an ordinary distribution, it carries an explicit factorization into its parts, marked by the capability `SupportsFactors`, which `FactoredDistribution` and `FactoredConditionalDistribution` implement generically.

Both carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's given fields against the fields produced by earlier factors, rather than stored. The joint's event declaration is an exposed `RecordSpec` over the disjoint union of the factors' declared output components (II.2, IV.2). Extraction and reconstruction preserve each factor's event packaging. Factor names are unique across the list. Conditioning a `FactoredConditionalDistribution` on all of its given fields yields a `FactoredDistribution`. Sampling and the log-prob capabilities are the intersection of the factors'. The moment capabilities are decided at construction and are present exactly when the joint's structure makes the moment derivable. An edge-free joint derives its moments componentwise, so it has a moment exactly when every factor does; jointly Gaussian factors are exact (VII.6); any other dependent joint carries no moment capability, since factor-wise conditional moments do not compose into a closed form. For example, with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `E[y] = e^{1/2}` is not computable from the factors' means.

```python
@runtime_checkable
class SupportsFactors(Protocol):
    @property
    def factors(self) -> tuple[Distribution | ConditionalDistribution, ...]: ...   # ordered

class FactoredDistribution(Distribution, SupportsFactors): ...
class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors): ...

# numeric markers for which of the (given, event) sides is numeric. For conditional distributions,  "Numeric" before "Distribution" marks the event as numeric and before "Conditional" marks the given as numeric.
class FactoredNumericDistribution(FactoredDistribution): ...                                # unconditional joint, event numeric
class FactoredConditionalNumericDistribution(FactoredConditionalDistribution): ...          # conditional joint, event numeric
class FactoredNumericConditionalDistribution(FactoredConditionalDistribution): ...          # conditional joint, given numeric
class FactoredFullyNumericConditionalDistribution(
        FactoredNumericConditionalDistribution, FactoredConditionalNumericDistribution): ... # conditional joint, both numeric
```

**Field versus factor.** A **field** is a named part of a draw, that is, a path in the event schema. A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint exposes up to two interfaces, never through the same operator.
- The **field interface** is available on every distribution: `d["intercept"]` is the field view of III.7, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, "coeffs")` returns the factor of that name, which is a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. There need be no factor for a given field, and no field for a given factor.

**Marginals of a joint.** Whether a marginal is exactly available depends on the factors' own marginal support and on the target's position in the dependence graph, so the factored classes resolve `_marginal` per path rather than wholesale. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out for free. What remains is integrating the extra ancestor fields back out, which is exact in three cases:
1. there are none, because the target is ancestrally closed, as for a root factor or an edge-free group;
2. the reduction lies within a single factor and delegates to that factor's own `SupportsMarginals`;
3. the affected factors admit closed-form integration, as when they are jointly Gaussian.

On any other path the exact route's guard declines, leaving the fallback available when sampling is supported and the fidelity controls permit it (VI.8).

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, instead of sitting in a parallel class tower. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two differ in the mathematics. Deciding moment presence at construction follows `D3 – Capability-based operations`: a capability is advertised exactly when the object can compute it.

### Notes

- *Group views.* The field interface also accepts an interior path, which names a group of fields rather than a single field. For example, when the event declaration nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns a view of the marginal over the whole group.

## IV.2 — Composition

### Contract

Composition builds a factored distribution from parts, written as an *expression*: a single binary operator `*` combines `Distribution`s and `ConditionalDistribution`s into a single joint (conditional) distribution. The kind of the result is derived from the operands. The base objects expose `*` via `__mul__`, which delegates to the operator.

**The `*` operator.** `A * B` composes two operands into a joint. It is **conditional-first**: the left operand may condition on the right, so `lik * prior` reads as the density p(y | β) · p(β), while the reverse, with the producer on the left of its consumer, is an error. Characterize each operand by its **produced slots** `F`, which are exactly the keys of its `event_spec.components` (II.2), and its **unmet given slots** `G`, which are empty for a `Distribution` and the given slots for a `ConditionalDistribution`. The dependency topology is fixed by the name sets; matched specs must also unify:

```
bound  = G_A ∩ F_B            # dependency edges: left A conditions on a name that right B produces
unmet  = (G_A − F_B) ∪ G_B    # residual exogenous givens — met by no factor
require  F_A ∩ F_B = ∅         # each name is produced exactly once
     and G_B ∩ F_A = ∅         # B must not consume a name A produces — else reorder (producer on the right)
law:  p(F_A, F_B | unmet) = p_A(F_A | bound ∪ (G_A − F_B)) · p_B(F_B | G_B)      # reads left → right
```

If `unmet ≠ ∅` then the result is a conditional distribution, or a plain distribution otherwise:

| `unmet` | result |
|---|---|
| `∅` | `FactoredDistribution` — a joint `Distribution` |
| `≠ ∅` | `FactoredConditionalDistribution` — a joint `ConditionalDistribution`, its `given_spec` exactly `unmet` |

`*` returns the **most specific** class, recomputed from the *flattened* factor graph at each step. `A * B * C` builds one flat N-factor joint, with independent factors commuting and dependent ones kept in conditional-first order. Same-named unmet givens unify into one slot of the joint: their specs must unify, a disagreement raising at composition, and binding the slot feeds every factor that names it — two givens that are different quantities are renamed apart first, as fields and levels already require. Symbolic dimensions follow the same rule: the operands share one dimension scope, so a name two factors both use is one dimension, bound once and required to agree (II.1), and two dimensions that are different quantities are renamed apart first with `with_dim_names`. The joint stores the factors so unified. The renaming is canonical, so derived names and fingerprints stay deterministic.

**Packing the joint.** The output is an exposed record of the components in canonical factor order, preserving each declaration's component order. Assembly extracts components from each factor's one draw; scoring reconstructs that factor's original event value before calling its density method. Extraction reads the declaration: a whole-term component is the draw itself, and an exposed record's components are its immediate children, so reconstructing all of a factor's components restores the draw's kind, structure, and coordinates. In particular, an array and a one-field record may both export `beta`, but their density implementations receive different declared event kinds. Multiple connections from one factor share the same draw. A name match is insufficient by itself: the supplying component's spec must unify with the receiving input slot's spec. The joint retains the declarations needed for reconstruction and never infers packaging from the number of fields.

**Naming the result.** A joint is *derived*, not created by the user, so `*` **auto-derives** its `name` deterministically from its factors. The factors are listed in **canonical order**, which is the conditional-first topological order of the flattened factor graph with incomparable factors ordered lexicographically by the fields they produce, and their names are joined by `·`. So `lik * prior` is named `lik·prior`, and because neither association nor the ordering of independent factors changes the canonical list, `A * B * C`, `(A * B) * C`, and `A * (B * C)` produce the same joint distribution.

Re-composition reads `name_is_auto`. An auto-named factored operand is **flattened**: its factors enter the new joint directly, its old name is discarded, and a fresh name is derived from the full factor list. An operand whose name the user has set with `with_name` is **not** flattened. It enters as a single factor under that name, and that name appears as one token in the parent's derived name. This grouping does not change its produced-component interface or any name-based connection; factor labels and component names are distinct. So `(lik * prior).with_name("posterior")` both labels the joint and, in any later composition, keeps it as the single factor `posterior`.

```python
def __mul__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product, where `bound = ∅`, is just an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. Deriving the name deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`), while `with_name` lets the user impose grouping where it carries intent. The conditional-first order is already a valid topological listing, so composition is associative and acyclicity is automatic, with no separate graph inference. Associativity rests on the `G_B ∩ F_A = ∅` requirement, under which the validity of `(A * B) * C` and `A * (B * C)` coincide. Composition is written as an expression so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and every result is a first-class joint that composes further (`D4 – Closed system of objects under operations`).

### Notes

- *Operator coexistence.* `*` also denotes scalar scaling on some objects, such as a random function or a linear operator. The two coexist by operand-type dispatch: `Distribution` and `ConditionalDistribution` operands compose, while scalar operands scale.

## IV.3 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or a backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current class to a requested target while preserving its law and its event declaration. A **converter** is a binary dispatch method (II.7) that declares the source types it converts *from* and the target types it converts *to*. A conversion is rarely unique, so each converter carries a **fidelity** on the shared scale of II.7: `exact` for an equivalent representation, `approximate` for a stand-in such as moment matching or a Monte Carlo representation. A caller may set `min_fidelity` as a floor, which `check` enforces.

**The registry.** The **converter registry** is the binary dispatch registry keyed on `(type(source), target)`. A target is a distribution class or a capability protocol (III.8). For a protocol target the registry considers the converters that promise the required guarded capability, and a source already satisfying the claim needs no conversion. A protocol target is feasible only when its claim and guard can be established, on the source or on the converter's promised result; class membership alone does not establish an instance-dependent capability.

**What a conversion preserves.** A conversion changes the representation and nothing else. The result carries the source's event declaration, its component names, kind, and packaging included, and realizes the same law up to the recorded fidelity. The event space never changes: flattening a record event to an array is the declared isomorphism of III.7, not a conversion.

**Check and execute.** A converter has a non-executing `check` and an `execute`. `check` returns a `ConversionInfo`: the promised target spec, the representation class when known, the capabilities guaranteed on the result, and, as pending requirements, whatever it cannot settle without the source's values. It never samples, fits, or evaluates a density. `execute` constructs the target. The function stack plans through `check` at normalization and constructs through `execute` at execution (V.4, V.9), and `convert` exposes the same two steps to the user (VI.10).

```python
@dataclass(frozen=True)
class ConversionInfo(MethodInfo):
    target_spec: DistributionSpec | None   # None when still unresolved
    target_class: type | None
    capabilities: tuple[type, ...]         # claims guaranteed by this conversion
    # pending requirements and local fidelity are inherited from MethodInfo

class Converter(BinaryDispatchMethod):
    name: str
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (source, target) types
    def check(self, source, target_type: type, *,
              min_fidelity: Fidelity | None = None) -> ConversionInfo: ...
    def execute(self, source, target_type: type) -> Distribution: ...             # the conversion itself

class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    # keyed on (type(source), target): the target is a distribution class, or a capability protocol (III.8)
    def convert(self, source, target_type: type,
                method: str | None = None, min_fidelity: Fidelity | None = None) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity makes the approximation explicit, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while an `approximate` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`). Realizing the registry as a subclass of the shared dispatch registry gives conversion registration, feasibility probing, prioritized selection, and cataloging without duplicating any of them (`D6 – Single source of truth`).
