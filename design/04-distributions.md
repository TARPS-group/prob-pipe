# Part IV — Constructing distributions

Part IV covers how distributions are constructed, from other distributions and from functions that a user writes.

| §    | Category           | Contents                                                                                              | Role                                                                                                   |
| ---- | ------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| IV.1 | From distributions | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from sub-distributions, with the factor and field access interfaces.              |
| IV.2 | From distributions | merging with `&`                                                                                      | Merges two terms or declarations by name, such as two (conditional) distributions into a joint one.    |
| IV.3 | From distributions | cross-type conversion (`converter_registry`)                                                          | Moving a distribution between representations, at a recorded fidelity.                                 |
| IV.4 | From functions     | `distribution` and `conditional_distribution`                                                         | A law built from a sampler, a density, or both, and a kernel built from a function that returns a law. |

## IV.1 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from sub-distributions**: it carries an explicit factorization into its parts, marked by the capability `SupportsFactors`, which `FactoredDistribution` and `FactoredConditionalDistribution` implement generically.

Both carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* from the components that feed each factor's slots, which the factors to its right offer in the conditional-first order (IV.2). The joint's event declaration is the merge of its factors' event declarations, a positional record over the disjoint union of their components (IV.2), and assembly and scoring rebuild each factor's draw at its declared kind. Each component belongs to exactly one factor; factor labels may repeat. Conditioning a `FactoredConditionalDistribution` on all of its given slots yields a `FactoredDistribution`. Sampling and the log-prob capabilities are the intersection of the factors'. A `FactoredConditionalDistribution`'s conditional capabilities pass its unfed slots to each factor's own conditional capability, and the joint's claim to each conditional capability is read from its factors' claims; a given that omits a slot or names one the joint lacks raises `KeyError`. The moment capabilities are decided at construction and are present exactly when the joint's structure makes the moment derivable. An edge-free joint derives its moments componentwise, so it has a moment exactly when every factor does; a joint of jointly Gaussian factors has its moments in closed form (VII.6); any other dependent joint carries no moment capability, since factor-wise conditional moments do not compose into a closed form. For example, with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `𝔼[y] = e^{1/2}` is not computable from the factors' means. The `by_component` route of VI.5 still computes exactly each component of such a joint's mean, variance, or quantile whose view has an exact route, as a root factor's view does.

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

**Field versus factor.** A **field** is a node of a draw that is not a group, named by its path (II.6). A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint has up to two interfaces, each through operators of its own.
- The **field interface** is available on every distribution: `d["intercept"]` is the view of III.7, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, component_name="intercept")` returns the complete factor named by one of `d`'s component names: a `Distribution`, or, for a dependent edge, a `ConditionalDistribution`. Several components may name the same factor; factors with no components remain accessible through `factors`.

**Renaming a joint.** `rename` renames a joint through its factors: the factor that offers a renamed component renames it, each factor that the component feeds renames the matching slot, and the result is the factored joint of the renamed factors over the same graph. A renamed slot of a conditional joint is renamed in every factor that names it. A rename that gathers components under a new node regroups the factors that offer them into a **grouped sub-joint**: the result has one factor per gathering node, which wraps that node. Its event declaration is positional over the one-field record of the group, so its one component is the group and its draws carry the group's level. The grouped sub-joint stays one factor of the joint. A sub-joint is labeled as `&` labels the joint of its factors (IV.2). A factor that a gathered component feeds conditions on the sub-joint through its renamed slot, which holds the node's whole record, so the dependencies carry over and the result stays factored. Conditioning on a node then takes the factored routes, as do the factor, the view, and the marginal at a node. A gathering whose groups condition on one another in a cycle has no factored form, as gathering `mu` and `theta` when `tau` conditions on `mu` and `theta` on `tau` does. Such a rename, like any rename the factors cannot carry, returns a law that renames the joint's values at its boundary. That law is the joint's law under the new names, and it claims no `SupportsFactors`. For example, renaming the eight-schools prior `Normal("mu", 0, 5) & HalfCauchy("tau", 0, 5) & Normal("theta_tilde", jnp.zeros(8), 1)` by `{"mu": "population/mu", "tau": "population/tau", "theta_tilde": "groups/theta_tilde"}` gives a joint of two factors: the sub-joint `population` of the `mu` and `tau` factors, whose draws are `{population: {mu, tau}}`, and the factor `groups`, which wraps the `theta_tilde` factor and draws `{groups: {theta_tilde}}`. In the centered model, where `theta` conditions on `mu` and `tau`, `groups` is a conditional sub-joint whose slot `population` holds `mu` and `tau`, and conditioning the prior on `population` binds that slot. The rename is also how a nested joint is built, one group at a time or all at once.

**Marginals of a joint.** Whether a marginal is exact depends on the factors' own marginal support and on the target's position in the dependence graph, so the factored classes resolve `_marginal` per path. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out without computation. What remains is integrating the extra ancestor fields back out, which is exact in three cases:
1. there are none, because the target is ancestrally closed, as for a root factor or an edge-free group;
2. no extra field feeds a factor of the closure, and each factor that holds extra fields is exact at its fields in the target, through its own `SupportsMarginals` or, for a `ConditionalDistribution` factor, `SupportsConditionalMarginals`;
3. the affected factors admit closed-form integration, as when they are jointly Gaussian.

In the second case the marginal is the joint of the reduced factors. At any other path the exact route's guard rejects the call, leaving the fallback available when sampling is supported and the fidelity controls permit it (VI.8).

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, within the one class hierarchy of distributions. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two differ in the mathematics. Deciding moment presence at construction follows `D3 – Capability-based operations`: a capability is advertised exactly when the object can compute it.

### Notes

- *Group views.* The field interface also accepts the path of a group. For example, when the event declaration nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns the view of the group, which offers the components `intercept` and `slope`, and `d[["coeffs"]]` returns the view that keeps the group's name.

## IV.2 — Merging distributions

### Contract

A **merge**, written `a & b`, combines two terms or two declarations by the names of their slots and components. It puts the components of both operands side by side, makes a slot that both operands take one slot, and lets each component of the right operand **feed** the slot of the same name of the left operand. It is written as a density is: `lik & prior` is `p(y | mu) · p(mu, tau)`. "Composition" keeps its meaning of `f ∘ g` and of a product of linear operators `A B`. This section defines the merge once, for every kind, and II.2, II.6, III.5, and III.6 point here.

**Which operands merge.** `&` merges two operands of one **merge category**:

| Merge category | Operands | Result |
|---|---|---|
| records | `Record`, `NumericRecord` | a record |
| laws and kernels | `Distribution`, `ConditionalDistribution` | a law, or a kernel when a required unfed slot is left |
| functions | `Function` | a function |
| batches | two batches over equal levels whose elements merge | a batch over the same levels |
| input declarations | two `InputSpec`s | an `InputSpec` |
| output declarations | two `OutputSpec`s | an `OutputSpec` |
| record specs | two `RecordSpec`s | a `RecordSpec` |
| specs of laws, kernels, functions, and batches | two of one of these | the spec of the merged term |

Operands of two categories raise `TypeError`, so `InputSpec & OutputSpec` raises, with a message that says to read an output declaration as slots with `InputSpec(output_spec)` (II.2). An array and an opaque value belong to no category, since they offer no component, and `&` on two `NumericArray`s keeps NumPy's elementwise meaning. A law with a point-mass factor has no joint density, so a record merged with a law and a function merged with a law raise `TypeError`.

Two batches have **equal levels** when they have the same level names, in the same order, with the same axis sizes. Their merge is the batch over those levels whose element at each position is the merge of the two elements there, each element read as one operand:
1. a base element as the record `{n: value}`, under the name `n` of its batch's element spec (II.5);
2. a record as itself;
3. a law, a kernel, or a function as one factor, with the declarations of its term spec.

Two batches of arrays therefore merge into a batch of records.

**Factors and unfed slots.** A merge of laws and kernels reads each operand as its **factors** (IV.1): the factors of a factored joint, apart from a grouped sub-joint, which stays one factor, and any other law or kernel as one factor. The **unfed slots** of an operand `t`, written `unfed(t)`, form the input declaration of the slots that `t`'s factors take and no factor of `t` feeds, optional ones included, in order of first appearance:
1. for a kernel, its `given_spec`;
2. for a law that a family builds, none;
3. for a joint law, the optional slots its factors left at their defaults, which are not slots of the law (III.7);
4. for a grouped sub-joint read as one factor, its own `given_spec` when it is a kernel, and none when it is a law.

For a record, a function, or a declaration, `unfed(t)` is its input declaration: none for a record, `f.input_spec` for a function, and `I` itself for an input declaration. A spec, and an element in a merge of batches, is read as one factor. `components(t)` is the set of the components of `t`'s output declaration, or of a record spec's top-level names. In `a & b`, writing `names(I)` for the slot names of an input declaration `I`:
- `fed = names(unfed(a)) ∩ components(b)` is the set of names that the right operand feeds;
- `shared = (names(unfed(a)) − fed) ∩ names(unfed(b))` is the set of names that both operands' factors take and neither feeds;
- `β` is the substitution of the merge's dimension scope (II.1), which starts from both operands' stored scopes and adds every binding that M6 and M7 make.

**The conditions.** `a & b` raises unless all of these hold:

| # | Condition | Error |
|---|---|---|
| M1 | `a` and `b` are of one category, and two batches have equal levels and elements of one category | `TypeError`, and `ValueError` for unequal levels |
| M2 | each law, kernel, or function states an output declaration with no type hole, and a function states its input declaration too; each output declaration is in the keyword form or positional over a `RecordSpec`, so its produced term is a base value or a record | `TypeError`, naming the type hole or the missing declaration, or saying to wrap the produced term under a name |
| M3 | no name is a component of both operands | `ValueError`, naming the component and both operands |
| M4 | **the order rule:** the right operand's factors take no unfed slot, required or optional, that the left operand offers: `names(unfed(b)) ∩ components(a) = ∅` | `ValueError`, saying to put the operand that offers the name on the right |
| M5 | each operand's unfed slots and components are disjoint | `ValueError`, saying to rename one |
| M6 | the spec of each component `c` of `b` that is in `fed` conforms to the spec of the slot `c` in `unfed(a)` (II.1) | `ValueError`, naming both specs |
| M7 | the specs in `unfed(a)` and in `unfed(b)` of each slot `s` that is in `shared` unify (II.1) | `ValueError`, naming both specs |
| M8 | all specs of both operands, with any stored scopes, form one dimension scope | `ValueError`, naming the dimension and both sizes |

For a merge of batches, M2 to M8 read the elements' specs. When both operands that M3 refuses are views of one parent, its message adds that a merge treats two views as independent factors, and that the joint law of both is the parent indexed once, as `hier["s1", "s2"]`. When only an operand's stored scope holds the name that M8 refuses, its message names that operand and suggests merging the operand's parts in another order, since `with_dim_names` cannot rename a stored name.

**The result.**

| # | Part | Value |
|---|---|---|
| R1 | Components | `output_spec(a & b) = β(output_spec(a) & output_spec(b))`: `a`'s components, then `b`'s; a component at the root named `n` becomes the child `n` |
| R2 | Unfed slots | `unfed(a & b) = β(unfed(a).without(*fed) & unfed(b))`: every unfed slot of the factors that no component feeds, `a`'s first; a shared slot carries the unification of its specs and is optional only when optional in every operand that has it; a fed slot leaves, whether required or optional |
| R3 | Kind | records give a record; laws and kernels give a kernel whose `given_spec` is `unfed(a & b)` when a required unfed slot is left, and otherwise a law; functions give a function whose `input_spec` is `unfed(a & b)`; batches give a batch over the same levels |
| R4 | Factors and scope | for laws and kernels, `a`'s factors followed by `b`'s, and the dimension scope, which the result stores |
| R5 | Label | the operands' labels joined with `·`, as **Labeling the result** states |

For laws and kernels, R3 reads:

| required slots of `unfed(a & b)` | result |
|---|---|
| `∅` | `FactoredDistribution` — a joint `Distribution`, whose factors take their optional slots at their defaults when it is sampled or scored |
| `≠ ∅` | `FactoredConditionalDistribution` — a joint `ConditionalDistribution`, with `unfed(a & b)` as its `given_spec` |

A kernel whose slots are all optional is still a kernel, while a merge that leaves only optional slots is a law, so `k_opt & Normal("z", 0.0, 1.0)` samples `k_opt` at its default although `sample(k_opt)` raises. A binding that supplies a shared slot fills it with one value, and a binding that omits a shared optional slot leaves each factor at its own default, since a default belongs to the term that owns it (II.2).

**The value of a merge.** Two records merge by top-level name, with nothing fed: `a & b` is the record of `a`'s top-level children followed by `b`'s. A draw `z` of a merge of laws and kernels at an input record `u` is a record with the components of `a` followed by those of `b`. The right operand is sampled first, since nothing feeds it:

```text
z_b ~ b( u↾unfed(b) )
z_a ~ a( u↾(unfed(a) − fed)  ⊎  rec_b(z_b)↾fed )
z   = rec_a(z_a) ⊎ rec_b(z_b)
```

Here `rec_t` turns `t`'s produced term into the record of its components, `{n: v}` for a component at the root named `n` and the record itself otherwise, and `dec_t` is its inverse. `r↾N` keeps the top-level children of `r` named in `N`, and `r ⊎ q` joins two records whose top-level names are disjoint. Where both operands have densities, the same law is

```text
p_{a&b}(z ; u)  =  p_a( dec_a(z↾components(a)) ; u↾(unfed(a) − fed) ⊎ z↾fed )  ·  p_b( dec_b(z↾components(b)) ; u↾unfed(b) )
```

So each factor's density receives its own draw, at its declared kind: an array and a one-field record may both offer `beta`, and their densities receive different kinds. Several slots fed from one factor share its one draw. A function is the kernel whose law at each value of its slots is a point mass at its return value, so a merge of functions follows the same rule deterministically, with `b` evaluated first and its components fed to `a`. A merge of batches follows its elements' rule at each position.

```python
model = lik & prior        # mu is fed; the components y, mu, tau; no unfed slot is left, so model is a law
prior & lik                # ValueError (M4): lik takes mu, which prior offers; write lik & prior
lik_nee & lik_lai          # shared = {theta}: one slot theta, the unification of its specs; a kernel
lik_opt & prior            # mu is fed; s stays unfed and optional; a law, sampled at s's default
Normal("x", 0.0, 1.0) & Normal("x", 2.0, 1.0)            # ValueError (M3): x is a component of both
lik & Record({"mu": 0.3})                                 # TypeError (M1): a kernel and a record
Record({"a": 1.0}) & Record({"b": 2.0, "g": {"c": 3.0}})  # Record {a, b, g: {c}}
Record({"g": {"c": 1.0}}) & Record({"g": {"e": 2.0}})     # ValueError (M3): g is a component of both
```

**Merging declarations.** `O & P` on two output declarations is the output declaration of two side-by-side terms, with nothing fed: the positional record of `O`'s components followed by `P`'s, under M2, M3, and M8. A component at the root named `n` becomes the child `n`, and a record contributes its children unchanged, so every path of an operand is a path of the result with the same spec. `I & J` on two input declarations is the input declaration of two terms that share their slots, with nothing fed: the slots of either in order of first appearance, under M7 and M8, a shared slot carrying the unification of its specs and staying optional only when optional in every operand that has it. `R & S` on two record specs is the union of their children, and a top-level name in both raises. A merge of laws and kernels builds its components and its unfed slots by these rules, so `(lik & prior).event_spec == lik.event_spec & prior.event_spec`. `&` on two `DistributionSpec`s, `ConditionalDistributionSpec`s, `FunctionSpec`s, or `BatchSpec`s gives the spec of the merged term, so a function that builds a model can declare its output before it runs. A merged law or kernel stores its dimension scope, while a merged declaration, record spec, or function applies `β` and keeps no binding.

```python
OutputSpec(x=NumericArraySpec(())) & grouped.event_spec
# OutputSpec(RecordSpec(x=..., a=..., g=RecordSpec(c=..., e=...))): x moves from the root to the path x
InputSpec(theta=("p",), x=()) & InputSpec(theta=(3,), z=())
# InputSpec(theta=NumericArraySpec((3,)), x=..., z=...): theta unifies, and p is bound to 3
OutputSpec(post.spec) & prior.event_spec   # TypeError (M2): the first produces a law; wrap it under a name
```

**Properties.** The rules give six properties, each under its condition:
1. **Associative when both groupings are valid.** For laws, kernels, records, record specs, output declarations, and batches of records or of base values, `(a & b) & c` and `a & (b & c)`, when both are valid, have the same factors, components in the same order, unfed slots with the same specs and optional flags, stored scope, and value. Their validity can differ only where the two groupings check different pairs of specs: at a name that `a` and `b` both take and `c` offers, whose specs in `a` and `b` do not unify while `c`'s conforms to each, and where two different free dimension names meet at one axis in one grouping but are bound first in the other. Input declarations and functions, which store no scope, also agree only when no dimension name that one grouping binds occurs free in the operand merged with later.
2. **The sides stay disjoint:** no name is both an unfed slot and a component of a merge.
3. **Commutative up to order** when neither operand feeds the other: `a & b` and `b & a` are valid together and give the same law or value, with components in another order.
4. **Paths are kept:** every path of an operand's output declaration is a path of the result's, with the same spec under `β`.
5. **Sampling commutes with merge** for laws that produce records and do not feed each other, when `a & b` is valid: `sample(a & b)` is distributed as `sample(a) & sample(b)`.
6. **Renaming commutes with merge:** `(a & b).rename(m) == a.rename(m_a) & b.rename(m_b)`, where `m` maps names to names injectively over both operands' names, `m_a` and `m_b` are its restrictions to each operand, and `m` renames no unfed slot of a law operand's factors.

**What a merge reads.** A merge reads names and factors, and never a label:
1. **a later merge feeds a joint's unfed optional slot**, so `(lik_opt & prior) & s_prior` feeds `s` and is the same law as `lik_opt & (prior & s_prior)`;
2. **a later merge can share it**, so for `M = lik_opt & Normal("mu", 0.0, 0.001)` and a kernel `k2` over the slot `s` that `M` neither feeds nor is fed by, `M & k2` is a kernel whose slot `s` is required, and for an unrelated kernel `kw(v | w)`, `M & kw` is a kernel with the optional slot `s`;
3. **equal laws can merge differently**, since `lik_opt & prior` equals `lik & prior` when `lik` is `lik_opt` at its default, while their merges with `s_prior` differ;
4. **a result keeps an unfed optional slot when it keeps, unchanged, the factor that takes it**, so a view, which is one factor, and a marginal that keeps the factors can merge differently;
5. **a merge of specs can be valid where the merge of terms raises**, since a `DistributionSpec` states no unfed slot of its factors, so `(a & b).spec == a.spec & b.spec` when neither operand's factors take an unfed slot that its input declaration does not state;
6. **a merge of batches reads each element as one factor**, so an element's unfed optional slots are not fed;
7. **two views of one parent merge as independent factors**, so `hier[["s1"]] & hier[["s2"]]` is the product of the two marginals, while `hier["s1", "s2"]` is their joint law.

A merge is not composition, which passes a whole result by position and returns only the outer result. It is not conditioning either: `condition_on(lik, {"mu": 0.5})` binds `mu` to a fixed value, while `lik & prior` keeps `mu` as a component, fed by a law. A merge never lifts: a component feeds a slot only when its spec conforms, since lifting the receiving term over a law-valued component would ignore that law's dependence on everything else the receiver takes.

**The most specific class.** `&` returns the **most specific** class, recomputed from the flattened factor graph at each step. A family supplies a more specific factored class, such as `FactoredMultivariateGaussian` (VII.6), by registering it at import with a predicate over the flattened factors, and `&` constructs the most specific registered class whose predicate holds. The refinement after binding (VII.6) reads the same registrations. `A & B & C` builds one flat joint of N factors, with independent factors commuting and dependent ones kept in conditional-first order. The joint stores its factors with their shared slots unified, and the unification is canonical, so derived names and fingerprints are deterministic. Two slots or two dimensions that are different quantities are renamed apart first, with `rename` and `with_dim_names`. A factor that is neither a `Distribution` nor a `ConditionalDistribution` raises `TypeError` at construction, and `__and__` returns `NotImplemented` for an operand of another category.

**The joint's event.** The joint's event declaration is the merge of its factors' event declarations (R1), so its draw holds the components in canonical factor order, each factor's in its declared order. Assembly extracts each factor's components from that factor's one draw, and scoring rebuilds each factor's draw at its declared kind before calling its density method. The joint keeps the declarations it needs to rebuild each draw and reads the form from them alone.

**Labeling the result.** The joint's label joins the operands' current labels with `·`, each grouped by the rules of II.4 so that it reads as one unit. So `lik & prior` is labeled `lik·prior`, `(lik & prior).with_label("posterior") & d` is labeled `posterior·d`, and `lik & d` for a law labeled `my model` is labeled `lik·[my model]`. Labels join associatively: an operand whose label is a product joins as it is, so `(lik & prior) & d` and `lik & (prior & d)` are both labeled `lik·prior·d`. Exchanging independent operands may change the label while preserving the joint law.

**The notation of a product.** A product that `&` or `joint` builds is unlabeled: its expression (II.4) is the product of its operands' expressions, so its notation joins their notations with `·`, as `lik(y | mu)·prior(mu)`. A product given a label, by `with_label` or by the `FactoredDistribution` constructor, displays by that label, as `model(y, mu)`. A law built from a product, such as a copy or a grouped sub-joint, keeps its expression. A product without a label that holds fixed paths (VI.6) displays by its grouped label and its signature, as `(lik·prior)(mu; y)`, since its factors' notations do not show the fixed paths. A marginal over the whole events of several factors is a product without a label whether or not the joint is labeled, as `marginal(model, ["a", "b"])` displays as `a(a)·b(b)` (VI.8). A product that a function returns takes the function's `output_label` as its label, so it displays as `predict(y, mu)` (III.3). A product without a label enters a further merge as its factors, so `(lik & prior) & d` displays as `lik(y | mu)·prior(mu)·d(z)`, and a labeled product enters as one operand, so `model & d` displays as `model(y, mu)·d(z)` and is labeled `model·d`.

A merge determines structure from the names of components and slots alone. Every operand enters as its flattened factors, so the joint's factors, its components, and the slots they feed are the same whatever the operands are called: `AB & C` and `AB.with_label(AB.label) & C` are one joint under one label, and `(lik & prior).with_label("posterior") & d` is the three-factor joint of `lik`, `prior`, and `d`. The only effect of `with_label` on a later merge is the text it contributes to the derived label.

```python
def __and__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product, where `fed = ∅`, is an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. One definition serves every category, so records, laws, functions, batches, and declarations merge by the same names under the same rules, and a merge of terms has the merge of their declarations as its declaration (`D6 – Single source of truth`). The order rule keeps the joint well formed: in a chain `a & b & c` each operand is fed only from its right, so the written order is a topological order of the factors, the density reads left to right, and the result's sides stay disjoint; associativity holds with or without the rule, under the conditions of property 1 (`D1 – Mathematical fidelity`). A fed component conforms to its slot, because the slot must admit every value it is fed, while a shared slot unifies, because one value fills both slots and neither operand is privileged (`D1 – Mathematical fidelity`). M2 holds because the result stores the value at each component as a field and must rebuild each operand's produced term from those values, which a base value and a record allow and a law, a kernel, a function, or a batch held under no name does not. Records merge shallowly, as laws do, so a group has one source, and `sample(a) & sample(b)` succeeds where `a & b` does (`C1 – Uniform interface to functions, distributions, and values`). Deriving the label deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`). An unlabeled product displays factor by factor because a single joined label followed by the joint's signature, as `lik·prior(y, mu)`, would read as one law over `y` and `mu`. A merge is written with an operator so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and every result is a first-class joint that merges further (`D4 – Closed system of objects under operations`). The operator is `&`, which reads as "both", because `*` is arithmetic on values (`C5 – Naming for unambiguous meaning`).

### Notes

- *Operator coexistence.* `*` scales: it multiplies two `NumericArray`s, scales a `LinOp` by a number, and scales a random function by a number, while `&` merges. On a `NumericRecord`, `&` is the merge and is not among the vector-space operators of III.5.
- *Interim whole forms in a merge.* Until every family wraps its draws (II.2), a merge accepts an operand whose event is an interim whole form and stores that operand's produced term whole, as the field of its name, so `Normal("z", 0.0, 1.0) & random_function` stores the random function under its name. The paths a merge keeps do not hold for such an operand.

## IV.3 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or a backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current class to a requested target while preserving its law and its event declaration. A **converter** is a binary dispatch method (II.7) that declares the source types it converts *from* and the target types it converts *to*. Conversions differ in fidelity, so each converter declares whether it is **exact** (II.7): exact for an equivalent representation, approximate for a stand-in such as moment matching or a Monte Carlo representation. A caller may set `exact_only`, which the registry's `check` enforces (II.7).

**The registry.** The **converter registry** is the binary dispatch registry keyed on `(type(source), target)`. A target is a distribution class or a capability protocol (III.8). Specificity on the target side is the position of the requested target in the declared target's method-resolution order, so a converter that declares the requested target itself is closest, and a structural match counts as least specific, as on the source side (II.7). For a protocol target the registry considers the converters that promise the required guarded capability. The registry's `check` and `execute` test the source first: a source that already satisfies the target, which for a protocol target means claiming the protocol and passing its guard, is reported feasible with no converter selected and is returned as it is, and `convert` gives it fresh identity (VI.10). A protocol target is feasible when both the claim of the protocol and its guard can be established, on the source or on the converter's promised result. A converter that promises a guarded protocol whose guard can be decided only once the converted law exists is reported unresolved, with that guard as the pending requirement (II.7).

**What a conversion preserves.** A conversion changes the representation alone. The result carries the source's event declaration, its component names, kind, and form included, and realizes the same law up to the recorded fidelity. It keeps the source's declared support, so a converter refuses a result on another support, as a moment-matched `Normal` fit to a law on the positive half-line would be. The converter option `check_support=False` overrides the refusal, and the result then has the family's support. Flattening a record event to an array changes the event space, so it is the declared isomorphism of III.7 rather than a conversion.

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

**A law from functions.** `distribution` builds a `Distribution` from a sampling function, a log-density, or both, as `function` builds a `Function` from a callable (V.1). It takes each function under the name of the operation it realizes, the event declaration, and the optional keywords `component` and `label`: `sample(key)` returns one draw at a PRNG key, at the kind the event declaration names, and `log_prob` or `unnormalized_log_prob` scores one value and returns a real scalar. A density receives an array for an array event, a `Record` for a record event, and the value itself for any other event. A call gives at least one function, at most one density, and an `event_spec` of any kind a `Distribution` declares, and a missing or non-callable function raises `TypeError`, which names it. An `OutputSpec` names its components and a bare `RecordSpec` is positional, offering its fields, while any other bare spec is read by `OutputSpec.default` under `component` (II.2), which such a spec requires and the other two refuse, both with `TypeError`. The label defaults to `ℙ`. The law claims the capability that each function given realizes:

- `sample`: `SupportsSampling`;
- `log_prob`: `SupportsLogProb`, which provides the unnormalized density as a normalized family's does;
- `unnormalized_log_prob`: `SupportsUnnormalizedLogProb`.

A law of a density alone is therefore unnormalized (III.8). A non-empty sample shape maps `sample` over keys split from the draw's key with `jax.vmap`. A sampler that does not trace in JAX, such as one that calls NumPy or an external program, and a sampler of an event that is not numeric are called at one key at a time instead, and their draws stack into the batch form of the event's kind. A density scores a batch of values along its leading axes, with `jax.vmap` when it traces.

Construction draws nothing and scores no value. It evaluates each function that traces abstractly, with `jax.eval_shape`: the abstract draw of `sample` must conform to `event_spec` (II.1), and each density must return a real scalar at a stand-in of one value of the event, or `ValueError` names the failed check. The abstract draw completes the declaration, as a family's parameters complete its own (III.7). A function whose `jax.eval_shape` fails does not trace, and construction reads nothing from it, so a declaration that no traced draw completes is complete as given, and a pending type raises `TypeError`.

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
    # at least one function and at most one density; component for a bare spec other than a record
```

**A kernel from a function.** `conditional_distribution` builds a `ConditionalDistribution` (III.9) from a function of its given values that returns a law. Its call form takes the function and an optional label, as in `conditional_distribution(lambda mu, tau: Normal("y", mu, tau), label="lik", given_spec=...)`, and its decorator form on a `def` takes the same keywords. The kernel's label defaults to the function's `__name__`, and to `ℙ` for a lambda, so the decorator form labels the kernel after the function. Each parameter of the function is a slot, and its spec is its entry in `given_spec`, which takes any of the four sources of II.2, or else its annotation, which must then be a term spec. A parameter with a default is an optional slot (II.2), which holds a constant of the model, and its default's value declares it when neither does. A parameter with no default and no declaration raises `TypeError`, whose message gives the `given_spec` entry that declares it. Construction evaluates the function once, abstractly, at a stand-in of each required slot's type and at the default of each optional slot, and reads three things from the law it returns:

1. the event declaration: the kernel declares the law's, or an explicit `event_spec` that names the law's components and to which the law's type conforms, where a bare spec other than a record names the law's one component, and a support that a given value sets is left undeclared;
2. the claims: the kernel claims `SupportsConditionalSampling`, `SupportsConditionalLogProb`, and `SupportsConditionalUnnormalizedLogProb` exactly when the law claims the capability each one twins;
3. the guards: each twin's guard reports what the law's guard of the capability reported.

Binding every required slot calls the function with each given value as the argument of that name, at the kind its slot declares, as a function's body receives a draw (V.5), and with its default for each optional slot left unbound. It returns the law the call returns, which must agree with the kernel's event declaration, so a law that departs from it raises `ValueError`. Binding fewer slots returns the kernel over the rest. In a joint, a factor that offers a component named as an optional slot feeds it (IV.2), so one kernel serves a model with the constant and a model with a prior on it. The function must be pure, so the law it returns is determined by its arguments. A joint that maps the kernel over many given values compiles the function and keeps the compiled program, and a call that reuses the program computes with the values that the globals the function reads had when it was compiled. A value that changes between calls is therefore a given slot.

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

A kernel whose function returns `distribution(...)` reads the law's claims as for any law, so the kernel of a simulator, `conditional_distribution(lambda rate: distribution(sample=lambda key: jax.random.poisson(key, rate, (10,)), event_spec=counts, component="y"), given_spec={"rate": positive_scalar})`, claims conditional sampling and no density.

### Rationale

A law built from functions claims only what its functions realize (`D3 – Capability-based operations`). Its construction checks each function that traces against the declaration, which every consumer of the law reads (`D5 – Explicit, carried structure`), and it draws nothing, so a kernel that builds a law at every binding calls the law's functions only when an operation asks for a draw or a density (`C3 – Computational detail hidden by default, available on demand`). A kernel built from a function of its given values is written as that function, as a `Function` is (`C1 – Uniform interface to functions, distributions, and values`), and reading its claims from the law the function returns makes it advertise only what that law supports (`D3 – Capability-based operations`).
