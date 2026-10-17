# Declarations and addressing

> **Status: reconciled plan, 2026-10-17.**
> This text replaces [`declarations.md`](declarations.md) and [`addressing.md`](addressing.md), as [`decisions.md`](decisions.md) records (2026-10-17).
> It merges three sources:
> 1. the body of [#571](https://github.com/TARPS-group/prob-pipe/issues/571), on declarations;
> 2. jhuggins's reply on #571 (2026-10-15);
> 3. the grammar-B addressing plan for [#573](https://github.com/TARPS-group/prob-pipe/issues/573) (2026-10-16).
>
> Where the sources disagree, the addressing plan wins, then the reply, then the body (decisions, 2026-10-17).
> Every decision in [`decisions.md`](decisions.md) binds this text, and [`gaps-and-conflicts.md`](gaps-and-conflicts.md) D1 records how each disagreement was resolved.
> The appendix lists what moved from each source.
> Code references are to `dev/overhaul` at `a703f15e`, and every "Today" example was run there.

A law in ProbPipe and every value derived from it should be indexed by the same paths.
The values derived from a law are:
- its draws, and batches of draws;
- its summaries;
- its views and marginals.

The names a term offers to the rest of a workflow should be the names inside its value.
Three rules make both hold by construction:

1. **A path is a position inside a value.**
2. **Paths go through every kind:** into a record's children, a law's draws, a function's outputs, and a batch's elements.
3. **Only an array or an opaque value is ever named as a whole.** Anything with parts offers its parts, or is wrapped in a record that holds its name.

The third rule is **grammar B** of jhuggins's [component interface doc](https://claude.ai/artifact/6TusK5RoZokDA1wk5sbdaq), which models specs, declarations, and paths in Rocq.
Names in code font such as `graft_identity_B` are lemmas of that Rocq development.

On top of these rules, the plan gives the declarations public operations:
- an `OutputSpec` becomes a mapping of its components, with `project`, `without`, and `&`;
- an `InputSpec` reads its slots one way from four sources, and merges with `&`;
- every batch stores the declaration of its elements.

## Contents

1. Scope
2. ProbPipe in brief, and the running examples
3. Core invariants
4. Definitions
5. The grammar rule
6. `InputSpec`: slots
7. `OutputSpec`: components
8. Reading a subscript
9. Indexing each kind
10. Indexing with several paths
11. Membership
12. Operations that take paths
13. Summaries
14. Composition with `&`
15. Batches and their element declarations
16. Deliberate exceptions
17. Vocabulary
18. Changes to `design/`, `STYLE_GUIDE.md`, and the CHANGELOG
19. Implementation
20. Alternatives considered
21. Open points
22. Appendix: what moved from each source

## 1. Scope

This plan decides two things:
1. **declarations:** how a term declares what it produces and what it consumes, and the operations on those declarations;
2. **addressing:** how a path or a subscript picks out a part of a term.

**In scope:**
- the definitions of term specs, producers, consumers, slots, components, declarations, paths, and subscripts;
- the grammar rule, which decides which declarations may name a value whole;
- the reading of slots, the size shorthand, and the operations on `InputSpec` and `OutputSpec`;
- what `[]` reads and returns on every kind, lists of paths, positions, and membership (`in`);
- the operations that take paths: `marginal`, `condition_on`, `log_prob`, `without`, `select`, and `rename`;
- what a summary is indexed by;
- composition with `&` for every kind;
- the declaration a batch stores for its elements, and how it enters equality and JAX;
- the vocabulary, the changes to `design/`, and the steps S0–S4 and G1–G5 of [`README.md`](README.md) § Steps.

**Out of scope:**
- how a computation runs, and where: the execution plan, [`execution-layer.md`](execution-layer.md) (#580);
- per-call state, such as `config` blocks and the call token: [`call-context-proposal.md`](call-context-proposal.md);
- the tutorials in `docs/tutorials/`, which are being rewritten separately (README directive 6).

The execution plan relies on this plan in two places.
An embeddable `Function` needs a complete `term_spec` (execution plan §6.2), and claims that name parts of an input use the subscript reader of § 8.
The execution plan uses two words that this plan does not: a **point** is one evaluation of a callable, and a **requester** is the code that asks for a set of points.
In this plan, a **consumer** is a term that declares what it consumes (§ 4.3).

PR #576 (summaries keep the law's paths) has merged into `dev/overhaul`.
PR #575 (`raw` as a property) removes `raw(path)`, so `raw` is not covered here.

Section numbers such as II.6 refer to the design reference in `design/`.

## 2. ProbPipe in brief, and the running examples

ProbPipe builds statistical workflows from laws (probability distributions), functions, the values they produce, and operations on all of them.
These are the terms the plan relies on before § 4 defines its own:

| Term | Meaning | Example |
| --- | --- | --- |
| **term** | Any object ProbPipe tracks: a value, a law, a kernel, a function, or a batch | |
| **law** | A probability distribution, the class `Distribution` | `Normal("x", 0.0, 1.0)` |
| **event** | What one draw of a law is: an array, or a record of named parts | the event of `Normal("x", 0.0, 1.0)` is a scalar |
| **draw** | One value sampled from a law | `sample(d)` |
| **record** | A tree of named values, the class `Record`; a nested value is named with `/` | `Record({"a": 1.0, "g": {"c": 2.0}})["g/c"]` |
| **opaque value** | A value that is neither numeric nor a record, such as a string or an arbitrary Python object | `"forest"` |
| **batch** | Many values of one kind, stacked along named axes called **levels** | `sample(d, sample_shape=100)` has the level `sample` |
| **kernel** | A conditional distribution, the class `ConditionalDistribution`: a law for each value of its inputs | `lik(y \| mu)` |
| **function** | A tracked callable, the class `Function`, with named inputs and a declared output | `Function(run_inference, output_spec=...)` |
| **joint law** | The law of several parts together; `&` composes laws and kernels into one | `lik & prior` |
| **view** | `d[...]`: a law over part of `d`'s event that keeps a reference to `d`; its class is `MarginalView` (today `FieldView`) | `prior["mu"]` |
| **marginal** | `marginal(d, ...)`: the same law as the view, detached from `d` | `marginal(prior, "mu")` |
| **conditioning** | `condition_on(d, values)`: the law of the rest of `d` given values for some of its parts | `condition_on(model, {"y": data})` |
| **summary** | A mean, variance, standard deviation, or quantile of a law | `mean(d)` |
| **route** | How an operation computes its result: by an exact method, or approximately by Monte Carlo from draws | an exact mean of a Normal |
| **label** | A display name for a term; no operation reads it | `mu ~ prior` |
| **JAX pytree** | How JAX sees a value: the arrays it transforms, plus static data it compares to decide whether two values have the same structure | |
| **factor** | One of the laws or kernels a joint law is composed from | `lik` and `prior` are the factors of `model` |
| **atoms** | The stored values an empirical law draws from | `EmpiricalDistribution(atoms)` |
| **fingerprint** | A content hash of a term, which caching and replay use to recognize the same computation | computed automatically |

The examples are written in the proposed API, so they compose with `&` and rename with `rename`.
A "Today" example uses today's API, `*` and `with_path_names`, and shows what `a703f15e` returns.

These terms appear throughout:

```python
from probpipe import (Normal, sample, mean, marginal, condition_on, log_prob,
                      conditional_distribution, Function, OutputSpec, InputSpec,
                      RecordSpec, NumericArraySpec)

# A law over two scalars. Its draws are records {mu, tau}.
prior = Normal("mu", 0.0, 5.0) & Normal("tau", 1.0, 1.0)

# A law with a nested event. Its draws are records {a, g: {c, e}}.
grouped = (Normal("a", 0.0, 1.0) & Normal("c", 0.0, 1.0) & Normal("e", 0.0, 1.0)) \
    .rename({"c": "g/c", "e": "g/e"})

# A hierarchical law. Its draws are records {s1: {theta}, s2: {theta}, mu}.
hier = (Normal("t1", 0.0, 1.0) & Normal("t2", 0.0, 1.0) & Normal("mu", 0.0, 1.0)) \
    .rename({"t1": "s1/theta", "t2": "s2/theta"})

# A kernel lik(y | mu), and the joint law model, whose draws are {y, mu, tau}.
lik = conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec={"mu": ()}, label="lik")
model = lik & prior
post = condition_on(model, {"y": data})    # a posterior whose draws are {mu, tau}
post_spec = post.spec                      # its DistributionSpec
```

A few sections also use terms whose construction does not matter to the example:

- `data`: an observed value of `y`.
- `emulator`: a function from `theta` to a record `{mean, var}`, returned by a function `fit_emulator`; `theta` is a value of its input.
- `lik_g`: a kernel `lik_g(y | g)` whose slot `g` takes a record `{c, e}`.
- `rm`: a random measure, a law whose draws hold a law under the name `G`.
- `gp`: a Gaussian process, a law whose draws hold a function under the name `f`.
- `run_inference`: a plain Python function that returns a posterior law like `post`.
- `sample_params`: a plain Python function of a random key that returns a dict `{"beta": ..., "sigma": ...}`.
- `bern`: a Bernoulli law, whose event is an `int32`.

"The tour" refers to the walkthrough notebook `review/overhaul_tour.ipynb`.

## 3. Core invariants

The design must guarantee thirteen properties.
In the table:
- `d` is a law, `t` is any term, and `f` is a function;
- `s` is a subscript, which is a path or a list of paths;
- `p` and `q` are paths;
- `ps` is a list of paths that is not empty, has no path inside another, and leaves something out.

§ 16 lists the cases where an invariant does not hold on purpose.

jhuggins's Rocq model proves I1 for laws, draws, and batches, and I2, I4, I5, and I10 for every kind of term.
It also proves the fact behind I12, that composition keeps every path of an operand except the root (`graft_identity_B`).
Some of its results about values hold in a model where laws are point masses.
I9 and I12 are rules this plan chooses, and the others follow from its definitions.

| # | Invariant | In code | Delivered by |
| --- | --- | --- | --- |
| I1 | **One set of paths.** A subscript valid on a law is valid on every value derived from it, and on its declaration. | `s` works on `d`, `sample(d)`, each element of `sample(d, sample_shape=n)`, `mean(d)`, and `d.event_spec.project(s)` | S1 for records and batches, S2 for declarations, G3 for every kind |
| I2 | **Indexing commutes with sampling.** A draw of a view is a draw of its parent, indexed. | `sample(d[s])` is distributed as `sample(d)[s]` | S1 for lists of paths, G1 for groups |
| I3 | **Indexing commutes with summaries.** | `mean(d)[s]` and `mean(d[s])` estimate the same quantity, and are equal when both take exact routes; likewise `variance`, `standard_deviation`, and `quantile` | #576 with S1 |
| I4 | **Positions and paths commute on a batch.** | `draws[i][s] == draws[s][i]` | S1 |
| I5 | **Paths chain**, across kinds. | `t[p][q] == t[p + "/" + q]`, including through a field that holds a law, a function, or a batch | S1 for records and batches, G1 for laws, G3 across kinds |
| I6 | **Declarations agree.** | `d[s].event_spec == d.event_spec.project(s) == marginal(d, s).event_spec` | S2, and G1 for groups |
| I7 | **A view and a marginal are one law.** | `marginal(d, s)` equals `d[s]`; one is detached, the other refers to `d` | S1 for lists of paths, G1 for groups |
| I8 | **Draws of a list of paths condition back.** | `condition_on(d, sample(d[ps]))` fixes the nodes in `ps` and nothing else | G4 |
| I9 | **The form of the subscript decides the result.** | `t["g"]` is the node; `t[["g"]]` is a record that holds it, as pandas `df["a"]` and `df[["a"]]` are | S1 |
| I10 | **A list of paths and `without` partition the leaf paths.** | Every leaf path of `t` is a leaf path of one and only one of `t[ps]` and `t.without(*ps)` | S1 for records, record specs, and batches; S2 for declarations; G4 for laws |
| I11 | **Conditioning removes what it fixes.** For a law, the fixed paths leave its event. For a kernel, bound slots, and the bound parts of a record slot, leave its `given_spec`. | `condition_on(d, given).event_spec == d.without(*fixed).event_spec`, where `fixed` are the event paths that `given` names, whichever route computes the result | G4 |
| I12 | **A component is the term at its address.** | `t.component(c)` is `t[address(c)]`, and `t` itself when the address is the root: `grouped.component("g") == grouped["g"]`, and `sample(Normal("x", 0.0, 1.0), sample_shape=4).component("x")` is the batch | G3 |
| I13 | **Indexing commutes with producing**, for every producer. | `f[s](x) == f(x)[s]` for a function, and `lik[s](v)` is `lik(v)[s]` for a kernel; I2 and I4 are the same rule for laws and batches | G3 for functions, G5 for kernels |

**What "equal" means.**
For values, `==` above means equal contents: the same spec and equal entries, compared with `jnp.array_equal` for arrays and field by field for records and batches.
For laws and functions, it means the same term: equal declarations and the same distribution or the same map.
Tests compare declarations with `==`, and distributions through exact summaries where they exist.

With the running examples, all of the following hold once every step has merged:

```python
draw = sample(grouped)

grouped["g"]["c"]                         # I5: the same law as grouped["g/c"]
sample(grouped["g"])                      # I2: a record {c, e}, like draw["g"]
draw["g"]["c"] == draw["g/c"]             # I5 on a value
mean(grouped)["g/c"]                      # I1, I3
grouped.event_spec.project(["a", "g/c"]) == grouped["a", "g/c"].event_spec   # I6
grouped.component("g") == grouped["g"]    # I12: the component g is at the path g

draws = sample(grouped, sample_shape=100)
draws[3]["g/c"] == draws["g/c"][3]       # I4

emulator["mean"](theta) == emulator(theta)["mean"]   # I13
```

## 4. Definitions

Two distinctions carry the plan:
1. A term spec is either a **base spec**, with no parts, or a **spec with parts**.
2. A **path** is a position inside a value, while a **component** is a name that a declaration offers to the rest of a workflow.

Grammar B makes the components and the paths coincide.
Each definition below says what changes from today.
The words *child* and *canonical order* keep their II.6 meanings: a child is an entry directly under a node, and the canonical order visits children depth first, in insertion order.

### 4.1 Term specs: base specs and specs with parts

A **term spec** is the type of one term: an instance of a `TermSpec` subclass, such as `NumericArraySpec`, `RecordSpec`, or `DistributionSpec` (II.1).

A **base spec** types a value with no named parts: a `NumericArraySpec` (an array) or an `OpaqueSpec` (an opaque value).
A value of a base spec is a **base term**.

A **spec with parts** types a term whose parts can be named.
There are five kinds, and every one except a record holds the declaration of what its term produces:

| Kind | Spec | Its parts are | Example |
| --- | --- | --- | --- |
| record | `RecordSpec` with at least one child | its children | a draw of `grouped`, `{a, g: {c, e}}` |
| law | `DistributionSpec`, holding the declaration of its event | the parts of one draw | `grouped` itself |
| kernel | `ConditionalDistributionSpec`, holding its slots and its event's declaration | the parts of the law it yields at each given value | `lik` |
| function | `FunctionSpec`, holding its slots and its output's declaration | the parts of what it returns | `emulator` |
| batch | `BatchSpec`, holding its element's declaration and its levels | the parts of one element | `sample(grouped, sample_shape=100)` |

A term whose spec has parts is a **term with parts**.
A law is always a term with parts, even `Normal("x", 0.0, 1.0)`, whose event is a single array.
That law has no paths below its root, and its one component `x` is at the root.

**Change from today:** none in the specs themselves. Today a `BatchSpec` holds a bare element spec, and § 15 makes it hold a declaration.

### 4.2 Trees and paths

| Term | Meaning | In a draw of `grouped`, `{a, g: {c, e}}` | Change from today |
| --- | --- | --- | --- |
| **name** | A non-empty string without `/`. Components, fields, groups, slots, and levels all have names. | `a`, `g`, `c`, `e` | none |
| **path** | The `/`-joined names from the top of a term down to a node. It is always a `str`. | `"a"`, `"g"`, `"g/c"` | a tuple is no longer a path |
| **node** | A position that a path names. The empty path, which names the term itself, is the **root**. | `a`, `g`, `g/c`, `g/e` | new glossary entry |
| **field** | A node of a record, named through records only, that is not itself a record. A field may hold a law, a function, or a batch, whose own paths continue below it. `keys()` lists a record's fields. | `a`, `g/c`, `g/e` | none |
| **group** | A node of a record that is itself a record. | `g` | one word replaces "interior node", "subtree", "field group", and "group of fields" |
| **leaf path** | A path with no path below it. | `a`, `g/c`, `g/e` | new glossary entry |

**Paths go through every kind.**
A path continues past a field that holds a term with parts, into that term's own paths:
- a law's paths are the paths of its draws;
- a kernel's are those of the law it yields;
- a function's are those of what it returns;
- a batch's are those of one element.

So the record `r = Record({"n": 3.0, "G": post})`, whose field `G` holds the posterior over `{mu, tau}`, has the paths `n`, `G`, `G/mu`, and `G/tau`.
`r["G/mu"]` is `r["G"]["mu"]`.
The fields of `r`, which `keys()` lists, are still only its own: `n` and `G`.
The leaf paths of `r` are `n`, `G/mu`, and `G/tau`.

**Change from today:** paths stop at a field, and a law, function, or batch held in a field is indexed only by a second `[]`.

### 4.3 Producers, consumers, and slots

A **producer** is a term that declares what it produces:
1. **a `Function`** declares its return value as `output_spec`;
2. **a `Distribution` or a `ConditionalDistribution`** declares one draw as `event_spec`.

An operation is a `Function`, so it is a producer too.

A **consumer** is a term that declares what it consumes:
1. **a `Function`** declares its parameters as `input_spec`;
2. **a `ConditionalDistribution`** declares what it conditions on as `given_spec`.

A term may be both.
A `Distribution` is a producer that consumes nothing.

A **slot** is a named place where a consumer receives one value:
- its name is a Python identifier that is not a keyword;
- its term spec types the values it admits;
- it is **required** or **optional**, and a binding may omit an optional slot, which then takes the consumer's default.

A `Function`'s slots are its parameters, and a `ConditionalDistribution`'s slots are its **given slots**.
A slot that holds a record is one slot, so one value binds to the whole slot.

An **`InputSpec`** declares a consumer's slots.
It is an ordered mapping from slot names to term specs, together with the set of optional slots.
Two `InputSpec`s are equal when they have the same slots, the same specs, and the same optional slots, in any order.

**Change from today:** "consumer" replaces the design's "map-like kind", and "slot" replaces "input slot". `design/glossary.md` gains entries for slot, producer, and consumer.

### 4.4 Declarations and components

A **declaration** is either of the two declarations a term carries:
- an **output declaration** is an `OutputSpec`: the spec of what a producer returns, together with the names it offers it under, as a law's `event_spec` or a function's `output_spec`;
- an **input declaration** is an `InputSpec`: a consumer's slots, as a kernel's `given_spec` or a function's `input_spec`.

The terms below concern output declarations.

| Term | Meaning | Example | Change from today |
| --- | --- | --- | --- |
| **whole declaration** | Offers its whole value under one name, written with a keyword. Under grammar B it may name only a base spec. | `OutputSpec(x=NumericArraySpec(()))` is the event of `Normal("x", 0.0, 1.0)` | today any spec may be named whole |
| **exposed declaration** | Offers the parts of a spec with parts, written positionally. An exposed law, kernel, function, or batch offers the components of the declaration it holds. | `OutputSpec(RecordSpec(a=..., g=...))` is the event of `grouped` | functions, and batches of arrays, cannot be exposed today |
| **component** | A name that a declaration offers, together with a term spec, or `None` while a type hole is pending. A whole declaration has one component, its name. An exposed record has one per child, which may be a group. An exposed law, kernel, function, or batch has the components of the declaration it holds. | `grouped`'s components are `a` and `g` | today II.2 and the glossary say "an exposed record's components are its fields", which is wrong for `g` |
| **address** | Where a component is in the value. It is the root for a whole declaration, and the child's path for an exposed record. For an exposed law, kernel, function, or batch, it is the address the component has in the declaration it holds. | `g` is at `"g"`; in `sample(Normal("x", 0.0, 1.0), sample_shape=4)`, whose element is whole `x`, the component `x` is at the root | new term |
| **component spec** | The spec of the declared value at the component's address. When the declared value is a law, as for a function that returns one, the value at an address is the view there, so the component's spec is a law's spec. A law's own event declaration is unaffected, since its declared value is a draw. | the component `mu` of a function returning `post` has the spec of `post["mu"]` | today a law's exposed components are typed by one draw's value, and #571 said the same |
| **wrap** | The one-field exposed record `OutputSpec(RecordSpec(name=spec))`. Its value is the record `{name: value}`, so the name is stored in the value. Under grammar B this is how a structured output gets one name. | `OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=())))` | new term |
| **type hole** | A component whose term spec is pending, written `None`, which a producer fills with `with_term_spec`. Only the keyword form admits one. | `OutputSpec(mean=None)` | `with_spec` becomes `with_term_spec` |

A component name is a name, so it may contain any character but `/`.
A component meets a slot only when its name is also a valid slot name.

Under grammar B, a component's address is the path of its own name, except where the name belongs to a whole base value, whose address is the root.
Composition places every path of an operand at the same path in the joint, apart from the root (`graft_identity_B`).
Today a whole declaration over structure breaks both facts: its name is outside its value, and § 5 shows the problems this causes.

A term's **term spec** in a declaration is the type of the produced term: `prior.event_spec.term_spec` is the type of one draw.
A tracked term's own `spec` is unchanged, so `prior.spec` is still the law's `DistributionSpec`.

**Composition matches by name.**
Composition meets a slot with the component of the same name, and the component's spec must unify with the slot's spec (IV.2).
The two roles differ in one rule:
1. **A component is produced once.** Two operands that produce one name raise `ValueError`.
2. **A slot may be shared.** Same-named slots of several operands become one slot, which carries the unification of their specs.

### 4.5 Subscripts

| Term | Meaning | Example | Change from today |
| --- | --- | --- | --- |
| **list of paths** | Several paths in a Python list or a tuple, to pick several nodes at once. | `["a", "g/c"]` | today a tuple of paths is accepted on laws only and is one nested path elsewhere; a Python list is refused everywhere |
| **position** | An integer, a slice, or a tuple of them, indexing a batch's axes. | `draws[3]`, `draws[0:10]` | II.5's prose also says "selection", which goes |
| **subscript** | What is written inside `[]`, Python's own term: a path, a list of paths, or, on a batch, a position. | `grouped["g/c"]`, `grouped["a", "g/c"]` | replaces "key" |

The public API gets no new noun for a list of paths: docstrings say "a path, or a list of paths".
The word "selection" is avoided because `select` is already a method with another meaning; the glossary keeps only *selection order* (II.7).
The word **key** is not used for subscripts, since ProbPipe already uses it for random keys.
It keeps its Python sense of a mapping's keys, as in `keys()` or the keys of a rename mapping.
Today the glossary also uses it for a field's path.

## 5. The grammar rule

### 5.1 The problem today

Today a law and its own values often disagree on what a subscript means, and a term can offer names that its value does not hold.
Every result below was run on `dev/overhaul` at `a703f15e`, where `*` composes laws:

| Expression | Result today | The related expression | Result today |
| --- | --- | --- | --- |
| `prior["mu", "tau"]` | the joint view of `mu` and `tau` | `sample(prior)["mu", "tau"]` | `KeyError`: read as the one path `mu/tau` |
| `grouped["g"]` | the view of the group `g` | `sample(grouped)["g"]` | `KeyError`: "'g' is a subtree, not a field; use at\_path()" |
| `sample(grouped, sample_shape=4)["g"]` | the batch of the group's values | `sample(grouped)["g"]` | `KeyError`, as above |
| `grouped["g"]["g/c"]` | the view of `c` | `grouped["g"]["c"]` | `KeyError`: "'c' is not an event path" |
| `sample(grouped["g"])["c"]` | an array | `sample(grouped["g"])["g/c"]` | `KeyError` |
| `mean(grouped)["g/c"]` | the mean of `c`, since #576 | `mean(grouped)["g"]` | `KeyError`: a record's `[]` stops at groups |
| `Normal("x", 0.0, 1.0)["x"]` | the law itself | `sample(Normal("x", 0.0, 1.0))["x"]` | `TypeError`: JAX does not support string indexing |

Four causes account for almost every row:

1. **A tuple means two things.** On a law, `["mu", "tau"]` picks two paths. On a record or a batch, the same tuple is one nested path, `mu/tau`.
2. **`Record[...]` stops at fields.** A law and a batch return a group for a path such as `"g"`, while a record raises.
3. **A name can be outside the value it names.** The view `grouped["g"]` is declared as a whole term named `g`, so it is indexed by `g/c`, while its draws are the record `{c, e}`, indexed by `c`. The same holds for any whole declaration over structure, such as a law declared `OutputSpec(params=RecordSpec(...))`: the law is indexed by `params/beta`, and its draws by `beta`.
4. **Summaries renamed their fields** to `mean(mu)` and so on. PR #576 fixed this.

Grammar B removes cause 3 at its source: a name may be outside its value only when the value is a base term, which has no paths to disagree with.
§ 8 removes causes 1 and 2.

### 5.2 The rule

**A whole declaration may name only a base spec.**
Anything with parts is offered by its parts, or wrapped in a one-field record that stores its name in the value.

```python
OutputSpec(x=NumericArraySpec(()))                     # whole, over a base spec: allowed
OutputSpec(RecordSpec(beta=(), sigma=()))              # exposed: components beta and sigma; a value is {beta, sigma}
OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=())))
                                                       # wrapped: one component params; a value is {params: {beta, sigma}}
OutputSpec(params=RecordSpec(beta=(), sigma=()))       # TypeError: params would name a record. Expose its fields,
                                                       # or wrap them under a field named params
```

Today the last line is allowed: `OutputSpec(params=RecordSpec(beta=(), sigma=()))` builds a whole declaration over a record.
The refusal comes with step G2, after the library has stopped writing that form (§ 5.8).

**Why.**
When a name is outside a structured value, the term has two sets of paths.
Its value is indexed from inside, by `beta`, while composition and renames see it from outside, by `params/beta`.
Grammar B keeps every name inside the value, or names a value that has no paths at all, so there is one set of paths:
- subscripts, rename keys, and draws agree (I12);
- no rename key can be ambiguous;
- the spec alone fixes whether a declaration is whole or exposed, so a declaration needs no flag to record it.

jhuggins's model proves each of these (`graft_identity_B`, `no_ambiguous_key_B`, `B_decl_determined`).

**Grammar B, not B′ (decided 2026-10-17).**
jhuggins's doc compares B with a close alternative, B′, which would also name whole any spec with no paths below its root: a batch of arrays, a law over a single array, or a random function whose output is one array.
B′ keeps a few more names, such as a Gaussian process's own name `f`.
It pays with a flag on every declaration, since the spec would no longer fix whether a declaration is whole or exposed.
ProbPipe takes B: a declaration is whole if and only if its spec is a base spec, and no flag records it.

**Views follow the same rule.**
`d[p]` names a base node whole, by the last segment of `p`, and exposes a node with parts.
`d[[p]]` is the wrap, which keeps the name (§ 9.5).

### 5.3 Structured outputs

The rule changes how a few kinds of output are written.
Each has two forms: expose the parts, or wrap the value to keep one name.

**A law over a record, named as a whole.**
Today `params_prior` can be declared `OutputSpec(params=RecordSpec(beta=(), sigma=()))`, and its draws are `{beta, sigma}`.
Under grammar B:

```python
# exposed: components beta and sigma; draws {beta, sigma}
params_prior = distribution(sample=sample_params,
                            event_spec=OutputSpec(RecordSpec(beta=(), sigma=())))

# wrapped: one component params; draws {params: {beta, sigma}}
def sample_params_wrapped(key):
    return {"params": sample_params(key)}
params_prior = distribution(sample=sample_params_wrapped,
                            event_spec=OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=()))))
params_prior["params/beta"]          # the view of beta; the same path on its draws
```

A kernel with the slot `params` meets the wrapped law, since that law offers `params`.
It does not meet the exposed one, which offers `beta` and `sigma`.

**A function that returns a law under one name.**
Today `Function(run_inference, output_spec=OutputSpec(posterior=post_spec))` returns the law under the component `posterior`.
Under grammar B a law has parts, so the declaration exposes it or wraps it:

```python
# exposed: the function returns the law, and offers its event's components
fit = Function(run_inference, output_spec=OutputSpec(post_spec))
fitted = fit(data)                  # the law; its components mu and tau are the views fitted["mu"] and fitted["tau"]
list(fit.output_spec)               # ["mu", "tau"]
fit.output_spec["mu"]               # the spec of post["mu"], a DistributionSpec

# wrapped: the function returns a record that holds the law
def run_inference_wrapped(data):
    return {"posterior": run_inference(data)}
fit = Function(run_inference_wrapped, output_spec=OutputSpec(RecordSpec(posterior=post_spec)))
fit(data)["posterior"]              # the law; a consumer with the slot posterior receives it
```

**The body builds the record (decided 2026-10-17).**
A function with a wrapped output returns the record itself, and the declaration checks it.
ProbPipe never wraps a returned value.
So `run_inference_wrapped` returns `{"posterior": law}`, and a body that returns the law alone fails the declaration's check.

**Exposing a law offers marginals.**
The exposed form offers `mu` and `tau` separately, each typed by its view's spec.
A consumer that meets them by name receives two marginal laws and loses the dependence between them.
To pass the joint posterior on as one thing, wrap it.

**A function that returns a function.**
`fit_emulator` returns `emulator`, which is a function from `theta` to `{mean, var}`.
Its declaration exposes the returned function's outputs, so its components are `mean` and `var`:

```python
fit_emulator = Function(train_gp, output_spec=OutputSpec(FunctionSpec(
    InputSpec(theta=theta_spec), OutputSpec(RecordSpec(mean=("n_days",), var=("n_days",))))))
list(fit_emulator.output_spec)       # ["mean", "var"]
emulator = fit_emulator(design=design, runs=runs)
emulator["mean"]                     # the Function theta -> emulator(theta)["mean"]   (step G3)
```

By I12, `fit_emulator.component("mean")` is `fit_emulator["mean"]`, which is the function that returns `emulator["mean"]`.
In turn, `emulator["mean"]` is the function from `theta` to the mean (I13).
Naming the returned function whole, as `OutputSpec(emulator=FunctionSpec(...))`, raises; wrap it to keep the name.
Today `OutputSpec(FunctionSpec(...))` raises `TypeError`, since a `FunctionSpec` cannot be exposed.

A consumer that meets a function's component with a slot that expects a value needs a rule that lifts by postcomposition, as jhuggins's reply notes.
Until that rule exists, a function's components meet only slots that declare functions.
The rule belongs with `&` on functions, after the workshop (§ 14.4).

**A law whose draws are laws, functions, or batches.**
The law's event rule of § 5.4 takes precedence here: a law's event is a whole base term or an exposed record.
So it can never expose a law, a function, or a batch, and under grammar B it cannot name one whole either.
The draws are therefore wrapped:
- a random measure's draws are laws, stored in a record: `{G: law}`;
- a random function's draws, such as a Gaussian process's, are functions, stored in a record: `{f: function}`; the law keeps offering `f`, and each draw gains a level;
- a bootstrap replicate law's draws are batches, stored in a record under the law's component, as `{b: batch}` for `BootstrapReplicateDistribution("b", ...)`;
- an empirical law whose atoms are laws or functions, and a minibatched target, wrap their draws the same way;
- lifting over a law a function that returns a law, a function, or a batch gives a law whose draws are of that kind, so its event wraps them under the function's name.

**A view at or below a field that holds a law or a function wraps (decided 2026-10-17).**
Paths go through every kind, so a single-path view can name a field that holds a law, as `rm["G"]` does, or a node below it, as `rm["G/mu"]` does.
Under the plain view rule such a view would expose what it names, a law or a function, which the law's event rule forbids (jhuggins's `sub_VB_breaks_rule`).
So such a view wraps, keeping the field's name: `rm["G"]` has the event `OutputSpec(RecordSpec(G=...))`, which is the declaration of `rm[["G"]]` too.
§ 21 asks which name a view strictly below the field keeps, such as `rm["G/mu"]`.

**A batch.**
A `BatchSpec` has parts, its element's, so a batch is always declared exposed, through its element's declaration.
This needs every batch to hold an element declaration (§ 15):
- a batch an operation produces stores its producer's declaration;
- a batch built from raw values gets one: a record element is exposed, and a base element is whole under the default name `element`.

So `sample(Normal("x", 0.0, 1.0), sample_shape=4)` offers `x` as today, and `NumericArrayBatch(values, "draw")` offers `element`.
A sweep or a function that returns a batch exposes the element's components, or wraps the batch to name it.
Today `OutputSpec(BatchSpec(NumericArraySpec(()), sample=3))` raises `TypeError`, since only a batch of records or laws can be exposed.

### 5.4 The law's event rule

**A law's event is a whole base term or an exposed record.**
This holds for the `event_spec` of every `Distribution` and `ConditionalDistribution`.
Composition stores the value at each component of a draw as a field of the joint's draw, and only these two forms give that value the component's spec.
A law's event declaration that exposes a law, a kernel, a function, or a batch raises `TypeError` at construction.

The rule removes a failure that exists today:

```python
# Today: a law whose draws are laws, declared to offer the parts of each draw
inner = Normal("mu", 0.0, 1.0)
laws_of_laws = distribution(sample=lambda key: Normal("mu", jax.random.normal(key), 1.0),
                            event_spec=OutputSpec(inner.spec))
joint = laws_of_laws * Normal("x", 0.0, 1.0)
joint.event_spec        # declares the field mu as a NumericArraySpec
sample(joint)           # ResultSchemaError: ... output/mu does not conform to NumericArraySpec(...): got Normal

# Proposed: the construction of laws_of_laws raises TypeError; wrap the draws instead
laws_of_laws = distribution(sample=lambda key: {"G": Normal("mu", jax.random.normal(key), 1.0)},
                            event_spec=OutputSpec(RecordSpec(G=inner.spec)))
```

The exposed forms of a law, a kernel, a function, and a batch declare what a `Function` returns (VI.0).
`RandomMeasure` declares its draws whole today, so the rule does not break it; grammar B moves it to the wrap in step G2.

**Change from today:** construction refuses an event declaration that exposes a law or a batch. No test under `tests/distributions/`, `tests/operations/`, `tests/families/`, or `tests/functions/` builds such a law, as #571 found.

### 5.5 Defaults by kind, and type holes

Two places build a declaration without a caller writing one: a producer's default, and a type hole filled later.

**Defaults.**
Today a producer whose output is not a record is declared whole under a default name: the function's name, or the law's one component.
`OutputSpec.default` builds these defaults, at 7 call sites in 6 modules.
Under grammar B the default depends on what is produced:

| What is produced | Default today | Default under grammar B |
| --- | --- | --- |
| a base term | whole under the default name | unchanged |
| a record | exposed | unchanged |
| a batch returned by a function or a sweep | whole under the default name | exposed through its element's declaration, with the producer's default name replacing a base element's whole name, so a function `f` that returns a batch of arrays still offers `f` |
| a law or a function returned by a function | whole under the function's name | exposed, which keeps the value and offers its components; the marginals caveat of § 5.3 applies |
| a law, a function, or a batch that is a law's event, as in lifting | whole under the default name | wrapped under the default name, as the law's event rule requires, so each draw gains a level |

A producer's declared or default name wins over a name stored in a batch's element.
A batch with no declaring producer keeps its stored element declaration.

```python
# Today
mk = Function(lambda x: Normal("y", x, 1.0), label="mk")
mk(Normal("x", 0.0, 1.0)).event_spec      # OutputSpec(mk=DistributionSpec(...)): whole over a law
OutputSpec.default(BatchSpec(NumericArraySpec(()), sample=3), component="f")
                                           # OutputSpec(f=BatchSpec(...)): whole over a batch

# Proposed
mk(Normal("x", 0.0, 1.0)).event_spec      # OutputSpec(RecordSpec(mk=DistributionSpec(...))): draws {mk: law}
OutputSpec.default(BatchSpec(NumericArraySpec(()), sample=3), component="f")
                                           # the exposed batch whose element is whole f; it offers f
```

**Type holes.**
`OutputSpec(x=None)` names a value whose type is filled in later, with `with_term_spec`:
1. filled with a base spec, it stays whole under `x`;
2. filled with a batch whose element is a base spec, it becomes the exposed batch whose element is whole `x`, so it offers `x`;
3. filled with any other spec with parts, it raises `TypeError`, and the message asks the caller to declare the record (decided 2026-10-17).

Case 3 is the refusal of a whole declaration over a spec with parts, so it comes with step G2 (§ 5.8).
Until then, filling a type hole with a spec with parts keeps today's whole form.
Case 2 renames nothing that the value holds, since a base element carries no name in its value; § 21 asks the maintainer to confirm it against the decision's wording.

### 5.6 Reading a bare `output_spec`

`output_spec=` accepts a bare term spec as well as an `OutputSpec`.
Today's rule (V.6 § The declared output, and III.3) reads a bare `RecordSpec` as exposed, and any other bare spec as whole under the callable's default component.
Under grammar B a bare spec reads by the defaults of § 5.5, with the callable's default component as the default name:

| Bare `output_spec` | Today | Under grammar B |
| --- | --- | --- |
| `NumericArraySpec((40,), "float32")` | whole under the callable's name | unchanged |
| `RecordSpec(a=..., b=...)` | exposed: components `a`, `b` | unchanged |
| `post_spec`, a `DistributionSpec` | whole under the callable's name | exposed: components `mu`, `tau` |
| a `FunctionSpec` | whole under the callable's name | exposed: the returned function's components |
| `BatchSpec(NumericArraySpec(()), sample=3)` | whole under the callable's name | exposed: the element is whole under the callable's name |

```python
# Today
Function(run_inference, output_spec=post_spec).output_spec
# OutputSpec(run_inference=DistributionSpec(event_spec=OutputSpec(NumericRecordSpec(mu=..., tau=...))))

# Proposed
Function(run_inference, output_spec=post_spec).output_spec
# OutputSpec(DistributionSpec(event_spec=OutputSpec(RecordSpec(mu=..., tau=...)))): components mu and tau
```

The execution plan's examples pass base specs, such as `output_spec=NumericArraySpec((40,), "float32")`, so they read as today.
A law's `event_spec=` keeps its own rule (III.7): a bare `RecordSpec` is exposed, and a whole event needs its component.
V.6 and III.3 change with step G1.

### 5.7 Prerequisites

Exposing a function's output, and a batch of arrays through its element, both depend on the rest of this plan:
- today a `FunctionSpec` cannot be exposed at all, and a `BatchSpec` only when its elements are records or laws;
- exposing a function needs the uniform exposure rule of § 7.2;
- exposing a batch of arrays needs every batch to hold an element declaration (§ 15, step S4).

### 5.8 What grammar B costs, before and after the workshop

The addressing plan swept `dev/overhaul` at `638f56bc` for every whole declaration over a spec with parts, by running the full test suite and every notebook with a probe on `OutputSpec`.
The changes fall into two classes.

**Form only, with the same names and values.**
A batch has parts under grammar B, so `sample(Normal("x", 0.0, 1.0), sample_shape=n)` and `quantile(...)`, which today declare a batch of arrays whole, become exposed batches.
The element's declaration offers `x`, so the batch still offers `x`, and its values do not change.
This class covers most of the 367 tests and 39 notebook cells that the sweep counted as batches of arrays.
The bootstrap replicate law's tests and cells, such as User Guide 08 cell 38 and tour cell 191, belong to the visible class below.

**Visible changes.**
These change what a term offers or what its draws look like:

| Change | Where | Step |
| --- | --- | --- |
| A view or marginal of a group offers its parts instead of the group's name | `_view_declaration`, `marginal` at a group; `tests/distributions/test_views.py` (79 tests), the empirical and path-rename marginal tests | G1 |
| A factor that holds a group (a "grouped sub-joint", § 17) is wrapped, so its draws gain the group's level: `factor(grouped, "g")` draws `{g: {c, e}}`. `marginal(grouped, "g")` must then be that factor's view at `g`, not the factor itself | `_factored.py`, `_regrouped` in `_views.py`, `test_regrouping_renames.py` (25 tests), the tour's hierarchical example | G1 |
| A law the library declares whole over a record, such as an approximate posterior's undeclared event, is exposed or wrapped | `inference/_approximate_distribution.py`, `distributions/_empirical.py`; 233 tests construct such a law, including `tests/core/test_specs.py` and `tests/distributions/test_views.py` | G1 |
| A function that returns a law or a function offers the returned term's components | `OutputSpec.default` and its callers, `_complete_output_spec` | G1 |
| Lifting a law-returning function over a law wraps its draws under the function's name | `functions/_broadcast.py` (V.5), including the tour's `predictive` example | G1 |
| A random function's draws are wrapped, `{f: function}` | `_gaussian.py`, `_random_functions.py`; 260 tests name a function whole, most of them in `test_gaussian.py` (109) and `test_event_declarations.py` (104) | G2 |
| A law whose draws are laws stores them in a record: random measures, empirical laws of laws, and minibatched targets | `_random_functions.py`, `_empirical.py`, `inference/_minibatch.py`; 181 tests name a law whole | G2 |
| A bootstrap replicate law, whose event is a batch, wraps its draws | `_resampling.py`; User Guide 08 and the tour | G2 |
| The keyword form refuses a spec with parts, directly or by filling a type hole | `core/_specs.py` | G2, last |

**Totals, from the sweep at `638f56bc`.**
In all, 1,046 tests in 72 files execute a whole declaration over a spec with parts, and 679 tests in 46 files remain once the form-only batches are set aside.
The counts in the table overlap, since one test can build several kinds.
The library has about 31 sites, most of them through `OutputSpec.default` and the filling of type holes.
No notebook writes such a declaration explicitly: 6 cells produce one through defaults, and 39 more only through the form-only batches.
In `design/`, about 22 passages state or show the whole form over parts: II.2, III.3, III.7, IV.1, V.6, VI.8, VII.2, VII.5, VII.6, and the glossary.

**The split at the workshop (decided 2026-10-17).**
Grammar B for the kinds the workshop uses merges before the workshop, as step G1:
- a view or marginal of a group exposes its parts;
- the library's laws declared whole over a record are exposed or wrapped;
- a factor that holds a group wraps it;
- a function that returns a law or a function exposes its output;
- a lifted law-returning function wraps its draws;
- the defaults by kind, and the reading of a bare `output_spec`.

Random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets follow after the workshop, as step G2.
G2 ends with the refusal of a whole declaration over a spec with parts.
These families declare their draws whole today, so refusing the form before they move would break them, and a special case that exempts them would be the bridging code that README directive 8 forbids.
Until G2, a user can still write the old form, and the library no longer does.

If time runs short, the lifting and default-output parts of G1 are the third cut, after E8 and E7 (README § Steps).

## 6. `InputSpec`: slots

### 6.1 The problem today

A consumer's slots have no single reading, and the operations users need are missing.
Take a law `growth_prior` over `K`, `r`, `phi`, and `sigma`, as in #571's example, and a kernel `shocks_given_sigma` that produces shocks `e` given `sigma`.
A likelihood that conditions on `K`, `r`, `phi`, and `e` declares its given slots today as follows:

```python
shock_spec = RecordSpec(shocks_given_sigma.event_spec.components)
likelihood_given_spec = growth_prior.event_spec.spec.without("sigma").merge(shock_spec)
```

`growth_prior.event_spec.spec` is a record because the law offers the record's parts.
The kernel names its event `e` whole, so its `.spec` is a `NumericArraySpec`, and its components are copied into a `RecordSpec` first.
The tutorials pick components with dict comprehensions and combine them with `{**inputs, ...}`, as #571 found.

Four defects follow, each run at `a703f15e`:

1. **A record read as slots is read by its leaf paths.** `conditional_distribution` reads `given_spec` with `.items()` (`_conditional.py:1392`), and a `RecordSpec` yields its leaf paths. Once the law is regrouped, the idiom above fails for a function `lik_fn(K, r, phi)` that returns the law of the data:

   ```python
   regrouped = growth_prior.with_path_names({"K": "growth/K", "r": "growth/r"})
   conditional_distribution(lik_fn, given_spec=regrouped.event_spec.spec.without("sigma"))
   # TypeError: given_spec of 'lik_fn': unknown parameters ['growth/K', 'growth/r']; available parameters: ['K', 'r', 'phi']
   ```

2. **`InputSpec` refuses a record.** `InputSpec(RecordSpec(K=(), r=()))` raises `TypeError: InputSpec expects a mapping of slot names to specs, got NumericRecordSpec`, although `conditional_distribution` accepts the same record as `given_spec`.
3. **The size shorthand is accepted in one place.** `RecordSpec(K=())` declares a scalar field, while `InputSpec(K=())`, `given_spec={"K": ()}`, and `OutputSpec(K=())` raise `TypeError`. Test modules define their own scalar spec, such as `SCALAR = NumericArraySpec(())`.
4. **`InputSpec.without` ignores a name that is not a slot.** `InputSpec(a=NumericArraySpec(())).without("zzz")` returns the slots unchanged.

### 6.2 One reading of slots

Every place that takes a consumer's slots reads them through the `InputSpec` constructor, which accepts four sources:

1. **An `InputSpec`**, as it is.
2. **A mapping** from slot names to term specs.
3. **A `RecordSpec`**, read by its children, the rule that the positional form of `OutputSpec` follows.
4. **An `OutputSpec`**, read by its components. A component name that is not a valid slot name raises `ValueError`, and a type hole raises `TypeError`.

Every slot of a mapping, a `RecordSpec`, or an `OutputSpec` source is required.
Three arguments take every source that the constructor takes:
- `conditional_distribution`'s `given_spec`;
- `Function`'s `input_spec`;
- `ConditionalDistributionSpec`'s `given_spec`.

A nested record therefore declares one slot per child, which removes defect 1:

```python
# Proposed
InputSpec(RecordSpec(growth=RecordSpec(K=(), r=()), phi=()))
# InputSpec(growth=RecordSpec(K=..., r=...), phi=...): two slots, growth holding a record
InputSpec(growth_prior.event_spec)                       # the slots K, r, phi, and sigma
conditional_distribution(lik_grouped_fn, given_spec=regrouped.event_spec.without("sigma"))
# builds, for a function lik_grouped_fn(growth, phi): the kernel's slots are growth and phi
```

`conditional_distribution` takes the slots' specs from `given_spec` and keeps its rule for optional slots: the parameters with defaults are the optional slots.
An `InputSpec` source that marks a slot optional whose parameter has no default raises `ValueError`, since the kernel has no default to use.

**Change from today:** `InputSpec` accepts a `RecordSpec` and an `OutputSpec`, and the three arguments read their source through it. Today `_slots_of` and `_complete_given_spec` in `_conditional.py`, and the `input_spec` reading in `values/_function_base.py` (line 1093), each read their argument their own way.

### 6.3 The size shorthand

Wherever a declaration takes a term spec, a tuple of sizes is the `NumericArraySpec` of that shape, as it already is for a `RecordSpec` field:

```python
InputSpec(K=(), e=("n_years",))                   # today: TypeError
conditional_distribution(lik_fn, given_spec={"K": (), "r": (), "phi": (), "e": (n_years,)})
                                                  # today: TypeError, "the given slot 'K' declares no term spec"
OutputSpec(K=())                                  # today: TypeError
```

The shorthand applies to an `InputSpec` slot, to `given_spec`, and to the keyword form of `OutputSpec`.
II.1's rule for a `RecordSpec` field applies unchanged, so the shorthand is a tuple only: `K=3` is refused, while a shape argument would read it as `(3,)`.
A slot declared by the shorthand sets no dtype and no support.
Composition still accepts it, since a spec that sets no dtype or support unifies with any dtype and support.

### 6.4 `&` on slots

`a & b` on two input declarations is the input declaration that requires what both require.
Each operand is read as the constructor reads it, so either may be any of the four sources:
1. The slots appear in the order of their first occurrence.
2. Same-named slots carry the unification of their specs, which admits only the values every operand admits. Specs that do not unify raise `ValueError`.
3. The operands share one dimension scope, so the sizes that unification binds are substituted in every slot (II.1, IV.2).
4. A slot is optional only when every operand that names it holds it optional, which is IV.2's rule for the unmet slots of a joint.

```python
InputSpec(theta=("p",), x=()) & InputSpec(theta=(3,), z=())
# InputSpec(theta=NumericArraySpec((3,)), x=..., z=...): theta unifies, and p binds to 3 in every slot
InputSpec(a=("n",)) & InputSpec(a=(2, 2))         # ValueError: the specs of a do not unify
```

`&` replaces #571's proposed `InputSpec.merge`, which was never built.
The joint's unmet slots in IV.2 follow the same rule, so the two share one implementation.

### 6.5 `without`

`InputSpec.without(*names)` returns the slots other than `names`, each optional slot staying optional.
A name that is not a slot raises `KeyError`, as `RecordSpec.without` does for a path.
A removal that leaves no slot stays valid, since a consumer may take no input.

**Change from today:** today `without` ignores a name that is not a slot. Its callers at `_factored.py:2463`, `_views.py:2954`, and `_conditional.py:1156` pass computed names, and step S2 checks that each passes only slots.

### 6.6 `rename` (after the workshop)

`InputSpec.rename(mapping=None, /, **kwargs)` renames slots, as indexing a transition kernel by time does:

```python
step.given_spec.rename(x_prev="x2")
```

It follows the rename rule of II.6, restricted to names, since slots are flat.
It belongs with spec-level `&` for model builders, after the workshop (§ 14.4, `deferred.md`).
A kernel's own `rename` already renames its given slots before then (IV.1).

### 6.7 The `InputSpec` API

```python
class InputSpec(Mapping[str, TermSpec]):        # slot names to term specs, in order
    def __init__(self, slots: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None, /,
                 **slots_by_name: TermSpec | tuple) -> None: ...
    @property
    def required(self) -> tuple[str, ...]: ...      # unchanged
    @property
    def optional(self) -> frozenset[str]: ...        # unchanged
    def with_optional(self, *names: str) -> InputSpec: ...   # unchanged
    def without(self, *names: str) -> InputSpec: ...         # an unknown name raises KeyError (new)
    def __and__(self, other: InputSpec | Mapping | RecordSpec | OutputSpec) -> InputSpec: ...   # new
    def rename(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> InputSpec: ...
    # new, after the workshop
```

Equality is unchanged: the same slots, specs, and optional slots, in any order, because binding reads slots by name.

## 7. `OutputSpec`: components

### 7.1 The problem today

1. **The interface has three forms with three APIs.** `OutputSpec.components` is a `mappingproxy` with no operations. `InputSpec` is a flat `Mapping` with `without` and `with_optional`. `RecordSpec` has the tree operations, and its mapping interface iterates leaf paths. A user converts between the three by hand, as § 6.1 shows.
2. **No public attribute says whether a declaration names its term whole.** For a law `d` whose event is `mu`, `OutputSpec(mu=d.spec)` and `OutputSpec(d.spec)` have the same component names, the same `exposes_record`, and the same `.spec`. They differ only in the spec of `mu`: the law's `DistributionSpec` in the first, and the event's `NumericArraySpec` in the second. Code outside `_specs.py` reads the private `OutputSpec._component_name` instead, on 29 lines of 10 modules at `a703f15e`, as `functions/_broadcast.py:356` does to detect an exposed law.
3. **The library uses private helpers for the operations users lack.** `_components_record` (`core/_specs.py:626`) is the conversion of § 6.1, with 7 call sites in 6 modules. `_view_declaration` (`distributions/_views.py:159`) computes a view's declaration and has no public counterpart. The joint's event declaration is assembled inline from its factors' components (`distributions/_factored.py:447`).
4. **`event_spec.spec` reads as a spec of a spec.** `OutputSpec.spec` is the type of the produced term, stored as `_term_spec`. `.event_spec.spec` occurs on 89 lines of `probpipe/` and 197 lines of `tests/`.

Grammar B already settles the second defect: a declaration is whole if and only if its term spec is a base spec, so the spec states the form and no flag is needed.

### 7.2 One rule for exposure

A spec exposes parts when it holds a declaration of what its term produces.
`OutputSpec(T)` exposes that declaration's components, and component `c` of a term is the term that yields its output at `c`:

| Kind of `T` | The declaration it holds | Component `c` of a term `t` (§ 9) |
| --- | --- | --- |
| `RecordSpec` | its children | the child `t[c]` |
| `DistributionSpec` | `event_spec` | the view `t[c]`, or `t` itself when the event is whole under `c` |
| `ConditionalDistributionSpec` | `event_spec` | the kernel whose law at each given value is the view at `c` |
| `FunctionSpec` | `output_spec` | the function `x ↦ t(x)[c]` |
| `BatchSpec` | the element declaration | the batch of each element's part at `c`, or `t` itself when the element is whole under `c` |
| `NumericArraySpec`, `OpaqueSpec` | none | named whole only |

The positional form exposes the parts that this table gives for its spec's kind, and the keyword form names a base spec whole.
The exposed forms apply recursively: a function that returns a law offers the law's event components.
Only the keyword form admits a type hole, so an exposed term always has a complete spec.

**Change from today:** a `FunctionSpec` and a `BatchSpec` of arrays become exposable (step G1, which needs S4 for batches). Today `_exposed_kind` (`core/_specs.py:285`) accepts a record, a law or a kernel, and a batch of records or laws.

### 7.3 A mapping of its components

Composition matches slots with components by name, so both declarations map names to term specs.
`OutputSpec` becomes a `Mapping` from component names to component specs, as `InputSpec` is a mapping from slot names to slot specs:

```python
prior.event_spec                              # OutputSpec(RecordSpec(mu=..., tau=...))
list(prior.event_spec)                        # ["mu", "tau"]
prior.event_spec["tau"]                       # the component spec of tau
"tau" in prior.event_spec                     # True: a component name
InputSpec(prior.event_spec)                   # the slots mu and tau
lik.given_spec.keys() & prior.event_spec.keys()   # {"mu"}: the slots prior meets
```

- `decl[c]` is the component spec of `c` (§ 4.4). For an exposed record, `c`'s address is the path `c`, so `decl["g"]` is the spec at the path `g` of the value, a group's `RecordSpec` included. For a whole declaration the address is the root, so `decl["x"]` is the base spec of the whole value, or `None` for a type hole.
- Iteration yields the component names in order. That order is the layout `to_vector` reads.
- `components`, `spec`, and `exposes_record` are removed, with no aliases (decided 2026-10-17). There is no `exposes` property, since the spec decides the form.

Until step G2 refuses the old form, a whole declaration over a spec with parts can still be built, as `RandomMeasure` builds one today.
So until then the declaration keeps its form privately, and code outside `core/_specs.py` reads the form through one private helper there, never through `_component_name`.
After G2 the helper reduces to the rule that a declaration is whole when its term spec is a base spec.

A declaration is the one term indexed by name rather than by path.
`decl.project(s)`, not `decl[s]`, builds the declaration of a part, which answers #571's open question 2.

### 7.4 `term_spec` and `with_term_spec`

`OutputSpec.spec` becomes `term_spec`, which is the glossary's term for the type of the produced term.
`term_spec` is `None` only while a type hole is pending.
`OutputSpec.with_spec` becomes `with_term_spec`, so the attribute and the method that completes it share one name.
Its rule is unchanged apart from the type holes of § 5.5: a type hole is filled, and a declared spec unifies with the produced one.

```python
# Today
prior.event_spec.spec                          # NumericRecordSpec(mu=..., tau=...)
OutputSpec(mean=None).with_spec(NumericArraySpec(()))

# Proposed
prior.event_spec.term_spec                     # RecordSpec(mu=..., tau=...)
OutputSpec(mean=None).with_term_spec(NumericArraySpec(()))
```

### 7.5 `project`

`decl.project(paths)` is the declaration of the view at `paths` (III.7): for every law `d`, `d[s].event_spec == d.event_spec.project(s)` (I6).
Its argument takes the subscripts that `d[...]` takes: a path, or a list of paths.

On an exposed record:
1. **One path to a base node** gives that node whole, under a component named by the path's last segment.
2. **One path to a group** gives the group's parts, exposed.
3. **One path at or below a field that holds a law or a function** gives the wrap that keeps the field's name (§ 5.3).
4. **A list of paths** gives an exposed record that keeps each node at its own path, in the declaration's canonical order, whatever order the paths are given in.

On an exposed law, kernel, function, or batch, the projection applies to the declaration it holds, and the result exposes the same kind of term.
That is the declaration of the law, the function, or the batch at the subscript.

On a whole declaration there is no path below the root, so a path raises `KeyError`.
Until step G3, `project` reads a whole declaration's name as `d[c]` reads it today, giving the declaration itself, so I6 holds at every step; G3 removes that reading from `[]` and `project` together (§ 9.5).

The errors are those of § 8: an empty list, a path given twice, and a path inside another raise `ValueError`, and a path that is not a path of the declaration raises `KeyError`.

```python
decl = grouped.event_spec                      # exposes {a, g: {c, e}}
decl.project("a")                              # OutputSpec(a=...)
decl.project("g/c")                            # OutputSpec(c=...)
decl.project("g")                              # OutputSpec(RecordSpec(c=..., e=...)): exposes c and e
decl.project(["g"])                            # OutputSpec(RecordSpec(g=RecordSpec(c=..., e=...))): exposes g
decl.project(["g/e", "a", "g/c"])              # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=..., e=...)))

hier.event_spec.project(["s1/theta", "s2/theta"])
# OutputSpec(RecordSpec(s1=RecordSpec(theta=...), s2=RecordSpec(theta=...)))

OutputSpec(post_spec).project("mu")            # OutputSpec(DistributionSpec(OutputSpec(mu=...))):
                                               # the declaration of fit["mu"], a function returning post["mu"]
```

`project` is not `Record.select`, which returns a dict for passing fields to a function as keywords (§ 12.5).

**Change from #571:** #571's `project` flattened each node to its last segment, in the order given, and refused two paths with the same last segment. `project("g")` gave the group whole under `g`. Grammar B keeps paths, uses the tree's order, and exposes a group (gaps D1 rows b and c).

### 7.6 `without`

`decl.without(*paths)` removes the nodes at `paths` and keeps every other node at its path.
What remains keeps its form:
1. **An exposed record** stays an exposed record, even when one node is left, so `project` and `without` stay complements (I10).
2. **An exposed law, kernel, function, or batch** removes the nodes from the declaration it holds.
3. **A whole declaration** has no path below its root, so it has nothing to remove: any path raises `KeyError`.

A path that is not a path of the declaration raises `KeyError`, and a removal that leaves no component raises `ValueError`, as `RecordSpec.without` does.
The paths are read by the reader of § 8, so a path given twice and a path inside another raise `ValueError`.

```python
growth_prior.event_spec.without("sigma")       # exposes the record of the other components
grouped.event_spec.without("g/e")              # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=...)))
OutputSpec(x=NumericArraySpec(())).without("x")   # KeyError: a whole declaration has only its root
```

**Change from #571:** #571 let a whole declaration remove nodes inside its term. Under grammar B a whole declaration names a base value, so nothing is below it (gaps D1 row m).

### 7.7 `&`

`a & b` on two output declarations is the exposed record whose children are the components of each operand, in operand order:
1. A whole base term `x` becomes the child `x`. An exposed record contributes its children.
2. A name that both operands produce raises `ValueError`.
3. An operand that exposes a law, a kernel, a function, or a batch raises `TypeError`, since a record child needs the complete spec of the value it holds. Wrap such an output to merge it.
4. A type hole raises `TypeError`, for the same reason.
5. The operands share one dimension scope.

For laws `a` and `b`, `(a & b).event_spec == a.event_spec & b.event_spec`, which is the joint's declaration (IV.2), with symbolic dimensions too.
§ 14 states `&` for every kind.

With `project`, `without`, and `&`, the given slots of § 6.1 are declared with no conversion:

```python
likelihood = conditional_distribution(
    likelihood_fn,
    given_spec=growth_prior.event_spec.without("sigma") & shocks_given_sigma.event_spec,
    label="likelihood",
)
```

The tutorial idioms map as follows:

| Today | Proposed |
| --- | --- |
| `{"sigma": growth_prior.event_spec.components["sigma"]}` | `growth_prior.event_spec.project("sigma")` |
| `{name: growth_prior.event_spec.components[name] for name in ["K", "r", "phi"]}` | `growth_prior.event_spec.project(["K", "r", "phi"])` |
| `{**inputs, "e": shocks_given_sigma.event_spec.components["e"]}` | `inputs & shocks_given_sigma.event_spec`, with `inputs` the projection above |
| `growth_prior.event_spec.spec.without("sigma")` | `growth_prior.event_spec.without("sigma")` |

In the third row, `&` raises `ValueError` when `inputs` already holds `e`, where the dict merge replaces the entry.
The tutorials are out of scope for the push, so they keep their idioms until the tutorial rewrite (README directive 6).

### 7.8 Equality and hashing

Two output declarations are equal when they have the same form, the same component names in order, and equal term specs:
- `OutputSpec(parameters=R)` and `OutputSpec(RecordSpec(parameters=R))` have the same components and produce different terms, so they are unequal. Under grammar B the first exists only for a base `R`.
- Equality is order-sensitive, because component order is the layout that `to_vector` reads. `InputSpec` equality ignores order, because binding reads slots by name.
- An `OutputSpec` equals only another `OutputSpec`, never a plain mapping, since a mapping does not state the form.
- The hash agrees with equality.

### 7.9 The `OutputSpec` API

```python
class OutputSpec(Mapping[str, TermSpec | None]):    # component names to component specs, in order
    @overload
    def __init__(self, term_spec: TermSpec, /) -> None: ...    # exposed: a record, law, kernel, function, or batch
    @overload
    def __init__(self, **term: TermSpec | tuple | None) -> None: ...   # whole: one keyword over a base spec,
                                                                       # a size tuple, or a type hole
    @property
    def term_spec(self) -> TermSpec | None: ...   # the produced term's type; None while a type hole is pending

    @classmethod
    def default(cls, spec: TermSpec, *, component: str) -> OutputSpec: ...   # the defaults of § 5.5
    def with_term_spec(self, spec: TermSpec) -> OutputSpec: ...
    def project(self, paths: str | Sequence[str]) -> OutputSpec: ...
    def without(self, *paths: str) -> OutputSpec: ...
    def rename(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> OutputSpec: ...
    def __and__(self, other: OutputSpec) -> OutputSpec: ...
    # unchanged: is_concrete, with_dim_sizes, with_dim_names
```

| Today | Proposed | Step |
| --- | --- | --- |
| `decl.spec` | `decl.term_spec` | S2 |
| `decl.components` | `decl` itself, a mapping | S2 |
| `decl.components[c]` | `decl[c]` | S2 |
| `decl.exposes_record` | `isinstance(decl.term_spec, RecordSpec)` for a declaration that is not whole, read through the private helper of § 7.3 until G2 | S2 |
| `decl._component_name` outside `_specs.py` | the private helper of § 7.3, which returns a whole declaration's one name or `None`; after G2, `next(iter(decl))` when `decl.term_spec` is a base spec | S2 |
| `decl.with_spec(spec)` | `decl.with_term_spec(spec)` | S2 |
| `_view_declaration(decl, s)` | `decl.project(s)` | S2 |
| `_components_record(decl)` | `RecordSpec(decl)` where a record of the components is needed | S2 |
| `decl.with_path_names(...)` | `decl.rename(...)` | S3 |
| #571's `OutputSpec.merge` (never built) | `a & b` | S2 (§ 21, O1) |

jhuggins's reply sketches the recursion behind `project`, `without`, and path reading as two methods on every term spec, `produced_declaration()` and `with_produced_declaration(declaration)`, with `None` for a base spec.
§ 21 asks whether they are public.

## 8. Reading a subscript

Every term that is indexed by path reads a subscript the same way:
- a `str` is one path;
- a list or tuple of `str` is a list of paths;
- an integer, a slice, or a tuple of them is a position, on a batch only.

A base term has no paths: a `NumericArray` keeps its own `[]`, which indexes its entries, and an opaque value has no `[]`.
One reader in `probpipe/core/_shapes.py`, beside `_as_names`, does all the parsing and owns every error message.
The separator `_PATH_SEP` is defined once, in `named_tree.py`; today eight other modules define their own.

```python
t["g/c"]             # a path: the node g/c
t["a", "g/c"]        # a list of paths (Python builds the tuple ("a", "g/c"))
t[["a", "g/c"]]      # the same, written as a Python list
t[("a",)]            # a list of one path: a record that holds a
draws[3]             # a position (batches only)
draws[0:10]          # positions 0 to 9
chains[1, 0:10]      # positions on two axes, for a batch with levels chain and draw
```

The reader accepts these types and nothing else:
- **A path** is a `str`. A subclass such as `numpy.str_` is converted with `str()`.
- **A position** is an `int`, anything with `__index__` (NumPy and JAX integer scalars included), a `slice`, or a tuple of those.
- `bool`, `None`, and `...` raise `TypeError`. `None` keeps its one meaning, a whole axis in `at_levels`.

It refuses a malformed subscript before anything is looked up:

| Subscript | Error | Reason |
| --- | --- | --- |
| `t[""]`, `t["g//c"]`, `t["/a"]` | `ValueError` | an empty path or an empty segment |
| `t[[]]`, `t[()]` | `ValueError` | no paths at all; on a batch too, so `t[tuple(paths)]` with no paths never returns everything |
| `t["a", "a"]` | `ValueError` | the same path twice |
| `t["g", "g/c"]` | `ValueError` | a path inside another would duplicate a variable, giving a degenerate joint law |
| `t["a", 0]` | `TypeError` | a path and a position in one subscript |
| `draws[[0, 2]]` | `TypeError` | a list of positions; there is no fancy indexing |
| `t[{"a"}]`, `t[iter(["a"])]` | `TypeError` | a set has no order and an iterator can be read only once, as `_as_names` already rules |

Only three places accept a list of paths: `[]`, `marginal`, and `project` on a declaration.
Every other argument takes one path as a `str`, and the same reader checks it, so `without("a", "a")` and `without("g", "g/c")` raise as `[]` does:

```python
rec.at_path("g", "c")         # separate segments, joined with "/" (unchanged)
rec.at_path(("g", "c"))       # TypeError: unpack the tuple, at_path(*segments)   (today: the path g/c)
rec.without("a", "g/e")       # removes several paths, one per argument (unchanged)
rec.without(("a", "g/e"))     # TypeError: pass the paths as separate arguments   (today: KeyError, the path a/g/e)
rec.without("a", "a")         # ValueError                                         (today: removes a)
condition_on(grouped, {"g/c": 0.5})    # a mapping from paths to values
```

A declaration is indexed by component name, not by path (§ 7.3).

**Change from today:** a tuple was one nested path on `Record`, `RecordSpec`, `RecordBatch`, and `Batch`, and a list of paths only on laws. A Python list was refused everywhere: `grouped[["g"]]` and `sample(grouped)[["a"]]` raise `TypeError` today.

## 9. Indexing each kind

Every term with parts resolves a path against its own paths and returns the node in its own kind:
- a record gives a record or a value;
- a law gives a law;
- a function gives a function;
- a batch gives a batch.

Paths name groups everywhere, `Record` included, and continue into a field that holds a law, a function, or a batch.

### 9.1 Records

```python
rec = sample(grouped)             # Record {a, g: {c, e}}

rec["a"]            # NumericArray, labeled "a"                    (unchanged)
rec["g/c"]          # NumericArray, labeled "g/c"                  (unchanged)
rec["g"]            # Record {c, e}, a view labeled "g"            (today: KeyError, "use at_path()")
rec["g"]["c"]       # the same as rec["g/c"]
rec["a", "g/c"]     # Record {a, g: {c}}, labeled "(a, g/c)"      (today: KeyError, the path a/g/c)
rec[["g"]]          # Record {g: {c, e}}                           (today: TypeError)
rec.keys()          # ("a", "g/c", "g/e"): fields only            (unchanged)
```

`rec.at_path("g")` stays as a synonym of `rec["g"]`, and `at_path("g", "c")` still takes separate segments.
Step S1 delivers this section.

### 9.2 A field that holds a law, a function, or a batch

A path continues into the term a field holds:

```python
r = Record({"n": 3.0, "G": post})    # post's draws are {mu, tau}

r["G"]              # post                                         (unchanged)
r["G/mu"]           # the view post["mu"]                          (today: KeyError)
r["G"]["mu"]        # the same view (I5)
```

Step G3 delivers this section.

### 9.3 Record specs and declarations

A `RecordSpec` indexes as a record does, returning specs.
A declaration is indexed by component name, and `project` takes a subscript:

```python
spec = rec.spec                       # RecordSpec {a, g: {c, e}}
spec["g"]                             # RecordSpec {c, e}                 (today: KeyError; step S1)

decl = grouped.event_spec             # exposes {a, g: {c, e}}
decl["g"]                             # RecordSpec {c, e}: the component spec of g   (step S2)
decl.project("g")                     # the declaration of grouped["g"]: exposes c and e   (step S2, the group case with G1)
decl.project(["a", "g/c"])            # exposes {a, g: {c}}   (step S2)
```

### 9.4 Batches

A batch indexes each of its elements by path.
Positions index the batch's own axes, as today.

```python
draws = sample(grouped, sample_shape=100)   # RecordBatch on the level "sample"

draws["a"]           # NumericArrayBatch                   (unchanged)
draws["g"]           # RecordBatch {c, e}                  (unchanged)
draws["a", "g/c"]    # RecordBatch {a, g: {c}}             (today: KeyError, read as the path a/g/c)
draws[3]             # Record {a, g: {c, e}}               (unchanged)
draws[3]["g"] == draws["g"][3]                            # I4 (today: the left side raises)
draws[()]            # ValueError: no paths at all         (today: the whole batch)
```

A batch of draws of `Normal("mu", 0.0, 1.0)` has no paths below its elements, so `draws["mu"]` raises `KeyError`: `mu` names the whole element, not a path inside it.
`draws.component("mu")` reads it (§ 9.7).
Today `draws["mu"]` raises `TypeError`, and #571 had proposed that it return the batch (gaps D1 row e).

A batch of laws indexes each law: `laws["mu"]` is the batch of the views `law["mu"]`.
Today it raises `TypeError` with the false message that the elements have no named fields.
A batch of functions does the same with each function.
These two arrive with step G5; the rest of this section is step S1.

### 9.5 Laws and views

`d[p]` returns a `MarginalView` (today `FieldView`): a law over the node at `p` that refers back to `d`.
Its declaration follows the grammar rule:
1. a base node is named whole by the last segment of `p`;
2. a node with parts is exposed, so the view offers its parts;
3. a node at or below a field that holds a law or a function is wrapped, keeping the field's name (§ 5.3).

To keep a group's name, index with a list of one path:

```python
grouped["a"]            # a whole term named a; its draws are arrays
grouped["g/c"]          # a whole term named c
grouped["g"]            # exposes c and e; its draws are records {c, e}        (today: a whole term g; step G1)
grouped[["g"]]          # exposes the record {g: {c, e}}: one component, g     (today: TypeError; step S1)
grouped["g"]["c"]       # the view of g/c                                      (today: KeyError; step G1)
grouped["a", "g/c"]     # exposes {a, g: {c}}: components a and g              (today: {a, c}; step S1)
marginal(grouped, ["a", "g/c"])   # the same law, detached; a list works       (today: TypeError, unhashable list; step S1)
```

This is I9 read as pandas reads it: `grouped["g"]` is the node, the group's contents, and `grouped[["g"]]` is a record that holds it, under its name.

```python
# Today
grouped["g"].event_spec
# OutputSpec(g=NumericRecordSpec(c=..., e=...)): a whole term g, so the view is indexed by "g/c"
#                                                while its draws are indexed by "c"
# Proposed
grouped["g"].event_spec      # OutputSpec(RecordSpec(c=..., e=...)): the view and its draws are both indexed by "c"
grouped[["g"]].event_spec    # OutputSpec(RecordSpec(g=RecordSpec(c=..., e=...)))
```

**A whole term's name is not a subscript.**
Today `Normal("x", 0.0, 1.0)["x"]` returns the law itself, since III.7 lets a consumer index a law by component whatever its form.
Under grammar B a whole declaration has no path below its root, so `x` is a name and not a path, and `Normal("x", 0.0, 1.0)["x"]` raises `KeyError`.
`Normal("x", 0.0, 1.0).component("x")` returns the law (I12).
Library code that indexes a law by a whole term's name today moves to the component reader in the same step.
This change comes with the public accessor in step G3, so a reader always has one way to read a component.

**`FieldView` becomes `MarginalView` (decided 2026-10-17).**
The class covers groups and lists of paths, not only fields, so "field" undersells it.
By I7 it is the marginal law at its subscript, kept as a view rather than detached, and the name pairs with `marginal`.
The rename changes the class, its repr, III.7, III.8, and the User Guide; `d[s]` is unchanged.
It merges in step S3.

### 9.6 Functions and kernels

A function is indexed by the paths of what it returns, and indexing comes after the call (I13).
A function with a declared output checks the subscript when it is indexed.
A function without one, whose `output_spec` is `None`, checks it when the indexed function is called, against what the call returns.

```python
emulator["mean"]             # the function theta -> emulator(theta)["mean"]   (today: TypeError, no [])
emulator["mean"](theta) == emulator(theta)["mean"]
emulator[["mean"]]           # the function theta -> Record {mean}
```

Functions gain `[]` in step G3.

A kernel is indexed the same way in step G5: `lik[s]` is the kernel that maps a given value `v` to the law `lik(v)[s]`.
Until then, a kernel is the one producer without `[]`, and the design states the exception (§ 16).

### 9.7 Reading a component

`t.component(c)` returns the term at component `c`'s address (I12), and `t` itself when the address is the root.
The name `component` is decided (2026-10-17).

| Term `t` | Its declaration | `t.component(c)` |
| --- | --- | --- |
| a record | exposes its children | `t[c]` |
| a law | its event declaration | the view `t[c]`, or `t` when the event is whole under `c` |
| a kernel | its event declaration | the kernel whose law at each given value is the view at `c`, or `t` when the event is whole |
| a function | its output declaration | the function `x ↦ t(x)[c]`, or `t` when the output is whole |
| a batch | its element declaration | the column `t[c]`, or `t` when the element is whole under `c` |

```python
grouped.component("g")                       # grouped["g"]: exposes c and e
Normal("x", 0.0, 1.0).component("x")         # the law itself
sample(Normal("x", 0.0, 1.0), sample_shape=4).component("x")   # the batch itself
rm.component("G")                            # rm["G"], which wraps: its event is {G: law}
gp.component("f")                            # gp["f"], which wraps: its event is {f: function}
```

The component's spec is the spec of what `component` returns: `decl[c] == t.component(c).spec` for a declaration whose declared value is `t`.
For a law's own event declaration, the declared value is a draw, so `d.event_spec[c]` is the spec of `sample(d).component(c)`.

**Change from #571:** #571 read a component by five cases, one per form, and jhuggins's reply gave each kind a private `_component`. Grammar B states one rule and makes the accessor public (gaps D1 row d). The accessor arrives with step G3.

## 10. Indexing with several paths

A list of paths keeps every chosen node at its own path, so the result is the original tree cut down to those nodes.
The result is always an exposed record (I9), and it is the complement of `without` (I10): every leaf path is in one and only one of the two.

```python
grouped["a", "g/c"]                  # a law over {a, g: {c}}
grouped.without("g/e")                # the same tree, reached by removing instead of choosing   (step G4)
sample(grouped)["a", "g/c"]           # Record {a, g: {c}}, labeled "(a, g/c)"

# I8: the draws condition back onto the law they came from
condition_on(grouped, sample(grouped["a", "g/c"]))   # fixes a and g/c; leaves g/e   (step G4)
```

In a hierarchical model, two groups often hold a parameter of the same name.
Keeping paths handles this:

```python
hier["s1/theta", "s2/theta"]          # a law over {s1: {theta}, s2: {theta}}
# today: ValueError, "more than one path ends in 'theta', which would give the result two fields of the same name"
```

The form of the subscript decides the result (I9).
One path gives the node, and a list of one path gives a record that holds it:

```python
grouped["g/c"]          # a law of c; its draws are arrays
grouped[["g/c"]]        # a law over {g: {c}}; its draws are records
```

The chosen nodes keep the original tree's order, whatever order the paths are given in, as `without` keeps the order of what it leaves.
Nodes under a shared group stay under it:

```python
grouped["g/e", "a", "g/c"]      # a law over {a, g: {c, e}}: the order of grouped
grouped.without("a")            # a law over {g: {c, e}}: the same order
```

A path through a field that holds a law, a function, or a batch cuts that term down too.
With `r = Record({"n": 3.0, "G": post})`, `r["n", "G/mu"]` is the record `{n, G}` whose `G` is `post[["mu"]]` (step G3).

A value indexed by several paths is labeled by its paths, as `(a, g/c)` (decided 2026-10-15).
Today `sample(grouped["a", "g/c"]).label` is `"(a, c) ~ Normal·Normal·Normal"`.

To pass chosen fields to a function as keyword arguments, use `select`, which flattens each path to its last segment (§ 12.5).

**Change from today:** a law indexed by several paths flattens each node to its last segment, so `grouped["a", "g/c"]` is a law over `{a, c}`, and two paths that end in the same name raise. VI.8 lists a marginal's fields "in the order of the paths"; that sentence changes to the tree's order. Step S1 delivers this section, apart from the steps marked.

## 11. Membership

`p in t` is true when `p` is a `str` and `t[p]` would return a node, and false otherwise.
That covers groups, and paths through a held law, function, or batch, on records, record specs, batches, laws, and functions.

```python
rec = sample(grouped)
"g/c" in rec          # True                                   (unchanged)
"g" in rec            # True: rec["g"] works                   (today: False)
("a", "g/c") in rec   # False: a list of paths is not a path   (False today too, as the path a/g/c)
"G/mu" in r           # True, for r = Record({"n": 3.0, "G": post})   (today: False; step G3)

draws = sample(grouped, sample_shape=100)
"a" in draws          # True                                   (today: False, by iterating the draws)

"a" in grouped        # True                                   (today: TypeError, "argument ... is not iterable")
```

Two kinds answer differently, because their `[]` is different:
- **A declaration** is a mapping of its components, so `"g" in grouped.event_spec` tests component names. For an exposed record these are its top-level paths. For a whole declaration they are not: `"x" in Normal("x", 0.0, 1.0).event_spec` is true, while `"x"` is not a path of the law.
- **A base term** keeps its own meaning: `in` on a `NumericArray` tests membership of an entry.

On every batch, `in` tests paths and never elements.
So a batch of arrays, which has no paths, answers `False` for every `str`, and `1.0 in sample(Normal("x", 0.0, 1.0), sample_shape=3)` changes from testing the draws to `False`.

Two consequences can surprise:
- `"x" in sample(Normal("x", 0.0, 1.0), sample_shape=4)` is `False`, although the batch offers the component `x`: `x` is its name, not a path inside it. `draws.component("x")` reads it.
- `"g/c" in grouped` is `True`, while `"g/c" in grouped.event_spec` is `False`: the law answers paths, and its declaration answers component names.

**Change from today:** a `Record` answers `in` for fields only, a batch falls back to iterating its elements, and a law raises. A Python `dict` satisfies both `p in m` if and only if `p in m.keys()`, and `p in m` if and only if `m[p]` succeeds. A `Record` cannot keep both, because `keys()` lists fields only while `[]` also returns groups; the plan keeps the second. Step S1 delivers this section, apart from the paths through held terms (G3).

## 12. Operations that take paths

Every operation reads its paths through the reader of § 8, so a subscript means the same thing wherever it is passed.
A few arguments also accept a name that is not a path:

| Argument | Paths | Names as well |
| --- | --- | --- |
| `t[...]`, `marginal(d, ...)`, `decl.project(...)` | one path, or a list of paths | none |
| `without(*paths)`, `select(*paths)`, `at_path(*segments)`, `is_field(path)` | one path per argument | none |
| the keys of `rename` | paths of the term | a whole term's name, which renames it |
| the keys of `condition_on(d, given)` | paths of `d`'s event | for a kernel, its slot names, each optionally followed by a path inside a record slot |
| composition with `&` | none | meets each component with the slot of the same name |

Under grammar B no rename key can be read both as a name and as a path.
A whole term is a base term, which has no path below its name, and every other component's name is its own path.
jhuggins's model proves this for renames (`no_ambiguous_key_B`).
The keys of `condition_on` mix slot names and event paths.
They stay unambiguous because IV.2 keeps a name from being both an unmet slot and a component, which is a separate rule.

### 12.1 `marginal`

`marginal(d, paths)` takes the subscripts that `d[paths]` takes and returns the same law, detached from `d` (I7).
The parameter is renamed from `field` to `paths`, documented as "a path, or a list of paths", as in `without(*paths)`.

```python
marginal(grouped, "g")              # the law of the group's contents, exposing c and e   (step G1)
marginal(grouped, ["g"])            # the law of the record {g: {c, e}}, exposing g
marginal(grouped, ["a", "g/c"])     # a Python list works  (today: TypeError, unhashable list)
marginal(grouped, ("a", "g/c"))     # the same call, the same result
```

A marginal over the whole events of several factors keeps the factors in the law's order.
Today `marginal(a_law & b_law, ("b", "a"))` lists the factors as `b(b)·a(a)`, in the order of the paths; under this plan it lists them as `a(a)·b(b)`.

**Change from today:** the parameter is named `field`, a list is refused, and several paths flatten to their last segments. A group's marginal is the factor that holds it, which is a whole term (VI.8). Under grammar B that factor wraps, so a group's marginal is the factor's view at the group (step G1).

### 12.2 `condition_on`

A given value is a mapping from paths to values, or a `Record`, read by its paths.
The result is declared as `d.without(*fixed)` is (I11, decided 2026-10-15), so it exposes a record of the remaining nodes whichever route computes it:

```python
condition_on(prior, {"mu": 0.1})    # a law over the record {tau}
                                    # (today: OutputSpec(tau=...), a whole term, from the exact Gaussian route;
                                    #  a Monte Carlo route returns a record instead)
post["mu"]                          # works for every posterior
```

Nested paths, records, and arrays are passed as follows:

```python
condition_on(model, {"y": data})                    # unchanged
condition_on(grouped, {"g/c": 0.5})                 # a nested path
                                                    # (today: ResolutionError, "conditioning on the nested field 'g/c' is not supported yet")
condition_on(grouped, sample(grouped["a", "g/c"]))  # a record read by its paths (I8)
condition_on(grouped, {"g": sample(grouped["g"])})  # one group: its draw is the record {c, e}, so name it
condition_on(grouped, sample(grouped[["g"]]))       # or draw the record {g: {c, e}} instead
condition_on(model, {"mu": sample(prior["mu"])})    # one array: name it too
condition_on(lik, mean(post[["mu"]]))               # a plug-in estimate of mu
```

I8 covers lists of paths only.
The draw of one group, `sample(grouped["g"])`, is the record `{c, e}`.
Passed alone, it would be read at the top level, where `c` and `e` are not paths of `grouped`; today that raises `ResolutionError: condition_on: unknown fields ['c', 'e']`.
The nested-path row states the contract.
Where a route cannot yet condition at a nested path, it raises `ResolutionError` naming the route, rather than refusing the path for every law.
Step G4 delivers this section.

### 12.3 `log_prob`

The value is read by its paths, so a summary scores where a draw scores whenever its spec fits the event's (#576):

```python
log_prob(prior, sample(prior))                # unchanged
log_prob(prior, mean(prior))                  # the score at the mean (works since #576)
log_prob(Normal("x", 0.0, 1.0), 0.3)          # a whole array takes the array    (unchanged)
log_prob(bern, mean(bern))                    # refused: the float32 mean cannot be cast to the int32 event
                                              # (today: ApplicabilityError, as here)
```

No step changes `log_prob`; it follows the reader of § 8.

### 12.4 `without`

`without(*paths)` removes nodes and keeps every other node at its path.
It applies to records, record specs, batches, output declarations, and laws, where it returns a view.
What remains keeps the form it had: removing from an exposed record leaves an exposed record, even when one node is left, so `without` and a list of paths stay complements (I10).

```python
rec.without("g/e")            # Record {a, g: {c}}
post.without("tau")           # a view of post over the record {mu}   (today: AttributeError, laws have no without)
```

`law.without` arrives with step G4; the rest is unchanged apart from the reader's errors (S1).

### 12.5 `select`

`select(*paths, **remap)` exists to pass fields to a function as keyword arguments, so it flattens each path to its last segment.
Today it names each entry by its full path, and `"g/c"` cannot be a keyword.
`RecordBatch.select` changes the same way; each of its entries stays a one-column batch, as today.

```python
rec.select("a", "g/c")        # {"a": ..., "c": ...}           (today: {"a": ..., "g/c": ...})
rec.select(x="g/c")           # {"x": ...}                    (unchanged)
sample(hier).select("s1/theta", "s2/theta")   # ValueError: both flatten to theta; name one with a keyword
                                              # (today: {"s1/theta": ..., "s2/theta": ...})
```

### 12.6 `rename`

`with_path_names` becomes `rename` (decided 2026-10-17), the name pandas (`df.rename`) and xarray (`ds.rename`) use.
It renames and moves nodes by their paths, and renames a whole term's name.
Under grammar B its keys are the term's paths, plus that one name, so a view's subscripts and its rename keys agree:

```python
Normal("x", 0.0, 1.0).rename(x="y")       # Normal(y): renames the whole term   (unchanged)
grouped.rename({"g/c": "g/k"})            # rename in place                     (unchanged)
grouped.rename({"g/c": "c"})              # move c to the top level             (unchanged)
grouped["g"].rename(c="k")                # a view exposing k and e
# today: KeyError, "with_path_names(): 'c' is not a path in this tree; the paths are ['g', 'g/c', 'g/e']"
```

Two cases follow from the rule:
1. **A whole base term's name cannot become a nested path**, since the term has no record to place it in. `Normal("x", 0.0, 1.0).rename(x="g/y")` raises `ValueError`, as it does today; compose it into a joint first.
2. **On a batch**, renaming the element's whole component renames the batch's one component: `draws.rename(x="y")` on draws of `Normal("x", 0.0, 1.0)` offers `y`.

The rename applies to every kind that has `with_path_names` today, in one mechanical PR with `&` and `MarginalView` (step S3): `NamedTree` and its families, `OutputSpec`, `Distribution`, `ConditionalDistribution`, and batches.
§ 21 asks whether `with_level_names` and `with_dim_names` follow.

## 13. Summaries

A summary that has one value per field of the event keeps the law's paths and declaration form, and the operation appears only in its label (decided 2026-10-15; PR #576, merged).
This holds for `mean`, `variance`, `standard_deviation`, and `quantile`, and `expectation(d, f)` takes the paths of `f`'s output.

```python
m = mean(grouped)
m["a"], m["g/c"]             # the means of a and g/c           (works since #576)
m["g"]                       # the record of means {c, e}       (today: KeyError, a record stops at groups; step S1)
mean(Normal("x", 0.0, 1.0))  # an array: the whole term x, like a draw
m.label                      # "𝔼[(a, g) ~ Normal·Normal·Normal]": the label says it is a mean
```

A summary binds wherever a draw of the same type binds, so plug-in estimates work, such as `condition_on(lik, mean(post[["mu"]]))` and `log_prob(prior, mean(prior))`.
The guard is the type, not the name.
A summary's spec is derived from the event's: a mean is floating, and its support is the hull of the event's support.
So a Bernoulli law's mean is refused where its `int32` event is required.
`condition_on(model, standard_deviation(post))` binds whenever the types fit; like passing the wrong array, it is the caller's mistake to make.

Results that are not one value per field are base terms under the operation's name: `cov(d)` is one array, and a score is declared under `log_prob`.

## 14. Composition with `&`

### 14.1 Why `&`

Today `*` composes laws and kernels: `lik * prior` is the joint law.
`*` is also arithmetic on values: it multiplies a `NumericArray`, a `NumericRecord`, or a `LinOp`.
A law's draw is a record, so `d1 * d2` composes two laws while `sample(d1) * sample(d2)` multiplies their draws.
`&` reads as "both", and it replaces `*` for composition (decided 2026-10-17), with no alias for `*`.
`*` keeps its arithmetic meanings: a Gaussian random function scaled by a number, and a `LinOp` scaled by a number.

### 14.2 The rule for every kind

`a & b` holds, produces, and requires what `a` and `b` do:
1. **A name both operands produce raises `ValueError`.**
2. **A name both operands consume is one slot.** Its specs unify, and it is optional only when optional in both.
3. **For laws and kernels, a name the left operand consumes and the right operand produces is matched (IV.2).** Functions pair and never wire, so for them such a name raises.
4. **The operands share one dimension scope.**

For every kind, `(a & b).spec == a.spec & b.spec`.
For two `Distribution`s, `(d1 & d2).event_spec == d1.event_spec & d2.event_spec`, with symbolic dimensions too.
For kernels the given side is `A.given_spec.without(*(A.given_spec.keys() & B.event_spec.keys())) & B.given_spec`.

| Operands | `a & b` | Step |
| --- | --- | --- |
| `Distribution`, `ConditionalDistribution` | the joint law or kernel, as `*` builds today | S3 |
| `Record`, `RecordSpec` | both sets of fields; a field in both raises | S3 |
| `OutputSpec` | both sets of components (§ 7.7) | S2 (§ 21, O1) |
| `InputSpec` | both sets of slots (§ 6.4) | S2 (§ 21, O1) |
| `DistributionSpec`, `ConditionalDistributionSpec` | the joint's spec | after the workshop |
| `Function`, `FunctionSpec` | the function that returns both outputs | after the workshop |
| `Batch`, `BatchSpec` | element by element, over equal levels | after the workshop |
| `NumericArray` | NumPy's elementwise `&`, unchanged | — |

```python
# Today
model = lik * prior
data_model = lik_nee * lik_lai

# Proposed
model = lik & prior                           # the joint law; its draws are {y, mu, tau}
data_model = lik_nee & lik_lai                # two likelihoods that share their parameters
list(data_model.given_spec)                   # ["theta"]: one shared slot
list(data_model.event_spec)                   # ["nee", "lai"]
data_model.given_spec == lik_nee.given_spec & lik_lai.given_spec
```

The joint's label and notation keep joining their operands with `·`, as `lik·prior` and `lik(y | mu)·prior(mu)`, since `·` there is the product of densities (§ 21, O8).

### 14.3 What composition does to paths

**A joint's paths are its operands' paths, side by side.**
The joint exposes a record whose children are the operands' components.
An exposed operand's paths appear unchanged.
A whole base term `x` becomes the child `x`: its value moves from the root to the path `x`.

```python
joint = Normal("x", 0.0, 1.0) & grouped
# joint's paths: x, a, g, g/c, g/e; its draws are {x, a, g: {c, e}}
joint["g/c"]            # the same node as grouped["g/c"]
```

Under the old grammar, a whole term over structure moved under its name in the same way, so every one of its paths changed: `params_prior["beta"]` became `joint["params/beta"]`.
Under grammar B that cannot happen, because only a base term is named whole, and a base term has no paths below the root.

**Composition meets components with slots by name, and a view offers what the grammar rule says it offers.**
A kernel whose slot is a whole group meets the group only through a list of one path:

```python
# lik_g(y | g): a kernel whose slot g takes a record {c, e}
lik_g & grouped[["g"]]      # meets the slot g: grouped[["g"]] offers g
lik_g & grouped["g"]        # does not meet it: grouped["g"] offers c and e
```

**Sampling commutes with `&` when both operands expose records.**
For independent laws `a` and `b` that each expose a record, `sample(a & b)` is distributed as `sample(a) & sample(b)`, where `&` on two records merges their fields.
For a whole base term it does not hold: `sample(Normal("x", 0.0, 1.0))` is an array, and `&` on arrays is NumPy's elementwise `&`.
jhuggins proves this boundary in a model where laws are point masses (`sample_commutes_iff`).

**Two views can offer the same name.**
`hier["s1"]` and `hier["s2"]` each offer `theta`, so `hier["s1"] & hier["s2"]` raises `ValueError`, since a joint cannot produce one name twice.
Today `hier["s1"] * hier["s2"]` composes, because each view is a whole term named after its group.
The joint law of both groups is `hier["s1", "s2"]`, which keeps the paths and the dependence between them.
`hier[["s1"]] & hier[["s2"]]` composes as well, but as the product of two marginals, so it is the joint only if the two groups are independent.

**An operand that exposes a law or a batch.**
`&` on declarations refuses an operand that exposes a law, a kernel, a function, or a batch (§ 7.7).
Grammar B exposes more outputs than today, such as a function that returns a law, so such outputs are wrapped before they merge.

### 14.4 After the workshop

These parts of `&` are design, but they merge after the workshop (`deferred.md`):
1. **`&` on `Function` and `FunctionSpec`:** the function that returns both outputs, with the shared slots unified.

   ```python
   center = Function(lambda y: jnp.mean(y), output_spec=OutputSpec(center=None))
   spread = Function(lambda y: jnp.std(y), output_spec=OutputSpec(spread=None))
   summaries = center & spread                   # the slot y, the outputs center and spread
   ```

   `pushforward(d, f & g)` is the joint law of `f(X)` and `g(X)`, which share `X`, while `pushforward(d, f) & pushforward(d, g)` is their independent product.
2. **`&` on `Batch` and `BatchSpec`:** element by element, over equal levels.

   ```python
   draws = sample(post, sample_shape=(1000,))    # elements {mu, tau}
   y_rep = simulate(draws)                       # elements {y}
   discrepancy(draws & y_rep)                    # elements {mu, tau, y}
   ```
3. **`&` on `DistributionSpec`, `ConditionalDistributionSpec`, and `FunctionSpec`, with `InputSpec.rename`:** a function that builds a model declares its output before it runs.

   ```python
   step = observation.spec & transition.spec     # y and x, given x_prev and theta
   renames = lambda t: {"x": f"x{t}", "y": f"y{t}", "x_prev": f"x{t - 1}"}
   chain = functools.reduce(operator.and_, [step.rename(renames(t)) for t in (3, 2, 1)])
   build_model = Function(make_chain, output_spec=OutputSpec(RecordSpec(model=chain & initial.spec & prior.spec)))
   ```

   The model is wrapped under `model`, since grammar B refuses `OutputSpec(model=...)` over a law.
4. **The lifting rule by postcomposition**, so a consumer that expects a value can meet a function's component (§ 5.3).

### 14.5 IV.2's rationale

IV.2 says that associativity rests on the rule that the right operand's slots avoid the left operand's components, `G_B ∩ F_A = ∅`.
jhuggins's Rocq model finds that a chain of kernels has the same slots, components, and validity in every grouping with or without that rule.
The rule has another purpose: it keeps the joint well formed.
A factor never conditions on a name that a factor to its left produces, so the conditional-first listing is a topological order of the dependence graph, and the joint's density reads from left to right.
Step S0 rewrites IV.2's rationale to say so.

## 15. Batches and their element declarations

### 15.1 What a batch stores today

A `BatchSpec` holds the spec of one element and the batch's named levels, but not the names its producer offered:

```python
draws = sample(Normal("mu", 0.0, 1.0), sample_shape=4)
draws.spec          # BatchSpec(NumericArraySpec(shape=(), dtype=float32, support=real), sample=4): nothing says "mu"
draws.components    # AttributeError
draws["mu"]         # TypeError: cannot index NumericArrayBatch by 'mu': its elements have no named fields
rb = sample(grouped, sample_shape=4)
rb["a"].spec        # BatchSpec(NumericArraySpec(...), sample=4): the column's name is not stored
```

II.5 already specifies that a batch an operation produces carries the producer's declaration, and that `Batch` has `components`; the code stores a bare `TermSpec` (`core/_batch.py:197`).

### 15.2 The element declaration

Every batch holds an element declaration (step S4):
1. **A batch an operation produces** stores its producer's declaration: the event declaration for draws (VI.3), and the completed output declaration for a sweep (V.6).
2. **A batch built from raw values** gets the default: a record element is exposed, a law or function element is exposed through its own declaration, and a base element is whole under the name `element`.
3. **A column of a batch of records** follows the view rule: `rb["a"]` stores the whole declaration `OutputSpec(a=...)`, and a group's column `rb["g"]` stores the exposed record `{c, e}`.

Grammar B applies to the stored declaration as to any other.
A batch of draws of `grouped` stores the exposed record `{a, g: {c, e}}`, and a batch of draws of `Normal("mu", 0.0, 1.0)` stores the whole term `mu`.

`BatchSpec` holds the declaration as `element`, which II.5 already names, and `element_spec` stays the type of one element, `element.term_spec`.
`Batch.components` returns the element declaration, a mapping of its components, as II.5 lists it.

```python
# Proposed
draws.spec.element      # OutputSpec(mu=NumericArraySpec(shape=(), dtype=float32, support=real))
draws.components        # the same declaration: list(draws.components) == ["mu"]
NumericArrayBatch(jnp.arange(4.0), "draw", label="xs").components   # OutputSpec(element=...)
rb["a"].components      # OutputSpec(a=...)
rb["g"].components      # OutputSpec(RecordSpec(c=..., e=...))
```

**What indexing uses.**
A batch is indexed by its element's paths, so it agrees with its law without reading the stored names (§ 9.4).
The stored names serve composition, binding, and the component accessor:

```python
draws["mu"]              # KeyError: mu names the whole element, not a path inside it
draws.component("mu")    # draws itself, as Normal("mu", 0.0, 1.0).component("mu") is the law   (step G3)
sample(grouped, sample_shape=4)["g/c"]    # a column, read through the element's paths
```

**Change from #571:** #571 kept a bare element spec for a batch built from raw values, so a batch of arrays had no components and could not be exposed. Grammar B gives every batch a declaration, and a raw batch of arrays offers `element` (gaps D1 row g). #571 also said `batch["mu"]` returns the batch; that is now `draws.component("mu")` (row e).

### 15.3 Names stay out of equality and JAX (decided 2026-10-17)

A batch's component names stay out of its equality, its hashing, the static data of its JAX pytree, and its fingerprint.

A batch is a JAX pytree, so JAX splits it into arrays and static data.
JAX compares the static data to decide whether two values have the same structure, which decides whether `jit` reuses a compiled program and whether the two branches of `lax.cond` may return the batch.
Today a batch's static data is its `BatchSpec`.
If the names were static data, draws of `Normal("x", 0.0, 1.0)` and of `Normal("y", 0.0, 1.0)` would become different structures:

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=3)
y = sample(Normal("y", 0.0, 1.0), sample_shape=3)
jax.tree_util.tree_map(jnp.subtract, x, y)   # works today, and keeps working;
                                             # with names in the static data it would raise "Mismatch custom node data"
```

A `jit`-compiled function would also recompile for each name, and the two branches of a `lax.cond` could not return draws under different names.

Keeping the names out has a cost, the one labels already pay, and only for the names of whole elements.
The names of a record element are stored in the element's `RecordSpec`, which is static data already, so they survive.
A whole element's name, as in a batch of arrays or a column such as `rb["a"]`, does not cross a raw JAX transform:

```python
jax.tree_util.tree_map(jnp.subtract, x, y)   # today: labeled '<no description>', as every raw rebuild is
jax.jit(g)(x)                                # comes back offering the default component element, not x
```

So `.component("x")` on that result, or binding it to a slot `x`, fails.
ProbPipe's own operations restore the names at the result, where they declare their result anyway.

The static data must not hold the name at all, not only compare without it.
JAX reuses the static data of the first call when it rebuilds the output of a cached compiled program, so a name kept there would come back from a later call made with another name.

This reverses #571's sentence that two batches whose elements differ only in component names are not equal (gaps D1 row h, D3).
jhuggins's model proves that `&` commutes with renaming every name at once (`dmerge_rename`).
§ 21 (O5) asks how the name is stored so that declarations stay name-sensitive while batches are not.

## 16. Deliberate exceptions

Eight cases fall outside the invariants on purpose.
Because paths go through every kind, indexing a law, a function, or a batch held in a field is not one of them; what such terms lack is summaries and conditioning.

| Case | What happens | Why |
| --- | --- | --- |
| Covariance | `cov(d)` is one base term, so `cov(d)[s]` raises | It is not one value per field. The block for `s` is `cov(d[s])`. |
| Monte Carlo summaries | `mean(d)[s]` and `mean(d[s])` estimate the same quantity but are not equal | Each call takes its own draws. |
| Summaries and conditioning at a law- or function-valued field | `mean(rm)` is undefined for a random measure's drawn laws, and `condition_on` cannot fix a drawn law or function | A mean of laws is not one value per field; I1, I3, and I8 hold for paths that end at values. |
| `&` on values of whole base terms | `sample(Normal("x", 0.0, 1.0) & Normal("y", 0.0, 1.0))` is a record `{x, y}`, but `sample(Normal("x", 0.0, 1.0)) & sample(Normal("y", 0.0, 1.0))` is NumPy's elementwise `&` of two arrays; the same holds for two batches of arrays | A base term carries no name in its value, so sampling commutes with `&` only when both operands expose records. |
| A batch of arrays through raw JAX | `jax.jit(g)(draws)` comes back offering `element` instead of `x` | Whole-element names stay out of the pytree (§ 15.3). |
| Labels | `sample(grouped["a", "g/c"])` and `sample(grouped)["a", "g/c"]` are draws of the same law under different labels | A label says how a term was made, for a reader; no operation reads it. |
| Kernels, until step G5 | `lik["y"]` raises `TypeError`, and so does a path through a field that holds a kernel | Kernel views come in a later step. |
| Python's mapping rule | `"g" in rec` is true, while `"g"` is not in `rec.keys()` | `keys()` lists fields only, and `in` agrees with `[]` (§ 11). |

**Before every step has merged.**
Some invariants hold only once their step merges, as § 3's last column states.
Between the workshop and step G3, for example, paths stop at a field that holds a law, and a function has no `[]`.
The design text states the target, and the design conformance tests mark each pending signature, so this interim is not an exception.

## 17. Vocabulary

One vocabulary runs through this plan and `design/`, and through the docstrings and messages of the code.

| Today | Proposed | Why |
| --- | --- | --- |
| "packaging", "packaged" | "what a declaration exposes"; the verb *exposes* | The code already says `exposes_record` and `_exposed_kind`, and the verb avoids the sense of Python packaging. |
| "produced slot", "output slot" (III.7, IV.2, VI.9) | "component" | A slot is a place where a consumer receives a value. |
| "input slot" | "slot" | A slot is always an input. |
| "map-like kind" (II.2) | "consumer" | It pairs with "producer". The execution plan says "requester" for the code that asks for a set of points. |
| "packaged sub-joint" (IV.1), "packaged joint" (IV.2) | "grouped sub-joint" | A rename that gathers components under one node creates it. IV.2's paragraph "Packing the joint" becomes "The joint's event". |
| "interior node", "subtree", "field group", "group of fields" | "group" | One word for a node of a record that is itself a record. |
| "key" for a field path | "path"; "subscript" for what `[]` takes | "Key" means a random key, and a mapping's keys. |
| "selection" for batch positions and for several paths | "positions" and "a list of paths" | `select` is a method with another meaning. |
| `FieldView` | `MarginalView` | It covers groups and lists of paths, and pairs with `marginal`. |
| `with_path_names` | `rename` | The name pandas and xarray use. |
| `*` for composition | `&` | `*` is arithmetic on values. |
| `OutputSpec.spec`, `with_spec` | `term_spec`, `with_term_spec` | The glossary's term for the produced term's type. |
| `marginal(d, field)` | `marginal(d, paths)` | It takes a path or a list of paths. |

**The glossary** (`design/glossary.md`) changes in step S0:
- **new entries:** base spec, spec with parts, node, group, leaf path, slot, producer, consumer, exposes, whole declaration, exposed declaration, address, component spec, wrap, position, subscript;
- **redefined:** *component* takes the definition of § 4.4 and keeps its second sense, the component laws of a `MixtureDistribution` (VII.3); *field*, *path*, and *declaration* take the definitions of § 4.2 and § 4.4; *marginal* takes "a path or a list of paths"; *view* names `MarginalView`; *type hole* names `with_term_spec`;
- **narrowed:** *key* keeps only the random key and a mapping's keys;
- **removed:** *packaging*, and *selection*, apart from *selection order* (II.7), which stays (gaps T3);
- **kept:** the occurrence path (V.8), *given slot* and *optional slot*, rewritten to say "slot".

The canonical-names table changes in the same step:
- the row "a field key within a tree, or the name assigned to a field: `field_name` or `key`" becomes "a path within a tree: `path`, and `paths` for several" and "the name assigned to a field: `field_name`";
- a new row, "the declaration a batch's elements satisfy: `BatchSpec.element`", beside the row for `element_spec`, which names the type.

The execution plan's glossary entries (point, requester, executor, and others) belong to its step E1, and its "automatic selection" points to *selection order* (gaps T3).

## 18. Changes to `design/`, `STYLE_GUIDE.md`, and the CHANGELOG

### 18.1 `design/`, section by section

II.6 becomes the one statement of paths and subscripts, and II.2 the one statement of the declarations and the grammar rule.
Every other section cites them instead of restating a rule (`STYLE_GUIDE.md` §10, rule 8).
Step S0 writes the target text of every row; the design conformance tests get pending entries for the signatures that change before the code does.

| Section | Today | Under this plan |
| --- | --- | --- |
| II.1 Term specs | The size shorthand is a `RecordSpec` field's alone | States the tuple shorthand once, for every place a declaration takes a term spec |
| II.2 Declarations | The keyword form names any spec whole; `spec`, `components`, `exposes_record`, `with_spec`; "packaging"; "map-like kind"; a whole term's paths start with its name; "an exposed record's components are its fields"; `OutputSpec(posterior=DistributionSpec(...))` as the example of a function returning a law; a law's exposed components typed by one draw's value | The definitions of § 4; the grammar rule; `OutputSpec` as a mapping with `term_spec`, `with_term_spec`, `project`, `without`, `rename`, and `&`; the exposure table of § 7.2; components, addresses, and component specs; the law's event rule; the four sources of slots; `InputSpec` `&`, `without`, and `rename`; defaults by kind and type holes; a returned law is exposed or wrapped |
| II.4 Labels and views | A field of a value takes its path as its label; batch positions are a "selection" | Adds the label of a value indexed by several paths, `(a, g/c)`; positions are "positions" |
| II.5 `Batch` | A tuple of names is one path; positions are a "selection"; `()` is the whole batch; a raw batch keeps a bare element spec; for a whole element, a name returns the batch | A batch is indexed by its element's paths and always holds an element declaration (`BatchSpec.element`); raw defaults; names out of equality and the pytree; `()` and lists of positions raise; `in` tests paths; the component accessor reads a whole element |
| II.6 `NamedTree` | A path may be a tuple; `[]` raises on a group; `in` tests fields; a key is a field's path; `without(path)` takes one path | Owns the definitions of § 4.2, the reading of a subscript, and `in`; `[]` returns groups; paths continue into a held law, function, or batch; a list of paths keeps its paths, in the tree's order; `without(*paths)` as the code has it; `rename` |
| III.3 `Function` | Output components default to the callable's name for any non-record; a bare term spec uses the same default; no `[]` | Defaults by kind (§ 5.5) and the bare-spec reading (§ 5.6, gaps T4); `input_spec` takes the four sources; `f[s]` indexes what `f` returns |
| III.5 `Record` | `select` names entries by full path | `[]` names groups and held terms; `select` flattens to last segments; `&` |
| III.6 `RecordBatch` | A tuple of names is one path | Cites II.6; a list of paths returns a `RecordBatch`; `select` flattens as on a record |
| III.7 `Distribution` | `d[c]` returns `d` for a whole term; "produced slots", "output slot"; a view of a group is a whole term; several paths flatten to last segments; `OutputSpec(parameters=RecordSpec(...))` has "the output slot `parameters` and the event path `parameters/beta`" | A view's declaration is `event_spec.project(s)`; a whole term's name is read by `component`, not `[]`; a view of a group exposes its parts; several paths keep their paths; `MarginalView`; cites II.2 for the law's event rule |
| III.8 View capabilities | Derivations use component-rooted paths | Derivations use the term's paths |
| III.9 Kernels | No `[]`; `given_spec` sources stated locally | Points to II.2 for the sources of `given_spec`; states the `[]` exception now and `lik[s]` as the target |
| III.10 Batches of laws | Silent on `[]` | `laws[s]` is the batch of views |
| IV.1 Joints | A "packaged sub-joint" declares a group as one whole term | A grouped sub-joint wraps its group, so its draws carry the group's level; the joint's event uses `&` |
| IV.2 Composition | `*`; "produced slots"; "Packing the joint"; associativity "rests on" `G_B ∩ F_A = ∅` | `&`; components; "The joint's event"; the rationale of § 14.5 (gaps T5); the operator-coexistence note says `*` scales and `&` composes |
| IV.4 Laws from functions | Sources of `given_spec` stated locally | Points to II.2 |
| V.5, V.6 Lifting and planning | Atoms and outputs named whole under a component; "The declared output" reads any non-record bare spec as whole | Defaults by kind: a lifted law of laws wraps its draws, and a sweep's batch is exposed through its element; the bare-spec reading of § 5.6 |
| VI.0 The operation model | "Selection commutes with a coordinatewise summary" | "Indexing commutes with a coordinatewise summary" |
| VI.6 `condition_on` | Given values are named by "keys of the event schema"; the result's declaration depends on the route | Given values are named by paths, nested ones included; the result is declared as `d.without(*fixed)` (I11) |
| VI.8 `marginal` | `marginal(d, field)`; one path or several, decided by count; a group's marginal is the factor that holds it, a whole term; fields "repackaged in the order of the paths" | `marginal(d, paths)`: a path or a list of paths, decided by form; a group's marginal exposes its parts, as the factor's view at the group; the tree's order |
| VII.2, VII.5, VII.6 Families | Empirical laws whose atoms are laws or functions, bootstrap laws, random functions, and random measures name their event whole | Their draws are wrapped in a record, as the law's event rule requires |
| Glossary | As § 17 lists | As § 17 lists |

Each use of "packaging", "packaged", "produced slot", "output slot", "map-like kind", and "input slot" in parts II–VII changes, about 40 lines at `a703f15e` (excluding Python packages in `package-structure.md`), including the II.2 row of Part II's summary table.

### 18.2 `STYLE_GUIDE.md`

| Section | Today | Under this plan |
| --- | --- | --- |
| §1.10 Record field iteration and path access | `"a/b"` is "a short form of the tuple form"; `[]` returns leaves only, so `at_path` is the way to a group; `with_path_names` | A path is a `str`; a tuple is a list of paths; `[]` and `in` name groups while `keys()` stays leaf-keyed; `rename` |
| §1.11 Distribution iteration | A `DistributionBatch` is indexed by position only | A batch of laws is also indexed by path (with step G5) |

`STYLE_GUIDE.md` describes the code as it stands (AGENTS.md), so § 21 (O7) asks whether §1.10 changes with S0 or with S1.

### 18.3 `docs/`

- `docs/api/declarations.md` opens with the definitions of slot, component, and what a declaration exposes (step S2).
- The User Guide notebooks are updated mechanically by each step that breaks them (README directive 6). User Guide 04 and 08 read `event_spec.components[...]` today.
- `docs/tutorials/` is out of scope, so its dict-comprehension idioms wait for the tutorial rewrite.

### 18.4 Breaking changes for the CHANGELOG

Every rename and removal is a clean break: no alias, no deprecation warning, and an entry under `### Changed (breaking)` (decided 2026-10-17).

| Change | Step |
| --- | --- |
| A tuple of `str` is no longer one nested path, on `[]`, `at_path`, `is_field`, `without`, and the keys of `with_path_names` (later `rename`) | S1 |
| A law indexed by several paths keeps their paths instead of flattening to last segments, in the tree's order | S1 |
| `marginal`'s parameter is renamed from `field` to `paths`, and a marginal of several factors lists them in the law's order | S1 |
| `draws[()]` raises instead of returning the whole batch | S1 |
| `without` refuses a path given twice and a path inside another | S1 |
| `select` flattens to last segments, on records and on record batches | S1 |
| `in` changes on records, batches, and laws | S1 |
| `OutputSpec.spec` is renamed `term_spec`, and `with_spec` is renamed `with_term_spec` | S2 |
| `OutputSpec.components` and `exposes_record` are removed: an `OutputSpec` is a mapping of its components | S2 |
| `InputSpec.without` raises `KeyError` for a name that is not a slot | S2 |
| A law's event declaration that exposes a law, a kernel, a function, or a batch raises `TypeError` | S2 |
| `&` replaces `*` for composing laws and kernels, and replaces `merge` on records (§ 21, O2) | S3 |
| `with_path_names` is renamed `rename` | S3 |
| `FieldView` is renamed `MarginalView` | S3 |
| A raw batch of arrays offers the default component `element`; two batches that differ only in a whole element's name are equal | S4 |
| A view or marginal of a group offers its parts, not the group's name | G1 |
| A factor that holds a group, and a lifted law-returning function, wrap their draws, so each draw gains a level | G1 |
| A function returning a law or a function offers the returned term's components, typed by their views; a consumer that met it by its name now needs a wrap | G1 |
| A bare law, function, or batch spec given as `output_spec` is exposed instead of whole | G1 |
| A random measure, a random function, a bootstrap law, an empirical law of laws, and a minibatched target wrap their draws | G2 |
| `OutputSpec(name=spec)` raises for a spec with parts, and so does filling a type hole with one | G2 |
| A whole term's name is no longer a subscript: `Normal("x", 0.0, 1.0)["x"]` raises | G3 |
| `condition_on` returns a law that exposes a record on every route | G4 |

## 19. Implementation

### 19.1 The steps and their order

The steps are those of [`README.md`](README.md) § Steps, and they run in the order of gaps F1 (decided 2026-10-17).
Every step is one PR into `dev/overhaul`.
Before the workshop: S0–S4 and G1.
After it: G2–G5 and the parts of `&` in § 14.4.

The order, with the execution steps that edit the same modules:
1. **S0** (design text) after this `plans/` PR merges.
2. **S1** together with **E2**. They share only `_conditional.py`, where S1 changes one constant and E2 the construction probe.
3. **S2**, announced in advance, since its renames change about 50 modules. E3's `core/` part runs alongside it.
4. **S3**, announced, after S2, so every open branch rebases once. E3's rewrite of `_broadcast.py` and `_sweep.py` starts after S2.
5. **E4** and **E5** after S2 and S3, since both edit `_conditional.py` and `_factored.py`.
6. **S4** after E3, since both rewrite the sweep.
7. **G1** after S2 and S4. § 21 (O9) proposes running it after E5, since both edit `functions/_broadcast.py`.

Each step adds property tests for the invariants it delivers, and marks the invariants that later steps deliver as expected failures until their step merges.
Each step updates the User Guide notebooks it breaks, mechanically, and records its breaking changes in the CHANGELOG (§ 18.4).

### 19.2 S0: the design text

**Delivers:** the target text of § 18.1, the glossary of § 17, and the `STYLE_GUIDE.md` rows of § 18.2 (or with S1, by O7).
II.2 states the declarations and the grammar rule, and II.6 the paths and the reading of a subscript.
IV.2's rationale takes the fix of § 14.5.
V.6 and III.3 state the bare-spec reading of § 5.6.

**Files:** `design/02-shared-abstractions.md`, `03-term-kinds.md`, `04-distributions.md`, `05-functions.md`, `06-operations.md`, `07-distribution-catalog.md`, `glossary.md`; `STYLE_GUIDE.md`.

**Done when:** `tests/docs/` and the four `test_design_conformance.py` files pass, with pending entries for the signatures that change before the code; `scripts/design/design_blocks.py check` passes for each edited section; and each candidate that `scripts/design/prose.py` reports is resolved under `STYLE_GUIDE.md` §10.

### 19.3 S1: one subscript reader

**Delivers:** § 8, § 9.1, § 9.3 for record specs, § 9.4 apart from batches of laws, the list-of-paths rule of § 9.5 and § 10, § 11 apart from held terms, `marginal(d, paths)` (§ 12.1), the reader's errors in `without` (§ 12.4), and `select` (§ 12.5).
`[]` returns groups on `Record` and `RecordSpec`; a list of paths keeps its paths on records, batches, and laws; `()` on a batch raises; `_view_declaration`'s list branch keeps paths and the tree's order.

**Invariants:** I4, I9, I5 on records and batches, I10 on records, record specs, and batches, I2 and I7 for lists of paths, and I3 with #576.

**Files:**
- `core/_shapes.py`: the one reader, beside `_as_names`;
- `core/named_tree.py`: the one `_PATH_SEP`, `__getitem__`, `__contains__`, `at_path`, `is_field`, `without`, the keys of `with_path_names`;
- `core/record.py`, `core/_record_spec.py`, `core/_record_batch.py`: groups, `select`;
- `core/_batch.py`: `__getitem__`, `__contains__`, `()`;
- `distributions/_views.py`, `distributions/_distribution.py`: lists of paths on laws, labels, `in` on laws;
- `operations/_marginal.py`: `paths`, lists;
- the eight modules that define their own `_PATH_SEP`: `operations/_marginal.py`, `operations/_condition.py`, `functions/_descendants.py`, `distributions/_empirical.py`, `distributions/_views.py`, `distributions/_factored.py`, `distributions/_conditional.py`, `families/_resampling.py`.

**Done when:** tests cover every row of § 8's error table on records, record specs, batches, and laws; `hier["s1/theta", "s2/theta"]` and `marginal(grouped, ["a", "g/c"])` give the laws of § 10; `draws[3]["g"] == draws["g"][3]`; `mean(grouped)["g"]` is the record of means; `"g" in sample(grouped)` and `"a" in grouped` are true; and no module but `named_tree.py` defines `_PATH_SEP`.

### 19.4 S2: `OutputSpec` as a mapping, and one reading of slots

**Delivers:** § 6.2–6.5 and § 7.3–7.8.
`OutputSpec` becomes a mapping; `spec` becomes `term_spec` and `with_spec` becomes `with_term_spec`; `components` and `exposes_record` go; `project` and `without` read through S1's reader; the `InputSpec` constructor takes the four sources and the size shorthand; `InputSpec.without` refuses an unknown name; the law's event rule refuses an exposed law, kernel, function, or batch; `&` on `InputSpec` and `OutputSpec` (by O1).
Its renames change about 50 modules, so it is announced in advance, as S3 is.

**Invariants:** I6 for the subscripts S1 supports, and I10 on declarations.

**Files:**
- `core/_specs.py`: the mapping, `term_spec`, `with_term_spec`, `project`, `without`, `&`, the event-rule check helpers;
- `values/_function_base.py`: the `input_spec` reading (line 1093), `FunctionSpec`'s checks of `output_spec`;
- `distributions/_conditional.py`: `_slots_of`, `_complete_given_spec`, the reading of `given_spec` (line 1392), `ConditionalDistributionSpec`'s `given_spec`, the event rule;
- `distributions/_distribution.py`: `DistributionSpec` and the event rule;
- `distributions/_views.py`: `_view_declaration` becomes `OutputSpec.project`;
- `distributions/_factored.py`: the joint's event (line 447) and its unmet slots through `&`;
- `core/_fingerprint.py` and every other module that reads `_component_name`, `.spec` on a declaration, `components`, or `exposes_record`: 29 lines read `_component_name` outside `_specs.py`, and `.event_spec.spec` appears on 89 lines of `probpipe/` and 197 of `tests/`;
- the 7 callers of `_components_record`, which use `RecordSpec(decl)` where they need a record;
- `docs/api/declarations.md`.

**Done when:**
- no module reads `OutputSpec._component_name` outside `_specs.py`;
- the law of § 5.4 raises `TypeError` at construction;
- a nested `RecordSpec` and an `OutputSpec` each declare a kernel's slots, and the regrouped prior of § 6.1 builds a kernel;
- an `OutputSpec` source with a component name that is not an identifier, or with a type hole, raises;
- `InputSpec(K=())`, `given_spec={"K": ()}`, and `OutputSpec(K=())` build, and `InputSpec(K=3)` raises;
- tests cover each error of § 7.5–7.7 and check `d[s].event_spec == d.event_spec.project(s)` and `(a & b).event_spec == a.event_spec & b.event_spec` over the running examples;
- `InputSpec.without("zzz")` raises `KeyError`, and its three callers pass only slots;
- the full suite passes.

### 19.5 S3: `&`, `rename`, and `MarginalView`

**Delivers:** one announced mechanical PR:
- `&` replaces `*` for composing laws and kernels: `Distribution.__mul__`, `ConditionalDistribution.__mul__`, and the composition branch of `GaussianRandomFunction.__mul__`, whose scaling branch stays;
- `&` on `Record` and `RecordSpec`, replacing `merge` (by O2);
- `with_path_names` becomes `rename` on every kind that has it;
- `FieldView` becomes `MarginalView`.

**Invariants:** none new; every existing test passes under the new names.

**Files:** `distributions/_distribution.py`, `distributions/_conditional.py`, `families/_gaussian.py`, `core/named_tree.py`, `core/_specs.py`, `distributions/_views.py`, and every caller: about 380 lines name `with_path_names` and about 260 name `FieldView` across `probpipe/`, `tests/`, `docs/`, and `design/`.

**Done when:** no `__mul__` composes laws; a search for `with_path_names` and `FieldView` finds nothing outside the CHANGELOG; the User Guide notebooks run; and the full suite passes.

### 19.6 S4: every batch stores its element declaration

**Delivers:** § 15.
`BatchSpec` holds `element`; draws (VI.3) and sweeps (V.6) store their producer's declaration; a raw batch gets the default; a column of a batch of records stores the view's declaration; `Batch.components`; whole-element names stay out of equality, hashing, the pytree's static data, and fingerprints (by O5).

**Invariants:** none of I1–I13 directly; it stores what I12 reads for batches (G3).

**Files:** `core/_batch.py`; the pytree registrations in `core/_numeric_array_batch.py`, `core/_record_batch.py`, `core/_numeric_record_batch.py`, and the other batch classes; `core/_fingerprint.py`; `functions/_sweep.py`, `functions/_result.py`; `operations/_sample.py`.

**Done when:**
- `sample(Normal("mu", 0.0, 1.0), sample_shape=4).components == OutputSpec(mu=...)`;
- `NumericArrayBatch(values, "draw").components` offers `element`;
- `rb["a"].components == OutputSpec(a=...)` and `rb["g"].components` exposes `{c, e}`;
- `jax.tree_util.tree_map(jnp.subtract, x, y)` works for draws named `x` and `y`, a jitted function compiles once for both, and a jitted call made with `y` after one made with `x` returns a batch offering `element`, never `x`;
- a rename of a batch's whole element keeps its fingerprint.

### 19.7 G1: grammar B for the kinds the workshop uses

**Delivers:** § 5.3–5.6 for the kinds the workshop uses, and § 9.5's view of a group:
1. a view or marginal of a group exposes its parts, and a group's marginal is the factor's view at the group;
2. the library's laws declared whole over a record are exposed or wrapped;
3. a grouped sub-joint wraps its group;
4. a `FunctionSpec`, and a `BatchSpec` of base elements, become exposable;
5. a function that returns a law or a function exposes its output;
6. a lifted law-returning function wraps its draws (cut third if time runs short);
7. `OutputSpec.default` follows the defaults by kind, and `_complete_output_spec` the bare-spec reading (the default-output part, cut third with item 6).

**Invariants:** I2, I5, I6, and I7 for groups.

**Files:** `distributions/_views.py` (`_view_declaration`, `_regrouped`), `operations/_marginal.py`, `distributions/_factored.py` (line 476), `inference/_approximate_distribution.py`, `distributions/_empirical.py`, `core/_specs.py` (`_exposed_kind`, `default`), `values/_function_base.py` (`_complete_output_spec`), `functions/_broadcast.py` (line 356), `functions/_result.py`, `distributions/_conditional.py` (line 1463), `distributions/_from_functions.py`.

**Done when:**
- `grouped["g"]["c"]` is the view of `g/c`, and `grouped["g"].event_spec == grouped.event_spec.project("g") == marginal(grouped, "g").event_spec`, exposing `c` and `e`;
- `factor(grouped, "g")` draws `{g: {c, e}}`, while `grouped`'s draws are unchanged;
- `Function(run_inference, output_spec=post_spec).output_spec` offers `mu` and `tau`, and a `fit_emulator` declaration offers `mean` and `var`;
- lifting `mk` over a law gives draws `{mk: law}`;
- a test-only probe on `OutputSpec`, run over the suite, finds no whole declaration over a spec with parts that the library builds, apart from the G2 families.

### 19.8 After the workshop

| Step | Delivers | Invariants | Files | Done when |
| --- | --- | --- | --- | --- |
| G2 | The G2 families wrap their draws (§ 5.3); then the keyword form and the filling of a type hole refuse a spec with parts (§ 5.2, § 5.5) | I1 for those families | `families/_random_functions.py`, `families/_gaussian.py`, `families/_resampling.py`, `distributions/_empirical.py`, `inference/_minibatch.py`, `core/_specs.py` | `rm`'s draws are `{G: law}` and `gp`'s are `{f: function}`; `OutputSpec(params=RecordSpec(beta=(), sigma=()))` raises `TypeError` with the message of § 5.2; `OutputSpec(x=None).with_term_spec(RecordSpec(a=()))` raises |
| G3 | Paths through every kind (§ 4.2, § 9.2); `Function[]` (§ 9.6); `component` (§ 9.7); a view at or below a field that holds a law or a function wraps; a whole term's name stops being a subscript | I1, I5 across kinds, I12, I13 for functions | `core/named_tree.py`, `core/record.py`, `core/_batch.py`, `values/_function_base.py`, `distributions/_views.py`, `core/tracked.py` | property tests of I1, I5, I12, and I13 over records, laws, functions, and batches |
| G4 | `condition_on` declared as `d.without(*fixed)` on every route, with nested paths route by route (§ 12.2); `law.without` (§ 12.4) | I8, I10 on laws, I11 | `operations/_condition.py`, the conditioning routes of the families, `distributions/_views.py` | property tests of I8, I10, and I11, including the Gaussian route |
| G5 | `[]` on batches of laws and of functions, and on kernels (§ 9.4, § 9.6) | I13 for kernels, I1 on batches of laws | `distributions/_batches.py`, `distributions/_conditional.py`, `core/_function_batch.py` | `laws["mu"]` is the batch of views; `lik[s](v)` is `lik(v)[s]` |
| — | `&` on functions and batches, spec-level `&`, `InputSpec.rename`, and lifting by postcomposition (§ 14.4) | — | `values/_function_base.py`, `core/_batch.py`, `core/_specs.py`, `functions/_broadcast.py` | the examples of § 14.4 run |

Before G1 and G2 are final, jhuggins's Rocq model should be extended to renames by path and to paths inside slots (`deferred.md`, gaps L1).

### 19.9 The required updates, as a checklist

This is #571's checklist, rewritten for the reconciled design.

- [ ] **The design reference (S0).** II.2 owns the definitions, the grammar rule, the exposure table, the law's event rule, `term_spec`, the sources of slots, and the operations. II.1 states the size shorthand once. II.5 states the element declaration and the raw defaults. II.6 owns paths and subscripts and states `without(*paths)`. III.3, III.9, and IV.4 point to II.2 for the sources of `input_spec` and `given_spec`. III.7 defines a view's declaration as `event_spec.project(s)`. IV.1 and IV.2 use the new vocabulary and define the joint's event with `&`. The glossary changes as § 17 lists. *(Done as § 19.2 states.)*
- [ ] **The subscript reader (S1).** *(Done as § 19.3 states.)*
- [ ] **`OutputSpec` as a mapping, and `term_spec` (S2).** Replace the reads of `exposes_record` (33 lines of `probpipe/`, 31 of `tests/`), of `_component_name` outside `_specs.py` (29 lines), and of `.spec` on a declaration (`.event_spec.spec` on 89 and 197 lines, `.output_spec.spec` on 4 and 24). Update the docstrings and messages that say "packaging", "packaged", or "map-like kind", such as the `InputSpec` docstring. *(Done when no module reads `_component_name` outside `_specs.py` and the full suite passes.)*
- [ ] **The law's event rule (S2).** Refuse an exposed law, kernel, function, or batch as the event declaration in the constructors of `Distribution`, `ConditionalDistribution`, `DistributionSpec`, and `ConditionalDistributionSpec`. *(Done when the law of § 5.4 raises `TypeError` at construction.)*
- [ ] **One reading of slots (S2).** The `InputSpec` constructor accepts the four sources and the size shorthand; `_slots_of`, `_complete_given_spec`, and the `input_spec` reading construct through it; the keyword form of `OutputSpec` accepts the shorthand. *(Done when a nested `RecordSpec` and an `OutputSpec` each declare a kernel's slots, the regrouped prior of § 6.1 builds a kernel, and an `OutputSpec` source with a non-identifier component or a type hole raises.)*
- [ ] **The operations (S2).** `OutputSpec.project`, `without`, and `&`; `InputSpec` `&`; `InputSpec.without` raises on an unknown name; `_view_declaration` becomes `project`; the joint's event uses `&`; the callers of `_components_record` use `RecordSpec(decl)`. *(Done when tests cover each raise and check the identities for `project` and `&`.)*
- [ ] **The renames (S3).** *(Done as § 19.5 states.)*
- [ ] **The batch's element declaration (S4).** *(Done as § 19.6 states.)*
- [ ] **Grammar B for the workshop's kinds (G1), and for the rest (G2).** *(Done as § 19.7 and § 19.8 state.)*
- [ ] **The documentation (S2, then each step).** `docs/api/declarations.md` opens with the definitions; the User Guide notebooks follow each step; the CHANGELOG records each breaking change of § 18.4. *(Done when `uv run mkdocs build --strict` and the User Guide notebook job pass.)*

## 20. Alternatives considered

1. **Grammar B′.** It would name whole any spec with no paths below its root, and keep a few names such as a Gaussian process's `f`. It needs a flag on every declaration, since the spec would no longer decide the form. Rejected (decided 2026-10-17).
2. **An `exposes` property**, with #571's four values `"whole"`, `"record"`, `"event"`, and `"element"`, or jhuggins's boolean `exposes_parts`. Under grammar B the term spec decides the form, so the property would only restate it (gaps D1 row a).
3. **Narrowing the law's event rule** to allow exposed laws and kernels with two or more components, so that a random function could expose its output and its draws stay functions. It changes what a joint's draw stores and needs its own analysis; a view of a law-valued field wraps instead (decided 2026-10-17).
4. **ProbPipe wraps a returned value** under a declared name, so that a body may return the law alone. It converts values behind the caller's back; the body builds the record instead (decided 2026-10-17).
5. **Names in a batch's equality and pytree**, as #571 proposed. Draws of `x` and `y` would then be different structures to JAX, so `jit` would recompile per name and `lax.cond` could not return either (decided 2026-10-17).
6. **`components` returns a `RecordSpec`.** A `RecordSpec` indexes leaf paths, so `components["growth"]` would raise for a component that is a record, and every reader of `components` would change.
7. **`RecordSpec` replaces `InputSpec`.** A record has no optional fields and no rule for identifier names, and its merge raises on a shared name, where slots unify.
8. **A given slot's type stays pending until composition fills it.** `conditional_distribution` evaluates the function at stand-ins of each slot's type to read its event declaration and its claims (IV.4), and III.7 requires a complete declaration, so a kernel with a pending slot would have no event declaration.
9. **`given_spec` holds specs for more names than the function has parameters.** This removes the need for `without`, and it accepts a misspelled name without an error.
10. **A law's event exposes laws, and each operation keeps what the declaration exposes.** A field of the joint's draw would then hold a law, and the joint's declaration would type it by the inner law's event, which is the failure of § 5.4.
11. **`produced slot` and `consumed slot`** in place of component and slot, with `OutputSpec.produced_slots`. The paired names state the matching, but the current names give each role its own word and keep the reading `decl[c]`; jhuggins's reply keeps component and slot.
12. **Keep `*`, and give declarations a `merge` method**, or support both operators for a release. `*` is arithmetic on values, and a release with both contradicts the clean-break rule (gaps D2, D5).
13. **Other names for `with_path_names`:** `with_names` does not say whose names, `with_field_names` is wrong because it also renames groups and whole terms, and `relabel` collides with "label".
14. **An error for an ambiguous rename key**, which the grammar-A plan needed. Under grammar B no key is ambiguous (`no_ambiguous_key_B`).

## 21. Open points

Each item below is a choice that the sources and [`decisions.md`](decisions.md) leave open.
The text above follows each recommendation and marks the place with the item's number.
Once the maintainer answers, the answer goes to `decisions.md` and the text changes in the same commit.

### O1. Which step adds `&` on `InputSpec` and `OutputSpec`

**Background.**
README § Steps gives step S2 "`InputSpec.merge`" and step S3 "`&` replaces `*` for laws, kernels, records, and declarations".
The decisions of 2026-10-17 say that `&` replaces both of #571's `merge` methods, and that S3 is one announced mechanical PR.
Neither `merge` method exists today.
`&` on a declaration is a new operation, not a rename, so it is behaviour rather than a mechanical change.

**Options.**
1. S2 adds `&` on `InputSpec` and `OutputSpec`, as new operations with their tests; S3 stays mechanical: `*` to `&` on laws and kernels, `merge` to `&` on records (O2), and the two renames.
2. S2 adds `InputSpec.merge` and `OutputSpec.merge`, and S3 renames them to `&`. A public method would exist for one step, against the clean-break rule.
3. S2 implements the merge rules privately, and S3 exposes them as `&`, adding behaviour to the mechanical PR.

**Recommendation.** Option 1. It keeps each PR one kind of change (README § Branches and PRs), and no public name exists for a single step only.

### O2. Whether `Record.merge` and `RecordSpec.merge` go when `&` arrives

**Background.**
`NamedTree.merge(other)` combines two trees and raises on a field in both (II.6).
jhuggins's reply gives `Record` and `RecordSpec` `&` with the same rule, and says `&` replaces "both `merge` methods", meaning #571's two proposals.
It says nothing of `NamedTree.merge`, which has 3 callers in `probpipe/` and 21 in `tests/`.

```python
# Today
RecordSpec(a=()).merge(RecordSpec(b=()))     # RecordSpec(a=(), b=())
Record({"a": 1.0}) & Record({"b": 2.0})      # TypeError: unsupported operand type(s) for &

# Proposed
RecordSpec(a=()) & RecordSpec(b=())          # RecordSpec(a=(), b=())
```

**Options.**
1. `&` replaces `merge` on records, in S3, with no alias.
2. Both stay, as two spellings of one operation.

**Recommendation.** Option 1, since the push gives each operation one spelling and breaks cleanly.

### O3. The name a view strictly below a law-valued field keeps

**Background.**
A view at or below a field that holds a law or a function wraps, keeping the field's name (decided 2026-10-17).
For a view at the field, the name is clear: `rm["G"]` draws `{G: law}`.
For a view below the field, the decision's wording does not say which name the wrap uses.
`rm["G/mu"]` is the law whose draws are the views `law["mu"]` of each drawn law.

**Options.**
1. The last segment of the path, as for a base node: `rm["G/mu"]` draws `{mu: law["mu"]}`. This matches `grouped["g/c"]`, which names `c`.
2. The field's name: `rm["G/mu"]` draws `{G: law["mu"]}`. This keeps the name of the field the draws come from.

**Recommendation.** Option 1. A single-path view names its node by the path's last segment in every other case, and `rm[["G/mu"]]` already gives the form that keeps `G`, with draws `{G: law[["mu"]]}`.

### O4. A type hole filled with a batch of base elements

**Background.**
The decision of 2026-10-17 says that a type hole filled with a spec that has parts raises.
A `BatchSpec` is a spec with parts.
The addressing plan carved out one case before that sentence: a type hole filled with a batch whose element is a base spec becomes the exposed batch whose element is whole under the declared name.

```python
f = Function(simulate_many, output_spec=OutputSpec(sims=None))   # returns a batch of arrays
# addressing plan: f's output exposes the batch, whose element is whole sims; it offers sims
# a literal reading of the decision: TypeError once G2 merges
```

The case wraps nothing and builds no record: a base element carries no name in its value, so the declared name only replaces the default `element`.

**Options.**
1. Keep the addressing plan's case, and read the decision as covering every other spec with parts.
2. Raise for every spec with parts, so a function that returns a batch of arrays declares `OutputSpec(BatchSpec(OutputSpec(sims=...), ...))` or relies on the default.

**Recommendation.** Option 1. The decision answered "who builds the record", and here there is no record to build.

### O5. How a batch's whole-element name is stored

**Background.**
A batch's component names stay out of its equality, hashing, pytree static data, and fingerprint (decided 2026-10-17).
A declaration must still distinguish names: `OutputSpec(BatchSpec(...))` for a batch whose element is whole `x` offers `x`, and one whose element is whole `y` offers `y`, so the two declarations should be unequal.
II.5 already names the stored declaration `BatchSpec.element`, and today a batch's static data is its `BatchSpec`.

**Options.**
1. `BatchSpec.element` holds the declaration. `BatchSpec` equality and hashing ignore a whole element's name. The pytree's static data holds the spec with that name reset to `element`, so a cached compiled program never returns another call's name. `OutputSpec` equality compares component names as well as the term spec (§ 7.8), so declarations stay name-sensitive.
2. The name is stored on the batch beside its label, and `BatchSpec` holds no whole-element name. A declaration that exposes a batch of base elements then has nowhere to store its name, apart from a producer's default.

**Recommendation.** Option 1. It keeps II.5's API, satisfies the decision, and keeps every declaration's names in its equality.

### O6. Whether `produced_declaration` and `with_produced_declaration` are public

**Background.**
jhuggins's reply gives every term spec two methods, `produced_declaration()` (the declaration a spec holds, or `None` for a base spec) and `with_produced_declaration(declaration)`.
`project`, `without`, path reading, and matching use only these.
The reply writes them as public methods of `TermSpec`.

**Options.**
1. Public, so a new kind of term spec can take part in exposure, and `spec.produced_declaration() is None` tests for a base spec.
2. Private (`_produced_declaration`), since users call `project`, `without`, `[]`, and `component`, and a new term spec kind is a contributor's change.

**Recommendation.** Option 2. The public operations cover every use a user has, and a smaller public API is easier to learn. A user who needs to test for a base spec checks `isinstance(spec, (NumericArraySpec, OpaqueSpec))`.

### O7. When `STYLE_GUIDE.md` §1.10 and §1.11 change

**Background.**
README § Steps puts the `STYLE_GUIDE.md` changes in step S0, with the design text.
AGENTS.md says that the contributor documents describe the code as it stands, while `design/` describes the target.
§1.10 says that `[]` returns leaves only and that a tuple is a path, and both stay true until S1 merges.

**Options.**
1. S0 changes `STYLE_GUIDE.md`, as README says, and the guide is ahead of the code until S1 merges.
2. S1 changes §1.10, and G5 changes §1.11's sentence on batches of laws, each with the code it describes.

**Recommendation.** Option 2, so the guide never describes code that does not exist.

### O8. Joint labels and notation keep `·`

**Background.**
Today a joint's label joins its operands' labels with `·`, as `lik·prior`, and its notation joins their notations, as `lik(y | mu)·prior(mu)` (IV.2).
The `·` mirrors the operator `*` and also reads as the product of densities.

**Options.**
1. Keep `·`, which states the product of densities whatever the operator is written as.
2. Join with `&`, as `lik & prior`.

**Recommendation.** Option 1. The label is mathematics for a reader, and `p(y | mu)·p(mu)` is the density of the joint.

### O9. G1's order relative to E5

**Background.**
Gaps F1 orders the steps that share modules, but does not place G1 against E4 and E5.
G1 changes how a lift declares its result in `functions/_broadcast.py`, and E5 puts `map_points` into lifting in the same module.
G1 can start once S2 and S4 merge; E5 can start once E4, S2, and S3 merge.

**Options.**
1. G1 after E5.
2. G1 before E5, and E5 rebases onto it.

**Recommendation.** Option 1. G1's change to the lift is small, E5's is large, and the lifting part of G1 is also the part cut third if time runs short.

### O10. `with_level_names` and `with_dim_names`

**Background.**
`with_path_names` becomes `rename`, which breaks the `with_` pattern of `with_label`, `with_level_names`, and `with_dim_names`.
The addressing plan noted that a separate change could rename the last two `rename_levels` and `rename_dims`.

**Options.**
1. Leave them unchanged, and record the idea in `deferred.md`.
2. Rename them in S3, with `rename`.

**Recommendation.** Option 1. No decision covers them, and S3 is already the widest PR of the push.

### O11. Step assignments that README § Steps does not name

**Background.**
README § Steps lists each step's content in one line, and three changes of this plan are not in any line:
1. `select` flattens to last segments (§ 12.5): this plan puts it in S1, with the reader.
2. A whole term's name stops being a subscript, as `Normal("x", 0.0, 1.0)["x"]` (§ 9.5): this plan puts it in G3, with the `component` accessor that replaces it, and `project` keeps today's reading until then (§ 7.5).
3. `in` on laws (§ 11): this plan puts it in S1.

**Recommendation.** Accept the three assignments, and add them to README's rows for S1 and G3.

## 22. Appendix: what moved from each source

### 22.1 From #571's body and jhuggins's reply

| Topic | #571's body | jhuggins's reply | This plan |
| --- | --- | --- | --- |
| Whether a declaration is whole | a property `exposes` with four values | a boolean `exposes_parts` | no property: a declaration is whole if and only if its term spec is a base spec (§ 5.2) |
| What the keyword form may name | any spec | any spec | a base spec only; a structured output is exposed or wrapped (§ 5.2) |
| `OutputSpec`'s interface | `components`, `exposes`, `term_spec` | a `Mapping`; `components`, `spec`, `exposes_record` removed | a `Mapping`, with `term_spec` (§ 7.3, § 7.4) |
| `project` with several paths | each node under its last segment, in the order given; a shared last segment raises | as the body | each node at its own path, in the tree's order; a path inside another raises (§ 7.5) |
| `project` with one path to a group | the group, whole under its last segment | as the body | the group's parts, exposed; `project(["g"])` keeps the name (§ 7.5) |
| The paths of a whole declaration | the component, then the paths inside its term, as `c/beta` | as the body | only the root (§ 4.4) |
| `without` on a whole declaration | removes nodes inside its term | as the body | nothing to remove (§ 7.6) |
| Reading a component | five cases by form | a private `_component` per kind | one rule, the public `t.component(c)` (§ 9.7) |
| The spec of a component of an exposed law | one draw's value at `c` | the view, without restating the spec | the view's spec, a `DistributionSpec` (§ 4.4) |
| `batch["mu"]` for draws of `Normal("mu", ...)` | returns the batch | as the body | raises; `draws.component("mu")` returns the batch (§ 15.2) |
| A batch built from raw values | a bare element spec, no components | as the body | a declaration; a base element is whole under `element` (§ 15.2) |
| Equality of batches that differ only in names | unequal | as the body | equal; names out of equality, hashing, the pytree, and fingerprints (§ 15.3) |
| Merging | `OutputSpec.merge`, `InputSpec.merge` | `&` replaces `*` and both `merge` methods | `&` (§ 6.4, § 7.7, § 14) |
| `&` on an operand that exposes a law or a batch | not covered | raises | raises; such outputs are wrapped to merge (§ 7.7) |
| The column of a group, `rb["g"]` | not covered | as the body | stores the exposed record `{c, e}` (§ 15.2) |
| A function's output | cannot be exposed | exposed through `FunctionSpec` | exposed, in G1 (§ 7.2) |
| `with_spec` | open question | `with_term_spec` | `with_term_spec` (§ 7.4) |
| Aliases | open question | none | none (§ 18.4) |
| The tutorials | use the new operations | — | out of scope; the User Guide is updated (§ 18.3) |
| `InputSpec.with_path_names` | — | proposed | `InputSpec.rename`, after the workshop (§ 6.6) |
| `&` on functions, batches, and specs | — | proposed | design, after the workshop (§ 14.4) |
| IV.2's rationale | — | — | associativity holds with or without `G_B ∩ F_A = ∅`; the rule keeps the joint well formed (§ 14.5) |

### 22.2 From the grammar-B addressing plan

| Topic | Addressing plan | This plan |
| --- | --- | --- |
| Status of grammar B | adopted, with jhuggins's ruling between B and B′ open | B, decided 2026-10-17 (§ 5.2) |
| Who builds the record of a wrapped output | open; leaning to the body | the body returns the record, decided (§ 5.3) |
| A type hole filled with a spec with parts | open; leaning to raise | raises, in G2; a batch of base elements is O4 (§ 5.5) |
| A view at or below a law- or function-valued field | open: wrap, or narrow #571's event rule | wraps, keeping the field's name, decided; the name below the field is O3 (§ 5.3) |
| The component accessor | `t.component(name)`, with the name a placeholder | `component`, decided (§ 9.7) |
| `rename` and `MarginalView` | recommended | decided, in S3 (§ 12.6, § 9.5) |
| Batch names out of equality and the pytree | recommended | decided; the storage is O5 (§ 15.3) |
| Clean break | open | decided (§ 18.4) |
| When kernels get `[]` | open | step G5 (§ 9.6) |
| Its scope | addressing only; #571 owned the reading of slots, `merge`, `term_spec`, and the vocabulary | both, in one text (§ 1) |
| The implementation sequence | six steps, tentative | S0–S4 before the workshop, G1 before, G2–G5 after (§ 19) |
| The workshop split | not covered | G1 before, G2 after, with the refusal of the old form last (§ 5.8) |
| A bare `output_spec` | not covered | read by the defaults by kind (§ 5.6, gaps T4) |
| IV.2's rationale | a note for #571 | the fix of § 14.5 (gaps T5) |
| "Selection" in the glossary | both senses removed | only *selection order* stays (gaps T3) |
| `mean(prior)["mu"]` in "The problem today" | `KeyError` | works since #576 merged; the table shows `mean(grouped)["g"]` instead (§ 5.1) |
| "Where this plan meets #571" | twelve shared points, each with an owner | gone: one text owns both (§ 22.1 records the outcomes) |
