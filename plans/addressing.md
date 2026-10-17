# ProbPipe addressing plan, grammar B

> **Status: imported source, not yet reconciled.**
> This file is the Claude Doc "ProbPipe addressing plan, grammar B" (revision 63, 2026-10-16), exported to Markdown without edits.
> From now on this file, not the Doc, is the copy that changes.
> [`gaps-and-conflicts.md`](gaps-and-conflicts.md) lists where it disagrees with [`declarations.md`](declarations.md) (#571) and with [`execution-layer.md`](execution-layer.md).
> Once those items are settled, this file and `declarations.md` are replaced by one reconciled text.

Oct 16, 2026 · @Andrew Roberts

A law in ProbPipe and every value derived from it (its draws, batches of draws, summaries, views and marginals) should be indexed by the same paths, and the names a term offers to the rest of a workflow should be exactly the names inside its value. Three rules make that hold by construction:

1. **A path is a position inside a value.**
2. **Paths go through every kind**: into a record's children, a law's draws, a function's outputs, and a batch's elements.
3. **Only an array or an opaque value is ever named as a whole.** Anything with parts offers its parts, or is wrapped in a record that holds its name.

The third rule is grammar B of jhuggins's [component interface doc](https://claude.ai/artifact/6TusK5RoZokDA1wk5sbdaq), which models specs, declarations and paths in Rocq and leans toward B while leaving the final ruling open. This plan adopts B and states what it implies for addressing; the grammar rule section explains why B rather than its close alternative B′. Names in code font such as `graft_identity_B` are lemmas in that Rocq development. This plan replaces the earlier [grammar-A addressing plan](https://claude.ai/artifact/PFHi8LYSRU1xgzGs8oYVCt) and keeps everything in it that grammar B does not change; an appendix lists the differences.

## Scope

This plan decides how ProbPipe addresses the parts of a term. It covers [GH #573](https://github.com/TARPS-group/prob-pipe/issues/573) and the parts of [GH #571](https://github.com/TARPS-group/prob-pipe/issues/571) that addressing depends on.

**In scope:**

- the definitions of base specs, specs with parts, paths, declarations and components;
- the grammar rule: which declarations may name a value whole;
- what `[]` reads and returns on every kind, lists of paths, and positions;
- membership (`in`);
- the operations that take paths: `marginal`, `condition_on`, `log_prob`, `without`, `select`, and renames;
- what a summary is indexed by, with PR 576;
- what `&` means for paths, and what a batch's stored declaration means for indexing.

**Out of scope, owned by #571:**

- the one reading of slots: the sources an `InputSpec` accepts, `given_spec`, and `input_spec`;
- `InputSpec.merge` and how shared slots unify;
- the full definition of `&` for every kind, beyond what it means for paths;
- the renames of `spec` to `term_spec`, and the removal of `components` and `exposes_record`;
- the vocabulary change from "packaging" to "exposes", and the tuple shorthand for sizes;
- \#571's rule for a law's event, and whether to narrow it.

The two plans meet in twelve places, chiefly `OutputSpec` as a mapping with `project`, what a declaration exposes, the batch's element declaration, and `&`. The section "Where this plan meets #571" lists each one and which plan owns it. PR 575 (`raw` as a property) and PR 576 (summaries keep the law's paths) land before this work; 575 removes `raw(path)`, so `raw` is not covered here. Section numbers such as II.6 refer to the design reference in `design/`.

## ProbPipe in brief

ProbPipe builds statistical workflows from laws (probability distributions), functions, the values they produce, and operations on all of them. These are the terms the plan relies on:

| Term | Meaning | Example |
| --- | --- | --- |
| **term** | Any object ProbPipe tracks: a value, a law, a kernel, a function or a batch |  |
| **law** | A probability distribution, the class `Distribution` | `Normal("x", 0.0, 1.0)` |
| **event** | What one draw of a law is: an array, or a record of named parts | the event of `Normal("x", 0.0, 1.0)` is a scalar |
| **draw** | One value sampled from a law | `sample(d)` |
| **record** | A tree of named values, the class `Record`; nested values are reached with `/` | `Record({"a": 1.0, "g": {"c": 2.0}})["g/c"]` |
| **opaque value** | A value that is neither numeric nor a record, such as a string or an arbitrary Python object | `"forest"` |
| **batch** | Many values of one kind, stacked along named axes called **levels** | `sample(d, sample_shape=100)` has the level `sample` |
| **kernel** | A conditional distribution, the class `ConditionalDistribution`: a law for each value of its inputs | `lik(y \| mu)` |
| **function** | A tracked callable, the class `Function`, with named inputs and a declared output | `Function(run_inference, output_spec=...)` |
| **joint law** | The law of several parts together. `&` composes laws and kernels into one; #571 proposes `&` in place of today's `*` | `lik & prior` |
| **view** | `d[...]`: a law over part of `d`'s event that keeps a reference to `d` (the class `FieldView`) | `prior["mu"]` |
| **marginal** | `marginal(d, ...)`: the same law as the view, but detached: it keeps no reference to `d` | `marginal(prior, "mu")` |
| **conditioning** | `condition_on(d, values)`: the law of the rest of `d` given values for some of its parts, such as a posterior | `condition_on(model, {"y": data})` |
| **summary** | A mean, variance, standard deviation or quantile of a law | `mean(d)` |
| **route** | How an operation computes its result: exactly, or approximately by Monte Carlo from draws | an exact mean of a Normal; a Monte Carlo mean of a posterior |
| **spec** | The type of a value: its shape, dtype and support, or its named structure | `NumericArraySpec(shape=())`, `RecordSpec(...)` |
| **producer, consumer** | A producer returns values under names (a law, a function, an operation); a consumer receives values under names (a function's parameters, a kernel's inputs) | `prior` produces `mu`; `lik` consumes `mu` |
| **label** | A display name for a term; no operation reads it | `mu ~ prior` |
| **JAX pytree** | How JAX sees a value: the arrays it transforms, plus static data it compares to decide whether two values have the same structure, which `jit` relies on |  |
| **factor** | One of the laws or kernels a joint law is composed from | lik and prior are the factors of model |
| **atoms** | The stored values an empirical law draws from, such as the draws a Monte Carlo posterior keeps | `EmpiricalDistribution(atoms)` |
| **fingerprint** | A content hash of a term, which caching and replay use to recognize the same computation | computed automatically |

The examples are written in the proposed API, so they use `&`. Two laws serve as running examples throughout:

```python
from probpipe import Normal, sample, mean, marginal, condition_on, log_prob

# A law over two scalars. Its draws are records {mu, tau}.
prior = Normal("mu", 0.0, 5.0) & Normal("tau", 1.0, 1.0)

# A law with a nested event. Its draws are records {a, g: {c, e}}.
grouped = (Normal("a", 0.0, 1.0) & Normal("c", 0.0, 1.0) & Normal("e", 0.0, 1.0)) \
    .with_path_names({"c": "g/c", "e": "g/e"})
```

A few sections also use terms whose construction does not matter to the point being made:

- `lik`: a kernel `lik(y | mu)`, the law of `y` for each value of `mu`.
- `model = lik & prior`: a joint law whose draws are `{y, mu, tau}`; `data` is an observed value of `y`.
- `post = condition_on(model, {"y": data})`: a posterior whose draws are `{mu, tau}`; `post_spec` is its spec.
- `hier`: a hierarchical law whose draws are `{s1: {theta}, s2: {theta}, mu}`.
- `emulator`: a function from `theta` to a record `{mean, var}`, returned by a function `fit_emulator`; `theta` is a value of its input.
- `lik_g`: a kernel `lik_g(y | g)` whose slot `g` takes a record `{c, e}`.
- `rm`: a random measure, a law whose draws are records `{G: law}`; `gp`: a Gaussian process, whose draws are records `{f: function}`.
- `run_inference`: a plain Python function that returns a posterior law; `sample_params` returns a dict `{"beta": ..., "sigma": ...}`.
- `bern`: a Bernoulli law, whose event is an `int32`.

"The tour" refers to the walkthrough notebook `review/overhaul_tour.ipynb`.

## Definitions

Two distinctions carry the plan. A spec is either a **base spec**, with no parts, or a **spec with parts**. And a **path** is a position inside a value, while a **component** is a name a declaration offers to the rest of a workflow. Grammar B makes the components and the paths coincide.

### Base specs and specs with parts

A **base spec** types a value with no named parts: a `NumericArraySpec` (an array) or an `OpaqueSpec` (an opaque value). A value of a base spec is a **base term**.

A **spec with parts** types a value whose parts can be named. There are five kinds, and every one except a record holds the declaration of what it produces:

| Kind | Spec | Its parts are | Example |
| --- | --- | --- | --- |
| record | `RecordSpec` with at least one child | its children | a draw of `grouped`, `{a, g: {c, e}}` |
| law | `DistributionSpec`, holding the declaration of its event | the parts of one draw | `grouped` itself |
| kernel | `ConditionalDistributionSpec`, holding its slots and its event's declaration | the parts of the law it yields at each given value | `lik` |
| function | `FunctionSpec`, holding its slots and its output's declaration | the parts of what it returns | `emulator` |
| batch | `BatchSpec`, holding its element's declaration and its levels | the parts of one element | `sample(grouped, sample_shape=100)` |

A term whose spec has parts is a **term with parts**. A law is always a term with parts, even `Normal("x", 0.0, 1.0)`, whose event is a single array: the law has no paths below its root, and its one component `x` sits at the root.

### Trees and paths

| Term | Meaning | In a draw of `grouped`, `{a, g: {c, e}}` | Change from today |
| --- | --- | --- | --- |
| **name** | A non-empty string without `/`. Components, fields, groups, slots and levels all have names. | `a`, `g`, `c`, `e` | none |
| **path** | The `/`-joined names from the top of a term down to a node. It is always a `str`. | `"a"`, `"g"`, `"g/c"` | a tuple is no longer a path |
| **node** | A position a path reaches. The empty path, the term itself, is the **root**. | `a`, `g`, `g/c`, `g/e` | new glossary entry |
| **field** | A node of a record, reached through records only, that is not itself a record. A field may hold a law, a function or a batch, whose own paths continue below it. `keys()` lists a record's fields. | `a`, `g/c`, `g/e` | none |
| **group** | A node of a record that is itself a record. | `g` | one word replaces "interior node", "subtree", "field group" and "group of fields" |

**Paths go through every kind.** A path continues past a field that holds a term with parts, into that term's own paths:

- a law's paths are the paths of its draws;
- a kernel's are those of the law it yields;
- a function's are those of what it returns;
- a batch's are those of one element.

So the record `r = Record({"n": 3.0, "G": post})`, whose field `G` holds the posterior over `{mu, tau}`, has the paths `n`, `G`, `G/mu` and `G/tau`, and `r["G/mu"]` is `r["G"]["mu"]`. The fields of `r`, which `keys()` lists, are still only its own: `n` and `G`. **Change from today:** paths stop at a field, and a law, function or batch held in a field is reached only by a second `[]`.

A **leaf path** is a path with no path below it. In a draw of `grouped` the leaf paths are the fields `a`, `g/c` and `g/e`. In `r` they are `n`, `G/mu` and `G/tau`.

### Declarations and components

| Term | Meaning | Example | Change from today |
| --- | --- | --- | --- |
| **declaration** | An `OutputSpec`: the spec of what a producer returns, and the names it offers it under. A law's declaration is its `event_spec`; a function's is its `output_spec`. | `grouped.event_spec` | none |
| **whole declaration** | Offers its whole value under one name, written with a keyword. Under grammar B it may name only a base spec. | `OutputSpec(x=NumericArraySpec(()))`, the event of `Normal("x", 0.0, 1.0)` | today any spec may be named whole |
| **exposed declaration** | Offers the parts of a spec with parts, written positionally. An exposed law, kernel, function or batch offers the components of the declaration it holds. | `OutputSpec(RecordSpec(a=..., g=...))`, the event of `grouped` | functions and batches cannot be exposed today |
| **component** | A name a declaration offers. A whole declaration has one, its name. An exposed record has one per child, which may be a group. An exposed law, kernel, function or batch has the components of the declaration it holds. | `grouped`'s components are `a` and `g` | today II.2 and the glossary say "an exposed record's components are its fields", which is wrong for `g` |
| **address** | Where a component sits in the value, defined by the form: the root for a whole declaration; the child's path for an exposed record; and for an exposed law, kernel, function or batch, the address the component has in the declaration it holds. | `g` sits at `"g"`; in `sample(Normal("x", 0.0, 1.0), sample_shape=4)`, whose element is whole `x`, the component `x` sits at the root | new term |
| **component spec** | The spec of the output at the component's address. When the declared value is itself a law, as for a function that returns one, the output at an address is the view there, so the component's spec is a law's spec, not an array spec. A law's own event declaration is unaffected: its value is a draw. | the component `mu` of a function returning `post` has the spec of `post["mu"]` | today a law's exposed components are typed by one draw's value (#571 says the same) |
| **wrap** | The one-field exposed record `OutputSpec(RecordSpec(name=spec))`. Its value is the record `{name: value}`, so the name is stored in the value. Under grammar B this is how a structured output gets one name. | `OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=())))` | new term |
| **slot** | A name under which a consumer receives a value. Composition meets a component with the slot of the same name. | `lik(y \| mu)` has the slot `mu` | new glossary entry (#571) |

Under grammar B, a component's address is the path of its own name, except where the name belongs to a whole base value, whose address is the root. jhuggins's model proves the related fact that composition places every path of an operand at the same path in the joint, apart from the root (`graft_identity_B`). Today a whole term over structure breaks both: its name sits outside its value, which is the root of most of the problems below.

### Subscripts

| Term | Meaning | Example | Change from today |
| --- | --- | --- | --- |
| **list of paths** | Several paths in a Python list or a tuple, to pick several nodes at once. | `["a", "g/c"]` | today a tuple of paths is accepted on laws only and is one nested path elsewhere; a Python list is refused everywhere |
| **position** | An integer, a slice, or a tuple of them, indexing a batch's axes. | `draws[3]`, `draws[0:10]` | II.5's view-identity and `at_levels` prose also say "selection", which goes |
| **subscript** | What is written inside `[]` (Python's own term): a path, a list of paths, or, on a batch, a position. | `grouped["g/c"]`, `grouped["a", "g/c"]` | replaces "key" |

The public API gets no new noun for a list of paths: docstrings say "a path, or a list of paths". The word "selection" is avoided because `select` is already a method with another meaning. The word **key** is not used for subscripts, since ProbPipe already uses it for random-number keys; it keeps only its Python sense of a mapping's keys, as in `keys()` or the keys of a rename mapping. Today the glossary also uses it for a field's path.

## The problem today

Today a law and its own values often disagree on what a subscript means, and a term can offer names that its value does not hold. Every result below runs on `dev/overhaul` (638f56bc), where `*` composes laws, since `&` does not exist yet.

| Expression | Result today | The related expression | Result today |
| --- | --- | --- | --- |
| `prior["mu", "tau"]` | joint view of `mu` and `tau` | `sample(prior)["mu", "tau"]` | `KeyError`: read as the one path `mu/tau` |
| `grouped["g"]` | view of the group `g` | `sample(grouped)["g"]` | `KeyError`: "'g' is a subtree, use at\_path()" |
| `sample(grouped, sample_shape=4)["g"]` | the batch of the group's values | `sample(grouped)["g"]` | `KeyError` (as above) |
| `grouped["g"]["g/c"]` | view of `c` | `grouped["g"]["c"]` | `KeyError` |
| `sample(grouped["g"])["c"]` | works | `sample(grouped["g"])["g/c"]` | `KeyError` |
| `prior["mu"]` | view of `mu` | `mean(prior)["mu"]` | `KeyError`: the field is named `mean(mu)` |
| `Normal("x", 0.0, 1.0)["x"]` | the law itself | `sample(Normal("x", 0.0, 1.0))["x"]` | `TypeError`: JAX does not support string indexing |

Four causes account for almost every row:

1. **A tuple means two things.** On a law, `["mu", "tau"]` picks two paths. On a record or a batch, the same tuple is one nested path, `mu/tau`.
2. **`Record[...]` stops at fields.** A law and a batch return a group for a path such as `"g"`; a record raises.
3. **A name can sit outside the value it names.** The view `grouped["g"]` is declared as a whole term named `g`, so it is indexed by `g/c`, while its draws are the record `{c, e}`, indexed by `c`. The same holds for any whole declaration over structure, such as a law declared `OutputSpec(params=RecordSpec(...))`: the law is indexed by `params/beta`, its draws by `beta`.
4. **Summaries rename their fields** to `mean(mu)` and so on. Open PR 576 already fixes this.

Grammar B removes cause 3 at its source: a name may sit outside its value only when the value is a base term, which has no paths for it to disagree with.

## Core invariants

The design must guarantee thirteen properties. In the table, `d` is a law, `t` is any term, `f` is a function, `s` is a subscript (a path or a list of paths), `p` and `q` are paths, and `ps` is a list of paths that is not empty, has no path inside another, and leaves something out. The exceptions section lists the cases where one deliberately does not hold.

jhuggins's Rocq model proves I1 for laws, draws and batches, and I2, I4, I5 and I10 for every kind of term, and the fact behind I12 that composition keeps every operand path except the root (`graft_identity_B`); some of its results about values hold in a model where laws are point masses. I9 and I12 are rules this plan chooses, and the others follow from its definitions.

| # | Invariant | In code |
| --- | --- | --- |
| I1 | **One set of paths.** A subscript valid on a law is valid on every value derived from it, and on its declaration. | `s` works on `d`, `sample(d)`, each element of `sample(d, sample_shape=n)`, `mean(d)` and `d.event_spec.project(s)` |
| I2 | **Indexing commutes with sampling.** A draw of a view is a draw of its parent, indexed. | `sample(d[s])` is distributed as `sample(d)[s]` |
| I3 | **Indexing commutes with summaries.** | `mean(d)[s]` and `mean(d[s])` estimate the same quantity, and are equal when both are computed exactly; likewise `variance`, `standard_deviation`, `quantile` |
| I4 | **Positions and paths commute on a batch.** | `draws[i][s] == draws[s][i]` |
| I5 | **Paths chain**, across kinds. | `t[p][q] == t[p + "/" + q]`, including through a field that holds a law, a function or a batch |
| I6 | **Declarations agree.** | `d[s].event_spec == d.event_spec.project(s) == marginal(d, s).event_spec` |
| I7 | **A view and a marginal are one law.** | `marginal(d, s)` equals `d[s]`; one is detached, the other refers to `d` |
| I8 | **Draws of a list of paths condition back.** | `condition_on(d, sample(d[ps]))` fixes exactly the nodes in `ps` |
| I9 | **The form of the subscript decides the result.** | `t["g"]` is the node; `t[["g"]]` is a record holding it, as pandas `df["a"]` and `df[["a"]]` are |
| I10 | **A list of paths and `without` partition the leaf paths.** | Every leaf path of `t` is a leaf path of exactly one of `t[ps]` and `t.without(*ps)` |
| I11 | **Conditioning removes what it fixes.** For a law, the fixed paths leave its event; for a kernel, bound slots, and the bound parts of a record slot, leave its `given_spec`. | `condition_on(d, given).event_spec == d.without(*fixed).event_spec`, where `fixed` are the event paths `given` names, whichever route computes the result |
| I12 | **A component is the term at its address.** | `t.component(c)` is `t[address(c)]`, and `t` itself when the address is the root: `grouped.component("g") == grouped["g"]`, and `sample(Normal("x", 0.0, 1.0), sample_shape=4).component("x")` is the batch |
| I13 | **Indexing commutes with producing**, for every producer. | `f[s](x) == f(x)[s]` for a function, and `lik[s](v)` is `lik(v)[s]` for a kernel once kernels have `[]`; I2 and I4 are the same rule for laws and batches |

**What "equal" means.** For values, `==` above means equal contents: the same spec and equal entries, compared with `jnp.array_equal` for arrays and field by field for records and batches. For laws and functions, it means the same term: equal declarations and the same distribution or the same map. Tests compare declarations exactly, and distributions through exact summaries where they exist.

With the running examples, all of the following hold under this plan:

```python
draw = sample(grouped)

grouped["g"]["c"]                         # I5: the same law as grouped["g/c"]
sample(grouped["g"])                      # I2: a record {c, e}, like draw["g"]
draw["g"]["c"] == draw["g/c"]             # I5 on a value
mean(grouped)["g/c"]                      # I1, I3: PR 576 makes this work
grouped.event_spec.project(["a", "g/c"]) == grouped["a", "g/c"].event_spec   # I6
grouped.component("g") == grouped["g"]    # I12: the component g sits at the path g

draws = sample(grouped, sample_shape=100)
draws[3]["g/c"] == draws["g/c"][3]       # I4

emulator["mean"](theta) == emulator(theta)["mean"]   # I13
```

## The grammar rule

**A whole declaration may name only a base spec.** Anything with parts is offered by its parts, or wrapped in a one-field record that stores its name in the value. This is grammar B.

```python
OutputSpec(x=NumericArraySpec(()))                     # whole, over a base spec: allowed
OutputSpec(RecordSpec(beta=(), sigma=()))              # exposed: components beta and sigma; a value is {beta, sigma}
OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=())))
                                                       # wrapped: one component params; a value is {params: {beta, sigma}}
OutputSpec(params=RecordSpec(beta=(), sigma=()))       # TypeError: params would name a record. Expose its fields,
                                                       # or wrap them under a field named params
```

**Why.** When a name sits outside a structured value, the term has two sets of paths. Its value is indexed from inside, by `beta`, while composition and renames see it from outside, by `params/beta`. Grammar B keeps every name inside the value, or names a value that has no paths at all, so there is one set of paths:

- subscripts, rename keys and draws agree (I12);
- no rename key can be ambiguous;
- the spec alone fixes whether a declaration is whole or exposed, so a declaration needs no flag to record it.

jhuggins's model proves each of these (`graft_identity_B`, `no_ambiguous_key_B`, `B_decl_determined`).

**Views follow the same rule.** `d[p]` names a base node whole, by the last segment of `p`, and exposes a node with parts. `d[[p]]` is the wrap, which keeps the name. See "Laws and views" below.

**Why B and not B′.** jhuggins's doc compares B with a close alternative, B′, which would also name whole any spec with no paths below its root: a batch of arrays, a law over a single array, or a random function whose output is one array. B′ keeps a few more names, such as a Gaussian process's own name `f`. It pays with a flag on every declaration, since the spec no longer fixes whether it is whole or exposed. This plan takes B for its single rule: a declaration is whole exactly when its spec is a base spec. The choice between them is still open in jhuggins's doc.

### Structured outputs under grammar B

The rule changes how a few kinds of output are written. Each has two forms: expose the parts, or wrap the value to keep one name.

**A law over a record, named as a whole.** Today `params_prior` is declared `OutputSpec(params=RecordSpec(beta=(), sigma=()))` and draws `{beta, sigma}`. Under B:

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

A kernel with the slot `params` meets the wrapped law, since that law offers `params`; it does not meet the exposed one, which offers `beta` and `sigma`.

**A function that returns a law under one name.** Today `Function(run_inference, output_spec=OutputSpec(posterior=post_spec))` returns the law under the component `posterior`. Under B a law has parts, so:

```python
# exposed: the function returns the law, and offers its event's components
fit = Function(run_inference, output_spec=OutputSpec(post_spec))
fitted = fit(data)                  # the law; its components mu and tau are the views fitted["mu"], fitted["tau"]

# wrapped: the function returns a record holding the law
def run_inference_wrapped(data):
    return {"posterior": run_inference(data)}
fit = Function(run_inference_wrapped, output_spec=OutputSpec(RecordSpec(posterior=post_spec)))
fit(data)["posterior"]              # the law; a consumer with the slot posterior receives it
```

**Exposing a law offers marginals.** The exposed form offers `mu` and `tau` separately, each typed by its view's spec. A consumer that meets them by name receives two marginal laws and loses the dependence between them. To pass the joint posterior on as one thing, wrap it.

**Open question: who builds the record.** In the wrapped form, either the function body returns `{"posterior": law}` and the declaration only checks it, or the function returns the law and ProbPipe wraps it under the declared name. The lean is the first, which converts nothing behind the caller's back.

**A function that returns a function.** `fit_emulator` returns `emulator`, a function from `theta` to `{mean, var}`. Its declaration exposes the returned function's outputs, so its components are `mean` and `var`. By I12, `fit_emulator.component("mean")` is `fit_emulator["mean"]`, the function that returns `emulator["mean"]`; and `emulator["mean"]` is itself the function from `theta` to the mean (I13). Naming the returned function whole, as `OutputSpec(emulator=FunctionSpec(...))`, raises; wrap it to keep the name.

**A law whose draws are laws, functions or batches.** Here #571's rule for a law's event takes precedence: a law's event is a whole term or an exposed record, so it can never expose a law, a function or a batch, and under B it cannot name one whole either. The draws are therefore wrapped:

- A random measure's draws are laws, stored in a record: `{G: law}`.
- A random function's draws, such as a Gaussian process's, are functions, stored in a record: `{f: function}`. The law keeps offering `f`, and each draw gains a level.
- A bootstrap replicate law's draws are batches, stored in a record: `{boot: batch}`.
- Lifting over a law a function that returns a law, a function or a batch gives a law whose draws are of that kind, so its event wraps them under the function's name.

If #571 narrows its event rule to exposed laws and kernels with two or more components, as jhuggins's open question 4 suggests, a random function could expose its output instead, and its draws would stay functions.

**A batch.** A `BatchSpec` has parts, its element's, so a batch is always declared exposed, through its element's declaration. This requires every batch to hold an element declaration. A batch an operation produces stores its producer's, as #571 proposes. A batch built from raw values, which #571 leaves with a bare element spec, gets one: a record element is exposed, and a base element is whole under the default name `element`. So `sample(Normal("x", 0.0, 1.0), sample_shape=4)` offers `x` as today, and `NumericArrayBatch(values, "draw")` offers `element`. A sweep or function that returns a batch exposes the element's components, or wraps the batch to name it.

### Defaults and pending types

Two places build whole declarations without a caller writing one, and grammar B gives each a rule:

- **Defaults.** Today, in sweeps and lifts, a producer whose output is not a record is declared whole under a default name: the function's name, or the law's one component. `OutputSpec.default` builds these, from 7 call sites in 6 files. Under B the default depends on what is produced:
  - a base term stays whole under the default name, as today;
  - a batch returned by a function or a sweep is exposed through its element's declaration, with the producer's default name replacing the element's whole name, so a function `f` returning a batch of arrays still offers `f`;
  - a law or a function returned by a function is exposed, which keeps the value and offers its components; the marginals caveat above applies;
  - a law, a function or a batch that is a law's event is wrapped under the default name, as #571's event rule requires, so each draw gains a level.
- **Pending types.** `OutputSpec(x=None)` names a value whose type is filled in later. Filled with a base spec, it stays whole. Filled with a batch whose element is a base spec, it becomes the exposed batch whose element is whole `x`: the declared name replaces the element's, so it offers `x`. Filled with any other spec with parts, it either raises or wraps the value under `x`. This is the same question as who builds the record, with the same lean: raise, and ask the caller to declare the record.

In both cases the producer's declared or default name wins over a name stored in the element; a batch with no declaring producer keeps its stored element declaration.

### What grammar B costs

A sweep of `dev/overhaul` (638f56bc) counted every whole declaration over a spec with parts, by running the full test suite and every notebook with a probe on `OutputSpec`. The changes fall into two classes.

**Form only, with the same names and values.** A batch has parts under B, so `sample(Normal("x", 0.0, 1.0), sample_shape=n)` and `quantile(...)`, which today declare a batch of arrays whole, become exposed batches. The element's declaration offers `x`, so the batch still offers `x`, and its values do not change. This class covers most of the 367 tests and 39 notebook cells that the sweep counted as batches of arrays; the bootstrap replicate law's tests and cells, such as user guide 08 cell 38 and tour cell 191, belong to the visible class below.

**Visible changes.** These change what a term offers or what its draws look like:

| Change | Where |
| --- | --- |
| A view or marginal of a group offers its parts instead of the group's name | `_view_declaration`, `marginal` at a group; `tests/distributions/test_views.py` (79 tests), the empirical and path-rename marginal tests |
| A factor that holds a group (IV.1's packaged sub-joint, a "grouped sub-joint" in #571's words) is wrapped, so its draws gain the group's level: `factor(grouped, "g")` draws `{g: {c, e}}`. `marginal(grouped, "g")` must then be that factor's view at `g`, not the factor itself | `_factored.py`, `_regrouped`, `test_regrouping_renames.py` (25 tests), the tour's hierarchical example |
| A law declared whole over a record, such as `params_prior`, is exposed or wrapped | 233 tests construct one, including `tests/core/test_specs.py:198` and `tests/distributions/test_views.py:100` |
| A law whose draws are laws (random measures, bootstrap laws, minibatched targets) stores them in a record | `_random_functions.py`, `_resampling.py`, `inference/_minibatch.py`; 181 tests name a law whole |
| A random function's draws are wrapped, `{f: function}` | `_gaussian.py`, `_random_functions.py`; 260 tests name a function whole; `test_gaussian.py` (109) and `test_event_declarations.py` (104) carry most of it |
| Lifting a law-returning function over a law wraps its draws under the function's name | `functions/_broadcast.py` (V.5), including the tour's `predictive` example |
| A function returning a law, in a sweep, offers the law's components instead of its own name | `OutputSpec.default` and its callers |
| A bootstrap replicate law, whose event is a batch, wraps its draws: `{boot: batch}` | `_resampling.py`; user guide 08 and the tour |

**Totals.** In all, 1,046 tests in 72 files execute a whole declaration over a spec with parts, and 679 tests in 46 files remain once the form-only batches are set aside. The counts in the table overlap, since one test can build several kinds. The library has about 31 sites, most of them through `OutputSpec.default` and the filling of pending types. No notebook writes such a declaration explicitly: 6 cells reach one through defaults, and 39 more only through the form-only batches. In `design/`, about 22 passages state or show the whole form over parts: II.2, III.3, III.7, IV.1, V.6, VI.8, VII.2, VII.5, VII.6 and the glossary.

**Prerequisites from #571.** Today a `FunctionSpec` cannot be exposed at all, and a `BatchSpec` only when its elements are records or laws: `OutputSpec(BatchSpec(NumericArraySpec(()), sample=3))` raises `TypeError`. Exposing a function's output, and a batch of arrays through its element, both depend on #571's exposure rules and on every batch holding an element declaration.

## Reading a subscript

Every term that is indexed by path reads a subscript the same way: a `str` is one path, a list or tuple of `str` is a list of paths, and integers and slices are positions (batches only). A base term has no paths: a `NumericArray` keeps its own `[]`, which indexes its entries, and an opaque value has no `[]`. One reader in `probpipe/core/_shapes.py`, beside `_as_names`, does all the parsing and owns every error message. The separator `_PATH_SEP` is defined once, in `named_tree.py`; today eight modules redefine it.

```python
t["g/c"]             # a path: the node g/c
t["a", "g/c"]        # a list of paths (Python builds the tuple ("a", "g/c"))
t[["a", "g/c"]]      # the same, written as a Python list
t[("a",)]            # a list of one path: a record holding a
draws[3]             # a position (batches only)
draws[0:10]          # positions 0 to 9
chains[1, 0:10]      # positions on two axes, for a batch with levels chain and draw
```

The reader accepts these types and nothing else:

- **A path** is a `str`. A subclass such as `numpy.str_` is converted with `str()`.
- **A position** is an `int`, anything with `__index__` (NumPy and JAX integer scalars included), a `slice`, or a tuple of those.
- `bool`, `None` and `...` raise `TypeError`. `None` keeps its one meaning, a whole axis in `at_levels`.

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

Only three places accept a list of paths: `[]`, `marginal`, and `project` on a declaration. Every other argument takes one path as a `str`, and the same reader checks it, so `without("a", "a")` and `without("g", "g/c")` raise as `[]` does:

```python
rec.at_path("g", "c")         # separate segments, joined with "/" (unchanged)
rec.at_path(("g", "c"))       # TypeError: unpack the tuple, at_path(*segments)
rec.without("a", "g/e")       # removes several paths, one per argument (unchanged)
rec.without(("a", "g/e"))     # TypeError: pass the paths as separate arguments
condition_on(grouped, {"g/c": 0.5})    # a mapping from paths to values
```

A declaration is the one thing indexed by name rather than by path: `decl["g"]` is the spec of the component `g`, as #571's reply makes `OutputSpec` a mapping of its components. For an exposed record, a component's address is the path of its name, so `decl["g"]` is the spec at the path `g` of the value. For a whole declaration the address is the root, so `decl["x"]` is the base spec of the whole value.

**Change from today:** a tuple was one nested path on `Record`, `RecordSpec`, `RecordBatch` and `Batch`, and a list of paths only on laws. A Python list was refused everywhere.

## Indexing each kind

Every term with parts resolves a path against its own paths and returns the node in its own kind: a record gives a record or a value, a law gives a law, a function gives a function, and a batch gives a batch. Paths reach groups everywhere, `Record` included, and continue into a field that holds a law, a function or a batch.

### Records

```python
rec = sample(grouped)             # Record {a, g: {c, e}}

rec["a"]            # NumericArray, labeled "a"                    (unchanged)
rec["g/c"]          # NumericArray, labeled "g/c"                  (unchanged)
rec["g"]            # Record {c, e}, a view labeled "g"            (today: KeyError)
rec["g"]["c"]       # the same as rec["g/c"]
rec["a", "g/c"]     # Record {a, g: {c}}, labeled "(a, g/c)"      (today: KeyError)
rec.keys()          # ("a", "g/c", "g/e"): fields only            (unchanged)
```

`rec.at_path("g")` stays as a synonym of `rec["g"]`, and `at_path("g", "c")` still takes separate segments.

### A field that holds a law, a function or a batch

A path continues into the term a field holds:

```python
r = Record({"n": 3.0, "G": post})    # post's draws are {mu, tau}

r["G"]              # post                                         (unchanged)
r["G/mu"]           # the view post["mu"]                          (today: KeyError)
r["G"]["mu"]        # the same view (I5)
```

### Record specs and declarations

A `RecordSpec` indexes as a record does, returning specs. A declaration is indexed by component name, and `project` takes a subscript:

```python
spec = rec.spec                       # RecordSpec {a, g: {c, e}}
spec["g"]                             # RecordSpec {c, e}                 (today: KeyError)

decl = grouped.event_spec             # exposes {a, g: {c, e}}
decl["g"]                             # RecordSpec {c, e}: the component g's spec
decl.project("g")                     # the declaration of grouped["g"]: exposes c and e
decl.project(["a", "g/c"])            # exposes {a, g: {c}}
```

### Batches

A batch indexes each of its elements by path. Positions index the batch's own axes, as today.

```python
draws = sample(grouped, sample_shape=100)   # RecordBatch on the level "sample"

draws["a"]           # NumericArrayBatch                   (unchanged)
draws["g"]           # RecordBatch {c, e}                  (unchanged)
draws["a", "g/c"]    # RecordBatch {a, g: {c}}             (today: KeyError, read as the path a/g/c)
draws[3]             # Record {a, g: {c, e}}               (unchanged)
draws[3]["g"] == draws["g"][3]                            # I4 (today: the left side raises)
```

A batch of laws indexes each law: `laws["mu"]` is the batch of the views `law["mu"]`. Today it raises `TypeError` with the false message that the elements have no named fields. A batch of functions does the same with each function.

### Laws and views

`d[p]` returns a `FieldView`: a law over the node at `p` that refers back to `d`. Its declaration follows the grammar rule. A base node is named whole by the last segment of `p`; a node with parts is exposed, so the view offers its parts. To keep a group's name, index with a list of one path:

```python
grouped["a"]            # a whole term named a; its draws are arrays
grouped["g/c"]          # a whole term named c
grouped["g"]            # exposes c and e; its draws are records {c, e}        (today: a whole term g)
grouped[["g"]]          # exposes the record {g: {c, e}}: one component, g
grouped["g"]["c"]       # the view of g/c                                      (today: KeyError)
grouped["a", "g/c"]     # exposes {a, g: {c}}: components a and g              (today: {a, c})
marginal(grouped, ["a", "g/c"])   # the same law, detached; a list works       (today: unhashable list)
```

This is I9 reading as pandas does. `grouped["g"]` is the node, the group's contents; `grouped[["g"]]` is a record holding it, under its name.

**Open question: rename `FieldView` to `MarginalView`.** The class covers groups and lists of paths, not only fields, so "field" undersells it. By I7 it is the marginal law at its subscript, kept as a view rather than detached. `MarginalView` says that, and pairs with `marginal`. The rename touches the class, its repr, III.7 and III.8, and the user guide; `d[s]` is unchanged.

### Functions

A function is indexed by the paths of what it returns, and indexing comes after the call (I13). A function with a declared output checks the subscript when it is indexed. A function without one, whose `output_spec` is `None`, checks it when the indexed function is called, against what the call returns.

```python
emulator["mean"]             # the function theta -> emulator(theta)["mean"]   (today: TypeError, no [])
emulator["mean"](theta) == emulator(theta)["mean"]
emulator[["mean"]]           # the function theta -> Record {mean}
```

### Kernels

A kernel is indexed the same way, in a later PR: `lik[s]` is the kernel that maps a given value `v` to the law `lik(v)[s]`. Until then, the design says a kernel is the one producer without `[]`.

## Indexing with several paths

A list of paths keeps every chosen node at its own path, so the result is the original tree cut down to those nodes. The result is always an exposed record (I9), and it is the exact complement of `without` (I10): every leaf path lands in exactly one of the two.

```python
grouped["a", "g/c"]                  # a law over {a, g: {c}}
grouped.without("g/e")                # the same tree, reached by removing instead of choosing
sample(grouped)["a", "g/c"]           # Record {a, g: {c}}, labeled "(a, g/c)"

# I8: the draws condition back onto the law they came from
condition_on(grouped, sample(grouped["a", "g/c"]))   # fixes a and g/c; leaves g/e
```

In a hierarchical model, two groups often hold a parameter of the same name. Keeping paths handles this:

```python
# hier's draws: {s1: {theta}, s2: {theta}, mu}
hier["s1/theta", "s2/theta"]          # a law over {s1: {theta}, s2: {theta}}
                                      # (today: ValueError, "more than one path ends in 'theta'")
```

The form of the subscript decides the result (I9). One path gives the node; a list of one path gives a record that holds it:

```python
grouped["g/c"]          # a law of c; its draws are arrays
grouped[["g/c"]]        # a law over {g: {c}}; its draws are records
```

The chosen nodes keep the original tree's order, whatever order the paths are given in, exactly as `without` keeps the order of what it leaves. Nodes under a shared group stay under it:

```python
grouped["g/e", "a", "g/c"]      # a law over {a, g: {c, e}}: the order of grouped
grouped.without("a")            # a law over {g: {c, e}}: the same order
```

A path through a field that holds a law, a function or a batch cuts that term down too. With `r = Record({"n": 3.0, "G": post})`, `r["n", "G/mu"]` is the record `{n, G}` whose `G` is `post[["mu"]]`.

To pass chosen fields to a function as keyword arguments, use `select`, which flattens each path to its last segment: `f(**rec.select("a", "g/c"))` calls `f(a=..., c=...)`.

**Change from today:** a law indexed by several paths flattens each node to its last segment, so `grouped["a", "g/c"]` is a law over `{a, c}`, and two paths ending in the same name raise. VI.8 says a marginal's fields are "repackaged in the order of the paths"; that sentence changes to the tree's order.

## Membership

`p in t` is true exactly when `p` is a `str` and `t[p]` would return a node. That covers groups and paths through a held law, function or batch, on records, record specs, batches, laws and functions.

```python
rec = sample(grouped)
"g/c" in rec          # True                                   (unchanged)
"g" in rec            # True: rec["g"] works                   (today: False)
("a", "g/c") in rec   # False: a list of paths is not a path   (False today too, as the path a/g/c)
"G/mu" in r           # True, for r = Record({"n": 3.0, "G": post})   (today: False)

draws = sample(grouped, sample_shape=100)
"a" in draws          # True                                   (today: False, by iterating the draws)

"a" in grouped        # True                                   (today: TypeError, laws have no `in`)
```

Two kinds answer differently, because their `[]` is different:

- **A declaration** is a mapping of its components, so `"g" in grouped.event_spec` tests component names. For an exposed record these are its top-level paths. For a whole term they are not: `"x" in Normal("x", 0.0, 1.0).event_spec` is true, while `"x"` is not a path of the law.
- **A base term** keeps its own meaning: `in` on a `NumericArray` tests membership of an entry.

On every batch, `in` tests paths and never elements. So a batch of arrays, which has no paths, answers `False` for every `str`, and `1.0 in sample(Normal("x", 0.0, 1.0), sample_shape=3)` changes from testing the draws to `False`.

Two consequences are worth stating, since they can surprise:

- `"x" in sample(Normal("x", 0.0, 1.0), sample_shape=4)` is `False`, although the batch offers the component `x`: `x` is its name, not a path inside it. `draws.component("x")` reads it.
- `"g/c" in grouped` is `True`, while `"g/c" in grouped.event_spec` is `False`: the law answers paths, its declaration answers component names.

**Change from today:** a `Record` answers `in` for fields only, a batch falls back to iterating its elements, and a law raises. A Python `dict` satisfies both `p in m` exactly when `p in m.keys()`, and `p in m` exactly when `m[p]` succeeds. A `Record` cannot keep both, because `keys()` lists fields only while `[]` also returns groups; the plan keeps the second.

## Operations that take paths

Every operation reads its paths through the same reader as `[]`, so a subscript means the same thing wherever it is passed. A few arguments also accept a name that is not a path:

| Argument | Paths | Names as well |
| --- | --- | --- |
| `t[...]`, `marginal(d, ...)` | one path, or a list of paths | none |
| `without(*paths)`, `select(*paths)`, `at_path(*segments)`, `is_field(path)` | one path per argument | none |
| the keys of `with_path_names` (to be renamed) | paths of the term | a whole term's name, which renames it |
| the keys of `condition_on(d, given)` | paths of `d`'s event | for a kernel, its slot names, each optionally followed by a path inside a record slot |
| composition with `&` | none | meets each component with the slot of the same name |

Under grammar B no rename key can be read both as a name and as a path: a whole term is a base term, which has no path below its name, and every other component's name is its own path. jhuggins's model proves this for renames (`no_ambiguous_key_B`). The keys of `condition_on` mix slot names and event paths; they stay unambiguous because IV.2 keeps a name from being both an unmet slot and a component, which is a separate rule. The earlier plan needed an error for ambiguous rename keys; this one does not.

### marginal

`marginal(d, paths)` takes exactly the subscripts `d[paths]` takes and returns the same law, detached from `d` (I7). The parameter is renamed from `field` to `paths`, documented as "a path, or a list of paths", as in `without(*paths)`.

```python
marginal(grouped, "g")              # the law of the group's contents, exposing c and e
marginal(grouped, ["g"])            # the law of the record {g: {c, e}}, exposing g
marginal(grouped, ["a", "g/c"])     # a Python list works  (today: TypeError, unhashable list)
marginal(grouped, ("a", "g/c"))     # the same call, the same result
```

### condition\_on

A given value is a mapping from paths to values, or a `Record`, read by its paths. The result is declared as `d.without(*fixed)` is (I11), so it exposes a record of the remaining nodes whichever route computes it:

```python
condition_on(prior, {"mu": 0.1})    # a law over the record {tau}
                                    # (today: a whole term tau, from the exact Gaussian route;
                                    #  a Monte Carlo route returns a record instead)
post["mu"]                          # works for every posterior
```

Nested paths, records and arrays are passed as follows:

```python
condition_on(model, {"y": data})                    # unchanged
condition_on(grouped, {"g/c": 0.5})                 # a nested path  (today: "nested field 'g/c' is not supported yet")
condition_on(grouped, sample(grouped["a", "g/c"]))  # a record read by its paths (I8)
condition_on(grouped, {"g": sample(grouped["g"])})  # one group: its draw is the record {c, e}, so name it
condition_on(grouped, sample(grouped[["g"]]))       # or draw the record {g: {c, e}} instead
condition_on(model, {"mu": sample(prior["mu"])})    # one array: name it too
condition_on(lik, mean(post[["mu"]]))               # a plug-in estimate of mu (PR 576)
```

I8 covers lists of paths only. The draw of one group, `sample(grouped["g"])`, is the record `{c, e}`; passed alone, it would be read at the top level, where `c` and `e` are not paths of `grouped` (today: `ResolutionError`, unknown fields). The nested-path row states the contract. Where a route cannot yet condition at a nested path, it raises `ResolutionError` naming the route, rather than refusing the path for every law.

### log\_prob

The value is read by its paths, so a summary scores where a draw scores whenever its spec fits the event's (PR 576):

```python
log_prob(prior, sample(prior))                # unchanged
log_prob(prior, mean(prior))                  # the score at the mean
log_prob(Normal("x", 0.0, 1.0), 0.3)          # a whole array takes the array    (unchanged)
log_prob(bern, mean(bern))                    # refused: the float32 mean cannot be cast to the int32 event
```

### without

`without(*paths)` removes nodes and keeps every other node at its path. It applies to records, record specs, batches, declarations and, new, laws, where it returns a view. What remains keeps the form it had: removing from an exposed record leaves an exposed record, even when one node is left, so `without` and a list of paths stay exact complements (I10).

```python
rec.without("g/e")            # Record {a, g: {c}}
post.without("tau")           # a view of post over the record {mu}   (today: laws have no without)
```

### select

`select(*paths, **remap)` exists to pass fields to a function as keyword arguments, so it flattens each path to its last segment. Today it names each entry by its full path, and `"g/c"` cannot be a keyword. `RecordBatch.select` changes the same way; each of its entries stays a one-column batch, as today.

```python
rec.select("a", "g/c")        # {"a": ..., "c": ...}           (today: {"a": ..., "g/c": ...})
rec.select(x="g/c")           # {"x": ...}                    (unchanged)
sample(hier).select("s1/theta", "s2/theta")   # ValueError: both flatten to theta; name one with a keyword
```

### Renames, and a new name for with\_path\_names

`with_path_names` renames and moves nodes by their paths, and renames a whole term's name. Under grammar B its keys are exactly the term's paths, plus that one name, so a view's subscripts and its rename keys agree:

```python
Normal("x", 0.0, 1.0).with_path_names(x="y")   # Normal(y): renames the whole term   (unchanged)
grouped.with_path_names({"g/c": "g/k"})        # rename in place                     (unchanged)
grouped.with_path_names({"g/c": "c"})          # move c to the top level             (unchanged)
grouped["g"].with_path_names(c="k")            # a view exposing k and e             (today: its keys are g and g/c)
```

Two cases follow from the rule. A whole base term's name cannot become a nested path, since the term has no record to place it in: `Normal("x", 0.0, 1.0).with_path_names(x="g/y")` raises, as it does today; compose it into a joint first. On a batch, renaming the element's whole component renames the batch's one component: `draws.with_path_names(x="y")` on draws of `Normal("x", 0.0, 1.0)` offers `y`.

Candidate names for the method:

| Name | For | Against |
| --- | --- | --- |
| `rename` (recommended) | The name pandas (`df.rename`) and xarray (`ds.rename`) use; short; a path target still moves a node | Breaks the `with_` pattern of `with_label`, `with_level_names` and `with_dim_names` |
| `with_names` | Keeps the `with_` pattern | Does not say whose names; easy to confuse with `with_label` |
| `with_field_names` | Says what is renamed | Wrong: it also renames groups and whole terms |
| `relabel` | Reads naturally | Collides with "label", which is the object's own name |

If `rename` is chosen, a separate change could rename `with_level_names` and `with_dim_names` to `rename_levels` and `rename_dims`, so all three read alike.

## Summaries

A summary that has one value per field of the event keeps the law's paths and declaration form, and the operation appears only in its label. Open PR 576 implements this for `mean`, `variance`, `standard_deviation` and `quantile`; `expectation(d, f)` takes the paths of `f`'s output.

```python
m = mean(grouped)
m["a"], m["g/c"]             # the means of a and g/c           (today: m["mean(a)"], m["mean(g)/c"])
m["g"]                       # the record of means {c, e}
mean(Normal("x", 0.0, 1.0))  # an array: the whole term x, like a draw
m.label                      # "𝔼[(a, g) ~ ...]": the label says it is a mean
```

A summary binds wherever a draw of the same type binds, so plug-in estimates work, such as `condition_on(lik, mean(post[["mu"]]))` and `log_prob(prior, mean(prior))`. The guard is the type, not the name. A summary's spec is derived from the event's: a mean is floating, and its support is the hull of the event's support. So a Bernoulli law's mean is refused where its `int32` event is required. `condition_on(model, standard_deviation(post))` binds whenever the types fit; like passing the wrong array, it is the caller's mistake to make.

Results that are not one value per field are base terms under the operation's name: with PR 576, `cov(d)` is declared under `cov`, and a score under `log_prob`. Today `cov(d)` is a bare array with no component.

**Change from today:** summaries name their fields `mean(a)`, `mean(g)/c`, and so on (VI.0 and VI.5), so `mean(d)[p]` fails for every path `p` of a law whose draws are records.

## Composition with `&`, and what it means for paths

\#571's reply replaces `*` with `&` for composing laws and kernels, and for merging declarations and records. #571 owns the full definition. This section states only what composition does to paths.

**A joint's paths are its operands' paths, side by side.** The joint exposes a record whose children are the operands' components. An exposed operand's paths appear unchanged. A whole base term `x` becomes the child `x`: its value moves from the root to the path `x`.

```python
joint = Normal("x", 0.0, 1.0) & grouped
# joint's paths: x, a, g, g/c, g/e; its draws are {x, a, g: {c, e}}
joint["g/c"]            # the same node as grouped["g/c"]
```

Under grammar A, a whole term over structure moved under its name in the same way, so every one of its paths changed: `params_prior["beta"]` became `joint["params/beta"]`. Under B that cannot happen, because only a base term is named whole, and a base term has no paths below the root.

**Composition meets components with slots by name, and a view offers what the grammar rule says it offers.** A kernel whose slot is a whole group meets the group only through a list of one path:

```python
# lik_g(y | g): a kernel whose slot g takes a record {c, e}
lik_g & grouped[["g"]]      # meets the slot g: grouped[["g"]] offers g
lik_g & grouped["g"]        # does not meet it: grouped["g"] offers c and e
```

**Sampling commutes with `&` when both operands expose records.** For independent laws `a` and `b` that each expose a record, `sample(a & b)` is distributed as `sample(a) & sample(b)`, where `&` on two records merges their fields. For a whole base term it does not hold: `sample(Normal("x", 0.0, 1.0))` is an array, and `&` on arrays is NumPy's elementwise `&`. jhuggins proves this boundary exactly, in a model where laws are point masses (`sample_commutes_iff`).

**Two views can offer the same name.** `hier["s1"]` and `hier["s2"]` each offer `theta`, so `hier["s1"] & hier["s2"]` raises `ValueError`, since a joint cannot produce one name twice. The joint law of both groups is `hier["s1", "s2"]`, which keeps the paths and the dependence between them. `hier[["s1"]] & hier[["s2"]]` composes as well, but as the product of two marginals, so it is the joint only if the two groups are independent.

jhuggins's model also checks the kernel rules of IV.2: a chain of kernels has the same slots, components and validity in every grouping. It finds that IV.2's rationale misattributes associativity, which holds with or without the rule that the right operand's slots avoid the left operand's components. That rule keeps the joint well formed. The fix belongs to IV.2, under #571.

## Batches and the declarations they store

A batch an operation produces stores its producer's declaration, as #571 proposes and its reply accepts. This plan does not need the stored declaration for indexing, and asks that its names stay out of a batch's equality and pytree structure.

**What a batch stores today.** A `BatchSpec` holds the spec of one element and the batch's named levels, but not the names its producer offered:

```python
draws = sample(Normal("mu", 0.0, 1.0), sample_shape=4)
draws.spec        # BatchSpec(NumericArraySpec(shape=()), sample=4): nothing says "mu"
```

**What #571 adds.** The batch stores the element declaration, here the whole base term `OutputSpec(mu=...)`, and gains `components`. Grammar B applies to the stored declaration as to any other: a batch of draws of `grouped` stores the exposed record `{a, g: {c, e}}`, and a batch of draws of `Normal("mu", 0.0, 1.0)` stores the whole term `mu`.

**What indexing uses.** A batch is indexed by its element's paths, so it agrees with its law without reading the stored names. The stored names serve composition and binding, and the component accessor:

```python
draws["mu"]              # KeyError: mu names the whole element, not a path inside it
draws.component("mu")    # draws itself, as Normal("mu", 0.0, 1.0).component("mu") is the law
sample(grouped, sample_shape=4)["g/c"]    # a column, read through the element's paths
```

\#571's text says `batch["mu"]` returns the batch. Under this plan that is `draws.component("mu")`, the same split as for laws.

**Recommendation: keep the names out of equality and the pytree.** `BatchSpec` is the static part of a batch's pytree, the part JAX compares to decide whether two values have the same structure. If the names were stored there, draws of `Normal("x", 0.0, 1.0)` and of `Normal("y", 0.0, 1.0)` would become different structures:

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=3)
y = sample(Normal("y", 0.0, 1.0), sample_shape=3)
jax.tree_util.tree_map(jnp.subtract, x, y)   # works today; would raise "Mismatch custom node data"
```

A `jit`-compiled function would also recompile for each name, and the two branches of a `lax.cond` could not return draws under different names. So the declaration is stored beside the spec, as a label is, and kept out of equality, hashing, the pytree's static data and fingerprints.

The price is the one labels already pay, but only for whole-element names. The names of a record element live inside the element's `RecordSpec`, which JAX's static data already carries, so they survive. A whole element's name, as in a batch of arrays or a column such as `rb["a"]`, is stored beside the spec, and a raw JAX transform rebuilds the batch without it, just as it loses the label today: the `tree_map` above returns a batch labeled `<no description>`. So `jax.jit(g)(sample(Normal("x", 0.0, 1.0), sample_shape=4))` comes back offering `element`, and `.component("x")` or binding it to a slot `x` fails. ProbPipe's own operations restore the names at the result boundary, where they declare their result anyway.

\#571 also says two batches whose elements differ only in component names stop being equal; this recommendation reverses that sentence. jhuggins's model proves that `&` commutes with renaming every name at once (`dmerge_rename`), but no other operation is proved to ignore names, so the recommendation awaits confirmation.

## Deliberate exceptions

Eight cases fall outside the invariants on purpose. Because paths go through every kind, indexing a law, a function or a batch held in a field is no longer one of them; what such terms still lack is summaries and conditioning.

| Case | What happens | Why |
| --- | --- | --- |
| Covariance | `cov(d)` is one base term named `cov` (with PR 576), so `cov(d)[s]` raises | It is not one value per field. The block for `s` is `cov(d[s])`. |
| Monte Carlo summaries | `mean(d)[s]` and `mean(d[s])` estimate the same quantity but are not equal | Each call takes its own draws (PR 576). |
| Summaries and conditioning at a law- or function-valued field | `mean(rm)` is undefined for a random measure's drawn laws, and `condition_on` cannot fix a drawn law or function | A mean of laws is not one value per field; I1, I3 and I8 hold for paths that end at values |
| `&` on values of whole base terms | `sample(Normal("x", 0.0, 1.0) & Normal("y", 0.0, 1.0))` is a record `{x, y}`, but `sample(Normal("x", 0.0, 1.0)) & sample(Normal("y", 0.0, 1.0))` is NumPy's elementwise `&` of two arrays; the same holds for two batches of arrays | A base term carries no name in its value; sampling commutes with `&` only when both operands expose records. |
| A batch of arrays through raw JAX | `jax.jit(g)(draws)` comes back offering `element` instead of `x` | Whole-element names are kept out of the pytree (Batches) |
| Labels | `sample(grouped["a", "g/c"])` and `sample(grouped)["a", "g/c"]` are draws of the same law under different labels | A label says how a term was made, for a reader; no operation reads it. |
| Kernels, until their `[]` lands | `lik["y"]` raises `TypeError`, and so does a path through a field that holds a kernel | Kernel views come in a later PR. |
| Python's mapping rule | `"g" in rec` is true, while `"g"` is not in `rec.keys()` | `keys()` lists fields only, and `in` agrees with `[]` (see Membership). |

## What changes from the current design

II.6 becomes the one statement of paths and subscripts, and II.2 the one statement of the grammar rule; every other section cites them instead of restating a rule.

| Section | Today | Under this plan |
| --- | --- | --- |
| II.2 Declarations | The keyword form names any spec whole; a whole term's paths start with its name; "an exposed record's components are its fields"; `OutputSpec(posterior=DistributionSpec(...))` is the example of a function returning a law; a law's exposed components are typed by one draw's value | The keyword form names only a base spec; the paths of a declaration are its value's paths; components and their addresses as defined above; a component is typed by the output at its address; a returned law is exposed or wrapped; defaults by kind |
| II.4 Labels and views | A field of a value takes its path as its label; batch positions are called a "selection" | Adds the label of a value indexed by several paths, `(a, g/c)`; positions are "positions" |
| II.5 `Batch` | A tuple of names is one path; positions are also called a "selection"; `()` is the whole batch; a raw batch has a bare element spec | A batch is indexed by its element's paths and always holds an element declaration; positions are "positions"; `()` and lists of positions raise; `in` tests paths |
| II.6 `NamedTree` | A path may be a tuple; `[]` raises on a group; `in` tests fields | Owns the definitions, the reading of a subscript, and `in`; `[]` returns groups; paths continue into a held law, function or batch; a list of paths keeps its paths, in the tree's order |
| III.3 `Function` | In sweeps and lifts, a function's output defaults to a whole term under its name; no `[]` | Defaults follow the grammar rule; `f[s]` indexes what `f` returns |
| III.5 `Record` | `select` names entries by full path | `[]` reaches groups and held terms; `select` flattens to last segments |
| III.6 `RecordBatch` | A tuple of names is one path | Cites II.6; a list of paths returns a `RecordBatch`; `select` flattens as on a record |
| III.7 `Distribution` | `d[c]` returns `d` for a whole term; a view of a group is a whole term; several paths flatten to last segments | A whole term's name is read by `component`, not `[]`; a view of a group exposes its parts; several paths keep their paths; `FieldView` possibly renamed `MarginalView` |
| III.8 View capabilities | Derivations use component-rooted paths | Derivations use the term's paths |
| III.9 Kernels | No `[]` | States the exception now; `lik[s]` later |
| III.10 Batches of laws | Silent on `[]` | `laws[s]` is the batch of views |
| IV.1, IV.2 Joints | A packaged sub-joint declares a group as one whole term | A factor that holds a group wraps it, so its draws carry the group's level; #571's vocabulary |
| V.5, V.6 Lifting and broadcasting | Atoms and outputs named whole under a component | Defaults by kind: a lifted law of laws wraps its draws; a sweep's batch is exposed through its element |
| VI.0, VI.5 Summaries | Fields named `mean(a)` | PR 576; its VI.0 text "Selection commutes" becomes "Indexing commutes" |
| VI.6 `condition_on` | Given values are named by "keys of the event schema"; the result's declaration depends on the route | Given values are named by paths, nested ones included; the result is declared as `d.without(*fixed)` (I11) |
| VI.8 `marginal` | `marginal(d, field: str)`; one path or several, decided by count; a group's marginal is the factor that holds it, a whole term; fields "repackaged in the order of the paths" | `marginal(d, paths)`: a path or a list of paths, decided by form; a group's marginal exposes its parts, as the factor's view at the group, since the factor itself now wraps; the tree's order |
| VII.2, VII.5, VII.6 Families | Empirical laws whose atoms are laws or functions, bootstrap laws, random functions and random measures name their event whole | Their draws are wrapped in a record, as #571's event rule requires |
| Glossary | path, key, field, component and selection each defined loosely or twice | Entries for base spec, spec with parts, node, group, leaf path, address, component spec, wrap, slot, position and subscript; key keeps only its random-number and mapping senses; selection and packaging removed; component redefined. The occurrence path (V.8) and "selection order" (II.7) keep their entries |
| `STYLE_GUIDE.md` §1.10, §1.11 | `"a/b"` is "a short form of the tuple form"; `[]` "reaches only leaves"; a `DistributionBatch` is indexed by position only | A path is a `str`; a tuple is a list of paths; `[]` and `in` reach groups while `keys()` stays leaf-keyed; a batch of laws is also indexed by path |

These changes break existing code, and `CHANGELOG.md` records each:

- `OutputSpec(name=spec)` raises for a spec with parts, and defaults follow the grammar rule instead of naming structured outputs whole.
- A view or marginal of a group offers its parts, not the group's name.
- A factor that holds a group, a random measure, a bootstrap law, a random function, and a lifted law-returning function wrap their draws, so each draw gains a level.
- A function returning a law offers the law's components, typed by their views; consumers that met it by its name now need a wrap.
- A raw batch of arrays offers the default component `element`.
- A whole term's name is no longer a subscript: `Normal("x", 0.0, 1.0)["x"]` raises.
- A tuple of `str` is no longer one nested path, on `[]`, `at_path`, `is_field`, `without`, and the keys of `with_path_names`.
- A law indexed by several paths keeps their paths instead of flattening to last segments.
- `draws[()]` raises instead of returning the whole batch.
- `without` refuses a path given twice and a path inside another.
- `select` flattens to last segments, on records and on record batches.
- `in` changes on records, batches and laws.
- `condition_on` returns a law that exposes a record on every route.
- `marginal`'s parameter is renamed from `field` to `paths`.

Whether the break is clean, with no warnings or compatibility shims for a release, is to be confirmed with jhuggins.

## Where this plan meets #571

The two plans share twelve points. For each, one plan owns the rule and the other depends on it; where this plan needs #571's text to change, the last column says how.

| Point | Owner | What this plan needs | Change to #571's text |
| --- | --- | --- | --- |
| `OutputSpec` as a mapping of its components, with `project` and `without` | #571 (reply) | `decl[c]` reads a component; `project` and `without` take subscripts read as `[]` reads them | `project` keeps each node's path and refuses no shared last segment, instead of flattening to last segments |
| What a declaration exposes | this plan (grammar B) | A whole declaration names only a base spec; everything with parts is exposed or wrapped | `exposes` needs no flag: the spec fixes the form (`B_decl_determined`) |
| The paths of a whole declaration | this plan | A whole term is a base term, so its only path is the root; composition places it at its name | #571's "Paths" item ("its component followed by the paths within its term") is replaced |
| Reading a component | this plan (I12) | `t.component(c)` is the term at `c`'s address, defined recursively | #571's five cases of "Reading a component" become this one rule; its case 5, a batch whose element is whole under `c`, gives the batch itself, as before |
| How a component is typed | this plan, following jhuggins's decision 5 | A component's spec is the spec of the output at its address; for an exposed law that is the view's spec | #571's "the value at `c` of one draw" changes for declarations whose value is a law |
| Every batch holds an element declaration | #571 (storage); this plan (indexing) | A batch is indexed by its element's paths; a raw batch's base element is whole under `element` | #571's sentence that a raw batch keeps a bare element spec, with no components, changes |
| Batch equality and the pytree | this plan (recommended, awaiting confirmation) | The stored names stay out of equality, hashing, the pytree and fingerprints | #571's "two batches whose elements differ only in component names stop being equal" is reversed |
| A column of a batch of records | #571 | `rb["a"]` carries `OutputSpec(a=...)`, as #571 says; a group's column `rb["g"]` carries the exposed record `{c, e}` | add the group case |
| `&` | #571 (reply) | Only the path rules in "Composition with `&`" | none; but #571 refuses `&` on an operand that exposes a law or a batch, and under B more outputs do, so such outputs must be wrapped to merge. IV.2's rationale fix rides with #571 |
| A law's event is whole or an exposed record | #571 | The rule takes precedence over grammar B's defaults: random measures, random functions and lifted laws of laws wrap their draws | none, but see the open question below |
| Lifting exposed components | #571 (its lifting rule) | A consumer that meets an exposed law's component receives a marginal law through lifting; more outputs are exposed under B, so more consumers rely on it | none |
| Slots | #571 | `condition_on` reaches inside a record slot by path | none; jhuggins's model has no paths inside slots yet |

**One conflict to resolve together.** Paths go through every kind, so a single-path view can land at or below a field that holds a law or a function: `rm["G"]` or `rm["G/mu"]` on a random measure, whose draws `{G: law}` hold laws, or `gp["f"]` on a Gaussian process. Under grammar B such a view exposes what it reaches, a law or a function, so its event exposes another law's or function's components, which #571's event rule forbids (jhuggins's `sub_VB_breaks_rule`). Either such a view wraps, keeping the field's name as `d[["G"]]` does, or #571 narrows its event rule to exposed laws and kernels with two or more components, as jhuggins's open question 4 suggests. Until this is ruled, I12 for random measures and random functions, `rm.component("G")` and `gp.component("f")`, is unsettled. The Decisions table lists it.

## Decisions

Thirteen decisions are settled for this plan. Nine more are recommended: this draft adopts them, and they await confirmation. Five are open. Each row points to the section that explains it.

| Decision | Status | Choice or recommendation |
| --- | --- | --- |
| A `str` is one path; a list or tuple of `str` is a list of paths (Reading a subscript) | Decided | As decided |
| The form of the subscript decides the result (I9) | Decided | As decided |
| A list of paths keeps each node at its own path (Indexing with several paths) | Decided | As decided |
| A list of paths refuses duplicates and a path inside another | Decided | As decided |
| A value indexed by several paths is labeled by its paths, `(a, g/c)` | Decided | As decided |
| Summaries keep the law's paths; the spec is the only guard (Summaries) | Decided | As decided; PR 576 |
| `()` and `[]` on a batch raise | Decided | As decided |
| `condition_on(d, given)` is declared as `d.without(*fixed)`, on every route (I11) | Decided | As decided |
| A list of paths and `without` partition the leaf paths (I10) | Decided | As decided |
| Nodes keep the original tree's order in every result | Decided | As decided |
| Grammar B: a whole declaration names only a base spec (The grammar rule) | Decided | Decided for this plan; jhuggins's final ruling between B and B′ is open |
| Paths go through a law, function or batch held in a field (Definitions) | Decided | As decided; answers jhuggins's open question 1 |
| Examples use `&` for composition | Decided | As decided here; the operator itself is #571's |
| A whole term's name is not a subscript; a public accessor reads components (Indexing each kind) | Recommended | `t.component(name)`; the method's name is a placeholder |
| A component's address is defined recursively, and its spec is the output at that address (Definitions) | Recommended | Yes, following jhuggins's decision 5 |
| Defaults by kind; #571's event rule wins over them (Defaults and pending types) | Recommended | Yes |
| Every batch holds an element declaration; a raw batch of arrays offers `element` (The grammar rule) | Recommended | Yes |
| Renames accept a whole term's name as well as paths (Operations) | Recommended | Yes; never ambiguous under B |
| `in` is true for groups and paths through held terms (Membership) | Recommended | Yes |
| The new name for `with_path_names` (Operations) | Recommended | `rename` |
| Rename `FieldView` to `MarginalView` (Indexing each kind) | Recommended | Yes |
| A batch's stored names stay out of equality, hashing, the pytree and fingerprints (Batches) | Recommended | Yes, accepting that raw JAX transforms drop them |
| Who builds the record when a structured output is wrapped, and what a pending type filled with a spec with parts becomes (The grammar rule) | Open | Lean: the caller declares and returns the record; a pending type raises |
| A view at or below a field holding a law or a function, and random functions, against #571's event rule (Where this plan meets #571) | Open | The view wraps and random functions wrap, or #571 narrows its event rule |
| When kernels get `[]` (Indexing each kind) | Open | A later PR; the design states the exception until then |
| A clean break, with no warnings or shims (What changes) | Open | Likely; to confirm with jhuggins |
| Renames by path and paths inside slots are not yet in jhuggins's model | Open | Extend the model before the design text is final |

## Implementation sequence

This order is tentative until the design is final. Every PR targets `dev/overhaul`. PRs 575 and 576 must land first. Each step adds property tests for the invariants it delivers, and marks the rest as expected failures until their step lands.

1. **Design.** II.2 states the grammar rule and II.6 the definitions and the reading of a subscript, and the other sections cite them; the glossary and `STYLE_GUIDE.md` §1.10 and §1.11 change. #571's text changes listed above land in the same pass, so that its `project` is never implemented with flattening. The design conformance tests get pending entries for signatures that change before the code does.
2. **Reading a subscript (#573).** One reader and one `_PATH_SEP`; `[]` returns groups on `Record` and `RecordSpec`; lists of paths keep their paths, on records, batches and laws; `in`; `marginal(d, paths)`; `()` raises on a batch. Delivers I4, I9, I5 within records, I10 on records, specs and batches, and I3 with PR 576.
3. **Grammar B in declarations**, with #571's `OutputSpec` and batch-declaration work, which it depends on. The keyword form refuses a spec with parts; every batch holds an element declaration; a view of a node with parts exposes it; defaults follow the grammar rule; code that declares structured outputs whole moves to exposing or wrapping. Delivers I2, I6, I7, and I5 on laws.
4. **Paths through every kind.** Paths continue into a law, function or batch held in a field; `Function` gains `[]`; the public component accessor. Delivers I1, I5 across kinds, I12, and I13 for functions.
5. **Conditioning and removal.** `condition_on` declared as `d.without(*fixed)`; nested paths, route by route; `law.without(...)`. Delivers I8, I11, and I10 on laws.
6. **Batches of laws, and kernels.** `laws[s]` and `lik[s]`. Delivers I13 for kernels.

Two bugs found along the way are separate tasks, not part of this sequence: the Gaussian route of `condition_on` accepts a given record with a missing or extra field, and inference on a whole-record prior fails with a wrong message. Grammar B removes the second one's cause.

## Appendix: how this plan differs from the grammar-A plan

The two plans read subscripts identically: jhuggins's model proves that the grammar does not change what a path selects (`proj_variant_invariant`). They differ in what names a term offers, and in how far paths reach.

| Topic | [Grammar-A plan](https://claude.ai/artifact/PFHi8LYSRU1xgzGs8oYVCt) | This plan |
| --- | --- | --- |
| Whole declarations | over any spec | over a base spec only; a structured output is exposed or wrapped |
| `grouped["g"]` | a whole term `g` over `{c, e}` | exposes `c` and `e`; `grouped[["g"]]` offers `g` |
| Paths of a whole term over structure | inside the value, while composition and renames used the name as a prefix: two path sets | no such term; one path set, and every component sits at its address (I12) |
| How a component is typed | not addressed | the spec of the output at its address |
| Rename keys | paths plus the whole term's name, with a `ValueError` for an ambiguous key | the same keys, which can never be ambiguous |
| A law, function or batch held in a field | paths stop at the field; an exception | paths continue into it (I5 across kinds) |
| Functions | no `[]` | `f[s]` indexes what `f` returns (I13) |
| Reading a component | an internal reader | a public accessor, `t.component(name)` (the name is a placeholder) |
| Indexing a declaration | `event_spec[s]` projected | `decl[c]` is a component's spec, and `decl.project(s)` projects |
| The batch's stored declaration | recommended dropping it from #571 | every batch holds one; names kept out of equality and the pytree (recommended) |
| Defaults | whole under a default name | by kind, with #571's event rule taking precedence |
| `params_prior`, declared `OutputSpec(params=RecordSpec(...))` | allowed; draws are `{beta, sigma}` | `TypeError`; wrapped, its draws are `{params: {beta, sigma}}` |
| A function returning a joint law under one name | allowed | exposed, offering marginals, or wrapped in a record (who builds the record is open) |
| Composition in examples | `*` | `&` |
| Invariants | I1 to I11 | I1 to I13: adds I12 (a component is the term at its address) and I13 (indexing commutes with producing); I1 and I6 use `project`; I10 counts leaf paths |
| Exceptions | six, including a law or batch held in a field and a draw that is a law | eight; indexing through held terms is no longer one, while summaries and conditioning at law-valued fields, and `&` on base-term values, are |

Unchanged from the earlier plan: how a subscript is read and its errors, lists of paths that keep their paths, the tree's order, `in`, conditioning declared as `d.without(*fixed)` (I11), summaries (PR 576), `select`, `()` raising on a batch, and the open questions on `MarginalView`, the name for `with_path_names`, kernel `[]`, and a clean break.
