# Declarations and addressing: inputs, outputs, and paths

> **Status: merged plan, 2026-10-18, provisional pending the maintainer's final read of the inputs and outputs plan.**
> This text supersedes [`declarations-and-addressing.md`](declarations-and-addressing.md) provisionally.
> It is that plan with the accepted proposal [`inputs-and-outputs.md`](inputs-and-outputs.md) merged in, taking the recommended answer to each of the proposal's six questions (§ 6.20), and one further decision: the elements' output spec of a batch is named `element_spec` (§ 6.21).
> The original stays in the repository, unchanged, so it can be recovered if the maintainer does not accept the merge; the git tag `plans-before-io` marks the state of `plans/` before the merge.
>
> The original plan replaced [`declarations.md`](declarations.md) and [`addressing.md`](addressing.md) (decisions, 2026-10-17), and merged three sources:
> 1. the body of [#571](https://github.com/TARPS-group/prob-pipe/issues/571), on declarations;
> 2. jhuggins's reply on #571 (2026-10-15);
> 3. the grammar-B addressing plan for [#573](https://github.com/TARPS-group/prob-pipe/issues/573) (2026-10-16).
>
> Where those sources disagreed, the addressing plan won, then the reply, then the body (decisions, 2026-10-17); [`gaps-and-conflicts.md`](gaps-and-conflicts.md) D1 records how.
> Where the inputs and outputs plan disagreed with the original, the inputs and outputs plan wins; § 7.5 lists every decision it reverses.
> Every other decision in [`decisions.md`](decisions.md) binds this text.
> Code references name functions; where they cite `file:line`, the line is at `6f8859fc`.
> The "Today" examples were run at `a703f15e` or, where the inputs and outputs plan added them, at `6f8859fc`.
> A citation "fact N" names fact N of [`inputs-and-outputs.md`](inputs-and-outputs.md) Appendix E, which records how each was checked; that plan's Appendix A holds the proofs of the merge properties of § 4.15, and its Appendix D the alternatives its three source designs chose.
> [`README.md`](README.md) § Steps owns the order of the steps, their dependencies, and the cut; this plan owns each step's content and its done-condition.
> The adversarial review of 2026-10-18 was applied to this text; the items it left for the maintainer carry the note "Pending the maintainer's decision", each with its entry in [`decisions.md`](decisions.md).
> Pending the maintainer's decision: the name of this file once it is accepted, and whether it then replaces `declarations-and-addressing.md` so that the links in #571 and #573 keep working, see `decisions.md`.

## Contents

1. Introduction and context
2. Core definitions
3. Invariants
4. General rules
5. Term by term
6. The API, examples, exceptions, vocabulary, steps, and decisions
7. Relation to previous plans
8. Transition plan: the interim rules between the steps

## 1. Introduction and context

### 1.1 What this plan settles

A law in ProbPipe and every value derived from it should be indexed by the same paths.
The values derived from a law are:
- its draws, and batches of draws;
- its summaries;
- its views and marginals.

The names a term offers to the rest of a workflow should be the names inside its value.
Three rules make both hold by construction:

1. **A path is a place inside a value.**
2. **Paths go through every kind:** into a record's children, a law's draws, a function's return values, and a batch's elements.
3. **Only an array or an opaque value is ever named from outside.** Anything with parts offers its parts, or is wrapped in a record that holds its name.

The third rule is **grammar B** of jhuggins's [component interface doc](https://claude.ai/artifact/6TusK5RoZokDA1wk5sbdaq), which models specs, output specs, and paths in Rocq.
Names in code font such as `graft_identity_B` are lemmas of that Rocq development.

On top of these rules, the plan defines what every term produces and receives, and one merge:
- every term **produces** a term: a law a draw, a kernel a draw at each value of its inputs, a function its return value, and a value itself (§ 2.5);
- a term's **inputs** are the named places where it receives values, listed by an `InputSpec`, and its **outputs** are the names under which it offers what it produces, stated by an `OutputSpec` (§ 2.6, § 2.11);
- `[]` takes paths on every kind, output specs included, and `without` removes nodes (§ 4.2, § 4.4, § 4.5);
- `a & b` puts the outputs of both operands side by side, and an output of `b` **feeds** the input of `a` with the same name (§ 2.15);
- every batch stores the output spec of its elements, `element_spec` (§ 5.8).

This plan decides two things:
1. **inputs and outputs:** what a term produces and receives, how its input spec and output spec state them, and the operations on those specs, merge included;
2. **addressing:** how a path or a list of paths picks out a part of a term.

### 1.2 Why: the problem today

Four problems motivate the plan: paths mean different things on a law and on its own values, a term's inputs have no single reading, an output spec lacks the operations users need, and a batch does not store the names its maker offered.

**Paths.**

Today a law and its own values often disagree on what a path means, and a term can offer names that its value does not hold.
Every result below was run on `dev/overhaul` at `a703f15e`, where `*` merges laws:

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
3. **A name can be outside the value it names.** The view `grouped["g"]` is declared as a whole term named `g`, so it is indexed by `g/c`, while its draws are the record `{c, e}`, indexed by `c`. The same holds for any keyword form over a spec with parts, such as a law declared `OutputSpec(params=RecordSpec(...))`: the law is indexed by `params/beta`, and its draws by `beta`. This plan calls `OutputSpec(n=T)`, for a spec with parts `T`, an **interim whole form**, since grammar B removes it; § 8.1 states its rules until then.
4. **Summaries renamed their fields** to `mean(mu)` and so on. PR #576 fixed this.

Grammar B removes cause 3 at its source: a name may be outside its value only when the value is a base value, which has no paths to disagree with.
§ 4.1 removes causes 1 and 2.

**Inputs.**

A term's inputs have no single reading, and the operations users need are missing.
Take the law `growth_prior` over `K`, `r`, `phi`, and `sigma`, as in #571's example, and the kernel `shocks_given_sigma` that produces shocks `e` given `sigma`.
A likelihood that conditions on `K`, `r`, `phi`, and `e` declares its inputs today as follows:

```python
shock_spec = RecordSpec(shocks_given_sigma.event_spec.components)
likelihood_given_spec = growth_prior.event_spec.spec.without("sigma").merge(shock_spec)
```

`growth_prior.event_spec.spec` is a record because the law offers the record's parts.
The kernel names its event `e` at the root, so its `.spec` is a `NumericArraySpec`, and its outputs are copied into a `RecordSpec` first.
The tutorials pick outputs with dict comprehensions and combine them with `{**inputs, ...}`, as #571 found.

Four defects follow, each run at `a703f15e`:

1. **A record read as inputs is read by its leaf paths.** `conditional_distribution` reads `given_spec` with `.items()` (`_slots_of` in `distributions/_conditional.py`), and a `RecordSpec` yields its leaf paths. Once the law is regrouped, the idiom above fails for a function `lik_fn(K, r, phi)` that returns the law of the data:

   ```python
   regrouped = growth_prior.with_path_names({"K": "growth/K", "r": "growth/r"})
   conditional_distribution(lik_fn, given_spec=regrouped.event_spec.spec.without("sigma"))
   # TypeError: given_spec of 'lik_fn': unknown parameters ['growth/K', 'growth/r']; available parameters: ['K', 'r', 'phi']
   ```

2. **`InputSpec` refuses a record.** `InputSpec(RecordSpec(K=(), r=()))` raises `TypeError: InputSpec expects a mapping of slot names to specs, got NumericRecordSpec`, although `conditional_distribution` accepts the same record as `given_spec` (fact 1).
3. **The size shorthand is accepted in one place.** `RecordSpec(K=())` declares a scalar field, while `InputSpec(K=())`, `given_spec={"K": ()}`, and `OutputSpec(K=())` raise `TypeError` (fact 4). Test modules define their own scalar spec, such as `SCALAR = NumericArraySpec(())`.
4. **`InputSpec.without` ignores a name that is not an input.** `InputSpec(a=NumericArraySpec(())).without("zzz")` returns the input spec unchanged (fact 2).

**Output specs.**

1. **The interface has three forms with three APIs.** `OutputSpec.components` is a `mappingproxy` with no operations. `InputSpec` is a flat `Mapping` with `without` and `with_optional`. `RecordSpec` has the tree operations, and its mapping interface iterates leaf paths. A user converts between the three by hand, as the inputs example above shows.
2. **No public attribute says whether an output spec names its term at the root.** For a law `d` whose event is `mu`, `OutputSpec(mu=d.spec)` and `OutputSpec(d.spec)` have the same output names, the same `exposes_record`, and the same `.spec`. They differ only in the spec of `mu`: the law's `DistributionSpec` in the first, and the event's `NumericArraySpec` in the second. Code outside `_specs.py` reads the private `OutputSpec._component_name` instead, on 19 lines of 6 modules at `6f8859fc`, as `_lift_result` in `functions/_broadcast.py` does to detect a positional law.
3. **The library uses private helpers for the operations users lack.** `_components_record` in `core/_specs.py` is the conversion in the inputs example above, with 7 call sites in 6 modules. `_view_declaration` in `distributions/_views.py` computes a view's output spec and has no public counterpart. The joint's event spec is assembled inline from its factors' outputs, in `_factor_graph` in `distributions/_factored.py`.
4. **`event_spec.spec` reads as a spec of a spec.** `OutputSpec.spec` is the type of the produced term, stored as `_term_spec`. `.event_spec.spec` occurs on 99 lines of `probpipe/` and 199 lines of `tests/` at `6f8859fc`.
5. **An output spec does not take paths.** `"mu" in prior.event_spec` raises `TypeError` (fact 13), and nothing gives the output spec of a part.

Grammar B already settles the second defect: an output spec is in the keyword form if and only if its term spec is a base spec, so the spec states the form and no flag is needed.

**Batches.**

A `BatchSpec` holds the term spec of one element and the batch's named levels, but not the names its maker offered:

```python
draws = sample(Normal("mu", 0.0, 1.0), sample_shape=4)
draws.spec          # BatchSpec(NumericArraySpec(shape=(), dtype=float32, support=real), sample=4): nothing says "mu"
draws.components    # AttributeError
draws["mu"]         # TypeError: cannot index NumericArrayBatch by 'mu': its elements have no named fields
grouped_draws = sample(grouped, sample_shape=4)
grouped_draws["a"].spec   # BatchSpec(NumericArraySpec(...), sample=4): the column's name is not stored
```

II.5 already specifies that a batch an operation produces carries its maker's output spec, and that `Batch` has `components`; the code stores a bare `TermSpec` in the field `BatchSpec.element_spec`, and `Batch.element_spec` reads it.

### 1.3 Scope

**In scope:**
- the definitions of term specs, the produced term, inputs, outputs, input specs, output specs, paths, lists of paths, and positions;
- the grammar rule, which decides which output specs may name a value from outside;
- the reading of inputs, the size shorthand, and the operations on `InputSpec` and `OutputSpec`;
- what `[]` reads and returns on every kind, lists of paths, positions, and membership (`in`);
- the operations that take paths: `marginal`, `condition_on`, `log_prob`, `without`, `select`, and `rename`;
- what a summary is indexed by;
- one definition of merging with `&`, for every kind;
- the output spec a batch stores for its elements, and how it enters equality and JAX;
- the vocabulary, the changes to `design/`, and the content and done-condition of the steps S0, S1, S2a, S2b, S3, S4, G1a, G1b, G2–G5, and R, the rename, of [`README.md`](README.md) § Steps;
- the interim rules that hold between the steps (§ 8).

**Out of scope:**
- how a computation runs, and where: the execution plan, [`execution-layer.md`](execution-layer.md) (#580);
- per-call state, such as `config` blocks and the call token: [`execution-layer.md`](execution-layer.md) § 6.16, "The call context";
- the tutorials in `docs/tutorials/`, which are being rewritten separately (README directive 6).

The execution plan relies on this plan in two places.
An embeddable `Function` needs a complete `term_spec` (execution plan §6.2), and claims that name parts of an input use the path reader of § 4.1.
The execution plan uses two words that this plan does not: an **evaluation point** is one evaluation of a callable, and a **requester** is the code that asks for a set of evaluation points.

PR #576 (summaries keep the law's paths) and PR #575 (`raw` as a property) have merged into `dev/overhaul`.
#575 removed `raw(path)`, so `raw` is not covered here.

### 1.4 How to read this plan

Sections 2 to 6 state the design once every step, before and after the workshop, has merged: each definition, rule, example, and table describes that state.
Section 2 defines the terms, in the order they depend on each other, and the general definition of merge.
Section 3 states the invariants that the definitions imply or the design chooses.
Section 4 states the rules that hold the same way for every kind, and section 5 goes through the kinds one by one, each in the same order: what the term is, its produced term, its parts, its inputs, its outputs, the specs it holds, indexing, merge, and examples.
Section 6 holds the API sketch, worked examples across kinds, the deliberate exceptions, the vocabulary, the changes to `design/` and the CHANGELOG, the steps, and the decisions.
Section 7 relates the plan to the plans and decisions it replaces.
Section 8 holds every interim rule, what holds before a step merges, grouped by the step that ends it; each step block of § 6.9 to § 6.15 names the rows of § 8 that it starts and ends.

A "Change from today" note says in one line what changes for users.

**The words (§ 6.20, Q1).**
This plan explains the design in its final words: input, output, input spec, output spec, event spec, `O.outputs`, `t.output(c)`, and `output_name=`.
The class names `InputSpec` and `OutputSpec` do not change.
The words that `design/` and the code use before the rename PR are interim rules: rows T25 and T26, and § 8.2.

The examples are written in the proposed API, so they merge with `&` and rename with `rename`.
A "Today" example uses today's API, `*` and `with_path_names`.
Today's output prints `NumericRecordSpec` for a record of numeric fields, a subclass of `RecordSpec`; the proposed examples write `RecordSpec` for short.

"The tour" refers to the walkthrough notebook `review/overhaul_tour.ipynb`.

Section numbers such as II.6 refer to the design reference in `design/`.

## 2. Core definitions

### 2.1 Terms and kinds

ProbPipe builds statistical workflows from laws (probability distributions), functions, the values they produce, and operations on all of them.
These are the terms the plan relies on before the rest of this section defines its own:

| Term | Meaning | Example |
| --- | --- | --- |
| **term** | Any object ProbPipe tracks: a value, a law, a kernel, a function, or a batch (II.4) | |
| **kind** | A sort of term, identified by the class of its term spec (II.1) | a law, a record |
| **term spec** | The type of one term, a `TermSpec` (II.1) | `NumericArraySpec((3,))`, `RecordSpec(a=())`, `DistributionSpec(...)` |
| **law** | A probability distribution, the class `Distribution`, which has no inputs (III.7) | `Normal("x", 0.0, 1.0)` |
| **event** | What one draw of a law is: an array, or a record of named parts | the event of `Normal("x", 0.0, 1.0)` is a scalar |
| **draw** | One value sampled from a law | `sample(d)` |
| **record** | A tree of named values, the class `Record`; a nested value is named with `/` (III.5) | `Record({"a": 1.0, "g": {"c": 2.0}})["g/c"]` |
| **opaque value** | A value that is neither numeric nor a record, such as a string or an arbitrary Python object | `"forest"` |
| **batch** | Many separate terms of one kind, stacked along named axes called **levels**; one of them is an **element** (II.5) | `sample(d, sample_shape=100)` has the level `sample` |
| **kernel** | A conditional distribution `K : S → P(T)`, the class `ConditionalDistribution`, with at least one input: a law for each value of what it conditions on (III.9) | `lik(y \| mu)` |
| **function** | A tracked callable, the class `Function`, whose `FunctionSpec` holds an `input_spec` and an `output_spec`, either of which may be `None` (III.3) | `Function(run_inference, output_spec=...)` |
| **joint law** | The law of several parts together; `&` merges laws and kernels into one | `lik & prior` |
| **merge** | What `&` does: it puts the outputs of both operands side by side, makes an input both operands take one input, and lets an output of the right operand feed the input of the left operand with the same name (§ 2.15). "Composition" keeps its meaning of `f ∘ g` and `A B` | `lik & prior`, `rec1 & rec2` |
| **view** | `d[...]`: a law over part of `d`'s event that keeps a reference to `d`; its class is `MarginalView` (today `FieldView`) | `prior["mu"]` |
| **marginal** | `marginal(d, ...)`: the same law as the view, detached from `d` (VI.8) | `marginal(prior, "mu")` |
| **conditioning** | `condition_on(d, given)`: the law of the rest of `d` given values for some of its parts | `condition_on(model, {"y": y_observed})` |
| **summary** | A mean, variance, standard deviation, or quantile of a law | `mean(d)` |
| **route** | How an operation computes its result: by an exact method, or approximately by Monte Carlo from draws | an exact mean of a Normal |
| **label** | A display name for a term; no operation reads it (C5, II.4) | `mu ~ prior` |
| **lifting** | Applying a function to a law or a batch where it expects a value (C4) | `f(prior)` |
| **JAX pytree** | How JAX sees a value: the arrays it transforms, plus static data it compares to decide whether two values have the same structure | |
| **factor** | One of the laws or kernels a joint law is merged from; a **factored joint** keeps its factors, as `FactoredDistribution` and `FactoredConditionalDistribution` do (IV.1) | `lik` and `prior` are the factors of `model` |
| **grouped sub-joint** | The one factor that a rename gathering outputs under one node makes of their factors (IV.1 calls it a packaged sub-joint; § 6.4 renames it) | the factor of `grouped` that holds `g` |
| **symbolic dimension**, **dimension scope** | A dimension size given by a name, such as `"n_obs"`; specs in one scope bind each name to one size, and a later occurrence that disagrees raises (II.1) | `NumericArraySpec(("n",))` |
| **distribution family** | A class of laws with its own constructor (VII) | `Normal`, `RandomMeasure` |
| **atoms** | The stored values an empirical law draws from | `EmpiricalDistribution(atoms)` |
| **fingerprint** | A content hash of a term, which caching and replay use to recognize the same computation | computed automatically |

The principles cited are C1 (uniform interface), C3 (detail hidden by default), C4 (lifting), C5 (names), D1 (mathematical fidelity), D2 (generality first), D5 (carried structure), and D6 (single source of truth), from `design/01-design-principles.md`.

Two distinctions carry the plan:
1. A term spec is either a **base spec**, with no parts, or a **spec with parts**.
2. A **path** is a place inside a value, while an **output** is a name under which a term offers a value it produces.

Grammar B makes the outputs and the paths coincide, apart from one base value named at the root.
Each definition uses § 2.1 and the definitions before it, and ends with an example; where one needs a later operation, it names the section that defines it.
Some definitions recurse on the nesting of specs; the recursion ends because specs nest finitely.
The words *child* and *canonical order* keep their II.6 meanings: a child is an entry directly under a node, and the canonical order visits children depth first, in insertion order.
A **name** is a non-empty `str` without `/` (II.2, II.6), and an **identifier** is a name that is a Python identifier and not a keyword (III.3).

### 2.2 The running examples

These terms appear throughout:

```python
from probpipe import (Normal, LogNormal, Beta, HalfNormal, sample, mean, marginal, condition_on, log_prob,
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
post = condition_on(model, {"y": y_observed})   # a posterior whose draws are {mu, tau}
post_spec = post.spec                           # its DistributionSpec

# A kernel with the required input mu and the optional input s, whose default is 1.0.
def lik_opt_fn(mu, s=1.0):
    return Normal("y", mu, s)
lik_opt = conditional_distribution(lik_opt_fn, given_spec={"mu": (), "s": ()}, label="lik_opt")
s_prior = Normal("s", 100.0, 0.001)             # a law over s, near 100

# A growth model's prior, and a kernel that produces shocks e given sigma.
growth_prior = (LogNormal("K", jnp.log(350.0), 0.4) & LogNormal("r", jnp.log(0.4), 0.4)
                & Beta("phi", 2.0, 2.0) & HalfNormal("sigma", 0.2))   # draws {K, r, phi, sigma}
shocks_given_sigma = conditional_distribution(
    lambda sigma: Normal("e", jnp.zeros(n_years), sigma),
    given_spec=growth_prior.event_spec["sigma"])                   # draws e, given sigma
```

A few sections also use terms whose construction does not matter to the example:

- `y_observed` and `data`: observed arrays; `n_years`: a number of years.
- `emulator`: a function from `theta` to a record `{mean, var}`, returned by a function `fit_emulator`, which a callable `train_gp` implements; `theta` is a value of its input, and `theta_spec` its spec.
- `rm`: a random measure, a law whose draws are records `{G: law}`, where each drawn law has the spec `law_spec` and draws `{mu, tau}`.
- `gp`: a Gaussian process, a law whose draws hold a function under the name `f`.
- `lik_g`: a kernel `lik_g(y | g)` whose input `g` takes a record `{c, e}`; `lik_G`: a kernel whose input `G` takes a law of spec `law_spec`.
- `lik_nee` and `lik_lai`: kernels that produce `nee` and `lai` given the same input `theta`.
- `likelihood`: a kernel that produces `y` given `K`, `r`, `phi`, and `e`, built in § 6.2.4.
- `run_inference`: a plain Python function that returns a posterior law like `post`.
- `sample_params`: a plain Python function of a random key that returns a dict `{"beta": ..., "sigma": ...}`.
- `bern`: a Bernoulli law, whose event is an `int32`.
- `post_draws`: `sample(post, sample_shape=1000)`, a batch of records `{mu, tau}`; `simulate`: a function from `mu` and `tau` to a record `{y}`, so `simulate(post_draws)` is a batch of records `{y}` over the same level.

### 2.3 Base specs and specs with parts

A **base spec** is a `NumericArraySpec` or an `OpaqueSpec`: the type of a term with no named parts.
A term whose spec is a base spec is a **base value**: an array or an opaque value.

A **spec with parts** is one of five:
1. a `RecordSpec` with at least one child;
2. a `DistributionSpec`;
3. a `ConditionalDistributionSpec`;
4. a `FunctionSpec`;
5. a `BatchSpec`.

An empty `RecordSpec` is neither, and an output spec refuses it (O10).
A law is always a term with parts, even `Normal("x", 0.0, 1.0)`, whose draws are single arrays: that law has no paths below its root, and its one output `x` is at the root (§ 2.11).

```python
NumericArraySpec(())                          # a base spec
RecordSpec(a=(), g=RecordSpec(c=(), e=()))    # a spec with parts
post.spec                                     # a spec with parts: a DistributionSpec
```

A term **has parts** when its term spec is a spec with parts; "parts" means nothing more in this plan.
A term **offers its parts** when its output spec is in the positional form: whatever outputs it offers are then named inside it (§ 2.11), not from outside, at the root (§ 2.7).

**Change from today:** a `BatchSpec` holds an output spec of its element, not a bare term spec (§ 5.8).

### 2.4 Nodes, fields, groups, paths, lists of paths, and positions

| Term | Meaning | In a draw of `grouped`, `{a, g: {c, e}}` | Change from today |
| --- | --- | --- | --- |
| **path** | The `/`-joined names from the top of a term down to a node. It is always a `str` (decided 2026-10-15). | `"a"`, `"g"`, `"g/c"` | a tuple is no longer a path |
| **node** | A place in a term that a path names. The **root** is the term itself; it has no path. | `a`, `g`, `g/c`, `g/e` | new glossary entry |
| **field** | A node of a record, named through records only, that is not itself a record. A field may hold a law, a kernel, a function, or a batch, whose own paths continue below it; such a field is a **held field** for short, and the term it holds a **held term**. `keys()` lists a record's fields. | `a`, `g/c`, `g/e`; in `record_with_law` below, `G` is a held field | none |
| **group** | A node of a record, other than the root, that is itself a record. | `g` | one word replaces "interior node", "subtree", "field group", and "group of fields" |
| **leaf path** | A path with no path below it. | `a`, `g/c`, `g/e` | a plain phrase, not a glossary term |
| `last(p)` | The last name of the path `p`; notation in this plan. | `last("g/c") == "c"` | — |

**Paths go through every kind (decided 2026-10-16).**
A path continues past a field that holds a term with parts, into that term's own paths:
- a law's paths are the paths of its draws;
- a kernel's are those of the draws of the law it gives;
- a function's are those of what it returns;
- a batch's are those of one element.

So the record `record_with_law = Record({"n": 3.0, "G": post})`, whose field `G` holds the posterior over `{mu, tau}`, has the paths `n`, `G`, `G/mu`, and `G/tau`.
`record_with_law["G/mu"]` is `record_with_law["G"]["mu"]`.
The fields of `record_with_law`, which `keys()` lists, are still only its own: `n` and `G`.
The leaf paths of `record_with_law` are `n`, `G/mu`, and `G/tau`.
§ 2.10 defines the paths of every spec.

**Change from today:** paths stop at a field, and a law, function, or batch held in a field is indexed only by a second `[]`.

| Term | Meaning | Example | Change from today |
| --- | --- | --- | --- |
| **list of paths** | Several paths in a Python list or a tuple, to pick several nodes at once. | `["a", "g/c"]` | today a tuple of paths is accepted on laws only and is one nested path elsewhere; a Python list is refused everywhere |
| **position** | An integer, a slice, or a tuple of them, indexing a batch's axes. | `draws[3]`, `draws[0:10]` | II.5's prose also says "selection", which goes |

`[]` takes a path, a list of paths, or, on a batch, a position. The three need no shared name, since one `[]` never mixes a path with a position (§ 4.1).
The public API gets no new noun for a list of paths: docstrings say "a path, or a list of paths" (decided 2026-10-18).
The word "selection" is avoided because `select` is already a method with another meaning; the glossary keeps only *selection order* (II.7).
The word **key** is not used for paths, since ProbPipe already uses it for random keys.
It keeps its Python sense of a mapping's keys, as in `keys()` or the keys of a rename mapping.
Today the glossary also uses it for a field's path.

### 2.5 The produced term, and columns

The **produced term** of a term is what one evaluation of it yields:

| Term | Its produced term |
| --- | --- |
| an array, an opaque value, or a record | the term itself |
| a law `d` | a random draw `X ~ d` |
| a kernel `k` | at each value `w` of its inputs, a random draw `X ~ k(w)` |
| a function `f` | at each value `w` of its inputs, its return value `f(w)` |
| a batch | the batch itself |

Here `w` is an **input record**: a record with one field for each input it binds.
The produced term of a law or a kernel is random, so a statement about it is a statement about the law of `X`.
Its type is the term spec its output spec states (§ 2.7).
A law's produced term is a draw, not the law, and a kernel's is a draw, not the law it gives; a joint law is a law over a joint draw, so this is the reading a merge needs.
An operation is a `Function`, so it has a produced term too.

The **column** of a batch at a path `p` is the batch, over the same levels, of each element's `[p]` (§ 4.2): the node at `p` for an element that is a value, and the view, kernel, or function at `p` for an element that is a law, a kernel, or a function.
A batch's paths are its elements' paths (§ 2.10), and the node of a batch at `p` is its column at `p`.

```python
sample(prior)                        # one draw of prior's produced term: Record {mu, tau}
sample(lik, given={"mu": 0.3})       # one draw of lik's produced term at mu = 0.3: an array
emulator(theta)                      # emulator's produced term at theta: Record {mean, var}
post_draws["mu"]                     # the column of post_draws at mu: a batch of arrays over the level sample
```

### 2.6 Inputs, input specs, and binding

An **input** is a named place where a term receives one value.
It has:
- a name, which is an identifier;
- a term spec, which types the values it admits;
- a flag: **required**, or **optional**.

A `Function`'s inputs are its parameters, and a `ConditionalDistribution`'s are what it conditions on.

**Binding** supplies values to a term's inputs by name: a call of a function, or `condition_on(k, given)` and the fused calls `sample(k, given=...)`, `log_prob(k, y, given=...)`, and `mean(k, given=...)` of a kernel (III.9).
A binding may omit an optional input, and the term then uses its own default.
Binding a kernel's last required input gives a law, whose unbound optional inputs take their defaults (III.9, IV.4): `condition_on(lik_opt, {"mu": 0.3})` is a `Normal` with `s = 1.0` (fact 24).
An input whose spec is a `RecordSpec` is one input: a binding fills it with one record, or binds part of it by paths into the record (III.9).
"Binds" keeps this sense, and a dimension's size is said to be bound, as II.1 says.

An **input spec** is an `InputSpec`: an ordered mapping from input names to term specs, together with the set of optional inputs.
Two input specs are equal when they have the same inputs, specs, and optional inputs, in any order, since binding reads inputs by name.
An input spec may have no inputs, and `InputSpec(RecordSpec())` is `InputSpec()`.
An `InputSpec` stores no defaults; the term that owns an optional input holds its default.

**Which inputs are optional** (§ 5.10, O9).
`conditional_distribution` and `Function` mark as optional each input whose parameter has a default (`_slots_of`, `distributions/_conditional.py:1366`, for kernels; O9 extends the rule to functions).
A source that marks optional an input whose parameter has no default raises `ValueError`, since the term has no default to use.
A kernel's inputs may all be optional, since III.9 requires only a non-empty `given_spec`; such a kernel is still a kernel, and `sample` asks for its inputs (facts 33 and 37).

```python
InputSpec(mu=(), s=()).with_optional("s")     # two inputs, s optional (the size shorthand is § 4.9)
lik.given_spec                                # InputSpec(mu=NumericArraySpec(())): one required input
lik_opt.given_spec                            # InputSpec(mu=..., s=...).with_optional("s")
condition_on(lik_opt, {"mu": 0.3})            # a law: Normal("y", 0.3, 1.0), with s at its default
```

**Change from today:** "input" replaces the design's "input slot", "slot", and "given slot", and the plan no longer uses "consumer" or "map-like kind" (§ 6.4).

### 2.7 Output specs and grammar B

An **output spec** is an `OutputSpec`.
It states two things about a term's produced term:
1. its **term spec**, read as `term_spec`;
2. a **name**, when that term spec is a base spec, or when it is pending in a type hole (below).

When the output spec is in the keyword form over a base spec, the name is the term's one output, and that output is **at the root**: the whole produced term is offered under it.
The constructor has two forms:
1. **the keyword form**, `OutputSpec(x=T)`, for a base spec `T`;
2. **the positional form**, `OutputSpec(T)`, for a spec with parts `T`, a `FunctionSpec` and a `BatchSpec` of base elements included (§ 2.11).

The term spec alone decides the form, so no flag records it.

**Grammar B.**
A name is stated outside a value only when the value is a base value, which has no paths for the name to disagree with.
A record carries its names inside it, as its children.
A law, a kernel, a function, or a batch carries no name of its own.
To give such a term one name, the code that returns it **wraps** it: it returns the one-field record `{name: term}`, declared `OutputSpec(RecordSpec(name=T))`.
ProbPipe never wraps a function body's return value itself (decided 2026-10-17); the body builds the record, and the output spec checks it.
The rule below states it, with its reason; § 7.1 relates grammar B to the grammars it replaces.

**The rule.** The keyword form may name only a base spec.
Anything with parts is offered by its parts, in the positional form, or wrapped in a one-field record that stores its name in the value.

**Why.**
When a name is outside a structured value, the term has two sets of paths.
Its value is indexed from inside, by `beta`, while merges and renames see it from outside, by `params/beta`.
The rule keeps every name inside the value, or names a value that has no paths at all, so there is one set of paths:
- the paths `[]` takes, rename keys, and draws agree (I1, I17, I18);
- no rename key can be ambiguous;
- the term spec alone fixes the form of an output spec, so an output spec needs no flag to record it.

jhuggins's model proves each of these (`graft_identity_B`, `no_ambiguous_key_B`, `B_decl_determined`).

**Views follow the same rule.**
`d[p]` names a base node by the last name of `p`, in the keyword form, and offers the parts of a node with parts, positionally.
`d[[p]]` is the wrap, which keeps the name (§ 5.5).

A **type hole** is the keyword form with `None` in place of the term spec, as `OutputSpec(mean=None)`: its name is stated, and its term spec is pending.
The function or law that owns it fills it with `with_term_spec` once it knows the type:
- filled with a base spec, it stays in the keyword form;
- filled with a batch built from raw values, whose elements a batch constructor names `element` by default (§ 5.8), it names the elements by the hole's name (O2);
- filled with any other spec with parts, it raises `TypeError`.

```python
OutputSpec(x=NumericArraySpec(()))                             # keyword form: one output x, at the root
OutputSpec(RecordSpec(beta=(), sigma=()))                      # positional form over a record
OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=())))   # the record, wrapped under params
OutputSpec(post.spec)                                          # positional form over a law: no name
OutputSpec(params=RecordSpec(beta=(), sigma=()))               # TypeError: "params would name a record; give
                                                               # OutputSpec(RecordSpec(...)) to offer its fields, or wrap it:
                                                               # OutputSpec(RecordSpec(params=...))"
OutputSpec(mean=None)                                          # a type hole
```

**Change from today:** the keyword form names any spec today, so the second-to-last line builds an interim whole form (§ 1.2), and the positional form refuses a `FunctionSpec` and a `BatchSpec` of base elements (`core/_specs.py:355`).

### 2.8 The output spec that a spec holds

Four specs with parts type a term that produces something, or holds elements, and each holds an output spec.
This plan writes `held(T)` for the output spec that `T` holds, and `with_held(T, E)` for `T` with `held(T)` replaced by the output spec `E` and everything else kept:

| `T` | `held(T)` | It describes | `with_held` keeps |
| --- | --- | --- | --- |
| `DistributionSpec` | `T.event_spec` | one draw | — |
| `ConditionalDistributionSpec` | `T.event_spec` | one draw, at one value of the inputs | `given_spec` |
| `FunctionSpec` | `T.output_spec`, which may be `None` | one return value | `input_spec` |
| `BatchSpec` | `T.element_spec`, the output spec of one element as a term (§ 2.9, § 5.8) | one element | the levels |

A `RecordSpec` holds no output spec, since its children are its structure, and a base spec holds none.
`held` and `with_held` are notation in this document; in code they are private helpers of `core/_specs.py`, which jhuggins's reply calls `produced_declaration` and `with_produced_declaration` (O11 keeps them private).

```python
held(post.spec)                                            # OutputSpec(RecordSpec(mu=..., tau=...))
with_held(post.spec, OutputSpec(mu=NumericArraySpec(())))  # the spec of a law whose draws are the array mu
held(post_draws.spec)                                      # OutputSpec(RecordSpec(mu=..., tau=...)): one element
```

### 2.9 The input spec and the output spec of each term

| Term `t` | Its input spec | Its output spec |
| --- | --- | --- |
| an array or an opaque value | none | none: a base value carries no name, and a label is not one (C5) |
| a record `r` | none | `OutputSpec(r.spec)`, derived and never stored (D6); none for an empty record |
| a law `d` | none (III.7) | `d.event_spec` |
| a kernel `k` | `k.given_spec`, with at least one input (III.9) | `k.event_spec` |
| a function `f` | `f.input_spec`, or `None` when undeclared | `f.output_spec`, or `None` when undeclared |
| a batch `bt` | its elements' input spec, read from the element's term spec: `given_spec` for a batch of kernels (III.10), `input_spec` for a batch of functions, and none for a batch of values or laws | `OutputSpec(bt.spec)`, derived; its outputs (§ 2.11) are its columns for a batch of records or of functions, one output at the root for a batch of base values, and none for a batch of laws or kernels |

In formulas, `input_spec(t)` is `t`'s input spec, or the empty `InputSpec()` when `t` has none; § 2.15's `names(I)` gives the set of an input spec's names.
`output_spec(t)` is `t`'s output spec.

**The element spec of a batch (§ 6.21).**
`bt.spec.element_spec`, and `bt.element_spec` on the batch, is the output spec of one element as a term, so `held(bt.spec) == bt.spec.element_spec`, and `element_spec.term_spec` is the element's term spec, which today's `BatchSpec.element_spec` holds:
- for an element that is a base value, which has no output spec of its own, the keyword form under the name its maker gave, or `element` (§ 5.8, O2, O3); this name is the one thing `element_spec` stores beyond the term spec;
- for an element that is a record, the record's own output spec, `OutputSpec(element_spec.term_spec)`;
- for an element that is a law, a kernel, or a function, the positional form over its spec, `OutputSpec(DistributionSpec(...))` and so on.

So for a batch of values, `bt.element_spec == output_spec(bt[j])` at every position `j`, with the batch supplying a base element's name.
For a batch of laws, kernels, or functions, each element's own output spec is the one its spec holds: `element_spec.term_spec.event_spec`, which III.10 gives as `DistributionBatch.event_spec` and `ConditionalDistributionBatch.event_spec`, or `element_spec.term_spec.output_spec` for functions.
The batch's own output spec, `OutputSpec(bt.spec)`, describes the batch as a produced term and offers its columns; `[]`, `t.output(c)`, and `InputSpec(o)` read it, while a merge of batches reads each element as one operand (§ 2.16).

**The attribute names.**
Each side has its attribute names, and each name says the role of what it describes:
- `event_spec` is the output spec of a law or a kernel, and describes a draw. A kernel and the function that returns its laws are one map `S → P(T)`, but their produced terms differ, a draw and a law, and the two names keep that difference visible.
- `output_spec` is the output spec of a function, and describes its return value.
- `element_spec` is the output spec of one element of a batch, as `event_spec` is of one draw.
- `given_spec` is the input spec of a kernel, named for what it is conditioned on, the word its own calls use: `condition_on(k, given)` and `sample(k, given=...)`.
- `input_spec` is the input spec of a function, and describes its arguments.

III.9 distinguishes conditioning from application, and D1 asks that a real distinction stay real in the interface, so the names stay.
Each is still an input spec or an output spec, and every rule below reads them as such.

**Disjoint sides.**
A kernel never has a name that is both an input and an output: `_check_disjoint_sides` refuses one at construction (`distributions/_conditional.py:131`).
A function does not check this today: `Function(lambda x: x, input_spec={"x": ...}, output_spec=OutputSpec(x=None))` builds.
Merge refuses such an operand (§ 2.17, M5).

```python
lik.given_spec                     # its input spec: the input mu
lik.event_spec                     # its output spec: OutputSpec(y=NumericArraySpec(())), one output y at the root
prior.event_spec                   # OutputSpec(RecordSpec(mu=..., tau=...))
post_draws.spec.element_spec       # OutputSpec(RecordSpec(mu=..., tau=...)): one element
post_draws.spec.element_spec.term_spec   # RecordSpec(mu=..., tau=...): today's BatchSpec.element_spec
```

### 2.10 The spec at a path

For a term spec `T` and a path `p`, the **spec at `p`**, written `spec_at(T, p)`, is a term spec.
Let `c` be the first name of `p`, and `rest` what follows `c/` in `p`, which is empty when `p` is `c`:
1. **`T` is a `RecordSpec`.** Let `child` be `T`'s child `c`; when `T` has no child `c`, `p` is not a path of `T`. When `rest` is empty, `spec_at(T, p)` is `child`. Otherwise `spec_at(T, p)` is `spec_at(child, rest)`.
2. **`T` holds an output spec `E` in the positional form.** Let `N = spec_at(E.term_spec, p)`. Then `spec_at(T, p)` is `with_held(T, E_p)`, where `E_p`, **the output spec of `E` at `p`**, is `OutputSpec(**{last(p): N})` when `N` is a base spec and `OutputSpec(N)` otherwise.
3. **In every other case**, `p` is not a path of `T`: a base spec has no paths; a held output spec in the keyword form over a base spec names its value at the root, with nothing below the name; and a `FunctionSpec` that declares no output has no paths known before it is called (§ 5.7).

The **paths** of `T` are the paths at which `spec_at` is defined.
The paths of an output spec `O` in the positional form are the paths of `O.term_spec`; an output spec in the keyword form over a base spec has none, since the name of an output at the root is not a path.
The paths of a term are the paths of its output spec.

Case 2 is how a path goes through every kind.
Below a held law, function, or batch, the spec at a path is the spec of a term of the same kind whose produced term is the node at that path.
So a law's paths are its draws' paths, a function's are its return values', and a batch's are its elements', since `held` of a `BatchSpec` describes one element as a term.

```python
R = RecordSpec(a=(), g=RecordSpec(c=(), e=()))
spec_at(R, "g")                                    # RecordSpec(c=(), e=())
spec_at(R, "g/c")                                  # NumericArraySpec(())
spec_at(post.spec, "mu")                           # DistributionSpec(OutputSpec(mu=...)): the spec of post["mu"]
spec_at(RecordSpec(n=(), G=post.spec), "G/mu")     # the same DistributionSpec, through the held law
spec_at(post_draws.spec, "mu")                     # the BatchSpec of the column mu, over the level sample
spec_at(Normal("x", 0.0, 1.0).spec, "x")           # undefined: x names the draw at the root, it is not a path
```

### 2.11 Outputs

The **value at an output** `c` of a produced term `v` is:
- `v` itself when `c` is at the root;
- the child `v[c]` when `v` is a record;
- the column at `c` when `v` is a batch;
- the function `x ↦` the value at `c` of `v(x)`, when `v` is a function.

A spec has outputs to offer when it is a non-empty record, or when it holds an output spec of what its term produces or holds (§ 2.8); the output `c` of a term is the term at that output (§ 4.11).
The **outputs** of an output spec `O` are names, and `O.outputs` maps each to its spec, in order:

| `O` | The outputs of `O` | The spec of the output `c`, `O.outputs[c]` |
| --- | --- | --- |
| keyword form, or a type hole, named `n` | the one name `n`, at the root | `O.term_spec`, or `None` in a type hole |
| positional over a `RecordSpec` `R` | its children's names, in `R`'s order | the child's spec |
| positional over a `BatchSpec` or a `FunctionSpec` `T` that holds `E` | the outputs of `E`; none when a `FunctionSpec` declares no output | `with_held(T, E_c)`, where `E_c` is `E` when `c` is `E`'s output at the root, and the output spec of `E` at the path `c` (§ 2.10) otherwise |
| positional over a `DistributionSpec` or a `ConditionalDistributionSpec` | none | — |

`outputs(t)` is the set of the outputs of `output_spec(t)`, empty when `t` has no output spec.
Only the keyword form admits a type hole, so a positional output spec always has a complete spec.

**The principle: the spec of an output is the spec of the produced term's value at that output.**
- For a record, the value at `c` is the child, so the spec of `c` is the child's spec.
- For a batch, the value at `c` is the column, so the spec of `c` is the column's `BatchSpec`.
- For a function as a produced term, the value at `c` is the output function, so the spec of `c` is its `FunctionSpec`.

**Why a function into records offers outputs, while a law over records does not.**
A function `f` that returns records `{mean, var}` is determined by its output functions: `f(x)` is `{mean: f_mean(x), var: f_var(x)}`, where `f_mean` is `x ↦ f(x)["mean"]`.
Splitting `f` into outputs therefore loses nothing, and a batch of records is likewise the record of its columns.
A law over records `{mu, tau}` is not determined by its marginals, since many joint laws share them.
So a produced term that is a law or a kernel offers no outputs (D1; § 6.20, Q2).
A batch of laws offers none either, since its `element_spec` describes an element as a law, positionally.
A law's own `event_spec` is unaffected: it describes a draw, not a law, so `prior.event_spec` offers `mu` and `tau`.

**The output at the root.**
`O` has an output at the root when it is in the keyword form, is a type hole, or is positional over a `BatchSpec` or a `FunctionSpec` that holds an `E` with an output at the root, under the same name.
Every other output of `O` is a path of `O` of one name.
**Grammar B's guarantee:** an output spec offers one output at the root and no paths, or outputs that are all paths of one name, never both.

```python
prior.event_spec.outputs                  # {mu: NumericArraySpec(()), tau: NumericArraySpec(())}
Normal("x", 0.0, 1.0).event_spec.outputs  # {x: NumericArraySpec(())}: one output, at the root
grouped.event_spec.outputs                # {a: NumericArraySpec(()), g: RecordSpec(c=(), e=())}
OutputSpec(post.spec).outputs             # {}: a produced term that is a law offers no outputs
emulator.output_spec.outputs              # {mean: ..., var: ...}
OutputSpec(post_draws.spec).outputs       # {mu: <the BatchSpec of the column mu>, tau: ...}: the columns
fit_emulator.output_spec.outputs          # {mean: FunctionSpec(InputSpec(theta=...), OutputSpec(mean=...)), var: ...}
```

**Change from today:** a law's positional outputs are typed today by one draw's value, as `OutputSpec(prior.spec).components` is `{mu: NumericArraySpec(()), tau: ...}`, and a batch's by its element's (fact 11).

### 2.12 The record of outputs, and record notation

This notation states a merge's value (§ 2.20).
Let `O` be an output spec in the keyword form or positional over a `RecordSpec`, and `v` a value of its term spec:
- `rec_O(v)` is the record with one field per output of `O`, holding the value at that output (§ 2.11): `v` itself for a record, and `{n: v}` for an output at the root named `n`;
- `dec_O(z)` is the inverse of `rec_O`: the field `n` of `z` for an output at the root named `n`, and `z` itself for a record.

For records `r` and `q` and a set of names `N`:
- `r↾N` is the record of the top-level children of `r` whose names are in `N`, in `r`'s order;
- `r ⊎ q` is the record of `r`'s top-level children followed by `q`'s, defined when no top-level name is in both.

```python
rec_O(0.3)            # for O = lik.event_spec, named y: Record {y: 0.3}
dec_O({"y": 0.3})     # 0.3
rec_O(sample(prior))  # for O = prior.event_spec: the draw {mu, tau} itself
```

### 2.13 Feeds

An output `c` of a term `b` **feeds** the input `c` of a term `a`, or the unfed input `c` of one of `a`'s factors (§ 2.15), when they share the name `c` and `a & b` is formed (§ 2.15): the value at `b`'s output `c` becomes the value received at `c`.
In `a & b` only the right operand's outputs feed, and only the left operand's inputs are fed (§ 2.21 says why).
"Feeds" is the one word for this relation; "meets", "is matched", "wires", "connects", and IV.2's "bound" are retired.

```python
inputs(lik) ∩ outputs(prior)        # {"mu"}: prior's output mu feeds lik's input mu in lik & prior
```

### 2.14 Dimension scopes, conformance, and unification

This plan uses one word per relation between specs: **conforms** for the directional check that a value of one spec is admitted by another, and **unify** for the symmetric relation whose result is the spec of a shared input.
§ 6.5.4 lists the passages of `design/` that S0 rewords to match.

**Dimension scope (II.1, unchanged).**
In one scope, a dimension name's size is bound by its first occurrence against a size, and a later occurrence that disagrees raises.
At one axis, two equal sizes agree, the same name agrees with itself, a name and a size bind the name, and two different sizes, or two different free names, raise `ValueError` (`_unify_array_shape`, `core/_spec_base.py:550`).
The scope's **substitution**, written `β`, replaces each bound name by its size, as `with_dim_sizes` does.
A merged law or kernel keeps its scope's bindings, as today's joint does (`_FactorGraph.scope`, `distributions/_factored.py:241`), and a later merge starts from the union of its operands' stored scopes, raising when they bind one name to two sizes (`_flattened`, `distributions/_factored.py:269`).
A merged input spec, output spec, record spec, or function stores no scope: `β` is applied to its specs and the bindings are not kept.
§ 4.17 states the consequences of both rules.

**Conformance (directional).**
A term spec `A` **conforms** to a term spec `B` when every value of type `A` is one that `B` admits, in one scope:
1. **arrays:** equal ranks, each axis agreeing by the scope rule above, and, when both set a dtype, `A`'s casting into `B`'s by NumPy's same-kind rule; supports are not compared, since a support is descriptive (III.5), as today;
2. **opaque values:** `B`'s type is `None` or a superclass of `A`'s, and `B`'s `meta` is `None` or equal to `A`'s (III.2);
3. **records:** the same child names, and each child of `A` conforms to the child of `B`;
4. **output specs:** the same form and name, and `A`'s term spec conforms to `B`'s; a type hole in `B` admits any term spec;
5. **laws:** `A`'s event spec conforms to `B`'s, so laws are covariant in what they produce;
6. **kernels and functions** (III.9's variance): the same input names; each input spec of `B` conforms to `A`'s input spec of the same name, and an input optional in `B` is optional in `A`, so inputs are contravariant; `A`'s output spec conforms to `B`'s, so outputs are covariant; a side that `B` leaves `None` admits any side of `A`, and a side that only `A` leaves `None` does not conform;
7. **batches:** the same levels, and `A`'s element term spec, `element_spec.term_spec`, conforms to `B`'s; a base element's name is not compared, as § 5.8 keeps names out of a batch's equality.

II.1's "binding by unification against a spec" is this relation, and today `_unify_specs(B, A, ...)` checks its array case (`core/_spec_base.py:525`); the joint calls it for each fed input with the output's spec as `A` (`distributions/_factored.py:437`).

**Unification (symmetric; IV.2's sense).**
Two term specs **unify** when their unification `U(A, B)` below exists, in one scope.
`U(A, B)` is the spec of a shared input, which "admits only the values every consuming factor admits" (IV.2), and it is defined by kind:
1. **arrays:** equal ranks and each axis agreeing by the scope rule; a dtype that only one sets, and, where both set one, the dtype of the spec that conforms to the other, the narrower by NumPy's safe cast when each conforms to the other (`float32` with `float64` gives `float32`), with no unification when neither conforms; a support only when both set the same one, and none otherwise;
2. **opaque values:** III.2's rule: the types are equal or one is `None`, and so are the `meta`s; `U` takes the known type and the set `meta`;
3. **records:** the same child names, child by child;
4. **output specs:** the same form and name, and `U` of the term specs; a type hole takes the other's term spec;
5. **laws:** `U` of the event specs;
6. **kernels and functions:** the same input names, with equal input specs and optional flags, and `U` of the output specs; a side that one leaves `None` takes the other's;
7. **batches:** the same levels, and `U` of the element term specs.

Any other pair, such as an array spec with a record spec, does not unify.
`U` is symmetric, and associative on the specs it accepts.
A spec `C` conforms to `U(A, B)` exactly when `U(A, B)` exists and `C` conforms to both `A` and `B`, provided no two different free dimension names occur at one axis among the three (§ 4.15 uses this; a probe over the standard NumPy dtypes checks the dtype case, fact 38).

**Change from today:** today's joint keeps the first operand's spec for a shared input instead of `U` (fact 8); since it checks a value bound to a shared input against every factor that consumes it, `U` changes only the reported `given_spec`, not which merges are valid (fact 39).

A spec declared by the size shorthand sets no dtype and no support, so it conforms to, and unifies with, a spec of any dtype and support of the same shape.

```python
NumericArraySpec((3,), dtype="float32")    # conforms to NumericArraySpec(("p",)), binding p to 3
NumericArraySpec((), dtype="int32")        # conforms to NumericArraySpec((), dtype="float32"): same-kind cast
NumericArraySpec((), dtype="float32")      # does not conform to NumericArraySpec((), dtype="int32")
NumericArraySpec((4,))                     # does not conform to NumericArraySpec((3,)): two different sizes
NumericArraySpec(("m",))                   # does not conform to NumericArraySpec(("n",)): two different names (II.1)
# U(NumericArraySpec((), dtype="float32"), NumericArraySpec((), dtype="float64")) == NumericArraySpec((), dtype="float32")
# U(NumericArraySpec((), dtype="float32"), NumericArraySpec((), dtype="int32")) == NumericArraySpec((), dtype="int32")
# U(OpaqueSpec(type=str), OpaqueSpec()) == OpaqueSpec(type=str)
```

### 2.15 Merge: the operator, factors, and the unfed inputs of factors

Today `*` merges laws and kernels: `lik * prior` is the joint law.
`*` is also arithmetic: it multiplies two `NumericArray`s, scales a `LinOp` by a number, and scales a Gaussian random function by a number.
So `d1 * d2` merges two laws, while `sample(d1) * sample(d2)` multiplies the two draws when both events are arrays.
On a `NumericRecord`, `*` is only scaling by a number: III.5 gives records the vector-space operators, `+` and `-` between records of one schema, scalar `*` and `/`, and `map`, so `r * 2` scales while `r * r` raises `TypeError`. The code does not implement these operators yet, and `&` is not among them.
`&` reads as "both", and it replaces `*` for merging laws and kernels (decided 2026-10-17), with no alias for `*`.
What `&` does is called a **merge** for every kind (decided 2026-10-18), and "composition" keeps its meaning of `f ∘ g` and of a product of linear operators `A B`; today the design reference calls both "composition" (`design/README.md`, the operations table).
`*` keeps its arithmetic meanings, and `GaussianRandomFunction.__mul__` keeps only its scaling branch.
Pending the maintainer's decision: whether `*` between two laws or kernels becomes a permanent refusal, whose `TypeError` says "* does not merge laws; write lik & prior", and whether a law and a kernel refuse `bool()`, so that `lik and prior` raises instead of silently returning `prior`; see `decisions.md`.

`a & b` combines two terms, or two specs, by the names of their inputs and outputs.
It is written right to left, as a density is: `lik & prior` is `p(y | mu) · p(mu, tau)`.
This section and the next five define it once, for every kind: which operands merge (§ 2.16), eight conditions (§ 2.17), merge on specs (§ 2.18), the result's inputs and outputs (§ 2.19), and the value (§ 2.20).

**Factors** (IV.1, IV.2).
A merge of laws and kernels reads each operand as its factors, as IV.2's flattening does today (`_flattened`, `distributions/_factored.py:269`).
The factors of a law or a kernel `t`, written `factors(t)`, are the factors of a factored joint, apart from a grouped sub-joint, which stays one factor (IV.2); every other law or kernel is one factor.

**The unfed inputs of the factors.**
`unfed(t)` is the input spec of the inputs that `t`'s factors take and no factor of `t` feeds, optional ones included, in order of first appearance.
It is a description that the merge rules read, not an attribute and not an input of `t`:
- for a kernel, `unfed(t)` is its `given_spec`;
- for a law built by a distribution family, it is empty;
- for a law that is a joint, it holds the optional inputs its factors left at their defaults: `unfed(lik_opt & prior)` holds `s`, although the law has no inputs (III.7);
- for a grouped sub-joint used as one factor, it is that factor's own `given_spec` when the sub-joint is a kernel, and empty when it is a law, so the optional inputs inside it are not read (fact 43).

For a record, a function, or a spec operand, `unfed(t)` is its input spec: none for a record, `f.input_spec` for a function, and `I` itself for an input spec.
A spec, and an element of a batch in a batch merge, is read as one factor, so `unfed` of a `DistributionSpec` is empty.
For a spec operand, `outputs(O)` is the set of an output spec `O`'s outputs, `outputs(R)` the set of a record spec `R`'s top-level names, and `outputs(I)` is empty.

In `a & b`, writing `names(I)` for the names of an input spec `I`:
- `fed = names(unfed(a)) ∩ outputs(b)` is the set of names that the right operand feeds (§ 2.13);
- `shared = (names(unfed(a)) − fed) ∩ names(unfed(b))` is the set of names both operands' factors take and neither feeds;
- `β` is the substitution of the merge's dimension scope (§ 2.14), which starts from both operands' stored scopes, if any, and adds every binding that M6 and M7 make.

### 2.16 Which operands merge

`&` merges two operands of one **merge category**, a grouping used only by this section's rules:

| Merge category | Operands | Result | Step |
| --- | --- | --- | --- |
| records | `Record`, `NumericRecord` | a record | S2b makes `merge` shallow; S3 renames it `&` |
| laws and kernels | `Distribution`, `ConditionalDistribution` | a law, or a kernel when a required unfed input is left | S3 (today's `*`) |
| functions | `Function` | a function | after the workshop |
| batches | two batches over equal levels whose elements merge | a batch over the same levels | `RecordBatch`: S2b rule, S3 operator; other batches after the workshop |
| input specs | two `InputSpec`s | an `InputSpec` | S2b |
| output specs | two `OutputSpec`s | an `OutputSpec` | S2b |
| record specs | two `RecordSpec`s | a `RecordSpec` | S2b rule, S3 operator |
| specs of laws, kernels, functions, and batches | two of one of these | the spec of the merged term | after the workshop |

Operands of two merge categories raise `TypeError`, so `InputSpec & OutputSpec` raises; to read an output spec as inputs, write `InputSpec(o)` (§ 6.20, Q3(ii)).
An array and an opaque value belong to no merge category: they offer no output, and `&` on two `NumericArray`s keeps NumPy's elementwise meaning (O7).

**Batches merge by their elements.**
Two batches have **equal levels** when they have the same level names, in the same order, with the same axis sizes, as `RecordBatch.merge` requires today (`core/_record_batch.py:783`).
For such batches, `a & b` is the batch, over the same levels, whose element at each position `j` is `ã_j & b̃_j`, where an element read as an operand `x̃_j` is:
- for an element that is a base value, the record `rec_E(x_j)` under the batch's `element_spec` `E` (§ 2.12), so it merges as a record;
- for a record, the record itself;
- for a law, a kernel, or a function, the element as one factor, with the input spec and output spec of the term spec `element_spec.term_spec` (§ 2.9).

The merge category, the conditions M2 to M8, `fed`, `shared`, and the result's inputs and outputs are those of the element merge, read once on those specs.
A batch's own outputs, its columns, serve `[]`, `t.output(c)`, and `InputSpec(o)`, and take no part in a merge.
Elements of two merge categories, such as records with laws, raise `TypeError`.
Today `&` on two batches of arrays raises `TypeError` (fact 44); under this plan they merge into a batch of records (§ 6.20, Q4(a)).

**Merges across categories are design for after the workshop.**
A record is the law that gives one draw, itself, with probability one, and a function is the kernel whose law at each input is a point mass at its return value.
By that reading, `lik & Record({"mu": 0.3})` and `f & d` have a meaning.
A law with a point-mass factor has no joint density, and IV.1's factors admit only laws and kernels, so these merges raise `TypeError` now, and `deferred.md` records the point-mass reading.

### 2.17 The conditions

`a & b` raises unless all of these hold:

| # | Condition | Error | Why |
| --- | --- | --- | --- |
| M1 | `a` and `b` are of one merge category (§ 2.16); two batches have equal levels and elements of one category | `TypeError`, which for an input spec with an output spec says to write `InputSpec(output_spec)` (§ 5.10, message 4); `ValueError` for unequal levels | Each category has its own value rule (§ 2.20) |
| M2 | **Complete specs, and produced terms that a record can hold.** Each law, kernel, or function operand states an output spec with no type hole, and a function also states its input spec. Each output spec is in the keyword form over a base spec or positional over a `RecordSpec`, so its produced term is a base value or a record. A record passes. A spec operand has no type hole | `TypeError`, which names the type hole or the missing spec, or says to wrap the output under a name | The result stores the value at each output as a field and must rebuild each operand's produced term from it (§ 2.21) |
| M3 | No name is an output of both: `outputs(a) ∩ outputs(b) = ∅` | `ValueError`, naming the output and both operands; when both operands are views of one parent, it adds "both are views of one law; for their joint law index once, as hier['s1', 's2']; merging two views treats them as independent" (§ 2.21) | One value needs one source |
| M4 | **The order rule:** the right operand's factors take no unfed input, required or optional, that the left one outputs: `names(unfed(b)) ∩ outputs(a) = ∅` | `ValueError`, which says to put the operand that outputs the name on the right, as today (`distributions/_factored.py:431`) | It keeps the result's sides disjoint and the written order a topological order of the factors (§ 2.21) |
| M5 | Each operand's unfed inputs and outputs are disjoint: `names(unfed(t)) ∩ outputs(t) = ∅` | `ValueError`, which says to rename one | Whether a name is fed or shared is then decided by names alone; III.9 already requires it of kernels |
| M6 | For each `c` in `fed`, the spec of `b`'s output `c` conforms to the spec at `c` in `unfed(a)` (§ 2.14) | `ValueError`, naming both specs | The input must admit every value it is fed |
| M7 | For each `s` in `shared`, `U` of the specs at `s` in `unfed(a)` and `unfed(b)` exists (§ 2.14) | `ValueError`, naming both specs | One value fills both inputs, so both must admit it |
| M8 | All specs of both operands form one dimension scope, with any stored scopes: no dimension name is bound to two sizes, and no two different free names occur at one axis (II.1) | `ValueError`, which names the dimension and both sizes; when a spec shows the name, it says to rename one apart with `with_dim_names`; when only an operand's stored scope holds it, it says so, names that operand, and suggests merging its parts in another order, since `with_dim_names` cannot rename a stored name (§ 4.17, item 2) | A dimension name is one quantity (II.1) |

For a merge of batches, M2 to M8 read the elements' specs (§ 2.16).

### 2.18 Merge on specs

**`O & P` on output specs** is the output spec of two side-by-side terms, with nothing fed, since an output spec has no inputs:

```text
O & P  =  OutputSpec(RecordSpec(O.outputs followed by P.outputs))
```

It applies M2 (each output spec in the keyword form over a base spec or positional over a `RecordSpec`, with no type hole), M3, and M8.
An output at the root, named `n`, becomes the child `n`, so its value moves from the root to the path `n`.
A record contributes its children unchanged, so every path of an operand is a path of the result with the same spec (`graft_identity_B`).

**`I & J` on input specs** is the input spec of two terms that share their inputs, with nothing fed, since an input spec has no outputs.
Both operands are `InputSpec`s (§ 6.20, Q3(ii)).
The inputs of either appear in the order of first appearance; a shared input carries `U` of its specs (M7) and is optional only when optional in every operand that has it; the operands share one dimension scope (M8), whose bindings the result does not keep (§ 4.17).
`&` replaces #571's proposed `InputSpec.merge`, which was never built, and the joint's unfed inputs follow the same rule, so the two share one implementation.

**`R & S` on record specs** is the union of the children; a top-level name in both raises (M3).
**`&` on `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, and `BatchSpec`** is the spec of the merged term, after the workshop (§ 6.16).
So `(a & b).spec == a.spec & b.spec` holds for records, and for the other categories under § 4.16's condition.

For laws and kernels, R1 and R2 read:

```python
(a & b).event_spec == β(a.event_spec & b.event_spec)
unfed(a & b) == β(unfed(a).without(*fed) & unfed(b))   # the given_spec, when the result is a kernel
```

**Change from today:** the joint keeps the first operand's spec for a shared input (`distributions/_factored.py:424`; fact 8), not § 2.14's `U`; one test asserts today's choice (fact 35).

```python
OutputSpec(x=NumericArraySpec(())) & grouped.event_spec
# OutputSpec(RecordSpec(x=..., a=..., g=RecordSpec(c=..., e=...)))
grouped.event_spec & OutputSpec(a=NumericArraySpec(()))     # ValueError (M3): a is an output of both
OutputSpec(post.spec) & prior.event_spec                    # TypeError (M2): the first produces a law; wrap it
InputSpec(theta=("p",), x=()) & InputSpec(theta=(3,), z=())
# InputSpec(theta=NumericArraySpec((3,)), x=..., z=...): theta unifies, and p is bound to 3 in both
InputSpec(a=("n",)) & InputSpec(a=(2, 2))                   # ValueError (M7): the ranks differ
InputSpec(o) & InputSpec(p)                                  # how to merge the inputs two output specs o and p feed
```

### 2.19 The result

| # | Part | Value |
| --- | --- | --- |
| R1 | Outputs | `output_spec(a & b) = β(output_spec(a) & output_spec(b))` (§ 2.18): `a`'s outputs, then `b`'s; an operand with no output spec, such as an empty record, contributes none; for batches, this holds of the element merge (§ 2.16), and the result's `element_spec` is the element merge's |
| R2 | Unfed inputs | `unfed(a & b) = β(unfed(a).without(*fed) & unfed(b))` (§ 2.18): every unfed input of the factors that no output feeds, optional ones included, `a`'s first; a shared one carries `U` of its specs and is optional only when optional in every operand that has it; a fed one leaves, whether it was required or optional; for batches, this holds of the element merge (§ 2.16) |
| R3 | Kind and inputs | records give a record; laws and kernels give a kernel whose `given_spec` is `unfed(a & b)` when a required unfed input is left, and otherwise a law, which has no inputs and whose factors take their optional inputs at their defaults when it is sampled or scored; functions give a function whose `input_spec` is `unfed(a & b)`; batches give a batch over the same levels whose elements are the element merges |
| R4 | Factors and scope | for laws and kernels, `factors(a)` followed by `factors(b)`, and the scope of § 2.15, which the result stores |
| R5 | Value | by category (§ 2.20) |
| R6 | Label | the operands' labels joined with `·` (IV.2; O12); no merge reads a label |

R3 follows IV.2's table, so a kernel whose inputs are all optional is a kernel alone, while a merge that leaves only its optional inputs is a law: `k_opt & Normal("z", 0.0, 1.0)` samples at `s = 1.0` although `sample(k_opt)` raises (fact 37). This is today's behaviour, and § 6.3 lists it.

**A shared optional input that a binding omits.**
One value fills a shared input when a binding supplies it.
When a binding omits a shared optional input, each factor takes its own default, since a default belongs to the term that owns it and an `InputSpec` stores none (§ 2.23, row 11).
This is today's behaviour (fact 26), and § 6.3 lists it.

```python
model = lik & prior
# fed = {mu}; outputs y, mu, tau; no unfed input left, so a law
data_model = lik_nee & lik_lai
# fed = {}; shared = {theta}: one input theta, U of its specs; outputs nee, lai: a kernel
list(data_model.given_spec)                      # ["theta"]
data_model.given_spec == lik_nee.given_spec & lik_lai.given_spec
lik_opt & prior                  # mu is fed; s stays unfed and optional: a law, with no inputs, sampled at s = 1.0
(lik_opt & prior) & s_prior      # s is fed now: y's spread follows s near 100, as today (§ 4.16)
prior & lik                      # ValueError (M4): lik takes mu, which prior outputs; write lik & prior
s_prior & (lik_opt & prior)      # ValueError (M4): lik_opt's unfed input s is an output of s_prior, on the left
Normal("x", 0.0, 1.0) & Normal("x", 2.0, 1.0)    # ValueError (M3): x is an output of both
lik & Normal("mu", jnp.zeros(3), 1.0)            # ValueError (M6): the output mu, of shape (3,), does not conform to lik's input mu
lik & Record({"mu": 0.3})                        # TypeError (M1): a kernel and a record are of two merge categories
echo = Function(lambda x: x, input_spec={"x": ()}, output_spec=OutputSpec(x=()))
twice = Function(lambda y: 2 * y, input_spec={"y": ()}, output_spec=OutputSpec(w=()))
echo & twice                                     # ValueError (M5): echo takes and outputs x   (after the workshop)
```

### 2.20 The value of a merge

Let `u` be an input record of `a & b`: a record with one field for each input of `a & b` that the binding supplies.
For a law, a kernel, or a function `t` and an input record `w`, `t(w)` is the law `t` gives at `w`, or the value it returns, where each factor fills each unfed input that `w` does not supply with its own default.
`x ~ L` says that `x` is a draw of the law `L`, and `p_t(x ; w)` is the density of `t(w)` at `x`.
`rec_a` and `dec_a` are `rec` and `dec` of `output_spec(a)` (§ 2.12), and likewise for `b`; `u↾N` and `⊎` are § 2.12's record notation, for a set of names `N`.

**Laws and kernels.**
A draw `z` of `a & b` at `u` is a record with the fields `outputs(a)` followed by `outputs(b)`.
Its law is defined by sampling the right operand first, since nothing feeds it:

```text
z_b ~ b( u↾names(unfed(b)) )
z_a ~ a( u↾(names(unfed(a)) − fed)  ⊎  rec_b(z_b)↾fed )
z   = rec_a(z_a) ⊎ rec_b(z_b)
```

Where both operands have densities, the same law is:

```text
p_{a&b}(z ; u)  =  p_a( dec_a(z↾outputs(a)) ; u↾(names(unfed(a)) − fed) ⊎ z↾fed )  ·  p_b( dec_b(z↾outputs(b)) ; u↾names(unfed(b)) )
```

IV.2 writes this as `p(F_A, F_B | unmet) = p_A(F_A | bound ∪ (G_A − F_B)) · p_B(F_B | G_B)`, where `F` and `G` are an operand's outputs and its factors' unfed inputs, `bound` is `fed`, and `unmet` is the result's unfed inputs; S0 writes `unfed` for `unmet` (§ 6.5.4).
Today's joint computes this on its sampling and scoring routes, over the factors of `_factor_graph` (`distributions/_factored.py:351`), with `_components_of` and `_event_of` (`:544`, `:557`).
Its exact conditioning route does not: `condition_on(lik_opt & s_prior, {"mu": 0.0})` draws `y` with `s` at its default though `s_prior` feeds `s` (fact 34). That is a bug under this definition and under today's IV.2; `deferred.md` records it, and S3's mechanical rename carries it unchanged.

```python
model = lik & prior
# z_prior ~ prior: {mu, tau};  z_lik ~ lik(mu = z_prior["mu"]): an array y
# a draw of model: {y, mu, tau};  p(y, mu, tau) = p_lik(y ; mu) · p_prior(mu, tau)
```

Records, functions, and batches read this rule for their kind: records have no inputs, so nothing is fed (§ 5.4); a function is the kernel whose every law is a point mass (§ 5.7); and two batches merge element by element (§ 5.8).

### 2.21 Why each rule

- **M2.** The merged produced term holds, at each output, the value there, and each operand's produced term must be rebuilt from those values, since a joint scores each factor's density on that factor's own draw (IV.2). A base value and a record are rebuilt without loss, by `dec`. A law, a kernel, a function, or a batch held under no name is not a record, so the merge would have to split it into its outputs or invent a name, and the caller wraps it instead, keeping it whole under a name of its choosing. M2's reason is this rebuilding; it does not rest on the rule that a law offers no outputs.
- **M3, and shared inputs.** An output offers a value, so two operands cannot offer one name: the result would have two values for it. An input asks for a value, and one value can fill the input of every operand that has it.
- **M4.** Feeding runs only from right to left. If `b`'s factors took an unfed input that `a` outputs, `a & b` would leave that input unfed while `a` outputs its name, so the result would have one name on both sides (I15). With M4, in a chain `a & b & c` each operand is fed only from its right, so the written order is a topological order of the factors, and the density reads left to right. Since feeding is one-way, two operands could never feed each other anyway, so preventing such a cycle is not M4's job. M4 also does not let merge build every acyclic graph module by module: with `cb = c & b`, for `c(z | y)`, `b = Normal("x")`, and `a(y | x)`, both `a & cb` and `cb & a` raise, while `c & a & b` builds the same graph from its factors (fact 29). IV.2 states the rule as `G_B ∩ F_A = ∅`, and the code enforces it, optional inputs included (fact 33).
- **IV.2's rationale.** IV.2 says that associativity rests on M4. jhuggins's Rocq model finds that a chain of kernels has the same inputs, outputs, and validity in every grouping with or without that rule, and associativity's proof does not use it (`inputs-and-outputs.md` Appendix A). The rule's purpose is the one above: it keeps the joint well formed, with the conditional-first listing a topological order of the dependence graph. Step S0 rewrites IV.2's rationale to say so, with § 4.15's statement of when validity can differ between groupings.
- **M5.** With M5, a name of `a` is fed exactly when `b` outputs it, shared exactly when `b`'s factors take it, and the two never overlap.
- **M6 is directional.** The output's values are given, and the input must admit each of them, so the output's spec conforms to the input's.
- **M7 is symmetric.** One value fills a shared input, so neither operand's spec is privileged.
- **M8.** A dimension name means one size wherever specs share one scope (II.1).
- **A merge never lifts.** An output feeds an input only when its spec conforms. Lifting the receiving term over a law-valued output would ignore that law's dependence on everything else the receiver gets, which only the merge itself knows. A lifting rule inside a merge stays after the workshop (§ 6.16).
- **Records merge shallowly** (§ 6.20, Q4(a)). A group is a value too. If each operand produced part of a group `g`, the value at `g` would have two sources, which M3 forbids at `g`. A merge of laws is already shallow: two laws that both output a group `g` raise today (`distributions/_factored.py:402`), while `NamedTree.merge` and `Record.merge` merge groups deeply (`core/named_tree.py:835`, `core/record.py:931`). Keeping both rules would let `sample(a) & sample(b)` succeed where `a & b` raises. A deep merge is still written by hand: `r1.without("g") & r2.without("g") & Record({"g": r1["g"] & r2["g"]})`.
- **Two views of one parent merge as independent factors.** `hier[["s1"]] & hier[["s2"]]` is the product of the two marginals, as `hier["s1"] * hier["s2"]` is today, and the joint law of both groups is `hier["s1", "s2"]`. A merge reads names and factors, and a view is one factor, so it cannot tell two views of one parent from two independent laws. § 6.3 lists this, and `deferred.md` records refusing such a merge as a possible later rule. `hier["s1"] & hier["s2"]` raises M3, since both offer `theta`, where today each view is a whole term named after its group. Wrapping the views is the natural fix for that error, and it silently drops the dependence between the groups, so M3's message, when both operands are views of one parent, points to `hier["s1", "s2"]` (§ 2.17).

### 2.22 What merge is not

- **Not composition.** `f ∘ g` passes `g`'s whole result to `f` by position and returns only `f`'s result, and the product of linear operators `A B` keeps the word "composition" too. A merge matches by name and returns both operands' outputs.
- **Not conditioning.** `condition_on(lik, {"mu": 0.5})` binds `mu` to a fixed value, and the result does not produce `mu`. In `lik & prior`, `mu` is fed by a law and stays an output.
- **Not renaming.** `joint(A, B, **renames)` stays `A & B.rename(**renames)`.

### 2.23 The two sides compared

Each concept on one side has its counterpart on the other, with the same structure, apart from the differences this table states.

| # | Input side | Output side | Why they differ |
| --- | --- | --- | --- |
| 1 | an input | an output | — |
| 2 | `InputSpec` | `OutputSpec` | — |
| 3 | several operands of a merge may share an input, which becomes one input | no two operands of a merge have one output | one value can serve many receivers, while one value needs one source |
| 4 | the left operand's inputs are fed | the right operand's outputs feed | a density is written `p(y \| mu) · p(mu)`, and a fixed direction makes the written order a topological order of the factors (§ 2.21) |
| 5 | names are identifiers | names are any names | an input binds a Python parameter, while an output becomes a record field; an output feeds an input only when its name is also an identifier, which `InputSpec(o)` checks |
| 6 | every input is named | only a base value is named from outside, at the root | an argument is always passed by name, while a produced base value carries no name (grammar B) |
| 7 | one form | two constructor forms, keyword and positional | follows from row 6 |
| 8 | flat names; a path reaches into a record input (§ 5.10) | an output is at the root or at a top-level path, with paths below it | a path is a place inside a produced term, while an input receives one value, part of which a path into a record input binds (III.9) |
| 9 | `I[name]` is a term spec, and `InputSpec` is a `Mapping` | `O[path]` is an output spec, `O.outputs[name]` is a term spec, and `OutputSpec` is not a `Mapping` | the node at a path of a produced term is itself described by an output spec, which needs a name when it is a base value; an input's value already has its input's name, and the input names are the keys |
| 10 | an input may be optional, and its default belongs to the term that owns it | an output is always present | a produced term has a value at every output; an `InputSpec` stores no default |
| 11 | a shared optional input that a binding omits takes each operand's own default | — | row 10: the defaults belong to the operands and cannot be compared; today's `*` does the same (fact 26) |
| 12 | no type holes | type holes in the keyword form | a function or a law fills its output's type once it runs, while an input's type must be known to check a binding |
| 13 | equality ignores order | equality keeps order | binding reads inputs by name, while output order is the layout `to_vector` reads |
| 14 | an input spec may have no inputs | an output spec's term spec is never an empty record, and `without` never removes every node | a term with no inputs is a law or a value, while an empty record describes nothing produced |
| 15 | a merge of laws and kernels also reads the unfed inputs of its operands' factors, which for a law are not inputs of the law (§ 2.15) | — | today's flattening (IV.2), kept before the workshop; § 4.16 states its consequences |
| 16 | a shared input's specs unify, symmetrically | a fed output's spec conforms to the input's, directionally | one value fills both inputs, so neither side is privileged; a fed value is given, and the input must admit it |
| 17 | a batch's inputs are its elements' inputs | a batch of records or of functions offers its columns, a batch of base values one output at the root, and a batch of laws or kernels none (§ 2.11) | one binding fills the same input of every element, while a produced batch is offered column by column; a merge of batches reads the elements on both sides (§ 2.16) |
| 18 | `InputSpec(o)` reads an output spec as the inputs it would feed | nothing reads an input spec as outputs | an output spec states a produced term, from which inputs follow; an input spec states no produced term, so no output spec follows from it |
| 19 | `InputSpec.rename` renames names (§ 5.10) | `OutputSpec.rename` renames and moves nodes by their paths, and renames the output at the root (§ 4.8) | row 8: inputs are flat, while outputs have paths below them |
| 20 | `given_spec`, `input_spec` | `event_spec`, `output_spec`, `element_spec` | each attribute names its kind's role (§ 2.9) |

The two sides share everything else: `[]` with one name or a list (§ 5.10, § 4.4), `without` (§ 5.10, § 4.5), `&` (§ 2.18), membership (§ 4.6), the size shorthand, and one dimension scope.

## 3. Invariants

The design must guarantee twenty-two properties.

**Symbols.**
`d` is a law, `t` a term other than an array, `f` a function, `k` a kernel, `bt` a batch, and `r` a record.
`a`, `b`, and `c` are operands of one merge category (§ 2.16).
`s` is a path or a list of paths, `p` and `q` are paths, `ps` is a list of paths that is not empty, has no path inside another, and leaves something out, and `j` is a position.
`I` is an input spec, `ns` a list of its input names, and `name` an output name.
`w` is an input record (§ 2.5), and `X` a draw (§ 2.5).
`given` is a mapping from paths of `d`'s event to values, and `fixed` is the set of those paths.
`m` is a rename, `key` a `str`, and `fed`, `unfed`, `names`, and `β` are as § 2.15 defines them.

The third column says whether the definitions imply the invariant ("follows") or this design chooses it ("rule"); the fourth names the step that delivers it.
§ 6.3 lists the cases where an invariant does not hold on purpose.

| # | Invariant | Follows or rule | Delivered by |
| --- | --- | --- | --- |
| I1 | **One set of paths.** A path or a list of paths valid on `d` is valid on `sample(d)`, on each element of `sample(d, sample_shape=n)`, on `mean(d)`, and on `d.event_spec` | follows from § 2.10: a term's paths are its output spec's | S1 for records and batches, S2b for output specs, G2 for the draws of the G2 families, G3 across kinds, G5 on batches of laws |
| I2 | **Indexing commutes with sampling.** `sample(d[s])` is distributed as `sample(d)[s]`, at every path, held fields included | follows: it is § 4.2's definition of `d[s]` | S1 for lists, G1a for groups, G3 at held terms |
| I3 | **Indexing commutes with summaries.** `mean(d)[s]` and `mean(d[s])` estimate one quantity, equal on exact routes; likewise `variance`, `standard_deviation`, and `quantile` | follows from I2 and the summaries rule of #576 | #576 with S1 |
| I4 | **Positions and paths commute on a batch.** `bt[j][s] == bt[s][j]` | follows from § 2.5 and § 4.2 | S1 |
| I5 | **Paths chain, across kinds.** `t[p][q] == t[p + "/" + q]`, including through a field that holds a law, a function, or a batch | follows from § 2.10, case 2 | S1 for records and batches, G1a for laws, G3 across kinds |
| I6 | **Output specs agree.** For a law, a kernel, a function, or a batch, `output_spec(t[s]) == output_spec(t)[s]`: `d[s].event_spec == d.event_spec[s] == marginal(d, s).event_spec`, `f[s].output_spec == f.output_spec[s]`, `k[s].event_spec == k.event_spec[s]`, and `bt[s].spec.element_spec == bt.spec.element_spec[s]`; for a record, `r[s].spec == r.spec[s]` | follows from § 4.2 and § 4.4 | S2b for the paths S1 supports, G1a for groups, G3 at held fields, S4 for batches, G3 for a batch's output at the root, G5 for kernels |
| I7 | **A view and a marginal are one law.** `marginal(d, s)` equals `d[s]`; one is detached, the other refers to `d` | rule (VI.8) | S1 for lists, G1a for groups |
| I8 | **Draws of a list of paths condition back.** `condition_on(d, sample(d[ps]))` fixes the nodes at `ps` and nothing else | rule (decided 2026-10-15) | G4 |
| I9 | **One path gives the node; a list gives a record that holds it.** `t["g"]` is the node, and `t[["g"]]` is the record `{g: ...}`, or the term whose produced term is that record, as pandas `df["a"]` and `df[["a"]]` are; for the output at the root, named `n`, `t[[n]]` is likewise the record `{n: ...}`, or the term that produces it | rule (decided 2026-10-15; O6) | S1; S4 for a batch's output at the root; G3 at held fields |
| I10 | **A list and `without` partition.** Every leaf path of `t` is a leaf path of exactly one of `t[ps]` and `t.without(*ps)`, on records, record specs, batches of records, output specs, and laws; every input of `I` is in exactly one of `I[ns]` and `I.without(*ns)` | follows from § 5.10, § 4.4, and § 4.5 | S1 for records, record specs, and batches; S2b for output and input specs; G4 for laws; after the workshop for records at paths through a held term |
| I11 | **Conditioning removes what it fixes.** `condition_on(d, given).event_spec == d.event_spec.without(*fixed)`, whichever route computes it; for a kernel, bound inputs, and the bound parts of a record input, leave its `given_spec` | rule (decided 2026-10-15) | G4 |
| I12 | **The term at an output.** For every term `t` with outputs, `t.output(name)` is `t[name]`, or `t` for the output at the root, and `output_spec(t).outputs[name]` is the term spec of the value at the output `name` of `t`'s produced term (§ 2.11): for a record or a batch, which is its own produced term, the spec of `t.output(name)` itself, so for `record_with_law` (§ 2.4) `outputs["G"]` is `post.spec`; for a law, a kernel, or a function, the term spec of `t.output(name)`'s produced term | follows from § 2.11 and § 4.11 | G3; G5 for kernels; S4 stores what it reads for batches |
| I13 | **Indexing commutes with producing.** `f[s](w) == f(w)[s]`; the law of `k[s]` at `w` is the law of `X`'s node at `s`, for `X ~ k(w)` | follows: it is § 4.2's definition | G3 for functions, G5 for kernels |
| I14 | **Indexing keeps inputs.** `f[s].input_spec == f.input_spec`; `k[s].given_spec == k.given_spec` | follows from § 4.2 | G3, G5 |
| I15 | **Sides are disjoint.** `names(unfed(t)) ∩ outputs(t) = ∅` for every kernel, law, and merge | rule for kernels (III.9, `distributions/_conditional.py:131`); follows for merges from M4 and M5 (§ 4.15, property 2) | today for kernels; S3 for merges |
| I16 | **The spec of a merge is the merge of the specs, after substitution.** For records, laws, kernels, and functions, `output_spec(a & b) == β(output_spec(a) & output_spec(b))` and `unfed(a & b) == β(unfed(a).without(*fed) & unfed(b))`; for batches, the same holds of the element merge; `(a & b).spec == a.spec & b.spec` when neither operand's factors take an unfed input that its input spec does not state (§ 4.16, item 5) | follows from R1 and R2 of § 2.19 | S2b for the specs, through `_factor_graph`; S3 for the operator; spec merges of laws after the workshop |
| I17 | **A merge keeps paths.** Every path `p` of `output_spec(a)` is a path of `output_spec(a & b)`, with `output_spec(a & b)[p] == β(output_spec(a)[p])`, and likewise for `b`; an output at the root moves to the path of its name | follows from § 2.18 (§ 4.15, property 4) | S2b for specs, S3 for laws; G1a and G2, which end the interim whole forms (T7, T16) |
| I18 | **Membership agrees with `[]`.** `key in t` if and only if `t[key]` succeeds, on every term other than an array, and on `RecordSpec`, `OutputSpec`, and `InputSpec`; `rename` takes the same paths, and the name of the output at the root | rule | S1 for terms and record specs, S2b for output and input specs, S3 for `rename`, G3 for paths through held terms |
| I19 | **Merge is associative when both groupings are valid**, under § 4.15, property 1's conditions: for input specs and functions, also when no dimension name one grouping binds occurs free in the operand merged with later; for batches, only of records or of base values | follows (§ 4.15, property 1) | holds today for `*`, apart from the shared-input specs of § 2.18; S3 |
| I20 | **Sampling commutes with merge.** `sample(a & b)` is distributed as `sample(a) & sample(b)` for laws that do not feed each other and produce records, when `a & b` is valid | follows from § 2.20 and the shallow record rule (§ 4.15, property 5) | S3, with S2b's shallow rule |
| I21 | **Merge commutes with renaming.** `(a & b).rename(m) == a.rename(m_a) & b.rename(m_b)` for laws, kernels, and records, under § 4.15, property 6's conditions on `m` | follows (§ 4.15, property 6) | S3; for functions, after the workshop |
| I22 | **Merge reads no label.** Relabeling an operand changes only the result's label | follows from § 2.17 and § 2.19, which read only names and factors | today |

I2, I4, I5, I6, and I13 follow from the definitions, so their tests check the implementation against the definition.
I2, I5, I6, and I9 hold with no exception at held fields, since no view wraps (§ 5.5).

jhuggins's Rocq model proves I1 for laws, draws, and batches, and I2, I4, I5, and I10 for every kind of term it models.
It also proves the fact behind I17, that a merge keeps every path of an operand except the root (`graft_identity_B`), and that `&` commutes with renaming every name at once (`dmerge_rename`), the fact behind I21.
Some of its results about values hold in a model where laws are point masses.
The model does not model specs, dimension names, or optional inputs, so I19's conditions are this plan's (§ 4.15), and it does not cover renames by path or paths inside inputs (`deferred.md`).

**What "equal" means.**
For values, `==` above means equal contents: the same spec and equal entries, compared with `jnp.array_equal` for arrays and field by field for records and batches.
For laws and functions, it means the same term: equal output specs and the same distribution or the same map.
A merge reads factors, which this equality does not compare (§ 4.16).
Tests compare specs with `==`, and distributions through exact summaries where they exist.

With the running examples, all of the following hold once every step has merged:

```python
draw = sample(grouped)

grouped["g"]["c"]                         # I5: the same law as grouped["g/c"]
sample(grouped["g"])                      # I2: a record {c, e}, like draw["g"]
draw["g"]["c"] == draw["g/c"]             # I5 on a value
mean(grouped)["g/c"]                      # I1, I3
grouped.event_spec[["a", "g/c"]] == grouped["a", "g/c"].event_spec   # I6
grouped.output("g") == grouped["g"]       # I12: the output g is at the path g
"g/c" in grouped.event_spec               # I18, as "g/c" in grouped

draws = sample(grouped, sample_shape=100)
draws[3]["g/c"] == draws["g/c"][3]       # I4

emulator["mean"](theta) == emulator(theta)["mean"]   # I13
(lik & prior).event_spec == lik.event_spec & prior.event_spec   # I16
```

## 4. General rules

Terms share one structure: a produced term, inputs, outputs, and paths (§ 2).
So most rules hold the same way for every kind, and this section states each of them once.
Section 5 says, kind by kind, only what is specific.

### 4.1 The path reader

Every term that is indexed by path reads what `[]` is given the same way, through one **path reader**:
- a `str` is one path;
- a list or tuple of `str` is a list of paths;
- an integer, a slice, or a tuple of them is a position, on a batch only.

A base value has no paths: a `NumericArray` keeps its own `[]`, which indexes its entries, and an opaque value has no `[]`.
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

It refuses malformed paths and positions before anything is looked up:

| Indexing | Error | Reason |
| --- | --- | --- |
| `t[""]`, `t["g//c"]`, `t["/a"]` | `ValueError` | an empty path or an empty segment |
| `t[[]]`, `t[()]` | `ValueError` | no paths at all; on a batch too, so `t[tuple(paths)]` with no paths never returns everything |
| `t["a", "a"]` | `ValueError` | the same path twice |
| `t["g", "g/c"]` | `ValueError` | a path inside another would duplicate a variable, giving a degenerate joint law |
| `t["a", 0]` | `TypeError` | a path and a position in one `[]` |
| `draws[[0, 2]]` | `TypeError` | a list of positions; there is no fancy indexing |
| `draws[jnp.array([0, 2])]`, `draws[np.array([0, 2])]` | `TypeError` | an integer array is a list of positions too; today it raises `TypeError` as well |
| `t[{"a"}]`, `t[iter(["a"])]` | `TypeError` | a set has no order and an iterator can be read only once, as `_as_names` already rules |

A path that does not exist raises `KeyError`.

**The reader's messages** (step S1, under `STYLE_GUIDE.md` §9.3):
- a missing path lists the available paths, as a law's message does today: "'habitat/forst' is not a path of 'site'; its paths: ['habitat', 'habitat/forest', 'habitat/wetland', 'forest']";
- when a missing path comes from a tuple whose `/`-joined form is a path of the term, as `site["habitat", "forest"]` on a record without a top-level `forest`, the message adds "a tuple is a list of paths; for the nested path write 'habitat/forest'". This hint changes no behaviour, so it is not bridging code;
- `draws[[0, 2]]`: "a list of positions is not supported; index one position or a slice";
- `t["g", "g/c"]`: "'g/c' is inside 'g'; pick one";
- `t["a", 0]`: "a path and a position cannot be mixed in one []".

Only three places accept a list of paths: `[]`, `marginal`, and `OutputSpec[]` (and, by names, `InputSpec[]`, § 5.10).
Every other argument takes one path as a `str`, and the same reader checks it, so `without("a", "a")` and `without("g", "g/c")` raise as `[]` does:

```python
draw.at_path("g", "c")        # separate segments, joined with "/" (unchanged)
draw.at_path(("g", "c"))      # TypeError: unpack the tuple, at_path(*segments)   (today: the path g/c)
draw.without("a", "g/e")      # removes several paths, one per argument (unchanged)
draw.without(("a", "g/e"))    # TypeError: pass the paths as separate arguments   (today: KeyError, the path a/g/e)
draw.without("a", "a")        # ValueError                                         (today: removes a)
condition_on(grouped, {"g/c": 0.5})    # a mapping from paths to values
```

An output spec is indexed by path too, as every term is (§ 4.4).

**Change from today:** a tuple is one nested path on records and batches today, a list of paths works only on laws, and a Python list is refused everywhere.

### 4.2 `[]` on terms

`[]` reads a path or a list of paths `s` through the path reader (§ 4.1), and its result depends on the kind:

| `t` | `t[s]` | Equation (I6) |
| --- | --- | --- |
| a record `r` | the node at `s` itself: a field's value or held term, or a group's record; for a list, the record of the chosen nodes | `r[s].spec == r.spec[s]` |
| a law `d` | the view, of class `MarginalView` (today `FieldView`, renamed in step S3): the law of `X`'s node at `s`, for `X ~ d`, which refers to `d`; `marginal(d, s)` is the same law detached | `d[s].event_spec == d.event_spec[s]` |
| a kernel `k` (step G5) | the kernel whose law at each input record `w` is the law of `X`'s node at `s`, for `X ~ k(w)` | `k[s].event_spec == k.event_spec[s]` and `k[s].given_spec == k.given_spec` |
| a function `f` (step G3) | the function `w ↦ f(w)[s]` | `f[s].output_spec == f.output_spec[s]` and `f[s].input_spec == f.input_spec` |
| a batch `bt` | the batch, over the same levels, of each element's node at `s`; for one path, the column | `bt[s].spec.element_spec == bt.spec.element_spec[s]` |

So on a law, a kernel, a function, or a batch, `t[s]` is the term of the same kind whose produced term is the node at `s` of `t`'s produced term.
A record is its own produced term, and its nodes are its values and held terms, so `r[s]` is the node itself: `r["a"]` is an array labeled `a`, and `record_with_law["G"]` is the law held there.
For a record, I6 is stated on its term spec, since a base child has no output spec; the name `a` is in `OutputSpec(r.spec)["a"]`, not in the array.
Indexing never changes a term's inputs, and the definition makes I2, I4, and I13 hold by construction.

One path gives the node; a list gives a record, or a term whose produced term is a record, that keeps each chosen node at its own path, in canonical order whatever order the list gives (decided 2026-10-15).
A path in a list that goes through a field holding a law, a function, or a batch cuts that held term down to the rest of the path, in the held term's own kind (`restrict`, § 4.4).
Paths name groups everywhere, `Record` included, and continue into a field that holds a law, a function, or a batch.

**The name of the output at the root is not a path.**
A keyword-form output spec has no path below its root, so its name is not a path: `Normal("x", 0.0, 1.0)["x"]` raises `KeyError`, and `"x" in Normal("x", 0.0, 1.0)` is false.
`Normal("x", 0.0, 1.0).output("x")` returns the law (I12), so a reader has one way to read an output (§ 4.11; § 6.20, Q5).
The list form at the root keeps today's tuple result: `Normal("x", 0.0, 1.0)[["x"]]`, like today's `Normal("x", 0.0, 1.0)[("x",)]`, is the law over the record `{x}` (O6; § 4.4, rule 3).

**Change from today:** `Normal("x", 0.0, 1.0)["x"]` returns the law itself, since III.7 lets a caller index a law by output name whatever its form.

### 4.3 Indexing with several paths

A list of paths keeps every chosen node at its own path, so the result is the original tree cut down to those nodes.
The result is always a record, or a term whose produced term is a record (I9), and it is the complement of `without` (I10): every leaf path is in one and only one of the two.

```python
grouped["a", "g/c"]                  # a law over {a, g: {c}}
grouped.without("g/e")                # the same tree, obtained by removing instead of choosing   (step G4)
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

Whether paths come as one path or as a list decides the result (I9).
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

A path through a field that holds a law, a function, or a batch cuts that term down too, in its own kind (`restrict`, § 4.4), and `without` removes a held field whose every leaf path it covers, as it removes an emptied group (`remove`, § 4.5).
With `record_with_law = Record({"n": 3.0, "G": post})`, `record_with_law["n", "G/mu"]` is the record `{n, G}` whose `G` is `post[["mu"]]` (step G3).

A value indexed by several paths is labeled by its paths, as `(a, g/c)` (decided 2026-10-15).
Today `sample(grouped["a", "g/c"]).label` is `"(a, c) ~ Normal·Normal·Normal"`.

To pass chosen fields to a function as keyword arguments, use `select`, which flattens each path to its last name (§ 5.4).

**Change from today:** a law indexed by several paths flattens each node to its last name, so `grouped["a", "g/c"]` is a law over `{a, c}`, and two paths that end in the same name raise (fact 16).

### 4.4 `[]` on an output spec

For an output spec `O` and a path or a list of paths `s`, `O[s]` is defined by four rules, read through the path reader (§ 4.1).
It replaces today's private `_view_declaration` (`distributions/_views.py:160`).

1. **One path `p`** of `O` gives the output spec of `O` at `p` (§ 2.10). So:
   - a path to a base node gives that node under its last name: `grouped.event_spec["g/c"]` is `OutputSpec(c=...)`;
   - a path to a group gives the group's record positionally, whose outputs are the group's children;
   - a path to a field that holds a law, a kernel, a function, or a batch gives that held term's spec positionally;
   - a path below such a field gives the spec of a term of the held term's kind, whose produced term is the node there.
2. **A list of paths `ps`** gives `OutputSpec(restrict(O.term_spec, ps))`, where `restrict` keeps each node at its own path:
   - for a `RecordSpec` `T`: the children of `T` that a path of `ps` names or enters, in `T`'s order; a child that `ps` names is kept whole, and a child that `ps` enters through the remaining paths `qs` is kept as `restrict(child, qs)`;
   - for a `T` that holds `E` in the positional form: `with_held(T, OutputSpec(restrict(E.term_spec, ps)))`.

   Every entry of `ps` other than the name that rule 3 admits must be a path of `O` (§ 2.10), and `O[ps]` checks this before `restrict` runs, raising `KeyError` otherwise. So for `O = OutputSpec(RecordSpec(G=Normal("x", 0.0, 1.0).spec))`, `O[["G/x"]]` raises `KeyError`: `x` names the drawn array at the root, and § 2.10, case 3 makes `G/x` no path of `O`.
3. **A list may also name the output at the root, at the top level only.** A list is read against the record of outputs (§ 2.12), which for an output at the root named `n` is `{n: v}`. So `O[[n]]` is `OutputSpec(RecordSpec(n=O.term_spec))` for an output spec in the keyword form, and `OutputSpec(with_held(T, E[[n]]))` for one positional over a batch or function spec `T` holding `E`. This is an extra rule beside rules 1 and 2 (O6): `Normal("x", 0.0, 1.0)[["x"]]` is the law over `{x}`. It applies only to the list that `O[...]` is given, never inside `restrict`.
4. **The name of the output at the root, alone, in the keyword form over a base spec.** `O[n]` raises `KeyError`, since the name is not a path (§ 2.10), and `t.output(n)` reads the term (§ 4.11).

`O[s]` adds no name that `O` does not already hold: rule 3 uses the name `O` states, and rules 1 and 2 add none.
The errors are the path reader's (§ 4.1), and a `str` that is not a path of `O` raises `KeyError`.
**Change from today:** `_view_declaration` flattens a list of paths to last names, and gives one path to a group whole.

**The property these rules give.**
For a law, a kernel, a function, or a batch `t` whose output spec is `O`, `O[s]` is the output spec of `t[s]` (I6).
For a record, the property is stated on term specs, `r[s].spec == r.spec[s]`, since a base child has no output spec.
`O[s]` never wraps, so the output spec of a function agrees with the value it returns (I13): `fit_wrapped.output_spec["posterior"]` is `OutputSpec(post_spec)`, as `fit_wrapped(y_observed)["posterior"]` is the law itself.

```python
O = grouped.event_spec                       # positional over RecordSpec {a, g: {c, e}}
O["a"]                                       # OutputSpec(a=...)
O["g/c"]                                     # OutputSpec(c=...)
O["g"]                                       # OutputSpec(RecordSpec(c=..., e=...)): outputs c and e
O[["g"]]                                     # OutputSpec(RecordSpec(g=RecordSpec(c=..., e=...))): output g
O[["g/e", "a", "g/c"]]                       # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=..., e=...)))
hier.event_spec[["s1/theta", "s2/theta"]]    # OutputSpec(RecordSpec(s1=RecordSpec(theta=...), s2=RecordSpec(theta=...)))
OutputSpec(post.spec)["mu"]                  # OutputSpec(DistributionSpec(OutputSpec(mu=...))): the spec of a law of mu
OutputSpec(RecordSpec(n=(), G=post.spec))[["n", "G/mu"]]
# OutputSpec(RecordSpec(n=..., G=DistributionSpec(OutputSpec(RecordSpec(mu=...))))): G holds post[["mu"]]'s spec
rm.event_spec["G"]                           # OutputSpec(law_spec): the law held at G, positionally
OutputSpec(x=NumericArraySpec(()))[["x"]]    # OutputSpec(RecordSpec(x=...)): rule 3
OutputSpec(x=NumericArraySpec(()))["x"]      # KeyError: x names the value at the root (rule 4)
O["x"]                                       # KeyError: x is not a path of grouped's draws
O[["g", "g/c"]]                              # ValueError: g/c is inside g
```

`O[s]` is not `Record.select`, which returns a dict for passing fields to a function as keywords (§ 5.4).

### 4.5 `without`

`t.without(*paths)` returns `t` in its own kind with the nodes at `paths` removed and every other node at its path, and its inputs are unchanged.
It exists on records, at every path, paths through a held term included; on record specs, output specs (§ 4.5), and input specs (§ 5.10); on batches of records and of laws; on laws, where it returns a view; and on kernels and functions.
What remains keeps the form it had: removing from a record leaves a record, even when one node is left, so `without` and a list of paths stay complements (I10).

```python
draw.without("g/e")           # Record {a, g: {c}}
post.without("tau")           # a view of post over the record {mu}   (today: AttributeError, laws have no without)
```

**On an output spec.**

`O.without(*ps)` is `OutputSpec(remove(O.term_spec, ps))`, where `remove` is defined on a spec with parts `T`:
1. **`T` is a `RecordSpec`:** each child that `ps` names is removed; a child all of whose leaf paths `ps` covers is removed, whether a group or a field holding a term; each other child that `ps` enters, through the remaining paths `qs`, is replaced by `remove(child, qs)`.
2. **`T` holds `E`:** `remove(T, ps)` is `with_held(T, E.without(*ps))`.

A record stays a record even when one node is left, so a list of paths and `without` are complements (I10).
A path that is not a path of `O` raises `KeyError`, so every path of an output spec in the keyword form over a base spec raises, since its name is not a path.
A removal that would leave `O` itself with nothing raises `ValueError`, as `RecordSpec.without` does.
The path reader's errors apply, so a path given twice and a path inside another raise `ValueError`.

```python
growth_prior.event_spec.without("sigma")      # OutputSpec(RecordSpec(K=..., r=..., phi=...))
grouped.event_spec.without("g/e")             # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=...)))
grouped.event_spec.without("g/c", "g/e")      # OutputSpec(RecordSpec(a=...)): the empty group goes
OutputSpec(RecordSpec(n=(), G=post.spec)).without("G/mu", "G/tau")   # OutputSpec(RecordSpec(n=...)): G goes
OutputSpec(x=NumericArraySpec(())).without("x")   # KeyError: x is not a path
```

### 4.6 Membership

`key in t` is true when `key` is a `str` and `t[key]` succeeds, and false otherwise (I18).
This holds on every term other than an array, and on `RecordSpec`, `OutputSpec`, and `InputSpec`.
That covers groups, and paths through a held law, function, or batch.
A record's `keys()` still lists its fields only, and an input spec's `keys()` lists its inputs; a `NumericArray` keeps its own `in`, which tests membership of an entry.

```python
draw = sample(grouped)
"g/c" in draw               # True                                   (unchanged)
"g" in draw                 # True: draw["g"] works                  (today: False)
("a", "g/c") in draw        # False: a list of paths is not a path   (False today too, as the path a/g/c)
"G/mu" in record_with_law   # True, for record_with_law of § 5.4     (today: False; step G3)

draws = sample(grouped, sample_shape=100)
"a" in draws          # True                                   (today: False, by iterating the draws)

"g/c" in grouped                   # True                      (today: TypeError, "argument ... is not iterable")
"g/c" in grouped.event_spec        # True: the same paths      (today: TypeError, fact 13)
"x" in Normal("x", 0.0, 1.0)       # False: x names the value at the root, not a path, as Normal("x", 0.0, 1.0)["x"] raises
"mu" in lik.given_spec             # True
```

On every batch, `in` tests paths and never elements.
So a batch of arrays, which has no paths, answers `False` for every `str`, and `1.0 in sample(Normal("x", 0.0, 1.0), sample_shape=3)` changes from testing the draws to `False`.
`"x" in sample(Normal("x", 0.0, 1.0), sample_shape=4)` is `False`, although the batch offers the output `x`: `x` is its name at the root, not a path inside it, and `draws.output("x")` reads it.

**Why `in` follows `[]`.**
A Python `dict` satisfies both `p in m` if and only if `p in m.keys()`, and `p in m` if and only if `m[p]` succeeds.
A `Record` cannot keep both, because `keys()` lists fields only while `[]` also returns groups; the plan keeps the second (§ 6.3).

**Change from today:** a `Record` answers `in` for fields only, a batch falls back to iterating its elements, and a law and an output spec raise.

### 4.7 Operations that take paths

Every operation reads its paths through the reader of § 4.1, so a path means the same thing wherever it is passed.
A few arguments also accept a name that is not a path:

| Argument | Paths | Names as well |
| --- | --- | --- |
| `t[...]`, `marginal(d, ...)`, `O[...]` | one path, or a list of paths | none, apart from a list that names the output at the root (§ 4.4, rule 3) |
| `without(*paths)`, `select(*paths)`, `at_path(*segments)`, `is_field(path)` | one path per argument | none |
| the keys of `rename` | paths of the term | the name of the output at the root, which renames it |
| the keys of `condition_on(d, given)` | paths of `d`'s event | for a kernel, its input names, each optionally followed by a path inside a record input |
| merging with `&` | none | feeds each input from the output of the same name |

No rename key can be read both as a name and as a path.
An output at the root is a base value, which has no path below its name, and every other output's name is its own path.
jhuggins's model proves this for renames (`no_ambiguous_key_B`).
The keys of `condition_on` mix input names and event paths.
They stay unambiguous because a kernel never has one name that is both an input and an output, and a merge keeps that so (I15), which is a separate rule.

### 4.8 `rename`

`with_path_names` becomes `rename` (decided 2026-10-17), the name pandas (`df.rename`) and xarray (`ds.rename`) use.
It renames and moves nodes by their paths, and renames the output at the root.
Its keys are the term's paths, plus that one name, so the paths a view takes and its rename keys agree (I18):

```python
Normal("x", 0.0, 1.0).rename(x="y")       # Normal(y): renames the output at the root   (unchanged)
grouped.rename({"g/c": "g/k"})            # rename in place                             (unchanged)
grouped.rename({"g/c": "c"})              # move c to the top level                     (unchanged)
grouped["g"].rename(c="k")                # a view offering k and e
# today: KeyError, "with_path_names(): 'c' is not a path in this tree; the paths are ['g', 'g/c', 'g/e']"
```

Two cases follow from the rule:
1. **The name of an output at the root cannot become a nested path**, since the base value has no record to place it in. `Normal("x", 0.0, 1.0).rename(x="g/y")` raises `ValueError`, as it does today; merge it into a joint first.
2. **On a batch**, renaming the elements' output at the root renames the batch's one output: `draws.rename(x="y")` on draws of `Normal("x", 0.0, 1.0)` offers `y`. The batch stores that name in its element spec (§ 5.8), and S4 delivers this case.

The rename applies to every kind that has `with_path_names` today, in one mechanical PR with `&` and `MarginalView` (step S3): `NamedTree` and its families, `OutputSpec`, `Distribution`, `ConditionalDistribution`, and batches.
O13 leaves `with_level_names` and `with_dim_names` unchanged.

### 4.9 The size shorthand

Wherever an input or the keyword form of `OutputSpec` takes a term spec, a tuple of sizes is the `NumericArraySpec` of that shape, as it already is for a `RecordSpec` field:

```python
InputSpec(K=(), e=("n_years",))                   # today: TypeError
conditional_distribution(lik_fn, given_spec={"K": (), "r": (), "phi": (), "e": (n_years,)})
                                                  # today: TypeError, "the given slot 'K' declares no term spec"
OutputSpec(K=())                                  # today: TypeError
```

The shorthand applies to an `InputSpec` input, to `given_spec`, and to the keyword form of `OutputSpec`.
II.1's rule for a `RecordSpec` field applies unchanged, so the shorthand is a tuple only: `K=3` is refused, while a shape argument would read it as `(3,)`.
An input declared by the shorthand sets no dtype and no support, so a merge accepts it against any dtype and support of its shape (§ 2.14).

### 4.10 Symbolic dimensions

`[]` and `without` keep every dimension name as it is, bound or free, and bind no size.
The kept nodes stay in the dimension scope of the term they came from: a name in `O[s]` is the same dimension as in `O`.
The free dimensions of `O[s]` are the free dimensions of `O` that occur in the kept nodes.
Substituting a size commutes with `[]`: `O.with_dim_sizes(n=3)[s] == O[s].with_dim_sizes(n=3)` when `n` is free in `O[s]`.
A view applies a dimension substitution to its parent (III.7).

```python
O = OutputSpec(RecordSpec(X=("obs", "features"), beta=("features",)))
O["beta"]                       # OutputSpec(beta=NumericArraySpec(("features",)))
O["beta"].term_spec.free_dims   # frozenset({"features"})
O["X"] & O["beta"]              # OutputSpec(RecordSpec(X=..., beta=...)): one dimension features again
```

### 4.11 The term at an output

`t.output(c)` is the term at the output `c`: `t` itself when `c` is at the root, and `t[c]` otherwise.
It arrives in step G3 on records, laws, functions, and batches, and with `[]` in step G5 on kernels (§ 5.6).
`t.output(c)` replaces the decided `t.component(c)` as the final name (§ 6.20, Q1, Q5).
Its use is the output at the root, which is not a path, so `[]` cannot read it; away from the root it is `t[c]`.
On an output spec there is no counterpart: `O.outputs[c]` gives the spec of an output, and `O[c]` the output spec at a path.
A name that is not an output raises `KeyError`, so a term whose produced term is a law offers nothing to `output`.

| Term `t` | `t.output(c)` |
| --- | --- |
| a record | `t[c]`, the child itself |
| a law or a kernel | the view `t[c]`, or `t` itself when its event's output `c` is at the root |
| a function | the function `w ↦ t(w)[c]`, or `t` itself when its output `c` is at the root |
| a batch | the column `t[c]`, or `t` itself when its elements' output `c` is at the root |

For every term `t` with outputs, `output_spec(t).outputs[c]` is the term spec of the value at the output `c` of `t`'s produced term (§ 2.11; I12):
- for a record or a batch, which is its own produced term, it is the spec of `t.output(c)` itself: `OutputSpec(record_with_law.spec).outputs["G"]` is `post.spec`, the spec of the law held there, not the spec of its draws;
- for a law, a kernel, or a function, it is the term spec of `t.output(c)`'s produced term: `prior.event_spec.outputs["mu"]` is `NumericArraySpec(())`, the spec of a draw of `prior["mu"]`.

```python
grouped.output("g")                                         # grouped["g"]: offers c and e
Normal("x", 0.0, 1.0).output("x")                           # the law itself
sample(Normal("x", 0.0, 1.0), sample_shape=4).output("x")   # the batch itself
fit = Function(run_inference, output_spec=post.spec)
fit.output("mu")                                            # KeyError: fit produces a law, which offers no outputs
rm[["G"]].output("G")                                       # rm["G"], a law whose draws are laws
```

A mixture is a law, so `mix.output("x")` reads its event's output `x`; the word "component" keeps only its sense of a mixture's component laws (VII.3), which settles O14.

### 4.12 Defaults by kind, and type holes

Two places build an output spec without a caller writing one: a default, and a type hole filled later.

**Defaults today.**
One classmethod, `OutputSpec.default(spec, *, component)`, builds every default.
It gives `OutputSpec(spec)` for a record, and the keyword form `OutputSpec(**{component: spec})` for any other spec.
Its callers play two roles:
1. **The output default**, for what a function returns: `_complete_output_spec` in `values/_function_base.py`, which reads a bare `output_spec`, and `_output_atoms` in `functions/_broadcast.py`, which declares the output of a lifted call.
2. **The event default**, for a law's whole-term event: `_whole_term_event` in `distributions/_distribution.py`, which the families call (random measures, empirical laws, bootstrap laws, laws from backends, converted and transformed laws, and GLM likelihoods); the event reading of `distribution` in `distributions/_from_functions.py`; `_agreed_event_spec` in `distributions/_conditional.py`; and `_event_repr_parts`, which decides whether a law's repr shows its event.

`functions/_result.py` calls it a third time, to declare the element of a batch it stacks under the name `element`; § 5.8 covers that case.

**One default.**
`OutputSpec.default` is the one public default (II.2), with the rule by kind below, and it serves events too.
The private event default, the event callers above, goes: each distribution family that names its own law's draws wraps them under the name its constructor takes (§ 5.5), and every other event default is `OutputSpec.default`.

| What is produced | Default today | `OutputSpec.default` |
| --- | --- | --- |
| a base value | keyword form under the default name | unchanged |
| a record | positional | unchanged |
| a batch | keyword form under the default name | positional through its element spec, named by the rule of O2 |
| a law, a kernel, or a function | keyword form under the default name | positional: a law or a kernel offers no outputs, and a function offers its outputs |

Which callers change in which step:
- **S4:** `functions/_result.py`, whose stacked batch stores its element spec (§ 5.8).
- **G1b:** the rule of `OutputSpec.default`, `_complete_output_spec`, `_output_atoms`, and `_lift_result` in `functions/_broadcast.py`, which declares a lifted law's event and wraps a returned law, kernel, function, or batch there.
- **G2:** the event default goes, and with it `_whole_term_event`'s own rule, the event reading of `distribution`, `_agreed_event_spec`, and `_event_repr_parts` move to `OutputSpec.default` or the families' wraps.

**Lifting a function whose produced term is not a value.**
A lift builds its own law, an `EmpiricalDistribution` of the evaluations, and names that law's draws itself.
When the returned term is a base value or a record, the lifted law's event spec is the function's output spec.
When it is a law, a kernel, a function, or a batch, which a record could not hold under no name (M2's test), the event is the record that holds the returned term whole under the callable's output name (`_callable_component`, `values/_function_base.py:230`).
With `include_inputs=True`, the record also holds each lifted argument under its parameter's name, as today (V.5).
The reason is C5 applied to a law the lift constructs, as the G2 families name theirs.
This does not contradict "ProbPipe never wraps a function body's return value" (decided 2026-10-17): the wrap names the lift's own law, not a value the body returns.

```python
def law_of_y(x):
    return Normal("y", x, 1.0)

# Today
Function(law_of_y)(Normal("x", 0.0, 1.0)).event_spec  # OutputSpec(law_of_y=DistributionSpec(...)): whole over a law
OutputSpec.default(BatchSpec(NumericArraySpec(()), sample=3), component="f")
                                                      # OutputSpec(f=BatchSpec(...)): whole over a batch

# Proposed
Function(law_of_y)(Normal("x", 0.0, 1.0)).event_spec  # OutputSpec(RecordSpec(law_of_y=DistributionSpec(...))): draws {law_of_y: law}
Function(law_of_y).using(include_inputs=True)(x=Normal("x", 0.0, 1.0)).event_spec
                                                      # OutputSpec(RecordSpec(x=..., law_of_y=DistributionSpec(...))), as today
OutputSpec.default(BatchSpec(NumericArraySpec(()), sample=3), component="f")
                                                      # the positional batch whose element is named f; it offers f
```

The default output name is the callable's name, never its label (`_callable_component` in `values/_function_base.py`).
So `Function(lambda x: ..., label="law_of_y")` has the default output name `f`, which is the name of a lambda.

**Type holes.**
`OutputSpec(x=None)` names a value whose type is filled in later, with `with_term_spec` (§ 2.7):
1. filled with a base spec, it stays in the keyword form under `x`;
2. filled with a batch whose element carries only the default name `element`, it names the element `x` (O2);
3. filled with any other spec with parts, it raises `TypeError`, and the message asks the caller to declare the record (decided 2026-10-17, amended by O2).

A function names what has no name of its own: a base value, and a batch whose element carries only the default name `element` (O2).

### 4.13 Reading a bare `output_spec`

`output_spec=` accepts a bare term spec as well as an `OutputSpec`.
Today's rule (V.6 § The declared output, and III.3) reads a bare `RecordSpec` positionally, and any other bare spec in the keyword form under the callable's default output name.
A bare spec reads by `OutputSpec.default` (§ 4.12), with the callable's default output name as the default name:

| Bare `output_spec` | Today | This plan |
| --- | --- | --- |
| `NumericArraySpec((40,), "float32")` | keyword form under the callable's name | unchanged |
| `RecordSpec(a=..., b=...)` | positional: outputs `a`, `b` | unchanged |
| `post_spec`, a `DistributionSpec` | keyword form under the callable's name | positional: a law, with no outputs |
| a `FunctionSpec` | keyword form under the callable's name | positional: the returned function's outputs |
| `BatchSpec(NumericArraySpec(()), sample=3)` | keyword form under the callable's name | positional: the element, which has only the default name, is named by the callable (O2) |

```python
# Today
Function(run_inference, output_spec=post_spec).output_spec
# OutputSpec(run_inference=DistributionSpec(event_spec=OutputSpec(NumericRecordSpec(mu=..., tau=...))))

# Proposed
Function(run_inference, output_spec=post_spec).output_spec
# OutputSpec(DistributionSpec(event_spec=OutputSpec(RecordSpec(mu=..., tau=...)))): a law, with no outputs
```

The execution plan's examples pass base specs, such as `output_spec=NumericArraySpec((40,), "float32")`, so they read as today.
A law's `event_spec=` keeps its own rule (III.7): a bare `RecordSpec` is positional, and an event at the root needs its name.
Step S0 writes the new rule into V.6 and III.3, and step G1b implements it.

### 4.14 Equality and hashing

Two output specs are equal when they have the same form, the same output names in order, and equal term specs:
- `OutputSpec(parameters=R)` and `OutputSpec(RecordSpec(parameters=R))` have the same outputs and describe different produced terms, so they are unequal. The first exists only for a base `R` (§ 2.7).
- Equality is order-sensitive, because output order is the layout that `to_vector` reads. `InputSpec` equality ignores order, because binding reads inputs by name.
- An `OutputSpec` equals only another `OutputSpec`, never a plain mapping, since a mapping does not state the form.
- The hash agrees with equality.

### 4.15 Merge properties

`inputs-and-outputs.md` Appendix A proves each property and checks it against the reviewers' examples; here are the statements, each with its condition.

1. **Associative, when both groupings are valid.** For laws, kernels, records, record specs, output specs, and batches of records or of base values, when `(a & b) & c` and `a & (b & c)` are both valid they have the same factors, outputs in the same order, unfed inputs with the same specs and optional flags, stored scope, and value. Their validity can differ, and only where the groupings check different pairs of specs: at a name that `a` and `b` both take and `c` outputs, whose specs in `a` and `b` do not unify while `c`'s conforms to each; and where two different free dimension names occur at one axis in one grouping but are bound first in the other. For input specs and functions, which store no scope, both groupings also agree only when no dimension name that one grouping binds occurs free in the operand it is merged with later (§ 4.17). For batches of laws or kernels it can fail, since a batch merge reads each element as one factor (§ 2.16) and so drops the element's factors and stored scope (§ 6.3).
2. **Merged operands keep their sides disjoint:** `names(unfed(a & b)) ∩ outputs(a & b) = ∅`.
3. **Commutative up to order** when neither operand feeds the other: `a & b` and `b & a` are valid together and have the same law or value, with outputs in a different order; their input specs are equal once a shared input carries `U` (§ 2.18). Their output specs are unequal, since output order is the layout `to_vector` reads.
4. **Paths are kept.** Every path `p` of `output_spec(a)` is a path of `output_spec(a & b)`, with `output_spec(a & b)[p] == β(output_spec(a)[p])`, and an output of `a` at the root, named `n`, moves to the path `n`; likewise for `b`.
5. **Sampling commutes with merge** for laws `a` and `b` that do not feed each other and produce records, when `a & b` is valid: `sample(a & b)` is distributed as `sample(a) & sample(b)`. jhuggins proves this boundary in a model where laws are point masses (`sample_commutes_iff`).
6. **Renaming commutes with merge.** `(a & b).rename(m) == a.rename(m_a) & b.rename(m_b)`, where `m` maps names to names injectively over the names of `a` and `b`, `m_a` and `m_b` are its restrictions to each operand's names, it renames inputs and outputs alike, and it renames no name that is an unfed input of a law operand's factors, since a law's `rename` does not rename it while the merged kernel's `rename` may (III.9): for `M = lik_opt & Normal("mu", 0.0, 0.001)` and `kw(v | w)`, `(M & kw).rename({"s": "s2"})` renames `s`, while `M.rename({}) & kw.rename({})` keeps it. It applies to laws, kernels, records, and functions. A move of a node to a path does not commute: `grouped` is built by a move, and moving `c` and `e` under `g` before the merge raises M3 at `g`.
7. **Merge reads factors, not only the law.** Two equal laws can merge differently when their factors differ (§ 4.16).
8. **Labels are never read.** Relabeling an operand changes only the result's label.

### 4.16 Reading factors: today's behaviour and its consequences

Before the workshop, a merge of laws and kernels reads its operands' factors, as IV.2's flattening does today, and this plan keeps that unchanged (§ 6.20, Q4(b)).
Every consequence is stated here; the long-term rule, a merge that leaves optional inputs shows them, is decided after the workshop (`deferred.md`).

1. **A later merge feeds an unfed optional input.** `(lik_opt & prior) & s_prior` feeds `s`, so `y`'s spread is near 100, the same law as `lik_opt & (prior & s_prior)` (fact 25). This keeps associativity (§ 4.15, property 1).
2. **A later merge shares it.** For `M = lik_opt & Normal("mu", 0.0, 0.001)`, a law with spread of `y` near 1, and `k2 = conditional_distribution(lambda s: Normal("w", 0.0, s), ...)`, which neither feeds nor is fed by `M`, `M & k2` is a kernel in `s`, required, and binding `s = 30` gives `y` a spread near 30 (fact 40). M7 also checks the specs of such a shared input, so a merge can fail on an input that no spec of `M` shows.
   - **It also appears with no sharing.** For an unrelated kernel `kw(v | w)`, `M & kw` is a kernel whose `given_spec` is `InputSpec(s, w).with_optional("s")`, by R3, so a law that has no inputs (III.7) gains a bindable input `s` by a merge with a term that does not mention it; binding `w = 0` and `s = 30` gives `y` a spread near 30 (fact 46).
3. **Equal laws merge differently.** `lik_opt & prior` and `lik & prior` are equal laws, since `lik` is `lik_opt` at `s = 1.0`, but after `& s_prior` the first gives `y` a spread near 100 and the second near 5 (property 7).
4. **Which results keep the factor depends on the operation and its route**, and not on any rule of `design/`. A result keeps an unfed optional input exactly when it keeps, unchanged, the factor that takes it (fact 41):

   | Law built from `M = lik_opt & Normal("mu", 0.0, 0.001)` | Keeps `lik_opt` as a factor | `& s_prior` gives `y` a spread near |
   | --- | --- | --- |
   | `M[["y", "mu"]]`, a view | no: a view is one factor | 1 |
   | `marginal(M, ["y", "mu"])` | yes: a factored joint | 100 |
   | `M.rename({"y": "w/y"})` | yes | 100 |
   | `condition_on(lik_opt & Normal("mu") & Normal("tau"), {"tau": 1.0})` | yes, on its exact route | 100 |
   | `condition_on(M, {"y": y_observed})` | no: an `EmpiricalDistribution` | — |
   | `condition_on(lik_opt & Gamma("z", 2.0, 1.0), {"mu": 0.0})` | no: `lik_opt` is bound | 1 |
   | a law regrouped so that `lik_opt` is inside a grouped sub-joint | no: the sub-joint is one factor | 1 |

   So a view and a marginal of one joint, which I7 calls one law, can merge differently. § 6.3 lists this against I7.
5. **A spec merge can be valid where the term merge raises.** `s_prior.spec & (lik_opt & prior).spec` reads a `DistributionSpec`, which states no unfed input, so it feeds nothing and is valid, while `s_prior & (lik_opt & prior)` raises M4. So `(a & b).spec == a.spec & b.spec` (I16) holds only when neither operand's factors take an unfed input that its input spec does not state.
6. **A batch merge reads each element as one factor** (§ 2.16), so an element's hidden unfed inputs are not fed in a batch merge.

Today's code behaves in every item as stated (facts 25, 40, 41, 43, 46); `condition_on`'s exact route on a joint kernel, which ignores a fed optional input, is the one bug (§ 2.20).

### 4.17 Dimension scopes across merges: today's behaviour and its consequences

A merged law or kernel keeps its scope, as today's joint does, while a merged input spec, output spec, record spec, or function applies `β` and keeps nothing (§ 2.14).
This plan adds no new storage before the workshop; the long-term rule is decided after it, leaning to storing a scope on every merge result (§ 6.20, Q6).

1. **Spec and function merges are associative only under a condition.** For `J1 = InputSpec(u=("n",))`, `J2 = InputSpec(u=(3,), q=("n",))`, and `J3 = InputSpec(w=("n",))`, `(J1 & J2) & J3` binds `n = 3` in the first merge and forgets it, so the result keeps `w: ("n",)`, while `J1 & (J2 & J3)` binds `n = 3` across all three and gives `w: (3,)`. Both groupings are valid, and they differ. Property 1 and I19 therefore state their condition for input specs and functions.
2. **A stored scope can refuse a merge over a name no spec shows.** `m1 = a & b3`, for `a(ya: ("n",) | u: ("n",))` and a law `b3` over `u: (3,)`, has the concrete event `{ya: (3,), u: (3,)}` and stores `n = 3`; `m2 = k & lv`, for `k(w: ("n",) | v: ("n",))` and a law `lv` over `v: (4,)`, stores `n = 4`. `m1 & m2` raises M8, and `m1.with_dim_names(n="m") & m2` raises the same, since neither event spec contains `n` (fact 42). `sample(m1) & sample(m2)` is a valid record merge, so property 5 holds only when `a & b` is valid. § 6.3 lists this.

Storing the scope is what makes laws and kernels associative (fact 36); forgetting it is what keeps spec merges simple.

## 5. Term by term

Each subsection below takes one kind of term, or one spec class, in the same order:
1. **What it is**;
2. **Produced term** (§ 2.5);
3. **Parts** (§ 2.3, § 2.4);
4. **Inputs** (§ 2.6, § 2.9);
5. **Outputs** (§ 2.11);
6. **Specs it holds** (§ 2.8);
7. **Indexing**: `[]`, `in`, `without`, and `rename`, which follow the rules of § 4, so only what is specific is stated;
8. **Merge** (§ 2.15 to § 2.20);
9. **Examples**, then any rules that belong to this kind alone.

### 5.1 Every kind placed

| Kind | Produced term (§ 2.5) | Inputs (§ 2.9) | Outputs (§ 2.11) | `[]` takes (§ 4.2) | `t[s]` is | `without` | `&` (§ 2.15) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array, `NumericArray` | itself | none | none | NumPy's positions into its entries | its entries, unchanged | none | NumPy's elementwise `&`, which is not a merge (O7) |
| opaque value, `Opaque` | itself | none | none | nothing | — | none | none |
| record, `Record` | itself | none | its children | paths, lists of paths | the node itself, or the record of the chosen nodes | yes | merge: both sets of fields |
| law, `Distribution` | a random draw | none | those of `event_spec` | paths, lists of paths | the view, a `MarginalView` | yes, a view (§ 4.5) | merge, read as its factors (§ 2.15); feeds the left operand |
| kernel, `ConditionalDistribution` | a random draw at each value of its inputs | `given_spec` | those of `event_spec` | paths, lists of paths | the kernel whose law at each input value is the view there | yes | merge, read as its factors; feeds and is fed |
| function, `Function` | its return value at each value of its inputs | `input_spec` | those of `output_spec` | paths, lists of paths | the function `w ↦ f(w)[s]` | yes | merge; feeds and is fed |
| batch, `Batch` | itself | its elements' | its columns for records or functions; one output at the root for base values; none for laws or kernels | positions, or paths of its elements | by position, a sub-batch or an element; by path, the batch of each element's node | for batches of records and of laws | merge of the elements, position by position, each read as one operand |

A spec is not a term: it produces and receives nothing.
It describes the terms it types, so each operation on a spec gives what the spec says about the same operation on those terms.

| Spec | What it describes | `[]` | `without` | `&` |
| --- | --- | --- | --- | --- |
| `NumericArraySpec`, `OpaqueSpec` | a base value | nothing | nothing | nothing |
| `RecordSpec` | the type of a record, read as outputs by `OutputSpec(R)` and as the inputs they feed by `InputSpec(R)` | a path gives the spec at that path; a list gives the cut-down `RecordSpec` | paths | the union of the children; a top-level name in both raises |
| `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, `BatchSpec` | a law, a kernel, a function, or a batch, through the output spec each holds (§ 2.8) | none of their own: index the held spec, as `post.spec.event_spec["mu"]` | none | the spec of the merged term |
| `InputSpec` | a term's inputs | a name gives its term spec; a list of names gives an `InputSpec` (§ 5.10) | names | the inputs of both, a shared input unified (§ 2.18) |
| `OutputSpec` | a term's produced term and its outputs | a path or a list of paths gives an `OutputSpec` (§ 4.4) | paths | the outputs of both (§ 2.18) |

### 5.2 Array

**What it is.** A `NumericArray`: a base value whose term spec is a `NumericArraySpec` (§ 2.3).

**Produced term.** The array itself.

**Parts.** None: an array has no paths.

**Inputs.** None.

**Outputs.** None: a base value carries no name, and a label is not one (C5; § 2.9). A term that produces an array names it in its own output spec, in the keyword form (§ 2.7).

**Specs it holds.** None.

**Indexing.** `[]` is NumPy's indexing of the array's entries, and `in` tests membership of an entry (§ 4.1, § 4.6); an array has no `without`.

**Merge.** None: an array belongs to no merge category, and `&` on two `NumericArray`s is NumPy's elementwise `&`, which is not a merge (O7; § 2.16, § 6.3).

**Examples.**

```python
x = sample(Normal("x", 0.0, 1.0))      # an array: the draw at the output x, which is at the root
mean(Normal("x", 0.0, 1.0))            # an array too, like a draw (§ 5.5)
sample(grouped)["a"]                   # an array, labeled "a"; the name a is in OutputSpec(...)["a"], not in the array
x & sample(Normal("y", 0.0, 1.0))      # TypeError: NumPy's & refuses floats, so two draws do not merge (§ 6.3)
```

### 5.3 Opaque value

**What it is.** An `Opaque`: a value that is neither numeric nor a record, such as a string or an arbitrary Python object, typed by an `OpaqueSpec` (§ 2.1, § 2.3).

**Produced term.** The value itself.

**Parts.** None.

**Inputs.** None.

**Outputs.** None, as for an array; a term that produces an opaque value names it in the keyword form (§ 2.7).

**Specs it holds.** None.

**Indexing.** An opaque value has no `[]` (§ 4.1) and no `without`.

**Merge.** None: an opaque value belongs to no merge category (§ 2.16). Conformance and unification of opaque specs follow III.2 (§ 2.14, rule 2 of each).

**Examples.**

```python
OutputSpec(habitat=OpaqueSpec(type=str))   # a term that produces a string offers it under the name habitat
U(OpaqueSpec(type=str), OpaqueSpec())      # OpaqueSpec(type=str): U takes the known type (§ 2.14)
```

### 5.4 Record

**What it is.** A `Record`: a tree of named values; a nested value is named with `/` (III.5). Its term spec is a `RecordSpec` (§ 5.9).

**Produced term.** The record itself.

**Parts.** Its nodes: its fields and groups, and, below a held field, the paths of the held term (§ 2.4). `keys()` lists its fields only.

**Inputs.** None.

**Outputs.** Its children: its output spec is `OutputSpec(r.spec)`, derived and never stored (D6), and an empty record has none (§ 2.9). A record is its own produced term, so `OutputSpec(r.spec).outputs[c]` is the spec of the child itself (§ 4.11).

**Specs it holds.** None: a `RecordSpec` holds no output spec, since its children are its structure (§ 2.8).

**Indexing.** `r[s]` is the node itself: a field's value or held term, or a group's record; a list gives the record of the chosen nodes (§ 4.2, § 4.3). I6 is stated on term specs, `r[s].spec == r.spec[s]`. `in` follows `[]`, while `keys()` lists fields (§ 4.6). `without` and `rename` follow § 4.5 and § 4.8. `select` flattens to last names, below.

**Merge.** Records form one merge category, and a record merges only with another record (§ 2.16). A record only offers values: records have no inputs, so nothing is fed, and `a & b = a ⊎ b` (§ 2.12), shallow, by top-level name (§ 2.21).

**Examples.**

```python
draw = sample(grouped)            # Record {a, g: {c, e}}

draw["a"]           # NumericArray, labeled "a"                    (unchanged)
draw["g/c"]         # NumericArray, labeled "g/c"                  (unchanged)
draw["g"]           # Record {c, e}, a view labeled "g"            (today: KeyError, "use at_path()")
draw["g"]["c"]      # the same as draw["g/c"]
draw["a", "g/c"]    # Record {a, g: {c}}, labeled "(a, g/c)"      (today: KeyError, the path a/g/c)
draw["g/e", "a"]    # Record {a, g: {e}}: grouped's order
draw[["g"]]         # Record {g: {c, e}}                           (today: TypeError)
draw.keys()         # ("a", "g/c", "g/e"): fields only            (unchanged)
```

`draw.at_path("g")` stays as a synonym of `draw["g"]`, and `at_path("g", "c")` still takes separate segments.
Step S1 delivers these examples.

**Held fields.**

A path continues into the term a field holds:

```python
record_with_law = Record({"n": 3.0, "G": post})   # post's draws are {mu, tau}

record_with_law["G"]            # post                                 (unchanged)
record_with_law["G/mu"]         # the view post["mu"]                  (today: KeyError)
record_with_law["G"]["mu"]      # the same view (I5)
record_with_law["n", "G/mu"]    # Record {n, G}, whose G is post[["mu"]]
```

Step G3 delivers paths through a held field.

**Merging two records.**

```python
Record({"a": 1.0}) & Record({"b": 2.0, "g": {"c": 3.0}})   # Record {a, b, g: {c}}
Record({"g": {"c": 1.0}}) & Record({"g": {"e": 2.0}})      # ValueError (M3): g is an output of both
```

**`select`.**

`select(*paths, **remap)` exists to pass fields to a function as keyword arguments, so it flattens each path to its last name.
Today it names each entry by its full path, and `"g/c"` cannot be a keyword.
`RecordBatch.select` changes the same way; each of its entries stays a one-column batch, as today.

```python
draw.select("a", "g/c")       # {"a": ..., "c": ...}           (today: {"a": ..., "g/c": ...})
draw.select(x="g/c")          # {"x": ...}                    (unchanged)
sample(hier).select("s1/theta", "s2/theta")   # ValueError: both flatten to theta; name one with a keyword
                                              # (today: {"s1/theta": ..., "s2/theta": ...})
```

### 5.5 Law

**What it is.** A `Distribution`: a probability law, which has no inputs (III.7). Its term spec is a `DistributionSpec`.

**Produced term.** A random draw `X ~ d` (§ 2.5).

**Parts.** The paths of its draws, which are the paths of `d.event_spec` (§ 2.4, § 2.10). A law over a single array has parts but no path below its root (§ 2.3).

**Inputs.** None (III.7). A joint law's factors may still take optional inputs at their defaults, which a later merge reads (`unfed`, § 2.15).

**Outputs.** Those of `d.event_spec`, which are a draw's outputs. A law that is itself the produced term of another term offers none (§ 2.11).

**Specs it holds.** `event_spec`, the output spec of one draw (§ 2.8, § 2.9).

**Indexing.** `d[s]` is the view, a `MarginalView`: the law of `X`'s node at `s`, which refers to `d`, with `d[s].event_spec == d.event_spec[s]` (§ 4.2, I6); `marginal(d, s)` is the same law, detached (I7). `in` follows `[]` (§ 4.6). `without` returns a view (§ 4.5). `rename` renames and moves nodes by their paths, and renames the output at the root (§ 4.8). `d.output(n)` reads the term at the output at the root (§ 4.11).

**Merge.** Laws and kernels form one merge category, and each operand is read as its factors (§ 2.15). A law feeds the left operand; the value is § 2.20's rule. M2 refuses a law whose draws are laws, kernels, functions, or batches held under no name (§ 2.17).

**Examples.**

```python
prior.event_spec                     # OutputSpec(RecordSpec(mu=..., tau=...)): the outputs mu and tau
grouped.event_spec.outputs           # {a: NumericArraySpec(()), g: RecordSpec(c=(), e=())}
Normal("x", 0.0, 1.0).output("x")    # the law itself: its output x is at the root
model = lik & prior                  # prior's output mu feeds lik's input mu; a law over {y, mu, tau}
```

**Views.**

`d[p]` returns a `MarginalView` (today `FieldView`): a law over the node at `p` that refers back to `d`.
Its event spec is `d.event_spec[p]` (§ 4.4):
1. a base node is named by the last name of `p`, in the keyword form;
2. a node with parts is positional, so the view offers its parts;
3. a node at or below a field that holds a law, a kernel, a function, or a batch is positional too, with no wrap: `rm["G"]` is a law whose draws are laws (below).

To keep a group's name, index with a list of one path:

```python
grouped["a"]            # a law in the keyword form, named a; its draws are arrays
grouped["g/c"]          # a law in the keyword form, named c
grouped["g"]            # offers c and e; its draws are records {c, e}         (today: a whole term g; step G1a)
grouped[["g"]]          # offers the record {g: {c, e}}: one output, g         (today: TypeError; step S1)
grouped["g"]["c"]       # the view of g/c                                      (today: KeyError; step G1a)
grouped["a", "g/c"]     # offers {a, g: {c}}: outputs a and g                  (today: {a, c}; step S1)
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

**A law over a record, named as a whole.**

Today `params_prior` can be declared `OutputSpec(params=RecordSpec(beta=(), sigma=()))`, and its draws are `{beta, sigma}`.
It is written positionally or wrapped:

```python
# positional: outputs beta and sigma; draws {beta, sigma}
params_prior = distribution(sample=sample_params,
                            event_spec=OutputSpec(RecordSpec(beta=(), sigma=())))

# wrapped: one output params; draws {params: {beta, sigma}}
def sample_params_wrapped(key):
    return {"params": sample_params(key)}
params_prior = distribution(sample=sample_params_wrapped,
                            event_spec=OutputSpec(RecordSpec(params=RecordSpec(beta=(), sigma=()))))
params_prior["params/beta"]          # the view of beta; the same path on its draws
```

In a merge, a kernel with the input `params` is fed by the wrapped law, since that law offers `params`.
The positional law offers `beta` and `sigma`, which feed inputs of those names.

**A law whose draws are laws, functions, or batches.**

Such a law can be declared positionally: a random measure whose draws are laws of spec `law_spec` has the event `OutputSpec(law_spec)`, which offers no outputs, and a merge refuses it (M2).
To give the draws a name, the law wraps them, and a distribution family names its own law's draws under the name its constructor takes (C5):
- a random measure constructed under the name `G` draws `{G: law}`;
- a random function's draws, such as a Gaussian process's, are functions, stored in a record: `{f: function}`; the law keeps offering `f`, and each draw gains a level;
- a bootstrap replicate law's draws are batches, stored in a record under its name, as `{b: batch}` for `BootstrapReplicateDistribution("b", ...)`;
- an empirical law whose atoms are laws or functions, and a minibatched target, wrap their draws the same way;
- lifting over a law a function that returns a law, a function, or a batch gives a law whose draws are of that kind, and the lift wraps them under the callable's output name (§ 4.12).

Today such a law fails in a merge, because a law's positional outputs are typed by one draw's value:

```python
# Today: a law whose draws are laws, declared positionally
mu_law = Normal("mu", 0.0, 1.0)
laws_of_laws = distribution(sample=lambda key: Normal("mu", jax.random.normal(key), 1.0),
                            event_spec=OutputSpec(mu_law.spec))
joint = laws_of_laws * Normal("x", 0.0, 1.0)
joint.event_spec        # declares the field mu as a NumericArraySpec
sample(joint)           # ResultSchemaError: ... output/mu does not conform to NumericArraySpec(...): got Normal

# Proposed: laws_of_laws builds, and offers no outputs; the merge raises TypeError (M2)
laws_of_laws & Normal("x", 0.0, 1.0)   # TypeError: the left operand produces a law; wrap it under a name
laws_of_laws = distribution(sample=lambda key: {"G": Normal("mu", jax.random.normal(key), 1.0)},
                            event_spec=OutputSpec(RecordSpec(G=mu_law.spec)))   # wrapped: offers G, and merges
```

A law that draws laws is legal; it can be named only by a list, as `rm[["G"]]`, or by the code that builds it wrapping its draws.

**Change from today:** a law-valued or kernel-valued produced term offers no outputs, so `OutputSpec(post_spec).components` is empty (fact 21).

**Views at fields that hold a law, a kernel, a function, or a batch.**

Paths go through every kind, so a single-path view can name a field that holds a law, as `rm["G"]` does, or a node below it, as `rm["G/mu"]` does.

A view at or below such a field follows the rule of § 4.2 like any other node, and no wrap is added (§ 6.20, Q2(i)).
So `rm["G"]` is a law whose draws are the drawn laws: its event spec is `OutputSpec(law_spec)`, which offers no outputs (§ 2.11).
To keep the field's name, index with a list: `rm[["G"]]` is the law over records `{G: law}`, which offers `G` and merges.

```python
rm["G"]                    # a law whose draws are laws; event_spec OutputSpec(law_spec); outputs {}   (G3)
sample(rm["G"])            # a law, distributed as sample(rm)["G"]            (I2, with no exception)
rm["G"]["mu"] == rm["G/mu"]                  # I5, with no exception
rm[["G"]]                  # a law over {G: law}; one output, G
lik_G & rm[["G"]]          # rm[["G"]]'s output G feeds lik_G's input G
lik_G & rm["G"]            # TypeError (M2): rm["G"] produces a law; index with [["G"]] to keep the name
```

**Change from today:** such a view is a whole term named by the field.

**`marginal`.**

`marginal(d, paths)` takes what `d[paths]` takes and returns the same law, detached from `d` (I7).
The parameter is renamed from `field` to `paths`, documented as "a path, or a list of paths", as in `without(*paths)`.

```python
marginal(grouped, "g")              # the law of the group's contents, offering c and e   (step G1a)
marginal(grouped, ["g"])            # the law of the record {g: {c, e}}, offering g
marginal(grouped, ["a", "g/c"])     # a Python list works  (today: TypeError, unhashable list)
marginal(grouped, ("a", "g/c"))     # the same call, the same result
```

A marginal over the whole events of several factors keeps the factors in the law's order.
Today `marginal(a_law & b_law, ("b", "a"))` lists the factors as `b(b)·a(a)`, in the order of the paths; under this plan it lists them as `a(a)·b(b)`.

**Change from today:** the parameter is named `field`, a list is refused, several paths flatten to their last names, and a group's marginal is the factor that holds it, whole under the group's name (VI.8).

**`condition_on`.**

A given value is a mapping from paths to values, or a `Record`, read by its paths.
The result's event spec is `d.event_spec.without(*fixed)` (I11, decided 2026-10-15), so it is a positional record of the remaining nodes whichever route computes it:

```python
condition_on(prior, {"mu": 0.1})    # a law over the record {tau}
                                    # (today: OutputSpec(tau=...), the keyword form, from the exact Gaussian route;
                                    #  a Monte Carlo route returns a record instead)
post["mu"]                          # works for every posterior
```

Nested paths, records, and arrays are passed as follows:

```python
condition_on(model, {"y": y_observed})              # unchanged
condition_on(grouped, {"g/c": 0.5})                 # a nested path
                                                    # (today: ResolutionError, "conditioning on the nested field 'g/c' is not supported yet")
condition_on(grouped, sample(grouped["a", "g/c"]))  # a record read by its paths (I8)
condition_on(grouped, {"g": sample(grouped["g"])})  # one group: its draw is the record {c, e}, so name it
condition_on(grouped, sample(grouped[["g"]]))       # or draw the record {g: {c, e}} instead
condition_on(model, {"mu": sample(prior["mu"])})    # one array: name it too
condition_on(lik, mean(post[["mu"]]))               # a plug-in estimate: the record {mu} binds lik's input mu
```

I8 covers lists of paths only.
The draw of one group, `sample(grouped["g"])`, is the record `{c, e}`.
Passed alone, it would be read at the top level, where `c` and `e` are not paths of `grouped`; today that raises `ResolutionError: condition_on: unknown fields ['c', 'e']`.
The nested-path row states the contract.
Where a route cannot yet condition at a nested path, it raises `ResolutionError` naming the route, rather than refusing the path for every law.
Binding a kernel keeps today's rule (§ 2.6).

**`log_prob`.**

The value is read by its paths, so a summary scores where a draw scores whenever its spec fits the event's (#576):

```python
log_prob(prior, sample(prior))                # unchanged
log_prob(prior, mean(prior))                  # the score at the mean (works since #576)
log_prob(Normal("x", 0.0, 1.0), 0.3)          # an output at the root takes the array    (unchanged)
log_prob(bern, mean(bern))                    # refused: the float32 mean cannot be cast to the int32 event
                                              # (today: ApplicabilityError, as here)
```

No step changes `log_prob`; it follows the reader of § 4.1.

**Summaries.**

A summary that has one value per field of the event keeps the law's paths and the form of its output spec, and the operation appears only in its label (decided 2026-10-15; PR #576, merged).
This holds for `mean`, `variance`, `standard_deviation`, and `quantile`, and `expectation(d, f)` takes the paths of `f`'s output.

```python
grouped_mean = mean(grouped)
grouped_mean["a"], grouped_mean["g/c"]   # the means of a and g/c       (works since #576)
grouped_mean["g"]                        # the record of means {c, e}   (today: KeyError, a record stops at groups; step S1)
mean(Normal("x", 0.0, 1.0))              # an array: the output x at the root, like a draw
grouped_mean.label                       # "𝔼[(a, g) ~ Normal·Normal·Normal]": the label says it is a mean
```

A summary binds wherever a draw of the same type binds, so plug-in estimates work, such as `condition_on(lik, mean(post[["mu"]]))` and `log_prob(prior, mean(prior))`.
The guard is the type, not the name.
A summary's spec is derived from the event's: a mean is floating, and its support is the hull of the event's support.
So a Bernoulli law's mean is refused where its `int32` event is required.
`condition_on(model, standard_deviation(post))` binds whenever the types fit; like passing the wrong array, it is the caller's mistake to make.

Results that are not one value per field are base values under the operation's name: `cov(d)` is one array, and a score is declared under `log_prob`.

**`FieldView` becomes `MarginalView` (decided 2026-10-17).**
The class covers groups and lists of paths, not only fields, so "field" undersells it.
By I7 it is the marginal law at its paths, kept as a view rather than detached, and the name pairs with `marginal`.
The rename changes the class, its repr, III.7, III.8, and the User Guide; `d[s]` is unchanged.
It merges in step S3.

### 5.6 Kernel

**What it is.** A `ConditionalDistribution`: a law for each value of what it conditions on, `K : S → P(T)`, with at least one input (III.9). Its term spec is a `ConditionalDistributionSpec`.

**Produced term.** At each value `w` of its inputs, a random draw `X ~ k(w)` (§ 2.5).

**Parts.** The paths of the draws of the law it gives (§ 2.4).

**Inputs.** `k.given_spec`, with at least one input. A parameter with a default is an optional input, and a kernel whose inputs are all optional is still a kernel (§ 2.6). Binding a kernel's last required input gives a law, whose unbound optional inputs take their defaults (§ 2.6). No name is both an input and an output (§ 2.9).

**Outputs.** Those of `k.event_spec`.

**Specs it holds.** `event_spec`, one draw at one value of the inputs, and `given_spec`, its input spec, named for what it is conditioned on (§ 2.8, § 2.9).

**Indexing.** `k[s]` is the kernel whose law at each input record `w` is the law of `X`'s node at `s`, for `X ~ k(w)`, and indexing keeps its inputs (§ 4.2, I13, I14). `in` follows `[]`. `without` follows § 4.5, and a kernel's own `rename` renames its inputs as well as its outputs (III.9). `k.output(c)` reads the term at an output (§ 4.11).

**Merge.** In the category of laws and kernels, read as its factors (§ 2.15); a kernel feeds and is fed. Its unfed inputs are its `given_spec` (§ 2.15), and a merge that leaves a required unfed input gives a kernel (§ 2.19, R3).

**Examples.**

```python
lik.given_spec                       # InputSpec(mu=NumericArraySpec(())): one required input
lik.event_spec                       # OutputSpec(y=NumericArraySpec(())): one output y, at the root
lik.output("y")                      # lik itself, since its output y is at the root
lik_opt.given_spec                   # InputSpec(mu=..., s=...).with_optional("s")
condition_on(lik_opt, {"mu": 0.3})   # a law: Normal("y", 0.3, 1.0), with s at its default
lik_g.given_spec["g/c"]              # NumericArraySpec(()): a path into the record input g (§ 5.10)
data_model = lik_nee & lik_lai       # a kernel in the shared input theta (§ 2.19)
```

### 5.7 Function

**What it is.** A `Function`: a tracked callable, whose `FunctionSpec` holds an `input_spec` and an `output_spec`, either of which may be `None` (III.3).

**Produced term.** Its return value at each value of its inputs (§ 2.5).

**Parts.** The paths of what it returns; a function that declares no output has no paths known before it is called (§ 2.10).

**Inputs.** `f.input_spec`, read through the `InputSpec` constructor (§ 5.10); a parameter with a default is an optional input, and an `input_spec` that marks optional a parameter without one raises `ValueError` (O9; § 2.6).

**Outputs.** Those of `f.output_spec`. A function into records is the record of its output functions, so it offers them; a function that returns a law offers none (§ 2.11).

**Specs it holds.** `output_spec`, one return value, and `input_spec` (§ 2.8, § 2.9). A bare term spec given as `output_spec` reads by `OutputSpec.default` (§ 4.13).

**Indexing.** `f[s]` is the function `w ↦ f(w)[s]`, with the same inputs (§ 4.2, I13, I14). `f.output(c)` reads the term at an output (§ 4.11). `without` follows § 4.5.

**Merge.** Functions form one merge category, and functions feed: a function is the kernel whose every law is a point mass (below).

**Examples.**

A function is indexed by the paths of what it returns, and indexing comes after the call (I13).
A function with a declared output checks the paths when it is indexed.
A function without one, whose `output_spec` is `None`, checks them when the indexed function is called, against what the call returns.

```python
emulator["mean"]             # the function theta -> emulator(theta)["mean"]   (today: TypeError, no [])
emulator["mean"](theta) == emulator(theta)["mean"]
emulator[["mean"]]           # the function theta -> Record {mean}
fit["mu"]                    # the function y -> run_inference(y)["mu"], for `fit` below
```

Functions gain `[]` in step G3.

**A function that returns a law.**

Today `Function(run_inference, output_spec=OutputSpec(posterior=post_spec))` returns the law under the name `posterior`.
A law has parts, so the output spec is positional over the law's spec, or the body wraps the law:

```python
# positional: the function returns the law, which offers no outputs
fit = Function(run_inference, output_spec=OutputSpec(post_spec))
fit.output_spec.outputs              # {}: a law, with no outputs (§ 2.11)
fit["mu"]                            # the function y -> run_inference(y)["mu"]   (step G3)
fit.output_spec["mu"] == fit["mu"].output_spec   # OutputSpec(DistributionSpec(OutputSpec(mu=...)))

# wrapped: the function returns a record that holds the law
def run_inference_wrapped(y_observed):
    return {"posterior": run_inference(y_observed)}
fit_wrapped = Function(run_inference_wrapped, output_spec=OutputSpec(RecordSpec(posterior=post_spec)))
fit_wrapped(y_observed)["posterior"]          # the law; an input posterior is fed it in a merge
fit_wrapped.output_spec["posterior"]          # OutputSpec(post_spec), as fit_wrapped(y)["posterior"] is the law
```

**The body builds the record (decided 2026-10-17).**
A function with a wrapped output returns the record itself, and the output spec checks it.
ProbPipe never wraps a function body's return value.
So `run_inference_wrapped` returns `{"posterior": law}`, and a body that returns the law alone fails the output spec's check.

**A returned law offers no outputs (§ 6.20, Q2(ii)).**
The positional form keeps the law whole and offers nothing: its marginals are not offered as outputs, so no merge receives `mu` and `tau` as two marginal laws that have lost the dependence between them, and no input is lifted over a marginal.
`fit["mu"]` still reads the function that returns the view, since `[]` takes paths on every kind.
A merge refuses `fit` (§ 2.17, M2), and `fit_wrapped` merges, offering `posterior`.
Today `OutputSpec(post_spec).components` is `{mu: NumericArraySpec(()), tau: NumericArraySpec(())}`, the specs of a draw's fields rather than of anything the function returns (fact 11).

**A function that returns a function.**
`fit_emulator` returns `emulator`, which is a function from `theta` to `{mean, var}`.
A function into records is the record of its output functions, so the positional form offers the returned function's outputs, `mean` and `var` (§ 2.11):

```python
fit_emulator = Function(train_gp, output_spec=OutputSpec(FunctionSpec(
    InputSpec(theta=theta_spec), OutputSpec(RecordSpec(mean=("n_days",), var=("n_days",))))))
list(fit_emulator.output_spec.outputs)       # ["mean", "var"]   (step G1b)
emulator = fit_emulator(design=design, runs=runs)
emulator["mean"]                             # the Function theta -> emulator(theta)["mean"]   (step G3)
```

By I12, `fit_emulator.output("mean")` is `fit_emulator["mean"]`, which is the function that returns `emulator["mean"]`.
In turn, `emulator["mean"]` is the function from `theta` to the mean (I13).
Naming the returned function from outside, as `OutputSpec(emulator=FunctionSpec(...))`, raises; wrap it to keep the name.
A merge refuses `fit_emulator`, since its produced term is a function, which a record holds only under a name (§ 2.17, M2).

**Change from today:** `OutputSpec(FunctionSpec(...))` raises `TypeError` today, since a `FunctionSpec` cannot be positional (fact 12).

**Merge of functions.**

A function is the kernel whose law at each input is a point mass at its return value (D2: a special case refines the general contract).
So the merge of two functions is the merge of two kernels, read deterministically, and § 2.20 gives its value; a function's optional inputs are always inputs of the function, as a kernel's are.

So the law rule of § 2.20 reads, for functions:

```text
w_b = rec_b( b(u↾names(unfed(b))) )
w_a = rec_a( a(u↾(names(unfed(a)) − fed) ⊎ w_b↾fed) )
(a & b)(u) = w_a ⊎ w_b
```

```python
center = Function(lambda y: jnp.mean(y), input_spec={"y": ("n",)}, output_spec=OutputSpec(center=()))
spread = Function(lambda y: jnp.std(y), input_spec={"y": ("n",)}, output_spec=OutputSpec(spread=()))
summaries = center & spread                        # input y, shared; outputs center and spread
standardize = Function(lambda y, center, spread: (y - center) / spread,
                       input_spec={"y": ("n",), "center": (), "spread": ()}, output_spec=OutputSpec(z=("n",)))
pipeline = standardize & summaries                 # summaries feeds center and spread; y is shared
pipeline(y=data)                                   # Record {z, center, spread}
pushforward(d, center & spread)                    # the joint law of both summaries of one draw of a law d over y
pushforward(d, center) & pushforward(d, spread)    # their independent product
```

**Complete specs.**
A function merges only with a declared input spec and a complete output spec (M2).
A function whose output spec is a type hole, such as `OutputSpec(center=None)`, does not merge: the result's output spec is a record of the operands' outputs, and a `RecordSpec` cannot hold a pending child (fact 28).
`&` raises `TypeError`, naming the hole and saying to give its spec.
A function with no output spec or no input spec, which `Function(f)` gives today (fact 20), raises `TypeError` too, saying which spec to pass.

**A function that returns a law** fails M2, since a law offers no outputs and a record holds it only under a name: `fit & center` raises, and `fit_wrapped`, whose body returns `{"posterior": law}`, merges (§ 6.2.6).

### 5.8 Batch

**What it is.** A `Batch`: many separate terms of one kind, stacked along named axes called levels; one of them is an element (II.5). Its term spec is a `BatchSpec`, which holds the element spec and the levels.

**Produced term.** The batch itself (§ 2.5).

**Parts.** Its elements' paths; the node of a batch at `p` is its column at `p` (§ 2.4, § 2.5). Positions index its own axes (§ 2.4).

**Inputs.** Its elements' inputs: `given_spec` for a batch of kernels (III.10), `input_spec` for a batch of functions, and none for a batch of values or laws (§ 2.9).

**Outputs.** `OutputSpec(bt.spec)`, derived: its columns for a batch of records or of functions, one output at the root for a batch of base values, and none for a batch of laws or kernels (§ 2.9, § 2.11).

**Specs it holds.** `element_spec`, the output spec of one element as a term (§ 2.9, below).

**Indexing.** By position, a sub-batch or an element; by path, the batch of each element's node (§ 4.2), so `bt[j][s] == bt[s][j]` (I4). `in` tests paths, never elements (§ 4.6). `without` exists for batches of records and of laws (§ 4.5). `rename` renames the elements' output at the root (§ 4.8, case 2).

**Merge.** Two batches over equal levels merge element by element, each element read as one operand (§ 2.16).

**Examples.**

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

A batch of draws of `Normal("mu", 0.0, 1.0)` has no paths below its elements, so `draws["mu"]` raises `KeyError`: `mu` names the output at the root, not a path inside an element.
The message says so: "'mu' names the elements of this batch as a whole, not a path inside them; the batch already holds mu's values (read the array with .raw)".
`draws.output("mu")` reads it (§ 4.11), and `draws.spec.element_spec["mu"]` raises `KeyError` too (§ 4.4, rule 4).
The list form at the root works on batches as on laws (§ 4.4, rule 3): `draws[["mu"]]` is the batch of records `{mu}`, since the batch stores the name (below).
Today `draws["mu"]` raises `TypeError` (fact 18).

A batch of laws indexes each law: `laws["mu"]` is the batch of the views `law["mu"]`.
Today it raises `TypeError` with the false message that the elements have no named fields.
A batch of functions does the same with each function.
These two arrive with step G5, and the list form at the root with S4; the rest of the examples above is step S1.

**The element spec.**

A `BatchSpec` has parts, its elements', so a batch's output spec is always positional, through its `element_spec`.
So `sample(Normal("x", 0.0, 1.0), sample_shape=4)` offers `x` as today, and `NumericArrayBatch(values, "draw", label="draws")` offers `element`.
A sweep or a function that returns a batch offers the batch's columns, or wraps the batch to name it.
**Change from today:** `OutputSpec(BatchSpec(NumericArraySpec(()), sample=3))` raises `TypeError` today, since only a batch of records or laws can be positional.

Every batch holds an **element spec**, `BatchSpec.element_spec`: the output spec of one element as a term (§ 2.9; step S4).
Today's `element_spec`, the element's term spec, becomes `element_spec.term_spec`, as `event_spec.term_spec` is a draw's term spec; S4 changes every use of the old meaning in the one PR (directive 7), `Batch.element_spec` included (§ 6.21).
1. **A batch an operation produces** stores its maker's output spec of an element: the law's event spec for draws (VI.3), and the completed output spec for a sweep (V.6).
2. **A batch built from raw values** gets the default: a record element is positional, a law, kernel, or function element is positional over its own spec, and a base element is named `element`, in the keyword form.
3. **A column of a batch of records** follows the rule for views of § 2.7: `grouped_draws["a"]` stores the keyword form `OutputSpec(a=...)`, and a group's column `grouped_draws["g"]` stores the positional record `{c, e}`.

A batch of records offers its columns, each typed by its column's `BatchSpec` (§ 2.11); today they are typed by the element (fact 11), and a probe of the whole suite found no test that depends on that typing (fact 30).
A group's column is positional, as the law's view of the same group is: `grouped_draws["g"].element_spec` and `grouped["g"].event_spec` both offer `c` and `e`.

The rule of § 2.7 applies to the stored element spec as to any other output spec.
A batch of draws of `grouped` stores the positional record `{a, g: {c, e}}`, and a batch of draws of `Normal("mu", 0.0, 1.0)` stores the keyword form under `mu`.
For a base element, the name is the one thing `element_spec` holds beyond the term spec; for every other element, `element_spec` is the positional form over the element's term spec, so `BatchSpec` still stores one thing (D6).
`Batch.components` (II.5) is not added: `bt.element_spec.outputs` lists the elements' outputs.
O8 says how the constructor takes an element spec.

```python
# Proposed
draws.spec.element_spec                # OutputSpec(mu=NumericArraySpec(shape=(), dtype=float32, support=real))
draws.spec.element_spec.term_spec      # NumericArraySpec(...): today's draws.spec.element_spec
list(draws.element_spec.outputs)       # ["mu"]
NumericArrayBatch(jnp.arange(4.0), "draw", label="xs").element_spec   # OutputSpec(element=...)
grouped_draws["a"].element_spec        # OutputSpec(a=...)
grouped_draws["g"].element_spec        # OutputSpec(RecordSpec(c=..., e=...))
OutputSpec(grouped_draws.spec).outputs["a"]   # the BatchSpec of the column a
laws.spec.element_spec                 # OutputSpec(law_spec), for a batch of laws each of spec law_spec: no outputs
laws.event_spec == laws.spec.element_spec.term_spec.event_spec   # III.10's event_spec of the elements
```

**What indexing uses.**
A batch is indexed by its elements' paths, so it agrees with its law without reading the stored names (above).
The stored names serve merging, binding, and the accessor `t.output(c)`:

```python
draws["mu"]              # KeyError: mu names the output at the root, not a path inside an element
draws.output("mu")       # draws itself, as Normal("mu", 0.0, 1.0).output("mu") is the law   (step G3)
sample(grouped, sample_shape=4)["g/c"]    # a column, read through the elements' paths
```

**Names stay out of a batch's equality and JAX (decided 2026-10-17).**

A batch's output names stay out of its equality, its hashing, the static data of its JAX pytree, and its fingerprint.

**Why.**
A batch is a JAX pytree, so JAX splits it into arrays and static data.
JAX compares the static data to decide whether two values have the same structure, which decides whether `jit` reuses a compiled JAX program and whether the two branches of `lax.cond` may return the batch.
Today a batch's static data is its `BatchSpec`.
If the names were static data, draws of `Normal("x", 0.0, 1.0)` and of `Normal("y", 0.0, 1.0)` would become different structures:

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=3)
y = sample(Normal("y", 0.0, 1.0), sample_shape=3)
jax.tree_util.tree_map(jnp.subtract, x, y)   # works today, and keeps working;
                                             # with names in the static data it would raise "Mismatch custom node data"
```

A `jit`-compiled function would also recompile for each name, and the two branches of a `lax.cond` could not return draws under different names.

**The cost.**
Keeping the names out costs what labels already cost, and only for the names of base elements.
The names of a record element are stored in the element's `RecordSpec`, which is static data already, so they survive.
A base element's name, as in a batch of arrays or a column such as `grouped_draws["a"]`, does not cross a raw JAX transform:

```python
jax.tree_util.tree_map(jnp.subtract, x, y)   # today: labeled '<no description>', as every raw rebuild is
jax.jit(lambda batch: batch)(x)              # comes back offering the default output element, not x
```

So `.output("x")` on that result, or feeding it to an input `x`, fails.
ProbPipe's own operations restore the names at their result, where they declare their result anyway.

**How the name is stored (O3).**
`BatchSpec` keeps the element spec, the name of a base element included, and its own equality and hash compare that name.
Every place where the decision keeps names out uses a name-free canonical form of the spec instead:
1. a batch's equality and hashing compare canonical specs;
2. fingerprints hash the canonical spec, as a `FunctionSpec` already hashes its output spec without names (`_update_value_spec` in `core/_fingerprint.py`);
3. every pytree flatten emits canonical static data, recursively: a batch's own `BatchSpec`, and any `BatchSpec` held inside other static data, such as the `RecordSpec` of a record whose field holds a batch, or a `DistributionSpec` or `FunctionSpec` whose output spec holds one.

Two reasons rule out comparing names in some places and not others inside one `BatchSpec`:
- JAX rebuilds the output of a cached compiled program from the static data of the first call, so a name kept in static data that compares without it comes back from a later call made under another name. A pytree whose static data compares equal across names shows this: `jax.jit(lambda b: b)` called with name `x` and then `y` returns `x` both times. The same holds one level down, for a record whose field holds a batch.
- Equal objects that answer differently are unsafe anywhere they serve as keys. The execution plan keys its probe results and its program store by input structure, which it defines through ProbPipe's canonical flatten (`execution-layer.md` § 5), so base-element names never enter those keys and a name never causes a recompile; a requester restores names on its results. A spec captured by a closure is compared in its canonical form for the same reason.

jhuggins's model proves that `&` commutes with renaming every name at once (`dmerge_rename`).
A requester restores each input's output spec from the stored input structure of the evaluation points, once a body can ask for tracked terms (O4, option 1); T31 holds the rule until then.

**Merge of batches.**

`(a & b)[j] = ã_j & b̃_j` at every position `j` (§ 2.16).
`RecordBatch.merge` does this today for batches of records, with equal levels (`core/_record_batch.py:754`, `:783`).
VI.11 exempts the operands of `*` from batch lifting (`design/06-operations.md:392`); S0 amends it to say that two batches merge element by element, and aligning other levels by name, with broadcasting, stays after the workshop.

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=3)
y = sample(Normal("y", 0.0, 1.0), sample_shape=3)
x & y                              # a batch of three records {x, y}: each element is {x: x_j} ⊎ {y: y_j}   (after the workshop)
post_draws & simulate(post_draws)  # elements {mu, tau} and {y} give elements {mu, tau, y}   (today: .merge(...))
```

### 5.9 `RecordSpec`

**What it is.** The term spec of a record. A spec is not a term: it produces and receives nothing, and each operation on it gives what it says about the same operation on the records it types (§ 5.1).

**Parts.** Its paths, as a record's (§ 2.10, case 1).

**Inputs and outputs.** It has one reading: `OutputSpec(R)` reads its children as outputs, and `InputSpec(R)` reads them as the inputs those outputs would feed (below).

**Specs it holds.** None (§ 2.8).

**Indexing.** A `RecordSpec` indexes as a record does, returning specs: a path gives the spec at that path, and a list gives the cut-down `RecordSpec`; `without` takes paths, and `in` follows `[]` (§ 4.6).

**Merge.** `R & S` is the union of the children; a top-level name in both raises (§ 2.18).

**Examples.**

```python
draw_spec = sample(grouped).spec      # RecordSpec {a, g: {c, e}}
draw_spec["g"]                        # RecordSpec {c, e}                 (today: KeyError; step S1)
RecordSpec(a=()) & RecordSpec(b=())   # RecordSpec(a=(), b=())
```

**A `RecordSpec` has one reading.**
`OutputSpec(R)` reads its children as outputs, and `InputSpec(R)` equals `InputSpec(OutputSpec(R))`: the inputs that those outputs would feed (§ 5.10).

```python
R = RecordSpec(growth=RecordSpec(K=(), r=()), phi=())
OutputSpec(R).outputs         # {growth: RecordSpec(K=(), r=()), phi: NumericArraySpec(())}
InputSpec(R)                  # two inputs, growth (a record) and phi: the same as InputSpec(OutputSpec(R))
```

### 5.10 `InputSpec`

**What it is.** The input spec of a term: an ordered mapping from input names to term specs, with the set of optional inputs (§ 2.6). It is a spec, not a term, and it stores no defaults.

**Parts.** Its inputs, which are flat names; a path reaches into a record input (below).

**Inputs and outputs.** It lists inputs, and states no produced term, so no output spec follows from it (§ 2.23, row 18).

**Indexing.** `[]` takes a name or a list of names, and a path into a record input; `in` follows `[]`; `without` and `rename` take names (below).

**Merge.** `I & J` is the input spec of two terms that share their inputs, with nothing fed (§ 2.18).

**Examples.**

```python
InputSpec(prior.event_spec)                   # the inputs mu and tau: the inputs prior's outputs could feed
InputSpec(theta=("p",), x=()) & InputSpec(theta=(3,), z=())   # theta unifies, and p is bound to 3 (§ 2.18)
```

**One reading of inputs: the four sources.**

Every place that takes a term's inputs reads them through the `InputSpec` constructor, which accepts four sources:

1. **An `InputSpec`**, as it is.
2. **A mapping** from input names to term specs.
3. **A `RecordSpec`**, read by its children, the rule that the positional form of `OutputSpec` follows.
4. **An `OutputSpec`**, read by its outputs (§ 2.11), each input typed by the spec of its output: the inputs that those outputs would feed.

The constructor tests for a `RecordSpec` and an `OutputSpec` before a plain mapping.
A `RecordSpec` is not a `Mapping`, and its tree interface iterates leaf paths, so reading it as a plain mapping would bring back defect 1; an `OutputSpec` must be read by its outputs.
Only the `OutputSpec` branch refuses a type hole, since only an output spec can hold one; every branch refuses a name that is not an identifier (§ 2.6).
`InputSpec(R)` equals `InputSpec(OutputSpec(R))` for every non-empty `RecordSpec` `R`, and `InputSpec(RecordSpec())` is `InputSpec()`.
An output name that is not an identifier raises `ValueError`, and a type hole raises `TypeError`.

Every input of a mapping, a `RecordSpec`, or an `OutputSpec` source is required.
`conditional_distribution` and `Function` then mark optional the inputs whose parameters have defaults, and refuse an optional input without one (§ 2.6, O9).
Three arguments take every source that the constructor takes:
- `conditional_distribution`'s `given_spec`;
- `Function`'s `input_spec`;
- `ConditionalDistributionSpec`'s `given_spec`.

A nested record therefore declares one input per child, which removes defect 1:

```python
InputSpec(RecordSpec(growth=RecordSpec(K=(), r=()), phi=()))
# InputSpec(growth=RecordSpec(K=..., r=...), phi=...): two inputs, growth holding a record
InputSpec(growth_prior.event_spec)                       # the inputs K, r, phi, and sigma
InputSpec(OutputSpec(post.spec))                         # InputSpec(): a produced term that is a law offers no outputs
conditional_distribution(lik_grouped_fn, given_spec=regrouped.event_spec.without("sigma"))
# builds, for a function lik_grouped_fn(growth, phi): the kernel's inputs are growth and phi
```

**The messages for the common mistakes (step S2b).**
The recommended idiom is `given_spec=law.event_spec`, cut down with `[...]` or `without` (§ 6.5.3).
Four mistakes will be frequent, and each message names the fix, under `STYLE_GUIDE.md` §9.3:
1. **An input the function does not take.** For `conditional_distribution(likelihood_fn, given_spec=growth_prior.event_spec)`, where `likelihood_fn` takes `K`, `r`, and `phi`: "given_spec of 'likelihood_fn' declares 'sigma', which the function does not take (it takes K, r, phi); leave it out with event_spec.without('sigma'), or pick the inputs with event_spec[['K', 'r', 'phi']]".
2. **The parts of a group, where the function takes the group.** When the declared inputs are the children of one group `g` and the function takes a parameter `g`, as for `given_spec=grouped.event_spec["g"]` and a function `f(g)`: "given_spec of 'f' declares c and e, the parts of the group g; to declare the input g, index with a list: event_spec[['g']]".
3. **A law, or a law's spec, as the source.** For `given_spec=growth_prior` or `given_spec=growth_prior.spec`: "given_spec takes an InputSpec, a mapping, a RecordSpec, or an OutputSpec, not a Distribution; for the outputs of a law, pass law.event_spec" (with "DistributionSpec" for the second).
4. **An input spec merged with an output spec** (§ 2.17, M1): "cannot merge an InputSpec with an OutputSpec; to read an output spec as the inputs it feeds, write InputSpec(output_spec)".

**Change from today:** `InputSpec` refuses a `RecordSpec` and an `OutputSpec`, the three arguments each read their source their own way (`_slots_of`, `_complete_given_spec`, `Function._initialize`), and `Function` does not mark a parameter with a default optional.

**Selecting inputs: `[]`.**

`InputSpec` stays a `Mapping` from input names to term specs: `keys()`, iteration, and `len` read the inputs.
Its `[]` reads the path reader's argument against the inputs:

| Argument | Result | Step |
| --- | --- | --- |
| an input's name | that input's term spec | unchanged |
| a list of input names | the input spec of those inputs, in this spec's order, each optional input staying optional | S2b |
| a path into a record input, such as `"g/c"` | the term spec at that path | G4, after the workshop |
| a list with a path into a record input | the input spec whose input holds the cut-down record; that input becomes required, since a default is a whole value | G4, after the workshop |

Paths into a record input come with partial binding of a record input (III.9) in step G4, so they are design for after the workshop.
A name that is not an input raises `KeyError`, and the path reader's errors apply (§ 4.1).

```python
lik.given_spec["mu"]                          # NumericArraySpec(()): a term spec
likelihood.given_spec[["K", "r"]]             # InputSpec(K=..., r=...)
lik_g.given_spec["g/c"]                       # NumericArraySpec(())   (step G4)
```

**`without`.**

`I.without(*names)` returns the inputs other than `names`, each optional input staying optional.
A name that is not an input raises `KeyError`, as `RecordSpec.without` does for a path.
A removal that leaves no input stays valid, since a term may take no input.
A path into a record input leaves that input holding the rest of the record, and the input becomes required.

```python
lik.given_spec.without("mu")                  # InputSpec(): valid
lik.given_spec.without("zzz")                 # KeyError                            (today: ignored)
```

**Change from today:** `without` ignores a name that is not an input (`core/_specs.py:191`).

**`rename`.**

`InputSpec.rename(mapping=None, /, **kwargs)` renames inputs, as indexing a transition kernel by time does:

```python
step.given_spec.rename(x_prev="x2")
```

It follows the rename rule of II.6, restricted to names, since inputs are flat.
It belongs with spec merges for model builders (§ 6.16).
A kernel's own `rename` renames its inputs too (III.9).

### 5.11 `OutputSpec`

**What it is.** The output spec of a term: the term spec of its produced term, and a name when that term spec is a base spec or pending (§ 2.7). It is a spec, not a term.

**Parts.** Its paths: those of its term spec in the positional form, and none in the keyword form over a base spec (§ 2.10).

**Inputs and outputs.** `O.outputs` maps each output's name to its spec (§ 2.11), and `InputSpec(O)` reads the outputs as the inputs they would feed (§ 5.10).

**Indexing.** `O[s]` takes a path or a list of paths and gives an output spec (§ 4.4); `in` follows `[]` (§ 4.6); `without` removes nodes (§ 4.5); `rename` renames and moves nodes by path, or renames the output at the root (§ 4.8). An `OutputSpec` is not a `Mapping`, and is not iterable (below).

**Merge.** `O & P` is the output spec of two side-by-side terms, with nothing fed (§ 2.18).

**Not iterable.**
Since it has `[]`, Python would otherwise iterate it by calling `O[0]`, `O[1]`, and so on, so `__iter__` raises `TypeError`: "an output spec is indexed by path and is not iterable; list its outputs with .outputs".
That is a defined refusal, not an alias.

**Examples.**

```python
prior.event_spec                              # OutputSpec(RecordSpec(mu=..., tau=...))
list(prior.event_spec.outputs)                # ["mu", "tau"]
prior.event_spec.outputs["tau"]               # NumericArraySpec(()): the spec of the output tau
prior.event_spec["tau"]                       # OutputSpec(tau=NumericArraySpec(())): the output spec at the path tau
"tau" in prior.event_spec                     # True: a path of prior's draws (§ 4.6)
InputSpec(prior.event_spec)                   # the inputs mu and tau
lik.given_spec.keys() & prior.event_spec.outputs.keys()   # {"mu"}: the inputs prior's outputs feed
```

- `O.outputs[c]` is the spec of the output `c`: for a positional record, the child's spec, a group's `RecordSpec` included; for the keyword form, the base spec of the whole value, or `None` for a type hole.
- The order of `outputs` is the layout `to_vector` reads.
- `spec`, `with_spec`, `with_path_names`, and `exposes_record` go, with no aliases (decided 2026-10-17); `with_path_names` becomes `rename` in S3. There is no `exposes` property, since the term spec decides the form.
- `OutputSpec` has no `output` method: `O.outputs[c]` gives the spec of an output, and `O[c]` the output spec at a path (§ 6.20, Q5).

Code outside `core/_specs.py` reads the form through one private helper there, `_root_output_name(output_spec) -> str | None`, which returns the name of the output at the root, or `None`, by the rule that an output spec is in the keyword form when its term spec is a base spec.
It replaces today's `_whole_term_component` in `distributions/_distribution.py` and every read of `_component_name` outside `core/_specs.py`.

```python
O = grouped.event_spec                # positional over {a, g: {c, e}}
O.outputs["g"]                        # RecordSpec {c, e}: the spec of the output g
O["g"]                                # OutputSpec(RecordSpec(c=..., e=...)): outputs c and e, as grouped["g"] offers   (step S2b)
O[["a", "g/c"]]                       # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=...)))   (step S2b)
```

**`term_spec` and `with_term_spec`.**

`OutputSpec.spec` becomes `term_spec`, which is the glossary's term for the type of the produced term.
`term_spec` is `None` only while a type hole is pending.
`OutputSpec.with_spec` becomes `with_term_spec`, so the attribute and the method that completes it share one name.
Its rule is unchanged apart from the type holes of § 4.12: a type hole is filled, and a stated spec is completed by the produced one, as II.2's completion does.

```python
# Today
prior.event_spec.spec                          # NumericRecordSpec(mu=..., tau=...)
OutputSpec(mean=None).with_spec(NumericArraySpec(()))

# Proposed
prior.event_spec.term_spec                     # RecordSpec(mu=..., tau=...)
OutputSpec(mean=None).with_term_spec(NumericArraySpec(()))
```

A tracked term's own `spec` is unchanged, so `prior.spec` is still the law's `DistributionSpec`, and `prior.event_spec.term_spec` is the type of one draw.

**Declaring inputs from output specs.**

`O & P` on two output specs is the output spec of two side-by-side terms, with nothing fed; § 2.18 states it with the other merges of specs.
With `[]`, `without`, and `&`, the inputs of § 1.2 are declared with no conversion:

```python
likelihood = conditional_distribution(
    likelihood_fn,
    given_spec=growth_prior.event_spec.without("sigma") & shocks_given_sigma.event_spec,
    label="likelihood",
)
```

§ 6.2.4 maps each tutorial idiom to its replacement.

### 5.12 The specs that hold an output spec

**What they are.** `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, and `BatchSpec`: the term specs of a law, a kernel, a function, and a batch. Each holds the output spec of what its term produces or holds, `held(T)`, and `with_held(T, E)` replaces it (§ 2.8).

**Parts.** The paths of the held output spec: below a held law, function, or batch, the spec at a path is the spec of a term of the same kind whose produced term is the node at that path (§ 2.10, case 2).

**Inputs and outputs.** A `ConditionalDistributionSpec` keeps its `given_spec`, and a `FunctionSpec` its `input_spec`; `OutputSpec(T)` offers the outputs § 2.11 gives for `T`'s kind.

**Indexing.** They have no `[]` and no `without` of their own: index the held spec, as `post.spec.event_spec["mu"]`.

**Merge.** `&` on two of one of these is the spec of the merged term (§ 2.18). Conformance and unification compare the held output specs (§ 2.14, rules 5 to 7).

**Examples.**

```python
held(post.spec)                        # OutputSpec(RecordSpec(mu=..., tau=...))
post.spec.event_spec["mu"]             # index the held spec: OutputSpec(mu=NumericArraySpec(()))
spec_at(post.spec, "mu")               # DistributionSpec(OutputSpec(mu=...)): the spec of post["mu"]
held(post_draws.spec)                  # OutputSpec(RecordSpec(mu=..., tau=...)): one element
```

## 6. The API, examples, exceptions, vocabulary, steps, and decisions

### 6.1 The API sketch

Only the public members this plan adds or changes are shown, in the final words (§ 6.4).

**`InputSpec`.**

```python
class InputSpec(Mapping[str, TermSpec]):
    """A term's inputs: each input's name and term spec, in order, with the optional ones."""

    def __init__(self, inputs: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None,
                 /, **inputs_by_name: TermSpec | tuple) -> None: ...
    # Reads one source (§ 5.10): an InputSpec as it is; a mapping; a RecordSpec by its children; an
    # OutputSpec by its outputs, each typed by its output's spec. InputSpec(R) == InputSpec(OutputSpec(R)).
    # Every input of a mapping, record spec, or output spec is required. ValueError for a name that is
    # not an identifier; TypeError for a type hole, any other source, or both forms at once.
    @property
    def required(self) -> tuple[str, ...]: ...                  # unchanged
    @property
    def optional(self) -> frozenset[str]: ...                   # unchanged
    def with_optional(self, *names: str) -> InputSpec: ...      # unchanged
    @overload
    def __getitem__(self, name: str, /) -> TermSpec: ...
    # That input's term spec; a path into a record input gives the spec there. KeyError for a name
    # that is not an input.
    @overload
    def __getitem__(self, names: Sequence[str], /) -> InputSpec: ...
    # The input spec of those inputs, in this spec's order, optional ones staying optional (S2b). The
    # path reader's errors; KeyError for a name that is not an input.
    def __contains__(self, name: object, /) -> bool: ...
    # Whether name is a str naming an input, or a path into a record input; False for a list or a
    # tuple. Mapping's default would answer True for ["mu"], since self[["mu"]] succeeds (S2b).
    def get(self, name: str, default: TermSpec | None = None, /) -> TermSpec | None: ...
    # self[name] when name in self, else default; a list or a tuple gives default, as for __contains__ (S2b).
    def without(self, *names: str) -> InputSpec: ...
    # The other inputs, optional ones staying optional; may leave none. KeyError for a name that is not
    # an input. A path into a record input leaves the rest of that record, and the input becomes
    # required.
    def __and__(self, other_input_spec: InputSpec, /) -> InputSpec: ...
    # The inputs of both, in order of first appearance; a shared input carries the unification of its
    # specs (§ 2.14) and is optional only when optional in every operand that has it; one dimension scope, not kept.
    # TypeError for an operand that is not an InputSpec; ValueError when two specs do not unify or a
    # dimension is bound to two sizes.
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> InputSpec: ...
    # Rename inputs, by the rename rule of II.6 restricted to names (§ 5.10).
    # equality: the same inputs, specs, and optional inputs, in any order (unchanged)
```

Equality is unchanged: the same inputs, specs, and optional inputs, in any order, because binding reads inputs by name.

**`OutputSpec`.**

```python
class OutputSpec:
    """The term spec of what a term produces, and the name it is offered under when it is a base value."""

    @overload
    def __init__(self, term_spec: TermSpec, /) -> None: ...
    # The positional form, over a spec with parts (§ 2.3): a non-empty RecordSpec, a DistributionSpec,
    # a ConditionalDistributionSpec, a FunctionSpec, or a BatchSpec, of base elements included. TypeError
    # for a base spec, which needs a name; ValueError for an empty RecordSpec or a child name that is not
    # a name.
    @overload
    def __init__(self, **named_output: TermSpec | tuple | None) -> None: ...
    # The keyword form: one keyword, the name of the output at the root, over a base spec, a size
    # tuple, or None for a type hole. A spec with parts raises TypeError naming the spec's kind:
    # "params would name a record; give
    # OutputSpec(RecordSpec(...)) to offer its fields, or wrap it: OutputSpec(RecordSpec(params=...))",
    # and for a law "G would name a law; give OutputSpec(law_spec) to offer it positionally, or wrap it".

    @property
    def term_spec(self) -> TermSpec | None: ...
    # The produced term's type; None only while a type hole is pending.
    @property
    def outputs(self) -> Mapping[str, TermSpec | None]: ...
    # Each output's name and spec, in order (§ 2.11). The children when the term spec is a RecordSpec;
    # the one name at the root in the keyword form; for a BatchSpec, the columns of a batch of records
    # or of functions, the one output at the root of a batch of base values, and none for a batch of laws
    # or kernels; the output functions when it is a FunctionSpec; none for the spec of a law or a kernel.
    def __getitem__(self, paths: str | Sequence[str], /) -> OutputSpec: ...
    # The output spec at paths, by § 4.4's four rules. The path reader's errors; KeyError for a str that
    # is not a path, the name of the output at the root included.
    def __contains__(self, path: object, /) -> bool: ...
    # Whether path is a str and self[path] succeeds.
    def __iter__(self) -> NoReturn: ...
    # TypeError: "an output spec is indexed by path and is not iterable; list its outputs with
    # .outputs" (§ 5.11).
    def without(self, *paths: str) -> OutputSpec: ...
    # The other nodes, each at its path (§ 4.5). KeyError for a path that is not a path of self, every
    # path of a keyword form over a base spec included; ValueError when nothing would be left.
    def __and__(self, other_output_spec: OutputSpec, /) -> OutputSpec: ...
    # The outputs of both, side by side, self's first (§ 2.18). TypeError for an operand that is not an
    # OutputSpec, a positional form over a law, kernel, function, or batch spec, or a type hole;
    # ValueError for a name in both, or a dimension bound to two sizes.
    def with_term_spec(self, term_spec: TermSpec, /) -> OutputSpec: ...
    # Fill a type hole, or complete the stated term spec with the produced one, as today's with_spec (II.2).
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> OutputSpec: ...
    # Today's with_path_names (step S3): rename or move nodes by their paths, or rename the output at the root.
    @classmethod
    def default(cls, term_spec: TermSpec, *, output_name: str) -> OutputSpec: ...
    # The output spec a function uses for its return value when given none: the keyword form under
    # output_name for a base spec; for a batch whose elements carry only the default name element, the
    # batch whose elements are named output_name (O2); the positional form for every other
    # spec with parts. The distribution families' events use it too (§ 4.12).
    # unchanged: is_concrete, with_dim_sizes, with_dim_names; free dimensions are term_spec.free_dims
    # equality: the same term spec and, for an output at the root, the same name; order-sensitive;
    # an OutputSpec equals only another OutputSpec, and the hash agrees (§ 4.14)
```

| Today | Proposed | Step |
| --- | --- | --- |
| `decl.spec` | `O.term_spec` | S2a |
| `decl.components` | `O.outputs`, with § 2.11's meaning: a produced law or kernel offers none, and a batch's outputs are typed by its columns | S2b, S4; the rename |
| `decl.exposes_record` | `isinstance(O.term_spec, RecordSpec)` | S2a |
| `decl._component_name` outside `_specs.py`, and `_whole_term_component(decl)` | the private helper `_root_output_name` of § 5.11, which returns the name of an output at the root or `None`: `next(iter(O.outputs))` when `O.term_spec` is a base spec | S2a; the rename |
| `decl.with_spec(spec)` | `O.with_term_spec(spec)` | S2a |
| `_view_declaration(decl, s)` | `O[s]` | S2b |
| `_components_record(decl)` | `RecordSpec(O.outputs)` where a record of the outputs is needed | S2a; the rename |
| `decl.with_path_names(...)` | `O.rename(...)` | S3 |
| #571's `OutputSpec.merge` (never built) | `O & P` | S2b (R1) |
| `OutputSpec.default(spec, component=...)` | `OutputSpec.default(spec, output_name=...)` | the rename |

**Terms.**

```python
class Record:
    def __and__(self, other_record: Record, /) -> Record: ...
    # The fields of both, self's first. TypeError for an operand that is not a record; ValueError for
    # a top-level name in both, groups included. Replaces merge (made shallow in S2b, renamed in S3).
# RecordSpec.__and__ and NamedTree.__and__: the same rule.

class Distribution:
    def __getitem__(self, paths: str | Sequence[str]) -> Distribution: ...
    # The view at paths (§ 4.2), whose event_spec is self.event_spec[paths]. KeyError for the name of the
    # output at the root, which output reads.
    def output(self, output_name: str, /) -> Distribution: ...
    # self[output_name], or self for the output at the root (G3). KeyError for a name that is not an output.
    def without(self, *paths: str) -> Distribution: ...      # step G4
    def __and__(self, other: Distribution | ConditionalDistribution, /) -> Distribution | ConditionalDistribution: ...
    # The merge of § 2.15, read over both operands' factors (§ 4.16): both sets of outputs side by side,
    # other's outputs feeding the inputs of this law's factors that no factor feeds. Raises as § 2.17
    # states; returns NotImplemented for an operand that is neither a law nor a kernel. Replaces __mul__ (S3).
# ConditionalDistribution: the same members; __getitem__ and output in step G5.

class Function:
    def __init__(self, fn, /, *, input_spec=None, output_spec: OutputSpec | TermSpec | None = None, **rest) -> None: ...
    # A bare TermSpec as output_spec reads as OutputSpec.default(term_spec, output_name=<the callable's
    # output name>) (§ 4.13). A parameter with a default is an
    # optional input, and input_spec marking optional a parameter without one raises ValueError (O9).
    def __getitem__(self, paths: str | Sequence[str]) -> Function: ...   # w -> self(w)[paths] (G3)
    def output(self, output_name: str, /) -> Function: ...                # G3
    def __and__(self, other_function: Function, /) -> Function: ...
    # § 2.20 and § 5.7 (after the workshop). TypeError for an operand with no input spec, no output spec,
    # or a type hole, naming it.
```

`factor(d, output_name)` takes the name that its parameter `component_name` takes today, and each distribution family takes the name of its output at the root as `output_name`, its first parameter, as `Normal("x", 0.0, 1.0)` does.
`conditional_distribution(fn, given_spec=...)`, `ConditionalDistributionSpec(given_spec, ...)`, and `Function(fn, input_spec=...)` read their argument through the `InputSpec` constructor (§ 5.10).

**Batches.**

```python
class BatchSpec(TermSpec):
    def __init__(self, element: TermSpec | OutputSpec, levels=None, /, **level_shapes) -> None: ...
    # element is the elements' term spec, or an OutputSpec that names base elements (keyword form) or
    # gives any other element positionally (O8). A bare base spec gets the name "element".
    @property
    def element_spec(self) -> OutputSpec: ...
    # The output spec of one element as a term (§ 2.9): for a base element, the keyword form under its
    # name; otherwise the positional form over the element's term spec (step S4). Its term_spec is
    # what today's element_spec returns.

class Batch:
    @property
    def element_spec(self) -> OutputSpec: ...          # a view on spec; its outputs replace II.5's Batch.components
    def __and__(self, other_batch: Batch, /) -> Batch: ...
    # The merge of the elements, position by position, each read as one operand (§ 2.16). ValueError
    # unless both have equal levels; TypeError for elements of different kinds. For a RecordBatch it is
    # today's merge, with the shallow rule (S2b), under its new name (S3).
```

A batch's own output spec is `OutputSpec(bt.spec)`, derived, whose outputs follow § 2.9: its columns for a batch of records or of functions, one output at the root for a batch of base values, and none for a batch of laws or kernels; a `DistributionBatch`'s and a `ConditionalDistributionBatch`'s `event_spec` is the event spec of their elements, `element_spec.term_spec.event_spec` (III.10).

### 6.2 Worked examples across kinds

Each example states what users write and what it returns once every step has merged, in the final words (§ 6.4); "Today" lines show `6f8859fc`.
Sections 2, 4, and 5 already show `grouped`, `hier`, the records, and the batches; this section gives the running examples that span several rules.

#### 6.2.1 `prior`, `lik`, `model`, and `post`

```python
prior.event_spec                     # OutputSpec(RecordSpec(mu=..., tau=...))
prior.event_spec.outputs             # {mu: NumericArraySpec(()), tau: NumericArraySpec(())}
prior.event_spec["mu"]               # OutputSpec(mu=NumericArraySpec(()))
InputSpec(prior.event_spec)          # InputSpec(mu=..., tau=...): the inputs prior's outputs could feed

model = lik & prior                  # Today: lik * prior
# fed = {mu}; outputs y, mu, tau; no unfed input left, so model is a law
model.event_spec == lik.event_spec & prior.event_spec        # I16: OutputSpec(RecordSpec(y=..., mu=..., tau=...))
prior & lik                          # ValueError (M4): lik takes mu, which prior outputs; put prior on the right

post = condition_on(model, {"y": y_observed})
post.event_spec                      # OutputSpec(RecordSpec(mu=..., tau=...)), on every route (I11, step G4)
post["mu"].event_spec == post.event_spec["mu"]               # I6: OutputSpec(mu=NumericArraySpec(()))
condition_on(lik, mean(post[["mu"]]))                        # a plug-in estimate: the record {mu} binds lik's input
hier["s1"] & hier["s2"]              # ValueError (M3): both output theta     (Today: they merge as whole terms s1, s2)
hier["s1", "s2"]                     # the joint law of both groups, which keeps their dependence
```

#### 6.2.2 `grouped`, and a kernel that takes a group

```python
grouped["g"].event_spec == grouped.event_spec["g"]           # OutputSpec(RecordSpec(c=..., e=...))   (G1a)
lik_g & grouped[["g"]]               # grouped[["g"]]'s output g feeds lik_g's input g
lik_g & grouped["g"]                 # nothing is fed: grouped["g"] outputs c and e, so lik_g keeps its input g
joint = Normal("x", 0.0, 1.0) & grouped
# a law over {x, a, g: {c, e}}; every path of grouped is unchanged (I17), and x moves from the root to the path x
joint["g/c"]                         # the same node as grouped["g/c"]
```

#### 6.2.3 Optional inputs across merges

```python
lik_opt.given_spec                   # InputSpec(mu=..., s=...).with_optional("s")
opt_model = lik_opt & prior          # a law: mu is fed, and s stays unfed at its default 1.0
opt_model & s_prior                  # s is fed: y's spread is near 100, as today (§ 4.16, item 1)
lik_opt & (prior & s_prior)          # the same law (I19)
condition_on(lik_opt, {"mu": 0.3})   # a law, with s at its default: binding keeps today's rule (§ 2.6)
```

#### 6.2.4 `growth_prior`, `shocks_given_sigma`, and the likelihood

A likelihood conditions on `K`, `r`, `phi`, and the shocks `e`, which `shocks_given_sigma` produces given `sigma`.
`likelihood_fn(K, r, phi, e)` returns the law of the data `y`.

```python
# Today (§ 1.2, and the workshop tutorials 3 and 4)
shock_spec = RecordSpec(shocks_given_sigma.event_spec.components)
likelihood_given_spec = growth_prior.event_spec.spec.without("sigma").merge(shock_spec)

# Proposed
likelihood = conditional_distribution(
    likelihood_fn,
    given_spec=growth_prior.event_spec.without("sigma") & shocks_given_sigma.event_spec,
    label="likelihood",
)
# the output at the root e becomes the child e; given_spec reads the four outputs as inputs K, r, phi, e
list(likelihood.given_spec)          # ["K", "r", "phi", "e"]

model = likelihood & shocks_given_sigma & growth_prior
# likelihood & shocks_given_sigma: e is fed; unfed inputs K, r, phi, sigma; outputs y, e
# ... & growth_prior: K, r, phi, and sigma are fed; no unfed input left, so model is a law
list(model.event_spec.outputs)       # ["y", "e", "K", "r", "phi", "sigma"]
shocks_given_sigma & likelihood      # ValueError (M4): likelihood takes e, which shocks_given_sigma outputs
```

The tutorial idioms map one to one; the tutorials keep theirs until their rewrite (README directive 6).
In the table, `kept = growth_prior.event_spec[["K", "r", "phi"]]`, an output spec.

| Today | Proposed |
| --- | --- |
| `given_spec=prior.event_spec.components` | `given_spec=prior.event_spec` |
| `{"sigma": growth_prior.event_spec.components["sigma"]}` | `growth_prior.event_spec["sigma"]` |
| `{name: growth_prior.event_spec.components[name] for name in ["K", "r", "phi"]}` | `kept`, that is `growth_prior.event_spec[["K", "r", "phi"]]` |
| `{**inputs, "e": shocks_given_sigma.event_spec.components["e"]}` | `kept & shocks_given_sigma.event_spec`, a merge of two output specs, which raises where `kept` already offers `e`, where the dict merge replaces the entry |
| `growth_prior.event_spec.spec.without("sigma")` | `growth_prior.event_spec.without("sigma")` |
| `list(model.event_spec.components)` | `list(model.event_spec.outputs)` |
| `posterior[tuple(prior.event_spec.components)]` | `posterior[tuple(prior.event_spec.outputs)]`, since a tuple of `str` is a list of paths |

#### 6.2.5 `emulator` and `fit_emulator`

```python
emulator["mean"]                     # the function theta -> emulator(theta)["mean"]   (step G3)
emulator["mean"](theta) == emulator(theta)["mean"]           # I13
emulator.output("mean") == emulator["mean"]                  # I12

fit_emulator = Function(train_gp, output_spec=OutputSpec(FunctionSpec(
    InputSpec(theta=theta_spec), OutputSpec(RecordSpec(mean=("n_days",), var=("n_days",))))))
list(fit_emulator.output_spec.outputs)       # ["mean", "var"]: the returned function's outputs (step G1b)
fit_emulator.output_spec["mean"]
# OutputSpec(FunctionSpec(InputSpec(theta=...), OutputSpec(mean=...))): the spec of the function emulator["mean"]
fit_emulator & center                # TypeError (M2): fit_emulator produces a function; wrap it to merge it
```

A function that returns a function offers outputs, while one that returns a law does not: a function into records is the record of its output functions, and a law over records is not the record of its marginals (§ 2.11).

#### 6.2.6 A function that returns a law, and lifting it

```python
fit = Function(run_inference, output_spec=post.spec)     # a bare law spec reads positionally (step G1b)
fit.output_spec.outputs              # {}: a law, with no outputs
fit["mu"]                            # the function y -> run_inference(y)["mu"]   (step G3)
fit.output_spec["mu"] == fit["mu"].output_spec               # OutputSpec(DistributionSpec(OutputSpec(mu=...)))

def run_inference_wrapped(y_observed):
    return {"posterior": run_inference(y_observed)}           # the body wraps the law under a name
fit_wrapped = Function(run_inference_wrapped, output_spec=OutputSpec(RecordSpec(posterior=post.spec)))
fit_wrapped.output_spec["posterior"] # OutputSpec(post.spec), as fit_wrapped(y)["posterior"] is the law
fit & center                         # TypeError (M2): fit produces a law; merge fit_wrapped instead

def law_of_y(x):
    return Normal("y", x, 1.0)
Function(law_of_y)(Normal("x", 0.0, 1.0)).event_spec
# OutputSpec(RecordSpec(law_of_y=DistributionSpec(...))): draws {law_of_y: law}   (step G1b; today: OutputSpec(law_of_y=...))
Function(law_of_y).using(include_inputs=True)(x=Normal("x", 0.0, 1.0)).event_spec
# OutputSpec(RecordSpec(x=..., law_of_y=DistributionSpec(...))): draws {x, law_of_y: law}, as today
```

Today `Function(run_inference, output_spec=post.spec).output_spec` is `OutputSpec(run_inference=DistributionSpec(...))`, and `OutputSpec(post.spec).components` is `{mu: NumericArraySpec(()), tau: NumericArraySpec(())}`, the specs of a draw's fields rather than of anything the function returns.

### 6.3 Deliberate exceptions

These cases fall outside the invariants on purpose.
Because paths go through every kind, indexing a law, a function, or a batch held in a field is not one of them; what such terms lack is summaries and conditioning.

| Case | What happens | Why |
| --- | --- | --- |
| Covariance | `cov(d)` is one base value, so `cov(d)[s]` raises | It is not one value per field; the block for `s` is `cov(d[s])` |
| Monte Carlo summaries | `mean(d)[s]` and `mean(d[s])` estimate one quantity but are not equal | Each call takes its own draws |
| Summaries and conditioning at a field that holds a law or a function | `mean(rm)` is undefined for a random measure's drawn laws, and `condition_on` cannot fix a drawn law or function | A mean of laws is not one value per field; I1, I3, and I8 hold for paths that end at values |
| `&` on base values | `sample(Normal("x", 0.0, 1.0) & Normal("y", 0.0, 1.0))` is a record `{x, y}`, but `sample(Normal("x", 0.0, 1.0)) & sample(Normal("y", 0.0, 1.0))` raises `TypeError`, since NumPy's `&` refuses floats, while two integer or boolean arrays combine bitwise; `(x & y)[j]` on two records is a field, while `x[j] & y[j]` on two arrays is a bitwise element | A base value carries no name, so I20 needs record outputs; O7 keeps NumPy's `&` |
| `&` on records and on arrays | `&` on two records, `NumericRecord`s included, merges their fields by top-level name, while `&` on two `NumericArray`s is elementwise | A record is a tree of named values (O7) |
| A batch of arrays through raw JAX | `jax.jit(lambda b: b)(draws)` comes back offering `element` | Base-element names stay out of the pytree (§ 5.8) |
| Labels | `sample(grouped["a", "g/c"])` and `sample(grouped)["a", "g/c"]` differ in label | No operation reads a label |
| Python's mapping rule | `"g" in draw` is true while `"g"` is not in `draw.keys()`; `"g/c" in lik_g.given_spec` is true while `"g/c"` is not in its `keys()` | `keys()` lists fields or inputs, and `in` agrees with `[]` (§ 4.6) |
| The list form at the root | `Normal("x", 0.0, 1.0)[["x"]]` succeeds while `sample(Normal("x", 0.0, 1.0))` is an array that NumPy indexes, and `OutputSpec(x=...).without("x")` raises | § 4.4's rule 3 is an extra rule at the top level; I1 and I10 cover paths |
| A shared optional input that a binding omits | each factor takes its own default, so one input holds two values, and the law at the defaults is not the kernel at any single value of that input | Defaults belong to the factors (§ 2.19) |
| A kernel whose inputs are all optional | `k_opt & Normal("z", 0.0, 1.0)` is a law, sampled at `s = 1.0`, while `sample(k_opt)` raises | IV.2's table decides the kind by the required inputs left (§ 2.19) |
| Two views of one parent | `hier[["s1"]] & hier[["s2"]]` is the product of the marginals | A merge reads names and factors, and a view is one factor (§ 2.21) |
| A view and a marginal of one joint (I7) | `M[["y", "mu"]] & s_prior` and `marginal(M, ["y", "mu"]) & s_prior` differ when `M` holds a factor with an unfed optional input `s` | Merge reads factors, and a view is one factor (§ 4.16, item 4) |
| A hidden unfed optional input, fed or shared later | `(lik_opt & prior) & s_prior` feeds `s`, `M & k2` shares it, and `M & kw` makes it a bindable input of the kernel result though `kw` does not mention it; equal laws merge differently | Today's flattening (§ 4.16, items 1 to 3) |
| Spec merges and term merges (I16) | `s_prior.spec & (lik_opt & prior).spec` is valid while `s_prior & (lik_opt & prior)` raises | A spec states no unfed input of its factors (§ 4.16, item 5) |
| Associativity of validity (I19) | in property 1's two cases, one grouping raises and the other merges | M7 and II.1's rule check different pairs in the two groupings |
| Batches of laws or kernels (I19) | for `A`, `B`, and `C` batches of `lik_opt`, `prior`, and `s_prior` over equal levels, `(A & B) & C` leaves `s` at its default, so `y`'s spread is near 1, while `A & (B & C)` feeds `s`, near 100; both are valid | A batch merge reads each element as one factor, with the input spec its term spec states, so the elements' factors and stored scopes are not read (§ 2.16, § 4.16, item 6) |
| Associativity of input specs and functions (I19) | `(J1 & J2) & J3` and `J1 & (J2 & J3)` are both valid and differ in `w`'s size | They store no dimension scope (§ 4.17, item 1) |
| A stored scope | `m1 & m2` raises over a dimension name that neither event spec shows, and `with_dim_names` cannot rename it away | Today's stored scope (§ 4.17, item 2); § 6.20, Q6 |

What holds before a step merges is not a deliberate exception; § 8 states it.

### 6.4 Vocabulary

One vocabulary runs through this plan and `design/`, and through the docstrings and messages of the code.
The final words are this plan's.
The column "When" says which step changes each row in `design/` and in the code.
A row marked "the rename" waits for the one rename PR after the workshop (§ 6.15); rows T25 and T26 give the words used until then.

| Today | Proposed | When | Why |
| --- | --- | --- | --- |
| "input slot", "slot", "given slot", "optional slot" | "input", "a kernel's input", "optional input" | S0, to "slot" (T25); the rename | Plain, and parallel with "output" |
| "component", "produced slot", "output slot" (III.7, IV.2, VI.9) | "output"; "component" keeps only a mixture's component laws (VII.3) | S0, to "component" (T25); the rename | Plain and parallel; settles O14 |
| "map-like kind" (II.2), "producer", "consumer" | not terms: "a term's inputs", "a term's produced term" | S0 in `design/`, S2a in docstrings | Every term produces a term (§ 2.5), so "producer" names nothing the plain words do not |
| "output declaration", "input declaration", "event declaration" | "output spec", "input spec", "event spec" | the rename | The class names; "declaration" then keeps only its broad sense of a static claim (II.7) |
| "whole declaration", "exposed declaration", "exposes", "packaging", "packaged" | the keyword form, the positional form; "offers its outputs" | S0 in `design/`, S2a in docstrings and messages (T25) | The term spec decides the form; "packaging" also suggests Python packaging |
| "address" | "the output at the root", or "a top-level path" | S0 (T25); the rename | Two places, two plain phrases |
| "component spec" | "the spec of an output" | S0 (T25); the rename | A plain phrase |
| "meets", "is matched", "wire", "bound", "connections" | "feeds" | S0 in `design/`, S3 in the code of merges | One word for one relation (§ 2.13) |
| IV.2's "unmet" | "unfed" (§ 2.15) | S0 in `design/`, S3 in the code of merges | One word for one concept |
| "unify" in four senses | "conforms", "unify", and "completes", one per relation (§ 6.5.4) | S0; the code's `_unify_specs` keeps its name | Directive 11 |
| "packaged sub-joint" (IV.1), "packaged joint" (IV.2) | "grouped sub-joint" | S0 in `design/`, G1a in the code | A rename that gathers outputs under one node creates it. IV.2's paragraph "Packing the joint" becomes "The joint's event" |
| "interior node", "subtree", "field group", "group of fields" | "group" | S0 in `design/`, S1 in the code | One word for a node of a record that is itself a record |
| "key" for a field path | "path", and "a path or a list of paths" for what `[]` takes | S0 in `design/`, S1 in the code | "Key" means a random key, and a mapping's keys |
| "selection" for batch positions and for several paths | "positions" and "a list of paths" | S0 in `design/`, S1 in the code | `select` is a method with another meaning |
| `FieldView` | `MarginalView` | S3 | It covers groups and lists of paths, and pairs with `marginal` |
| `with_path_names` | `rename` | S3 | The name pandas and xarray use |
| `*` for merging laws and kernels | `&` | S3 | `*` is arithmetic on values |
| "composition" for what `&` or `*` does to laws, kernels, and records | "merge" | S0 in `design/`, S3 in the code | "Composition" means `f ∘ g` and `A B`; `&` matches names |
| `OutputSpec.spec`, `with_spec` | `term_spec`, `with_term_spec` | S2a | The glossary's term for the produced term's type |
| `OutputSpec.components`, `decl[c]` | `O.outputs`, `O.outputs[c]`; `O[s]` takes paths | the rename for `outputs`; S2b for `O[s]` | § 5.11 |
| the decided `t.component(c)` | `t.output(c)` | the rename, or G3 if the rename merges first (§ 6.15) | § 4.11 |
| `BatchSpec.element_spec` (a term spec), O15's `element_declaration`, II.5's `Batch.components` | `BatchSpec.element_spec`, an output spec, whose `term_spec` is the old meaning; `element_spec.outputs` | S4; the rename for `outputs` (T26) | § 5.8, § 6.21 |
| the families' keyword `component=`, `factor(d, component_name)`, `OutputSpec.default(component=)` | `output_name` | the rename | § 6.20, Q1 |
| `marginal(d, field)` | `marginal(d, paths)` | S1 | It takes a path or a list of paths |
| the private names `_exposed_components`, `_component_name`, `_components_of`, `_callable_component`, `_root_component_name` (§ 5.11), and `_ByComponent` (`operations/_moments.py`), and the page `docs/api/declarations.md` | `_exposed_outputs`, `_output_name`, `_outputs_of`, `_callable_output_name`, `_root_output_name`, and `_ByOutput`, and the page `docs/api/input_and_output_specs.md` | the rename | README directive 5, rule 5 |

#### 6.4.1 Who changes the glossary

Several steps edit `design/glossary.md`, each for its own terms.
This plan owns two of them:
- step S0 owns the path terms and the terms of this plan's rules listed below, in today's words;
- the rename PR after the workshop replaces *slot*, *component*, and the three kinds of declaration with *input*, *output*, *input spec*, *output spec*, and *event spec*.

The execution plan owns the others, and `execution-layer.md` § 13 lists them: step P0b adds "call context" in #587; step E3a adds "evaluation context", "static setting", and "inert setting"; step E1 adds the other execution terms, such as evaluation point, requester, executor, call token, and JAX program with embedded calls; and step E0 changes the entries whose words it renames.

**The glossary changes of step S0:**
- **new entries:** base spec, spec with parts, node, group, column, produced term, feeds, merge, conforms, unify (II.1 and IV.2, as § 2.14 defines them), and *unfed*, said of an input of a joint's factors that no factor feeds (§ 2.15), replacing IV.2's "unmet"; *slot* is added and *component* redefined with this plan's meanings of input and output;
- **redefined:** *field* and *path* take § 2.4's meanings; *name* lists the names of slots, components, fields, and levels; *marginal* takes "a path or a list of paths"; *view* names `MarginalView`; *type hole* names `with_term_spec`; *signature* says "a law's components, a kernel's components with `|` and its slots, or a function's parameters";
- **narrowed:** *key* keeps only the random key and a mapping's keys;
- **removed:** *packaging*; *key* in its sense of a field's path; *selection*, apart from *selection order* (II.7), which stays (gaps T3);
- **not added**, although the original plan proposed them: producer, consumer, exposes, whole declaration, exposed declaration, address, component spec, leaf path, wrap (which stays a plain verb), position, and list of paths (plain phrases, as decided 2026-10-18). "Merge category" and "interim whole form" are words of this plan's rules, not glossary terms;
- **kept until the rename:** *declaration*, with its sentences on `InputSpec` and `OutputSpec`, and "event declaration" in *conversion* and *event* (§ 8.2);
- **kept:** the occurrence path (V.8).

**The glossary changes of the rename PR:**
- *slot* becomes *input*, and *component* becomes *output*; *component* keeps only a mixture's component laws (VII.3);
- *declaration* keeps only its broad sense of a static claim (II.7), and its sentences on `InputSpec` and `OutputSpec` point to the new entries *input spec* and *output spec*;
- *conversion* and *event* say "event spec" for "event declaration";
- *name* and *signature* say "inputs" and "outputs".

The canonical-names table changes in S0:
- the row "a field key within a tree, or the name assigned to a field: `field_name` or `key`" becomes "a path within a tree: `path`, and `paths` for several" and "the name assigned to a field: `field_name`";
- the row for the name of a family's output at the root reads `component`;
- the row for `element_spec` says it is the output declaration of one element of a batch, whose `term_spec` is the element's type;
- the row that says "event declaration" is unchanged.

The rename PR then changes three rows: "a distribution's event spec, an `OutputSpec`: `event_spec`" replaces the row that says "event declaration"; the family's row reads `output_name`; and the `element_spec` row says "output spec".

The execution plan's "automatic selection" points to *selection order* (gaps T3).

**Two words keep a second sense.**
- *Component* names the component laws of a `MixtureDistribution`, which its constructor takes as `components=`. A mixture is a law, so `mix.output("x")` reads its event's output `x`, not a component law (O14).
- *Element* names an item of a batch, and the default output name `element` of a raw batch (§ 5.8). The attribute `element_spec` describes one element, so the senses agree (O15, as replaced by § 6.21).

### 6.5 Changes to `design/`, `STYLE_GUIDE.md`, and `docs/`

#### 6.5.1 `design/`, section by section

II.6 becomes the one statement of paths, lists of paths, and how `[]` reads them, and II.2 the one statement of input specs, output specs, and the grammar rule.
Every other section cites them instead of restating a rule (`STYLE_GUIDE.md` §10, rule 8).
Step S0 writes the target text of every row in today's words, "slot", "component", and the three kinds of declaration, with this plan's meanings (§ 8.2); § 6.9 says how the signatures that change before the code does are recorded.

| Section | Today | Under this plan |
| --- | --- | --- |
| II.1 Term specs | The size shorthand is a `RecordSpec` field's alone; "binding by unification" | States the tuple shorthand once, for every place an input or the keyword form takes a term spec; "conforms" for the directional check (§ 6.5.4) |
| II.2 Declarations | The keyword form names any spec whole; `spec`, `components`, `exposes_record`, `with_spec`; "packaging"; "map-like kind"; a whole term's paths start with its name; "an exposed record's components are its fields"; `OutputSpec(posterior=DistributionSpec(...))` as the example of a function returning a law; a law's positional outputs typed by one draw's value | The definitions of § 2: the produced term, inputs and input specs, outputs and output specs with their two forms, the interim whole form until G2 (§ 8.1), and type holes; the grammar rule; `OutputSpec` indexed by path, with `term_spec`, `with_term_spec`, the attribute of outputs (`components` until the rename, T26), `[]`, `in`, `without`, `rename`, and `&`; the rule of § 2.11, with no output offered by a produced law or kernel and no law's event rule; the four sources of inputs; `InputSpec[]` with a list of names, `&`, `without`, and `rename`; one `OutputSpec.default` and type holes; a returned law is positional, with no outputs, or wrapped; "completes" for II.2's completion (§ 6.5.4) |
| II.4 Labels and views | A field of a value takes its path as its label; batch positions are a "selection" | Adds the label of a value indexed by several paths, `(a, g/c)`; positions are "positions" |
| II.5 `Batch` | A tuple of names is one path; positions are a "selection"; `()` is the whole batch; a raw batch keeps a bare element spec; for a whole element, a name returns the batch; `Batch.components` | A batch is indexed by its elements' paths and always holds an element spec, `BatchSpec.element_spec`, an output spec whose `term_spec` is the element's type; a batch's outputs are its columns; raw defaults; `BatchSpec` compares names, while a batch's equality, hashing, fingerprint, and pytree use the name-free canonical spec (§ 5.8); `()` and lists of positions raise; `in` tests paths; `t.component(c)` reads the output at the root; no `Batch.components` |
| II.6 `NamedTree` | A path may be a tuple; `[]` raises on a group; `in` tests fields; a key is a field's path; `without(path)` takes one path | Owns the definitions of § 2.4, how `[]` reads paths, and `in`; `[]` returns groups; paths continue into a held law, function, or batch; a list of paths keeps its paths, in the tree's order; `without(*paths)` as the code has it; `rename`; `&` is shallow |
| III.3 `Function` | Output names default to the callable's name for any non-record; a bare term spec uses the same default; no `[]` | `OutputSpec.default` by kind (§ 4.12) and the bare-spec reading (§ 4.13, gaps T4); `input_spec` takes the four sources, with O9's rule for optional inputs; `f[s]` indexes what `f` returns |
| III.5 `Record` | `select` names entries by full path; `merge` is deep | `[]` names groups and held terms; `select` flattens to last names; `&`, shallow, by top-level name |
| III.6 `RecordBatch` | A tuple of names is one path | Cites II.6; a list of paths returns a `RecordBatch`; `select` flattens as on a record; `&` merges element by element |
| III.7 `Distribution` | `d[c]` returns `d` for a whole term; one `OutputSpec.default` serves a whole-term event; "produced slots", "output slot"; a view of a group is a whole term; several paths flatten to last names; `OutputSpec(parameters=RecordSpec(...))` has "the output slot `parameters` and the event path `parameters/beta`"; the leaf check "unifies" | A view's event spec is `event_spec[s]`; the name of the output at the root is read by `component`, not `[]`, from G3 (T18); a view of a group offers its parts; several paths keep their paths; a view at a field that holds a law, a kernel, a function, or a batch is positional, with no wrap; `MarginalView`; the leaf check "conforms" |
| III.8 View capabilities | Derivations use component-rooted paths | Derivations use the term's paths |
| III.9 Kernels | No `[]`; `given_spec` sources stated locally | Points to II.2 for the sources of `given_spec`; states the `[]` exception now and `lik[s]` as the target |
| III.10 Batches of laws | Silent on `[]` | `laws[s]` is the batch of views; `event_spec` is `element_spec.term_spec.event_spec` |
| IV.1 Joints | A "packaged sub-joint" declares a group as one whole term | A grouped sub-joint wraps its group, so its draws carry the group's level, and stays one factor; the joint's event uses `&` |
| IV.2 Composition, retitled "Merging distributions" | `*`; "produced slots"; "Packing the joint"; "unmet"; associativity "rests on" `G_B ∩ F_A = ∅`; "matched specs that do not unify" | `&`, with the one definition of § 2.15: the conditions M1–M8, the result, and the value; flattening kept as today's behaviour, with § 4.16's consequences; conformance for a fed input and unification for a shared one; "unfed" for "unmet"; "The joint's event"; the rationale of § 2.21 (gaps T5); the operator-coexistence note says `*` scales and `&` merges; "merge" replaces "composition" for `&` throughout |
| IV.4 Laws from functions | Sources of `given_spec` stated locally; "unifies" for the abstract draw and an explicit event spec | Points to II.2; "conforms" (§ 6.5.4) |
| V.5, V.6 Lifting and planning | Atoms and outputs named whole under a component; "The declared output" reads any non-record bare spec as whole; planning "unifies" | Defaults by kind: a lifted law of laws wraps its draws under the callable's output name, for the reason of § 4.12 (C5), and a sweep's batch is positional through its element spec; the bare-spec reading of § 4.13; "conforms" |
| VI.0 The operation model | "Selection commutes with a coordinatewise summary"; the slot "unifies" | "Indexing commutes with a coordinatewise summary"; "conforms" |
| VI.6 `condition_on` | Given values are named by "keys of the event schema"; the result's output spec depends on the route; "unmet givens" | Given values are named by paths, nested ones included; the result's event spec is `d.event_spec.without(*fixed)` (I11); "unfed" |
| VI.8 `marginal` | `marginal(d, field)`; one path or several, decided by count; a group's marginal is the factor that holds it, a whole term; fields "repackaged in the order of the paths" | `marginal(d, paths)`: a path or a list of paths, decided by form; a group's marginal offers its parts, as the factor's marginal at the group; the tree's order |
| VI.9 | "unmet givens" | "unfed" |
| VI.11 Batch lifting | The operands of `*` are exempt from batch lifting | Two batches merge element by element, over equal levels; aligning other levels by name stays after the workshop |
| VII.2, VII.5, VII.6 Families | Empirical laws whose atoms are laws or functions, bootstrap laws, random functions, and random measures name their event whole; VII.2's declared type "unifies" | Their draws are wrapped in a record under the name their constructor takes (C5); "completes" |
| Glossary | As § 6.4.1 lists | As § 6.4.1 lists |

Each use of "packaging", "packaged", "produced slot", "output slot", "map-like kind", and "input slot" in parts II–VII changes, about 40 lines at `a703f15e` (excluding Python packages in `package-structure.md`), including the II.2 row of Part II's summary table, plus the 29 lines of § 6.5.4.

#### 6.5.2 `STYLE_GUIDE.md`

| Section | Today | Under this plan |
| --- | --- | --- |
| §1.10 Record field iteration and path access | `"a/b"` is "a short form of the tuple form"; `[]` returns leaves only, so `at_path` is the way to a group; `with_path_names` | A path is a `str`; a tuple is a list of paths; `[]` and `in` name groups while `keys()` stays leaf-keyed; `rename` (S1 for paths, S3 for `rename`) |
| §1.11 Distribution iteration | A `DistributionBatch` is indexed by position only | A batch of laws is also indexed by path (G5) |

`STYLE_GUIDE.md` describes the code as it stands (AGENTS.md), so each row changes with the step that changes the code, not with S0 (R4).

#### 6.5.3 `docs/`

- `docs/api/declarations.md` opens with the definitions of an input, an output, and the two forms of an output spec, in the words of § 8.2, so "slot", "component", "input declaration", and "output declaration" (step S2b); the rename PR after the workshop renames the page `docs/api/input_and_output_specs.md`.
- **One idiom to teach.** After S2b five spellings declare the same inputs: `prior.event_spec`, `prior.event_spec.components`, `prior.event_spec.term_spec`, `InputSpec(prior.event_spec)`, and a dict comprehension. `docs/api/declarations.md` and the docstrings of `conditional_distribution` and `Function` show one idiom, `given_spec=law.event_spec` cut down with `[...]`, `without`, and `&`, and list the other sources only as accepted (step S2b). `at_path(p)` stays a synonym of `[p]` (§ 5.4), and its docstring says that `[]` is the recommended form (step S1).
- The User Guide notebooks are updated mechanically by each step that breaks them (README directive 6). User Guide 04 and 08 read `event_spec.components[...]` today; five User Guide notebooks read `.components` and four pass `component=`. S2b moves the `given_spec` idioms to the one idiom above (§ 6.11), and the rename PR updates what remains.
- `docs/tutorials/` is out of scope, so its dict-comprehension idioms wait for the tutorial rewrite (§ 6.2.4 maps them).

#### 6.5.4 One word per relation in `design/`

S0 gives each relation between specs one word, as § 2.14 states, and rewords these passages; the last but one row also replaces IV.2's "unmet", one concept that this plan calls "unfed":

| Sense | Passages in `design/` | S0 writes |
| --- | --- | --- |
| a directional check that a value or a spec is admitted, binding dimension names as it goes | II.1's binding (`02-shared-abstractions.md:23`, and the comments at `:39` and `:40`); III.7's leaf check (`03-term-kinds.md:461`); IV.4's abstract draw and explicit event spec (`04-distributions.md:162`, `:179`); V.6's planning (`05-functions.md:14`, `:42`, `:149`, `:204`); VI.0's slot (`06-operations.md:160`); VI.6's `given` (`:286`) | "conforms", and for several specs at once "conform in one dimension scope", II.1's words |
| the symmetric spec of values that both admit | IV.2's same-named givens (`04-distributions.md:86`, `:87`); III.2's opaque specs (`03-term-kinds.md:64`, the sentence that begins "Two opaque specs unify") | "unify", with § 2.14's `U` |
| both relations in one sentence | IV.2's merge errors (`04-distributions.md:83`, "matched specs that do not unify") | "an output that does not conform to the input it feeds, or shared inputs that do not unify", in today's words until the rename |
| completing a declared spec with a produced one, keeping the declared dtype and support | II.2's `with_spec` comment and completion (`02-shared-abstractions.md:87`, `:113`); III.2's clause on a declared `meta` (`03-term-kinds.md:64`); VII.2's declared type (`07-distribution-catalog.md:60`) | "completes", II.2's own heading, with "conforms" for its check |
| an input of a joint's factors that no factor feeds | IV.2's "unmet" (`04-distributions.md:18`, `:66`, `:70`, `:73`, `:76`, `:78`, `:81`, `:85`); VI.6's and VI.9's "unmet givens" (`06-operations.md:294`, `:368`) | "unfed", with § 2.15's `unfed(t)`; the variable `unmet` in IV.2's formulas becomes `unfed` |
| names of today's code | `package-structure.md:29`, `:34`, `:142` ("dimension unification", "unification (III.5)") | unchanged, since they describe `_unify_specs` and `_unify_array_shape` |

The line numbers are at `6f8859fc`.
The code's `_unify_specs` keeps its name until a step changes it (README directive 5, rule 5).

### 6.6 Breaking changes for the CHANGELOG

Every rename and removal is a clean break: no alias, no deprecation warning, and an entry under `### Changed (breaking)` (decided 2026-10-17).
A change is **silent** when the same call still runs and returns something else, so a user finds out late, if at all.
For each silent row, the PR body and the CHANGELOG entry show the same call before and after, as a line a user can copy; for example, for the first row and `site = Record({"habitat": {"forest": 0.7, "wetland": 0.3}, "forest": 2.0})`, `site["habitat", "forest"]` is the nested value `0.7` today and the record `{habitat: {forest, wetland}, forest}` after S1 (§ 4.1).
S1's `[]` docstring and `STYLE_GUIDE.md` §1.10 say "a tuple is a list of paths; a nested path is written with /".

| Change | Silent? | Step |
| --- | --- | --- |
| A tuple of `str` is no longer one nested path, on `[]`, `at_path`, `is_field`, `without`, and the keys of `with_path_names` (later `rename`) | yes, where each name of the tuple is also a top-level path | S1 |
| A law indexed by several paths keeps their paths instead of flattening to last names, in the tree's order | yes | S1 |
| A value picked by several paths is labeled by its paths, as `(a, g/c)` instead of `(a, c)` (§ 4.3) | yes | S1 |
| `marginal`'s parameter is renamed from `field` to `paths`, and a marginal of several factors lists them in the law's order | the order: yes | S1 |
| `draws[()]` raises instead of returning the whole batch | no | S1 |
| `without` refuses a path given twice and a path inside another | no | S1 |
| `select` flattens to last names, on records and on record batches: `draw.select("a", "g/c")` gives the key `c`, not `g/c` | yes | S1 |
| `in` changes on records, batches, and laws: `"g" in draw` is true, and `1.0 in draws` is false | yes | S1 |
| `OutputSpec.spec` is renamed `term_spec`, `with_spec` is renamed `with_term_spec`, and `exposes_record` is removed | no | S2a |
| `OutputSpec[]` takes a path or a list of paths, and `in` tests paths | no | S2b |
| `InputSpec.without` raises `KeyError` for a name that is not an input | no | S2b |
| A `RecordSpec` given as `given_spec` or `input_spec` is read by its children, not its leaf paths | no: today's reading raises where the two differ | S2b |
| `Function` marks a parameter with a default as an optional input, so `f.input_spec.optional` and `required` change, and an `input_spec` that marks optional a parameter without a default raises `ValueError` (O9) | the optional inputs: yes | S2b |
| A produced law or kernel offers no outputs: `OutputSpec(law_spec).components` is empty, and a merge refuses a law whose event is positional over a law, a kernel, a function, or a batch | the empty outputs: yes | S2b |
| `merge` on records, record specs, and record batches is shallow: a top-level name in both raises, groups included | no | S2b |
| `&` replaces `*` for merging laws and kernels, and replaces `merge` on `NamedTree`, `Record`, `RecordSpec`, and `RecordBatch` | no | S3 |
| The `joint` operation's route `"compose"` becomes `"merge"`, so `provenance.metadata["route"]` changes value, and a recipe recorded before S3 names a route that no longer exists | yes | S3 |
| `with_path_names` is renamed `rename` | no | S3 |
| `FieldView` is renamed `MarginalView` | no | S3 |
| `BatchSpec`'s first parameter is renamed from `element_spec` to `element`, and takes an `OutputSpec` or a term spec (§ 6.1) | no | S4 |
| `BatchSpec.element_spec` and `Batch.element_spec` are an output spec, whose `term_spec` is the old value; a raw batch of arrays offers the default output `element`; a batch's outputs are typed by their columns; two batches that differ only in a base element's name are equal | the equality: yes | S4 |
| A view or marginal of a group offers its parts, not the group's name | yes | G1a |
| A factor that holds a group wraps its draws, so each draw gains a level | yes | G1a |
| A lifted law-returning function wraps its draws under the callable's output name | yes | G1b |
| A function returning a law has a positional output with no outputs, and one returning a function offers the returned function's outputs; code that received it by its name now needs a wrap | no | G1b |
| A bare law, function, or batch spec given as `output_spec` is positional instead of named | yes | G1b |
| A random measure, a random function, a bootstrap law, an empirical law of laws, and a minibatched target wrap their draws under the name their constructor takes | yes | G2 |
| `OutputSpec(name=spec)` raises for a spec with parts, and so does filling a type hole with one | no | G2 |
| `[]` and `OutputSpec[]` no longer accept the name of the output at the root: `Normal("x", 0.0, 1.0)["x"]` raises; a view at a field that holds a law, a kernel, a function, or a batch is positional | no | G3 |
| `"x" in Normal("x", 0.0, 1.0)` becomes false, as `in` follows `[]` | yes | G3 |
| `condition_on` returns a law whose event is a positional record on every route | yes, on the routes that return the keyword form today | G4 |
| "slot" and "component" become "input" and "output": `components` becomes `outputs`, `component=` becomes `output_name=`, and `t.component(c)` becomes `t.output(c)`, if G3 has merged; the entry maps each name of the workshop's code to its new name | no | the rename after the workshop |

### 6.7 The steps

[`README.md`](README.md) § Steps owns the order of the steps, their dependencies, the critical path, and the cut.
This section owns what each step delivers, the invariants it delivers, its files, its demonstration, and its done-condition.
Every step is one PR into `dev/overhaul`.
Each step block names the rows of § 8 that the step starts and ends; a done-condition names an interim behaviour only by its row.

**Rules for every step.**
- **Words.** Each step before the rename PR (§ 6.15) writes code names, docstrings, messages, and design text in the words of § 8.2's right-hand column (T25, T26), and each step before S3 writes `*`, `merge`, `with_path_names`, and `FieldView` (T3). The step blocks below use these names.
- **Property tests of the invariants.** S1 creates `tests/test_paths_and_specs_invariants.py`, with one test per invariant I1–I22 over the running examples of § 2.2. It sits at the top of `tests/`, as `tests/test_messages.py` does, since the invariants span `core/`, `distributions/`, and `functions/`. Each test of an invariant, or of the part of one, that a later step delivers is marked `pytest.mark.pending(reason="<step>: <what it awaits>", raises=<the exception today's code raises>)`, the repository's strict expected failure (`tests/conftest.py`), so it fails loudly once delivered. Each later step removes the marks of the invariants its block lists, splitting a test where steps deliver parts of one invariant.
- **The User Guide and the CHANGELOG.** Each step updates the User Guide notebooks it breaks, mechanically, and records its breaking changes in the CHANGELOG, with the before-and-after line § 6.6 asks for each silent change.
- **Mechanical steps.** A mechanical step changes names and call sites only, so its done-condition is that the full suite passes with no test changed apart from the renamed names.
- **The tutorials.** No step edits `docs/tutorials/` (README directive 6). Every search below that covers `docs/` leaves out `docs/tutorials/` and the CHANGELOG, and a PR that breaks a tutorial says so in its body, since the advisory tutorials job may fail until the tutorial rewrite merges.
- **Demonstration.** Each step block names its demonstration: a short snippet, run on the branch, whose code and output the PR body shows.

**What the inputs and outputs merge gates, and its schedule risk.**
The merge changes only this plan, so it gates S0, and not step PL or the execution steps P0b, PV, E0, E1, E2, and E3a.
Through S0 it gates every step that depends on S1, S2a, or S2b, directly or through another step, by README § Steps: every declaration step, and E3b, E4, and the execution steps after E4.
S0 starts once the maintainer accepts this merged text, so the acceptance heads the never-cut chain PL → S0 → S1 → S2b → S3, and every day it waits after PL merges delays each of those steps by a day.
With the checkpoint on 2026-10-20 and the freeze at the end of 2026-10-25, the acceptance needs to come by the checkpoint for the chain to keep its slack; if it has not, the checkpoint of 2026-10-22 should weigh starting S0 from the original plan, with the inputs and outputs changes after the workshop.
Pending the maintainer's decision: the deadline for the acceptance, and what the fallback is, see `decisions.md`.

### 6.8 What grammar B costs, before and after the workshop

The addressing plan swept `dev/overhaul` at `638f56bc` for every interim whole form, by running the full test suite and every notebook with a probe on `OutputSpec`.
The changes fall into two classes.

**Form only, with the same names and values.**
A batch has parts under grammar B, so `sample(Normal("x", 0.0, 1.0), sample_shape=n)` and `quantile(...)`, which today declare a batch of arrays in the keyword form, become positional batches.
The element spec offers `x`, so the batch still offers `x`, and its values do not change.
This class covers most of the 367 tests and 39 notebook cells that the sweep counted as batches of arrays.
The bootstrap replicate law's tests and cells, such as User Guide 08 cell 38 and tour cell 191, belong to the visible class below.

**Visible changes.**
These change what a term offers or what its draws look like:

| Change | Where | Step |
| --- | --- | --- |
| A view or marginal of a group offers its parts instead of the group's name | `_view_declaration`, `marginal` at a group; `tests/distributions/test_views.py` (79 tests), the empirical and path-rename marginal tests | G1a |
| A factor that holds a group (a grouped sub-joint, § 6.4) is wrapped, so its draws gain the group's level: `factor(grouped, "g")` draws `{g: {c, e}}`, and its event is positional over its record; it stays one factor. `marginal(grouped, "g")` must then be that factor's marginal at `g`, not the factor itself | `_factored.py` (including `_flattened`'s test for a grouped sub-joint, `distributions/_factored.py:300`), `_regrouped` in `_views.py`, `test_regrouping_renames.py` (25 tests), the tour's hierarchical example | G1a |
| A law the library declares in an interim whole form over a record, such as an approximate posterior's undeclared event, is positional or wrapped (pending the maintainer's decision on which, see `decisions.md`) | `_posterior_declaration` and `make_posterior` in `inference/_approximate_distribution.py`, `_marginal` in `distributions/_empirical.py`; 233 tests construct such a law, most by hand, including `tests/core/test_specs.py` and `tests/distributions/test_views.py` | G1a |
| A function that returns a law has a positional output with no outputs, and one that returns a function offers the returned function's outputs | `OutputSpec.default` and its output callers, `_complete_output_spec` and `_output_atoms` | G1b |
| Lifting a law-returning function over a law wraps its draws under the callable's output name | `_lift_result` in `functions/_broadcast.py` (V.5), including the tour's `predictive` example | G1b |
| A random function's draws are wrapped, `{f: function}` | `_gaussian.py`, `_random_functions.py`; 260 tests name a function whole, most of them in `test_gaussian.py` (109) and `test_event_declarations.py` (104) | G2 |
| A law whose draws are laws stores them in a record under its constructor's name: random measures, empirical laws of laws, and minibatched targets | `_random_functions.py`, `_empirical.py`, `inference/_minibatch.py`; 181 tests name a law whole | G2 |
| A bootstrap replicate law, whose event is a batch, wraps its draws | `_resampling.py`; User Guide 08 and the tour | G2 |
| The keyword form refuses a spec with parts, directly or by filling a type hole, and the private event default goes | `core/_specs.py` | G2, last |

**Totals, from the sweep at `638f56bc`.**
In all, 1,046 tests in 72 files execute an interim whole form, and 679 tests in 46 files remain once the form-only batches are set aside.
The counts in the table overlap, since one test can build several kinds.
The library has about 31 sites, most of them through `OutputSpec.default` and the filling of type holes.
No notebook writes such a form explicitly: 6 cells produce one through defaults, and 39 more only through the form-only batches.
In `design/`, about 22 passages state or show the whole form over parts: II.2, III.3, III.7, IV.1, V.6, VI.8, VII.2, VII.5, VII.6, and the glossary.

**The split at the workshop (decided 2026-10-17).**
Grammar B for the kinds the workshop uses merges before the workshop, in two steps.
[`README.md`](README.md) § Steps owns their order, their dependencies, and the cut.

Step G1a is the fifth cut if time runs short:
- a view or marginal of a group offers its parts;
- the library's laws declared in an interim whole form over a record are positional or wrapped (pending the maintainer's decision on which, see `decisions.md`);
- a grouped sub-joint wraps its group, and stays one factor.

Step G1b is the third cut if time runs short:
- a function that returns a law has a positional output with no outputs, and one that returns a function offers its outputs;
- a lifted law-returning function wraps its draws;
- `OutputSpec.default` by kind, and the reading of a bare `output_spec`.

Random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets follow after the workshop, as step G2, which then removes the private event default of § 4.12.
G2 ends with the refusal of the interim whole form.
These families declare their draws in that form today, so refusing it before they move would break them, and a special case that exempts them would be the bridging code that README directive 8 forbids.
§ T says which forms each part of the library still builds until its step (T5 to T19).

**Prerequisites.**

Accepting a function's output, and a batch of arrays, in the positional form both depend on the rest of this plan:
- today a `FunctionSpec` cannot be positional at all, and a `BatchSpec` only when its elements are records or laws;
- accepting them needs the uniform rule of § 2.11, which step G1b implements;
- a batch of arrays needs every batch to hold an element spec (§ 5.8, step S4).

### 6.9 S0: the design text

**Delivers:** the target text of § 6.5.1, the rewording of § 6.5.4, and the glossary changes that § 6.4.1 assigns to S0.
II.2 states input specs, output specs, and the grammar rule, and II.6 the paths and how `[]` reads them; together with II.5, III.3, III.5, III.7, IV.2, IV.4, and VI.11 they state this plan's rules.
IV.2's rationale takes the fix of § 2.21, and IV.2 is retitled "Merging distributions": "merge" replaces "composition" for `&` throughout `design/`, and the operations table of `design/README.md` splits "merge" (`p(x | y) · p(y)`) from "composition" (`f ∘ g`, `A B`).
V.6 and III.3 state the bare-spec reading of § 4.13.
S0 writes in today's words, by the mapping of § 8.2 (T25, T26); the rules are this plan's.
The rename PR after the workshop changes the words in `design/`, the glossary, and the code together.
S0 does not change `STYLE_GUIDE.md`, which changes with the code (§ 6.5.2).

**Files:** `design/02-shared-abstractions.md`, `03-term-kinds.md`, `04-distributions.md`, `05-functions.md`, `06-operations.md`, `07-distribution-catalog.md`, `design/README.md`, and `glossary.md`; the `_PENDING` tables of the four `test_design_conformance.py` files.

**Demonstration:** the PR body quotes II.2's statement of the grammar rule and II.6's statement of how `[]` reads a path and a list of paths.

**Done when:**
- `tests/docs/` and the four `test_design_conformance.py` files pass. Those files check the code blocks of III.3, III.7–III.10, IV.1–IV.4, V.1–V.12, VI.0, and VII.1–VII.9 against the code through `scripts/design/design_blocks.py`. Each declaration that S0 changes ahead of the code gets an entry in the `_PENDING` table of its file, whose reason names the step that delivers it; the entry marks the test `pending`, a strict expected failure, so the delivering step must remove the entry.
- For II.2, II.5, II.6, III.5, and III.6, which no conformance test covers, the PR body gives the output of `uv run python scripts/design/design_blocks.py check II.2 II.5 II.6 III.5 III.6` on the base branch and on S0's branch, and names, for each line the branch adds, the step that delivers it. `check` has no pending list and exits with status 1 whenever it reports a difference, as it already does on the base branch for II.2, II.5, II.6, and III.5, so the condition is on the report, not the exit status. A new line that no step delivers is a defect of S0's text. VI.6, VI.8, and VI.11 have no code blocks to check.
- A search of `design/` finds none of "packaging" (apart from Python packaging in `package-structure.md`), "packaged", "produced slot", "output slot", "input slot", "map-like kind", "unmet", "whole declaration", "exposed declaration", `exposes_record`, and `.project(`; the words of T25 remain, with this plan's meanings (§ 8.2).
- Each candidate that `scripts/design/prose.py` reports is resolved under `STYLE_GUIDE.md` §10.

### 6.10 S1: one path reader

**Delivers:**
- the reader of § 4.1, with its errors and messages, and one `_PATH_SEP`;
- § 5.4 and § 5.9 for record specs: `[]` returns groups on `Record` and `RecordSpec`;
- § 5.8 apart from batches of laws and the list form at a batch's root (S4), including `()` raising on a batch and the message for `draws["mu"]`;
- the list-of-paths rule of § 5.5 and § 4.3 on records, batches, and laws, with `_view_declaration`'s list branch keeping paths and the tree's order;
- § 4.6 for terms and record specs, apart from paths through held terms, `in` on laws included;
- `marginal(d, paths)` (§ 5.5), the reader's errors in `without` (§ 4.5), and `select` (§ 5.4);
- the docstrings of § 6.5.3 and § 6.6: `at_path` recommends `[]`, and `[]` says "a tuple is a list of paths; a nested path is written with /";
- `STYLE_GUIDE.md` §1.10 for paths (§ 6.5.2);
- the module of invariant tests (§ 6.7).

**Transition rows:** starts T18 for `in`, since S1 adds `in` on laws; `Normal("x", 0.0, 1.0)[["x"]]` is the law over the record `{x}` by § 4.4, rule 3, as today's tuple form gives it (O6).

**Invariants:** I4, I9, I5 on records and batches, I10 on records, record specs, and batches, I2 and I7 for lists of paths, I1 for records and batches, I18 for terms and record specs, and I3 with #576.

**Files:**
- `core/_shapes.py`: the one reader, beside `_as_names`;
- `core/named_tree.py`: the one `_PATH_SEP`, `__getitem__`, `__contains__`, `at_path`, `is_field`, `without`, the keys of `with_path_names`;
- `core/record.py`, `core/_record_spec.py`, `core/_record_batch.py`: groups, `select`;
- `core/_batch.py`: `__getitem__`, `__contains__`, `()`;
- `distributions/_views.py`, `distributions/_distribution.py`: lists of paths on laws, labels, `in` on laws;
- `operations/_marginal.py`: `paths`, lists;
- the eight modules that define their own `_PATH_SEP`: `operations/_marginal.py`, `operations/_condition.py`, `functions/_descendants.py`, `distributions/_empirical.py`, `distributions/_views.py`, `distributions/_factored.py`, `distributions/_conditional.py`, `families/_resampling.py`;
- the literal `"/"` separators: about 20 occurrences in about 14 modules, such as `families/_mixture.py`, `functions/_broadcast.py`, `inference/_bayesflow_common.py`, `validation/_calibration.py`, and `diagnostics/_loo.py`;
- `STYLE_GUIDE.md` §1.10;
- `tests/test_paths_and_specs_invariants.py`, new.

**Demonstration:** for `draw = sample(grouped)`, `draw["g"]`, `draw["a", "g/c"]`, and `"g" in draw`, and `hier["s1/theta", "s2/theta"]`.

**Done when:**
- tests cover every row of § 4.1's error table on records, record specs, batches, and laws, and each message of § 4.1, the tuple hint included;
- `hier["s1/theta", "s2/theta"]` and `marginal(grouped, paths=["a", "g/c"])` give the laws of § 4.3;
- `draws[3]["g"] == draws["g"][3]`, and `mean(grouped)["g"]` is the record of means;
- `"g" in sample(grouped)`, `"a" in grouped`, and `"a" in sample(grouped, sample_shape=4)` are true, and `1.0 in sample(Normal("x", 0.0, 1.0), sample_shape=3)` is false;
- `sample(Normal("mu", 0.0, 1.0), sample_shape=4)["mu"]` raises `KeyError` with the message of § 5.8;
- `draw.select("a", "g/c")` is `{"a": ..., "c": ...}`, and `sample(hier).select("s1/theta", "s2/theta")` raises `ValueError`;
- `Normal("x", 0.0, 1.0)[["x"]]` keeps today's result;
- no module but `named_tree.py` defines `_PATH_SEP`, and each literal `"/"` that splits or joins a path uses the one separator and reader; any other literal `"/"`, such as a DataTree group path, is named in the PR as unrelated to paths;
- `tests/test_paths_and_specs_invariants.py` exists, the tests of S1's invariants pass, and every other test in it is marked `pending` with its step.

### 6.11 S2a and S2b: `term_spec`, and the operations on input and output specs

S2a is announced and mechanical; S2b changes behaviour.

**S2a delivers:**
- `spec` becomes `term_spec`, `with_spec` becomes `with_term_spec`, and `exposes_record` goes;
- one private helper, `_root_component_name(output_spec) -> str | None` in `core/_specs.py` (§ 5.11), returns the name of the output at the root, or `None`; it replaces `_whole_term_component` in `distributions/_distribution.py`, on 20 lines of 4 modules of `distributions/`, and every read of `_component_name` outside `core/_specs.py`;
- the 7 callers of `_components_record` use `RecordSpec(O.components)`, and the helper goes (§ 6.1).

`components` keeps its name, and is not rewritten into a mapping; S2a renames none of the words of § 8.2, which wait for the rename after the workshop.
Its renames change about 50 modules, which is why it is announced.

**S2a transition rows:** starts T17.

**S2a files:** `core/_specs.py`, `core/_fingerprint.py`, and every module that reads the renamed names: `.event_spec.spec` on 99 lines of `probpipe/` and 199 of `tests/`, `.output_spec.spec` on 4 lines of `probpipe/` and 24 of `tests/`, `exposes_record` on 33 and 31, and `._component_name` on 19 lines of 6 modules (counts at `6f8859fc`, fact 22); `_whole_term_component` in `distributions/_distribution.py`, `_factored.py`, `_views.py`, and `_empirical.py`; and the callers of `_components_record`.

**S2a demonstration:** `prior.event_spec.term_spec` is the `RecordSpec` of a draw, and `prior.event_spec.spec` raises `AttributeError`.

**S2a is done when:** a search of `probpipe/`, `tests/`, and `docs/` finds no `._component_name` outside `core/_specs.py`, no `_whole_term_component`, no `_components_record`, no `exposes_record`, and no `.spec` and no `with_spec` on an `OutputSpec`; the docstrings and messages that say "packaging", "packaged", or "map-like kind", such as the `InputSpec` docstring, are updated; and the full suite passes.

**S2b delivers** § 5.10 and § 4.9, with § 5.10's messages, the `__contains__` and `get` of § 6.1, and § 5.11, § 4.4, § 4.5, and § 4.14, with `OutputSpec.__iter__`'s refusal, the record merge of § 2.21, and the documentation of § 6.5.3.
It keeps the original plan's content minus the law's event rule, plus the items below, since it is on both critical chains (S0 → S1 → S2b → S3, and S2b → S4 and S2b → E4):

| Item | Original plan | This plan | Files | Call sites and tests |
| --- | --- | --- | --- | --- |
| `OutputSpec.__getitem__`, `__contains__`, `without`, `__and__`, and `__iter__`'s refusal | `project`, `without`, `&` | the same logic under `[]`, plus `in` and the refusal | `core/_specs.py`, `distributions/_views.py` | `_view_declaration`'s 2 call sites (`distributions/_views.py:1057`, `:1063`) move to `O[s]`, apart from one path to a group (T5) and one path to a held field (T19) |
| `InputSpec` sources, size shorthand, `&`, strict `without`, O9, and § 5.10's messages | yes | the same; `&` keeps today's first-operand spec for a shared input (T29) | `core/_specs.py`, `values/_function_base.py`, `distributions/_conditional.py` | 4 readers (`_slots_of`, `_complete_given_spec`, `ConditionalDistributionSpec.__init__`'s `given_spec`, `Function._initialize`); 3 callers of `InputSpec.without`; tests `tests/core/test_specs.py:366` and `:1151` change |
| `InputSpec[]` with a list of names, `__contains__`, and `get` | no | yes | `core/_specs.py` | none |
| The joint's event and inputs through `&` | yes | the same | `distributions/_factored.py` | `_factor_graph` (1 function) |
| The law's event rule | yes, in four constructors | not done (§ 5.5) | — | — |
| A law-valued or kernel-valued produced term offers no outputs | no | yes, replacing the event rule | `core/_specs.py` (`_exposed_components`, 3 call sites) | 3 tests change (fact 21) |
| Shallow record merge | no (S3, deep) | yes | `core/named_tree.py`, `core/record.py`, `core/_record_batch.py` | 3 library call sites; 2 tests and 1 message test change (fact 5) |
| `docs/api/declarations.md`, with § 6.5.3's one idiom | yes | the same, in the words of § 8.2 | `docs/api/declarations.md` | — |
| The User Guide's `given_spec` idioms | no | yes: `given_spec=<law>.event_spec.components` becomes `given_spec=<law>.event_spec`, and `{"<name>": <law>.event_spec.components["<name>"]}` becomes `<law>.event_spec["<name>"]`, in User Guide 04, 06, 07, 08, and 09, wherever the page's output is unchanged; each is a one-token change, so it counts as mechanical, and these cells then need nothing from the rename | the five notebooks | — |

The three callers of `InputSpec.without` are `_bound_factor` in `distributions/_factored.py`, and the `_condition_on` methods of a view in `distributions/_views.py` and of a kernel in `distributions/_conditional.py`; S2b checks that each passes only inputs.

So S2b edits eight Python modules, the docs page, and five notebooks, against the original plan's six modules and the docs page, and changes 6 tests beyond the original plan's.
With S2b's two new behaviours patched in, every User Guide and Get Started notebook that runs here passed, and neither behaviour was exercised (fact 45).
Everything else waits: the column typing of a batch's outputs goes to S4, the `condition_on` bug of § 2.20 to `deferred.md`, and functions' merges stay after the workshop.
Pending the maintainer's decision: whether S2b splits into S2b-out, the operations on output specs and the record merge, which S3, S4, and E4 need, and S2b-in, the reading of inputs, which merges by the freeze; see `decisions.md`.

**S2b invariants:** I6 for the paths S1 supports, I10 on output and input specs, I1 for output specs, I18 for output and input specs, and I16 and I17 for the specs.

**S2b transition rows:** ends T1 and T2; starts T18 for `O[n]`, T29, and the gaps of `O[s]` in T5 and T19.

**S2b demonstration:** the likelihood of § 5.11, declared with `given_spec=growth_prior.event_spec.without("sigma") & shocks_given_sigma.event_spec`, and its `given_spec`.

**S2b is done when:**
- `grouped.event_spec["g/c"] == OutputSpec(c=...)`, `grouped.event_spec[["g"]]` has the one output `g`, `"g/c" in grouped.event_spec`, and `grouped.event_spec.without("g/e")` keeps `{a, g: {c}}`;
- `list(prior.event_spec)` raises `TypeError` with § 5.11's message;
- `d[s].event_spec == d.event_spec[s]` and `(a * b).event_spec == a.event_spec & b.event_spec` over the running examples, for the paths S1 supports, none of which goes through a held term, leaving out one path to a group (T5) and one path to a held field (T19);
- `OutputSpec(post_spec).components == {}`, and `fit_wrapped.output_spec["posterior"] == OutputSpec(post_spec)`;
- `laws_of_laws`, the law of § 5.5 whose event is positional over a law, builds, with `laws_of_laws.event_spec.components == {}`; `laws_of_laws * Normal("x", 0.0, 1.0)` raises `TypeError`, whose message says to wrap the draws under a name; and the wrapped `laws_of_laws` of § 5.5 merges;
- `Normal("z", 0.0, 1.0) * random_function` still merges, as T15 states;
- the `InputSpec` constructor's four sources and the size shorthand work, `InputSpec(K=())`, `given_spec={"K": ()}`, and `OutputSpec(K=())` build and `InputSpec(K=3)` raises, `InputSpec(a=NumericArraySpec(())).without("zzz")` raises `KeyError` and its three callers pass only inputs, `likelihood.given_spec[["K", "r"]]` is an `InputSpec`, an `OutputSpec` source with an output name that is not an identifier, or with a type hole, raises, and an optional input whose parameter has no default raises `ValueError`, for kernels and functions (O9);
- `["mu"] in lik.given_spec` and `("mu",) in lik.given_spec` are `False`, and `lik.given_spec.get(["mu"])` is `None`;
- a nested `RecordSpec` and an `OutputSpec` each declare a kernel's inputs, through `conditional_distribution` and through `ConditionalDistributionSpec(RecordSpec(...), ...)` and `ConditionalDistributionSpec(OutputSpec(...), ...)`, and the regrouped law of § 1.2 builds a kernel;
- each of § 5.10's four messages has a test;
- `Record({"g": {"c": 1.0}}).merge(Record({"g": {"e": 2.0}}))` raises `ValueError`, and so do `RecordSpec.merge` and `RecordBatch.merge` for a shared top-level name;
- tests cover each error of § 4.4, § 4.5, § 5.11, and § 2.18;
- the five User Guide notebooks run with the new idioms;
- the full suite passes.

### 6.12 S3: `&`, `rename`, and `MarginalView`

**Delivers:** one announced mechanical PR:
- `&` replaces `*` for merging laws and kernels: `Distribution.__mul__`, `ConditionalDistribution.__mul__`, and the merge branch of `GaussianRandomFunction.__mul__`, whose scaling branch stays;
- "merge" replaces "composition" in the code that implements `&`: `distributions/_composition.py` becomes `distributions/_merge.py`, `tests/distributions/test_composition.py` becomes `test_merge.py`, the `joint` operation's route `"compose"` becomes `"merge"` with its functions `_can_compose`, `_compose`, and `_composed_expression`, and their docstrings and comments; composition of linear operators and of functions keeps its name;
- `&` replaces `merge` on `NamedTree`, `Record`, `RecordSpec`, and `RecordBatch`, with S2b's shallow rule;
- `with_path_names` becomes `rename` on every kind that has it, and `STYLE_GUIDE.md` §1.10 says `rename`;
- `FieldView` becomes `MarginalView`;
- the error and warning messages of merges and renames say "merge", "&", and "rename", in the words of § 8.2.

Pending the maintainer's decision: whether `*` between two laws or kernels stays as a permanent refusal that names `&` (§ 2.15), see `decisions.md`.

**Invariants:** I16 for the operator, I17 for laws, and I18 for `rename`; I15, I19, I20, and I21 hold for the merges and renames it delivers, and every existing test passes under the new names.

**Transition rows:** ends T3.

**Files:** `distributions/_distribution.py`, `distributions/_conditional.py`, `distributions/_composition.py`, `operations/_joint.py`, `families/_gaussian.py`, `core/named_tree.py`, `core/record.py`, `core/_record_batch.py`, `core/_specs.py`, `distributions/_views.py`, and every caller: about 400 lines name `with_path_names` and about 260 name `FieldView` across `probpipe/`, `tests/`, `docs/`, and `design/`.

**Demonstration:** `model = lik & prior`, and the message of `prior & lik`.

**Done when:**
- no `__mul__` merges laws, and no code name, docstring, or comment calls `&` or a joint law's construction a composition;
- a search finds no `with_path_names`, no `FieldView`, no `.merge(` on a `NamedTree`, a `Record`, a `RecordSpec`, or a `RecordBatch`, and no `*` between two laws or kernels, in `probpipe/`, `tests/`, `docs/`, and `design/`, leaving out `docs/tutorials/` and the CHANGELOG (§ 6.7);
- no error or warning message calls a merge a product or a composition, or suggests `*` or `with_path_names`; the order message (M4) reads "write likelihood & prior: likelihood takes K, which prior produces, and in a & b only b's components feed a's slots";
- the User Guide notebooks run, and the full suite passes.

### 6.13 S4: every batch stores its element spec

**Delivers:** § 5.8:
- `BatchSpec.element_spec` and `Batch.element_spec` become the output spec of one element (§ 2.9), and every use of their old meaning, the element's term spec, becomes `element_spec.term_spec` in the same PR (directive 7);
- `BatchSpec` stores the name of a base element, and the positional form for every other element; draws (VI.3), sweeps (V.6), and the batches `functions/_result.py` stacks store the name their maker gave; a raw batch gets the default `element`, which its repr does not show; a column of a batch of records stores the view's element spec;
- a batch of records offers its columns, each typed by its column's `BatchSpec` (`_exposed_components` for batches);
- the list form at a batch's root, `draws[["x"]]` (§ 5.8), and renaming a base element's name by `rename` (§ 4.8, case 2; T3 if S3 has not merged);
- the name-free canonical spec of § 5.8 in a batch's equality and hashing, in fingerprints, and in every pytree flatten (O3).

Pending the maintainer's decision: whether S4 is marked announced and kept as one PR, with the mechanical rewrite of `element_spec` to `element_spec.term_spec` in its own commit, or split into an announced, mechanical S4a and a behavioural S4b; see `decisions.md`.
Pending the maintainer's decision: how evaluation points run when a callable returns a law, a function, or a batch, and the order of S4 and E5, which both rewrite `functions/_sweep.py` and `functions/_result.py`; see `decisions.md`.

**Invariants:** I6 for batches, apart from the output at the root (T20); I9 for a batch's output at the root; and it stores what I12 reads for batches (G3).

**Transition rows:** ends T4; starts T6 and T20.

**Files:** `core/_batch.py`, `core/_specs.py` for `_exposed_components`; the pytree registrations of every batch class and of every tracked kind whose static data can hold a `BatchSpec`, such as `core/_numeric_array_batch.py`, `core/_record_batch.py`, `core/_numeric_record_batch.py`, and `core/record.py`; `distributions/_batches.py`; `core/_fingerprint.py`; `functions/_sweep.py`, `functions/_result.py`; `operations/_sample.py`; and every other reader of `element_spec`: 364 lines in 28 files of `probpipe/`, 517 lines of `tests/`, and 11 lines of `design/`, counted at `9bf48ae0`.

**Demonstration:** `sample(Normal("mu", 0.0, 1.0), sample_shape=4).element_spec`, and `jax.tree_util.tree_map(jnp.subtract, x, y)` on draws of `Normal("x", 0.0, 1.0)` and `Normal("y", 0.0, 1.0)`.

**Done when:**
- `sample(Normal("mu", 0.0, 1.0), sample_shape=4).spec.element_spec == OutputSpec(mu=...)`;
- `NumericArrayBatch(values, "draw", label="draws").spec.element_spec` offers `element`, and the batch's repr does not show the default name `element`;
- `grouped_draws["a"].spec.element_spec == OutputSpec(a=...)`, and `grouped_draws["g"]`'s is `OutputSpec(RecordSpec(c=..., e=...))`, while `grouped["g"]` is as T6 states;
- `OutputSpec(grouped_draws.spec).components["a"]` is the `BatchSpec` of the column `a`;
- `sample(Normal("x", 0.0, 1.0), sample_shape=4)[["x"]]` is a `RecordBatch` of `{x}`;
- for `x` the draws of `Normal("x", 0.0, 1.0)`, `x.rename(x="y").element_spec == OutputSpec(y=...)`, with an unchanged fingerprint;
- a `DistributionBatch`'s `event_spec == spec.element_spec.term_spec.event_spec`, and its `spec.element_spec.term_spec` is the `DistributionSpec` that `spec.element_spec` held before;
- a search of `probpipe/`, `tests/`, and `design/` finds no use of `element_spec` that expects a term spec;
- `x.spec != y.spec` for draws of `Normal("x", 0.0, 1.0)` and `Normal("y", 0.0, 1.0)`, while `x == y` holds for equal values;
- `jax.tree_util.tree_map(jnp.subtract, x, y)` works, a jitted function compiles once for both, and a jitted call made with `y` after one made with `x` returns a batch offering `element`, never `x`;
- the same holds for a record whose field holds such a batch.

### 6.14 G1a and G1b: grammar B for the kinds the workshop uses

**G1a delivers**, and is the fifth cut if time runs short:
1. a view or marginal of a group offers its parts, and a group's marginal is the factor's marginal at the group (§ 5.5);
2. the library's laws declared in an interim whole form over a record are positional or wrapped (§ 6.8); pending the maintainer's decision on which, see `decisions.md`;
3. a grouped sub-joint wraps its group, so its event is positional over its record, and stays one factor;
4. M3's message for two views of one parent (§ 2.17, § 2.21).

For item 3: today `_flattened` flattens a factored joint only when its event is not an interim whole form (the test `_whole_term_component(factor.event_spec) is None`, `distributions/_factored.py:300`, which S2a writes with `_root_component_name`), so a grouped sub-joint, whose event is one today, stays one factor.
Once a grouped sub-joint's event is positional, that test no longer finds it, so G1a replaces the test with a structural one.
A factored joint is flattened exactly when the names of its event's outputs are the names of its factors' outputs:

```python
set(d.event_spec.components) == {name for f in graph.factors for name in f.event_spec.components}
```

A grouped sub-joint places its factors' outputs under its group, so its outputs, `{g}`, differ from its factors' outputs, `{c, e}`, and it stays one factor.
An ordinary joint's outputs are its factors' outputs (§ 2.19, R1), whatever its dimension scope binds, so it is flattened as today.

**G1a invariants:** I2, I5, I6, and I7 for groups, and I17 for the grouped sub-joint.

**G1a transition rows:** ends T5, T6, T7, T8, and T9, and T16 for the grouped sub-joint; T19 continues until G3.

**G1a files:** `_view_declaration` and `_regrouped` in `distributions/_views.py`; `operations/_marginal.py`; `_joint_declaration`, `_flattened`, and M3's message in `distributions/_factored.py`; `_posterior_declaration` and `make_posterior` in `inference/_approximate_distribution.py`; `_marginal` in `distributions/_empirical.py`; `tests/_interim_form_probe.py`, new, and `tests/conftest.py`.

**G1a demonstration:** `grouped["g"].event_spec` and `sample(grouped["g"])`, which agree on the paths `c` and `e`.

**G1a is done when:**
- `grouped["g"]["c"]` is the view of `g/c`, and `grouped["g"].event_spec == grouped.event_spec["g"] == marginal(grouped, "g").event_spec`, offering `c` and `e`;
- `factor(grouped, "g")` draws `{g: {c, e}}`, its event is positional over its record, and it stays one factor of `grouped`, and of `grouped` merged with `Normal("z", 0.0, 1.0)`, while `grouped`'s draws are unchanged;
- a law regrouped so that `lik_opt` is inside a grouped sub-joint (§ 4.16, item 4; fact 43), merged with `s_prior`, keeps `y`'s spread near 1;
- `hier["s1"]` merged with `hier["s2"]` raises M3 with the message for views of one parent;
- `uv run pytest -n 3 --interim-form-probe` passes.

**G1b delivers**, and is the third cut if time runs short:
1. a `FunctionSpec`, and a `BatchSpec` of base elements, are accepted positionally (§ 2.11);
2. a function that returns a law declares a positional law output with no outputs, and one that returns a function offers the returned function's outputs (§ 5.7);
3. `OutputSpec.default` follows § 6.1, and `_complete_output_spec` reads a bare `output_spec` by it (§ 4.12, § 4.13);
4. `_lift_result` wraps a returned law, kernel, function, or batch under the callable's output name (§ 4.12).

**G1b transition rows:** ends T10 and T11; T14 continues until G2.

**G1b invariants:** none of I1–I22 directly; it makes outputs offer the names inside their values, which I12 reads in G3.

**G1b files:** `core/_specs.py` (`_exposed_kind`, `default`); `values/_function_base.py` (`_complete_output_spec`); `_output_atoms` and `_lift_result` in `functions/_broadcast.py`; the probe's allowance table.

**G1b demonstration:** `Function(run_inference, output_spec=post_spec).output_spec`, and the draws of `law_of_y` lifted over `Normal("x", 0.0, 1.0)` (§ 6.2.6).

**G1b is done when:**
- `Function(run_inference, output_spec=post_spec).output_spec == OutputSpec(post_spec)`, with no outputs, and a `fit_emulator` output spec offers `mean` and `var`;
- lifting `def law_of_y(x)` over a law gives draws `{law_of_y: law}`, and with `include_inputs=True` draws `{x, law_of_y: law}`;
- `OutputSpec.default(BatchSpec(NumericArraySpec(()), sample=3), component="f")` offers `f`;
- G1b's rows are gone from the probe's allowance, and `uv run pytest -n 3 --interim-form-probe` passes.

**The interim-form probe.**
G1a adds a test-only probe: the opt-in pytest plugin `tests/_interim_form_probe.py`, registered in `tests/conftest.py` through `pytest_plugins` and enabled by the option `--interim-form-probe`.
CI does not run it, so it adds no time to the blocking jobs; each step below that changes its allowance runs it once and states the result in its PR body.
When enabled, it wraps the keyword branch of `OutputSpec.__init__` and `with_term_spec` while the suite runs.
For each interim whole form, it records the first stack frame outside `core/_specs.py` that built it, since the innermost frame is always inside `OutputSpec`'s own code:
- a frame in `tests/` means the test built the form by hand, which T12 keeps legal, so the probe ignores it;
- a frame in `probpipe/` means the library built it, and the probe checks the frame's function and the kind of the spec (law, function, batch, or record) together against the allowance, so a group view that regresses to the interim form is caught while a view at a law-valued field is allowed.

Under `pytest-xdist`, each worker writes what it finds to a file in pytest's temporary directory, and at the end of the session the controller reads the files, prints each library-built interim form outside the allowance with its frame and kind, and makes the session fail.

The allowance is a module-level table in the plugin: one row per library site, each with the step that ends it.
At any time the allowance is the rows whose step has not merged, so the G1b sites stay allowed after G3 when G1b has not merged, and if G1b is cut its rows stay until it merges after the workshop.
Each step deletes its own rows in its PR:

| Library sites that may still build the interim whole form | Until |
| --- | --- |
| the G1b sites: `_complete_output_spec`, `_output_atoms`, `_lift_result` (T11) | G1b |
| a view at a field that holds a law, a kernel, a function, or a batch (`_view_declaration`; T19) | G3 |
| the G2 families and the private event default: `_whole_term_event`, the event reading of `distribution`, `_agreed_event_spec` (T14) | G2 |

No other library site may build the form once G1a merges: a group view, the grouped sub-joint, and the laws of § 6.8's third row are not in the table.
G2 deletes the plugin with its last rows, since the keyword form then refuses the interim whole form.

### 6.15 After the workshop

Each step's demonstration is the first item of its done-condition, run in the PR body.

**G2.**
*Delivers:* each G2 family wraps its draws under the name its constructor takes, so its event is positional over `RecordSpec(<name>=...)` (§ 5.5); then the keyword form and the filling of a type hole refuse a spec with parts (§ 2.7, § 4.12), with a message that names the kind of the spec, and the private event default goes.
*Invariants:* I1 for the draws of those families; I17 for their interim forms; grammar B's guarantee.
*Transition rows:* ends T12 to T17, and with them the interim whole form of § 8.1.
*Files:* `_whole_term_event`, `_event_repr_parts` (`distributions/_distribution.py`), `distribution` (`distributions/_from_functions.py`), `_agreed_event_spec` (`distributions/_conditional.py`), `families/_random_functions.py`, `families/_gaussian.py`, `families/_resampling.py`, `distributions/_empirical.py`, `inference/_minibatch.py`, `core/_specs.py`, and `tests/_interim_form_probe.py`, which goes.
*Done when:*
- `rm.event_spec == OutputSpec(RecordSpec(G=law_spec))` for a random measure `rm` constructed under the name `G`, and `sample(rm)` is the record `{G: law}`;
- `gp`'s draws are `{f: function}`; `BootstrapReplicateDistribution("b", ...)` draws `{b: batch}`; an empirical law whose atoms are laws, and a minibatched target, draw records under their constructors' names;
- `OutputSpec(params=RecordSpec(beta=(), sigma=()))` raises `TypeError` with the message of § 6.1, and `OutputSpec(G=law_spec)` raises with the same message for a law: "G would name a law; give OutputSpec(law_spec) to offer it positionally, or wrap it: OutputSpec(RecordSpec(G=...))";
- `OutputSpec(x=None).with_term_spec(RecordSpec(a=()))` raises;
- a search finds no `_whole_term_event`, and the probe is removed.

**G3.**
*Delivers:*
- paths through every kind (§ 2.4, § 5.4), and `Function[]` (§ 5.7);
- the accessor `t.component(c)` on records, laws, functions, and batches (§ 4.11), spelled `t.output(c)` if the rename has merged;
- `[]` and `OutputSpec[]` stop reading the name of the output at the root in the keyword form over a base spec (§ 4.2, § 4.4, rule 4), while an interim whole form keeps the reading of § 8.1 (T16);
- a view at or below a field that holds a law, a kernel, a function, or a batch is positional, with no wrap (§ 5.5), and `_view_declaration` goes, since `O[s]` then serves every path.

*Invariants:* I1 and I5 across kinds; I2, I6, and I9 at held fields; I12; I13 and I14 for functions; I18 for paths through held terms.
*Transition rows:* ends T18, T19, T20, and T21.
*Files:* `core/named_tree.py`, `core/record.py`, `core/_batch.py`, `core/_specs.py`, `values/_function_base.py`, `distributions/_views.py`, `distributions/_distribution.py`, `core/tracked.py`, and the probe's allowance table.
*Done when:*
- the property tests of I1, I5, I12, and I13 pass. I12's test checks `output_spec(t).outputs[name]` against `t.output(name).spec` for a record or a batch, including `record_with_law`, whose `outputs["G"]` is `post.spec`, and against `t.output(name).event_spec.term_spec` for a law and `t.output(name).output_spec.term_spec` for a function;
- for a law `L` declared `OutputSpec(RecordSpec(G=law_spec))`, built in the test from a sampler, `L["G"].event_spec == OutputSpec(law_spec)`, `sample(L["G"])` is a law, and `L["G"]["mu"] == L["G/mu"]`; the test law does not depend on G2's families;
- for `record_with_law` (§ 5.4), `record_with_law["G/mu"]` is the view `post["mu"]`, `record_with_law["n", "G/mu"]` is the record `{n, G}` whose `G` is `post[["mu"]]`, and `"G/mu" in record_with_law` is true;
- `Normal("x", 0.0, 1.0)["x"]` raises `KeyError`, `Normal("x", 0.0, 1.0).component("x")` is the law, and `draws.component("mu") is draws` for draws of `Normal("mu", 0.0, 1.0)`;
- § 8.1, rule 3 still holds for a random measure `rm_whole` declared `OutputSpec(G=law_spec)` (T16);
- a search finds no `_view_declaration`, and `uv run pytest -n 3 --interim-form-probe` passes with G3's rows gone.

**G4.**
*Delivers:* the result of `condition_on` has the event spec `d.event_spec.without(*fixed)` on every route, with nested paths route by route (§ 5.5); `law.without` (§ 4.5); paths into a record input in `InputSpec.__getitem__`, `__contains__`, and `without` (§ 5.10, § 5.10).
*Invariants:* I8, I10 on laws, I11.
*Transition rows:* ends T22 and T23.
*Files:* `operations/_condition.py`, the conditioning routes of the families, `distributions/_views.py`, `core/_specs.py`.
*Done when:* property tests of I8, I10, and I11 pass, including on the Gaussian route; `lik_g.given_spec["g/c"]` is `NumericArraySpec(())`, and `lik_g.given_spec.without("g/c")` leaves `g` holding `{e}`, required.

**G5.**
*Delivers:* `[]` on batches of laws and of functions, and `[]` and the accessor `t.component(c)` (or `t.output(c)`) on kernels (§ 5.8, § 5.6, § 4.11); `STYLE_GUIDE.md` §1.11.
*Invariants:* I12, I13, and I14 for kernels, I6 for kernels, I1 on batches of laws.
*Transition rows:* ends T24.
*Files:* `distributions/_batches.py`, `distributions/_conditional.py`, `core/_function_batch.py`.
*Done when:*
- `laws["mu"]` is the batch of the views `law["mu"]`, and a batch of functions indexed by a path is the batch of each function indexed there;
- the law of `lik[s]` at `w` is the law of the node at `s` of a draw of `lik(w)`, and `lik[s].given_spec == lik.given_spec` (I14);
- `lik.component("y")`, or `lik.output("y")`, is `lik`.

**The rename.**
*Delivers:* one announced PR after the workshop, mechanical apart from the judgement of each word's sense:
- in `design/`, the glossary, the code, its docstrings and messages, and the User Guide: "slot" and "component" become "input" and "output", and "input declaration", "output declaration", and "event declaration" become "input spec", "output spec", and "event spec" (§ 8.2, § 6.4.1);
- `components` becomes `outputs`, `component=` becomes `output_name=`, `factor`'s `component_name` becomes `output_name`, and `t.component(c)` becomes `t.output(c)`, if G3 has merged;
- the private names and the docs page of § 6.4's last row;
- in `execution-layer.md` and the code it governs, its terms built on "declaration", such as "output declaration" and "complete output declaration", in the same way;
- a CHANGELOG entry that maps each name of the workshop's code to its new name (§ 6.6).

*Order.* The rename and G3 may merge in either order, and each adapts to the other: if the rename merges first, G3 adds `t.output(c)` directly, and no `t.component(c)` ever exists; if G3 merges first, it adds `t.component(c)`, and the rename renames it.
Pending the maintainer's decision: whether the rename must precede G2 to G5, so that `t.component(c)` never ships, see `decisions.md`.
*Invariants:* none.
*Transition rows:* ends T25 and T26.
*Files:* `design/`, `probpipe/`, `tests/`, `docs/` apart from `docs/tutorials/` (§ 6.7), the User Guide notebooks, and `plans/execution-layer.md`.
*Done when:* a search finds no `.components` on an `OutputSpec`, no `component=` keyword, no `t.component(` accessor, and no "slot", "component", "input declaration", "output declaration", or "event declaration" in the senses § 8.2 maps, with a list by sense of every remaining occurrence, such as a mixture's component laws and `__slots__`; and the full suite passes.

**The other work after the workshop.**
`&` on functions and other batches, spec merges, `InputSpec.rename`, and `without` on the other kinds (§ 6.16, § 6.3), and the items of `deferred.md` from the inputs and outputs plan, in `values/_function_base.py`, `core/_batch.py`, and `core/_specs.py`; done when the examples of § 5.7 and § 6.16 run. This work ends T27, T28, T29, and T30; T31 ends when the rest of #506 merges.

Before G1a, G1b, and G2 are final, jhuggins's Rocq model should be extended to renames by path and to paths inside inputs (`deferred.md`, gaps L1).

### 6.16 `&` after the workshop

These parts of `&` are design, but they merge after the workshop (`deferred.md`):
1. **`&` on `Function`:** functions feed, as § 5.7 defines, with complete specs; `center & spread` and `standardize & summaries` of § 5.7 are the examples.
2. **`&` on batches other than `RecordBatch`:** element by element, over equal levels, two batches of arrays giving a batch of records (§ 2.16); aligning other levels by name, with broadcasting, stays later still.
3. **`&` on `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, and `BatchSpec`, with `InputSpec.rename`:** a function that builds a model declares its output before it runs.

   ```python
   step = observation.spec & transition.spec     # y and x, given x_prev and theta
   renames = lambda t: {"x": f"x{t}", "y": f"y{t}", "x_prev": f"x{t - 1}"}
   chain = functools.reduce(operator.and_, [step.rename(renames(t)) for t in (3, 2, 1)])
   build_model = Function(make_chain, output_spec=OutputSpec(RecordSpec(model=chain & initial.spec & prior.spec)))
   ```

   The model is wrapped under `model`, since a merge refuses a produced law (M2) and grammar B refuses `OutputSpec(model=...)` over a law.
4. **The point-mass reading of merges across categories** (§ 2.16), and a lifting rule inside a merge, so an input that expects a value can be fed a law- or function-valued output (§ 2.21).
5. **The long-term rules** for optional inputs across merges (§ 4.16), the spec of a shared input (§ 2.18), and the dimension scopes of merge results (§ 4.17).

### 6.17 The required updates, as a checklist

This is #571's checklist, rewritten for the merged design.

- [ ] **The design reference (S0).** II.2 owns the definitions, the grammar rule, the rule of § 2.11, `term_spec`, the sources of inputs, and the operations. II.1 states the size shorthand once. II.5 states the element spec and the raw defaults. II.6 owns paths and how `[]` reads them and states `without(*paths)`. III.3, III.9, and IV.4 point to II.2 for the sources of `input_spec` and `given_spec`. III.7 defines a view's event spec as `event_spec[s]`. IV.1 and IV.2 use the new vocabulary, define merge as § 2.15 does, and define the joint's event with `&`. § 6.5.4's passages are reworded. The glossary changes as § 6.4.1 lists. All in today's words (§ 8.2). *(Done as § 6.9 states.)*
- [ ] **The path reader (S1).** *(Done as § 6.10 states.)*
- [ ] **`term_spec` (S2a).** Replace the reads of `exposes_record`, of `_component_name` outside `_specs.py`, and of `.spec` on an output spec. Update the docstrings and messages that say "packaging", "packaged", or "map-like kind". *(Done as § 6.11 states for S2a.)*
- [ ] **One reading of inputs (S2b).** The `InputSpec` constructor accepts the four sources and the size shorthand; `_slots_of`, `_complete_given_spec`, and the `input_spec` reading construct through it; the keyword form of `OutputSpec` accepts the shorthand; `InputSpec[]` takes a list of names. *(Done as § 6.11 states for S2b.)*
- [ ] **The operations (S2b).** `OutputSpec[]`, `in`, `without`, and `&`; `InputSpec` `&`; `InputSpec.without` raises on an unknown name; `_view_declaration` becomes `O[s]`; the joint's event uses `&`; a produced law or kernel offers no outputs; record merges are shallow. *(Done when tests cover each raise and check the equations I6 and I16.)*
- [ ] **The renames (S3).** *(Done as § 6.12 states.)*
- [ ] **The batch's element spec (S4).** *(Done as § 6.13 states.)*
- [ ] **Grammar B for the workshop's kinds (G1a, G1b), and for the rest (G2).** *(Done as § 6.14 and § 6.15 state.)*
- [ ] **The documentation (S2b, then each step).** `docs/api/declarations.md` opens with the definitions; the User Guide notebooks follow each step; the CHANGELOG records each breaking change of § 6.6. *(Done when `uv run mkdocs build --strict` and the User Guide notebook job pass.)*

### 6.18 Decisions: how they are recorded, and those resolved by README

> **Decided 2026-10-18:** the maintainer accepted every recommendation of the original plan's § 21, O1–O16 included (`decisions.md`).
> **Decided 2026-10-18, provisional pending the maintainer's final read of the inputs and outputs plan:** the inputs and outputs plan is merged with the recommended answer to each of its questions Q1–Q6 (§ 6.20), and the elements' output spec of a batch is named `element_spec` (§ 6.21). § 7.5 lists the earlier decisions this reverses.

Each item keeps its options as the record of why.
The text above follows each decision and marks the place with the item's number.
A question raised later goes to `decisions.md`, and the text changes in the same commit.

#### R1. Which step adds `&` on `InputSpec` and `OutputSpec`

Step S2b adds them, as new operations with their tests.
The decisions of 2026-10-17 say that `&` replaces both of #571's `merge` methods, which were never built.
`&` on a spec is therefore new behaviour, not a rename, and it belongs in the behaviour step rather than the mechanical S3.
No public `merge` method exists for a single step.

#### R2. Whether `merge` on records goes when `&` arrives (replaced)

The original resolution kept `NamedTree.merge`'s deep rule under `&` in S3.
*Replaced by § 6.20, Q4(a):* step S2b makes `merge` shallow on `NamedTree`, `Record`, `RecordSpec`, and `RecordBatch`, so a top-level name in both raises, groups included, and step S3 replaces `merge` with `&` on all four, with no alias.
The library calls the deep merge only inside its own implementations (`core/record.py:948`, `core/_record_batch.py:792`), and 21 test lines call `merge`, of which 2 assert the deep rule (fact 5).

```python
# Today
RecordSpec(a=()).merge(RecordSpec(b=()))     # RecordSpec(a=(), b=())
Record({"g": {"c": 1.0}}).merge(Record({"g": {"e": 2.0}}))   # {g: {c, e}}
Record({"a": 1.0}) & Record({"b": 2.0})      # TypeError: unsupported operand type(s) for &

# Proposed
RecordSpec(a=()) & RecordSpec(b=())          # RecordSpec(a=(), b=())
Record({"g": {"c": 1.0}}) & Record({"g": {"e": 2.0}})        # ValueError (M3): g is an output of both
```

#### R3. The name a view below a law-valued field keeps (replaced)

The original resolution, from the decision of 2026-10-17, had such a view wrap, keeping the field's name, so `rm["G/mu"]` drew `{G: law["mu"]}`.
*Replaced by § 6.20, Q2(i):* no view wraps; from G3, `rm["G/mu"]` is the law whose draws are the views `law["mu"]` of the drawn laws, and `rm["G"]["mu"] == rm["G/mu"]` (§ 5.5).

#### R4. When `STYLE_GUIDE.md` §1.10 and §1.11 change

S0 does not change `STYLE_GUIDE.md`.
AGENTS.md says that the contributor documents describe the code as it stands, while `design/` describes the target.
So §1.10 changes with S1 for paths and with S3 for `rename`, and §1.11 changes with G5 (§ 6.5.2).

#### R5. G1's order relative to E5

G1 is split.
G1b changes how a lift declares its result in `functions/_broadcast.py`, the module where E5 puts `evaluate_at_points` into lifting, so G1b depends on E5.
G1a does not edit `_broadcast.py`, so it depends only on S2b and S4.

#### R6. Step assignments that README § Steps did not name

README § Steps names them:
- S1 makes `select` flatten to last names (§ 5.4) and adds `in` on laws (§ 4.6);
- G3 makes `[]` stop accepting the name of the output at the root, with the accessor `t.component(c)` that replaces it, spelled `t.output(c)` if the rename has merged (§ 4.2), and `OutputSpec[]`, new in S2b, reads that name as the spec itself until then, mirroring `Distribution.__getitem__` (§ 4.4, rule 4).

### 6.19 The original plan's open points, decided 2026-10-18

#### O1. `project` never wraps; a law's view adds the wrap (lapsed)

The original plan put the wrap of a view at a law-valued field into the view, not into `project`, so a function's output spec agreed with its return value (I13).
*Lapsed:* `project` is replaced by `OutputSpec[]`, which never wraps, and no view wraps either (§ 5.5), so I6 holds as written, with no wrap at any field.

#### O2. One naming rule for outputs whose one output is at the root

**Background.**
Three cases name the one output of an output spec at the root: a function that returns a batch of base elements; a function that returns a law whose event is at the root; and a type hole.
After S4, a base element's name is part of the batch's stored element spec, so filling a hole with a named batch replaces a stored name.

**Options.**
1. The function's name wins for batches: a function returning draws of `x` offers its own name, while a law keeps its own.
2. The returned term's names always win: a function names only a base value, and a type hole filled with any spec with parts raises.
3. **A function names what has no name of its own:** a base value, and a batch whose element carries only the default name `element`. A batch whose element was named by its maker keeps that name, and a law always keeps its own. A type hole filled with a raw batch names its element; filled with any other spec with parts, it raises.

```python
def draws(n):
    return sample(Normal("x", 0.0, 1.0), sample_shape=n)    # option 1: offers draws; options 2 and 3: offers x

def simulate(theta):
    return NumericArrayBatch(run_simulator(theta), "sim", label="sims")    # options 1 and 3: offers simulate; option 2: offers element
```

**Decision.** Option 3: one rule for every kind, a sensible name for a simulator that returns a raw batch, and it amends the decision of 2026-10-17 on type holes only where no named value is renamed. § 2.7, § 4.12, and § 4.13 follow it.

#### O3. How a batch's base-element name is stored

**Options.**
1. `BatchSpec` holds the name, but its equality and hash ignore it, and a batch's own flatten resets it to `element`. This leaks names through the static data of records that hold batches, and makes unsafe cache keys.
2. The name is stored on the batch beside its label. An output spec positional over a batch then has nowhere to store the name without a flag.
3. **`BatchSpec` holds the name and compares it in its own equality and hash;** a batch's equality and hashing, fingerprints, and every pytree flatten, recursively, use a name-free canonical spec.

**Decision.** Option 3 (§ 5.8). The name is stored in `BatchSpec.element_spec` (§ 6.21).

#### O4. Whether an executor restores the names a transform drops

**Options.**
1. The requester restores each input's output spec from the stored input structure of the evaluation points on the far side of every executor.
2. **Bodies run by an executor receive raw values only**, as the decisions already say, so no body sees names; a body that asks for a tracked batch waits for the rest of #506, after the workshop.

**Decision.** Option 2 before the workshop, with option 1 as the rule once #506 lets a body ask for tracked terms; the execution plan's §6.7 states the same.

#### O5. Views at fields that hold a batch or a kernel (lapsed)

The original plan extended the wrap of views at law-valued fields to fields that hold a batch or a kernel.
*Lapsed:* no view wraps; a view at any held field is positional from G3 (§ 5.5).

#### O6. `d[[name]]` on the output at the root

**Options.**
1. **The list form accepts the name of the output at the root and returns the wrap**, `OutputSpec(RecordSpec(x=...))`, so I9's record form exists at the root.
2. The list form refuses the name with `KeyError`, as the single form does from G3.

**Decision.** Option 1, as an extra rule at the top level, read against the record of outputs (§ 4.4, rule 3). It keeps today's tuple result, and the name is never ambiguous under grammar B (`no_ambiguous_key_B`).

#### O7. `&` on records and on arrays

**Background.**
`&` on two `Record`s, `NumericRecord`s included, merges their fields, while `&` on two `NumericArray`s is NumPy's elementwise "and", which refuses float arrays.
III.5 gives a `NumericRecord` the vector-space operators, `+` and `-` between records of one schema, scalar `*` and `/`, and `map`, and `&` is not among them, so merging takes no elementwise meaning away from records (decided 2026-10-18).

**Options.**
1. **Keep both meanings, and state the difference as an exception** (§ 6.3).
2. Restrict `&` on records to `Record` and `RecordSpec`, and refuse it on `NumericRecord`.

**Decision.** Option 1. Two records merge by top-level name (§ 5.4).

#### O8. How `BatchSpec` takes an element spec

**Options.**
1. **The first positional argument takes an `OutputSpec` or a bare `TermSpec`;** a bare spec gets the raw default of § 5.8.
2. A keyword, `BatchSpec(element_spec, element=OutputSpec(...), **levels)`.
3. No constructor form; an element spec is set with a method.

**Decision.** Option 1 (§ 6.1). It mirrors how `OutputSpec` and `InputSpec` read their sources, and adds no keyword that could collide with a level name.

#### O9. Optional inputs of `Function.input_spec`

**Options.**
1. **`Function` takes the kernel's rule:** the parameters with defaults are the optional inputs, and an optional input without a default raises `ValueError`.
2. `Function` keeps reading optional inputs from its `input_spec` alone.

**Decision.** Option 1, so the one reading of inputs has one rule for optional inputs (§ 2.6).

#### O10. An empty `RecordSpec` in an output spec

**Options.**
1. **Refuse an empty `RecordSpec` in `OutputSpec`,** as `without` refuses to remove every node.
2. Treat it as a positional record with no outputs.
3. Treat it as a base spec, nameable in the keyword form.

**Decision.** Option 1. A term that produces nothing has no use in a merge. An `InputSpec` with no inputs stays valid, since a term may take no input.

#### O11. Whether `held` and `with_held` are public

jhuggins's reply gives every term spec two methods, `produced_declaration()` and `with_produced_declaration(declaration)`, which this plan writes `held` and `with_held` (§ 2.8).

**Options.**
1. Public, so a new kind of term spec can take part in the rules, and `held(spec) is None` tests for a base spec.
2. **Private**, since users call `[]`, `without`, `&`, and `output`, and a new kind of term spec is a contributor's change.

**Decision.** Option 2. A user who needs to test for a base spec checks `isinstance(spec, (NumericArraySpec, OpaqueSpec))`.

#### O12. Joint labels and notation keep `·`

**Options.**
1. **Keep `·`**, which states the product of densities whatever the operator is written as.
2. Join with `&`, as `lik & prior`.

**Decision.** Option 1 (§ 2.19, R6). The label is mathematics for a reader, and `p(y | mu)·p(mu)` is the density of the joint.

#### O13. `with_level_names` and `with_dim_names`

**Options.**
1. **Leave them unchanged**, and record the idea of `rename_levels` and `rename_dims` in `deferred.md`.
2. Rename them in S3, with `rename`.

**Decision.** Option 1. No decision covers them, and S3 is already the widest PR of the push.

#### O14. `component` and a mixture's component laws (settled)

A `MixtureDistribution` takes its component laws as `components=`, while the accessor reads the term at an output.
The original decision kept both senses for now, with "component law" in the glossary and the docstring, the accessor refusing an integer with a message that points to the mixture's laws, and the rename of the mixture's parameter recorded in `deferred.md`.
*Settled by § 6.20, Q1:* until the rename after the workshop, the original decision stands for `t.component(c)`; after it, the accessor is `t.output(c)`, and "component" keeps only the sense of a mixture's component laws, so the collision is gone.

#### O15. The senses of "element", and the attribute's name (replaced)

The original decision named II.5's attribute `BatchSpec.element_declaration`, beside `element_spec`, which named the element's type.
*Replaced by § 6.21:* the attribute is `BatchSpec.element_spec`, an output spec, and the element's type is `element_spec.term_spec`.

#### O16. G2's refusal needs G3

Until G3, a view at a field that holds a law, a kernel, a function, or a batch keeps today's interim whole form, such as `joint["G"]` declared `OutputSpec(G=DistributionSpec(...))` (§ 5.5).
G2 ends by refusing the interim whole form, so it must follow G3; README § Steps makes G2 depend on G1b and G3.

### 6.20 The inputs and outputs questions, decided 2026-10-18 (provisional)

Each question below was settled on 2026-10-18 with the inputs and outputs plan's recommendation, provisional pending the maintainer's final read of the inputs and outputs plan; `inputs-and-outputs.md` § 14 gives each in full.

| Question | Options | Decision |
| --- | --- | --- |
| Q1. The words "input" and "output", and when they change | 1 all in S2a; 2 all after the workshop, in one PR; 3 keep "slot" and "component" | 2 |
| Q2. (i) The event rule and the view wrap; (ii) the outputs of a law-valued produced term | (i) drop or keep; (ii) none or marginals | (i) drop; (ii) none |
| Q3. (i) `OutputSpec` indexing; (ii) `InputSpec &`'s operands | (i) by paths with `outputs`, the mapping with `project`, or a mapping with `project` renamed; (ii) two input specs, or any source | (i) by paths; (ii) two input specs |
| Q4. Merge rules: (a) shallow records; (b) optional inputs across merges; (c) functions feed; (d) the spec of a shared input | (a) shallow, mixed, or deep; (b) today's factors, IV.2 strictly, or visible optional inputs; (c) feed or refuse; (d) `U` after the workshop or in S2b | (a) shallow, with batches of arrays merging as records; (b) today's factors for the push, visible optional inputs after; (c) feed; (d) after |
| Q5. The output at the root | 1 `[]` paths only from G3, with `t.output(c)`; 2 `[]` reads the name permanently; 3 no accessor, `t[[c]]` | 1 |
| Q6. Dimension scopes of merge results, after the workshop | store a scope on every merge result, or substitute and forget everywhere | decide after the workshop, leaning to store |

#### Q1. The words, and when they change

"Slot" and "component" are not parallel, "slot" is a technical word, and "component" also names a mixture's component laws; "input" and "output" are plain and parallel, and already spelled in `InputSpec`, `OutputSpec`, `input_spec`, and `output_spec`.
The design does not depend on the words; the question is when they change, since directive 7 requires a PR that changes a name to change every use, and directive 11 requires the code to use the glossary's words.
The rename covers, at `6f8859fc`: `.components` on 151 lines of `probpipe/` and 347 of `tests/`; `component=` with 493 uses in 81 files; "component" on 1,653 lines of `probpipe/` and "slot" on 806, each a judgement by sense; 66 lines with "slot" and 149 with "component" in `design/`; and 5 User Guide notebooks (fact 22).

**Options.**
1. The full rename in S2a, before the workshop; S2a, on both critical chains, then carries about 4,500 lines of judgement days before the freeze.
2. **Defer the words everywhere to one announced PR right after the workshop**, `design/` and the glossary included, with S0 writing this plan's rules in today's words by the mapping of § 8.2. The cost is one more breaking rename after the workshop, with a CHANGELOG entry.
3. Keep "slot" and "component" permanently, with the mapping of option 2 as the final words.

**Decision.** Option 2. It satisfies directives 7 and 11 at every merge, keeps the critical path free of line-by-line judgement, and still adopts the plain words.
It renames three decided names after the workshop: `t.component(c)` becomes `t.output(c)`, `component=` becomes `output_name=`, and `components` becomes `outputs`.

#### Q2. Law-valued produced terms

A function can return a law, as `fit` does, and a law can draw laws, as a random measure does.
The original plan offered such a produced term's marginals as outputs, typed by its views, refused at construction a law whose event offers a law's parts, and wrapped a view at a law-valued field under the field's name.
M2 refuses in a merge every operand whose produced term is a law, whatever outputs it offers (§ 2.21), so neither the outputs rule nor the event rule protects merges.

**(i) The event rule and the view wrap.** Options: drop both, or keep both.
Dropping them makes a law that draws laws legal, makes a view at a held law positional, so I2, I5, I6, and I9 hold at held fields with no exception from G3, lets O1, O5, and R3 lapse, and leaves one `OutputSpec.default` from G2.
Its cost: a law of laws can be named only by a list, as `rm[["G"]]`, or by the code that builds it wrapping its draws.
**Decision:** drop both (§ 5.5).

**(ii) The outputs of a law-valued produced term.** Options: none, or the marginals, typed by the views.
Offering none keeps `InputSpec(o)` from typing inputs as marginal laws and `o.outputs` from listing marginals that cannot be passed on separately (D1).
Its cost: `fit.output("mu")` and `fit.output_spec.outputs["mu"]` are gone, while `fit["mu"]` still reads the function returning the view; three tests that assert those outputs change (fact 21).
**Decision:** none (§ 2.11, § 5.7).

```python
# The original plan                              # This plan
list(fit.output_spec)         # [mu, tau]         fit.output_spec.outputs      # {}
rm["G"].event_spec            # {G: law}          rm["G"].event_spec           # OutputSpec(law_spec): a law of laws
sample(rm["G"])               # Record {G: law}   sample(rm["G"])              # a law, distributed as sample(rm)["G"]
rm[["G"]].event_spec          # {G: law}          rm[["G"]].event_spec         # {G: law}
```

#### Q3. The operations on the two specs

**(i) How an output spec is indexed.**
The original plan made `OutputSpec` a `Mapping` of output names, so `decl[c]` read an output's spec, and added `project(s)` for the output spec at paths.
An output spec was then the one object whose `[]` did not take paths: `d[s].event_spec == d.event_spec.project(s)`, and `"g/c" in grouped.event_spec` was false while `"g/c" in grouped` was true; and the maintainer disliked the name `project`.
Options: (1) `O[s]` takes paths, as on every term, and an attribute maps names to specs; (2) the `Mapping` with `project`; (3) a `Mapping` with `project` renamed, such as `O.part(s)`.
**Decision:** option 1 (§ 5.11, § 4.4): `[]` keeps one meaning, I6 reads as written, `in` agrees with `[]`, and no new method name is needed.

```python
# Option 1 (this plan)                        # Option 2 (the original plan)
grouped.event_spec["g/c"]                     grouped.event_spec.project("g/c")
grouped.event_spec.outputs["g"]               grouped.event_spec["g"]
"g/c" in grouped.event_spec   # True          "g/c" in grouped.event_spec   # False
```

**(ii) The operands of `InputSpec &`.**
The original plan let either operand be any of the four sources, so `o & i` merged an output spec, read as inputs, with an input spec.
Options: (1) both operands are `InputSpec`s, and `InputSpec(o) & i` states the conversion; (2) any source.
**Decision:** option 1 (§ 2.18): each side's `&` stays within its side, and `o & p` keeps one meaning, the merge of two output specs.

#### Q4. Merge rules

**(a) Shallow `&` on records, and its timing.**
Today `merge` on records is deep: `{g: {c}}` and `{g: {e}}` merge into `{g: {c, e}}` (fact 5), while a merge of laws is shallow (fact 6).
Options: (1) shallow, delivered on `NamedTree.merge`, `Record.merge`, and `RecordBatch.merge` in S2b, so that S3 stays a rename; (2) deep for records and record specs, shallow for laws and output specs, so `&` means two things and `OutputSpec(R1) & OutputSpec(R2)` differs from `OutputSpec(R1 & R2)`; (3) deep for both, which changes what a factor graph is.
The same question covers batches of base values, which merge as records, so two batches of arrays give a batch of records, where the original plan had them raise or combine bitwise.
**Decision:** option 1, with batches of arrays merging as records (§ 2.16, § 2.21; R2).

**(b) Optional inputs across merges.**
`lik_opt & prior` is a law, so it has no inputs (III.7), yet its factor `lik_opt` still reads `s` at its default.
Options: (1) today's factors: a merge reads its operands' factors, and the unfed inputs of the factors, optional ones included, can be fed or shared by a later merge, with § 4.16's consequences; (2) IV.2 read strictly: an unfed optional input takes its default and leaves, which breaks associativity for optional inputs; (3) visible optional inputs: a merge that leaves optional inputs is a term that shows them, as a kernel all of whose inputs are optional, with `sample` and `log_prob` at their defaults.
**Decision:** option 1 for the push, since it is today's behaviour and needs no code; option 3 as the long-term design, to decide after the workshop (`deferred.md`).

**(c) Functions feed.**
The original plan had functions "pair and never wire", with no reason given.
Options: (1) functions feed, as point-mass kernels, after the workshop; (2) a fed name raises for functions.
**Decision:** option 1, by D2: the function rule is the kernel rule, read deterministically (§ 5.7).

**(d) The spec of a shared input.**
Today the joint keeps the first operand's spec for a shared input; validity is unaffected, since every consuming factor checks a bound value (§ 2.14).
Options: (1) keep today's choice in S2b's new `InputSpec &`, and switch to `U` after the workshop; (2) implement `U` in S2b, which changes one test (fact 35).
**Decision:** option 1, to keep S2b to the original plan's content (`deferred.md`).

#### Q5. The output at the root

From step G3, `[]` takes paths only, so `Normal("x", 0.0, 1.0)["x"]`, which returns the law today, raises, and an accessor reads the term at an output.
Under grammar B the name of an output at the root never collides with a path (`no_ambiguous_key_B`).

**Options.**
1. **`[]` takes paths only from G3; `t.output(c)` on terms reads the term at an output**, and output specs get no accessor, since `O.outputs[c]` and `O[c]` cover them. Away from the root, `t.output(c)` and `t[c]` are two spellings of one term.
2. `[]` takes paths and the name of the output at the root, permanently. For a record `{n, G}` whose field `G` holds `Normal("x", 0.0, 1.0)`, `t["G"]["x"]` would work while `t["G/x"]` raises, so I5 gains an exception, `in` mixes names with paths, and `[]` means two things.
3. No accessor at all; generic code writes `t[[c]]`, which gives the term whose produced term is the record `{c: ...}`, so code that wants the law `Normal("x", 0.0, 1.0)` back from its output `x` has no way to ask for it.

**Decision.** Option 1 (§ 4.2, § 4.11): `[]` keeps one meaning and I5 keeps no exception; the price is one plain method on terms and the G3 break already scheduled.

#### Q6. Dimension scopes of merge results, after the workshop

Today a merged law or kernel stores its dimension scope, and this plan keeps that; a merged input spec or function forgets its bindings (§ 4.17).
Storing is what makes laws and kernels associative (fact 36); it also lets a merge fail on a name no spec shows, and `with_dim_names` cannot rename the stored name today.

**Options.**
1. Store a scope on every merge result, input specs and functions included, as private state that equality ignores, and let `with_dim_names` rename stored names too.
2. Substitute and forget everywhere, laws and kernels included, which loses associativity for laws and kernels.

**Decision.** Decide after the workshop, leaning to option 1, with `with_dim_names` renaming stored names too; nothing before the workshop needs either (`deferred.md`).

### 6.21 `element_spec` (decided 2026-10-18, provisional)

**Background.**
A batch needs one output spec for its elements, to hold a base element's name (§ 5.8).
The original plan named it `element_declaration` (O15), beside today's `element_spec`, which holds the element's term spec; the inputs and outputs plan renamed it `element_output_spec`.

**Decision.**
The elements' output spec is named `element_spec`, since it is an output spec, as `event_spec` is: `event_spec` describes one draw, and `element_spec` one element.
Today's `BatchSpec.element_spec`, the element's term spec, becomes `element_spec.term_spec`, as `event_spec.term_spec` is a draw's term spec.
Step S4 changes every use in one PR (directive 7), on `BatchSpec` and on `Batch`; the name involves no word of § 8.2, so S4 introduces it directly, with no interim name.

**How this plan defines it.**
`element_spec` is the output spec of one element as a term, `held(bt.spec)` (§ 2.8, § 2.9): the keyword form under a base element's name, and the positional form over every other element's term spec.
This makes `element_spec.term_spec` equal today's `element_spec` for every kind of element, which is what lets S4 change each use mechanically; for a batch of laws today's `element_spec` is the laws' `DistributionSpec`.
The inputs and outputs plan defined `element_output_spec` as each element's own output spec, which for a batch of laws is the laws' `event_spec`, whose `term_spec` is a draw's spec, not the law's.
For a batch of values the two definitions agree; for a batch of laws, kernels, or functions, each element's own output spec is the one its term spec holds, `element_spec.term_spec.event_spec` or `element_spec.term_spec.output_spec`, which III.10's `DistributionBatch.event_spec` already exposes.
`inputs-and-outputs.md` was revised to the same definition, so the two texts agree.

### 6.22 Alternatives considered

Items 1 to 17 are the original plan's; items 3, 10, 11, 15, and 17 are superseded by the inputs and outputs merge, each with its reason.
Items 18 to 26 are the alternatives the inputs and outputs plan weighed; its Appendix D lists every choice its three source designs made.

1. **Grammar B′.** It would name from outside any spec with no paths below its root, and keep a few names such as a Gaussian process's `f`. It needs a flag on every output spec, since the term spec would no longer decide the form. Rejected (decided 2026-10-17).
2. **An `exposes` property**, with #571's four values `"whole"`, `"record"`, `"event"`, and `"element"`, or jhuggins's boolean `exposes_parts`. Under grammar B the term spec decides the form, so the property would only restate it (gaps D1 row a).
3. **Narrowing the law's event rule** to allow positional laws and kernels with two or more outputs. *Superseded:* the event rule itself is removed (§ 5.5; § 6.20, Q2(i)), so there is nothing to narrow; a produced law offers no outputs, and M2 refuses it in a merge.
4. **ProbPipe wraps a returned value** under a declared name, so that a body may return the law alone. It converts values behind the caller's back; the body builds the record instead (decided 2026-10-17).
5. **Names in a batch's equality and pytree**, as #571 proposed. Draws of `x` and `y` would then be different structures to JAX, so `jit` would recompile per name and `lax.cond` could not return either (decided 2026-10-17).
6. **`components` returns a `RecordSpec`.** A `RecordSpec` indexes leaf paths, so `components["growth"]` would raise for an output that is a record, and every reader of `components` would change.
7. **`RecordSpec` replaces `InputSpec`.** A record has no optional fields and no rule for identifier names, and its merge raises on a shared name, where inputs unify.
8. **A kernel's input type stays pending until a merge fills it.** `conditional_distribution` evaluates the function at stand-ins of each input's type to read its event spec and its claims (IV.4), and III.7 requires a complete event spec, so a kernel with a pending input would have no event spec.
9. **`given_spec` holds specs for more names than the function has parameters.** This removes the need for `without`, and it accepts a misspelled name without an error.
10. **A law's event offers the outputs of laws, and each operation keeps what the output spec offers.** *Superseded:* the failure it caused, a joint's field typed by the inner law's event (§ 5.5), cannot arise once a produced law offers no outputs and M2 refuses it in a merge.
11. **`produced slot` and `consumed slot`** in place of component and slot. *Superseded:* "input" and "output" are the final words (§ 6.4; § 6.20, Q1).
12. **Keep `*`, and give the specs a `merge` method**, or support both operators for a release. `*` is arithmetic on values, and a release with both contradicts the clean-break rule (gaps D2, D5).
13. **Other names for `with_path_names`:** `with_names` does not say whose names, `with_field_names` is wrong because it also renames groups and the output at the root, and `relabel` collides with "label".
14. **An error for an ambiguous rename key**, which the grammar-A plan needed. Under grammar B no key is ambiguous (`no_ambiguous_key_B`).
15. **`project` wraps at a field that holds a law or a function**, as a law's view did. *Superseded:* `project` is replaced by `OutputSpec[]`, and no view wraps (§ 5.5), so `O[s]` and `d[s].event_spec` agree with no wrap (I6).
16. **A batch's base-element name stored beside the spec, as a label is**, which the addressing plan proposed. An output spec positional over a batch would have nowhere to hold the name without a flag (§ 5.8, O3).
17. **One `OutputSpec.default` for outputs and events.** *Superseded, and adopted:* the original plan kept two because a law's event with parts had to wrap under the event rule; with the rule removed, one default serves both from G2, and the private event default goes (§ 4.12).
18. **Keep "slot" and "component" permanently.** § 6.20, Q1, option 3.
19. **`[]` keeps reading the name of the output at the root permanently.** § 6.20, Q5, option 2: it gives I5 an exception and `[]` two meanings.
20. **`OutputSpec` as a mapping of names, with `project`**, or with `project` renamed. § 6.20, Q3(i), options 2 and 3.
21. **`InputSpec &` takes any of the four sources.** § 6.20, Q3(ii), option 2.
22. **Deep merge on every kind.** § 6.20, Q4(a), option 3: a factor would then produce part of an output, which IV.1 forbids.
23. **One attribute pair, `input_spec` and `output_spec`, on every kind.** It renames `event_spec` on 885 lines and `given_spec` on 210 lines of `probpipe/`, and drops a distinction III.9 states (§ 2.9).
24. **Merge functions with type holes, checking the output specs at the first call.** The merged output spec would be a record of pending children, which `RecordSpec` cannot hold (fact 28).
25. **A merge that sorts the flattened factors topologically**, so a kernel can be added between two merged operands (fact 29). It changes IV.2's conditional-first rule; `deferred.md` records it.
26. **`element_output_spec`, or `BatchSpec.output_spec`, for the elements' output spec.** The first is long beside `event_spec`; the second reads as the batch's own output spec. `element_spec` says what it describes, one element, as `event_spec` describes one draw (§ 6.21).

## 7. Relation to previous plans

This section records where the plan's rules came from, and what it changed in the plans and decisions it replaces.
None of it adds a rule; sections 2 to 6 state every rule.

### 7.1 Grammars A, B, and B′

The one grammar question is when a value may be named from outside, as `OutputSpec(name=T)` names a term of spec `T`:
- **Grammar A**, today's code, names any term spec from outside, so a structured term has a name outside it and paths inside it. It survives only as the interim whole form of § 8.1, until G2 refuses it.
- **Grammar B**, from jhuggins's component interface doc, names from outside only a base value (§ 2.7). This plan adopts it (decided 2026-10-17).
- **Grammar B′** would also name from outside any spec with no paths below its root. It was rejected on 2026-10-17, for the reason below.

jhuggins's doc compares B with a close alternative, B′, which would also name from outside any spec with no paths below its root: a batch of arrays, a law over a single array, or a random function whose output is one array.
B′ keeps a few more names, such as a Gaussian process's own name `f`.
It pays with a flag on every output spec, since the term spec would no longer fix the form.
ProbPipe takes B: an output spec is in the keyword form if and only if its term spec is a base spec, and no flag records it.

Under grammar A, an interim whole form over structure moved under its name in a merge, so every one of its paths changed: `params_prior["beta"]` became `joint["params/beta"]`.
Under grammar B that cannot happen, because only a base value is named from outside, and a base value has no paths below the root.

### 7.2 From #571's body and jhuggins's reply

| Topic | #571's body | jhuggins's reply | This plan |
| --- | --- | --- | --- |
| Whether an output spec names its value whole | a property `exposes` with four values | a boolean `exposes_parts` | no property: the keyword form if and only if the term spec is a base spec (§ 2.7) |
| What the keyword form may name | any spec | any spec | a base spec only; a structured output is positional or wrapped (§ 2.7) |
| `OutputSpec`'s interface | `components`, `exposes`, `term_spec` | a `Mapping`; `components`, `spec`, `exposes_record` removed | indexed by path, with `outputs` and `term_spec`; not a `Mapping` (§ 5.11) |
| `project` with several paths | each node under its last name, in the order given; a shared last name raises | as the body | `O[ps]`: each node at its own path, in the tree's order; a path inside another raises (§ 4.4) |
| `project` with one path to a group | the group, whole under its last name | as the body | `O["g"]`: the group's parts, positionally; `O[["g"]]` keeps the name (§ 4.4) |
| The paths of a whole declaration | the name, then the paths inside its term, as `c/beta` | as the body | the keyword form has none; an interim whole form keeps today's until G2 (§ 2.7) |
| `without` on a whole declaration | removes nodes inside its term | as the body | nothing to remove in the keyword form (§ 4.5) |
| Reading a component | five cases by form | a private `_component` per kind | one rule, the public `t.output(c)` (§ 4.11) |
| The spec of an output of a positional law | one draw's value at `c` | the view, without restating the spec | none: a produced law offers no outputs (§ 2.11) |
| `batch["mu"]` for draws of `Normal("mu", ...)` | returns the batch | as the body | raises; `draws.output("mu")` returns the batch (§ 5.8) |
| A batch built from raw values | a bare element spec, no outputs | as the body | an element spec; a base element is named `element` (§ 5.8) |
| Equality of batches that differ only in names | unequal | as the body | equal; names out of equality, hashing, the pytree, and fingerprints (§ 5.8) |
| Merging | `OutputSpec.merge`, `InputSpec.merge` | `&` replaces `*` and both `merge` methods | `&`, with one definition (§ 2.15) |
| `&` on an operand that offers a law or a batch | not covered | raises | raises (M2); such outputs are wrapped to merge (§ 2.17) |
| The column of a group, `grouped_draws["g"]` | not covered | as the body | stores the positional record `{c, e}` (§ 5.8) |
| A function's output | cannot be positional | positional through `FunctionSpec` | positional, in G1b; a returned function offers its outputs (§ 2.11) |
| `with_spec` | open question | `with_term_spec` | `with_term_spec` (§ 5.11) |
| Aliases | open question | none | none (§ 6.6) |
| The tutorials | use the new operations | — | out of scope; the User Guide is updated (§ 6.5.3) |
| `InputSpec.with_path_names` | — | proposed | `InputSpec.rename`, after the workshop (§ 5.10) |
| `&` on functions, batches, and specs | — | proposed | design, after the workshop; functions feed (§ 6.16) |
| IV.2's rationale | — | — | associativity holds with or without `G_B ∩ F_A = ∅`; the rule keeps the joint well formed (§ 2.21) |

The changes from #571, by section of this plan:

- § 4.4, `[]` on an output spec: #571's `project` flattened each node to its last name, in the order given, and refused two paths with the same last name. `project("g")` gave the group under the name `g`. Grammar B keeps paths, uses the tree's order, and offers a group's parts (gaps D1 rows b and c).

- § 4.5, `without`: #571 let a whole declaration remove nodes inside its term. The keyword form names a base value, so nothing is below it (gaps D1 row m).

- § 4.11, the term at an output: #571 read a component by five cases, one per form, and jhuggins's reply gave each kind a private `_component`. This plan states one rule and makes the accessor public (gaps D1 row d).

- § 5.8, the element spec: #571 kept a bare element spec for a batch built from raw values, so a batch of arrays had no outputs and could not be positional. This plan gives every batch an element spec, and a raw batch of arrays offers `element` (gaps D1 row g). #571 also said `batch["mu"]` returns the batch; that is now `draws.output("mu")` (row e).

- § 5.8, names out of a batch's equality: this reverses #571's sentence that two batches whose elements differ only in output names are not equal (gaps D1 row h, D3).

- § 5.11, `OutputSpec`: `O[s]` answers #571's open question 2: the output spec of a part is `O[s]`.

### 7.3 From the grammar-B addressing plan

| Topic | Addressing plan | This plan |
| --- | --- | --- |
| Status of grammar B | adopted, with jhuggins's ruling between B and B′ open | B, decided 2026-10-17 (§ 2.7) |
| Who builds the record of a wrapped output | open; leaning to the body | the body returns the record, decided (§ 5.7) |
| A type hole filled with a spec with parts | open; leaning to raise | raises, in G2; a batch whose element has only the default name is named by the hole (O2) |
| A view at or below a law- or function-valued field | open: wrap, or narrow #571's event rule | neither: no view wraps and the event rule is removed; positional from G3 (§ 5.5; § 6.20, Q2) |
| The accessor | `t.component(name)`, with the name a placeholder | `t.output(c)`, spelled `t.component(c)` until the rename (§ 4.11) |
| `rename` and `MarginalView` | recommended | decided, in S3 (§ 4.8, § 5.5) |
| Batch names out of equality and the pytree | recommended; the name "stored beside the spec, as a label is" | decided; the name is stored in `BatchSpec.element_spec`, and a name-free canonical spec serves equality, hashing, fingerprints, and every flatten (§ 5.8, O3) |
| Clean break | open | decided (§ 6.6) |
| When kernels get `[]` | open | step G5 (§ 5.6) |
| Its scope | addressing only; #571 owned the reading of inputs, `merge`, `term_spec`, and the vocabulary | both, in one text (§ 1.3) |
| The implementation sequence | six steps, tentative | S0, S1, S2a, S2b, S3, S4, G1a, and G1b before the workshop, G2–G5 after; README § Steps owns the order (§ 6.7) |
| The workshop split | not covered | G1a and G1b before, G2 after, with the refusal of the interim whole form last (§ 6.8) |
| Defaults | one rule by kind | one `OutputSpec.default`, from G1b for outputs and G2 for events (§ 4.12) |
| Membership and `[]` on the name of the output at the root | the name is not a path | the same, from G3; until then the old reading stays (§ 2.7) |
| A bare `output_spec` | not covered | read by `OutputSpec.default` (§ 4.13, gaps T4) |
| IV.2's rationale | a note for #571 | the fix of § 2.21 (gaps T5) |
| "Selection" in the glossary | both senses removed | only *selection order* stays (gaps T3) |
| `mean(prior)["mu"]` in "The problem today" | `KeyError` | works since #576 merged; the table shows `mean(grouped)["g"]` instead (§ 1.2) |
| "Where this plan meets #571" | twelve shared points, each with an owner | gone: one text owns both (§ 7.2 records the outcomes) |

**Departure from the addressing plan.**
The addressing plan stored the name "beside the spec, as a label is".
This plan stores it in `BatchSpec`, because an output spec needs it: `OutputSpec(BatchSpec(...))` for a batch whose element is named `x` offers `x`, and an `OutputSpec` cannot hold a name outside its term spec without a flag, which grammar B rules out.
The decision's four exclusions still hold, through the canonical form.

### 7.4 From the inputs and outputs proposal

| `inputs-and-outputs.md` | This plan |
| --- | --- |
| § 1 Summary, § 2 the maintainer's questions | the header, § 1.1, and § 6.20 |
| § 3 Decisions reversed | § 7.5 |
| § 4.1 Concepts, and the path reader | § 2.1, § 4.1 |
| § 4.2 Running examples | § 2.2 |
| § 5.1–5.13 Definitions | § 2.3 to § 2.14, and § 2.23 |
| § 6 Every kind placed | § 5.1 |
| § 7.1 `[]` on terms, § 7.2 views at held fields | § 4.2, § 5.5 |
| § 7.3 `[]` on an output spec | § 4.4 |
| § 7.4 `[]` on an input spec, its sources, the shorthand | § 5.10, § 4.9 |
| § 7.5 `without` | § 5.10, § 4.5 |
| § 7.6 Membership, § 7.7 the term at an output, § 7.8 symbolic dimensions | § 4.6, § 4.11, § 4.10 |
| § 8 Merge | § 2.15 to § 2.22, § 4.15 to § 4.17 |
| § 9 Invariants | § 3 |
| § 10 API sketch | § 6.1 |
| § 11 Worked examples | § 6.2 |
| § 12.1, § 12.2 Differences and renames | applied throughout; § 6.4 |
| § 12.3 The word "unify" | § 6.5.4 |
| § 13 Effect on the push | § 6.7, § 6.11 |
| § 14 Open questions | § 6.20 |
| Appendix A Proofs | cited from § 4.15 |
| Appendix B Deliberate exceptions | § 6.3 |
| Appendix C Edits to the plan | applied; C.3 in § 6.4.1, C.4 in `deferred.md` |
| Appendix D Alternatives | § 6.22, items 18–26 |
| Appendix E Facts | cited as "fact N" |

**The judgement calls of the merge.**

The merge of the inputs and outputs plan into this text made four calls of its own, which the maintainer's acceptance covers with the rest:
1. `element_spec` is the output spec of one element as a term, so `element_spec.term_spec` equals today's `element_spec` for every kind of element, a batch of laws included; the inputs and outputs plan's `element_output_spec` was each element's own output spec, which for a batch of laws is the laws' event spec (§ 6.21).
2. Conformance and unification of batches compare the elements' term specs, not a base element's name (§ 2.14, rule 7 of each).
3. The interim-form probe gained a row for G3, a view at a held field, beside the rows of G1b and G2 (§ 6.14).
4. The worked examples became § 6.2 and the appendix § 7; the inputs and outputs plan's proofs, alternatives, and facts are cited from it, not copied.

### 7.5 Differences from `declarations-and-addressing.md`, and the decisions reversed

The original plan, `declarations-and-addressing.md`, stays in the repository for recovery.
This plan differs from it where the inputs and outputs plan won; the table lists every decision that reverses.

| # | Decision | Date and source | This plan | Question |
| --- | --- | --- | --- | --- |
| 1 | The words "slot" and "component", "input declaration" and "output declaration", and the glossary terms producer, consumer, exposes, whole declaration, exposed declaration, address, and component spec | the original plan's § 4.3, § 4.4, and § 17, accepted 2026-10-18 | input, output, input spec, output spec, after the rename; none of the glossary terms (§ 6.4) | Q1 |
| 2 | The accessor `t.component(c)` | `decisions.md`, 2026-10-17 | `t.output(c)` on terms, after the rename; no accessor on output specs (§ 4.11) | Q1, Q5 |
| 3 | `BatchSpec.element_declaration` (O15), and `Batch.components` (II.5; the original plan's § 15.2 and S4) | the original plan's O15 and S4, accepted 2026-10-18; II.5 | `BatchSpec.element_spec`, an output spec, whose `term_spec` is today's `element_spec`; `element_spec.outputs` (§ 5.8) | Q1; § 6.21 |
| 4 | The outputs of a produced term that is a law: its event's outputs, typed by the views | the original plan's § 4.4 and § 5.3, accepted 2026-10-18 | none (§ 2.11) | Q2(ii) |
| 5 | The law's event rule: "#571's rule that a law's event is whole or an exposed record stands unchanged" | `decisions.md`, 2026-10-17; the original plan's § 5.4 | removed (§ 5.5) | Q2(i) |
| 6 | "A view at or below a field that holds a law or a function wraps, keeping the field's name" | `decisions.md`, 2026-10-17; the original plan's § 5.3, R3, and O5 | no view wraps (§ 5.5) | Q2(i) |
| 7 | Two defaults, an output default and a private event default that wraps (the original plan's alternative 17 rejected one default) | the original plan's § 5.5 and § 20, accepted 2026-10-18 | one `OutputSpec.default`; the private event default keeps today's rule until G2 and then goes (§ 4.12) | Q2(i) |
| 8 | `OutputSpec` is a `Mapping` of output names, with `project`; `components` is removed | the original plan's § 7.3 and § 7.5, and README's S2a, accepted 2026-10-18 | indexed by paths; the attribute `outputs`, spelled `components` until the rename (§ 5.11) | Q3(i) |
| 9 | Either operand of `InputSpec &` may be any of the four sources | the original plan's § 6.4, accepted 2026-10-18 | both operands are `InputSpec`s (§ 2.18) | Q3(ii) |
| 10 | `&` on records keeps `NamedTree.merge`'s deep rule | the original plan's R2, accepted 2026-10-18 | shallow (§ 2.21) | Q4(a) |
| 11 | `&` on records arrives in the mechanical S3, and `&` on batches after the workshop | the original plan's § 19.5 and § 14.4, accepted 2026-10-18 | S2b makes `merge` shallow on records and on `RecordBatch`, and S3 renames it `&` | Q4(a) |
| 12 | Functions "pair and never wire" | the original plan's § 14.2, rule 3, accepted 2026-10-18 | functions feed, after the workshop (§ 5.7) | Q4(c) |
| 13 | Two batches of arrays raise or combine bitwise under `&` | the original plan's § 16, accepted 2026-10-18 | they merge into a batch of records, element by element, after the workshop (§ 2.16) | Q4(a) |

**Kept, with a restated reason.**
The lifted law-returning function keeps the original plan's G1b done-condition: its draws are `{law_of_y: law}`, under the callable's output name, because the lift builds its own law and names its own record (C5), as `include_inputs=True` already does today (§ 4.12).
The decision of 2026-10-17, "a function with a wrapped output returns the record itself … ProbPipe never wraps a returned value", is about a function body's return value, and it stands.

**Kept, unchanged.**
O9: `Function` reads optional inputs by the kernel's rule, and a source that marks optional an input whose parameter has no default raises `ValueError`.
III.7 and III.9: a law has no inputs, and a kernel has at least one; binding a kernel's last required input gives a law whose optional inputs take their defaults (IV.4).
II.1: dimension names bind in one scope, and two different names at one axis raise.
IV.2's flattening: a merge of laws and kernels reads its operands' factors (§ 4.16).
Every other accepted item of `decisions.md` and of § 6.18 and § 6.19 stands, apart from the rows above and the items marked lapsed, settled, or replaced (O1, O5, O14, O15, R2, R3).

**The original plan's rules that this plan drops.**

- **The law's event rule.** The original plan added a rule, decided 2026-10-17, that a law's event is a base value in the keyword form or a positional record, and refused at construction a law's event spec that is positional over a law, a kernel, a function, or a batch.
That rule is removed (§ 6.20, Q2(i)): a produced term that is a law or a kernel offers no outputs (§ 2.11), and a merge refuses any operand whose produced term a record could hold only under a name (§ 2.17, M2), so neither needs the rule.

- **The wrap of views at held fields.** The original plan wrapped these views under the field's name from G3, because its law's event rule forbade a law whose event spec is positional over a law (decided 2026-10-17; its R3 and O5).
Here a produced term that is a law offers no outputs, and M2 refuses it in a merge, so neither the event rule nor the wrap is needed, and that decision is reversed (row 6 above).

- **`project`.** `O[s]` replaces the original plan's `project`, which the maintainer found poorly named (§ 5.11, § 4.4).

- **"Functions pair and never wire".** The original plan's rule gave no reason, and this plan finds none (§ 6.20, Q4(c)).

- **A row of its deliberate exceptions.** The original plan's row "Views at a field that holds a law or a function: I2, I5, and I9 hold only up to the wrap" is gone, since no view wraps (§ 5.5).

## 8. Transition plan: the interim rules between the steps

Sections 2 to 6 state the design once every step has merged.
Between the steps, README directive 8 asks that deferred work keep today's behaviour, with no shim and no bridging code, and the families of G2 move to grammar B only after the workshop, so their old form stays legal until G2.
An **interim rule** states what holds before a named step merges; every rule below ends at one, and the body's rule then holds alone.

The design text states the target, and the design conformance tests mark each pending signature (§ 6.9), so an interim rule is not a deliberate exception (§ 6.3).
§ 3's last column says which step delivers each invariant, and each step block of § 6.9 to § 6.15 names the rows below that it starts and ends.
The decision records of § 6.18 to § 6.21 and the records of § 7 keep their wording.

In each table, **From** names the step that introduces the operation a row qualifies, and "—" means the behaviour is today's; **Until** names the step whose merge ends the row.
A row's examples write `&` and `rename`, which are `*` and `with_path_names` before S3 (T3), and code names, which are today's before R (T26).
The groups follow the order of the steps; README § Steps makes G2 follow G3 (O16).

**Rows that end at S2b**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T1 | A produced law or kernel offers outputs: the fields of one draw, typed by one draw's values, so `OutputSpec(post_spec).components` is `{mu: NumericArraySpec(()), tau: NumericArraySpec(())}`. A merge accepts such an operand: `laws_of_laws * Normal("x", 0.0, 1.0)` (§ 5.5) builds, and sampling it raises `ResultSchemaError` | output specs positional over a `DistributionSpec` or a `ConditionalDistributionSpec`; merges of laws and kernels | — | S2b | today's reading (fact 11); S2b gives a produced law or kernel no outputs (§ 2.11, § 5.5), and M2 refuses it in a merge (§ 2.17) |
| T2 | `merge` on records is deep: `Record({"g": {"c": 1.0}}).merge(Record({"g": {"e": 2.0}}))` is `{g: {c, e}}` | `NamedTree`, `Record`, `RecordSpec`, and `RecordBatch` | — | S2b | today's rule (fact 5); S2b makes it shallow, so that S3's `&` is only a rename (§ 2.21; § 6.20, Q4(a)) |

**Rows that end at S3**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T3 | Today's operator and names: `*` merges laws and kernels, as `lik * prior`; `merge` merges records, shallow from S2b (T2); `with_path_names` renames; a view is a `FieldView`. A step that merges before S3 writes these names in its code, messages, and done-condition, and S4, if it merges first, adds the renaming of a base element's name (§ 4.8, case 2) to `with_path_names` | laws, kernels, `NamedTree`, `Record`, `RecordSpec`, `RecordBatch`, `OutputSpec`, and batches | — | S3 | S3 is one announced mechanical PR that renames them all at once (§ 6.12) |

**Rows that end at S4**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T4 | A batch stores only its element's term spec, so it holds no base element's name: on draws of `Normal("mu", 0.0, 1.0)`, `draws[["mu"]]` raises `KeyError`, and a base element's name cannot be renamed. A batch of records offers its columns typed by the element's fields, not by the columns' `BatchSpec`s | every batch, and `OutputSpec(bt.spec)` | — | S4 | today's `BatchSpec` (§ 1.2; fact 11); S4 stores the element spec (§ 5.8) |

**Rows that end at G1a**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T5 | A view or a marginal at one path to a group of a law is an interim whole form (§ 8.1) named by the group: `grouped["g"].event_spec` is `OutputSpec(g=RecordSpec(c=..., e=...))`, so the view is indexed by `g/c` while its draws are indexed by `c`; `grouped["g"]["c"]` raises `KeyError`; and `marginal(grouped, "g")` is the factor that holds `g`. So I2, I5, I6, and I7 do not hold for groups. From S2b, `grouped.event_spec["g"]` offers `c` and `e` while `grouped["g"]` offers `g`, since S2b's switch of `_view_declaration` to `O[s]` leaves out one path to a group | views and marginals of laws at one path to a group; `_view_declaration` | — (S2b for `O[s]`) | G1a | today's behaviour; G1a changes what a law's view offers, which users see (§ 6.8) |
| T6 | A group's column and the law's view of the same group disagree: `grouped_draws["g"].element_spec` offers `c` and `e`, while `grouped["g"].event_spec` offers `g` (T5) | a batch of draws, and the law it was drawn from | S4 | G1a | S4 stores element specs, and G1a changes what a law's view offers; the two steps are separate on purpose (§ 5.8) |
| T7 | A grouped sub-joint's event is an interim whole form: `grouped`'s factor `F` declares `OutputSpec(g=NumericRecordSpec(c=..., e=...))` (fact 23), and `factor(grouped, "g")` draws `{c, e}`. `_flattened` keeps it one factor by testing that its event has an output at the root (`distributions/_factored.py:300`; written with `_root_component_name` from S2a). I17 and § 4.15, property 4 fail for it: in `Normal("a", 0.0, 1.0) & F`, `F.event_spec["g"]` is `F.event_spec`, while the merge's `["g"]` is the group | the factor of a law that holds a group (§ 6.4) | — | G1a | today's form; G1a wraps the group, and keeps the sub-joint one factor by a structural test (§ 6.14) |
| T8 | The library's laws declared in an interim whole form over a record keep it, such as an approximate posterior's undeclared event (`_posterior_declaration` and `make_posterior` in `inference/_approximate_distribution.py`, and `_marginal` in `distributions/_empirical.py`) | those library sites | — | G1a | today's form; G1a makes them positional or wrapped (§ 6.8), pending the maintainer's decision on which (`decisions.md`) |
| T9 | Two views of one parent merge as whole terms named after their groups: `hier["s1"] & hier["s2"]` builds and offers `s1` and `s2`, and M3 has no message for views of one parent | merges of two views of one law | — | G1a | today's views of a group are whole terms (T5); once G1a makes both offer `theta`, M3 raises, and its message points to `hier["s1", "s2"]` (§ 2.21) |

**Rows that end at G1b**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T10 | The positional form refuses a `FunctionSpec` and a `BatchSpec` of base elements with `TypeError` (`core/_specs.py:355`): `OutputSpec(FunctionSpec(...))` raises (fact 12), and so does `OutputSpec(BatchSpec(NumericArraySpec(()), sample=3))`, so a batch of base values has no output spec of its own | `OutputSpec` in the positional form; `OutputSpec(bt.spec)` for a batch of base values | — | G1b, after S4 for batches | today's `_exposed_kind`; accepting them needs § 2.11's uniform rule, and a batch of arrays needs S4's element spec (§ 6.8) |
| T11 | Outputs keep today's default. `OutputSpec.default` gives the keyword form under the default name for a batch, a law, a kernel, or a function; a bare `output_spec` other than a record reads in the keyword form under the callable's name, so `Function(run_inference, output_spec=post_spec).output_spec` is `OutputSpec(run_inference=DistributionSpec(...))`; and a lifted law-returning function names its event whole, so `Function(law_of_y)(Normal("x", 0.0, 1.0)).event_spec` is `OutputSpec(law_of_y=DistributionSpec(...))`. `_complete_output_spec`, `_output_atoms`, and `_lift_result` build these interim whole forms. If G1b is cut, they keep the old form until G1b merges after the workshop | `OutputSpec.default`, a function's output spec, and a lifted call's event | — | G1b | today's rule (§ 4.12, § 4.13); G1b changes the output default once |

**Rows that end at G2**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T12 | The keyword form accepts a spec with parts and builds an interim whole form, which follows § 8.1: `OutputSpec(params=RecordSpec(beta=(), sigma=()))` builds, and so does `OutputSpec(emulator=FunctionSpec(...))`. A user, and a test, can still write it | `OutputSpec(n=T)` for a spec with parts `T` | — | G2 | the G2 families declare their draws in this form, so refusing it before they move would break them, and a special case that exempts them would be bridging code (§ 6.8); G2 refuses it last |
| T13 | Filling a type hole with a spec with parts, other than a batch whose element has only the default name (O2), gives an interim whole form instead of raising `TypeError` | `with_term_spec` on a type hole | — | G2 | case 3 of § 4.12's type holes is the refusal of the interim whole form, so it comes with G2 |
| T14 | The G2 families, random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets, declare their events in an interim whole form through the private event default (`_whole_term_event`, the event reading of `distribution`, `_agreed_event_spec`, and `_event_repr_parts`, § 4.12), which keeps today's rule: the keyword form under the default name for a law, a function, or a batch. An empirical law of laws that a user builds keeps the form too. So `rm.event_spec["G"]` is `rm.event_spec` (§ 8.1, rule 3) | the G2 families, and the private event default | — | G2 | current behaviour kept until its step, not bridging code: G2 moves each family to a wrap under the name its constructor takes, then the event default goes, so it changes once (§ 4.12) |
| T15 | A merge accepts an interim whole form and stores its produced term whole, as the field of its name (§ 8.1, rule 5): M2 passes it, and so does `O & P` on output specs (§ 2.18). `Normal("z", 0.0, 1.0) & random_function`, and `rm_whole & Normal("x", 0.0, 1.0)` for a random measure `rm_whole` declared `OutputSpec(G=law_spec)`, merge as today (fact 27) | merges of laws and kernels, and of output specs | — | G2 | it keeps today's merges of the G2 families working (§ 2.17) |
| T16 | Grammar B's guarantee (§ 2.11), I17, and § 4.15, property 4 fail for an interim whole form, and I5 fails at its name, after G3 as before: `rm_whole["G"]` is `rm_whole`, so `rm_whole["G"]["mu"]` raises `KeyError` while `rm_whole["G/mu"]` is the view at `mu` | every interim whole form (T5, T7, T8, T11 to T14, T19) | — | G2; G1a for the grouped sub-joint (T7) | today's reading, which § 8.1 states |
| T17 | Code outside `core/_specs.py` reads an output spec's form through the private helper `_root_component_name` (§ 5.11), which reads the stored form, since an interim whole form can still be built, as `RandomMeasure` builds one; the replacement of `exposes_record` reads the form through it too | `_root_component_name` and its callers | S2a | G2 | after G2 the helper reduces to the rule that an output spec is in the keyword form when its term spec is a base spec (§ 5.11); R renames it (T26) |

**Rows that end at G3**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T18 | `[]` reads the name of the output at the root of an output spec in the keyword form over a base spec: `Normal("x", 0.0, 1.0)["x"]` is the law, and `OutputSpec(x=NumericArraySpec(()))["x"]` is the spec itself (§ 4.4, rule 4), so `t[...]`, `marginal(d, ...)`, and `O[...]` accept that name; `"x" in Normal("x", 0.0, 1.0)` is true; and no accessor `t.output(c)` exists | laws and output specs | S1 for `in` on laws; S2b for `O[n]` | G3 | `O[n]` mirrors `Distribution.__getitem__` (`distributions/_distribution.py:1015`), so I6 holds for laws at every step; G3 removes the reading from both at once, with the accessor that replaces it, so a reader always has one way to read an output (§ 6.20, Q5), and library code that indexes a law by that name moves to the accessor in the same step; so it is not bridging code |
| T19 | A view at or below a field that holds a law, a kernel, a function, or a batch keeps today's interim whole form named by the field: `d[p].event_spec == OutputSpec(**{last(p): d.event_spec[p].term_spec})`, which is I6's interim form at these fields, so I2, I5, I6, and I9 have an exception there (§ 8.1 has an example). S2b's switch of `_view_declaration` to `O[s]` leaves out one path to such a field, and G1a's rule that a view of a node with parts offers its parts applies to groups, not to these fields | views of laws at held fields; `_view_declaration` | — | G3 | today's behaviour; G2's refusal of the interim whole form would refuse these views, so G2 follows G3 (O16). **Inconsistent:** § 6.8 said "Until G2, the G2 families still build the interim whole form and a user can still write it; every other part of the library has stopped once G1b merges", while this row, § 5.5, and the probe's allowance (§ 6.14) let `_view_declaration` build the form until G3, which follows the workshop |
| T20 | A batch's output at the root: `draws["mu"]` raises `KeyError`, while `draws.spec.element_spec["mu"]` is that output spec, by T18, so I6 fails for a batch's output at the root | batches of base values | S4 | G3 | § 4.4, rule 4's reading mirrors laws (T18), and `[]` on a batch refuses the name today |
| T21 | Paths stop at a field that holds a law, a function, or a batch, whose term is indexed only by a second `[]`: `record_with_law["G/mu"]` raises `KeyError`, and `"G/mu" in record_with_law` is false. A function has no `[]`: `emulator["mean"]` raises `TypeError` | terms with held fields; functions | — | G3 | today's behaviour; G3 delivers paths through every kind and `Function[]` (§ 5.4, § 5.7) |

**Rows that end at G4**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T22 | An input whose spec is a `RecordSpec` receives one value whole: no binding fills part of it, and `InputSpec[]`, `in`, and `without` take input names only, so `lik_g.given_spec["g/c"]` raises `KeyError` and `"g/c" in lik_g.given_spec` is false | record inputs of kernels and functions; `InputSpec` | — | G4 | partial binding of a record input (III.9) and paths into it come together in G4 (§ 5.10) |
| T23 | A law has no `without`, so `post.without("tau")` raises `AttributeError`; and `condition_on` keeps today's result: its event spec depends on the route, as `condition_on(prior, {"mu": 0.1})` gives `OutputSpec(tau=...)` on the exact Gaussian route, and a nested path raises `ResolutionError`. So I8, I10 on laws, and I11 do not hold yet | laws; `condition_on` | — | G4 | today's behaviour; G4 delivers § 5.5 and `law.without` (§ 4.5) |

**Rows that end at G5**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T24 | A kernel has no `[]` and no accessor: `lik["y"]` raises `TypeError`, and so does a path through a field that holds a kernel; a kernel is the one term with a produced term and no `[]`, and the design states the exception. A batch of laws or of functions has no `[]` by path: `laws["mu"]` raises `TypeError`, with the false message that the elements have no named fields | kernels; batches of laws and of functions | — | G5 | kernel views come in G5 (§ 5.8, § 5.6) |

**Rows that end at R, the rename**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T25 | `design/`, the glossary, docstrings, messages, the User Guide, and `execution-layer.md` keep today's words, with this plan's meanings: an input is a "slot", an output a "component", and an input spec, an output spec, and an event spec are an "input declaration", an "output declaration", and an "event declaration". Every instruction to an implementer that writes design text, a docstring, or a message, and every step block of § 6.7, uses these words; § 8.2 gives the mapping | `design/`, the glossary, the prose of the code, `execution-layer.md`, § 6.5, and § 6.7 to § 6.17 | — | R | § 6.20, Q1: one announced rename PR right after the workshop keeps directives 7 and 11 at every merge, and keeps the critical path free of judgement by sense |
| T26 | Code names keep today's spelling: `O.components` for `O.outputs`, with § 2.11's meaning; `component=` for `output_name=`, in `OutputSpec.default` and in the families; `factor(d, component_name)`; `t.component(c)` for `t.output(c)` when G3 merges before R, so `mix.component("x")` reads a mixture's output `x`; `bt.element_spec.components`; the private names of § 6.4's last row, `_root_component_name` among them; the page `docs/api/declarations.md`; and the message of `OutputSpec.__iter__`, which says ".components" | `OutputSpec`, terms, the families, private helpers, and the docs page | — | R | as T25; `element_spec` is a final name, which S4 introduces directly |

**Rows that end after the workshop, with the other work of § 6.15 that `deferred.md` lists**

| # | What | Applies to | From | Until | Why |
| --- | --- | --- | --- | --- | --- |
| T27 | `&` exists only on records, laws and kernels, `RecordBatch`, and input, output, and record specs. Two functions do not merge; two batches of arrays raise `TypeError` (fact 44); and the specs of laws, kernels, functions, and batches have no `&`, so `(a & b).spec == a.spec & b.spec` holds only for records | functions; batches other than `RecordBatch`; the specs of laws, kernels, functions, and batches | — | after the workshop | § 6.16 and `deferred.md` |
| T28 | `InputSpec` has no `rename`, while a kernel's own `rename` renames its inputs (III.9); so § 4.15, property 6, and I21 do not cover functions | `InputSpec`; functions | — | after the workshop | it belongs with spec merges for model builders (§ 5.10, § 6.16); `rename` replaces `with_path_names`, which `OutputSpec` has today and `InputSpec` does not |
| T29 | A shared input carries the first operand's spec, not `U` (§ 2.14), in `InputSpec &` and so in the joint, whose inputs S2b builds through it; S2b changes no joint's behaviour | `InputSpec &`, and the `given_spec` of merged laws and kernels | S2b | after the workshop | today's joint keeps the first operand's spec (`distributions/_factored.py:424`; fact 8), and validity is unaffected (fact 39); the switch changes one test (fact 35), so it stays out of S2b (§ 6.20, Q4(d)) |
| T30 | `without` does not exist on functions, kernels, or batches of laws, nor `Record.without` at a path through a held term, so I10 does not apply to them, and a held law, function, or batch lacks removal | functions, kernels, batches of laws, and records at held paths | — | after the workshop | `deferred.md` |
| T31 | An executor need not restore the names a transform drops: a body that an executor runs receives raw values only, so no body sees names | executors and requesters | — | after the workshop, once the rest of #506 lets a body ask for tracked terms | O4; the execution plan's § 6.7 |

### 8.1 The interim whole form

An **interim whole form** is `OutputSpec(n=T)` for a spec with parts `T`: the keyword form naming a structured term from outside, which today's code accepts (§ 1.2) and grammar B removes (§ 2.7).
Today's random measures, random functions, Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets declare their events this way, and so does the grouped sub-joint factor of a regrouped law, as `grouped`'s factor `OutputSpec(g=NumericRecordSpec(c=..., e=...))` (fact 23).
The grouped sub-joint's form ends with G1a, which wraps it (T7); the G2 families' form ends with G2, which moves them to wraps and then refuses the form (T12 to T16).
Until then the form keeps today's reading in `[]`, `without`, `rename`, and `&`, which is current behaviour.
For `O = OutputSpec(n=T)`:
1. **outputs:** one output, `n`, at the root, whose spec is `T`, so `O.outputs` is `{n: T}`;
2. **paths:** `n`, and `n/q` for each path `q` of `T`; for a spec `T′` that holds `O` (§ 2.8), `spec_at(T′, n)` is `T′` itself, and `spec_at(T′, n + "/" + q)` is `with_held(T′, O_q)`, where `O_q` is built from `N = spec_at(T, q)` as § 2.10's case 2 builds `E_p`, privately, so the positional form's public refusal of a `FunctionSpec` and of a batch of base elements until G1b (T10) does not apply;
3. **indexing:** `O[n]` is `O`, after G3 as before, since G3's change of § 4.4, rule 4 concerns only the keyword form over a base spec (T18); `O["n/q"]` is `O_q`, built as in rule 2; a list `ps` of these paths gives `OutputSpec(restrict(RecordSpec(n=T), ps))`, with `restrict` as § 4.4 defines it; and inside `restrict`, a spec `T′` that holds `O` gives `with_held(T′, O[ps])`;
4. **removal:** `O.without(n)` raises `ValueError`, since nothing would be left; for paths `n/q`, `O.without(*ps)` is `OutputSpec(n=remove(T, qs))`, with `remove` as § 4.5 defines it and `qs` the paths below `n`, so the name stays;
5. **the record of outputs** (§ 2.12): `rec_O(v)` is `{n: v}`, and `dec_O(z)` is the field `n` of `z`, so a merge stores the produced term whole, as the field `n` (T15);
6. **grammar B's guarantee fails:** the form offers an output at the root and has paths below it (T16).

```python
P = OutputSpec(params=RecordSpec(beta=(), sigma=()))   # an interim whole form (T12)
P["params"] is P; P["params/beta"]                     # P itself; OutputSpec(beta=...)

# params_prior declared OutputSpec(params=RecordSpec(beta=..., sigma=...)), as today
params_prior["params/beta"]          # the view of beta, while its draws {beta, sigma} are indexed by beta
params_prior["beta"]                 # KeyError: "'beta' is not an event path of 'ℙ'; its fields: ['params']"
params_prior.with_path_names({"params/beta": "params/b"})   # renames inside the whole term; rename from S3

# whole_laws draws laws and is declared OutputSpec(G=Normal("mu", 0.0, 1.0).spec), as today (T19)
joint = whole_laws * Normal("x", 0.0, 1.0)      # its event is {G: law, x}
joint["G"].event_spec                            # OutputSpec(G=DistributionSpec(...)): the interim whole form

rm.event_spec["G"]       # rm.event_spec itself, while rm's event is an interim whole form (T14)
rm_whole["G"]            # rm_whole, for a random measure declared OutputSpec(G=law_spec), after G3 as before
rm_whole["G/mu"]         # the view at mu of the drawn law
rm_whole["G"]["mu"]      # KeyError: I5 fails at the name (T16)
```

### 8.2 The words before the rename

The design does not depend on the words.
Every step that merges before the rename PR writes this plan's rules in today's words, so `dev/overhaul` is coherent at every merge (directives 7 and 11).
One announced PR after the workshop (§ 6.15) then renames the words in `design/`, the glossary, the code, and `execution-layer.md` together, with a CHANGELOG entry.
This plan's own definitions and explanations use the left-hand column below.
Every instruction to an implementer that writes design text, a docstring, or a message before the rename uses the right-hand column, and the step blocks of § 6.7 use its names, since they describe code merged before the rename:

| This plan | Before the rename |
| --- | --- |
| input, optional input, a kernel's input | slot, optional slot, given slot |
| output; the output at the root | component; the component at the root |
| input spec, output spec, event spec | input declaration, output declaration, event declaration; the class names `InputSpec` and `OutputSpec`, and the attributes `input_spec`, `output_spec`, `given_spec`, and `event_spec`, are already final |
| the spec of an output | the spec of a component |
| `O.outputs` | `O.components`, with this plan's meaning (§ 2.11) |
| `t.output(c)` | `t.component(c)`, when G3 merges before the rename (§ 6.15) |
| `output_name=` | `component=` |
| `element_spec` | `element_spec`, a final name that S4 introduces with its new meaning |
| feeds, produced term, conforms, unify, unfed, keyword form, positional form, group, merge | the same words, which name no renamed thing; the steps of § 6.7 bring them in where § 6.4's table says |

For example, S0 writes in II.2 "an output declaration in the positional form offers its components", and the rename PR changes that sentence to "an output spec in the positional form offers its outputs".
Where § 6.7 describes S0 and S2a, it says which words each writes.
