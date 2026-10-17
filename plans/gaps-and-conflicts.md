# Gaps and conflicts across the plans

This file lists every place where the three plans disagree with each other, are out of date, or leave a decision open that implementation needs.
The `plans/` PR merges once each item here is settled and the plans are updated to match.
Settled items move to [`decisions.md`](decisions.md), and the plan texts change in the same commit.

Each item has the same parts:
- **Background:** what the reader needs to know to follow the item.
- **The problem:** what disagrees or is missing, with code where it helps.
- **Options** and a **recommendation**.
- **Blocks:** which implementation steps cannot start until the item is settled. The steps are named in [`README.md`](README.md) § Steps.
- **Status:** open, recommended, or settled.

The items fall into five groups:

| Group | What it holds | Items |
|---|---|---|
| D | Decisions needed before implementation can start | D1–D9 |
| T | Text fixes that need no new decision | T1–T6 |
| F | Steps that edit the same files, and the order that avoids collisions | F1 |
| W | Gaps in how the push itself works | W1–W5 |
| L | Known gaps that can wait until after the workshop | L1–L6 |

---

## Orientation: the three plans in one page

**The execution layer** ([`execution-layer.md`](execution-layer.md), issue #580) decides *how* a computation runs.
Today ProbPipe cannot run code that JAX cannot compile, such as a simulator called through `subprocess`, and each of fourteen places in the library decides on its own whether to vectorize or loop.
The plan adds:
- **execution claims** on a `Function`, such as `traceable=False`;
- **point sets**, a batch of evaluations of one callable;
- **executors**, pluggable runners for point sets (sequential, threads, a user's Slurm cluster), chosen with `using(executor=...)`;
- the **embedded call**, which runs opaque code inside a compiled JAX program through a host callback, so a joint's compiled sampler or a BlackJAX kernel can send a batch of simulator runs to a cluster;
- the **call token**, a small integer passed into compiled programs so a callback can find the settings of the call it belongs to.

**Declarations** ([`declarations.md`](declarations.md), issue #571) decide *how terms describe what they produce and consume*.
A law's `event_spec` and a function's `output_spec` are `OutputSpec`s, which offer named **components**.
A kernel's `given_spec` and a function's `input_spec` are `InputSpec`s, which declare named **slots**.
Composition meets a component with the slot of the same name.
#571 gives these declarations public operations (`project`, `without`, merging), one way of reading slots, and the batch's stored element declaration.
jhuggins's reply makes `OutputSpec` a mapping and proposes `&` for composition.

**Addressing** ([`addressing.md`](addressing.md), issue #573) decides *how a path or a subscript picks out a part of a term*.
Today `x["a", "b"]` means one nested path on a record and two paths on a law.
The plan gives one reading of a subscript everywhere, and adopts **grammar B**: only an array or an opaque value may be named as a whole, and anything with parts offers its parts or is wrapped in a one-field record.

Three terms recur below:
- A **whole** declaration names one value under one name, written with a keyword: `OutputSpec(x=NumericArraySpec(()))`.
- An **exposed** declaration offers the parts of its value as components, written positionally: `OutputSpec(RecordSpec(a=..., g=...))` offers `a` and `g`.
- A **base spec** types a value with no named parts: a `NumericArraySpec` or an `OpaqueSpec`. Every other spec has parts.
- A **view** `d[p]` is the law of the part of `d` at path `p`, which keeps a reference to `d` (today the class `FieldView`).
- A component's **address** is the position of its value inside the produced value: the root for a whole declaration, and the path of its name for an exposed record.

The addressing plan states its scope against #571 explicitly, so the two were meant to fit.
They were written two days apart, however, and #571's text was not updated when grammar B was adopted.
The execution plan was written alongside them, and uses some words and examples that the other two have since changed.

---

## D. Decisions needed before implementation

### D1. There is no single current text for declarations

**Background.**
The declarations design exists in three texts:
1. #571's issue body (2026-10-13);
2. jhuggins's reply on #571 (2026-10-15), which accepts most of the body and changes four things;
3. the grammar-B addressing plan (2026-10-16), whose section "Where this plan meets #571" lists twelve points and changes #571's text at several of them.

None of the three has been updated to match the later ones.
An implementer reading #571 alone would build several things that the later texts replaced.

**Rule of precedence (settled 2026-10-17).**
Where the texts disagree, the grammar-B addressing plan wins, then jhuggins's reply, then #571's body.
Under this rule every disagreement below is resolved, and the table records the outcome.
The old positions stay in the table so the reconciled text can say what changes from #571.

| # | Topic | #571 body | jhuggins's reply | Grammar B, which takes precedence | Resolution |
|---|---|---|---|---|---|
| a | Whether a declaration is whole or exposed | a property `exposes` with four values: `"whole"`, `"record"`, `"event"`, `"element"` | a boolean `exposes_parts` | no property: the spec decides, since a declaration is whole exactly when its spec is a base spec | ✅ grammar B: no flag |
| b | `project` with several paths | a record of the nodes, each under its last segment, in the order given; two paths with the same last segment raise | as the body | each node stays at its own path, in the tree's order; a path inside another raises | ✅ grammar B |
| c | `project` with one path to a group | the group, named whole by its last segment | as the body | the group's parts, exposed; `project(["g"])` keeps the name | ✅ grammar B |
| d | Reading the term at a component | five cases by what the declaration exposes: the term itself, a record's child, a law's view, a kernel's laws, a batch's elements | a private method `_component` per kind | one rule: a public `t.component(c)`, equal to `t[address(c)]` | ✅ grammar B. The method's name is still a placeholder (below). |
| e | `batch["mu"]` on a batch of draws of `Normal("mu", ...)` | returns the batch | as the body | raises, since `mu` is a name and not a path; `draws.component("mu")` returns the batch | ✅ grammar B |
| f | The spec of a component when the declared value is a law | the spec of one draw's value at the component | reads the term at component `c` of a law as the view `t[c]`, without restating the spec | the spec of the law's view at the component, a `DistributionSpec`, following jhuggins's decision 5 in his component-interface doc | ✅ grammar B |
| g | A batch built from raw values | keeps a bare element spec with no components, so a batch of arrays cannot be exposed | as the body | every batch holds an element declaration; a base element is whole under the default name `element` | ✅ grammar B |
| h | Equality of batches whose elements differ only in component names | not equal | as the body | equal: names stay out of equality, hashing, and the JAX pytree | ✅ grammar B, confirmed in D3. |
| i | Merging declarations | `OutputSpec.merge`, `InputSpec.merge` | `&` replaces `*` and both `merge` methods | uses `&` in its examples | ✅ `&`, on which the reply and grammar B agree. D2 keeps the renames and the timing. |
| j | The paths of a whole declaration | the component followed by the paths inside the term, as `c/beta` | as the body | only the root, since a whole declaration names a base value | ✅ grammar B |
| k | The column of a group in a batch of records, `rb["g"]` | not covered (a field's column carries `OutputSpec(a=...)`) | as the body | carries the exposed record `{c, e}` | ✅ grammar B |
| l | `&` on an operand that exposes a law or a batch | not covered | raises | as the reply; since grammar B exposes more outputs, such outputs must be wrapped to merge | ✅ the reply and grammar B agree |
| m | `OutputSpec.without` on a whole declaration | removes nodes inside its term | as the body | has nothing to remove, since a whole declaration names a base value with no paths below it | ✅ grammar B |

Two examples show what the resolution changes from #571.

Row b, with a hierarchical law `hier` whose draws are `{s1: {theta}, s2: {theta}, mu}`:

```python
hier.event_spec.project(("s1/theta", "s2/theta"))
# #571 body: ValueError, since both paths end in "theta"
# grammar B: the record {s1: {theta}, s2: {theta}}, which keeps both parameters apart
```

Row c, with a law `grouped` whose draws are `{a, g: {c, e}}`:

```python
grouped["g"].event_spec
# #571 body: OutputSpec(g=RecordSpec(c=..., e=...)), a whole term named g,
#            so the view is indexed by "g/c" while its draws are indexed by "c"
# grammar B: OutputSpec(RecordSpec(c=..., e=...)), which offers c and e,
#            so the view and its draws are both indexed by "c"
grouped[["g"]].event_spec   # grammar B: OutputSpec(RecordSpec(g=RecordSpec(c=..., e=...)))
```

**What remains open in D1.** Two small things, neither a conflict between the texts:
1. **The form of the text.** Either one reconciled text, `plans/declarations-and-addressing.md`, replaces `declarations.md` and `addressing.md`, and #571 and #573 link to it; or the two texts stay separate and #571's body is edited row by row. Recommendation: one reconciled text, since the two plans meet in twelve places and that boundary caused most of the disagreement.
2. **The name of the component accessor.** The addressing plan writes `t.component(name)` and calls the name a placeholder. Recommendation: keep `component`. It pairs with the glossary term, and `d[...]` already reads paths, so the accessor needs a name that says it reads a component.

**Blocks:** S2, S3, S4, and every later step of the declarations and addressing track, until the reconciled text exists.

**Status:** ✅ settled 2026-10-17. The conflicts are resolved by precedence, the plan becomes one reconciled text, and the accessor is named `component`.

---

### D2. Should `&` replace `*` for composition, and should the pending renames merge with it?

**Background.**
Today `*` composes laws and kernels: `likelihood * prior` is the joint law.
`*` is also arithmetic on values: it multiplies a `NumericArray` by another and scales a `LinOp`.
So `d1 * d2` composes two laws, while `sample(d1) * sample(d2)` multiplies their draws.
jhuggins's reply on #571 proposes `&`, read as "both", for composition, and extends it to records, declarations, `InputSpec`, functions, and batches:

```python
model = likelihood & prior                      # the joint law, as likelihood * prior is today
spec = prior.event_spec.without("sigma") & shocks.event_spec   # replaces OutputSpec.merge
summaries = center & spread                     # one Function that returns both outputs
```

Rule for every kind: `(a & b).spec == a.spec & b.spec`.
A name that both operands produce raises, and same-named slots of both operands become one slot.

The addressing plan writes its examples with `&` and marks the operator itself as #571's decision.
The execution plan's examples still use `*`.

Grammar B notes one exception: `&` on two arrays is NumPy's bitwise "and".
So `sample(Normal("x", 0, 1)) & sample(Normal("y", 0, 1))` combines two arrays elementwise, while `Normal("x", 0, 1) & Normal("y", 0, 1)` is the joint law.

Two renames are also pending, both user-visible and both recommended by the addressing plan:
- `with_path_names` becomes `rename`, as in pandas and xarray;
- `FieldView` becomes `MarginalView`, since it covers groups and lists of paths.

The parameter of `marginal(d, field)` also becomes `paths`, which belongs to step S1 because it changes what `marginal` accepts.

**The problem.**
`&` changes the most visible line of every model a user writes.
If it merges after the workshop, attendees learn `*` and then unlearn it.
The renames have the same timing problem.
The decision is not yours alone: jhuggins proposed `&`, and the addressing plan only adopted it for its examples.

**Options.**
1. Adopt `&` for laws, kernels, records, and declarations before the workshop. Remove `*` for composition with no alias. Add `&` on functions and batches after the workshop.
2. Keep `*` for composition, and give declarations a `merge` method.
3. Support both for a release.

**Recommendation.** Option 1, settled with jhuggins.
Put `&` and the two renames in one mechanical PR (step S3), announced in advance. Step S2 is also a wide rename (`components` and `exposes_record` go and `.spec` becomes `term_spec`, on about 380 lines), so it is announced the same way (item F1).
Option 3 contradicts the clean-break rule of item D5.

**Blocks:** S3.

**Status:** ✅ settled 2026-10-17. `&` replaces `*` for composition (D1 row i). `with_path_names` becomes `rename`, and `FieldView` becomes `MarginalView`. All three change in one announced mechanical PR, step S3. Whether `*` keeps an alias is D5, and whether S3 comes before the workshop is D9.

---

### D3. Do a batch's component names stay out of equality and the JAX pytree?

**Background.**
A batch is a JAX pytree, so JAX splits it into arrays and static data.
JAX compares the static data to decide whether two values have the same structure, which decides whether `jit` reuses a compiled program and whether the two branches of `lax.cond` may return the batch.
A batch's static data is its `BatchSpec`.

#571 has every batch that an operation produces store its producer's declaration, so that a batch of draws of `Normal("mu", 0, 1)` still knows its component is `mu`.
That fixes a real defect: today `sample(Normal("mu", 0, 1), sample_shape=4)` forgets the name.

**The problem.**
#571 says two batches whose elements differ only in component names are not equal, and it keeps the names out of fingerprints.
The addressing plan recommends the opposite: keep the names out of equality, hashing, the pytree's static data, and fingerprints.
If the names were static data:

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=4)
y = sample(Normal("y", 0.0, 1.0), sample_shape=4)
# names in the pytree: x and y have different structures, so a jitted function
# compiles once per name, and lax.cond cannot return x from one branch and y from the other
```

Keeping them out has a cost, the one labels already pay.
A raw JAX transform rebuilds the batch from its static data, so the name of a whole element is lost:

```python
jax.jit(g)(x)                 # comes back offering the default component "element", not "x"
```

The names of a record element are stored in its `RecordSpec`, which is static data already, so they survive.
ProbPipe's own operations restore names at their result boundary.

**Options.**
1. Names out of equality, hashing, the pytree, and fingerprints (addressing plan).
2. Names in equality and the pytree (#571).

**Recommendation.** Option 1.
jhuggins's Rocq model proves only that `&` commutes with renaming, so this needs his agreement.

**Blocks:** S4.

**Status:** ✅ settled 2026-10-17: option 1. A batch's component names stay out of equality, hashing, the pytree's static data, and fingerprints.

---

### D4. Grammar B's open rulings, and whether to enforce grammar B before the workshop

**Background.**
Under grammar B, a whole declaration may name only a base spec.
`OutputSpec(params=RecordSpec(beta=(), sigma=()))` raises.
A law over that record either offers `beta` and `sigma`, or is wrapped as `OutputSpec(RecordSpec(params=RecordSpec(...)))`, whose draws are `{params: {beta, sigma}}`.
This keeps one set of paths: a term's subscripts, its rename keys, and its draws all agree.

jhuggins's component-interface doc models this in Rocq and leans toward B, but leaves two rulings open, and the addressing plan leaves a third.

**The problem.** Three rulings are open, and a fourth question is about timing.

1. **B or B′.**
   B′ would also let a spec with no paths below its root be named whole: a batch of arrays, a law over one array, or a random function whose output is one array.
   B′ keeps a few names, such as a Gaussian process's own name `f`.
   It costs a flag on every declaration, since the spec no longer decides whole or exposed.
   The addressing plan chooses B, and jhuggins has not ruled.

2. **A view of a field that holds a law.**
   A random measure `rm` draws records `{G: law}`.
   Paths go through every kind, so `rm["G"]` is a view whose value is a law.
   Under grammar B that view exposes the inner law's parts, which makes its event expose a law.
   #571's event rule forbids that: a law's event is whole or an exposed record, because a joint stores each component of a draw as a field.
   The same holds for a Gaussian process `gp`, whose draws are `{f: function}`.

   ```python
   rm["G"]        # grammar B: exposes the inner law's components
                  # #571's event rule: refused
   ```

   Two ways out: such a view wraps, keeping the name as `rm[["G"]]` does, or #571 narrows its event rule to allow exposed laws and kernels with two or more components.
   Until this is ruled, `rm.component("G")` and `gp.component("f")` are undefined.

3. **Who builds the record when an output is wrapped.**
   A function that returns a posterior under one name is declared `OutputSpec(RecordSpec(posterior=post_spec))`.
   Either its body returns `{"posterior": law}` and the declaration only checks it, or the body returns the law and ProbPipe wraps it.
   The same question decides what a pending type `OutputSpec(x=None)` becomes when it is filled with a spec that has parts.
   This question is the addressing plan's own. It leans to the first: the body returns the record, and a pending type filled with parts raises.

4. **Timing.**
   The addressing plan counted 679 tests in 46 files that run a whole declaration over a spec with parts, an upper bound on the tests that change, and about 31 library sites.
   It changes what users see in several places: the view of a group, laws declared whole over a record, random measures, Gaussian processes, and functions that return laws.

**Options for timing.**
1. Settle rulings 1–3 and the design text before the workshop, and enforce grammar B after it. The workshop material avoids the forms that change.
2. Enforce grammar B before the workshop.

**Recommendation.**
Ruling 1: B.
Ruling 2: such a view wraps, which keeps #571's event rule as it stands. Narrowing the rule may be sound once row f of D1 types a component by its view, but it changes what a joint's draw stores, so it needs its own analysis.
Ruling 3: the body returns the record.
Timing: option 1. The size and the open rulings make option 2 a risk to everything else on the schedule.

**Blocks:** steps G1 and G2.

**Status:** ✅ settled 2026-10-17.
1. Grammar B, not B′.
2. A view of a field that holds a law or a function wraps, keeping the field's name, so #571's event rule stands unchanged.
3. The body of a function with a wrapped output returns the record, and the declaration checks it. A pending type filled with a spec that has parts raises.
4. Grammar B for the kinds the workshop uses lands before it (step G1). Random measures, random functions, and Gaussian processes do not appear in the workshop, so they wait until after it (step G2).

The refusal of a whole declaration over a spec with parts waits for G2 as well. Random measures and Gaussian processes declare their draws whole today, so refusing that form before they are migrated would break them, and a special case that exempts them would be the bridging code directive 8 forbids. Until G2, a user can still write the old form, and the library no longer does.

---

### D5. Clean break, with no deprecation aliases?

**Background.**
ProbPipe has not released on PyPI.
#448 set the rule for 0.2.0 as breaking changes with no compatibility shims, and draft PR #549 prepares 0.2.0.
jhuggins's reply on #571 answers "no deprecated aliases" for its renames.
The addressing plan lists fourteen breaking changes and marks "a clean break" as open, to confirm with jhuggins.
The execution plan's Q2 asks whether `dispatch` and `max_workers` go before 0.2.0.

**The problem.**
Each plan asks the question separately, and the answer should be one rule for the push.
It also decides whether 0.2.0 waits for the push.

**Options.**
1. A clean break everywhere, each change recorded under `### Changed (breaking)` in the CHANGELOG, and 0.2.0 released after the push.
2. Deprecation warnings for one release.

**Recommendation.** Option 1.
A shim is the kind of bridging code that directive 8 of the README forbids for deferred work, and the same reasoning applies to a change made now.

**Blocks:** S2, S3, E3.

**Status:** ✅ settled 2026-10-17: option 1. Every rename and removal in the push is a clean break with no alias or warning, recorded under `### Changed (breaking)` in the CHANGELOG, and 0.2.0 is released after the push.

---

### D6. The execution plan's open questions on the pre-workshop path

**Background.**
The execution plan asks eighteen questions in its section 9.
Two are decided: Q1, the name `executor`, and Q17, at most one run per distinct point inside a call.
Q12 moved to issue #585.
The questions below block a pre-workshop step.
Each is explained from scratch in the plan's section 9, so here each gets its context, the recommendation, and the step it blocks.

| Q | Question | Recommendation | Blocks |
|---|---|---|---|
| Q5 | Where the code goes. The import order puts `distributions/` before `functions/`, and joints must use the executor registry. | Point sets and the registry go in `core/`; `functions/` installs the workflow scope into `core` at import, as the call engine is installed today. No import goes against the layering. | E2, E3 |
| Q13 | What happens with a callable that claims nothing. Tutorial code claims nothing. | Probe it once per input structure, in `check` as well as in the call, and store the outcome. A callable that draws from NumPy's global generator is recorded as opaque, with a warning. | E2 |
| Q16 | Names that collide with ProbPipe's vocabulary | ✅ *probe result* for a stored probe outcome; "the point's random key" with no separate name; `functions/_point_keys.py` for the module that `_managed.py` and the key broker `_broker.py` merge into (#524) | E2, E3 |
| Q15 | The executor class's name | `PointExecutor`, since users often import `concurrent.futures.Executor` too | E3 |
| Q3 | Which callables an executor named on an operation applies to. `condition_on.using(executor="slurm")` must send the simulator to the cluster without the user rebuilding the model. | Every opaque callable inside the call. An executor set on a `Function` itself wins, so a cheap statistic can stay local. A mapping by label can come later. | E3 |
| Q2 | When `dispatch` and `max_workers` go | ✅ Settled by D5: removed outright, and 0.2.0 waits for the push | E3 |
| Q8 | Which executor a ProbPipe call made inside a point uses | Automatic selection, unless the call names one, so a cluster job does not submit cluster jobs | E3, E5 |
| Q4 | Host callbacks, staged execution, or both. Staged execution stores `G(θ)` as a deterministic field of a joint. | Callbacks now. Deterministic fields change what a joint's event holds, so they go to a separate issue (L5 below). | E4 |
| Q7 | What a failed point means | The call raises once its point set completes, naming the point, and a compiled program stops at once. Dropping failed points is opt-in, after the workshop, for point sets built in Python only. | E4 |
| Q14 | Whether a traceable model may use gradient methods without claiming gradients. A strict reading of V.11, which says a claim is never inferred, would take NUTS away from every tutorial model. | Yes, as today. Claims only narrow it: an opaque step needs a supplied derivative, and an explicit claim restricts which inputs gradients use. V.11's text says so. | E6 |
| Q18 | A JAX setting, such as `jax_enable_x64`, changed during a call | Raise, naming the setting and both values | E8 |

Questions Q6 (the memo across calls), Q9 (Prefect), Q10 (a Gibbs method), and Q11 (who owns the differentiability claim) belong to post-workshop steps.

**Status:** ✅ settled 2026-10-17: every recommendation in the table is accepted, Q5 and Q14 included. Q16's module is named `functions/_point_keys.py`, since `_work_scope.py` was too vague.

---

### D7. Per-call settings: #587's `config` blocks and the execution plan's call context

**Background.**
PR #587 (jhuggins, ready to merge after #586) groups the global settings under `probpipe.config` and adds temporary blocks:

```python
with probpipe.config.sampling.using(num_samples=4096):
    mean.using(method="monte_carlo")(y)
```

A block is stored in a context variable.
Since PR #547, a call carries a small context into the worker threads it dispatches to, holding controls such as the inherited `exact_only`.
#587 extends it to `_CarriedContext(func, inherited, record, settings)`: the active blocks are `settings`, and `record` is #586's record of the Monte Carlo draws a call makes, which provenance reports.
#587's design text adds: "A setting read while JAX traces a function is its value at trace time."

The execution plan carries per-call settings the same way within Python: a call context in a context variable, inherited by worker threads.
Inside a compiled program it cannot, for two reasons its verification found:
- under `pmap`, a host callback runs on XLA's own threads, which cannot see context variables;
- a value read while tracing becomes a constant of the compiled program, and a cached program then serves later calls with the first call's value. The plan's prototype hit this bug.

So the plan passes a **call token** into every program as an argument, and the host looks the call's settings up by token (principle P11: per-call values are data at every boundary).

**The problem.** Three things need to fit together.
1. **Callbacks cannot see the carried context.** An embedded call's host side finds its settings by token. Unless the call context registered under the token also holds the active `config` blocks and the Monte Carlo record, a callback on an XLA thread reads the session values instead of the block's, and its draws go unrecorded.
2. **"Its value at trace time" meets program reuse.** A program store reuses one compiled program across calls. A setting read inside a traced function and baked in would then leak from one call into the next. A setting that changes what a compiled program computes must be part of the program's cache key or be passed as an argument.
3. **Where executor settings are kept.** #587 adds `config.prefect`, which holds the Prefect task runner and auto-detects Ray or Dask. The execution plan's Q9 retires that auto-detection and turns Prefect's runners into a `prefect` executor, after the workshop. It is also open whether `executor` and `executor_options` get a `config.execution` section for session defaults.
4. **When Prefect's auto-detection goes.** Step E3 replaces the dispatch modes before the workshop, while the `prefect` executor comes after it. Retiring the auto-detection in E3 would remove behaviour before its replacement exists, which directive 8 rules out.

**Options.**
1. One call context, the execution plan's, that captures #587's carried context (its active blocks and its Monte Carlo record) when the call starts, and travels by token into callbacks and by shipping into worker processes. Settings that change a compiled program join the program-store key. No `config.execution` section for now. Prefect's auto-detection stays unchanged until the `prefect` executor replaces it.
2. Two mechanisms side by side.

**Recommendation.** Option 1, agreed with jhuggins before step E3 starts.
#586 and #587 merge first, and the execution plan is updated to cite `config` (item T2).

**Blocks:** E3, E4.

**Status:** in progress. Settled 2026-10-17:
1. There is one call context, and its core logic is implemented once. #587's carried context and the execution plan's call context become the same mechanism, with no duplicated or redundant code.
2. The field `record` of #587's `_CarriedContext` is renamed, since "record" collides with ProbPipe's `Record`; the class `_MonteCarloRecord` has the same collision.

An independent design agent is writing the precise contract to `plans/call-context-proposal.md`: what the context holds, its lifecycle and precedence, its transport across threads, processes, Prefect, and host callbacks, the rule for settings read while tracing, the names, and what changes in #587 versus the execution steps. Once the maintainer accepts it, it is folded into the execution plan, and a comment on #587 describes the unified design to jhuggins.

---

### D8. Prerequisites the execution plan names that have no PR

**Background.**
The execution plan's section 10 lists issues that must merge first or alongside.
None of them has a PR, so the plan cannot wait for them.

| Issue | What it is | What the pre-workshop steps need from it | Recommendation |
|---|---|---|---|
| #524 | Finishing the function engine. Includes moving `_execution_contract.py` into `_execution.py` and merging `_managed.py` into the existing key broker `_broker.py`. | E3 rewrites these modules, so moving them afterwards would rewrite them twice. | E3's first commit does the two moves mechanically, using Q16's name. The rest of #524 stays after the workshop. |
| #554 | Explicit-key randomness: a `key` parameter declares randomness, and the workflow supplies the key. | E6's `InferenceTarget` gives a method its run key through a `key` parameter (#554 §8). A `vectorized` callable receives stacked keys. | E6 implements the run-key delivery of #554 §8 only. The rest of #554 follows the workshop. |
| #522 | `LinOp` as a `Function`, and the differentiability claim. | Nothing, if Q14 is answered as recommended. | After the workshop, with the supplied derivatives. |
| #506 | What a function body receives: raw values by default. | Point sets carry raw values, which open PR #575 reads through the `raw` property. | Point sets present raw values; the rest of #506 after the workshop. |
| #451 | One n-ary dispatch registry class. | The executor registry has one key, so today's unary registry serves. | No dependency. |

**Status:** ✅ settled 2026-10-17: every recommendation in the table is accepted, with the partial #554. `deferred.md` § Open issues the push touches records what each issue keeps for after the workshop.

---

### D9. Which steps happen before the workshop

**Background.**
On 2026-10-16 you set the execution core: executor dispatch for traceable and opaque functions, and the public target interface.
The declarations and addressing work had no workshop cut yet.
The principle proposed on 2026-10-17: change everything a user types before the workshop, and leave behaviour a user rarely meets until after.

**The proposal.** The steps are defined in [`README.md`](README.md) § Steps.

| Step | What | Before the workshop? |
|---|---|---|
| S1 | One subscript reader: a `str` is one path, a list or tuple is a list of paths that keep their paths, `[]` returns groups, `in`, `marginal(d, paths)` | yes |
| S2 | `OutputSpec` as a mapping, `term_spec`, `project` and `without`, one reading of slots, the size shorthand, `InputSpec.merge` | yes |
| S3 | `&` and the two renames, one mechanical PR | yes, if D2 is settled |
| S4 | Batches store their element declaration | yes |
| G1 | Grammar B for the kinds the workshop uses | yes (D4) |
| G2 | Grammar B for random measures, random functions, Gaussian processes, and bootstrap laws; then the refusal of the old form | after (D4) |
| G3 | Paths through every kind, `Function[]`, `component` | after |
| G4 | `condition_on` declared as `without`; `law.without` | after |
| G5 | `[]` on batches of laws and on kernels | after |
| E1–E7 | The execution core of the plan's section 13 | yes |
| E8 | The JAX settings contract | yes, but the first to cut if time runs short |

**Status:** ✅ settled 2026-10-17. Before the workshop: S0–S4, G1, and E1–E8. After it: G2–G5 and the execution plan's post-workshop list. If time runs short, the cuts are E8 first, then E7, then the lifting and default-output parts of G1; S0–S4 and E1–E6 are never cut.

---

## T. Text fixes that need no new decision

### T1. "Consumer" means two things

#571 defines a **consumer** as a term that declares what it consumes: a `Function` through `input_spec`, or a `ConditionalDistribution` through `given_spec`.
It replaces the design's phrase "map-like kind".

The execution plan's glossary defines a **consumer** as the code that asked for a point set, such as a lifted call, a joint, or an inference method.
It uses the word eleven times in that sense.

Both senses would end up in `design/glossary.md`.
**Fix:** the execution plan says **requester** for the code that asks for a point set.
The declarations sense keeps "consumer", since it pairs with "producer".

### T2. The execution plan is out of date in six places

1. Its examples compose with `*` (`prior = Normal(...) * HalfNormal(...)`). They change to whatever D2 decides.
2. It cites `n_broadcast_samples`, which #586 replaces with `max_evaluations` for lifting and `num_samples` for Monte Carlo summaries. Q12 and issue #585 need the same change.
3. It cites Prefect's auto-detection at `core/config.py:104-121`, which #587 moves to `config.prefect`.
4. Its note says the questions are numbered Q1 to Q15; there are eighteen.
5. §5 and Q16 justify writing "random key" because "key" alone means a field key. The addressing plan removes that sense from the glossary, so the justification changes, though the decision of 2026-10-16 to tolerate "key" stands.
6. Its header and §12 say its verification scripts are in `.claude/plans/580-verification/` and that the document is kept in the repository. The first is a gitignored local path (item W4). The second becomes true with this PR.

### T3. "Selection" keeps one sense

The glossary defines *selection* twice: positions of a batch, and several paths of a law.
The addressing plan removes both senses, since `select` is a method with another meaning, and keeps *selection order* (II.7), the ranking by which a registry tries its methods.
The execution plan's "automatic selection" of an executor uses the II.7 sense, which stays.
**Fix:** the glossary keeps only *selection order*, and the execution plan's glossary entry for "automatic selection" points to it.

### T4. Grammar B changes how a bare `TermSpec` given as `output_spec` is read

`output_spec=` accepts a bare term spec as well as an `OutputSpec`.
Today's rule (V.1 in `design/05-functions.md`, and III.3) is that a bare `RecordSpec` exposes its fields, and any other bare spec is whole under the callable's name.
The execution plan relies on this when it writes `output_spec=NumericArraySpec((40,), "float32")`.
Grammar B's "defaults by kind" changes the rule for specs with parts: a bare law, function, or batch spec becomes exposed instead of whole.
**Fix:** the reconciled declarations text restates the rule for bare specs under grammar B, and V.1 and III.3 change with step G1. The execution plan's examples use base specs, so they are unaffected.

### T5. IV.2's rationale misattributes associativity

jhuggins's Rocq model finds that a chain of kernels is associative with or without IV.2's rule that the right operand's slots avoid the left operand's components.
That rule keeps the joint well formed, which is a different reason.
**Fix:** the design text of step S2 or S3 corrects IV.2's rationale.

### T6. A point and a managed work item name the same thing

V.8 keys every draw made inside a **managed work item**: a point evaluation, a cell of a sweep, or one evaluation of a lifted call (`functions/_managed.py`, `make_managed_work_items`).
Each work item adds a segment to the occurrence path, so a draw inside it gets the same key wherever the item runs.
The execution plan calls these same units **points**, and gathers them into point sets that an executor runs.
Once step E3 lands, every managed work item is a point, so the design would have two words for one thing.
**Fix:** the execution design text (step E1) replaces "managed work item" with "point" in V.8 and the glossary, and the module that merges `_managed.py` and `_broker.py` is named after what it does for a point (D6, Q16).

**Status:** ✅ settled 2026-10-17. "Point" replaces "managed work item" everywhere, and the module is `functions/_point_keys.py`.

---

## F. Steps that edit the same files

### F1. The order that avoids collisions

Several steps edit the same modules.
Run in the wrong order, each rebase redoes the other's work.

| Module | Steps that edit it |
|---|---|
| `functions/_broadcast.py`, `functions/_sweep.py`, `functions/_execution.py` | S2 (the renames: `_broadcast.py` reads `_component_name` 17 times), E3 (executors replace the dispatch modes), E5 (`map_points` in lifting and sweeps), S4 (sweeps store their element declaration), G1 (defaults by kind) |
| about 41 modules that read `.spec`, `components`, `exposes_record`, or `_component_name` | S2 |
| `core/_specs.py` | S2 (most of it, including `project`), S3 (`&` on declarations) |
| `core/_shapes.py`, `named_tree.py`, `Record`, `RecordSpec`, and the 9 modules that define their own `_PATH_SEP` | S1 |
| `core/_batch.py` | S1, S4 |
| `distributions/_distribution.py` (`__mul__`) | S3 |
| `distributions/_conditional.py` | S1 (`_PATH_SEP`), S2 (one reading of slots: `_slots_of`, `_complete_given_spec`), S3 (`&` replaces `__mul__`), E2 (its construction probe), E4 (building a kernel over an embedded call) |
| `distributions/_factored.py` | S1 (`_PATH_SEP`), S2 (the joint's event from its factors' components), S3, E5 (`map_points` at joint factors) |
| `distributions/_views.py` | S1, S2 (`_view_declaration` becomes `project`) |
| `functions/_function.py` | E2 (the probes it holds today), E3 |
| `values/_function_base.py` | E2 (claims on `Function`), E3 (the controls and the call context), S2 (the reading of `input_spec`) |

**Proposed order.**
1. S1 and E2 together. They share only `_conditional.py`, where S1 changes one constant and E2 the construction probe.
2. S2, announced in advance like S3, since its renames change about 41 modules.
3. E3 in two parts: the point sets and the registry in `core/` start alongside S2, and the rewrite of the dispatch in `_broadcast.py` and `_sweep.py` starts after S2 merges.
4. S3 at an announced time, after S2, so every open branch rebases once.
5. E4 and E5 after S2 and S3, since both edit `_conditional.py` and `_factored.py`.
6. S4 after E3, since both rewrite the sweep.

**Status:** recommended.

---

## W. Gaps in how the push works

### W1. CONTRIBUTING keeps planning files off branches

`CONTRIBUTING.md` § Opening the PR says scratch planning artifacts stay off the branch, and § Plan first puts a significant plan in a GitHub issue.
`plans/` is a deliberate exception: three plans that must agree, many PRs, and sessions that need the plans in their checkout.
**This PR** adds one sentence to CONTRIBUTING naming the exception and its lifecycle: `plans/` exists while a push runs and is deleted when its sections have moved into `design/`.
**Status:** recommended; the sentence is in this PR for review.

### W2. The tutorials notebooks block every PR

The CI job `notebooks (tutorials)` runs `docs/tutorials` whenever `probpipe/` changes, and it blocks the merge.
Directive 6 puts the tutorials outside this work, and you are rewriting them on your own branch (#548 and `dev/workshop-tutorials`).
Every breaking step will fail that job.

**Options.**
1. Make the tutorials leg advisory while the push runs, and restore it when the rewritten tutorials merge.
2. Each step makes minimal mechanical fixes to the tutorials, which then conflict with your rewrite.

**Recommendation.** Option 1. The `user_guide` and `get_started` legs stay blocking.
**Status:** open; your decision.

### W3. Who updates the User Guide when a step breaks it

The User Guide notebooks lag behind the code, and they run in a blocking CI leg.
**Recommendation:** each step updates the User Guide notebooks it breaks, mechanically and with no rewriting, so the leg stays green and the guide never documents a removed API.
**Status:** recommended.

### W4. The execution plan's verification scripts

Appendix A of the execution plan cites 21 scripts and a prototype (`proto/`), about 120 KB, in the gitignored `.claude/plans/580-verification/`.
They are evidence for the plan's ✅ claims, and the steps reuse the prototype's solver as a test stand-in.
They fail `ruff check` in 180 places, so copying them as they are would break the ruff gate.

**Options.**
1. Add them under `plans/execution-layer-verification/`, excluded from ruff.
2. Leave them local, and let each step port what it needs into `tests/`.

**Recommendation.** Option 2. The tests of each step replace them, and the plan keeps describing what they showed.
**Status:** open.

### W5. The Claude Docs copies

The execution plan and the addressing plan also exist as Claude Docs, and jhuggins has commented on the execution Doc (his comment on a lighter pyabc route is recorded in L4 below).
From this PR on, the files in `plans/` are the copies that change.
**Recommendation:** add a line at the top of each Doc pointing to the file in `plans/`, and stop editing the Docs.
**Status:** open; needs your permission to edit the Docs.

---

## L. Known gaps that can wait

These are recorded in [`deferred.md`](deferred.md) and become issues after the workshop.

- **L1. The Rocq model does not cover renames by path or paths inside slots.** The addressing plan asks to extend the model before the design text is final, which matters for steps G1 and G2.
- **L2. When kernels get `[]`.** `lik["y"]` raises until step G5. The design states the exception until then.
- **L3. Two bugs the addressing plan found.** The Gaussian route of `condition_on` accepts a given record with a missing or extra field, and inference on a law declared whole over a record fails with a wrong message. Grammar B removes the second one's cause.
- **L4. A lighter route for pyabc.** jhuggins's comment on the execution Doc (2026-10-16) proposes a `pyabc.sampler.Sampler` subclass that seeds NumPy's global generator per evaluation, instead of reimplementing pyabc's loop. A 60-line prototype gave bitwise-identical populations when run sequentially, with chunks in reverse order, on 4 processes, and with two batch sizes. The thread executor cannot run these points, since threads share NumPy's global generator. This would shrink the plan's post-workshop pyabc step.
- **L5. Two issues the execution plan proposes and nobody has opened:** deterministic fields of a joint (Q4), and a Gibbs method (Q10).
- **L6. An exact result lost to a budget** is tracked in #585, which needs the `max_evaluations` wording of item T2.
