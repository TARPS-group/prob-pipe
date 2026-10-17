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
| A | Decisions needed before implementation can start | A1–A9 |
| B | Text fixes that need no new decision | B1–B5 |
| C | Steps that edit the same files, and the order that avoids collisions | C1 |
| P | Process gaps in how the push itself runs | P1–P5 |
| E | Known gaps that can wait until after the workshop | E1–E6 |

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

The addressing plan states its scope against #571 explicitly, so the two were meant to fit.
They were written two days apart, however, and #571's text was not updated when grammar B was adopted.
The execution plan was written alongside them, and uses some words and examples that the other two have since changed.

---

## A. Decisions needed before implementation

### A1. There is no single current text for declarations

**Background.**
The declarations design exists in three texts:
1. #571's issue body (2026-10-13);
2. jhuggins's reply on #571 (2026-10-15), which accepts most of the body and changes four things;
3. the grammar-B addressing plan (2026-10-16), whose section "Where this plan meets #571" lists twelve points and changes #571's text at several of them.

None of the three has been updated to match the later ones.
An implementer reading #571 alone would build several things that the later texts replaced.

**The problem.** The texts disagree on nine points.

| # | Topic | #571 body | jhuggins's reply | Grammar B |
|---|---|---|---|---|
| a | Whether a declaration is whole or exposed | a property `exposes` with four values: `"whole"`, `"record"`, `"event"`, `"element"` | a boolean `exposes_parts` | no property: the spec decides, since a declaration is whole exactly when its spec is a base spec |
| b | `project` with several paths | a record of the nodes, each under its last segment, in the order given; two paths with the same last segment raise | as the body | each node stays at its own path, in the tree's order; a path inside another raises |
| c | `project` with one path to a group | the group, named whole by its last segment | as the body | the group's parts, exposed; `project(["g"])` keeps the name |
| d | Reading the term at a component | five cases, one per kind | a private method `_component` per kind | one rule, a public `t.component(c)`, equal to `t[address(c)]` |
| e | `batch["mu"]` on a batch of draws of `Normal("mu", ...)` | returns the batch | as the body | raises, since `mu` is a name and not a path; `draws.component("mu")` returns the batch |
| f | The spec of a component when the declared value is a law | the spec of one draw's value at the component | as the body | the spec of the law's view at the component, a `DistributionSpec` |
| g | A batch built from raw values | keeps a bare element spec with no components, so a batch of arrays cannot be exposed | as the body | every batch holds an element declaration; a base element is whole under the default name `element` |
| h | Equality of batches whose elements differ only in component names | not equal | as the body | equal: names stay out of equality, hashing, and the JAX pytree (item A3) |
| i | Merging declarations | `OutputSpec.merge`, `InputSpec.merge` | `&` replaces `*` and both `merge` methods (item A2) | uses `&` in its examples, and leaves the operator to #571 |

Two examples show what is at stake.

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

**Options.**
1. Write one reconciled text, `plans/declarations-and-addressing.md`, which replaces `declarations.md` and `addressing.md`. #571 and #573 then link to it.
2. Keep two texts and edit #571's body to match grammar B row by row.

**Recommendation.** Option 1.
The two plans meet in twelve places, so the boundary between them causes most of the disagreement.
You decided rows b and c on 2026-10-15, and adopted grammar B on 2026-10-16, so rows a to g follow grammar B.
Rows h and i wait on items A3 and A2.
Row a also depends on item A4: if jhuggins rules for grammar B′ instead of B, a declaration needs a flag again.

**Blocks:** S2, S3, S4, and every later step of the declarations and addressing track.

**Status:** open. The reconciled text is written once A2, A3, and A4 are settled.

---

### A2. Should `&` replace `*` for composition, and should the pending renames merge with it?

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

Three renames are also pending, all user-visible:
- `with_path_names` becomes `rename`, as in pandas and xarray (addressing plan, recommended);
- `FieldView` becomes `MarginalView`, since it covers groups and lists of paths (addressing plan, recommended);
- the parameter of `marginal(d, field)` becomes `paths` (addressing plan, decided).

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
Put `&` and the three renames in one mechanical PR (step S3), so the codebase changes its vocabulary once.
Option 3 contradicts the clean-break rule of item A5.

**Blocks:** S3. The reconciled declarations text (A1, row i).

**Status:** open; needs jhuggins.

---

### A3. Do a batch's component names stay out of equality and the JAX pytree?

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

**Status:** recommended; needs jhuggins.

---

### A4. Grammar B's open rulings, and whether to enforce grammar B before the workshop

**Background.**
Under grammar B, a whole declaration may name only a base spec.
`OutputSpec(params=RecordSpec(beta=(), sigma=()))` raises.
A law over that record either offers `beta` and `sigma`, or is wrapped as `OutputSpec(RecordSpec(params=RecordSpec(...)))`, whose draws are `{params: {beta, sigma}}`.
This keeps one set of paths: a term's subscripts, its rename keys, and its draws all agree.

jhuggins's component-interface doc models this in Rocq and leans toward B, but leaves three things open.

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
   The plan leans to the first: the body returns the record, and a pending type filled with parts raises.

4. **Timing.**
   Enforcing grammar B changes 679 tests in 46 files and about 31 library sites.
   It changes what users see in several places: the view of a group, laws declared whole over a record, random measures, Gaussian processes, and functions that return laws.

**Options for timing.**
1. Settle rulings 1–3 and the design text before the workshop, and enforce grammar B after it. The workshop material avoids the forms that change.
2. Enforce grammar B before the workshop.

**Recommendation.**
Ruling 1: B.
Ruling 2: such a view wraps, because narrowing the event rule reopens the defect that rule fixed (problem 4 of #571, a joint whose draws fail their declaration).
Ruling 3: the body returns the record.
Timing: option 1. The size and the open rulings make option 2 a risk to everything else on the schedule.

**Blocks:** the grammar-B step (G1 in the README, after the workshop under the recommendation). Row a of A1.

**Status:** open; rulings 1 and 2 need jhuggins.

---

### A5. Clean break, with no deprecation aliases?

**Background.**
ProbPipe has not released on PyPI.
#448 set the rule for 0.2.0 as breaking changes with no compatibility shims, and draft PR #549 prepares 0.2.0.
jhuggins's reply on #571 answers "no deprecated aliases" for its renames.
The addressing plan lists thirteen breaking changes and marks "a clean break" as open, to confirm with jhuggins.
The execution plan's Q2 asks whether `dispatch` and `max_workers` go before 0.2.0.

**The problem.**
Each plan asks the question separately, and the answer should be one rule for the push.
It also decides whether 0.2.0 waits for the push.

**Options.**
1. A clean break everywhere, each change recorded under `### Changed (breaking)` in the CHANGELOG, and 0.2.0 released after the push.
2. Deprecation warnings for one release.

**Recommendation.** Option 1.
A shim would also break directive 8 of the README, which forbids bridging code.

**Blocks:** S2, S3, E3.

**Status:** open; confirm with jhuggins.

---

### A6. The execution plan's open questions on the pre-workshop path

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
| Q16 | Names that collide with ProbPipe's vocabulary | *probe result* for a stored probe outcome; `functions/_work_scope.py` for the module of the workflow scope that work enters, instead of #524's `_broker.py` merge target; "the point's random key" with no separate name | E2, E3 |
| Q15 | The executor class's name | `PointExecutor`, since users often import `concurrent.futures.Executor` too | E3 |
| Q3 | Which callables an executor named on an operation applies to. `condition_on.using(executor="slurm")` must send the simulator to the cluster without the user rebuilding the model. | Every opaque callable inside the call. An executor set on a `Function` itself wins, so a cheap statistic can stay local. A mapping by label can come later. | E3 |
| Q2 | When `dispatch` and `max_workers` go | Before 0.2.0, removed outright (item A5) | E3 |
| Q4 | Host callbacks, staged execution, or both. Staged execution stores `G(θ)` as a deterministic field of a joint. | Callbacks now. Deterministic fields change what a joint's event holds, so they go to a separate issue (E5 below). | E4 |
| Q7 | What a failed point means | The call raises once its point set completes, naming the point, and a compiled program stops at once. Dropping failed points is opt-in, after the workshop, for point sets built in Python only. | E4 |
| Q14 | Whether a traceable model may use gradient methods without claiming gradients. A strict reading of V.11 would take NUTS away from every tutorial model. | Yes, as today. Claims only narrow it: an opaque step needs a supplied derivative. V.11 states this as an exception. | E6 |
| Q18 | A JAX setting, such as `jax_enable_x64`, changed during a call | Raise, naming the setting and both values | E8 |

Questions Q6 (the memo across calls), Q8, Q9 (Prefect), Q10 (a Gibbs method), and Q11 (who owns the differentiability claim) belong to post-workshop steps.

**Status:** open; your decision on each, with jhuggins on Q5 and Q14.

---

### A7. Per-call settings: #587's `config` blocks and the execution plan's call context

**Background.**
PR #587 (jhuggins, ready to merge after #586) groups the global settings under `probpipe.config` and adds temporary blocks:

```python
with probpipe.config.sampling.using(num_samples=4096):
    mean.using(method="monte_carlo")(y)
```

A block is stored in a context variable.
The call context that #547 carries into worker threads, which already carries the inherited `exact_only`, now carries the active blocks too.
#587's design text adds: "A setting read while JAX traces a function is its value at trace time."

The execution plan carries per-call settings the same way within Python: a call context in a context variable, inherited by worker threads.
Inside a compiled program it cannot, for two reasons its verification found:
- under `pmap`, a host callback runs on XLA's own threads, which cannot see context variables;
- a value read while tracing becomes a constant of the compiled program, and a cached program then serves later calls with the first call's value. The plan's prototype hit this bug.

So the plan passes a **call token** into every program as an argument, and the host looks the call's settings up by token (principle P11: per-call values are data at every boundary).

**The problem.** Three things need to fit together.
1. **Callbacks cannot see `config` blocks.** An embedded call's host side finds its settings by token. Unless the call context registered under the token also holds the active `config` blocks, a callback on an XLA thread reads the session values instead of the block's.
2. **"Its value at trace time" meets program reuse.** A program store reuses one compiled program across calls. A setting read inside a traced function and baked in would then leak from one call into the next. A setting that changes what a compiled program computes must be part of the program's cache key or be passed as an argument.
3. **Where executor settings are kept.** #587 adds `config.prefect`, which holds the Prefect task runner and auto-detects Ray or Dask. The execution plan's Q9 retires that auto-detection and turns Prefect's runners into a `prefect` executor, after the workshop. It is also open whether `executor` and `executor_options` get a `config.execution` section for session defaults.

**Options.**
1. One call context, the execution plan's, that captures #587's active blocks when the call starts and travels by token into callbacks and by shipping into worker processes. Settings that change a compiled program join the program-store key. No `config.execution` section for now.
2. Two mechanisms side by side.

**Recommendation.** Option 1, agreed with jhuggins before step E3 starts.
#586 and #587 merge first, and the execution plan is updated to cite `config` (item B2).

**Blocks:** E3, E4.

**Status:** open; needs jhuggins.

---

### A8. Prerequisites the execution plan names that have no PR

**Background.**
The execution plan's section 10 lists issues that must merge first or alongside.
None of them has a PR, so the plan cannot wait for them.

| Issue | What it is | What the pre-workshop steps need from it | Recommendation |
|---|---|---|---|
| #524 | Finishing the function engine. Includes moving `_execution_contract.py` into `_execution.py` and merging `_managed.py` into `_broker.py`. | E3 rewrites these modules, so moving them afterwards would rewrite them twice. | E3's first commit does the two moves mechanically, using Q16's name. The rest of #524 stays after the workshop. |
| #554 | Explicit-key randomness: a `key` parameter declares randomness, and the workflow supplies the key. | E6's `InferenceTarget` gives a method its run key through a `key` parameter (#554 §8). A `vectorized` callable receives stacked keys. | E6 implements the run-key delivery of #554 §8 only. The rest of #554 follows the workshop. |
| #522 | `LinOp` as a `Function`, and the differentiability claim. | Nothing, if Q14 is answered as recommended. | After the workshop, with the supplied derivatives. |
| #506 | What a function body receives: raw values by default. | Point sets carry raw values, as #575 already makes `raw` a property. | Point sets present raw values; the rest of #506 after the workshop. |
| #451 | One n-ary dispatch registry class. | The executor registry has one key, so today's unary registry serves. | No dependency. |

**Status:** recommended; your confirmation, especially for the partial #554.

---

### A9. Which steps happen before the workshop

**Background.**
On 2026-10-16 you set the execution core: executor dispatch for traceable and opaque functions, and the public target interface.
The declarations and addressing work had no workshop cut yet.
The principle proposed on 2026-10-17: change everything a user types before the workshop, and leave behaviour a user rarely meets until after.

**The proposal.** The steps are defined in [`README.md`](README.md) § Steps.

| Step | What | Before the workshop? |
|---|---|---|
| S1 | One subscript reader: a `str` is one path, a list or tuple is a list of paths that keep their paths, `[]` returns groups, `in`, `marginal(d, paths)` | yes |
| S2 | `OutputSpec` as a mapping, `term_spec`, `project` and `without`, one reading of slots, the size shorthand, `InputSpec.merge` | yes |
| S3 | `&` and the three renames, one mechanical PR | yes, if A2 is settled |
| S4 | Batches store their element declaration | yes if A3 is settled by the end of day 2, otherwise the first item after |
| G1 | Grammar-B enforcement and wrapping | after (A4) |
| G2 | Paths through every kind, `Function[]`, `component` | after |
| G3 | `condition_on` declared as `without`; `law.without` | after |
| G4 | `[]` on batches of laws and on kernels | after |
| E1–E7 | The execution core of the plan's section 13 | yes |
| E8 | The JAX settings contract | yes, but the first to cut if time runs short |

**Status:** open; your decision.

---

## B. Text fixes that need no new decision

### B1. "Consumer" means two things

#571 defines a **consumer** as a term that declares what it consumes: a `Function` through `input_spec`, or a `ConditionalDistribution` through `given_spec`.
It replaces the design's phrase "map-like kind".

The execution plan's glossary defines a **consumer** as the code that asked for a point set, such as a lifted call, a joint, or an inference method.
It uses the word eleven times in that sense.

Both senses would end up in `design/glossary.md`.
**Fix:** the execution plan says **requester** for the code that asks for a point set.
The declarations sense keeps "consumer", since it pairs with "producer".

### B2. The execution plan is out of date in four places

1. Its examples compose with `*` (`prior = Normal(...) * HalfNormal(...)`). They change to whatever A2 decides.
2. It cites `n_broadcast_samples`, which #586 replaces with `max_evaluations` for lifting and `num_samples` for Monte Carlo summaries. Q12 and issue #585 need the same change.
3. It cites Prefect's auto-detection at `core/config.py:104-121`, which #587 moves to `config.prefect`.
4. Its header and §12 say its verification scripts are in `.claude/plans/580-verification/` and that the document is kept in the repository. The first is a gitignored local path (item P4). The second becomes true with this PR.

### B3. "Selection" keeps one sense

The glossary defines *selection* twice: positions of a batch, and several paths of a law.
The addressing plan removes both senses, since `select` is a method with another meaning, and keeps *selection order* (II.7), the ranking by which a registry tries its methods.
The execution plan's "automatic selection" of an executor uses the II.7 sense, which stays.
**Fix:** the glossary keeps only *selection order*, and the execution plan's glossary entry for "automatic selection" points to it.

### B4. How a bare `TermSpec` given as `output_spec` is read

The execution plan's examples declare `output_spec=NumericArraySpec((40,), "float32")`, a bare term spec rather than an `OutputSpec`.
Neither #571 nor the addressing plan says how a bare term spec given as `output_spec` is read.
Under grammar B the natural reading is a whole declaration under the function's default name for a base spec, and an exposed declaration otherwise.
**Fix:** the reconciled declarations text states this reading, and the execution plan cites it.

### B5. IV.2's rationale misattributes associativity

jhuggins's Rocq model finds that a chain of kernels is associative with or without IV.2's rule that the right operand's slots avoid the left operand's components.
That rule keeps the joint well formed, which is a different reason.
**Fix:** the design text of step S2 or S3 corrects IV.2's rationale.

---

## C. Steps that edit the same files

### C1. The order that avoids collisions

Several steps edit the same modules.
Run in the wrong order, each rebase redoes the other's work.

| Module | Steps that edit it |
|---|---|
| `functions/_broadcast.py`, `functions/_sweep.py`, `functions/_execution.py` | E3 (executors replace the dispatch modes), E5 (`map_points` in lifting and sweeps), S4 (sweeps store their element declaration), G1 (defaults by kind) |
| `core/_specs.py` | S1 (`project` reads subscripts), S2 (most of it), S3 (`&` on declarations) |
| `core/_batch.py` | S1, S4 |
| `distributions/_conditional.py` | S2 (one reading of slots: `_slots_of`, `_complete_given_spec`), S3 (`&` replaces `__mul__`), E4 (building a kernel over an embedded call) |
| `distributions/_factored.py` | S2 (the joint's event from its factors' components), S3, E5 (`map_points` at joint factors) |
| `distributions/_views.py` | S1, S2 (`_view_declaration` becomes `project`) |
| `values/_function_base.py` | E2 (claims on `Function`), E3 (the controls and the call context), S2 (the reading of `input_spec`) |

**Proposed order.**
1. S1 and E2 together: they share no module.
2. S2 and E3 together: they share only `values/_function_base.py`, in different functions.
3. S3 at an announced time, after S2, so every open branch rebases once.
4. E4 and E5 after S2 and S3, since both edit `_conditional.py` and `_factored.py`.
5. S4 after E3, since both rewrite the sweep.

**Status:** recommended.

---

## P. Process gaps

### P1. CONTRIBUTING keeps planning files off branches

`CONTRIBUTING.md` § Opening the PR says scratch planning artifacts stay off the branch, and § Plan first puts a significant plan in a GitHub issue.
`plans/` is a deliberate exception: three plans that must agree, many PRs, and sessions that need the plans in their checkout.
**This PR** adds one sentence to CONTRIBUTING naming the exception and its lifecycle: `plans/` exists while a push runs and is deleted when its sections have moved into `design/`.
**Status:** recommended; the sentence is in this PR for review.

### P2. The tutorials notebooks block every PR

The CI job `notebooks (tutorials)` runs `docs/tutorials` whenever `probpipe/` changes, and it blocks the merge.
Directive 6 puts the tutorials outside this work, and you are rewriting them on your own branch (#548 and `dev/workshop-tutorials`).
Every breaking step will fail that job.

**Options.**
1. Make the tutorials leg advisory while the push runs, and restore it when the rewritten tutorials merge.
2. Each step makes minimal mechanical fixes to the tutorials, which then conflict with your rewrite.

**Recommendation.** Option 1. The `user_guide` and `get_started` legs stay blocking.
**Status:** open; your decision.

### P3. Who updates the User Guide when a step breaks it

The User Guide notebooks lag behind the code, and they run in a blocking CI leg.
**Recommendation:** each step updates the User Guide notebooks it breaks, mechanically and with no rewriting, so the leg stays green and the guide never documents a removed API.
**Status:** recommended.

### P4. The execution plan's verification scripts

Appendix A of the execution plan cites 21 scripts and a prototype (`proto/`), about 120 KB, in the gitignored `.claude/plans/580-verification/`.
They are evidence for the plan's ✅ claims, and the steps reuse the prototype's solver as a test stand-in.
They fail `ruff check` in 180 places, so copying them as they are would break the ruff gate.

**Options.**
1. Add them under `plans/execution-layer-verification/`, excluded from ruff.
2. Leave them local, and let each step port what it needs into `tests/`.

**Recommendation.** Option 2. The tests of each step replace them, and the plan keeps describing what they showed.
**Status:** open.

### P5. The Claude Docs copies

The execution plan and the addressing plan also exist as Claude Docs, and jhuggins has commented on the execution Doc (his comment on a lighter pyabc route is recorded in E4 below).
From this PR on, the files in `plans/` are the copies that change.
**Recommendation:** add a line at the top of each Doc pointing to the file in `plans/`, and stop editing the Docs.
**Status:** open; needs your permission to edit the Docs.

---

## E. Known gaps that can wait

These are recorded in [`deferred.md`](deferred.md) and become issues after the workshop.

- **E1. The Rocq model does not cover renames by path or paths inside slots.** The addressing plan asks to extend the model before the design text is final, which matters for step G1 and later.
- **E2. When kernels get `[]`.** `lik["y"]` raises until step G4. The design states the exception until then.
- **E3. Two bugs the addressing plan found.** The Gaussian route of `condition_on` accepts a given record with a missing or extra field, and inference on a law declared whole over a record fails with a wrong message. Grammar B removes the second one's cause.
- **E4. A lighter route for pyabc.** jhuggins's comment on the execution Doc (2026-10-16) proposes a `pyabc.sampler.Sampler` subclass that seeds NumPy's global generator per evaluation, instead of reimplementing pyabc's loop. A 60-line prototype gave bitwise-identical populations across executors. This would shrink the plan's post-workshop pyabc step.
- **E5. Two issues the execution plan proposes and nobody has opened:** deterministic fields of a joint (Q4), and a Gibbs method (Q10).
- **E6. An exact result lost to a budget** is tracked in #585, which needs the `max_evaluations` wording of item B2.
