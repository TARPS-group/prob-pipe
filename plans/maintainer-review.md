# For the maintainer: what needs your opinion

> **Answered 2026-10-18:** the maintainer accepted every recommendation below (`decisions.md`). This file is deleted before the `plans/` PR merges.

This file lists what needs your review after the work of 2026-10-17. It is a working file of the `plans/` PR and is deleted before that PR merges.
Answer in chat by item number, for example "M2: accept OD1–OD8 as recommended, but OD5 option B".
Each item says why it matters, what is recommended, and where to read more.

## What happened while you were away

1. **The reconciled plan was written.** [`declarations-and-addressing.md`](declarations-and-addressing.md) replaces #571's body, jhuggins's reply, and the grammar-B plan, following the precedence you set (grammar B, then the reply, then the body). It defines everything from first principles and gives each step a done-condition.
2. **The call-context proposal was reviewed and revised three times.** [`call-context-proposal.md`](call-context-proposal.md), § 13, lists every revision.
3. **Four adversarial reviews ran, each with scratch tests against JAX and the code:**
   - **Round 1:** the execution plan and the call context. It found two blockers: no token was passed to embedded calls inside the inference runners, and the detection of JAX programs with embedded calls was unsound.
   - **Round 2a:** the reconciled plan.
   - **Round 2b:** consistency across every plan.
   - **Round 3:** verification of all the fixes.

   Every confident error the reviews found is fixed, and each fix is logged in the file it changed. What remains below is either a decision for you or a change made on your behalf that you should confirm.
4. **The branch is up to date.** `dev/overhaul` (with #575, #474, and #583) is merged into it, and everything is pushed to [PR #593](https://github.com/TARPS-group/prob-pipe/pull/593). `tests/docs/` passes.

---

## A. Decisions that block the plan

### M1. The workshop date and the code freeze

**Why.** The plans say "about two weeks" and "October 2026". The cut order (E8, then E7, then G1b) has no trigger without a date. The critical path is eight merges long on each chain, with no slack ([`README.md`](README.md) § Steps).

**Asked.** The workshop date and the freeze date. The charter then sets a checkpoint two working days before the freeze, where the cut is decided.

**Answered 2026-10-18:** the workshop is on 2026-10-27, and the freeze is at the end of 2026-10-25. Checkpoints on 2026-10-20 (progress) and 2026-10-22 (the cut).

### M2. The call-context plan

**Why.** It is the one mechanism that carries per-call state (`config` blocks, inherited controls, the `num_samples` its Monte Carlo estimates used, the executor choice, the call token, the JAX settings) into threads, processes, Prefect, and host callbacks inside compiled JAX programs. It unifies jhuggins's #587 with the execution plan, so nothing is built twice. Read the proposal in full; its § 1 summarizes it in eleven statements.

**The design in brief:**
- **One value.** One frozen `_CallContext`, in one context variable, in `core/_call_context.py`. #587's three context variables become three of its fields: `config_blocks`, `passed_controls`, and `num_samples_used`. The execution steps add their own fields to the same type.
- **Who opens one.** Every `Function` call opens a child context. A `config.using` block or a derived route's scope derives a changed copy and opens no call.
- **One precedence rule:** the call's explicit setting, then a value passed down, then the innermost block, then the session value, then the default. `exact_only` accumulates.
- **Transport:**
  - within a thread, the context variable;
  - across threads, Prefect, and processes, the evaluation context as picklable data;
  - into compiled JAX programs, a `uint32` call token, looked up in a table.
- **Tokens only where they are needed.** A call registers a token only when it runs a program that stages an embedded call, found by searching the program's jaxpr. Only such a call waits for its programs before it closes, so every other call keeps JAX's asynchronous dispatch.
- **Settings while tracing.** Each `config` setting declares its kind:
  - *traced*, such as `num_samples`: part of the compiled JAX program's key;
  - *presentation*, such as `notation.max_depth`: readable anywhere, never part of a key.
- **Renames:**
  - `record` → `num_samples_used`;
  - `_MonteCarloRecord` → `_NumSamplesUsed`;
  - `_CarriedContext` → `_ContextBoundFunction`.

**The open decisions** (each is explained in § 11 of the proposal):

| | Decision | Recommendation |
|---|---|---|
| OD1 | `config` blocks are a field of the one context, not a separate variable | Accept |
| OD2 | The provenance mode becomes a per-call setting, which fixes a bug in #587 (M4) | Accept, as step PV |
| OD3 | A callback in a JAX program the user compiles runs under the default context on every thread | Accept: otherwise `jit` and `pmap` read different settings |
| OD4 | Process workers return the `num_samples` their Monte Carlo estimates used with their outcomes, starting in E3a | Accept |
| OD5 | #587 creates the one context before it merges, rather than a follow-up PR (M3) | Accept |
| OD6 | A compiled JAX program's key holds the passed controls and the static settings only | Accept (option A′) |
| OD7 | An embedded call inside a program the `jax` executor compiles uses the call's executor, so `growth.using(executor="slurm")` sends the opaque `forward` inside it to the cluster | Accept; Q8 then covers only calls an evaluation point makes in Python |
| OD8 | The program store holds `jax.jit` objects, which JAX itself keys by its global settings | Accept |

**Asked.**
1. Accept or amend OD1–OD8.
2. Approve folding the accepted proposal into `execution-layer.md` (step CC, which gates E1 and E3a).

### M3. How #587 absorbs the call context, and the comment to jhuggins

**Why.** OD5 asks jhuggins's ready-to-merge #587 to restructure its context code, about a day of work. Step P0b is that work.

**Options.**
1. #587 does it before merging. You could push the commits to his branch yourself; the draft comment offers this.
2. A follow-up PR after #587.

**Recommended.** Option 1.

**Asked.**
1. Pick an option.
2. Approve posting the draft comment in the proposal's § 12 to #587, after you have read it. Nothing has been posted.

### M4. The provenance-mode bug (step PV)

**Why.** Inside an open `workflow_run`, `config.provenance.using(mode=OFF)` reads OFF, but provenance is still recorded, because the run keeps the mode it copied when it opened (verified in #587's code).

**Options.**
1. The mode becomes a per-call setting, in a small PR after #587.
2. The mode is fixed for a whole run, and a block inside a run raises.

**Recommended.** Option 1, unless the per-run rule matters to replay. The draft comment asks jhuggins exactly this.

### M5. The reconciled plan's open points

**Why.** Implementers of S2a through G1b would hit each of these. Each is explained from scratch in [`declarations-and-addressing.md`](declarations-and-addressing.md) § 21. The text already follows each recommendation, marked with the item's number.

| | Question | Recommendation |
|---|---|---|
| O1 | Does `project` wrap at a field that holds a law or a function? Wrapping there breaks I13 for functions (`fit["posterior"]` would be declared a record while `fit(data)["posterior"]` is the law). | No. `project` never wraps, and a law's view adds the wrap on top. |
| O2 | One naming rule for outputs whose one component is at the root | A producer names only what has no name of its own: a base term, or a batch whose element has only the default name `element`. Otherwise the returned term keeps its names. This **amends your decision** that a type hole filled with any spec with parts raises. |
| O3 | How a batch's whole-element name is stored | In `BatchSpec`, whose own equality is name-sensitive. A batch's equality, fingerprints, and every pytree flatten use a name-free canonical form, recursively. This departs from the grammar-B plan's "beside the spec, as a label is", which leaked names through nested specs in a test. |
| O4 | Do executors restore the declarations of their inputs? | Not before the workshop; sets of evaluation points present raw values. |
| O5 | Views at fields that hold a batch or a kernel | Extend the wrap to them, as for laws and functions. |
| O6 | `Normal("x")[["x"]]` on a whole base term | Accept the whole name and return the wrap, which is what today's tuple form gives. |
| O7 | `&` merges records but is elementwise on arrays | Keep both, and document the difference as a § 16 exception. |
| O8 | How `BatchSpec` takes an element declaration | Its first argument accepts an `OutputSpec` or a bare `TermSpec`. |
| O9 | Optional slots for `Function.input_spec` | The kernel's rule: parameters with defaults are optional. |
| O10 | An empty `RecordSpec` in an `OutputSpec` | Refuse it. |
| O11 | Should `produced_declaration` be public? | No, private. |
| O12 | Joint labels | Keep `·`. |
| O13 | `with_level_names` and `with_dim_names` | Leave them, and defer `rename_levels` and `rename_dims`. |
| O14 | `t.component(c)` next to a mixture's component laws | Keep both senses, and defer renaming the mixture's `components=`. |
| O15 | The name `BatchSpec.element` | Rename it `element_declaration`, since "element" also means one item of a batch. |
| O16 | G2's refusal must follow G3 | Resolved in the charter. |

The plan also lists R1–R6: points the charter has already resolved. Confirm them in passing; R4, the timing of the `STYLE_GUIDE.md` changes, is the only real choice among them.

### M6. The restructured steps

**Why.** The reviews found that several steps mixed mechanical renames with new behaviour, against the charter's own rule, or put cuttable work in front of work that must not be cut.

**The new steps**, all in [`README.md`](README.md) § Steps:

| Step | What it is |
|---|---|
| S2a, S2b | S2 split: S2a is the announced mechanical rename (`term_spec`, the mapping interface, about 50 modules); S2b is the behaviour |
| E0 | A new mechanical step: #524's module moves, and "managed work item" renamed "evaluation point" in the code |
| E3a, E3b | E3 split: E3a builds the `core/` pieces, and E3b rewrites dispatch after S2a |
| G1a, G1b | G1 split: G1a covers group views, library laws, and grouped sub-joints, and is never cut; G1b covers lifting and default outputs, comes after E5, and is cut third |
| CC | Accept and fold the call context; it gates E1 and E3a |
| P0b, PV | The work of M3 and M4 |

**Consequences.**
- **E4 is larger.** It now threads the token through the inference runners, which were post-workshop work. Without this, `condition_on.using(method="blackjax_rwmh", executor="slurm")` would silently run sequentially.
- **S3 and E4 may merge in either order.**

**Asked.** Confirm the structure and the cut order.

**Update 2026-10-18.** You chose to aim for every step, with the fundamental steps first and clean off-ramps. `README.md` § Steps now proposes:
- **never cut:** CC, P0b, S0, S1, S2a, S2b, S3, E0, E1, E2, E3a, E3b, E4, and E6;
- **off-ramps**, cut in order: E8, E7, G1b, E5, G1a, S4.

Two judgements to check:
- **E6 is in the never-cut group**, because you named the target interface as core on 2026-10-16.
- **E5 is cut before G1a and S4**, because E4's embedded calls already let opaque code run inside joints and lifts; E5 only consolidates the remaining "vectorize or loop" sites.

### M7. A GitHub tracking issue for step status

**Why.** If every step PR edited the status table in `README.md`, parallel PRs would conflict, and a row would show its state only after its PR merged. `CONTRIBUTING.md` already names a tracking issue's checklist as the tracker for a multi-PR plan.

**Asked.** May a tracking issue be created when this PR merges? It is an outward-facing action, so it waits for your approval.

### M8. When this PR may merge

**Recommended.** It merges once every gaps item except D7 is settled and O1–O16 above are answered. D7, the call context, then gates only the execution steps, through step CC, so the declarations track can start without waiting for it.

**Asked.** Confirm, or keep D7 as a condition of this PR.

---

## B. Changes made on your behalf: please confirm

### M9. The execution plan ([`execution-layer.md`](execution-layer.md); Appendix D logs every change)

- **§ 13 restated.** It now lists steps E0–E8 with done-conditions, and points to the charter for order and cut.
- **Section 14 rows tagged by step.** The acceptance example that ran a tutorial now uses a User Guide model.
- **Corrections the call-context work verified:**
  - failures are decided by the failures recorded on the call, never by the exception's type;
  - tokens are `uint32`;
  - a call with a token waits for its programs before releasing the token;
  - a `process` worker's initializer applies four JAX settings before it unpickles anything, since settings applied after unpickling turned float64 into float32 in a test.
- **Marked post-workshop:** the common executor options `retries`, `retry_errors`, and `bisect` (until then an executor refuses them as unknown), SBC's evaluation points, one per inference run, and presenting arguments in the form #506 specifies (until then sets of evaluation points present raw values).
- **Moved into the core:** the output check per evaluation point and its warning when a `float64` output is cast to `float32`, in E4.

### M10. The call-context revisions beyond the first draft

Read the proposal's § 13:
- tokens only for JAX programs with embedded calls;
- the narrowed cache key (OD6);
- the name `_ContextBoundFunction`;
- `embedded_calls` in place of `callbacks`, since `jax.debug.print` is also a callback;
- presentation settings stay readable while tracing, because labelled terms read `notation.max_depth` inside traced code today;
- the program store's full key arrives with the store in E3a;
- detection traces with `jax.jit(f).trace(...)`, because a lowered program carries no jaxpr.

### M11. The reconciled plan's provisional resolutions

Beyond the open points of M5:

- **The decision that the wrap keeps the field's name** has consequences. I2, I5, and I9 hold only up to the wrap for laws whose draws hold laws: for example, `rm["G"]["mu"]` raises while `rm["G/mu"]` works. § 16 now states this as an exception.
- **Two defaults.** An output default changes in G1b. A private event default keeps today's whole form for random measures, bootstrap laws, and empirical laws of laws until G2. Without the split, G1 would silently change the families you deferred.
- **Behaviour before G2 and G3.** A whole declaration over parts, legal until G2, keeps today's component-rooted paths (`params/beta`). `Normal("x")["x"]` keeps working until G3.
- **I10 is narrowed.** `without` exists for records, record specs, batches, declarations, and laws only. The other kinds are deferred.
- **`STYLE_GUIDE.md` changes with S1, S3, and G5,** not with S0, so the guide never describes code that does not exist yet (R4).

### M12. Process

- **The handoff protocol is rewritten.**
  - It works for steps with several dependencies.
  - It allows stacking on exactly one open dependency, and stops when there are several.
  - Two handoffs for the same step are caught by the tracking issue.
  - This session creates the handoffs for S0, CC, and E0 when the PR merges.
  - Where the chip tool is missing, the fallback prompt goes in a PR comment, not the PR body.
- **The superseded sources.** `declarations.md` and `addressing.md` stay for comparison and are deleted before merge, if you agree.
- **CI.** The tutorials notebooks are advisory. The `get_started` and User Guide notebooks stay blocking, and steps keep them running.
- **`CONTRIBUTING.md`** gains one sentence naming `plans/` as the exception to keeping planning files off branches.

---

## C. For jhuggins

| Topic | Status | Action |
|---|---|---|
| #587 and the call context (OD5, OD2) | waits on M2–M4 | post the § 12 comment once approved |
| `&` replaces `*`, `rename`, `MarginalView` | decided; `&` was his proposal | tell him |
| Grammar B, not B′; the wrap keeps the field's name | decided by you | tell him, since he left B versus B′ open in his Rocq document |
| Batch names stay out of equality | decided; reverses a sentence of #571 | tell him |
| Q5 (code in `core/`) and Q14 (traceable models keep gradient methods) | accepted | tell him; both were marked worth agreeing with him |
| The pyabc route he proposed on the execution Doc | recorded in `deferred.md` | none before the workshop |

---

## D. Findings worth knowing

None of these needs a decision before the workshop.

- **"Point" became "evaluation point"** on 2026-10-18, a decision of the maintainer and jhuggins, and the plans use the new terms.
- **#571's reply is partly wrong about `*`.** It motivates `&` by saying `*` multiplies records, but `NumericRecord` has no `*`. `&` on float arrays raises `TypeError` rather than computing an elementwise "and". The reconciled plan states both.
- **`GaussianRandomFunction.__mul__` both composes and scales.** S3 keeps only the scaling.
- **VI.8 orders a marginal's factors by path order** today (`marginal(model, ("b", "a"))` gives `b(b)·a(a)`). This changes under the tree-order rule.
- **II.5 already specifies** `BatchSpec.element` and `Batch.components`, so the design is ahead of the code there.
- **#587's PR text gives an inaccurate reason** for carrying the context explicitly. Prefect's thread runner does copy the context; the real reasons are pickling for process runners and keeping the workflow frame out. The draft comment says so.
- **The verification scripts are kept locally** in the main checkout's gitignored `.claude/plans/` (execution plan and call context).
- **Unverified:**
  - the grammar-B plan's sweep counts (1,046 and 679 tests);
  - the Rocq lemma claims;
  - any behaviour on a GPU or TPU;
  - Prefect with Ray or Dask runners.
- **Two published HTML artifacts carry older copies of the plans** ("ProbPipe execution layer design (#580)" and "ProbPipe addressing plan, grammar B"). Only the two Claude Docs got a pointer note. Say if the artifacts should get one too.

---

## E. New since your review: provisional decisions from the call-context review (2026-10-18)

Two adversarial reviews of the folded call context (`execution-layer.md` § 6.16) found two blockers and a dozen major issues, all fixed. Five fixes needed a design choice; each is written into § 6.16 as the design and listed in its "Decisions" table, awaiting your confirmation.

| | Choice made | Why | Alternative |
|---|---|---|---|
| M13 | **Decided 2026-10-18:** settings are **static** (part of the cache key: `sampling.*`) or **inert** (cannot change a computed value: `notation.*`, `provenance.*`, `prefect.*`); a call opened inside a staging trace runs inline | | |
| M14 | **Decided 2026-10-18:** notes made while tracing are captured and applied once per run; a later trace capturing a different note raises | | |
| M15 | **Decided 2026-10-18:** shipped work carries the session's settings in effect when the call opened | | |
| M16 | **Decided 2026-10-18:** two merge rules: derived routes collect their formula's calls; evaluation points always merge into their call, on success | | |
| M17 | **Decided 2026-10-18:** executor and options resolve as one unit; `using(executor=None)` restores inheritance; opt out by naming an executor; traceable callables always `jax` | | |

Smaller choices the reviser made, for a glance:
- **The evaluation run table (E7):** bounded at 65,536 entries with least-recently-used eviction. Concurrent misses on one evaluation point de-duplicate in flight.
- **Root calls:** marked by a `call_depth` field and `is_root_call()`.
- **Process pools:** keyed by the shipped JAX settings and reused across calls.
- **Renames in P0b:** a table of renames for vague names in the code it changes, for example `_plan_point` → `_plan_route_request` and `_passing_down` → `_passing_controls_to_formula_calls`.
- **The formula rename:** P0b writes "identity", the term current on `dev/overhaul`, and E0 then renames every occurrence, P0b's text included, so there is no partial migration.
- **PV's new field:** PV adds `session_config_values` with the `provenance` section only, and E3a extends it to every section.

A second round of review then found no blockers but eight problems, all fixed, and a third round confirmed the fixes and found one more problem, also fixed. Together they added five more provisional choices:

| | Choice made | Alternative |
|---|---|---|
| M18 | **Decided 2026-10-18:** a backward pass after its call closed uses the default context, documented clearly when supplied derivatives land | |
| M19 | **Decided 2026-10-18:** refusal raises when the executor was named for this callable (`applies_only_to`), falls back with a warning when inherited | |
| M20 | **Decided 2026-10-18:** evaluation points read the snapshot taken when the root call opened, under every executor | |
| M21 | **Decided 2026-10-18:** `prefect` is never shipped to worker processes; `UnshippableSettingError` accepted; a Prefect review is deferred | |
| M22 | While ProbPipe is already tracing, an inner call's body runs inline in the outer trace instead of in a separately stored JAX program, so an executor named on a traceable `Function` inside a kernel (`growth.using(executor="slurm")`) reaches the opaque code it calls; only the outermost JAX program is cached. JAX compiles the inner code into the outer program anyway, so nothing is lost | Keep a separate inner JAX program and put the inner call's executor choice into its cache key |
