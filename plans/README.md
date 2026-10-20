# The contracts push

This folder holds the shared plans for a sequence of PRs that puts three areas of ProbPipe on precise contracts before the workshop of 2026-10-27:
1. **the execution layer:** how a computation runs, including code JAX cannot compile, and where (#580);
2. **addressing:** how a path or a list of paths picks out a part of a term (#573);
3. **inputs and outputs:** what terms produce and receive, the input specs and output specs that state it, and the operations on them, merge included (#571).

Earlier work in these areas was planned one change at a time, which left inconsistent designs, duplicated code, and an API that is hard to learn.
This push plans the three areas together, and every PR in it is judged against the whole plan, not only its own step.

Every session that works on the push reads this file in full before anything else.

## The files

| File | What it holds |
|---|---|
| `README.md` | This charter: the directives, the steps, their order and dependencies, the cut order, and the handoff protocol. It is the one owner of the order and the cut; every other file points here |
| [`decisions.md`](decisions.md) | Every decision taken during the push, with its date and reason, and the queue of questions waiting for the maintainer |
| [`gaps-and-conflicts.md`](gaps-and-conflicts.md) | The places where the plans disagree or leave a decision open. This PR merges once each is settled. |
| [`execution-layer.md`](execution-layer.md) | The execution-layer design (#580) |
| [`declarations-and-addressing-io.md`](declarations-and-addressing-io.md) | The declarations and addressing design (#571, #573) with the inputs and outputs plan merged in, provisionally, rewritten on 2026-10-19 with the design unchanged: Part I the concepts (the naming rule, inputs and outputs, paths and lists of paths, one definition of `&`), Part II the code, and the appendices, B holding steps S0, S1, S2a, S2b, S3, S4, G1a, G1b, and G2–G5 |
| [`declarations-and-addressing.md`](declarations-and-addressing.md) | The declarations and addressing plan as it stood before the inputs and outputs merge, kept unchanged for recovery until the maintainer accepts the merged copy; the git tag `plans-before-io` marks that state |
| [`inputs-and-outputs.md`](inputs-and-outputs.md) | The accepted proposal merged into `declarations-and-addressing-io.md`, kept for the proofs, alternatives, and checked facts that the merged copy cites |
| [`deferred.md`](deferred.md) | Work found or postponed during the push, to become issues after the workshop |

Four superseded sources are still on this branch and are deleted before this PR merges (step PL): `declarations.md` and `addressing.md`, which `declarations-and-addressing.md` replaced (gaps D1); `call-context-proposal.md`, folded into `execution-layer.md` § 6.16; and `maintainer-review.md`, whose recommendations `decisions.md` records. Git keeps their history.
`declarations-and-addressing-io.md` supersedes `declarations-and-addressing.md` provisionally (decisions, 2026-10-18). PL keeps `declarations-and-addressing.md` and `inputs-and-outputs.md` until the maintainer accepts the merge, and the next checkpoint PR deletes the recovery copy. What the merged plan's file is called after acceptance is pending the maintainer's decision N32 in `decisions.md`.

## Standing directives

These apply to every person and agent working on the push.
They add to `CONTRACTS.md`, `STYLE_GUIDE.md`, and the implementation guidelines of `execution-layer.md` §12, which still apply in full.

1. **Read the full design first.** Before working on a step, read this file, `decisions.md`, and the current plans in full: `declarations-and-addressing-io.md` and `execution-layer.md`, whose § 6.16 holds the call context. Read them whole, not only the sections the step implements. The superseded `declarations.md`, `addressing.md`, `call-context-proposal.md`, and `maintainer-review.md`, while they remain, are not plans; skip them. `declarations-and-addressing.md`, kept for recovery, and `inputs-and-outputs.md`, merged into the `-io` copy, are not current plans either; read them only for what the `-io` copy cites. Check that the work fits the broader goals of the push and of `design/`.
2. **Design gaps go to the maintainer.** The plans will have gaps that implementation exposes. A gap that needs a real design decision is never settled by the implementer alone: follow § When to stop and ask. A fix to a gap serves the long-term design, never only the step at hand.
3. **Contracts, not patches.** Write well-designed, coherent contracts. Never patch or work around a deeper design problem. If the problem is within the push's scope, fix it; otherwise record it in `deferred.md` for work after the workshop.
4. **PR descriptions written for people.** Give the context, and do not assume the reviewer knows ProbPipe's internals. Show specific examples of new behaviour, such as code before and after. Avoid jargon and dense blocks of text. The reasoning should flow, and the summary of changes should read naturally.
5. **Code quality is judged, not only behaviour.**
    - No duplicated code. Before writing a helper, search the codebase for one that already does the job. Reviewers check for duplicates across the codebase, not only within the diff.
    - Every name says what the thing is or does, private ones included, local variables and parameters too, and public names most of all. `STYLE_GUIDE.md` §1.12 states the general rule; this push adds five concrete ones:
        1. **Qualify a generic word with its object.** `current_context`, not `current`; `parent_context`, not `parent`; `config_values`, not `values`; `resolved_controls`, not `options`. Generic words such as `current`, `parent`, `data`, `result`, `value`, `item`, `obj`, `info`, `state`, `handle`, `helper`, `tmp`, and `other` never stand alone.
        2. **Name a function for what it does.** A wrapper is named for the work it does, as `run_in_child_call_context`, not for its position or the moment it runs, as `entered`, `wrapper`, `inner`, or `_impl`. `_combine` fails: combine what?
        3. **Single letters only for mathematics or a short loop.** `d`, `x`, and `theta` may name a law or a value as the mathematics names it, in an example or a formula that defines them; `i` may index a loop of a few lines. Nowhere else.
        4. **One name for one thing.** A name never reuses a term that ProbPipe defines for something else, such as "record", "event", "key", "component", or "element"; check `design/glossary.md` before choosing one.
        5. **Fix the vague names in the code a step changes.** A step renames, under rules 1 to 4, every vague name that already exists in the code it changes: in the functions, classes, and modules it edits, local variables and parameters included, and every use of each renamed name. A vague name outside that code is recorded in `deferred.md` rather than renamed piecemeal.
    - A public docstring is written for users, and carries no design reasoning.
    - A private object of central importance, such as `_CallContext`, gets a full docstring too: what it is, its lifecycle, its invariants, and each attribute, so a reader can follow the code without the plans.
    - An inline comment appears only where it adds something the code and the docstring do not already say.
    - A module puts its most important public objects first and its private helpers at the bottom.
6. **The tutorials are out of scope; the User Guide and Get Started are kept running.** `docs/tutorials/` is being rewritten separately, so no step edits it, and its CI notebook job is advisory until the rewrite merges. The notebooks of `docs/user_guide/` and `docs/get_started/` run in blocking CI jobs: a step that breaks one updates it mechanically, without rewriting it, so no page documents a removed API. A mechanical update includes replacing a renamed term in the page's prose and printed labels, such as "field view" after `FieldView` becomes `MarginalView`. A step's searches over `docs/` leave out `docs/tutorials/` and the CHANGELOG, and a PR that breaks a tutorial says so, since the advisory tutorials job may then fail. *Which branch the workshop's materials use, and how they follow the push, is pending the maintainer's decision N17 in `decisions.md`.*
7. **No partial migrations.** A PR that changes a name or a meaning changes every use of it in the same PR, so `dev/overhaul` is always coherent and could be used at the workshop at any merge.
8. **Deferring keeps the current behaviour.** Work postponed to after the workshop leaves the current code unchanged and gets an entry in `deferred.md`. It never gets bridging code, such as an alias, a shim, or a special case that serves only the interim.
9. **Tests and a demonstration in every PR.** Each PR adds the tests that assert its contract (shapes, orderings, and error cases, beyond the happy path), and its body includes a short runnable demonstration with its output.
10. **Every PR body says how it fits the plan.** A section "Fit with the plan" lists:
    - the plan sections the PR implements, each linked with a permalink at a commit (press `y` on GitHub), or, after PL merges, at `dev/overhaul`;
    - any departure from them;
    - the questions and decisions it raised, and where (§ When to stop and ask);
    - the existing code it reused, and any duplicate it removed.

    The title and the rest of the body name each change by its behaviour, never by a step label such as "S3" (`STYLE_GUIDE.md` §10, rule 9).
11. **One vocabulary.** Terms come from `design/glossary.md` and the plans. A PR that needs a new term adds it to the glossary in the same PR, and checks that it collides with no existing term.
12. **Broad reviews at checkpoints only.** `/review-all` runs four reviews and costs much time and many tokens. Run it when it is a good moment to assess the push broadly, such as after a group of steps has merged, never after each PR. A single PR gets `/review-pr` or `/check-pr`.
13. **Review before asking for review.** Open the PR as a draft and run `/review-pr` on it. When `plans/README.md` exists at the PR's base, the skill also checks the PR against these directives, `decisions.md`, and the step's plan section; when it does not, see § Ending a session, step 2. Fix each finding, or answer it in the PR, and then mark the PR ready for review. The maintainer reviews a PR once it is marked ready.
14. **Replace stale code rather than work around it.** This push overhauls these parts of ProbPipe, and the goal is code a person can read through and follow. Code that conflicts with the plans, or that has bad names, poor organization, or duplicated logic, is rewritten or removed, not patched around and not edited piecemeal to fit. Writing a clean replacement from scratch is often the better choice, within the step's scope; a larger rewrite outside it is recorded in `deferred.md` or raised with the maintainer.

## When to stop and ask

Ask the maintainer before doing any of the following:
- changing a public name, signature, or default that the step's plan section does not already state;
- changing a contract or an invariant that a plan or `design/` states;
- adding, removing, or redefining a term in the glossary;
- departing from an item that `decisions.md` records as settled;
- adding a special case, an alias, or a workaround to get past a problem.

To ask, comment on the tracking issue (§ Status), or, before it exists, list the question in the PR body under "Questions for the maintainer", and say so in the session. A step PR does not edit `decisions.md` (§ Ending a session); the checkpoint PR copies each question into the queue of `decisions.md` and each answer into its settled table.
Unless the question blocks all further work on the step, do not wait for the answer: make the most conservative choice consistent with the plans, state the choice and its alternative with the question, and go on.
Write the question for a reader without the session's context:
1. the background, from first principles;
2. today's behaviour and the proposed behaviour, in code;
3. the options, each explained;
4. a recommendation, with the reason.

A choice that changes none of the above, such as a private helper's structure, is the implementer's. Record it in the PR body.

## Branches and PRs

- Every step is one PR into `dev/overhaul`, from the branch the step tables below name, cut from `origin/dev/overhaul`. The one exception is P0b, a PR into #587's branch, `dev/config-sections` (§ Steps started before this PR merged).
- A step may stack on a dependency whose PR is open but not merged, and retargets `dev/overhaul` once the dependency merges. Say so in the PR body. If exactly one dependency is still open, the step stacks on it; if more than one is open, the session stops and tells the maintainer. *A rule that lets a step build in parallel with dependencies that only fix the merge order is pending the maintainer's decision N7 in `decisions.md`.*
- A mechanical PR, such as a rename across the codebase, stays separate from a PR that changes behaviour, so each is quick to review.
- An **announced** step is mechanical, and a script that its PR body includes, such as `sed` or `rope` commands, generates it. It merges at a time stated in advance on the tracking issue (§ Status), and the session tells the maintainer in its final message. In the two hours before that time no other PR of the push merges. A branch that conflicts with it afterwards does not resolve the conflict by hand: it merges `dev/overhaul` with the announced PR's changes discarded, then runs the announced PR's script on its own branch. When an announced PR goes stale, its author rebuilds it the same way, by running the script again on the current base, not by rebasing. The merge slots:
    - E0 merges right after #587;
    - S2a right after S1;
    - S3 right after S2b, before S4, G1a, E4, or E5 opens its PR.
- PRs of the push merge with a merge commit, never squash or rebase, because branches stack.
- Before merging, the merger updates the branch with `dev/overhaul` if `dev/overhaul` has moved since the branch's last CI run, and waits for CI to pass. `dev/overhaul` has no required checks and no rule that a branch be up to date, so a PR whose CI passed against an old base could otherwise break it without a textual conflict.
- When a PR merges, each PR stacked on it is retargeted by its session or the coordinator, with `gh pr edit <n> --base dev/overhaul` and then a merge of `origin/dev/overhaul`; GitHub does not retarget them, since merged branches are not deleted.
- Bring a draft up to date by rebasing. Bring a PR marked ready up to date by merging `dev/overhaul` into it, so a reviewer's history is kept.
- The maintainer merges PRs into `dev/overhaul`. jhuggins merges PRs into his own branches (P0b into #587).
- When the workshop's code freezes, `dev/overhaul` is tagged, and the workshop uses the tag.

## Steps

**Provisional** until this PR merges.
"Before" marks a step planned before the workshop.
A step may not start until every step and PR it depends on has merged into `dev/overhaul`, or it stacks as § Branches and PRs allows.
Each step's content and its done-condition are stated where the step's plan section says; this table owns only the order, the dependencies, the branch, and the cut.
The declarations and addressing steps follow `declarations-and-addressing-io.md`, which merged the inputs and outputs plan on 2026-10-18; the original `declarations-and-addressing.md` is kept unchanged for recovery, pending the maintainer's acceptance of the merge, and the tag `plans-before-io` marks the state before it.

> **Pending decisions on the schedule.** An overnight review on 2026-10-18 found that the never-cut set, as the steps are drawn, does not fit the week (`decisions.md` N1), and proposed splitting E3a, E4, and S2b (N2, N3, N5), making E4's performance check an off-ramp (N4), letting E6 build on E2 instead of E4 (N6), and separating dependencies a step builds on from those that only fix the merge order (N7). Until the maintainer decides, the tables below stand as written.

**Dates.** The workshop is on 2026-10-27, and the code freezes at the end of Sunday 2026-10-25, when `dev/overhaul` is tagged. Checkpoints are on 2026-10-20 and 2026-10-22; the second decides any cut. Whether 2026-10-24 and 2026-10-25 are working days for reviews is pending the maintainer's decision N13.

**Priorities and off-ramps.** The push aims to merge every step marked "before" by the freeze, with several sessions working in parallel. At most three implementing sessions run at once on one machine, plus one coordinator, since the full suite and its runs on each branch and its base share one 16 GB machine. When a slot frees, start the never-cut step that frees the longest chain; off-ramps start only when no never-cut step can start. The most fundamental contracts start first and are never cut; the rest are off-ramps, cut in a fixed order if the checkpoint of 2026-10-22 finds time short.

1. **Never cut**, started first as their dependencies allow: P0b, PV, S0, S1, S2a, S2b, S3, E0, E1, E2, E3a, E3b, E4, and E6. These are the declarations and paths a user types, the call context, executor dispatch for traceable and opaque functions, the boundary, and the public target interface.
2. **Off-ramps**, cut in this order: E8, then E7, then G1b, then E5, then G1a, then S4. Each cut keeps the steps above it buildable: cutting E5 also cuts G1b, which depends on it, and cutting S4 also cuts G1a. If E5 is cut, the sites it names keep today's own vectorize-or-loop decision under E3b's executors, and its **Joints (E5)** test is marked as an expected failure (`execution-layer.md` § 13).

A clean cut means:
- the step's PR does not merge; its branch is kept, and its work resumes after the workshop;
- the current behaviour stays, with no bridging code (directive 8);
- `deferred.md` gains the step, and the tracking issue marks it cut;
- a session never starts a step whose cut is already decided.

Since every merge leaves `dev/overhaul` coherent (directive 7), the code can be frozen after any merge.

**The critical path.** Three chains decide the date:
1. **Execution:** #586 → P0b → #587 → PV → E3a → E3b → E4 → E6, with E1 → E2 joining at E3a. #586 and #587 are jhuggins's PRs, so their merge dates are agreed with him on 2026-10-19 (pending N11; #586 conflicts with `dev/overhaul` tonight).
2. **Declarations:** the maintainer's acceptance of the inputs and outputs merge → S0 → S1 and S2a → S2b → S3, and S2b with E3b → S4 → G1a. Each day the acceptance waits delays this chain and, through S2a and S2b, the execution chain from E3b on (`declarations-and-addressing-io.md` B.7; pending N12).
3. **Where they join:** S2a gates E3b, and S2b gates E4 and S4.

E0 gates E3a, E1 gates E2, and E2 gates E3a; none of them runs beside the chains. The off-ramps E5 and G1b follow E4.

### Before any step

| Step | What | Depends on | Branch |
|---|---|---|---|
| P0 | The in-flight PRs #586 and #587 merge (#575, #583, and #474 merged on 2026-10-17). #586 and #587 record their renames in the CHANGELOG, and their user-visible changes (`n_broadcast_samples` → `max_evaluations` and `num_samples`, `set_default_n_broadcast_samples` → `config.sampling`, `*_config` → `config.*`, the default Monte Carlo count 256 → 1024) are part of any tutorial migration (N17) | — | `dev/split-sample-controls`, `dev/config-sections` |
| P0b | The call context is created as `execution-layer.md` § 6.16 describes, in #587 before it merges. jhuggins asked on 2026-10-18 that the push make these changes on his branch; the maintainer chose to open them as a draft PR into `dev/config-sections` rather than push to it, and jhuggins merges it into #587 | #586 | `dev/call-context` |
| PV | The provenance mode becomes a per-call setting, and the `workflow_run` frame and the Prefect payload stop keeping their own copy (decided 2026-10-18; `execution-layer.md` § 6.16, "The provenance mode per call (step PV)", states it and its done-condition); PV adds the call context's `session_config_values` for every section but `prefect`, which evaluation points read, and E3a ships it to process workers. *Pending the maintainer's decision N9: jhuggins asks on #587 that a block changing the mode inside an open run raise instead* | P0, P0b | `dev/provenance-per-call` |
| PL | This `plans/` PR merges first, by 2026-10-19: the tracking issue, the advisory tutorials leg (gaps W2), and `/review-pr`'s plan check all depend on it. It deletes `declarations.md`, `addressing.md`, `call-context-proposal.md`, and `maintainer-review.md`, and keeps `declarations-and-addressing.md` and `inputs-and-outputs.md` until the acceptance of the merge | every item of `gaps-and-conflicts.md` and every open point of the declarations and addressing plan, all settled on 2026-10-18; PL does not wait on the inputs and outputs questions, which gate S0 instead | `dev/contracts-plans` |

### Declarations and addressing

| Step | What | Plan section | Depends on | Before | Branch |
|---|---|---|---|---|---|
| S0 | Design text: II.2 states the grammar rule, input specs, and output specs indexed by path, II.6 the paths and how `[]` reads them; IV.2 retitled "Merging distributions", with the one definition of merge, "merge" for `&`, and "unfed" for "unmet"; "conforms", "unify", and "completes", one word per relation; the glossary; all in today's words "slot", "component", and "input, output, and event declaration", which one PR renames after the workshop; the reworded passages of § 6.5.4; and `_PENDING` entries in the conformance tests for the signatures that change ahead of the code | `declarations-and-addressing-io.md` B.6 S0 | PL; the maintainer's acceptance of the inputs and outputs merge (`declarations-and-addressing-io.md`, provisional 2026-10-18) | yes | recorded on the tracking issue when its PR opens |
| S1 | One path reader and one `_PATH_SEP`; `[]` returns groups on `Record` and `RecordSpec`; lists of paths keep their paths; `in`, laws included; `marginal(d, paths)`; `select` flattens to last names; `()` on a batch raises | B.6 S1 | S0 | yes | `dev/path-reader` |
| S2a | **Announced, mechanical.** `spec` becomes `term_spec` and `with_spec` becomes `with_term_spec`; `exposes_record` goes; every read of `_component_name` outside `_specs.py` moves to one private helper; `components` keeps its name | B.6 S2a | S0 | yes | `dev/term-spec-rename` |
| S2b | `OutputSpec[]`, `in`, and `without` through S1's reader; one reading of inputs, with list selection on `InputSpec`; the size shorthand; `&` on `InputSpec` and `OutputSpec`; a produced law or kernel offers no outputs; `merge` on records, record specs, and `RecordBatch` becomes shallow; the User Guide's `given_spec` idioms (User Guide 04, 06, 07, 08, 09). *Splitting it into output and input parts is pending N5* | B.6 S2b | S1, S2a | yes | `dev/input-output-specs` |
| S3 | **Announced, mechanical.** `&` replaces `*` for laws and kernels, and `merge` on `NamedTree`, `Record`, `RecordSpec`, and `RecordBatch`; "merge" replaces "composition" in the code that implements `&`; `rename`; `MarginalView`. S3 and E4 may merge in either order; the second rebases. *Whether `*` between laws keeps a refusing `__mul__` is pending N19* | B.6 S3 | S2b | yes | `dev/merge-operator` |
| S4 | Every batch stores its element spec: `BatchSpec.element_spec` becomes an output spec, whose `term_spec` is the old meaning, changed at every use; a batch's outputs are its columns; the list form at a batch's root (`draws[["x"]]`) and renaming a base element's name. *Whether it is announced, or split into a mechanical and a behaviour part, is pending N15* | B.6 S4 | S2b, E3b | yes, cut sixth | `dev/batch-element-spec` |
| G1a | Grammar B for views and marginals of groups, the library's laws in an interim whole form over a record made positional or wrapped (pending N31), and factors that hold a group, which stay one factor; M3's message for two views of one parent; the opt-in interim-form probe `tests/_interim_form_probe.py` | B.6 G1a | S2b, S4 | yes, cut fifth | `dev/grammar-b-views` |
| G1b | Grammar B for outputs: a function that returns a law has a positional output with no outputs, one that returns a function offers its outputs, a lifted law-returning function wraps its draws, one `OutputSpec.default` by kind, and the reading of a bare `output_spec` | B.6 G1b | G1a, E5 | yes, cut third | `dev/grammar-b-outputs` |
| G2 | Grammar B for the remaining kinds, then enforcement: random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets wrap their draws under the name their constructor takes; then the keyword form of `OutputSpec` refuses a spec with parts, once G3 has moved views at law-valued fields off the old form, and the private event default goes | B.6 G2 | G1b, G3 | after | chosen after the workshop |
| G3 | Paths through every kind; `Function[]`; the public accessor `t.component(c)` (`t.output(c)` after the rename), and `[]` and `OutputSpec[]` stop reading the name of the output at the root; a view at a field that holds a law, a kernel, a function, or a batch is positional, with no wrap; `_view_declaration` goes. *Its order against the rename is pending N16* | B.6 G3 | G1a | after | chosen after the workshop |
| G4 | `condition_on`'s result has the event spec `d.event_spec.without(*fixed)`; `law.without`; paths into a record input of an `InputSpec`; only merge and rename keep factors | B.6 G4 | G1a | after | chosen after the workshop |
| G5 | `[]` on batches of laws, on batches of functions, and on kernels, and the accessor on kernels | B.6 G5 | G3 | after | chosen after the workshop |
| R | **Announced.** The rename: "slot", "component", "input declaration", "output declaration", and "event declaration" become "input", "output", "input spec", "output spec", and "event spec" in `design/`, the glossary, the code, its messages, and `execution-layer.md`; `components` → `outputs`, `component=` → `output_name=`, and `t.component(c)` → `t.output(c)` if G3 has merged. Either order against G3 works, and each adapts; whether it must precede G2–G5 is pending N16 | § 6.15 | the workshop | after | chosen after the workshop |

### The execution layer

| Step | What | Plan section | Depends on | Before | Branch |
|---|---|---|---|---|---|
| E0 | **Announced, mechanical.** #524's two module moves (`_execution_contract.py` into `_execution.py`; `_managed.py` and `_broker.py` into `_evaluation_keys.py`), and "managed work item" renamed "evaluation point" in every code name, docstring, message, and test, in V.8, `design/package-structure.md`, and the glossary; a derived operation's "identity" renamed its "formula", and the identity route its "derived route", in `operations/`, `values/`, `functions/`, VI.0, and the glossary, `_formula_source` included | `execution-layer.md` § 13 | PL, P0 | yes | `dev/evaluation-point-rename` |
| E1 | Design text: V.9 rewritten around sets of evaluation points and executors, the new V.13, the controls in V.2 and VI.0, the glossary, and the removal of "dispatch mode" and "managed work item" from every section of `design/` | § 13 | PL, P0 | yes | `dev/execution-design-text` |
| E2 | Claims and the probe result: `traceable`, `vectorized`, one probe per callable and input structure, shared by `using` copies and run by `check`, replacing the five probes, `is_jax_traceable` included; the error for a false claim | § 13 | E1 | yes | `dev/execution-claims` |
| E3a | In `core/`: sets of evaluation points, the executor registry and its built-in executors, the call context's executor choice and evaluation context, the program store of `jax.jit` objects that traces and lowers each JAX program before its first run, with the static settings, which include the passed controls and the call's resolved static controls, in its cache key, and the JAX settings and session snapshot shipped to process workers; the failure rule and output check for sets built in Python; the controls `executor` and `executor_options`. *Splitting it is pending N2* | § 13 | E0, E1, E2, P0b, PV | yes | `dev/executor-registry` |
| E3b | The executors replace the dispatch modes in `_broadcast.py`, `_sweep.py`, and `_execution.py`; `dispatch` and `max_workers` go; the closed execution contract goes; `EvaluationChunk` carries the evaluation context, Prefect tasks included. *Whether it also moves lifting and sweeps onto `evaluate_at_points` is pending N8* | § 13 | E3a, S2a | yes | `dev/executors-in-lifting` |
| E4 | The boundary: embedded calls and call tokens in every JAX program ProbPipe opens that runs user or route code, the inference runners included; the search of the jaxpr; lazy token registration; failures that stop a compiled JAX program; the output check on the host side. *Splitting it, and its performance check, are pending N3 and N4* | § 13 | E2, E3b, S2b | yes | `dev/embedded-calls` |
| E5 | `evaluate_at_points` in lifting and sweeps, joint factors, laws built from functions, and expectations; if cut, those sites keep their own decision under E3b's executors | § 13 | E4 | yes, cut fourth | `dev/evaluate-at-points` |
| E6 | The public target interface: `InferenceTarget` and `make_posterior`, with the run key passed through a method's `key` parameter, as #554's § 8 states. *Its interface and its dependency are pending N23 and N6* | § 13 | E4 | yes | `dev/inference-target` |
| E7 | At most one run per distinct evaluation point inside a call | § 13 | E4 | yes, cut second | `dev/evaluation-run-table` |
| E8 | The JAX settings contract: the check at close and on the host side, and provenance | § 13 | E4 | yes, cut first | `dev/jax-settings-contract` |

The post-workshop work of both tracks is listed in `deferred.md`. The rename (row R) comes right after the workshop; its order against G3 is pending N16.

## Status

A GitHub tracking issue holds each step's state and PR, as `CONTRIBUTING.md` § PR Workflow describes for a multi-PR plan. The maintainer approved it on 2026-10-18; it is created when this PR merges. Step PRs never edit § Steps, so parallel PRs do not conflict on this file.

### Steps started before this PR merged

The maintainer approved starting steps from the plans branch on 2026-10-18, before PL merges. Each opened, or will open, as a draft PR, and lists its questions in its body under "Questions for the maintainer", since `decisions.md` is not on its base. State on the night of 2026-10-18:

| Step | PR | Branch | Base | What happens next |
|---|---|---|---|---|
| P0b | #597 (draft) | `dev/call-context` | `dev/config-sections` (#587) | A PR into #587's branch, the one exception to "every step is one PR into `dev/overhaul`". The maintainer or jhuggins reviews it, and jhuggins merges it into #587; nothing was pushed to `dev/config-sections` itself. |
| E0 | #598 (draft) | `dev/evaluation-point-rename` | `dev/config-sections` | Once P0b is in #587, it merges `dev/config-sections` and renames P0b's "identity". It retargets `dev/overhaul` when #587 merges. |
| E1 | #596 (draft) | `dev/execution-design-text` | `dev/config-sections` | Once P0b is in #587, it merges `dev/config-sections`, taking P0b's V.2 sentence and glossary entry "call context". It retargets `dev/overhaul` when #587 merges. |
| S1 | none yet (draft PR planned) | `dev/path-reader` | `dev/overhaul` | Built ahead of S0, against `declarations-and-addressing-io.md` B.6, S1. It merges after S0, and merges `dev/overhaul` once #587 merges. |
| S0 | none yet | recorded when its PR opens | `dev/overhaul` | Being written from the merged `-io` plan; it waits for the maintainer's acceptance of the merge (N12). |

When this PR merges, its session creates the tracking issue with these rows, and creates handoffs for PV only; it does not create handoffs for the steps above.

## Starting a session on a step

1. Read this file in full, then `decisions.md`, then the current plans (directive 1).
2. Check the tracking issue, and `gh pr list --state all --head <the step's branch>` with the branch the step tables name: if a PR or branch for the step already exists, stop and tell the maintainer.
3. Check that every dependency of the step has merged: `gh pr view <number> --json state,mergedAt` for each. If exactly one is open but not merged, stack on it as § Branches and PRs allows; if more than one is open, stop and report which.
4. `git fetch origin`, then create the step's branch, named as the step tables say, from `origin/dev/overhaul`, or from the dependency's branch when stacking. The session's worktree may have started on another branch, such as `main`; never base a step on it.
5. Read the code the step changes, and the code that already does anything similar (directive 5).
6. **A session the maintainer opened from a handoff chip** first returns an implementation plan to him and waits for his approval before writing code: what the step will change, file by file, the order of the work, the tests and demonstration, and any question for him. **A session another agent started automatically**, which usually runs while the maintainer is away and has a well-scoped task, skips this and proceeds; it reports its plan and its choices in the PR body, which the maintainer reviews when he is back.

## Ending a session

1. Open the PR into `dev/overhaul` as a draft, with a body that follows directives 4, 9, and 10 and the PR template. Its CHANGELOG entries go at the **end** of the existing section for their type, and never add a heading that `[Unreleased]` already has; each breaking change goes under `### Changed (breaking)` with a before-and-after line a user can copy, and a silent change shows the same call before and after (`execution-layer.md` § 13.1, `declarations-and-addressing-io.md` C.5).
2. Run `/review-pr` on it, fix or answer each finding, and mark it ready for review (directive 13). If `plans/` is not on the PR's base, run `/review-pr` from a checkout of a branch that has `plans/`, or paste `plans/README.md` into the review's context, so the plan check runs.
3. Comment on the tracking issue with the PR link.
4. Post any decision or question, and any postponed work, as a comment on the tracking issue (before it exists, in the PR body). Step PRs do not edit `decisions.md` or `deferred.md`, which every parallel PR would otherwise conflict on: at each checkpoint, one small plans PR moves the comments into them. A plan correction that a step needs goes in that step's PR, in the section it corrects, and the checkpoint PR adds its revision-log row.
5. Create the handoffs (below).

## Handoffs

When a session marks its PR ready for review, it creates a handoff for each step that lists its own step among its dependencies and whose other dependencies have each merged or are open as PRs.
Two sessions may then create a handoff for the same step; the new session's check of the tracking issue and the step's branch stops the second.
A handoff whose dependencies have not merged when it is opened stacks on them or stops (§ Starting a session); the maintainer opens it again later.
When this `plans/` PR merges, its session creates the handoff for PV only; P0b, E0, E1, S1, and S0 are already under way (§ Steps started before this PR merged), and a handoff must not start them again.
A coordinator session may start step sessions directly, at most three at a time (§ Priorities and off-ramps), choosing never-cut steps first and the step that frees the longest chain first.

In the Claude desktop app, a handoff is a task chip made with the `spawn_task` tool: the maintainer clicks it to open a new session in a fresh worktree, so a chip starts nothing until he does.
Where `spawn_task` is not available, post the same prompt as a comment on the PR, not in its body, since the prompt names step labels (directive 10).

The chip's title names the step, such as "Implement step E3a: evaluation points and the executor registry".
Its prompt stands alone, since the new session sees nothing of the old one, and follows this template:

```text
You are starting step <ID> (<title>) of ProbPipe's contracts push.

Repository: <absolute path of the main checkout>. Work in the fresh worktree this session opened.

1. Read plans/README.md in full and follow its directives. Then read plans/decisions.md
   and the current plans in full (README directive 1). This step is specified in
   <plan file and section>, which also states its done-condition.
2. Check the tracking issue (<link>) and `gh pr list --state all --head <branch>`, where
   <branch> is the step's branch in plans/README.md § Steps: if a PR or branch for this
   step already exists, stop.
3. This step depends on <each dependency: step, PR number>. Check each with
   `gh pr view <number> --json state,mergedAt`. Base the branch on origin/dev/overhaul
   when all have merged; stack on <branch> if exactly one dependency, <dependency>, is still
   open, as plans/README.md § Branches and PRs allows; if more than one is open, stop and
   tell the maintainer.
4. Run `git fetch origin` and create the branch <branch> from the base chosen above.
   Do not build on the branch this worktree started on.
5. Context from each dependency:
   - <step, PR link>: what it delivered; what it found that this step needs to know;
     the decisions it recorded; what it left open.
6. Before writing code, analyze the step and return an implementation plan to the
   maintainer (what changes, file by file; the order of work; the tests and demonstration;
   your questions), then wait for his approval (plans/README.md § Starting a session, item 6).
   Then do the step as its plan section describes. Ask the maintainer whenever
   plans/README.md § When to stop and ask applies, without waiting unless the question
   blocks the step. The PR body's "Fit with the plan" section names <the plan sections>.
7. Finish as plans/README.md § Ending a session describes, including the handoffs.
```

## Checkpoints

The checkpoints are on 2026-10-20 and 2026-10-22; the second decides the cut. *Criteria that make each checkpoint's decision mechanical, and a cut early enough to save work, are pending the maintainer's decision N14 in `decisions.md`.*
At a checkpoint:
1. run `/review-all` on the latest merged work, or a fresh-eyes review of the merged code against every plan;
2. update the plans and this file where the code has taught something, and move the tracking issue's decisions, questions, and postponed work into `decisions.md` and `deferred.md` (§ Ending a session, step 4);
3. update the steps that remain.

## When the push ends

Steps S0 and E1 move the design into `design/`, which stays the single source of the target design; later steps update `design/` with their code.
Before `plans/` is deleted, one PR re-evaluates the repository's rule documents against this push: `STYLE_GUIDE.md`, `CONTRACTS.md`, `CONTRIBUTING.md`, `AGENTS.md`, and the skills in `.claude/skills/`. It moves into them every directive of this file that proved useful and is not already there, such as the code-quality rules of directive 5 and the rule of directive 14, so the standards outlive the push. The same PR removes what the push added to the `review-pr` skill (its plan check and the matching line in `.claude/skills/README.md`), since `plans/` no longer exists.
Once that PR has merged, every step before the workshop has merged, and the post-workshop work is recorded, `plans/` is deleted; git keeps its history.
The items left in `deferred.md` become GitHub issues.
