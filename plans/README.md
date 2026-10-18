# The contracts push

This folder holds the shared plans for a sequence of PRs that puts three areas of ProbPipe on precise contracts before the workshop of 2026-10-27:
1. **the execution layer:** how a computation runs, including code JAX cannot compile, and where (#580);
2. **addressing:** how a path or a subscript picks out a part of a term (#573);
3. **declarations:** how terms declare what they produce and consume, and the operations on those declarations (#571).

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
| [`declarations-and-addressing.md`](declarations-and-addressing.md) | The reconciled declarations and addressing design (#571, #573): grammar B, slots and components, subscripts and paths, `&`, and steps S0, S1, S2a, S2b, S3, S4, G1a, G1b, and G2–G5 |
| [`declarations.md`](declarations.md), [`addressing.md`](addressing.md) | The two source texts the reconciled design replaced, kept for comparison until the maintainer has reviewed the reconciled text |
| [`call-context-proposal.md`](call-context-proposal.md) | The design of the one call context that carries per-call state, under review; it is folded into `execution-layer.md` once accepted |
| [`deferred.md`](deferred.md) | Work found or postponed during the push, to become issues after the workshop |

`declarations-and-addressing.md` replaces `declarations.md` and `addressing.md` (gaps D1); the two sources are deleted once the maintainer has reviewed it.

## Standing directives

These apply to every person and agent working on the push.
They add to `CONTRACTS.md`, `STYLE_GUIDE.md`, and the implementation guidelines of `execution-layer.md` §12, which still apply in full.

1. **Read the full design first.** Before working on a step, read this file, `decisions.md`, and the current plans in full: `declarations-and-addressing.md`, `execution-layer.md`, and `call-context-proposal.md` until it is folded into the execution plan. Read them whole, not only the sections the step implements. The superseded `declarations.md` and `addressing.md` are not plans; skip them. Check that the work fits the broader goals of the push and of `design/`.
2. **Design gaps go to the maintainer.** The plans will have gaps that implementation exposes. A gap that needs a real design decision is never settled by the implementer alone: follow § When to stop and ask. A fix to a gap serves the long-term design, never only the step at hand.
3. **Contracts, not patches.** Write well-designed, coherent contracts. Never patch or work around a deeper design problem. If the problem is within the push's scope, fix it; otherwise record it in `deferred.md` for work after the workshop.
4. **PR descriptions written for people.** Give the context, and do not assume the reviewer knows ProbPipe's internals. Show specific examples of new behaviour, such as code before and after. Avoid jargon and dense blocks of text. The reasoning should flow, and the summary of changes should read naturally.
5. **Code quality is judged, not only behaviour.**
    - No duplicated code. Before writing a helper, search the codebase for one that already does the job. Reviewers check for duplicates across the codebase, not only within the diff.
    - Every name says what the thing is or does, private ones included. A name such as `_combine` fails: combine what?
    - A public docstring is written for users, and carries no design reasoning.
    - A private object of central importance, such as `_CallContext`, gets a full docstring too: what it is, its lifecycle, its invariants, and each attribute, so a reader can follow the code without the plans.
    - An inline comment appears only where it adds something the code and the docstring do not already say.
    - A module puts its most important public objects first and its private helpers at the bottom.
6. **The tutorials are out of scope; the User Guide and Get Started are kept running.** `docs/tutorials/` is being rewritten separately, so no step edits it, and its CI notebook job is advisory until the rewrite merges. The notebooks of `docs/user_guide/` and `docs/get_started/` run in blocking CI jobs: a step that breaks one updates it mechanically, without rewriting it, so no page documents a removed API.
7. **No partial migrations.** A PR that changes a name or a meaning changes every use of it in the same PR, so `dev/overhaul` is always coherent and could be used at the workshop at any merge.
8. **Deferring keeps the current behaviour.** Work postponed to after the workshop leaves the current code unchanged and gets an entry in `deferred.md`. It never gets bridging code, such as an alias, a shim, or a special case that serves only the interim.
9. **Tests and a demonstration in every PR.** Each PR adds the tests that assert its contract (shapes, orderings, and error cases, beyond the happy path), and its body includes a short runnable demonstration with its output.
10. **Every PR body says how it fits the plan.** A section "Fit with the plan" lists:
    - the plan sections the PR implements;
    - any departure from them;
    - the decisions it recorded in `decisions.md`;
    - the existing code it reused, and any duplicate it removed.

    The title and the rest of the body name each change by its behaviour, never by a step label such as "S3" (`STYLE_GUIDE.md` §10, rule 9).
11. **One vocabulary.** Terms come from `design/glossary.md` and the plans. A PR that needs a new term adds it to the glossary in the same PR, and checks that it collides with no existing term.
12. **Broad reviews at checkpoints only.** `/review-all` runs four reviews and costs much time and many tokens. Run it when it is a good moment to assess the push broadly, such as after a group of steps has merged, never after each PR. A single PR gets `/review-pr` or `/check-pr`.
13. **Review before asking for review.** Open the PR as a draft and run `/review-pr` on it. When `plans/README.md` exists, the skill also checks the PR against these directives, `decisions.md`, and the step's plan section. Fix each finding, or answer it in the PR, and then mark the PR ready for review. The maintainer reviews a PR once it is marked ready.
14. **Replace stale code rather than work around it.** This push overhauls these parts of ProbPipe, and the goal is code a person can read through and follow. Code that conflicts with the plans, or that has bad names, poor organization, or duplicated logic, is rewritten or removed, not patched around and not edited piecemeal to fit. Writing a clean replacement from scratch is often the better choice, within the step's scope; a larger rewrite outside it is recorded in `deferred.md` or raised with the maintainer.

## When to stop and ask

Stop and ask the maintainer before doing any of the following:
- changing a public name, signature, or default;
- changing a contract or an invariant that a plan or `design/` states;
- adding, removing, or redefining a term in the glossary;
- departing from an item that `decisions.md` records as settled;
- adding a special case, an alias, or a workaround to get past a problem.

To ask, add an entry to the queue in `decisions.md` and say so in the session.
Write the question for a reader without the session's context:
1. the background, from first principles;
2. today's behaviour and the proposed behaviour, in code;
3. the options, each explained;
4. a recommendation, with the reason.

A choice that changes none of the above, such as a private helper's structure, is the implementer's. Record it in the PR body.

## Branches and PRs

- Every step is one PR into `dev/overhaul`, from a branch `dev/<short-description>` cut from `origin/dev/overhaul`.
- A step may stack on a dependency whose PR is open but not merged, and retargets `dev/overhaul` once the dependency merges. Say so in the PR body. Code steps commonly stack on the design-text PRs S0 and E1.
- A mechanical PR, such as a rename across the codebase, stays separate from a PR that changes behaviour, so each is quick to review. The steps marked **announced** below are wide: before opening one, the session comments on the tracking issue (§ Status) and tells the maintainer in its final message, so open branches rebase once.
- When the workshop's code freezes, `dev/overhaul` is tagged, and the workshop uses the tag.

## Steps

**Provisional** until this PR merges.
"Before" marks a step planned before the workshop.
A step may not start until every step and PR it depends on has merged into `dev/overhaul`, or it stacks as § Branches and PRs allows.
Each step's content and its done-condition are stated where the step's plan section says; this table owns only the order, the dependencies, and the cut.

**Dates.** The workshop is on 2026-10-27, and the code freezes at the end of 2026-10-25, when `dev/overhaul` is tagged. A checkpoint on 2026-10-20 reviews progress, and a checkpoint on 2026-10-22 decides any cut.

**Priorities and off-ramps.** The push aims to merge every step marked "before" by the freeze, with several people and sessions working in parallel. The most fundamental contracts start first and are never cut; the rest are off-ramps, cut in a fixed order if the checkpoint of 2026-10-22 finds time short.

1. **Never cut**, started first as their dependencies allow: CC, P0b, S0, S1, S2a, S2b, S3, E0, E1, E2, E3a, E3b, E4, and E6. These are the declarations and subscripts a user types, the call context, executor dispatch for traceable and opaque functions, the boundary, and the public target interface.
2. **Off-ramps**, cut in this order: E8, then E7, then G1b, then E5, then G1a, then S4. Each cut keeps the steps above it buildable: cutting E5 also cuts G1b, which depends on it, and cutting S4 also cuts G1a.

A clean cut means:
- the step's PR does not merge; its branch is kept, and its work resumes after the workshop;
- the current behaviour stays, with no bridging code (directive 8);
- `deferred.md` gains the step, and the tracking issue marks it cut;
- a session never starts a step whose cut is already decided.

Since every merge leaves `dev/overhaul` coherent (directive 7), the code can be frozen after any merge.

**The critical path** has two chains. PL → S0 → S1 → S2b → S3, with S2b and E3b → S4 → G1a. PL → CC → E1 → E3a → E3b → E4 → E5 → G1b. S2a, E0, and E2 run beside them.

### Before any step

| Step | What | Depends on |
|---|---|---|
| P0 | The in-flight PRs #586 and #587 merge (#575, #583, and #474 merged on 2026-10-17) | — |
| P0b | The call context is created as `call-context-proposal.md` § 7.2 describes: in #587 before it merges if the maintainer chooses OD5 option A, in a follow-up PR if option B (decision pending) | #586 |
| PV | The provenance mode becomes a per-call setting, and the `workflow_run` frame stops keeping its own copy (call-context OD2, pending) | P0, P0b |
| PL | This `plans/` PR merges | every item of `gaps-and-conflicts.md` except D7, and the open points O1–O16 of `declarations-and-addressing.md` |
| CC | The maintainer accepts `call-context-proposal.md` (gaps D7), and a PR folds it into `execution-layer.md` | PL |

### Declarations and addressing

| Step | What | Plan section | Depends on | Before |
|---|---|---|---|---|
| S0 | Design text: II.2 states the grammar rule and the declarations, II.6 the paths and the reading of a subscript; the glossary | `declarations-and-addressing.md` § 19.2 | PL | yes |
| S1 | One subscript reader and one `_PATH_SEP`; `[]` returns groups on `Record` and `RecordSpec`; lists of paths keep their paths; `in`, laws included; `marginal(d, paths)`; `select` flattens to last segments; `()` on a batch raises | § 19.3 | S0 | yes |
| S2a | **Announced, mechanical.** `OutputSpec` becomes a mapping of its components; `spec` becomes `term_spec`; every read of `components`, `exposes_record`, and `_component_name` outside `_specs.py` moves to the mapping | § 19.4 | S0 | yes |
| S2b | `project` and `without` through S1's reader; one reading of slots; the size shorthand; `&` on `InputSpec` and `OutputSpec`; the refusal of a law's or a kernel's event declaration that exposes a law, a kernel, a function, or a batch | § 19.4 | S1, S2a | yes |
| S3 | **Announced, mechanical.** `&` replaces `*` for laws and kernels, and `merge` on records; `rename`; `MarginalView`. S3 and E4 may merge in either order; the second rebases | § 19.5 | S2b | yes |
| S4 | Every batch stores its element declaration | § 19.6 | S2b, E3b | yes |
| G1a | Grammar B for views and marginals of groups, the library's laws declared whole over a record, and factors that hold a group | § 19.7 | S2b, S4 | yes, never cut |
| G1b | Grammar B for outputs: a function that returns a law or a function exposes its output, a lifted law-returning function wraps its draws, defaults by kind, and the reading of a bare `output_spec` | § 19.7 | G1a, E5 | yes, cut third |
| G2 | Grammar B for the remaining kinds, then enforcement: random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets wrap their draws; then the keyword form of `OutputSpec` refuses a spec with parts, once G3 has moved views at law-valued fields off the old form | § 19.8 | G1b, G3 | after |
| G3 | Paths through every kind; `Function[]`; the public `component` accessor, and a whole term's name stops being a subscript; a view of a field that holds a law or a function wraps | § 19.8 | G1a | after |
| G4 | `condition_on` declared as `d.without(*fixed)`; `law.without` | § 19.8 | G1a | after |
| G5 | `[]` on batches of laws and on kernels | § 19.8 | G3 | after |

### The execution layer

| Step | What | Plan section | Depends on | Before |
|---|---|---|---|---|
| E0 | **Announced, mechanical.** #524's two module moves (`_execution_contract.py` into `_execution.py`; `_managed.py` and `_broker.py` into `_evaluation_keys.py`), and "managed work item" renamed "evaluation point" in every code name, docstring, and test | `execution-layer.md` § 13 | PL, P0 | yes |
| E1 | Design text: V.9 rewritten around sets of evaluation points and executors, the new V.13, the controls in V.2 and VI.0, the glossary | § 13 | CC | yes |
| E2 | Claims and the probe result: `traceable`, `vectorized`, one probe per callable and input structure, shared by `using` copies and run by `check`, replacing the five probes, `is_jax_traceable` included; the error for a false claim | § 13 | E1 | yes |
| E3a | In `core/`: sets of evaluation points, the executor registry and its built-in executors, the call context's executor choice and evaluation context, the program store of `jax.jit` objects that traces and lowers each program before its first run, with the passed controls and the traced settings in its cache key, and the JAX settings shipped to process workers; the controls `executor` and `executor_options` | § 13 | E0, E1, P0b, PV | yes |
| E3b | The executors replace the dispatch modes in `_broadcast.py`, `_sweep.py`, and `_execution.py`; `dispatch` and `max_workers` go; the closed execution contract goes | § 13 | E3a, S2a | yes |
| E4 | The boundary: embedded calls and call tokens in every JAX program ProbPipe opens that runs user or route code, the inference runners included; the search of the jaxpr; lazy token registration; failures; the output check per evaluation point | § 13 | E2, E3b, S2b | yes |
| E5 | `evaluate_at_points` in lifting and sweeps, joint factors, laws built from functions, and expectations | § 13 | E4 | yes |
| E6 | The public target interface: `InferenceTarget` and `make_posterior`, with the run key passed as #554 §8 states | § 13 | E4 | yes |
| E7 | At most one run per distinct evaluation point inside a call | § 13 | E4 | yes, cut second |
| E8 | The JAX settings contract: the check at close and on the host side, and provenance | § 13 | E4 | yes, cut first |

The post-workshop work of both tracks is listed in `deferred.md`.

## Status

A GitHub tracking issue holds each step's state and PR, as `CONTRIBUTING.md` § PR Workflow describes for a multi-PR plan. The maintainer creates it when this PR merges (pending the maintainer's approval). Step PRs never edit § Steps, so parallel PRs do not conflict on this file.

## Starting a session on a step

1. Read this file in full, then `decisions.md`, then the current plans (directive 1).
2. Check the tracking issue: if a PR or branch for the step already exists, stop and tell the maintainer.
3. Check that every dependency of the step has merged: `gh pr view <number> --json state,mergedAt` for each. If one is open but not merged, stack on it as § Branches and PRs allows, or stop and report which if stacking is not sensible.
4. `git fetch origin`, then create the step's branch from `origin/dev/overhaul`, or from the dependency's branch when stacking. The session's worktree may have started on another branch, such as `main`; never base a step on it.
5. Read the code the step changes, and the code that already does anything similar (directive 5).

## Ending a session

1. Open the PR into `dev/overhaul` as a draft, with a body that follows directives 4, 9, and 10 and the PR template.
2. Run `/review-pr` on it, fix or answer each finding, and mark it ready for review (directive 13).
3. Comment on the tracking issue with the PR link.
4. Record any decision in `decisions.md`, and any postponed work in `deferred.md`, in the same PR.
5. Create the handoffs (below).

## Handoffs

When a session marks its PR ready for review, it creates a handoff for each step that lists its own step among its dependencies and whose other dependencies have each merged or are open as PRs.
Two sessions may then create a handoff for the same step; the new session's check of the tracking issue stops the second.
A handoff whose dependencies have not merged when it is opened stacks on them or stops (§ Starting a session); the maintainer opens it again later.
When this `plans/` PR merges, its session creates the handoffs for S0, CC, and E0, and for P0b and PV once the maintainer has decided OD5 and OD2. The session that opens CC's PR creates E1's handoff. A handoff that stopped because a dependency had not merged is opened again by the maintainer.

In the Claude desktop app, a handoff is a task chip made with the `spawn_task` tool: the maintainer clicks it to open a new session in a fresh worktree.
Where `spawn_task` is not available, post the same prompt as a comment on the PR, not in its body, since the prompt names step labels (directive 10).

The chip's title names the step, such as "Implement step E3a: evaluation points and the executor registry".
Its prompt stands alone, since the new session sees nothing of the old one, and follows this template:

```text
You are starting step <ID> (<title>) of ProbPipe's contracts push.

Repository: <absolute path of the main checkout>. Work in the fresh worktree this session opened.

1. Read plans/README.md in full and follow its directives. Then read plans/decisions.md
   and the current plans in full (README directive 1). This step is specified in
   <plan file and section>, which also states its done-condition.
2. Check the tracking issue (<link>), or, if it does not exist yet, `gh pr list --search <branch name>`:
   if a PR or branch for this step already exists, stop.
3. This step depends on <each dependency: step, PR number>. Check each with
   `gh pr view <number> --json state,mergedAt`. Base the branch on origin/dev/overhaul
   when all have merged; stack on <branch> if exactly one dependency, <dependency>, is still open, as
   plans/README.md § Branches and PRs allows; otherwise stop and tell the maintainer.
4. Run `git fetch origin` and create the branch dev/<name> from the base chosen above.
   Do not build on the branch this worktree started on.
5. Context from each dependency:
   - <step, PR link>: what it delivered; what it found that this step needs to know;
     the decisions it recorded; what it left open.
6. Do the step as its plan section describes. Stop and ask the maintainer whenever
   plans/README.md § When to stop and ask applies. The PR body's "Fit with the plan"
   section names <the plan sections>.
7. Finish as plans/README.md § Ending a session describes, including the handoffs.
```

## Checkpoints

A checkpoint is a moment to assess the push broadly, such as after S2b and E3b have merged, and two working days before the freeze, when the cut is decided.
At a checkpoint:
1. run `/review-all` on the latest merged work, or a fresh-eyes review of the merged code against every plan;
2. update the plans and this file where the code has taught something;
3. update the steps that remain.

## When the push ends

Steps S0 and E1 move the design into `design/`, which stays the single source of the target design; later steps update `design/` with their code.
Before `plans/` is deleted, one PR re-evaluates the repository's rule documents against this push: `STYLE_GUIDE.md`, `CONTRACTS.md`, `CONTRIBUTING.md`, `AGENTS.md`, and the skills in `.claude/skills/`. It moves into them every directive of this file that proved useful and is not already there, such as the code-quality rules of directive 5 and the rule of directive 14, so the standards outlive the push.
Once that PR has merged, every step before the workshop has merged, and the post-workshop work is recorded, `plans/` is deleted; git keeps its history.
The items left in `deferred.md` become GitHub issues.
