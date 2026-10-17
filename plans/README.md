# The contracts push

This folder holds the shared plans for a sequence of PRs that puts three areas of ProbPipe on precise contracts before the October 2026 workshop:
1. **the execution layer:** how a computation runs, including code JAX cannot compile, and where (#580);
2. **addressing:** how a path or a subscript picks out a part of a term (#573);
3. **declarations:** how terms declare what they produce and consume, and the operations on those declarations (#571).

Earlier work in these areas was planned one change at a time, which left inconsistent designs, duplicated code, and an API that is hard to learn.
This push plans the three areas together, and every PR in it is judged against the whole plan, not only its own step.

Every session that works on the push reads this file in full before anything else.

## The files

| File | What it holds |
|---|---|
| `README.md` | This charter: the directives, the steps and their order, the handoff protocol, and the status of each step |
| [`decisions.md`](decisions.md) | Every decision taken during the push, with its date and reason, and the queue of questions waiting for the maintainer |
| [`gaps-and-conflicts.md`](gaps-and-conflicts.md) | The places where the plans disagree or leave a decision open. This PR merges once each is settled. |
| [`execution-layer.md`](execution-layer.md) | The execution-layer design (#580) |
| [`declarations.md`](declarations.md) | The declarations design: #571's body and jhuggins's reply |
| [`addressing.md`](addressing.md) | The addressing design, grammar B (#573) |
| [`deferred.md`](deferred.md) | Work found or postponed during the push, to become issues after the workshop |

`declarations.md` and `addressing.md` are replaced by one reconciled text once the items of `gaps-and-conflicts.md` are settled (item D1).

## Standing directives

These apply to every person and agent working on the push.
They add to `CONTRACTS.md`, `STYLE_GUIDE.md`, and the implementation guidelines of `execution-layer.md` §12, which still apply in full.

1. **Read the full design first.** Before working on a step, read this file, `decisions.md`, and every plan in this folder, not only the sections the step implements. Check that the work fits the broader goals of the push and of `design/`.
2. **Design gaps go to the maintainer.** The plans will have gaps that implementation exposes. A gap that needs a real design decision is never settled by the implementer alone: follow § When to stop and ask. A fix to a gap serves the long-term design, never only the step at hand.
3. **Contracts, not patches.** Write well-designed, coherent contracts. Never patch or work around a deeper design problem. If the problem is within the push's scope, fix it; otherwise record it in `deferred.md` for work after the workshop.
4. **PR descriptions written for people.** Give the context, and do not assume the reviewer knows ProbPipe's internals. Show specific examples of new behaviour, such as code before and after. Avoid jargon and dense blocks of text. The reasoning should flow, and the summary of changes should read naturally.
5. **Code quality is judged, not only behaviour.**
    - No duplicated code. Before writing a helper, search the codebase for one that already does the job. Reviewers check for duplicates across the codebase, not only within the diff.
    - Every name says what the thing is or does, private ones included. A name such as `_combine` fails: combine what?
    - A public docstring is written for users, and carries no design reasoning.
    - An inline comment appears only where it adds something the code and the docstring do not already say.
    - A module puts its most important public objects first and its private helpers at the bottom.
6. **The tutorials are out of scope; the User Guide is kept running.** `docs/tutorials/` is being rewritten separately, so no step edits it, and its CI notebook job is advisory until the rewrite merges. The User Guide notebooks run in a blocking CI job: a step that breaks one updates it mechanically, without rewriting it, so the guide never documents a removed API.
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
- A step whose dependency is open but not merged may stack on that PR's branch, and retargets `dev/overhaul` once the dependency merges. Say so in the PR body.
- A mechanical PR, such as a rename across the codebase, stays separate from a PR that changes behaviour, so each is quick to review. Announce a wide mechanical PR before it merges, so open branches rebase once.
- When the workshop's code freezes, `dev/overhaul` is tagged, and the workshop uses the tag.

## Steps

**Provisional:** the steps and their order are final once `gaps-and-conflicts.md` is settled.
"Before" marks a step planned before the workshop. A step may not start until every step and PR it depends on has merged into `dev/overhaul`, or it stacks as § Branches and PRs allows.

If time runs short before the workshop, steps are cut in this order: E8, then E7, then the lifting and default-output parts of G1, which move to after the workshop. S0–S4 and E1–E6 are never cut.

### Before any step

| Step | What | Depends on | State |
|---|---|---|---|
| P0 | The in-flight PRs merge: #575, #583, #474, #584, #586, #587 | — | open |
| PL | This `plans/` PR merges | the items of `gaps-and-conflicts.md` | open |

### Declarations and addressing

| Step | What | Plan sections | Depends on | Before | PR | State |
|---|---|---|---|---|---|---|
| S0 | Design text: II.2 states the grammar rule and the declarations, II.6 the paths and the reading of a subscript; the glossary and `STYLE_GUIDE.md` §1.10–§1.11 | reconciled text | PL | yes | | |
| S1 | One subscript reader and one `_PATH_SEP`; `[]` returns groups on `Record` and `RecordSpec`; lists of paths keep their paths; `in`; `marginal(d, paths)`; `()` on a batch raises | addressing: Reading a subscript, Indexing with several paths, Membership | S0 | yes | | |
| S2 | `OutputSpec` as a mapping; `term_spec`; `project` and `without` through S1's reader; one reading of slots; the size shorthand; `InputSpec.merge`. Its renames change about 41 modules, so it is announced like S3 | declarations; reconciled text | S1 | yes | | |
| S3 | `&` replaces `*` for laws, kernels, records, and declarations; `rename` and `MarginalView`, in one announced mechanical PR | declarations Part 2 §3 | S2 | yes | | |
| S4 | Every batch stores its element declaration | declarations; addressing: Batches | S2, E3 | yes | | |
| G1 | Grammar B for the kinds the workshop uses: a view or marginal of a group exposes its parts; the library's laws declared whole over a record are exposed or wrapped; a factor that holds a group wraps it; a function that returns a law or a function exposes its output; a lifted law-returning function wraps its draws; defaults by kind; the reading of a bare `output_spec` | addressing: The grammar rule, Structured outputs, Defaults | S2, S4 | yes | | |
| G2 | Grammar B for the remaining kinds, then enforcement: random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets wrap their draws; then the keyword form of `OutputSpec` refuses a spec with parts | addressing: The grammar rule, What grammar B costs | G1 | after | | |
| G3 | Paths through every kind; `Function[]`; the public `component` accessor; a view of a field that holds a law or a function wraps | addressing: Indexing each kind; gaps D4 | G1 | after | | |
| G4 | `condition_on` declared as `d.without(*fixed)`; `law.without` | addressing: Operations that take paths | G1 | after | | |
| G5 | `[]` on batches of laws and on kernels | addressing: Indexing each kind | G3 | after | | |

### The execution layer

| Step | What | Plan sections | Depends on | Before | PR | State |
|---|---|---|---|---|---|---|
| E1 | Design text: V.9 rewritten around point sets and executors, the new V.13, the controls in V.2 and VI.0, the glossary | §11, §13 item 1 | PL | yes | | |
| E2 | Claims and the probe result: `traceable`, `vectorized`, one probe per callable and input structure, shared by `using` copies and run by `check`; the error for a false claim | §6.2 | E1 | yes | | |
| E3 | Point sets and the executor registry: `jax`, `sequential`, `thread`, `process` replace the dispatch modes; the controls `executor` and `executor_options`; the closed execution contract goes. First commit: #524's two module moves | §6.3, §6.7, §6.8 | E1, P0 | yes | | |
| E4 | The boundary: embedded calls and call tokens in every program ProbPipe opens; a failure raises `PointEvaluationError` and stops a compiled program | §6.4, §6.9 | E2, E3, S3 | yes | | |
| E5 | `map_points` in lifting and sweeps, joint factors, laws built from functions, and expectations | §6.1, §13 item 5 | E4 | yes | | |
| E6 | The public target interface: `InferenceTarget` and `make_posterior`, with the run key passed as #554 §8 states | §6.12 | E2, E3 | yes | | |
| E7 | At most one run per distinct point inside a call | §6.5, Q17 | E4 | yes, cut second | | |
| E8 | The JAX settings contract | §6.15 | E3 | yes, cut first | | |

The execution plan's post-workshop steps are listed in its §13 and in `deferred.md`.

## Starting a session on a step

1. Read this file in full, then `decisions.md`, then the plans (directive 1).
2. Check that every dependency of the step has merged: `gh pr view <number> --json state,mergedAt` for each PR in the step's row. If one has not, stop and report which, unless the step may stack on it.
3. `git fetch origin`, then create the step's branch from `origin/dev/overhaul`. The session's worktree may have started on another branch, such as `main`; never base a step on it.
4. Read the code the step changes, and the code that already does anything similar (directive 5).

## Ending a session

1. Open the PR into `dev/overhaul`, with a body that follows directives 4, 9, and 10 and the PR template.
2. Update the step's row in § Steps: the PR link and the state.
3. Record any decision in `decisions.md`, and any postponed work in `deferred.md`, in the same PR.
4. Create the handoffs (below).

## Handoffs

A session that opens a PR creates a handoff for each step whose only unmet dependency is that PR.
In the Claude desktop app, a handoff is a task chip made with the `spawn_task` tool: the maintainer clicks it to open a new session in a fresh worktree.
Where `spawn_task` is not available, put the same prompt in the PR body under "Handoff".

The chip's title names the step, such as "Implement step E3: point sets and the executor registry".
Its prompt stands alone, since the new session sees nothing of the old one, and follows this template:

```text
You are starting step <ID> (<title>) of ProbPipe's contracts push.

Repository: <absolute path of the main checkout>. Work in the fresh worktree this session opened.

1. Read plans/README.md in full and follow its directives. Then read plans/decisions.md
   and all the plans in plans/, not only the sections for this step:
   <the plan sections this step implements>.
2. This step depends on <PR numbers and steps>. Check each with
   `gh pr view <number> --json state,mergedAt`. If any has not merged into dev/overhaul,
   stop and tell the maintainer which one; do not start the work.
3. Run `git fetch origin` and create the branch dev/<name> from origin/dev/overhaul.
   Do not build on the branch this worktree started on.
4. Context from the previous step:
   - what it delivered: <summary, PR link>;
   - what it found that this step needs to know: <surprises, constraints>;
   - decisions it recorded: <entries in plans/decisions.md>;
   - what it left open: <threads>.
5. Do the step as plans/README.md § Steps describes it. Stop and ask the maintainer
   whenever plans/README.md § When to stop and ask applies.
6. Finish as plans/README.md § Ending a session describes, including the handoffs.
```

## Checkpoints

A checkpoint is a moment to assess the push broadly, such as after S2 and E3 have merged, or before the workshop freeze.
At a checkpoint:
1. run `/review-all` on the latest merged work, or a fresh-eyes review of the merged code against every plan;
2. update the plans and this file where the code has taught something;
3. update the steps that remain.

## When the push ends

Each step's design text moves its sections from `plans/` into `design/`, which stays the single source of the target design.
When every section has moved, `plans/` is deleted; git keeps its history.
The items left in `deferred.md` become GitHub issues.
