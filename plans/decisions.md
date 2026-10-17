# Decisions

This file records every decision taken for the contracts push, and the questions waiting for one.
A decision recorded here binds every step: departing from it means asking again (README § When to stop and ask).
When a decision changes a plan, the plan's text changes in the same commit.

## Waiting for a decision

The open items are in [`gaps-and-conflicts.md`](gaps-and-conflicts.md) until this PR merges.
After that, a question raised during implementation is added here, written as README § When to stop and ask describes.

| Raised | By | Question | Where it is explained |
|---|---|---|---|
| | | | |

## Settled

Newest first.

| Date | Decision | Area | Source |
|---|---|---|---|
| 2026-10-17 | Per-call state is carried by one call context whose core logic is implemented once: #587's carried context and the execution plan's call context become one mechanism. Its precise contract is being designed in `plans/call-context-proposal.md` (gaps D7). | execution | maintainer |
| 2026-10-17 | No name uses "record" for anything but ProbPipe's `Record`: #587's `_CarriedContext.record` and `_MonteCarloRecord` are renamed. | process | maintainer |
| 2026-10-17 | "Point" replaces "managed work item" everywhere: in V.8, the glossary, the code, and its names. The module that `_managed.py` and `_broker.py` merge into is `functions/_point_keys.py`. | execution | maintainer |
| 2026-10-17 | The execution plan's recommendations for Q3, Q4, Q5, Q7, Q8, Q13, Q14, Q15, Q16, and Q18 are accepted, as `gaps-and-conflicts.md` D6 states them. | execution | maintainer |
| 2026-10-17 | Every rename and removal in the push is a clean break: no alias, no deprecation warning, and a `### Changed (breaking)` CHANGELOG entry. This answers the execution plan's Q2: `dispatch` and `max_workers` are removed outright. Release 0.2.0 (#549) follows the push. | process | maintainer |
| 2026-10-17 | Grammar B, not B′: a declaration is whole exactly when its spec is a base spec, with no flag. | addressing | maintainer |
| 2026-10-17 | A view at or below a field that holds a law or a function wraps, keeping the field's name; #571's rule that a law's event is whole or an exposed record stands unchanged. | addressing | maintainer |
| 2026-10-17 | A function with a wrapped output returns the record itself, and the declaration checks it; ProbPipe never wraps a returned value. A pending type filled with a spec that has parts raises. | declarations | maintainer |
| 2026-10-17 | Grammar B for the kinds the workshop uses lands before the workshop (step G1). Random measures, random functions, and Gaussian processes follow after it, together with the refusal of the old whole form (step G2). | addressing | maintainer |
| 2026-10-17 | A batch's component names stay out of its equality, hashing, the static data of its JAX pytree, and its fingerprint. A raw JAX transform therefore drops a whole element's name, as it drops a label, and ProbPipe's operations restore names at their result. This reverses #571's sentence that such batches are unequal. | declarations | maintainer |
| 2026-10-17 | `&` replaces `*` for composition; `with_path_names` becomes `rename`; `FieldView` becomes `MarginalView`. The three change in one mechanical PR (step S3). | declarations, addressing | maintainer |
| 2026-10-17 | The public accessor of the term at a component is `t.component(c)`. | addressing | maintainer |
| 2026-10-17 | The declarations and addressing plans become one reconciled text, `plans/declarations-and-addressing.md`, which #571 and #573 link to. | declarations, addressing | maintainer |
| 2026-10-17 | Where #571's body, jhuggins's reply on #571, and the grammar-B addressing plan disagree, the addressing plan wins, then the reply, then the body. This resolves the thirteen rows of gaps D1. | declarations, addressing | maintainer |
| 2026-10-17 | Every step is a PR into `dev/overhaul`, with no separate branch for the push; the workshop uses a tag cut at the freeze. | process | maintainer |
| 2026-10-17 | The plans of the push are kept in `plans/` in the repository, which replaces the Claude Docs and the local drafts as the copy that changes. | process | maintainer |
| 2026-10-17 | `/review-all` runs only at checkpoints, never after each PR. | process | maintainer |
| 2026-10-17 | A session that opens a PR creates a handoff chip for each step it unblocks; the next session checks that its dependencies have merged before it starts. | process | maintainer |
| 2026-10-17 | `docs/tutorials/` is out of scope for the push; the maintainer rewrites it separately. | process | maintainer |
| 2026-10-16 | Grammar B is adopted for the addressing plan: a whole declaration names only a base spec. jhuggins's final ruling between B and B′ is still open (gaps D4). | addressing | maintainer |
| 2026-10-16 | Paths go through a law, a function, or a batch held in a field. | addressing | maintainer |
| 2026-10-16 | The addressing plan writes composition as `&`; the operator itself is #571's decision (gaps D2). | addressing | maintainer |
| 2026-10-16 | Before the workshop, the execution core is executor dispatch for traceable and opaque functions, and the public target interface. | execution | maintainer, review 1 |
| 2026-10-16 | Q17: inside one call, an opaque `Function` runs at most once per distinct point, by default, and this is in the core. | execution | maintainer, review 2 |
| 2026-10-16 | Q12, an exact result lost to a budget, is a general rule of route resolution, tracked in #585 and outside the execution design. | execution | maintainer, review 2 |
| 2026-10-16 | "Key" may mean a random key; the collision with a mapping's keys is tolerated, and "random key" is written only where it could be ambiguous. | execution | maintainer, review 2 |
| 2026-10-16 | Q1: the selected runner is an **executor**, chosen with `executor=` and `executor_options=`. | execution | maintainer, review 1 |
| 2026-10-15 | A `str` is one path; a list or tuple of `str` is a list of paths. | addressing | maintainer |
| 2026-10-15 | The form of the subscript decides the result: `t["g"]` is the node, `t[["g"]]` a record that holds it. | addressing | maintainer |
| 2026-10-15 | A list of paths keeps each node at its own path, in the original tree's order, and refuses a duplicate path or a path inside another. | addressing | maintainer |
| 2026-10-15 | A value indexed by several paths is labeled by its paths, as `(a, g/c)`. | addressing | maintainer |
| 2026-10-15 | Summaries keep the law's paths, and the spec is the only guard on where a summary binds (PR #576, merged 2026-10-17). | addressing | maintainer |
| 2026-10-15 | `()` and `[]` as a subscript on a batch raise. | addressing | maintainer |
| 2026-10-15 | `condition_on(d, given)` is declared as `d.without(*fixed)`, on every route. | addressing | maintainer |
| 2026-10-15 | A list of paths and `without` partition the leaf paths. | addressing | maintainer |
