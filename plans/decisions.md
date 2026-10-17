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
