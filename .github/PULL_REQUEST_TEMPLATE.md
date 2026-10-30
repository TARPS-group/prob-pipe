## Summary

<!-- One short paragraph describing what this PR does and why. -->

## Examples

<!--
A short runnable example of each change of behavior, with its real output:
the same call before and after for a changed behavior, and a call that uses a
new feature. Write N/A when the PR has no behavior to run (CONTRIBUTING.md
§ Opening the PR).
-->

## Linked issue(s)

<!--
Use one of:
  Closes #<num>     — issues this PR resolves entirely
  Part of #<num>    — tracking issues this contributes to
  Refs #<num>       — context/discussion that is not closed by this PR
-->

## Contract assessment

<!--
For each abstraction this PR adds or changes, say whether its contract is
unambiguous, and record each decision that resolved an ambiguity
(CONTRIBUTING.md § Changing the code, rule 2). Write "None" when no contract changes.
-->

## Test plan

<!--
What verifies the change?
- New tests added (pytest paths)
- Existing tests changed
- Manual verification steps (if applicable)
- For docs PRs: `mkdocs serve` checked locally
-->

## Breaking changes

<!--
List any user-visible API changes. If none, write "None."
If there are any, add the `kind:breaking-change` label.
-->

## Documentation

<!--
Tick what applies; leave blank if not relevant.
-->

- [ ] User Guide / tutorials updated where relevant
- [ ] CHANGELOG entry added in this PR for a user-visible change
- [ ] A docs page an AI assistant drafted or changed carries the review label that CONTRIBUTING.md § Documentation gives for the change

## Checklist

- [ ] PR title follows `<type>(<scope>): <subject>` (e.g. `refactor(core): ...`, `feat(inference): ...`)
- [ ] Linked to an issue above — or N/A for a small standalone fix (see CONTRIBUTING.md)
- [ ] `ruff check` and `ruff format --check` pass (CONTRIBUTING.md § Linting & pre-commit)

The contract items follow CONTRIBUTING.md § Changing the code and STYLE_GUIDE.md:

- [ ] The design decides: docstrings and names follow the design's target contract and vocabulary, and each conflict, gap, public name, or amendment is raised with the maintainer
- [ ] The contract of every changed abstraction was settled before coding, and the contract assessment records each decision
- [ ] A contract found wrong, missing, or unclear in `design/` or a docstring is fixed or raised with the maintainer, and interim code is labeled as an interim implementation detail
- [ ] Docstrings state each new and changed contract in full and open with the API (STYLE_GUIDE.md §3.4)
- [ ] Tests assert each new and changed contract, its shapes, orderings, and error cases included (STYLE_GUIDE.md §8.7), and names follow STYLE_GUIDE.md §1.12
