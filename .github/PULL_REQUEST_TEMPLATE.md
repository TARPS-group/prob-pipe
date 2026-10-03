## Summary

<!-- One short paragraph describing what this PR does and why. -->

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
(CONTRACTS.md directives 1 and 3). Write "None" when no contract changes.
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

The contract items follow the directives of CONTRACTS.md:

- [ ] Directive 1: the contract of every changed abstraction was clear before coding, and the contract assessment records each ambiguity resolved
- [ ] Directive 2: new and changed contracts are documented in full in their docstrings (types, shapes, orderings, raises, invariants), and each docstring opens with the API
- [ ] Directive 3: the code obeys each documented contract, tests assert it (shapes, orderings, and error cases), and names follow `design/glossary.md` § Canonical names
- [ ] Directive 4: a contract found wrong, missing, or unclear in `design/` or a docstring is fixed or raised with the maintainer
- [ ] Directive 5: docstrings describe the design's target contract and terms, and any temporary coexistence is labeled as an interim implementation detail
