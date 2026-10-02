---
name: check-pr
description: Run the pre-PR checks of a ProbPipe branch and draft its PR body from the template. Use before opening a PR or updating its description.
allowed-tools: Read Grep Glob Write Bash(git *) Bash(gh *) Bash(uv run *) Bash(python3 *) Bash(pre-commit *)
argument-hint: [base-branch]
---

# ProbPipe pre-PR check

Check the current branch against base branch **$ARGUMENTS** (if no base is
given, use `main`), then draft the PR body. Each step cites the document that
owns its rule; read that section rather than relying on memory of it.

This skill changes no code. It reports findings, writes the draft body to a
file, and opens no PR unless the user asks.

## Step 1: Gather the change

```
git log --oneline <base>..HEAD
git diff --stat <base>...HEAD
git diff --name-only <base>...HEAD
```

Note what the branch changes and which packages, design sections, and
documents it touches. Read every changed file in full.

## Step 2: Check the branch, the title, and the commits

- The branch name follows `CONTRIBUTING.md` § Branch naming.
- The proposed PR title follows `CONTRIBUTING.md` § Opening the PR.
- Commit messages and the PR text are self-contained, as that section requires.

`scripts/ci/pr_hygiene.py` checks the title and the branch as the advisory CI
job does:

```
PR_TITLE="<title>" PR_BRANCH="$(git branch --show-current)" PR_BODY="" python3 scripts/ci/pr_hygiene.py
```

Its body findings are expected at this step, since the body is drafted in Step 5.

## Step 3: Verify

Follow `AGENTS.md` § Verify:

1. select the tests with `python3 scripts/ci/import_graph.py test-targets`, and
   add each changed test file;
2. run them, and compare the pass, skip, and xfail counts with the base branch;
3. run `tests/docs/`, and for a change to `design/` run the `design-check` skill;
4. run the ruff gate.

Record each command and its counts for the test plan.

## Step 4: Check the contracts and the documentation

- For each abstraction the branch adds or changes, apply the directives of
  `CONTRACTS.md` and note whether its contract is unambiguous. List each
  ambiguity and the decision that resolved it, for the contract assessment.
- Audit each touched docstring against `CONTRACTS.md` directive 2: its Raises
  section, its shapes, and drift from the behavior.
- Check the CHANGELOG entry against `CONTRIBUTING.md` § Opening the PR.
- Check that the documents a convention change touches are updated, as
  `CONTRIBUTING.md` § Documentation requires.

## Step 5: Draft the PR body

Copy `.github/PULL_REQUEST_TEMPLATE.md` and fill each section:

- **Summary**: what the PR does and why, motivated from the repository's code,
  merged PRs, `design/`, and open issues;
- **Linked issue(s)**: one of the three forms that the template's comment
  states;
- **Contract assessment**: the result of Step 4;
- **Test plan**: the commands of Step 3 and their counts;
- **Breaking changes**, **Documentation**, and **Checklist**: tick an item only
  when a step above verified it.

Write the body to a file, which `gh pr create --body-file` or
`gh pr edit --body-file` takes, and run `scripts/ci/pr_hygiene.py` again with
`PR_BODY="$(cat <file>)"`.

## Step 6: Report

Present the findings, grouped as blocking (a failing test or check) and
recommended, followed by the path of the draft body. Wait for the user before
pushing or opening the PR.
