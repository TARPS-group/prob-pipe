---
name: design-check
description: Check the design sections a ProbPipe change edits or realizes against the code and the design's conventions, with scripts/design/design_blocks.py, the conformance tests, and the ledger. Use when a change edits design/ or a module that realizes a design section.
allowed-tools: Read Grep Glob Bash(git *) Bash(uv run *) Bash(python3 *)
argument-hint: [base-branch or section ids]
---

# ProbPipe design check

Check the design sections that a change edits or realizes. **$ARGUMENTS** is a base
branch (default `main`) or a list of section ids, such as `III.7 VI.3`.

This check is read-only: it reports findings and edits nothing.

## Step 1: Find the sections

- A change to `design/` edits each section whose text a hunk of
  `git diff -U0 <base>...HEAD -- design/` changes, and that section is the
  nearest `## <id> — ` heading above the hunk.
- A change to `probpipe/` realizes the sections that
  `design/package-structure.md` § The tree lists beside each changed module.

## Step 2: Compare each section's declarations with the code

```
uv run python scripts/design/design_blocks.py list <ids>
uv run python scripts/design/design_blocks.py check <ids> --module <module> [--module <module> ...]
```

`list` prints what the sections declare. `check` reports each declared class,
member, field, and parameter that the code lacks or declares differently, and
its `--module` options are the modules that `design/package-structure.md`
places the sections in, in order. The four conformance tests hold those module
lists and their known exceptions, so run them too:

```
uv run pytest tests/distributions/test_design_conformance.py tests/families/test_design_conformance.py tests/functions/test_design_conformance.py tests/operations/test_design_conformance.py
```

## Step 3: Run the ledger and the documentation tests

```
uv run python scripts/design/ledger.py
uv run pytest tests/docs
```

The ledger lists the stubs, the pending tests, and the stale uses of removed
names in the docs. `tests/docs/` checks the pointers that the rule documents
make into `design/`.

## Step 4: Check the design's conventions

Read `design/README.md` § Conventions and check each section of Step 1 against
it, and check that each term `design/glossary.md` defines keeps its defined
sense. Then run the prose checker on the changed design files:

```
uv run python scripts/design/prose.py design/<file>.md ...
```

It prints candidates, not verdicts: judge each one against `STYLE_GUIDE.md` §10
and `design/README.md` § Conventions, and report the ones that hold.

## Step 5: Report

For each section, list its findings. Classify a disagreement between the
design and the code as `design/README.md` and `CONTRACTS.md` directive 5 state:
the design ahead of the code, which is expected, or a defect of the design or
of the code. Name the command that found each finding.
