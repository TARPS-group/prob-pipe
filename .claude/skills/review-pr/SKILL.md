---
name: review-pr
description: Review a ProbPipe PR for documentation, tests, API consistency, philosophy adherence, and code quality. Use when asked to review a PR or check for issues.
allowed-tools: Read Grep Glob Bash(gh *) Bash(git *) Agent
argument-hint: [pr-number]
---

# ProbPipe PR Review

Review pull request **$ARGUMENTS** (if no number given, detect the PR for the
current branch using `gh pr view`).

## Instructions

You are performing a read-only review. **Do not edit any files.** Your job is to
analyze the PR and present a structured report of findings with concrete
suggestions. The user will decide which suggestions to act on before any changes
are made.

## Step 1: Gather context

### 1a. Read project conventions (do this first)

Read the following files in full. These are the **authoritative source of truth**
for all naming, style, architecture, and API conventions. Every check you perform
in Step 2 must be grounded in what these documents say — do not rely on your own
prior knowledge of ProbPipe conventions, as they may have changed.

- `STYLE_GUIDE.md` — naming, imports, types, protocols, testing, module layout
- `CONTRIBUTING.md` — architecture overview, design principles, package
  structure, dependency graph, registry patterns, PR workflow

### 1b. Fetch the PR

```
gh pr view $ARGUMENTS --json title,body,baseRefName,headRefName,files
gh pr diff $ARGUMENTS
```

Identify which files were added, modified, or deleted. Read the full contents of
every modified and newly added file so you have complete context (not just the
diff hunks).

### 1c. Understand existing abstractions

Before checking for redundant code, familiarize yourself with the abstractions
already available in the codebase. Scan these areas for classes, utilities, and
patterns that the PR's code should be using rather than reimplementing:

- `probpipe/__init__.py` — the public API surface
- `probpipe/core/` — base classes, protocols, ops, registries
- Any utility modules (`_weights.py`, `_utils.py`, `_array_utils.py`, etc.)

## Step 2: Run the review checklist

Work through **every** category below. For each, note specific findings with
file paths and line numbers. If a category has no issues, say so briefly.

### 2.1 ProbPipe philosophy and conventions

Check that the PR adheres to every convention documented in `STYLE_GUIDE.md` and
`CONTRIBUTING.md`. These include (but are not limited to) — always defer to what
the docs actually say over this summary:

- **Design principles** — immutability, ops-not-methods, protocol-based dispatch,
  private method convention, etc. (see CONTRIBUTING.md "Design principles")
- **Naming** — protocols, ops, implementation functions, classes, modules,
  reserved parameter names, the `.n` property convention (see STYLE_GUIDE.md)
- **Imports** — `from __future__ import annotations`, relative internals, import
  order, optional dependency patterns, `TYPE_CHECKING` guards
- **Type annotations** — modern Python 3.12+ syntax, project type aliases
- **Subpackage dependency graph** — no illegal cross-imports (see STYLE_GUIDE.md
  section 6 and CONTRIBUTING.md)
- **Registry patterns** — if the PR adds or modifies registry-dispatched
  behavior, check it follows the documented registry conventions (converter
  registry, inference method registry, etc.)
- **Docstrings** — NumPy-style, module docstrings, section separators
- **`__all__` exports** — updated in `__init__.py` when new public symbols are
  added

### 2.2 Documentation

- Are new or modified public classes, functions, and modules documented with
  NumPy-style docstrings (`Parameters`, `Returns`, `Raises`)?
- Are existing docstrings still accurate after the changes, or have they become
  stale (e.g., parameter added but not documented, behavior changed but docstring
  not updated)?
- Do new modules have a module-level docstring explaining their purpose?
- If the PR adds user-facing features, are the relevant docs pages in `docs/`
  updated?
- **Convention docs consistency** — Does the PR introduce changes that affect
  project-wide conventions (e.g., new abstractions, new patterns, changes to the
  class hierarchy, new registry types, renamed or removed APIs, new reserved
  parameter names, new module layout)? If so, are `STYLE_GUIDE.md` and/or
  `CONTRIBUTING.md` updated to reflect those changes? Flag any case where a PR
  changes how things are done but leaves the convention docs describing the old
  way.

### 2.3 Test coverage

- Does every new public function/class/method have corresponding tests?
- Are there edge cases that lack test coverage (empty inputs, boundary values,
  error paths)?
- Have any existing tests become stale — testing behavior that no longer matches
  the implementation?
- Do tests follow project conventions (see STYLE_GUIDE.md section 8)?

### 2.4 Duplicate and redundant code

Using the abstractions you identified in Step 1c, check whether the PR
reimplements logic that already exists. Common patterns to watch for:

- Custom weight handling instead of using existing weight abstractions
- Manual sampling loops instead of using `WorkflowFunction` broadcasting
- Bespoke protocol checks instead of `isinstance` with existing protocols
- Re-implementing ops logic instead of calling the ops
- Duplicating registry dispatch logic instead of using existing registries

Also flag:
- Copy-pasted code that should be factored into a shared helper
- Unnecessary abstractions or over-engineering for what the PR actually needs

### 2.5 AI artifact comments

Flag any comments that look like AI thinking artifacts rather than intentional
documentation:

- Comments starting with "wait", "hmm", "actually", "let me think", etc.
- Overly verbose explainer comments that restate what the code obviously does
- `# TODO` or `# FIXME` comments that were not in the original code and seem
  like AI planning artifacts rather than genuine action items
- Comments that narrate the development process rather than explain the code

### 2.6 General concerns

- Are there edge cases that could cause runtime errors (e.g., empty arrays,
  shape mismatches, division by zero)?
- Could any changes break backward compatibility for existing users?
- Are there performance concerns (unnecessary copies, unvectorized loops over
  large arrays, repeated recompilation)?
- Are error messages clear and actionable when protocol checks fail?

### 2.7 Code quality

- Can any code be simplified without losing clarity?
- Are there overly complex expressions that should be broken up?
- Is there dead code, unused imports, or unreachable branches?
- Are there opportunities to use JAX idioms more effectively (e.g., `vmap`
  instead of Python loops, `jnp` instead of `np` where JIT is intended)?

## Step 3: Present findings

Organize your findings into a structured report with this format:

```
## PR Review: <PR title>

### Summary
<1-2 sentence overall assessment>

### Findings

#### Critical (must fix)
- ...

#### Recommended (should fix)
- ...

#### Minor (nice to have)
- ...

### Suggested Changes
<Numbered list of concrete, actionable changes with file paths and line numbers.
Group related changes together.>
```

**Do not make any changes.** Present the report and wait for the user to decide
which suggestions to implement. Once the user approves specific items, then
proceed with edits.
