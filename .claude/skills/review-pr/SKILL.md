---
name: review-pr
description: Review a ProbPipe PR for documentation, tests, API consistency, philosophy adherence, and code quality. Use when asked to review a PR or check for issues.
allowed-tools: Read Grep Glob Bash(gh *) Bash(git *) Agent
argument-hint: [pr-number]
---

# ProbPipe PR Review

Review pull request **$ARGUMENTS** (if no number given, detect the PR for the current branch using `gh pr view`).

## Instructions

You are performing a read-only review. **Do not edit any files.** Your job is to
analyze the PR and present a structured report of findings with concrete
suggestions. The user will decide which suggestions to act on before any changes
are made.

## Step 1: Gather context

Fetch the PR metadata and diff:

```
gh pr view $ARGUMENTS --json title,body,baseRefName,headRefName,files
gh pr diff $ARGUMENTS
```

Identify which files were added, modified, or deleted. Read the full contents of
every modified and newly added file so you have complete context (not just the
diff hunks).

Also read these project references — they define the conventions you are checking
against:

- `STYLE_GUIDE.md`
- `CONTRIBUTING.md`

## Step 2: Run the review checklist

Work through **every** category below. For each, note specific findings with
file paths and line numbers. If a category has no issues, say so briefly.

### 2.1 Documentation

- Are new or modified public classes, functions, and modules documented with
  NumPy-style docstrings (`Parameters`, `Returns`, `Raises`)?
- Are existing docstrings still accurate after the changes, or have they become
  stale (e.g., parameter added but not documented, behavior changed but docstring
  not updated)?
- Do new modules have a module-level docstring explaining their purpose?
- If the PR adds user-facing features, are the relevant docs pages in `docs/`
  updated?

### 2.2 Test coverage

- Does every new public function/class/method have corresponding tests?
- Are there edge cases that lack test coverage (empty inputs, boundary values,
  error paths)?
- Have any existing tests become stale — testing behavior that no longer matches
  the implementation?
- Do tests follow project conventions: `Test*` classes, pytest fixtures,
  parametrized distribution families, `pytest.importorskip` for optional deps?
- Is `__all__` updated in `__init__.py` files when new public symbols are added?

### 2.3 ProbPipe philosophy

Check that the PR adheres to the core design principles:

- **Immutability** — Distribution parameters are fixed at construction. No
  mutation of distribution state after `__init__`.
- **Ops, not methods** — Public operations (`sample`, `mean`, `log_prob`,
  `condition_on`, etc.) live in `probpipe.core.ops` as `WorkflowFunction`
  instances. Users call `mean(dist)`, never `dist.mean()` or `dist._mean()`.
- **Protocol-based dispatch** — Capabilities are declared via `@runtime_checkable`
  protocols (`SupportsSampling`, `SupportsLogProb`, etc.). Concrete classes
  implement the underscore methods (`_sample`, `_log_prob`, `_mean`).
- **Private method convention** — Protocol methods use single underscore prefix.
  Public API is always through ops.
- **WorkflowFunction patterns** — Standalone workflow functions follow the
  `_<name>_impl` + `<name> = WorkflowFunction(...)` pattern.

### 2.4 API consistency

- Do new names follow STYLE_GUIDE.md conventions? (`Supports<Cap>` for protocols,
  `_method` for protocol methods, `snake_case` for ops, CamelCase for classes,
  `_underscore.py` for private modules.)
- Are reserved parameter names (`seed`, `n_broadcast_samples`, `include_inputs`)
  avoided in user-defined functions?
- Does the `.n` property convention hold for new finite-sample distribution
  classes?
- Are imports structured correctly? (`from __future__ import annotations` first,
  relative internal imports, proper grouping.)
- Are `__all__` exports updated?
- Is the subpackage dependency graph respected? (See STYLE_GUIDE.md section 6.)

### 2.5 Duplicate and redundant code

- Does the PR reimplement logic that already exists in ProbPipe's abstractions?
  Common offenders:
  - Custom weight handling instead of using the `Weights` class
  - Manual sampling loops instead of using `WorkflowFunction` broadcasting
  - Bespoke protocol checks instead of using `isinstance` with existing protocols
  - Re-implementing ops logic instead of calling the ops
- Is there copy-pasted code that should be factored into a shared helper?
- Are there unnecessary abstractions or over-engineering for what the PR needs?

### 2.6 AI artifact comments

Flag any comments that look like AI thinking artifacts rather than intentional
documentation:

- Comments starting with "wait", "hmm", "actually", "let me think", etc.
- Overly verbose explainer comments that restate what the code obviously does
- `# TODO` or `# FIXME` comments that were not in the original code and seem
  like AI planning artifacts rather than genuine action items
- Comments that narrate the development process rather than explain the code

### 2.7 General concerns

- Are there edge cases that could cause runtime errors (e.g., empty arrays,
  shape mismatches, division by zero)?
- Could any changes break backward compatibility for existing users?
- Are there performance concerns (unnecessary copies, unvectorized loops over
  large arrays, repeated recompilation)?
- Are error messages clear and actionable when protocol checks fail?

### 2.8 Code quality

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
