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

Read the following files **from the PR's base ref** — e.g.
`git show origin/main:STYLE_GUIDE.md` after `git fetch origin main` — not from
the local checkout: a worktree copy may be stale relative to the branch the PR
merges into. These are the **authoritative source of truth** for all naming,
style, architecture, and API conventions. Every check you perform in Step 2
must be grounded in what these documents say — do not rely on your own prior
knowledge of ProbPipe conventions, as they may have changed.

- `AGENTS.md` — the map from each task to the document that owns its rules
- `CONTRACTS.md` — the contract directives every PR follows
- `STYLE_GUIDE.md` — naming, imports, types, protocols, testing, module layout
- `CONTRIBUTING.md` — the PR workflow, test quality, documentation, and CI
- `design/README.md`, `design/glossary.md`, and the design section of each
  abstraction the PR changes, which `design/package-structure.md` § The tree
  lists beside each module

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

- `probpipe/__init__.py` — the public API
- the packages of `design/package-structure.md` § The tree that the PR changes
- Any utility modules (`_weights.py`, `_array_utils.py`, `_dtype.py`, etc.)

## Step 2: Run the review checklist

Work through **every** category below. For each, note specific findings with
file paths and line numbers. If a category has no issues, say so briefly. Each
category names the document section that owns its rules; check the PR against
that section as it stands at the base ref.

### 2.1 ProbPipe philosophy and conventions

- **Design** — each changed abstraction follows its design section, and the
  principles of `design/01-design-principles.md`. Classify a disagreement as
  `CONTRACTS.md` directive 5 does: the design decides.
- **Contracts** — the directives of `CONTRACTS.md`.
- **Naming** — `STYLE_GUIDE.md` §1, naming accuracy included, and
  `design/glossary.md` § Canonical names.
- **Imports** — `STYLE_GUIDE.md` §4.
- **Type annotations** — `STYLE_GUIDE.md` §5. Also check for *missing* type
  hints: all new or modified public function signatures (parameters and return
  types) and class attributes should be annotated.
- **Subpackage dependency graph** — `STYLE_GUIDE.md` §6.
- **Registry patterns** — design II.7 and `docs/api/extending.md`.
- **Docstrings** — `CONTRACTS.md` directive 2 and `STYLE_GUIDE.md` §3.
- **`__all__` exports** — `STYLE_GUIDE.md` §9.1.

### 2.2 Documentation

- Are new or modified public classes, functions, and modules documented as
  `CONTRACTS.md` directive 2 requires?
- Are existing docstrings still accurate after the changes, or have they become
  stale (e.g., parameter added but not documented, behavior changed but docstring
  not updated)?
- Do new modules have a module-level docstring explaining their purpose?
- If the PR adds user-facing features, are the relevant docs pages in `docs/`
  updated, as `CONTRIBUTING.md` § Documentation requires?
- **Convention docs consistency** — Does the PR change how things are done (new
  abstractions, new patterns, renamed or removed APIs, new module layout)? If
  so, are the documents that state the convention updated: `STYLE_GUIDE.md`,
  `CONTRIBUTING.md`, `CONTRACTS.md`, `AGENTS.md`, or `design/`? Flag any case
  where a PR changes how things are done but leaves a document describing the
  old way.

### 2.3 Test coverage

- Does every new public function/class/method have corresponding tests?
- Are there edge cases that lack test coverage (empty inputs, boundary values,
  error paths)?
- Have any existing tests become stale — testing behavior that no longer matches
  the implementation?
- Do the tests assert each contract, as `CONTRACTS.md` directive 3 requires?
- Do the tests follow `STYLE_GUIDE.md` §8 and `CONTRIBUTING.md` § Test quality?
  A loose tolerance, a shape-only assertion on inference output where a
  statistical check is feasible, and a dispatch change without an equivalence
  test are findings.

### 2.4 Duplicate and redundant code

Using the abstractions you identified in Step 1c, check whether the PR
reimplements logic that already exists. Common patterns to watch for:

- Custom weight handling instead of using existing weight abstractions
- Manual sampling loops instead of using `Function` broadcasting
- Bespoke protocol checks instead of `isinstance` with existing protocols
- Re-implementing operation logic instead of calling the operations
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
- Comments and docstrings that break `CONTRIBUTING.md` § Code comments &
  docstrings or the writing rules of `STYLE_GUIDE.md` §10

### 2.6 General concerns

- Are there edge cases that could cause runtime errors (e.g., empty arrays,
  shape mismatches, division by zero)?
- Could any changes break backward compatibility for existing users?
- Are there performance concerns (unnecessary copies, unvectorized loops over
  large arrays, repeated recompilation)?
- Do new and changed error and warning messages follow `STYLE_GUIDE.md` §9.3?

### 2.7 Code quality

- Can any code be simplified without losing clarity?
- Are there overly complex expressions that should be broken up?
- Is there dead code, unused imports, or unreachable branches?
- Are there opportunities to use JAX idioms more effectively (e.g., `vmap`
  instead of Python loops, `jnp` instead of `np` where JIT is intended)?

### 2.8 PR body and branch hygiene

- Do the title, the branch, and the body follow `CONTRIBUTING.md` § Opening the
  PR and § Branch naming, and does the body keep every section and checklist
  item of `.github/PULL_REQUEST_TEMPLATE.md`? `scripts/ci/pr_hygiene.py` checks
  their form, as the advisory CI job does.
- Does the title/description match the **final** diff? Flag drift in either
  direction: features described but not present, changes present but
  undescribed.
- Does the contract assessment cover each abstraction the PR changes, as
  `CONTRACTS.md` directive 3 requires?
- Did scratch artifacts leak into the diff — `*_plan.md` files, references to
  local plan directories, leftover debug scripts?
- When the PR pins a version (CI action, tool), is the pin consistent with the
  same pin elsewhere in the repo, and reachable by the automated bumpers
  (Dependabot, `pre-commit autoupdate`)? An inline pin no bumper parses will
  silently drift.

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
