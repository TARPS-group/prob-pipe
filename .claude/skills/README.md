# Claude Code Skills

This directory contains project-level [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills)
(slash commands) available to all ProbPipe contributors. Skills are invoked
from the Claude Code CLI or IDE extension.

## Available Skills

### `/review-pr [number]`

Performs a structured review of a ProbPipe pull request.

**When to use:** Before merging a PR, or when reviewing someone else's PR for
issues and consistency with ProbPipe conventions.

**What it checks:**
- ProbPipe philosophy and convention adherence (read from `AGENTS.md`,
  `CONTRACTS.md`, `STYLE_GUIDE.md`, `CONTRIBUTING.md`, and `design/` at review
  time, so it stays current as conventions evolve)
- Documentation completeness and staleness — including whether the PR should
  have updated a rule document or `design/`
- Test coverage gaps
- Duplicate or redundant code (checks against existing abstractions in the
  codebase)
- AI artifact comments (thinking-aloud comments, unnecessary narration)
- General concerns (edge cases, backward compatibility, performance)
- Code quality (simplification, dead code, JAX idioms)
- When `plans/` exists, the PR's fit with the running multi-PR push: its step, the
  standing directives of `plans/README.md`, and the settled decisions

**Usage:**
```
/review-pr          # reviews the PR for the current branch
/review-pr 103      # reviews PR #103
```

**Behavior:** Claude analyzes the PR and presents a structured report organized
by severity (critical, recommended, minor) with concrete suggestions including
file paths and line numbers. No changes are made until the user approves
specific suggestions.

This skill can also be triggered automatically when asking Claude to review a
PR in general conversation.

---

### `/audit-tests [file-or-pattern]`

Audits the ProbPipe test suite for quality, correctness, and completeness.

**When to use:** Periodically to check test suite health, after a major
refactor, or to audit specific test files before a release.

**What it checks:**
- **Test gaps** — missing coverage for public APIs, edge cases, error paths,
  protocol compliance
- **Stale tests** — tests referencing renamed/removed APIs or outdated behavior
- **Duplicate tests** — redundant tests (vs legitimate parametrization, which
  is intentional)
- **Misleading tests** — tests that don't actually test what their name claims,
  or use trivially-true assertions
- **Sham tests** — tests that mock away the thing being tested, inflate
  coverage without validating behavior, or use trivial inputs designed to pass
- **Mathematical correctness** — validates that tests use independent baselines
  (analytical formulas, scipy, numerical approximations) rather than
  self-referential checks; flags inappropriate tolerances
- **Style and conventions** — checks against `STYLE_GUIDE.md` §8, the
  contract tests of `CONTRACTS.md` directive 3, and existing suite patterns

**Usage:**
```
/audit-tests                          # audits the entire test suite
/audit-tests tests/families/test_continuous.py # audits a specific file
/audit-tests tests/families/test_*            # audits by pattern
```

**Behavior:** Claude reads both the test files and the source code they
exercise, then presents a structured report organized by severity (critical,
gaps, recommended, minor). No changes are made until the user approves specific
suggestions.

This skill runs only when its slash command invokes it.

---

### `/review-all [number]`

Runs four independent PR reviews in parallel and merges them into one
deduplicated, severity-tiered report.

**When to use:** Before merging a non-trivial PR, or when correctness matters
enough to want an adversarial second, third, and fourth opinion. For a quick look,
`/review-pr` alone is enough — `/review-all` is the heavier, multi-lens pass.

**What it does:**
- Fans out four concurrent reviews, each invoking a different skill:
  - `/review-pr` — ProbPipe conventions, docs, tests, duplicate code, AI-artifact comments
  - `/review` — general correctness, conventions, performance, test coverage, security
  - `/code-review` — focused, high-effort correctness bug-hunt of the diff
  - `/audit-tests` — test-suite quality: gaps, weak assertions, stale/duplicate tests, assertion correctness
- Consolidates the four into a single report: deduplicates overlapping
  findings, re-tiers by consensus severity, attributes each finding to the
  lens(es) that raised it, surfaces disagreements between lenses, and keeps a
  "verified correct" section so the sound core is clear.

**Usage:**
```
/review-all          # reviews the PR for the current branch
/review-all 273      # reviews PR #273
```

**Behavior:** Read-only. Presents the consolidated report and waits for the user
to decide what to act on. Four agents cost many tokens, so this skill runs only
when its slash command invokes it, and a general "review this PR" request routes
to `/review-pr`.

---

### `/check-pr [base-branch]`

Runs the pre-PR procedure on the current branch and drafts the PR body.

**When to use:** Before opening a PR, or before updating a PR's description.

**What it does:**
- Checks the branch, the title, and the commits against `CONTRIBUTING.md`, with
  `scripts/ci/pr_hygiene.py`
- Runs the verification steps of `AGENTS.md` § Verify
- Audits the changed contracts and docstrings against `CONTRACTS.md`
- Drafts the body from `.github/PULL_REQUEST_TEMPLATE.md`, with the contract
  assessment and the test plan filled in, and writes it to a file

**Usage:**
```
/check-pr          # compares the branch with main
/check-pr dev/x    # compares the branch with dev/x
```

**Behavior:** Changes no code and opens no PR unless asked.

---

### `/design-check [base-branch or section ids]`

Checks the design sections a change edits or realizes against the code and the
design's conventions.

**When to use:** When a change edits `design/` or a module that realizes a
design section.

**What it does:**
- Finds the sections from the diff, or takes them as arguments
- Runs `scripts/design/design_blocks.py` `list` and `check` on them, and the
  five conformance tests
- Runs `scripts/design/ledger.py` and `tests/docs/`
- Checks those sections against `design/README.md` § Conventions, and runs
  `scripts/design/prose.py` on the changed design files

**Usage:**
```
/design-check            # the sections the branch changes, against main
/design-check III.7 VI.3 # the named sections
```

**Behavior:** Read-only; reports each finding with the command that found it.

---

### `/design-pipeline [problem description or project to resume]`

Helps a user design a ProbPipe data analysis pipeline for their own problem.

**When to use:** When starting a new analysis with ProbPipe, or when turning an
existing model and dataset into a ProbPipe pipeline. The model can be a paper,
a simulator, or code, such as a PyMC, Stan, or JAX model.

**Other agents:** The instructions are written for any agent that can read
files and run commands. `SKILL.md` is the Claude Code entry point, which maps
the slash command's argument and the agent's tools to the agent-neutral
instructions. Another agent starts from the prompt of
`docs/get_started/design_assistant.md`, which clones ProbPipe, installs it,
builds the corpus, and reads the instructions.

**Files of `design-pipeline/`:**
- `design-pipeline/INSTRUCTIONS.md`: the role, the ground rules, the table of
  phases, and the resume protocol;
- `design-pipeline/phases/0-load.md` to `design-pipeline/phases/5-feedback.md`:
  one file per phase, which the agent reads when the phase starts, and
  `design-pipeline/phases/4-existing-pipeline.md`, which takes the place of
  the proposal's steps for a pipeline that the user already runs;
- `design-pipeline/CAPABILITIES.md`: the routes by which a user's model enters
  ProbPipe, and each need of a pipeline with its ProbPipe entry points, its
  example page, and its workflow stage;
- `design-pipeline/RESOURCES.md`: the Bayesian workflow checklist and the
  annotated bibliography;
- `design-pipeline/BACKGROUND-RUBRIC.md`,
  `design-pipeline/DESIGN-DOC-TEMPLATE.md`, and
  `design-pipeline/MEMORY-FORMAT.md`: the background rubric, the
  design-document template, and the memory and workspace format;
- `design-pipeline/build_corpus.py`: flattens the README, the documentation
  site, and the design reference into part files of bounded length, and
  writes beside them an index of the exports and inference methods and a
  manifest of the checkout's version and of the line of each heading;
- `design-pipeline/check_symbols.py`: checks the ProbPipe names and call
  arguments of a pipeline script or notebook, and of the modules beside it
  that it imports, against the installed package, and reports each call of
  an export that is not implemented.

**What it does:**
1. **Loads knowledge on demand:** replies first, builds the corpus, reads the
   index of exports and inference methods, the overview tier, and
   `CAPABILITIES.md`, and searches the rest of the corpus when a question
   needs it.
2. **Intake:** asks about the problem, the scientific and analysis goals, and
   the constraints.
3. **Materials:** asks for pointers to the data and to existing models,
   inspects the data, and extracts the model structure from papers or code.
4. **Working preferences:** proposes how it will work with the user, such as
   how much it explains, from what the user says of their experience on the
   dimensions of the [rubric](design-pipeline/BACKGROUND-RUBRIC.md), and
   confirms the preferences in one line of the proposal.
5. **Initial proposal:** writes a versioned design document
   ([template](design-pipeline/DESIGN-DOC-TEMPLATE.md)) and a runnable
   `<workspace>/pipeline.py` or `<workspace>/pipeline.ipynb`, checks it with
   `check_symbols.py`, and runs it in stages: small budgets under
   `STAGE=smoke`, then the validation on simulated data, and the fit to the
   real data only after the validation passes. For a pipeline that the user
   already runs, it first reproduces the pipeline's results through ProbPipe,
   and then adds ProbPipe's checks one at a time.
6. **Feedback loop:** revises the design and the code from the user's
   feedback, with a changelog, until the user is done.

**Usage:**
```
/design-pipeline
/design-pipeline estimating disease transmission rates from weekly case counts
/design-pipeline resume flu-transmission      # resumes a project
```

**Behavior:** Keeps two memories across sessions
([format](design-pipeline/MEMORY-FORMAT.md)):
- a person memory, at `~/.probpipe-design/user.md` or under
  `$PROBPIPE_DESIGN_HOME`, which holds the working preferences, the experience
  that the user stated, the environment, and an index of projects;
- a project memory in each workspace, `~/probpipe-projects/<project>/` by
  default, which holds the decisions, the options ruled out, the corpus
  version, and where the work stopped.

It asks before it creates person memory, and a returning user skips the
questions already answered. It starts in quick start mode, asks at most three
questions in a message, each with a default, and answers `status` with where
the project stands. It reads user-supplied data and code as untrusted
input, and asks before it runs code or installs anything. Before it creates a
workspace inside a git repository, it checks that git ignores the workspace.

This skill runs only when its slash command invokes it.

## Adding New Skills

To add a new skill, create a directory under `.claude/skills/` with a
`SKILL.md` file:

```
.claude/skills/<skill-name>/SKILL.md
```

See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/skills)
for the skill file format. Commit the skill to version control so all
contributors have access.
