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
- ProbPipe philosophy and convention adherence (read from `STYLE_GUIDE.md` and
  `CONTRIBUTING.md` at review time, so it stays current as conventions evolve)
- Documentation completeness and staleness — including whether the PR should
  have updated `STYLE_GUIDE.md` or `CONTRIBUTING.md`
- Test coverage gaps
- Duplicate or redundant code (checks against existing abstractions in the
  codebase)
- AI artifact comments (thinking-aloud comments, unnecessary narration)
- General concerns (edge cases, backward compatibility, performance)
- Code quality (simplification, dead code, JAX idioms)

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
- **Style and conventions** — checks against `STYLE_GUIDE.md` testing
  conventions and existing suite patterns

**Usage:**
```
/audit-tests                          # audits the entire test suite
/audit-tests tests/test_continuous.py # audits a specific file
/audit-tests tests/test_joint*        # audits by pattern
```

**Behavior:** Claude reads both the test files and the source code they
exercise, then presents a structured report organized by severity (critical,
gaps, recommended, minor). No changes are made until the user approves specific
suggestions.

This skill must be explicitly invoked — it will not auto-trigger.

## Adding New Skills

To add a new skill, create a directory under `.claude/skills/` with a
`SKILL.md` file:

```
.claude/skills/<skill-name>/SKILL.md
```

See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/skills)
for the skill file format. Commit the skill to version control so all
contributors have access.
