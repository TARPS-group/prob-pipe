---
name: audit-tests
description: Audit the ProbPipe test suite for gaps, stale tests, duplicates, style issues, weak assertions, and mathematical correctness.
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash(pytest *) Bash(git *) Agent
argument-hint: [test-file-or-pattern]
---

# ProbPipe Test Suite Audit

Audit the test suite for quality, correctness, and completeness.

**Scope:** If `$ARGUMENTS` is provided, audit only the specified files or
patterns (e.g., `tests/test_continuous.py`, `tests/test_joint*`). Otherwise,
audit the entire `tests/` directory.

## Instructions

You are performing a read-only audit. **Do not edit any files.** Your job is to
analyze the tests and present a structured report of findings with concrete
suggestions. The user will decide which suggestions to act on before any changes
are made.

## Step 1: Gather context

### 1a. Read project conventions

Read the testing conventions section of `STYLE_GUIDE.md` and the architecture
overview in `CONTRIBUTING.md`. These define the expected test structure, naming,
fixture patterns, and project abstractions that tests should exercise.

### 1b. Understand what is being tested

Read the source files that the tests under audit are exercising. You need to
understand the actual behavior in order to judge whether tests are correct,
complete, and non-stale. For a full audit, scan `probpipe/__init__.py` for the
public API surface, then read source files as needed per test file.

### 1c. Read the tests

Read every test file in scope. Also read:

- `tests/conftest.py` — shared fixtures
- `pyproject.toml` — pytest configuration, markers, xdist settings

## Step 2: Run the audit checklist

Work through **every** category below for each test file in scope. Note specific
findings with file paths, line numbers, and the test function/class name. If a
category has no issues for a file, say so briefly.

### 2.1 Test gaps

Identify source code that lacks test coverage:

- Public classes, functions, or methods with no corresponding tests
- Code paths exercised only in the "happy path" — missing tests for edge cases
  (empty inputs, zero-length arrays, boundary values, single-element collections,
  degenerate parameters)
- Error paths: are `TypeError`, `ValueError`, and other expected exceptions
  tested? Do tests verify that the correct exception type and message are raised?
- Protocol compliance: if a class implements protocols (`SupportsSampling`,
  `SupportsLogProb`, etc.), is each protocol method tested?
- New abstractions or registries: are registration, dispatch, priority ordering,
  and fallback behavior all tested?

### 2.2 Stale tests

Identify tests that have become outdated:

- Tests that reference APIs, parameters, classes, or behaviors that no longer
  exist or have been renamed
- Tests whose assertions pass but no longer validate the intended behavior
  because the implementation changed (e.g., testing a default value that was
  changed, or asserting a shape that is no longer the expected shape)
- Tests that import from private modules that have been restructured
- Fixtures that create objects with outdated constructor signatures

### 2.3 Duplicate and redundant tests

- Tests that assert the exact same behavior as another test (possibly in a
  different file)
- Test files that overlap significantly in what they cover — could they be
  consolidated?
- Parametrized tests that redundantly test the same code path with different
  values when a single representative value would suffice (distinguish this from
  legitimate parametrization across distribution families, which is intentional)

### 2.4 Tests that don't test what they claim

- Tests whose name or docstring describes one behavior but whose assertions
  verify something different or something trivially true
- Tests that assert only that code runs without error but don't verify
  correctness of the output (e.g., calling `sample()` but never checking the
  result beyond its existence)
- Tests that use `assert True`, `assert result is not None`, or similarly weak
  assertions where a meaningful value check is possible
- Tests with overly loose tolerances (e.g., `atol=1.0`) that would pass even
  with incorrect results

### 2.5 Sham tests

Flag tests that appear to have been added to make the test suite pass or inflate
coverage rather than to validate real behavior:

- Tests that mock away the very thing they should be testing
- Tests that only verify mock call counts without checking actual behavior
- Tests that assert implementation details (e.g., internal method calls) rather
  than observable behavior
- Tests that construct trivial inputs specifically designed to make assertions
  pass, when realistic inputs would fail
- Tests in `test_coverage_gaps.py` or similar that merely touch code paths
  without meaningful assertions

### 2.6 Mathematical correctness

For tests of mathematical operations (distributions, inference, linear algebra,
expectations, etc.), check:

- **Independent baselines** — Does the test validate results against an
  independent calculation? Valid baselines include:
  - Analytical formulas (e.g., known mean/variance for standard distributions)
  - A different numerical library (e.g., scipy.stats, numpy, TFP directly)
  - A numerical approximation (e.g., finite-difference gradient check, Monte
    Carlo estimate with enough samples for a tight tolerance)
  - Known identities or invariants (e.g., `log_prob` consistency with `prob`,
    KL divergence non-negativity, law of total expectation)

  The baseline should **not** use the same code path as the implementation being
  tested. For example, testing `mean(Normal(0, 1))` by checking it equals
  `Normal(0, 1).loc` just verifies passthrough, not correctness.

- **Tolerance appropriateness** — Are `atol`/`rtol` values tight enough to
  catch bugs but loose enough for floating-point and Monte Carlo noise? Flag
  both overly loose (masks bugs) and overly tight (flaky) tolerances.

- **Shape and broadcasting** — Do tests verify output shapes, especially for
  batched operations, broadcasting, and multi-dimensional inputs?

- **Edge cases specific to math** — Degenerate parameters (zero variance, unit
  matrices), extreme values (very large/small), numerical stability (log-space
  operations, underflow/overflow).

### 2.7 Style and convention compliance

Check tests against the conventions in `STYLE_GUIDE.md` section 8 and patterns
established in the existing suite:

- **Structure** — Are tests organized in `Test*` classes? Are related tests
  grouped logically?
- **Fixtures** — Are fixtures used appropriately? Parametrized fixtures for
  distribution families? Shared fixtures in conftest.py vs local fixtures?
- **Naming** — Do test names clearly describe the behavior being tested?
  `test_<method>_<scenario>` pattern?
- **Imports** — Importing from `probpipe` (public API), not from private modules?
- **Optional dependencies** — Using `pytest.importorskip()` or pytest markers
  for optional backends?
- **Assertions** — Using `np.testing.assert_allclose` for numerical comparisons
  (not bare `assert` with manual tolerance)?

## Step 3: Present findings

Organize your findings into a structured report with this format:

```
## Test Audit: <scope description>

### Summary
<1-2 sentence overall assessment of test suite health>

### Findings

#### Critical (tests that are wrong, misleading, or mask bugs)
- ...

#### Gaps (missing tests that should exist)
- ...

#### Recommended (improvements to existing tests)
- ...

#### Minor (style, naming, cleanup)
- ...

### Suggested Changes
<Numbered list of concrete, actionable changes. For each:
 - The file and test function/class
 - What is wrong or missing
 - What the fix or new test should look like (brief description, not full code)
Group related changes together.>
```

**Do not make any changes.** Present the report and wait for the user to decide
which suggestions to implement. Once the user approves specific items, then
proceed with edits.
