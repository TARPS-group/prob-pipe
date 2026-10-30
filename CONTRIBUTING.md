# Contributing to ProbPipe

## PR Workflow

### 1. Plan first (for significant changes)

If the change is significant, start by opening a GitHub **issue**
containing the plan: motivation, proposed design, trade-offs, and a
task checklist for the implementation work. Tag with an appropriate
label (typically `enhancement`).
[#196](https://github.com/TARPS-group/prob-pipe/issues/196) is a
representative model.

The issue is the durable record of intent and stays open across the
implementation, even when the work spans multiple PRs — its task
checklist is the cross-PR progress tracker. This is a cleaner split
of concerns than carrying the plan in a PR: the plan document outlives
any single PR, and implementation PRs stay focused on code rather
than doubling as design-discussion threads.

### 2. Request a review

Tag **@jhuggins** and/or **@arob5** on the issue. Feedback lands in
the issue thread.

### 3. Wait for approval before implementing

Do not start implementing until the plan issue is approved. Once
approved, open implementation PR(s) referencing the issue (`Refs
#N` in the PR body for stage PRs, `Closes #N` on the final one).
Check off tasks on the issue as PRs land; the issue closes when the
last task is done.

### What counts as "significant"?

- New subpackages or modules
- Changes to protocols or the base class hierarchy
- New external dependencies
- Architectural changes

For small fixes (typos, bug fixes, test additions), skip the plan step and
go straight to an implementation PR.

### Changing the code

Each abstraction and operation of ProbPipe has a contract: its API, shapes,
return types, error cases, invariants, and parameter names. The section of
`design/` that owns an abstraction states its target contract,
`design/package-structure.md` § Correspondence to the implementation maps each
module to that section, and the module's docstrings state the contract as
implemented. Every change to `probpipe/` follows these rules, whoever makes it.

1. **The design decides.** Where `design/` disagrees with the code or with a
   contributor document, the design decides, and the change moves the code
   toward the design without asking. Docstrings and names follow the design's
   target contract and vocabulary, not the behavior of code elsewhere that the
   design will change. Ask the maintainer in four cases:
   - a conflict, where two parts of the design disagree;
   - a gap, where the design leaves a real choice open;
   - a public name, where the change names or renames a public API;
   - an amendment, where the change would amend the design.
2. **Settle the contract before coding.** Before writing code, make the contract
   of every abstraction the change modifies or calls clear: its shapes, its single and
   batched behavior, its orderings, its error cases, its return types, and its
   invariants. An ambiguity is settled by asking the maintainer or by stating a
   choice, and the PR's contract assessment records each decision.
3. **Fix the contract rather than code around it.** A contract in `design/` or
   in a docstring that turns out wrong, missing, or unclear is fixed in the
   documentation or raised with the maintainer. Code that conflicts with the
   design, or that has vague names, poor organization, or duplicated logic, is
   rewritten within the change's scope rather than patched around. Work outside
   the scope gets an issue.
4. **Label interim code.** A change that must coexist with code that does not
   yet follow the design may describe that coexistence in a docstring as an
   *interim implementation detail*. The note never defines the contract and never blurs a
   distinction the design draws.
5. **Write a question for a reader without the context.** A question to the
   maintainer, on the issue or the PR, gives the background from first
   principles, today's behavior and the proposed behavior in code, the options
   with what each means, and a recommendation with its reason. Until it is
   answered, a change that the question does not block makes the most
   conservative choice consistent with the design, and says so beside the
   question.

`STYLE_GUIDE.md` states the conventions of the code itself:

- names, in `STYLE_GUIDE.md` §1.12;
- the layout of a module, in `STYLE_GUIDE.md` §2;
- what a docstring states, in `STYLE_GUIDE.md` §3.4;
- the tests that assert a contract, in `STYLE_GUIDE.md` §8.7.

### Opening the PR

A few conventions keep PRs consistent; the PR template
(`.github/PULL_REQUEST_TEMPLATE.md`) is the checklist for them:

- **Title** follows `<type>(<scope>): <subject>` — e.g.
  `feat(inference): add BayesFlow backend`, `fix(core): guard empty record`,
  `docs(contributing): document PR conventions`. Common types are `feat`,
  `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, and `ci`; the scope is
  the affected subpackage or area.
- **Description = final state.** The title and body describe the change as
  it stands, and are updated whenever the scope shifts during review — a
  stale description is a review blocker. The body gives a reviewer who does
  not know ProbPipe's internals the context of the change, in prose that
  reads in order rather than in dense lists. The PR text follows the writing
  rules of `STYLE_GUIDE.md` §10, whose rules 9 and 10 keep it free of plan
  labels and self-contained. Scratch planning artifacts, such as `*_plan.md`
  files, stay off the branch. The one exception is `plans/`, which holds the shared
  plans of a multi-PR push while it runs and is deleted once its sections
  have moved into `design/` (`plans/README.md`).
- **Examples** — the body shows each change of behavior as a short runnable
  example with its real output. A changed behavior shows the same call before
  and after, and a new feature shows a call that uses it. A PR with no behavior
  to run, such as a CI or documentation change, writes N/A in that section.
- **Contract assessment** — for each abstraction the PR adds or changes, the
  body says whether its contract is unambiguous and records each decision that
  settled an ambiguity (§ Changing the code, rule 2). A PR that changes no
  contract writes None.
- **Review before requesting review** — open the PR as a draft and run the
  `review-pr` skill on it, or review it against these documents by hand. Fix
  or answer each finding, and then mark the PR ready for review.
- **CHANGELOG** — a PR with a user-visible change adds its entry to
  `CHANGELOG.md` in the same PR, under `## [Unreleased]` and the one heading
  of its change type, such as `### Added` or `### Changed (breaking)`.
- **Labels** — `area:*` labels are auto-applied from the changed paths (see
  [Labels](#labels) below); set `kind:*` / `status:*` by hand, and always add
  `kind:breaking-change` if the PR changes a user-visible API.
- **Linked issue** — reference the plan/tracking issue (`Refs #N` on stage
  PRs, `Closes #N` on the final one). A small standalone fix that skipped the
  plan step (above) has no issue to link; leave that section and its checklist
  item as N/A.

### Branch naming

Use `dev/<short-kebab-case-description>` for branches. Examples:
`dev/record-array-views`, `dev/pr-129-review-fixes`,
`dev/bijector-for-constraint`. Keep the description short (3–5 words)
and tied to the change, not the author or date.

Claude Code auto-generates branch names like
`claude/<adjective-name>-<hash>`. Rename these to a `dev/...` name
**before** opening a PR, or via the GitHub web UI's Branches page
(Branches → pencil icon). The REST `rename` API does **not** redirect
open PRs to the new branch — it silently closes them
(see [#157](https://github.com/TARPS-group/prob-pipe/pull/157) for an
example). If you must rename after a PR is open, use the web UI.

### Labels

ProbPipe uses three label families:

- **`area:*`** — the affected subsystem (`area:core`, `area:distributions`,
  `area:records`, `area:inference`, `area:workflow`, `area:orchestration`,
  `area:diagnostics`, `area:provenance`, `area:docs`,
  `area:infrastructure`). On PRs these are **applied automatically** by
  `.github/workflows/labeler.yml` from the changed file paths (mapping in
  `.github/labeler.yml`), so a PR that touches several areas gets several
  `area:*` labels. On *issues*, apply them by hand — the auto-labeler runs
  on PRs only.
- **`kind:*`** — the nature of the change (`kind:refactor`,
  `kind:breaking-change`, `kind:deprecation`, `kind:tracking`). Always
  human-set; paths cannot infer intent.
- **`status:*`** — workflow state (`status:blocked`, `status:needs-design`,
  `status:needs-review`). Human-set.

`enhancement` and `documentation` remain the catch-all tags for issues.

---

## Development Setup

### Installation

ProbPipe uses [uv](https://docs.astral.sh/uv/) for environment + dependency
management. The dependency tree is locked in `uv.lock` — CI installs from
the same lockfile so a contributor's local env and CI agree.

```bash
# One-time: install uv (see https://docs.astral.sh/uv/getting-started/installation/).
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the dev environment (.venv at the project root):
uv sync --extra dev --extra nutpie            # core + test + nutpie
uv sync --extra dev --extra nutpie --extra pymc   # + pymc backend
```

The `pip install -e ".[dev]"` path still works for contributors with an
existing pip-based setup, but uv is the recommended path. Optional backends
not required to run the test suite locally (their tests skip when the backend
is absent): `bridgestan`, `pymc`, `bayesflow` (amortized SBI; Python 3.12–3.13
only). In CI, `bridgestan` and `bayesflow` are exercised in their own dedicated
legs (the `stan` and `bayesflow` jobs).

These commands build the **`probpipe-core`** distribution (the repository root);
the friendly `probpipe` name is a separate code-less metapackage that bundles the
backends — see *Package Structure* below.

### Everyday commands

`AGENTS.md` § Commands lists the everyday commands, such as those that run the
tests, the linter, and the docs build. `AGENTS.md` § Verify states how to choose
and check the tests a change needs.

### Test quality

- Correctness tests use the **tightest tolerances that pass reliably** —
  a loose tolerance is a review flag, not a convenience.
- Cover structured cases (multi-field Records, mixed scalar/vector
  parameters), error paths, and **equivalence across executors**
  (jax vs sequential) — path divergence has produced real
  silently-wrong-results bugs.
- Inference code gets a statistical sanity check (parameter estimates and
  uncertainty roughly correct on a known target), not just shape
  assertions.
- Coverage targets more than 90% of each module.

### Test quality for numerical code

Coverage is necessary but not sufficient: tests of mathematical behavior
must check correctness against an **independent baseline** (an analytic
result, an exact reference computation, a known invariant, or finite
differences for gradients) — and where the claim is distributional,
check both location and spread. Tolerances on stochastic or trained
estimators are **measured, not guessed**: run the test's configuration
across a few seeds, bound the observed spread with modest margin, and
document the measured range in a comment next to the assertion. Full
conventions in [STYLE_GUIDE.md § 8.6](STYLE_GUIDE.md#86-numerical-correctness-and-tolerances).

### Code formatting

Formatting is owned by **`ruff format`** (Black-style) — don't hand-format
Python. The `ruff-format` pre-commit hook reformats on commit, and CI checks the
formatting with the blocking gate of *Linting & pre-commit* below. A few
specifics: the line
length is 100 (`[tool.ruff]` in `pyproject.toml`); ruff keeps code on one line
when it fits and explodes imports / call arguments one-item-per-line when it does
not; string quotes normalize to double. Notebooks are excluded
(`[tool.ruff.format] exclude` in `pyproject.toml`), so the docs' tutorial cells
keep their compact, hand-curated layout.

### Code comments & docstrings

Comments state constraints and contracts the code cannot express. Match the
comment density of the surrounding code, and when in doubt, delete: an
over-explained obvious line is worse than no comment.

The prose of comments and docstrings follows the writing rules of
`STYLE_GUIDE.md` §10. `STYLE_GUIDE.md` §3.4 states what a docstring
documents, and it bans references to PRs and issues in code.

### Linting & pre-commit

Linting uses [ruff](https://docs.astral.sh/ruff/) (configured in
`pyproject.toml`). Install pre-commit as a uv tool, then install the hooks, once:

```bash
uv tool install pre-commit
pre-commit install
```

The hook script calls the Python interpreter that ran `pre-commit install`. With
`uvx pre-commit install`, that interpreter is kept in the uv cache, so once
`uv cache clean` deletes it, every commit fails with "`pre-commit` not found"
unless another `pre-commit` is on your `PATH`. A `pre-commit` already installed by
Homebrew or pipx works too.

Once the hooks are installed, `ruff` (lint + format), a few file-hygiene hooks,
the `pydoclint` docstring check of STYLE_GUIDE.md §3.3, and the
`no-issue-numbers` hook of STYLE_GUIDE.md §3.4 run on your staged
files at commit time. The hooks see only the files you're
changing, so a commit is checked without re-linting the whole tree.
`AGENTS.md` § Commands runs the linter and the hooks over the whole tree.

A full `pre-commit run --all-files` run may report pre-existing file-hygiene nits (trailing
whitespace, end-of-file) in files you did not change; the fixer hooks clean those
as the relevant files are next edited.

**The ruff gate is blocking.** The `lint & format` CI job runs `ruff check .`
and `ruff format --check .` over the whole tree, which is clean under both, so a
lint violation or a misformatted file fails the build. Rule selection and
per-file ignores are in `[tool.ruff.lint]` in `pyproject.toml`, and the
pre-commit hooks apply the same checks to your staged files at commit time.

**The docstring check is blocking.** The same CI job runs
`pre-commit run pydoclint --all-files`, which checks every module of `probpipe/`.
The hook's entry in `.pre-commit-config.yaml` holds its version and options, so a
local run and CI check the same rules.

### Type checking

Type checking uses [pyright](https://microsoft.github.io/pyright/)
(configured in `pyrightconfig.json`, scoped to the `probpipe` package), and
`AGENTS.md` § Commands runs it in the synced environment.

CI pins a specific pyright version for a reproducible baseline, so a
local run on a newer pyright may report a slightly different count — pin
to match CI (`pyright[nodejs]==<version from ci.yml>`) if you need exact
parity.

Like ruff, **pyright is advisory in CI for now** — the `typecheck
(advisory)` job reports type issues (and shows the count in the run's
job summary) but does not gate merges. The source carries a type-debt
baseline (much of it noise from JAX/TFP untyped attributes), so enforcing
immediately would block unrelated work. The plan is to burn the baseline
down, then tighten `typeCheckingMode` in `pyrightconfig.json` and make the
gate blocking. New code should be clean under the current `basic` mode
where practical.

ProbPipe ships a `py.typed` marker, so the package's annotations are
consumed by downstream users' type checkers — keeping the public API
well-typed is user-facing quality, not just an internal nicety.

### Documentation

`AGENTS.md` § Commands builds and previews the docs site. API docs use
`mkdocstrings` directives in `docs/api/*.md` referencing fully-qualified
Python paths.

A behavior or API change and its documentation ship in the **same PR**:
docstrings, the user-guide notebooks, README / `docs/index.md`, the
CHANGELOG, and STYLE_GUIDE.md / CONTRIBUTING.md when conventions change.
Examples and notebooks show idiomatic usage — never add a compat shim to
keep an example running against an old API.

The prose of the docs follows the writing rules of `STYLE_GUIDE.md` §10, and
each notebook follows three more:

1. **Labeled output:** every printed line names what it shows, as
   `print("mean:", value)` does, or each value gets a cell or a table row of
   its own.
2. **Public names:** output uses public names and says what each number means,
   such as a level's size printed as `levels={'school': 8}`.
3. **No design citations:** a notebook for users cites no section of `design/`
   and no decision identifier, except where it reports a bug against the design.

Each page of the site and README.md carries a review label, which tells users
how far to trust the page and tells maintainers what to review next. A page is
in one of three states:

1. **AI-generated:** an AI assistant drafted the page, and no maintainer has
   reviewed it. Its first line is
   `> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.`
2. **Human-validated:** a maintainer has read the whole page as it renders, run
   its code, and found it correct. Its last line names the reviewer and the
   date, as in `> **Human-validated** by Jonathan Huggins on 2026-10-20.`
3. **AI-revised:** an AI assistant changed a validated page in more than two
   sections. Its first line replaces the validated label and names that review,
   as in
   `> **AI-revised.** An AI assistant changed this page after Jonathan Huggins reviewed it on 2026-10-20, and no maintainer has reviewed the changes yet. Please report errors on the issue tracker.`
   A pair of comments marks each changed part: `<!-- unreviewed: what changed -->`
   before it and `<!-- /unreviewed -->` after it in markdown, and
   `# unreviewed: what changed` and `# /unreviewed` in a notebook's code cell.
   The note after the colon is optional.

An AI assistant's change to one or two sections of a validated page keeps the
page validated, and the first line under each changed section's heading is
`> **AI-generated section.** An AI assistant changed this section after the page was reviewed, and no maintainer has reviewed the change yet.`
A change to a third section makes the page AI-revised. A maintainer who reviews
a page removes its other labels and markers and closes it with the validated
label, and a maintainer's own edit keeps a page validated and updates the date.

A notebook's labels are in its markdown cells, the opening label on the first
line of the first markdown cell and the closing label on the last line of the
last. An API page's labels cover the docstrings it renders.
`tests/docs/test_review_labels.py` checks every page, and
`python scripts/docs/review_status.py` lists each page with its status, its
reviewer and date, and the sections or parts left to review.

### Prefect orchestration

ProbPipe ships with Prefect orchestration **off** by default — every
`Function` runs in-process unless the caller opts in. Two
ways to opt in:

```python
# Per-process (notebook, REPL, script):
import probpipe
probpipe.config.prefect.workflow_kind = probpipe.WorkflowKind.TASK
```

```bash
# Per-deployment (Docker, systemd, CI), read once at import:
export PROBPIPE_WORKFLOW_KIND=task   # or flow / off / default
```

Per-workflow overrides via
`@function(workflow_kind=probpipe.WorkflowKind.TASK)` and
explicit `Function(..., workflow_kind=probpipe.WorkflowKind.FLOW)`
are unaffected by either of the above. String aliases such as `"task"` /
`"flow"` are not accepted; use `WorkflowKind` enum members explicitly.

The off-by-default behaviour exists because the prior auto-detect path
("Prefect importable → tasks enabled") confused notebook and REPL
users who happened to have Prefect on `sys.path` as a transitive
dependency: every `sample(...)` then tried to reach
`http://127.0.0.1:4200/api/` and raised `httpx.ConnectError`. See
[#182](https://github.com/TARPS-group/prob-pipe/issues/182) for the
full rationale.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`):

- Tests on Python 3.12 and 3.14 for a pull request, and on 3.12, 3.13, and 3.14
  for a push to `main` or `dev/overhaul` and for the nightly run of `main`.
  Each version's selected tests run in four `test-shard` jobs, and a `test (<version>)`
  job reports the version's result (see *Test shards* below)
- Installs via `uv sync --frozen` from `uv.lock` (single source of truth
  for pinned dependency versions, shared between local dev and CI)
- Test job uses extras `dev,nutpie,pymc,pyabc`. The notebooks job is a two-leg
  matrix that runs in parallel — a `user_guide` leg (`dev,nutpie`) for
  `docs/user_guide` and a `tutorials` leg (`dev,nutpie,bayesflow,pymc,pyabc`) for
  `docs/tutorials` — each scoped to its own directory with independent
  change detection, so an unrelated leg is skipped (`bridgestan` is installed
  only in the `stan` leg, below)
- A separate `bayesflow` leg (Python 3.12 and 3.13 only — BayesFlow caps
  `<3.14`) syncs `dev,nutpie,bayesflow` and runs every test marked `bayesflow`.
  The other jobs skip a test that needs the extra, so such a test carries
  `@pytest.mark.bayesflow`
- A separate `stan` leg (Python 3.12) syncs `dev,nutpie,stan,pymc`, caches the
  `~/.bridgestan` build and a CmdStan build under `~/.cmdstan` (the version in the
  job's `CMDSTAN_VERSION`), and runs every test marked `stan` against a real
  BridgeStan backend; coverage uploads under a `stan` flag. `tests/conftest.py`
  marks a test that requests the `_stanc` or `_stan_toolchain` fixture, and a
  test that needs BridgeStan another way carries `@pytest.mark.stan`. Gated like
  the bayesflow leg — runs on pushes to main, foundational changes, or Stan-file
  changes
- A separate `test (lowest versions)` leg (Python 3.12) syncs `dev` from the
  lock, replaces each core dependency of `pyproject.toml` with the lowest version
  that its specifier admits, and runs `tests/functions`, `tests/converters`, and
  `tests/operations`. The other jobs install the locked versions, so a call to an
  API that is newer than a floor fails only in this leg. Fix such a failure by
  raising the floor in `pyproject.toml` or by calling an API that the floor
  provides. The leg runs on every push, on the nightly schedule, on dependency
  changes, and on changes under `probpipe/` or `tests/`
- Only the Python 3.12 shards measure coverage. The `coverage` job combines
  their data, enforces the 88% floor on a full-suite run, and uploads one report
  to Codecov
- The `lint & format` job runs the ruff gate and the docstring check of
  *Linting & pre-commit*
- Both the `test` and `notebooks` jobs choose what to run via a shared,
  unit-tested AST import-graph helper — `scripts/ci/import_graph.py` (tests
  in `tests/ci/`) — so a change to a source file also exercises the tests and
  notebooks that transitively import it. Edit that committed helper, not inline
  workflow scripts.
- The `test` job also selects the tests that read files rather than import the
  changed modules. A change to `design/`, to `scripts/design/`, or to the
  conformance helper `tests/_design_conformance.py` runs the five
  `test_design_conformance.py` files, `tests/docs/`, and `tests/ci/`, and a
  change to `probpipe/`, a rule document, an agent file, or the CHANGELOG runs
  `tests/docs/`.
- The `test` job runs the full suite on a push, on the nightly run, and for a
  pull request that changes a dependency file (`pyproject.toml`, `uv.lock`, or
  `setup.py`), `tests/conftest.py`, or `probpipe/__init__.py`. A change to
  `scripts/` or `.github/workflows/` runs `tests/ci/`.
- The `design ledger (report)` job runs `scripts/design/ledger.py` and shows its
  counts of stubs, pending tests, and stale docs in the job summary. It fails
  only when the script errors.
- The `PR hygiene (advisory)` job of `.github/workflows/pr-hygiene.yml` checks the
  PR title, branch, and body against this guide and the PR template through
  `scripts/ci/pr_hygiene.py`. It reports each finding as a warning and does not
  gate merges.

Docs build (`.github/workflows/docs.yml`) with `uv run mkdocs build --strict`.

### Test shards

`pytest-split` divides the selected tests into four groups of about equal
duration, using the per-test durations recorded in `.test_durations`. A test
missing from the file counts as the average duration, so new tests keep the
groups roughly balanced, and a group with no tests passes. Refresh the file
when the shard times in CI drift apart, such as after adding a slow test module:

```bash
uv run pytest -o addopts="" -n 4 --store-durations --clean-durations
```

`--clean-durations` drops the durations of removed tests. Run the command with
the extras of the `test` job installed, so that no test is skipped, and commit
the updated `.test_durations`. To run one shard locally, add
`--splits 4 --group <k> --splitting-algorithm least_duration` to the pytest command.

### Updating dependencies

`uv.lock` is committed and CI uses `--frozen`, so a dependency bump needs an
explicit lockfile update:

```bash
uv lock --upgrade-package <name>    # bump one package within pyproject.toml constraints
uv lock --upgrade                   # refresh the whole lock
```

Commit the resulting `uv.lock` change alongside the `pyproject.toml` change.

---

## Package Structure

```
probpipe/
├── core/           # Shared abstractions: specs, records, batches, identity, dispatch
├── values/         # The Function base, FunctionSpec, and pure argument binding
├── linalg/         # Linear operators
├── distributions/  # The Distribution base, capabilities, factored laws, batches, conversion
├── functions/      # The Function engine and workflow scopes
├── operations/     # The operations and their routes
├── families/       # The distribution catalog
├── record/         # Parameter-sweep designs
├── inference/      # Inference methods (BlackJAX, TFP, nutpie, CmdStan, PyMC, pyabc, BayesFlow)
├── validation/     # Predictive checks, calibration, and model comparison
├── diagnostics/    # Posterior diagnostics, ArviZ interop, and diagnostic views
├── testing/        # check_executor, the conformance check of a user's executor
├── custom_types.py # Array, PRNGKey, ArrayLike type aliases
└── _array_utils.py, _dtype.py, _weights.py  # Internal helpers
```

STYLE_GUIDE.md §2.3 states which modules are private and which are public, and
STYLE_GUIDE.md §6 states the allowed import directions between the packages.
`probpipe/__init__.py` lists the full public API.

### Distributions: `probpipe-core` and `probpipe`

The repository builds **two** distributions from one tree, both exposing the
same `probpipe` import package above:

- **`probpipe-core`** — the root `pyproject.toml`. The minimal distribution: the
  JAX base only, with every inference backend an optional extra. This is what
  `uv sync` / `pip install -e .` build, so it is the distribution you develop and
  test against.
- **`probpipe`** — a code-less metapackage in
  `packaging/probpipe/pyproject.toml`. It pins `probpipe-core==<version>` and
  adds the backends the docs exercise (`pymc`, `nutpie`, and marker-guarded
  `bayesflow`), so `pip install probpipe` runs every example and tutorial. It
  ships no modules of its own.

The two versions move in lockstep: when bumping `version`, update **both**
`pyproject.toml` files and keep the metapackage's `probpipe-core==` pin equal to
the core version, which `tests/test_version.py` checks. Build each with:

```bash
uv build                      # probpipe-core (repository root)
uv build packaging/probpipe   # probpipe (metapackage)
```

---

## Releasing

The workflow `.github/workflows/release.yml` publishes both distributions to
PyPI when a tag `vX.Y.Z` is pushed. It runs these steps:

1. It builds the sdist and the wheel of each distribution.
2. It checks that the tag matches both versions.
3. It runs `twine check` on the four files.
4. It installs both wheels into a fresh environment and imports `probpipe`.
5. It uploads `probpipe-core` and then `probpipe` through PyPI Trusted Publishing.

### One-time setup

A trusted publisher on PyPI has these fields:

| Field | Value |
|---|---|
| Owner | `TARPS-group` |
| Repository name | `prob-pipe` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

1. On PyPI, add this trusted publisher to each of the projects `probpipe-core`
   and `probpipe`. Before a project's first upload, add it as a pending publisher
   under **Publishing** in the account settings.
2. In the repository's settings on GitHub, create the environment `pypi` under
   **Environments**. A required reviewer on it makes each upload wait for approval.
3. To test the workflow on TestPyPI, repeat both steps with TestPyPI and the
   environment `testpypi`, and then run the workflow by hand with its `testpypi`
   input checked.

### Each release

1. On a branch, set `version` in both `pyproject.toml` files and the
   metapackage's `probpipe-core==` pins to the new version, as
   *Distributions: `probpipe-core` and `probpipe`* describes.
2. In `CHANGELOG.md`, move the entries under `## [Unreleased]` to a heading
   `## [X.Y.Z] - YYYY-MM-DD` with the release date.
3. In `CITATION.cff`, set `version` and `date-released`.
4. Open a PR into `main` and merge it once CI passes.
5. Tag the merge commit and push the tag, which starts the release workflow:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. Approve the `pypi` deployment if the environment requires a reviewer.
7. Check the release in a fresh environment:

   ```bash
   python -m venv /tmp/check && /tmp/check/bin/pip install probpipe==X.Y.Z
   /tmp/check/bin/python -c "import probpipe; print(probpipe.__version__)"
   ```

---

## Architecture

The design reference in [`design/`](design/README.md) describes the
architecture in its target state:

- [Part I](design/01-design-principles.md): the design principles;
- Parts II to VII: the shared abstractions, the term kinds, the distributions,
  the `Function` engine, the operations, and the distribution catalog;
- [`design/package-structure.md`](design/package-structure.md): the package
  layout, whose § Correspondence to the implementation maps each module to its
  target.

[Extending ProbPipe](docs/api/extending.md) documents the extension points
and their registries.

---

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed coding conventions.
