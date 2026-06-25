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
  stale description is a review blocker. No internal process jargon
  ("Phase 1b", plan-file references, review-round narration): an outside
  reader must be able to follow the description on its own. User-visible
  changes get a CHANGELOG entry, and scratch planning artifacts
  (`*_plan.md` files, references to local plan directories) never land on
  the branch.
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

### Running Tests

```bash
uv run pytest                              # parallel via xdist
uv run pytest -p no:xdist -o "addopts="    # disable parallel for debugging
uv run pytest tests/test_foo.py -x -v      # single file, stop on first failure
```

`uv run` executes inside the synced `.venv` without manual activation; you
can also `source .venv/bin/activate` once per shell and then just type
`pytest`.

### Test quality

- Correctness tests use the **tightest tolerances that pass reliably** —
  a loose tolerance is a review flag, not a convenience.
- Cover structured cases (multi-field Records, mixed scalar/vector
  parameters), error paths, and **equivalence across dispatch paths**
  (jax vs sequential) — path divergence has produced real
  silently-wrong-results bugs.
- Inference code gets a statistical sanity check (parameter estimates and
  uncertainty roughly correct on a known target), not just shape
  assertions.

### Coverage

```bash
uv run pytest --cov=probpipe --cov-report=term-missing
```

Target: >90% on all modules.

### Test quality for numerical code

Coverage is necessary but not sufficient: tests of mathematical behavior
must check correctness against an **independent baseline** (an analytic
result, an exact reference computation, a known invariant, or finite
differences for gradients) — and where the claim is distributional,
check both location and spread. Tolerances on stochastic or trained
components are **measured, not guessed**: run the test's configuration
across a few seeds, bound the observed spread with modest margin, and
document the measured range in a comment next to the assertion. Full
conventions in [STYLE_GUIDE.md § 8.6](STYLE_GUIDE.md#86-numerical-correctness-and-tolerances).

### Code formatting

Formatting is owned by **`ruff format`** (Black-style) — don't hand-format
Python. The `ruff-format` pre-commit hook reformats on commit; run it directly
with:

```bash
uv run ruff format .          # reformat the tree
uv run ruff format --check .  # verify (this is what CI enforces)
```

Both `ruff format --check` and `ruff check` (the linter) are **blocking** CI
steps, so a misformatted file or a lint violation fails the build. A few
specifics: the line
length is 100 (`[tool.ruff]` in `pyproject.toml`); ruff keeps code on one line
when it fits and explodes imports / call arguments one-item-per-line when it does
not; string quotes normalize to double. Notebooks are excluded
(`[tool.ruff.format] exclude` in `pyproject.toml`), so the docs' tutorial cells
keep their compact, hand-curated layout.

### Code comments & docstrings

Comments state constraints and contracts the code cannot express — not
the development process. Match the comment density of the surrounding
code, and when in doubt, delete: an over-explained obvious line is worse
than no comment.

- **No process narration.** Never record provenance in code — which PR
  or plan phase introduced a line, which review comment prompted it
  ("addressed in review", "previously this was..."). Such comments are
  noise the moment the PR merges. (Citing an issue that documents a
  known limitation or a non-obvious rationale is fine — that is a
  constraint, not history.)
- **Describe what something *is*, not what it *isn't*.** Negative
  documentation ("this is not a mixture") usually signals that the name
  or design needs fixing instead.
- **Public docstrings describe behavior and usage, not implementation
  internals.** Internals discussion belongs on private helpers, or
  nowhere.

### Linting & pre-commit

Linting uses [ruff](https://docs.astral.sh/ruff/) (configured in
`pyproject.toml`). Install the pre-commit hooks once:

```bash
uvx pre-commit install      # or: pre-commit install
```

Thereafter `ruff` (lint + format) plus a few file-hygiene hooks run on your staged
files at commit time. The hooks see only the files you're changing, so a commit is
checked without re-linting the whole tree. To run manually:

```bash
uv run ruff check .              # lint the whole tree (uses the uv.lock-pinned ruff)
uvx pre-commit run --all-files   # run every hook over everything
```

`ruff check .` and `ruff format --check .` are clean tree-wide. A full
`--all-files` run may still surface pre-existing file-hygiene nits (trailing
whitespace, end-of-file) in files you did not touch; the fixer hooks clean those
as the relevant files are next edited.

Both `ruff check` (lint) and `ruff format` are enforced:

- **`ruff check` (lint) is blocking in CI.** The `lint & format` job runs
  `ruff check .` over the whole tree, which is at zero violations; a new
  violation fails the build. Rule selection and per-file ignores live in
  `[tool.ruff.lint]` in `pyproject.toml`. The pre-commit hook flags the same
  issues on your staged files at commit time.
- **`ruff format` is enforced.** Formatting is owned by `ruff format` (see
  *Code formatting* above): `ruff format --check` is a blocking CI step and the
  `ruff-format` pre-commit hook reformats on commit.

### Type checking

Type checking uses [pyright](https://microsoft.github.io/pyright/)
(configured in `pyrightconfig.json`, scoped to the `probpipe` package).
Run it locally in the synced environment:

```bash
uv run --with 'pyright[nodejs]' pyright
```

CI pins a specific pyright version for a reproducible baseline, so a
local run on a newer pyright may report a slightly different count — pin
to match CI (`pyright[nodejs]==<version from ci.yml>`) if you need exact
parity.

Like ruff, **pyright is advisory in CI for now** — the `typecheck
(advisory)` job reports type issues (and surfaces the count in the run's
job summary) but does not gate merges. The source carries a type-debt
baseline (much of it noise from JAX/TFP untyped attributes), so enforcing
immediately would block unrelated work. The plan is to burn the baseline
down, then tighten `typeCheckingMode` in `pyrightconfig.json` and make the
gate blocking. New code should be clean under the current `basic` mode
where practical.

ProbPipe ships a `py.typed` marker, so the package's annotations are
consumed by downstream users' type checkers — keeping the public surface
well-typed is user-facing quality, not just an internal nicety.

### Documentation

```bash
uv run mkdocs build --strict   # build docs, fail on warnings
uv run mkdocs serve            # local preview
```

API docs use `mkdocstrings` directives in `docs/api/*.md` referencing
fully-qualified Python paths.

A behavior or API change and its documentation ship in the **same PR**:
docstrings, the user-guide notebooks, README / `docs/index.md`, the
CHANGELOG, and STYLE_GUIDE.md / CONTRIBUTING.md when conventions change.
Examples and notebooks show idiomatic usage — never add a compat shim to
keep an example running against an old API.

### Prefect orchestration

ProbPipe ships with Prefect orchestration **off** by default — every
`WorkflowFunction` runs in-process unless the caller opts in. Two
ways to opt in:

```python
# Per-process (notebook, REPL, script):
import probpipe
probpipe.prefect_config.workflow_kind = probpipe.WorkflowKind.TASK
```

```bash
# Per-deployment (Docker, systemd, CI), read once at import:
export PROBPIPE_WORKFLOW_KIND=task   # or flow / off / default
```

Per-workflow overrides via
`@workflow_function(workflow_kind=probpipe.WorkflowKind.TASK)` and
explicit `WorkflowFunction(..., workflow_kind=probpipe.WorkflowKind.FLOW)`
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

- Tests on Python 3.12, 3.13, and 3.14
- Installs via `uv sync --frozen` from `uv.lock` (single source of truth
  for pinned dependency versions, shared between local dev and CI)
- Test job uses extras `dev,nutpie,pymc`. The notebooks job is a two-leg
  matrix that runs in parallel — a `user_guide` leg (`dev,nutpie`) for
  `docs/user_guide` and a `tutorials` leg (`dev,nutpie,bayesflow,pymc`) for
  `docs/tutorials` — each scoped to its own directory with independent
  change detection, so an unrelated leg is skipped (`bridgestan` is installed
  only in the `stan` leg, below)
- A separate `bayesflow` leg (Python 3.12 and 3.13 only — BayesFlow caps
  `<3.14`) syncs `dev,nutpie,bayesflow` and runs the amortized-SBI tests
- A separate `stan` leg (Python 3.12) syncs `dev,nutpie,stan`, caches the
  `~/.bridgestan` build, and runs StanModel's compile-backed tests against a
  real BridgeStan backend; coverage uploads under a `stan` flag. Gated like the
  bayesflow leg — runs on pushes to main, foundational changes, or Stan-file
  changes
- Coverage uploaded to Codecov
- The `lint & format` job runs `ruff check` (lint) and `ruff format --check`;
  both are **blocking** (see *Linting & pre-commit*)
- Both the `test` and `notebooks` jobs choose what to run via a shared,
  unit-tested AST import-graph helper — `scripts/ci/import_graph.py` (tests
  in `tests/ci/`) — so a change to a source file also exercises the tests and
  notebooks that transitively import it. Edit that committed helper, not inline
  workflow scripts.

Docs build (`.github/workflows/docs.yml`) with `uv run mkdocs build --strict`.

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
├── core/           # Base abstractions: Distribution, protocols, ops, node, transition
├── distributions/  # Concrete distributions (continuous, discrete, multivariate, ...)
├── record/         # Record-adjacent constructions: parameter-sweep Designs
├── modeling/       # Model wrappers (SimpleModel, StanModel, PyMCModel, likelihoods)
├── inference/      # Inference methods + registry (BlackJAX, TFP, nutpie, RWMH)
├── converters/     # Distribution conversion registry
├── linalg/         # Linear algebra for random functions
├── custom_types.py # Array, PRNGKey, ArrayLike type aliases
└── _utils.py, _array_utils.py, _weights.py  # Internal helpers
```

Within subpackages that contain multiple implementation files
(`modeling/`, `inference/`, `converters/`), implementation modules use
a leading underscore (`_simple.py`, `_blackjax_rwmh.py`).  The package
`__init__.py` re-exports the public API so users import from
`probpipe` or from subpackage `__init__` modules, never from
underscore modules directly.  See `probpipe/__init__.py` for the
full public API surface.

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
the core version. Build each with:

```bash
uv build                      # probpipe-core (repository root)
uv build packaging/probpipe   # probpipe (metapackage)
```

---

## Architecture Overview

### Design principles

1. **Distributions are immutable** — parameters fixed at construction;
   operations return new distributions. The one documented exception
   is `Distribution._auxiliary` (a `DataTree` of post-construction
   metadata): validators and diagnostic ops (e.g.,
   `predictive_check`) attach their results in-place under named
   groups (`auxiliary["predictive_check"]`, future `auxiliary["loo"]`,
   ...). This is a deliberate carve-out — the alternative of returning
   a renamed clone for every diagnostic would break source/identity
   tracking that downstream code relies on. Treat `_auxiliary` as
   append-only; never mutate other state post-construction.
2. **Operations are standalone workflow functions** — `sample()`, `mean()`,
   `log_prob()`, `condition_on()` are `WorkflowFunction` instances in
   `probpipe/core/ops.py`.
3. **Capabilities via protocols** — distributions declare support through
   `@runtime_checkable` protocols (e.g., `SupportsSampling`,
   `SupportsLogProb`, `SupportsMean`). Operations check protocols at
   dispatch time.  Protocols are dynamically included on composite
   distributions (`ProductDistribution`, `TransformedDistribution`)
   based on component capabilities.

   Most protocols are *instance-level*: the contract is "this
   `Distribution` instance can do X" and the runtime check is
   `isinstance(my_dist, SupportsX)`. A small number are *class-level*:
   `SupportsArrayBackend` declares `_make_array_backend` as a
   `@classmethod`, so the contract is "this class can produce a
   batched form" and the runtime check is on the class itself
   (`isinstance(MyDistribution, SupportsArrayBackend)`). New
   protocols default to instance-level; reach for class-level only
   when the capability is genuinely a class concern (e.g., a fused-
   storage factory that doesn't depend on per-instance state).
4. **Private method convention** — protocols define `_method()` (e.g.,
   `_sample`, `_log_prob`, `_mean`). The public API is via ops:
   `sample(dist)`, not `dist.sample()`.
5. **Record and Distributions are parallel** — `Record` is the universal
   container for non-random structured data; `Distribution` is the
   universal container for random quantities. Both support named fields,
   `select()` for workflow function splatting, and JAX pytree
   traversal.  The full pipeline (prior → inference → posterior
   predictive) produces named, provenance-tracked objects at every step.
   **Field access is bracket-only**: use `record["x"]`, `array["x"]`,
   `dist["x"]`.  Attribute access (`__getattr__`) was removed from
   `Record` and `RecordDistribution` because it shadowed methods and
   properties like `.mean`, `.var`, `.fields`, `.map`, and produced
   confusing errors.
6. **Every distribution is named** — `Distribution.__init__` requires a
   non-empty `name: str`.  Leaf distributions (Normal, Gamma, etc.)
   require an explicit `name=` at construction.  Composite distributions
   (ProductDistribution, EmpiricalDistribution, TransformedDistribution,
   etc.) auto-generate a name from their components when one is not
   provided.  `ProductDistribution` validates that each component
   distribution's `name` matches its keyword key (e.g.,
   `ProductDistribution(x=Normal(0, 1, name="x"))`).
7. **Uniform output wrap at the WorkflowFunction boundary** — every
   `@workflow_function` return is coerced into the
   `Record | RecordArray | Distribution` contract before it reaches the
   caller.  Scalars and `jnp.ndarray`s become
   `NumericRecord({fn_name: value})` (no sweep) or
   `NumericRecordArray({fn_name: arr}, batch_shape=sweep_shape)`
   (swept); `dict` / `list` / `tuple` promote via `_make_stack`;
   existing `Record` / `RecordArray` / `Distribution` values pass
   through unchanged.  The field name is always the function's own
   name.  Single-field `NumericRecord` / `NumericRecordArray` / `Record`
   expose shims (`__jax_array__`, `__float__`, `__call__`, `.shape`,
   `.dtype`, `.ndim`) so `jnp.array(log_prob(d, v))`,
   `float(mean(d))`, and `sample(grf)(X)` stay terse.
8. **Array inputs vectorize with the product rule** — when a
   `@workflow_function` is called with array-valued inputs
   (`RecordArray` or `DistributionArray` with nonempty
   `batch_shape`) passed to slots whose hints don't match the
   batched type, the WorkflowFunction layer dispatches cell-by-cell
   and stacks the returns.  Multiple array inputs combine by the
   **product rule** (Cartesian full factorial); the sweep's
   `batch_shape` is the concatenation of each array arg's
   `batch_shape`.  Scalar `Distribution` inputs marginalise via
   Monte Carlo, unchanged.  A `DistributionArray` is always treated
   as `Array[Distribution]` (never marginalised in-place), so
   `sample(da)` / `mean(da)` / `log_prob(da, v)` are handled
   uniformly by the sweep path rather than by DistArray-specific
   methods.

### Key abstractions

| Abstraction | Description |
|-------------|-------------|
| `Distribution[T]` | Generic base parameterized by value type; provides `event_template` and `auxiliary` properties |
| `Record` | Named, immutable, JAX-pytree container for structured non-random values; leaves stored verbatim (no coercion); `select()` for workflow function splatting |
| `NumericRecord` (subclass of `Record`) | Post-construction invariant: every leaf is a `jax.Array` (constructor coerces via `jnp.asarray`). Adds `to_vector` / `vector_size` (the numeric 1-D serialization; the inverse `from_vector` lives on `NumericEventTemplate`). Captures backend metadata (xarray dims/coords, pandas index) via the aux registry; `to_native()` reverses the conversion to a permissive `Record`. `Record.to_numeric()` is the symmetric forward path. |
| `RecordArray` | Batch of `Record` elements with a `EventTemplate`; integer index → element, field index → batched array |
| `NumericRecordArray` (subclass of `RecordArray`) | Batch of `NumericRecord` elements; adds `to_vector` / `mean` / `var` |
| `EventTemplate` | Structural skeleton (field names, per-field shapes or `None`); `NumericEventTemplate.from_vector` rebuilds a numeric value from its 1-D vector without an example instance |
| `RecordDistribution` | Record-based distribution base; `fields`, `__getitem__` → `_RecordDistributionView`, `select()` / `select_all()` for correlated broadcasting. A `Distribution` represents one random variable; use `DistributionArray` for collections. |
| `_RecordDistributionView` | Lightweight component reference; dynamic protocol support matching parent capabilities |
| `NumericRecordDistribution` | Numeric-array distribution base; per-field `dtypes`, `supports`, `event_shapes`; base for all TFP-backed distributions |
| `FlatNumericRecordDistribution` | Refinement of `NumericRecordDistribution` enforcing the flat contract: single field, `event_shape == (N,)`. Carries `flat_size` and `as_record_distribution(template=…)` — the inverse of `as_flat_distribution()`, lifting a flat distribution to a Record-keyed view under a user-supplied `NumericEventTemplate`. Algorithms that consume a flat parameter vector (MCMC, optimisers, VI / Pathfinder / Laplace surrogates) should declare their input as this type. Natively-multivariate parametrics (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `VonMisesFisher`) and `FlattenedDistributionView` all implement it. |
| `FlattenedDistributionView` | A `FlatNumericRecordDistribution` produced by `nrd.as_flat_distribution()`. Wraps any base distribution and exposes flat-vector samples / log-probs (`event_shape == (event_size,)`), delegating through the base. |
| `NumericRecordDistributionView` | The inverse view, produced by `FlatNumericRecordDistribution.as_record_distribution(template=…)`. Lifts a flat distribution to a Record-keyed structure; samples come back as `NumericRecord` / `NumericRecordArray` keyed by `template.fields`. |
| `DistributionArray` | Shape-indexed `Array[Distribution]`; exposes only the container surface (indexing, iteration, `batch_shape`, `event_shape`, `components`). Vectorized ops are delivered by the `WorkflowFunction` sweep layer — passing a `DistributionArray` to an op whose hint is a scalar `Distribution` / protocol triggers cell-by-cell dispatch, and outputs stack into `NumericRecordArray` / `RecordArray` / (nested) `DistributionArray`. Produced by parameter-sweep workflow functions whose inner call returns a `Distribution`. |
| `JointEmpirical` / `NumericJointEmpirical` | Weighted joint samples distribution. Generic base supports only sampling + conditioning; the numeric subclass adds exact `SupportsMean` / `SupportsVariance`. `JointEmpirical(...)` dispatches to `NumericJointEmpirical` when every field is numeric. (Empirical distributions do not claim `SupportsLogProb`; use `from_distribution(emp, KDEDistribution, …)` for a density.) |
| `EmpiricalDistribution[T]` / `RecordEmpiricalDistribution` | Weighted empirical distribution. Generic base over arbitrary sample type ``T``; Record-based specialisation adds `event_shapes`, exact moments (`SupportsMean` / `SupportsVariance` / `SupportsCovariance`), and TFP-style shape semantics. Numeric-array sources auto-wrap as a single-field Record (requires `name=`). Two views on the stored draws: `samples` (structured `NumericRecord`, per-field access via `samples[name]`) and `flat_samples` (flat `(n, dim)` matrix across all fields, in insertion order). Use `flat_samples` for stacked-matrix idioms like `post.flat_samples.mean(axis=0)` for per-parameter posterior summaries. |
| `BootstrapReplicateDistribution[T]` / `RecordBootstrapReplicateDistribution` | N-fold product over a source: each draw is a bootstrapped dataset of `n` i.i.d. observations. Accepts a `Record`, `RecordEmpiricalDistribution`, numeric array, or any `SupportsSampling` source (in which case `n` is mandatory). |
| `WorkflowFunction` | Orchestration-aware function wrapper (Prefect off by default; see the Prefect-orchestration section above); groups views by parent for correlated broadcasting |
| `Module` | Stateful workflow-aware base class (see `@workflow_method`) |
| Protocols | `SupportsSampling`, `SupportsLogProb`, `SupportsMean`, `SupportsConditioning`, etc.; dynamic inclusion on `ProductDistribution` and `TransformedDistribution` |
| `BaseDispatchRegistry` | Abstract base for priority-based registries: holds registration, priority management (incl. opt-in-only sentinel + override warnings), and the `check`/`execute` loop. Arity-specific subclasses override `_cache_key`, `_find_methods`, and `_format_key`. |
| `UnaryDispatchRegistry` | Single-argument dispatch registry; dispatches on the type of the first positional argument. Used by the inference method registry. |
| `BinaryDispatchRegistry` | Two-argument dispatch registry; dispatches on the joint type of the first two positional args via paired `((left_types,), (right_types,))` pre-filters. |
| `ProbabilisticModel` | Base for models (extends `Distribution`; provides `fields`) |
| `SimpleGenerativeModel` | Simulator-only model wrapper for SBI/ABC (prior + `GenerativeLikelihood`) |
| `IncrementalConditioner` | Stateful `Module` for sequential Bayesian updating via `update()` / `update_all()` |
| `iterate` / combinators | Iterative distribution transformation; `with_conversion`, `with_resampling` |
| `Design` / `FullFactorialDesign` (`probpipe.record`) | `RecordArray` subclass carrying per-field marginals; `FullFactorialDesign(**marginals)` materialises the Cartesian product as a sweep-ready `RecordArray`. Pipe into a `WorkflowFunction` as a single `Record`-typed arg to trigger the WF sweep path. |

### Inference method registry

`condition_on` dispatches inference via a pluggable **inference method
registry** (`inference_method_registry`).  Each method declares
`supported_types`, a `priority`, and `check()`/`execute()` methods.
The registry tries methods in descending priority order; the first
whose `check()` returns `feasible=True` wins.

Models no longer implement `_condition_on` directly — conditioning is
handled entirely by registered methods.  The removed protocol
`SupportsConditionableComponents` is no longer part of the public API;
use `fields` and the inference registry instead.

Priorities follow a semantic convention (issue #189): values above
``50`` mark *exact* methods, values in ``(0, 50]`` mark *inexact*
methods, and ``0`` is the opt-in-only sentinel (selectable by name but
skipped during auto-dispatch). The contributor-facing tier criteria
for picking a number when registering a new method live under
[Extending ProbPipe → Setting priority for a new method](docs/api/extending.md#setting-priority-for-a-new-method).

Built-in methods:

| Priority | Name | Backend | Applies to |
|----------|------|---------|------------|
| 88 | `nutpie_nuts` | nutpie | `StanModel`, `PyMCModel` |
| 85 | `blackjax_nuts` | BlackJAX | Any `SupportsLogProb` (JAX-traceable) |
| 82 | `cmdstan_nuts` | CmdStanPy | `StanModel` |
| 82 | `pymc_nuts` | PyMC | `PyMCModel` |
| 75 | `blackjax_elliptical_slice` | BlackJAX | `SimpleModel` + Gaussian prior + JAX-traceable likelihood |
| 55 | `blackjax_rwmh` | BlackJAX | Any `SupportsLogProb` (eager fallback for non-traceable targets) |
| 45 | `blackjax_sgld` | BlackJAX | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` |
| 0 | `blackjax_hmc` | BlackJAX | Any `SupportsLogProb` (JAX-traceable); opt-in only via `method=` |
| 0 | `blackjax_sghmc` | BlackJAX | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=`; opt-in only via `method=` |
| 0 | `pymc_advi` | PyMC | `PyMCModel`; opt-in only via `method=` |
| 0 | `tfp_nuts` | TFP | Any `SupportsLogProb` (JAX-traceable); opt-in only via `method=` |
| 0 | `tfp_hmc` | TFP | Any `SupportsLogProb` (JAX-traceable); opt-in only via `method=` |

**Amortized SBI dispatches two ways.** Trained amortized posterior estimators
(`learn_amortized_posterior` → `BayesFlowModel`, the `[bayesflow]` extra)
implement `SupportsConditioning` directly, so `condition_on(model, observed)` is
a single forward pass through the trained network. Because `condition_on` checks
`SupportsConditioning` *before* the inference-method registry, these estimators
short-circuit it and register no method. The learned NLE/NRE likelihoods
(`learn_amortized_likelihood` / `learn_amortized_ratio` → `BayesFlowLikelihood`
/ `BayesFlowRatio`) take the opposite route: they are ordinary
`ConditionallyIndependentLikelihood` components, so
`SimpleModel(prior, learned)` + `condition_on` selects a sampler through the
registry table above (typically `blackjax_nuts` — the learned scores are
JAX-traceable) with no new registry entries.

### Converter priority system

The `ConverterRegistry` dispatches conversions by trying registered
`Converter` subclasses in descending **priority** order. The first
converter whose `check()` returns `feasible=True` wins. Built-in
priorities:

| Priority | Converter | Role |
|----------|-----------|------|
| 200 | `ProtocolConverter` | Intercepts protocol targets (e.g., `SupportsLogProb`), resolves to a concrete type, and delegates back to the registry |
| 100 | `ProbPipeConverter` | ProbPipe-to-ProbPipe conversions (same-class passthrough or cross-family moment-matching) |
| 50 | `TFPConverter` | Bidirectional TFP ↔ ProbPipe conversions |
| 25 | `ScipyConverter` | Bidirectional scipy.stats ↔ ProbPipe conversions (optional) |

When adding a new converter, choose a priority that reflects its
specificity – higher priority means it is tried first. Protocol-level
converters should be above concrete-type converters.

### Aux registry

The `aux_registry` in `probpipe.core._array_backend` is a flat
`dict[type, AuxHooks]` mapping a leaf type to a `(capture, restore)`
pair. It is used only for round-tripping backend-specific metadata
across the `Record` ↔ `NumericRecord` boundary — `jnp.asarray` drops
xarray dims/coords/attrs and pandas index/columns/dtypes, and the
aux registry restores them.

```python
from probpipe import register_aux

register_aux(
    MyArrayLike,
    capture=lambda leaf: {"label": leaf.label},
    restore=lambda arr, aux: MyArrayLike(arr, label=aux["label"]),
)
```

Built-in registrations cover `xarray.DataArray`, `pandas.Series`,
and `pandas.DataFrame`. Lookup walks the MRO of `type(obj)`, so
registering a base class also covers its subclasses.

This registry is **not** a behavioural-dispatch hierarchy — it has
no priority system, no feasibility check, no `execute()`. Use a
`BaseDispatchRegistry` subclass (`UnaryDispatchRegistry` for the
inference registry, `BinaryDispatchRegistry` for two-argument dispatch)
or `ConverterRegistry` (distribution conversion) when behaviour
dispatch is needed.

### Constraint → Bijector registry

The `_CONSTRAINT_BIJECTOR_REGISTRY` in
`probpipe.distributions._bijector_dispatch` is a flat
`dict[type | Constraint, BijectorFactory]` mapping a `Constraint`
subclass *or* a specific `Constraint` instance to a factory that
returns a TFP bijector mapping ℝⁿ to that constraint's support.
Lookup precedence is:

1. **Exact instance match** — e.g., a registration on the singleton
   `positive` overrides the type-level `_Positive` default.
2. **Type match via MRO** — most-specific subclass first.
3. `NotImplementedError` if neither matches.

Use `register_bijector(key, factory)` to add or override a default;
re-registering the same key silently overwrites. The registry
mirrors PyTorch's `constraint_registry` semantics.

Like the aux registry, this is **not** a behavioural-dispatch
hierarchy: there is no priority system or feasibility check. Reach
for a `BaseDispatchRegistry` subclass (`UnaryDispatchRegistry` /
`BinaryDispatchRegistry`) or `ConverterRegistry` instead when
dispatch needs to consider the input value, environment, or
installed backends.

### Generic vs Record-based pattern

Some distribution families have a generic base parameterised over the
sample type ``T`` and a Record-based specialisation:

- `EmpiricalDistribution[T]` / `RecordEmpiricalDistribution`
- `BootstrapReplicateDistribution[T]` / `RecordBootstrapReplicateDistribution`

The generic base carries only type-agnostic features (sampling,
expectation). The Record-based variant adds `event_shapes`, `dim`,
`dtypes`, `support`, and moment protocols (`SupportsMean`,
`SupportsVariance`, `SupportsCovariance`).

**Automatic factory dispatch.** Constructing the generic base with
a numeric array or a `Record` automatically returns the Record-based
subclass:

```python
EmpiricalDistribution(jnp.ones((100, 3)), name="theta")
# → returns RecordEmpiricalDistribution; auto-wraps the array as
#   Record(theta=arr)

EmpiricalDistribution(Record(x=jnp.zeros((50,)), y=jnp.zeros((50,))))
# → returns RecordEmpiricalDistribution; multi-field record
```

`__new__` on the generic base implements the dispatch. Non-array,
non-Record inputs (lists of objects, opaque sequences) stay in the
generic base. The numeric-array path requires `name=` so the
auto-wrapped Record has a meaningful field key.

`BootstrapReplicateDistribution[T]` additionally accepts a
`SupportsSampling` source (e.g. `Normal(0, 1, name="x")`); each
replicate is `n` i.i.d. draws from `source._sample`. `n` is
mandatory in this case (no canonical observation count).

### Framework abstraction hierarchy

Three rules govern how the framework's universal types relate.

1. **One random variable per `Distribution`.** A single `Distribution`
   instance represents one random quantity. To carry a *collection*
   of distributions (a parameter sweep, a per-component posterior,
   ...), wrap them in a `DistributionArray`. The rule is enforced
   structurally: `Distribution` has no `batch_shape` accessor, and
   TFP-backed constructors raise `ValueError` if their parameters
   imply a non-empty `tfd.Distribution.batch_shape`. Use
   `DistributionArray.from_batched_params` (or the per-class
   `Normal.from_batched_params(...)` alias) for batched
   constructions.

2. **Two implementations per concept.** Each abstraction has at most
   two concrete pairs:
   - a generic implementation parameterised over `T`
     (`EmpiricalDistribution[T]`,
     `BootstrapReplicateDistribution[T]`, `Distribution[T]`),
   - and a Record-based specialisation
     (`RecordEmpiricalDistribution`,
     `RecordBootstrapReplicateDistribution`, `RecordDistribution`).

   No third "numeric-array" variant. Records *are* array-based — a
   single numeric array becomes a single-field Record at the
   constructor boundary.

   `NumericRecordDistribution` additionally has the
   `FlatNumericRecordDistribution` *refinement* — not a third
   implementation, just a tighter-typed subset (single field,
   `event_shape == (N,)`). Natively-multivariate parametrics
   (`MultivariateNormal`, `Dirichlet`, `Multinomial`,
   `VonMisesFisher`) and `FlattenedDistributionView` implement the
   refinement; scalar parametrics route through `as_flat_distribution()`
   first. Algorithms that consume a flat parameter vector should
   declare their input as `FlatNumericRecordDistribution` so
   receiver typing — not a runtime shape probe — enforces the
   contract.

3. **Iteration is a Record-family convention.** `Record`,
   `NumericRecord`, `RecordArray`, `NumericRecordArray` iterate field
   names dict-style. `DistributionArray` is positional (``len(da)``
   is the leading-axis size, ``prod(da.batch_shape)`` is the total
   cell count; access via ``da[i]``). Every other `Distribution`
   subclass — including `EmpiricalDistribution`,
   `BootstrapReplicateDistribution`, marginals — is non-iterable;
   finite-sample subclasses (see STYLE_GUIDE §1.9) expose stored
   samples on `.samples` / `.draws()` with `.n` reporting the count;
   parametric distributions do not have `.n`.

---

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed coding conventions.
