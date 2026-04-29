# Contributing to ProbPipe

## PR Workflow

### 1. Plan first (for significant changes)

If the change is significant, start by creating a PR that contains only a
plan document at the repo root named `<PR-description>_plan.md`. The plan
should describe the motivation, proposed design, and any trade-offs.

### 2. Request a review

Request a review from **@jhuggins** and/or **@arob5**. You will receive
feedback on the PR or the plan document.

### 3. Wait for approval before implementing

If you submitted a plan, do **not** start implementing until the plan is
approved. Once approved, open a follow-up PR (or update the existing one)
with the implementation and request another review.

### What counts as "significant"?

- New subpackages or modules
- Changes to protocols or the base class hierarchy
- New external dependencies
- Architectural changes

For small fixes (typos, bug fixes, test additions), skip the plan step and
go straight to an implementation PR.

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

---

## Development Setup

### Installation

```bash
pip install -e ".[dev]"          # core + test deps
pip install -e ".[dev,nutpie]"   # + nutpie for MCMC
```

Optional backends (not required for tests): `bridgestan`, `pymc`.

### Running Tests

```bash
pytest                           # parallel via xdist (configured in pyproject.toml)
pytest -p no:xdist -o "addopts=" # disable parallel for debugging
pytest tests/test_foo.py -x -v   # single file, stop on first failure
```

### Coverage

```bash
pytest --cov=probpipe --cov-report=term-missing
```

Target: >90% on all modules.

### Documentation

```bash
mkdocs build --strict   # build docs, fail on warnings
mkdocs serve            # local preview
```

API docs use `mkdocstrings` directives in `docs/api/*.md` referencing
fully-qualified Python paths.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`):

- Tests on Python 3.12 and 3.13
- Installs `.[dev,nutpie,sbi]` (bridgestan and pymc are not included)
- Coverage uploaded to Codecov

Docs build (`.github/workflows/docs.yml`) with `mkdocs build --strict`.

---

## Package Structure

```
probpipe/
├── core/           # Base abstractions: Distribution, protocols, ops, node, transition
├── distributions/  # Concrete distributions (continuous, discrete, multivariate, ...)
├── record/         # Record-adjacent constructions: parameter-sweep Designs
├── modeling/       # Model wrappers (SimpleModel, StanModel, PyMCModel, likelihoods)
├── inference/      # Inference methods + registry (TFP, nutpie, RWMH, sbijax)
├── converters/     # Distribution conversion registry
├── linalg/         # Linear algebra for random functions
├── custom_types.py # Array, PRNGKey, ArrayLike type aliases
└── _utils.py, _array_utils.py, _weights.py  # Internal helpers
```

Within subpackages that contain multiple implementation files
(`modeling/`, `inference/`, `converters/`), implementation modules use
a leading underscore (`_simple.py`, `_rwmh.py`).  The package
`__init__.py` re-exports the public API so users import from
`probpipe` or from subpackage `__init__` modules, never from
underscore modules directly.  See `probpipe/__init__.py` for the
full public API surface.

---

## Architecture Overview

### Design principles

1. **Distributions are immutable** — parameters fixed at construction;
   operations return new distributions.
2. **Operations are standalone workflow functions** — `sample()`, `mean()`,
   `log_prob()`, `condition_on()` are `WorkflowFunction` instances in
   `probpipe/core/ops.py`.
3. **Capabilities via protocols** — distributions declare support through
   `@runtime_checkable` protocols (e.g., `SupportsSampling`,
   `SupportsLogProb`, `SupportsMean`). Operations check protocols at
   dispatch time.  Protocols are dynamically included on composite
   distributions (`ProductDistribution`, `TransformedDistribution`)
   based on component capabilities.
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
   properties like `.mean`, `.var`, `.fields`, `.flatten`, and produced
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
| `Distribution[T]` | Generic base parameterized by value type; provides `record_template` and `auxiliary` properties |
| `Record` | Named, immutable, JAX-pytree container for structured non-random values; leaves stored verbatim (no coercion); `select()` for workflow function splatting |
| `NumericRecord` (subclass of `Record`) | Post-construction invariant: every leaf is a `jax.Array` (constructor coerces via `jnp.asarray`). Adds `flatten` / `unflatten` / `flat_size`. Captures backend metadata (xarray dims/coords, pandas index) via the aux registry; `to_native()` reverses the conversion to a permissive `Record`. `Record.to_numeric()` is the symmetric forward path. |
| `RecordArray` | Batch of `Record` elements with a `RecordTemplate`; integer index → element, field index → batched array |
| `NumericRecordArray` (subclass of `RecordArray`) | Batch of `NumericRecord` elements; adds `flatten` / `mean` / `var` |
| `RecordTemplate` | Structural skeleton (field names, per-field shapes or `None`); enables `NumericRecord.unflatten` without an example instance |
| `RecordDistribution` | Record-based distribution base; `fields`, `__getitem__` → `_RecordDistributionView`, `select()` / `select_all()` for correlated broadcasting; `.n` = cells in `batch_shape` |
| `_RecordDistributionView` | Lightweight component reference; dynamic protocol support matching parent capabilities |
| `NumericRecordDistribution` | Numeric-array distribution base; per-field `dtypes`, `supports`, `event_shapes`; base for all TFP-backed distributions |
| `DistributionArray` | Shape-indexed `Array[Distribution]`; exposes only the container surface (indexing, iteration, `batch_shape`, `event_shape`, `n`, `components`). Vectorized ops are delivered by the `WorkflowFunction` sweep layer — passing a `DistributionArray` to an op whose hint is a scalar `Distribution` / protocol triggers cell-by-cell dispatch, and outputs stack into `NumericRecordArray` / `RecordArray` / (nested) `DistributionArray`. Produced by parameter-sweep workflow functions whose inner call returns a `Distribution`. |
| `JointEmpirical` / `NumericJointEmpirical` | Weighted joint samples distribution. Generic base supports only sampling + conditioning; the numeric subclass adds Gaussian-approximation `SupportsLogProb` and exact `SupportsMean` / `SupportsVariance`. `JointEmpirical(...)` dispatches to `NumericJointEmpirical` when every field is numeric. |
| `WorkflowFunction` | Orchestration-aware function wrapper; groups views by parent for correlated broadcasting |
| `Module` | Stateful workflow-aware base class (see `@workflow_method`) |
| Protocols | `SupportsSampling`, `SupportsLogProb`, `SupportsMean`, `SupportsConditioning`, etc.; dynamic inclusion on `ProductDistribution` and `TransformedDistribution` |
| `MethodRegistry` | Generic priority-based dispatch; used by the inference method registry |
| `ProbabilisticModel` | Base for models (extends `Distribution`; provides `fields`) |
| `SimpleGenerativeModel` | Simulator-only model wrapper for SBI/ABC (prior + `GenerativeLikelihood`) |
| `IncrementalConditioner` | Stateful `Module` for sequential Bayesian updating via `update()` / `update_all()` |
| `iterate` / combinators | Iterative distribution transformation; `with_conversion`, `with_resampling` |
| `sbi_learn_conditional` / `sbi_learn_likelihood` | SBI workflow functions; return `DirectSamplerSBIModel` or `SimpleModel` with neural likelihood |
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

Built-in methods:

| Priority | Name | Backend | Applies to |
|----------|------|---------|------------|
| 100 | `tfp_nuts` | TFP | Any `SupportsLogProb` (JAX-traceable) |
| 90 | `tfp_hmc` | TFP | Any `SupportsLogProb` (JAX-traceable) |
| 80 | `nutpie_nuts` | nutpie | `StanModel`, `PyMCModel` |
| 70 | `cmdstan_nuts` | CmdStanPy | `StanModel` |
| 60 | `pymc_nuts` | PyMC | `PyMCModel` |
| 50 | `tfp_rwmh` | TFP | Any `SupportsLogProb` |
| 40 | `sbijax_smcabc` | sbijax | `SimpleGenerativeModel` |
| 35 | `pymc_advi` | PyMC | `PyMCModel` |

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
no priority system, no feasibility check, no `execute()`. Use
`MethodRegistry` (inference) or `ConverterRegistry` (distribution
conversion) when behaviour dispatch is needed.

### Generic vs array-specific pattern

Some distribution families have a generic base and an array-specific
subclass:

- `EmpiricalDistribution[T]` / `NumericEmpiricalDistribution`
- `BootstrapReplicateDistribution[T]` / `ArrayBootstrapReplicateDistribution`

The generic base carries only type-agnostic features (sampling, expectation).
The numeric variant adds `event_shape`, `dim`, `dtype`, `support`, and moment
protocols (`SupportsMean`, `SupportsVariance`, `SupportsCovariance`).

**Automatic factory dispatch:** Constructing a generic base with a numeric
array automatically returns the numeric-specific subclass:

```python
EmpiricalDistribution(jnp.ones((100, 3)))
# → returns NumericEmpiricalDistribution

BootstrapReplicateDistribution(jnp.ones((50, 2)))
# → returns ArrayBootstrapReplicateDistribution
```

This is implemented via `__new__` on the generic base classes.  Non-numeric
inputs (e.g. lists of objects, numpy object arrays) remain as the generic
base class.  Calling the numeric subclass directly (e.g.
`NumericEmpiricalDistribution(...)`) is unaffected by the dispatch.

---

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed coding conventions.
