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
- Installs `.[dev,nutpie]` (bridgestan and pymc are not included)
- Coverage uploaded to Codecov

Docs build (`.github/workflows/docs.yml`) with `mkdocs build --strict`.

---

## Package Structure

```
probpipe/
├── __init__.py              # Public API re-exports
├── custom_types.py          # Array, PRNGKey, ArrayLike type aliases
├── _utils.py                # Internal utilities
├── _array_utils.py          # Array manipulation helpers
├── _weights.py              # Weights class + weighted operations
│
├── core/                    # Core abstractions (no specialized distributions)
│   ├── distribution.py        # Re-export facade for all core distribution symbols
│   ├── _distribution_base.py  # Distribution[T] base, global settings
│   ├── _array_distributions.py # PyTreeArrayDistribution, ArrayDistribution,
│   │                          #   BootstrapDistribution, FlattenedView
│   ├── _empirical.py          # EmpiricalDistribution[T], ArrayEmpiricalDistribution,
│   │                          #   BootstrapReplicateDistribution[T]
│   ├── _broadcast_distributions.py # BroadcastDistribution, marginal types
│   ├── _joint.py              # JointDistribution, ProductDistribution, DistributionView
│   ├── _random_functions.py   # RandomFunction[X,Y], ArrayRandomFunction
│   ├── protocols.py           # @runtime_checkable protocol definitions
│   ├── ops.py                 # Built-in ops: sample, mean, log_prob, etc.
│   ├── constraints.py         # Constraint hierarchy (real, positive, etc.)
│   ├── node.py                # Node, Module, WorkflowFunction, @workflow_function, @workflow_method, @abstract_workflow_method
│   ├── modeling.py            # Likelihood, GenerativeLikelihood, IncrementalConditioner
│   └── provenance.py          # Provenance tracking
│
├── distributions/           # Concrete distribution implementations
│   ├── continuous.py        # Normal, Gamma, Beta, etc.
│   ├── discrete.py          # Bernoulli, Poisson, Categorical, etc.
│   ├── multivariate.py      # MultivariateNormal, Dirichlet, Wishart, etc.
│   ├── joint.py             # SequentialJointDistribution, JointEmpirical, etc.
│   ├── transformed.py       # TransformedDistribution
│   ├── gaussian_random_function.py  # GaussianRandomFunction, LinearBasisFunction
│   └── _tfp_base.py         # TFPDistribution mixin
│
├── modeling/                # Probabilistic model wrappers
│   ├── _base.py             # ProbabilisticModel base class
│   ├── _simple.py           # SimpleModel (prior + likelihood)
│   ├── _simple_generative.py  # SimpleGenerativeModel (prior + simulator, for SBI/ABC)
│   ├── _stan.py             # StanModel (BridgeStan, optional dep)
│   ├── _pymc.py             # PyMCModel (PyMC, optional dep)
│   └── _likelihood.py       # Likelihood + GenerativeLikelihood protocols
│
├── inference/               # Inference algorithms
│   ├── _mcmc_distribution.py  # ApproximateDistribution + make_posterior
│   ├── _rwmh.py             # Random-walk Metropolis-Hastings
│   ├── _nutpie.py           # Nutpie-backed NUTS (optional dep)
│   └── _sbijax.py           # sbijax-backed SBI: sbi_learn_conditional
│                            #   (NPE/FMPE/CMPE), sbi_learn_likelihood
│                            #   (NLE/NRE), SbiSMCABCMethod (optional dep)
│
├── converters/              # Distribution conversion registry
│   ├── _registry.py         # Converter ABC, registry, and metadata types
│   ├── _protocol.py         # Protocol-based conversion (SupportsLogProb, etc.)
│   ├── _probpipe.py         # ProbPipe ↔ ProbPipe conversions
│   ├── _tfp.py              # TFP conversions
│   └── _scipy.py            # SciPy conversions
│
└── linalg/                  # Linear algebra for random functions
    ├── linear_operator.py   # LinOp hierarchy
    ├── operations.py        # Functional interface
    └── utils.py             # add_diag_jitter, symmetrize_pd
```

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
   dispatch time.
4. **Private method convention** — protocols define `_method()` (e.g.,
   `_sample`, `_log_prob`, `_mean`). The public API is via ops:
   `sample(dist)`, not `dist.sample()`.

### Key abstractions

| Abstraction | Description |
|-------------|-------------|
| `Distribution[T]` | Generic base parameterized by value type |
| `ArrayDistribution` | Single-array specialization with TFP shape conventions |
| `WorkflowFunction` | Orchestration-aware function wrapper |
| Protocols | `SupportsSampling`, `SupportsLogProb`, `SupportsMean`, `SupportsConditioning`, etc. |
| `MethodRegistry` | Generic priority-based dispatch; used by the inference method registry |
| `ProbabilisticModel` | Base for models (extends `Distribution` + `SupportsNamedComponents`) |

### Inference method registry

`condition_on` dispatches inference via a pluggable **inference method
registry** (`inference_method_registry`).  Each method declares
`supported_types`, a `priority`, and `check()`/`execute()` methods.
The registry tries methods in descending priority order; the first
whose `check()` returns `feasible=True` wins.

Models no longer implement `_condition_on` directly — conditioning is
handled entirely by registered methods.  The removed protocol
`SupportsConditionableComponents` is no longer part of the public API;
use `SupportsNamedComponents` and the inference registry instead.

Built-in methods:

| Priority | Name | Backend | Applies to |
|----------|------|---------|------------|
| 100 | `tfp_nuts` | TFP | Any `SupportsLogProb` (JAX-traceable) |
| 90 | `tfp_hmc` | TFP | Any `SupportsLogProb` (JAX-traceable) |
| 80 | `nutpie_nuts` | nutpie | `StanModel`, `PyMCModel` |
| 70 | `cmdstan_nuts` | CmdStanPy | `StanModel` |
| 60 | `pymc_nuts` | PyMC | `PyMCModel` |
| 50 | `tfp_rwmh` | TFP | Any `SupportsLogProb` |
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

### Generic vs array-specific pattern

Some distribution families have a generic base and an array-specific
subclass:

- `EmpiricalDistribution[T]` / `ArrayEmpiricalDistribution`
- `BootstrapReplicateDistribution[T]` / `ArrayBootstrapReplicateDistribution`

The generic base carries only type-agnostic features (sampling, expectation).
The array variant adds `event_shape`, `dim`, `dtype`, `support`, and moment
protocols (`SupportsMean`, `SupportsVariance`, `SupportsCovariance`).

**Automatic factory dispatch:** Constructing a generic base with a numeric
array automatically returns the array-specific subclass:

```python
EmpiricalDistribution(jnp.ones((100, 3)))
# → returns ArrayEmpiricalDistribution

BootstrapReplicateDistribution(jnp.ones((50, 2)))
# → returns ArrayBootstrapReplicateDistribution
```

This is implemented via `__new__` on the generic base classes.  Non-numeric
inputs (e.g. lists of objects, numpy object arrays) remain as the generic
base class.  Calling the array subclass directly (e.g.
`ArrayEmpiricalDistribution(...)`) is unaffected by the dispatch.

---

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed coding conventions.
