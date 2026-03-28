# CLAUDE.md

## Project Overview

ProbPipe is a JAX-based library for probabilistic data-processing pipelines. It provides immutable distribution objects, protocol-based capabilities, and workflow functions for sampling, inference, and conditioning.

## Architecture

### Core Design Principles

- **Distributions are immutable** — parameters fixed at construction; operations return new distributions.
- **Operations are workflow functions** — `sample()`, `mean()`, `log_prob()`, `condition_on()` are standalone `WorkflowFunction` instances in `probpipe/core/ops.py`.
- **Capabilities via protocols** — distributions declare support through `@runtime_checkable` protocols in `probpipe/core/protocols.py`. Operations check protocols at dispatch time.
- **Private method convention** — protocols define `_method()` (e.g., `_sample`, `_log_prob`, `_mean`). Public API is via ops: `sample(dist)`, not `dist.sample()`.

### Package Structure

```
probpipe/
├── core/
│   ├── distribution.py    # Distribution[T] base, ArrayDistribution, EmpiricalDistribution, TFPDistribution, etc.
│   ├── protocols.py       # @runtime_checkable protocols: SupportsSampling, SupportsLogProb, SupportsMean, etc.
│   ├── ops.py             # Standalone WorkflowFunction instances: sample, mean, log_prob, condition_on, etc.
│   ├── node.py            # Node, Module, WorkflowFunction, @wf decorator
│   ├── modeling.py        # Likelihood, GenerativeLikelihood, IterativeForecaster (legacy re-exports)
│   ├── _modeling_legacy.py # Legacy MCMCSampler, RWMH (deprecated, backward compat)
│   └── provenance.py      # Provenance tracking utilities
├── distributions/          # Concrete distributions: continuous, discrete, multivariate, joint, transformed
├── modeling/               # Probabilistic model wrappers
│   ├── _base.py           # ProbabilisticModel(Distribution, SupportsConditionableComponents)
│   ├── _simple.py         # SimpleModel: prior + likelihood, dynamic protocol delegation
│   ├── _stan.py           # StanModel: BridgeStan-backed (optional dep: bridgestan)
│   └── _pymc.py           # PyMCModel: PyMC wrapper (optional dep: pymc)
├── inference/              # Inference algorithms
│   ├── _mcmc_distribution.py # MCMCApproximateDistribution(EmpiricalDistribution) with chain structure
│   ├── _diagnostics.py    # MCMCDiagnostics dataclass
│   ├── _rwmh.py           # Standalone RWMH workflow function
│   └── _nutpie.py         # Nutpie-backed NUTS sampling (optional dep: nutpie)
├── converters/             # Distribution conversion registry
└── linalg/                 # Linear algebra utilities for random functions
```

### Key Protocols

| Protocol | Key Method | Description |
|----------|-----------|-------------|
| `SupportsSampling` | `_sample(key, sample_shape)` | Can draw samples |
| `SupportsLogProb` | `_log_prob(value)` | Has normalized log-probability |
| `SupportsMean` | `_mean()` | Has exact mean |
| `SupportsConditioning` | `_condition_on(observed, **kw)` | Can condition on data |
| `SupportsNamedComponents` | `component_names`, `__getitem__` | Has named sub-distributions |
| `SupportsConditionableComponents` | `conditionable_components` | Model with observable components |

## Development

### Setup

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

Tests use `pytest-xdist` with `-n auto --dist worksteal` by default (configured in `pyproject.toml` addopts).

### Coverage

```bash
pytest --cov=probpipe --cov-report=term-missing
```

Target: >90% on all modules.

### Docs

```bash
mkdocs build --strict   # build docs, fail on warnings
mkdocs serve            # local preview
```

API docs use `mkdocstrings` directives in `docs/api/*.md` referencing fully-qualified Python paths.

## Conventions

- **WorkflowFunction reserved parameter names**: Do not use `seed` as a parameter name — use `random_seed` instead. `WorkflowFunction` reserves `seed` internally.
- **Optional dependencies**: Use try-import pattern with helpful error messages. Tests for optional deps use `pytest.importorskip()` or mock-based approaches with `patch.dict(sys.modules)`.
- **Protocol checks**: Use `isinstance(obj, Protocol)` for instance checks. `issubclass` does not work with protocols that have non-method members (properties).
- **Test isolation**: Stan/nutpie tests use `object.__new__()` + manual attribute setting to avoid requiring compiled Stan models. PyMC tests use `pytest.importorskip("pymc")`.

## CI

GitHub Actions workflow in `.github/workflows/ci.yml`:
- Tests on Python 3.12 and 3.13
- Installs `.[dev,nutpie]` (nutpie included; bridgestan/pymc are not)
- Coverage uploaded to Codecov

Docs build in `.github/workflows/docs.yml` with `mkdocs build --strict`.
