# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)

ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

Most workflows for probabilistic inference -- including validation procedures -- can be described in terms of **distributions** (priors, posteriors, data products), **fixed inputs** (data, hyperparameters), **operations** that transform distributions (conditioning, pushforwards, expectations), and **differentiation** with respect to fixed inputs. However, *implementing* these workflows faces algorithm challenges (many algorithms for the same operation, across incompatible packages) and representation challenges (algorithms require or output specific distribution formats). ProbPipe lets you build pipelines in terms of these abstract components while managing the underlying representations and algorithms automatically.

**[Documentation](https://tarps-group.github.io/prob-pipe/)** | **[Getting Started](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** | **[API Reference](https://tarps-group.github.io/prob-pipe/api/distributions/)**

## Key Features

- **Protocol-based distributions** -- capabilities declared via `@runtime_checkable` protocols (`SupportsSampling`, `SupportsLogProb`, `SupportsMean`, ...), enabling structural subtyping across backends
- **Automatic uncertainty propagation** -- `@workflow_function` broadcasting: pass a distribution where a function expects a concrete value and get a distribution back
- **MCMC inference** -- NUTS/HMC with automatic gradient-free RWMH fallback; diagnostics (acceptance rate, divergences, tree depth) on every run
- **Multiple backends** -- native TFP, nutpie, Stan (via BridgeStan), and PyMC models, all unified behind `condition_on`
- **Predictive checking** -- `predictive_check` for prior and posterior predictive checks with test statistics and p-values
- **Sequential Bayesian updating** -- `IncrementalConditioner` chains posterior updates across data batches
- **Automatic distribution conversion** -- converter registry for moment-matching and sampling-based conversion between distribution types
- **JAX-native** -- `vmap`, `jit`, `grad` throughout; TFP substrate for distribution math
- **Provenance tracking** -- every distribution records its lineage from inputs through operations
- **Prefect orchestration** -- distribute pipeline steps across machines without code changes

## Installation

Requires Python >= 3.12 (tested on 3.12 and 3.13).

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

Core dependencies: JAX and TensorFlow Probability. ProbPipe uses [tfp-nightly](https://pypi.org/project/tfp-nightly/), which is the [recommended approach](https://github.com/tensorflow/probability/issues/1994#issuecomment-3129033043) for TFP on JAX since stable TFP releases are tied to TensorFlow and often lag behind JAX.

Optional extras:

```bash
pip install .[dev]       # pytest, jupyter, matplotlib, graphviz
pip install .[prefect]   # Prefect orchestration backend
pip install .[stan]      # Stan models via BridgeStan + CmdStanPy
pip install .[pymc]      # PyMC model integration
pip install .[nutpie]    # nutpie MCMC sampler
```

## Quick Example

```python
import jax
import jax.numpy as jnp
from probpipe import (
    MultivariateNormal, SimpleModel, workflow_function,
    condition_on, mean, variance,
)

# 1. Define a model: prior + likelihood
class LinearLikelihood:
    def log_likelihood(self, params, data):
        x, y = data[:, 0], data[:, 1]
        return jnp.sum(-0.5 * (y - (params[0] + params[1] * x)) ** 2)

prior = MultivariateNormal(loc=jnp.zeros(2), cov=10.0 * jnp.eye(2))
model = SimpleModel(prior, LinearLikelihood())

# 2. Condition on data -- runs NUTS automatically
data = jnp.column_stack([jnp.linspace(0, 1, 20), 1.0 + 2.0 * jnp.linspace(0, 1, 20)])
posterior = condition_on(model, data)

# 3. Propagate uncertainty through a prediction function
@workflow_function
def predict(params, x):
    return params[0] + params[1] * x

predictive = predict(params=posterior, x=0.5)
mean(predictive)       # posterior predictive mean
variance(predictive)   # posterior predictive variance
```

## Next Steps

- **[Getting Started tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** -- iterative Bayesian model building with ProbPipe
- **[API Reference](https://tarps-group.github.io/prob-pipe/api/distributions/)** -- full class and function documentation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR workflow, and guidelines.
