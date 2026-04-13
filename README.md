# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)

ProbPipe is a Python framework for building scalable probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

Most workflows for probabilistic inference can be described in terms of **distributions**, **fixed inputs**, **operations** that transform distributions, and **differentiation** with respect to fixed inputs. Implementing these workflows, however, is harder than describing them:
1. **Algorithmic challenges**: There are many possible algorithms for common operations, with varying trade-offs that need to be explored in a problem-specific manner. ProbPipe provides a unified framework for comparing and using such algorithms (e.g., for posterior inference) with reasonable defaults. 
2. **Representational challenges**: Algorithms require -- and output -- specific distribution formats that are not always with other parts of the workflow. ProbPipe manages representations and algorithms automatically, while still allowing the user to override the defaults when necessary.

**[Documentation](https://tarps-group.github.io/prob-pipe/)** | **[Getting Started Tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** | **[API Reference](https://tarps-group.github.io/prob-pipe/)**

## Key Features

- **Named structured values** -- `Values` is the universal container for non-random structured data, just as `Distribution` is for random quantities. Both support named fields and `select()` for workflow function splatting. Named distributions automatically produce named posterior draws: `posterior.draws()` returns `Values(params=...)`.
- **Protocol-based distributions** -- A distribution's capabilities are declared via `@runtime_checkable` protocols (`SupportsSampling`, `SupportsLogProb`, `SupportsMean`, ...), enabling structural subtyping across representations. Composite distributions dynamically include protocols based on their components' capabilities.
- **Automatic uncertainty propagation** -- With `@workflow_function` broadcasting, a user can pass a distribution where a function expects a concrete value, propagating that uncertainty into the function's output. Use `dist.select("x", "y")` to pass named components while preserving correlation.
- **Pluggable inference** -- A single `condition_on` interface is backed by an inference registry that auto-selects the best-available algorithm (NUTS, HMC, ADVI, and more), taking into account a distribution's compatibility with available backends (TFP, nutpie, Stan, PyMC). Override with `method=` when you want control; inference diagnostics are attached to the output distribution.
- **Automatic distribution conversion** -- A converter registry converts between distribution types, using a similar registry-backed approach to `condition_on`.
- **JAX-native** -- Workflows and array-based distributions are compatible with JAX (`vmap`, `jit`, `grad`), with always-on support for TFP distributions and inference methods.
- **Provenance tracking** -- Every distribution records its lineage through operations, ensuring traceability of the workflow.
- **Prefect orchestration** -- Enable Prefect to distribute pipeline steps across machines without code changes.



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
pip install .[nutpie]    # nutpie Markov chain Monte Carlo (MCMC) sampler
```

## Quick Example

```python
import jax, jax.numpy as jnp, numpy as np
import tensorflow_probability.substrates.jax.glm as tfp_glm
from probpipe import (
    MultivariateNormal, GLMLikelihood, SimpleModel,
    workflow_function, condition_on, mean,
)

# 1. Define a logistic regression model (non-conjugate)
x_obs = jax.random.normal(jax.random.PRNGKey(42), shape=(80,))
y_obs = jax.random.bernoulli(jax.random.PRNGKey(1), jax.nn.sigmoid(-1 + 2 * x_obs)).astype(jnp.float32)
X = jnp.column_stack([jnp.ones_like(x_obs), x_obs])

# Named prior — produces named posterior draws automatically
prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
model = SimpleModel(prior, GLMLikelihood(tfp_glm.Bernoulli(), X))

# 2. Condition on data — runs NUTS automatically
posterior = condition_on(model, y_obs, num_results=2000, num_warmup=1000, random_seed=0)
draws = posterior.draws()   # Values(beta=array(2000, 2)) — named!
draws.beta.mean(axis=0)     # Array([-1.38, 1.77], dtype=float32)

# 3. Propagate uncertainty — select() preserves posterior correlation
@workflow_function
def predict_prob(beta, x):
    return jax.nn.sigmoid(beta[0] + beta[1] * x)

x_grid = jnp.linspace(-3, 3, 100)
predictive = predict_prob(**posterior.select("beta"), x=x_grid)
predictive      # EmpiricalDistribution(n=2000) over predicted P(y=1|x)
```

Broadcasting samples from the posterior and evaluates the function for each draw, returning the full predictive distribution:

```python
import matplotlib.pyplot as plt

S = np.array(predictive.samples)  # (2000, 100) — one curve per posterior draw
lo, hi = np.percentile(S, [5, 95], axis=0)
plt.fill_between(x_grid, lo, hi, alpha=0.3, label="90% CI")
plt.plot(x_grid, S.mean(axis=0), lw=2, label="Posterior mean")
plt.plot(x_grid, jax.nn.sigmoid(-1.0 + 2.0 * x_grid), "k--", label="True")
plt.scatter(np.array(x_obs), np.array(y_obs), s=12, alpha=0.4, label="Data")
plt.xlabel("x"); plt.ylabel("P(y = 1 | x)"); plt.legend(fontsize=8)
```

![Posterior predictive](docs/assets/images/readme_logistic.png)

## Next Steps

- **[Getting Started Tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** -- iterative Bayesian model building with ProbPipe
- **[API Reference](https://tarps-group.github.io//)** -- full class and function documentation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR workflow, and guidelines.
