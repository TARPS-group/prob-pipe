# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)

ProbPipe is a Python framework for building scalable probabilistic pipelines with automated uncertainty quantification.

### Why ProbPipe?

Most workflows for probabilistic inference can be described in terms of **distributions**, **fixed values** (data, hyperparameters, covariates), and **operations** that transform distributions. Implementing these workflows, however, is harder than describing them:

1. **Algorithmic challenges** -- there are many possible algorithms for common operations, with varying trade-offs that need to be explored in a problem-specific manner. A posterior could be approximated using different MCMC algorithms, variational inference, or sequential Monte Carlo.
2. **Representational challenges** -- algorithms require -- and output -- specific formats for both distributions and fixed values that are not always compatible with other parts of the workflow. Fixed values may be named parameter vectors, covariate matrices, or structured observations -- and different algorithms expect different representations.

### Simplification via abstraction

ProbPipe addresses these challenges through a single design principle: **simplification via abstraction**. There are just three core types:

- **`Distribution`** -- the universal representation of random quantities (priors, posteriors, data-generating processes). A distribution's capabilities are declared via protocols (`SupportsSampling`, `SupportsLogProb`, ...), and ProbPipe converts between representations as needed.
- **`Record`** -- the universal container for non-random structured data (observed datasets, hyperparameters, design matrices). `Record` is the deterministic counterpart of `Distribution`.
- **`WorkflowFunction`** -- operations that take distributions or fixed values as input and return distributions or fixed values as output. Decorate any function with `@workflow_function` and ProbPipe automatically propagates uncertainty: pass a `Distribution` where the function expects a concrete value, and the output becomes a `Distribution` over results.

`Distribution` and `Record` follow the same syntax for accessing their components and passing those components into a `WorkflowFunction`, so they can easily be interchanged. Both support **named fields** and a **`select()`** method for splatting (e.g., `predict(**posterior.select("intercept", "slope"))`). The implementation details -- algorithms, data and distribution representations -- are invisible to the user, while remaining fully configurable when control is needed.

### Built-in operations

ProbPipe provides a set of built-in **ops** -- special workflow functions that dispatch based on a distribution's protocols:

- **`condition_on`** -- condition a model on observed data, automatically selecting the best inference algorithm (or specify one with `method=`).
- **`mean`**, **`variance`**, **`cov`**, **`expectation`** -- compute distributional summaries, with automatic Monte Carlo fallback when exact computation is unavailable.
- **`sample`**, **`log_prob`** -- draw samples or evaluate densities through a uniform interface.
- **`from_distribution`** -- convert between distribution representations via the converter registry.
- **`predictive_check`** -- built-in prior and posterior predictive checking.

**[Documentation](https://tarps-group.github.io/prob-pipe/)** | **[Getting Started Tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** | **[API Reference](https://tarps-group.github.io/prob-pipe/)**

## Quick Example

Logistic regression with named parameters: simulate data, fit the model, and propagate posterior uncertainty through a prediction.

```python
import jax, jax.numpy as jnp
import tensorflow_probability.substrates.jax.glm as tfp_glm
from probpipe import (
    Normal, ProductDistribution, GLMLikelihood, SimpleModel,
    workflow_function, condition_on, mean,
)

# --- Simulate data from a logistic regression ---
beta_true = jnp.array([-1.0, 2.0])  # intercept, slope
x = jax.random.normal(jax.random.PRNGKey(0), shape=(200,))
X = jnp.column_stack([jnp.ones_like(x), x])

likelihood = GLMLikelihood(tfp_glm.Bernoulli(), X, seed=1)
y = likelihood.generate_data(beta_true, 200).astype(jnp.float32)

# --- 1. Build a model with named parameters ---
prior = ProductDistribution(
    intercept=Normal(loc=0.0, scale=5.0, name="intercept"),
    slope=Normal(loc=0.0, scale=5.0, name="slope"),
)
model = SimpleModel(prior, likelihood)

# --- 2. Condition on data (auto-selects NUTS) ---
posterior = condition_on(model, y)
draws = posterior.draws()            # Record(intercept=..., slope=...)
draws.intercept.mean()               # -0.93  (true: -1.0)
draws.slope.mean()                   #  2.18  (true:  2.0)

# --- 3. Propagate uncertainty through a prediction ---
@workflow_function
def predict_prob(intercept, slope, x):
    return jax.nn.sigmoid(intercept + slope * x)

x_new = jnp.linspace(-3, 3, 50)
predictive = predict_prob(**posterior.select('intercept', 'slope'), x=x_new)
# predictive is a Distribution over predicted P(y=1|x) curves
```

`predict_prob` is a `@workflow_function`: ProbPipe samples from the posterior and evaluates the function for each draw, returning the full predictive distribution. Plotting the result:

```python
import numpy as np, matplotlib.pyplot as plt

S = np.array(predictive.samples)     # (n_draws, 50)
lo, hi = np.percentile(S, [5, 95], axis=0)
plt.fill_between(np.array(x_new), lo, hi, alpha=0.3, label='90% CI')
plt.plot(np.array(x_new), S.mean(axis=0), lw=2, label='Posterior mean')
true_curve = jax.nn.sigmoid(beta_true[0] + beta_true[1] * x_new)
plt.plot(np.array(x_new), np.array(true_curve), 'k--', label='True')
plt.scatter(np.array(x), np.array(y), s=12, alpha=0.4, label='Data')
plt.xlabel('x'); plt.ylabel('P(y = 1 | x)'); plt.legend(fontsize=8)
```

![Posterior predictive](docs/assets/images/readme_logistic.png)

## Key Features

- **Protocol-based dispatch** -- a distribution's capabilities are declared via `@runtime_checkable` protocols (`SupportsSampling`, `SupportsLogProb`, `SupportsMean`, ...). Operations like `condition_on` and `from_distribution` use these protocols to auto-select the best algorithm from a pluggable registry. Override with `method=` when you want control.
- **Multiple backends** -- the inference registry spans TFP (NUTS, HMC, RWMH), nutpie, CmdStan, PyMC (NUTS, ADVI), and simulation-based inference (SMC-ABC via sbijax). Swap backends without changing model code.
- **Automatic distribution conversion** -- a converter registry converts between distribution representations (e.g., MCMC samples to KDE) as needed, using protocol-based dispatch analogous to `condition_on`.
- **JAX-native** -- distributions and workflow functions are compatible with JAX (`vmap`, `jit`, `grad`), with built-in support for TFP distributions.
- **Provenance tracking** -- each distribution records how it was created (algorithm, parents, metadata), enabling full lineage tracing from any result back to its inputs.
- **Prefect orchestration** -- distribute pipeline steps across machines and CPUs without code changes.

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

## Next Steps

- **[Getting Started Tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** -- iterative Bayesian model building with ProbPipe
- **[API Reference](https://tarps-group.github.io/prob-pipe/)** -- full class and function documentation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR workflow, and guidelines.
