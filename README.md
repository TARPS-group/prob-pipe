# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)

ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

**[Documentation](https://tarps-group.github.io/prob-pipe/)** | **[Tutorials](https://tarps-group.github.io/prob-pipe/tutorials/)** | **[API Reference](https://tarps-group.github.io/prob-pipe/api/distributions/)**

## Key Features

- **20+ probability distributions** wrapping TensorFlow Probability with a uniform interface, all fully JAX-differentiable
- **Automatic uncertainty propagation** through workflow nodes via sample broadcasting
- **Joint distributions** with independent, autoregressive, and Gaussian conditioning support
- **Built-in MCMC inference** with NUTS, HMC, and random-walk Metropolis-Hastings, plus optional Stan and PyMC backends
- **Bagged posteriors** for reproducible inference under model misspecification
- **Provenance tracking** recording full lineage from any result back to its inputs
- **Prefect orchestration** for distributing pipeline steps across machines

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

## Walkthrough: Bayesian Linear Regression

This walkthrough demonstrates ProbPipe's approach to posterior inference, uncertainty propagation, and reproducible analysis using a Bayesian linear regression example.

### 1. Define the model

ProbPipe models are assembled from a prior distribution and a likelihood. Operations like `sample()`, `log_prob()`, `mean()`, and `condition_on()` are standalone workflow functions — you call `mean(dist)`, not `dist.mean()`.

```python
import jax
import jax.numpy as jnp
from probpipe import (
    MultivariateNormal, SimpleModel, EmpiricalDistribution,
    JointBootstrapDistribution,
    condition_on, sample, mean, variance, log_prob,
)
from probpipe.modeling import Likelihood
from probpipe.core.node import wf

class LinearRegressionLikelihood(Likelihood):
    @wf
    def log_likelihood(self, params, data):
        x, y = data[:, :-1], data[:, -1]
        predicted = x @ params[1:] + params[0]
        return jnp.sum(-0.5 * (y - predicted) ** 2)

prior = MultivariateNormal(loc=jnp.zeros(3), cov=10.0 * jnp.eye(3))
model = SimpleModel(prior, LinearRegressionLikelihood())
```

### 2. Fit the standard posterior

Conditioning the model on data runs MCMC (NUTS when gradients are available, gradient-free MH otherwise) and returns an `MCMCApproximateDistribution` — a posterior distribution with chain structure and diagnostics.

```python
# Synthetic data: y = 1.0 + 2.0*x1 - 0.5*x2 + noise
key = jax.random.PRNGKey(42)
N = 100
x = jax.random.normal(key, shape=(N, 2))
y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * jax.random.normal(key, shape=(N,))
data = jnp.column_stack([x, y])

posterior = condition_on(model, data, num_results=1000, num_warmup=500, random_seed=0)
mean(posterior)       # Array([1.009, 1.943, -0.549], dtype=float32)
variance(posterior)   # Array([0.0097, 0.0143, 0.0120], dtype=float32)
```

### 3. Bagged posteriors for reproducible inference

Under model misspecification, standard Bayesian posteriors can be unreliable — credible sets from replicate datasets may not overlap. The *bagged posterior* ([Huggins & Miller, 2024](https://doi.org/10.1214/24-EJS2237)) averages over posteriors conditioned on bootstrapped datasets, yielding reproducible uncertainty quantification.

ProbPipe makes this natural. `JointBootstrapDistribution` represents the bootstrap sampling distribution over datasets. Broadcasting `condition_on` over bootstrap datasets produces multiple posteriors that can be aggregated:

```python
# Wrap the observed data as a bootstrap sampling distribution
bootstrap_data = JointBootstrapDistribution(EmpiricalDistribution(data))

# Each bootstrap sample is a full dataset; condition_on broadcasts over them
bagged_posterior = condition_on(model, bootstrap_data)
```

Because `JointBootstrapDistribution` implements `SupportsSampling`, the `condition_on` workflow function automatically broadcasts: it draws bootstrap datasets and runs MCMC on each, returning a collection of posterior distributions.

We can visualize the bagged posterior by overlaying the individual bootstrap posterior densities:

```python
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
param_names = [r"$\beta_0$ (intercept)", r"$\beta_1$ (slope 1)", r"$\beta_2$ (slope 2)"]
true_values = [1.0, 2.0, -0.5]

for j, (ax, name, truth) in enumerate(zip(axes, param_names, true_values)):
    for post in bagged_posterior[:10]:
        draws = post.samples[:, j]
        kde = gaussian_kde(np.array(draws))
        xs = np.linspace(float(draws.min()) - 0.3, float(draws.max()) + 0.3, 200)
        ax.plot(xs, kde(xs), alpha=0.6)
    ax.axvline(truth, color="black", linestyle="--", label="true value")
    ax.set_xlabel(name)
axes[0].set_ylabel("Density")
axes[0].legend()
fig.suptitle("Bagged Posterior: 10 Bootstrap Posterior Densities")
fig.tight_layout()
```

![Bagged posterior densities](docs/assets/images/bagged_posterior_densities.png)

The bootstrap posterior densities are tightly clustered — the sampling variability (spread across bootstrap replicates) is small compared to the width of each individual posterior. This indicates the standard posterior is already stable and reproducible for this well-specified model. In such cases the bagged posterior is wider than necessary, making it somewhat conservative. This is expected: bagging provides a safety net under misspecification, at the cost of mild overcoverage when the model is correct.

### 4. Use an external sampler

ProbPipe supports multiple inference backends. For Stan models, `condition_on` uses CmdStan's NUTS sampler. You can also use nutpie for Stan or PyMC models:

```python
from probpipe.modeling import StanModel
from probpipe.inference import condition_on_nutpie

# Stan model: condition_on uses CmdStan's built-in NUTS
stan_mod = StanModel("my_model.stan", data={"N": N})
stan_posterior = condition_on(stan_mod, {"x": x, "y": y})

# Or use nutpie (Rust-based NUTS) for the same model
nutpie_posterior = condition_on_nutpie(stan_mod, {"x": x, "y": y}, num_results=2000)
```

### 5. Propagate posterior uncertainty

Once you have a posterior (standard or bagged), propagate uncertainty through downstream computations. When a workflow function receives a distribution, ProbPipe automatically broadcasts over samples:

```python
from probpipe import WorkflowFunction

def predict_impl(params, x_new):
    return x_new @ params[1:] + params[0]

predict = WorkflowFunction(func=predict_impl)

# Predictions propagate posterior uncertainty automatically
x_new = jnp.array([0.5, -0.3])
predictive = predict(params=posterior, x_new=x_new)

# predictive is a BroadcastDistribution (joint over inputs and output)
# Call marginalize() to get the output distribution
mean(predictive.marginalize())       # Array(2.1497, dtype=float32)
variance(predictive.marginalize())   # Array(0.0151, dtype=float32)
```

### 6. Provenance tracking

Every distribution records its lineage automatically. Provenance chains enable full reproducibility — you can trace any result back to the data and model that produced it:

```python
from probpipe import provenance_ancestors

posterior.source
# Provenance('nuts', parents=[MultivariateNormal])

provenance_ancestors(predictive)
# [MCMCApproximateDistribution(num_chains=1, ..., algorithm=nuts, ...),
#  MultivariateNormal(event_shape=(3,))]
```

## Architecture

ProbPipe's architecture separates the **representational layer** (distributions as immutable data) from the **algorithmic layer** (operations on distributions as workflow functions):

- **Distributions are immutable** — parameters fixed at construction; operations return new distributions.
- **Operations are workflow functions** — `sample()`, `mean()`, `log_prob()`, `condition_on()` are standalone `WorkflowFunction` instances in `probpipe/core/ops.py`.
- **Capabilities via protocols** — distributions declare support through `@runtime_checkable` protocols (e.g., `SupportsSampling`, `SupportsLogProb`, `SupportsMean`). Operations check protocols at dispatch time.
- **Private method convention** — protocols define `_method()` (e.g., `_sample`, `_log_prob`, `_mean`). The public API is via ops: `sample(dist)`, not `dist.sample()`.

## Contributing

Contributions are welcome! To get started:

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/prob-pipe.git
   cd prob-pipe
   pip install -e .[dev]
   ```
2. Create a feature branch, make your changes, and ensure tests pass:
   ```bash
   git checkout -b my-feature
   pytest
   ```
3. Open a pull request against `main`.
