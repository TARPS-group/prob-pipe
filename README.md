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

Because `JointBootstrapDistribution` implements `SupportsSampling`, the `condition_on` workflow function automatically broadcasts: it draws bootstrap datasets and runs MCMC on each, returning the **bagged posterior** — a mixture distribution that averages over the individual bootstrap posteriors. The individual posteriors are accessible via `.components`.

To see why bagging matters, compare the well-specified Gaussian data above with a **misspecified** scenario — heavy-tailed noise that violates the model's Gaussian assumption:

```python
import numpy as np
rng = np.random.default_rng(42)
x_ht = jax.random.normal(jax.random.PRNGKey(99), shape=(N, 2))
noise_ht = rng.standard_t(df=2, size=N)
y_ht = 1.0 + 2.0 * np.array(x_ht[:, 0]) - 0.5 * np.array(x_ht[:, 1]) + noise_ht
data_ht = jnp.column_stack([x_ht, jnp.array(y_ht, dtype=jnp.float32)])

posterior_ht = condition_on(model, data_ht, num_results=1000, num_warmup=500, random_seed=0)

bootstrap_ht = JointBootstrapDistribution(EmpiricalDistribution(data_ht))
bagged_posterior_ht = condition_on(model, bootstrap_ht)
```

Plotting both cases side by side:

```python
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from probpipe import sample

param_names = [r"$\beta_0$ (intercept)", r"$\beta_1$ (slope 1)", r"$\beta_2$ (slope 2)"]
true_values = [1.0, 2.0, -0.5]

fig, all_axes = plt.subplots(2, 3, figsize=(12, 7))

for row, (label, bagged, axes) in enumerate([
    ("Well-specified (Gaussian noise)", bagged_posterior, all_axes[0]),
    ("Misspecified (heavy-tailed noise)", bagged_posterior_ht, all_axes[1]),
]):
    individual_posteriors = bagged.components

    for j, (ax, name, truth) in enumerate(zip(axes, param_names, true_values)):
        for i, post in enumerate(individual_posteriors[:10]):
            draws = post.samples[:, j]
            kde = gaussian_kde(np.array(draws))
            xs = np.linspace(float(draws.min()) - 0.3, float(draws.max()) + 0.3, 200)
            ax.plot(xs, kde(xs), alpha=0.3, lw=0.8, color="steelblue",
                    label="individual posteriors" if i == 0 else None)

        bagged_draws = np.array(sample(bagged, sample_shape=(2000,))[:, j])
        kde_bag = gaussian_kde(bagged_draws)
        xs = np.linspace(float(bagged_draws.min()) - 0.3, float(bagged_draws.max()) + 0.3, 200)
        ax.plot(xs, kde_bag(xs), color="darkorange", lw=2.5, label="bagged posterior")

        ax.axvline(truth, color="black", linestyle="--", lw=1.5, label="true value")
        ax.set_xlabel(name)
    axes[0].set_ylabel("Density")
    axes[0].set_title(label, fontsize=10, loc="left")
    if row == 0:
        axes[0].legend(fontsize=8)
fig.suptitle("Bagged Posterior vs. Individual Bootstrap Posteriors")
fig.tight_layout()
```

![Bagged posterior densities](docs/assets/images/bagged_posterior_densities.png)

**Top row (well-specified):** The bootstrap posteriors are tightly clustered and the bagged posterior nearly coincides with each individual one. Here bagging is mildly conservative — the extra width is small.

**Bottom row (misspecified):** Heavy-tailed noise causes individual posteriors to spread apart (the outlier-sensitive Gaussian likelihood produces different estimates for different bootstrap datasets). The bagged posterior is noticeably wider, correctly reflecting the additional uncertainty from model misspecification.

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

When a `WorkflowFunction` receives a distribution where it expects a concrete value, ProbPipe automatically broadcasts over samples. Start with a reusable prediction function:

```python
from probpipe import WorkflowFunction
import numpy as np

def predict(params, x):
    return x @ params[1:] + params[0]

predict_wf = WorkflowFunction(func=predict)

# Posterior uncertainty propagates automatically
x_new = jnp.array([[0.5, -0.3]])
predictive = predict_wf(params=posterior, x_new=x_new)
mean(predictive)       # posterior predictive mean
variance(predictive)   # posterior predictive variance
```

This same pattern makes **posterior predictive checks** natural. Separate the pipeline into reusable pieces — simulation and test statistics — with `predict` driving the generative model:

```python
x_obs, y_obs = np.array(data_ht[:, :-1]), np.array(data_ht[:, -1])
rng_ppc = np.random.default_rng(123)

def simulate_replicate(params):
    """Draw y_rep from the model: y_rep ~ N(predict(params, x), I)."""
    return np.array(predict(params, x_obs)) + rng_ppc.normal(size=len(x_obs))

# Test statistics on residuals
def max_abs_residual(residuals):
    return float(np.max(np.abs(residuals)))

def excess_kurtosis(residuals):
    centered = residuals - np.mean(residuals)
    return float(np.mean((centered / np.std(residuals))**4) - 3.0)

# Compose: simulate replicate, compute residuals, apply statistic
def ppc_max_impl(params):
    y_rep = simulate_replicate(params)
    return max_abs_residual(y_rep - np.array(predict(params, x_obs)))

def ppc_kurt_impl(params):
    y_rep = simulate_replicate(params)
    return excess_kurtosis(y_rep - np.array(predict(params, x_obs)))

ppc_max = WorkflowFunction(func=ppc_max_impl, n_broadcast_samples=500)
ppc_kurt = WorkflowFunction(func=ppc_kurt_impl, n_broadcast_samples=500)

ppc_max_dist = ppc_max(params=posterior_ht)
ppc_kurt_dist = ppc_kurt(params=posterior_ht)
```

Compare to the observed residual statistics:

```python
params_hat = np.array(mean(posterior_ht))
resid_obs = y_obs - np.array(predict(params_hat, x_obs))
obs_max = max_abs_residual(resid_obs)      # 6.18
obs_kurt = excess_kurtosis(resid_obs)      # 2.70

p_max = float(np.mean(np.array(ppc_max_dist.samples) >= obs_max))    # 0.000
p_kurt = float(np.mean(np.array(ppc_kurt_dist.samples) >= obs_kurt)) # 0.000
```

Both p-values are extreme — the Gaussian model cannot reproduce the large residuals or heavy tails in the observed data:

![PPC checks](docs/assets/images/ppc_checks.png)

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
