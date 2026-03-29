# Getting Started

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

Core dependencies: JAX and TensorFlow Probability (nightly).

### Optional extras

```bash
pip install .[dev]       # pytest, jupyter, matplotlib, graphviz, docs
pip install .[prefect]   # Prefect orchestration backend
pip install .[stan]      # Stan models via BridgeStan + CmdStanPy
pip install .[pymc]      # PyMC model integration
pip install .[nutpie]    # nutpie MCMC sampler
```

## Walkthrough: Bayesian Linear Regression

This walkthrough demonstrates ProbPipe's approach to posterior inference, uncertainty propagation, and reproducible analysis. We'll fit a Bayesian linear regression, compute bagged posteriors for robust uncertainty quantification, and propagate posterior uncertainty through predictions.

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

The `SimpleModel` dynamically delegates to the prior's protocols: if the prior supports `SupportsLogProb`, the model composes prior log-probability with the likelihood for gradient-based inference. If the prior supports `SupportsSampling`, the model can generate prior predictive samples.

### 2. Fit the posterior

Condition the model on data to run MCMC. When the model supports log-probability gradients (as here), `condition_on` uses NUTS. Otherwise it falls back to gradient-free Metropolis-Hastings.

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

The result is an `MCMCApproximateDistribution` — an `EmpiricalDistribution` with chain structure and diagnostics:

```python
posterior.num_chains       # 1
posterior.diagnostics      # MCMCDiagnostics(algorithm=nuts, accept_rate=0.954, ...)
posterior.draws(chain=0)   # Array of shape (1000, 3)
```

### 3. Bagged posteriors for reproducible inference

Under model misspecification, standard Bayesian posteriors can be unreliable — credible sets from replicate datasets may not overlap
([Huggins & Miller, 2024](https://doi.org/10.1214/24-EJS2237)).
The *bagged posterior* averages over posteriors conditioned on bootstrapped datasets,
yielding reproducible uncertainty quantification.

ProbPipe makes this natural. `JointBootstrapDistribution` represents the bootstrap sampling distribution over datasets — each sample is a full bootstrapped dataset drawn i.i.d. with replacement:

```python
# Wrap the observed data as a bootstrap sampling distribution
bootstrap_data = JointBootstrapDistribution(EmpiricalDistribution(data))

# Broadcasting condition_on over bootstrap datasets produces bagged posteriors
bagged_posterior = condition_on(model, bootstrap_data)
```

By default, `JointBootstrapDistribution` sets the bootstrap dataset size equal to the original dataset size (the standard nonparametric bootstrap). You can customize it — for example, using `n=int(N**0.95)` as recommended for BayesBag model selection:

```python
bootstrap_data = JointBootstrapDistribution(EmpiricalDistribution(data), n=int(N**0.95))
bootstrap_data.n   # 79
```

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

![Bagged posterior densities](assets/images/bagged_posterior_densities.png)

The bootstrap posterior densities are tightly clustered — the sampling variability (spread across bootstrap replicates) is small compared to the width of each individual posterior. This indicates the standard posterior is already stable and reproducible for this well-specified model. In such cases the bagged posterior is wider than necessary, making it somewhat conservative. This is expected: bagging provides a safety net under misspecification, at the cost of mild overcoverage when the model is correct.

### 4. Use external samplers

ProbPipe supports multiple inference backends. Stan models use CmdStan's built-in NUTS sampler for `condition_on`. You can also use nutpie (a Rust-based NUTS implementation) for Stan or PyMC models:

```python
from probpipe.modeling import StanModel
from probpipe.inference import condition_on_nutpie

# Stan model — condition_on delegates to CmdStan's NUTS
stan_mod = StanModel("my_model.stan", data={"N": N})
stan_posterior = condition_on(stan_mod, {"x": x, "y": y})

# Or use nutpie for potentially faster sampling
nutpie_posterior = condition_on_nutpie(stan_mod, {"x": x, "y": y}, num_results=2000)
```

### 5. Propagate posterior uncertainty

Once you have a posterior, propagate uncertainty through downstream computations. When a `WorkflowFunction` receives a distribution where it expects a concrete value, ProbPipe automatically broadcasts over samples:

```python
from probpipe import WorkflowFunction

def predict_impl(params, x_new):
    return x_new @ params[1:] + params[0]

predict = WorkflowFunction(func=predict_impl)

# Posterior uncertainty propagates automatically
x_new = jnp.array([0.5, -0.3])
predictive = predict(params=posterior, x_new=x_new)

# predictive is an EmpiricalDistribution over predicted values
mean(predictive)       # Array(2.1497, dtype=float32)
variance(predictive)   # Array(0.0151, dtype=float32)
```

This works with any function — the science stays clean, and ProbPipe handles the uncertainty bookkeeping.

### 6. Track provenance

Every distribution records its lineage automatically. Provenance chains enable full reproducibility:

```python
from probpipe import provenance_ancestors

posterior.source
# Provenance('nuts', parents=[MultivariateNormal])

provenance_ancestors(predictive)
# [MCMCApproximateDistribution(...), MultivariateNormal(event_shape=(3,))]
```

### 7. Differentiate through distributions

Since everything is built on JAX, you can compute gradients of distribution operations — useful for sensitivity analysis, MLE, and variational inference:

```python
from probpipe import Normal

sensor = Normal(loc=0.0, scale=0.5)

# Score function: d/d(x) log p(x)
jax.grad(lambda x: log_prob(sensor, x))(0.5)   # Array(-2.0, dtype=float32)
```

### 8. Interoperate with TFP and scipy

The converter registry handles automatic bidirectional conversion between ProbPipe, raw TFP, and scipy.stats distributions:

```python
from probpipe import converter_registry, Normal
import tensorflow_probability.substrates.jax.distributions as tfd
import scipy.stats as ss

# Raw TFP and scipy distributions work directly in workflows
conc_from_tfp = predict_wf(params=tfd.Normal(loc=0.0, scale=1.0))

# Convert ProbPipe distributions to scipy for use with scipy tools
from scipy.stats._distn_infrastructure import rv_frozen
sp = converter_registry.convert(sensor, rv_frozen)
sp.ppf(0.975)  # 0.9800 (95th percentile via scipy)
```

## Next Steps

Explore the [tutorials](tutorials.md) for in-depth coverage of distributions, joint models, conditioning, automatic differentiation, and modular inference pipelines.
