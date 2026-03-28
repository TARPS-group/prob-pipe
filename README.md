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
- **Provenance tracking** recording full lineage from any result back to its inputs
- **Prefect orchestration** for distributing pipeline steps across machines

## Philosophy

Scientific discovery and real-world decision-making increasingly depend on complex, end-to-end inferential pipelines that integrate heterogeneous data, fit probabilistic models, propagate uncertainty, and validate predictions. Yet high-quality uncertainty quantification (UQ) is rarely achieved at scale because such pipelines are difficult to construct in a way that is simultaneously flexible, reliable, and scalable. Current workflows are typically assembled in an ad hoc manner, with UQ added only partially – if at all – and with limited statistical validation.

ProbPipe aims to overturn this paradigm. Just as probabilistic programming systems (Stan, PyMC, NumPyro) made Bayesian inference in complex models accessible to non-experts, deep learning frameworks (PyTorch, TensorFlow) enabled rapid model development across domains, and automatic differentiation systems (JAX, Autograd) made gradient-based inference practical at scale, ProbPipe provides general-purpose abstractions for probabilistic **workflows** – making composability, scalability, and reproducibility the default. Its design is driven by five principles:

1. **Reusable inferential components.** Workflows are expressed in terms of modular, swappable statistical or algorithmic units rather than low-level orchestration primitives. Users can change the likelihood model, swap the inference algorithm, or explore a different prior without restructuring the pipeline.
2. **Interoperability with the Python ecosystem.** ProbPipe is designed to work with existing ML, probabilistic, and orchestration libraries. Modules can serve as thin wrappers around other packages, and automatic conversion among distributional representations (parametric distributions, Monte Carlo samples, amortized posteriors) removes a major source of brittleness in current pipelines.
3. **End-to-end uncertainty propagation.** Once uncertainty is introduced, it is represented and propagated through all downstream steps. When a workflow node expects a concrete value but receives a distribution, ProbPipe automatically broadcasts over samples – users write deterministic functions and get UQ for free.
4. **Seamless scalability.** The same pipeline scales in two complementary directions without code changes: *computationally*, by vectorizing operations across samples via JAX, and *operationally*, by distributing workflow steps across machines via Prefect orchestration. Both can be active simultaneously, so a pipeline prototyped on a laptop can move to a cluster with no restructuring.
5. **Provenance and reproducibility.** Every distribution records how it was created – operation, parents, parameters – enabling full lineage tracing from any result back to its inputs.

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
pip install .[stan]      # Stan models via BridgeStan
pip install .[pymc]      # PyMC model integration
pip install .[nutpie]    # nutpie MCMC sampler
```

## Quick Start

### Distributions

ProbPipe wraps 20+ TensorFlow Probability distributions with a uniform interface. All distributions follow TFP shape semantics (`sample_shape + batch_shape + event_shape`) and are fully JAX-differentiable. Bijector-based transforms create constrained distributions (e.g., log-normal for positive quantities).

```python
from probpipe import Normal, TransformedDistribution, sample, log_prob
import jax
import tensorflow_probability.substrates.jax.bijectors as tfb

# Model measurement uncertainty: sensor reads temperature with noise
sensor_noise = Normal(loc=0.0, scale=0.5, name='sensor_noise')

# Log-normal model for positive-valued quantities (e.g., concentrations)
log_conc = Normal(loc=2.0, scale=0.3, name='log_concentration')
concentration = TransformedDistribution(log_conc, tfb.Exp())

# Sample and evaluate
samples = sample(sensor_noise, key=jax.random.PRNGKey(0), sample_shape=(1000,))
samples.shape                  # (1000,)
log_prob(sensor_noise, 0.5)    # Array(-0.7257913, dtype=float32)

# Fully differentiable: compute the score function
jax.grad(lambda x: log_prob(sensor_noise, x))(0.5)  # Array(-2.0, dtype=float32)
```

### Workflows and Broadcasting

When a workflow node receives a distribution where it expects a concrete value, ProbPipe automatically draws samples and evaluates the function for each, returning an `EmpiricalDistribution`. Here, uncertain velocity and time propagate through a distance calculation.

```python
from probpipe import Normal, TransformedDistribution, WorkflowFunction, mean, variance, expectation
import tensorflow_probability.substrates.jax.bijectors as tfb

# distance = velocity * time
def compute_distance(velocity, time):
    return velocity * time

wf = WorkflowFunction(func=compute_distance)

# Log-normal time ensures non-negative values
log_time = Normal(loc=1.6, scale=0.04)          # log-seconds
time = TransformedDistribution(log_time, tfb.Exp())  # ~ 5.0 seconds

result = wf(
    velocity=Normal(loc=10.0, scale=1.0),  # m/s, uncertain
    time=time,                             # seconds, non-negative
)
# result is an EmpiricalDistribution with 128 samples
mean(result)       # Array(49.875435, dtype=float32)
variance(result)   # Array(26.283531, dtype=float32)

# Compute expectations with automatic error tracking
# On the EmpiricalDistribution, this is exact (weighted sum over samples)
expectation(result, lambda x: x)  # Array(49.875435, dtype=float32)

# On a parametric distribution, Monte Carlo returns a BootstrapDistribution
# that captures the sampling error
velocity = Normal(loc=10.0, scale=1.0)
ex = expectation(velocity, lambda x: x**2, num_evaluations=5000)
mean(ex)       # ~101.0  -- point estimate of E[V²] = loc² + scale² = 101
variance(ex)   # ~0.08   -- MC error variance (decreases with more evaluations)
```

### Joint Distributions and Conditioning

Build hierarchical models with autoregressive dependence, or fuse multiple noisy observations via exact Gaussian conditioning.

```python
from probpipe import Normal, SequentialJointDistribution, JointGaussian, condition_on, mean
import jax.numpy as jnp

# Hierarchical model: population mean height -> individual measurement
hierarchical = SequentialJointDistribution(
    population_mean=Normal(loc=170.0, scale=10.0),
    measurement=lambda population_mean: Normal(loc=population_mean, scale=5.0),
)
hierarchical.event_shape       # (2,)
hierarchical.component_names   # ('population_mean', 'measurement')

# Sensor fusion: two noisy sensors measure the same latent signal
# signal has var=1; sensor1 adds noise var=1; sensor2 adds noise var=4
cov = jnp.array([
    [1.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 5.0],
])
jg = JointGaussian(mean=jnp.zeros(3), cov=cov, signal=1, sensors=2)

# Observe sensors read [2.5, 3.0] -> infer the latent signal
posterior = condition_on(jg, sensors=jnp.array([2.5, 3.0]))
mean(posterior)         # Array([1.4444445], dtype=float32)
posterior.covariance    # Array([[0.44444442]], dtype=float32)
```

### Bayesian Inference

Bayesian linear regression in a few lines. Define a likelihood, pair it with a prior, build a `SimpleModel`, and condition on data. The model automatically selects NUTS when the likelihood is JAX-traceable and falls back to gradient-free Metropolis-Hastings otherwise.

```python
import jax
import jax.numpy as jnp
from probpipe import MultivariateNormal, SimpleModel, condition_on, mean, wf
from probpipe.core.modeling import Likelihood

class LinearRegressionLikelihood(Likelihood):
    @wf
    def log_likelihood(self, params, data):
        x, y = data[:, 0], data[:, 1]
        predicted = params[0] + params[1] * x       # y = w0 + w1*x
        return jnp.sum(-0.5 * (y - predicted) ** 2)

# Synthetic data: y = 1.0 + 2.0*x + noise
key = jax.random.PRNGKey(42)
x = jnp.linspace(0, 1, 20)
y = 1.0 + 2.0 * x + 0.3 * jax.random.normal(key, shape=(20,))
data = jnp.column_stack([x, y])

prior = MultivariateNormal(loc=jnp.zeros(2), cov=10.0 * jnp.eye(2))
model = SimpleModel(prior, LinearRegressionLikelihood())
posterior = condition_on(model, data, num_results=500, num_warmup=200, random_seed=0)
mean(posterior)     # Array([1.09, 1.99], dtype=float32)  ≈ [1.0, 2.0]
posterior.source    # Provenance('nuts', parents=[SimpleModel])
```

### Provenance Tracking

Every distribution records its lineage automatically. Provenance chains can be serialized to JSON or traversed programmatically, enabling full reproducibility.

```python
from probpipe import Normal, TransformedDistribution, provenance_ancestors
import tensorflow_probability.substrates.jax.bijectors as tfb

# Model a positive rate parameter via log-transform
log_rate = Normal(loc=1.0, scale=0.5, name='log_rate')
rate = TransformedDistribution(log_rate, tfb.Exp())

rate.source
# Provenance('transform', parents=[log_rate])

rate.source.to_dict()
# {'operation': 'transform',
#  'parents': [{'type': 'Normal', 'name': 'log_rate'}],
#  'metadata': {'bijector': 'Exp'}}

provenance_ancestors(rate)
# [Normal(name='log_rate', event_shape=())]
```

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
