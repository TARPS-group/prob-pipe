# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)

ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

## Philosophy

Scientific discovery and real-world decision-making increasingly depend on complex, end-to-end inferential pipelines that integrate heterogeneous data, fit probabilistic models, propagate uncertainty, and validate predictions. Yet high-quality uncertainty quantification (UQ) is rarely achieved at scale because such pipelines are difficult to construct in a way that is simultaneously flexible, reliable, and scalable. Current workflows are typically assembled in an ad hoc manner, with UQ added only partially -- if at all -- and with limited statistical validation.

ProbPipe aims to overturn this paradigm. Just as probabilistic programming systems (Stan, PyMC, NumPyro) made Bayesian inference in complex models accessible to non-experts, and deep learning frameworks (PyTorch, TensorFlow, JAX) enabled rapid model development across domains, ProbPipe provides general-purpose abstractions for probabilistic *workflows* -- making composability, scalability, and reproducibility the default. Its design is driven by three requirements:

- **Reusable inferential components.** Workflows are expressed in terms of modular, swappable statistical or algorithmic units rather than low-level orchestration primitives. Users can change the likelihood model, swap the inference algorithm, or explore a different prior without restructuring the pipeline.
- **Interoperability with the Python ecosystem.** ProbPipe is designed to work with existing ML, probabilistic, and orchestration libraries. Modules can serve as thin wrappers around other packages, and automatic conversion among distributional representations (parametric distributions, Monte Carlo samples, amortized posteriors) removes a major source of brittleness in current pipelines.
- **End-to-end uncertainty propagation.** Once uncertainty is introduced, it is represented and propagated through all downstream steps. When a workflow node expects a concrete value but receives a distribution, ProbPipe automatically broadcasts over samples -- users write deterministic functions and get UQ for free.
- **Seamless scalability.** The same pipeline scales in two complementary directions without code changes: *computationally*, by vectorizing operations across samples via JAX, and *operationally*, by distributing workflow steps across machines via Prefect orchestration. Both can be active simultaneously, so a pipeline prototyped on a laptop can move to a cluster with no restructuring.
- **Provenance and reproducibility.** Every distribution records how it was created -- operation, parents, parameters -- enabling full lineage tracing from any result back to its inputs.

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

Core dependencies: JAX, NumPy, SciPy, TensorFlow Probability (nightly, [recommended for JAX users](https://github.com/tensorflow/probability/issues/1994#issuecomment-3129033043)).

Optional extras:

```bash
pip install .[dev]       # pytest, jupyter, matplotlib, graphviz
pip install .[prefect]   # Prefect orchestration backend
```

## Quick Start

### Distributions

ProbPipe wraps 20+ TensorFlow Probability distributions with a uniform interface. All distributions follow TFP shape semantics (`sample_shape + batch_shape + event_shape`) and are fully JAX-differentiable.

```python
from probpipe import Normal, MultivariateNormal, TransformedDistribution
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

# Scalar and multivariate distributions
prior = Normal(loc=0.0, scale=1.0)
mvn = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))

# Transform to enforce positivity
positive_prior = TransformedDistribution(prior, tfb.Exp())

# Sample and evaluate
samples = prior.sample(jax.random.PRNGKey(0), (1000,))  # shape: (1000,)
lp = prior.log_prob(0.5)                                # -1.0439...

# Fully differentiable
score = jax.grad(prior.log_prob)(0.5)                    # -0.5
```

### Joint Distributions and Conditioning

Joint distributions compose marginals into a single object. Components can be independent (product), autoregressive (sequential), or jointly Gaussian with exact closed-form conditioning.

```python
from probpipe import Normal, ProductDistribution, SequentialJointDistribution, JointGaussian
import jax.numpy as jnp

# Independent components
joint = ProductDistribution(
    mu=Normal(loc=0.0, scale=1.0),
    sigma=Normal(loc=1.0, scale=0.5),
)

# Autoregressive dependence: x depends on z
seq = SequentialJointDistribution(
    z=Normal(loc=0.0, scale=1.0),
    x=lambda z: Normal(loc=z, scale=0.5),
)

# Exact Gaussian conditioning: p(y | x)
jg = JointGaussian(mean=jnp.zeros(4), cov=jnp.eye(4), x=2, y=2)
posterior = jg.condition_on(x=jnp.array([1.0, 2.0]))
posterior.mean()  # Array([0., 0.], dtype=float32)
```

### Workflows and Broadcasting

When a workflow node receives a distribution where it expects a scalar, ProbPipe automatically draws samples and calls the function once per sample, returning an `EmpiricalDistribution` over the outputs.

```python
from probpipe import Normal, Workflow

def simulate(mu: float, sigma: float) -> float:
    return mu + sigma * 0.1

wf = Workflow(func=simulate)
result = wf(mu=Normal(loc=0.0, scale=1.0), sigma=Normal(loc=1.0, scale=0.1))
# result is an EmpiricalDistribution with 128 samples
result.mean()  # ~0.1
```

### Bayesian Inference

The built-in MCMC sampler supports NUTS, HMC, and random-walk Metropolis-Hastings. Define a likelihood module, pair it with a prior, and sample.

```python
from probpipe import MultivariateNormal, wf
from probpipe.core.modeling import Likelihood, MCMCSampler
import jax.numpy as jnp

class GaussianLikelihood(Likelihood):
    @wf
    def log_likelihood(self, params, data):
        return jnp.sum(-0.5 * (data - params) ** 2)

prior = MultivariateNormal(loc=jnp.zeros(2), cov=25.0 * jnp.eye(2))
sampler = MCMCSampler(algorithm='nuts', num_results=500, num_warmup=200)
data = jnp.array([1.0, 2.0])

posterior = sampler(prior=prior, likelihood=GaussianLikelihood(), data=data)
posterior.mean()          # Array([~1.0, ~2.0], dtype=float32)
posterior.source          # Provenance('nuts', parents=[MultivariateNormal])
```

### Provenance Tracking

Every distribution records its lineage automatically. Provenance chains can be serialized to JSON or traversed programmatically.

```python
from probpipe import Normal, TransformedDistribution, provenance_ancestors
import tensorflow_probability.substrates.jax.bijectors as tfb

base = Normal(loc=0.0, scale=1.0, name='base')
positive = TransformedDistribution(base, tfb.Exp())

positive.source           # Provenance('transform', parents=[base])
positive.source.to_dict() # {'operation': 'transform',
                           #  'parents': [{'type': 'Normal', 'name': 'base'}],
                           #  'metadata': {'bijector': 'Exp'}}
provenance_ancestors(positive)  # [Normal(name='base', event_shape=())]
```

## Example Notebooks

| Notebook | Topic |
|----------|-------|
| [01_distributions](examples/01_distributions.ipynb) | Distribution basics, shape semantics, support checking, conversion |
| [02_transformations](examples/02_transformations.ipynb) | Bijectors, transformed distributions, provenance chains |
| [03_joint_distributions](examples/03_joint_distributions.ipynb) | Joint distributions, conditioning, correlated broadcasting |
| [04_broadcasting](examples/04_broadcasting.ipynb) | Broadcasting vectorization, enumeration, auto-detection |
| [05_autodiff](examples/05_autodiff.ipynb) | JAX autodiff: score functions, sensitivity analysis, MLE, variational inference |
| [06_modular_forecasting](examples/06_modular_forecasting.ipynb) | Modular inference pipeline, swappable likelihoods, posterior predictive checks |

