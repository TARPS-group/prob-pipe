# ProbPipe

ProbPipe is a Python workflow management system for probabilistic modeling and uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

## Philosophy

ProbPipe is a **workflow manager first**. It provides a modular, composable architecture for building probabilistic pipelines using existing new inference frameworks and packages that, by default, are not compatible. The library builds on TensorFlow Probability (TFP) with a JAX backend for distributions and sampling, and provides:

- **Distributions as first-class objects.** All 20+ built-in distribution types (continuous, discrete, multivariate, joint, transformed, empirical) follow TFP shape semantics and are fully JAX-differentiable.
- **Automatic uncertainty propagation.** When a `Workflow` node expects a concrete value but receives a distribution, ProbPipe automatically broadcasts over samples — users write deterministic functions and get uncertainty quantification for free.
- **Seamless scalability.** The same pipeline runs on a laptop via JAX vectorization or scales to a cluster via optional Prefect orchestration. The `"auto"` backend probes JAX traceability and Prefect availability, then picks the fastest execution strategy.
- **Composable modules.** Complex pipelines are built from swappable components, making it easy to change the likelihood model, swap the inference algorithm, or explore a different prior — the pipeline structure stays the same, including sensitivity analyses, predictive checks, and other upstream and downstream steps commonly included in scientific workflows.
- **Provenance tracking.** Every distribution records how it was created (operation, parents, other parameters), which enables full full lineage tracing and improves reproducibility.

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

Core dependencies: JAX, NumPy, SciPy, TensorFlow Probability.

Optional extras:

```bash
pip install .[dev]       # pytest, jupyter, matplotlib, graphviz
pip install .[prefect]   # Prefect orchestration backend
```

## Quick Start

### Distributions

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
samples = prior.sample(jax.random.PRNGKey(0), (1000,))
lp = prior.log_prob(0.5)

# Fully differentiable
score = jax.grad(prior.log_prob)(0.5)
```

### Joint Distributions and Conditioning

```python
from probpipe import ProductDistribution, SequentialJointDistribution, JointGaussian

# Independent components
joint = ProductDistribution(
    mu=Normal(loc=0.0, scale=1.0),
    sigma=Normal(loc=1.0, scale=0.5),
)

# Autoregressive dependence
seq = SequentialJointDistribution(
    z=Normal(loc=0.0, scale=1.0),
    x=lambda z: Normal(loc=z, scale=0.5),
)

# Exact Gaussian conditioning
jg = JointGaussian(mean=jnp.zeros(4), cov=jnp.eye(4), x=2, y=2)
posterior = jg.condition_on(x=jnp.array([1.0, 2.0]))
```

### Workflows and Broadcasting

```python
from probpipe import Normal, wf

@wf
def simulate(mu: float, sigma: float) -> float:
    return mu + sigma * 0.1  # simplified model

# Pass distributions where concrete values are expected —
# ProbPipe broadcasts automatically
result = simulate(mu=Normal(loc=0.0, scale=1.0), sigma=Normal(loc=1.0, scale=0.1))
# result is an EmpiricalDistribution of outputs
```

### Bayesian Inference Pipeline

```python
from probpipe import Normal, MultivariateNormal, wf
from probpipe.core.modeling import Likelihood, MCMCSampler, IterativeForecaster

class GaussianLikelihood(Likelihood):
    @wf
    def log_likelihood(self, params, data):
        return jnp.sum(-0.5 * (data - params) ** 2)

prior = MultivariateNormal(loc=jnp.zeros(2), cov=25.0 * jnp.eye(2))
sampler = MCMCSampler(algorithm='nuts', num_results=1000, num_warmup=500)
posterior = sampler(prior=prior, likelihood=GaussianLikelihood(), data=data)

# Sequential updating
forecaster = IterativeForecaster(prior=prior, likelihood=likelihood, approx_post=sampler)
for batch in data_batches:
    posterior = forecaster.update(data=batch)
```

### Provenance Tracking

```python
from probpipe import provenance_ancestors

# Every distribution records its lineage
print(posterior.source)           # Provenance('nuts', parents=[prior])
print(posterior.source.to_dict()) # JSON-serializable

ancestors = provenance_ancestors(posterior)
# [prior] — full ancestry chain
```

## Subpackages

| Subpackage | Contents |
|------------|----------|
| `probpipe.distributions` | Distribution ABC, 23 TFP-backed distributions, EmpiricalDistribution, TransformedDistribution, joint distributions, constraints |
| `probpipe.core` | Workflow/Module/Node (DAG execution), MCMCSampler, Likelihood, IterativeForecaster |
| `probpipe.provenance` | `provenance_ancestors`, `provenance_dag` (lineage utilities) |

## Example Notebooks

| Notebook | Topic |
|----------|-------|
| [01_distributions](examples/01_distributions.ipynb) | Distribution basics, shape semantics, support checking, conversion |
| [02_transformations](examples/02_transformations.ipynb) | Bijectors, transformed distributions, provenance chains |
| [03_joint_distributions](examples/03_joint_distributions.ipynb) | Joint distributions, conditioning, correlated broadcasting |
| [04_broadcasting](examples/04_broadcasting.ipynb) | Broadcasting backends, enumeration, auto-detection |
| [05_autodiff](examples/05_autodiff.ipynb) | JAX autodiff: score functions, sensitivity analysis, MLE, variational inference |
| [06_modular_forecasting](examples/06_modular_forecasting.ipynb) | Modular inference pipeline, swappable likelihoods, posterior predictive checks |

## License

MIT
