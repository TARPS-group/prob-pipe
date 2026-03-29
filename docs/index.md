ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

## Why ProbPipe?

Scientific discovery and real-world decision-making increasingly depend on complex, end-to-end inferential pipelines that integrate heterogeneous data, fit probabilistic models, propagate uncertainty, and validate predictions. Yet high-quality uncertainty quantification (UQ) is rarely achieved at scale because such pipelines are difficult to construct in a way that is simultaneously flexible, reliable, and scalable.

ProbPipe provides general-purpose abstractions for probabilistic *workflows* -- making composability, scalability, and reproducibility the default. Its design is driven by five requirements:

- **Reusable inferential components.** Workflows are expressed in terms of modular, swappable statistical units rather than low-level orchestration primitives.
- **Interoperability with the Python ecosystem.** Modules can wrap existing ML and probabilistic libraries, and automatic conversion among distributional representations removes brittleness.
- **End-to-end uncertainty propagation.** When a workflow node expects a concrete value but receives a distribution, ProbPipe automatically broadcasts over samples -- users write deterministic functions and get UQ for free.
- **Seamless scalability.** The same pipeline scales computationally (JAX vectorization) and operationally (Prefect orchestration) without code changes.
- **Provenance and reproducibility.** Every distribution records how it was created, enabling full lineage tracing from any result back to its inputs.

## Quick Example

Fit a Bayesian model, then propagate posterior uncertainty through a prediction function:

```python
import jax
import jax.numpy as jnp
from probpipe import (
    MultivariateNormal, SimpleModel, WorkflowFunction,
    condition_on, mean, variance, sample,
)
from probpipe.modeling import Likelihood
from probpipe.core.node import wf

# 1. Define a model
class LinearLikelihood(Likelihood):
    @wf
    def log_likelihood(self, params, data):
        x, y = data[:, 0], data[:, 1]
        return jnp.sum(-0.5 * (y - (params[0] + params[1] * x)) ** 2)

prior = MultivariateNormal(loc=jnp.zeros(2), cov=10.0 * jnp.eye(2))
model = SimpleModel(prior, LinearLikelihood())

# 2. Fit the posterior
data = jnp.column_stack([jnp.linspace(0, 1, 20), ...])  # observed (x, y) pairs
posterior = condition_on(model, data)
mean(posterior)       # ≈ [intercept, slope]

# 3. Propagate uncertainty through predictions
predict_wf = WorkflowFunction(func=lambda params, x: params[0] + params[1] * x)
predictive = predict_wf(params=posterior, x=0.5)

mean(predictive)       # point prediction at x=0.5
variance(predictive)   # posterior predictive uncertainty
```

The result is an `EmpiricalDistribution` -- a first-class distribution object that can be passed into downstream workflow nodes, triggering further uncertainty propagation.

## Next Steps

- [Getting Started](getting-started.md) -- installation and a complete posterior inference walkthrough
- [Tutorials](tutorials.md) -- guided notebooks covering distributions, transforms, joint models, and more
- [API Reference](api/distributions.md) -- full class and function documentation
