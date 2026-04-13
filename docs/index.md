ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

## Why ProbPipe?

### The challenge

Most workflows for probabilistic inference -- including validation procedures -- can be described in terms of four **abstract components**:

1. **Distributions** -- priors, posteriors, data products ($\pi, p, q, \nu, \mu, \dots$)
2. **Fixed inputs** -- data and hyperparameters ($\boldsymbol{X}, \boldsymbol{Y}, \alpha_0, \beta_0, \dots$)
3. **Operations** that transform distributions, possibly depending on fixed inputs -- conditioning, pushforwards through functions, taking expectations
4. **Differentiation** with respect to fixed inputs

Using these abstractions, even complex workflows can be written down succinctly. However, *implementing* them requires concrete representations of distributions and algorithms for operations -- each creating its own challenge:

- **Algorithmic challenges.** There are usually many possible algorithms for a given operation. A posterior could be approximated using different Markov chain Monte Carlo (MCMC) algorithms, variational inference methods, or sequential Monte Carlo. These are implemented across many packages and often not designed to be directly compatible.
- **Representational challenges.** Algorithms often require or output specific distribution representations that are not compatible with other parts of the workflow. For example, MCMC outputs a discrete approximation to a distribution, but many MCMC algorithms require continuous representations of prior distributions.

### ProbPipe's approach

ProbPipe manages representations and algorithms automatically by default, while giving you control over these choices when you want it:

- **`Values`** is the universal container for named, structured non-random data. `Distribution` is the universal container for random quantities. Both support named fields and `select()` for workflow function splatting.  Together they ensure the full pipeline — from prior specification through inference to posterior predictive checks — produces named, provenance-tracked objects at every step.
- **`Distribution`s** are generic and support subsets of capabilities via `@runtime_checkable` protocols (`SupportsSampling`, `SupportsLogProb`, `SupportsConditioning`, ...). This means external distribution types (TensorFlow Probability (TFP), scipy) can participate without inheriting from ProbPipe base classes.
- **`WorkflowFunction`s** natively handle conversion between distribution representations and automatically compute **pushforward distributions** when functions defined on fixed inputs receive distributions as arguments.  Use `dist.select("x", "y")` to pass named components while preserving correlation.
- **`Module`s** wrap multiple workflow functions with shared state, enabling reusable inferential components.

## Quick Example

Fit a Bayesian model with named parameters, then propagate posterior uncertainty through a prediction function:

```python
import jax
import jax.numpy as jnp
from probpipe import (
    MultivariateNormal, SimpleModel, workflow_function,
    condition_on, mean, variance,
)

# 1. Define a model with a named prior
class LinearLikelihood:
    def log_likelihood(self, params, data):
        x, y = data[:, 0], data[:, 1]
        return jnp.sum(-0.5 * (y - (params[0] + params[1] * x)) ** 2)

prior = MultivariateNormal(loc=jnp.zeros(2), cov=10.0 * jnp.eye(2), name="beta")
model = SimpleModel(prior, LinearLikelihood())

# 2. Fit the posterior — named prior produces named draws
posterior = condition_on(model, data)
draws = posterior.draws()       # Values(beta=array(num_draws, 2))
draws.beta.mean(axis=0)         # posterior mean for [intercept, slope]

# 3. Propagate uncertainty — select() preserves posterior correlation
@workflow_function
def predict(beta, x):
    return beta[0] + beta[1] * x

predictive = predict(**posterior.select("beta"), x=0.5)
mean(predictive)       # posterior predictive mean
variance(predictive)   # posterior predictive variance
```

When a `WorkflowFunction` receives a distribution where it expects a concrete value, it automatically broadcasts over samples and returns the output distribution.  Using `select()` ensures that correlated parameters from the same posterior are sampled jointly.

## Next Steps

- [Getting Started tutorial](tutorials/getting_started.ipynb) -- iterative Bayesian model building following the Bayesian Workflow (Gelman et al., 2020)
- [Reference Notebooks](reference_notebooks.md) -- in-depth coverage of specific ProbPipe features
- [API Reference](api/distributions.md) -- full class and function documentation
