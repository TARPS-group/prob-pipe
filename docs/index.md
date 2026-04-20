ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

## Why ProbPipe?

### The challenge

Most workflows for probabilistic inference -- including validation procedures -- can be described in terms of four **abstract components**:

1. **Distributions** -- priors, posteriors, data products ($\pi, p, q, \nu, \mu, \dots$)
2. **Fixed values** -- data and hyperparameters ($\boldsymbol{X}, \boldsymbol{Y}, \alpha_0, \beta_0, \dots$)
3. **Operations** that transform distributions, possibly depending on fixed values -- conditioning, pushforwards through functions, taking expectations
4. **Differentiation** with respect to fixed values

Using these abstractions, even complex workflows can be written down succinctly. However, *implementing* them requires concrete representations of both distributions and fixed values, plus algorithms for operations -- each creating its own challenge:

- **Algorithmic challenges.** There are usually many possible algorithms for a given operation. A posterior could be approximated using different Markov chain Monte Carlo (MCMC) algorithms, variational inference methods, or sequential Monte Carlo. These are implemented across many packages and often not designed to be directly compatible.
- **Representational challenges.** Algorithms often require or output specific representations for both distributions and fixed values that are not compatible with other parts of the workflow. For example, MCMC outputs a discrete approximation to a distribution, but many MCMC algorithms require continuous representations of prior distributions. Similarly, fixed values may need to be structured as named parameter vectors, flat arrays, or covariate matrices depending on the algorithm — and downstream analysis requires named access to specific components of the result.

### ProbPipe's approach

ProbPipe manages representations and algorithms automatically by default, while giving you control over these choices when you want it:

- **`Record`** is the universal container for named, structured non-random data. `Distribution` is the universal container for random quantities. Both support named fields and `select()` for workflow function splatting.  Together they ensure the full pipeline — from prior specification through inference to posterior predictive checks — produces named, provenance-tracked objects at every step.
- **`Distribution`s** are generic and support subsets of capabilities via `@runtime_checkable` protocols (`SupportsSampling`, `SupportsLogProb`, `SupportsConditioning`, ...). This means external distribution types (TensorFlow Probability (TFP), scipy) can participate without inheriting from ProbPipe base classes.
- **`WorkflowFunction`s** natively handle conversion between distribution representations and automatically compute **pushforward distributions** when functions defined on fixed inputs receive distributions as arguments.  Use `dist.select("x", "y")` to pass named components while preserving correlation.
- **`Module`s** wrap multiple workflow functions with shared state, enabling reusable inferential components.

## Quick Example

Fit a Bayesian model with named parameters, then propagate posterior uncertainty through a prediction function:

```python
import jax
import jax.numpy as jnp
import numpy as np
from probpipe import (
    Normal, ProductDistribution, SimpleModel, workflow_function,
    condition_on, mean, variance,
)
from probpipe.modeling import Likelihood

# 1. Define a model with a named prior. ``params`` arrives flattened
#    in sorted-field order (here ``intercept`` then ``slope``).
class LinearLikelihood(Likelihood):
    def log_likelihood(self, params, data):
        x, y = data[:, 0], data[:, 1]
        intercept, slope = params
        return jnp.sum(-0.5 * (y - (intercept + slope * x)) ** 2)

prior = ProductDistribution(
    intercept=Normal(loc=0.0, scale=3.0, name="intercept"),
    slope=Normal(loc=0.0, scale=3.0, name="slope"),
)
model = SimpleModel(prior, LinearLikelihood())

# Simulated data for the snippet
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (50,))
y = 0.7 + 1.3 * x + 0.2 * jax.random.normal(key, (50,))
data = jnp.column_stack([x, y])

# 2. Fit the posterior — named prior produces named draws
posterior = condition_on(model, data)
draws = posterior.draws()                   # NumericRecordArray(intercept, slope)
draws["intercept"].mean()                   # posterior mean of intercept
draws["slope"].mean()                       # posterior mean of slope

# 3. Propagate uncertainty — select() preserves posterior correlation
@workflow_function
def predict(intercept, slope, x):
    return intercept + slope * x

predictive = predict(**posterior.select("intercept", "slope"), x=0.5)
float(mean(predictive))                     # posterior predictive mean
float(jnp.sqrt(variance(predictive)))       # posterior predictive std
```

A `WorkflowFunction`'s broadcasting rule is simple and uniform: array-valued inputs
(a `RecordArray` or `DistributionArray`) drive a cell-by-cell sweep, multiple array
inputs combine by the product rule, and scalar `Distribution` inputs marginalise via
Monte Carlo. Every return is wrapped at the decorator boundary into the
`Record | RecordArray | Distribution` contract with the function's name as the
single field name — so `mean(d)` returns `NumericRecord(mean=...)`, and numeric
access stays terse via `float(...)`, `jnp.array(...)`, and `.shape` / `.dtype` shims.
Using `select()` ensures that correlated parameters from the same posterior are
sampled jointly.

## Next Steps

- [Getting Started tutorial](tutorials/getting_started.ipynb) -- iterative Bayesian model building following the Bayesian Workflow (Gelman et al., 2020)
- [Reference Notebooks](reference_notebooks.md) -- in-depth coverage of specific ProbPipe features
- [API Reference](api/distributions.md) -- full class and function documentation
