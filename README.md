# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20683559-blue)](https://doi.org/10.5281/zenodo.20683559)

<!-- --8<-- [start:intro] -->
ProbPipe is a Python framework that makes it easy to build probabilistic workflows with automated uncertainty quantification.
Workflows are built from familiar mathematical objects, such as distributions, conditional distributions, and functions, and standard operations such as composition, conditioning, and expectation.
Each ProbPipe object declares its computational capabilities based on its type, such as drawing random samples for a distribution, evaluating the density for a conditional distribution, or inversion for a function.
By default, ProbPipe hides computational details, but it provides access to them when you need it.
<!-- --8<-- [end:intro] -->

## What you can do

<!-- --8<-- [start:capabilities] -->
Carrying out a probabilistic analysis means translating its mathematics into a computational procedure.
In practice you have to choose among algorithms with different trade-offs and convert between the formats that different tools expect.
ProbPipe streamlines and automates those tasks, so you can:

- **Switch and compare algorithms without rewriting the model.** The `condition_on(...)` operation computes a model posterior with an algorithm it selects, based on that model's capabilities (e.g., log density evaluation or forward sampling). Calling `condition_on.with_options(method=method)(...)` swaps in another algorithm named `method`, such as stochastic-gradient Langevin dynamics (for large data sets) or random-walk Metropolis (for a log density without gradients).
- **Use the models and data you already have.** A model written in PyMC or Stan can be wrapped as a ProbPipe object that operations like `condition_on` can be used with. Data can be in pandas or xarray objects.
- **Push uncertainty through any Python function.** Write a forecast or a decision rule in plain Python as a function of known inputs, such as one value of each parameter, and decorate it with `@function` to make it a ProbPipe-native function. Called on a posterior, or on any other distribution of its inputs, the function returns the distribution of its output, which ProbPipe computes by evaluating the function at draws from that distribution.
- **Get exact answers when feasible.** ProbPipe computes a result exactly when a formula or an exact algorithm applies, such as the update to a discrete distribution, and approximately otherwise.
- **Know how each result was computed.** Every result records how it was computed and whether that computation is exact. Moreover, uniform random seed control helps improve reproducibility.
<!-- --8<-- [end:capabilities] -->

## A short example

<!-- --8<-- [start:quick-example] -->
Consider a Bayesian logistic regression for the [Challenger O-ring data](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster), which consists of temperatures of 23 shuttle launches and an indicator as to whether an O-ring was damaged.
In January 1986, the Space Shuttle Challenger broke apart shortly after launch because an O-ring seal in one of its rocket boosters failed during unusually cold weather.
Given the data from the 23 earlier launches, we forecast the probability of O-ring damage at the estimated temperature of 31°F near the rocket booster at the time of that launch.

First, we specify the model and condition it on the data:

```python
import jax.numpy as jnp
import pandas as pd

from probpipe import Normal, condition_on, mean, workflow_run
from probpipe.families import BernoulliFamily, glm_likelihood

# One row per launch: the temperature, and whether an O-ring was damaged.
data = pd.read_csv("docs/tutorials/data/challenger.csv")
temperature = jnp.asarray(data["temperature"], dtype=jnp.float32)
damage = jnp.asarray(data["damage"], dtype=jnp.int32)

# The design matrix: a column of ones for the intercept, and the temperatures for the slope.
X = jnp.stack([jnp.ones_like(temperature), temperature], axis=1)

# The prior: a normal distribution of the coefficients beta = (intercept, slope).
prior = Normal("beta", jnp.zeros(2), jnp.array([10.0, 1.0]))

# The likelihood: the distribution of the 23 damage indicators given beta, a logistic regression.
likelihood = glm_likelihood("damage", BernoulliFamily(), X=X)

# The model is the joint distribution of beta and the damage indicators.
model = likelihood * prior

# Conditioning the model on the observed damage gives the posterior of beta.
# The seed makes the sampler's random draws reproducible.
with workflow_run(seed=0):
    posterior = condition_on(model, {"damage": damage})

print("posterior mean of beta:", mean(posterior["beta"]).raw())

# Every result records how it was computed.
for name, result in [("mean of the prior", mean(prior)), ("posterior", posterior)]:
    prov_meta = result.provenance.metadata
    method = f", method={prov_meta['method']}" if "method" in prov_meta else ""
    print(f"{name} provenance: route={prov_meta['route']}{method}, exact={prov_meta['exact']}")
```

```text
posterior mean of beta: [11.602147   -0.18273503]
mean of the prior provenance: route=closed_form, exact=True
posterior provenance: route=inference_methods, method=blackjax_nuts, exact=False
```

The prior and the likelihood are both ProbPipe objects: a distribution of the coefficients, and a conditional distribution of the data given the coefficients (constructed with `glm_likelihood`).
Multiplying them with `*` constructs the joint model, and `condition_on` computes the posterior.
The output also tracks how ProbPipe computed each result.
The prior's mean has a formula, so `mean` computed it exactly.
The posterior of a logistic regression has none, so `condition_on` generated approximate draws from the posterior using BlackJAX's No-U-Turn Sampler.

We can easily compare many different posterior inference algorithms:

```python
from probpipe.diagnostics import add_mcmc_diagnostics

# Use the same code to get results from any available method.
print(f"{'method':<15}{'slope mean':>12}{'largest R-hat':>16}")
for method in ["blackjax_nuts", "tfp_nuts", "blackjax_rwmh"]:
    with workflow_run(seed=0):
        method_posterior = condition_on.with_options(method=method)(model, {"damage": damage})
    # Adds R-hat and effective sample size diagnostics to the posterior object.
    add_mcmc_diagnostics(method_posterior)
    mean_slope = float(mean(method_posterior["beta"]).raw()[1])
    largest_rhat = max(method_posterior.diagnostics.mcmc.rhat.values())
    print(f"{method:<15}{mean_slope:>12.3f}{largest_rhat:>16.3f}")
```

```text
method           slope mean   largest R-hat
blackjax_nuts        -0.183           1.009
tfp_nuts             -0.186           1.003
blackjax_rwmh        -0.083           1.973
```

BlackJAX and TensorFlow Probability's implementations of the No-U-Turn Sampler give similar inferences about the slope.
Random-walk Metropolis reports a different slope. But it shouldn't be trusted since the largest R-hat is nearly 2, indicating the chains didn't mix.

Next, we illustrate how to forecast using the posterior.
The probability of damage at 31°F is a function of the coefficients, which we write in plain JAX:

```python
import jax

from probpipe import function, quantile


# The probability of damage at 31°F, for one pair of coefficients.
@function
def challenger_damage_probability(beta: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(beta[0] + beta[1] * 31.0)


# Called on the posterior, the function returns the distribution of the probability.
with workflow_run(seed=2):
    damage_prob = challenger_damage_probability(posterior["beta"])

print("P(damage at 31°F), posterior mean:", round(float(mean(damage_prob)), 3))
print("P(damage at 31°F), 90% interval:", quantile(damage_prob, jnp.array([0.05, 0.95])).raw())

# The forecast records how it was computed.
prov_meta = damage_prob.provenance.metadata
print(f"forecast provenance: route={prov_meta['route']}, exact={prov_meta['exact']}")
```

```text
P(damage at 31°F), posterior mean: 0.964
P(damage at 31°F), 90% interval: [0.8174724  0.99998176]
forecast provenance: route=sampling_lift, exact=False
```

We wrote `challenger_damage_probability` for one pair of coefficients, and ProbPipe automatically lifted it to operate on the posterior distribution: it evaluated the function at draws from the posterior distribution, and returned the distribution of the results.
Its provenance shows it was generated using this sampling approach and that it is not exact.
<!-- --8<-- [end:quick-example] -->

## Installation

ProbPipe requires Python 3.12 or later. It is not yet on PyPI, so it installs from GitHub, and until a release the documentation describes the `dev/overhaul` branch:

```bash
pip install "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

The [installation page](https://tarps-group.github.io/prob-pipe/get_started/installation/) gives the commands for Google Colab and uv, and the extras that add backends such as Stan and PyMC.

## Learn more

- [The quickstart](https://tarps-group.github.io/prob-pipe/get_started/quickstart/) works through this example in ten minutes, with diagnostics and a predictive check.
- [The tutorials](https://tarps-group.github.io/prob-pipe/tutorials/01_first_analysis/) follow one analysis of a moose population, from a first model to forecasts that update as each count arrives.
- [The API reference](https://tarps-group.github.io/prob-pipe/api/) documents every public name.
- [How to cite ProbPipe](https://tarps-group.github.io/prob-pipe/cite/), [getting help](https://tarps-group.github.io/prob-pipe/help/), and [contributing](CONTRIBUTING.md).

> **Human-validated** by Jonathan Huggins on 2026-10-04.
