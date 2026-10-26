# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20683559-blue)](https://doi.org/10.5281/zenodo.20683559)

<!-- --8<-- [start:intro] -->
ProbPipe is a Python framework that enables you to connect the tools you already use into a single, coherent probabilistic workflow, from data preparation through prediction and decision-making.
ProbPipe makes workflows maintainable and reliable by keeping the mathematics separate from the computational implementation.
You express the workflow in mathematical terms: distributions, conditional distributions, and functions, combined by operations such as conditioning, expectation, and quantiles.
Each mathematical object declares what it can do, such as drawing samples or evaluating a density, and each operation uses those capabilities to choose how to compute its result.
You can specify a different algorithm when you want to, and your own Python functions, simulators, and PyMC, Stan, or TFP models work without a rewrite.
Every result records how it was computed, and random number seeds are managed automatically, so workflows are traceable and reproducible.
<!-- --8<-- [end:intro] -->

## What you can do

<!-- --8<-- [start:capabilities] -->
Carrying out a probabilistic analysis means translating its mathematics into a computational procedure.
In practice you have to choose among algorithms with different trade-offs and convert between the formats that different tools expect.
ProbPipe streamlines and automates those tasks, so you can:

- **Switch and compare algorithms without rewriting the model.** The `condition_on(...)` operation computes a model posterior with an algorithm it selects, based on that model's capabilities (e.g., log density evaluation or forward sampling). Calling `condition_on.using(method=method)(...)` swaps in another algorithm named `method`, such as stochastic-gradient Langevin dynamics (for large data sets) or random-walk Metropolis (for a log density without gradients).
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
prior = Normal("beta", jnp.zeros(2), jnp.array([10.0, 1.0]), label="prior")

# The likelihood: the distribution of the 23 damage indicators given beta, a logistic regression.
likelihood = glm_likelihood("damage", BernoulliFamily(), X=X, label="likelihood")

# The model is the joint distribution of beta and the damage indicators.
model = likelihood & prior

# Conditioning the model on the observed damage gives the posterior of beta.
# The seed makes the sampler's random draws reproducible.
with workflow_run(seed=0):
    posterior = condition_on(model, {"damage": damage})

print("posterior mean of beta:", mean(posterior["beta"]).raw)

# Every result records how it was computed.
print("provenance of the prior's mean:")
print(mean(prior).provenance)
print("provenance of the posterior:")
print(posterior.provenance)
```

```text
posterior mean of beta: [11.66378    -0.18358812]
provenance of the prior's mean:
1. mean(prior, given)  route closed_form, exact
provenance of the posterior:
1. condition_on(likelihood·prior, given)  route inference_methods, method blackjax_nuts, approximate
```

The prior and the likelihood are both ProbPipe objects: a distribution of the coefficients, and a conditional distribution of the data given the coefficients (constructed with `glm_likelihood`).
Merging them with `&` constructs the joint model, labeled `likelihood·prior`, and `condition_on` computes the posterior.
Printing a result's provenance lists each step that produced it, with how ProbPipe computed the step.
The prior's mean has a formula, so `mean` computed it exactly.
The posterior of a logistic regression has none, so `condition_on` generated approximate draws from the posterior using BlackJAX's No-U-Turn Sampler.

We can easily compare many different posterior inference algorithms:

```python
from probpipe.diagnostics import add_mcmc_diagnostics

# Use the same code to get results from any available method.
# Each method is a different algorithm, so the fits run one at a time.
comparison = {}
for method in ["blackjax_nuts", "tfp_nuts", "blackjax_rwmh"]:
    with workflow_run(seed=0):
        method_posterior = condition_on.using(method=method)(model, {"damage": damage})
    # Adds R-hat and effective sample size diagnostics to the posterior object.
    add_mcmc_diagnostics(method_posterior)
    comparison[method] = {
        "slope mean": float(mean(method_posterior["beta"]).raw[1]),
        "largest R-hat": max(method_posterior.diagnostics.mcmc.rhat.values()),
    }
print(pd.DataFrame(comparison).T.round(3))
```

```text
               slope mean  largest R-hat
blackjax_nuts      -0.184          1.005
tfp_nuts           -0.182          1.008
blackjax_rwmh      -0.181          1.130
```

BlackJAX and TensorFlow Probability's implementations of the No-U-Turn Sampler give similar inferences about the slope.
Random-walk Metropolis gives a similar slope, but its largest R-hat of 1.13 is well above the usual threshold of 1.01, so its chains have not mixed and its draws need more iterations before they can be trusted.

Next, we illustrate how to forecast using the posterior.
The probability of damage at 31°F is a function of the coefficients, which we write in plain JAX:

```python
import jax

from probpipe import function, quantile


# The probability of damage at 31°F, for one pair of coefficients.
@function
def challenger_damage_probability(beta: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(beta[0] + beta[1] * 31.0)


# Called on the posterior, which is a distribution, the function returns the
# distribution of the probability.
with workflow_run(seed=2):
    damage_prob = challenger_damage_probability(posterior["beta"])

print("P(damage at 31°F), posterior mean:", round(float(mean(damage_prob)), 3))
print("P(damage at 31°F), 90% interval:", quantile(damage_prob, jnp.array([0.05, 0.95])).raw)

# The forecast records each step from the posterior to it.
print("provenance of the forecast:")
print(damage_prob.provenance)
```

```text
P(damage at 31°F), posterior mean: 0.974
P(damage at 31°F), 90% interval: [0.86161655 0.999984  ]
provenance of the forecast:
1. likelihood·prior = condition_on(likelihood·prior, given)  route inference_methods, method blackjax_nuts, approximate
2. broadcast(challenger_damage_probability, 1['beta'])  route sampling_lift, approximate
```

We wrote `challenger_damage_probability` for one pair of coefficients, and ProbPipe automatically lifted it to operate on the posterior distribution: it evaluated the function at draws from the posterior distribution, and returned the distribution of the results.
Its provenance lists both steps: the fit, which gives approximate draws, and the function evaluated at those draws (`sampling_lift`), so the forecast is approximate too.
<!-- --8<-- [end:quick-example] -->

## Installation

ProbPipe requires Python 3.12 or later. It is not yet on PyPI, so it installs from GitHub, and until a release the documentation describes the `main` branch:

```bash
pip install "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@main"
```

The [installation page](https://tarps-group.github.io/prob-pipe/get_started/installation/) provides guidance for installing ProbPipe directly using uv or with Google Colab.

## Learn more

- [The quickstart](https://tarps-group.github.io/prob-pipe/get_started/quickstart/) works through this example in more detail, including demonstrating how to use built-in diagnostics and perform a predictive check.
- [The design assistant](https://tarps-group.github.io/prob-pipe/get_started/design_assistant/) is a set of instructions for an AI coding agent, which designs and builds a ProbPipe pipeline for your own problem and data with you.
- [The API reference](https://tarps-group.github.io/prob-pipe/api/) provides complete documentation of ProbPipe capabilities.
- [How to cite ProbPipe](https://tarps-group.github.io/prob-pipe/cite/), [getting help](https://tarps-group.github.io/prob-pipe/help/), and [contributing](CONTRIBUTING.md).

> **Human-validated** by Jonathan Huggins on 2026-10-25.
