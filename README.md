> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20683559-blue)](https://doi.org/10.5281/zenodo.20683559)

<!-- --8<-- [start:intro] -->
ProbPipe is a Python framework that makes it easy to build probabilistic pipelines with automated uncertainty quantification.
Pipelines are built from the usual mathematical objects, such as distributions, conditional distributions, and functions, and the standard mathematical operations, such as composition, conditioning, and expectation.
Each ProbPipe object declares the computational capabilities of its type, such as drawing random samples from a distribution, evaluating the density of a conditional distribution, or inverting a function.
By default, ProbPipe hides computational complexity, but it provides access to the computational details when you need them.
<!-- --8<-- [end:intro] -->

## What you can do

<!-- --8<-- [start:capabilities] -->
Carrying out a probabilistic analysis means turning its mathematics into computation: choosing among algorithms with different trade-offs, and converting between the formats that different tools expect.
ProbPipe takes on that work, so you can:

- **Switch and compare algorithms without rewriting the model.** `condition_on` fits a model with an algorithm it selects, and one option, `with_options(method=...)`, swaps in another, such as TensorFlow Probability's NUTS or random-walk Metropolis. Comparing algorithms takes a loop, not a rewrite.
- **Use the models and data you already have.** A model written in PyMC or Stan becomes a ProbPipe object that the same calls fit, data can arrive in pandas or xarray objects, and a sampler's draws come with ArviZ data for diagnostics and plots.
- **Push uncertainty through any Python function.** A forecast or a decision rule written for fixed inputs, applied to a posterior, returns the distribution of its output.
- **Get exact answers where they exist.** ProbPipe computes a result exactly when a formula or an exact algorithm applies, such as the update of a posterior held as weighted draws, and approximately otherwise.
- **Know how each result was computed.** Every result records how it was computed and whether that computation is exact, and a seed reproduces its random draws.
<!-- --8<-- [end:capabilities] -->

## How it works

<!-- --8<-- [start:approach] -->
ProbPipe keeps the mathematics separate from the computation.
A model states only the mathematics, and each operation chooses how to compute its result, so changing the algorithm never changes the model.
Six ideas make this work:

1. **Mathematical objects:** distributions, conditional distributions, functions, and values are ProbPipe objects, and each one names its components, such as the coefficients and the response of a regression. A batch holds several objects of one kind, such as a set of scenarios, on named axes.
2. **One vocabulary of operations:** `*` composes a conditional distribution with a distribution into their joint distribution, `condition_on` conditions, and summaries such as `mean` and `quantile` describe a law. Each operation applies to every object that supports it mathematically and returns another ProbPipe object, so results compose.
3. **Computation from capabilities:** an operation computes its result from what its inputs can do, by a closed form where one exists, and otherwise by an exact algorithm or an approximate method from a registry of backends such as BlackJAX, Stan, and PyMC. The choice is automatic, `check` reports it before a call runs, and `with_options` overrides it.
4. **Lifting:** an ordinary Python function applied to distributions returns the distribution of its output, and applied to a batch it returns the batch of its outputs.
5. **Traceable, reproducible results:** every result records the operation, the route, and the inputs that produced it, and `workflow_run(seed=...)` makes its random draws reproducible.
6. **Native Python and existing packages:** functions are plain Python, typically JAX code, and data can arrive in pandas or xarray objects. A model written in PyMC or Stan becomes a ProbPipe object, and a sampler's draws are available as ArviZ data.
<!-- --8<-- [end:approach] -->

## Quick example

<!-- --8<-- [start:quick-example] -->
A Bayesian logistic regression on the [Challenger O-ring data](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster) relates the temperature of 23 shuttle launches to whether an O-ring was damaged.
Its posterior then gives the probability of damage at 31°F, far below the 53°F of the coldest of those launches.

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

The prior and the likelihood are both ProbPipe objects: a distribution of the coefficients, and a conditional distribution of the data given the coefficients, which `glm_likelihood` builds for us.
Multiplying them with `*` gives the model, and `condition_on` turns the model into the posterior.
The output also shows how ProbPipe computed each result.
The prior's mean has a formula, so `mean` computed it exactly.
The posterior of a logistic regression has none, so `condition_on` drew from it with BlackJAX's No-U-Turn Sampler, and the posterior is approximate: it is held as the sampler's draws.

The model doesn't name an algorithm, so we can switch it.
Here we fit the same model with three algorithms and compare the results:

```python
from probpipe.diagnostics import add_mcmc_diagnostics

# The same model and data, conditioned with three algorithms.
for method in ["blackjax_nuts", "tfp_nuts", "blackjax_rwmh"]:
    with workflow_run(seed=0):
        fit = condition_on.with_options(method=method)(model, {"damage": damage})
    add_mcmc_diagnostics(fit)  # R-hat and effective sample sizes, recorded on the fit
    slope = float(mean(fit["beta"]).raw()[1])
    largest_rhat = max(fit.diagnostics.mcmc.rhat.values())
    print(f"{method}: slope={slope:.3f}, largest R-hat={largest_rhat:.2f}")
```

```text
blackjax_nuts: slope=-0.183, largest R-hat=1.01
tfp_nuts: slope=-0.186, largest R-hat=1.00
blackjax_rwmh: slope=-0.083, largest R-hat=1.97
```

BlackJAX's and TensorFlow Probability's implementations of the No-U-Turn Sampler agree on the slope.
Random-walk Metropolis reports a different slope, and its R-hat near 2 says why: its chains never mixed, so we shouldn't trust its answer.
Only `method` changed between the three fits, because the model states what to compute and ProbPipe decides how.

Next, we forecast.
The probability of damage at 31°F is a function of the coefficients, which we write in plain JAX:

```python
import jax

from probpipe import function, quantile


# The probability of damage at 31°F, for one pair of coefficients.
@function
def damage_probability(beta: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(beta[0] + beta[1] * 31.0)


# Called on the posterior, the function returns the distribution of the probability.
with workflow_run(seed=1):
    risk = damage_probability(posterior["beta"])

print("P(damage at 31°F), posterior mean:", round(float(mean(risk)), 3))
print("P(damage at 31°F), 90% interval:", quantile(risk, jnp.array([0.05, 0.95])).raw())

# The forecast records how it was computed, too.
prov_meta = risk.provenance.metadata
print(f"risk provenance: route={prov_meta['route']}, exact={prov_meta['exact']}")
```

```text
P(damage at 31°F), posterior mean: 0.96
P(damage at 31°F), 90% interval: [0.77571553 0.9999759 ]
risk provenance: route=sampling_lift, exact=False
```

We wrote `damage_probability` for one pair of coefficients, and ProbPipe lifted it to the posterior: it evaluated the function at draws of the coefficients and returned the distribution of the results, so the forecast carries the posterior's uncertainty.
Its provenance says it came from those draws, so it too is approximate, and running the code again with the same seeds reproduces it.
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
