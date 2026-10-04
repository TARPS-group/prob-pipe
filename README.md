> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20683559-blue)](https://doi.org/10.5281/zenodo.20683559)

<!-- --8<-- [start:intro] -->
ProbPipe is a Python framework for probabilistic pipelines with automated uncertainty quantification, in which the objects of probability theory are the objects of the program.
A model is written as it is on paper, as distributions, conditional distributions, and functions, and mathematical operations such as composition, conditioning, and expectation combine them.
Each object declares what it can compute, such as draws, a density, or exact moments, and each operation uses those capabilities: it computes its result exactly where it can, and otherwise it selects an approximate method, such as Markov chain Monte Carlo.
Every result is another ProbPipe object, which records how it was computed.
<!-- --8<-- [end:intro] -->

## The approach

<!-- --8<-- [start:approach] -->
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
    method = f", method {prov_meta['method']}" if "method" in prov_meta else ""
    print(f"{name} provenance: route={prov_meta['route']}{method}, exact={prov_meta['exact']}")
```

```text
posterior mean of beta: [11.602147   -0.18273503]
mean of the prior provenance: route=closed_form, exact=True
posterior provenance: route=inference_methods, method blackjax_nuts, exact=False
```

The prior and the likelihood are both ProbPipe objects: a distribution of the coefficients, and a conditional distribution of the data given the coefficients, which `glm_likelihood` builds for us.
Multiplying them with `*` gives the model, and `condition_on` turns the model into the posterior.
The output also shows how ProbPipe computed each result.
The prior's mean has a formula, so `mean` computed it exactly.
The posterior of a logistic regression has none, so `condition_on` drew from it with BlackJAX's No-U-Turn Sampler, and the posterior is approximate: it is held as the sampler's draws.

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
