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
1. **Mathematical objects:** distributions, conditional distributions, functions, and values are ProbPipe objects, and each one names its components, such as the intercept and the slope of a regression.
2. **One vocabulary of operations:** `*` composes a conditional distribution with a distribution into their joint distribution, `condition_on` conditions, and summaries such as `mean` and `quantile` describe a law. Each operation applies to every object that supports it mathematically and returns another ProbPipe object, so results compose.
3. **Computation from capabilities:** an operation computes its result from what its inputs can do, by a closed form where one exists, and otherwise by an exact algorithm or an approximate method from a registry of backends such as BlackJAX, Stan, and PyMC. The choice is automatic, `check` reports it before a call runs, and `with_options` overrides it.
4. **Lifting:** an ordinary Python function applied to distributions returns the distribution of its output.
5. **Traceable, reproducible results:** every result records the operation, the route, and the inputs that produced it, and `workflow_run(seed=...)` makes its random draws reproducible.
<!-- --8<-- [end:approach] -->

## Quick example

<!-- --8<-- [start:quick-example] -->
A Bayesian logistic regression on the [Challenger O-ring data](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster) relates the temperature of 23 shuttle launches to whether an O-ring was damaged.
Its posterior then gives the probability of damage at 31°F, far below the 53°F of the coldest of those launches.
The example uses each idea of the approach.

```python
import jax
import jax.numpy as jnp
import pandas as pd

from probpipe import (
    Bernoulli,
    Normal,
    NumericArraySpec,
    condition_on,
    conditional_distribution,
    function,
    mean,
    quantile,
    workflow_run,
)

data = pd.read_csv("docs/tutorials/data/challenger.csv")
temperature = jnp.asarray(data["temperature"], dtype=jnp.float32)
damage = jnp.asarray(data["damage"], dtype=jnp.int32)

# A distribution of the intercept and the slope of the log-odds of damage.
prior = Normal("intercept", 0.0, 10.0) * Normal("slope", 0.0, 1.0)

# A conditional distribution of the 23 damage indicators, given the intercept and the slope.
likelihood = conditional_distribution(
    "damage",
    lambda intercept, slope: Bernoulli("damage", logits=intercept + slope * temperature),
    given_spec={"intercept": NumericArraySpec(()), "slope": NumericArraySpec(())},
)

# Their composition is the joint distribution of the parameters and the data.
model = likelihood * prior


# An ordinary function of two numbers.
@function
def damage_probability(intercept, slope):
    return jax.nn.sigmoid(intercept + slope * 31.0)


with workflow_run(seed=0):
    prior_mean = mean(prior)
    posterior = condition_on(model, {"damage": damage})
    risk = damage_probability(posterior["intercept"], posterior["slope"])

print("posterior mean of the slope:", round(float(mean(posterior["slope"])), 3))
print("P(damage at 31°F), posterior mean:", round(float(mean(risk)), 3))
print("P(damage at 31°F), 90% interval:", quantile(risk, jnp.array([0.05, 0.95])).values)

# How each result was computed, from its provenance.
for name, result in [("mean of the prior", prior_mean), ("posterior", posterior), ("risk", risk)]:
    record = result.provenance.metadata
    method = f", method {record['method']}" if "method" in record else ""
    print(f"{name}: route {record['route']}{method}, exact {record['exact']}")
```

```text
posterior mean of the slope: -0.182
P(damage at 31°F), posterior mean: 0.962
P(damage at 31°F), 90% interval: [0.82085985 0.9999324 ]
mean of the prior: route closed_form, exact True
posterior: route bayes, method blackjax_nuts, exact False
risk: route sampling_lift, exact False
```

1. **Objects and composition:** `prior` is a distribution and `likelihood` a conditional distribution, each naming its components, and `likelihood * prior` is their joint distribution.
2. **Capabilities decide the computation:** the prior's mean has a closed form, so `mean` computes it exactly. The posterior of a logistic regression has none, so `condition_on` selects BlackJAX's No-U-Turn Sampler, and the posterior is a distribution held as its draws.
3. **Lifting:** `damage_probability` is a function of two numbers, and called on the posterior's two components it returns the distribution of the probability of damage, evaluated at joint draws of the intercept and the slope.
4. **Provenance:** each result records its route and whether it is exact, and the seed of `workflow_run` reproduces the draws.
<!-- --8<-- [end:quick-example] -->

## Installation

ProbPipe requires Python 3.12 or later. It is not yet on PyPI, so it installs from GitHub, and until a release the documentation describes the `dev/overhaul` branch:

```bash
pip install "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

The [installation page](https://tarps-group.github.io/prob-pipe/get_started/installation/) gives the commands for Google Colab and uv, and the extras that add backends such as Stan and PyMC.

## Learn more

- [The quickstart](https://tarps-group.github.io/prob-pipe/get_started/quickstart/) works through this example in ten minutes, with diagnostics and a predictive check.
- [The tutorials](https://tarps-group.github.io/prob-pipe/tutorials/01_first_analysis/) fit, check, and forecast a model of a moose population.
- [The API reference](https://tarps-group.github.io/prob-pipe/api/) documents every public name.
- [How to cite ProbPipe](https://tarps-group.github.io/prob-pipe/cite/), [getting help](https://tarps-group.github.io/prob-pipe/help/), and [contributing](CONTRIBUTING.md).
