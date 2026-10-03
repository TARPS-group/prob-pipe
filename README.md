> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20683559-blue)](https://doi.org/10.5281/zenodo.20683559)

<!-- --8<-- [start:intro] -->
ProbPipe is a Python framework for probabilistic pipelines with automated uncertainty quantification.
A pipeline is built from distributions, records of fixed values, and ordinary Python functions, and operations such as `condition_on`, `mean`, and `sample` act on all of them through one interface.
Each operation computes its result exactly where it can and otherwise selects a method, such as BlackJAX's NUTS sampler, and every result records how it was produced.
<!-- --8<-- [end:intro] -->

## Quick example

<!-- --8<-- [start:quick-example] -->
A Bayesian logistic regression on the [Challenger O-ring data](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster) relates the temperature of 23 shuttle launches to whether an O-ring was damaged.
Its posterior then gives the probability of damage at the Challenger's launch temperature of 31°F.

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

# A prior over the intercept and the slope of the log-odds of O-ring damage.
prior = Normal("intercept", 0.0, 10.0) * Normal("slope", 0.0, 1.0)

# Each launch's damage indicator, given the intercept and the slope.
likelihood = conditional_distribution(
    "damage",
    lambda intercept, slope: Bernoulli("damage", logits=intercept + slope * temperature),
    given_spec={"intercept": NumericArraySpec(()), "slope": NumericArraySpec(())},
)


@function
def damage_probability(intercept, slope):
    return jax.nn.sigmoid(intercept + slope * 31.0)


with workflow_run(seed=0):
    posterior = condition_on(likelihood * prior, {"damage": damage})
    risk = damage_probability(posterior["intercept"], posterior["slope"])

print("posterior mean of the slope:", round(float(mean(posterior["slope"])), 3))
print("P(damage at 31°F), posterior mean:", round(float(mean(risk)), 3))
print("P(damage at 31°F), 90% interval:", quantile(risk, jnp.array([0.05, 0.95])).values)
```

```text
posterior mean of the slope: -0.189
P(damage at 31°F), posterior mean: 0.959
P(damage at 31°F), 90% interval: [0.72666454 0.99999535]
```

`damage_probability` is an ordinary function of two numbers.
Called on the posterior's two fields, it is evaluated at joint draws of the intercept and the slope, and it returns the distribution of the probability of damage.
<!-- --8<-- [end:quick-example] -->

## Installation

ProbPipe requires Python 3.12 or later and is not yet on PyPI, so it installs from GitHub:

```bash
pip install "git+https://github.com/TARPS-group/prob-pipe.git"
```

## Learn more

- [Documentation](https://tarps-group.github.io/prob-pipe/)
- [API reference](https://tarps-group.github.io/prob-pipe/api/)
- [How to cite ProbPipe](https://tarps-group.github.io/prob-pipe/cite/)
- [Getting help](https://tarps-group.github.io/prob-pipe/help/)
- [Contributing](CONTRIBUTING.md)
