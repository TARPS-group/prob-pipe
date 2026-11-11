> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# User guide

ProbPipe is a Python library for computing with uncertainty.
You write the mathematics: distributions, models, and ordinary Python functions.
ProbPipe chooses how to compute each result, records how it computed it, and lets you change the method in one line.

## ProbPipe at a glance

On the morning of its last launch, the Space Shuttle Challenger stood at 31°F.
The four steps below estimate the probability of O-ring damage at that temperature from the 23 earlier launches.
The first block loads the data and states the likelihood, a logistic regression of damage on temperature, and the [quickstart](../get_started/quickstart.ipynb) explains each line of it.

```python
import jax
import jax.numpy as jnp
import pandas as pd

from probpipe import Normal, condition_on, function, mean, quantile, workflow_run
from probpipe.families import BernoulliFamily, glm_likelihood

data = pd.read_csv("docs/tutorials/data/challenger.csv")
temperature = jnp.asarray(data["temperature"], dtype=jnp.float32)
damage = jnp.asarray(data["damage"], dtype=jnp.int32)
X = jnp.stack([jnp.ones_like(temperature), temperature], axis=1)
likelihood = glm_likelihood("damage", BernoulliFamily(), X=X, label="likelihood")
```

### A function of one value accepts a distribution

The probability of damage is a function of the coefficients `beta`, written in plain JAX for one value of them:

```python
@function
def damage_probability(beta: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(beta[0] + beta[1] * 31.0)


prior = Normal("beta", jnp.zeros(2), jnp.array([10.0, 1.0]), label="prior")
with workflow_run(seed=0):
    print(mean(damage_probability(prior)))  # 0.528
```

Called on a distribution of `beta`, the function returns the distribution of the probability.
Under the prior alone, the expected probability of damage is close to one half.

### A model merges its parts, and one call fits it

```python
model = likelihood & prior  # the joint distribution of beta and the damage indicators
with workflow_run(seed=0):
    posterior = condition_on(model, {"damage": damage})
    forecast = damage_probability(posterior["beta"])

print(mean(forecast))  # 0.972
print(quantile(forecast, jnp.array([0.05, 0.95])).raw)  # [0.855 1.000]
```

After conditioning on the earlier launches, the same function gives an expected probability of 0.97.

### One line changes the algorithm, and the model stays the same

```python
with workflow_run(seed=0):
    tfp_posterior = condition_on.using(method="tfp_nuts")(model, {"damage": damage})
```

ProbPipe chose BlackJAX's No-U-Turn Sampler for the first posterior.
The `using` call runs TensorFlow Probability's No-U-Turn Sampler on the same model, and other registered methods, such as random-walk Metropolis, are selected the same way.

### Every result records how it was computed

```python
print(forecast.provenance)
```

```text
1. likelihood·prior = condition_on(likelihood·prior, given)  route inference_methods, method blackjax_nuts, approximate
2. broadcast(damage_probability, 1['beta'])          route sampling_lift, approximate
```

ProbPipe computes a result by a formula where one exists, such as the prior's mean, and marks it exact.
Otherwise ProbPipe computes it by an algorithm and marks it approximate, as the provenance of the forecast shows for both of its steps.

## ProbPipe and the libraries it uses

1. **NumPy and JAX:** your functions are written in them, and ProbPipe calls each function on values or on distributions.
2. **PyMC, Stan, and TensorFlow Probability:** ProbPipe wraps a model written in PyMC or Stan and fits it with that library's sampler, and TensorFlow Probability's samplers are among the methods that `condition_on` selects from.
3. **pandas, xarray, and ArviZ:** your data can arrive in pandas or xarray objects, and a sampler's draws are available as ArviZ data.

## How the guide is organized

The user guide explains ProbPipe one idea at a time, each with a small example.
The chapters are ordered so that each one uses only ideas from the chapters before it.
The [tutorials](../tutorials/01_first_analysis.ipynb) follow one analysis from start to finish, and the [API reference](../api/index.md) documents every public name.

### Choose a reading path

1. **Propagate uncertainty through a computation:** read chapters 1 to 3, from [Key concepts](01_key_concepts.ipynb) to [Propagating uncertainty through functions](03_functions_and_lifting.ipynb).
2. **Fit and check a model:** read chapters 4 to 7, from [Your data: values, records, and batches](04_values_and_records.ipynb) to [Checking a fit](07_checking_results.ipynb).
3. **Reuse, update, and reproduce results:** read chapters 8 and 9, [Working with results](08_conversion_and_updating.ipynb) and [Tracking and reproducing results](09_how_results_are_computed.ipynb).

The [specs](03_functions_and_lifting.ipynb#specs) in chapter 3 are for declaring or inspecting inputs and outputs.

### Chapters

| Chapter | What it covers |
|---|---|
| 1. [Key concepts](01_key_concepts.ipynb) | Terms, lifting, Monte Carlo error, provenance, and seeds, on one example. |
| 2. [Distributions](02_distributions.ipynb) | The parametric families, the operations on a distribution, and distributions built from a sampler or a density. |
| 3. [Propagating uncertainty through functions](03_functions_and_lifting.ipynb) | ProbPipe-native functions, calling them on distributions and batches, `pushforward`, `expectation`, and the specs that declare inputs and outputs. |
| 4. [Your data: values, records, and batches](04_values_and_records.ipynb) | Numeric data and records, pandas and xarray fields, and batches of values, records, and distributions. |
| 5. [Building a model](05_building_a_model.ipynb) | Conditional distributions, merging a model with `&`, and the fields and marginals of a joint model. |
| 6. [Fitting a model to data](06_conditioning_and_inference.ipynb) | Conditioning models on observations and selecting inference methods. |
| 7. [Checking a fit](07_checking_results.ipynb) | Checking inference, evaluating predictions, and testing calibration. |
| 8. [Working with results](08_conversion_and_updating.ipynb) | Distribution representations, resampling, and sequential updates. |
| 9. [Tracking and reproducing results](09_how_results_are_computed.ipynb) | Inspecting computations, controlling approximation and randomness, provenance, labels and notation, and immutability. |
| [Ray via Prefect](ray.md) | Running ProbPipe functions on Ray workers through Prefect. |

Each notebook opens in Google Colab from the badge at its top.
