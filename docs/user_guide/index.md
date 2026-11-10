> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# User guide

The user guide explains ProbPipe one idea at a time, each with a small example.
You describe the quantities, the uncertainty about them, and the computations on them, and ProbPipe chooses how to compute each result.
The chapters are ordered so that each one uses only ideas from the chapters before it.
The [tutorials](../tutorials/01_first_analysis.ipynb) follow one analysis from start to finish, and the [API reference](../api/index.md) documents every public name.

## Choose a reading path

1. **Propagate uncertainty through a computation:** read chapters 1 to 3, from [Uncertainty in a computation](01_uncertainty_in_a_computation.ipynb) to [Propagating uncertainty through functions](03_functions_and_lifting.ipynb).
2. **Fit and check a model:** read chapters 4 to 7, from [Your data: values, records, and batches](04_values_and_records.ipynb) to [Checking a fit](07_checking_results.ipynb).
3. **Reuse, update, and reproduce results:** read chapters 8 and 9, [Working with results](08_conversion_and_updating.ipynb) and [Tracking and reproducing results](09_how_results_are_computed.ipynb).

The [specs](03_functions_and_lifting.ipynb#specs) in chapter 3 are for declaring or inspecting inputs and outputs.

## Chapters

| Chapter | What it covers |
|---|---|
| 1. [Uncertainty in a computation](01_uncertainty_in_a_computation.ipynb) | Values, distributions, and functions in one example. |
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
