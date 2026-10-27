> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# User guide

The user guide explains ProbPipe one feature at a time.
Each chapter explains a calling rule or a group of operations with a small example.
You describe the mathematical objects and operations; ProbPipe selects a computation from the available representations and capabilities.
Start with the short example in [Terms](01_terms.ipynb), which combines a known value with a distribution through a function.
Its later sections provide optional reference on term properties.
The [API reference](../api/index.md) documents every public name.

## Choose a reading path

1. **Propagate uncertainty through a computation:** read [Distributions](03_distributions.ipynb), then [Functions on values and distributions](05_functions_and_lifting.ipynb).
2. **Fit and check a model:** read [Joint models](04_joint_models.ipynb), then [Conditioning and inference methods](06_conditioning_and_inference.ipynb) and [Checking results](07_checking_results.ipynb).
3. **Understand a result's computation:** read [How a result was computed](09_how_results_are_computed.ipynb) for approximation controls, reproducible draws, and provenance.

Consult [Values and records](02_values_and_records.ipynb) when you need to prepare data or extract results.
The detailed specs in [Terms](01_terms.ipynb#specs) are for declaring or inspecting inputs and outputs.

## Chapters

| Chapter | What it covers |
|---|---|
| [Terms](01_terms.ipynb) | Values, distributions, and functions; optional reference on term properties. |
| [Values and records](02_values_and_records.ipynb) | Numeric data and records, including pandas and xarray fields. |
| [Distributions](03_distributions.ipynb) | The parametric families, the operations on a distribution, and distributions built from a sampler or a density. |
| [Joint models and conditional distributions](04_joint_models.ipynb) | Conditional distributions, merging a model with `&`, and the fields and marginals of a joint model. |
| [Functions on values and distributions](05_functions_and_lifting.ipynb) | ProbPipe-native functions, calling them on distributions and batches, `pushforward`, and `expectation`. |
| [Conditioning and inference methods](06_conditioning_and_inference.ipynb) | Conditioning models on observations and selecting inference methods. |
| [Checking results](07_checking_results.ipynb) | Checking inference, evaluating predictions, and testing calibration. |
| [Converting and updating distributions](08_conversion_and_updating.ipynb) | Distribution representations and sequential updates. |
| [How a result was computed](09_how_results_are_computed.ipynb) | Inspecting computations and controlling approximation and randomness. |
| [Ray via Prefect](ray.md) | Running ProbPipe functions on Ray workers through Prefect. |

Each notebook opens in Google Colab from the badge at its top.
