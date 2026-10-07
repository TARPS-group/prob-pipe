> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# User guide

The user guide explains ProbPipe one feature at a time, on small examples.
Each chapter covers one kind of object or one group of operations, so you can read the chapter you need when you need it.
[Terms](01_terms.ipynb) introduces the properties that every object shares, so read it first.
The [tutorials](../tutorials/01_first_analysis.ipynb) instead follow one analysis from start to finish, and the [API reference](../api/index.md) documents every public name.

| Chapter | What it covers |
|---|---|
| [Terms](01_terms.ipynb) | The label, spec, and stored data of every object, immutability, batches, and provenance. |
| [Values and records](02_values_and_records.ipynb) | Arrays, records of named fields, non-numeric values, and pandas and xarray data. |
| [Distributions](03_distributions.ipynb) | The parametric families, the operations on a distribution, and distributions built from a sampler or a density. |
| [Joint models and conditional distributions](04_joint_models.ipynb) | Conditional distributions, composing a model with `*`, and the fields and marginals of a joint model. |
| [Functions on values and distributions](05_functions_and_lifting.ipynb) | ProbPipe-native functions, calling them on distributions and batches, `evaluate`, and `expectation`. |
| [Conditioning and inference methods](06_conditioning_and_inference.ipynb) | `condition_on`, exact conditioning, choosing an inference method, existing PyMC models, and simulation-based inference. |
| [Checking results](07_checking_results.ipynb) | MCMC diagnostics, predictive checks, model comparison, and simulation-based calibration. |
| [Converting and updating distributions](08_conversion_and_updating.ipynb) | Empirical distributions, `convert`, resampling, and updating a posterior as data arrive. |
| [How a result was computed](09_how_results_are_computed.ipynb) | Routes, options, reproducible random draws, and provenance. |
| [Ray via Prefect](ray.md) | Running ProbPipe functions on Ray workers through Prefect. |

Every chapter runs on Google Colab from the badge at its top.
