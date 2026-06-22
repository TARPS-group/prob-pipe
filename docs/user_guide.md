# User Guide

These notebooks provide in-depth coverage of specific ProbPipe features. For a guided introduction, start with the [Getting Started tutorial](tutorials/getting_started.ipynb).

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Distribution basics](user_guide/01_distributions.ipynb) | The seven distribution ops (`sample`, `log_prob`, `mean`, `variance`, `cov`, `expectation`, `condition_on`); univariate, multivariate, and non-parametric (`Empirical` / `Bootstrap` / `BootstrapReplicate`) families; the `SupportsX` protocol family; `from_distribution` for converting between representations. |
| 2 | [Records and Record Distributions](user_guide/02_records.ipynb) | The 2×2 of structured containers: `Record` / `NumericRecord` (non-random values) paired with `RecordDistribution` / `NumericRecordDistribution` (random named-component distributions), plus the `RecordArray` and `DistributionArray` "array of" forms. |
| 3 | [Broadcasting and workflow functions](user_guide/03_broadcasting.ipynb) | Automatic uncertainty propagation, empirical enumeration, cartesian products, vectorization backends, and seeded reproducibility. |
| 4 | [Joint distributions](user_guide/04_joint_distributions.ipynb) | `ProductDistribution`, `SequentialJointDistribution`, `JointGaussian`, `JointEmpirical` / `NumericJointEmpirical`; component views; `condition_on`; flat-vector interop. |
| 5 | [External backends](user_guide/05_external_backends.ipynb) | How `condition_on` dispatches to BlackJAX NUTS (the default), Stan, PyMC, and nutpie; pinning a specific method; the inference method registry. |
| 6 | [Converting between representations](user_guide/06_converting_representations.ipynb) | Bijectors + `TransformedDistribution`, `from_distribution` moment matching, and the converter registry for satisfying protocols like `SupportsLogProb`. |
| 7 | [Sequential updating](user_guide/07_sequential_updating.ipynb) | Batch-wise Bayesian updating with `IncrementalConditioner`, auto KDE conversion, and provenance chain. |
| 8 | [JAX interop](user_guide/08_jax_interop.ipynb) | End-to-end JAX gradients through distributions: score functions, sensitivity analysis, maximum likelihood estimation, and variational inference. |
| 9 | [Bagged posteriors](user_guide/09_bagged_posteriors.ipynb) | `BootstrapReplicateDistribution`, broadcasting `condition_on` over resampled datasets, and the between- vs. within-replicate spread as a stability / misspecification diagnostic. |
| 10 | [Random functions and Gaussian emulators](user_guide/10_random_functions_and_emulators.ipynb) | `RandomFunction` / `GaussianRandomFunction` / `LinearBasisFunction`; joint-input / joint-output modes; algebraic operations; fitting to data; GP emulators and synthetic-likelihood surrogates for simulation-based inference. |
| 11 | [Scalability with Prefect](user_guide/11_prefect_scalability.ipynb) | How ProbPipe's global Prefect configuration distributes bagged posterior fits automatically: set orchestration once, and every `WorkflowFunction` in the system uses it. |
