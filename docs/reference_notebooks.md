# Reference Notebooks

These notebooks provide in-depth coverage of specific ProbPipe features. For a guided introduction, start with the [Getting Started tutorial](tutorials/getting_started.ipynb).

The set below matches the 10-notebook target tracked in [issue #127](https://github.com/TARPS-group/prob-pipe/issues/127).

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Distribution basics](examples/01_distributions.ipynb) | `sample` / `log_prob` / `mean` / `variance` / `cov` / `expectation`; parametric families; support checking; the `SupportsX` protocol family. |
| 2 | [Records and the Record family](examples/02_records.ipynb) | `Record`, `NumericRecord`, `RecordArray`, `NumericRecordArray`, `RecordTemplate` — the structured-value containers that flow through every ProbPipe workflow. |
| 3 | [Broadcasting and workflow functions](examples/03_broadcasting.ipynb) | Automatic uncertainty propagation, empirical enumeration, cartesian products, vectorization backends, and seeded reproducibility. |
| 4 | [Joint distributions](examples/04_joint_distributions.ipynb) | `ProductDistribution`, `SequentialJointDistribution`, `JointGaussian`, `JointEmpirical` / `NumericJointEmpirical`; component views; `condition_on`; flat-vector interop. |
| 5 | [External backends](examples/05_external_backends.ipynb) | How `condition_on` dispatches to TFP NUTS, Stan, PyMC, nutpie, and sbijax; pinning a specific method; the inference method registry. |
| 6 | [Converting between representations](examples/06_converting_representations.ipynb) | Bijectors + `TransformedDistribution`, `from_distribution` moment matching, and the converter registry for satisfying protocols like `SupportsLogProb`. |
| 7 | [Sequential updating](examples/07_sequential_updating.ipynb) | Batch-wise Bayesian updating with `IncrementalConditioner`, auto KDE conversion, and provenance chain. |
| 8 | [JAX interop](examples/08_jax_interop.ipynb) | End-to-end JAX gradients through distributions: score functions, sensitivity analysis, maximum likelihood estimation, and variational inference. |
| 9 | [Bagged posteriors](examples/09_bagged_posteriors.ipynb) | `BootstrapReplicateDistribution`, broadcasting `condition_on` over resampled datasets, and the between- vs. within-replicate spread as a stability / misspecification diagnostic. |
| 10 | [Random functions and Gaussian emulators](examples/10_random_functions_and_emulators.ipynb) | `RandomFunction` / `GaussianRandomFunction` / `LinearBasisFunction`; joint-input / joint-output modes; algebraic operations; fitting to data; GP emulators and synthetic-likelihood surrogates for simulation-based inference. |
