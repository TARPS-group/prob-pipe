# Reference Notebooks

These notebooks provide in-depth coverage of specific ProbPipe features. For a guided introduction, start with the [Getting Started tutorial](tutorials/getting_started.ipynb).

The list below is converging toward the 10-notebook target set tracked in [issue #127](https://github.com/TARPS-group/prob-pipe/issues/127). Notebooks flagged **refresh pending** are legacy Batch C entries — they still run against the current codebase but will get a light touch-up in follow-up PRs.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Distribution basics](examples/01_distributions.ipynb) | `sample` / `log_prob` / `mean` / `variance` / `cov` / `expectation`; parametric families; support checking; the `SupportsX` protocol family. |
| 2 | [Records and the Record family](examples/02_records.ipynb) | `Record`, `NumericRecord`, `RecordArray`, `NumericRecordArray`, `RecordTemplate` — the structured-value containers that flow through every ProbPipe workflow. |
| 3 | [Broadcasting](examples/04_broadcasting.ipynb) | Automatic uncertainty propagation, empirical enumeration, cartesian products, backend auto-detection, and seeded reproducibility. _Refresh pending._ |
| 4 | [Joint distributions](examples/04_joint_distributions.ipynb) | `ProductDistribution`, `SequentialJointDistribution`, `JointGaussian`, `JointEmpirical` / `NumericJointEmpirical`; component views; `condition_on`; flat-vector interop. |
| 5 | [External backends](examples/05_external_backends.ipynb) | How `condition_on` dispatches to TFP NUTS, Stan, PyMC, nutpie, and sbijax; pinning a specific method; the inference method registry. |
| 6 | [Converting between representations](examples/06_converting_representations.ipynb) | Bijectors + `TransformedDistribution`, `from_distribution` moment matching, and the converter registry for satisfying protocols like `SupportsLogProb`. |
| 7 | [Sequential Updating](examples/10_sequential_updating.ipynb) | Batch-wise Bayesian updating with `IncrementalConditioner`, auto KDE conversion, and provenance chain. _Refresh pending._ |
| 8 | [Automatic Differentiation](examples/05_autodiff.ipynb) | End-to-end JAX gradients through distributions: score functions, sensitivity analysis, maximum likelihood estimation, and variational inference. _Refresh pending._ |
| 9 | [Bagged posteriors](examples/09_bagged_posteriors.ipynb) | `BootstrapReplicateDistribution`, broadcasting `condition_on` over resampled datasets, and the between- vs. within-replicate spread as a stability / misspecification diagnostic. |
| 10 | [Emulators](examples/07_emulators.ipynb) / [Random Functions](examples/08_random_functions.ipynb) | Gaussian random functions and emulators for expensive simulators. _Refresh pending — will merge into a single notebook in the next PR._ |
