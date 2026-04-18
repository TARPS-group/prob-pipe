# Reference Notebooks

These notebooks provide in-depth coverage of specific ProbPipe features. For a guided introduction, start with the [Getting Started tutorial](tutorials/getting_started.ipynb).

The list below is converging toward the 10-notebook target set tracked in [issue #127](https://github.com/TARPS-group/prob-pipe/issues/127). Notebooks flagged **rewrite pending** are legacy Batch B/C entries — they will be replaced or refreshed in follow-up PRs.

| Notebook | Description |
|----------|-------------|
| [Distribution basics](examples/01_distributions.ipynb) | `sample` / `log_prob` / `mean` / `variance` / `cov` / `expectation`; parametric families; support checking; the `SupportsX` protocol family. |
| [Records and the Record family](examples/02_records.ipynb) | `Record`, `NumericRecord`, `RecordArray`, `NumericRecordArray`, `RecordTemplate` — the structured-value containers that flow through every ProbPipe workflow. |
| [Joint distributions](examples/04_joint_distributions.ipynb) | `ProductDistribution`, `SequentialJointDistribution`, `JointGaussian`, `JointEmpirical` / `NumericJointEmpirical`; component views; `condition_on`; flat-vector interop. |
| [Broadcasting](examples/04_broadcasting.ipynb) | Automatic uncertainty propagation, empirical enumeration, cartesian products, backend auto-detection, and seeded reproducibility. |
| [Transformations](examples/02_transformations.ipynb) | Bijector-based transforms (Exp, Sigmoid, Softplus), chaining transforms, and transforming empirical distributions. _Rewrite pending — will become "Converting between representations" in Batch B._ |
| [Automatic Differentiation](examples/05_autodiff.ipynb) | End-to-end JAX gradients through distributions: score functions, sensitivity analysis, maximum likelihood estimation, and variational inference. _Rewrite pending._ |
| [Emulators](examples/07_emulators.ipynb) | Gaussian random functions and emulators for expensive simulators. _Rewrite pending — will merge with Random Functions in Batch C._ |
| [Random Functions](examples/08_random_functions.ipynb) | Distribution over functions, algebraic operations on Gaussian random functions, and workflow function broadcasting. _Rewrite pending._ |
| [Sequential Updating](examples/10_sequential_updating.ipynb) | Batch-wise Bayesian updating with `IncrementalConditioner`, auto KDE conversion, and provenance chain. _Rewrite pending._ |
