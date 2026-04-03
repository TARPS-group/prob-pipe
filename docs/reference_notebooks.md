# Reference Notebooks

These notebooks provide in-depth coverage of specific ProbPipe features. For a guided introduction, start with the [Getting Started tutorial](tutorials/getting_started.ipynb).

| Notebook | Description |
|----------|-------------|
| [Distributions](examples/01_distributions.ipynb) | Distribution basics, shape semantics, support checking, and moment-matching conversion between distribution types. |
| [Transformations](examples/02_transformations.ipynb) | Bijector-based transforms (Exp, Sigmoid, Softplus), chaining transforms, and transforming empirical distributions. |
| [Joint Distributions](examples/03_joint_distributions.ipynb) | Composing marginals into joint distributions, autoregressive dependence, exact Gaussian conditioning, and correlated broadcasting. |
| [Broadcasting](examples/04_broadcasting.ipynb) | Automatic uncertainty propagation, empirical enumeration, cartesian products, backend auto-detection, and seeded reproducibility. |
| [Automatic Differentiation](examples/05_autodiff.ipynb) | End-to-end JAX gradients through distributions: score functions, sensitivity analysis, maximum likelihood estimation, and variational inference. |
| [Modular Forecasting](examples/06_modular_forecasting.ipynb) | Building a full inference pipeline with swappable likelihoods, MCMC sampling, iterative Bayesian updating, and posterior predictive checks. |
| [Emulators](examples/07_emulators.ipynb) | Gaussian random functions and emulators for expensive simulators. |
| [Random Functions](examples/08_random_functions.ipynb) | Distribution over functions, algebraic operations on Gaussian random functions, and workflow function broadcasting. |
| [PyTree Distributions](examples/09_pytree_array_distributions.ipynb) | Distributions over structured JAX pytrees with shared batch shapes and per-leaf event shapes. |
