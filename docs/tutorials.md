# Tutorials

These notebooks walk through ProbPipe's core features with runnable examples.

| Notebook | Description |
|----------|-------------|
| [Distributions](examples/01_distributions.ipynb) | Distribution basics, shape semantics, support checking, and moment-matching conversion between distribution types. |
| [Transformations](examples/02_transformations.ipynb) | Bijector-based transforms (Exp, Sigmoid, Softplus), chaining transforms, and transforming empirical distributions. |
| [Joint Distributions](examples/03_joint_distributions.ipynb) | Composing marginals into joint distributions, autoregressive dependence, exact Gaussian conditioning, and correlated broadcasting. |
| [Broadcasting](examples/04_broadcasting.ipynb) | Automatic uncertainty propagation, empirical enumeration, cartesian products, backend auto-detection, and seeded reproducibility. |
| [Automatic Differentiation](examples/05_autodiff.ipynb) | End-to-end JAX gradients through distributions: score functions, sensitivity analysis, maximum likelihood estimation, and variational inference. |
| [Modular Forecasting](examples/06_modular_forecasting.ipynb) | Building a full inference pipeline with swappable likelihoods, MCMC sampling, iterative Bayesian updating, and posterior predictive checks. |
