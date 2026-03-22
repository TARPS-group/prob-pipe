# Changelog

All notable changes to ProbPipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-03-21

Initial release with TensorFlow Probability / JAX backend.

### Added

- **Distribution framework**: `Distribution` ABC with TFP shape semantics (`sample_shape + batch_shape + event_shape`), `TFPDistribution` mixin, `EmpiricalDistribution` (direct JAX).
- **23 distribution wrappers**: 14 continuous (Normal, Beta, Gamma, etc.), 5 discrete (Bernoulli, Binomial, Poisson, etc.), 4 multivariate (MultivariateNormal, Dirichlet, Multinomial, Wishart, VonMisesFisher).
- **Constraints**: `Constraint` base class with partial-order compatibility checking. Built-in singletons (`real`, `positive`, `unit_interval`, `simplex`, etc.) and factories (`interval()`, `greater_than()`).
- **Transformed distributions**: `TransformedDistribution` with TFP bijectors (Exp, Sigmoid, Softplus, Shift, Scale, Chain). Automatic support derivation.
- **Joint distributions**: `ProductDistribution` (independent), `SequentialJointDistribution` (autoregressive), `JointEmpirical` (weighted joint samples), `JointGaussian` (exact analytical conditioning). `DistributionView` for component access, `ConditionedComponent` for conditioning.
- **Workflows and broadcasting**: `WorkflowFunction` with automatic uncertainty propagation. Multi-backend broadcasting (`jax` vectorization, `loop`, `prefect` orchestration). Auto-detection of JAX traceability. Empirical enumeration with budget-aware cartesian product.
- **Bayesian inference**: `MCMCSampler` with NUTS/HMC via TFP + auto-fallback to gradient-free RW-MH. `Likelihood` base class. `IterativeForecaster` for sequential Bayesian updating.
- **Provenance tracking**: Automatic lineage on all creation paths (transforms, broadcasting, conditioning, inference). `Provenance.to_dict()` / `from_dict()` serialization. `provenance_ancestors()` and `provenance_dag()` utilities.
- **Documentation**: MkDocs Material site with API reference, getting started guide, and 6 tutorial notebooks (distributions, transformations, joint distributions, broadcasting, autodiff, modular forecasting).
- **CI/CD**: GitHub Actions CI with pytest + coverage, Codecov integration, automated docs deployment to GitHub Pages.
