# Distributions

Built-in distribution classes, grouped by family. Base classes and the
protocols distributions implement live under
[Extending ProbPipe](../extending.md).

## Sections

| Section | Classes |
|---|---|
| [Continuous](continuous.md) | `Normal`, `Beta`, `Gamma`, `InverseGamma`, `Exponential`, `LogNormal`, `StudentT`, `Uniform`, `Cauchy`, `Laplace`, `HalfNormal`, `HalfCauchy`, `Pareto`, `TruncatedNormal` |
| [Discrete](discrete.md) | `Bernoulli`, `Binomial`, `Poisson`, `Categorical`, `NegativeBinomial` |
| [Multivariate](multivariate.md) | `MultivariateNormal`, `Dirichlet`, `Multinomial`, `Wishart`, `VonMisesFisher` |
| [Composite and joint](composite.md) | `ProductDistribution`, `SequentialJointDistribution`, `TransformedDistribution`, `JointGaussian` |
| [Empirical and bootstrap](empirical.md) | `EmpiricalDistribution`, `RecordEmpiricalDistribution`, `KDEDistribution`, `BootstrapDistribution`, `BootstrapReplicateDistribution`, `RecordBootstrapReplicateDistribution`, `JointEmpirical`, `NumericJointEmpirical` |
| [Random functions](random_functions.md) | `RandomFunction`, `ArrayRandomFunction`, `GaussianRandomFunction`, `LinearBasisFunction`, `RandomMeasure`, `NumericRandomMeasure` |
