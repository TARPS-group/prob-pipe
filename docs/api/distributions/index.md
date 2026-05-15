# Distributions

ProbPipe's built-in distribution families, plus the composite, empirical, and
random-function classes that build on them.

The base classes (`Distribution`, `RecordDistribution`,
`NumericRecordDistribution`, `TFPDistribution`) and the
[protocols](../extending.md#protocols) that distributions implement are
documented under [Extending ProbPipe](../extending.md) — they are the surface
you implement against when adding a new distribution, not the surface you
use to construct one.

## Sections

| Section | What's there |
|---|---|
| [Continuous](continuous.md) | `Normal`, `Beta`, `Gamma`, `InverseGamma`, `Exponential`, `LogNormal`, `StudentT`, `Uniform`, `Cauchy`, `Laplace`, `HalfNormal`, `HalfCauchy`, `Pareto`, `TruncatedNormal` |
| [Discrete](discrete.md) | `Bernoulli`, `Binomial`, `Poisson`, `Categorical`, `NegativeBinomial` |
| [Multivariate](multivariate.md) | `MultivariateNormal`, `Dirichlet`, `Multinomial`, `Wishart`, `VonMisesFisher` |
| [Composite and joint](composite.md) | `ProductDistribution`, `SequentialJointDistribution`, `TransformedDistribution`, `JointGaussian` |
| [Empirical and bootstrap](empirical.md) | `EmpiricalDistribution`, `RecordEmpiricalDistribution`, `KDEDistribution`, the `BootstrapDistribution` family, `JointEmpirical`, `NumericJointEmpirical` |
| [Random functions](random_functions.md) | `RandomFunction`, `ArrayRandomFunction`, `GaussianRandomFunction`, `LinearBasisFunction`, `RandomMeasure`, `NumericRandomMeasure` |
