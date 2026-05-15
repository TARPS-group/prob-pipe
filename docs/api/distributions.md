# Distributions

ProbPipe's built-in distribution families, plus the composite, empirical, and
random-function classes that build on them.

The base classes (`Distribution`, `RecordDistribution`,
`NumericRecordDistribution`, `TFPDistribution`) and the
[protocols](extending.md#protocols) that distributions implement are
documented under [Extending ProbPipe](extending.md) — they are the surface
you implement against when adding a new distribution, not the surface you
use to construct one.

## Continuous

::: probpipe.Normal

::: probpipe.Beta

::: probpipe.Gamma

::: probpipe.InverseGamma

::: probpipe.Exponential

::: probpipe.LogNormal

::: probpipe.StudentT

::: probpipe.Uniform

::: probpipe.Cauchy

::: probpipe.Laplace

::: probpipe.HalfNormal

::: probpipe.HalfCauchy

::: probpipe.Pareto

::: probpipe.TruncatedNormal

## Discrete

::: probpipe.Bernoulli

::: probpipe.Binomial

::: probpipe.Poisson

::: probpipe.Categorical

::: probpipe.NegativeBinomial

## Multivariate

::: probpipe.MultivariateNormal

::: probpipe.Dirichlet

::: probpipe.Multinomial

::: probpipe.Wishart

::: probpipe.VonMisesFisher

## Composite and joint

Composite distributions combine named components into a joint over a
[`Record`](records.md). Component access uses `dist["name"]` (returns a
lightweight view that preserves correlation across broadcast samples — see
[Internals](internals.md)) and `dist.select("x", "y")` for workflow-function
splatting.

### ProductDistribution

::: probpipe.ProductDistribution

### SequentialJointDistribution

::: probpipe.SequentialJointDistribution

### TransformedDistribution

::: probpipe.TransformedDistribution

For unconstrained-to-constrained reparameterization (e.g. for MCMC / VI), see
[Constraints → Bijectors](constraints.md#bijectors-for-unconstrained-reparameterization).

### JointGaussian

::: probpipe.JointGaussian

## Empirical and bootstrap

Finite-sample distributions backed by stored draws or by an underlying data
source. Each finite-sample class exposes `.n` (count of stored items — see
the convention in `STYLE_GUIDE.md` §1.9) and is non-iterable (use `.samples`
/ `.draws()` / `.components` for collection-style access).

### EmpiricalDistribution

::: probpipe.EmpiricalDistribution

::: probpipe.RecordEmpiricalDistribution

### KDEDistribution

::: probpipe.KDEDistribution

### Bootstrap

::: probpipe.BootstrapDistribution

::: probpipe.BootstrapReplicateDistribution

::: probpipe.RecordBootstrapReplicateDistribution

### Joint empirical

::: probpipe.JointEmpirical

::: probpipe.NumericJointEmpirical

## Random functions

Distributions over function-valued random variables. `RandomFunction` is the
base; `GaussianRandomFunction` is the workhorse for GP emulation and BO.
`RandomMeasure` represents a measure-valued random variable — used by SBI
and posterior-sample workflows.

### RandomFunction (base classes)

::: probpipe.RandomFunction

::: probpipe.ArrayRandomFunction

### GaussianRandomFunction

::: probpipe.GaussianRandomFunction

::: probpipe.LinearBasisFunction

### Random measures

::: probpipe.RandomMeasure

::: probpipe.NumericRandomMeasure
