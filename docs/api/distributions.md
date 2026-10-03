> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Distributions and families

This page documents the distribution classes, the capabilities a distribution claims, and the families ProbPipe ships.
The distributions over functions and over distributions are on [Random functions](random_functions.md), and the operations on a distribution, such as `sample` and `mean`, are on [Operations](operations.md).

## Distributions

::: probpipe.Distribution

::: probpipe.NumericDistribution

::: probpipe.EmpiricalDistribution

::: probpipe.DistributionBatch

::: probpipe.distributions.FieldView
    options:
      show_root_full_path: true

## Conditional distributions

::: probpipe.conditional_distribution

::: probpipe.distributions.ConditionalDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.ConditionalNumericDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.NumericConditionalDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.FullyNumericConditionalDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.ConditionalDistributionBatch
    options:
      show_root_full_path: true

## Factored distributions

`A * B` composes two laws into a factored joint.

::: probpipe.distributions.FactoredDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.FactoredNumericDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.FactoredConditionalDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.FactoredConditionalNumericDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.FactoredNumericConditionalDistribution
    options:
      show_root_full_path: true

::: probpipe.distributions.FactoredFullyNumericConditionalDistribution
    options:
      show_root_full_path: true

## Capabilities

A capability is a protocol that names one implementation method, such as `_sample` for `SupportsSampling`.
A class claims a capability by defining that method, and it claims a conditioning capability by inheriting it.

::: probpipe.SupportsSampling

::: probpipe.SupportsLogProb

::: probpipe.SupportsUnnormalizedLogProb

::: probpipe.SupportsMean

::: probpipe.SupportsVariance

::: probpipe.SupportsCovariance

::: probpipe.SupportsQuantile

::: probpipe.SupportsExpectation

::: probpipe.SupportsRandomLogProb

::: probpipe.SupportsRandomUnnormalizedLogProb

::: probpipe.SupportsExactConditioning

::: probpipe.SupportsApproximateConditioning

::: probpipe.distributions.SupportsMarginals
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsFactors
    options:
      show_root_full_path: true

## Capabilities of a conditional distribution

Each capability of a conditional distribution takes the given value `given` before the arguments of its unconditional counterpart.

::: probpipe.distributions.SupportsConditionalSampling
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalLogProb
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalUnnormalizedLogProb
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalMean
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalVariance
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalCovariance
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalQuantile
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalExpectation
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalRandomLogProb
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalRandomUnnormalizedLogProb
    options:
      show_root_full_path: true

::: probpipe.distributions.SupportsConditionalMarginals
    options:
      show_root_full_path: true

## Continuous families

`TFPDistribution` adapts a TensorFlow Probability distribution, and each family of this section and the next two is a subclass of it.

::: probpipe.TFPDistribution

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

## Discrete families

::: probpipe.Bernoulli

::: probpipe.Binomial

::: probpipe.Poisson

::: probpipe.Categorical

::: probpipe.NegativeBinomial

## Multivariate families

::: probpipe.MultivariateNormal

::: probpipe.Dirichlet

::: probpipe.Multinomial

::: probpipe.Wishart

::: probpipe.VonMisesFisher

## Resampling families

::: probpipe.BootstrapReplicateDistribution

::: probpipe.BootstrapDistribution

::: probpipe.KDEDistribution

::: probpipe.families.SmoothingKernel
    options:
      show_root_full_path: true

::: probpipe.families.GaussianKernel
    options:
      show_root_full_path: true

::: probpipe.families.EpanechnikovKernel
    options:
      show_root_full_path: true

## Mixtures

::: probpipe.families.MixtureDistribution
    options:
      show_root_full_path: true

## Transformed distributions

::: probpipe.BijectorTransformedDistribution

::: probpipe.families.LinearPushforwardDistribution
    options:
      show_root_full_path: true

## Gaussian joints

::: probpipe.families.FactoredMultivariateGaussian
    options:
      show_root_full_path: true

## Conditional families

::: probpipe.families.LinearGaussianConditional
    options:
      show_root_full_path: true

::: probpipe.families.glm_likelihood
    options:
      show_root_full_path: true

::: probpipe.families.GLMFamily
    options:
      show_root_full_path: true

::: probpipe.families.GaussianFamily
    options:
      show_root_full_path: true

::: probpipe.families.BernoulliFamily
    options:
      show_root_full_path: true

::: probpipe.families.PoissonFamily
    options:
      show_root_full_path: true

## Program-defined families

`StanModel` and `PyMCModel` import from `probpipe` as well as from `probpipe.families`.

::: probpipe.families.StanModel
    options:
      show_root_full_path: true

::: probpipe.families.PyMCModel
    options:
      show_root_full_path: true

::: probpipe.UnnormalizedDistribution
