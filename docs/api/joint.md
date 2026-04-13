# Joint Distributions

All joint distributions inherit from
[`ValuesDistribution`][probpipe.core._values_distribution.ValuesDistribution]
and return [`Values`][probpipe.core.values.Values] from `_sample()`.

Component access uses `dist["name"]` (returns a
[`_ValuesDistributionView`][probpipe.core._values_distribution._ValuesDistributionView])
and `dist.select("x", "y")` for workflow function broadcasting.

::: probpipe.distributions._product.ProductDistribution

::: probpipe.distributions._sequential_joint.SequentialJointDistribution

::: probpipe.distributions._joint_empirical.JointEmpirical

::: probpipe.distributions._joint_gaussian.JointGaussian
