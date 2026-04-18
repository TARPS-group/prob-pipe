# Joint Distributions

All joint distributions inherit from
[`RecordDistribution`][probpipe.core._record_distribution.RecordDistribution]
and return [`Record`][probpipe.core.record.Record] from `_sample()`.

Component access uses `dist["name"]` (returns a lightweight view that
preserves correlation across broadcast samples — see
[Internals](internals.md)) and `dist.select("x", "y")` for workflow
function broadcasting.

::: probpipe.distributions._product.ProductDistribution

::: probpipe.distributions._sequential_joint.SequentialJointDistribution

::: probpipe.distributions._joint_empirical.JointEmpirical

::: probpipe.distributions._joint_empirical.NumericJointEmpirical

::: probpipe.distributions._joint_gaussian.JointGaussian
