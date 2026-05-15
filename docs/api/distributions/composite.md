# Composite and joint

Composite distributions combine named components into a joint over a
[`Record`](../records.md). Component access uses `dist["name"]` (returns a
lightweight view that preserves correlation across broadcast samples — see
[Internals](../internals.md)) and `dist.select("x", "y")` for
workflow-function splatting.

::: probpipe.ProductDistribution

::: probpipe.SequentialJointDistribution

::: probpipe.TransformedDistribution

For unconstrained-to-constrained reparameterization (e.g. for MCMC / VI), see
[Constraints → Bijectors](../constraints.md#bijectors-for-unconstrained-reparameterization).

::: probpipe.JointGaussian
