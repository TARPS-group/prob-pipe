# Composite and joint

Distributions combining named components into a joint over a
[`Record`](../records.md). Component access: `dist["name"]` returns a view
(see [Internals](../internals.md) for the correlation semantics);
`dist.select("x", "y")` splats into a Function.

::: probpipe.ProductDistribution

::: probpipe.SequentialJointDistribution

::: probpipe.TransformedDistribution

The [bijector](../constraints.md#bijectors-for-unconstrained-reparameterization)
machinery pairs naturally with `TransformedDistribution` for
unconstrained-to-constrained reparameterization.

::: probpipe.JointGaussian
