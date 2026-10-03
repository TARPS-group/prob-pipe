> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Operations

This page documents the operations, such as `sample`, `mean`, and `condition_on`, and the errors a call raises when no route of an operation applies or the result is undefined.
A call of an operation selects one of its routes, which are its implementations, and ranks exact routes before approximate ones.
The `convert` operation is on [Conversion](conversion.md), the inference methods that `condition_on` selects are on [Inference methods](inference.md), and the registry of the operations and their routes is on [Registries for extensions](extending.md).

## Sampling

::: probpipe.sample

## Densities

::: probpipe.log_prob

::: probpipe.unnormalized_log_prob

::: probpipe.prob

::: probpipe.unnormalized_prob

::: probpipe.random_log_prob

::: probpipe.random_unnormalized_log_prob

## Moments and expectations

::: probpipe.mean

::: probpipe.variance

::: probpipe.cov

::: probpipe.quantile

::: probpipe.expectation

## Conditioning

::: probpipe.condition_on

## Joints, marginals, and mixtures

::: probpipe.joint

::: probpipe.marginal

::: probpipe.factor

::: probpipe.mixture

## Maps

::: probpipe.evaluate

::: probpipe.inverse

::: probpipe.log_det_jacobian

## Errors

::: probpipe.ResolutionError

::: probpipe.MathematicalDomainError
