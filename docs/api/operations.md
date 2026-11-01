> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Operations

This page documents the operations, such as `sample`, `mean`, and `condition_on`, and the errors a call raises when no route of an operation applies or the result is undefined.
A call of an operation selects one of its routes, which are its implementations, and ranks exact routes before approximate ones.
The `convert` operation is on [Conversion](conversion.md), the inference methods that `condition_on` selects are on [Inference methods](inference.md), and the registry of the operations and their routes is on [Registries for extensions](extending.md).

For a kernel, `given=` first tries the operation's native exact conditional
capabilities, then exact conditioning followed by an exact operation.
The latter requires exact input binding and any necessary normalization.
If either needs an approximation, the operation can select its Monte Carlo
route using the requested `num_samples`.
Delegation to a bound law records the fidelity of binding, normalization,
and the operation together.
An explicit `method` selects the downstream operation's route; `method_options`
also passes to binding and normalization.

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

::: probpipe.standard_deviation

::: probpipe.cov

::: probpipe.quantile

::: probpipe.expectation

## Summary tables

`summarize` collects several of the moments above for one law in one record, which displays as a table.
It is a function over the operations, so it has no routes of its own.

::: probpipe.summarize

## Conditioning

::: probpipe.condition_on

## Joints, marginals, and mixtures

::: probpipe.joint

::: probpipe.marginal

::: probpipe.factor

::: probpipe.mixture

## Maps

::: probpipe.pushforward

::: probpipe.inverse

::: probpipe.log_det_jacobian

## Errors

::: probpipe.ResolutionError

::: probpipe.MathematicalDomainError
