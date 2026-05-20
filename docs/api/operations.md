# Operations

Standalone workflow functions for sampling, density evaluation, moments,
conditioning, and conversion. Each op dispatches via the matching
[protocol](extending.md#protocols), participates in
[broadcasting](../examples/03_broadcasting.ipynb), and is subject to
[Prefect orchestration](workflows.md) when configured.

## Sampling

::: probpipe.core.ops.sample

## Density evaluation

::: probpipe.core.ops.log_prob

::: probpipe.core.ops.prob

::: probpipe.core.ops.unnormalized_log_prob

::: probpipe.core.ops.unnormalized_prob

::: probpipe.core.ops.random_log_prob

::: probpipe.core.ops.random_unnormalized_log_prob

## Moments and expectations

::: probpipe.core.ops.mean

::: probpipe.core.ops.variance

::: probpipe.core.ops.cov

::: probpipe.core.ops.expectation

## Conditioning

::: probpipe.core.ops.condition_on

`condition_on` dispatches inference via the
[inference-method registry](inference.md#inference-methods); override the
auto-selection with `method="<name>"`.

## Conversion

::: probpipe.core.ops.from_distribution

Backed by the [converter registry](converters.md).
