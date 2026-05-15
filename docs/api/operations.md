# Operations

Built-in operations are standalone workflow functions that form the primary
public API. Each op checks the relevant [protocol](extending.md#protocols)
at dispatch time and delegates to the distribution's private implementation
method. All ops participate in [broadcasting](../examples/03_broadcasting.ipynb)
and Prefect orchestration ([Workflows](workflows.md)).

```python
from probpipe import sample, mean, log_prob, condition_on
```

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

`condition_on` dispatches inference via the inference-method registry; see
[Modeling and inference → Inference methods](inference.md#inference-methods)
for the built-ins and how to override the auto-selected method.

## Conversion

::: probpipe.core.ops.from_distribution

See [Conversion and interop](converters.md) for the converter registry and
the bidirectional converters that back this op.
