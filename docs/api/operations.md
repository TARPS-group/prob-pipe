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
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

## Density evaluation

::: probpipe.core.ops.log_prob
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.prob
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.unnormalized_log_prob
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.unnormalized_prob
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.random_log_prob
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.random_unnormalized_log_prob
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

## Moments and expectations

::: probpipe.core.ops.mean
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.variance
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.cov
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

::: probpipe.core.ops.expectation
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

## Conditioning

::: probpipe.core.ops.condition_on
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

`condition_on` dispatches inference via the inference-method registry; see
[Modeling and inference → Inference methods](inference.md#inference-methods)
for the built-ins and how to override the auto-selected method.

## Conversion

::: probpipe.core.ops.from_distribution
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

See [Conversion and interop](converters.md) for the converter registry and
the bidirectional converters that back this op.
