# Operations

Built-in operations are standalone workflow functions that form the primary public API.
Each operation checks the relevant protocol at dispatch time and delegates to the
distribution's private implementation method. All operations participate in
[broadcasting](../examples/03_broadcasting.ipynb) and Prefect orchestration.

```python
from probpipe import sample, mean, log_prob, condition_on
```

## Sampling

::: probpipe.core.ops.sample
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false

## Density Evaluation

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

## Moments and Expectations

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

## Conversion

::: probpipe.core.ops.from_distribution
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
