# Operations

Built-in operations are standalone workflow functions that form the primary public API.
Each operation checks the relevant protocol at dispatch time and delegates to the
distribution's private implementation method. All operations participate in
[broadcasting](../examples/04_broadcasting.ipynb) and Prefect orchestration.

```python
from probpipe import sample, mean, log_prob, condition_on
```

## Sampling

::: probpipe.core.ops._sample_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "sample"

## Density Evaluation

::: probpipe.core.ops._log_prob_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "log_prob"

::: probpipe.core.ops._prob_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "prob"

::: probpipe.core.ops._unnormalized_log_prob_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "unnormalized_log_prob"

::: probpipe.core.ops._unnormalized_prob_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "unnormalized_prob"

## Moments and Expectations

::: probpipe.core.ops._mean_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "mean"

::: probpipe.core.ops._variance_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "variance"

::: probpipe.core.ops._cov_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "cov"

::: probpipe.core.ops._expectation_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "expectation"

## Conditioning

::: probpipe.core.ops._condition_on_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "condition_on"

## Conversion

::: probpipe.core.ops._from_distribution_impl
    options:
      show_root_heading: true
      heading_level: 3
      show_root_full_path: false
      show_name: true
      name: "from_distribution"
