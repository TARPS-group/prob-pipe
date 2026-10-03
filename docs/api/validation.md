> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Validation

This page documents the predictive checks, the scores of an approximate posterior against a reference posterior, and simulation-based calibration.
The diagnostics recorded on a posterior, such as R-hat and leave-one-out cross-validation, are on [Diagnostics](diagnostics.md).

## Predictive checks

::: probpipe.predictive_check

## Scores against a reference posterior

::: probpipe.validation.Reference
    options:
      show_root_full_path: true

::: probpipe.validation.score_posterior
    options:
      show_root_full_path: true

::: probpipe.validation.standardized_mean_error
    options:
      show_root_full_path: true

::: probpipe.validation.relative_cov_error
    options:
      show_root_full_path: true

::: probpipe.validation.std_ratios
    options:
      show_root_full_path: true

::: probpipe.validation.sliced_wasserstein
    options:
      show_root_full_path: true

::: probpipe.validation.mmd
    options:
      show_root_full_path: true

::: probpipe.validation.ksd
    options:
      show_root_full_path: true

## Calibration and coverage

::: probpipe.validation.simulation_based_calibration
    options:
      show_root_full_path: true

::: probpipe.validation.SBCResult
    options:
      show_root_full_path: true

::: probpipe.validation.interval_coverage
    options:
      show_root_full_path: true
