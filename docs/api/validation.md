# Validation

Utilities for validating inference methods and models: predictive checking, and
posterior-vs-reference comparison metrics that score an approximation against a
trusted reference (analytic, long-NUTS, or sandwich). These answer "does this
method recover the right posterior?", as opposed to per-fit convergence
diagnostics, which assess a single fitted posterior.

## Predictive checks

::: probpipe.validation.predictive_check

## Reference posteriors

::: probpipe.validation.Reference

## Comparison metrics

The metrics group by what the reference must carry: the moment metrics need the
reference's high-precision `(mean, cov)`; the sample metrics need reference
draws; the kernel Stein discrepancy needs only the target score `∇ log π`. All
return JAX arrays and are jit-compatible; `score_posterior` aggregates a chosen
set into a scorecard, skipping any whose reference pieces are absent.

::: probpipe.validation.standardized_mean_error

::: probpipe.validation.relative_cov_error

::: probpipe.validation.std_ratios

::: probpipe.validation.sliced_wasserstein

::: probpipe.validation.mmd

::: probpipe.validation.ksd

::: probpipe.validation.score_posterior
