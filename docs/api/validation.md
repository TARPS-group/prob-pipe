# Validation

Utilities for validating inference methods and models: predictive checking, and
posterior-vs-reference comparison metrics that score an approximation against a
trusted reference (analytic, long-NUTS, or sandwich). These answer "does this
method recover the right posterior?", as opposed to per-fit convergence
diagnostics, which assess a single fitted posterior.

## Predictive checks

When `key` is omitted, `predictive_check` delegates randomness to the workflow
broker only for the exact built-in `GLMLikelihood`. Custom or otherwise opaque
generative likelihoods—including `GLMLikelihood` subclasses—must pass an
explicit `key=`. Inheriting the built-in `generate_data` method is not enough
to certify that a subclass preserves its stochastic-effect descriptor. The
same boundary applies to `simulation_based_calibration` and the diagnostic
helper `add_ppc`.

::: probpipe.validation.predictive_check

## Reference posteriors

::: probpipe.validation.Reference

## Comparison metrics

The metrics group by what the reference must carry: the moment metrics need the
reference's high-precision `(mean, cov)`; the sample metrics need reference
draws; the kernel Stein discrepancy needs only the target score `∇ log π`. All
return JAX arrays and are jit-compatible; `score_posterior` aggregates a chosen
set into a scorecard, skipping any whose reference pieces are absent.

When sliced Wasserstein scoring is active, an omitted `score_posterior` key is
workflow-owned. A bare call therefore receives a fresh ephemeral root instead
of the fixed key used by earlier releases. Reproducible benchmark scorecards,
including calls from `probpipe-benchmark`, should use an enclosing
`workflow_run(seed=...)` or pass `key=` explicitly.

::: probpipe.validation.standardized_mean_error

::: probpipe.validation.relative_cov_error

::: probpipe.validation.std_ratios

::: probpipe.validation.sliced_wasserstein

::: probpipe.validation.mmd

::: probpipe.validation.ksd

::: probpipe.validation.score_posterior

## Calibration and coverage

Method self-consistency checks: simulation-based calibration drives the inference
method over many `(θ★, data, posterior)` replications and tests whether the rank
of the truth among the posterior draws is uniform; interval coverage checks
whether central credible intervals contain the truth at their nominal rate.

::: probpipe.validation.simulation_based_calibration

::: probpipe.validation.SBCResult

::: probpipe.validation.interval_coverage
