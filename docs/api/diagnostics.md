# Diagnostics

Utilities for attaching Bayesian diagnostics to fitted posteriors in place.
The public diagnostic operations mutate `posterior._auxiliary` and return
`None`; users normally read the results back through `posterior.diagnostics`.
ArviZ-compatible data are exposed through `posterior.arviz_data`.

## MCMC diagnostics

Use these operations to compute convergence and Monte Carlo accuracy summaries
for a fitted posterior.

::: probpipe.diagnostics.add_rhat

::: probpipe.diagnostics.add_ess

::: probpipe.diagnostics.add_mcse

::: probpipe.diagnostics.add_mcmc_diagnostics

## Predictive and LOO diagnostics

::: probpipe.diagnostics.add_ppc

::: probpipe.diagnostics.add_loo

## Diagnostic views

`posterior.diagnostics` returns a structured view over the diagnostics subtree.
The concrete views expose MCMC, posterior predictive check, and LOO results
without requiring users to traverse `_auxiliary` directly.

::: probpipe.diagnostics.DiagnosticsView

::: probpipe.diagnostics.MCMCView

::: probpipe.diagnostics.PPCView

::: probpipe.diagnostics.LOOView

::: probpipe.diagnostics.DiagnosticRunView

::: probpipe.diagnostics.NotComputed
