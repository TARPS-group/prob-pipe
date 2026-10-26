> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Diagnostics

The diagnostic functions compute a diagnostic of a posterior and record it in the posterior's annotations, and `posterior.diagnostics` returns a view that reads them.
This page documents the MCMC, predictive, and leave-one-out diagnostics and the views, and the checks of an approximation against a reference posterior are on [Validation](validation.md).

## MCMC diagnostics

::: probpipe.diagnostics.add_rhat
    options:
      show_root_full_path: true

::: probpipe.diagnostics.add_ess
    options:
      show_root_full_path: true

::: probpipe.diagnostics.add_mcse
    options:
      show_root_full_path: true

::: probpipe.diagnostics.add_mcmc_diagnostics
    options:
      show_root_full_path: true

## Predictive and leave-one-out diagnostics

::: probpipe.diagnostics.add_ppc
    options:
      show_root_full_path: true

::: probpipe.diagnostics.add_loo
    options:
      show_root_full_path: true

## Views

::: probpipe.diagnostics.DiagnosticsView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.MCMCView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.PPCView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.LOOView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.DiagnosticRunView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.NotComputed
    options:
      show_root_full_path: true

## View helpers

::: probpipe.diagnostics.views.DataTreeView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.views.DatasetView
    options:
      show_root_full_path: true

::: probpipe.diagnostics.views.read_scalar
    options:
      show_root_full_path: true

::: probpipe.diagnostics.views.read_indexed
    options:
      show_root_full_path: true

::: probpipe.diagnostics.views.read_json_attr
    options:
      show_root_full_path: true
