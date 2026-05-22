# Modeling and inference

Models, likelihoods, inference methods, iterative transformations, and
predictive checks. Registering a new inference method is documented under
[Extending ProbPipe → Custom inference methods](extending.md#custom-inference-methods).

## Models

::: probpipe.ProbabilisticModel

::: probpipe.SimpleModel

::: probpipe.SimpleGenerativeModel

## Likelihoods

::: probpipe.Likelihood

::: probpipe.ConditionallyIndependentLikelihood

::: probpipe.GenerativeLikelihood

::: probpipe.GLMLikelihood

::: probpipe.IncrementalConditioner

## Inference methods

[`condition_on`](operations.md#conditioning) dispatches through the
inference-method registry: methods are tried in descending priority order
and the first whose `check()` returns `feasible=True` runs. Pass
`method="<name>"` to override the auto-selection;
`inference_method_registry.set_priorities(...)` reorders the table at
runtime.

Priorities follow a single-axis convention: values above `50` mark *exact*
methods, values in `(0, 50]` mark *inexact* methods, and `0` means
opt-in only (selectable by name but skipped during auto-dispatch). The
five-axis selection criteria and the tier ranges contributors should use
when choosing a number for a new method are documented under
[Extending ProbPipe → Setting priority for a new method](extending.md#setting-priority-for-a-new-method).

**Built-in methods:**

| Name | Priority | Requires | Backend |
|------|----------|----------|---------|
| `nutpie_nuts` | 88 | `StanModel` or `PyMCModel` + nutpie | nutpie |
| `blackjax_nuts` | 85 | `SupportsLogProb` + JAX-traceable | BlackJAX |
| `cmdstan_nuts` | 82 | `StanModel` + cmdstanpy | CmdStan |
| `pymc_nuts` | 82 | `PyMCModel` + pymc | PyMC |
| `tfp_rwmh` | 55 | `SupportsLogProb` | hand-rolled Python |
| `blackjax_sgld` | 45 | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` | BlackJAX |
| `sbijax_smcabc` | 5 | `SimpleGenerativeModel` + sbijax | sbijax |
| `blackjax_hmc` | 0 | `SupportsLogProb` + JAX-traceable | BlackJAX (opt-in only) |
| `blackjax_sghmc` | 0 | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` | BlackJAX (opt-in only) |
| `pymc_advi` | 0 | `PyMCModel` + pymc | PyMC (opt-in only) |
| `tfp_nuts` | 0 | `SupportsLogProb` + JAX-traceable | TFP (opt-in only) |
| `tfp_hmc` | 0 | `SupportsLogProb` + JAX-traceable | TFP (opt-in only) |

::: probpipe.ApproximateDistribution

::: probpipe.rwmh

::: probpipe.condition_on_nutpie

::: probpipe.sbi_learn_conditional

::: probpipe.sbi_learn_likelihood

::: probpipe.MinibatchedDistribution

::: probpipe.inference_method_registry
    options:
      show_root_heading: true

## Iterative transformations

Step functions folded over inputs by `iterate`, with `with_conversion` and
`with_resampling` as step-function wrappers.

::: probpipe.iterate

::: probpipe.with_conversion

::: probpipe.with_resampling

## Predictive checks

::: probpipe.predictive_check
