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

**Built-in methods:**

| Name | Priority | Requires | Backend |
|------|----------|----------|---------|
| `tfp_nuts` | 100 | `SupportsLogProb` + JAX-traceable | TFP |
| `tfp_hmc` | 90 | `SupportsLogProb` + JAX-traceable | TFP |
| `nutpie_nuts` | 80 | `StanModel` or `PyMCModel` + nutpie | nutpie |
| `cmdstan_nuts` | 70 | `StanModel` + cmdstanpy | CmdStan |
| `pymc_nuts` | 60 | `PyMCModel` + pymc | PyMC |
| `tfp_rwmh` | 50 | `SupportsLogProb` | TFP |
| `sbijax_smcabc` | 40 | `SimpleGenerativeModel` + sbijax | sbijax |
| `pymc_advi` | 35 | `PyMCModel` + pymc | PyMC |
| `blackjax_sgld` | 30 | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` | BlackJAX |
| `blackjax_sghmc` | 25 | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` | BlackJAX |

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
