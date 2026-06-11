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
| `blackjax_elliptical_slice` | 75 | `SimpleModel` + Gaussian prior + JAX-traceable likelihood | BlackJAX |
| `blackjax_rwmh` | 55 | `SupportsLogProb` (eager fallback for non-traceable targets) | BlackJAX |
| `blackjax_sgld` | 45 | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` | BlackJAX |
| `blackjax_hmc` | 0 | `SupportsLogProb` + JAX-traceable | BlackJAX (opt-in only) |
| `blackjax_sghmc` | 0 | `SimpleModel` + `ConditionallyIndependentLikelihood` + `batch_size=` | BlackJAX (opt-in only) |
| `pymc_advi` | 0 | `PyMCModel` + pymc | PyMC (opt-in only) |
| `tfp_nuts` | 0 | `SupportsLogProb` + JAX-traceable | TFP (opt-in only) |
| `tfp_hmc` | 0 | `SupportsLogProb` + JAX-traceable | TFP (opt-in only) |

::: probpipe.ApproximateDistribution

::: probpipe.rwmh

::: probpipe.elliptical_slice

::: probpipe.condition_on_nutpie

::: probpipe.MinibatchedDistribution

::: probpipe.inference_method_registry
    options:
      show_root_heading: true

## Amortized SBI

Simulation-based inference with trained amortized estimators (requires the
`[bayesflow]` extra; Python 3.12–3.13 only). All three learners take the same
inputs — a `RecordDistribution` prior and a `GenerativeLikelihood` simulator —
and share one offline simulation pipeline; they differ in what the network
estimates, and therefore in how posterior draws are produced:

| Learner | Estimates | Returns | Posterior draws via |
|---|---|---|---|
| `learn_amortized_posterior` | the posterior directly (NPE, FMPE, or CMPE) | `BayesFlowModel` | `condition_on(model, observed)` — one network forward pass, no MCMC |
| `learn_amortized_likelihood` | the likelihood (NLE) | `BayesFlowLikelihood` | `condition_on(SimpleModel(prior, learned), data)` — gradient-based MCMC through the learned density |
| `learn_amortized_ratio` | the likelihood-to-evidence ratio (NRE) | `BayesFlowRatio` | same as NLE; the evidence constant cancels in MCMC |

Choosing between them: NPE is amortized over observations — the one trained
model conditions on any (single) observation with no retraining and no cost
beyond a forward pass. NLE and NRE instead pay an MCMC run per dataset (the
learned `log_likelihood` is jax-grad-transparent, so the standard NUTS
machinery applies) but handle **multi-observation datasets** natively: both
wrappers are `ConditionallyIndependentLikelihood`s, so per-row scores sum.
Between those two, NLE learns an actual density and needs continuous
observations of at least two dimensions; NRE's classifier has no dimension
minimum and handles discrete-valued observations, but its values are
log-ratios — valid for conditioning, **not** for absolute-likelihood uses
such as model comparison or LOO/WAIC.

::: probpipe.learn_amortized_posterior

::: probpipe.BayesFlowModel

::: probpipe.learn_amortized_likelihood

::: probpipe.BayesFlowLikelihood

::: probpipe.learn_amortized_ratio

::: probpipe.BayesFlowRatio

## Iterative transformations

Step functions folded over inputs by `iterate`, with `with_conversion` and
`with_resampling` as step-function wrappers.

::: probpipe.iterate

::: probpipe.with_conversion

::: probpipe.with_resampling

## Predictive checks

::: probpipe.predictive_check
