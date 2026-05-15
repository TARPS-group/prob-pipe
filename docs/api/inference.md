# Modeling and inference

Probabilistic models, likelihoods, inference methods, iterative transformations,
and predictive checks.

The extension surface — `MethodRegistry`, `Method`, `MethodInfo`, and the
custom-method walkthrough — lives on [Extending ProbPipe → Custom inference
methods](extending.md#custom-inference-methods).

## Models

`ProbabilisticModel` is the abstract base for models; `SimpleModel` is the
prior + likelihood workhorse used in the tutorials. `SimpleGenerativeModel`
is the simulator-only wrapper for likelihood-free / SBI workflows.

::: probpipe.ProbabilisticModel

::: probpipe.SimpleModel

::: probpipe.SimpleGenerativeModel

## Likelihoods

::: probpipe.Likelihood

::: probpipe.GenerativeLikelihood

::: probpipe.GLMLikelihood

::: probpipe.IncrementalConditioner

## Inference methods

The **inference method registry** is the dispatch system behind
[`condition_on()`](operations.md#conditioning). When you call
`condition_on(model, data)`, the registry auto-selects the highest-priority
feasible method. Pass `method="tfp_nuts"` (or any registered name) to
override.

```python
from probpipe import condition_on
from probpipe.inference import inference_method_registry

# Auto-select best method
posterior = condition_on(model, data, num_results=1000)

# Override with a specific method
posterior = condition_on(model, data, method="tfp_rwmh", num_results=1000)

# List available methods (highest priority first)
inference_method_registry.list_methods()

# Adjust priority so RWMH is tried before NUTS
inference_method_registry.set_priorities(tfp_rwmh=200, tfp_nuts=50)
```

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

`ApproximateDistribution` is the common posterior result type returned by
`condition_on`. `rwmh` and `condition_on_nutpie` are direct entry points
for the corresponding MCMC backends. `sbi_learn_conditional` and
`sbi_learn_likelihood` are likelihood-free entry points for
`SimpleGenerativeModel` instances.

::: probpipe.ApproximateDistribution

::: probpipe.rwmh

::: probpipe.condition_on_nutpie

::: probpipe.sbi_learn_conditional

::: probpipe.sbi_learn_likelihood

::: probpipe.inference_method_registry
    options:
      show_root_heading: true

## Iterative transformations

Step functions are folded over inputs to produce sequences of distributions.
`iterate` is the workhorse; `with_conversion` and `with_resampling` are
step-function wrappers that compose with it.

::: probpipe.iterate

::: probpipe.with_conversion

::: probpipe.with_resampling

## Predictive checks

::: probpipe.predictive_check
