# Inference

## Modeling

::: probpipe.modeling.Likelihood

::: probpipe.modeling.GenerativeLikelihood

::: probpipe.modeling.IncrementalConditioner

## Probabilistic Models

::: probpipe.modeling.ProbabilisticModel

::: probpipe.modeling.SimpleModel

## Inference Method Registry

The **inference method registry** is the dispatch system behind
[`condition_on()`](operations.md#conditioning).  When you call
`condition_on(model, data)`, the registry auto-selects the highest-priority
feasible method.  Pass `method="tfp_nuts"` (or any registered name) to
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

### Built-in methods

| Name | Priority | Requires | Backend |
|------|----------|----------|---------|
| `tfp_nuts` | 100 | `SupportsLogProb` + JAX-traceable | TFP |
| `tfp_hmc` | 90 | `SupportsLogProb` + JAX-traceable | TFP |
| `nutpie_nuts` | 80 | StanModel or PyMCModel + nutpie | nutpie |
| `cmdstan_nuts` | 70 | StanModel + cmdstanpy | CmdStan |
| `pymc_nuts` | 60 | PyMCModel + pymc | PyMC |
| `tfp_rwmh` | 50 | `SupportsLogProb` | TFP |
| `pymc_advi` | 35 | PyMCModel + pymc | PyMC |

::: probpipe.core._registry.MethodRegistry
    options:
      show_root_heading: true
      heading_level: 3

::: probpipe.core._registry.Method
    options:
      show_root_heading: true
      heading_level: 3

::: probpipe.core._registry.MethodInfo
    options:
      show_root_heading: true
      heading_level: 3

## MCMC

::: probpipe.inference.InferenceDiagnostics

::: probpipe.inference.MCMCApproximateDistribution

::: probpipe.inference.rwmh

::: probpipe.inference.condition_on_nutpie
