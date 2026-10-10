> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Inference methods

This page documents the inference methods a user calls directly, the functions of amortized and sequential inference, and the registry from which `condition_on` selects a method.
`condition_on` itself is on [Operations](operations.md), and `MinibatchedDistribution`, from which the stochastic-gradient methods estimate gradients, is on [Random functions](random_functions.md).
`InferenceMethod`, which a new method subclasses, is on [Registries for extensions](extending.md).

## MCMC

::: probpipe.condition_on_nutpie

::: probpipe.rwmh

::: probpipe.elliptical_slice

## Amortized simulation-based inference

::: probpipe.learn_amortized_posterior

::: probpipe.learn_amortized_likelihood

::: probpipe.BayesFlowLikelihood

::: probpipe.learn_amortized_ratio

::: probpipe.BayesFlowRatio

## Sequential updating

::: probpipe.iterate

::: probpipe.with_conversion

::: probpipe.with_resampling

## The inference-method registry

`inference_method_registry.list_methods()` names the registered methods, and `condition_on.with_options(method=name)` runs the method of that name instead of the one the registry selects.

::: probpipe.inference_method_registry
