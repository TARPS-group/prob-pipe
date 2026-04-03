"""RWMH inference method for the registry."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb
from ._registry import InferenceMethod
from ._tfp_mcmc import _get_init_state, _get_prior


class TFPRWMHMethod(InferenceMethod):
    """Gradient-free random-walk Metropolis–Hastings.

    Feasible when the distribution (or its prior) supports
    ``SupportsLogProb``.  Used as a fallback when gradient-based
    methods are not applicable.
    """

    @property
    def name(self) -> str:
        return "tfp_rwmh"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return 50

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        prior = _get_prior(dist)
        if not isinstance(prior, SupportsLogProb):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SupportsLogProb")
        if observed is not None and isinstance(observed, dict):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Does not support dict-based conditioning")
        try:
            _get_init_state(prior, kwargs.get("init"), observed)
        except Exception as e:
            return MethodInfo(feasible=False, method_name=self.name,
                              description=str(e))
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        from ._rwmh import rwmh as _rwmh

        prior = _get_prior(dist)
        log_prob_fn = None
        if hasattr(dist, "_likelihood"):
            lik = dist._likelihood
            log_prob_fn = lambda params, d: lik.log_likelihood(params=params, data=d)

        init = kwargs.get("init")
        if init is None:
            init = _get_init_state(prior, None, observed)

        return _rwmh._func(
            prior, observed,
            log_prob_fn=log_prob_fn,
            num_results=kwargs.get("num_results", 1000),
            num_warmup=kwargs.get("num_warmup", 500),
            num_chains=kwargs.get("num_chains", 1),
            step_size=kwargs.get("step_size", 0.1),
            init=init,
            random_seed=kwargs.get("random_seed", 0),
        )
