"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

from typing import Any

import arviz as az
import jax.numpy as jnp

from ..core._registry import MethodInfo
from ._mcmc_distribution import MCMCApproximateDistribution, make_posterior
from ._registry import InferenceMethod


def _ensure_cmdstanpy():
    """Import cmdstanpy or raise a helpful error."""
    try:
        import cmdstanpy
        return cmdstanpy
    except ImportError as e:
        raise ImportError(
            "cmdstanpy is required for Stan sampling. "
            "Install it with: pip install probpipe[stan]"
        ) from e


class CmdStanNutsMethod(InferenceMethod):
    """CmdStanPy-backed NUTS for Stan models."""

    def __init__(self) -> None:
        from ..modeling._stan import StanModel
        self._model_type = StanModel

    @property
    def name(self) -> str:
        return "cmdstan_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type,)

    @property
    def priority(self) -> int:
        return 70

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._model_type):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires StanModel")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> MCMCApproximateDistribution:
        cmdstanpy = _ensure_cmdstanpy()

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 1000)
        num_chains = kwargs.get("num_chains", 4)
        random_seed = kwargs.get("random_seed", 0)

        # Merge model's fixed data with observed values
        data = {**(dist._stan_data or {})}
        if isinstance(observed, dict):
            data.update(observed)

        model = cmdstanpy.CmdStanModel(stan_file=dist._stan_file)
        fit = model.sample(
            data=data,
            chains=num_chains,
            iter_sampling=num_results,
            iter_warmup=num_warmup,
            seed=random_seed,
            show_console=False,
        )

        chains = []
        for c in range(num_chains):
            chain_draws = jnp.asarray(
                fit.draws(concat_chains=False)[c], dtype=jnp.float32,
            )
            chains.append(chain_draws)

        inference_data = az.from_cmdstanpy(fit)

        return make_posterior(
            chains, parents=(dist,), algorithm="cmdstan_nuts",
            inference_data=inference_data,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )
