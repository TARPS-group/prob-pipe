"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

from typing import Any

import arviz_base as azb
import jax.numpy as jnp

from ..core._dispatch import Feasibility
from ..families._programs import _StanPosterior
from ..operations._condition import InferenceMethod
from ._approximate_distribution import ApproximateDistribution, make_posterior


def _import_cmdstanpy():
    """Import cmdstanpy or raise a helpful error."""
    try:
        import cmdstanpy

        return cmdstanpy
    except ImportError as e:
        raise ImportError(
            "cmdstanpy is required for Stan sampling. "
            "Install it with: pip install probpipe-core[stan]"
        ) from e


class CmdStanNutsMethod(InferenceMethod):
    """CmdStanPy-backed NUTS, registered as ``cmdstan_nuts`` at priority 82.

    Applies to a Stan program's posterior at its data, the target a
    ``StanModel`` curries to; cmdstanpy is imported at execution.

    Notes
    -----
    An optimised backend: Stan-compiled NUTS through the cmdstanpy subprocess
    interface. Below ``nutpie_nuts`` (88) and ``blackjax_nuts`` (85) because
    of the subprocess overhead, and tied with ``pymc_nuts`` (82), which
    applies to a disjoint model class.
    """

    @property
    def name(self) -> str:
        return "cmdstan_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (_StanPosterior,)

    @property
    def priority(self) -> int:
        return 82

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a Stan program's posterior at its data."""
        if not isinstance(target, _StanPosterior):
            return Feasibility(feasible=False, description="Requires a StanModel's posterior")
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> ApproximateDistribution:
        """Stan's NUTS on the target's program at its data, through cmdstanpy."""
        cmdstanpy = _import_cmdstanpy()

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 1000)
        num_chains = kwargs.get("num_chains", 4)
        random_seed = kwargs.get("random_seed", 0)

        model = cmdstanpy.CmdStanModel(stan_file=target.stan_file)
        fit = model.sample(
            data=dict(target.data),
            chains=num_chains,
            iter_sampling=num_results,
            iter_warmup=num_warmup,
            seed=random_seed,
            show_console=False,
        )

        chains = []
        for c in range(num_chains):
            chain_draws = jnp.asarray(fit.draws(concat_chains=False)[c])
            chains.append(chain_draws)

        inference_data = azb.from_cmdstanpy(fit)

        return make_posterior(
            chains,
            parents=(target,),
            algorithm="cmdstan_nuts",
            annotations=inference_data,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )
