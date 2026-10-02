"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import arviz_base as azb
import jax.numpy as jnp
import numpy as np

from ..core._dispatch import Feasibility
from ..core._specs import OutputSpec
from ..families._programs import _parameter_record_at, _StanPosterior
from ..operations._condition import InferenceMethod
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import integer_seed, run_seed


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

    _method_options = ("num_chains", "num_results", "num_warmup", "random_seed")

    @property
    def name(self) -> str:
        return "cmdstan_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (_StanPosterior,)

    @property
    def priority(self) -> int:
        return 82

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a Stan program's posterior at its data, and cmdstanpy is installed."""
        if not isinstance(target, _StanPosterior):
            return Feasibility(feasible=False, description="Requires a StanModel's posterior")
        if "cmdstanpy" not in sys.modules and importlib.util.find_spec("cmdstanpy") is None:
            return Feasibility(feasible=False, description="cmdstanpy is not installed")
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> ApproximateDistribution:
        """Stan's NUTS on the target's program at its data, through cmdstanpy.

        The posterior keeps the target's parameter record: each chain holds the
        draws of the parameter blocks alone, each in its own shape, and no
        sampler, transformed, or generated column.
        """
        self._check_options(kwargs)
        cmdstanpy = _import_cmdstanpy()

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 1000)
        num_chains = kwargs.get("num_chains", 4)
        random_seed = integer_seed(run_seed(kwargs, self.name))

        model = cmdstanpy.CmdStanModel(stan_file=target.stan_file)
        fit = model.sample(
            data=dict(target.data),
            chains=num_chains,
            iter_sampling=num_results,
            iter_warmup=num_warmup,
            seed=random_seed,
            show_console=False,
        )

        # stan_variable concatenates the chains in chain order, each draw in the
        # variable's own shape.
        names = list(target.event_spec.components)
        draws = {name: np.asarray(fit.stan_variable(name)) for name in names}
        per_chain = len(draws[names[0]]) // num_chains
        chains = [
            jnp.concatenate(
                [
                    jnp.reshape(draws[name][c * per_chain : (c + 1) * per_chain], (per_chain, -1))
                    for name in names
                ],
                axis=-1,
            )
            for c in range(num_chains)
        ]
        shapes = {name: draws[name].shape[1:] for name in names}
        event_spec = OutputSpec(_parameter_record_at(target.event_spec.spec, shapes))

        return make_posterior(
            chains,
            parents=(target,),
            method="cmdstan_nuts",
            annotations=azb.from_cmdstanpy(fit),
            event_spec=event_spec,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )
