"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import arviz_base as azb
import jax.numpy as jnp
import numpy as np

from ..core._dispatch import Feasibility
from ..custom_types import PRNGKey
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..families._programs import _StanPosterior
from ..operations._condition import InferenceMethod
from ..operations._inference_target import InferenceTarget
from ._approximate_distribution import make_posterior
from ._inference_utils import integer_seed, joint_and_given


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

    Applies to a Stan program's posterior at its data, the target that
    binding a ``StanModel``'s data gives; cmdstanpy is imported at execution.

    Notes
    -----
    An optimised backend: Stan-compiled NUTS through the cmdstanpy subprocess
    interface. Below ``nutpie_nuts`` (88) and ``blackjax_nuts`` (85) because
    of the subprocess overhead, and tied with ``pymc_nuts`` (82), which
    applies to a disjoint model class.
    """

    _method_options = ("num_chains", "num_results", "num_warmup")

    @property
    def description(self) -> str:
        return "NUTS through CmdStan, for a Stan program's posterior at its data."

    @property
    def name(self) -> str:
        return "cmdstan_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return 82

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a Stan program's posterior at its data, and cmdstanpy is installed."""
        program_posterior, given = joint_and_given(target)
        if not isinstance(program_posterior, _StanPosterior) or given is not None:
            return Feasibility(
                feasible=False, description="the target must be a StanModel's posterior"
            )
        if "cmdstanpy" not in sys.modules and importlib.util.find_spec("cmdstanpy") is None:
            return Feasibility(
                feasible=False,
                description="cmdstanpy is not installed; pip install probpipe-core[stan]",
            )
        return Feasibility(feasible=True)

    def execute(
        self, target: InferenceTarget, /, *, key: PRNGKey, **options: Any
    ) -> EmpiricalDistribution:
        """Stan's NUTS on the target's program at its data, through cmdstanpy.

        The posterior keeps the target's parameter record: each chain holds the
        draws of the parameter blocks alone, each in its own shape, and no
        sampler, transformed, or generated column.
        """
        self._check_options(options)
        cmdstanpy = _import_cmdstanpy()

        num_results = options.get("num_results", 1000)
        num_warmup = options.get("num_warmup", 1000)
        num_chains = options.get("num_chains", 4)
        random_seed = integer_seed(key)
        program_posterior, _ = joint_and_given(target)

        model = cmdstanpy.CmdStanModel(stan_file=program_posterior.stan_file)
        fit = model.sample(
            data=dict(program_posterior.data),
            chains=num_chains,
            iter_sampling=num_results,
            iter_warmup=num_warmup,
            seed=random_seed,
            show_console=False,
        )

        # stan_variable concatenates the chains in chain order, each draw in the
        # variable's own shape.
        names = list(program_posterior.event_spec.outputs)
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

        return make_posterior(
            target,
            chains,
            method="cmdstan_nuts",
            annotations=azb.from_cmdstanpy(fit),
            parameter_shapes=shapes,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )
