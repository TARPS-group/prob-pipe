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
from ._inference_utils import integer_seed, joint_and_given, nuts_controls


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


def _add_reached_max_treedepth(trace: Any, max_tree_depth: int) -> None:
    """Mark, in place, each draw of *trace* whose tree reached *max_tree_depth*, as ``reached_max_treedepth``.

    A trace with no ``tree_depth`` in its ``sample_stats`` group is left unmarked.
    """
    try:
        stats = trace["sample_stats"]
    except (KeyError, TypeError):
        return
    if "tree_depth" in stats.data_vars:
        stats["reached_max_treedepth"] = stats["tree_depth"] >= max_tree_depth


class CmdStanNutsMethod(InferenceMethod):
    """CmdStanPy-backed NUTS, registered as ``cmdstan_nuts`` at priority 82.

    Applies to a Stan program's posterior at its data, the target that
    binding a ``StanModel``'s data gives; cmdstanpy is imported at execution.
    Its ``method_options`` are:

    - ``num_results``, ``num_warmup``, and ``num_chains``: the draw, warmup,
      and chain counts, which are 1000, 1000, and 4 unless set;
    - ``target_accept_prob``: the acceptance probability that Stan's step-size
      adaptation targets, a number strictly between 0 and 1, which CmdStan
      receives as ``adapt_delta`` and which is Stan's 0.8 unless set;
    - ``max_tree_depth``: the maximum depth of the NUTS tree, a positive
      integer, which CmdStan receives as ``max_treedepth`` and which is
      Stan's 10 unless set.

    The run's sample statistics hold Stan's ``treedepth__`` under ArviZ's name
    ``tree_depth``, and ``reached_max_treedepth``, which marks each draw whose
    tree reached ``max_tree_depth``. Stan records no reason a trajectory
    stopped, so a draw whose trajectory turned at its last doubling also
    counts.

    Notes
    -----
    An optimised backend: Stan-compiled NUTS through the cmdstanpy subprocess
    interface. Below ``nutpie_nuts`` (88) and ``blackjax_nuts`` (85) because
    of the subprocess overhead, and tied with ``pymc_nuts`` (82), which
    applies to a disjoint model class.
    """

    _method_options = (
        "max_tree_depth",
        "num_chains",
        "num_results",
        "num_warmup",
        "target_accept_prob",
    )

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

        Parameters
        ----------
        target : InferenceTarget
            The law to sample, the model at its observed values.
        key : PRNGKey
            The run's random key, which seeds the sampler.
        **options : Any
            The call's ``method_options``, which ``CmdStanNutsMethod`` lists.

        Returns
        -------
        EmpiricalDistribution
            The posterior, with its atoms on the levels ``chain`` and ``draw``
            and the sampler's trace as its annotations.

        Raises
        ------
        ImportError
            If cmdstanpy is not installed.
        TypeError
            If an option is one the method does not read, or
            ``target_accept_prob`` or ``max_tree_depth`` has the wrong type.
        ValueError
            If ``target_accept_prob`` is not strictly between 0 and 1, or
            ``max_tree_depth`` is less than 1.
        """
        self._check_options(options)
        target_accept_prob, max_tree_depth = nuts_controls(self.name, options)
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
            adapt_delta=target_accept_prob,
            max_treedepth=max_tree_depth,
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

        trace = azb.from_cmdstanpy(fit)
        _add_reached_max_treedepth(trace, max_tree_depth)
        return make_posterior(
            target,
            chains,
            method="cmdstan_nuts",
            annotations=trace,
            parameter_shapes=shapes,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )
