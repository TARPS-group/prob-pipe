"""Nutpie-backed MCMC: standalone function + registry method."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core._dispatch import Feasibility
from ..custom_types import ArrayLike, PRNGKey
from ..distributions._empirical import EmpiricalDistribution
from ..functions import function
from ..operations._condition import InferenceMethod, _inference_run_key
from ..operations._inference_target import InferenceTarget, _described, _UnnormalizedConditional
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    extract_chain_columns,
    integer_seed,
    joint_and_given,
    observed_target,
    refuse_seed_keywords,
    require_trace_variables,
)
from ._pymc_method import _pymc_parameter_shapes

logger = logging.getLogger(__name__)

__all__ = ["NutpieNutsMethod", "condition_on_nutpie"]


# ---------------------------------------------------------------------------
# Standalone Function
# ---------------------------------------------------------------------------


@function
def condition_on_nutpie(
    model: Any,
    data: ArrayLike | None = None,
    *,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    **kwargs: Any,
) -> EmpiricalDistribution:
    """MCMC sampling via nutpie (Rust-based NUTS).

    Accepts a :class:`~probpipe.families.StanModel` or its posterior, bound
    to *data* when it is a kernel, or a :class:`~probpipe.families.PyMCModel`
    at the observed values *data*. The run's seed is drawn from a
    workflow-owned random event, so ``workflow_run(seed=...)`` reproduces the
    chains, and an unscoped call runs fresh ones.

    Parameters
    ----------
    model : StanModel or PyMCModel
        The program to sample: a Stan program, its posterior at its data, or
        a PyMC model.
    data : Mapping or None
        The data that bind a Stan program, or the observed values of a PyMC
        model.
    num_results : int
        The number of draws per chain.
    num_warmup : int
        The number of tuning steps per chain.
    num_chains : int
        The number of chains.
    **kwargs : Any
        Further keyword arguments of ``nutpie.sample``, such as
        ``progress_bar``; its ``seed`` is the run's.

    Returns
    -------
    EmpiricalDistribution
        The chains of the program's parameters, with nutpie's trace as the
        annotations.

    Raises
    ------
    ImportError
        If nutpie is not installed.
    TypeError
        If *model* is neither a Stan program nor a PyMC model, or *kwargs*
        holds ``random_seed`` or ``seed``.
    """
    refuse_seed_keywords("condition_on_nutpie", kwargs)
    return _nutpie_posterior(
        model,
        data,
        observed_target(model, data),
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        random_seed=integer_seed(_inference_run_key("nutpie_nuts")),
        **kwargs,
    )


def _nutpie_posterior(
    model: Any,
    data: Any,
    target: InferenceTarget,
    *,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    random_seed: int,
    **kwargs: Any,
) -> EmpiricalDistribution:
    """nutpie's posterior of *model* at *data*, the posterior of *target*.

    Parameters
    ----------
    model : Any
        The program: a ``StanModel``, a Stan program's posterior, or a
        ``PyMCModel``.
    data : Any
        A ``StanModel``'s data, as a dict, or the observed values of a
        ``PyMCModel``; a Stan program's posterior ignores it, since its data
        are bound.
    target : InferenceTarget
        The target of *model* at *data*, whose parameters the chains draw and
        which the posterior's provenance names as its parent.
    num_results : int
        Number of draws each chain keeps, which nutpie takes as ``draws``.
    num_warmup : int
        Number of tuning steps each chain runs, which nutpie takes as ``tune``.
    num_chains : int
        Number of chains.
    random_seed : int
        The seed nutpie's sampler takes.
    **kwargs : Any
        Further options of ``nutpie.sample``, such as ``progress_bar``.

    Returns
    -------
    EmpiricalDistribution
        The posterior on the levels ``chain`` and ``draw``, whose annotations
        hold nutpie's trace.

    Raises
    ------
    ImportError
        If nutpie is not installed.
    """
    try:
        import nutpie
    except ImportError as e:
        raise ImportError(
            "nutpie is required for condition_on_nutpie. Install it with: pip install nutpie"
        ) from e

    compiled, pymc_build = _compile_for_nutpie(model, data)

    # A PyMC model's shapes come from its build at the data, before sampling,
    # so a dynamic-RV or non-concrete model fails fast. A Stan model's come from
    # the trace.
    if pymc_build is not None:
        parameter_shapes = _pymc_parameter_shapes(target, model, pymc_build)
    else:
        parameter_shapes = None
    parameter_names = list(target.event_spec.outputs)

    trace = nutpie.sample(
        compiled,
        draws=num_results,
        tune=num_warmup,
        chains=num_chains,
        seed=random_seed,
        **kwargs,
    )
    # nutpie sorts the trace's variables by name, so the parameters are extracted
    # by name, in the target's order, without a program's transformed and
    # generated variables.
    require_trace_variables(trace, parameter_names)
    if parameter_shapes is None:
        parameter_shapes = {
            name: np.shape(trace.posterior[name].values)[2:] for name in parameter_names
        }
    chains, _ = _extract_chains(trace, num_chains, keep_names=parameter_names)

    return make_posterior(
        target,
        chains,
        method="nutpie_nuts",
        annotations=trace,
        parameter_shapes=parameter_shapes,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_for_nutpie(model: Any, data: Any) -> tuple[Any, Any | None]:
    """Compile a model for nutpie sampling.

    Returns ``(compiled, pymc_build)``. ``pymc_build`` is the
    data-conditioned ``pm.Model`` for PyMCModel targets (so the caller
    can derive a matching parameter record), and ``None`` for Stan
    targets, which nutpie compiles from the program's file and then gives
    the program's data.
    """
    from ..families._programs import StanModel, _StanPosterior, _to_numpy

    if isinstance(model, StanModel) and isinstance(data, dict):
        # Binding a Stan program's data gives the posterior.
        model, data = model._condition_on(data), None
    if isinstance(model, _StanPosterior):
        import nutpie

        from ..families._programs import _BRIDGESTAN_MAKE_ARGS

        compiled = nutpie.compile_stan_model(
            filename=model.stan_file, extra_compile_args=list(_BRIDGESTAN_MAKE_ARGS)
        )
        return compiled.with_data(**{k: _to_numpy(v) for k, v in model.data.items()}), None

    if hasattr(model, "_pymc_model"):
        import nutpie

        pymc_build = model._pymc_model(data=data)
        return nutpie.compile_pymc_model(pymc_build), pymc_build

    raise TypeError(
        f"condition_on_nutpie: model must be a StanModel or PyMCModel; got {type(model).__name__}"
    )


def _extract_chains(
    trace: Any,
    num_chains: int,
    *,
    keep_names: list[str] | None = None,
) -> tuple[list, list]:
    """Extract per-chain sample arrays from a nutpie ArviZ trace.

    Parameters
    ----------
    trace : nutpie ArviZ trace
        Trace exposing a ``posterior`` group.
    num_chains : int
        Number of chains to extract.
    keep_names : list of str or None
        If given, extract exactly these variables, in this order, instead
        of ``posterior.data_vars`` order (which nutpie sorts
        alphabetically). Callers pass the parameter names, so the chain
        columns align with the template and omit a Stan program's
        transformed and generated variables.

    Returns
    -------
    tuple[list, list]
        Per-chain concatenated arrays, and the resolved variable-name
        order.
    """
    if not hasattr(trace, "posterior"):
        raise TypeError(
            f"cannot extract chains from the nutpie trace: it has no posterior group; "
            f"got {type(trace).__name__}"
        )
    if keep_names is not None:
        param_names = list(keep_names)
    else:
        param_names = list(trace.posterior.data_vars)
    return extract_chain_columns(trace, param_names, num_chains), param_names


# ---------------------------------------------------------------------------
# Registry method
# ---------------------------------------------------------------------------


class NutpieNutsMethod(InferenceMethod):
    """nutpie-backed NUTS, registered as ``nutpie_nuts`` at priority 88.

    Applies to a Stan program's posterior at its data, and to a ``PyMCModel``
    target at its observed values; infeasible while nutpie is not installed.

    Its ``method_options`` are the draw, warmup, and chain counts, and
    ``progress_bar``, which passes to nutpie's sampler; an unset
    ``progress_bar`` leaves nutpie's default.

    Notes
    -----
    An optimised backend: Rust-implemented NUTS with in-process gradients,
    faster than every other registered NUTS backend on its applicable model
    class, so it ranks above all of them.
    """

    _method_options = ("num_chains", "num_results", "num_warmup", "progress_bar")

    def __init__(self) -> None:
        from ..families._programs import PyMCModel, _StanPosterior

        self._pymc_model_type = PyMCModel
        self._supported = (_StanPosterior, PyMCModel)

    @property
    def description(self) -> str:
        return "NUTS through nutpie, for a Stan program or a PyMC model."

    @property
    def name(self) -> str:
        return "nutpie_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return (_UnnormalizedConditional,)

    @property
    def priority(self) -> int:
        return 88

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a Stan or PyMC program, or a PyMC one at its observed values."""
        dist, given = joint_and_given(target)
        if not isinstance(dist, self._supported):
            return Feasibility(
                feasible=False,
                description=f"the model must be a StanModel or PyMCModel; got {_described(dist)}",
            )
        if given is not None and not isinstance(dist, self._pymc_model_type):
            return Feasibility(
                feasible=False,
                description=(
                    "nutpie cannot condition a StanModel on more values: it samples every "
                    "parameter at the data the StanModel was built with. Pass data to "
                    "StanModel(...) instead"
                ),
            )
        try:
            import nutpie  # noqa: F401
        except ImportError:
            return Feasibility(
                feasible=False, description="nutpie is not installed; pip install nutpie"
            )
        return Feasibility(feasible=True)

    def execute(
        self, target: InferenceTarget, /, *, key: PRNGKey, **kwargs: Any
    ) -> EmpiricalDistribution:
        """nutpie's NUTS on the program the target carries, at the observed values it binds.

        The posterior's provenance names the target as its parent.
        """
        self._check_options(kwargs)
        dist, observed = joint_and_given(target)
        seed = integer_seed(key)
        return _nutpie_posterior(dist, observed, target, random_seed=seed, **kwargs)
