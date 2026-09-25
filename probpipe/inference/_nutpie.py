"""Nutpie-backed MCMC: standalone function + registry method."""

from __future__ import annotations

import logging
from typing import Any

from ..core._dispatch import Feasibility
from ..core._specs import OutputSpec
from ..core.node import function
from ..custom_types import ArrayLike
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import extract_chain_columns, posterior_var_order
from ._registry import InferenceMethod

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
    random_seed: int = 0,
    **kwargs: Any,
) -> ApproximateDistribution:
    """MCMC sampling via nutpie (Rust-based NUTS).

    Accepts a :class:`~probpipe.modeling.StanModel` or
    :class:`~probpipe.modeling.PyMCModel`.
    """
    try:
        import nutpie
    except ImportError as e:
        raise ImportError(
            "nutpie is required for condition_on_nutpie. Install it with: pip install nutpie"
        ) from e

    compiled, pymc_build = _compile_for_nutpie(model, data)

    # Build the parameter record in canonical field order from the
    # conditioned build before sampling (fail fast on a dynamic-RV /
    # non-concrete model). A Stan model declares its own parameters.
    if pymc_build is not None:
        param_names = list(model._conditioned_param_names(pymc_build))
        event_spec = OutputSpec(model._parameter_record_for(pymc_build, param_names))
    else:
        param_names = None
        event_spec = getattr(model, "event_spec", None)

    trace = nutpie.sample(
        compiled,
        draws=num_results,
        tune=num_warmup,
        chains=num_chains,
        seed=random_seed,
        **kwargs,
    )

    # Extract in nutpie's natural ``data_vars`` order (it sorts
    # alphabetically); ``field_order`` lets make_posterior realign columns
    # to the parameters by name, so we don't depend on the orders matching.
    if param_names is not None:
        field_order = posterior_var_order(trace, param_names)
        chains, _ = _extract_chains(trace, num_chains, keep_names=field_order)
    else:
        field_order = None
        chains, _ = _extract_chains(trace, num_chains)

    return make_posterior(
        chains,
        parents=(model,),
        algorithm="nutpie_nuts",
        annotations=trace,
        event_spec=event_spec,
        field_order=field_order,
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
    targets.
    """
    if hasattr(model, "_bridgestan_model"):
        import nutpie

        if isinstance(data, dict):
            # Keep the data the model was built with — StanModel(name, file, data=...)
            # stores it on ``_stan_data`` — and let the conditioning data
            # override key-by-key, mirroring the CmdStan method. Without this
            # the rebuilt BridgeStan model would see only the conditioning data
            # and fail on (or silently misuse) the construction-time variables.
            data = {**(model._stan_data or {}), **data} or None
        return nutpie.compile_stan_model(model._bridgestan_model(data=data)), None

    if hasattr(model, "_pymc_model"):
        import nutpie

        pymc_build = model._pymc_model(data=data)
        return nutpie.compile_pymc_model(pymc_build), pymc_build

    raise TypeError(
        f"condition_on_nutpie does not support {type(model).__name__}. "
        f"Expected a StanModel or PyMCModel."
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
        alphabetically). PyMC callers pass the param names so chain
        columns align with the template; Stan callers pass ``None``.

    Returns
    -------
    tuple[list, list]
        Per-chain concatenated arrays, and the resolved variable-name
        order.
    """
    if not hasattr(trace, "posterior"):
        raise TypeError(f"Cannot extract chains from nutpie trace of type {type(trace).__name__}")
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

    Applies to a ``StanModel`` or ``PyMCModel`` whose modeling backend is
    installed; infeasible while nutpie is not installed.

    Notes
    -----
    An optimised backend: Rust-implemented NUTS with in-process gradients,
    faster than every other registered NUTS backend on its applicable model
    class, so it ranks above all of them.
    """

    def __init__(self) -> None:
        types: list[type] = []
        try:
            from ..modeling._stan import StanModel

            types.append(StanModel)
        except ImportError:
            pass
        try:
            from ..modeling._pymc import PyMCModel

            types.append(PyMCModel)
        except ImportError:
            pass
        self._supported = tuple(types)

    @property
    def name(self) -> str:
        return "nutpie_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return self._supported

    @property
    def priority(self) -> int:
        return 88

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> Feasibility:
        if not isinstance(dist, self._supported):
            return Feasibility(feasible=False, description="Requires StanModel or PyMCModel")
        try:
            import nutpie  # noqa: F401
        except ImportError:
            return Feasibility(feasible=False, description="nutpie not installed")
        return Feasibility(feasible=True)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        return condition_on_nutpie.apply(dist, observed, **kwargs)
