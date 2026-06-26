"""Nutpie-backed MCMC: standalone function + registry method."""

from __future__ import annotations

import logging
from typing import Any

from ..core._registry import MethodInfo
from ..core.node import workflow_function
from ..custom_types import ArrayLike
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import extract_chain_columns, posterior_var_order
from ._registry import InferenceMethod

logger = logging.getLogger(__name__)

__all__ = ["NutpieNutsMethod", "condition_on_nutpie"]


# ---------------------------------------------------------------------------
# Standalone WorkflowFunction
# ---------------------------------------------------------------------------


@workflow_function
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

    # Build the template in canonical field order from the conditioned
    # build before sampling (fail fast on a dynamic-RV / non-concrete
    # model). Stan models carry their own event_template, if any.
    if pymc_build is not None:
        param_names = list(model._conditioned_param_names(pymc_build))
        event_template = model._event_template_for(pymc_build, param_names)
    else:
        param_names = None
        event_template = getattr(model, "event_template", None)

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
    # to the template by name, so we don't depend on the orders matching.
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
        auxiliary=trace,
        event_template=event_template,
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
    can derive a matching ``event_template``), and ``None`` for Stan
    targets.
    """
    if hasattr(model, "_bridgestan_model"):
        import nutpie

        if isinstance(data, dict):
            # Keep the data the model was built with — StanModel(file, data=...)
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
    """Registry method for nutpie-backed NUTS."""

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
        # Tier 81-90 (optimised backend; Rust-implemented NUTS with
        # in-process gradients, faster than every other registered
        # NUTS backend on its applicable model class). Top of the
        # tier at 88.
        return 88

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._supported):
            return MethodInfo(
                feasible=False, method_name=self.name, description="Requires StanModel or PyMCModel"
            )
        try:
            import nutpie  # noqa: F401
        except ImportError:
            return MethodInfo(
                feasible=False, method_name=self.name, description="nutpie not installed"
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        return condition_on_nutpie._func(dist, observed, **kwargs)
