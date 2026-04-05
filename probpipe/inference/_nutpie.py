"""Nutpie-backed MCMC: standalone function + registry method."""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

from ..core._registry import MethodInfo
from ..core.provenance import Provenance
from ..core.node import workflow_function
from ..custom_types import ArrayLike
from ._diagnostics import InferenceDiagnostics, extract_arviz_diagnostics
from ._mcmc_distribution import MCMCApproximateDistribution
from ._registry import InferenceMethod

logger = logging.getLogger(__name__)

__all__ = ["condition_on_nutpie", "NutpieNutsMethod"]


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
) -> MCMCApproximateDistribution:
    """MCMC sampling via nutpie (Rust-based NUTS).

    Accepts a :class:`~probpipe.modeling.StanModel` or
    :class:`~probpipe.modeling.PyMCModel`.
    """
    try:
        import nutpie
    except ImportError as e:
        raise ImportError(
            "nutpie is required for condition_on_nutpie. "
            "Install it with: pip install nutpie"
        ) from e

    compiled = _compile_for_nutpie(model, data)
    trace = nutpie.sample(
        compiled, draws=num_results, tune=num_warmup,
        chains=num_chains, seed=random_seed, **kwargs,
    )

    chains, param_names = _extract_chains(trace, num_chains)
    diagnostics = extract_arviz_diagnostics(
        trace, algorithm="nutpie_nuts",
        num_results=num_results, num_chains=num_chains,
    )

    result = MCMCApproximateDistribution(
        chains, diagnostics=diagnostics, inference_data=trace,
        name="posterior",
    )
    result.with_source(Provenance(
        "nutpie_nuts", parents=(model,),
        metadata={"num_results": num_results, "num_warmup": num_warmup,
                  "num_chains": num_chains},
    ))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_for_nutpie(model: Any, data: Any) -> Any:
    """Compile a model for nutpie sampling."""
    if hasattr(model, "_bridgestan_model"):
        import nutpie
        return nutpie.compile_stan_model(model._bridgestan_model(data=data))

    if hasattr(model, "_pymc_model"):
        import nutpie
        return nutpie.compile_pymc_model(model._pymc_model(data=data))

    raise TypeError(
        f"condition_on_nutpie does not support {type(model).__name__}. "
        f"Expected a StanModel or PyMCModel."
    )


def _extract_chains(trace: Any, num_chains: int) -> tuple[list, list]:
    """Extract per-chain sample arrays from a nutpie ArviZ trace."""
    if hasattr(trace, "posterior"):
        posterior = trace.posterior
        param_names = list(posterior.data_vars)
        chains = []
        for c in range(num_chains):
            chain_arrays = []
            for name in param_names:
                vals = posterior[name].values[c]
                if vals.ndim == 1:
                    vals = vals[:, None]
                else:
                    vals = vals.reshape(vals.shape[0], -1)
                chain_arrays.append(vals)
            chains.append(jnp.concatenate(chain_arrays, axis=-1))
        return chains, param_names
    raise TypeError(
        f"Cannot extract chains from nutpie trace of type {type(trace).__name__}"
    )


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
        return 80

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._supported):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires StanModel or PyMCModel")
        try:
            import nutpie  # noqa: F401
        except ImportError:
            return MethodInfo(feasible=False, method_name=self.name,
                              description="nutpie not installed")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> MCMCApproximateDistribution:
        return condition_on_nutpie._func(dist, observed, **kwargs)
