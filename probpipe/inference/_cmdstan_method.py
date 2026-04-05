"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core._registry import MethodInfo
from ..core.provenance import Provenance
from ._diagnostics import InferenceDiagnostics
from ._mcmc_distribution import MCMCApproximateDistribution
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


def _extract_cmdstan_diagnostics(
    fit: Any, num_results: int, num_chains: int,
) -> InferenceDiagnostics:
    """Extract diagnostics from a CmdStanMCMC fit object."""
    kwargs: dict[str, Any] = {}
    n_total = num_results * num_chains
    accept_rate = None

    try:
        method_vars = fit.method_variables()
    except Exception:
        method_vars = {}

    if "accept_stat__" in method_vars:
        ar = jnp.asarray(method_vars["accept_stat__"]).reshape(-1)
        accept_rate = ar
        kwargs["log_accept_ratio"] = jnp.log(jnp.clip(ar, 1e-10, 1.0))
    else:
        kwargs["log_accept_ratio"] = jnp.zeros(n_total)

    if "stepsize__" in method_vars:
        kwargs["step_size"] = jnp.asarray(method_vars["stepsize__"]).reshape(-1)

    if "divergent__" in method_vars:
        diverging = jnp.asarray(method_vars["divergent__"], dtype=jnp.bool_).reshape(-1)
        kwargs["diverging"] = diverging
        kwargs["n_divergences"] = int(jnp.sum(diverging))

    if "treedepth__" in method_vars:
        kwargs["tree_depth"] = jnp.asarray(method_vars["treedepth__"]).reshape(-1)

    if "n_leapfrog__" in method_vars:
        kwargs["n_steps"] = jnp.asarray(method_vars["n_leapfrog__"]).reshape(-1)

    if "energy__" in method_vars:
        kwargs["energy"] = jnp.asarray(method_vars["energy__"]).reshape(-1)

    if "lp__" in method_vars:
        kwargs["lp"] = jnp.asarray(method_vars["lp__"]).reshape(-1)

    diag = InferenceDiagnostics(algorithm="cmdstan_nuts", **kwargs)
    if accept_rate is not None:
        diag["_accept_rate_override"] = float(jnp.mean(accept_rate))
    return diag


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

        diagnostics = _extract_cmdstan_diagnostics(fit, num_results, num_chains)
        result = MCMCApproximateDistribution(
            chains, diagnostics=diagnostics, inference_data=fit,
            name="posterior",
        )
        result.with_source(Provenance(
            "cmdstan_nuts", parents=(dist,),
            metadata={"num_results": num_results, "num_warmup": num_warmup,
                      "num_chains": num_chains, "algorithm": "cmdstan_nuts"},
        ))
        return result
