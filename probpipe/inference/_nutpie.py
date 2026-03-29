"""Nutpie-backed MCMC sampling workflow function."""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

from ..core.distribution import Provenance
from ..core.node import WorkflowFunction
from ..custom_types import ArrayLike
from ._diagnostics import MCMCDiagnostics
from ._mcmc_distribution import MCMCApproximateDistribution

logger = logging.getLogger(__name__)

__all__ = ["condition_on_nutpie"]


def _condition_on_nutpie_impl(
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

    Accepts a :class:`~probpipe.modeling.StanModel` (via BridgeStan)
    or a :class:`~probpipe.modeling.PyMCModel`.

    Parameters
    ----------
    model : StanModel or PyMCModel
        Probabilistic model to sample from.
    data : array-like or None
        Observed data to condition on.  Passed to the model's
        conditioning interface.
    num_results : int
        Number of post-warmup draws per chain (default 1000).
    num_warmup : int
        Number of warmup draws per chain (default 500).
    num_chains : int
        Number of independent chains (default 4).
    random_seed : int
        Random seed (default 0).
    **kwargs
        Additional keyword arguments passed to ``nutpie.sample``.

    Returns
    -------
    MCMCApproximateDistribution
        Posterior samples with chain structure and diagnostics.
    """
    try:
        import nutpie
    except ImportError as e:
        raise ImportError(
            "nutpie is required for condition_on_nutpie. "
            "Install it with: pip install nutpie"
        ) from e

    # Determine the compiled model for nutpie
    compiled = _compile_for_nutpie(model, data)

    # Run sampling
    trace = nutpie.sample(
        compiled,
        draws=num_results,
        tune=num_warmup,
        chains=num_chains,
        seed=random_seed,
        **kwargs,
    )

    # Extract chains from the InferenceData / trace object
    chains, param_names = _extract_chains(trace, num_chains)

    diagnostics = _extract_nutpie_diagnostics(trace, num_results, num_chains)

    result = MCMCApproximateDistribution(
        chains,
        diagnostics=diagnostics,
        name="posterior",
    )
    result.with_source(
        Provenance(
            "condition_on_nutpie",
            parents=(model,),
            metadata={
                "num_results": num_results,
                "num_warmup": num_warmup,
                "num_chains": num_chains,
            },
        )
    )
    return result


def _compile_for_nutpie(model: Any, data: Any) -> Any:
    """Compile a model for nutpie sampling."""
    # StanModel path: uses BridgeStan
    if hasattr(model, "_bridgestan_model"):
        import nutpie

        return nutpie.compile_stan_model(model._bridgestan_model(data=data))

    # PyMCModel path: uses PyMC compilation
    if hasattr(model, "_pymc_model"):
        import nutpie

        pm_model = model._pymc_model(data=data)
        return nutpie.compile_pymc_model(pm_model)

    raise TypeError(
        f"condition_on_nutpie does not support {type(model).__name__}. "
        f"Expected a StanModel or PyMCModel."
    )


def _extract_chains(trace: Any, num_chains: int) -> tuple[list, list]:
    """Extract per-chain sample arrays from a nutpie trace.

    Returns (chains, param_names) where each chain is an array of
    shape (num_draws, num_params).
    """
    # nutpie returns an ArviZ InferenceData object
    if hasattr(trace, "posterior"):
        posterior = trace.posterior
        param_names = list(posterior.data_vars)

        chains = []
        for c in range(num_chains):
            chain_arrays = []
            for name in param_names:
                vals = posterior[name].values[c]  # (num_draws, *param_shape)
                if vals.ndim == 1:
                    vals = vals[:, None]
                else:
                    vals = vals.reshape(vals.shape[0], -1)
                chain_arrays.append(vals)
            chain_concat = jnp.concatenate(chain_arrays, axis=-1)
            chains.append(chain_concat)
        return chains, param_names

    raise TypeError(
        f"Cannot extract chains from nutpie trace of type {type(trace).__name__}"
    )


def _extract_nutpie_diagnostics(
    trace: Any, num_results: int, num_chains: int,
) -> MCMCDiagnostics:
    """Extract diagnostics from a nutpie ArviZ InferenceData trace.

    Populates core fields (``log_accept_ratio``, ``step_size``) and
    stores additional NUTS-specific diagnostics (divergences, tree depth,
    energy, leapfrog steps, log-posterior) in ``extra``.
    """
    extra: dict[str, Any] = {}
    n_total = num_results * num_chains

    stats = getattr(trace, "sample_stats", None)

    if stats is None:
        return MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(n_total),
            step_size=0.0,
            algorithm="nutpie_nuts",
        )

    # -- Core fields ---------------------------------------------------------
    # acceptance_rate: (chains, draws) → flat array, then log
    if "acceptance_rate" in stats:
        accept_rate = jnp.asarray(stats["acceptance_rate"].values).reshape(-1)
        log_accept_ratio = jnp.log(jnp.clip(accept_rate, 1e-10, 1.0))
    else:
        accept_rate = None
        log_accept_ratio = jnp.zeros(n_total)

    if "step_size" in stats:
        step_size = jnp.asarray(stats["step_size"].values).reshape(-1)
    else:
        step_size = jnp.zeros(0)

    # -- Extra diagnostics ---------------------------------------------------
    if "diverging" in stats:
        diverging = jnp.asarray(stats["diverging"].values, dtype=jnp.bool_)
        extra["diverging"] = diverging.reshape(-1)
        extra["n_divergences"] = int(jnp.sum(extra["diverging"]))

    if "tree_depth" in stats:
        extra["tree_depth"] = jnp.asarray(
            stats["tree_depth"].values
        ).reshape(-1)

    if "n_steps" in stats:
        extra["n_steps"] = jnp.asarray(
            stats["n_steps"].values
        ).reshape(-1)

    if "energy" in stats:
        extra["energy"] = jnp.asarray(
            stats["energy"].values
        ).reshape(-1)

    if "energy_error" in stats:
        extra["energy_error"] = jnp.asarray(
            stats["energy_error"].values
        ).reshape(-1)

    if "lp" in stats:
        extra["lp"] = jnp.asarray(stats["lp"].values).reshape(-1)

    diag = MCMCDiagnostics(
        log_accept_ratio=log_accept_ratio,
        step_size=step_size,
        algorithm="nutpie_nuts",
        extra=extra,
    )

    # If we got raw acceptance rates, set the override for accuracy
    if accept_rate is not None:
        diag._numpy_accept_rate = float(jnp.mean(accept_rate))

    return diag


condition_on_nutpie = WorkflowFunction(func=_condition_on_nutpie_impl, name="condition_on_nutpie")
