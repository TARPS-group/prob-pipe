"""TFP-backed inference methods: NUTS, HMC, and mean-field VI."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb, SupportsMean
from ..core.provenance import Provenance
from ..custom_types import Array, ArrayLike
from ._diagnostics import InferenceDiagnostics
from ._mcmc_distribution import MCMCApproximateDistribution
from ._registry import InferenceMethod


# ---------------------------------------------------------------------------
# Helpers (extracted from SimpleModel)
# ---------------------------------------------------------------------------

def _is_jax_traceable(fn: Any, init_state: jnp.ndarray) -> bool:
    """Probe whether *fn* can be traced by JAX."""
    try:
        jax.make_jaxpr(fn)(init_state)
        return True
    except Exception:
        return False


def _get_init_state(
    dist: Any, init: ArrayLike | None, data: ArrayLike | None,
) -> jnp.ndarray:
    """Determine an initial chain state from the distribution or data."""
    if init is not None:
        return jnp.atleast_1d(jnp.asarray(init, dtype=jnp.float32))

    if isinstance(dist, SupportsMean):
        try:
            m = dist._mean()
            return jnp.atleast_1d(jnp.asarray(m, dtype=jnp.float32))
        except Exception:
            pass

    if data is not None:
        return jnp.atleast_1d(jnp.mean(jnp.asarray(data), axis=0))

    raise ValueError(
        "Cannot determine initial state: provide init=, "
        "a distribution with SupportsMean, or observed data."
    )


def _build_target_log_prob(dist: Any, observed: Any) -> Any:
    """Build a target_log_prob_fn(params) from *dist* and *observed*.

    Handles three cases:

    1. **SimpleModel-like** (has ``_prior`` and ``_likelihood``):
       ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``
    2. **Bare SupportsLogProb with data**: ``dist._log_prob(params)``
       (data is assumed already folded in, e.g., via closure).
    3. **Joint over (params, data)**: ``dist._log_prob((params, data))``
    """
    # SimpleModel pattern: separate prior + likelihood
    if hasattr(dist, "_prior") and hasattr(dist, "_likelihood"):
        data = jnp.asarray(observed) if observed is not None else None

        def target_log_prob_fn(params):
            lp = dist._prior._log_prob(params)
            if data is not None:
                lp = lp + dist._likelihood.log_likelihood(params=params, data=data)
            return lp

        return target_log_prob_fn

    # Joint _log_prob((params, data))
    if observed is not None:
        data = jnp.asarray(observed)
        return lambda params: dist._log_prob((params, data))

    # Bare SupportsLogProb (no data)
    return dist._log_prob


def _run_tfp_chains(
    target_log_prob_fn: Any,
    init_state: jnp.ndarray,
    *,
    algorithm: str,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    step_size: float,
    random_seed: int,
) -> tuple[list, InferenceDiagnostics]:
    """Run TFP-backed MCMC chains."""
    if algorithm == "nuts":
        inner_kernel = tfp_mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
        )
    elif algorithm == "hmc":
        inner_kernel = tfp_mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=10,
        )
    else:
        raise ValueError(f"algorithm must be 'nuts' or 'hmc', got {algorithm!r}")

    num_adapt = int(0.8 * num_warmup) if num_warmup > 0 else 0
    if num_adapt > 0:
        kernel = tfp_mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adapt,
            target_accept_prob=0.75,
        )
    else:
        kernel = inner_kernel

    chains = []
    key = jax.random.PRNGKey(random_seed)
    last_trace = None

    for c in range(num_chains):
        key, chain_key = jax.random.split(key)
        samples, trace = tfp_mcmc.sample_chain(
            num_results=num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=num_warmup,
            seed=chain_key,
            trace_fn=lambda _, kr: kr,
        )
        chains.append(samples)
        last_trace = trace

    diagnostics = _extract_diagnostics(last_trace, algorithm)
    return chains, diagnostics


def _extract_diagnostics(trace: Any, algorithm: str) -> InferenceDiagnostics:
    """Extract diagnostics from a TFP trace."""
    results = trace

    if hasattr(results, "new_step_size"):
        step_size = results.new_step_size
        results = results.inner_results
    elif hasattr(results, "step_size"):
        step_size = results.step_size
    else:
        step_size = jnp.nan

    log_accept_ratio = getattr(results, "log_accept_ratio", jnp.array(jnp.nan))
    is_accepted = getattr(results, "is_accepted", None)

    kwargs: dict[str, Any] = {
        "log_accept_ratio": log_accept_ratio,
        "step_size": step_size,
    }
    if is_accepted is not None:
        kwargs["is_accepted"] = is_accepted
    return InferenceDiagnostics(algorithm=algorithm, **kwargs)


def _make_posterior(
    chains: list,
    diagnostics: InferenceDiagnostics,
    parents: tuple,
    algorithm: str,
    **meta: Any,
) -> MCMCApproximateDistribution:
    """Wrap chains + diagnostics into an MCMCApproximateDistribution with provenance."""
    result = MCMCApproximateDistribution(
        chains, diagnostics=diagnostics, name="posterior",
    )
    result.with_source(
        Provenance(algorithm, parents=parents, metadata={"algorithm": algorithm, **meta})
    )
    return result


# ---------------------------------------------------------------------------
# Helpers for prior extraction
# ---------------------------------------------------------------------------

def _get_prior(dist: Any) -> Any:
    """Extract the prior from a model, or return dist itself."""
    return dist._prior if hasattr(dist, "_prior") else dist


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


class _TFPGradientMethod(InferenceMethod):
    """Base for TFP gradient-based MCMC methods (NUTS, HMC).

    Parameterized by algorithm name and priority.  check() is kept
    cheap — only a protocol check and JAX traceability probe.
    """

    def __init__(self, algorithm: str, method_name: str, method_priority: int):
        self._algorithm = algorithm
        self._method_name = method_name
        self._method_priority = method_priority

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return self._method_priority

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, SupportsLogProb):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SupportsLogProb")
        try:
            target = _build_target_log_prob(dist, observed)
            init = _get_init_state(_get_prior(dist), kwargs.get("init"), observed)
            if not _is_jax_traceable(target, init):
                return MethodInfo(feasible=False, method_name=self.name,
                                  description="Log-prob is not JAX-traceable")
        except Exception as e:
            return MethodInfo(feasible=False, method_name=self.name,
                              description=str(e))
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> MCMCApproximateDistribution:
        target = _build_target_log_prob(dist, observed)
        prior = _get_prior(dist)
        init = _get_init_state(prior, kwargs.get("init"), observed)

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 1)

        chains, diagnostics = _run_tfp_chains(
            target, init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=kwargs.get("step_size", 0.1),
            random_seed=kwargs.get("random_seed", 0),
        )
        return _make_posterior(
            chains, diagnostics, parents=(prior,), algorithm=self._algorithm,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )


def TFPNutsMethod() -> _TFPGradientMethod:
    """TFP No-U-Turn Sampler (gradient-based MCMC)."""
    return _TFPGradientMethod("nuts", "tfp_nuts", 100)


def TFPHmcMethod() -> _TFPGradientMethod:
    """TFP Hamiltonian Monte Carlo."""
    return _TFPGradientMethod("hmc", "tfp_hmc", 90)
