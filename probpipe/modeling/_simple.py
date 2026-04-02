"""SimpleModel: construct a model from a prior and likelihood."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core.distribution import Distribution
from ..core.provenance import Provenance
from ..core.protocols import (
    SupportsLogProb,
    SupportsMean,
)
from ..custom_types import Array, ArrayLike
from ..inference._diagnostics import InferenceDiagnostics
from ..inference._mcmc_distribution import MCMCApproximateDistribution
from ._base import ProbabilisticModel
from ._likelihood import Likelihood

__all__ = ["SimpleModel"]


class SimpleModel[P, D](ProbabilisticModel[tuple[P, D]], SupportsLogProb):
    """Probabilistic model as a joint distribution over (parameters, data).

    A ``SimpleModel[P, D]`` is a ``Distribution[tuple[P, D]]`` — the joint
    distribution $p(\\theta, y) = p(\\theta) \\, p(y \\mid \\theta)$.
    The prior must support :class:`SupportsLogProb` so that the joint
    log-density is always computable.

    **Named components:** ``"parameters"`` (the prior) and ``"data"``
    (the likelihood).  Only ``"data"`` is conditionable.

    **Log-prob:** ``_log_prob((params, data))`` returns the joint
    log-density: ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``.

    **Conditioning:** ``_condition_on(data)`` fixes the data and runs
    MCMC (NUTS/HMC or RWMH fallback) to approximate the posterior
    $p(\\theta \\mid y)$.

    Parameters
    ----------
    prior : Distribution[P] that supports SupportsLogProb
        Prior distribution over model parameters.
    likelihood : Likelihood[P, D]
        Must have a ``log_likelihood(params, data)`` method.
    name : str or None
        Model name for provenance.
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        prior: Distribution[P],
        likelihood: Likelihood[P, D],
        *,
        name: str | None = None,
    ):
        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"SimpleModel requires a prior that supports SupportsLogProb, "
                f"got {type(prior).__name__}"
            )
        self._prior = prior
        self._likelihood = likelihood
        self._name_str = name

    # -- Distribution interface ---------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name_str

    # -- SupportsNamedComponents interface ----------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        return ("parameters", "data")

    def __getitem__(self, key: str) -> Any:
        if key == "parameters":
            return self._prior
        if key == "data":
            return self._likelihood
        raise KeyError(
            f"Unknown component: {key!r}; "
            f"available: {self.component_names}"
        )

    # -- SupportsConditionableComponents interface --------------------------

    @property
    def conditionable_components(self) -> dict[str, bool]:
        return {"parameters": False, "data": True}

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("parameters",)

    # -- SupportsLogProb interface -----------------------------------------

    def _log_prob(self, value: tuple[P, D]) -> Array:
        """Joint log-density: prior log-prob + log-likelihood.

        Parameters
        ----------
        value : tuple[P, D]
            A ``(params, data)`` pair.
        """
        params, data = value
        lp = self._prior._log_prob(params)
        ll = self._likelihood.log_likelihood(params=params, data=data)
        return lp + ll

    # -- Conditioning -------------------------------------------------------

    def _condition_on(self, observed: D, /, **kwargs: Any) -> MCMCApproximateDistribution:
        """Condition on observed data, returning posterior samples.

        Uses TFP NUTS/HMC when the posterior is JAX-traceable,
        falls back to RWMH otherwise.

        Parameters
        ----------
        observed : array-like
            Observed data.
        **kwargs
            Overrides for MCMC parameters: ``num_results`` (default 1000),
            ``num_warmup`` (default 500), ``num_chains`` (default 1),
            ``step_size`` (default 0.1), ``init`` (initial state),
            ``random_seed`` (default 0), ``algorithm`` (``"nuts"`` or ``"hmc"``).
        """
        data = jnp.asarray(observed)

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 1)
        step_size = kwargs.get("step_size", 0.1)
        init = kwargs.get("init", None)
        random_seed = kwargs.get("random_seed", 0)
        algorithm = kwargs.get("algorithm", "nuts")

        def target_log_prob_fn(params):
            return self._log_prob((params, data))

        # Determine initial state
        init_state = self._get_init_state(init, data)

        # Check JAX traceability for gradient-based sampling
        if self._is_jax_traceable(target_log_prob_fn, init_state):
            chains, diagnostics = self._run_tfp_chains(
                target_log_prob_fn,
                init_state,
                algorithm=algorithm,
                num_results=num_results,
                num_warmup=num_warmup,
                num_chains=num_chains,
                step_size=step_size,
                random_seed=random_seed,
            )
        else:
            # Fall back to RWMH — always pass init_state so RWMH doesn't
            # need to access event_shape (which SimpleModel doesn't have).
            from ..inference._rwmh import _rwmh_impl

            return _rwmh_impl(
                self._prior,
                data,
                log_prob_fn=lambda params, d: self._likelihood.log_likelihood(
                    params=params, data=d
                ),
                num_results=num_results,
                num_warmup=num_warmup,
                num_chains=num_chains,
                step_size=step_size,
                init=init_state,
                random_seed=random_seed,
            )

        result = MCMCApproximateDistribution(
            chains,
            diagnostics=diagnostics,
            name="posterior",
        )
        result.with_source(
            Provenance(
                algorithm,
                parents=(self._prior,),
                metadata={
                    "num_results": num_results,
                    "num_warmup": num_warmup,
                    "num_chains": num_chains,
                    "algorithm": algorithm,
                },
            )
        )
        return result

    # -- Internal helpers ---------------------------------------------------

    def _get_init_state(
        self, init: ArrayLike | None, data: ArrayLike
    ) -> jnp.ndarray:
        """Determine the initial chain state."""
        if init is not None:
            return jnp.atleast_1d(jnp.asarray(init, dtype=jnp.float32))

        if isinstance(self._prior, SupportsMean):
            try:
                m = self._prior._mean()
                return jnp.atleast_1d(jnp.asarray(m, dtype=jnp.float32))
            except Exception:
                pass

        return jnp.atleast_1d(jnp.mean(jnp.asarray(data), axis=0))

    @staticmethod
    def _is_jax_traceable(
        target_log_prob_fn: Any, init_state: jnp.ndarray
    ) -> bool:
        """Probe whether the target can be traced by JAX."""
        try:
            jax.make_jaxpr(target_log_prob_fn)(init_state)
            return True
        except Exception:
            return False

    @staticmethod
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

    def __repr__(self) -> str:
        prior_name = type(self._prior).__name__
        lik_name = type(self._likelihood).__name__
        return f"SimpleModel(prior={prior_name}, likelihood={lik_name})"


def _extract_diagnostics(trace: Any, algorithm: str) -> InferenceDiagnostics:
    """Extract diagnostics from TFP trace."""
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
