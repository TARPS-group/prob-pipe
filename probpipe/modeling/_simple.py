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
    SupportsSampling,
)
from ..custom_types import Array, ArrayLike
from ..inference._diagnostics import InferenceDiagnostics
from ..inference._mcmc_distribution import MCMCApproximateDistribution
from ._base import ProbabilisticModel

__all__ = ["SimpleModel"]


class SimpleModel(ProbabilisticModel):
    """Probabilistic model constructed from a prior and likelihood.

    Dynamically determines protocol support based on the capabilities
    of the *prior* distribution:

    - If *prior* supports :class:`SupportsLogProb`, the model supports
      ``_log_prob`` (prior log-prob + likelihood log-likelihood).
    - If *prior* supports :class:`SupportsSampling`, the model supports
      ``_sample`` (prior predictive sampling).
    Conditioning uses TFP NUTS/HMC when the log-posterior is
    JAX-traceable, and falls back to gradient-free RWMH otherwise.

    Parameters
    ----------
    prior : Distribution
        Prior distribution over model parameters.
    likelihood : callable
        Must have a ``log_likelihood(params, data)`` method (e.g., a
        :class:`~probpipe.modeling.Likelihood` module).
    data_names : tuple of str or None
        Names of the data components.  Defaults to ``("data",)``.
    name : str or None
        Model name for provenance.
    """

    def __init__(
        self,
        prior: Distribution,
        likelihood: Any,
        *,
        data_names: tuple[str, ...] | None = None,
        name: str | None = None,
    ):
        self._prior = prior
        self._likelihood = likelihood
        self._data_names = data_names or ("data",)
        self._name_str = name

    # -- Distribution interface ---------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name_str

    # -- SupportsNamedComponents interface ----------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        return self.parameter_names + self._data_names

    def __getitem__(self, key: str) -> Any:
        if key == "prior" or key in self.parameter_names:
            return self._prior
        if key == "likelihood":
            return self._likelihood
        raise KeyError(f"Unknown component: {key!r}")

    # -- SupportsConditionableComponents interface --------------------------

    @property
    def conditionable_components(self) -> dict[str, bool]:
        result = {}
        for name in self.parameter_names:
            result[name] = False  # parameters are optional to condition on
        for name in self._data_names:
            result[name] = True  # data is required
        return result

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        # Derive from prior's component names if available, else generic
        if hasattr(self._prior, "component_names"):
            return self._prior.component_names
        return ("params",)

    # -- Dynamic protocol support -------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Event shape of the parameter space."""
        return self._prior.event_shape

    def _log_prob(self, value: Any) -> Array:
        """Log-probability of parameters under the prior.

        Available only when the prior supports :class:`SupportsLogProb`.
        To get the full posterior log-prob, use ``_posterior_log_prob``
        with data.
        """
        if not isinstance(self._prior, SupportsLogProb):
            raise TypeError(
                f"Prior {type(self._prior).__name__} does not support log_prob"
            )
        return self._prior._log_prob(value)

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        """Prior predictive sample.

        Available only when the prior supports :class:`SupportsSampling`.
        """
        if not isinstance(self._prior, SupportsSampling):
            raise TypeError(
                f"Prior {type(self._prior).__name__} does not support sampling"
            )
        return self._prior._sample(key, sample_shape)

    # -- Conditioning -------------------------------------------------------

    def _posterior_log_prob(self, params: Any, data: ArrayLike) -> Array:
        """Unnormalized log-posterior: prior + likelihood."""
        data_jnp = jnp.asarray(data)
        lp = self._prior._log_prob(params)
        ll = self._likelihood.log_likelihood(params=params, data=data_jnp)
        return lp + ll

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> MCMCApproximateDistribution:
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
            return self._posterior_log_prob(params, data)

        # Determine initial state
        init_state = self._get_init_state(init, data)

        # Check JAX traceability for gradient-based sampling
        if isinstance(self._prior, SupportsLogProb) and self._is_jax_traceable(
            target_log_prob_fn, init_state
        ):
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
            # Fall back to RWMH
            from ..inference._rwmh import _rwmh_impl

            return _rwmh_impl(
                self,
                data,
                log_prob_fn=lambda params, d: self._likelihood.log_likelihood(
                    params=params, data=d
                ),
                num_results=num_results,
                num_warmup=num_warmup,
                num_chains=num_chains,
                step_size=step_size,
                init=init,
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
        # Build kernel
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

        # Step-size adaptation
        num_adapt = int(0.8 * num_warmup) if num_warmup > 0 else 0
        if num_adapt > 0:
            kernel = tfp_mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=inner_kernel,
                num_adaptation_steps=num_adapt,
                target_accept_prob=0.75,
            )
        else:
            kernel = inner_kernel

        # Run chains
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

        # Extract diagnostics from last chain
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
