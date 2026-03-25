"""
Modeling components for ProbPipe.

Provides abstract interfaces for likelihoods and posterior approximation,
concrete MCMC samplers (TFP-backed NUTS/HMC with automatic fallback),
and an iterative forecasting module.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions.distribution import ArrayDistribution, EmpiricalDistribution, Provenance
from ..distributions.multivariate import MultivariateNormal
from .node import AbstractModule, Module, WorkflowFunction, abstractwf, wf

logger = logging.getLogger(__name__)

__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "ApproximatePosterior",
    "MCMCSampler",
    "MCMCDiagnostics",
    "RWMH",
    "IterativeForecaster",
]


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class Likelihood(AbstractModule):
    """Abstract module for computing log-likelihood of data given parameters."""

    @abstractwf
    def log_likelihood(self, params: ArrayLike, data: ArrayLike) -> float:
        ...


class GenerativeLikelihood(AbstractModule):
    """Abstract module for generating synthetic data given parameters."""

    @abstractwf
    def generate_data(self, params: ArrayLike, n_samples: int) -> ArrayLike:
        ...


# ---------------------------------------------------------------------------
# Approximate posterior base
# ---------------------------------------------------------------------------


class ApproximatePosterior(WorkflowFunction, ABC):
    """Abstract base for all posterior approximation methods."""

    def __init__(
        self,
        *,
        workflow_kind: str | None = None,
        name: str = "compute_posterior",
        **bind: Any,
    ):
        super().__init__(
            func=self._compute_posterior,
            workflow_kind=workflow_kind,
            name=name,
            bind=bind,
        )

    @abstractwf
    def _compute_posterior(
        self,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> EmpiricalDistribution:
        ...


# ---------------------------------------------------------------------------
# MCMCDiagnostics
# ---------------------------------------------------------------------------


@dataclass
class MCMCDiagnostics:
    """Post-sampling diagnostics from an MCMC run.

    Attributes
    ----------
    log_accept_ratio : Array
        Per-sample log Metropolis-Hastings acceptance ratio.
    step_size : Array or float
        Final adapted step size(s).
    is_accepted : Array or None
        Per-sample boolean accept/reject flags (if available).
    algorithm : str
        Name of the algorithm that produced these diagnostics.
    """

    log_accept_ratio: Array
    step_size: Array | float
    is_accepted: Array | None = None
    algorithm: str = "unknown"

    @property
    def accept_rate(self) -> float:
        """Empirical acceptance rate."""
        # RWMH fallback stores exact accept rate
        if hasattr(self, "_numpy_accept_rate"):
            return self._numpy_accept_rate
        if self.is_accepted is not None:
            return float(jnp.mean(self.is_accepted))
        return float(
            jnp.mean(jnp.exp(jnp.minimum(self.log_accept_ratio, 0.0)))
        )

    @property
    def final_step_size(self) -> float:
        """Mean final adapted step size."""
        return float(jnp.mean(jnp.asarray(self.step_size)))

    def summary(self) -> str:
        """One-line summary of diagnostics."""
        return (
            f"algorithm={self.algorithm}, "
            f"accept_rate={self.accept_rate:.3f}, "
            f"final_step_size={self.final_step_size:.4f}"
        )


# ---------------------------------------------------------------------------
# MCMCSampler  (TFP-backed NUTS / HMC with automatic fallback)
# ---------------------------------------------------------------------------


class MCMCSampler(ApproximatePosterior):
    """
    Gradient-based MCMC posterior approximation using TFP kernels.

    Runs NUTS (default) or HMC with automatic step-size adaptation during
    warmup.  If the target log-prob is not JAX-traceable (e.g. wraps scipy
    or external code), automatically falls back to gradient-free random-walk
    Metropolis-Hastings.

    Returns an :class:`EmpiricalDistribution` with provenance and
    diagnostics attached to ``self.diagnostics``.

    Parameters
    ----------
    algorithm : ``"nuts"`` or ``"hmc"``
        MCMC algorithm to use (default ``"nuts"``).  Ignored when the
        likelihood is not JAX-traceable (always falls back to RW-MH).
    num_results : int
        Number of posterior samples to retain after warmup (default 1000).
    num_warmup : int
        Number of warmup (adaptation + burn-in) steps (default 500).
    init : ArrayLike or None
        Initial chain state.  If *None*, ``prior.mean()`` is tried first,
        then the column-wise data mean.
    step_size : float
        Initial leapfrog step size (default 0.1).  Adapted during warmup.
    num_leapfrog_steps : int
        Leapfrog steps per HMC proposal (default 10).  Ignored for NUTS.
    target_accept_prob : float
        Target Metropolis acceptance probability for adaptation (default 0.75).
    seed : int
        Random seed for JAX PRNG (default 0).
    """

    def __init__(
        self,
        algorithm: str = "nuts",
        num_results: int = 1000,
        num_warmup: int = 500,
        init: ArrayLike | None = None,
        step_size: float = 0.1,
        num_leapfrog_steps: int = 10,
        target_accept_prob: float = 0.75,
        seed: int = 0,
        workflow_kind: str | None = None,
    ):
        if algorithm not in ("nuts", "hmc"):
            raise ValueError(f"algorithm must be 'nuts' or 'hmc', got {algorithm!r}")
        if num_results <= 0:
            raise ValueError("num_results must be > 0")
        if num_warmup < 0:
            raise ValueError("num_warmup must be >= 0")

        super().__init__(
            workflow_kind=workflow_kind,
            algorithm=algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            init=init,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            target_accept_prob=target_accept_prob,
            seed=seed,
        )
        self.algorithm = algorithm
        self.num_results = num_results
        self.num_warmup = num_warmup
        self.init = init
        self.step_size = float(step_size)
        self.num_leapfrog_steps = int(num_leapfrog_steps)
        self.target_accept_prob = float(target_accept_prob)
        self.seed = int(seed)

        self.diagnostics: MCMCDiagnostics | None = None

    # -- internal helpers ---------------------------------------------------

    def _get_init_state(
        self,
        prior: ArrayDistribution,
        data: ArrayLike,
    ) -> jnp.ndarray:
        """Determine the initial chain state (always at least 1-D, floating)."""
        if self.init is not None:
            return jnp.atleast_1d(jnp.asarray(self.init, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32))

        # Try prior mean
        try:
            m = prior.mean()
            return jnp.atleast_1d(jnp.asarray(m))
        except (NotImplementedError, Exception):
            pass

        # Fall back to data column mean
        return jnp.atleast_1d(jnp.mean(jnp.asarray(data), axis=0))

    def _is_jax_traceable(
        self,
        target_log_prob_fn: Callable,
        init_state: jnp.ndarray,
    ) -> bool:
        """Probe whether *target_log_prob_fn* can be traced by JAX."""
        try:
            jax.make_jaxpr(target_log_prob_fn)(init_state)
            return True
        except Exception:
            return False

    def _build_kernel(
        self,
        target_log_prob_fn: Callable,
        init_state: jnp.ndarray,
    ) -> tuple:
        """Build an MCMC kernel, falling back to ``None`` if not JAX-traceable.

        Returns ``(kernel, used_algorithm)`` where *kernel* is ``None``
        when the target is not JAX-traceable (caller should use the
        numpy-based RWMH path instead).
        """
        if self._is_jax_traceable(target_log_prob_fn, init_state):
            # Gradient-based kernel
            if self.algorithm == "nuts":
                inner_kernel = tfp_mcmc.NoUTurnSampler(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=self.step_size,
                )
            else:
                inner_kernel = tfp_mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=self.step_size,
                    num_leapfrog_steps=self.num_leapfrog_steps,
                )

            # Wrap with dual-averaging step-size adaptation
            num_adapt = int(0.8 * self.num_warmup) if self.num_warmup > 0 else 0
            if num_adapt > 0:
                kernel = tfp_mcmc.DualAveragingStepSizeAdaptation(
                    inner_kernel=inner_kernel,
                    num_adaptation_steps=num_adapt,
                    target_accept_prob=self.target_accept_prob,
                )
            else:
                kernel = inner_kernel

            return kernel, self.algorithm
        else:
            logger.info(
                "Target log-prob is not JAX-traceable; falling back to "
                "numpy-based random-walk Metropolis-Hastings (no gradients)."
            )
            return None, "rwmh_fallback"

    @staticmethod
    def _extract_diagnostics(
        trace: Any,
        used_algorithm: str,
    ) -> MCMCDiagnostics:
        """Extract diagnostics from the trace returned by sample_chain."""
        # The trace structure depends on the kernel stack.  We walk
        # through common wrapper layers to find the inner results.
        results = trace

        # Unwrap DualAveragingStepSizeAdaptation
        if hasattr(results, "new_step_size"):
            step_size = results.new_step_size
            results = results.inner_results
        elif hasattr(results, "step_size"):
            step_size = results.step_size
        else:
            step_size = jnp.nan

        # Extract accept info
        log_accept_ratio = getattr(results, "log_accept_ratio", jnp.array(jnp.nan))
        is_accepted = getattr(results, "is_accepted", None)

        return MCMCDiagnostics(
            log_accept_ratio=log_accept_ratio,
            step_size=step_size,
            is_accepted=is_accepted,
            algorithm=used_algorithm,
        )

    # -- main implementation ------------------------------------------------

    @wf
    def _compute_posterior(
        self,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> EmpiricalDistribution:
        data_jnp = jnp.asarray(data)

        # Build unnormalized log-posterior
        def target_log_prob_fn(params):
            lp = prior.log_prob(params)
            ll = likelihood.log_likelihood(params=params, data=data_jnp)
            return lp + ll

        # Determine initial state and build kernel
        init_state = self._get_init_state(prior, data_jnp)
        kernel, used_algorithm = self._build_kernel(target_log_prob_fn, init_state)

        if kernel is not None:
            # JAX-traceable: run TFP sample_chain
            samples, trace = self._run_tfp_chain(kernel, init_state)
            self.diagnostics = self._extract_diagnostics(trace, used_algorithm)
        else:
            # Not JAX-traceable: fall back to gradient-free RW-MH
            samples, accept_rate = self._run_numpy_rwmh(
                target_log_prob_fn, init_state
            )
            self.diagnostics = MCMCDiagnostics(
                log_accept_ratio=jnp.zeros(self.num_results),
                step_size=self.step_size,
                is_accepted=None,
                algorithm=used_algorithm,
            )
            # Overwrite accept_rate with the actual value
            self.diagnostics._numpy_accept_rate = accept_rate

        # Build and return EmpiricalDistribution with provenance
        posterior = EmpiricalDistribution(samples, name="posterior")
        posterior.with_source(Provenance(
            used_algorithm,
            parents=(prior,),
            metadata={
                "num_results": self.num_results,
                "num_warmup": self.num_warmup,
                "algorithm": used_algorithm,
            },
        ))
        return posterior

    def _run_tfp_chain(
        self,
        kernel,
        init_state: jnp.ndarray,
    ) -> tuple:
        """Run TFP sample_chain and return (samples, trace)."""
        key = jax.random.PRNGKey(self.seed)
        return tfp_mcmc.sample_chain(
            num_results=self.num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=self.num_warmup,
            seed=key,
            trace_fn=lambda _, kr: kr,
        )

    def _run_numpy_rwmh(
        self,
        target_log_prob_fn: Callable,
        init_state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """JAX-based random-walk MH for non-JAX-traceable likelihoods.

        Returns ``(samples, accept_rate)``.
        """
        key = jax.random.PRNGKey(self.seed)
        d = init_state.shape[0]
        mu_curr = jnp.asarray(init_state)
        logp_curr = float(target_log_prob_fn(mu_curr))

        kept: list[jnp.ndarray] = []
        accepts = 0
        total_steps = self.num_warmup + self.num_results

        for t in range(total_steps):
            key, subkey_prop, subkey_accept = jax.random.split(key, 3)
            noise = jax.random.normal(subkey_prop, shape=(d,), dtype=mu_curr.dtype)
            mu_prop = mu_curr + self.step_size * noise
            logp_prop = float(target_log_prob_fn(mu_prop))

            u = jax.random.uniform(subkey_accept, dtype=mu_curr.dtype)
            if jnp.log(u) < min(0.0, logp_prop - logp_curr):
                mu_curr = mu_prop
                logp_curr = logp_prop
                accepts += 1

            if t >= self.num_warmup:
                kept.append(mu_curr)

        accept_rate = accepts / total_steps
        samples = jnp.stack(kept)
        return samples, accept_rate


# ---------------------------------------------------------------------------
# Legacy RWMH  (gradient-free)
# ---------------------------------------------------------------------------


class RWMH(ApproximatePosterior):
    """
    Random-walk Metropolis-Hastings (gradient-free).

    Useful when the likelihood wraps external code that isn't JAX-traceable.
    For JAX-traceable likelihoods, prefer :class:`MCMCSampler` which uses
    NUTS/HMC with automatic adaptation.

    .. deprecated::
        Use :class:`MCMCSampler` for new code.  ``MCMCSampler`` will
        automatically fall back to gradient-free sampling when needed.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        n_steps: int = 10_000,
        burn_in: int = 2_000,
        thin: int = 5,
        init: ArrayLike | None = None,
        seed: int = 0,
        workflow_kind: str | None = None,
    ):
        super().__init__(
            workflow_kind=workflow_kind,
            step_size=step_size,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            init=init,
            seed=seed,
        )
        self.step_size = float(step_size)
        self.n_steps = int(n_steps)
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.init = None if init is None else jnp.asarray(init, dtype=jnp.float32)
        self.seed = int(seed)

        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.thin <= 0:
            raise ValueError("thin must be > 0")

    @wf
    def _compute_posterior(
        self,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> EmpiricalDistribution:
        data = jnp.asarray(data, dtype=jnp.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected data shape (n, d), got {data.shape}")
        d = data.shape[1]

        def log_post(mu):
            mu = jnp.asarray(mu, dtype=jnp.float32)
            if mu.shape != (d,):
                raise ValueError(f"Expected params shape {(d,)}, got {mu.shape}")
            lp_val = float(jnp.asarray(prior.log_prob(mu)).sum())
            ll_val = float(likelihood.log_likelihood(params=mu, data=data))
            return lp_val + ll_val

        # Initialize
        if self.init is not None:
            mu_curr = self.init
            if mu_curr.shape != (d,):
                raise ValueError(f"init must have shape {(d,)}, got {mu_curr.shape}")
        else:
            mu_curr = None
            try:
                m = jnp.asarray(prior.mean(), dtype=jnp.float32)
                if m.shape == (d,):
                    mu_curr = m
            except (NotImplementedError, Exception):
                pass
            if mu_curr is None:
                mu_curr = jnp.mean(data, axis=0)

        logp_curr = log_post(mu_curr)

        key = jax.random.PRNGKey(self.seed)
        kept = []
        accepts = 0
        burn_in = min(self.burn_in, self.n_steps)

        for t in range(self.n_steps):
            key, subkey_prop, subkey_accept = jax.random.split(key, 3)
            mu_prop = mu_curr + self.step_size * jax.random.normal(subkey_prop, shape=(d,))
            logp_prop = log_post(mu_prop)

            log_alpha = logp_prop - logp_curr
            if jnp.log(jax.random.uniform(subkey_accept)) < min(0.0, log_alpha):
                mu_curr = mu_prop
                logp_curr = logp_prop
                accepts += 1

            if t >= burn_in and ((t - burn_in) % self.thin == 0):
                kept.append(mu_curr)

        if len(kept) == 0:
            raise ValueError(
                "No samples retained; increase n_steps or reduce burn_in/thin."
            )

        samples = jnp.stack(kept)
        self.accept_rate = accepts / float(self.n_steps)

        posterior = EmpiricalDistribution(samples, name="posterior")
        posterior.with_source(Provenance(
            "rwmh",
            parents=(prior,),
            metadata={
                "n_steps": self.n_steps,
                "burn_in": self.burn_in,
                "thin": self.thin,
                "accept_rate": self.accept_rate,
            },
        ))
        return posterior


# ---------------------------------------------------------------------------
# IterativeForecaster
# ---------------------------------------------------------------------------


class IterativeForecaster(Module):
    """Iteratively update posterior given new data batches."""

    def __init__(
        self,
        *,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        generative_likelihood: GenerativeLikelihood,
        approx_post: ApproximatePosterior,
        workflow_kind: str | None = None,
    ):
        self._curr_posterior: ArrayDistribution = prior
        self._generative_likelihood = generative_likelihood

        super().__init__(
            likelihood=likelihood,
            generative_likelihood=generative_likelihood,
            approx_post=approx_post,
            prior=prior,
            workflow_kind=workflow_kind,
        )

    @property
    def curr_posterior(self) -> ArrayDistribution:
        return self._curr_posterior

    @wf
    def update(
        self,
        approx_post: ApproximatePosterior,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> ArrayDistribution:
        post_dist = approx_post(
            prior=self._curr_posterior, likelihood=likelihood, data=data
        )
        self._curr_posterior = post_dist
        return post_dist
