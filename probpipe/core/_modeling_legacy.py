"""Legacy MCMC classes, kept for backward compatibility.

.. deprecated::
    Use :mod:`probpipe.modeling` and :mod:`probpipe.inference` instead.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..custom_types import Array, ArrayLike
from .distribution import ArrayDistribution, EmpiricalDistribution, Provenance
from .modeling import ApproximatePosterior, Likelihood
from .node import wf
from ..inference._diagnostics import MCMCDiagnostics

logger = logging.getLogger(__name__)

__all__ = ["MCMCSampler", "RWMH"]


class MCMCSampler(ApproximatePosterior):
    """Gradient-based MCMC posterior approximation using TFP kernels.

    .. deprecated::
        Use :class:`~probpipe.modeling.SimpleModel` with
        :func:`~probpipe.core.ops.condition_on` instead.
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

    def _get_init_state(self, prior, data):
        if self.init is not None:
            return jnp.atleast_1d(jnp.asarray(self.init, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32))
        try:
            m = prior._mean()
            return jnp.atleast_1d(jnp.asarray(m))
        except Exception:
            pass
        return jnp.atleast_1d(jnp.mean(jnp.asarray(data), axis=0))

    def _is_jax_traceable(self, target_log_prob_fn, init_state):
        try:
            jax.make_jaxpr(target_log_prob_fn)(init_state)
            return True
        except Exception:
            return False

    def _build_kernel(self, target_log_prob_fn, init_state):
        if self._is_jax_traceable(target_log_prob_fn, init_state):
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
    def _extract_diagnostics(trace, used_algorithm):
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
        return MCMCDiagnostics(
            log_accept_ratio=log_accept_ratio,
            step_size=step_size,
            is_accepted=is_accepted,
            algorithm=used_algorithm,
        )

    @wf
    def _compute_posterior(
        self,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> EmpiricalDistribution:
        data_jnp = jnp.asarray(data)

        def target_log_prob_fn(params):
            lp = prior._log_prob(params)
            ll = likelihood.log_likelihood(params=params, data=data_jnp)
            return lp + ll

        init_state = self._get_init_state(prior, data_jnp)
        kernel, used_algorithm = self._build_kernel(target_log_prob_fn, init_state)

        if kernel is not None:
            samples, trace = self._run_tfp_chain(kernel, init_state)
            self.diagnostics = self._extract_diagnostics(trace, used_algorithm)
        else:
            samples, accept_rate = self._run_numpy_rwmh(target_log_prob_fn, init_state)
            self.diagnostics = MCMCDiagnostics(
                log_accept_ratio=jnp.zeros(self.num_results),
                step_size=self.step_size,
                is_accepted=None,
                algorithm=used_algorithm,
            )
            self.diagnostics._numpy_accept_rate = accept_rate

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

    def _run_tfp_chain(self, kernel, init_state):
        key = jax.random.PRNGKey(self.seed)
        return tfp_mcmc.sample_chain(
            num_results=self.num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=self.num_warmup,
            seed=key,
            trace_fn=lambda _, kr: kr,
        )

    def _run_numpy_rwmh(self, target_log_prob_fn, init_state):
        key = jax.random.PRNGKey(self.seed)
        d = init_state.shape[0]
        mu_curr = jnp.asarray(init_state)
        logp_curr = float(target_log_prob_fn(mu_curr))
        kept = []
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


class RWMH(ApproximatePosterior):
    """Random-walk Metropolis-Hastings (gradient-free).

    .. deprecated::
        Use :func:`probpipe.inference.rwmh` instead.
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
            lp_val = float(jnp.asarray(prior._log_prob(mu)).sum())
            ll_val = float(likelihood.log_likelihood(params=mu, data=data))
            return lp_val + ll_val

        if self.init is not None:
            mu_curr = self.init
            if mu_curr.shape != (d,):
                raise ValueError(f"init must have shape {(d,)}, got {mu_curr.shape}")
        else:
            mu_curr = None
            try:
                m = jnp.asarray(prior._mean(), dtype=jnp.float32)
                if m.shape == (d,):
                    mu_curr = m
            except Exception:
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
