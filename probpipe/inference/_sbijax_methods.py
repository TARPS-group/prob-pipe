"""sbijax inference methods for the ProbPipe registry."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..core._registry import MethodInfo
from ._mcmc_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod
# Importing _sbijax_adapters also imports sbijax (with rcParams workaround).
from ._sbijax_adapters import (
    adapt_prior,
    adapt_simulator,
    coerce_observable,
    default_distance_fn,
    default_summary_fn,
    extract_chains,
    is_tfp_backed,
)
import sbijax

__all__: list[str] = []


class SbiSMCABCMethod(InferenceMethod):
    """Non-amortized SMC-ABC via sbijax.

    Operates on :class:`~probpipe.modeling.SimpleGenerativeModel` —
    requires a prior that supports sampling and a
    :class:`~probpipe.modeling.GenerativeLikelihood`.

    Unlike amortized methods (NPE, NLE), SMCABC runs from scratch on
    each ``condition_on`` call with no prior training step.
    """

    @property
    def name(self) -> str:
        return "sbijax_smcabc"

    def supported_types(self) -> tuple[type, ...]:
        from ..modeling._simple_generative import SimpleGenerativeModel
        return (SimpleGenerativeModel,)

    @property
    def priority(self) -> int:
        return 40

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        from ..modeling._simple_generative import SimpleGenerativeModel

        if not isinstance(dist, SimpleGenerativeModel):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="Requires SimpleGenerativeModel",
            )
        if not is_tfp_backed(dist["parameters"]):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="Prior must be TFP-backed for sbijax",
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(
        self, dist: Any, observed: Any, **kwargs: Any
    ) -> ApproximateDistribution:
        import inspect

        prior_fn = adapt_prior(dist["parameters"])
        simulator_fn = adapt_simulator(dist["data"])
        fns = (prior_fn, simulator_fn)

        random_seed = kwargs.get("random_seed", 0)
        key = jax.random.PRNGKey(random_seed)

        summary_fn = kwargs.get("summary_fn", default_summary_fn)
        distance_fn = kwargs.get("distance_fn", default_distance_fn)
        sbi_model = sbijax.SMCABC(fns, summary_fn, distance_fn)

        # Workaround for sbijax bug: _chol_factor returns a scalar when
        # parameters are 1D because jnp.cov reduces (1, n) to ().
        # Patch to ensure the covariance is always at least 2D.
        def _patched_chol_factor(particles, cov_scale):
            from jax.flatten_util import ravel_pytree
            flat = jax.vmap(lambda x: ravel_pytree(x)[0])(particles)
            cov = jnp.atleast_2d(jnp.cov(flat.T)) * cov_scale
            return jnp.linalg.cholesky(cov)

        sbi_model._chol_factor = _patched_chol_factor

        observable = coerce_observable(observed)

        # Forward only kwargs sample_posterior actually accepts.
        sig = inspect.signature(sbi_model.sample_posterior)
        accepted = set(sig.parameters) - {"rng_key", "observable", "self"}
        smcabc_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        posterior_idata, _ = sbi_model.sample_posterior(
            key, observable=observable, **smcabc_kwargs,
        )

        chains = extract_chains(posterior_idata)

        return make_posterior(
            chains,
            parents=(dist["parameters"],),
            algorithm="sbijax_smcabc",
            inference_data=posterior_idata,
        )
