"""BlackJAX-backed elliptical slice sampling for Gaussian-prior models.

Elliptical slice sampling ([Murray, Adams & MacKay 2010](https://proceedings.mlr.press/v9/murray10a.html))
is a gradient-free MCMC kernel restricted to models whose prior is
a multivariate Gaussian. The kernel is structurally **tuning-free**:
no step size, no mass matrix, no per-model hyperparameter. At each
step it draws an auxiliary Gaussian sample from the prior, constructs
an ellipse passing through the current state and the auxiliary sample,
and uses slice sampling on the angle around the ellipse to land at a
state whose likelihood passes a uniformly-drawn slice height.

The combination "self-tuning + asymptotically uniform mixing on the
ellipse" gives ESS strong performance on its narrow feasibility class
(Gaussian-prior latent-variable models — Bayesian linear regression,
GP hyperparameter posteriors with Gaussian hyperpriors, latent-Gaussian
models). When applicable, ESS dominates RWMH on the same target and is
often competitive with NUTS at a fraction of the per-step cost.

ProbPipe registers this method as ``blackjax_elliptical_slice``;
:class:`BlackJAXESSMethod` states its priority and feasibility class.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from ..core._dispatch import Feasibility
from ..core.record import Record
from ..custom_types import Array, ArrayLike
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import FactoredDistribution
from ..families import MultivariateNormal, Normal
from ..operations._condition import InferenceMethod, _UnnormalizedConditional
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    as_prng_key,
    build_mcmc_datatree,
    get_init_state,
    is_jax_traceable,
    likelihood_flat,
    model_factors,
    observed_target,
    parallel_chain_map,
    run_seed,
)

logger = logging.getLogger(__name__)

__all__ = ["BlackJAXESSMethod", "elliptical_slice"]


# ---------------------------------------------------------------------------
# Gaussian-prior detection
# ---------------------------------------------------------------------------


def _gaussian_prior_params(prior: Distribution) -> tuple[Array, Array] | None:
    """Extract ``(mean, cov)`` if *prior* is Gaussian; ``None`` otherwise.

    Parameters are returned in the flat-vector layout matching the
    convention used by the other MCMC backends — concatenation in
    the order of the prior's components for composite priors.

    Recognises:

    * :class:`~probpipe.MultivariateNormal` — ``(loc, cov)`` directly.
    * :class:`~probpipe.Normal` — ``(loc, diag(scale**2))`` with the scalar
      promoted to a length-1 vector.
    * a factored joint whose factors are each themselves recognised, as a
      :class:`~probpipe.families.FactoredMultivariateGaussian` is —
      block-diagonal assembly in the order of its components, the factors
      being independent.

    Returns ``None`` for any other distribution: mixtures of Gaussians,
    conditional Gaussians whose covariance depends on other parameters,
    Gamma / Beta / Dirichlet / non-Gaussian priors, and improper
    priors (which have no ``_sample`` to draw the auxiliary from).
    """
    if isinstance(prior, MultivariateNormal):
        loc = jnp.atleast_1d(jnp.asarray(prior.loc))
        cov = jnp.atleast_2d(jnp.asarray(prior.cov))
        return loc, cov

    if isinstance(prior, Normal):
        loc = jnp.atleast_1d(jnp.asarray(prior.loc))
        scale = jnp.atleast_1d(jnp.asarray(prior.scale))
        return loc, jnp.diag(scale**2)

    if isinstance(prior, FactoredDistribution):
        locs: list[Array] = []
        covs: list[Array] = []
        for component in prior.factors:
            if not isinstance(component, Distribution):
                return None
            sub = _gaussian_prior_params(component)
            if sub is None:
                return None
            locs.append(sub[0])
            covs.append(sub[1])
        mean = jnp.concatenate(locs)
        cov = jsl.block_diag(*covs)
        return mean, cov

    return None


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def _run_ess_chains(
    loglikelihood_fn,
    init_position: Array,
    prior_mean: Array,
    prior_cov: Array,
    *,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    random_seed: int,
) -> tuple[list[Array], list[Array] | None, dict[str, np.ndarray]]:
    """Run ``num_chains`` ESS chains via ``lax.scan`` + ``vmap``.

    Returns ``(chains, warmup_chains_or_None, sample_stats_dict)``.
    """
    sampler = blackjax.elliptical_slice(
        loglikelihood_fn,
        mean=prior_mean,
        cov=prior_cov,
    )
    key = as_prng_key(random_seed)
    chain_keys = jax.random.split(key, num_chains)

    def run_one_chain(chain_key):
        warmup_key, sample_key = jax.random.split(chain_key)
        state = sampler.init(init_position)

        def step(state, k):
            state, info = sampler.step(k, state)
            return state, (state.position, info.subiter)

        if num_warmup > 0:
            warmup_keys = jax.random.split(warmup_key, num_warmup)
            state, (warmup_positions, _warmup_subiter) = jax.lax.scan(
                step,
                state,
                warmup_keys,
            )
        else:
            warmup_positions = jnp.empty((0, init_position.shape[0]), dtype=init_position.dtype)

        sample_keys = jax.random.split(sample_key, num_results)
        _, (positions, subiter) = jax.lax.scan(step, state, sample_keys)
        return positions, warmup_positions, subiter

    positions_all, warmups_all, subiter_all = parallel_chain_map(run_one_chain, chain_keys)
    chains = [positions_all[c] for c in range(num_chains)]
    warmups = [warmups_all[c] for c in range(num_chains)] if num_warmup > 0 else None
    sample_stats = {"subiter": np.asarray(subiter_all)}
    return chains, warmups, sample_stats


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------


def elliptical_slice(
    model: Distribution,
    data: Record | Mapping[str, Any],
    *,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    init: ArrayLike | None = None,
    random_seed: int | None = None,
) -> EmpiricalDistribution:
    """Elliptical slice sampling of a joint with a Gaussian prior, at observed fields.

    The joint is a factored one, such as ``likelihood * prior``. The factors that
    produce the observed fields are the likelihood, and the others are the
    prior, which must be Gaussian, as :func:`_gaussian_prior_params` recognizes.

    Parameters
    ----------
    model : Distribution
        The factored joint, whose factors that produce no observed field form a
        ``MultivariateNormal``, a ``Normal``, or a factored joint over those.
    data : Record or Mapping[str, Any]
        The observed values, keyed by the fields they bind.
    num_results, num_warmup, num_chains
        MCMC tuning parameters.
    init
        Initial chain state in the flat parameter vector. Defaults to
        a sample from the prior.
    random_seed
        Seed for chain initialisation and sampling RNG. Omitted, the run's
        seed is a workflow-owned random event, which ``workflow_run`` fixes.

    Returns
    -------
    EmpiricalDistribution
        Posterior samples with chain structure and an annotations
        ArviZ-shaped ``DataTree`` carrying per-step ``subiter`` counts
        (the inner shrinkage iterations BlackJAX performed before
        accepting the proposal).

    Raises
    ------
    TypeError
        If the joint at *data* has no prior and likelihood factors, or the prior
        is not Gaussian.
    """
    return _elliptical_slice(
        observed_target(model, data),
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        init=init,
        random_seed=run_seed({"random_seed": random_seed}, "blackjax_ess"),
    )


def _elliptical_slice(
    target: Any,
    *,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    init: ArrayLike | None,
    random_seed: int,
) -> EmpiricalDistribution:
    """Elliptical slice chains on the unnormalized conditional *target* of a factored joint."""
    factors = model_factors(target)
    if factors is None:
        raise TypeError(
            "elliptical_slice requires a factored joint at observed values of its fields, "
            "whose other factors form the prior"
        )
    gp = _gaussian_prior_params(factors.prior)
    if gp is None:
        raise TypeError(
            f"elliptical_slice requires a Gaussian prior; got {type(factors.prior).__name__}"
        )
    prior_mean, prior_cov = gp
    init_state = get_init_state(factors.prior, init, random_seed=random_seed)

    # ESS consumes the log-likelihood alone — the prior is folded into
    # the proposal mechanism via the ellipse construction.
    loglikelihood_fn = likelihood_flat(factors)

    chains, warmups, sample_stats = _run_ess_chains(
        loglikelihood_fn,
        init_state,
        prior_mean,
        prior_cov,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        random_seed=random_seed,
    )

    annotations = build_mcmc_datatree(chains, sample_stats, warmup_chains=warmups)
    return make_posterior(
        chains,
        parents=(target,),
        method="elliptical_slice",
        annotations=annotations,
        event_spec=target.event_spec,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
    )


# ---------------------------------------------------------------------------
# Registry method
# ---------------------------------------------------------------------------


class BlackJAXESSMethod(InferenceMethod):
    """Elliptical slice sampling on top of ``blackjax.elliptical_slice``.

    Registered as ``blackjax_elliptical_slice`` at priority 75. Applies to the
    unnormalized conditional of a factored joint at observed fields whose other
    factors form a Gaussian prior, detected by :func:`_gaussian_prior_params`,
    and whose likelihood is JAX-traceable; ``check()`` enforces that class, not
    the priority.

    Notes
    -----
    Self-tuning and robust without per-model hyperparameter selection, so it
    ranks above ``blackjax_rwmh`` (55) and below the NUTS backends (82–88).
    """

    _method_options = ("init", "num_chains", "num_results", "num_warmup", "random_seed")

    @property
    def name(self) -> str:
        return "blackjax_elliptical_slice"

    def supported_types(self) -> tuple[type, ...]:
        return (_UnnormalizedConditional,)

    @property
    def priority(self) -> int:
        return 75

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target is a joint with a Gaussian prior at data, with a traceable likelihood."""
        factors = model_factors(target)
        if factors is None:
            return Feasibility(
                feasible=False,
                description=(
                    "ESS requires a factored joint at observed values of its fields, whose "
                    "other factors form the prior"
                ),
            )
        gp = _gaussian_prior_params(factors.prior)
        if gp is None:
            return Feasibility(
                feasible=False,
                description=f"ESS requires a Gaussian prior; got {type(factors.prior).__name__}",
            )
        # The runner traces the BlackJAX ESS step under ``lax.scan``;
        # there's no eager fallback. Catching non-traceable likelihoods
        # here lets auto-dispatch slide down to RWMH instead.
        try:
            if not is_jax_traceable(likelihood_flat(factors), jnp.asarray(gp[0])):
                return Feasibility(
                    feasible=False,
                    description="Log-likelihood is not JAX-traceable",
                )
        except Exception as e:
            return Feasibility(
                feasible=False,
                description=str(e),
            )
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> EmpiricalDistribution:
        """Elliptical slice chains on the joint the target conditions, at its data."""
        self._check_options(kwargs)
        return _elliptical_slice(
            target,
            num_results=kwargs.get("num_results", 1000),
            num_warmup=kwargs.get("num_warmup", 500),
            num_chains=kwargs.get("num_chains", 4),
            init=kwargs.get("init"),
            random_seed=run_seed(kwargs, self.name),
        )
