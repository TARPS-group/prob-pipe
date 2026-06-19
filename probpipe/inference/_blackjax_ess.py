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

ProbPipe registers this method at priority 75 (tier 71-80: self-tuning,
converges robustly without per-model hyperparameter selection). Its
``check()`` is strict — ``SimpleModel`` only, Gaussian prior detected
by :func:`_gaussian_prior_params`, observed data required — so
auto-dispatch only fires when the kernel is genuinely applicable.
"""

from __future__ import annotations

import logging
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..custom_types import Array, ArrayLike
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import (
    build_likelihood_flat,
    build_mcmc_datatree,
    extract_event_template,
    get_init_state,
    get_prior,
    is_jax_traceable,
    is_simple_model,
    parallel_chain_map,
)
from ._registry import InferenceMethod

logger = logging.getLogger(__name__)

__all__ = ["BlackJAXESSMethod", "elliptical_slice"]


# ---------------------------------------------------------------------------
# Gaussian-prior detection
# ---------------------------------------------------------------------------


def _gaussian_prior_params(prior: Distribution) -> tuple[Array, Array] | None:
    """Extract ``(mean, cov)`` if *prior* is Gaussian; ``None`` otherwise.

    Parameters are returned in the flat-vector layout matching the
    convention used by the other MCMC backends — concatenation in
    ``event_template.fields`` order for composite priors.

    Recognises:

    * :class:`~probpipe.distributions.MultivariateNormal` — ``(loc, cov)``
      directly.
    * :class:`~probpipe.distributions.JointGaussian` — a named multi-field
      Gaussian *with cross-covariance*. Its ``(mean_vector, covariance)``
      are already laid out in ``event_template.fields`` order, so they
      plug straight in; unlike a ``ProductDistribution`` of Gaussians, the
      off-diagonal cross-field covariance is preserved.
    * :class:`~probpipe.distributions.Normal` — ``(loc, diag(scale**2))``
      with the scalar / batch promoted to a length-1 vector.
    * :class:`~probpipe.distributions.ProductDistribution` whose
      components are each themselves recognised — block-diagonal
      assembly preserving the field order (the independent case).

    Returns ``None`` for any other distribution: mixtures of Gaussians,
    conditional Gaussians whose covariance depends on other parameters,
    Gamma / Beta / Dirichlet / non-Gaussian priors, and improper
    priors (which have no ``_sample`` to draw the auxiliary from).
    """
    from ..distributions import (
        JointGaussian,
        MultivariateNormal,
        Normal,
        ProductDistribution,
    )

    if isinstance(prior, MultivariateNormal):
        loc = jnp.atleast_1d(jnp.asarray(prior.loc))
        cov = jnp.atleast_2d(jnp.asarray(prior.cov))
        return loc, cov

    if isinstance(prior, JointGaussian):
        mean = jnp.atleast_1d(jnp.asarray(prior.mean_vector))
        cov = jnp.atleast_2d(jnp.asarray(prior.covariance))
        return mean, cov

    if isinstance(prior, Normal):
        loc = jnp.atleast_1d(jnp.asarray(prior.loc))
        scale = jnp.atleast_1d(jnp.asarray(prior.scale))
        return loc, jnp.diag(scale**2)

    if isinstance(prior, ProductDistribution):
        locs: list[Array] = []
        covs: list[Array] = []
        for _name, component in prior.components.items():
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
    key = jax.random.PRNGKey(random_seed)
    chain_keys = jax.random.split(key, num_chains)

    def run_one_chain(chain_key):
        warmup_key, sample_key = jax.random.split(chain_key)
        state = sampler.init(init_position)

        def step(state, k):
            state, info = sampler.step(k, state)
            return state, (state.position, info.subiter)

        if num_warmup > 0:
            warmup_keys = jax.random.split(warmup_key, num_warmup)
            state, (warmup_positions, warmup_subiter) = jax.lax.scan(
                step,
                state,
                warmup_keys,
            )
        else:
            warmup_positions = jnp.empty((0, init_position.shape[0]), dtype=init_position.dtype)
            warmup_subiter = jnp.empty((0,), dtype=jnp.int32)

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
    data: ArrayLike,
    *,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    init: ArrayLike | None = None,
    random_seed: int = 0,
) -> ApproximateDistribution:
    """Elliptical slice sampling for Gaussian-prior ``SimpleModel`` targets.

    Parameters
    ----------
    model : SimpleModel
        Must have a Gaussian prior recognised by
        :func:`_gaussian_prior_params` (``MultivariateNormal``,
        ``JointGaussian``, ``Normal``, or a ``ProductDistribution`` over
        those).
    data
        Observed data, passed to ``model.likelihood.log_likelihood``.
    num_results, num_warmup, num_chains
        MCMC tuning parameters.
    init
        Initial chain state in the flat parameter vector. Defaults to
        a sample from the prior.
    random_seed
        Seed for chain initialisation and sampling RNG.

    Returns
    -------
    ApproximateDistribution
        Posterior samples with chain structure and an auxiliary
        ArviZ-shaped ``DataTree`` carrying per-step ``subiter`` counts
        (the inner shrinkage iterations BlackJAX performed before
        accepting the proposal).
    """
    if not is_simple_model(model):
        raise TypeError(
            "elliptical_slice requires a SimpleModel "
            "(bare SupportsUnnormalizedLogProb has no prior/likelihood "
            "decomposition)"
        )
    prior = model._prior
    likelihood = model._likelihood
    gp = _gaussian_prior_params(prior)
    if gp is None:
        raise TypeError(f"elliptical_slice requires a Gaussian prior; got {type(prior).__name__}")
    if data is None:
        raise TypeError("elliptical_slice requires observed data")

    prior_mean, prior_cov = gp
    init_state = get_init_state(model, init, random_seed=random_seed)

    # ESS consumes the log-likelihood alone — the prior is folded into
    # the proposal mechanism via the ellipse construction.
    loglikelihood_fn = build_likelihood_flat(prior, likelihood, data)

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

    auxiliary = build_mcmc_datatree(chains, sample_stats, warmup_chains=warmups)
    event_template = extract_event_template(model)
    return make_posterior(
        chains,
        parents=(prior,),
        algorithm="elliptical_slice",
        auxiliary=auxiliary,
        event_template=event_template,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
    )


# ---------------------------------------------------------------------------
# Registry method
# ---------------------------------------------------------------------------


class BlackJAXESSMethod(InferenceMethod):
    """Elliptical slice sampling on top of ``blackjax.elliptical_slice``.

    Tier 71-80 (self-tuning, converges robustly without per-model
    hyperparameter selection). Priority 75. The narrow Gaussian-prior
    feasibility class is enforced in ``check()``, not by priority.
    """

    @property
    def name(self) -> str:
        return "blackjax_elliptical_slice"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return 75

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not is_simple_model(dist):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description=(
                    "ESS requires a SimpleModel; bare "
                    "SupportsUnnormalizedLogProb has no prior/likelihood "
                    "decomposition"
                ),
            )
        prior = get_prior(dist)
        gp = _gaussian_prior_params(prior)
        if gp is None:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description=(f"ESS requires a Gaussian prior; got {type(prior).__name__}"),
            )
        if observed is None:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="ESS requires observed data for the likelihood",
            )
        if isinstance(observed, dict):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="Does not support dict-based conditioning",
            )
        # The runner traces the BlackJAX ESS step under ``lax.scan``;
        # there's no eager fallback. Catching non-traceable likelihoods
        # here lets auto-dispatch slide down to RWMH instead.
        try:
            likelihood = dist._likelihood
            flat_init = jnp.asarray(gp[0])
            loglikelihood_fn = build_likelihood_flat(prior, likelihood, observed)
            if not is_jax_traceable(loglikelihood_fn, flat_init):
                return MethodInfo(
                    feasible=False,
                    method_name=self.name,
                    description="Log-likelihood is not JAX-traceable",
                )
        except Exception as e:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description=str(e),
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        return elliptical_slice(
            dist,
            observed,
            num_results=kwargs.get("num_results", 1000),
            num_warmup=kwargs.get("num_warmup", 500),
            num_chains=kwargs.get("num_chains", 1),
            init=kwargs.get("init"),
            random_seed=kwargs.get("random_seed", 0),
        )
