"""sbijax integration for ProbPipe.

This module consolidates everything needed to use the `sbijax` library as
an inference backend:

* Adapter helpers that bridge ProbPipe distributions / generative
  likelihoods to the dict-based, TFP-flavoured API that sbijax expects.
* :func:`sbi_learn_conditional` and :func:`sbi_learn_likelihood`,
  workflow functions that train amortized conditional posterior or
  likelihood emulators from a prior and simulator.
* Two return-type tracks for trained models:

  - **Direct samplers** (NPE, FMPE, CMPE) — sbijax samples directly from
    the trained network with no MCMC step.  We return a
    :class:`_DirectSamplerSBIModel` that implements
    :class:`~probpipe.core.protocols.SupportsConditioning`, so
    ``condition_on(model, obs)`` produces fast amortized samples.
  - **MCMC-required** (NLE, NRE) — sbijax only learns a likelihood
    (NLE) or likelihood-ratio (NRE), and posterior sampling needs MCMC.
    Rather than hard-coding sbijax's blackjax-based MCMC we wrap the
    trained network as a :class:`~probpipe.modeling.Likelihood` and
    return a :class:`~probpipe.modeling.SimpleModel`.  ``condition_on``
    then dispatches through the standard probpipe inference registry,
    so users can pick NUTS / nutpie / RWMH / etc.

* :class:`SbiSMCABCMethod`, a non-amortized SMC-ABC inference method
  registered with ProbPipe's inference registry.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .._utils import prod
from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.node import workflow_function
from ..core.protocols import SupportsConditioning
from ..custom_types import Array, PRNGKey
from ..modeling._likelihood import GenerativeLikelihood, Likelihood
from ..modeling._simple import SimpleModel
from ._mcmc_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod

# sbijax's ``__init__`` calls ``matplotlib.style.use`` with a stylesheet
# that sets ``text.usetex: True``, which breaks plotting on systems
# without LaTeX.  Snapshot and restore rcParams around the import so the
# side effect is contained to this module-load.
import matplotlib as _mpl

_saved_rcparams = dict(_mpl.rcParams)
import sbijax  # noqa: E402
from sbijax.nn import make_cm, make_cnf, make_maf, make_mlp  # noqa: E402

_mpl.rcParams.update(_saved_rcparams)
del _saved_rcparams, _mpl

__all__ = [
    "sbi_learn_conditional",
    "sbi_learn_likelihood",
    "SbiSMCABCMethod",
    "DirectSamplerSBIModel",
    "PARAM_KEY",
]


PARAM_KEY = "model_parameters"
"""Key used for the parameter dimension in sbijax prior/posterior dicts."""


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _is_tfp_backed(dist: object) -> bool:
    """Check whether a distribution exposes the underlying TFP distribution."""
    return hasattr(dist, "_tfp_dist")


def _coerce_observable(observed: Any) -> Array:
    """Coerce observed data to a JAX array with at least one dimension."""
    return jnp.atleast_1d(jnp.asarray(observed))


def _adapt_prior(prior: Distribution) -> Callable[[], Any]:
    """Convert a ProbPipe TFP-backed distribution to an sbijax prior factory."""
    from tensorflow_probability.substrates.jax import distributions as tfd

    if not _is_tfp_backed(prior):
        raise TypeError(
            f"adapt_prior requires a TFP-backed distribution, "
            f"got {type(prior).__name__}"
        )
    tfp_dist = prior._tfp_dist
    # sbijax requires parameters to have at least one event dimension.
    if tfp_dist.event_shape.rank == 0:
        tfp_dist = tfd.Sample(tfp_dist, sample_shape=[1])

    def prior_fn() -> Any:
        return tfd.JointDistributionNamed(
            {PARAM_KEY: tfp_dist}, batch_ndims=0,
        )

    return prior_fn


def _adapt_simulator(
    simulator: GenerativeLikelihood,
) -> Callable[[PRNGKey, dict[str, Array]], Array]:
    """Convert a ProbPipe GenerativeLikelihood to an sbijax simulator function."""

    def _single_sim(seed: PRNGKey, params: Array) -> Array:
        return jnp.atleast_1d(simulator.generate_data(params, 1, key=seed)[0])

    def simulator_fn(seed: PRNGKey, theta: dict[str, Array]) -> Array:
        params = theta[PARAM_KEY]
        seeds = jax.random.split(seed, params.shape[0])
        return jax.vmap(_single_sim)(seeds, params)

    return simulator_fn


def _default_summary_fn(data: Array) -> Array:
    """Identity summary statistics — pass data through unchanged."""
    return jnp.atleast_2d(data)


def _default_distance_fn(s1: Array, s2: Array) -> Array:
    """Euclidean distance between summary statistics."""
    return jnp.sqrt(jnp.sum((s1 - s2) ** 2, axis=-1))


def _extract_chains(
    posterior_idata: Any, param_key: str = PARAM_KEY
) -> list[Array]:
    """Extract per-chain posterior samples from an ArviZ DataTree."""
    posterior = posterior_idata.posterior
    samples = posterior[param_key].values  # (chains, draws, *event)
    return [jnp.asarray(samples[i]) for i in range(samples.shape[0])]


def _prior_ndim(prior: Distribution) -> int:
    """Flat parameter dimensionality from a prior's event shape."""
    if hasattr(prior, "event_shape") and prior.event_shape:
        return prod(tuple(prior.event_shape))
    return 1


# ---------------------------------------------------------------------------
# Direct-sampler track: NPE, FMPE, CMPE
# ---------------------------------------------------------------------------


# Each entry is (sbi_class, default_network_factory, ndim_source) where
# ndim_source is one of:
#   "theta" — factory called with the parameter dimension (NPE/FMPE/CMPE)
#   "data"  — factory called with the observation dimension (NLE)
#   "none"  — factory called with no arguments (NRE → MLP classifier)
_DIRECT_SAMPLER_BUILDERS: dict[
    str, tuple[Callable[..., Any], Callable[..., Any], str]
] = {
    "npe": (sbijax.NPE, make_maf, "theta"),
    "fmpe": (sbijax.FMPE, make_cnf, "theta"),
    "cmpe": (sbijax.CMPE, make_cm, "theta"),
}

_MCMC_SBI_BUILDERS: dict[
    str, tuple[Callable[..., Any], Callable[..., Any], str]
] = {
    "nle": (sbijax.NLE, make_maf, "data"),
    "nre": (sbijax.NRE, make_mlp, "none"),
}


class DirectSamplerSBIModel(Distribution, SupportsConditioning):
    """Trained amortized SBI model that samples directly from a neural network.

    Wraps a trained sbijax model whose ``sample_posterior`` is a direct
    forward-pass on the trained density estimator (no MCMC under the
    hood) — currently NPE, FMPE, and CMPE.  Implements
    :class:`~probpipe.core.protocols.SupportsConditioning` so that
    ``condition_on(model, observed)`` returns posterior samples in one
    forward pass per observation.
    """

    def __init__(
        self,
        sbijax_model: Any,
        params: Any,
        prior: Distribution,
        *,
        algorithm: str,
        n_samples: int = 4000,
        random_seed: int = 0,
    ):
        self._sbijax_model = sbijax_model
        self._params = params
        self._prior = prior
        self._algorithm = algorithm
        self._n_samples = n_samples
        self._random_seed = random_seed

    @property
    def name(self) -> str:
        return f"DirectSamplerSBIModel({self._algorithm})"

    def _condition_on(
        self, observed: Any, /, **kwargs: Any
    ) -> ApproximateDistribution:
        n_samples = kwargs.get("n_samples", self._n_samples)
        random_seed = kwargs.get("random_seed", self._random_seed)
        key = jax.random.PRNGKey(random_seed)

        observable = _coerce_observable(observed)
        posterior_idata, _ = self._sbijax_model.sample_posterior(
            key, self._params, observable=observable, n_samples=n_samples,
        )
        chains = _extract_chains(posterior_idata)
        return make_posterior(
            chains,
            parents=(self._prior,),
            algorithm=self._algorithm,
            inference_data=posterior_idata,
        )

    def __repr__(self) -> str:
        return (
            f"DirectSamplerSBIModel(algorithm={self._algorithm!r}, "
            f"n_samples={self._n_samples})"
        )


# ---------------------------------------------------------------------------
# MCMC-required track: NLE, NRE → SimpleModel(prior, NeuralLikelihood)
# ---------------------------------------------------------------------------


class _NLELikelihood:
    """Wrap a trained sbijax NLE model as a :class:`Likelihood`.

    The trained network represents :math:`p(y \\mid \\theta)`.  We expose
    its ``log_prob`` so that probpipe's standard inference registry can
    drive MCMC over the parameters.
    """

    def __init__(self, sbijax_model: Any, params: Any):
        self._sbijax_model = sbijax_model
        self._params = params

    def log_likelihood(self, params: Array, data: Array) -> Array:
        observable = jnp.atleast_2d(jnp.asarray(data))
        flat, _ = ravel_pytree(params)
        theta = jnp.tile(flat, [observable.shape[0], 1])
        lp = self._sbijax_model.model.apply(
            self._params,
            rng=None,
            method="log_prob",
            y=observable,
            x=theta,
        )
        return jnp.sum(lp)


class _NRELikelihood:
    """Wrap a trained sbijax NRE model as a :class:`Likelihood`.

    The trained network represents the log-density-ratio
    :math:`\\log r(y, \\theta) \\propto \\log p(y \\mid \\theta) - \\log p(y)`.
    We return only the ratio term — the prior log-density is added by
    :class:`~probpipe.modeling.SimpleModel`.
    """

    def __init__(self, sbijax_model: Any, params: Any):
        self._sbijax_model = sbijax_model
        self._params = params

    def log_likelihood(self, params: Array, data: Array) -> Array:
        observable = jnp.atleast_2d(jnp.asarray(data))
        flat, _ = ravel_pytree(params)
        theta = jnp.tile(flat, [observable.shape[0], 1]).reshape(
            observable.shape[0], -1
        )
        joint = jnp.concatenate([observable, theta], axis=-1)
        lr = self._sbijax_model.model.apply(
            self._params, joint, is_training=False,
        )
        return jnp.sum(lr)


# ---------------------------------------------------------------------------
# Workflow functions
# ---------------------------------------------------------------------------


ConditionalSBIMethod = Literal["npe", "fmpe", "cmpe"]
LikelihoodSBIMethod = Literal["nle", "nre"]


def _probe_data_ndim(
    prior: Distribution, simulator: GenerativeLikelihood, key: PRNGKey
) -> int:
    """Probe the flat observation dimensionality via one simulator call."""
    probe_params = prior._tfp_dist.sample(seed=key)
    probe_data = simulator.generate_data(
        jnp.asarray(probe_params), 1, key=key
    )[0]
    return int(jnp.size(probe_data))


def _train(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    builder: Callable[..., Any],
    default_net: Callable[..., Any],
    ndim_source: str,
    n_simulations: int,
    n_iter: int,
    batch_size: int,
    network_factory: Callable[..., Any] | None,
    random_seed: int,
    fit_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Shared simulate-and-fit step for both workflow functions."""
    if not _is_tfp_backed(prior):
        raise TypeError(
            f"sbi training requires a TFP-backed prior, "
            f"got {type(prior).__name__}"
        )
    fns = (_adapt_prior(prior), _adapt_simulator(simulator))
    factory = network_factory or default_net
    key = jax.random.PRNGKey(random_seed)
    key_probe, key_sim, key_fit = jax.random.split(key, 3)
    if ndim_source == "theta":
        network = factory(_prior_ndim(prior))
    elif ndim_source == "data":
        network = factory(_probe_data_ndim(prior, simulator, key_probe))
    elif ndim_source == "none":
        network = factory()
    else:
        raise ValueError(f"Unknown ndim_source: {ndim_source!r}")
    sbi_model = builder(fns, network)

    data, _ = sbi_model.simulate_data(key_sim, n_simulations=n_simulations)
    params, _ = sbi_model.fit(
        key_fit,
        data=data,
        n_iter=n_iter,
        batch_size=batch_size,
        **fit_kwargs,
    )
    return sbi_model, params


@workflow_function
def sbi_learn_conditional(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    method: ConditionalSBIMethod = "npe",
    n_simulations: int = 10_000,
    n_iter: int = 1000,
    batch_size: int = 128,
    network_factory: Callable[..., Any] | None = None,
    n_samples: int = 4000,
    random_seed: int = 0,
    **fit_kwargs: Any,
) -> DirectSamplerSBIModel:
    """Learn the conditional posterior :math:`p(\\theta \\mid y)` directly.

    Trains an amortized conditional density estimator (NPE / FMPE / CMPE)
    that samples directly from the trained network with no MCMC step.
    Returns a :class:`DirectSamplerSBIModel` implementing
    :class:`~probpipe.core.protocols.SupportsConditioning`, so
    ``condition_on(result, observed)`` produces fast amortized samples.

    Parameters
    ----------
    prior : Distribution
        Prior distribution over model parameters.  Must be TFP-backed.
    simulator : GenerativeLikelihood
        Must have ``generate_data(params, n_samples, *, key)`` method.
    method : str
        Conditional density estimator: ``"npe"`` (Neural Posterior
        Estimation), ``"fmpe"`` (Flow Matching Posterior Estimation), or
        ``"cmpe"`` (Consistency Model Posterior Estimation).
    n_simulations : int
        Number of (parameter, data) pairs to simulate for training.
    n_iter : int
        Number of training iterations.
    batch_size : int
        Training batch size.
    network_factory : callable or None
        Factory that returns an sbijax network.  Called with the
        parameter dimension for NPE/FMPE/CMPE.  If ``None``, a
        method-appropriate sbijax default is used.
    n_samples : int
        Default number of posterior samples per ``condition_on`` call.
    random_seed : int
        Base random seed for simulation, training, and sampling.
    **fit_kwargs
        Additional keyword arguments passed to ``model.fit()``.
    """
    method_lower = method.lower()
    if method_lower not in _DIRECT_SAMPLER_BUILDERS:
        raise ValueError(
            f"Unknown conditional SBI method: {method!r}. "
            f"Supported: {sorted(_DIRECT_SAMPLER_BUILDERS)}."
        )
    builder, default_net, ndim_source = _DIRECT_SAMPLER_BUILDERS[method_lower]

    sbi_model, params = _train(
        prior, simulator,
        builder=builder, default_net=default_net, ndim_source=ndim_source,
        n_simulations=n_simulations, n_iter=n_iter, batch_size=batch_size,
        network_factory=network_factory, random_seed=random_seed,
        fit_kwargs=fit_kwargs,
    )

    return DirectSamplerSBIModel(
        sbi_model,
        params,
        prior,
        algorithm=f"sbijax_{method_lower}",
        n_samples=n_samples,
        random_seed=random_seed,
    )


@workflow_function
def sbi_learn_likelihood(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    method: LikelihoodSBIMethod = "nle",
    n_simulations: int = 10_000,
    n_iter: int = 1000,
    batch_size: int = 128,
    network_factory: Callable[..., Any] | None = None,
    random_seed: int = 0,
    return_likelihood_only: bool = False,
    **fit_kwargs: Any,
) -> SimpleModel | Likelihood:
    """Learn an emulated likelihood :math:`p(y \\mid \\theta)` from simulations.

    Trains a neural likelihood (NLE) or likelihood-ratio (NRE) estimator.
    Because these methods only emulate the likelihood, posterior sampling
    requires a separate MCMC step — by returning a
    :class:`~probpipe.modeling.SimpleModel`, ``condition_on`` dispatches
    through the standard probpipe inference registry, letting the user
    pick any MCMC backend (NUTS, nutpie, RWMH, ...).

    Parameters
    ----------
    prior : Distribution
        Prior distribution over model parameters.  Must be TFP-backed.
        Used both to draw training parameters and (unless
        ``return_likelihood_only=True``) as the prior of the returned
        :class:`SimpleModel`.
    simulator : GenerativeLikelihood
        Must have ``generate_data(params, n_samples, *, key)`` method.
    method : str
        Likelihood estimator: ``"nle"`` (Neural Likelihood Estimation)
        or ``"nre"`` (Neural Ratio Estimation).
    n_simulations : int
        Number of (parameter, data) pairs to simulate for training.
    n_iter : int
        Number of training iterations.
    batch_size : int
        Training batch size.
    network_factory : callable or None
        Factory that returns an sbijax network.  Called with the
        observation (data) dimension for NLE, and with no arguments for
        NRE.  If ``None``, a method-appropriate sbijax default is used.
    random_seed : int
        Base random seed for simulation and training.
    return_likelihood_only : bool
        If ``True``, return just the trained
        :class:`~probpipe.modeling.Likelihood` object, with no prior
        attached.  Useful when the caller wants to combine the emulated
        likelihood with a different prior, plug it into an
        :class:`~probpipe.modeling.IncrementalConditioner`, etc.  If
        ``False`` (the default), return a
        :class:`~probpipe.modeling.SimpleModel` wrapping the original
        prior and the trained likelihood.
    **fit_kwargs
        Additional keyword arguments passed to ``model.fit()``.
    """
    method_lower = method.lower()
    if method_lower not in _MCMC_SBI_BUILDERS:
        raise ValueError(
            f"Unknown likelihood SBI method: {method!r}. "
            f"Supported: {sorted(_MCMC_SBI_BUILDERS)}."
        )
    builder, default_net, ndim_source = _MCMC_SBI_BUILDERS[method_lower]

    sbi_model, params = _train(
        prior, simulator,
        builder=builder, default_net=default_net, ndim_source=ndim_source,
        n_simulations=n_simulations, n_iter=n_iter, batch_size=batch_size,
        network_factory=network_factory, random_seed=random_seed,
        fit_kwargs=fit_kwargs,
    )

    likelihood: Likelihood
    if method_lower == "nle":
        likelihood = _NLELikelihood(sbi_model, params)
    else:  # "nre"
        likelihood = _NRELikelihood(sbi_model, params)

    if return_likelihood_only:
        return likelihood
    return SimpleModel(prior, likelihood, name=f"sbijax_{method_lower}")


# ---------------------------------------------------------------------------
# SMC-ABC inference method
# ---------------------------------------------------------------------------


class SbiSMCABCMethod(InferenceMethod):
    """SMC-ABC via sbijax.

    Operates on :class:`~probpipe.modeling.SimpleGenerativeModel` —
    requires a prior that supports sampling and a
    :class:`~probpipe.modeling.GenerativeLikelihood`.
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

    def check(
        self, dist: Any, observed: Any, **kwargs: Any
    ) -> MethodInfo:
        from ..modeling._simple_generative import SimpleGenerativeModel

        if not isinstance(dist, SimpleGenerativeModel):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="Requires SimpleGenerativeModel",
            )
        if not _is_tfp_backed(dist["parameters"]):
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

        prior_fn = _adapt_prior(dist["parameters"])
        simulator_fn = _adapt_simulator(dist["data"])
        fns = (prior_fn, simulator_fn)

        random_seed = kwargs.get("random_seed", 0)
        key = jax.random.PRNGKey(random_seed)

        summary_fn = kwargs.get("summary_fn", _default_summary_fn)
        distance_fn = kwargs.get("distance_fn", _default_distance_fn)
        sbi_model = sbijax.SMCABC(fns, summary_fn, distance_fn)

        # Workaround for sbijax bug: ``_chol_factor`` returns a scalar
        # when parameters are 1D because ``jnp.cov`` reduces (1, n) to
        # ().  Patch to ensure the covariance is always at least 2D.
        def _patched_chol_factor(particles: Any, cov_scale: float) -> Array:
            flat = jax.vmap(lambda x: ravel_pytree(x)[0])(particles)
            cov = jnp.atleast_2d(jnp.cov(flat.T)) * cov_scale
            return jnp.linalg.cholesky(cov)

        sbi_model._chol_factor = _patched_chol_factor

        observable = _coerce_observable(observed)

        # Forward only kwargs sample_posterior actually accepts.
        sig = inspect.signature(sbi_model.sample_posterior)
        accepted = set(sig.parameters) - {"rng_key", "observable", "self"}
        smcabc_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        posterior_idata, _ = sbi_model.sample_posterior(
            key, observable=observable, **smcabc_kwargs,
        )

        chains = _extract_chains(posterior_idata)
        return make_posterior(
            chains,
            parents=(dist["parameters"],),
            algorithm="sbijax_smcabc",
            inference_data=posterior_idata,
        )
