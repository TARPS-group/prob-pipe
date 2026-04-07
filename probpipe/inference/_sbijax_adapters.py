"""Adapter helpers bridging ProbPipe distributions to sbijax's API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from ..modeling._likelihood import GenerativeLikelihood

__all__: list[str] = []

PARAM_KEY = "theta"
"""Key used for the parameter dimension in sbijax prior/posterior dicts."""


def import_sbijax() -> Any:
    """Import sbijax, undoing its global matplotlib style side effect.

    sbijax's ``__init__`` calls ``matplotlib.style.use`` with a stylesheet that
    sets ``text.usetex: True``, which breaks plotting on systems without
    a LaTeX installation. We restore the rcParams that the style touched.
    """
    import matplotlib as mpl
    saved = dict(mpl.rcParams)
    import sbijax  # noqa: F401
    mpl.rcParams.update(saved)
    return sbijax


def is_tfp_backed(dist: Any) -> bool:
    """Check whether a distribution has an underlying TFP distribution."""
    return hasattr(dist, "_tfp_dist")


def coerce_observable(observed: Any) -> Array:
    """Coerce observed data to a JAX array with at least one dimension."""
    return jnp.atleast_1d(jnp.asarray(observed))


def adapt_prior(prior: Any) -> Callable[[], Any]:
    """Convert a ProbPipe distribution to an sbijax prior factory.

    sbijax expects a zero-argument callable returning a TFP
    ``JointDistributionNamed``.  This function extracts the underlying
    TFP distribution and wraps it accordingly.

    Parameters
    ----------
    prior : Distribution
        A ProbPipe ``TFPDistribution`` (or subclass) with a ``_tfp_dist``
        attribute.

    Returns
    -------
    Callable[[], tfd.JointDistributionNamed]
        Zero-arg factory suitable for sbijax model constructors.
    """
    from tensorflow_probability.substrates.jax import distributions as tfd

    if not is_tfp_backed(prior):
        raise TypeError(
            f"adapt_prior requires a TFP-backed distribution, "
            f"got {type(prior).__name__}"
        )
    tfp_dist = prior._tfp_dist
    # sbijax requires parameters to have at least one event dimension.
    # Wrap scalar distributions so samples are shape (1,) not ().
    if tfp_dist.event_shape.rank == 0:
        tfp_dist = tfd.Sample(tfp_dist, sample_shape=[1])

    def prior_fn():
        return tfd.JointDistributionNamed(
            {PARAM_KEY: tfp_dist}, batch_ndims=0,
        )

    return prior_fn


def adapt_simulator(
    simulator: GenerativeLikelihood,
) -> Callable[[PRNGKey, dict], Array]:
    """Convert a ProbPipe GenerativeLikelihood to an sbijax simulator function.

    sbijax expects ``simulator_fn(seed, params_dict) -> data`` where
    ``params_dict`` is a dict keyed by prior parameter names (e.g.
    ``{"theta": array}``).  This wrapper adapts from the dict-keyed
    format to the ProbPipe ``generate_data(params, n_samples, *, key)``
    signature.

    Parameters
    ----------
    simulator : GenerativeLikelihood
        Must have ``generate_data(params, n_samples, *, key)`` method.

    Returns
    -------
    Callable[[PRNGKey, dict], Array]
        Function suitable for sbijax model constructors.
    """

    def _single_sim(seed: PRNGKey, params: Array) -> Array:
        return jnp.atleast_1d(simulator.generate_data(params, 1, key=seed).squeeze(0))

    def simulator_fn(seed: PRNGKey, theta: dict) -> Array:
        params = theta[PARAM_KEY]
        seeds = jax.random.split(seed, params.shape[0])
        return jax.vmap(_single_sim)(seeds, params)

    return simulator_fn


def default_summary_fn(data: Array) -> Array:
    """Identity summary statistics — pass data through unchanged."""
    return jnp.atleast_2d(data)


def default_distance_fn(s1: Array, s2: Array) -> Array:
    """Euclidean distance between summary statistics."""
    return jnp.sqrt(jnp.sum((s1 - s2) ** 2, axis=-1))


def extract_chains(posterior_idata: Any, param_key: str = PARAM_KEY) -> list[Array]:
    """Extract posterior samples from ArviZ InferenceData as a list of chain arrays.

    Parameters
    ----------
    posterior_idata : InferenceData
        ArviZ ``InferenceData`` returned by sbijax's ``sample_posterior``.
    param_key : str
        Key under the posterior group to extract.

    Returns
    -------
    list of Array
        Per-chain sample arrays, each shaped ``(num_draws, *event_shape)``.
    """
    posterior = posterior_idata.posterior
    samples = posterior[param_key].values  # shape: (chains, draws, *event)
    return [jnp.asarray(samples[i]) for i in range(samples.shape[0])]
