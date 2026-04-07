"""train_sbi: workflow function for training amortized SBI models."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import jax

from ..core.distribution import Distribution
from ..core.node import workflow_function
from ..core.protocols import SupportsSampling
from ..modeling._likelihood import GenerativeLikelihood
from ._sbijax_adapters import adapt_prior, adapt_simulator, is_tfp_backed
from ._sbijax_distribution import TrainedSBIModel
import sbijax
from sbijax.nn import make_maf

__all__ = ["train_sbi"]


@workflow_function
def train_sbi(
    prior: Distribution,
    simulator: GenerativeLikelihood,
    *,
    method: Literal["npe", "nle"] = "npe",
    n_simulations: int = 10_000,
    n_iter: int = 1000,
    batch_size: int = 128,
    network_factory: Callable | None = None,
    n_samples: int = 4000,
    random_seed: int = 0,
    **fit_kwargs: Any,
) -> TrainedSBIModel:
    """Train an amortized SBI model from a prior and simulator.

    Returns a :class:`TrainedSBIModel` that implements
    :class:`~probpipe.core.protocols.SupportsConditioning`, so
    ``condition_on(result, observed)`` produces fast posterior samples
    without re-training.

    Parameters
    ----------
    prior : Distribution
        Prior distribution over model parameters.  Must be TFP-backed.
    simulator : GenerativeLikelihood
        Must have ``generate_data(params, n_samples, *, key)`` method.
    method : str
        SBI algorithm: ``"npe"`` (Neural Posterior Estimation) or
        ``"nle"`` (Neural Likelihood Estimation).
    n_simulations : int
        Number of (parameter, data) pairs to simulate for training.
    n_iter : int
        Number of training iterations.
    batch_size : int
        Training batch size.
    network_factory : callable or None
        Factory for the density estimator neural network.  If ``None``,
        defaults to ``sbijax.nn.make_maf(ndim)`` where ``ndim`` is
        inferred from the prior's event shape.
    n_samples : int
        Default number of posterior samples when conditioning.
    random_seed : int
        Base random seed for simulation, training, and sampling.
    **fit_kwargs
        Additional keyword arguments passed to ``model.fit()``.

    Returns
    -------
    TrainedSBIModel
        Trained model wrapping the sbijax estimator and fitted parameters.
    """
    if not is_tfp_backed(prior):
        raise TypeError(
            f"train_sbi requires a TFP-backed prior, got {type(prior).__name__}"
        )

    prior_fn = adapt_prior(prior)
    simulator_fn = adapt_simulator(simulator)
    fns = (prior_fn, simulator_fn)

    ndim = 1
    if hasattr(prior, "event_shape") and prior.event_shape:
        ndim = math.prod(prior.event_shape)

    if network_factory is None:
        network = make_maf(ndim)
    else:
        network = network_factory(ndim)

    method_lower = method.lower()
    if method_lower == "npe":
        sbi_model = sbijax.NPE(fns, network)
    elif method_lower == "nle":
        sbi_model = sbijax.NLE(fns, network)
    else:
        raise ValueError(
            f"Unknown SBI method: {method!r}. Supported: 'npe', 'nle'."
        )

    algorithm_name = f"sbijax_{method_lower}"

    key = jax.random.PRNGKey(random_seed)
    key_sim, key_fit = jax.random.split(key)

    data, _ = sbi_model.simulate_data(key_sim, n_simulations=n_simulations)

    params, _ = sbi_model.fit(
        key_fit,
        data=data,
        n_iter=n_iter,
        batch_size=batch_size,
        **fit_kwargs,
    )

    return TrainedSBIModel(
        sbi_model,
        params,
        prior,
        algorithm=algorithm_name,
        n_samples=n_samples,
        random_seed=random_seed,
    )
