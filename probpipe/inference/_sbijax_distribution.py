"""TrainedSBIModel: amortized posterior from simulation-based inference."""

from __future__ import annotations

from typing import Any

import jax

from ..core.distribution import Distribution
from ..core.protocols import SupportsConditioning
from ._mcmc_distribution import ApproximateDistribution, make_posterior
from ._sbijax_adapters import coerce_observable, extract_chains

__all__ = ["TrainedSBIModel"]


class TrainedSBIModel(Distribution, SupportsConditioning):
    """Trained amortized SBI model.

    Wraps a trained sbijax model and its parameters.  Implements
    :class:`~probpipe.core.protocols.SupportsConditioning` so that
    ``condition_on(model, observed)`` uses the trained neural network
    for fast posterior sampling — no re-training needed per observation.

    Parameters
    ----------
    sbijax_model : object
        A fitted sbijax model (e.g., ``sbijax.NPE``, ``sbijax.NLE``).
    params : dict
        Trained neural network parameters from ``model.fit()``.
    prior : Distribution
        The original prior distribution (for provenance).
    algorithm : str
        Name of the SBI algorithm (e.g., ``"sbijax_npe"``).
    n_samples : int
        Default number of posterior samples per ``_condition_on`` call.
    random_seed : int
        Base random seed for posterior sampling.
    """

    def __init__(
        self,
        sbijax_model: Any,
        params: dict,
        prior: Distribution,
        *,
        algorithm: str = "sbijax_npe",
        n_samples: int = 4000,
        random_seed: int = 0,
    ):
        self._sbijax_model = sbijax_model
        self._params = params
        self._prior = prior
        self._algorithm = algorithm
        self._n_samples = n_samples
        self._random_seed = random_seed
        self._name = f"TrainedSBIModel({algorithm})"

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> ApproximateDistribution:
        """Sample posterior using the trained amortized model.

        Parameters
        ----------
        observed : array-like
            Observed data to condition on.
        **kwargs
            Optional overrides: ``n_samples``, ``random_seed``.

        Returns
        -------
        ApproximateDistribution
            Posterior samples with chain structure and an attached
            ArviZ ``DataTree`` of inference data.
        """
        n_samples = kwargs.get("n_samples", self._n_samples)
        random_seed = kwargs.get("random_seed", self._random_seed)
        key = jax.random.PRNGKey(random_seed)

        observable = coerce_observable(observed)

        posterior_idata, _ = self._sbijax_model.sample_posterior(
            key, self._params, observable=observable, n_samples=n_samples,
        )

        chains = extract_chains(posterior_idata)

        return make_posterior(
            chains,
            parents=(self._prior,),
            algorithm=self._algorithm,
            inference_data=posterior_idata,
        )

    def __repr__(self) -> str:
        return (
            f"TrainedSBIModel(algorithm={self._algorithm!r}, "
            f"n_samples={self._n_samples})"
        )
