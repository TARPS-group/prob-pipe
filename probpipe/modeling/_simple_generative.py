"""SimpleGenerativeModel: construct a model from a prior and generative likelihood."""

from __future__ import annotations

from typing import Any

from ..core.protocols import SupportsSampling
from ._base import ProbabilisticModel
from ._likelihood import GenerativeLikelihood

__all__ = ["SimpleGenerativeModel"]


class SimpleGenerativeModel[P, D](ProbabilisticModel[tuple[P, D]], SupportsSampling):
    """Generative probabilistic model as a joint over (parameters, data).

    A ``SimpleGenerativeModel[P, D]`` pairs a prior that supports
    sampling with a :class:`GenerativeLikelihood` that can generate
    synthetic data given parameters.  Unlike :class:`SimpleModel`, this
    does **not** require a log-density — making it suitable for
    simulation-based inference (SBI) and approximate Bayesian
    computation (ABC) methods.

    **Sampling:** implements :class:`~probpipe.core.protocols.SupportsSampling`
    via the obvious joint draw — sample parameters from the prior, then
    call ``likelihood.generate_data(params, ...)`` for the data.

    **Named components:** ``"parameters"`` (the prior) and ``"data"``
    (the generative likelihood).  Only ``"data"`` is conditionable.

    **Conditioning:** Use ``condition_on(model, data)`` — the inference
    method registry auto-selects an appropriate SBI or ABC method.
    ``SimpleGenerativeModel`` does not implement ``SupportsConditioning``
    directly.

    Parameters
    ----------
    prior : SupportsSampling[P]
        Prior distribution over model parameters.  Must support sampling.
    likelihood : GenerativeLikelihood[P, D]
        Must have a ``generate_data(params, n_samples, *, key)`` method.
    name : str or None
        Model name for provenance.
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def _sample(self, key, sample_shape=()):
        """Draw one joint ``(params, data)`` sample.

        Samples ``params`` from the prior, then calls
        ``likelihood.generate_data(params, 1, key=...)`` for the data.
        Returns a ``(params, data)`` tuple.

        Batched draws (non-empty ``sample_shape``) are not supported
        because ``GenerativeLikelihood.generate_data`` takes a single
        params tensor by contract; users who need a batched simulator
        run should drive a Python loop around ``_sample(key, ())``.
        """
        import jax

        if sample_shape != ():
            raise NotImplementedError(
                "SimpleGenerativeModel._sample(..., sample_shape=...) with a "
                "non-empty sample_shape is not supported. Loop over "
                "_sample(key, ()) with independent keys instead."
            )
        k_prior, k_data = jax.random.split(key)
        params = self._prior._sample(k_prior, ())
        data = self._likelihood.generate_data(params, 1, key=k_data)[0]
        return (params, data)

    def __init__(
        self,
        prior: SupportsSampling[P],
        likelihood: GenerativeLikelihood[P, D],
        *,
        name: str | None = None,
    ):
        if not isinstance(prior, SupportsSampling):
            raise TypeError(
                f"SimpleGenerativeModel requires a prior that supports SupportsSampling, "
                f"got {type(prior).__name__}"
            )
        if not isinstance(likelihood, GenerativeLikelihood):
            raise TypeError(
                f"SimpleGenerativeModel requires a GenerativeLikelihood, "
                f"got {type(likelihood).__name__}"
            )
        self._prior = prior
        self._likelihood = likelihood
        self._name_str = name

    # -- Distribution interface ---------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name_str

    # -- Named components interface ------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return ("parameters", "data")

    def __getitem__(self, key: str) -> Any:
        if key == "parameters":
            return self._prior
        if key == "data":
            return self._likelihood
        raise KeyError(
            f"Unknown component: {key!r}; "
            f"available: {self.fields}"
        )

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("parameters",)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        prior_name = type(self._prior).__name__
        lik_name = type(self._likelihood).__name__
        return f"SimpleGenerativeModel(prior={prior_name}, likelihood={lik_name})"
