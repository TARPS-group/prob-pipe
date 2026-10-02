"""Shared helpers for the BayesFlow surrogate tests.

Kept out of the ``test_*`` modules so the two BayesFlow test files
(``test_bayesflow_likelihoods.py`` / ``test_bayesflow_posteriors.py``) share a
single implementation rather than duplicating it: the flat parameter vector of
a per-draw record, and the simulator kernel the learners train on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from probpipe import NumericArraySpec, OutputSpec
from probpipe.custom_types import Array, PRNGKey
from probpipe.distributions import ConditionalDistribution, Distribution
from probpipe.distributions._capabilities import SupportsConditionalSampling, SupportsSampling


def theta_vec(params: Any) -> Array:
    """Coerce a per-draw ``params`` object to its 1-D parameter vector.

    Mirrors ``BayesFlowLikelihood._theta_row``: structured records serialize via
    :meth:`~probpipe.NumericRecord.to_vector`; raw array-likes are ravelled.

    Parameters
    ----------
    params : Any
        A per-draw ``NumericRecord`` / ``NumericRecordBatch`` (training and
        predictive paths) or a flat array-like (the gradient-MCMC path).

    Returns
    -------
    Array
        The 1-D parameter vector.
    """
    return params.to_vector() if hasattr(params, "to_vector") else jnp.ravel(jnp.asarray(params))


class _SimulatedLaw(Distribution, SupportsSampling):
    """The law of one observation that a simulator draws, which only samples."""

    def __init__(self, name: str, event_spec: OutputSpec, draw: Callable[[PRNGKey], Array]):
        super().__init__(name, event_spec)
        object.__setattr__(self, "_draw", draw)

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        if sample_shape == ():
            return self._draw(key)
        keys = jax.random.split(key, math.prod(sample_shape))
        draws = jnp.stack([self._draw(k) for k in keys])
        return draws.reshape(*sample_shape, *draws.shape[1:])


class SimulatorKernel(ConditionalDistribution, SupportsConditionalSampling):
    """The kernel of one observation that ``simulate(params, key)`` draws.

    Its given slots are the components of *prior*, and its event is one
    observation of *shape*. The learners pass the given values as the record of
    the prior's fields, so *simulate* may read them by field, by leaf path, or as
    one flat vector through :func:`theta_vec`.
    """

    def __init__(
        self,
        prior: Any,
        shape: tuple[int, ...],
        simulate: Callable[[Any, PRNGKey], Array],
        *,
        name: str = "observation",
    ) -> None:
        super().__init__(
            name,
            dict(prior.event_spec.components),
            OutputSpec(**{name: NumericArraySpec(shape)}),
        )
        object.__setattr__(self, "_simulate", simulate)

    def _condition_on(self, given: Any, /, **options: Any) -> _SimulatedLaw:
        return _SimulatedLaw(self.label, self.event_spec, lambda key: self._simulate(given, key))

    def _conditional_sample(
        self, given: Any, key: PRNGKey, sample_shape: tuple[int, ...] = ()
    ) -> Array:
        return self._condition_on(given)._sample(key, sample_shape)
