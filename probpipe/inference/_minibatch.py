"""Minibatched random measure for stochastic-gradient inference.

Provides:

* :class:`MinibatchedDistribution` — a ``RandomMeasure`` whose
  draws are unbiased stochastic surrogates of the full-data
  unnormalized log-posterior. Consumed by stochastic-gradient MCMC
  kernels and (future) tempered SMC.
* :class:`_FixedMinibatchDistribution` (private) — one realisation of
  the measure, holding a single fixed minibatch.
* :class:`_RandomMinibatchLogProb` (private) — the
  ``RandomFunction`` from parameter records to arrays returned by
  ``random_unnormalized_log_prob(measure)``; its ``_sample(key)``
  yields a deterministic unnormalized-log-density callable for one
  minibatch.

For a model with prior :math:`p(\\theta)` and likelihood
:math:`p(\\mathcal{D} \\mid \\theta) = \\prod_i p(d_i \\mid \\theta)`,
the measure :math:`M` has draws :math:`\\tilde{D}_B` whose
unnormalized log-density is

.. math::

    \\log \\tilde{D}_B(\\theta) = \\log p(\\theta)
                                  + \\frac{N}{b} \\sum_{d \\in B}
                                    \\log p(d \\mid \\theta),

where :math:`B \\subset \\mathcal{D}` is a uniform random size-:math:`b`
subset of the data. The :math:`N/b` rescaling makes the gradient an
unbiased estimator of the full-data log-posterior gradient.

The likelihood is a kernel whose observations are conditionally independent
along the leading axis of its event, as a GLM likelihood's are, so the
log-density of a subset of them is read through the kernel's
``_observation_log_prob(given, value, rows)``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import jax
import jax.numpy as jnp

from ..core._specs import NumericArraySpec, OpaqueSpec, OutputSpec
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import _components_of
from ..families._random_functions import RandomFunction, RandomMeasure
from ..values._function_base import FunctionSpec

__all__ = ["MinibatchedDistribution"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_size(data: Any) -> int:
    """The number of observations in *data*, the length of its leading axis.

    Parameters
    ----------
    data : array-like
        The observed value of the likelihood's event.

    Returns
    -------
    int
        The dataset size ``N``, which bounds the minibatch size and scales a
        minibatch's log-likelihood by ``N / b``.

    Raises
    ------
    ValueError
        If *data* has no leading axis.
    """
    shape = jnp.shape(data)
    if not shape:
        raise ValueError(
            "MinibatchedDistribution takes observations along a leading axis; got a scalar"
        )
    return int(shape[0])


def _draw_indices(
    key: PRNGKey,
    n: int,
    batch_size: int,
    *,
    with_replacement: bool,
) -> Array:
    """Draw ``batch_size`` indices uniformly from ``range(n)``."""
    if with_replacement:
        return jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n)
    # Without replacement: random permutation, take first batch_size.
    return jax.random.permutation(key, n)[:batch_size]


def _parameter_declaration(prior: Any, component: str) -> OutputSpec:
    """The declaration of the parameters *prior* is a law over.

    A prior that is not a distribution declares nothing, so its parameters are
    opaque under *component*.
    """
    try:
        return prior.event_spec
    except AttributeError:
        return OutputSpec(**{component: OpaqueSpec()})


def _reads_observations(likelihood: Any) -> bool:
    """Whether *likelihood* is a kernel that scores a subset of its observations."""
    return isinstance(likelihood, ConditionalDistribution) and callable(
        getattr(likelihood, "_observation_log_prob", None)
    )


def _likelihood_given(prior: Any, likelihood: ConditionalDistribution, theta: Any) -> dict:
    """The likelihood's given values at *theta*, a draw of *prior*."""
    components = _components_of(prior.event_spec, theta)
    return {slot: components[slot] for slot in likelihood.given_spec}


# ---------------------------------------------------------------------------
# MinibatchedDistribution — the outer random measure
# ---------------------------------------------------------------------------


class MinibatchedDistribution(
    RandomMeasure,
    SupportsRandomUnnormalizedLogProb,
):
    """Random measure realised by uniform minibatching.

    A draw from this measure is a *fixed-minibatch target* — an
    unnormalized stochastic surrogate of the full-data unnormalized
    log-posterior, rescaled by ``N / b`` so the gradient is an
    unbiased estimator. **Not** a posterior in the strict (normalized)
    sense.

    For a model with prior :math:`p(\\theta)` and likelihood
    :math:`p(\\mathcal{D} \\mid \\theta) = \\prod_i p(d_i \\mid \\theta)`,
    a draw's unnormalized log-density is

    .. math::

        \\log \\tilde{D}_B(\\theta) = \\log p(\\theta)
                                      + \\frac{N}{b}
                                        \\sum_{d \\in B}
                                          \\log p(d \\mid \\theta).

    Parameters
    ----------
    label : str
        Distribution label.
    prior : SupportsLogProb
        Prior distribution over parameters; provides the log-prior
        term :math:`\\log p(\\theta)`.
    likelihood : ConditionalDistribution
        The kernel of the observations given the prior's fields, whose
        observations are conditionally independent along the leading axis of
        its event and which scores a subset of them through
        ``_observation_log_prob(given, value, rows)``, as the kernel
        :func:`~probpipe.families.glm_likelihood` returns does.
    data : array-like
        The observed value of the likelihood's event, with the observations
        along its leading axis, of length ``>= batch_size``.
    batch_size : int
        Minibatch size :math:`b`. Must be ``1 <= b <= len(data)``.
    with_replacement : bool, default False
        Sample minibatch indices with replacement. Default is
        without-replacement (uniform permutation, take first ``b``).

    Raises
    ------
    TypeError
        If ``prior`` is not :class:`~probpipe.SupportsLogProb`, or
        ``likelihood`` is not a kernel that scores a subset of its
        observations.
    ValueError
        If ``data`` has no leading axis, or ``batch_size`` is not in
        ``[1, len(data)]``.
    """

    def __init__(
        self,
        label: str,
        prior: SupportsLogProb,
        likelihood: ConditionalDistribution,
        data: ArrayLike,
        batch_size: int,
        *,
        with_replacement: bool = False,
    ):
        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"MinibatchedDistribution requires prior to satisfy "
                f"SupportsLogProb; got {type(prior).__name__}."
            )
        if not _reads_observations(likelihood):
            raise TypeError(
                f"MinibatchedDistribution requires a likelihood kernel whose observations "
                f"are conditionally independent and which scores a subset of them, such as "
                f"glm_likelihood's kernel; got {type(likelihood).__name__}."
            )

        n = _data_size(data)
        if batch_size < 1 or batch_size > n:
            raise ValueError(f"batch_size must be in [1, len(data)={n}]; got {batch_size}.")

        self._prior = prior
        self._likelihood = likelihood
        self._data = data
        self._n = n
        self._batch_size = int(batch_size)
        self._with_replacement = bool(with_replacement)
        self._rescale_factor = float(self._n / batch_size)
        # A draw is a law over the prior's parameters, declared as the prior
        # declares them.
        self._draw_event_spec = _parameter_declaration(prior, "parameters")

        super().__init__(label, DistributionSpec(self._draw_event_spec))

    # -- read-only metadata --------------------------------------------------

    @property
    def dataset_size(self) -> int:
        """Total number of observations in the dataset (``len(data)``).

        Named ``dataset_size`` rather than ``num_atoms`` (the
        finite-sample-size convention used by
        :class:`EmpiricalDistribution` and siblings) because
        :class:`MinibatchedDistribution` is not a finite-sample
        distribution; it doesn't hold a finite collection of
        realisations.
        """
        return self._n

    @property
    def batch_size(self) -> int:
        """Minibatch size :math:`b`."""
        return self._batch_size

    @property
    def with_replacement(self) -> bool:
        """Whether minibatch indices are drawn with replacement."""
        return self._with_replacement

    @property
    def prior(self) -> SupportsLogProb:
        """The prior distribution over parameters."""
        return self._prior

    @property
    def likelihood(self) -> ConditionalDistribution:
        """The kernel of the conditionally independent observations."""
        return self._likelihood

    @property
    def data(self) -> Any:
        """The full dataset (not the minibatched view)."""
        return self._data

    # -- Internal draw -------------------------------------------------------

    def _draw_one(self, key: PRNGKey) -> _FixedMinibatchDistribution:
        """Draw one minibatch and return the corresponding fixed-minibatch target."""
        rows = _draw_indices(
            key,
            self._n,
            self._batch_size,
            with_replacement=self._with_replacement,
        )
        return _FixedMinibatchDistribution(
            prior=self._prior,
            likelihood=self._likelihood,
            data=self._data,
            rows=rows,
            rescale_factor=self._rescale_factor,
            label=f"{self.label}/draw",
            event_spec=self._draw_event_spec,
        )

    # -- SupportsRandomUnnormalizedLogProb -----------------------------------

    def _random_unnormalized_log_prob(self) -> _RandomMinibatchLogProb:
        return _RandomMinibatchLogProb(self)

    # -- repr ----------------------------------------------------------------

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The prior, the likelihood, the dataset size, and the minibatch size."""
        return [
            ("prior", repr(self._prior)),
            ("likelihood", repr(self._likelihood)),
            ("dataset_size", repr(self._n)),
            ("batch_size", repr(self._batch_size)),
        ]


# ---------------------------------------------------------------------------
# _FixedMinibatchDistribution — one minibatch's stochastic-surrogate target
# ---------------------------------------------------------------------------


class _FixedMinibatchDistribution(
    Distribution,
    SupportsUnnormalizedLogProb,
):
    """One sampled inner distribution from a :class:`MinibatchedDistribution`.

    Holds a single fixed minibatch :math:`B`, the indices of its observations,
    and the rescale factor :math:`N / b`. Its unnormalized log-density at
    parameters :math:`\\theta` is

    .. math::

        \\log p(\\theta) +
        \\frac{N}{b} \\sum_{d \\in B} \\log p(d \\mid \\theta),

    which is an unbiased stochastic surrogate (in expectation over
    :math:`B`) of the full-data unnormalized log-posterior. **Not** a
    posterior in the strict (normalized) sense.

    Returned by :meth:`MinibatchedDistribution._draw_one`; users do not
    construct this class directly.
    """

    def __init__(
        self,
        prior: SupportsLogProb,
        likelihood: ConditionalDistribution,
        data: Any,
        rows: Array,
        rescale_factor: float,
        *,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if not label:
            label = "fixed_minibatch_distribution"
        if event_spec is None:
            event_spec = _parameter_declaration(prior, "parameters")
        super().__init__(label, event_spec)
        self._prior = prior
        self._likelihood = likelihood
        self._data = data
        self._rows = rows
        self._rescale_factor = rescale_factor

    @property
    def prior(self) -> SupportsLogProb:
        """The prior distribution carried from the parent measure."""
        return self._prior

    @property
    def likelihood(self) -> ConditionalDistribution:
        """The likelihood kernel carried from the parent measure."""
        return self._likelihood

    @property
    def rows(self) -> Array:
        """The indices of the observations in this realisation's minibatch."""
        return self._rows

    @property
    def rescale_factor(self) -> float:
        """Rescaling factor :math:`N / b`."""
        return self._rescale_factor

    def _unnormalized_log_prob(self, theta: Any) -> Array:
        """Stochastic-surrogate unnormalized log-density at ``theta``, a draw of the prior."""
        given = _likelihood_given(self._prior, self._likelihood, theta)
        batch = self._likelihood._observation_log_prob(given, self._data, self._rows)
        return self._prior._log_prob(theta) + self._rescale_factor * batch

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The prior, the likelihood, and the factor that rescales the minibatch to the dataset."""
        return [
            ("prior", repr(self._prior)),
            ("likelihood", repr(self._likelihood)),
            ("rescale_factor", f"{self._rescale_factor:.3g}"),
        ]


# ---------------------------------------------------------------------------
# _RandomMinibatchLogProb — a RandomFunction from parameter records to arrays
# ---------------------------------------------------------------------------


class _RandomMinibatchLogProb(
    RandomFunction,
    SupportsSampling,
):
    """The function-valued random variable :math:`\\theta \\mapsto \\log \\tilde{D}_B(\\theta)`.

    Returned by :meth:`MinibatchedDistribution._random_unnormalized_log_prob`.

    * :meth:`_sample` (``key``, ``sample_shape=()``) returns a
      *deterministic* unnormalized log-density callable for one
      minibatch draw — the primary form stochastic-gradient kernels
      consume.
    * :meth:`__call__` (``theta``) returns an array-valued distribution
      over log-density estimates at a fixed :math:`\\theta`. That
      distribution's :meth:`_sample` draws minibatched
      log-density values, so its Monte-Carlo mean recovers
      :math:`\\log p_\\text{full}(\\theta)`.
    """

    def __init__(self, measure: MinibatchedDistribution):
        super().__init__(
            f"{measure.label}/random_log_prob", OutputSpec(random_log_prob=FunctionSpec())
        )
        self._measure = measure

    # -- RandomFunction.__call__ --------------------------------------------

    def __call__(self, theta: Any) -> _MinibatchLogProbAtPoint:
        """Distribution over log-density values at a fixed ``theta``."""
        return _MinibatchLogProbAtPoint(self._measure, theta)

    # -- SupportsSampling (returns a callable) ------------------------------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Callable[[Any], Array]:
        """Return a deterministic ``theta -> log~D_B(theta)`` callable.

        Non-empty ``sample_shape`` is not supported — drawing a batch
        of log-density callables would require returning a structure
        of functions, which is awkward to type. Users who need
        multiple draws should call repeatedly with split keys.
        """
        if sample_shape != ():
            raise NotImplementedError(
                "Batched _sample of _RandomMinibatchLogProb (sample_shape != ()) "
                "is not supported. Call with split keys instead."
            )
        inner = self._measure._draw_one(key)
        # Return the bound method as a deterministic callable.
        return inner._unnormalized_log_prob

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The minibatched law whose log-density this random function draws."""
        return [("measure", repr(self._measure))]


# ---------------------------------------------------------------------------
# _MinibatchLogProbAtPoint — the distribution over log-density values at a fixed theta
# ---------------------------------------------------------------------------


class _MinibatchLogProbAtPoint(Distribution, SupportsSampling):
    """Distribution over minibatched log-density values at a fixed ``theta``.

    Returned by ``_RandomMinibatchLogProb(theta)`` — the two-argument
    form of :func:`~probpipe.random_unnormalized_log_prob`.
    Sampling draws minibatch indices, computes the rescaled per-datum
    sum, and returns the scalar log-density value.

    Monte Carlo mean over enough draws recovers
    :math:`\\log p_\\text{full}(\\theta)` (the full-data unnormalized
    log-posterior at ``theta``) — i.e. this is the unbiased
    log-density estimator the random-measure machinery promises.
    """

    def __init__(self, measure: MinibatchedDistribution, theta: Any):
        # A draw is one scalar log-density value.
        super().__init__(f"{measure.label}@theta", OutputSpec(log_prob=NumericArraySpec(())))
        self._measure = measure
        self._theta = theta

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw minibatched log-density values at the fixed ``theta``."""

        def _one_draw(k: PRNGKey) -> Array:
            inner = self._measure._draw_one(k)
            return inner._unnormalized_log_prob(self._theta)

        if sample_shape == ():
            return _one_draw(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        vals = jax.vmap(_one_draw)(keys)
        return vals.reshape(sample_shape)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The minibatched law whose log-density this law draws at a fixed point."""
        return [("measure", repr(self._measure))]
