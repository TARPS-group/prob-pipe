"""Minibatched random measure for stochastic-gradient inference.

Provides:

* :class:`MinibatchedDistribution` — a ``RandomMeasure[Record]`` whose
  draws are unbiased stochastic surrogates of the full-data
  unnormalized log-posterior. Consumed by stochastic-gradient MCMC
  kernels and (future) tempered SMC.
* :class:`_FixedMinibatchDistribution` (private) — one realisation of
  the measure, holding a single fixed minibatch.
* :class:`_RandomMinibatchLogProb` (private) — the
  ``RandomFunction[Record, Array]`` returned by
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
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..core._distribution_base import Distribution
from ..core._random_functions import RandomFunction
from ..core._random_measures import RandomMeasure
from ..core._record_array import RecordArray
from ..core.protocols import (
    SupportsLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from ..core.record import Record
from ..custom_types import Array, ArrayLike, PRNGKey

if TYPE_CHECKING:
    from ..core.protocols import ConditionallyIndependentLikelihood

__all__ = ["MinibatchedDistribution"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_size(data: Any) -> int:
    """Return the leading-axis length of *data*.

    Accepts a ``RecordArray`` (uses ``batch_shape[0]``), a flat
    ``Record`` of equal-leading-axis array leaves, an array-like with
    ``.shape``, or any object with ``__len__``. Nested Records are
    rejected — minibatching expects a flat-field layout.
    """
    if isinstance(data, RecordArray):
        return data.batch_shape[0]
    if isinstance(data, Record):
        for f in data.fields:
            leaf = data[f]
            if isinstance(leaf, Record):
                raise ValueError(
                    f"MinibatchedDistribution requires a flat Record "
                    f"(no nested fields). Got nested Record at field "
                    f"{f!r}; flatten the structure or use a "
                    f"RecordArray instead."
                )
        leading = {jnp.asarray(data[f]).shape[0] for f in data.fields}
        if len(leading) != 1:
            raise ValueError(
                f"Record leaves have differing leading-axis lengths: "
                f"{ {f: jnp.asarray(data[f]).shape[0] for f in data.fields} }. "
                f"All leaves must share a common N for minibatching."
            )
        return leading.pop()
    if hasattr(data, "shape") and len(data.shape) > 0:
        return data.shape[0]
    return len(data)


def _index_along_leading(data: Any, indices: Array) -> Any:
    """Index along the leading axis. Works for Records, arrays, RecordArrays.

    Returns a plain ``Record`` (not a ``RecordArray``) when the source
    is Record-shaped — the minibatch needs a flat dict of indexed
    leaves; per-datum vmap dispatches over the leading axis of each.
    ``RecordArray.__getitem__`` doesn't accept array indices, but since
    ``RecordArray`` subclasses ``Record``, the Record branch picks it
    up via field-name access.
    """
    if isinstance(data, Record):
        # Covers Record and RecordArray (the latter via subclass).
        return Record({f: jnp.asarray(data[f])[indices] for f in data.fields})
    return jnp.asarray(data)[indices]


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


# ---------------------------------------------------------------------------
# MinibatchedDistribution — the outer random measure
# ---------------------------------------------------------------------------


class MinibatchedDistribution(
    RandomMeasure[Record],
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
    prior : SupportsLogProb
        Prior distribution over parameters; provides the log-prior
        term :math:`\\log p(\\theta)`.
    likelihood : ConditionallyIndependentLikelihood
        Likelihood that factorises as
        :math:`\\log p(\\mathcal{D} \\mid \\theta) = \\sum_i \\log p(d_i \\mid \\theta)`;
        supplies the per-datum log-density used in the rescaled sum.
    data : array-like, Record, or RecordArray
        Observed data. Indexed along its leading axis to draw
        minibatches; must have leading-axis length ``>= batch_size``.
    batch_size : int
        Minibatch size :math:`b`. Must be ``1 <= b <= len(data)``.
    with_replacement : bool, default False
        Sample minibatch indices with replacement. Default is
        without-replacement (uniform permutation, take first ``b``).
    name : str, optional
        Distribution name.

    Raises
    ------
    TypeError
        If ``prior`` is not :class:`~probpipe.SupportsLogProb` or
        ``likelihood`` is not
        :class:`~probpipe.ConditionallyIndependentLikelihood`.
    ValueError
        If ``batch_size`` is not in ``[1, len(data)]``.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        prior: SupportsLogProb,
        likelihood: ConditionallyIndependentLikelihood,
        data: ArrayLike | Record | RecordArray,
        batch_size: int,
        *,
        with_replacement: bool = False,
        name: str | None = None,
    ):
        from ..core.protocols import ConditionallyIndependentLikelihood

        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"MinibatchedDistribution requires prior to satisfy "
                f"SupportsLogProb; got {type(prior).__name__}."
            )
        if not isinstance(likelihood, ConditionallyIndependentLikelihood):
            raise TypeError(
                f"MinibatchedDistribution requires likelihood to satisfy "
                f"ConditionallyIndependentLikelihood; got "
                f"{type(likelihood).__name__}. Implement "
                f"per_datum_log_likelihood(params, datum) on the "
                f"likelihood class."
            )

        # Validate data + batch_size.
        n = _data_size(data)
        if batch_size < 1 or batch_size > n:
            raise ValueError(f"batch_size must be in [1, len(data)={n}]; got {batch_size}.")

        self._prior = prior
        self._likelihood = likelihood
        self._data = data
        self._n = int(n)
        self._batch_size = int(batch_size)
        self._with_replacement = bool(with_replacement)
        self._rescale_factor = float(self._n / batch_size)

        if name is None:
            name = f"MinibatchedDistribution(batch_size={batch_size})"
        super().__init__(name=name)

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
    def likelihood(self) -> ConditionallyIndependentLikelihood:
        """The conditionally-independent likelihood."""
        return self._likelihood

    @property
    def data(self) -> Any:
        """The full dataset (not the minibatched view)."""
        return self._data

    # -- Internal draw -------------------------------------------------------

    def _draw_one(self, key: PRNGKey) -> _FixedMinibatchDistribution:
        """Draw one minibatch and return the corresponding fixed-minibatch target."""
        indices = _draw_indices(
            key,
            self._n,
            self._batch_size,
            with_replacement=self._with_replacement,
        )
        batch = _index_along_leading(self._data, indices)
        return _FixedMinibatchDistribution(
            prior=self._prior,
            likelihood=self._likelihood,
            batch=batch,
            rescale_factor=self._rescale_factor,
            name=f"{self.name}/draw",
        )

    # -- SupportsRandomUnnormalizedLogProb -----------------------------------

    def _random_unnormalized_log_prob(self) -> _RandomMinibatchLogProb:
        return _RandomMinibatchLogProb(self)

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MinibatchedDistribution(prior={type(self._prior).__name__}, "
            f"likelihood={type(self._likelihood).__name__}, "
            f"dataset_size={self._n}, batch_size={self._batch_size})"
        )


# ---------------------------------------------------------------------------
# _FixedMinibatchDistribution — one minibatch's stochastic-surrogate target
# ---------------------------------------------------------------------------


class _FixedMinibatchDistribution(
    Distribution[Record],
    SupportsUnnormalizedLogProb,
):
    """One sampled inner distribution from a :class:`MinibatchedDistribution`.

    Holds a single fixed minibatch :math:`B` and the rescale factor
    :math:`N / b`. Its unnormalized log-density at parameters
    :math:`\\theta` is

    .. math::

        \\log p(\\theta) +
        \\frac{N}{b} \\sum_{d \\in B} \\log p(d \\mid \\theta),

    which is an unbiased stochastic surrogate (in expectation over
    :math:`B`) of the full-data unnormalized log-posterior. **Not** a
    posterior in the strict (normalized) sense.

    Returned by :meth:`MinibatchedDistribution._sample`; users do not
    construct this class directly.
    """

    _sampling_cost: str = "free"  # log_prob is closed-form
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        prior: SupportsLogProb,
        likelihood: ConditionallyIndependentLikelihood,
        batch: Any,
        rescale_factor: float,
        *,
        name: str | None = None,
    ):
        super().__init__(
            name=name or "fixed_minibatch_distribution",
        )
        self._prior = prior
        self._likelihood = likelihood
        self._batch = batch
        self._rescale_factor = rescale_factor

    @property
    def prior(self) -> SupportsLogProb:
        """The prior distribution carried from the parent measure."""
        return self._prior

    @property
    def likelihood(self) -> ConditionallyIndependentLikelihood:
        """The CIL likelihood carried from the parent measure."""
        return self._likelihood

    @property
    def batch(self) -> Any:
        """The fixed minibatch this realisation was built from."""
        return self._batch

    @property
    def rescale_factor(self) -> float:
        """Rescaling factor :math:`N / b`."""
        return self._rescale_factor

    def _unnormalized_log_prob(self, theta: Any) -> Array:
        """Stochastic-surrogate unnormalized log-density at ``theta``."""
        per_datum = jax.vmap(
            self._likelihood.per_datum_log_likelihood,
            in_axes=(None, 0),
        )(theta, self._batch)
        return self._prior._log_prob(theta) + self._rescale_factor * jnp.sum(per_datum)

    def __repr__(self) -> str:
        return f"_FixedMinibatchDistribution(rescale_factor={self._rescale_factor:.3g})"


# ---------------------------------------------------------------------------
# _RandomMinibatchLogProb — RandomFunction[Record, Array]
# ---------------------------------------------------------------------------


class _RandomMinibatchLogProb(
    RandomFunction[Record, Array],
    SupportsSampling,
):
    """The function-valued random variable :math:`\\theta \\mapsto \\log \\tilde{D}_B(\\theta)`.

    Returned by :meth:`MinibatchedDistribution._random_unnormalized_log_prob`.

    * :meth:`_sample` (``key``, ``sample_shape=()``) returns a
      *deterministic* unnormalized log-density callable for one
      minibatch draw — the primary form stochastic-gradient kernels
      consume.
    * :meth:`__call__` (``theta``) returns a ``Distribution[Array]``
      over log-density estimates at a fixed :math:`\\theta`. The
      ``Distribution[Array]``'s :meth:`_sample` draws minibatched
      log-density values, so its Monte-Carlo mean recovers
      :math:`\\log p_\\text{full}(\\theta)`.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(self, measure: MinibatchedDistribution):
        super().__init__(name=f"{measure.name}/random_log_prob")
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

    def __repr__(self) -> str:
        return f"_RandomMinibatchLogProb(measure={self._measure.name})"


# ---------------------------------------------------------------------------
# _MinibatchLogProbAtPoint — Distribution[Array] over log-density at fixed theta
# ---------------------------------------------------------------------------


class _MinibatchLogProbAtPoint(Distribution[Array], SupportsSampling):
    """Distribution over minibatched log-density values at a fixed ``theta``.

    Returned by ``_RandomMinibatchLogProb(theta)`` — the two-argument
    form of :func:`~probpipe.core.ops.random_unnormalized_log_prob`.
    Sampling draws minibatch indices, computes the rescaled per-datum
    sum, and returns the scalar log-density value.

    Monte Carlo mean over enough draws recovers
    :math:`\\log p_\\text{full}(\\theta)` (the full-data unnormalized
    log-posterior at ``theta``) — i.e. this is the unbiased
    log-density estimator the random-measure machinery promises.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(self, measure: MinibatchedDistribution, theta: Any):
        super().__init__(name=f"{measure.name}@theta")
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

    def __repr__(self) -> str:
        return f"_MinibatchLogProbAtPoint(measure={self._measure.name})"
