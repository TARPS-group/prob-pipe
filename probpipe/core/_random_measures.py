"""
Random measure abstractions for ProbPipe.

Provides:
  - ``RandomMeasure[T]``         – Distribution over distributions on ``T``.
  - ``NumericRandomMeasure``     – Specialization where ``T = Array``; adds
                                   ``inner_support`` and ``inner_event_shape``.

A *random measure* is a distribution-valued random variable. Formally,
for a sample space ``T``, ``M`` is a probability distribution over the
space of probability distributions on ``T``; a draw ``D ~ M`` is itself
a ``Distribution[T]``.

Layering: shape and support semantics
-------------------------------------

A regular ``Distribution[T]`` may carry ``support`` / ``event_shape``
(these enter the hierarchy at
:class:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution`,
not on the abstract base).  A ``RandomMeasure[T]`` describes
*two* layers simultaneously:

* **Outer layer** — the random measure itself. A draw is a
  ``Distribution[T]``. Outer ``support`` / ``event_shape`` would
  carry no useful tensor content for a distribution-valued draw,
  so ``RandomMeasure`` exposes only the inner-layer metadata —
  matching the abstract ``Distribution`` base.

* **Inner layer** — properties of the inner ``Distribution[T]`` draws.
  ``inner_support`` (the support of every inner ``D``'s samples) and
  ``inner_event_shape`` (the inner ``D``'s ``event_shape`` when ``T`` is
  array-like) live on :class:`NumericRandomMeasure`, mirroring how
  ``support`` and ``event_shape`` enter the regular distribution
  hierarchy at :class:`NumericRecordDistribution` rather than the base.

Batches of random measures use the same machinery as batches of any
other ``Distribution`` — wrap them in a
:class:`~probpipe.core._distribution_array.DistributionArray`. There is
no separate ``batch_shape`` on ``RandomMeasure`` itself.

Protocol opt-in
---------------

The base class declares no required methods (matching the
:class:`~probpipe.core._random_functions.RandomFunction` precedent —
sampling a function or a distribution can be intractable).  Concrete
subclasses opt into capabilities via the protocols in
:mod:`probpipe.core.protocols`:

* :class:`~probpipe.core.protocols.SupportsSampling` — implement
  ``_sample(key, sample_shape)`` returning a ``Distribution[T]`` for
  ``sample_shape == ()`` and a ``DistributionArray`` of shape
  ``sample_shape`` otherwise.
* :class:`~probpipe.core.protocols.SupportsMean` — implement ``_mean()``
  returning the marginalised ``Distribution[T]`` ``D̄(A) = ∫ D(A) dM(D)``.
  This is the natural sample-type-polymorphic specialisation of
  ``mean``: a draw from a ``RandomMeasure[T]`` is itself a
  ``Distribution[T]``, so its expected value is a ``Distribution[T]``.
  Array-path ``_mean`` implementations elsewhere in the hierarchy are
  unaffected.
* :class:`~probpipe.core.protocols.SupportsRandomLogProb` /
  :class:`~probpipe.core.protocols.SupportsRandomUnnormalizedLogProb` —
  implement ``_random_log_prob`` / ``_random_unnormalized_log_prob``
  returning a :class:`~probpipe.core._random_functions.RandomFunction`.
  The matching ops accept an optional ``value`` argument that, when
  supplied, calls the returned random function and yields a
  ``Distribution[Array]`` directly (mirroring ``log_prob(dist, value)``);
  subclasses still implement only the zero-argument method.

Forward compatibility
---------------------

A finite-support ``RandomMeasure[T]`` (a discrete distribution over a
finite set of inner distributions) is exactly a mixture distribution;
its expected distribution is the same shape as a
:class:`~probpipe.core._broadcast_distributions._MixtureMarginal`. A
future ``MixtureDistribution[T]`` may inherit from ``RandomMeasure[T]``
or be a closely related construct. Likewise ``condition_on(rm,
observed) -> RandomMeasure`` should compose via the existing
``condition_on`` op machinery; not implemented here.
"""

from __future__ import annotations

from abc import abstractmethod

from ..custom_types import Array
from ._distribution_base import Distribution
from .constraints import Constraint
from .tracked import auto_name

# ---------------------------------------------------------------------------
# RandomMeasure[T]
# ---------------------------------------------------------------------------


class RandomMeasure[T](Distribution[Distribution[T]]):
    """A distribution over probability distributions on ``T``.

    A draw ``D ~ M`` is itself a ``Distribution[T]``.  Capabilities
    (sampling, expected distribution, random log-density) are declared
    via the protocols in :mod:`probpipe.core.protocols` — subclasses opt
    in by implementing the corresponding ``_method`` and inheriting the
    matching ``Supports*`` protocol.

    The base class does not expose outer ``support`` / ``event_shape`` /
    ``batch_shape``: those concepts apply to tensor-valued distributions
    and have no useful content for a distribution-valued one. Inner
    metadata (``inner_support``, ``inner_event_shape``) lives on
    :class:`NumericRandomMeasure` when ``T`` is array-like.

    Batches of random measures should use
    :class:`~probpipe.core._distribution_array.DistributionArray`, which
    treats ``RandomMeasure`` instances as scalar ``Distribution``
    components like any other.
    """

    def __init__(self, *, name: str | None = None, name_is_auto: bool = False):
        # Default only when no name was supplied; a subclass passing a
        # derived name with name_is_auto=True keeps its flag.
        if name is None:
            name, name_is_auto = auto_name(name, type(self).__name__)
        super().__init__(name=name, name_is_auto=name_is_auto)


# ---------------------------------------------------------------------------
# NumericRandomMeasure — RandomMeasure over array-valued inner samples
# ---------------------------------------------------------------------------


class NumericRandomMeasure(RandomMeasure[Array]):
    """A random measure whose inner draws are array-valued distributions.

    Adds two pieces of metadata that are only meaningful when the inner
    sample type is array-like:

    * ``inner_support`` — the :class:`~probpipe.core.constraints.Constraint`
      that every inner ``Distribution[Array]``'s samples satisfy.
    * ``inner_event_shape`` — the ``event_shape`` shared by the inner
      distributions; the shape of one sample drawn from any ``D ~ M``.

    These mirror :class:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution`'s
    ``support`` / ``event_shape`` for the random-measure layer.
    Subclasses must override both.
    """

    @property
    @abstractmethod
    def inner_support(self) -> Constraint:
        """Support shared by every inner ``Distribution[Array]``'s samples."""
        ...

    @property
    @abstractmethod
    def inner_event_shape(self) -> tuple[int, ...]:
        """``event_shape`` shared by the inner distributions."""
        ...
