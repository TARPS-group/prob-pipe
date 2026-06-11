"""Protocol definitions for distribution capabilities.

Each protocol declares a capability that a distribution may support.
Operations in :mod:`probpipe.core.ops` check these protocols via
``isinstance`` to determine what computations are valid.

All protocols are ``@runtime_checkable`` so that external distribution
types (TFP, scipy) can satisfy them via structural subtyping without
inheriting from ProbPipe base classes.

**Naming convention:** Protocol methods use an underscore prefix
(``_sample``, ``_log_prob``, ``_mean``, …) to distinguish the
primitive implementation from the public workflow-function API in
:mod:`probpipe.core.ops`.

**Orchestration hints:** ``SupportsSampling`` defines class-attribute
defaults for orchestration preferences.  Distribution subclasses
override as needed.

Protocol hierarchy
------------------

::

    SupportsSampling          standalone; single _sample protocol method

    SupportsExpectation       standalone; E[f(X)] computation

    SupportsMean              standalone; exact _mean()
    SupportsVariance          standalone; exact _variance()
    SupportsCovariance        standalone; exact _cov()

    SupportsUnnormalizedLogProb
        ↑ inherits
    SupportsLogProb           provides _unnormalized_log_prob via _log_prob

    SupportsArrayBackend      classmethod-level capability; declares that a
                              ``Distribution`` subclass can produce a fused
                              storage backend for ``DistributionArray``

The moment protocols (SupportsMean, SupportsVariance, SupportsCovariance)
are independent of SupportsExpectation.  The ops layer falls back to
MC estimation via SupportsExpectation when the exact protocol is absent.
Concrete classes that want default MC implementations can use the
``@compute_expectation`` decorator on their ``_mean``/``_variance``/
``_cov`` methods — but this is opt-in, not required by the protocol.

"""

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Protocol,
    runtime_checkable,
)

import jax.numpy as jnp

from ..custom_types import Array, PRNGKey

if TYPE_CHECKING:
    from ._distribution_base import Distribution


# ---------------------------------------------------------------------------
# Decorator for default moment implementations via expectation
# ---------------------------------------------------------------------------

def compute_expectation(method):
    """Decorator providing a default moment implementation via ``expectation``.

    The decorated method should return the function ``f`` to pass to
    ``self.expectation(f, return_dist=False)``.  Any setup (e.g.
    computing the mean before computing the variance) can be done in
    the method body before the ``return``.

    Example::

        @compute_expectation
        def _mean(self):
            return lambda x: x
    """

    @functools.wraps(method)
    def wrapper(self):
        f = method(self)
        return self._expectation(f, return_dist=False)

    return wrapper


# ---------------------------------------------------------------------------
# Expectation & sampling
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsExpectation(Protocol):
    """Distribution that can compute ``E[f(X)]``."""

    def _expectation(self, f: Any, *, key: Any, num_evaluations: Any,
                     return_dist: Any) -> Any: ...


@runtime_checkable
class SupportsSampling(Protocol):
    """Distribution that can produce samples via ``_sample(key, sample_shape)``.

    Only requires ``_sample(key, sample_shape)``; concrete classes choose
    their own implementation strategy (TFP batched sampling, index
    resampling, vmap over a local single-draw helper, etc.).

    Does NOT extend :class:`SupportsExpectation` — not all samplable
    distributions support array-valued expectations (e.g., random functions).
    Classes that support both should inherit both protocols.

    Return-type convention
    ----------------------
    The shape of the return value depends on whether the distribution
    emits structured samples and whether the caller asks for a batch:

    =====================  =======================  =========================================
    Distribution kind      ``sample_shape == ()``   ``sample_shape == (S1, S2, ...)``
    =====================  =======================  =========================================
    Numeric (raw array)    ``Array[*event_shape]``  ``Array[*sample_shape, *event_shape]``
    ``RecordDistribution`` ``Record`` / ``NumericRecord``  ``NumericRecordArray(batch_shape=sample_shape)``
    =====================  =======================  =========================================

    To draw a single sample, call ``_sample(key, ())``. Implementations
    that find it clearer to factor out a single-draw helper should
    define it as a private method (e.g. ``_one_bootstrap``) and have
    ``_sample`` dispatch on ``sample_shape`` internally.
    """

    _sampling_cost: ClassVar[str]  # "low", "medium", "high"
    _preferred_orchestration: ClassVar[str | None]  # "task", "flow", or None

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        """Draw sample(s) from this distribution.

        Parameters
        ----------
        key : PRNGKey
            JAX PRNG key.
        sample_shape : tuple of int
            Shape prefix for independent draws.

        Returns
        -------
        Any
            See class-level "Return-type convention".
        """
        ...


# ---------------------------------------------------------------------------
# Density evaluation
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsUnnormalizedLogProb(Protocol):
    """Distribution with an unnormalized log-density.

    Provides ``_unnormalized_log_prob(value)``.
    """

    def _unnormalized_log_prob(self, value: Any) -> Array: ...


@runtime_checkable
class SupportsLogProb(SupportsUnnormalizedLogProb, Protocol):
    """Distribution with a (normalized) log-density.

    Extends :class:`SupportsUnnormalizedLogProb` because any distribution
    with a normalized density also has an unnormalized one (they coincide).
    The base :class:`~probpipe.core.distribution.Distribution` class
    provides ``_unnormalized_log_prob`` defaulting to ``_log_prob``.

    """

    def _log_prob(self, value: Any) -> Array: ...

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """Default: delegates to ``_log_prob``."""
        return self._log_prob(value)


# ---------------------------------------------------------------------------
# Moment protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsMean(Protocol):
    """Distribution with an exact mean via ``_mean()``.

    The return type is ``T``-shaped where ``T`` is the distribution's
    sample type. For the common cases this is:

    * :class:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution`
      and friends (``T = Array``) — returns :class:`~probpipe.custom_types.Array`.
    * :class:`~probpipe.core._record_distribution.RecordDistribution` and
      friends (``T = Record``) — returns :class:`~probpipe.record.Record`.
    * :class:`~probpipe.core._random_measures.RandomMeasure[T]`
      (``T = Distribution[T]``) — returns the marginalised
      ``Distribution[T]`` with marginal ``D̄(A) = ∫ D(A) dM(D)``.

    The protocol is sample-type-polymorphic by design: the array-valued
    and structured paths are unchanged; ``RandomMeasure`` opts in by
    implementing ``_mean`` to return its expected distribution.

    Independent of :class:`SupportsExpectation`.  The ops layer falls
    back to MC estimation via ``SupportsExpectation`` when this protocol
    is absent.  Concrete classes that want the MC default can apply
    ``@compute_expectation`` to their ``_mean`` implementation (only
    valid when ``T`` is array-like).
    """

    def _mean(self) -> Any: ...


@runtime_checkable
class SupportsVariance(Protocol):
    """Distribution with an exact variance via ``_variance()``.

    Independent of :class:`SupportsExpectation`.  The ops layer falls
    back to MC estimation via ``SupportsExpectation`` when this protocol
    is absent.  Concrete classes that want the MC default can apply
    ``@compute_expectation`` to their ``_variance`` implementation.
    """

    def _variance(self) -> Any: ...


@runtime_checkable
class SupportsCovariance(Protocol):
    """Distribution with an exact covariance via ``_cov()``.

    Independent of :class:`SupportsExpectation`.  The ops layer falls
    back to MC estimation via ``SupportsExpectation`` when this protocol
    is absent.  Concrete classes that want the MC default can apply
    ``@compute_expectation`` to their ``_cov`` implementation.
    """

    def _cov(self) -> Any: ...


# ---------------------------------------------------------------------------
# Random-measure protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsRandomLogProb(Protocol):
    """Distribution over distributions with a random (normalized) log-density.

    For a ``RandomMeasure[T]`` ``M``, ``_random_log_prob`` returns the
    random function ``x ↦ log D(x)`` where ``D ~ M`` as a
    :class:`~probpipe.core._random_functions.RandomFunction`. The op
    layer (:func:`~probpipe.core.ops.random_log_prob`) optionally
    forwards an input *value* by calling the returned random function;
    that two-argument convenience is purely op-layer sugar — concrete
    subclasses implement only the zero-argument method here.

    Mirrors :class:`SupportsLogProb` for the random-measure setting.
    """

    def _random_log_prob(self) -> Any: ...


@runtime_checkable
class SupportsRandomUnnormalizedLogProb(Protocol):
    """Distribution over distributions with a random unnormalized log-density.

    Mirrors :class:`SupportsUnnormalizedLogProb` for the random-measure
    setting. ``_random_unnormalized_log_prob`` returns the random
    function ``x ↦ log D̃(x)`` where ``D̃`` is the unnormalized density
    of a draw ``D ~ M``, as a
    :class:`~probpipe.core._random_functions.RandomFunction`. The op
    layer (:func:`~probpipe.core.ops.random_unnormalized_log_prob`)
    optionally forwards an input *value* by calling the returned random
    function; that two-argument convenience is purely op-layer sugar —
    concrete subclasses implement only the zero-argument method here.
    """

    def _random_unnormalized_log_prob(self) -> Any: ...


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsConditioning(Protocol):
    """Distribution that has a fast, built-in ``condition_on`` path.

    Implemented by distributions whose ``_condition_on`` produces a
    posterior without calling into the inference registry — either
    closed-form (conjugate updates, joint Gaussian marginalization)
    or amortized (e.g., a pre-trained SBI posterior that just runs a
    forward pass).  When ``condition_on(dist, observed)`` is called
    and *dist* implements this protocol, the built-in path is used
    directly; otherwise the inference method registry selects an
    algorithm (NUTS, RWMH, variational, ...).

    Probabilistic models whose conditioning requires on-the-fly MCMC
    or variational inference should **not** implement this protocol —
    let the registry handle algorithm selection instead.
    """

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Array backend (fused storage for DistributionArray)
# ---------------------------------------------------------------------------


@runtime_checkable
class _DistributionArrayBackend(Protocol):
    """Internal storage backend that ``DistributionArray`` consumes.

    A backend owns the *batched* parameters of a homogeneous
    ``DistributionArray`` and delivers vectorised ops directly — TFP's
    native batch axis, a single ``RecordEmpiricalDistribution`` with a
    leading batch dim, etc. It carries no ``name`` / ``provenance``
    and lives only as the contract between a distribution class's
    :meth:`SupportsArrayBackend._make_array_backend` and the array
    consumer.

    Backends are private to the library. User code never imports or
    constructs them; they exist solely so a
    :class:`~probpipe.DistributionArray` can fuse storage instead of
    materialising one ``Distribution`` per cell.

    Required surface
    ----------------
    Every backend exposes ``batch_shape``, ``event_shape``, ``cell``,
    and the ``_sample``/``_log_prob``/``_mean``/``_variance``/``_cov``
    methods that mirror whichever moment / density protocols the
    underlying distribution class supports. ``DistributionArray``
    introspects via ``isinstance`` and forwards to whichever ones are
    present.

    ``cell(index)`` materialises a fresh **scalar** ``Distribution``
    (i.e. ``batch_shape == ()``) for the cell at ``index``. Used by
    ``DistributionArray.__getitem__`` and by the WF sweep when
    cell-level dispatch is needed.
    """

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    def cell(self, index: int | tuple[int, ...]) -> "Distribution":
        """Fabricate a scalar ``Distribution`` for the cell at ``index``."""
        ...


@runtime_checkable
class SupportsArrayBackend(Protocol):
    """Distribution class that supports efficient batched construction.

    Used by :meth:`DistributionArray.from_batched_params` to fuse
    storage when the caller's components are homogeneous instances of
    the same class. Implementations construct an internal
    :class:`_DistributionArrayBackend` that owns the batched parameters
    and the vectorised ops; ``DistributionArray`` becomes a thin
    consumer.

    Distribution classes that don't implement this protocol still work
    in a ``DistributionArray`` via the literal-array fallback (one
    ``Distribution`` instance per cell) — slower but correct.

    The protocol attaches to the **class**, not to instances. The
    runtime check is ``isinstance(MyDistribution, SupportsArrayBackend)``
    (i.e. the class itself implements ``_make_array_backend``).
    ``isinstance(an_instance, SupportsArrayBackend)`` returns
    ``True`` too — instances inherit class attributes, and
    ``runtime_checkable`` just looks for the named attribute — but
    the result is misleading because the contract is at class
    scope.

    The protocol is internal to the library; user code never calls
    ``_make_array_backend`` directly. ``DistributionArray`` is the
    sole consumer.

    Examples
    --------
    A distribution class declares the capability by implementing the
    classmethod::

        class MyDistribution(Distribution[T]):
            @classmethod
            def _make_array_backend(
                cls,
                *,
                name: str,
                batch_shape: tuple[int, ...],
                **batched_params,
            ) -> _DistributionArrayBackend:
                return _MyArrayBackend(
                    cls=cls, name=name, batch_shape=batch_shape,
                    **batched_params,
                )
    """

    @classmethod
    def _make_array_backend(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...],
        **batched_params: Any,
    ) -> _DistributionArrayBackend:
        """Construct an array backend for this class.

        Parameters
        ----------
        name : str
            Base name; per-cell distributions auto-suffix as
            ``f"{name}_{i}"``.
        batch_shape : tuple of int
            The leading shape of the batched parameters. The backend
            stores parameters with this shape prepended to each
            ``cls(**kwargs)``-style argument.
        **batched_params
            Same keys as ``cls(**kwargs)`` would take, but with
            ``batch_shape`` prepended to each array argument.

        Returns
        -------
        _DistributionArrayBackend
            Backend instance owning the batched parameters and
            delivering vectorised ops.
        """
        ...


def protocols_supported_by_all(
    leaves: list, candidates: tuple[type, ...],
) -> tuple[type, ...]:
    """Return the subset of *candidates* that every leaf satisfies.

    Used by dynamic-protocol factories (``ProductDistribution``,
    ``SequentialJointDistribution``, ``TransformedDistribution``,
    ``_RecordDistributionView``, ``FlattenedDistributionView``) when building a
    cached subclass whose protocol bases track the capabilities of the
    underlying distribution(s). Pass in the leaves to check and the
    tuple of ``SupportsFoo`` protocols to test against; get back the
    protocols that are satisfied by every leaf, in the given order.
    """
    return tuple(p for p in candidates if all(isinstance(l, p) for l in leaves))


# ---------------------------------------------------------------------------
# Likelihoods and generative simulators
# ---------------------------------------------------------------------------


@runtime_checkable
class Likelihood[P, D](Protocol):
    """Protocol for computing log-likelihood of data given parameters.

    Generic in ``P`` (parameter type) and ``D`` (data type).
    Any class that defines ``log_likelihood(params, data) -> float``
    satisfies this protocol.
    """

    def log_likelihood(self, params: P, data: D) -> float: ...


@runtime_checkable
class ConditionallyIndependentLikelihood[P, D](Likelihood[P, D], Protocol):
    """Likelihood whose observations are conditionally independent given
    the parameters.

    Formally, for observations ``y_1, ..., y_N`` the joint log-density
    factorises into a sum of per-observation log-densities:

    .. math::

        \\log p(y_1, \\ldots, y_N \\mid \\theta)
            = \\sum_{i=1}^N \\log p(y_i \\mid \\theta).

    The "conditionally" refers to conditioning on the parameters ``θ``:
    the ``y_i`` are independent *given* ``θ``, not marginally. For
    regression-style likelihoods each datum carries a covariate ``x_i``
    that the per-observation density depends on; the factorisation then
    reads ``Σ_i log p(y_i | x_i, θ)``, with the covariates treated as
    fixed inputs. This is the "conditionally independent" case rather
    than the stricter "i.i.d." (where every ``p(y_i | θ)`` is identical).

    Required by :class:`~probpipe.MinibatchedDistribution` for
    stochastic-gradient inference, and useful independently for held-out
    predictive log-likelihoods, leave-one-out cross-validation, and
    PSIS-LOO. Implementations expose :meth:`per_datum_log_likelihood`;
    :func:`_default_per_datum_log_likelihood` is a length-1-batch fallback
    for likelihoods that prefer a default over an efficient override.
    """

    def per_datum_log_likelihood(self, params: P, datum: Any) -> Any:
        """Log-density of a single datum given parameters.

        Parameters
        ----------
        params : P
            Model parameters.
        datum : Any
            One observation; its shape depends on the data format the
            likelihood was built against (a row ``(x_i, y_i)`` for a
            regression model, a single value for a scalar response).

        Returns
        -------
        Array
            Scalar log-density of the datum under ``params``.
        """
        ...


def _default_per_datum_log_likelihood(
    likelihood: Likelihood,
    params: Any,
    datum: Any,
) -> Any:
    """Default per-datum log-likelihood — evaluate ``log_likelihood`` on a length-1 batch.

    Fallback for :class:`ConditionallyIndependentLikelihood`
    implementations that don't have a row-specific shortcut. Adds a
    leading axis to ``datum`` via ``jax.tree.map(lambda x: x[None, ...], datum)``
    and calls ``likelihood.log_likelihood(params, batch)``. Less
    efficient than an override that evaluates the family directly on
    the un-reshaped datum (no length-1-batch wrap, no associated
    broadcasting overhead inside ``log_likelihood``).
    """
    import jax
    batch = jax.tree.map(lambda x: x[None, ...], datum)
    return likelihood.log_likelihood(params, batch)


@runtime_checkable
class GenerativeLikelihood[P, D](Protocol):
    """Protocol for generating synthetic data given parameters.

    Generic in ``P`` (parameter type) and ``D`` (data type).
    Any class that defines
    ``generate_data(params, num_observations, *, key) -> D``
    satisfies this protocol.
    """

    def generate_data(
        self, params: P, num_observations: int,
        *, key: PRNGKey | None = None,
    ) -> D:
        """Generate ``num_observations`` synthetic data points from ``params``.

        Parameters
        ----------
        params : P
            Model parameters.
        num_observations : int
            Number of data points to generate.
        key : PRNGKey or None
            JAX PRNG key for reproducible generation.
        """
        ...


__all__ = [
    "compute_expectation",
    "SupportsExpectation",
    "SupportsSampling",
    "SupportsUnnormalizedLogProb",
    "SupportsLogProb",
    "SupportsMean",
    "SupportsVariance",
    "SupportsCovariance",
    "SupportsRandomLogProb",
    "SupportsRandomUnnormalizedLogProb",
    "SupportsConditioning",
    "SupportsArrayBackend",
    "Likelihood",
    "ConditionallyIndependentLikelihood",
    "GenerativeLikelihood",
    "protocols_supported_by_all",
]
