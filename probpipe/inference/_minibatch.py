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
from functools import partial
from math import ceil, log1p, prod
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ..core._expression import Applied, constant
from ..core._repr import type_name
from ..core._specs import NumericArraySpec, OpaqueSpec, OutputSpec
from ..core.tracked import TrackedTerm
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import _outputs_of
from ..families._random_functions import RandomFunction, RandomMeasure
from ..operations._inference_target import _described
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
        raise ValueError("the observed data must have a leading axis of observations; got a scalar")
    return int(shape[0])


@partial(jax.jit, static_argnames=("n", "batch_size", "with_replacement"))
def _draw_indices(
    key: PRNGKey,
    n: int,
    batch_size: int,
    *,
    with_replacement: bool,
) -> Array:
    """Draw ordered indices uniformly from ``range(n)``.

    Parameters
    ----------
    key : PRNGKey
        The random key. Reusing it with the same arguments reproduces the draw.
    n : int
        Population size, positive and static for JAX.
    batch_size : int
        Number of indices, between one and *n*, static for JAX.
    with_replacement : bool
        Whether indices may repeat, static for JAX. Otherwise the indices
        are distinct, with a uniformly random order.

    Returns
    -------
    Array
        Integer indices of shape ``(batch_size,)`` in ``[0, n)``.

    Notes
    -----
    Sampling without replacement retains the first occurrences from batches
    of independent candidates. If a batch supplies too few distinct indices,
    further candidates supplement the retained indices. Sorting identifies
    duplicates while preserving the original order of first occurrences.
    Candidate arrays have size proportional to the smaller of ``batch_size``
    and ``n - batch_size``. Their sizes depend only on these static dimensions.
    Batches larger than half the population sample the excluded indices, then
    shuffle their complement. The complement mask, output and permutation
    still require O(n) space. A full batch is a population permutation.
    """
    if with_replacement:
        return jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n)
    if batch_size == n:
        return jax.random.permutation(key, n)
    if batch_size <= n // 2:
        return _sample_unique_indices(key, n, batch_size)
    subset_key, order_key = jax.random.split(key)
    excluded = _sample_unique_indices(subset_key, n, n - batch_size)
    included = jnp.ones((n,), dtype=bool).at[excluded].set(False)
    indices = jnp.nonzero(included, size=batch_size)[0]
    return jax.random.permutation(order_key, indices)


def _unique_prefix(candidates: Array, size: int, n: int) -> Array:
    """The first *size* distinct candidates, padded with the sentinel *n*.

    Candidates form a one-dimensional array of at least *size* integers in
    ``[0, n]``. The sentinel is ignored when identifying first occurrences.
    The returned array has shape ``(size,)``
    and retains the candidates' original order.
    """
    count = candidates.shape[0]
    values, positions = jax.lax.sort_key_val(candidates, jnp.arange(count), is_stable=True)
    first = jnp.concatenate((jnp.ones((1,), dtype=bool), values[1:] != values[:-1]))
    positions = jnp.sort(jnp.where(first & (values < n), positions, count))[:size]
    padded = jnp.concatenate((candidates, jnp.full((1,), n, dtype=candidates.dtype)))
    return padded[positions]


def _candidate_count(n: int, size: int) -> int:
    """Candidate allocation for static dimensions ``1 <= size <= n // 2``."""
    # N log(N / (N - size)) estimates the draws needed to collect size values.
    # The surplus limits retries even near half the population; its maximum
    # remains proportional to size. Retaining first occurrences makes every
    # ordered subset equally likely, regardless of this allocation choice.
    return max(size + 16, ceil(-1.25 * n * log1p(-size / n)))


def _sample_unique_indices(key: PRNGKey, n: int, size: int) -> Array:
    """Draw *size* distinct indices in uniform random order using bounded arrays.

    The static dimensions satisfy ``1 <= size <= n // 2``. The returned
    integer array has shape ``(size,)`` with values in ``[0, n)``.
    Reusing a key and dimensions reproduces the draw on the same backend.
    """
    if size == 1:
        return jax.random.randint(key, (1,), 0, n)
    count = _candidate_count(n, size)
    key, subkey = jax.random.split(key)
    selected = _unique_prefix(jax.random.randint(subkey, (count,), 0, n), size, n)

    def supplement(state: tuple[PRNGKey, Array]) -> tuple[PRNGKey, Array]:
        key, selected = state
        key, subkey = jax.random.split(key)
        candidates = jnp.concatenate((selected, jax.random.randint(subkey, (count,), 0, n)))
        return key, _unique_prefix(candidates, size, n)

    return jax.lax.while_loop(lambda state: state[1][-1] == n, supplement, (key, selected))[1]


def _parameter_declaration(prior: Any, output_name: str) -> OutputSpec:
    """The event spec of the parameters *prior* is a law over.

    A prior that is not a distribution declares nothing, so its parameters are
    opaque under *output_name*.
    """
    try:
        return prior.event_spec
    except AttributeError:
        return OutputSpec(**{output_name: OpaqueSpec()})


def _reads_observations(likelihood: Any) -> bool:
    """Whether *likelihood* is a kernel that scores a subset of its observations."""
    return isinstance(likelihood, ConditionalDistribution) and callable(
        getattr(likelihood, "_observation_log_prob", None)
    )


def _subset_scoring_reason(likelihood: Any) -> str:
    """Why *likelihood*, which :func:`_reads_observations` rejects, cannot be minibatched."""
    return (
        "the likelihood must be a ConditionalDistribution that can score a subset of its "
        f"observations, such as one from glm_likelihood; got {_described(likelihood)}"
    )


def _likelihood_given(prior: Any, likelihood: ConditionalDistribution, theta: Any) -> dict:
    """The likelihood's given values at *theta*, a draw of *prior*."""
    outputs = _outputs_of(prior.event_spec, theta)
    return {input_name: outputs[input_name] for input_name in likelihood.given_spec}


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
    label : str, optional
        The measure's label. By default the label is ``minibatch``, and the
        notation shows the construction over the prior and the likelihood, as
        ``minibatch(prior(beta), lik(y | beta), batch_size=40)``. A prior that
        is not a ``Distribution`` has no label to show, so it requires one.
    with_replacement : bool, default False
        Sample minibatch indices with replacement. By default each draw
        selects a uniform size-``b`` subset in random order. Different draws
        may reuse observations.

    Raises
    ------
    TypeError
        If ``prior`` is not :class:`~probpipe.SupportsLogProb`, or
        ``likelihood`` is not a kernel that scores a subset of its
        observations, or ``label`` is ``None`` for a prior that is not a
        ``Distribution``.
    ValueError
        If ``data`` has no leading axis, or ``batch_size`` is not in
        ``[1, len(data)]``.
    """

    #: The constructor takes no output name, so the repr shows none.
    _repr_output_name: ClassVar[str | None] = None

    def __init__(
        self,
        prior: SupportsLogProb,
        likelihood: ConditionalDistribution,
        data: ArrayLike,
        batch_size: int,
        *,
        label: str | None = None,
        with_replacement: bool = False,
    ):
        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"MinibatchedDistribution: prior must have a log-density (SupportsLogProb), "
                f"got {_described(prior)}"
            )
        if not _reads_observations(likelihood):
            raise TypeError(f"MinibatchedDistribution: {_subset_scoring_reason(likelihood)}")

        n = _data_size(data)
        if batch_size < 1 or batch_size > n:
            raise ValueError(f"batch_size must be in [1, len(data)={n}]; got {batch_size}")

        expression = None
        if label is None:
            if not isinstance(prior, TrackedTerm):
                raise TypeError(
                    f"MinibatchedDistribution cannot take its label from a prior of type "
                    f"{type_name(prior)}, which has no label; pass label=..."
                )
            expression = Applied(
                "minibatch",
                (prior._embedded_expression(), likelihood._embedded_expression()),
                (("batch_size", constant(int(batch_size))),),
            )

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

        # A draw is one fixed-minibatch target, the measure's one output.
        super().__init__("target", DistributionSpec(self._draw_event_spec), label=label)
        if expression is not None:
            self._store_expression(expression)

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

    def _derived_label(self) -> str | None:
        """The label the constructor gives when it is given none: ``minibatch``, for a prior that has a label."""
        return "minibatch" if isinstance(self._prior, TrackedTerm) else None

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
        super().__init__(
            event_spec,
            label=label,
        )
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
            "random_log_prob", FunctionSpec(), label=f"{measure.label}/random_log_prob"
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
                f"sample_shape={sample_shape} is not supported: the random log-density of a "
                f"MinibatchedDistribution draws one function at a time. Split the key and "
                f"sample once per key."
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
        super().__init__(
            OutputSpec(log_prob=NumericArraySpec(())),
            label=f"{measure.label}@theta",
        )
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
