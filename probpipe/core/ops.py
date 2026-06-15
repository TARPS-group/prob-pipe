"""Built-in operations for distribution computation.

The value-free / dispatch ops (``sample``, ``mean``, ``variance``, ``cov``,
``expectation``, ``condition_on``, ``from_distribution``) are
:class:`~probpipe.core.node.WorkflowFunction` instances created via the
``@workflow_function`` decorator, so every call automatically participates
in broadcasting and Prefect orchestration when a distribution argument is
passed where a concrete value is expected.

The density ops (``log_prob``, ``prob``, ``unnormalized_log_prob``,
``unnormalized_prob``, and the ``random_*_log_prob`` ops) are
:class:`_DensityOp` wrappers over inner ``_<op>_impl`` WorkflowFunctions. The
wrapper adds the positional-or-keyword value form — ``log_prob(dist, value)``
or ``log_prob(dist, field=value, ...)`` (packed via
:meth:`~probpipe.core._distribution_base.Distribution._pack_value`);
broadcasting still happens because the inner impl is the WorkflowFunction.
``dist`` and ``value`` are positional-only so field names are free for the
keyword form. Per-call ProbPipe controls (``seed`` / ``n_broadcast_samples`` /
``include_inputs``) use ``with_options`` —
``log_prob.with_options(seed=0)(dist, value)`` — the same idiom as the
dispatch ops.

Usage::

    from probpipe import sample, mean, log_prob, condition_on

    dist = Normal(loc=0.0, scale=1.0, name="x")
    s = sample(dist, key=jax.random.PRNGKey(0), sample_shape=(100,))
    m = mean(dist)
    lp = log_prob(dist, jnp.array(1.5))   # positional
    lp = log_prob(dist, x=1.5)            # keyword
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .._utils import _auto_key
from ..custom_types import Array, PRNGKey
from .distribution import Distribution, RandomFunction
from .node import workflow_function
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)

__all__ = [
    "condition_on",
    "cov",
    "expectation",
    "from_distribution",
    "log_prob",
    "mean",
    "prob",
    "random_log_prob",
    "random_unnormalized_log_prob",
    "sample",
    "unnormalized_log_prob",
    "unnormalized_prob",
    "variance",
]


# ---------------------------------------------------------------------------
# Public API — value-free ops are WorkflowFunctions directly; the density
# ops (below) are wrappers over inner _<op>_impl WorkflowFunctions
# ---------------------------------------------------------------------------


@workflow_function
def sample(
    dist: SupportsSampling,
    *,
    key: PRNGKey | None = None,
    sample_shape: int | tuple[int, ...] = (),
) -> Any:
    """Draw samples from a distribution.

    Parameters
    ----------
    dist : SupportsSampling
        Distribution to sample from.
    key : PRNGKey, optional
        JAX PRNG key.  Auto-generated if ``None``.
    sample_shape : int or tuple of int
        Shape prefix for independent draws.  A scalar ``N`` is treated
        as sugar for ``(N,)``, matching the convention used by numpy,
        JAX, scipy, and TFP.
    """
    if not isinstance(dist, SupportsSampling):
        raise TypeError(
            f"{type(dist).__name__} does not support sampling "
            f"(does not implement SupportsSampling)"
        )
    if isinstance(sample_shape, int):
        sample_shape = (sample_shape,)
    if key is None:
        key = _auto_key()
    return dist._sample(key, sample_shape)


# -- density ops: positional/keyword value form + with_options controls ------
#
# Each density op (``log_prob`` and friends) is a :class:`_DensityOp` — a
# callable wrapper over an inner ``@workflow_function`` impl
# (``_<op>_impl``).  The wrapper resolves the positional-or-keyword value and
# forwards to the inner impl, so broadcasting and orchestration are unchanged.
#
# Per-call ProbPipe controls are applied with ``with_options`` — the same
# idiom as the dispatch ops (``condition_on.with_options(...)``) and the
# non-deprecated control path (STYLE_GUIDE §1.8)::
#
#     log_prob.with_options(seed=0)(dist, value)
#
# Keeping controls off the call signature leaves the keyword namespace free
# for field names: a distribution field that happens to be named ``seed`` /
# ``n_broadcast_samples`` / ``include_inputs`` is never silently swallowed by
# the workflow layer.


def _resolve_value(
    op_name: str,
    dist: Any,
    value: Any,
    field_kwargs: dict[str, Any],
    *,
    allow_none: bool = False,
) -> Any:
    """Resolve a density op's value from the positional or keyword form.

    Keyword form packs ``field_kwargs`` into a single draw via
    ``dist._pack_value``; positional form passes ``value`` through. Raises
    if both forms are given. With ``allow_none=False`` (default) a missing
    value also raises; ``allow_none=True`` (the ``random_*`` ops) lets
    ``value=None`` through so the bare random function is returned.
    """
    if field_kwargs:
        if value is not None:
            raise TypeError(
                f"{op_name}: pass either a positional value or field kwargs, "
                f"not both."
            )
        if "value" in field_kwargs and "value" not in getattr(dist, "fields", ()):
            # ``value`` was the pre-#228 keyword name; it is now positional-only.
            raise TypeError(
                f"{op_name}: `value` is positional-only — call "
                f"{op_name}(dist, value), not {op_name}(dist, value=...)."
            )
        return dist._pack_value(**field_kwargs)
    if value is None and not allow_none:
        raise TypeError(
            f"{op_name}: a value is required — pass it positionally or as "
            f"field keyword arguments."
        )
    return value


class _DensityOp:
    """A density op: the positional/keyword value form over an inner
    :class:`~probpipe.core.node.WorkflowFunction`, plus a ``with_options``
    control path.

    Calling the op resolves the value — a positional ``value`` or named
    ``field_kwargs`` packed via :meth:`Distribution._pack_value` — and
    forwards to the inner impl, so broadcasting and orchestration are
    unchanged. :meth:`with_options` returns a callable that applies the given
    WorkflowFunction controls (``seed`` / ``n_broadcast_samples`` /
    ``include_inputs``) and accepts the same value form, mirroring
    ``condition_on.with_options(...)`` and the other dispatch ops.
    """

    def __init__(
        self, impl: Any, *, name: str, doc: str | None, allow_none: bool = False,
    ):
        self._impl = impl
        self._allow_none = allow_none
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = doc

    def _invoke(
        self, impl: Any, dist: Any, value: Any, field_kwargs: dict[str, Any],
    ) -> Any:
        value = _resolve_value(
            self.__name__, dist, value, field_kwargs, allow_none=self._allow_none,
        )
        return impl(dist, value)

    def __call__(self, dist: Any, value: Any = None, /, **field_kwargs: Any) -> Any:
        return self._invoke(self._impl, dist, value, field_kwargs)

    def with_options(self, **options: Any):
        """Return a callable that applies WorkflowFunction *options* for one
        call and accepts the same positional/keyword value form as the op."""
        configured = self._impl.with_options(**options)

        def call(dist: Any, value: Any = None, /, **field_kwargs: Any) -> Any:
            return self._invoke(configured, dist, value, field_kwargs)

        call.__name__ = call.__qualname__ = f"{self.__name__}.with_options"
        return call

    def __repr__(self) -> str:
        return f"<density op {self.__name__}>"


def _density_op(impl: Any, *, allow_none: bool = False):
    """Build a public density op from a stub carrying the op's name and
    docstring.

    The decorated stub's body is never executed — it is only the natural
    place to write the op's signature and docstring. The returned
    :class:`_DensityOp` provides the call behaviour (value resolution +
    ``with_options``) over *impl*, the inner ``_<op>_impl`` WorkflowFunction.
    """

    def decorate(stub: Any) -> _DensityOp:
        return _DensityOp(
            impl, name=stub.__name__, doc=stub.__doc__, allow_none=allow_none,
        )

    return decorate


@workflow_function(name="log_prob")
def _log_prob_impl(dist: SupportsLogProb, value: Any) -> Array:
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(f"{type(dist).__name__} does not support log_prob")
    return dist._log_prob(value)


# The decorated stubs below are name/docstring carriers; ``_density_op``
# replaces each with a ``_DensityOp`` over the inner impl (the body never runs).


@_density_op(_log_prob_impl)
def log_prob(dist, value=None, /, **field_kwargs):
    """Evaluate the normalized log-density at *value*.

    Two call forms:

    * **Positional** — ``log_prob(dist, value)``; *value* is a single draw
      of *dist*'s sample type, or a batched form (which broadcasts).
    * **Keyword** — ``log_prob(dist, field=value, ...)`` builds a single
      draw from named fields via :meth:`Distribution._pack_value`
      (single-field → bare value; multi-field → ``Record``). Use the
      positional form for batched evaluation.

    To override ProbPipe controls for one call, use ``with_options`` (the
    same idiom as the dispatch ops)::

        log_prob.with_options(seed=0)(dist, value)
    """


@workflow_function(name="prob")
def _prob_impl(dist: SupportsLogProb, value: Any) -> Array:
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support prob (missing _log_prob method)"
        )
    return jnp.exp(dist._log_prob(value))


@_density_op(_prob_impl)
def prob(dist, value=None, /, **field_kwargs):
    """Evaluate the density at *value* (``exp(log_prob)``).

    See :func:`log_prob` for the positional and keyword call forms and the
    ``with_options`` control path.
    """


@workflow_function(name="unnormalized_log_prob")
def _unnormalized_log_prob_impl(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_log_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return dist._unnormalized_log_prob(value)


@_density_op(_unnormalized_log_prob_impl)
def unnormalized_log_prob(dist, value=None, /, **field_kwargs):
    """Evaluate the unnormalized log-density at *value*.

    See :func:`log_prob` for the positional and keyword call forms and the
    ``with_options`` control path.
    """


@workflow_function(name="unnormalized_prob")
def _unnormalized_prob_impl(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return jnp.exp(dist._unnormalized_log_prob(value))


@_density_op(_unnormalized_prob_impl)
def unnormalized_prob(dist, value=None, /, **field_kwargs):
    """Evaluate the unnormalized density at *value*
    (``exp(unnormalized_log_prob)``).

    See :func:`log_prob` for the positional and keyword call forms and the
    ``with_options`` control path.
    """


@workflow_function
def mean(dist: SupportsMean) -> Any:
    """Compute ``E[X]`` where ``X ~ dist``.

    The return type is ``T``-shaped where ``T`` is *dist*'s sample type:

    * Numeric distributions (``T = Array``) — returns
      :class:`~probpipe.custom_types.Array`.
    * Structured distributions (``T = Record``) — returns
      :class:`~probpipe.record.Record`.
    * :class:`~probpipe.core._random_measures.RandomMeasure[T]` (``T``
      itself a :class:`~probpipe.core._distribution_base.Distribution[T]`)
      — returns the marginalised ``Distribution[T]`` with marginal
      ``D̄(A) = ∫ D(A) dM(D)``.

    Requires the distribution to implement :class:`SupportsMean`.
    """
    if not isinstance(dist, SupportsMean):
        raise TypeError(
            f"{type(dist).__name__} does not support mean "
            f"(does not implement SupportsMean)"
        )
    return dist._mean()


@workflow_function
def variance(dist: SupportsVariance) -> Any:
    """Compute Var[X].

    Requires the distribution to implement :class:`SupportsVariance`.
    """
    if not isinstance(dist, SupportsVariance):
        raise TypeError(
            f"{type(dist).__name__} does not support variance "
            f"(does not implement SupportsVariance)"
        )
    return dist._variance()


@workflow_function
def cov(dist: SupportsCovariance) -> Array:
    """Compute the covariance matrix.

    Requires the distribution to implement :class:`SupportsCovariance`.
    """
    if not isinstance(dist, SupportsCovariance):
        raise TypeError(
            f"{type(dist).__name__} does not support covariance "
            f"(does not implement SupportsCovariance)"
        )
    return dist._cov()


@workflow_function
def expectation(
    dist: SupportsExpectation,
    f: Any,
    *,
    key: PRNGKey | None = None,
    num_evaluations: int | None = None,
    return_dist: bool | None = None,
) -> Any:
    """Compute E[f(X)] where X ~ dist."""
    if not isinstance(dist, SupportsExpectation):
        raise TypeError(
            f"{type(dist).__name__} does not support expectation"
        )
    return dist._expectation(
        f, key=key, num_evaluations=num_evaluations, return_dist=return_dist,
    )


@workflow_function(name="random_log_prob")
def _random_log_prob_impl(
    dist: SupportsRandomLogProb, value: Any = None,
) -> RandomFunction | Distribution:
    if not isinstance(dist, SupportsRandomLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support random_log_prob "
            f"(does not implement SupportsRandomLogProb)"
        )
    rf = dist._random_log_prob()
    return rf if value is None else rf(value)


@_density_op(_random_log_prob_impl, allow_none=True)
def random_log_prob(dist, value=None, /, **field_kwargs):
    """Return the random (normalized) log-density of a random measure.

    For a ``RandomMeasure[T]`` ``M`` with draws ``D ~ M``, the random
    function ``x ↦ log D(x)`` is itself a callable returning a
    distribution over scalars at every input.

    When *value* is omitted, returns that callable as a
    :class:`~probpipe.core._random_functions.RandomFunction`. When
    *value* is provided (positionally, or built from ``field_kwargs`` via
    :meth:`Distribution._pack_value`), returns the ``Distribution[Array]``
    over ``log D(value)`` directly — equivalent to
    ``random_log_prob(dist)(value)``. The positional and keyword forms
    mirror :func:`log_prob`; per-call controls use
    ``random_log_prob.with_options(...)``.

    Concrete subclasses implement a single method
    ``_random_log_prob()`` returning a ``RandomFunction``; the optional
    *value* dispatch lives entirely in this op, not on the protocol.
    """


@workflow_function(name="random_unnormalized_log_prob")
def _random_unnormalized_log_prob_impl(
    dist: SupportsRandomUnnormalizedLogProb, value: Any = None,
) -> RandomFunction | Distribution:
    if not isinstance(dist, SupportsRandomUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support random_unnormalized_log_prob "
            f"(does not implement SupportsRandomUnnormalizedLogProb)"
        )
    rf = dist._random_unnormalized_log_prob()
    return rf if value is None else rf(value)


@_density_op(_random_unnormalized_log_prob_impl, allow_none=True)
def random_unnormalized_log_prob(dist, value=None, /, **field_kwargs):
    """Return the random unnormalized log-density of a random measure.

    For a ``RandomMeasure[T]`` ``M`` with draws ``D ~ M``, the random
    function ``x ↦ log D̃(x)`` (where ``D̃`` is the unnormalized density
    of ``D``) is itself a callable returning a distribution over
    scalars at every input.

    When *value* is omitted, returns that callable as a
    :class:`~probpipe.core._random_functions.RandomFunction`. When
    *value* is provided (positionally, or built from ``field_kwargs`` via
    :meth:`Distribution._pack_value`), returns the ``Distribution[Array]``
    over ``log D̃(value)`` directly — equivalent to
    ``random_unnormalized_log_prob(dist)(value)``. The positional and
    keyword forms mirror :func:`unnormalized_log_prob`; per-call controls
    use ``random_unnormalized_log_prob.with_options(...)``.

    Concrete subclasses implement a single method
    ``_random_unnormalized_log_prob()`` returning a ``RandomFunction``;
    the optional *value* dispatch lives entirely in this op, not on
    the protocol.
    """


def _split_data_kwargs(
    dist: Distribution,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate named data kwargs from inference kwargs.

    Uses the distribution's ``fields`` as the sole signal:
    any kwarg whose name matches a component name is data (conditioning
    target); everything else is an inference parameter.

    Returns ``(data_kwargs, inference_kwargs)``.
    """
    comp_names = frozenset(
        dist.fields if hasattr(dist, 'fields') else ()
    )
    data_kwargs = {k: v for k, v in kwargs.items() if k in comp_names}
    inference_kwargs = {k: v for k, v in kwargs.items() if k not in comp_names}
    return data_kwargs, inference_kwargs


@workflow_function
def condition_on(
    dist: Distribution,
    observed: Any = None,
    *,
    method: str | None = None,
    **kwargs: Any,
) -> Distribution:
    """Condition a distribution on observed values.

    Observed data can be passed positionally or as named keyword
    arguments::

        # Positional (backward compatible):
        condition_on(model, y_obs)

        # Named data kwargs — bundled into Record(X=..., y=...):
        condition_on.with_options(n_broadcast_samples=16)(
            model, X=bootstrap["X"], y=bootstrap["y"],
        )

    When named data kwargs are distribution views from the same parent,
    the workflow function broadcasting machinery samples the parent once
    and distributes the fields, preserving joint correlation.

    Dispatch priority:

    1. **Explicit override** — ``method="tfp_nuts"`` (or any registered
       name) routes directly to the named inference method.
    2. **Exact conditioning** — if *dist* implements
       ``SupportsConditioning``, its ``_condition_on`` is called for a
       closed-form result (e.g., conjugate updates, joint marginalization).
    3. **Registry auto-select** — the inference method registry picks
       the highest-priority feasible algorithm (NUTS, HMC, RWMH, etc.).

    Parameters
    ----------
    dist : Distribution
        Distribution or model to condition.  Need not implement
        ``SupportsConditioning`` — the registry provides inference
        methods for common model types.
    observed : Any
        Observed values to condition on.
    method : str or None
        If provided, use the named inference method from the registry
        instead of the default dispatch.
    **kwargs
        Inference parameters (e.g., ``num_results``, ``num_warmup``,
        ``random_seed``) and/or named data kwargs.  Any kwarg whose
        name matches a distribution component name is treated as
        observed data; everything else is an inference parameter.
    """
    from ..inference import inference_method_registry
    from .record import Record

    # Separate data kwargs (names matching fields) from
    # inference kwargs (everything else like num_results, num_warmup).
    data_kwargs, inference_kwargs = _split_data_kwargs(dist, kwargs)

    # Explicit method override → always use the registry
    if method is not None:
        if data_kwargs:
            if observed is not None:
                raise ValueError(
                    "Cannot provide both positional `observed` and named "
                    f"data kwargs ({', '.join(data_kwargs)})"
                )
            observed = Record(data_kwargs)
        return inference_method_registry.execute(
            dist, observed, method=method, **inference_kwargs
        )

    # Exact conditioning (conjugate updates, joint marginalization, etc.)
    # All kwargs pass through to _condition_on — it handles its own
    # validation (e.g., ProductDistribution raises KeyError on unknown names).
    if isinstance(dist, SupportsConditioning):
        return dist._condition_on(observed, **data_kwargs, **inference_kwargs)

    # Registry auto-selects the best approximate inference algorithm.
    # Data kwargs are bundled into observed as a Record object.
    if data_kwargs:
        if observed is not None:
            raise ValueError(
                "Cannot provide both positional `observed` and named "
                f"data kwargs ({', '.join(data_kwargs)})"
            )
        observed = Record(data_kwargs)
    return inference_method_registry.execute(dist, observed, **inference_kwargs)


@workflow_function
def from_distribution(
    source: Distribution,
    target_type: type,
    *,
    key: Any | None = None,
    check_support: bool = True,
    **kwargs: Any,
) -> Any:
    """Convert *source* into an instance of *target_type*.

    Delegates to the global converter registry.

    Parameters
    ----------
    source : Distribution
        Source distribution to convert.
    target_type : type
        The target distribution class.
    key : PRNGKey, optional
        JAX PRNG key for sampling-based conversion.
    check_support : bool
        If ``True`` (default), verify the supports are compatible.
    **kwargs
        Additional keyword arguments passed to the converter.
    """
    from ..converters import converter_registry
    if key is None:
        key = _auto_key()
    return converter_registry.convert(
        source, target_type, key=key, check_support=check_support, **kwargs
    )
