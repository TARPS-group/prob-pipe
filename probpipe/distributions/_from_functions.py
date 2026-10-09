"""A law built from a sampling function, a log-density, or both (IV.4).

``distribution(label, sample=..., log_prob=..., event_spec=...)`` returns the
law of the functions it is given. The law claims the capability each function
realizes, and construction checks each function that traces in JAX against
the event declaration, abstractly.

Provides:
  - ``distribution`` – the law of a sampler, a density, or both, over a
    declared event.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from ..core._record_spec import RecordSpec
from ..core._repr import format_value
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ..core._specs import OutputSpec
from ..custom_types import Array, PRNGKey
from ._capabilities import (
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    _capability_subclass,
)
from ._conditional import _argument
from ._distribution import Distribution
from ._factored import _each_value, _flatten_draws, _leading_axes, _raw_record

__all__ = ["distribution"]


# ---------------------------------------------------------------------------
# Draws and scores
# ---------------------------------------------------------------------------


def _raw_draw(draw: Any, spec: TermSpec) -> Any:
    """*draw* in the raw form *spec* declares: each array leaf as an array, any other leaf as it is."""
    if isinstance(spec, RecordSpec):
        children = _raw_record(draw)
        return {name: _raw_draw(children[name], child) for name, child in spec.children.items()}
    if isinstance(spec, NumericArraySpec):
        return jnp.asarray(draw)
    return draw


def _column(spec: TermSpec, draws: Sequence[Any]) -> Any:
    """The raw draws *draws* of *spec*, stacked along a new leading axis.

    An array leaf stacks into an array, and any other leaf into a NumPy array
    of objects, the column form of a batch of values that are not arrays.
    """
    if isinstance(spec, RecordSpec):
        return {
            name: _column(child, [draw[name] for draw in draws])
            for name, child in spec.children.items()
        }
    if isinstance(spec, NumericArraySpec):
        if draws:
            return jnp.stack(draws)
        dtype = spec.dtype if spec.dtype is not None else jnp.result_type(float)
        return jnp.zeros((0, *spec.shape), dtype)
    column = np.empty(len(draws), dtype=object)
    for index, draw in enumerate(draws):
        column[index] = draw
    return column


def _with_sample_axes(columns: Any, shape: tuple[int, ...]) -> Any:
    """*columns*, whose leaves lead with one axis of draws, with that axis split into *shape*."""

    def split(leaf: Any) -> Any:
        if isinstance(leaf, np.ndarray) and leaf.dtype == object:
            return leaf.reshape((*shape, *leaf.shape[1:]))
        return jnp.reshape(leaf, (*shape, *jnp.shape(leaf)[1:]))

    return jax.tree.map(split, columns)


def _draw(self: _FunctionLaw, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
    """Draws of the sampler: one at *key*, or one at each key split from it.

    A non-empty *sample_shape* splits *key* into one key per draw and leads
    each leaf with the sample axes. A sampler that traces is mapped over the
    keys with ``jax.vmap``, and any other is called at one key at a time, its
    draws stacked into the batch form of the event's kind.
    """
    spec = self.event_spec.spec
    shape = tuple(sample_shape)
    if not shape:
        return _raw_draw(self._sampler(key), spec)
    keys = jax.random.split(key, math.prod(shape))
    if self._sampler_traces:
        columns = jax.vmap(lambda one: _raw_draw(self._sampler(one), spec))(keys)
    else:
        columns = _column(spec, [_raw_draw(self._sampler(one), spec) for one in keys])
    return _with_sample_axes(columns, shape)


def _score(law: _FunctionLaw, density: Callable[[Any], Array], value: Any) -> Array:
    """*density* at *value*, or at each value of a batch along its leading axes.

    *density* receives each value at the kind the event declares: an array, a
    ``Record``, or the value itself. A batch is scored with ``jax.vmap`` when
    the density traces and one value at a time otherwise, and the result has
    the batch's leading axes. A batch is recognized at an array leaf, so a
    value of an event without one is scored whole.
    """
    spec = law.event_spec.spec
    if isinstance(spec, NumericArraySpec):
        value = jnp.asarray(value)
    axes = _leading_axes(value, spec) or ()
    if not axes:
        return jnp.asarray(density(_argument(spec, law.label, value)))
    count = math.prod(axes)
    scores = _each_value(
        lambda one: jnp.asarray(density(_argument(spec, law.label, one))),
        count,
        _flatten_draws(_raw_record(value), axes),
        traceable=law._density_traces,
    )
    return jnp.reshape(scores, axes)


def _normalized_density(self: _FunctionLaw, value: Any) -> Array:
    """The log-density of *value*, or of each value of a batch along its leading axes."""
    return _score(self, self._log_density, value)


def _unnormalized_density(self: _FunctionLaw, value: Any) -> Array:
    """The log-density of *value* up to an additive constant, or of each value of a batch."""
    return _score(self, self._unnormalized_log_density, value)


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------


class _FunctionLaw(Distribution):
    """The law of a sampler, a density, or both, which :func:`distribution` builds.

    Each instance is of the subclass that claims the capabilities its functions
    realize, as :func:`distribution` states.

    Parameters
    ----------
    label : str
        The law's label.
    event_spec : OutputSpec
        The complete declaration of one draw.
    sample : callable or None
        The sampler, as :func:`distribution` takes it.
    log_prob : callable or None
        The normalized log-density, as :func:`distribution` takes it.
    unnormalized_log_prob : callable or None
        The log-density up to an additive constant, as :func:`distribution`
        takes it.
    sampler_traces : bool
        Whether *sample* traces in JAX, so a sample shape maps it with
        ``jax.vmap`` rather than with a Python loop.
    density_traces : bool
        Whether the density traces in JAX, so a batch is scored with
        ``jax.vmap`` rather than with a Python loop.
    """

    _capability_table: ClassVar = {
        SupportsSampling: {"_sample": _draw},
        SupportsLogProb: {"_log_prob": _normalized_density},
        SupportsUnnormalizedLogProb: {"_unnormalized_log_prob": _unnormalized_density},
    }

    def __new__(
        cls,
        label: str,
        event_spec: OutputSpec,
        *,
        sample: Callable[[PRNGKey], Any] | None,
        log_prob: Callable[[Any], Array] | None,
        unnormalized_log_prob: Callable[[Any], Array] | None,
        sampler_traces: bool,
        density_traces: bool,
    ) -> _FunctionLaw:
        functions = {
            SupportsSampling: sample,
            SupportsLogProb: log_prob,
            SupportsUnnormalizedLogProb: unnormalized_log_prob,
        }
        claimed = [protocol for protocol, function in functions.items() if function is not None]
        return object.__new__(_capability_subclass(_FunctionLaw, claimed))

    def __init__(
        self,
        label: str,
        event_spec: OutputSpec,
        *,
        sample: Callable[[PRNGKey], Any] | None,
        log_prob: Callable[[Any], Array] | None,
        unnormalized_log_prob: Callable[[Any], Array] | None,
        sampler_traces: bool,
        density_traces: bool,
    ) -> None:
        super().__init__(label, event_spec)
        self._sampler = sample
        self._log_density = log_prob
        self._unnormalized_log_density = unnormalized_log_prob
        self._sampler_traces = sampler_traces
        self._density_traces = density_traces

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The functions the law was built from, by name."""
        functions = {
            "sample": self._sampler,
            "log_prob": self._log_density,
            "unnormalized_log_prob": self._unnormalized_log_density,
        }
        return [(name, format_value(f)) for name, f in functions.items() if f is not None]


# ---------------------------------------------------------------------------
# The abstract checks
# ---------------------------------------------------------------------------


def _abstract_spec(draw: Any) -> TermSpec:
    """The spec of an abstract draw: each array's shape and dtype, in its records."""
    if isinstance(draw, Mapping):
        return RecordSpec({name: _abstract_spec(child) for name, child in draw.items()})
    return NumericArraySpec(tuple(draw.shape), draw.dtype)


def _sampler_declaration(
    label: str, sample: Callable[[PRNGKey], Any], declaration: OutputSpec
) -> tuple[OutputSpec, bool]:
    """*declaration* completed from the abstract draw of *sample*, and whether *sample* traces.

    Only a declaration that is numeric or pending is read abstractly. A sampler
    whose ``jax.eval_shape`` fails does not trace, and *declaration* is
    returned as it is, as it is for an event that is not numeric.

    Parameters
    ----------
    label : str
        The law's label, which error messages name.
    sample : callable
        The sampler, which returns one draw at a PRNG key.
    declaration : OutputSpec
        The event declaration, whose type may be pending.

    Returns
    -------
    declaration : OutputSpec
        The declaration that ``OutputSpec.with_spec`` completes with the abstract
        draw's spec, or *declaration* as it is when no abstract draw is read.
    traces : bool
        Whether *sample* traces in JAX.

    Raises
    ------
    ValueError
        If the abstract draw does not conform to *declaration*.
    """
    if declaration.spec is not None and not isinstance(declaration.spec, NumericSpec):
        return declaration, False
    try:
        abstract = jax.eval_shape(
            lambda key: jax.tree.map(jnp.asarray, _raw_record(sample(key))), jax.random.key(0)
        )
    except Exception:
        # The sampler does not trace; a failure of its own raises at a draw.
        return declaration, False
    spec = _abstract_spec(abstract)
    try:
        return declaration.with_spec(spec), True
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"sample of {label!r} returns draws that do not match event_spec: {error}; the "
            f"draws are {spec!r}"
        ) from None


def _event_declaration(label: str, event_spec: Any) -> OutputSpec:
    """*event_spec* as an output declaration, a bare spec completed as ``Distribution`` completes it.

    Parameters
    ----------
    label : str
        The law's label, which is the component of a bare spec.
    event_spec : OutputSpec or TermSpec
        The declaration :func:`distribution` received.

    Returns
    -------
    OutputSpec
        *event_spec* itself when it is an ``OutputSpec``, and otherwise the
        default declaration of the bare spec under the component *label*.

    Raises
    ------
    TypeError
        If *event_spec* is neither an ``OutputSpec`` nor a ``TermSpec``.
    """
    if isinstance(event_spec, OutputSpec):
        return event_spec
    if isinstance(event_spec, TermSpec):
        return OutputSpec.default(event_spec, component=label)
    raise TypeError(
        f"event_spec of {label!r} must be an OutputSpec or a TermSpec; got "
        f"{type(event_spec).__name__}"
    )


def _stand_in(spec: TermSpec) -> Any:
    """The abstract stand-in of one value of *spec*, or ``None`` when an array leaf has none.

    An array leaf of fixed shape stands in as its shape and dtype, and a record
    as the mapping of its fields' stand-ins.
    """
    if isinstance(spec, RecordSpec):
        children = {name: _stand_in(child) for name, child in spec.children.items()}
        return None if any(child is None for child in children.values()) else children
    if isinstance(spec, NumericArraySpec) and not spec.free_dims:
        dtype = spec.dtype if spec.dtype is not None else jnp.result_type(float)
        return jax.ShapeDtypeStruct(tuple(spec.shape), dtype)
    return None


def _density_traces(
    label: str, name: str, density: Callable[[Any], Array], declaration: OutputSpec
) -> bool:
    """Whether *density* traces at a stand-in of one value of the event, where it must give a scalar.

    A declaration without a stand-in, such as one with a value that is no
    array, reads no density, which then scores a batch one value at a time.

    Parameters
    ----------
    label : str
        The law's label, which error messages name.
    name : str
        The density's keyword, ``"log_prob"`` or ``"unnormalized_log_prob"``,
        which error messages name.
    density : callable
        The log-density, which receives one value at the kind the event
        declares.
    declaration : OutputSpec
        The complete event declaration.

    Returns
    -------
    bool
        ``True`` when *density* traces at the stand-in and returns a real
        scalar there, and ``False`` otherwise.

    Raises
    ------
    ValueError
        If *density* traces and returns anything but a real scalar.
    """
    spec = declaration.spec
    stand_in = _stand_in(spec)
    if stand_in is None:
        return False
    try:
        score = jax.eval_shape(
            lambda value: jnp.asarray(density(_argument(spec, label, value))), stand_in
        )
    except Exception:
        # The density does not trace; a failure of its own raises at a value.
        return False
    real = jnp.issubdtype(score.dtype, jnp.floating) or jnp.issubdtype(score.dtype, jnp.integer)
    if score.shape != () or not real:
        raise ValueError(
            f"{name} of {label!r} must return a real scalar for one value of the event; it "
            f"returns an array of shape {score.shape} and dtype {score.dtype}"
        )
    return True


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


# A plain function rather than a Function: the distribution layer is below
# functions/, which defines the @function decorator.
def distribution(
    label: str,
    /,
    *,
    sample: Callable[[PRNGKey], Any] | None = None,
    log_prob: Callable[[Any], Array] | None = None,
    unnormalized_log_prob: Callable[[Any], Array] | None = None,
    event_spec: OutputSpec | TermSpec,
) -> Distribution:
    """Build a ``Distribution`` from a sampling function, a log-density, or both.

    ``sample(key)`` returns one draw at a PRNG key, at the kind the event
    declaration names, as an array for an array event or a mapping or
    ``Record`` for a record event. ``log_prob(value)`` returns the normalized
    log-density of one value as a real scalar, and
    ``unnormalized_log_prob(value)`` returns one up to an additive constant. A density receives an array for an array event, a
    ``Record`` for a record event, and the value itself for any other event.

    The law claims the capability each function given realizes:

    - *sample*: ``SupportsSampling``;
    - *log_prob*: ``SupportsLogProb``, which provides the unnormalized density
      too;
    - *unnormalized_log_prob*: ``SupportsUnnormalizedLogProb``.

    A law of a density alone is therefore unnormalized, and ``sample``,
    ``convert``, and ``condition_on`` normalize it through the
    inference-method registry.

    A non-empty sample shape maps *sample* over keys split from the draw's key
    with ``jax.vmap``. A sampler that does not trace in JAX, such as one that
    calls NumPy or an external program, and a sampler of an event that is not
    numeric are called at one key at a time instead, and their draws stack into
    the batch form of the event's kind. A density scores a batch of values
    along its leading axes, with ``jax.vmap`` when it traces.

    Construction draws nothing and scores no value. It evaluates each function
    that traces abstractly, with ``jax.eval_shape``: the abstract draw of
    *sample* must unify with *event_spec*, which it completes by filling a
    pending type and setting a dtype the declaration leaves unset, and each
    density must return a real scalar at a stand-in of one value of the event.
    A function whose ``jax.eval_shape`` fails does not trace, and construction
    reads nothing from it.

    A kernel of a simulator samples and has no density::

        simulator = conditional_distribution(
            "y",
            lambda rate: distribution(
                "y",
                sample=lambda key: jax.random.poisson(key, rate, (10,)),
                event_spec=NumericArraySpec((10,), jnp.int32, non_negative_integer),
            ),
            given_spec={"rate": NumericArraySpec((), jnp.float32, positive)},
        )

    Parameters
    ----------
    label : str
        The law's label.
    sample : callable, optional
        ``sample(key)``, one draw of the law at a PRNG key.
    log_prob : callable, optional
        ``log_prob(value)``, the normalized log-density of one value.
    unnormalized_log_prob : callable, optional
        ``unnormalized_log_prob(value)``, the log-density of one value up to an
        additive constant.
    event_spec : OutputSpec or TermSpec
        The declaration of one draw, a bare spec completed as ``Distribution``
        completes one.

    Returns
    -------
    Distribution
        The law, which claims the capability of each function given.

    Raises
    ------
    TypeError
        If *label* is not a non-empty string; if no function is given, a given
        function is not callable, or both densities are given; if *event_spec*
        is not a spec; or if *event_spec* has a pending type and no sampler
        that traces fills it.
    ValueError
        If the abstract draw of *sample* does not conform to *event_spec*, or a
        density that traces returns anything but a real scalar.
    """
    if not isinstance(label, str) or not label:
        raise TypeError(f"distribution: label must be a non-empty string; got {label!r}")
    functions = {
        "sample": sample,
        "log_prob": log_prob,
        "unnormalized_log_prob": unnormalized_log_prob,
    }
    given = {name: function for name, function in functions.items() if function is not None}
    if not given:
        raise TypeError(
            f"distribution {label!r} needs at least one of sample, log_prob, or "
            f"unnormalized_log_prob"
        )
    for name, function in given.items():
        if not callable(function):
            raise TypeError(
                f"the {name} of {label!r} must be callable; got {type(function).__name__}"
            )
    if log_prob is not None and unnormalized_log_prob is not None:
        raise TypeError(f"distribution {label!r}: pass log_prob or unnormalized_log_prob, not both")
    declaration = _event_declaration(label, event_spec)
    sampler_traces = False
    if sample is not None:
        declaration, sampler_traces = _sampler_declaration(label, sample, declaration)
    if declaration.spec is None:
        raise TypeError(
            f"event_spec of {label!r} does not declare a type, and only a sample function that "
            f"traces in JAX can infer it; give the type, such as NumericArraySpec(()) for a "
            f"real scalar"
        )
    density_traces = False
    for name in ("log_prob", "unnormalized_log_prob"):
        if name in given:
            density_traces = _density_traces(label, name, given[name], declaration)
    return _FunctionLaw(
        label,
        declaration,
        sample=sample,
        log_prob=log_prob,
        unnormalized_log_prob=unnormalized_log_prob,
        sampler_traces=sampler_traces,
        density_traces=density_traces,
    )
