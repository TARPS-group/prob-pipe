"""A law built from a sampling function, a log-density, or both (IV.4).

``distribution(sample=..., log_prob=..., event_spec=..., output_name=...)`` returns the
law of the functions it is given. The law claims the capability each function
realizes, and construction checks each function that traces in JAX against
the event spec, abstractly.

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

from ..core._execution_claims import (
    ProbeResults,
    Traceability,
    probe_closure,
    spec_stand_in,
)
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
from ._distribution import DEFAULT_LABEL, Distribution, _given_label, _value_at_declared_kind
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
    spec = self.event_spec.term_spec
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
    spec = law.event_spec.term_spec
    if isinstance(spec, NumericArraySpec):
        value = jnp.asarray(value)
    axes = _leading_axes(value, spec) or ()
    if not axes:
        return jnp.asarray(density(_value_at_declared_kind(spec, law.label, value)))
    count = math.prod(axes)
    scores = _each_value(
        lambda one: jnp.asarray(density(_value_at_declared_kind(spec, law.label, one))),
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
        The complete event spec of one draw.
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
    _default_label: ClassVar[str | None] = DEFAULT_LABEL

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
        super().__init__(
            event_spec,
            label=label,
        )
        self._sampler = sample
        self._log_density = log_prob
        self._unnormalized_log_density = unnormalized_log_prob
        self._sampler_traces = sampler_traces
        self._density_traces = density_traces

    def _derived_traceability(self) -> Traceability:
        """Whether the law's density traced when the law was built, or its sampler for a law without one.

        A law's traceability is that of the method that scores it, as for any
        law (:meth:`Distribution._derived_traceability`), so a law with a
        density that traces is traceable whatever its sampler does.
        """
        has_density = self._log_density is not None or self._unnormalized_log_density is not None
        traces = self._density_traces if has_density else self._sampler_traces
        return Traceability.TRACEABLE if traces else Traceability.HOST_ONLY

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


#: The probe results of the functions laws are built from, keyed by each
#: function's code and the structure of what it captures, so the law a kernel
#: builds anew at each given value is probed once (V.13, probe rule 3).
_FUNCTION_PROBE_RESULTS = ProbeResults(max_entries=1024)


def _abstract_spec(draw: Any) -> TermSpec:
    """The spec of an abstract draw: each array's shape and dtype, in its records."""
    if isinstance(draw, Mapping):
        return RecordSpec({name: _abstract_spec(child) for name, child in draw.items()})
    return NumericArraySpec(tuple(draw.shape), draw.dtype)


def _sampler_declaration(
    label: str, sample: Callable[[PRNGKey], Any], declaration: OutputSpec
) -> tuple[OutputSpec, bool]:
    """*declaration* completed from the abstract draw of *sample*, and whether *sample* traces.

    Only an event spec whose type is numeric or pending is read abstractly,
    from the probe of *sample* at one key, which closures of one code share. A
    sampler whose probe fails does not trace, and *declaration* is returned as
    it is, as it is for an event that is not numeric.

    Parameters
    ----------
    label : str
        The law's label, which error messages name.
    sample : callable
        The sampler, which returns one draw at a PRNG key.
    declaration : OutputSpec
        The event spec, whose type may be pending.

    Returns
    -------
    declaration : OutputSpec
        The event spec that ``OutputSpec.with_term_spec`` completes with the
        abstract draw's spec, or *declaration* as it is when no abstract draw is
        read.
    traces : bool
        Whether *sample* traces in JAX.

    Raises
    ------
    ValueError
        If the abstract draw does not conform to *declaration*.
    """
    if declaration.term_spec is not None and not isinstance(declaration.term_spec, NumericSpec):
        return declaration, False
    probe_result = probe_closure(
        _FUNCTION_PROBE_RESULTS,
        sample,
        lambda sampler, key: jax.tree.map(jnp.asarray, _raw_record(sampler(key))),
        jax.random.key(0),
        label=label,
        role="sample",
    )
    if not probe_result.traces:
        # The sampler does not trace; a failure of its own raises at a draw.
        return declaration, False
    spec = _abstract_spec(probe_result.abstract_output)
    try:
        return declaration.with_term_spec(spec), True
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"sample of {label!r} returns draws that do not match event_spec: {error}; the "
            f"draws are {spec!r}"
        ) from None


def _event_declaration(output_name: str | None, event_spec: Any) -> OutputSpec:
    """*event_spec* as an output spec, a bare spec of a whole term under *output_name*.

    Parameters
    ----------
    output_name : str or None
        The name of a whole-term event's output, which :func:`distribution`
        received.
    event_spec : OutputSpec or TermSpec
        The event spec :func:`distribution` received.

    Returns
    -------
    OutputSpec
        *event_spec* itself when it is an ``OutputSpec``, the positional form
        over a bare ``RecordSpec``, and otherwise the bare spec under *output_name*.

    Raises
    ------
    TypeError
        If *event_spec* is neither an ``OutputSpec`` nor a ``TermSpec``, a bare
        spec other than a record comes without *output_name*, or *output_name*
        comes with an ``OutputSpec`` or a ``RecordSpec``, which name their own
        outputs.
    """
    if isinstance(event_spec, OutputSpec | RecordSpec):
        if output_name is not None:
            raise TypeError(
                f"distribution: {type(event_spec).__name__} names its own outputs, so pass "
                f"no output_name; got output_name={output_name!r}"
            )
        return event_spec if isinstance(event_spec, OutputSpec) else OutputSpec(event_spec)
    if isinstance(event_spec, TermSpec):
        if output_name is None:
            raise TypeError(
                f"distribution: a whole-term event of type {type(event_spec).__name__} needs an "
                f"output name; pass output_name='name'"
            )
        if not isinstance(output_name, str):
            raise TypeError(f"distribution: output_name must be a string; got {output_name!r}")
        return OutputSpec.default(event_spec, output_name=output_name)
    raise TypeError(
        f"distribution: event_spec must be an OutputSpec or a TermSpec; got "
        f"{type(event_spec).__name__}"
    )


def _density_traces(
    label: str, name: str, density: Callable[[Any], Array], declaration: OutputSpec
) -> bool:
    """Whether *density* traces at a stand-in of one value of the event, where it must give a scalar.

    An event spec without a stand-in, such as one with a value that is no
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
        The complete event spec.

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
    spec = declaration.term_spec
    stand_in = spec_stand_in(spec)
    if stand_in is None:
        return False
    probe_result = probe_closure(
        _FUNCTION_PROBE_RESULTS,
        density,
        lambda scoring_function, value: jnp.asarray(
            scoring_function(_value_at_declared_kind(spec, label, value))
        ),
        stand_in,
        label=label,
        role=name,
    )
    if not probe_result.traces:
        # The density does not trace; a failure of its own raises at a value.
        return False
    score = probe_result.abstract_output
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
    *,
    sample: Callable[[PRNGKey], Any] | None = None,
    log_prob: Callable[[Any], Array] | None = None,
    unnormalized_log_prob: Callable[[Any], Array] | None = None,
    event_spec: OutputSpec | TermSpec,
    output_name: str | None = None,
    label: str | None = None,
) -> Distribution:
    """Build a ``Distribution`` from a sampling function, a log-density, or both.

    ``sample(key)`` returns one draw at a PRNG key, at the kind the event
    spec names, as an array for an array event or a mapping or
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

    Construction draws nothing and scores no value. It traces each function
    once, without values, to learn whether JAX can trace it: the abstract draw
    of *sample* must unify with *event_spec*, which it completes by filling a
    pending type and setting a dtype it leaves unset, and each
    density must return a real scalar at a stand-in of one value of the event.
    A function that does not trace is read nothing from, and it makes the law
    run one value at a time. Every law built from the same code with captured
    values of the same shapes and dtypes, such as the law a kernel builds at
    each given value, shares that trace, so the trace runs once, unless the
    function reads a captured value, as ``jnp.zeros(int(n))`` does. A function
    that draws from NumPy's or Python's global random generator is treated as
    one that does not trace, with a warning.

    A kernel of a simulator samples and has no density::

        simulator = conditional_distribution(
            lambda rate: distribution(
                sample=lambda key: jax.random.poisson(key, rate, (10,)),
                event_spec=NumericArraySpec((10,), jnp.int32, non_negative_integer),
                output_name="y",
            ),
            label="simulator",
            given_spec={"rate": NumericArraySpec((), jnp.float32, positive)},
        )

    Parameters
    ----------
    sample : callable, optional
        ``sample(key)``, one draw of the law at a PRNG key.
    log_prob : callable, optional
        ``log_prob(value)``, the normalized log-density of one value.
    unnormalized_log_prob : callable, optional
        ``unnormalized_log_prob(value)``, the log-density of one value up to an
        additive constant.
    event_spec : OutputSpec or TermSpec
        The spec of one draw. An ``OutputSpec`` names its outputs, the fields
        of a bare ``RecordSpec`` are the outputs, and any other bare spec is a
        whole term under *output_name*.
    output_name : str, optional
        The name of the output of a whole-term event declared by a bare spec
        other than a record; required for one, and refused with any other
        *event_spec*.
    label : str, optional
        The law's label, ``ℙ`` by default.

    Returns
    -------
    Distribution
        The law, which claims the capability of each function given.

    Raises
    ------
    TypeError
        If *label* is not a non-empty string; if no function is given, a given
        function is not callable, or both densities are given; if *event_spec*
        is not a spec; if *output_name* is missing for a whole-term event or given
        for another; or if *event_spec* has a pending type and no sampler that
        traces fills it.
    ValueError
        If the abstract draw of *sample* does not conform to *event_spec*, or a
        density that traces returns anything but a real scalar.
    """
    label = _given_label(label, DEFAULT_LABEL, owner="distribution")
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
    declaration = _event_declaration(output_name, event_spec)
    sampler_traces = False
    if sample is not None:
        declaration, sampler_traces = _sampler_declaration(label, sample, declaration)
    if declaration.term_spec is None:
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
