"""Execution claims, the probe, and its stored result (V.13).

A ``Function`` states at construction how it may be run: whether JAX can trace
it, and whether it takes a stack of evaluation points at once. Those
statements are its :class:`ExecutionClaims`. A callable that states nothing is
*probed*: traced once with ``jax.make_jaxpr`` on stand-ins for its inputs, to
learn whether JAX can trace it. The outcome, a :class:`ProbeResult`, is stored
by the callable's *input structure*, so a later call or ``check`` with inputs
of the same structure reads it instead of tracing again.

The module lives in ``core`` because every package probes: the engine probes a
``Function`` before it maps it, a law built from functions probes its sampler
and its densities, and a law without a stated traceability probes its density.

Provides:
  - :class:`ExecutionClaims` – the claims a ``Function`` states.
  - :class:`Traceability` – traceable, embeddable, or host-only, with their meet.
  - :class:`ProbeResult` and :class:`ProbeResults` – a stored probe outcome, and
    the bounded store of them.
  - :func:`probe` – the one probe, with its rules for Python numbers and for the
    global random generators.
  - :func:`closure_probe_key` – the key under which closures of one code share a
    probe result.
  - :func:`spec_stand_in` – the abstract stand-in of one value of a spec.
  - :func:`raise_for_false_traceable_claim` – the error for a false claim.
"""

from __future__ import annotations

import enum
import random
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from types import FunctionType
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import numpy as np

from ._record_spec import RecordSpec
from ._spec_base import NumericArraySpec, NumericSpec, TermSpec

__all__ = [
    "NO_CLAIMS",
    "TRACING_ERRORS",
    "ExecutionClaims",
    "ProbeResult",
    "ProbeResults",
    "Traceability",
    "closure_probe_key",
    "execution_claims",
    "probe",
    "raise_for_false_traceable_claim",
    "spec_stand_in",
]


# ---------------------------------------------------------------------------
# Claims, traceability, and the probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionClaims:
    """The statements a ``Function`` makes, when it is built, about how it may be run.

    The claims are fixed for the ``Function``'s life and are never controls.

    Attributes
    ----------
    traceable : bool or None
        ``True`` states that JAX can trace the callable: ``jax.jit`` and
        ``jax.vmap`` both succeed for its input shapes. ``False`` states that
        it cannot, so ProbPipe never traces or probes it. ``None`` states
        nothing, and the callable is probed once per input structure.
    vectorized : bool
        Whether the callable accepts every array argument with an extra
        leading axis of evaluation points and returns its outputs stacked on
        that axis, output row ``i`` depending only on input row ``i``.
    differentiable : NumericSpec or None
        The differentiability claim: the numeric inputs gradients flow
        through, or ``None`` for no claim.
    """

    traceable: bool | None = None
    vectorized: bool = False
    differentiable: NumericSpec | None = None


#: The claims of a callable that states nothing.
NO_CLAIMS = ExecutionClaims()


def execution_claims(
    *,
    function_name: str,
    traceable: bool | None,
    vectorized: bool,
    differentiable: NumericSpec | None,
) -> ExecutionClaims:
    """The claims *function_name* states, checked.

    Parameters
    ----------
    function_name : str
        The label of the ``Function`` that states them, which messages name.
    traceable : bool or None
        The ``traceable`` claim.
    vectorized : bool
        The ``vectorized`` claim.
    differentiable : NumericSpec or None
        The differentiability claim.

    Returns
    -------
    ExecutionClaims
        The claims.

    Raises
    ------
    TypeError
        If *traceable* is neither a bool nor ``None``, or *vectorized* is not a
        bool.
    """
    if traceable is not None and not isinstance(traceable, bool):
        raise TypeError(
            f"traceable of {function_name!r} must be True, False, or None, "
            f"got {type(traceable).__name__}"
        )
    if not isinstance(vectorized, bool):
        raise TypeError(
            f"vectorized of {function_name!r} must be True or False, got {type(vectorized).__name__}"
        )
    if traceable is None and not vectorized and differentiable is None:
        return NO_CLAIMS
    return ExecutionClaims(traceable, vectorized, differentiable)


class Traceability(enum.Enum):
    """How a computation may be run inside a JAX program, from the most permissive down.

    The members are ordered, and :meth:`meet` gives the least permissive of
    several, which is the traceability of a computation built from them.

    Attributes
    ----------
    TRACEABLE
        JAX can trace it, so it is compiled, vectorized, and differentiated.
    EMBEDDABLE
        JAX cannot trace it, but each opaque ``Function`` in it declares an
        output that a host callback can return.
    HOST_ONLY
        It runs only on concrete values, outside any JAX program.
    """

    TRACEABLE = 2
    EMBEDDABLE = 1
    HOST_ONLY = 0

    @classmethod
    def meet(cls, *traceabilities: Traceability) -> Traceability:
        """The least permissive of *traceabilities*, ``TRACEABLE`` for none.

        Parameters
        ----------
        *traceabilities : Traceability
            The traceabilities of a computation's parts.

        Returns
        -------
        Traceability
            The traceability of the whole.
        """
        return min(
            traceabilities, key=lambda traceability: traceability.value, default=cls.TRACEABLE
        )


@dataclass(frozen=True)
class ProbeResult:
    """The stored outcome of one probe.

    Attributes
    ----------
    traces : bool
        Whether JAX traced the callable at the probed input structure.
    ints_static : bool
        Whether it traced only with its Python ``int`` arguments static.
    output : Any
        The abstract output, a tree of ``jax.ShapeDtypeStruct``, when it
        traced, and ``None`` otherwise.
    error : BaseException or None
        The error the trace raised when it did not trace, without its
        traceback.
    draws_global_random : bool
        Whether the callable drew from NumPy's or Python's global random
        generator while it was traced, which makes it opaque.
    retry_with_static_ints : bool
        Whether this result only records that the form with ``int``\\ s as
        data failed, so the form with them static decides.
    """

    traces: bool
    ints_static: bool = False
    output: Any = None
    error: BaseException | None = None
    draws_global_random: bool = False
    retry_with_static_ints: bool = False

    @property
    def traceability(self) -> Traceability:
        """``TRACEABLE`` when the callable traced, and ``HOST_ONLY`` otherwise."""
        return Traceability.TRACEABLE if self.traces else Traceability.HOST_ONLY


class ProbeResults:
    """The probe results of one owner, by input structure, bounded and safe for several threads.

    The store keeps at most *max_entries* results and drops the least
    recently used one beyond that. It is shared by every copy of its owner
    made in the same process, and arrives empty in another process, since a
    probe result is cheap to compute again and holds a lock that cannot be
    pickled.

    Parameters
    ----------
    max_entries : int
        The bound on the number of stored results.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max_entries
        self._results: OrderedDict[Hashable, ProbeResult] = OrderedDict()
        self._lock = threading.Lock()

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a new, empty store with the same bound."""
        return (ProbeResults, (self._max_entries,))

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)

    def get(self, structure: Hashable) -> ProbeResult | None:
        """The result stored under *structure*, or ``None``.

        Parameters
        ----------
        structure : Hashable
            The input structure, with whatever else the owner keys by.

        Returns
        -------
        ProbeResult or None
            The stored result.
        """
        with self._lock:
            result = self._results.get(structure)
            if result is not None:
                self._results.move_to_end(structure)
            return result

    def put(self, structure: Hashable, result: ProbeResult) -> None:
        """Store *result* under *structure*, dropping the least recently used result past the bound.

        Parameters
        ----------
        structure : Hashable
            The input structure, with whatever else the owner keys by.
        result : ProbeResult
            The outcome to store.
        """
        with self._lock:
            self._results[structure] = result
            self._results.move_to_end(structure)
            while len(self._results) > self._max_entries:
                self._results.popitem(last=False)


def probe(
    store: ProbeResults | None,
    trace: Callable[[Any], Any],
    arguments: Any,
    *,
    label: str,
    key: Hashable = (),
) -> ProbeResult:
    """Whether *trace* traces at *arguments*, read from *store* or probed and stored there.

    A probe traces *trace* once with ``jax.make_jaxpr``, taking every numeric
    array, stand-in, and Python ``float`` of *arguments* as data and passing
    every other leaf as it is. It follows the probe rules of V.13:

    1. A Python ``int`` is data first. When the probe fails with an ``int`` as
       data, it is retried once with every ``int`` static, as one probe, and
       the result is stored under the input structure that holds the ``int``
       values.
    2. A callable that draws from NumPy's or Python's global random generator
       while it is traced is recorded as opaque, and a warning names it. The
       generators are put back to their states before the probe.

    The probe runs the callable's Python up to the first operation that fails
    on a tracer.

    Parameters
    ----------
    store : ProbeResults or None
        Where results are read and stored; ``None`` stores nothing.
    trace : Callable
        Called with the arguments, rebuilt with tracers for their data.
    arguments : Any
        The arguments, a tree of arrays, stand-ins, and other values.
    label : str
        The callable's label, which the warning names.
    key : Hashable
        What else the result depends on besides the input structure, such as
        the route the call runs.

    Returns
    -------
    ProbeResult
        The stored or new outcome.
    """
    data_form = _probe_arguments(arguments, ints_static=False)
    data_key = None if store is None or data_form.structure is None else (key, data_form.structure)
    stored = None if data_key is None else store.get(data_key)
    if stored is not None and not stored.retry_with_static_ints:
        return stored
    if stored is None:
        result = _trace_once(trace, data_form)
        if result.draws_global_random:
            _warn_global_random(label)
        if result.traces or result.draws_global_random or not data_form.has_int_data:
            if data_key is not None:
                store.put(data_key, result)
            return result
        if data_key is not None:
            store.put(data_key, ProbeResult(traces=False, retry_with_static_ints=True))
    static_form = _probe_arguments(arguments, ints_static=True)
    static_key = (
        None if store is None or static_form.structure is None else (key, static_form.structure)
    )
    static_stored = None if static_key is None else store.get(static_key)
    if static_stored is not None:
        return static_stored
    static_result = _trace_once(trace, static_form)
    if static_result.draws_global_random:
        _warn_global_random(label)
    static_result = ProbeResult(
        traces=static_result.traces,
        ints_static=static_result.traces,
        output=static_result.output,
        error=static_result.error,
        draws_global_random=static_result.draws_global_random,
    )
    if static_key is not None:
        store.put(static_key, static_result)
    return static_result


def closure_probe_key(callable_: Callable[..., Any]) -> Hashable:
    """The key under which every closure of *callable_*'s code with captures of one structure shares a probe result.

    A plain function counts by its code and by the structure of what it
    captures: the values in its closure cells and its defaults. A captured
    array counts by its shape and dtype, and a hashable value by its value.
    Any other callable, or a function that captures any other value, counts by
    itself, so it shares its result only with itself.

    Parameters
    ----------
    callable_ : Callable
        The closure, such as the density of a law built inside a kernel.

    Returns
    -------
    Hashable
        The key.
    """
    if isinstance(callable_, FunctionType):
        captured = [cell.cell_contents for cell in callable_.__closure__ or ()]
        captured.extend(callable_.__defaults__ or ())
        captured.extend((callable_.__kwdefaults__ or {}).values())
        try:
            return (
                "code",
                callable_.__code__,
                tuple(_captured_value_key(value) for value in captured),
            )
        except (_Unkeyable, ValueError):
            pass
    try:
        hash(callable_)
    except TypeError:
        return ("unshared", object())
    return ("callable", callable_)


def spec_stand_in(spec: TermSpec | None) -> Any:
    """The abstract stand-in of one value of *spec*, or ``None`` when an array leaf has none.

    An array leaf of fixed shape stands in as its shape and dtype, the default
    float dtype when it sets none, and a record as the mapping of its fields'
    stand-ins. A spec of any other kind, and an array with a dimension not yet
    bound, have none.

    Parameters
    ----------
    spec : TermSpec or None
        The spec of the value.

    Returns
    -------
    jax.ShapeDtypeStruct or dict or None
        The stand-in.
    """
    if isinstance(spec, RecordSpec):
        children = {name: spec_stand_in(child) for name, child in spec.children.items()}
        return None if any(child is None for child in children.values()) else children
    if isinstance(spec, NumericArraySpec) and not spec.free_dims:
        dtype = spec.dtype if spec.dtype is not None else jnp.result_type(float)
        return jax.ShapeDtypeStruct(tuple(spec.shape), dtype)
    return None


#: What each JAX tracing error says the callable did, most specific first.
_TRACING_FAILURES: tuple[tuple[type[Exception], str], ...] = (
    (jax.errors.TracerArrayConversionError, "it converts a traced value to NumPy"),
    (jax.errors.TracerBoolConversionError, "it branches in Python on a traced value"),
    (jax.errors.TracerIntegerConversionError, "it converts a traced value to a Python int"),
    (jax.errors.NonConcreteBooleanIndexError, "its output shape depends on a traced value"),
    (jax.errors.ConcretizationTypeError, "it reads the value of a traced input"),
)


#: The JAX errors that mean a callable cannot be traced.
TRACING_ERRORS: tuple[type[Exception], ...] = tuple(error for error, _ in _TRACING_FAILURES)


def raise_for_false_traceable_claim(function_name: str, error: Exception) -> NoReturn:
    """Raise the error for a callable that claims ``traceable=True`` but failed to trace.

    Parameters
    ----------
    function_name : str
        The label of the ``Function`` that makes the claim.
    error : Exception
        The JAX tracing error, one of :data:`TRACING_ERRORS`, which the raised
        error is chained to.

    Raises
    ------
    TypeError
        Always, naming the claim, what the callable did, and JAX's error.
    """
    failure = next(
        description for kind, description in _TRACING_FAILURES if isinstance(error, kind)
    )
    raise TypeError(
        f"{function_name!r} claims traceable=True, but tracing it failed: {failure} "
        f"({type(error).__name__}). Remove the claim, or claim traceable=False"
    ) from error


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_numeric_dtype(dtype: Any) -> bool:
    """Whether *dtype* is a number or a PRNG key, which a probe takes as data."""
    return jnp.issubdtype(dtype, jax.dtypes.prng_key) or np.dtype(dtype).kind in "biufc"


def _array_leaf_key(leaf: Any) -> tuple[Any, ...]:
    """The part of the input structure an array leaf contributes: its shape, dtype, and kind."""
    origin = "tracer" if isinstance(leaf, jax.core.Tracer) else "array"
    return (
        origin,
        tuple(leaf.shape),
        str(leaf.dtype),
        bool(getattr(leaf, "weak_type", False)),
    )


def _is_array_leaf(leaf: Any) -> bool:
    """Whether *leaf* is an array a probe takes as data, a stand-in of one included."""
    if isinstance(leaf, jax.Array | np.ndarray | np.generic | jax.ShapeDtypeStruct):
        return _is_numeric_dtype(leaf.dtype)
    return False


def _is_python_int(leaf: Any) -> bool:
    """Whether *leaf* is a Python ``int``, a ``bool`` included."""
    return isinstance(leaf, int)


class _Unkeyable(Exception):
    """An argument has no part in an input structure: it is neither an array nor hashable."""


def _static_leaf_key(leaf: Any) -> tuple[Any, ...]:
    """The part of the input structure a leaf passed as it is contributes: its type and value."""
    try:
        hash(leaf)
    except TypeError:
        raise _Unkeyable from None
    return ("static", type(leaf), leaf)


@dataclass(frozen=True)
class _ProbeArguments:
    """A call's arguments split into the data a probe traces and the values it passes as they are.

    Attributes
    ----------
    structure : Hashable or None
        The input structure of the arguments in this form, or ``None`` when an
        argument is neither an array nor hashable.
    data_leaves : tuple
        The leaves the probe traces, in leaf order.
    rebuild : Callable
        Rebuilds the arguments from values of the data leaves.
    has_int_data : bool
        Whether a Python ``int`` is among the data leaves.
    """

    structure: Hashable | None
    data_leaves: tuple[Any, ...]
    rebuild: Callable[[tuple[Any, ...]], Any]
    has_int_data: bool


def _probe_arguments(arguments: Any, *, ints_static: bool) -> _ProbeArguments:
    """*arguments* split for a probe, with Python ``int``\\ s as data or as static values.

    A numeric array, a stand-in of one, and a Python ``float`` or ``complex``
    are data, the float a weakly typed scalar. A Python ``int`` is data unless
    *ints_static*. Every other leaf is passed as it is, and enters the input
    structure by its value.
    """
    leaves, treedef = jax.tree.flatten(arguments)
    data_positions: list[int] = []
    leaf_keys: list[tuple[Any, ...]] = []
    structure_known = True
    has_int_data = False
    for position, leaf in enumerate(leaves):
        if _is_array_leaf(leaf):
            data_positions.append(position)
            leaf_keys.append(_array_leaf_key(leaf))
        elif isinstance(leaf, float | complex) or (_is_python_int(leaf) and not ints_static):
            data_positions.append(position)
            leaf_keys.append(("scalar", type(leaf)))
            has_int_data = has_int_data or _is_python_int(leaf)
        else:
            try:
                leaf_keys.append(_static_leaf_key(leaf))
            except _Unkeyable:
                structure_known = False

    def rebuild(data_values: tuple[Any, ...]) -> Any:
        rebuilt = list(leaves)
        for position, value in zip(data_positions, data_values, strict=True):
            rebuilt[position] = value
        return jax.tree.unflatten(treedef, rebuilt)

    return _ProbeArguments(
        structure=(treedef, tuple(leaf_keys)) if structure_known else None,
        data_leaves=tuple(leaves[position] for position in data_positions),
        rebuild=rebuild,
        has_int_data=has_int_data,
    )


def _global_random_states() -> tuple[Any, Any]:
    """The states of NumPy's and Python's global random generators."""
    return np.random.get_state(), random.getstate()


def _states_equal(before: tuple[Any, Any], after: tuple[Any, Any]) -> bool:
    """Whether two snapshots of the global random states are the same."""
    numpy_before, python_before = before
    numpy_after, python_after = after
    same_numpy = numpy_before[0] == numpy_after[0] and all(
        np.array_equal(part_before, part_after)
        for part_before, part_after in zip(numpy_before[1:], numpy_after[1:], strict=True)
    )
    return same_numpy and python_before == python_after


def _restore_global_random_states(states: tuple[Any, Any]) -> None:
    """Put NumPy's and Python's global random generators back to *states*."""
    numpy_state, python_state = states
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def _trace_once(trace: Callable[[Any], Any], probe_arguments: _ProbeArguments) -> ProbeResult:
    """One attempt of a probe: *trace* traced at the data leaves of *probe_arguments*."""
    states_before = _global_random_states()
    try:
        _, output = jax.make_jaxpr(
            lambda data_leaves: trace(probe_arguments.rebuild(data_leaves)), return_shape=True
        )(probe_arguments.data_leaves)
    except Exception as error:
        result = ProbeResult(traces=False, error=error.with_traceback(None))
    else:
        result = ProbeResult(traces=True, output=output)
    if not _states_equal(states_before, _global_random_states()):
        _restore_global_random_states(states_before)
        return ProbeResult(traces=False, draws_global_random=True)
    return result


def _warn_global_random(label: str) -> None:
    """Warn that the callable *label* draws from a global random generator."""
    warnings.warn(
        f"{label!r} draws from NumPy's global random generator, so it is treated as opaque; "
        f"its noise would otherwise be fixed into the compiled JAX program. Draw from a key "
        f"instead, or claim traceable=False",
        UserWarning,
        stacklevel=3,
    )


def _captured_value_key(value: Any) -> Hashable:
    """The part of a closure's probe key one captured value contributes.

    An array counts by its shape and dtype, and a hashable value by its value;
    a ``Function`` is hashable by its identity, and its claims are fixed. Any
    other value raises :class:`_Unkeyable`.
    """
    if _is_array_leaf(value) or isinstance(value, jax.core.Tracer):
        return _array_leaf_key(value)
    return _static_leaf_key(value)
