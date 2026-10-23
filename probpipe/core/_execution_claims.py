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
  - :class:`Traceability` – traceable, embeddable, or host-only, ordered from the
    most permissive down.
  - :class:`ProbeResult` and :class:`ProbeResults` – a stored probe outcome, and
    the bounded store of them.
  - :func:`probe` – the one probe, with its rules for Python numbers and for the
    global random generators.
  - :func:`probe_closure` – the probe of a closure, which closures of one code
    share.
  - :func:`spec_stand_in` – the abstract stand-in of one value of a spec.
  - :class:`FalseTraceableClaimError` – the error for a false claim.
"""

from __future__ import annotations

import enum
import hashlib
import random
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import cached_property
from types import CellType, FunctionType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._record_spec import RecordSpec
from ._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ._specs import OutputSpec

__all__ = [
    "NO_CLAIMS",
    "TRACING_ERRORS",
    "ExecutionClaims",
    "FalseTraceableClaimError",
    "ProbeResult",
    "ProbeResults",
    "Traceability",
    "checked_execution_claims",
    "probe",
    "probe_closure",
    "spec_stand_in",
]


# ---------------------------------------------------------------------------
# Claims and traceability
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
        Whether the callable evaluates a stack of evaluation points in one
        call, returning their outputs stacked on a leading axis, with output
        row ``i`` depending only on evaluation point ``i``.
    differentiable : NumericSpec or None
        The differentiability claim: the numeric inputs gradients flow
        through, or ``None`` for no claim.
    """

    traceable: bool | None = None
    vectorized: bool = False
    differentiable: NumericSpec | None = None


#: The claims of a callable that states nothing.
NO_CLAIMS = ExecutionClaims()


def checked_execution_claims(
    *,
    function_name: str,
    traceable: bool | None,
    vectorized: bool,
    differentiable: NumericSpec | None,
) -> ExecutionClaims:
    """The claims *function_name* states, after checking that each has an admitted value.

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
            f"traceable of {function_name!r} must be True, False, or None, got {traceable!r}"
        )
    if not isinstance(vectorized, bool):
        raise TypeError(
            f"vectorized of {function_name!r} must be True or False, got {vectorized!r}"
        )
    if traceable is None and not vectorized and differentiable is None:
        return NO_CLAIMS
    return ExecutionClaims(traceable, vectorized, differentiable)


class Traceability(enum.Enum):
    """How a computation may be run inside a JAX program, from the most permissive down.

    The members are ordered, and :meth:`least_permissive` gives the least permissive of
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
    def least_permissive(cls, *traceabilities: Traceability) -> Traceability:
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


# ---------------------------------------------------------------------------
# The probe and its results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """The stored outcome of one probe.

    Attributes
    ----------
    traces : bool
        Whether JAX traced the callable at the probed input structure.
    ints_static : bool
        Whether it traced only with its integer scalar arguments static.
    abstract_output_leaves : tuple of jax.ShapeDtypeStruct or None
        The leaves of the callable's result as JAX traced it, when it traced,
        and ``None`` otherwise.
    abstract_output_tree : PyTreeDef or None
        The tree structure of that result, which :attr:`abstract_output`
        rebuilds it from.
    error : BaseException or None
        The error the trace raised when it did not trace, without its
        traceback.
    draws_global_random : bool
        Whether the callable drew from NumPy's or Python's global random
        generator while it was traced, which makes it opaque.
    retry_with_static_ints : bool
        Whether this result only records that the form with integer scalars as
        data failed, so the form with them static decides.
    """

    traces: bool
    ints_static: bool = False
    abstract_output_leaves: tuple[jax.ShapeDtypeStruct, ...] | None = None
    abstract_output_tree: Any = None
    error: BaseException | None = None
    draws_global_random: bool = False
    retry_with_static_ints: bool = False

    @property
    def abstract_output(self) -> Any:
        """The callable's result as JAX traced it, a tree of ``jax.ShapeDtypeStruct``, or ``None``.

        It is rebuilt only when read, since a ProbPipe term in the result, such
        as a ``NumericArray``, cannot hold a ``jax.ShapeDtypeStruct``; read it
        only for a result of raw arrays and mappings.
        """
        if self.abstract_output_leaves is None:
            return None
        return jax.tree.unflatten(self.abstract_output_tree, self.abstract_output_leaves)

    @property
    def traceability(self) -> Traceability:
        """``TRACEABLE`` when the callable traced, and ``HOST_ONLY`` otherwise."""
        return Traceability.TRACEABLE if self.traces else Traceability.HOST_ONLY


class ProbeResults:
    """The probe results of one owner, by cache key, bounded and safe for several threads.

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

    def stored_result(self, cache_key: Hashable) -> ProbeResult | None:
        """The result stored under *cache_key*, or ``None``.

        Parameters
        ----------
        cache_key : Hashable
            The input structure, with whatever else the owner's results depend on.

        Returns
        -------
        ProbeResult or None
            The stored result.
        """
        with self._lock:
            stored_result = self._results.get(cache_key)
            if stored_result is not None:
                self._results.move_to_end(cache_key)
            return stored_result

    def store_result(self, cache_key: Hashable, probe_result: ProbeResult) -> None:
        """Store *probe_result* under *cache_key*, dropping the least recently used result past the bound.

        Parameters
        ----------
        cache_key : Hashable
            The input structure, with whatever else the owner's results depend on.
        probe_result : ProbeResult
            The outcome to store.
        """
        with self._lock:
            self._results[cache_key] = probe_result
            self._results.move_to_end(cache_key)
            while len(self._results) > self._max_entries:
                self._results.popitem(last=False)


def probe(
    probe_results: ProbeResults | None,
    traced_function: Callable[[Any], Any],
    arguments: Any,
    *,
    label: str,
    cache_key_prefix: Hashable = (),
) -> ProbeResult:
    """Whether *traced_function* traces at *arguments*, read from *probe_results* or probed and stored there.

    A probe traces *traced_function* once with ``jax.make_jaxpr``, taking every
    numeric array, stand-in, and Python ``float`` of *arguments* as data and
    passing every other leaf as it is, so the result holds for any arguments of
    the same input structure. It follows the probe rules of V.13:

    1. An integer scalar, a Python ``int`` or a NumPy integer, is data first.
       When the probe fails with one as data, it is retried once with every
       integer scalar static, as one probe, and that result is stored under
       the input structure that holds the integers' values.
    2. A callable that draws from NumPy's or Python's global random generator
       while it is traced is recorded as opaque, and a warning names it. The
       generators are put back to their states before the probe.

    The probe runs the callable's Python up to the first operation that fails
    on a tracer. A false ``traceable=True`` claim inside it is not a probe
    outcome, and its error propagates.

    Parameters
    ----------
    probe_results : ProbeResults or None
        Where results are read and stored; ``None`` stores nothing.
    traced_function : Callable
        Called with the arguments, rebuilt with tracers for their data.
    arguments : Any
        The arguments, a tree of arrays, stand-ins, and other values.
    label : str
        The callable's label, which the warning names.
    cache_key_prefix : Hashable
        What else the result depends on besides the input structure, such as
        the route the call runs.

    Returns
    -------
    ProbeResult
        The stored or new outcome.

    Raises
    ------
    FalseTraceableClaimError
        If a ``Function`` that claims ``traceable=True`` fails while the probe
        traces it.
    """
    data_form = _probe_arguments(arguments, ints_static=False)
    data_cache_key = (
        None
        if probe_results is None or data_form.structure is None
        else (cache_key_prefix, data_form.structure)
    )
    stored_result = None if data_cache_key is None else probe_results.stored_result(data_cache_key)
    if stored_result is not None and not stored_result.retry_with_static_ints:
        return stored_result
    if stored_result is None:
        data_result = _trace_once(traced_function, data_form, label)
        if data_result.traces or data_result.draws_global_random or not data_form.has_int_data:
            if data_cache_key is not None:
                probe_results.store_result(data_cache_key, data_result)
            return data_result
        if data_cache_key is not None:
            probe_results.store_result(
                data_cache_key, ProbeResult(traces=False, retry_with_static_ints=True)
            )
    static_form = _probe_arguments(arguments, ints_static=True)
    static_cache_key = (
        None
        if probe_results is None or static_form.structure is None
        else (cache_key_prefix, static_form.structure)
    )
    stored_static_result = (
        None if static_cache_key is None else probe_results.stored_result(static_cache_key)
    )
    if stored_static_result is not None:
        return stored_static_result
    static_attempt = _trace_once(traced_function, static_form, label)
    static_result = ProbeResult(
        traces=static_attempt.traces,
        ints_static=static_attempt.traces,
        abstract_output_leaves=static_attempt.abstract_output_leaves,
        abstract_output_tree=static_attempt.abstract_output_tree,
        error=static_attempt.error,
        draws_global_random=static_attempt.draws_global_random,
    )
    if static_cache_key is not None:
        probe_results.store_result(static_cache_key, static_result)
    return static_result


def probe_closure(
    probe_results: ProbeResults,
    closure: Callable[..., Any],
    call_closure: Callable[[Callable[..., Any], Any], Any],
    arguments: Any,
    *,
    label: str,
    role: str,
) -> ProbeResult:
    """Whether ``call_closure(closure, arguments)`` traces, sharing the result among closures of one code.

    Closures of one code, such as the density of the law a kernel builds anew
    at each given value, share a probe result when what they capture has one
    structure (V.13, probe rule 3):

    1. The probe first traces the closure with each captured array as data, so
       its result, the abstract output included, holds for any captured values
       of the same shapes and dtypes. A captured hashable value counts by its
       value, and a captured ``Function`` by itself.
    2. A closure that fails with its captured arrays as data, because it reads
       one's value, as ``jnp.zeros(int(n))`` does, is probed again as it is, and
       that result is shared only by closures that capture equal arrays.
    3. A closure that captures any other value, a plain callable included, and
       a callable that is no plain function, are probed for themselves, and
       their results are not stored.

    Parameters
    ----------
    probe_results : ProbeResults
        Where results are read and stored.
    closure : Callable
        The closure, such as a law's sampler or density.
    call_closure : Callable
        ``call_closure(callable, arguments)``, the computation the probe traces,
        given the closure or a copy of it with other captured values.
    arguments : Any
        The arguments, as :func:`probe` takes them.
    label : str
        The label the warning of :func:`probe` names.
    role : str
        What the closure is to its owner, such as ``"sample"``, which keeps the
        results of one code in two roles apart.

    Returns
    -------
    ProbeResult
        The stored or new outcome.
    """
    captures = _closure_captures(closure)
    if captures is None:
        return probe(
            None,
            lambda probed_arguments: call_closure(closure, probed_arguments),
            arguments,
            label=label,
        )
    with_captures_as_data = probe(
        probe_results,
        lambda probed_inputs: call_closure(
            captures.closure_with_arrays(probed_inputs[1]), probed_inputs[0]
        ),
        (arguments, captures.captured_arrays),
        label=label,
        cache_key_prefix=("captures as data", role, closure.__code__, captures.static_cache_key),
    )
    if with_captures_as_data.traces or with_captures_as_data.draws_global_random:
        return with_captures_as_data
    value_cache_key = captures.value_cache_key
    return probe(
        None if value_cache_key is None else probe_results,
        lambda probed_arguments: call_closure(closure, probed_arguments),
        arguments,
        label=label,
        cache_key_prefix=(
            "captures as constants",
            role,
            closure.__code__,
            captures.static_cache_key,
            value_cache_key,
        ),
    )


def spec_stand_in(spec: TermSpec | None, *, unbound_size: int | None = None) -> Any:
    """The abstract stand-in of one value of *spec*, or ``None`` when an array leaf has none.

    An array leaf stands in as its shape and dtype, the default float dtype
    when it sets none, and a record as the mapping of its fields' stand-ins. A
    dimension not yet bound takes *unbound_size*, and without one the array has
    no stand-in. A spec of any other kind has none.

    Parameters
    ----------
    spec : TermSpec or None
        The spec of the value.
    unbound_size : int or None
        The size a dimension name not yet bound stands in with, or ``None``
        to give no stand-in for an array with such a dimension.

    Returns
    -------
    jax.ShapeDtypeStruct or dict or None
        The stand-in.
    """
    if isinstance(spec, RecordSpec):
        children = {
            name: spec_stand_in(child, unbound_size=unbound_size)
            for name, child in spec.children.items()
        }
        return None if any(child is None for child in children.values()) else children
    if isinstance(spec, NumericArraySpec) and (not spec.free_dims or unbound_size is not None):
        dtype = spec.dtype if spec.dtype is not None else jnp.result_type(float)
        shape = tuple(size if isinstance(size, int) else unbound_size for size in spec.shape)
        return jax.ShapeDtypeStruct(shape, dtype)
    return None


# ---------------------------------------------------------------------------
# A false claim
# ---------------------------------------------------------------------------


class FalseTraceableClaimError(TypeError):
    """A ``Function`` claims ``traceable=True``, but JAX failed to trace it.

    Use :meth:`for_tracing_error` to build one from JAX's error, which the
    raised error is chained to.
    """

    @classmethod
    def for_tracing_error(
        cls, function_name: str, tracing_error: Exception
    ) -> FalseTraceableClaimError:
        """The error for *function_name*, saying what JAX's *tracing_error* shows the callable did.

        Parameters
        ----------
        function_name : str
            The label of the ``Function`` that makes the claim.
        tracing_error : Exception
            The JAX tracing error, one of :data:`TRACING_ERRORS`.

        Returns
        -------
        FalseTraceableClaimError
            The error, whose message names the claim, what the callable did,
            and JAX's error.
        """
        failure = next(
            (
                description
                for error_class, description in _TRACING_FAILURES
                if isinstance(tracing_error, error_class)
            ),
            "it reads the value of a traced input",
        )
        return cls(
            f"{function_name!r} claims traceable=True, but tracing it failed: {failure} "
            f"({type(tracing_error).__name__}). Remove the claim, or claim traceable=False"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


#: What each JAX tracing error says the callable did, most specific first.
_TRACING_FAILURES: tuple[tuple[type[Exception], str], ...] = (
    (jax.errors.TracerArrayConversionError, "it converts a traced value to NumPy"),
    (jax.errors.TracerBoolConversionError, "it branches in Python on a traced value"),
    (jax.errors.TracerIntegerConversionError, "it converts a traced value to a Python int"),
    (jax.errors.NonConcreteBooleanIndexError, "its output shape depends on a traced value"),
    (jax.errors.ConcretizationTypeError, "it reads the value of a traced input"),
)

#: The JAX errors that mean a callable cannot be traced.
TRACING_ERRORS: tuple[type[Exception], ...] = tuple(
    error_class for error_class, _ in _TRACING_FAILURES
)


def _is_numeric_dtype(dtype: Any) -> bool:
    """Whether *dtype* is a number or a PRNG key, which a probe takes as data."""
    return jnp.issubdtype(dtype, jax.dtypes.prng_key) or np.dtype(dtype).kind in "biufc"


def _is_array_leaf(leaf: Any) -> bool:
    """Whether *leaf* is an array a probe takes as data, a stand-in of one included."""
    if isinstance(leaf, jax.Array | np.ndarray | np.generic | jax.ShapeDtypeStruct):
        return _is_numeric_dtype(leaf.dtype)
    return False


def _is_integer_scalar(leaf: Any) -> bool:
    """Whether *leaf* is a Python ``int``, a ``bool`` included, or a NumPy integer scalar."""
    return isinstance(leaf, int | np.integer | np.bool_)


def _array_leaf_cache_key(leaf: Any) -> tuple[Any, ...]:
    """The part of an input structure an array leaf gives: whether it is traced, its shape, and its dtype."""
    origin = "tracer" if isinstance(leaf, jax.core.Tracer) else "array"
    return (
        origin,
        tuple(leaf.shape),
        str(leaf.dtype),
        bool(getattr(leaf, "weak_type", False)),
    )


class _NoCacheKey(Exception):
    """A value has no part in a cache key: it is neither an array nor hashable."""


def _static_leaf_cache_key(leaf: Any) -> tuple[Any, ...]:
    """The part of an input structure a leaf passed as it is gives: its type and value."""
    try:
        hash(leaf)
    except TypeError:
        raise _NoCacheKey from None
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
    rebuild_arguments : Callable
        Rebuilds the arguments from values of the data leaves.
    has_int_data : bool
        Whether an integer scalar is among the data leaves.
    """

    structure: Hashable | None
    data_leaves: tuple[Any, ...]
    rebuild_arguments: Callable[[tuple[Any, ...]], Any]
    has_int_data: bool


def _probe_arguments(arguments: Any, *, ints_static: bool) -> _ProbeArguments:
    """*arguments* split for a probe, with integer scalars as data or as static values.

    A numeric array, a stand-in of one, and a Python ``float`` or ``complex``
    are data, a Python number a weakly typed scalar. An integer scalar is data
    unless *ints_static*. Every other leaf is passed as it is, and enters the
    input structure by its value.
    """
    leaves, treedef = jax.tree.flatten(arguments)
    data_positions: list[int] = []
    leaf_cache_keys: list[tuple[Any, ...]] = []
    structure_known = True
    has_int_data = False
    for position, leaf in enumerate(leaves):
        if _is_integer_scalar(leaf):
            if ints_static:
                leaf_cache_keys.append(_static_leaf_cache_key(leaf))
                continue
            data_positions.append(position)
            leaf_cache_keys.append(("integer", type(leaf)))
            has_int_data = True
        elif _is_array_leaf(leaf):
            data_positions.append(position)
            leaf_cache_keys.append(_array_leaf_cache_key(leaf))
        elif isinstance(leaf, float | complex):
            data_positions.append(position)
            leaf_cache_keys.append(("scalar", type(leaf)))
        else:
            try:
                leaf_cache_keys.append(_static_leaf_cache_key(leaf))
            except _NoCacheKey:
                structure_known = False

    def rebuild_arguments(data_values: tuple[Any, ...]) -> Any:
        rebuilt_leaves = list(leaves)
        for position, data_value in zip(data_positions, data_values, strict=True):
            rebuilt_leaves[position] = data_value
        return jax.tree.unflatten(treedef, rebuilt_leaves)

    return _ProbeArguments(
        structure=(treedef, tuple(leaf_cache_keys)) if structure_known else None,
        data_leaves=tuple(leaves[position] for position in data_positions),
        rebuild_arguments=rebuild_arguments,
        has_int_data=has_int_data,
    )


def _global_random_states() -> tuple[Any, Any]:
    """The states of NumPy's and Python's global random generators."""
    return np.random.get_state(), random.getstate()


def _random_states_equal(states_before: tuple[Any, Any], states_after: tuple[Any, Any]) -> bool:
    """Whether two snapshots of the global random states are the same."""
    numpy_state_before, python_state_before = states_before
    numpy_state_after, python_state_after = states_after
    same_numpy_state = numpy_state_before[0] == numpy_state_after[0] and all(
        np.array_equal(part_before, part_after)
        for part_before, part_after in zip(
            numpy_state_before[1:], numpy_state_after[1:], strict=True
        )
    )
    return same_numpy_state and python_state_before == python_state_after


def _restore_global_random_states(states: tuple[Any, Any]) -> None:
    """Put NumPy's and Python's global random generators back to *states*."""
    numpy_state, python_state = states
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def _trace_once(
    traced_function: Callable[[Any], Any], probe_arguments: _ProbeArguments, label: str
) -> ProbeResult:
    """One attempt of a probe: *traced_function* traced at the data leaves of *probe_arguments*.

    The global random generators are put back to their states before the
    attempt, so another thread that draws from them while it runs loses those
    draws; a probe runs only while a call is being planned.
    """
    states_before = _global_random_states()
    output_trees: list[Any] = []

    def traced_output_leaves(data_leaves: tuple[Any, ...]) -> list[Any]:
        # The output is returned as its leaves, and its tree kept here, since
        # JAX would otherwise rebuild the tree with abstract leaves, which a
        # ProbPipe term in it cannot hold.
        output_leaves, output_tree = jax.tree.flatten(
            traced_function(probe_arguments.rebuild_arguments(data_leaves))
        )
        output_trees.append(output_tree)
        return output_leaves

    try:
        closed_jaxpr = jax.make_jaxpr(traced_output_leaves)(probe_arguments.data_leaves)
    except FalseTraceableClaimError:
        raise
    except Exception as error:
        attempt = ProbeResult(traces=False, error=error.with_traceback(None))
    else:
        attempt = ProbeResult(
            traces=True,
            abstract_output_leaves=tuple(
                jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
                for aval in closed_jaxpr.out_avals
            ),
            abstract_output_tree=output_trees[-1],
        )
    if not _random_states_equal(states_before, _global_random_states()):
        _restore_global_random_states(states_before)
        warnings.warn(
            f"{label!r} draws from NumPy's or Python's global random generator, so ProbPipe "
            f"does not compile it with JAX. Draw from a JAX key instead, or pass "
            f"traceable=False",
            UserWarning,
            stacklevel=4,
        )
        return ProbeResult(traces=False, draws_global_random=True)
    return attempt


@dataclass(frozen=True)
class _ClosureCaptures:
    """What a plain function captures, split into the arrays a probe can trace and the rest.

    Attributes
    ----------
    captured_arrays : tuple
        The captured arrays, in the order of the closure's cells.
    static_cache_key : Hashable
        Every captured value but the arrays, by value, with each array's place
        and shape and dtype.
    closure_with_arrays : Callable
        A copy of the closure that captures other arrays in their places.
    """

    captured_arrays: tuple[Any, ...]
    static_cache_key: Hashable
    closure_with_arrays: Callable[[tuple[Any, ...]], Callable[..., Any]]

    @cached_property
    def value_cache_key(self) -> Hashable | None:
        """The captured arrays by their contents, or ``None`` when one is traced.

        Hashing copies every captured array to the host, so it runs only for a
        closure whose probe with its captured arrays as data failed.
        """
        return _array_contents_cache_key(self.captured_arrays)


def _closure_captures(closure: Callable[..., Any]) -> _ClosureCaptures | None:
    """What *closure* captures, or ``None`` when it shares no probe result with another closure.

    A closure shares nothing when it is no plain function, or when it captures
    a value that is neither an array, a ``Function``, nor another hashable
    value that is not callable.
    """
    if not isinstance(closure, FunctionType):
        return None
    cell_values = [cell.cell_contents for cell in closure.__closure__ or ()]
    defaults = (*(closure.__defaults__ or ()), *(closure.__kwdefaults__ or {}).values())
    array_positions: list[int] = []
    static_parts: list[Hashable] = []
    try:
        for position, cell_value in enumerate(cell_values):
            if _is_array_leaf(cell_value) or isinstance(cell_value, jax.core.Tracer):
                array_positions.append(position)
                static_parts.append(_array_leaf_cache_key(cell_value))
            else:
                static_parts.append(_captured_static_cache_key(cell_value))
        static_parts.extend(_captured_static_cache_key(default) for default in defaults)
    except (_NoCacheKey, ValueError):
        return None
    captured_arrays = tuple(cell_values[position] for position in array_positions)

    def closure_with_arrays(replacement_arrays: tuple[Any, ...]) -> Callable[..., Any]:
        rebuilt_values = list(cell_values)
        for position, replacement in zip(array_positions, replacement_arrays, strict=True):
            rebuilt_values[position] = replacement
        rebuilt = FunctionType(
            closure.__code__,
            closure.__globals__,
            closure.__name__,
            closure.__defaults__,
            tuple(CellType(rebuilt_value) for rebuilt_value in rebuilt_values),
        )
        rebuilt.__kwdefaults__ = closure.__kwdefaults__
        return rebuilt

    return _ClosureCaptures(
        captured_arrays=captured_arrays,
        static_cache_key=tuple(static_parts),
        closure_with_arrays=closure_with_arrays,
    )


def _captured_static_cache_key(captured_value: Any) -> Hashable:
    """The part of a closure's cache key a captured value that is not an array gives.

    A ``Function`` counts by itself, since its claims are fixed; any other
    hashable value counts by its value. Another callable, whose behaviour is
    not its value, and an unhashable value raise :class:`_NoCacheKey`, so the
    closure shares nothing.
    """
    if callable(captured_value) and getattr(captured_value, "_execution_claims", None) is None:
        raise _NoCacheKey
    return _static_leaf_cache_key(captured_value)


def _array_contents_cache_key(arrays: tuple[Any, ...]) -> Hashable | None:
    """*arrays* by their shapes, dtypes, and contents, or ``None`` when one is traced."""
    if any(isinstance(array, jax.core.Tracer) for array in arrays):
        return None
    contents_cache_key = []
    for array in arrays:
        if jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jax.random.key_data(array)
        host_array = np.asarray(array)
        digest = hashlib.blake2b(np.ascontiguousarray(host_array).tobytes(), digest_size=16)
        contents_cache_key.append((host_array.shape, str(host_array.dtype), digest.hexdigest()))
    return tuple(contents_cache_key)


def _check_output_float_width(output_spec: OutputSpec | None, origin: str) -> None:
    """Refuse an output declared float64 while JAX has 64-bit floats disabled."""
    if jax.config.x64_enabled or output_spec is None:
        return
    from ._batch import BatchSpec

    def declares_float64(spec: TermSpec | None) -> bool:
        if isinstance(spec, NumericArraySpec):
            return spec.dtype is not None and spec.dtype.kind == "f" and spec.dtype.itemsize == 8
        if isinstance(spec, RecordSpec):
            return any(declares_float64(child) for child in spec.children.values())
        if isinstance(spec, BatchSpec):
            return declares_float64(spec.element_spec.term_spec)
        return False

    if declares_float64(output_spec.term_spec):
        raise ValueError(
            f"{origin!r} declares a float64 output while 64-bit floats are off. "
            "Set jax_enable_x64=True before calling it, or declare a float32 output"
        )
