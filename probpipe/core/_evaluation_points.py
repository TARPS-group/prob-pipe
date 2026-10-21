"""Sets of evaluation points, the chunks an executor runs, and their outcomes (V.9).

An *evaluation point* is one evaluation of one callable at one set of
arguments. The evaluation points of one callable that one request asks for,
such as 200 evaluations of a simulator at 200 parameter values, form one
:class:`EvaluationPoints`. An executor runs it by its :class:`EvaluationChunk`
objects, contiguous slices that can be pickled, and returns one
:class:`EvaluationOutcomes`: each evaluation point's output, or the
:class:`EvaluationFailure` that says why it has none.

The module lives in ``core`` because every package asks for evaluations. The
workflow scope of each evaluation point, which gives a draw made inside it its
random key, belongs to the function engine, which installs it here when it is
imported (:func:`_install_evaluation_point_scopes`).

Provides:
  - :class:`EvaluationPoints` – a set of evaluation points of one callable.
  - :class:`EvaluationChunk` – a contiguous slice of one, which pickles.
  - :class:`SharedPayload` – the callable and the arguments every evaluation
    point shares.
  - :class:`EvaluationOutcomes` – the outcomes of a chunk or of a whole set.
  - :class:`EvaluationFailure` – why one evaluation point has no output.
  - :class:`EvaluationError` – the error a call raises for failed evaluation
    points.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np

from ._array_backend import _is_numeric_leaf
from ._call_context import (
    _CallContext,
    _collecting_call_accumulators,
    _EvaluationLog,
    _NumSamplesUsed,
    _under_call_context,
    current_call_context,
)
from ._dispatch import Feasibility
from ._execution_claims import NO_CLAIMS, ExecutionClaims
from ._record_spec import RecordSpec
from ._spec_base import NumericArraySpec, OpaqueSpec, TermSpec
from ._specs import OutputSpec

__all__ = [
    "EvaluationChunk",
    "EvaluationError",
    "EvaluationFailure",
    "EvaluationOutcomes",
    "EvaluationPoints",
    "SharedPayload",
]

#: Why an evaluation point has no output: its code raised, its output failed
#: the check against the output spec, the infrastructure lost it, or it ran
#: out of time.
FailureKind = Literal["error", "invalid", "lost", "timeout"]

#: How many failed evaluation points an ``EvaluationError`` message lists.
_LISTED_FAILURES = 5


class EvaluationPoints:
    """The evaluation points of one callable that one request asks for.

    An evaluation point is one evaluation of the callable at one set of
    arguments. A set of evaluation points holds the callable, the arguments
    every evaluation point shares, and the arguments that vary, stacked on a
    leading axis, so it describes its evaluation points without building
    each one. An executor runs it by its chunks, and never reads or changes
    the arguments, so the executor decides only how the evaluation points
    run, never what they compute.

    A set of evaluation points is built by the code that asks for the
    evaluations, such as a lifted call, under the call that runs it: each
    evaluation point then runs under that call's settings, and a ProbPipe
    draw made inside it takes its random key from its place in the workflow,
    whatever executor runs it.

    Parameters
    ----------
    fn : callable
        The callable, which each evaluation point calls with its arguments by
        keyword.
    varying : Mapping of str to Any
        Each argument that differs between evaluation points, its values
        stacked on a leading axis: every array leaf of one argument has one
        row per evaluation point.
    shared : Mapping of str to Any, optional
        Each argument that every evaluation point receives unchanged.
    origin : str
        The name of the callable or of the site that asks for the evaluations,
        which messages and reports use.
    output_spec : OutputSpec or None, optional
        The callable's output spec, which each output is checked against.
    claims : ExecutionClaims, optional
        The callable's execution claims; none by default.
    traceable : bool or None, optional
        Whether JAX can trace the callable for these arguments, as its claim
        or its stored probe result says; ``None`` when that is unknown, and
        the set then runs as opaque code.
    point_noun : str, optional
        The word for one evaluation point in messages, such as ``"draw"`` or
        ``"grid point"``.

    Raises
    ------
    ValueError
        If the varying arguments do not all have the same number of rows.

    Examples
    --------
    >>> import numpy as np
    >>> points = EvaluationPoints(
    ...     lambda theta, scale: scale * np.sum(theta),
    ...     varying={"theta": np.arange(6.0).reshape(3, 2)},
    ...     shared={"scale": 2.0},
    ...     origin="total",
    ... )
    >>> len(points), [(chunk.start, chunk.stop) for chunk in points.chunks(chunk_size=2)]
    (3, [(0, 2), (2, 3)])
    """

    origin: str
    claims: ExecutionClaims
    output_spec: OutputSpec | None

    def __init__(
        self,
        fn: Callable[..., Any],
        /,
        *,
        varying: Mapping[str, Any],
        shared: Mapping[str, Any] | None = None,
        origin: str,
        output_spec: OutputSpec | None = None,
        claims: ExecutionClaims = NO_CLAIMS,
        traceable: bool | None = None,
        point_noun: str = "evaluation point",
    ) -> None:
        self.origin = origin
        self.claims = claims
        self.output_spec = output_spec
        self.point_noun = point_noun
        self._traceable = claims.traceable if claims.traceable is not None else traceable
        self._varying = MappingProxyType(dict(varying))
        self._num_evaluation_points = _num_rows(self._varying, origin)
        self._shared_payload = SharedPayload(fn, MappingProxyType(dict(shared or {})))
        self._call_context = current_call_context()
        evaluation_log = self._call_context.evaluation_log
        self._set_ordinal = 0 if evaluation_log is None else evaluation_log.next_set_ordinal(origin)
        self._point_scopes = _EVALUATION_POINT_SCOPES_FACTORY(self._num_evaluation_points)
        self._shipping_feasibility: Feasibility | None = None

    def __len__(self) -> int:
        """The number of evaluation points."""
        return self._num_evaluation_points

    def __repr__(self) -> str:
        return (
            f"EvaluationPoints(origin={self.origin!r}, "
            f"num_evaluation_points={self._num_evaluation_points})"
        )

    @property
    def traceable(self) -> bool:
        """Whether JAX can trace the callable for these evaluation points; ``False`` when unknown."""
        return self._traceable is True

    def chunks(
        self, *, chunk_size: int | None = None, num_chunks: int | None = None
    ) -> list[EvaluationChunk]:
        """Split the evaluation points into contiguous chunks, in order.

        Parameters
        ----------
        chunk_size : int, optional
            The number of evaluation points in each chunk; the last chunk may
            hold fewer.
        num_chunks : int, optional
            The number of chunks, as equal in size as possible; at most one
            per evaluation point.

        Returns
        -------
        list of EvaluationChunk
            The chunks, which together hold every evaluation point once, in
            order. Without *chunk_size* and *num_chunks*, one chunk holds them
            all.

        Raises
        ------
        ValueError
            If both *chunk_size* and *num_chunks* are given, or either is not a
            positive integer.
        """
        if chunk_size is not None and num_chunks is not None:
            raise ValueError(
                "cannot split evaluation points by both chunk_size and num_chunks. Give one of them"
            )
        num_points = self._num_evaluation_points
        if num_points == 0:
            return []
        if chunk_size is not None:
            _checked_positive_integer(chunk_size, "chunk_size")
            starts = range(0, num_points, chunk_size)
            bounds = [(start, min(start + chunk_size, num_points)) for start in starts]
        else:
            if num_chunks is not None:
                _checked_positive_integer(num_chunks, "num_chunks")
            chunk_count = min(num_chunks or 1, num_points)
            base_size, remainder = divmod(num_points, chunk_count)
            bounds = []
            start = 0
            for chunk_index in range(chunk_count):
                stop = start + base_size + (1 if chunk_index < remainder else 0)
                bounds.append((start, stop))
                start = stop
        return [EvaluationChunk(self, start, stop) for start, stop in bounds]

    def shared_payload(self) -> SharedPayload:
        """The callable and the arguments every evaluation point shares.

        An executor sends it once to each worker that runs chunks of this set,
        and passes it to :meth:`EvaluationChunk.run`; an executor that runs
        chunks in the calling process passes it without copying.

        Returns
        -------
        SharedPayload
            The same object on every call.
        """
        return self._shared_payload

    def shipping_check(self) -> Feasibility:
        """Whether the chunks and the shared payload can be pickled, for another process.

        The check pickles the callable, the shared arguments, the call's
        settings, and the first evaluation point's arguments, and runs
        nothing. Its outcome is computed once.

        Returns
        -------
        Feasibility
            Feasible when everything pickles; otherwise infeasible, with the
            reason in its description.
        """
        if self._shipping_feasibility is None:
            self._shipping_feasibility = self._compute_shipping_feasibility()
        return self._shipping_feasibility

    def _compute_shipping_feasibility(self) -> Feasibility:
        """The outcome :meth:`shipping_check` computes once."""
        first_arguments = self._arguments_in(0, min(1, self._num_evaluation_points))
        shipped_parts = {
            "its callable": self._shared_payload.fn,
            "its shared arguments": dict(self._shared_payload.shared_arguments),
            "its varying arguments": first_arguments,
            "its call's settings": self._call_context.evaluation_context(),
        }
        for part_name, part in shipped_parts.items():
            try:
                cloudpickle.dumps(part)
            except Exception as pickling_error:
                return Feasibility(
                    feasible=False,
                    description=(
                        f"{part_name} cannot be pickled "
                        f"({type(pickling_error).__name__}: {pickling_error})"
                    ),
                    actionable=True,
                )
        return Feasibility(feasible=True)

    def _arguments_in(self, start: int, stop: int) -> dict[str, Any]:
        """The varying arguments of the evaluation points ``start`` to ``stop``, stacked."""
        return {
            argument_name: jax.tree.map(lambda leaf: leaf[start:stop], stacked_values)
            for argument_name, stacked_values in self._varying.items()
        }

    def _arguments_at(self, point_index: int) -> dict[str, Any]:
        """The varying arguments of the evaluation point *point_index*."""
        return {
            argument_name: jax.tree.map(lambda leaf: leaf[point_index], stacked_values)
            for argument_name, stacked_values in self._varying.items()
        }

    def _point_identity(self, point_index: int) -> tuple[str, int, int]:
        """The identity of evaluation point *point_index* within its call."""
        return (self.origin, self._set_ordinal, point_index)


@dataclass(frozen=True)
class SharedPayload:
    """The callable of a set of evaluation points and the arguments every evaluation point shares.

    An executor treats it as opaque: it sends it to its workers and passes it
    to :meth:`EvaluationChunk.run`.

    Attributes
    ----------
    fn : callable
        The callable.
    shared_arguments : Mapping of str to Any
        The arguments every evaluation point receives unchanged.
    """

    fn: Callable[..., Any]
    shared_arguments: Mapping[str, Any]

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the shared arguments as a plain dict, read-only again when unpickled."""
        return (_unpickled_shared_payload, (self.fn, dict(self.shared_arguments)))


class EvaluationChunk:
    """A contiguous slice of a set of evaluation points, which an executor runs as one unit.

    A chunk pickles, so an executor may run it in another process or on
    another machine; a pickled chunk carries only its own evaluation points'
    arguments and the settings of the call that asked for them. Running it
    does the same wherever it runs.

    Attributes
    ----------
    start : int
        The point index of its first evaluation point.
    stop : int
        One past the point index of its last evaluation point.
    """

    def __init__(self, evaluation_points: EvaluationPoints, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop
        self._origin = evaluation_points.origin
        self._set_ordinal = evaluation_points._set_ordinal
        self._point_noun = evaluation_points.point_noun
        self._output_spec = evaluation_points.output_spec
        self._vectorized = evaluation_points.claims.vectorized
        self._evaluation_context = evaluation_points._call_context.evaluation_context()
        self._point_scopes = evaluation_points._point_scopes
        self._arguments_source: EvaluationPoints | Mapping[str, Any] = evaluation_points

    def __len__(self) -> int:
        """The number of evaluation points in the chunk."""
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"EvaluationChunk(origin={self._origin!r}, start={self.start}, stop={self.stop})"

    def run(self, shared_payload: SharedPayload) -> EvaluationOutcomes:
        """Run every evaluation point of the chunk, in order, and return their outcomes.

        Each evaluation point runs under the settings of the call that asked
        for it, and a ProbPipe draw made inside it takes its random key from
        the evaluation point's place in the workflow. An exception its code
        raises, an ``Exception`` and never a ``KeyboardInterrupt``, is that
        evaluation point's failure, and the chunk goes on with the next one.
        Each output is checked against the callable's output spec. A callable
        that claims ``vectorized`` is called once for the whole chunk, and if
        that call raises, every evaluation point of the chunk fails with its
        error.

        Parameters
        ----------
        shared_payload : SharedPayload
            The set's :meth:`EvaluationPoints.shared_payload`.

        Returns
        -------
        EvaluationOutcomes
            One outcome per evaluation point of the chunk, in order.
        """
        chunk_context = _context_with_fresh_accumulators(self._evaluation_context)
        with _under_call_context(chunk_context):
            if self._vectorized:
                point_results = self._run_vectorized(shared_payload)
            else:
                point_results = [
                    self._run_point(shared_payload, point_index)
                    for point_index in range(self.start, self.stop)
                ]
        outputs: list[Any] = []
        failures: dict[int, EvaluationFailure] = {}
        lossy_casts: set[tuple[str, str]] = set()
        for point_index, point_result in zip(range(self.start, self.stop), point_results):
            if isinstance(point_result, EvaluationFailure):
                failures[point_index] = point_result
                outputs.append(None)
            else:
                checked_output, point_lossy_casts = point_result
                outputs.append(checked_output)
                lossy_casts |= point_lossy_casts
        return EvaluationOutcomes(
            start=self.start,
            outputs=tuple(outputs),
            failures=MappingProxyType(failures),
            _num_samples_used=chunk_context.num_samples_used,
            _evaluation_log=chunk_context.evaluation_log,
            _claim_reports=tuple(self._point_scopes.claim_reports()),
            _lossy_casts=frozenset(lossy_casts),
        )

    def _run_point(
        self, shared_payload: SharedPayload, point_index: int
    ) -> tuple[Any, frozenset[tuple[str, str]]] | EvaluationFailure:
        """The checked output of one evaluation point, or its failure.

        The evaluation point notes into accumulators of its own, which merge
        into the chunk's only when its output passes the check.
        """
        try:
            with (
                _collecting_call_accumulators("always"),
                self._point_scopes.point_scope(point_index),
            ):
                output = shared_payload.fn(
                    **shared_payload.shared_arguments, **self._arguments_at(point_index)
                )
                return _checked_point_output(
                    output, self._output_spec, self._output_path(point_index)
                )
        except _InvalidPointOutput as invalid_output:
            return self._failure(point_index, "invalid", invalid_output.check_error)
        except Exception as error:
            return self._failure(point_index, "error", error)

    def _run_vectorized(
        self, shared_payload: SharedPayload
    ) -> list[tuple[Any, frozenset[tuple[str, str]]] | EvaluationFailure]:
        """The checked outputs of the chunk's evaluation points, from one call on the stacked arguments."""
        num_rows = len(self)
        try:
            with _collecting_call_accumulators("always"), self._point_scopes.vectorized_scope():
                stacked_output = shared_payload.fn(
                    **_stacked_shared_arguments(shared_payload.shared_arguments, num_rows),
                    **self._arguments_in_chunk(),
                )
                output_rows = _output_rows(stacked_output, num_rows, self._origin)
        except Exception as error:
            return [
                self._failure(point_index, "error", error)
                for point_index in range(self.start, self.stop)
            ]
        point_results: list[tuple[Any, frozenset[tuple[str, str]]] | EvaluationFailure] = []
        for point_index, output_row in zip(range(self.start, self.stop), output_rows):
            try:
                point_results.append(
                    _checked_point_output(
                        output_row, self._output_spec, self._output_path(point_index)
                    )
                )
            except _InvalidPointOutput as invalid_output:
                point_results.append(
                    self._failure(point_index, "invalid", invalid_output.check_error)
                )
        return point_results

    def _arguments_at(self, point_index: int) -> dict[str, Any]:
        """The varying arguments of evaluation point *point_index*."""
        if isinstance(self._arguments_source, EvaluationPoints):
            return self._arguments_source._arguments_at(point_index)
        row = point_index - self.start
        return {
            argument_name: jax.tree.map(lambda leaf: leaf[row], stacked_values)
            for argument_name, stacked_values in self._arguments_source.items()
        }

    def _arguments_in_chunk(self) -> dict[str, Any]:
        """The varying arguments of every evaluation point of the chunk, stacked."""
        if isinstance(self._arguments_source, EvaluationPoints):
            return self._arguments_source._arguments_in(self.start, self.stop)
        return dict(self._arguments_source)

    def _output_path(self, point_index: int) -> str:
        """How a message names the output of evaluation point *point_index*."""
        return f"the output of {self._origin!r} at {self._point_noun} {point_index}"

    def _failure(self, point_index: int, kind: FailureKind, error: Exception) -> EvaluationFailure:
        """The failure of evaluation point *point_index*, of *kind*, caused by *error*."""
        return EvaluationFailure(
            point_index=point_index,
            point_identity=(self._origin, self._set_ordinal, point_index),
            kind=kind,
            error_type=type(error).__qualname__,
            message=str(error),
            error=error,
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the chunk with its own evaluation points' arguments, to run in another process.

        Pickling reserves the chunk's evaluation points with the call that
        asked for them, so the draws made inside them in the other process
        take the keys they take in this one, and the keys they claim come back
        with the outcomes.
        """
        chunk_state = {
            "start": self.start,
            "stop": self.stop,
            "origin": self._origin,
            "set_ordinal": self._set_ordinal,
            "point_noun": self._point_noun,
            "output_spec": self._output_spec,
            "vectorized": self._vectorized,
            "evaluation_context": self._evaluation_context,
            "point_scopes": self._point_scopes.shipped(range(self.start, self.stop)),
            "varying_arguments": self._arguments_in_chunk(),
        }
        return (_unpickled_chunk, (chunk_state,))


@dataclass(frozen=True)
class EvaluationFailure:
    """Why one evaluation point has no output.

    Attributes
    ----------
    point_index : int
        Its position within its set of evaluation points.
    point_identity : tuple of (str, int, int)
        Its unique name within the call: the callable, which of the call's
        sets of evaluation points of that callable it belongs to, counting
        from 0, and its point index.
    kind : {"error", "invalid", "lost", "timeout"}
        ``"error"`` when its code raised; ``"invalid"`` when its output did
        not match the output spec; ``"lost"`` when the infrastructure lost
        it; ``"timeout"`` when it ran out of time.
    error_type : str
        The class name of the exception, which survives sending it between
        processes.
    message : str
        The exception's message.
    error : Exception or None
        The exception itself; ``None`` when it could not be sent back from the
        process that ran the evaluation point.
    """

    point_index: int
    point_identity: tuple[str, int, int]
    kind: FailureKind
    error_type: str
    message: str
    error: Exception | None

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the failure, leaving out an exception that does not pickle."""
        error = self.error
        if error is not None:
            try:
                cloudpickle.dumps(error)
            except Exception:
                error = None
        return (
            EvaluationFailure,
            (
                self.point_index,
                self.point_identity,
                self.kind,
                self.error_type,
                self.message,
                error,
            ),
        )


@dataclass(frozen=True)
class EvaluationOutcomes:
    """The outcomes of a run of contiguous evaluation points: an output or a failure for each.

    A chunk's run returns one, and an executor merges its chunks' outcomes
    into one for the whole set, with :meth:`merge`.

    Attributes
    ----------
    start : int
        The point index of the first evaluation point covered.
    outputs : tuple
        Each evaluation point's output, in order; ``None`` where one failed.
    failures : Mapping of int to EvaluationFailure
        The failure of each evaluation point that has no output, by point
        index.
    """

    start: int
    outputs: tuple[Any, ...]
    failures: Mapping[int, EvaluationFailure]
    _num_samples_used: _NumSamplesUsed | None = field(default=None, repr=False, compare=False)
    _evaluation_log: _EvaluationLog | None = field(default=None, repr=False, compare=False)
    _claim_reports: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    _lossy_casts: frozenset[tuple[str, str]] = field(default=frozenset(), repr=False, compare=False)

    @classmethod
    def lost(cls, chunk: EvaluationChunk, reason: str) -> EvaluationOutcomes:
        """The outcomes of a chunk whose every evaluation point the infrastructure lost.

        Parameters
        ----------
        chunk : EvaluationChunk
            The chunk that did not finish, such as a cluster job that died.
        reason : str
            Why, for the failures' messages.

        Returns
        -------
        EvaluationOutcomes
            A ``"lost"`` failure for each evaluation point of *chunk*.
        """
        failures = {
            point_index: EvaluationFailure(
                point_index=point_index,
                point_identity=(chunk._origin, chunk._set_ordinal, point_index),
                kind="lost",
                error_type="",
                message=reason,
                error=None,
            )
            for point_index in range(chunk.start, chunk.stop)
        }
        return cls(
            start=chunk.start,
            outputs=(None,) * len(chunk),
            failures=MappingProxyType(failures),
        )

    @classmethod
    def merge(cls, chunk_outcomes: Iterable[EvaluationOutcomes]) -> EvaluationOutcomes:
        """The outcomes of several runs as one, in the order of the evaluation points.

        Parameters
        ----------
        chunk_outcomes : iterable of EvaluationOutcomes
            The outcomes of contiguous runs that together cover a range of
            evaluation points once, in any order.

        Returns
        -------
        EvaluationOutcomes
            The outcomes sorted by ``start`` and joined.

        Raises
        ------
        ValueError
            If the runs overlap or leave a gap.
        """
        ordered_outcomes = sorted(chunk_outcomes, key=lambda outcomes: outcomes.start)
        if not ordered_outcomes:
            return cls(start=0, outputs=(), failures=MappingProxyType({}))
        next_start = ordered_outcomes[0].start
        for outcomes in ordered_outcomes:
            if outcomes.start != next_start:
                raise ValueError(
                    f"cannot merge outcomes that start at {outcomes.start} after outcomes that "
                    f"end at {next_start}. Merge the outcomes of contiguous chunks that cover "
                    "each evaluation point once"
                )
            next_start = outcomes.start + len(outcomes.outputs)
        merged_num_samples_used = _NumSamplesUsed()
        merged_evaluation_log = _EvaluationLog()
        for outcomes in ordered_outcomes:
            if outcomes._num_samples_used is not None:
                merged_num_samples_used.merge_from(outcomes._num_samples_used)
            if outcomes._evaluation_log is not None:
                merged_evaluation_log.merge_from(outcomes._evaluation_log)
        return cls(
            start=ordered_outcomes[0].start,
            outputs=tuple(output for outcomes in ordered_outcomes for output in outcomes.outputs),
            failures=MappingProxyType(
                {
                    point_index: failure
                    for outcomes in ordered_outcomes
                    for point_index, failure in outcomes.failures.items()
                }
            ),
            _num_samples_used=merged_num_samples_used,
            _evaluation_log=merged_evaluation_log,
            _claim_reports=tuple(
                report for outcomes in ordered_outcomes for report in outcomes._claim_reports
            ),
            _lossy_casts=frozenset().union(
                *(outcomes._lossy_casts for outcomes in ordered_outcomes)
            ),
        )


class EvaluationError(RuntimeError):
    """Evaluation points of a call failed.

    The call ran every evaluation point and kept the outputs that succeeded,
    then raised. The error is chained to the first failure's exception when it
    is available, so the traceback points into the code that raised.

    Attributes
    ----------
    failures : tuple of EvaluationFailure
        Every failed evaluation point, in order.
    outputs : tuple
        Each evaluation point's output, in order, ``None`` where one failed,
        kept for diagnosis.

    Examples
    --------
    A simulator that fails at one of 200 draws raises::

        EvaluationError: evaluation of 'forward' failed at 1 of 200 evaluation points:
        draw 17: CalledProcessError: Command '...' returned non-zero exit status 3.
    """

    def __init__(
        self,
        failures: Iterable[EvaluationFailure],
        outputs: Iterable[Any],
        *,
        origin: str,
        point_noun: str = "evaluation point",
    ) -> None:
        self.failures = tuple(sorted(failures, key=lambda failure: failure.point_index))
        self.outputs = tuple(outputs)
        listed_failures = "; ".join(
            f"{point_noun} {failure.point_index}: "
            + (
                f"{failure.error_type}: {failure.message}"
                if failure.error_type
                else failure.message
            )
            for failure in self.failures[:_LISTED_FAILURES]
        )
        unlisted_count = len(self.failures) - _LISTED_FAILURES
        if unlisted_count > 0:
            listed_failures += f"; and {unlisted_count} more"
        super().__init__(
            f"evaluation of {origin!r} failed at {len(self.failures)} of {len(self.outputs)} "
            f"evaluation points: {listed_failures}"
        )


# ---------------------------------------------------------------------------
# The workflow scopes of evaluation points, which the function engine installs
# ---------------------------------------------------------------------------


class _EvaluationPointScopes(Protocol):
    """The workflow scope of each evaluation point of one set, which gives a draw inside it its key.

    The function engine implements it and installs a factory of it with
    :func:`_install_evaluation_point_scopes`. One instance belongs to one set
    of evaluation points.
    """

    def point_scope(self, point_index: int) -> AbstractContextManager[None]:
        """The scope evaluation point *point_index* runs in."""
        ...

    def vectorized_scope(self) -> AbstractContextManager[None]:
        """The scope one call on stacked evaluation points runs in, which refuses workflow draws."""
        ...

    def mapped_scope(self) -> AbstractContextManager[None]:
        """The scope a ``jax.vmap`` over the evaluation points runs in, which refuses workflow draws."""
        ...

    def shipped(self, point_indices: range) -> _EvaluationPointScopes:
        """Scopes for *point_indices* that pickle, reserved for a run in another process."""
        ...

    def claim_reports(self) -> Iterable[Any]:
        """The records of the draws the evaluation points made, from a run in another process."""
        ...

    def accept_claim_reports(self, claim_reports: Iterable[Any]) -> None:
        """Add the draws that runs in other processes made to the call's record of draws."""
        ...

    def close(self) -> None:
        """Release what the evaluation points still hold, once every chunk has run or failed."""
        ...


class _UnscopedEvaluationPoints:
    """The scopes of evaluation points that run outside every workflow: none of them installs anything."""

    def point_scope(self, point_index: int) -> AbstractContextManager[None]:
        return nullcontext()

    def vectorized_scope(self) -> AbstractContextManager[None]:
        return nullcontext()

    def mapped_scope(self) -> AbstractContextManager[None]:
        return nullcontext()

    def shipped(self, point_indices: range) -> _UnscopedEvaluationPoints:
        return self

    def claim_reports(self) -> tuple[Any, ...]:
        return ()

    def accept_claim_reports(self, claim_reports: Iterable[Any]) -> None:
        return None

    def close(self) -> None:
        return None


def _unscoped_evaluation_points(num_evaluation_points: int) -> _EvaluationPointScopes:
    """The scopes of a set of evaluation points when no function engine installed its own."""
    return _UnscopedEvaluationPoints()


#: Builds the workflow scopes of a new set of evaluation points from its size.
_EVALUATION_POINT_SCOPES_FACTORY: Callable[[int], _EvaluationPointScopes] = (
    _unscoped_evaluation_points
)


def _install_evaluation_point_scopes(factory: Callable[[int], _EvaluationPointScopes]) -> None:
    """Make *factory* build the workflow scopes of every new set of evaluation points.

    The function engine calls this when it is imported, as it installs its
    call engine, so ``core`` never imports the workflow scope.
    """
    global _EVALUATION_POINT_SCOPES_FACTORY
    _EVALUATION_POINT_SCOPES_FACTORY = factory


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _InvalidPointOutput(Exception):
    """An evaluation point's output does not match its output spec.

    Raised inside the evaluation point's scope, so its accumulators are
    dropped, and caught as an ``"invalid"`` failure.
    """

    def __init__(self, check_error: Exception) -> None:
        super().__init__(str(check_error))
        self.check_error = check_error


def _checked_point_output(
    output: Any, output_spec: OutputSpec | None, output_path: str
) -> tuple[Any, frozenset[tuple[str, str]]]:
    """*output* checked against *output_spec*, with the casts to declared dtypes that lose precision.

    A mapping returned for a record output is laid out as a ``Record`` in the
    declared order, and an array whose dtype differs from a declared one is
    cast to it.

    Raises
    ------
    _InvalidPointOutput
        If *output* does not conform to the output spec's type.
    """
    if output_spec is None or output_spec.term_spec is None:
        return output, frozenset()
    term_spec = output_spec.term_spec
    try:
        term_spec._bind_dims_from_value(output, {}, output_path)
    except ValueError as check_error:
        raise _InvalidPointOutput(check_error) from check_error
    lossy_casts: set[tuple[str, str]] = set()
    return _cast_to_declared(output, term_spec, lossy_casts), frozenset(lossy_casts)


def _cast_to_declared(output: Any, term_spec: TermSpec, lossy_casts: set[tuple[str, str]]) -> Any:
    """*output*, which conforms to *term_spec*, laid out and cast as *term_spec* declares.

    Each cast that loses precision adds its pair of dtype names to *lossy_casts*.
    """
    from .record import Record

    if isinstance(term_spec, RecordSpec):
        children = output.children if isinstance(output, Record) else output
        laid_out_fields = {
            child_name: _cast_to_declared(children[child_name], child_spec, lossy_casts)
            for child_name, child_spec in term_spec.children.items()
        }
        if isinstance(output, Record) and all(
            laid_out_fields[child_name] is children[child_name] for child_name in laid_out_fields
        ):
            return output
        return Record(laid_out_fields)
    if not isinstance(term_spec, NumericArraySpec) or term_spec.dtype is None:
        return output
    actual_dtype = np.result_type(output)
    declared_dtype = np.dtype(term_spec.dtype)
    if actual_dtype == declared_dtype:
        return output
    if not np.can_cast(actual_dtype, declared_dtype, casting="safe"):
        lossy_casts.add((str(actual_dtype), str(declared_dtype)))
    if isinstance(output, jax.Array):
        return jnp.asarray(output, dtype=declared_dtype)
    return np.asarray(output).astype(declared_dtype)


def _holds_only_values(term_spec: TermSpec | None) -> bool:
    """Whether *term_spec* describes values: arrays, opaque values, or records of them.

    A spec that holds the type of a law, a kernel, a function, or a batch
    anywhere does not.
    """
    if term_spec is None or isinstance(term_spec, (NumericArraySpec, OpaqueSpec)):
        return True
    if isinstance(term_spec, RecordSpec):
        return all(_holds_only_values(child_spec) for child_spec in term_spec.children.values())
    return False


def _context_with_fresh_accumulators(evaluation_context: _CallContext) -> _CallContext:
    """*evaluation_context* with new accumulators that collect calls when its call's do."""
    enclosing_num_samples_used = evaluation_context.num_samples_used
    fresh_context = evaluation_context.with_fresh_call_accumulators()
    if enclosing_num_samples_used is not None and fresh_context.num_samples_used is not None:
        fresh_context.num_samples_used.collects_calls = enclosing_num_samples_used.collects_calls
    return fresh_context


def _num_rows(varying: Mapping[str, Any], origin: str) -> int:
    """The number of rows every array leaf of *varying* has on its leading axis."""
    row_counts = {
        np.shape(leaf)[0] if np.ndim(leaf) > 0 else None
        for stacked_values in varying.values()
        for leaf in jax.tree.leaves(stacked_values)
    }
    if not row_counts:
        raise ValueError(
            f"cannot build evaluation points of {origin!r} without varying arguments. "
            "Give each argument that differs between evaluation points, stacked on a "
            "leading axis"
        )
    if None in row_counts or len(row_counts) != 1:
        raise ValueError(
            f"the varying arguments of {origin!r} have different numbers of rows, "
            f"{sorted(count for count in row_counts if count is not None)}. Stack every "
            "varying argument with one row per evaluation point"
        )
    return int(next(iter(row_counts)))


def _stacked_shared_arguments(shared_arguments: Mapping[str, Any], num_rows: int) -> dict[str, Any]:
    """Each numeric leaf of *shared_arguments* as a read-only view repeated over *num_rows* rows.

    A leaf that is not numeric, such as a string, is passed once, unchanged.
    """

    def stacked_leaf(leaf: Any) -> Any:
        if not _is_numeric_leaf(leaf):
            return leaf
        host_array = np.asarray(leaf)
        return np.broadcast_to(host_array, (num_rows, *host_array.shape))

    return {
        argument_name: jax.tree.map(stacked_leaf, argument_value)
        for argument_name, argument_value in shared_arguments.items()
    }


def _output_rows(stacked_output: Any, num_rows: int, origin: str) -> list[Any]:
    """The *num_rows* rows of a vectorized callable's stacked output."""
    leaf_row_counts = {
        np.shape(leaf)[0] if np.ndim(leaf) > 0 else None for leaf in jax.tree.leaves(stacked_output)
    }
    if leaf_row_counts != {num_rows}:
        raise ValueError(
            f"{origin!r} claims vectorized=True but returned an output whose leading axes "
            f"have {sorted(str(count) for count in leaf_row_counts)} rows for {num_rows} "
            "evaluation points. Return the outputs stacked with one row per evaluation point"
        )
    return [
        jax.tree.map(lambda leaf, row_index=row: leaf[row_index], stacked_output)
        for row in range(num_rows)
    ]


def _checked_positive_integer(count: Any, count_name: str) -> None:
    """Raise if *count* is not a positive integer."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{count_name} must be a positive integer, got {count!r}")


def _unpickled_shared_payload(
    fn: Callable[..., Any], shared_arguments: dict[str, Any]
) -> SharedPayload:
    """The shared payload that :meth:`SharedPayload.__reduce__` pickled."""
    return SharedPayload(fn, MappingProxyType(shared_arguments))


def _unpickled_chunk(chunk_state: Mapping[str, Any]) -> EvaluationChunk:
    """The chunk that :meth:`EvaluationChunk.__reduce__` pickled, in the process that unpickles it."""
    chunk = EvaluationChunk.__new__(EvaluationChunk)
    chunk.start = chunk_state["start"]
    chunk.stop = chunk_state["stop"]
    chunk._origin = chunk_state["origin"]
    chunk._set_ordinal = chunk_state["set_ordinal"]
    chunk._point_noun = chunk_state["point_noun"]
    chunk._output_spec = chunk_state["output_spec"]
    chunk._vectorized = chunk_state["vectorized"]
    chunk._evaluation_context = chunk_state["evaluation_context"]
    chunk._point_scopes = chunk_state["point_scopes"]
    chunk._arguments_source = chunk_state["varying_arguments"]
    return chunk


def _warn_lossy_casts(
    lossy_casts: Iterable[tuple[str, str]], *, origin: str, evaluation_log: _EvaluationLog | None
) -> None:
    """Warn, once per call, about each cast to a declared dtype that lost precision."""
    for actual_dtype, declared_dtype in sorted(lossy_casts):
        warning_key = ("lossy cast", origin, actual_dtype, declared_dtype)
        if evaluation_log is not None and not evaluation_log.is_first_warning(warning_key):
            continue
        warnings.warn(
            f"{origin!r} returned {actual_dtype} outputs where its output spec declares "
            f"{declared_dtype}, so they were cast and lost precision. Return {declared_dtype}, "
            "or declare the dtype the callable returns",
            UserWarning,
            stacklevel=2,
        )
