"""Tests for sets of evaluation points, their chunks, and their outcomes."""

import pickle
import threading

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EvaluationError,
    EvaluationFailure,
    EvaluationOutcomes,
    EvaluationPoints,
    NumericArraySpec,
    OutputSpec,
    Record,
    RecordSpec,
)
from probpipe.core._execution_claims import ExecutionClaims


def _points(fn, num_points=5, **options):
    """A set of *num_points* evaluation points of *fn*, varying ``x`` over rows of ``[i, i + 1]``."""
    return EvaluationPoints(
        fn,
        varying={"x": np.arange(2.0 * num_points).reshape(num_points, 2)},
        shared={"scale": 3.0},
        origin="fn",
        **options,
    )


def _run(evaluation_points, **chunk_options):
    """The outcomes of every chunk of *evaluation_points*, merged."""
    shared_payload = evaluation_points.shared_payload()
    return EvaluationOutcomes.merge(
        chunk.run(shared_payload) for chunk in evaluation_points.chunks(**chunk_options)
    )


def _scaled_sum(x, scale):
    return scale * np.sum(x)


class TestBuildingASet:
    def test_its_length_is_the_number_of_rows_of_its_varying_arguments(self):
        assert len(_points(_scaled_sum, num_points=7)) == 7

    def test_varying_arguments_with_different_numbers_of_rows_are_refused(self):
        with pytest.raises(ValueError, match="different numbers of rows"):
            EvaluationPoints(
                _scaled_sum,
                varying={"x": np.zeros((3, 2)), "y": np.zeros(4)},
                origin="fn",
            )

    def test_a_set_without_varying_arguments_is_refused(self):
        with pytest.raises(ValueError, match="without varying arguments"):
            EvaluationPoints(_scaled_sum, varying={}, origin="fn")

    def test_the_shared_payload_is_one_object_holding_the_callable_and_the_shared_arguments(self):
        evaluation_points = _points(_scaled_sum)
        shared_payload = evaluation_points.shared_payload()
        assert shared_payload is evaluation_points.shared_payload()
        assert shared_payload.fn is _scaled_sum
        assert dict(shared_payload.shared_arguments) == {"scale": 3.0}

    def test_a_claim_of_traceability_wins_over_a_probe_result(self):
        claimed_opaque = _points(
            _scaled_sum, claims=ExecutionClaims(traceable=False), traceable=True
        )
        assert claimed_opaque.traceable is False
        assert _points(_scaled_sum, traceable=True).traceable is True
        assert _points(_scaled_sum).traceable is False


class TestChunks:
    def test_by_default_one_chunk_holds_every_evaluation_point(self):
        assert [(chunk.start, chunk.stop) for chunk in _points(_scaled_sum).chunks()] == [(0, 5)]

    def test_a_chunk_size_cuts_contiguous_chunks_in_order(self):
        chunks = _points(_scaled_sum).chunks(chunk_size=2)
        assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 2), (2, 4), (4, 5)]

    def test_a_number_of_chunks_cuts_chunks_as_equal_as_possible(self):
        chunks = _points(_scaled_sum, num_points=7).chunks(num_chunks=3)
        assert [len(chunk) for chunk in chunks] == [3, 2, 2]

    def test_there_are_never_more_chunks_than_evaluation_points(self):
        assert len(_points(_scaled_sum, num_points=2).chunks(num_chunks=8)) == 2

    def test_a_chunk_size_and_a_number_of_chunks_together_are_refused(self):
        with pytest.raises(ValueError, match="both chunk_size and num_chunks"):
            _points(_scaled_sum).chunks(chunk_size=2, num_chunks=2)

    @pytest.mark.parametrize("bad_count", [0, -1, 1.5, True])
    def test_a_count_that_is_not_a_positive_integer_is_refused(self, bad_count):
        with pytest.raises(ValueError, match="must be a positive integer"):
            _points(_scaled_sum).chunks(chunk_size=bad_count)

    def test_running_the_chunks_gives_each_output_in_point_order(self):
        outcomes = _run(_points(_scaled_sum), chunk_size=2)
        assert outcomes.start == 0
        np.testing.assert_array_equal(outcomes.outputs, [3.0, 15.0, 27.0, 39.0, 51.0])
        assert dict(outcomes.failures) == {}

    def test_a_pickled_chunk_carries_only_its_own_arguments_and_runs_the_same(self):
        evaluation_points = _points(_scaled_sum, num_points=40)
        chunk = evaluation_points.chunks(chunk_size=4)[3]
        unpickled_chunk = cloudpickle.loads(cloudpickle.dumps(chunk))
        assert unpickled_chunk._arguments_source["x"].shape == (4, 2)
        shared_payload = pickle.loads(pickle.dumps(evaluation_points.shared_payload()))
        assert unpickled_chunk.run(shared_payload).outputs == chunk.run(shared_payload).outputs


class TestFailuresInAChunk:
    def test_an_evaluation_point_that_raises_fails_alone_and_the_chunk_goes_on(self):
        def fail_at_two(x, scale):
            if x[0] == 2.0:
                raise ValueError("bad point")
            return scale * np.sum(x)

        outcomes = _run(_points(fail_at_two))
        assert set(outcomes.failures) == {1}
        failure = outcomes.failures[1]
        assert failure.kind == "error"
        assert failure.error_type == "ValueError"
        assert failure.message == "bad point"
        assert failure.point_identity == ("fn", 0, 1)
        assert outcomes.outputs[1] is None
        assert outcomes.outputs[2] == 27.0

    def test_a_chunk_catches_exceptions_but_not_keyboard_interrupts(self):
        def interrupted(x, scale):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _run(_points(interrupted))

    def test_lost_outcomes_fail_every_evaluation_point_of_the_chunk(self):
        chunk = _points(_scaled_sum).chunks(chunk_size=2)[1]
        outcomes = EvaluationOutcomes.lost(chunk, "the job was killed")
        assert outcomes.start == 2
        assert outcomes.outputs == (None, None)
        assert {failure.kind for failure in outcomes.failures.values()} == {"lost"}
        assert outcomes.failures[3].message == "the job was killed"

    def test_a_failure_whose_exception_does_not_pickle_arrives_without_it(self):
        class HoldsALock(Exception):
            def __init__(self):
                super().__init__("holds a lock")
                self.lock = threading.Lock()

        failure = EvaluationFailure(
            0, ("fn", 0, 0), "error", "HoldsALock", "holds a lock", HoldsALock()
        )
        restored = cloudpickle.loads(cloudpickle.dumps(failure))
        assert restored.error is None
        assert restored.message == "holds a lock"


class TestMergingOutcomes:
    def test_outcomes_merge_in_point_order_whatever_order_they_arrive_in(self):
        chunks = _points(_scaled_sum).chunks(chunk_size=2)
        shared_payload = _points(_scaled_sum).shared_payload()
        merged = EvaluationOutcomes.merge(reversed([chunk.run(shared_payload) for chunk in chunks]))
        np.testing.assert_array_equal(merged.outputs, [3.0, 15.0, 27.0, 39.0, 51.0])

    def test_outcomes_that_leave_a_gap_are_refused(self):
        chunks = _points(_scaled_sum).chunks(chunk_size=2)
        shared_payload = _points(_scaled_sum).shared_payload()
        with pytest.raises(ValueError, match="cannot merge outcomes that start at 4"):
            EvaluationOutcomes.merge([chunks[0].run(shared_payload), chunks[2].run(shared_payload)])


class TestTheOutputCheck:
    def test_an_output_of_the_wrong_shape_fails_its_evaluation_point_only(self):
        def wrong_at_three(x, scale):
            return np.zeros(3) if x[0] == 6.0 else scale * x

        outcomes = _run(
            _points(wrong_at_three, output_spec=OutputSpec(y=NumericArraySpec((2,), "float64")))
        )
        assert set(outcomes.failures) == {3}
        assert outcomes.failures[3].kind == "invalid"
        assert "the output of 'fn' at evaluation point 3" in outcomes.failures[3].message

    def test_a_mapping_returned_for_a_record_is_laid_out_in_the_declared_order(self):
        def unordered_mapping(x, scale):
            return {"b": np.sum(x), "a": scale * x[0]}

        output_spec = OutputSpec(RecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(())))
        outcomes = _run(_points(unordered_mapping, output_spec=output_spec))
        first_output = outcomes.outputs[0]
        assert isinstance(first_output, Record)
        assert first_output.keys() == ("a", "b")

    def test_an_output_of_another_dtype_is_cast_to_the_declared_one(self):
        def float64_output(x, scale):
            return np.asarray(scale * x, dtype=np.float64)

        outcomes = _run(
            _points(float64_output, output_spec=OutputSpec(y=NumericArraySpec((2,), "float32")))
        )
        assert np.asarray(outcomes.outputs[0]).dtype == np.float32
        assert outcomes._lossy_casts == frozenset({("float64", "float32")})

    def test_without_an_output_spec_outputs_are_not_checked(self):
        outcomes = _run(_points(lambda x, scale: "any value"))
        assert outcomes.outputs == ("any value",) * 5


class TestVectorizedCallables:
    def test_a_vectorized_callable_is_called_once_per_chunk_with_every_argument_stacked(self):
        received_shapes = []

        def whole_chunk(x, scale):
            received_shapes.append((np.shape(x), np.shape(scale)))
            return np.asarray(scale)[:, np.newaxis] * x

        outcomes = _run(
            _points(whole_chunk, claims=ExecutionClaims(traceable=False, vectorized=True)),
            chunk_size=2,
        )
        assert received_shapes == [((2, 2), (2,)), ((2, 2), (2,)), ((1, 2), (1,))]
        np.testing.assert_array_equal(outcomes.outputs[4], [24.0, 27.0])

    def test_a_shared_argument_arrives_as_a_read_only_view(self):
        writeable_flags = []

        def whole_chunk(x, scale):
            writeable_flags.append(scale.flags.writeable)
            return x

        _run(_points(whole_chunk, claims=ExecutionClaims(traceable=False, vectorized=True)))
        assert writeable_flags == [False]

    def test_when_a_vectorized_call_raises_every_evaluation_point_of_the_chunk_fails(self):
        def raises(x, scale):
            raise RuntimeError("solver crashed")

        outcomes = _run(
            _points(raises, claims=ExecutionClaims(traceable=False, vectorized=True)), chunk_size=3
        )
        assert set(outcomes.failures) == {0, 1, 2, 3, 4}
        assert {failure.message for failure in outcomes.failures.values()} == {"solver crashed"}

    def test_a_vectorized_output_with_the_wrong_number_of_rows_fails_the_chunk(self):
        outcomes = _run(
            _points(
                lambda x, scale: jnp.zeros((1, 2)),
                claims=ExecutionClaims(traceable=False, vectorized=True),
            )
        )
        assert len(outcomes.failures) == 5
        assert "one row per evaluation point" in outcomes.failures[0].message


class TestEvaluationError:
    def _failures(self, point_indices):
        return [
            EvaluationFailure(
                index, ("forward", 0, index), "error", "CalledProcessError", "exit status 3", None
            )
            for index in point_indices
        ]

    def test_its_message_names_the_callable_the_count_and_each_failure_in_the_requesters_words(
        self,
    ):
        error = EvaluationError(
            self._failures([17]), [None] * 200, origin="forward", point_noun="draw"
        )
        assert str(error) == (
            "evaluation of 'forward' failed at 1 of 200 evaluation points: "
            "draw 17: CalledProcessError: exit status 3"
        )

    def test_its_message_lists_five_failures_and_counts_the_rest(self):
        error = EvaluationError(self._failures(range(8)), [None] * 8, origin="forward")
        assert str(error).count("CalledProcessError") == 5
        assert str(error).endswith("; and 3 more")

    def test_it_keeps_the_failures_in_order_and_is_a_runtime_error(self):
        error = EvaluationError(self._failures([4, 1]), [None] * 5, origin="forward")
        assert isinstance(error, RuntimeError)
        assert [failure.point_index for failure in error.failures] == [1, 4]


class TestShippingCheck:
    def test_a_set_whose_callable_pickles_can_be_shipped(self):
        assert _points(_scaled_sum).shipping_check().feasible is True

    def test_a_set_whose_callable_holds_a_lock_cannot_be_shipped_and_says_why(self):
        lock = threading.Lock()

        def holds_a_lock(x, scale):
            with lock:
                return scale * x

        feasibility = _points(holds_a_lock).shipping_check()
        assert feasibility.feasible is False
        assert feasibility.description.startswith("its callable cannot be pickled")
