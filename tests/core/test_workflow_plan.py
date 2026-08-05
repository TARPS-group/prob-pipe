"""Tests for Function broadcast planning."""

from __future__ import annotations

import inspect
from typing import Any

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import DistributionArray, Normal, NumericRecord, NumericRecordBatch
from probpipe.core import _workflow_call
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._workflow_distribution_normalization import (
    normalize_distribution_values,
)
from probpipe.core._workflow_plan import ArrayBroadcastGroup, build_broadcast_plan
from probpipe.core.distribution import Distribution
from probpipe.core.protocols import SupportsSampling


def _numeric_record_array(field: str, values: range) -> NumericRecordBatch:
    return NumericRecordBatch.stack(
        [NumericRecord("nr", **{field: float(value)}) for value in values], level_name="draw"
    )


def _ref(name: str) -> _workflow_call.WorkflowInputRef:
    return _workflow_call.WorkflowInputRef(name)


def _plan(values, hints=None):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(
        signature,
        hints=hints,
    )
    return build_broadcast_plan(values=values, signature_info=signature_info)


class TestBroadcastRegime:
    def test_plain_values_do_not_broadcast(self):
        plan = _plan({"x": 1.0})

        assert plan.regime == "none"
        assert plan.dist_args == ()
        assert plan.array_args == ()
        assert plan.array_groups == ()
        assert plan.sweep_batch_shape == ()
        assert plan.n_sweep == 1

    def test_distribution_value_selects_distribution_regime(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        plan = _plan({"x": dist})

        assert plan.regime == "distribution"
        assert plan.dist_args == (_ref("x"),)
        assert plan.array_args == ()

    def test_record_array_value_selects_sweep_regime(self):
        values = {"p": _numeric_record_array("x", range(4))}

        plan = _plan(values)

        assert plan.regime == "sweep"
        assert plan.dist_args == ()
        assert plan.array_args == (_ref("p"),)
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("p"),),
                batch_shape=(4,),
                size=4,
                level_names=("p",),
                axis_groups=((4,),),
            ),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_array_and_distribution_values_select_nested_regime(self):
        values = {
            "p": _numeric_record_array("x", range(4)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }

        plan = _plan(values)

        assert plan.regime == "nested"
        assert plan.dist_args == (_ref("noise"),)
        assert plan.array_args == (_ref("p"),)


class TestHintClassification:
    def test_distribution_hints_skip_scalar_distribution_broadcast(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        concrete = _plan({"x": dist}, {"x": Distribution})
        protocol = _plan({"x": dist}, {"x": SupportsSampling})

        assert concrete.regime == "none"
        assert protocol.regime == "none"

    def test_array_hints_skip_array_sweep(self):
        ra = _numeric_record_array("x", range(4))
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2,),
            loc=jnp.asarray([0.0, 1.0]),
            scale=1.0,
            name="d",
        )

        record_plan = _plan({"p": ra}, {"p": NumericRecordBatch})
        dist_plan = _plan({"d": da}, {"d": DistributionArray})
        any_plan = _plan({"p": ra}, {"p": Any})

        assert record_plan.regime == "none"
        assert dist_plan.regime == "none"
        assert any_plan.regime == "none"


class TestArrayGrouping:
    def test_sibling_views_zip_into_one_group(self):
        ra = NumericRecordBatch.stack(
            [NumericRecord("nr", x=float(i), y=float(2 * i)) for i in range(4)], level_name="draw"
        )

        plan = _plan({"x": ra.view("x"), "y": ra.view("y")})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("x"), _ref("y"))
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("x"), _ref("y")),
                batch_shape=(4,),
                size=4,
                level_names=("x",),
                axis_groups=((4,),),
            ),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_views_from_different_parents_use_product_shape(self):
        ra_a = _numeric_record_array("a", range(3))
        ra_b = _numeric_record_array("b", range(2))

        plan = _plan({"a": ra_a.view("a"), "b": ra_b.view("b")})

        assert plan.regime == "sweep"
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("a"),),
                batch_shape=(3,),
                size=3,
                level_names=("a",),
                axis_groups=((3,),),
            ),
            ArrayBroadcastGroup(
                arg_refs=(_ref("b"),),
                batch_shape=(2,),
                size=2,
                level_names=("b",),
                axis_groups=((2,),),
            ),
        )
        assert plan.sweep_batch_shape == (3, 2)
        assert plan.n_sweep == 6

    def test_distribution_array_uses_sweep_group(self):
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="d",
        )

        plan = _plan({"d": da})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("d"),)
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("d"),),
                batch_shape=(2, 3),
                size=6,
                level_names=("d",),
                axis_groups=((2, 3),),
            ),
        )
        assert plan.sweep_batch_shape == (2, 3)
        assert plan.n_sweep == 6


class TestPlanPurity:
    def test_planner_does_not_convert_or_mutate_external_distributions(self):
        external = tfd.Normal(loc=0.0, scale=1.0)
        values = {"x": external}

        signature = inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
        signature_info = _workflow_call.make_signature_info_from_signature(signature)
        raw_plan = build_broadcast_plan(
            values=values,
            signature_info=signature_info,
        )
        normalized = normalize_distribution_values(
            values=values,
            signature_info=signature_info,
        )
        normalized_plan = build_broadcast_plan(
            values=normalized,
            signature_info=signature_info,
        )

        assert values["x"] is external
        assert raw_plan.regime == "none"
        assert normalized_plan.regime == "distribution"


def _batch(level: str = "draw", n: int = 3, **fields) -> Any:
    from probpipe.core._numeric_record_batch import NumericRecordBatch
    from probpipe.core.event_template import EventTemplate

    fields = fields or {"x": jnp.arange(float(n))}
    return NumericRecordBatch(
        dict(fields),
        (level,),
        element_spec=EventTemplate({name: value.shape[1:] for name, value in fields.items()}),
    )


class TestBatchGrouping:
    """A batch has no parent pointer, so grouping follows its level names."""

    def test_sibling_select_all_views_zip_into_one_group(self):
        batch = _batch(x=jnp.arange(3.0), y=jnp.arange(3.0) * 10)
        views = batch.select_all()

        plan = _plan({"x": views["x"], "y": views["y"]})

        assert plan.regime == "sweep"
        assert len(plan.array_groups) == 1
        assert plan.array_groups[0].arg_refs == (_ref("x"), _ref("y"))
        assert plan.sweep_batch_shape == (3,)
        assert plan.n_sweep == 3

    def test_a_batch_zips_with_its_own_view(self):
        batch = _batch(x=jnp.arange(3.0), y=jnp.arange(3.0) * 10)

        plan = _plan({"whole": batch, "x": batch.select("x")["x"]})

        assert len(plan.array_groups) == 1
        assert plan.n_sweep == 3

    def test_batches_with_no_level_in_common_form_a_product(self):
        plan = _plan({"a": _batch("outer", 3), "b": _batch("inner", 2)})

        assert len(plan.array_groups) == 2
        assert plan.sweep_batch_shape == (3, 2)
        assert plan.n_sweep == 6

    def test_one_level_name_at_two_sizes_is_refused(self):
        """Two batches naming the same level claim to range over the same thing,
        so disagreeing about its size is a mistake rather than a product."""
        import pytest

        with pytest.raises(ValueError, match="batched differently"):
            _plan({"a": _batch("draw", 3), "b": _batch("draw", 2)})


class TestAnnotationDispatch:
    """The hint says what the body accepts whole, so the value answers it."""

    def test_the_exact_annotation_skips_the_sweep(self):
        from probpipe.core._numeric_record_batch import NumericRecordBatch

        batch = _batch()
        plan = _plan({"p": batch}, hints={"p": NumericRecordBatch})

        assert plan.regime == "none"

    def test_the_other_family_member_does_not_skip_it(self):
        batch = _batch()
        plan = _plan({"p": batch}, hints={"p": NumericRecordArray})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("p"),)

    def test_a_record_array_annotated_as_a_batch_is_swept(self):
        from probpipe.core._record_batch import RecordBatch

        ra = _numeric_record_array("x", range(3))
        plan = _plan({"p": ra}, hints={"p": RecordBatch})

        assert plan.regime == "sweep"


class TestPartialLevelOverlap:
    def test_a_shared_level_across_different_level_sets_is_refused(self):
        """Aligning one shared level across differently-leveled operands is not
        built; a product would read the shared name as two unrelated axes and
        mint the same level twice."""
        from probpipe.core.event_template import EventTemplate

        two = NumericRecordBatch(
            {"x": jnp.arange(6.0).reshape(2, 3)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            axis_groups=((2,), (3,)),
        )
        one = _batch("draw", 3)

        with pytest.raises(ValueError, match="share the level 'draw'"):
            _plan({"a": two, "b": one})

    def test_a_levelless_operand_does_not_own_its_parameter_name_as_a_level(self):
        """An operand with no levels of its own cannot share one.

        Its multiplicity is anonymous, so it aligns with nothing by name and
        products with everything. Standing the parameter's name in for the levels
        it does not have collides with a real level of that name on another
        operand — and refuses a call whose two axes are simply independent, on
        the strength of a level neither operand disagrees about.

        A ``DistributionArray`` is the levelless operand that outlives the
        cutover: it is swept by its ``batch_shape`` without being a ``Batch``, so
        it carries no level names of its own either before or after the batch
        types stop being records.
        """
        levelless = DistributionArray.from_batched_params(
            Normal, batch_shape=(2,), loc=jnp.asarray([0.0, 1.0]), scale=1.0, name="d"
        )
        batch = _batch("draw", 3)

        plan = _plan({"draw": levelless, "other": batch})

        assert plan.regime == "sweep"
        # Two independent multiplicities: a product, not a zip.
        assert len(plan.array_groups) == 2

    def test_the_same_levels_at_different_geometries_are_refused(self):
        """The flat shape can agree while the partition does not; zipping would
        hand the output whichever partition arrived first."""
        from probpipe.core.event_template import EventTemplate

        ga = NumericRecordBatch(
            {"x": jnp.zeros((2, 3, 4))},
            ("a", "b"),
            element_spec=EventTemplate(x=()),
            axis_groups=((2,), (3, 4)),
        )
        gb = NumericRecordBatch(
            {"y": jnp.zeros((2, 3, 4))},
            ("a", "b"),
            element_spec=EventTemplate(y=()),
            axis_groups=((2, 3), (4,)),
        )

        with pytest.raises(ValueError, match="same levels but are batched differently"):
            _plan({"a": ga, "b": gb})


class TestBatchAnnotationsSuppressTheSweep:
    def test_the_abstract_batch_annotation_takes_the_value_whole(self):
        from probpipe.core._batch import Batch

        plan = _plan({"p": _batch()}, hints={"p": Batch})

        assert plan.regime == "none"

    def test_a_generic_alias_answers_by_its_origin(self):
        from probpipe.core._batch import Batch

        plan = _plan({"p": _batch()}, hints={"p": Batch[dict]})

        assert plan.regime == "none"


class TestOptionalBatchAnnotations:
    def test_an_optional_container_annotation_still_takes_the_value_whole(self):
        """``Batch | None`` names the container as surely as ``Batch``: the value
        answers whichever arm it satisfies, and ``None`` answers none."""
        from typing import Optional

        from probpipe.core._batch import Batch
        from probpipe.core._record_batch import RecordBatch

        batch = _batch()
        hints = (Batch | None, RecordBatch | None, Optional[Batch], Batch[dict] | None)  # noqa: UP045
        for hint in hints:
            plan = _plan({"p": batch}, hints={"p": hint})

            assert plan.regime == "none", hint

    def test_a_non_container_union_still_sweeps(self):
        from probpipe import Record

        plan = _plan({"p": _batch()}, hints={"p": Record | None})

        assert plan.regime == "sweep"
