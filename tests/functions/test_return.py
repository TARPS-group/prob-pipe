"""Return, step 8 of the stack (design V.10).

Return validates the produced terms against the completed declaration and
wraps a raw host into the kind its spec names. It labels the result by
``output_label`` and gives it provenance, or returns it detached under
``raw=True``. A result that violates its declaration raises ResultKindError or
ResultSchemaError, which are return-contract defects.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    Batch,
    BatchSpec,
    Distribution,
    DistributionSpec,
    Function,
    FunctionSpec,
    Gamma,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    Opaque,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
    ResultKindError,
    ResultSchemaError,
    workflow_run,
)
from probpipe.core.constraints import positive

from ._design_helpers import error_of, standard_normal

SCALAR = NumericArraySpec(())


class TestTheKindDirectedWrap:
    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (jnp.ones(2), NumericArray),
            ({"a": 1.0}, Record),
            (lambda x: x, Function),
            ("text", Opaque),
        ],
        ids=["array", "mapping", "callable", "other"],
    )
    def test_a_raw_return_is_wrapped_into_its_kind(self, value, kind):
        result = Function(
            lambda: value,
            label="produce",
        )()

        assert isinstance(result, kind)

    @pytest.mark.parametrize(
        "value", [[1.0, 2.0], (1.0, 2.0), {1.0, 2.0}], ids=["list", "tuple", "set"]
    )
    def test_a_returned_collection_is_opaque(self, value):
        assert isinstance(
            Function(
                lambda: value,
                label="produce",
            )(),
            Opaque,
        )

    @pytest.mark.parametrize("value", [2.0, jnp.ones(2), lambda x: x])
    def test_an_explicit_opaque_declaration_accepts_raw_numeric_values_and_callables(self, value):
        wrapped = Function(
            lambda: value,
            output_spec=OutputSpec(produce=OpaqueSpec()),
            label="produce",
        )

        assert isinstance(wrapped(), Opaque)
        assert wrapped.apply() is value

    def test_a_declared_batch_still_accepts_a_raw_sequence(self):
        wrapped = Function(
            lambda: [1.0, 2.0],
            output_spec=OutputSpec(produce=BatchSpec(SCALAR, row=2)),
            label="produce",
        )

        result = wrapped()

        assert isinstance(result, NumericArrayBatch)
        assert result.level_names == ("row",)
        np.testing.assert_array_equal(result.values, [1.0, 2.0])

    def test_an_array_under_one_component_stays_an_array(self):
        wrapped = Function(
            lambda: jnp.ones(2),
            output_spec=OutputSpec(beta=None),
            label="f",
        )

        assert isinstance(wrapped(), NumericArray)

    def test_a_one_field_record_stays_a_record(self):
        wrapped = Function(
            lambda: {"beta": jnp.ones(2)},
            output_spec=RecordSpec(beta=NumericArraySpec((2,))),
            label="f",
        )

        assert isinstance(wrapped(), Record)

    def test_a_tracked_return_keeps_its_kind_under_a_fresh_identity(self):
        stored = NumericArray(
            jnp.ones(2),
            label="stored",
        )
        result = Function(
            lambda: stored,
            output_label="loaded",
            label="load",
        )()

        assert isinstance(result, NumericArray)
        assert result is not stored
        assert result.label == "loaded" and stored.label == "stored"
        assert result.value is stored.value


class TestLabelAndProvenance:
    def test_a_result_is_labeled_by_output_name(self):
        wrapped = Function(
            lambda x: 2.0 * x,
            output_label="prediction",
            label="predict",
        )

        assert wrapped(1.0).label == "prediction"

    def test_a_lifted_result_is_labeled_by_output_name(self):
        wrapped = Function(
            lambda x: 2.0 * x,
            output_label="prediction",
            n_broadcast_samples=6,
            dispatch="sequential",
            label="predict",
            output_spec=OutputSpec(prediction=None),
        )

        with workflow_run(seed=0):
            law = wrapped(standard_normal())
        rows = wrapped(
            NumericArrayBatch(
                jnp.arange(3.0),
                "row",
                element_spec=SCALAR,
                label="rows",
            )
        )

        assert isinstance(law, Distribution) and law.label == "prediction"
        assert isinstance(rows, Batch) and rows.label == "prediction"

    def test_a_returned_product_takes_the_output_label_as_its_label(self):
        """The product displays by the label, and the returned object is left unlabeled."""
        product = Normal("a", 0.0, 1.0) * Gamma("b", 2.0, 1.0)
        result = Function(
            lambda: product,
            label="predict",
        )()

        assert result.label == "predict()"
        assert str(result) == result.notation == "predict()"
        assert product.notation == "Normal(a)·Gamma(b)"

    def test_each_product_of_a_sweep_displays_by_its_element_label(self):
        def predict(loc):
            return Normal("a", loc, 1.0) * Gamma("b", 2.0, 1.0)

        rows = NumericArrayBatch(
            jnp.arange(2.0),
            "row",
            element_spec=SCALAR,
            label="rows",
        )
        result = Function(
            predict,
            label="predict",
        )(rows)

        assert result[0].notation == "predict(rows)[row=0]"

    def test_provenance_records_the_function_its_dependencies_and_its_inputs(self):
        wrapped = Function(
            lambda x, y: x + y,
            label="add",
        )
        tracked = NumericArray(
            jnp.ones(2),
            label="a",
        )

        provenance = wrapped(tracked, 3.0).provenance

        assert [parent.label for parent in provenance.parents] == ["add", "a"]
        assert set(provenance.inputs) == {"y"}

    @pytest.mark.pending(
        reason="provenance records the call's resolved controls", raises=AssertionError
    )
    def test_provenance_records_the_resolved_controls(self):
        wrapped = Function(
            lambda x, y: x + y,
            dispatch="sequential",
            label="add",
        )

        controls = wrapped(1.0, 2.0).provenance.controls

        assert controls.get("dispatch") == "sequential"
        assert controls.get("n_broadcast_samples") == Function.DEFAULT_N_BROADCAST_SAMPLES


class TestRaw:
    def test_raw_returns_a_returned_function_detached(self):
        def inner(x):
            return x

        result = Function(
            lambda: inner,
            label="make",
        ).with_options(raw=True)()

        assert result is inner

    def test_raw_returns_an_array_result_as_its_backing_array(self):
        result = Function(
            lambda x: 2.0 * x,
            label="double",
        ).with_options(raw=True)(jnp.ones(2))

        np.testing.assert_allclose(np.asarray(result), 2.0)
        assert not isinstance(result, NumericArray)


class TestResultErrors:
    def test_the_errors_are_return_contract_defects(self):
        assert issubclass(ResultKindError, TypeError)
        assert issubclass(ResultSchemaError, ValueError)
        assert not issubclass(ResultKindError, ApplicabilityError)
        assert not issubclass(ResultSchemaError, ApplicabilityError)

    def test_an_incompatible_shape_raises_result_schema_error(self):
        wrapped = Function(
            lambda: jnp.ones(3),
            output_spec=OutputSpec(f=NumericArraySpec((2,))),
            label="f",
        )

        with pytest.raises(ResultSchemaError, match="output"):
            wrapped()

    @pytest.mark.parametrize("dispatch", ["jax", "sequential", "thread", "auto"])
    @pytest.mark.parametrize("regime", ["broadcast", "sweep"])
    def test_a_lifted_output_violation_raises_result_schema_error_under_every_dispatch(
        self, regime, dispatch
    ):
        wrapped = Function(
            lambda x: jnp.ones(3),
            output_spec=OutputSpec(f=NumericArraySpec((2,))),
            dispatch=dispatch,
            n_broadcast_samples=6,
            label="f",
        )
        operand = (
            standard_normal()
            if regime == "broadcast"
            else NumericRecordBatch(
                {"x": jnp.arange(3.0)},
                "row",
                element_spec=RecordSpec(x=()),
                label="rows",
            )
        )

        with workflow_run(seed=0), pytest.raises(ResultSchemaError, match="output"):
            wrapped(operand)

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [(jnp.int32, jnp.float32), (jnp.bool_, jnp.float32), (jnp.float32, jnp.int32)],
        ids=["integer-for-float", "bool-for-float", "float-for-integer"],
    )
    def test_a_returned_dtype_of_another_kind_raises_result_schema_error(self, returned, declared):
        wrapped = Function(
            lambda: jnp.ones((), dtype=returned),
            output_spec=OutputSpec(f=NumericArraySpec((), declared)),
            label="f",
        )

        with pytest.raises(ResultSchemaError, match="dtype"):
            wrapped()
        with pytest.raises(ValueError, match="dtype"):
            wrapped.apply()

    def test_a_returned_dtype_of_the_declared_kind_keeps_the_declaration(self):
        wrapped = Function(
            lambda: jnp.ones((), dtype=jnp.float32),
            output_spec=OutputSpec(f=NumericArraySpec((), jnp.float64)),
            label="f",
        )

        assert wrapped().spec == NumericArraySpec((), jnp.float64)

    def test_a_record_field_of_another_dtype_kind_raises_result_schema_error(self):
        wrapped = Function(
            lambda: {"y": jnp.ones((), dtype=jnp.int32)},
            output_spec=RecordSpec(y=NumericArraySpec((), jnp.float32)),
            label="f",
        )

        with pytest.raises(ResultSchemaError, match="output/y dtype int32"):
            wrapped()

    def test_a_batch_of_another_dtype_kind_raises_result_schema_error(self):
        returned = NumericArrayBatch(
            jnp.arange(3, dtype=jnp.int32),
            "row",
            element_spec=NumericArraySpec(()),
            label="rows",
        )
        declared = BatchSpec(NumericArraySpec((), jnp.float32), returned.spec.levels)

        with pytest.raises(ResultSchemaError, match="dtype int32"):
            Function(
                lambda: returned,
                output_spec=OutputSpec.default(declared, component="f"),
                label="f",
            )()

    @pytest.mark.parametrize("dispatch", ["jax", "sequential"])
    def test_a_lifted_dtype_of_another_kind_raises_result_schema_error(self, dispatch):
        wrapped = Function(
            lambda x: jnp.ones((), dtype=jnp.int32),
            output_spec=OutputSpec(f=NumericArraySpec((), jnp.float32)),
            dispatch=dispatch,
            n_broadcast_samples=6,
            label="f",
        )

        with workflow_run(seed=0), pytest.raises(ResultSchemaError, match="dtype"):
            wrapped(standard_normal())

    def test_a_violated_support_raises_result_schema_error(self):
        wrapped = Function(
            lambda: -jnp.ones(2),
            output_spec=OutputSpec(f=NumericArraySpec((2,), support=positive)),
            label="f",
        )

        with pytest.raises(ResultSchemaError, match="support"):
            wrapped()

    def test_an_output_dimension_bound_twice_raises_result_schema_error(self):
        wrapped = Function(
            lambda x: x[:-1],
            input_spec={"x": NumericArraySpec(("obs",))},
            output_spec=OutputSpec(y=NumericArraySpec(("obs",))),
            label="f",
        )

        with pytest.raises(ResultSchemaError, match="output/y"):
            wrapped(jnp.ones(4))

    def test_rows_that_disagree_on_an_output_dimension_raise_result_schema_error(self):
        wrapped = Function(
            lambda x: jnp.ones(int(x) + 1),
            output_spec=OutputSpec(y=NumericArraySpec(("k",))),
            dispatch="sequential",
            label="f",
        )
        rows = NumericArrayBatch(
            jnp.arange(2.0),
            "row",
            element_spec=SCALAR,
            label="rows",
        )

        with pytest.raises(ResultSchemaError):
            wrapped(rows)

    def test_a_wrong_returned_kind_raises_result_kind_error(self):
        wrapped = Function(
            lambda: "text",
            output_spec=OutputSpec(f=NumericArraySpec(())),
            label="f",
        )

        assert isinstance(error_of(wrapped), ResultKindError)

    @pytest.mark.parametrize("mode", ["plain", "raw", "apply"])
    @pytest.mark.parametrize(
        ("declaration", "value"),
        [
            pytest.param(
                SCALAR,
                Opaque(
                    "text",
                    label="stored",
                ),
                id="tracked-opaque-for-array",
            ),
            pytest.param(SCALAR, {"y": 1.0}, id="mapping-for-array"),
            pytest.param(RecordSpec(y=SCALAR), 1.0, id="array-for-record"),
            pytest.param(FunctionSpec(), 1.0, id="array-for-function"),
            pytest.param(DistributionSpec(OutputSpec(y=SCALAR)), 1.0, id="array-for-distribution"),
            pytest.param(BatchSpec(SCALAR, row=2), jnp.ones(2), id="array-for-batch"),
            pytest.param(OpaqueSpec(), {"y": 1.0}, id="mapping-for-opaque"),
            pytest.param(
                OpaqueSpec(),
                NumericArray(
                    1.0,
                    label="stored",
                ),
                id="tracked-array-for-opaque",
            ),
        ],
    )
    def test_an_overall_kind_mismatch_is_distinguished_from_a_schema_error(
        self, declaration, value, mode
    ):
        wrapped = Function(
            lambda: value,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, component="produce"),
            label="produce",
        )
        invoke = wrapped.apply if mode == "apply" else wrapped.with_options(raw=mode == "raw")

        with pytest.raises(ValueError if mode == "apply" else ResultKindError, match="output"):
            invoke()

    @pytest.mark.parametrize("dispatch", ["jax", "sequential", "thread", "auto"])
    @pytest.mark.parametrize("regime", ["broadcast", "sweep"])
    @pytest.mark.parametrize("raw", [False, True])
    def test_a_lifted_kind_mismatch_raises_result_kind_error(self, dispatch, regime, raw):
        wrapped = Function(
            lambda x: {"y": x},
            output_spec=OutputSpec.default(SCALAR, component="produce"),
            dispatch=dispatch,
            n_broadcast_samples=6,
            raw=raw,
            label="produce",
        )
        operand = (
            standard_normal()
            if regime == "broadcast"
            else NumericArrayBatch(
                jnp.arange(3.0),
                "row",
                element_spec=SCALAR,
                label="rows",
            )
        )

        with (
            workflow_run(seed=0),
            pytest.raises(ResultKindError, match=r"output.*NumericArraySpec"),
        ):
            wrapped(operand)

    @pytest.mark.parametrize("value", [{"y": "text"}, {"other": 1.0}])
    def test_a_field_kind_or_structure_mismatch_remains_a_schema_error(self, value):
        wrapped = Function(
            lambda: value,
            output_spec=RecordSpec(y=SCALAR),
            label="produce",
        )

        with pytest.raises(ResultSchemaError, match="output"):
            wrapped()

    @pytest.mark.parametrize("error", [TypeError("body failed"), ValueError("body failed")])
    @pytest.mark.parametrize("regime", ["plain", "broadcast", "sweep"])
    def test_an_exception_from_the_body_is_propagated_unchanged(self, error, regime):
        def body(x):
            raise error

        wrapped = Function(
            body,
            output_spec=OutputSpec.default(SCALAR, component="produce"),
            dispatch="sequential",
            label="produce",
        )
        operand = {
            "plain": 1.0,
            "broadcast": standard_normal(),
            "sweep": NumericArrayBatch(
                jnp.arange(3.0),
                "row",
                element_spec=SCALAR,
                label="rows",
            ),
        }[regime]

        with workflow_run(seed=0), pytest.raises(type(error)) as raised:
            wrapped(operand)

        assert raised.value is error
