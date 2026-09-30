"""Function declarations, independent names, and the value/engine boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    DistributionSpec,
    Function,
    FunctionSpec,
    InputSpec,
    Module,
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
    function,
    workflow_method,
    workflow_run,
)
from probpipe.core.constraints import positive


class TestFunctionDeclarations:
    def test_required_name_and_raw_representation(self):
        def add(x, /, *, y=2):
            return x + y

        wrapped = Function("add", add)
        assert list(inspect.signature(Function.__init__).parameters)[1:3] == ["name", "fn"]
        assert wrapped.raw() is add
        assert wrapped.apply(3) == 5
        assert inspect.signature(wrapped) == inspect.signature(add)
        with pytest.raises(TypeError):
            Function(fn=add)

    def test_names_are_independent(self, full_provenance_mode):
        @function(name="predict", output_name="prediction", output_spec=OutputSpec(mean=None))
        def predict_impl(x):
            return x + 1

        renamed = predict_impl.with_name("renamed_predict")
        result = renamed(2)
        assert isinstance(result, NumericArray)
        assert result.name == "prediction"
        assert result.provenance.parents[0].parent is renamed
        assert float(result) == 3
        assert renamed.output_name == "prediction"
        assert renamed.output_spec is predict_impl.output_spec
        assert renamed.output_spec.components == {"mean": None}
        assert result.with_name("display").name == "display"
        assert predict_impl.output_spec.components == {"mean": None}

    def test_default_output_name_is_captured_once(self):
        wrapped = Function("score", lambda x: x, output_spec=NumericArraySpec(()))
        renamed = wrapped.with_name("other")
        assert renamed.output_name == "score"
        assert renamed.output_spec.components == {"score": NumericArraySpec(())}
        assert renamed(4).name == "score"

    def test_decorator_can_be_reused_with_its_name_override(self):
        decorate = function(name="shared", output_name="value")
        first = decorate(lambda: 1)
        second = decorate(lambda: 2)
        assert first.name == second.name == "shared"
        assert first.output_name == second.output_name == "value"
        assert float(first()) == 1
        assert float(second()) == 2

    def test_returned_function_keeps_its_own_output_contract(self):
        returned = Function("inner", lambda: 3, output_spec=NumericArraySpec(()))
        factory = Function("factory", lambda: returned, output_name="created")
        result = factory()
        assert isinstance(result, Function)
        assert result.name == result.__name__ == "created"
        assert result.output_name == "inner"
        assert result.spec is returned.spec
        assert returned.name == returned.__name__ == "inner"
        assert result().name == "inner"

    def test_raw_callable_return_receives_its_declared_function_spec(self):
        declaration = FunctionSpec(
            InputSpec(x=NumericArraySpec(())), OutputSpec(value=NumericArraySpec(()))
        )
        wrapped = Function("factory", lambda: lambda x: x + 1, output_spec=declaration)
        result = wrapped()
        assert isinstance(result, Function)
        assert result.spec == declaration
        assert float(result(2)) == 3
        with pytest.raises(ValueError, match="input/x"):
            result(jnp.ones(2))

    def test_returned_array_batch_receives_its_declared_element_spec(self):
        from probpipe import BatchSpec
        from probpipe.core.constraints import positive

        stored = NumericArrayBatch("stored", jnp.ones(2), "draw", element_spec=NumericArraySpec(()))
        declared = BatchSpec(
            NumericArraySpec((), support=positive), stored.axis_groups, stored.level_names
        )
        wrapped = Function("load", lambda: stored, output_spec=declared)
        result = wrapped()
        assert wrapped.apply() is stored
        assert result.spec == declared
        assert stored.element_spec.support is None

    def test_labels_are_outside_spec_equality(self):
        declaration = OutputSpec(mean=NumericArraySpec(()))
        left = Function("left", lambda x: x, output_spec=declaration, output_name="a")
        right = Function("right", lambda x: x, output_spec=declaration, output_name="b")
        assert left.spec == right.spec == FunctionSpec(output_spec=declaration)

    @pytest.mark.parametrize("exposed", [True, False])
    def test_single_field_record_keeps_its_kind_and_exposure(self, exposed):
        record_spec = RecordSpec(x=())
        declaration = record_spec if exposed else OutputSpec(bundle=record_spec)
        wrapped = Function("pack", lambda x: {"x": x}, output_spec=declaration)
        result = wrapped(2)
        assert isinstance(result, Record)
        assert result.name == "pack"
        assert float(result["x"]) == 2
        assert tuple(wrapped.output_spec.components) == (("x",) if exposed else ("bundle",))

    def test_type_hole_is_resolved_per_call(self):
        wrapped = Function("identity", lambda x: x, output_spec=OutputSpec(value=None))
        assert isinstance(wrapped(3), NumericArray)
        assert isinstance(wrapped("text"), Opaque)
        assert wrapped.output_spec.spec is None

    def test_shared_and_output_only_dimensions(self):
        wrapped = Function(
            "append",
            lambda x: jnp.concatenate([x, x]),
            input_spec={"x": NumericArraySpec(("n",))},
            output_spec=NumericArraySpec(("m",)),
        )
        assert wrapped.input_spec == InputSpec(x=NumericArraySpec(("n",)))
        assert wrapped(jnp.ones(3)).shape == (6,)
        assert wrapped(jnp.ones(2)).shape == (4,)
        assert wrapped.output_spec.spec.free_dims == {"m"}

    def test_returned_term_is_copied_and_relabelled(self):
        value = NumericArray("stored", jnp.array([1.0, 2.0]))
        wrapped = Function("load", lambda: value, output_name="loaded")
        assert wrapped.apply() is value
        result = wrapped()
        assert result is not value
        assert result.value is value.value
        assert result.name == "loaded"
        assert value.name == "stored"

    def test_input_and_output_kinds_are_checked(self):
        wrapped = Function(
            "identity",
            lambda x: x,
            input_spec={"x": NumericArraySpec((2,))},
            output_spec=RecordSpec(x=(2,)),
        )
        with pytest.raises(ValueError, match="input"):
            wrapped.apply(jnp.ones(3))
        with pytest.raises(ValueError, match="RecordSpec"):
            wrapped.apply(jnp.ones(2))

    def test_options_preserve_declarations(self):
        wrapped = Function("value", lambda x: x, output_spec=OpaqueSpec())
        view = wrapped.with_options(dispatch="thread", max_workers=2)
        assert view.options["dispatch"] == "thread"
        assert wrapped.options["dispatch"] == "auto"
        assert view.spec is wrapped.spec

    def test_base_does_not_import_the_engine(self):
        import probpipe.values._function_base as base

        tree = ast.parse(Path(base.__file__).read_text())
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all("functions" not in ast.unparse(node) for node in imports)


class TestLiftedNames:
    @pytest.mark.parametrize("dispatch", ["sequential", "jax", "thread"])
    def test_sweep_keeps_array_kind_and_result_label(self, dispatch):
        wrapped = Function(
            "double",
            lambda x: x["x"] * 2,
            output_name="doubled",
            output_spec=OutputSpec(value=NumericArraySpec(())),
            dispatch=dispatch,
        )
        from probpipe import NumericRecordBatch

        rows = NumericRecordBatch(
            "inputs", {"x": jnp.arange(3.0)}, "case", element_spec=RecordSpec(x=())
        )
        result = wrapped(rows)
        assert isinstance(result, NumericArrayBatch)
        assert result.name == "doubled"
        np.testing.assert_array_equal(result.values, [0.0, 2.0, 4.0])

    def test_empty_sweep_uses_declared_array_kind(self):
        wrapped = Function(
            "double",
            lambda x: x * 2,
            output_name="doubled",
            output_spec=NumericArraySpec(()),
            dispatch="sequential",
        )
        rows = NumericArrayBatch("inputs", jnp.empty(0), "case", element_spec=NumericArraySpec(()))
        result = wrapped(rows)
        assert isinstance(result, NumericArrayBatch)
        assert result.name == "doubled"
        assert result.batch_shape == (0,)

    @pytest.mark.parametrize(
        ("declaration", "kind"),
        [
            (RecordSpec(x=()), "NumericRecordBatch"),
            (OpaqueSpec(), "OpaqueBatch"),
            (FunctionSpec(), "FunctionBatch"),
        ],
    )
    def test_empty_sweep_keeps_other_declared_kinds(self, declaration, kind):
        def unexpected(x):
            raise AssertionError("An empty sweep must not evaluate the body")

        wrapped = Function(
            "empty",
            unexpected,
            output_name="results",
            output_spec=declaration,
            dispatch="sequential",
        )
        rows = NumericArrayBatch("inputs", jnp.empty(0), "case", element_spec=NumericArraySpec(()))
        result = wrapped(rows)
        assert type(result).__name__ == kind
        assert result.name == "results"
        assert result.element_spec == declaration
        assert result.batch_shape == (0,)

    @pytest.mark.parametrize("declaration", [OutputSpec(value=None), NumericArraySpec(("n",))])
    def test_empty_sweep_cannot_infer_missing_output_type_or_shape(self, declaration):
        wrapped = Function("empty", lambda x: x, output_spec=declaration, dispatch="sequential")
        rows = NumericArrayBatch("inputs", jnp.empty(0), "case", element_spec=NumericArraySpec(()))
        with pytest.raises(ValueError, match="concrete output_spec"):
            wrapped(rows)

    def test_broadcast_has_independent_label_and_component(self):
        wrapped = Function(
            "double",
            lambda x: x * 2,
            output_name="doubled",
            output_spec=OutputSpec(value=NumericArraySpec(())),
            dispatch="sequential",
            n_broadcast_samples=8,
        )
        with workflow_run(seed=1):
            result = wrapped(Normal("x", 0, 1))
        assert result.name == "doubled"
        assert result.fields == ("value",)
        assert result.num_atoms == 8

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    @pytest.mark.parametrize("renamed", [None, "renamed", "M.pair"])
    def test_list_sweep_levels_use_output_name(self, dispatch, renamed):
        rows = NumericRecordBatch(
            "inputs", {"x": jnp.arange(3.0)}, "rows", element_spec=RecordSpec(x=())
        )
        wrapped = Function(
            "pair",
            lambda x: [x["x"], x["x"] + 1.0],
            output_name="outs",
        )
        if renamed is not None:
            wrapped = wrapped.with_name(renamed)
        result = wrapped.with_options(dispatch=dispatch)(rows)
        assert isinstance(result, NumericArrayBatch)
        assert result.name == "outs"
        assert result.level_names == ("rows", "outs")
        np.testing.assert_array_equal(
            result.values,
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        )


class TestCompletedOutputDeclarations:
    @pytest.fixture
    def rows(self):
        return NumericRecordBatch(
            "inputs", {"x": jnp.arange(1.0, 4.0)}, "case", element_spec=RecordSpec(x=())
        )

    @pytest.mark.parametrize("mode", ["plain", "sweep", "broadcast"])
    def test_returned_laws_use_declaration_unification_across_paths(self, rows, mode):
        stored = Normal("y", jnp.asarray(0.0, dtype="float32"), 1.0)
        declaration = DistributionSpec(
            OutputSpec(y=NumericArraySpec((), dtype="float64", support=positive))
        )
        factory = Function(
            "factory",
            lambda x: stored,
            output_spec=declaration,
            dispatch="sequential",
            n_broadcast_samples=3,
        )
        operand = {"plain": rows[0], "sweep": rows, "broadcast": Normal("x", 0.0, 1.0)}[mode]
        with workflow_run(seed=0):
            result = factory(operand)
        laws = (result,) if mode == "plain" else result.components
        for law in laws:
            assert law.spec is stored.spec
            assert law.event_spec.components["y"].dtype == np.dtype("float32")
            assert law.event_spec.components["y"].support != positive

    @pytest.mark.parametrize("dispatch", ["sequential", "thread"])
    def test_swept_returned_functions_enforce_the_declared_contract(self, rows, dispatch):
        stored = Function("inner", lambda: -1.0)
        declaration = FunctionSpec(
            output_spec=OutputSpec(value=NumericArraySpec((), support=positive))
        )
        factory = Function(
            "factory", lambda row: stored, output_spec=declaration, dispatch=dispatch
        )

        single = factory(rows[0])
        batch = factory(rows)
        assert batch.element_spec == declaration
        for result in (single, *batch):
            assert result is not stored
            assert result.spec == declaration
            with pytest.raises(ValueError, match="support positive"):
                result()
        assert stored.output_spec is None
        assert factory.apply(rows[0]) is stored

    @pytest.mark.parametrize("dispatch", ["sequential", "thread"])
    def test_swept_existing_arrays_keep_declared_support(self, rows, dispatch):
        stored = NumericArray("stored", jnp.ones(2))
        declaration = NumericArraySpec((2,), support=positive)
        wrapped = Function("load", lambda row: stored, output_spec=declaration, dispatch=dispatch)
        result = wrapped(rows)
        assert result.element_spec == wrapped(rows[0]).spec == declaration
        np.testing.assert_array_equal(result.values, np.ones((3, 2)))
        assert stored.spec.support is None

    @pytest.mark.parametrize("dispatch", ["sequential", "jax", "thread"])
    def test_sweep_completes_output_only_dimensions_per_call(self, rows, dispatch):
        declaration = RecordSpec(stats=RecordSpec(y=("width",)))
        wrapped = Function(
            "pack",
            lambda row, width: {"stats": {"y": jnp.full((width,), row["x"])}},
            output_spec=declaration,
            dispatch=dispatch,
        )
        for width in (2, 4):
            result = wrapped(rows, width)
            assert result.element_spec == RecordSpec(stats=RecordSpec(y=(width,)))
            np.testing.assert_array_equal(
                result["stats/y"], np.repeat([[1.0], [2.0], [3.0]], width, axis=1)
            )
        assert wrapped.output_spec.spec is declaration
        assert declaration.free_dims == {"width"}

    @pytest.mark.parametrize("dispatch", ["sequential", "jax", "thread"])
    @pytest.mark.parametrize("kind", ["array", "record", "hole", "record_hole"])
    def test_broadcast_completes_dimensions_and_preserves_component(self, dispatch, kind):
        declaration = {
            "array": OutputSpec(component=NumericArraySpec(("width",))),
            "record": OutputSpec(RecordSpec(component=("width",))),
            "hole": OutputSpec(component=None),
            "record_hole": OutputSpec(component=None),
        }[kind]

        def body(x):
            value = jnp.stack([x, x + 1])
            return {"component": value} if kind in ("record", "record_hole") else value

        wrapped = Function(
            "f",
            body,
            output_name="result",
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            joint = wrapped.with_options(include_inputs=True)(Normal("x", 0, 1))
        result = joint.marginalize()
        assert result.name == "result"
        assert result.fields == ("component",)
        assert result.event_spec.spec["component"].shape == (2,)
        np.testing.assert_allclose(
            result.samples["component"][:, 1], result.samples["component"][:, 0] + 1, rtol=0, atol=0
        )
        assert wrapped.output_spec is declaration
        if kind in ("hole", "record_hole"):
            assert declaration.spec is None
        else:
            assert declaration.spec.free_dims == {"width"}

    @pytest.mark.parametrize("tracked", [False, True])
    def test_type_hole_accepts_a_nested_record(self, tracked):
        value = {"stats": {"mean": 2.0}}
        if tracked:
            value = Record("stored", value)
        wrapped = Function("load", lambda: value, output_spec=OutputSpec(bundle=None))
        result = wrapped()
        assert isinstance(result, Record)
        assert float(result["stats/mean"]) == 2.0
        assert wrapped.output_spec.components == {"bundle": None}
        assert wrapped.apply() is value

    @pytest.mark.parametrize("tracked", [False, True])
    def test_returned_function_declaration_must_match_its_signature(self, tracked):
        def body(x):
            return x

        returned = Function("inner", body) if tracked else body
        declaration = FunctionSpec(input_spec=InputSpec(y=NumericArraySpec(())))
        factory = Function("factory", lambda: returned, output_spec=declaration)
        with pytest.raises(ValueError, match=r"input_spec slots.*signature parameters"):
            factory()
        if tracked:
            assert returned.input_spec is None

    @pytest.mark.parametrize("bound", [False, True])
    def test_returned_function_checks_defaults_and_bindings(self, bound):
        default = jnp.ones(2)
        returned = (
            Function("inner", lambda x: x, bind={"x": default})
            if bound
            else Function("inner", lambda x=default: x)
        )
        factory = Function(
            "factory",
            lambda: returned,
            output_spec=FunctionSpec(input_spec=InputSpec(x=NumericArraySpec(()))),
        )
        with pytest.raises(ValueError, match=r"default/x|construction binding/x"):
            factory()
        assert returned.input_spec is None

    @pytest.mark.parametrize("dispatch", ["sequential", "thread"])
    def test_nested_lift_carries_completed_output_dimensions(self, rows, dispatch):
        declaration = RecordSpec(component=("width",))
        wrapped = Function(
            "nested",
            lambda row, x: {"component": jnp.full((2,), x + row["x"])},
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            result = wrapped(rows, Normal("x", 0, 1))
        assert result.event_spec.spec.leaf_shapes == {"component": (2,)}
        assert result.batch_shape == (3,)
        for marginal in result:
            assert marginal.event_spec.spec == result.event_spec.spec
            values = marginal.samples["component"]
            np.testing.assert_array_equal(values[:, 0], values[:, 1])
        assert declaration.free_dims == {"width"}

    def test_output_only_dimensions_must_agree_across_sweep_rows(self, rows):
        wrapped = Function(
            "ragged",
            lambda row: {"value": jnp.ones(int(row["x"]))},
            output_spec=RecordSpec(value=("width",)),
            dispatch="sequential",
        )
        with pytest.raises(ValueError, match="already bound"):
            wrapped(rows)


class TestModuleReturnInference:
    @pytest.mark.parametrize("sequence", [[1, 2], (1, 2), []])
    def test_method_sequence_uses_the_same_inference_as_a_function(self, sequence):
        class Example(Module):
            @workflow_method
            def numbers(self):
                return sequence

        method = Example().numbers
        ordinary = Function("numbers", lambda: sequence)
        result = method()
        expected = ordinary()
        assert method.output_spec is None
        assert method.name == "Example.numbers"
        assert result.name == method.output_name == "numbers"
        assert type(result) is type(expected)
        assert result.spec == expected.spec
        if sequence:
            np.testing.assert_array_equal(result.values, expected.values)
