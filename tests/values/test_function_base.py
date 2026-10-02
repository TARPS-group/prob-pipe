"""Function declarations, independent names, and the value/engine boundary."""

from __future__ import annotations

import ast
import inspect
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    DistributionSpec,
    EmpiricalDistribution,
    Function,
    FunctionSpec,
    Gamma,
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
    ProductDistribution,
    Record,
    RecordSpec,
    WorkflowKind,
    function,
    workflow_method,
    workflow_run,
)
from probpipe.core.constraints import positive, real


class TestFunctionSpecMatching:
    @pytest.mark.parametrize("from_value", [False, True])
    @pytest.mark.parametrize(
        ("expected", "actual", "message"),
        [
            (
                FunctionSpec(InputSpec(x=NumericArraySpec(()))),
                Function("actual", lambda y: y, input_spec={"y": NumericArraySpec(())}),
                "incompatible input slots",
            ),
            (
                FunctionSpec(output_spec=OutputSpec(left=None)),
                Function("actual", lambda: 1, output_spec=OutputSpec(right=None)),
                "incompatible output components",
            ),
            (
                FunctionSpec(output_spec=OutputSpec(component=None)),
                Function("actual", lambda: {"component": 1}, output_spec=RecordSpec(component=())),
                "incompatible output components",
            ),
            (
                FunctionSpec(output_spec=OutputSpec(RecordSpec(left=()))),
                Function("actual", lambda: {"right": 1}, output_spec=RecordSpec(right=())),
                "incompatible output components",
            ),
        ],
        ids=["slots", "component-names", "whole-vs-exposed", "exposed-fields"],
    )
    def test_incompatible_declarations_raise(self, expected, actual, message, from_value):
        with pytest.raises(ValueError, match=message):
            if from_value:
                expected.bind_dims_from_value(actual)
            else:
                expected.bind_dims_from_spec(actual.spec)

    def test_matching_declarations_bind_dimensions_without_changing_labels(self):
        expected = FunctionSpec(
            InputSpec(x=NumericArraySpec(("n",))),
            OutputSpec(component=NumericArraySpec(("n",))),
        )
        actual = Function(
            "display",
            lambda x: x,
            input_spec={"x": NumericArraySpec((3,))},
            output_spec=OutputSpec(component=NumericArraySpec((3,))),
        )
        for value in (actual, actual.with_name("another")):
            assert expected.bind_dims_from_value(value) == actual.spec
            assert expected.bind_dims_from_spec(value.spec) == actual.spec
        assert expected.free_dims == {"n"}
        assert actual.name == "display"


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

    @pytest.fixture(params=["partial", "instance"])
    def unnamed_callable(self, request):
        def add(a, b):
            return a + b

        class AddOne:
            def __call__(self, x):
                return x + 1

        if request.param == "partial":
            return partial(add, 1)
        return AddOne()

    def test_decorator_accepts_unnamed_callable_with_explicit_name(self, unnamed_callable):
        wrapped = function(name="add1")(unnamed_callable)
        assert wrapped.name == "add1"
        assert wrapped.output_name == "add1"
        assert wrapped.raw() is unnamed_callable
        assert wrapped.apply(114514) == 114515

    @pytest.mark.parametrize("with_parentheses", [True, False])
    def test_decorator_requires_name_for_unnamed_callable(self, unnamed_callable, with_parentheses):
        decorate = function() if with_parentheses else function
        with pytest.raises(ValueError, match="an explicit 'name'"):
            decorate(unnamed_callable)

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

    def test_plain_engine_resolves_to_off(self, monkeypatch):
        import probpipe.values._function_base as base

        monkeypatch.setattr(base, "_call_engine", base._plain_call)
        monkeypatch.setattr(base, "_apply_scope", base.nullcontext)
        monkeypatch.setattr(base, "_workflow_kind_resolver", base._plain_workflow_kind)
        value = object()
        wrapped = Function("value", lambda: value, workflow_kind=WorkflowKind.TASK)
        assert wrapped.effective_workflow_kind is WorkflowKind.OFF
        assert wrapped() is value
        base.install_call_engine(base._plain_call)
        assert wrapped.effective_workflow_kind is WorkflowKind.OFF


class TestLiftedNames:
    @pytest.mark.parametrize("sliced", [False, True])
    def test_returned_batch_relabels_its_view_root(self, sliced, full_provenance_mode):
        stored = NumericArrayBatch(
            "pts",
            jnp.arange(6.0).reshape(2, 3),
            ("chain", "row"),
            axes_per_level=(1, 1),
            element_spec=NumericArraySpec(()),
        )
        if sliced:
            stored = stored[1]
        original_name = stored.name
        factory = Function("factory", lambda: stored, output_name="f")
        result = factory()
        assert result.name == "f"
        assert result[0].name == ("f[row=0]" if sliced else "f[chain=0]")
        if not sliced:
            assert result[0][1].name == "f[chain=0, row=1]"
        assert result.provenance.parents[0].parent is factory
        assert result[0].provenance is result.provenance
        assert stored.name == original_name
        assert stored.provenance is None
        renamed = result.with_name("display")
        assert renamed[0].name.startswith("display[")
        assert renamed.provenance.operation == "with_name"
        np.testing.assert_array_equal(result.values, stored.values)

    def test_returned_function_relabels_python_names(self, full_provenance_mode):
        stored = Function("inner", lambda: 1, output_name="value")
        factory = Function("factory", lambda: stored, output_name="result")
        result = factory()
        assert result.name == result.__name__ == result.__qualname__ == "result"
        assert result.output_name == "value"
        assert result.provenance.parents[0].parent is factory
        assert stored.name == stored.__name__ == "inner"

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


class TestLiftedInputDeclarations:
    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    def test_declared_array_input_accepts_single_leaf_empirical(self, dispatch):
        values = jnp.asarray([0.0, 1.0, 2.0])
        law = EmpiricalDistribution("theta", values)
        predict = Function(
            "predict",
            lambda theta: 2 * theta,
            input_spec={"theta": NumericArraySpec(())},
            dispatch=dispatch,
        )
        result = predict(theta=law)
        assert result.num_atoms == 3
        np.testing.assert_array_equal(result.samples["predict"], 2 * values)

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    def test_declared_functions_compose_over_a_law(self, dispatch):
        F = Function(
            "F",
            lambda theta: theta + 1,
            input_spec={"theta": NumericArraySpec(())},
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        G = Function(
            "G",
            lambda x: x * 2,
            input_spec={"x": NumericArraySpec(())},
            dispatch=dispatch,
        )
        with workflow_run(seed=114514):
            intermediate = F(Normal("theta", 0.0, 1.0))
            result = G(intermediate)
        assert intermediate.num_atoms == 8
        assert result.num_atoms == 8
        np.testing.assert_array_equal(
            result.samples["G"],
            2 * (intermediate.samples["F"]),
        )


class TestCompletedOutputDeclarations:
    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    @pytest.mark.parametrize("exposed", [False, True])
    def test_broadcast_preserves_record_component_exposure(self, dispatch, exposed):
        import jax

        from probpipe import sample

        record = RecordSpec(field=NumericArraySpec((2,), dtype="float32"))
        declaration = OutputSpec(record) if exposed else OutputSpec(bundle=record)
        factory = Function(
            "factory",
            lambda x: {"field": jnp.stack([x, x + 1])},
            output_name="results",
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        completed = RecordSpec(field=NumericArraySpec((2,), dtype="float32", support=real))
        expected = OutputSpec(completed) if exposed else OutputSpec(bundle=completed)
        assert result.event_spec == expected
        assert result.fields == (("field",) if exposed else ("bundle",))
        if not exposed:
            assert result["bundle"] is result
        draws = sample(result, key=jax.random.PRNGKey(1), sample_shape=(4,))
        np.testing.assert_allclose(draws["field"][:, 1], draws["field"][:, 0] + 1, rtol=0, atol=0)
        assert factory.output_spec is declaration

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    @pytest.mark.parametrize("kind", ["function", "opaque", "record", "batch"])
    def test_broadcast_keeps_non_numeric_component_kind(self, dispatch, kind):
        values = {
            "function": Function("inner", lambda x: x + 1),
            "opaque": Opaque("stored", "payload", spec=OpaqueSpec(meta="text")),
            "record": Record("stored", field=Opaque("leaf", "payload")),
            "batch": NumericArrayBatch(
                "stored", jnp.arange(2.0), "row", element_spec=NumericArraySpec(())
            ),
        }
        stored = values[kind]
        declaration = OutputSpec(component=stored.spec)
        factory = Function(
            "factory",
            lambda x: stored,
            output_name="results",
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        assert result.event_spec == declaration
        assert result["component"] is result
        assert len(result.items) == 8
        assert all(type(item) is type(stored) for item in result.items)
        if kind == "function":
            assert result.items[0].apply(2) == 3
        assert stored.name in ("inner", "stored")

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    def test_joint_broadcast_constructs_completed_output_declaration(self, dispatch):
        factory = Function(
            "factory",
            lambda x: jnp.stack([x, x + 1, x + 2]),
            output_name="results",
            output_spec=OutputSpec(component=NumericArraySpec(("width",))),
            dispatch=dispatch,
            n_broadcast_samples=8,
            include_inputs=True,
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        assert result.name == "results"
        assert result.event_spec.spec["_output"] == NumericArraySpec((3,))
        assert factory.output_spec.spec.free_dims == {"width"}

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    def test_enumerated_joint_constructs_completed_output_declaration(self, dispatch):
        factory = Function(
            "factory",
            lambda x: jnp.stack([x, x + 1, x + 2]),
            output_name="results",
            output_spec=OutputSpec(component=None),
            dispatch=dispatch,
            include_inputs=True,
        )
        result = factory(EmpiricalDistribution("x", jnp.arange(3.0)))
        assert result.name == "results"
        assert result.event_spec.spec["_output"] == NumericArraySpec((3,), dtype="float32")
        assert factory.output_spec.spec is None

    @pytest.mark.parametrize("kind", ["array", "record", "batch"])
    @pytest.mark.parametrize("mode", ["plain", "sweep"])
    def test_shape_only_output_preserves_array_metadata(self, kind, mode):
        from probpipe import BatchSpec

        leaf = NumericArraySpec((3,), dtype="float32", support=positive)
        values = jnp.ones(3, dtype="float32")
        if kind == "array":
            stored = NumericArray("stored", values, spec=leaf)
            declaration = NumericArraySpec((3,))
        elif kind == "record":
            stored = Record("stored", stats=Record("stats", x=NumericArray("x", values, spec=leaf)))
            declaration = RecordSpec(stats=RecordSpec(x=NumericArraySpec((3,))))
        else:
            stored = NumericArrayBatch("stored", values[None, :], "item", element_spec=leaf)
            declaration = BatchSpec(NumericArraySpec((3,)), ((1,),), ("item",))
        original_spec = stored.spec
        factory = Function(
            "factory", lambda row: stored, output_spec=declaration, dispatch="sequential"
        )
        if mode == "plain":
            result = factory(0)
            actual = result.spec
        else:
            rows = NumericArrayBatch(
                "rows", jnp.arange(2.0), "row", element_spec=NumericArraySpec(())
            )
            result = factory(rows)
            actual = result.element_spec
        if kind == "record":
            actual = actual["stats/x"]
        elif kind == "batch" and mode == "plain":
            actual = actual.element_spec
        assert actual == leaf
        assert stored.spec is original_spec
        assert factory.output_spec.spec is declaration
        assert factory.apply(0) is stored

    @pytest.mark.parametrize("values", [[1.0, 2.0], (1.0, 2.0), []])
    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    def test_sequence_type_hole_matches_undeclared_return(self, values, dispatch):
        ordinary = Function("f", lambda row: values, output_name="items", dispatch=dispatch)
        declared = Function(
            "f",
            lambda row: values,
            output_name="items",
            output_spec=OutputSpec(component=None),
            dispatch=dispatch,
        )
        rows = NumericRecordBatch(
            "rows", {"x": jnp.arange(2.0)}, "row", element_spec=RecordSpec(x=())
        )
        for operand in (0, rows):
            if not values and dispatch == "jax" and operand is rows:
                continue  # Empty opaque batches are not JAX values.
            expected = ordinary(operand)
            result = declared(operand)
            assert type(result) is type(expected)
            assert result.spec == expected.spec
            if values:
                np.testing.assert_array_equal(result.values, expected.values)
        assert declared.output_spec.spec is None
        assert declared.apply(0) is values

    @pytest.mark.parametrize("mode", ["plain", "sweep", "broadcast"])
    @pytest.mark.parametrize("declared_side", [None, "input", "output"])
    def test_returned_function_preserves_unspecified_declarations(self, mode, declared_side):
        inputs = InputSpec(x=NumericArraySpec(()))
        outputs = OutputSpec(value=NumericArraySpec((), support=positive))
        inner = Function("inner", lambda x: x, input_spec=inputs, output_spec=outputs)
        declaration = FunctionSpec(
            input_spec=inputs if declared_side == "input" else None,
            output_spec=outputs if declared_side == "output" else None,
        )
        factory = Function(
            "outer", lambda row: inner, output_spec=declaration, dispatch="sequential"
        )
        if mode == "plain":
            returned = [factory(0)]
        elif mode == "sweep":
            rows = NumericArrayBatch(
                "rows", jnp.arange(2.0), "row", element_spec=NumericArraySpec(())
            )
            returned = list(factory(rows))
        else:
            returned = factory(EmpiricalDistribution("row", jnp.arange(2.0))).items
        for result in returned:
            assert result is not inner
            assert result.spec == inner.spec
            with pytest.raises(ValueError):
                result.apply(jnp.ones(2))
            with pytest.raises(ValueError, match="support positive"):
                result.apply(-1.0)
        assert inner.input_spec is inputs
        assert inner.output_spec is outputs
        assert factory.apply(0) is inner

    @pytest.fixture
    def rows(self):
        return NumericRecordBatch(
            "inputs", {"x": jnp.arange(1.0, 4.0)}, "case", element_spec=RecordSpec(x=())
        )

    @pytest.mark.parametrize("mode", ["plain", "sweep", "broadcast"])
    def test_returned_laws_use_declaration_unification_across_paths(self, rows, mode):
        stored = Gamma("y", jnp.asarray(1.0, dtype="float32"), 1.0)
        declaration = DistributionSpec(
            OutputSpec(y=NumericArraySpec((), dtype="float64", support=real))
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
            assert law.event_spec.components["y"].support == positive

    @pytest.mark.parametrize("mode", ["apply", "plain", "sweep", "broadcast"])
    def test_returned_law_rejects_incompatible_declared_support(self, rows, mode):
        stored = Normal("y", 0.0, 1.0)
        factory = Function(
            "factory",
            lambda x: stored,
            output_spec=DistributionSpec(OutputSpec(y=NumericArraySpec((), support=positive))),
            dispatch="sequential",
            n_broadcast_samples=8,
        )
        operand = {
            "apply": rows[0],
            "plain": rows[0],
            "sweep": rows,
            "broadcast": Normal("x", 0.0, 1.0),
        }[mode]
        invoke = factory.apply if mode == "apply" else factory

        with (
            workflow_run(seed=0),
            pytest.raises(
                ValueError, match=r"output/factory/y support real does not conform to positive"
            ),
        ):
            invoke(operand)
        assert stored.event_spec.components["y"].support == real

    @pytest.mark.parametrize("whole_record", [False, True])
    def test_returned_law_checks_nested_component_support(self, whole_record):
        stored = ProductDistribution(params=ProductDistribution(y=Normal("y", 0.0, 1.0)))
        declared = RecordSpec(params=RecordSpec(y=NumericArraySpec((), support=positive)))
        if whole_record:
            from probpipe import Distribution

            stored = Distribution("bundle", OutputSpec(bundle=stored.event_spec.spec))
            event_spec = OutputSpec(bundle=declared)
        else:
            event_spec = OutputSpec(declared)
        factory = Function("factory", lambda: stored, output_spec=DistributionSpec(event_spec))

        with pytest.raises(ValueError, match=r"params/y support real does not conform to positive"):
            factory.apply()

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
        assert (
            result.element_spec
            == wrapped(rows[0]).spec
            == NumericArraySpec((2,), dtype=stored.dtype, support=positive)
        )
        assert wrapped.output_spec.spec is declaration
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
            if kind == "record_hole":
                return {"field": value}
            return {"component": value} if kind == "record" else value

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
        field = "field" if kind == "record_hole" else "component"
        record = RecordSpec({field: NumericArraySpec((2,), dtype="float32", support=real)})
        expected = OutputSpec(component=record) if kind == "record_hole" else OutputSpec(record)
        assert result.event_spec == expected
        np.testing.assert_allclose(
            result.samples[field][:, 1], result.samples[field][:, 0] + 1, rtol=0, atol=0
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
    @pytest.mark.parametrize("raw", [False, True])
    def test_returned_function_declaration_must_match_its_signature(self, tracked, raw):
        def body(x):
            return x

        returned = Function("inner", body) if tracked else body
        declaration = FunctionSpec(input_spec=InputSpec(y=NumericArraySpec(())))
        factory = Function("factory", lambda: returned, output_spec=declaration)
        with pytest.raises(ValueError, match=r"input_spec slots.*signature parameters"):
            (factory.apply if raw else factory)()
        if tracked:
            assert returned.input_spec is None

    @pytest.mark.parametrize("bound", [False, True])
    @pytest.mark.parametrize("raw", [False, True])
    def test_returned_function_checks_defaults_and_bindings(self, bound, raw):
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
            (factory.apply if raw else factory)()
        assert returned.input_spec is None

    def test_apply_validates_a_callable_without_invoking_or_wrapping_it(self):
        def returned(*, x=1.0):
            raise AssertionError("A return contract must not execute the callable")

        factory = Function(
            "factory",
            lambda: returned,
            output_spec=FunctionSpec(input_spec=InputSpec(x=NumericArraySpec(()))),
        )
        assert factory.apply() is returned
        assert factory().signature == inspect.signature(returned)

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
