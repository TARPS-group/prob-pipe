"""Function declarations, independent names, and the value/engine boundary."""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
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
    OpaqueBatch,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
    WorkflowKind,
    function,
    sample,
    workflow_method,
    workflow_run,
)
from probpipe.core.constraints import positive, real


def _unnamed_callables():
    """A partial and a callable instance, neither of which has a ``__name__``."""

    def add(a, b):
        return a + b

    class AddOne:
        def __call__(self, x):
            return x + 1

    return {"partial": partial(add, 1), "instance": AddOne()}


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
        for value in (actual, actual.with_label("another")):
            assert expected.bind_dims_from_value(value) == actual.spec
            assert expected.bind_dims_from_spec(value.spec) == actual.spec
        assert expected.free_dims == {"n"}
        assert actual.label == "display"


class TestCallEngineInstallation:
    @pytest.fixture
    def base(self, monkeypatch):
        """The value layer with no engine installed, restored after the test."""
        import probpipe.values._function_base as base

        monkeypatch.setattr(base, "_call_engine", base._plain_call)
        monkeypatch.setattr(base, "_check_engine", base._plain_check)
        monkeypatch.setattr(base, "_apply_scope", nullcontext)
        monkeypatch.setattr(base, "_invoke_engine", base._plain_invoke)
        monkeypatch.setattr(base, "_workflow_kind_engine", base._plain_workflow_kind)
        return base

    @pytest.fixture
    def installation(self, base):
        events = []

        class Engine:
            def __call__(self, function, /, *args, **kwargs):
                events.append("call")
                return function.apply(*args, **kwargs)

            @staticmethod
            @contextmanager
            def apply_scope():
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")

            @staticmethod
            def workflow_kind(function):
                return WorkflowKind.TASK

        engine = Engine()
        base.install_call_engine(engine)
        return base, engine, events

    def test_installing_the_installed_engine_again_changes_nothing(self, installation):
        base, engine, events = installation
        base.install_call_engine(engine)
        value = object()
        wrapped = Function("value", lambda: value, workflow_kind=WorkflowKind.OFF)
        assert wrapped() is value
        assert wrapped.apply() is value
        assert events == ["call", "enter", "exit", "enter", "exit"]
        assert wrapped.effective_workflow_kind is WorkflowKind.TASK

    @pytest.mark.parametrize(
        ("replacement", "error", "message"),
        [
            (lambda function: None, RuntimeError, "already installed"),
            (None, TypeError, "must be callable"),
        ],
        ids=["another-engine", "not-callable"],
    )
    def test_a_refused_installation_keeps_the_installed_engine(
        self, installation, replacement, error, message
    ):
        base, _, events = installation
        with pytest.raises(error, match=message):
            base.install_call_engine(replacement)
        wrapped = Function("value", lambda: 7)
        assert wrapped() == 7
        assert events == ["call", "enter", "exit"]
        assert wrapped.effective_workflow_kind is WorkflowKind.TASK

    def test_without_an_engine_a_call_evaluates_plainly_with_orchestration_off(self, base):
        value = object()
        wrapped = Function("value", lambda: value, workflow_kind=WorkflowKind.TASK)
        assert wrapped.effective_workflow_kind is WorkflowKind.OFF
        assert wrapped() is value


class TestFunctionDeclarations:
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"output_label": ""}, "output_label must be a non-empty string"),
            ({"output_label": 3}, "output_label must be a non-empty string"),
            ({"output_spec": 3}, "output_spec must be an OutputSpec, TermSpec, or None"),
        ],
    )
    def test_invalid_output_options(self, options, message):
        with pytest.raises(TypeError, match=message):
            Function("value", lambda: 1, **options)

    @pytest.mark.parametrize("component", ["", "group/value"])
    def test_invalid_explicit_output_component(self, component):
        with pytest.raises(ValueError, match="component names must be non-empty and contain no"):
            Function(
                "value", lambda: 1, output_spec=OutputSpec(**{component: NumericArraySpec(())})
            )

    def test_invalid_default_output_component_reports_its_name(self):
        with pytest.raises(ValueError, match="got 'group/value'"):
            Function("group/value", lambda: 1, output_spec=NumericArraySpec(()))

    @pytest.mark.parametrize("label", ["Model.fit", "<lambda>"])
    def test_non_identifier_output_component_is_allowed(self, label):
        wrapped = Function(label, lambda: 1, output_spec=NumericArraySpec(()))
        assert tuple(wrapped.output_spec.components) == (label,)
        assert wrapped().label == label

    def test_decorated_lambda_keeps_its_default_component(self):
        wrapped = function(output_spec=NumericArraySpec(()))(lambda: 1)
        assert wrapped.label == "<lambda>"
        assert wrapped.output_spec == OutputSpec(**{"<lambda>": NumericArraySpec(())})
        assert float(wrapped()) == 1

    @pytest.mark.parametrize("kind", ["partial", "instance"])
    def test_an_unnamed_callable_wraps_under_an_explicit_name(self, kind):
        wrapped = function(label="add1")(_unnamed_callables()[kind])
        assert (wrapped.label, wrapped.output_label) == ("add1", "add1")
        assert float(wrapped(2.0)) == 3.0

    @pytest.mark.parametrize("with_parentheses", [True, False], ids=["called", "bare"])
    @pytest.mark.parametrize("kind", ["partial", "instance"])
    def test_an_unnamed_callable_needs_an_explicit_name(self, kind, with_parentheses):
        decorate = function() if with_parentheses else function
        with pytest.raises(TypeError, match="explicit label"):
            decorate(_unnamed_callables()[kind])

    def test_required_name_and_raw_representation(self):
        def add(x, /, *, y=2):
            return x + y

        wrapped = Function("add", add)
        assert list(inspect.signature(Function.__init__).parameters)[1:3] == ["label", "fn"]
        assert wrapped.raw() is add
        assert wrapped.apply(3) == 5
        assert inspect.signature(wrapped) == inspect.signature(add)
        with pytest.raises(TypeError):
            Function(fn=add)

    def test_names_are_independent(self, full_provenance_mode):
        @function(label="predict", output_label="prediction", output_spec=OutputSpec(mean=None))
        def predict_impl(x):
            return x + 1

        renamed = predict_impl.with_label("renamed_predict")
        result = renamed(2)
        assert isinstance(result, NumericArray)
        assert result.label == "prediction"
        assert result.provenance.parents[0].parent is renamed
        assert float(result) == 3
        assert renamed.output_label == "prediction"
        assert renamed.output_spec is predict_impl.output_spec
        assert renamed.output_spec.components == {"mean": None}
        assert result.with_label("display").label == "display"
        assert predict_impl.output_spec.components == {"mean": None}

    def test_default_output_name_is_captured_once(self):
        wrapped = Function("score", lambda x: x, output_spec=NumericArraySpec(()))
        renamed = wrapped.with_label("other")
        assert renamed.output_label == "score"
        assert renamed.output_spec.components == {"score": NumericArraySpec(())}
        assert renamed(4).label == "score"

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    @pytest.mark.parametrize("lift", ["sweep", "broadcast"])
    def test_a_relabeled_lift_records_the_called_function(
        self, dispatch, lift, full_provenance_mode
    ):
        original = Function(
            "predict",
            (lambda x: x["value"] + 1) if lift == "sweep" else (lambda x: x + 1),
            output_label="prediction",
            output_spec=OutputSpec(component=NumericArraySpec(())),
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        relabeled = original.with_label("display")
        source = (
            NumericRecordBatch(
                "inputs", {"value": jnp.arange(3.0)}, "row", element_spec=RecordSpec(value=())
            )
            if lift == "sweep"
            else Normal("x", 0.0, 1.0)
        )

        with workflow_run(seed=0):
            result = relabeled(source)

        assert result.label == "prediction"
        assert result.provenance.parents[0].parent is relabeled
        assert result.provenance.parents[1].parent is source
        assert relabeled.output_spec is original.output_spec
        assert (original.label, relabeled.label) == ("predict", "display")
        assert original.output_label == relabeled.output_label == "prediction"
        if lift == "sweep":
            assert result.level_names == ("row",)
            np.testing.assert_array_equal(np.asarray(result.values), [1.0, 2.0, 3.0])
        else:
            assert tuple(result.event_spec.components) == ("component",)
            assert result.num_atoms == 8

    def test_decorator_can_be_reused_with_its_name_override(self):
        decorate = function(label="shared", output_label="value")
        first = decorate(lambda: 1)
        second = decorate(lambda: 2)
        assert first.label == second.label == "shared"
        assert first.output_label == second.output_label == "value"
        assert float(first()) == 1
        assert float(second()) == 2

    def test_returned_function_keeps_its_own_output_contract(self):
        returned = Function("inner", lambda: 3, output_spec=NumericArraySpec(()))
        factory = Function("factory", lambda: returned, output_label="created")
        result = factory()
        assert isinstance(result, Function)
        assert result.label == result.__name__ == "created"
        assert result.output_label == "inner"
        assert result.spec is returned.spec
        assert returned.label == returned.__name__ == "inner"
        assert result().label == "inner"

    def test_raw_callable_return_receives_its_declared_function_spec(self):
        declaration = FunctionSpec(
            InputSpec(x=NumericArraySpec(())), OutputSpec(value=NumericArraySpec(()))
        )
        wrapped = Function("factory", lambda: lambda x: x + 1, output_spec=declaration)
        result = wrapped()
        assert isinstance(result, Function)
        assert result.spec == declaration
        assert float(result(2)) == 3
        with pytest.raises(ApplicabilityError, match="input/x"):
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

    @pytest.mark.parametrize("invalid_value", [0.0, -1.0])
    @pytest.mark.parametrize("raw", [False, True])
    def test_returned_array_batch_checks_every_value_against_support(self, invalid_value, raw):
        from probpipe import BatchSpec

        values = jnp.array([1.0, invalid_value])
        stored = NumericArrayBatch("stored", values, "draw", element_spec=NumericArraySpec(()))
        declaration = BatchSpec(
            NumericArraySpec((), support=positive), stored.axis_groups, stored.level_names
        )
        wrapped = Function("load", lambda: stored, output_spec=declaration)
        with pytest.raises(ValueError, match="output/load does not conform to declared support"):
            (wrapped.apply if raw else wrapped)()
        assert stored.element_spec.support is None
        assert stored.label == "stored"
        np.testing.assert_array_equal(stored.values, values)
        assert wrapped.output_spec.spec == declaration

    def test_labels_are_outside_spec_equality(self):
        declaration = OutputSpec(mean=NumericArraySpec(()))
        left = Function("left", lambda x: x, output_spec=declaration, output_label="a")
        right = Function("right", lambda x: x, output_spec=declaration, output_label="b")
        assert left.spec == right.spec == FunctionSpec(output_spec=declaration)

    @pytest.mark.parametrize("exposed", [True, False])
    def test_single_field_record_keeps_its_kind_and_exposure(self, exposed):
        record_spec = RecordSpec(x=())
        declaration = record_spec if exposed else OutputSpec(bundle=record_spec)
        wrapped = Function("pack", lambda x: {"x": x}, output_spec=declaration)
        result = wrapped(2)
        assert isinstance(result, Record)
        assert result.label == "pack"
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
        wrapped = Function("load", lambda: value, output_label="loaded")
        assert wrapped.apply() is value
        result = wrapped()
        assert result is not value
        assert result.value is value.value
        assert result.label == "loaded"
        assert value.label == "stored"

    def test_input_and_output_kinds_are_checked(self):
        wrapped = Function(
            "identity",
            lambda x: x,
            input_spec={"x": NumericArraySpec((2,))},
            output_spec=RecordSpec(x=(2,)),
        )
        with pytest.raises(ValueError, match="input"):
            wrapped.apply(jnp.ones(3))
        with pytest.raises(ValueError, match="must be a record or a mapping of fields"):
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
            output_label="doubled",
            output_spec=OutputSpec(value=NumericArraySpec(())),
            dispatch=dispatch,
        )
        from probpipe import NumericRecordBatch

        rows = NumericRecordBatch(
            "inputs", {"x": jnp.arange(3.0)}, "case", element_spec=RecordSpec(x=())
        )
        result = wrapped(rows)
        assert isinstance(result, NumericArrayBatch)
        assert result.label == "doubled"
        np.testing.assert_array_equal(result.values, [0.0, 2.0, 4.0])

    def test_empty_sweep_uses_declared_array_kind(self):
        wrapped = Function(
            "double",
            lambda x: x * 2,
            output_label="doubled",
            output_spec=NumericArraySpec(()),
            dispatch="sequential",
        )
        rows = NumericArrayBatch("inputs", jnp.empty(0), "case", element_spec=NumericArraySpec(()))
        result = wrapped(rows)
        assert isinstance(result, NumericArrayBatch)
        assert result.label == "doubled"
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
            output_label="results",
            output_spec=declaration,
            dispatch="sequential",
        )
        rows = NumericArrayBatch("inputs", jnp.empty(0), "case", element_spec=NumericArraySpec(()))
        result = wrapped(rows)
        assert type(result).__name__ == kind
        assert result.label == "results"
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
            output_label="doubled",
            output_spec=OutputSpec(value=NumericArraySpec(())),
            dispatch="sequential",
            n_broadcast_samples=8,
        )
        with workflow_run(seed=1):
            result = wrapped(Normal("x", 0, 1))
        assert result.label == "doubled"
        assert tuple(result.event_spec.components) == ("value",)
        assert result.num_atoms == 8

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    @pytest.mark.parametrize("renamed", [None, "renamed", "M.pair"])
    def test_list_rows_are_opaque_elements_under_output_name(self, dispatch, renamed):
        rows = NumericRecordBatch(
            "inputs", {"x": jnp.arange(3.0)}, "rows", element_spec=RecordSpec(x=())
        )
        wrapped = Function(
            "pair",
            lambda x: [x["x"], x["x"] + 1.0],
            output_label="outs",
        )
        if renamed is not None:
            wrapped = wrapped.with_label(renamed)
        result = wrapped.with_options(dispatch=dispatch)(rows)
        assert isinstance(result, OpaqueBatch)
        assert result.label == "outs"
        assert result.level_names == ("rows",)
        assert [float(value) for value in result[1].value] == [1.0, 2.0]

    def test_the_mapped_dispatch_refuses_a_list_row(self):
        rows = NumericRecordBatch(
            "inputs", {"x": jnp.arange(3.0)}, "rows", element_spec=RecordSpec(x=())
        )
        wrapped = Function("pair", lambda x: [x["x"], x["x"] + 1.0], dispatch="jax")

        with pytest.raises(ValueError, match="dispatch='jax'"):
            wrapped(rows)


class TestLiftedInputDeclarations:
    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    def test_a_declared_array_input_lifts_an_empirical_law_over_arrays(self, dispatch):
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
        np.testing.assert_array_equal(np.asarray(result.atoms), 2 * values)

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    def test_declared_functions_compose_over_a_law(self, dispatch):
        first = Function(
            "first",
            lambda theta: theta + 1,
            input_spec={"theta": NumericArraySpec(())},
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        second = Function(
            "second",
            lambda x: x * 2,
            input_spec={"x": NumericArraySpec(())},
            dispatch=dispatch,
        )
        with workflow_run(seed=114514):
            intermediate = first(Normal("theta", 0.0, 1.0))
            result = second(intermediate)
        assert intermediate.num_atoms == 8
        assert result.num_atoms == 8
        np.testing.assert_array_equal(np.asarray(result.atoms), 2 * np.asarray(intermediate.atoms))

    def test_a_law_over_one_field_records_lifts_only_into_a_record_slot(self):
        """A one-field record stays a record (II.2), so an array slot refuses its law."""
        atoms = NumericRecordBatch("atoms", {"theta": jnp.asarray([0.0, 1.0, 2.0])}, "draw")
        law = EmpiricalDistribution("posterior", atoms)
        as_array = Function(
            "as_array",
            lambda theta: 2 * theta,
            input_spec={"theta": NumericArraySpec(())},
            dispatch="sequential",
        )
        as_record = Function(
            "as_record",
            lambda theta: 2 * theta["theta"],
            input_spec={"theta": RecordSpec(theta=())},
            dispatch="sequential",
        )
        with pytest.raises(ApplicabilityError, match=r"'theta' accepts NumericArraySpec"):
            as_array(theta=law)
        result = as_record(theta=law)
        np.testing.assert_array_equal(np.asarray(result.atoms), [0.0, 2.0, 4.0])


class TestCompletedOutputDeclarations:
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
        if mode == "plain":
            laws = (result,)
        elif mode == "sweep":
            laws = tuple(result)
        else:
            laws = tuple(result.atoms)
        for law in laws:
            assert law.spec is stored.spec
            assert law.event_spec.components["y"].dtype == np.dtype("float32")
            assert law.event_spec.components["y"].support == positive

    @pytest.mark.parametrize("mode", ["apply", "plain", "sweep", "broadcast"])
    def test_a_returned_law_outside_the_declared_support_is_refused(self, rows, mode):
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
            pytest.raises(ValueError, match=r"factory/y support real does not conform to positive"),
        ):
            invoke(operand)
        assert stored.event_spec.components["y"].support == real

    def test_a_returned_joint_is_checked_field_by_field(self):
        stored = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        declared = RecordSpec(a=NumericArraySpec((), support=positive), b=NumericArraySpec(()))
        factory = Function(
            "factory", lambda: stored, output_spec=DistributionSpec(OutputSpec(declared))
        )
        with pytest.raises(
            ValueError, match=r"factory/a support real does not conform to positive"
        ):
            factory.apply()

    def test_a_returned_law_over_a_whole_record_is_checked_field_by_field(self):
        from probpipe import Distribution

        joint = Normal("y", 0.0, 1.0) * Normal("z", 0.0, 1.0)
        stored = Distribution("bundle", OutputSpec(bundle=joint.event_spec.spec))
        declared = RecordSpec(y=NumericArraySpec((), support=positive), z=NumericArraySpec(()))
        factory = Function(
            "factory", lambda: stored, output_spec=DistributionSpec(OutputSpec(bundle=declared))
        )
        with pytest.raises(
            ValueError, match=r"factory/bundle/y support real does not conform to positive"
        ):
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
        # The declaration leaves the dtype open, so completion takes the produced one (II.2).
        completed = NumericArraySpec((2,), dtype=stored.dtype, support=positive)
        assert result.element_spec == wrapped(rows[0]).spec == completed
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
            output_label="result",
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            joint = wrapped.with_options(include_inputs=True)(Normal("x", 0, 1))
        assert joint.label == "result"
        assert tuple(joint.event_spec.components) == ("x", "component")
        column = joint._rows["component/component" if kind == "record_hole" else "component"]
        assert column.shape == (8, 2)
        np.testing.assert_allclose(column[:, 1], column[:, 0] + 1, rtol=0, atol=0)
        assert wrapped.output_spec is declaration
        if kind in ("hole", "record_hole"):
            assert declaration.spec is None
        else:
            assert declaration.spec.free_dims == {"width"}

    @pytest.mark.parametrize(
        "returned",
        [[1.0, 2.0], (1.0, 2.0), [], jnp.ones(3, dtype="float32"), np.arange(3, dtype="int32")],
        ids=["list", "tuple", "empty-list", "float32-array", "int32-array"],
    )
    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    def test_a_type_hole_completes_as_an_undeclared_return_does(self, returned, dispatch):
        """A type hole takes the type the kind-directed wrap gives the return, dtype included."""
        undeclared = Function("f", lambda row: returned, output_label="items", dispatch=dispatch)
        declared = Function(
            "f",
            lambda row: returned,
            output_label="items",
            output_spec=OutputSpec(items=None),
            dispatch=dispatch,
        )
        rows = NumericRecordBatch(
            "rows", {"x": jnp.arange(2.0)}, "row", element_spec=RecordSpec(x=())
        )
        operands = [0]
        if dispatch != "jax" or not isinstance(returned, (list, tuple)):
            # The mapped dispatch refuses a sequence row.
            operands.append(rows)
        for operand in operands:
            expected = undeclared(operand)
            result = declared(operand)
            assert type(result) is type(expected)
            assert result.spec == expected.spec
        assert declared.output_spec.spec is None
        assert declared.apply(0) is returned

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    def test_a_lift_that_includes_its_inputs_completes_the_output_declaration(self, dispatch):
        factory = Function(
            "factory",
            lambda x: jnp.stack([x, x + 1, x + 2]),
            output_label="results",
            output_spec=OutputSpec(component=NumericArraySpec(("width",))),
            dispatch=dispatch,
            n_broadcast_samples=8,
            include_inputs=True,
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        assert result.label == "results"
        assert result.event_spec.spec["component"] == NumericArraySpec((3,), dtype="float32")
        assert factory.output_spec.spec.free_dims == {"width"}

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    def test_an_enumerated_lift_that_includes_its_inputs_fills_a_type_hole(self, dispatch):
        factory = Function(
            "factory",
            lambda x: jnp.stack([x, x + 1, x + 2]),
            output_label="results",
            output_spec=OutputSpec(component=None),
            dispatch=dispatch,
            include_inputs=True,
        )
        result = factory(EmpiricalDistribution("x", jnp.arange(3.0)))
        assert result.label == "results"
        assert result.event_spec.spec["component"] == NumericArraySpec((3,), dtype="float32")
        assert factory.output_spec.spec is None

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax", "auto"])
    @pytest.mark.parametrize("exposed", [False, True])
    def test_a_lift_keeps_the_declared_record_exposure(self, dispatch, exposed):
        record = RecordSpec(field=NumericArraySpec((2,), dtype="float32"))
        declaration = OutputSpec(record) if exposed else OutputSpec(bundle=record)
        factory = Function(
            "factory",
            lambda x: {"field": jnp.stack([x, x + 1])},
            output_label="results",
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
            draws = sample(result, sample_shape=(4,))
        assert result.event_spec == declaration
        assert tuple(result.event_spec.components) == (("field",) if exposed else ("bundle",))
        if not exposed:
            assert result["bundle"] is result
        column = np.asarray(draws["field"])
        np.testing.assert_allclose(column[:, 1], column[:, 0] + 1, rtol=0, atol=0)
        assert factory.output_spec is declaration

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    @pytest.mark.parametrize(
        "kind",
        [
            "function",
            "opaque",
            "record",
            pytest.param(
                "batch",
                marks=pytest.mark.pending(
                    reason="the empirical law of a lifted function that returns a batch"
                ),
            ),
        ],
    )
    def test_a_lift_keeps_a_non_numeric_component_kind(self, dispatch, kind):
        stored = {
            "function": lambda: Function("inner", lambda x: x + 1),
            "opaque": lambda: Opaque("stored", "payload", spec=OpaqueSpec(meta="text")),
            "record": lambda: Record("stored", field=Opaque("leaf", "payload")),
            "batch": lambda: NumericArrayBatch(
                "stored", jnp.arange(2.0), "row", element_spec=NumericArraySpec(())
            ),
        }[kind]()
        declaration = OutputSpec(component=stored.spec)
        factory = Function(
            "factory",
            lambda x: stored,
            output_label="results",
            output_spec=declaration,
            dispatch=dispatch,
            n_broadcast_samples=8,
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        assert result.event_spec == declaration
        assert result["component"] is result
        assert result.num_atoms == 8
        assert all(type(atom) is type(stored) for atom in result.atoms)
        if kind == "function":
            assert float(result.atoms[0].apply(2)) == 3
        assert stored.label == ("inner" if kind == "function" else "stored")

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
            values = marginal.atoms["component"].raw()
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
        assert method.label == "Example.numbers"
        assert result.label == method.output_label == "numbers"
        assert type(result) is type(expected)
        assert result.spec == expected.spec
        assert result.value == expected.value == sequence


@pytest.mark.parametrize("kind", ["array", "record", "batch"])
@pytest.mark.parametrize("mode", ["plain", "sweep"])
def test_a_shape_only_declaration_keeps_the_returned_terms_dtype_and_support(kind, mode):
    """Completion unifies the declared type with the produced one (II.2)."""
    from probpipe import BatchSpec, positive

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
    factory = Function(
        "factory", lambda row: stored, output_spec=declaration, dispatch="sequential"
    )
    if mode == "plain":
        actual = factory(0).spec
    else:
        rows = NumericArrayBatch("rows", jnp.arange(2.0), "row", element_spec=NumericArraySpec(()))
        actual = factory(rows).element_spec
    if kind == "record":
        actual = actual["stats/x"]
    elif kind == "batch" and mode == "plain":
        actual = actual.element_spec
    assert actual == leaf
    assert factory.output_spec.spec is declaration


@pytest.mark.parametrize("mode", ["plain", "sweep", "lift"])
@pytest.mark.parametrize("declared_side", [None, "input", "output"])
def test_a_returned_function_keeps_the_declarations_its_result_leaves_open(mode, declared_side):
    """A side the outer declaration leaves unspecified keeps the returned function's own."""
    from probpipe import EmpiricalDistribution, positive

    inputs = InputSpec(x=NumericArraySpec(()))
    outputs = OutputSpec(value=NumericArraySpec((), support=positive))
    inner = Function("inner", lambda x: x, input_spec=inputs, output_spec=outputs)
    declaration = FunctionSpec(
        input_spec=inputs if declared_side == "input" else None,
        output_spec=outputs if declared_side == "output" else None,
    )
    factory = Function("outer", lambda row: inner, output_spec=declaration, dispatch="sequential")
    if mode == "plain":
        returned = [factory(0)]
    elif mode == "sweep":
        rows = NumericArrayBatch("rows", jnp.arange(2.0), "row", element_spec=NumericArraySpec(()))
        returned = list(factory(rows))
    else:
        returned = list(factory(EmpiricalDistribution("row", jnp.arange(2.0))).atoms)
    for result in returned:
        assert result is not inner
        assert result.spec == inner.spec
        with pytest.raises(ValueError, match="support positive"):
            result.apply(-1.0)
    assert inner.input_spec is inputs
    assert inner.output_spec is outputs


@pytest.mark.parametrize("sliced", [False, True])
def test_a_returned_batch_is_relabeled_as_the_root_of_its_views(sliced, full_provenance_mode):
    stored = NumericArrayBatch(
        "pts",
        jnp.arange(6.0).reshape(2, 3),
        ("chain", "row"),
        axes_per_level=(1, 1),
        element_spec=NumericArraySpec(()),
    )
    if sliced:
        stored = stored[1]
    original_label = stored.label
    factory = Function("factory", lambda: stored, output_label="f")
    result = factory()
    assert result.label == "f"
    assert result[0].label == ("f[row=0]" if sliced else "f[chain=0]")
    if not sliced:
        assert result[0][1].label == "f[chain=0, row=1]"
    assert result.provenance.parents[0].parent is factory
    assert stored.label == original_label
    assert stored.provenance is None
    relabeled = result.with_label("display")
    assert relabeled[0].label.startswith("display[")
    assert relabeled.provenance.operation == "with_label"
    np.testing.assert_array_equal(result.values, stored.values)


def test_a_returned_function_is_relabeled_with_its_python_names(full_provenance_mode):
    stored = Function("inner", lambda: 1, output_label="value")
    factory = Function("factory", lambda: stored, output_label="result")
    result = factory()
    assert result.label == result.__name__ == result.__qualname__ == "result"
    assert result.output_label == "value"
    assert result.provenance.parents[0].parent is factory
    assert stored.label == stored.__name__ == "inner"
