"""Function input and output specs, independent names, and the value/engine boundary."""

from __future__ import annotations

import ast
import copy
import inspect
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    Beta,
    DistributionSpec,
    EmpiricalDistribution,
    EvaluationError,
    Function,
    FunctionSpec,
    Gamma,
    InputSpec,
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
    ResolutionError,
    ResultSchemaError,
    Uniform,
    WorkflowKind,
    function,
    sample,
    workflow_run,
)
from probpipe.core._fingerprint import fingerprint
from probpipe.core.constraints import interval, positive, real, unit_interval
from probpipe.core.tracked import _NO_DESCRIPTION

#: The executor controls a lift runs under in these tests. ``jax`` maps a
#: traceable function, ``sequential`` and ``thread`` evaluate a function that
#: claims ``traceable=False`` one evaluation point at a time, and the empty
#: controls select the executor automatically.
_JAX = pytest.param({"executor": "jax"}, id="jax")
_SEQUENTIAL = pytest.param({"traceable": False, "executor": "sequential"}, id="sequential")
_THREAD = pytest.param({"traceable": False, "executor": "thread"}, id="thread")
_AUTOMATIC = pytest.param({}, id="automatic")


def _module_level_add(x, y=1.0):
    return x + y


def _unnamed_callables():
    """A partial and a callable instance, neither of which has a ``__name__``."""

    def add(a, b):
        return a + b

    class AddOne:
        def __call__(self, x):
            return x + 1

    return {"partial": partial(add, 1), "instance": AddOne()}


class TestNotation:
    """A function reads as its label followed by its parameters, which ``str()`` returns."""

    def test_a_function_reads_by_its_label_and_its_parameters(self):
        predict = Function(
            lambda x, y: x + y,
            label="predict",
        )
        assert str(predict) == predict.notation == "predict(x, y)"

    def test_a_relabeled_function_reads_by_its_new_label(self):
        assert (
            Function(
                lambda x, y: x + y,
                label="predict",
            )
            .with_label("fit")
            .notation
            == "fit(x, y)"
        )

    def test_the_decorated_function_reads_by_its_name(self):
        @function
        def predict(x, theta=1.0):
            return x * theta

        assert predict.notation == "predict(x, theta=1.0)"

    def test_a_function_of_no_parameters_reads_as_an_empty_call(self):
        assert (
            Function(
                lambda: 1.0,
                label="draw",
            ).notation
            == "draw()"
        )

    def test_the_signature_stays_the_python_signature(self):
        predict = Function(
            lambda x, y: x + y,
            label="predict",
        )
        assert isinstance(predict.signature, inspect.Signature)
        assert list(predict.signature.parameters) == ["x", "y"]

    def test_the_repr_keeps_the_label_first(self):
        predict = Function(
            lambda x, y: x + y,
            label="predict",
        )
        assert repr(predict) == "Function('predict', parameters=('x', 'y'))"


class TestFunctionSpecMatching:
    @pytest.mark.parametrize("from_value", [False, True])
    @pytest.mark.parametrize(
        ("expected", "actual", "message"),
        [
            (
                FunctionSpec(InputSpec(x=NumericArraySpec(()))),
                Function(
                    lambda y: y,
                    input_spec={"y": NumericArraySpec(())},
                    label="actual",
                ),
                "the expected FunctionSpec has the inputs",
            ),
            (
                FunctionSpec(output_spec=OutputSpec(left=None)),
                Function(
                    lambda: 1,
                    output_spec=OutputSpec(right=None),
                    label="actual",
                ),
                "the expected FunctionSpec has the outputs",
            ),
            (
                FunctionSpec(output_spec=OutputSpec(quantity=None)),
                Function(
                    lambda: {"quantity": 1},
                    output_spec=RecordSpec(quantity=()),
                    label="actual",
                ),
                "incompatible outputs",
            ),
            (
                FunctionSpec(output_spec=OutputSpec(RecordSpec(left=()))),
                Function(
                    lambda: {"right": 1},
                    output_spec=RecordSpec(right=()),
                    label="actual",
                ),
                "the expected FunctionSpec has the outputs",
            ),
        ],
        ids=["inputs", "output-names", "keyword-vs-positional", "positional-fields"],
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
            OutputSpec(quantity=NumericArraySpec(("n",))),
        )
        actual = Function(
            lambda x: x,
            input_spec={"x": NumericArraySpec((3,))},
            output_spec=OutputSpec(quantity=NumericArraySpec((3,))),
            label="display",
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
        wrapped = Function(
            lambda: value,
            workflow_kind=WorkflowKind.OFF,
            label="value",
        )
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
        wrapped = Function(
            lambda: 7,
            label="value",
        )
        assert wrapped() == 7
        assert events == ["call", "enter", "exit"]
        assert wrapped.effective_workflow_kind is WorkflowKind.TASK

    def test_without_an_engine_a_call_evaluates_plainly_with_orchestration_off(self, base):
        value = object()
        wrapped = Function(
            lambda: value,
            workflow_kind=WorkflowKind.TASK,
            label="value",
        )
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
            Function(
                lambda: 1,
                **options,
                label="value",
            )

    @pytest.mark.parametrize("output_name", ["", "group/value"])
    def test_invalid_explicit_output_name(self, output_name):
        with pytest.raises(ValueError, match="output names must be non-empty and contain no"):
            Function(
                lambda: 1,
                output_spec=OutputSpec(**{output_name: NumericArraySpec(())}),
                label="value",
            )

    @pytest.mark.parametrize("name", ["class", "lambda", "None"])
    def test_a_keyword_name_infers_the_result_output(self, name):
        def value():
            return 1.0

        value.__name__ = name
        wrapped = Function(value, output_spec=NumericArraySpec(()))
        assert wrapped.label == name
        assert tuple(wrapped.output_spec.outputs) == ("result",)

    @pytest.mark.parametrize("name", ["group/value", "Model.fit", ""])
    def test_a_non_identifier_name_infers_the_result_output(self, name):
        def value():
            return 1.0

        value.__name__ = name
        wrapped = Function(value, output_spec=NumericArraySpec(()), label="value")
        assert tuple(wrapped.output_spec.outputs) == ("result",)

    @pytest.mark.parametrize("label", ["Model.fit", "<lambda>"])
    def test_a_non_identifier_label_leaves_the_inferred_output(self, label):
        wrapped = Function(lambda: 1, output_spec=NumericArraySpec(()), label=label)
        assert tuple(wrapped.output_spec.outputs) == ("f",)
        assert wrapped().label == f"{label}()"

    def test_decorated_lambda_keeps_its_default_output(self):
        wrapped = function(output_spec=NumericArraySpec(()))(lambda: 1)
        assert wrapped.label == "𝒻"
        assert wrapped.output_spec == OutputSpec(f=NumericArraySpec(()))
        assert float(wrapped()) == 1

    @pytest.mark.parametrize("kind", ["partial", "instance"])
    def test_an_unnamed_callable_infers_the_result_output(self, kind):
        wrapped = Function(
            _unnamed_callables()[kind], output_spec=NumericArraySpec(()), label="add1"
        )
        assert tuple(wrapped.output_spec.outputs) == ("result",)
        assert wrapped._output_name == "result"

    def test_a_bound_method_infers_its_method_name(self):
        class Model:
            def fit(self, x):
                return x + 1.0

        wrapped = Function(Model().fit, output_spec=NumericArraySpec(()))
        assert wrapped.label == "fit"
        assert tuple(wrapped.output_spec.outputs) == ("fit",)

    def test_an_undeclared_output_infers_the_callable_name(self):
        def score(x):
            return x + 1.0

        wrapped = Function(score, label="other")
        assert wrapped.output_spec is None
        assert wrapped._output_name == "score"

    def test_with_label_leaves_the_inferred_output(self):
        def score(x):
            return x

        wrapped = Function(score, output_spec=NumericArraySpec(()))
        renamed = wrapped.with_label("other")
        assert renamed.output_spec.outputs == {"score": NumericArraySpec(())}
        assert renamed._output_name == "score"
        assert renamed(4).label == "other(4)"

    @pytest.mark.parametrize("kind", ["partial", "instance"])
    def test_an_unnamed_callable_wraps_under_an_explicit_name(self, kind):
        wrapped = function(label="add1")(_unnamed_callables()[kind])
        assert (wrapped.label, wrapped.output_label) == ("add1", "add1")
        assert float(wrapped(2.0)) == 3.0

    @pytest.mark.parametrize("with_parentheses", [True, False], ids=["called", "bare"])
    @pytest.mark.parametrize("kind", ["partial", "instance"])
    def test_an_unnamed_callable_needs_an_explicit_name(self, kind, with_parentheses):
        decorate = function() if with_parentheses else function
        callable_ = _unnamed_callables()[kind]
        with pytest.raises(
            TypeError,
            match=f"^Function needs an explicit label for a {type(callable_).__name__}, which has "
            r"no name; pass label=\.\.\.$",
        ):
            decorate(callable_)

    def test_required_name_and_raw_representation(self):
        def add(x, /, *, y=2):
            return x + y

        wrapped = Function(
            add,
            label="add",
        )
        assert list(inspect.signature(Function.__init__).parameters)[1:3] == ["fn", "label"]
        assert wrapped.raw is add
        assert wrapped.apply(3) == 5
        assert inspect.signature(wrapped) == inspect.signature(add)
        assert Function(add).label == "add"

    def test_a_label_before_the_callable_names_the_new_form(self):
        def g(x):
            return x

        with pytest.raises(TypeError, match=r"'g'.*Function\(fn, label='g'\)"):
            Function("g", g)

    def test_a_label_before_a_callable_given_as_fn_names_the_new_form(self):
        def g(x):
            return x

        with pytest.raises(TypeError, match=r"'g'.*Function\(fn, label='g'\)"):
            Function("g", fn=g)

    def test_a_label_after_the_callable_names_the_keyword_form(self):
        def g(x):
            return x

        with pytest.raises(
            TypeError,
            match=r"^Function got the string 'lab' as a second positional argument; "
            r"pass the label by keyword: Function\(fn, label='lab'\)$",
        ):
            Function(g, "lab")

    @pytest.mark.parametrize("keyword", ["fn", "func"])
    def test_a_callable_given_by_keyword_names_the_positional_form(self, keyword):
        def g(x):
            return x

        with pytest.raises(
            TypeError,
            match=rf"^Function got the callable as {keyword}=, but takes it as its first "
            r"positional argument; write Function\(fn, label=...\)$",
        ):
            Function(**{keyword: g})
        with pytest.raises(TypeError, match=r"write Function\(fn, label='g'\)$"):
            Function(**{keyword: g, "label": "g"})

    def test_a_function_copies_after_the_argument_check(self):
        wrapped = Function(_module_level_add, label="add")
        assert copy.copy(wrapped).label == "add"

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
        assert renamed.output_spec.outputs == {"mean": None}
        assert result.with_label("display").label == "display"
        assert predict_impl.output_spec.outputs == {"mean": None}

    def test_default_output_name_is_captured_once(self):
        def score(x):
            return x

        wrapped = Function(score, output_spec=NumericArraySpec(()), output_label="shown")
        renamed = wrapped.with_label("other")
        assert renamed.output_label == "shown"
        assert renamed.output_spec.outputs == {"score": NumericArraySpec(())}
        assert renamed(4).label == "shown"

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _JAX, _AUTOMATIC])
    @pytest.mark.parametrize("lift", ["sweep", "broadcast"])
    def test_a_relabeled_lift_records_the_called_function(
        self, execution, lift, full_provenance_mode
    ):
        original = Function(
            (lambda x: x["value"] + 1) if lift == "sweep" else (lambda x: x + 1),
            output_label="prediction",
            output_spec=OutputSpec(quantity=NumericArraySpec(())),
            **execution,
            max_evaluations=8,
            label="predict",
        )
        relabeled = original.with_label("display")
        source = (
            NumericRecordBatch(
                {"value": jnp.arange(3.0)},
                "row",
                element=RecordSpec(value=()),
                label="inputs",
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
            np.testing.assert_array_equal(np.asarray(result.raw), [1.0, 2.0, 3.0])
        else:
            assert tuple(result.event_spec.outputs) == ("quantity",)
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
        returned = Function(
            lambda: 3,
            output_spec=OutputSpec(inner=NumericArraySpec(())),
            label="inner",
        )
        factory = Function(
            lambda: returned,
            output_label="created",
            label="factory",
        )
        result = factory()
        assert isinstance(result, Function)
        assert result.label == result.__name__ == "created"
        assert result.output_label == "created"
        assert result.spec is returned.spec
        assert returned.label == returned.__name__ == "inner"
        assert result().label == "created()"

    def test_raw_callable_return_receives_its_declared_function_spec(self):
        declaration = FunctionSpec(
            InputSpec(x=NumericArraySpec(())), OutputSpec(value=NumericArraySpec(()))
        )
        wrapped = Function(
            lambda: lambda x: x + 1,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="factory"),
            label="factory",
        )
        result = wrapped()
        assert isinstance(result, Function)
        assert result.spec == declaration
        assert float(result(2)) == 3
        with pytest.raises(ApplicabilityError, match="input/x"):
            result(jnp.ones(2))

    def test_returned_array_batch_receives_its_declared_element_spec(self):
        from probpipe import BatchSpec
        from probpipe.core.constraints import positive

        stored = NumericArrayBatch(
            jnp.ones(2),
            "draw",
            element=NumericArraySpec(()),
            label="stored",
        )
        declared = BatchSpec(NumericArraySpec((), support=positive), stored.spec.levels)
        wrapped = Function(
            lambda: stored,
            output_spec=OutputSpec.default(declared, output_name="load"),
            label="load",
        )
        result = wrapped()
        assert wrapped.apply() is stored
        assert result.spec == declared
        assert stored.element_spec.term_spec.support is None

    @pytest.mark.parametrize("invalid_value", [0.0, -1.0])
    @pytest.mark.parametrize("raw", [False, True])
    def test_returned_array_batch_checks_every_value_against_support(self, invalid_value, raw):
        from probpipe import BatchSpec

        values = jnp.array([1.0, invalid_value])
        stored = NumericArrayBatch(
            values,
            "draw",
            element=NumericArraySpec(()),
            label="stored",
        )
        declaration = BatchSpec(NumericArraySpec((), support=positive), stored.spec.levels)
        wrapped = Function(
            lambda: stored,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="load"),
            label="load",
        )
        with pytest.raises(ValueError, match="output/load does not conform to declared support"):
            (wrapped.apply if raw else wrapped)()
        assert stored.element_spec.term_spec.support is None
        assert stored.label == "stored"
        np.testing.assert_array_equal(stored.raw, values)
        assert wrapped.output_spec.term_spec == declaration

    def test_labels_are_outside_spec_equality(self):
        declaration = OutputSpec(mean=NumericArraySpec(()))
        left = Function(
            lambda x: x,
            output_spec=declaration,
            output_label="a",
            label="left",
        )
        right = Function(
            lambda x: x,
            output_spec=declaration,
            output_label="b",
            label="right",
        )
        assert left.spec == right.spec == FunctionSpec(output_spec=declaration)

    @pytest.mark.parametrize("exposed", [True, False])
    def test_single_field_record_keeps_its_kind_and_exposure(self, exposed):
        record_spec = RecordSpec(x=())
        declaration = record_spec if exposed else OutputSpec(bundle=record_spec)
        wrapped = Function(
            lambda x: {"x": x},
            output_spec=declaration,
            label="pack",
        )
        result = wrapped(2)
        assert isinstance(result, Record)
        assert result.label == "pack(2)"
        assert float(result["x"]) == 2
        assert tuple(wrapped.output_spec.outputs) == (("x",) if exposed else ("bundle",))

    def test_an_unspecified_output_type_is_resolved_per_call(self):
        wrapped = Function(
            lambda x: x,
            output_spec=OutputSpec(value=None),
            label="identity",
        )
        assert isinstance(wrapped(3), NumericArray)
        assert isinstance(wrapped("text"), Opaque)
        assert wrapped.output_spec.term_spec is None

    def test_shared_and_output_only_dimensions(self):
        wrapped = Function(
            lambda x: jnp.concatenate([x, x]),
            input_spec={"x": NumericArraySpec(("n",))},
            output_spec=OutputSpec(append=NumericArraySpec(("m",))),
            label="append",
        )
        assert wrapped.input_spec == InputSpec(x=NumericArraySpec(("n",)))
        assert wrapped(jnp.ones(3)).shape == (6,)
        assert wrapped(jnp.ones(2)).shape == (4,)
        assert wrapped.output_spec.term_spec.free_dims == {"m"}

    def test_returned_term_is_copied_and_relabelled(self):
        value = NumericArray(
            jnp.array([1.0, 2.0]),
            label="stored",
        )
        wrapped = Function(
            lambda: value,
            output_label="loaded",
            label="load",
        )
        assert wrapped.apply() is value
        result = wrapped()
        assert result is not value
        assert result.raw is value.raw
        assert result.label == "loaded"
        assert value.label == "stored"

    def test_input_and_output_kinds_are_checked(self):
        wrapped = Function(
            lambda x: x,
            input_spec={"x": NumericArraySpec((2,))},
            output_spec=RecordSpec(x=(2,)),
            label="identity",
        )
        with pytest.raises(ValueError, match="input"):
            wrapped.apply(jnp.ones(3))
        with pytest.raises(ValueError, match="must be a record or a mapping of fields"):
            wrapped.apply(jnp.ones(2))

    def test_options_preserve_declarations(self):
        wrapped = Function(
            lambda x: x,
            output_spec=OutputSpec(value=OpaqueSpec()),
            label="value",
        )
        view = wrapped.using(executor="thread", executor_options={"max_workers": 2})
        assert view.options["executor"] == "thread"
        assert view.options["executor_options"] == {"max_workers": 2}
        assert wrapped.options["executor"] is None
        assert view.spec is wrapped.spec

    def test_base_does_not_import_the_engine(self):
        import probpipe.values._function_base as base

        tree = ast.parse(Path(base.__file__).read_text())
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all("functions" not in ast.unparse(node) for node in imports)


class TestLiftedNames:
    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _JAX, _THREAD])
    def test_sweep_keeps_array_kind_and_result_label(self, execution):
        wrapped = Function(
            lambda x: x["x"] * 2,
            output_label="doubled",
            output_spec=OutputSpec(value=NumericArraySpec(())),
            **execution,
            label="double",
        )
        from probpipe import NumericRecordBatch

        rows = NumericRecordBatch(
            {"x": jnp.arange(3.0)},
            "case",
            element=RecordSpec(x=()),
            label="inputs",
        )
        result = wrapped(rows)
        assert isinstance(result, NumericArrayBatch)
        assert result.label == "doubled"
        np.testing.assert_array_equal(result.raw, [0.0, 2.0, 4.0])

    def test_empty_sweep_uses_declared_array_kind(self):
        wrapped = Function(
            lambda x: x * 2,
            output_label="doubled",
            output_spec=OutputSpec(doubled=NumericArraySpec(())),
            executor="sequential",
            label="double",
        )
        rows = NumericArrayBatch(
            jnp.empty(0),
            "case",
            element=NumericArraySpec(()),
            label="inputs",
        )
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
            unexpected,
            output_label="results",
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="results"),
            executor="sequential",
            label="empty",
        )
        rows = NumericArrayBatch(
            jnp.empty(0),
            "case",
            element=NumericArraySpec(()),
            label="inputs",
        )
        result = wrapped(rows)
        assert type(result).__name__ == kind
        assert result.label == "results"
        assert result.element_spec.term_spec == declaration
        assert result.batch_shape == (0,)

    @pytest.mark.parametrize("declaration", [OutputSpec(value=None), NumericArraySpec(("n",))])
    def test_empty_sweep_cannot_infer_missing_output_type_or_shape(self, declaration):
        wrapped = Function(
            lambda x: x,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="empty"),
            executor="sequential",
            label="empty",
        )
        rows = NumericArrayBatch(
            jnp.empty(0),
            "case",
            element=NumericArraySpec(()),
            label="inputs",
        )
        with pytest.raises(ValueError, match="concrete output_spec"):
            wrapped(rows)

    def test_broadcast_has_independent_label_and_output(self):
        wrapped = Function(
            lambda x: x * 2,
            output_label="doubled",
            output_spec=OutputSpec(value=NumericArraySpec(())),
            executor="sequential",
            max_evaluations=8,
            label="double",
        )
        with workflow_run(seed=1):
            result = wrapped(Normal("x", 0, 1))
        assert result.label == "doubled"
        assert tuple(result.event_spec.outputs) == ("value",)
        assert result.num_atoms == 8

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _AUTOMATIC])
    @pytest.mark.parametrize("renamed", [None, "renamed", "M.pair"])
    def test_list_rows_are_opaque_elements_under_output_name(self, execution, renamed):
        rows = NumericRecordBatch(
            {"x": jnp.arange(3.0)},
            "rows",
            element=RecordSpec(x=()),
            label="inputs",
        )
        wrapped = Function(
            lambda x: [x["x"], x["x"] + 1.0],
            output_label="outs",
            **execution,
            label="pair",
        )
        if renamed is not None:
            wrapped = wrapped.with_label(renamed)
        result = wrapped(rows)
        assert isinstance(result, OpaqueBatch)
        assert result.label == "outs"
        assert result.level_names == ("rows",)
        assert [float(value) for value in result[1].raw] == [1.0, 2.0]

    def test_the_jax_executor_refuses_a_list_row(self):
        rows = NumericRecordBatch(
            {"x": jnp.arange(3.0)},
            "rows",
            element=RecordSpec(x=()),
            label="inputs",
        )
        wrapped = Function(
            lambda x: [x["x"], x["x"] + 1.0],
            executor="jax",
            label="pair",
        )

        with pytest.raises(
            ResolutionError,
            match=r"executor 'jax' cannot run the evaluation points of 'pair': "
            r"JAX cannot trace 'pair': tracing it raised TypeError",
        ):
            wrapped(rows)


class TestLiftedInputDeclarations:
    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _AUTOMATIC])
    def test_a_declared_array_input_lifts_an_empirical_law_over_arrays(self, execution):
        values = jnp.asarray([0.0, 1.0, 2.0])
        law = EmpiricalDistribution(values, output_name="theta")
        predict = Function(
            lambda theta: 2 * theta,
            input_spec={"theta": NumericArraySpec(())},
            **execution,
            label="predict",
            output_spec=OutputSpec(predict=None),
        )
        result = predict(theta=law)
        assert result.num_atoms == 3
        np.testing.assert_array_equal(np.asarray(result.atoms), 2 * values)

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _AUTOMATIC])
    def test_declared_functions_compose_over_a_law(self, execution):
        first = Function(
            lambda theta: theta + 1,
            input_spec={"theta": NumericArraySpec(())},
            **execution,
            max_evaluations=8,
            label="first",
            output_spec=OutputSpec(first=None),
        )
        second = Function(
            lambda x: x * 2,
            input_spec={"x": NumericArraySpec(())},
            **execution,
            label="second",
            output_spec=OutputSpec(second=None),
        )
        with workflow_run(seed=114514):
            intermediate = first(Normal("theta", 0.0, 1.0))
            result = second(intermediate)
        assert intermediate.num_atoms == 8
        assert result.num_atoms == 8
        np.testing.assert_array_equal(np.asarray(result.atoms), 2 * np.asarray(intermediate.atoms))

    def test_a_law_over_one_field_records_lifts_only_into_a_record_input(self):
        """A one-field record stays a record (II.2), so an array input refuses its law."""
        atoms = NumericRecordBatch(
            {"theta": jnp.asarray([0.0, 1.0, 2.0])},
            "draw",
            label="atoms",
        )
        law = EmpiricalDistribution(atoms, label="posterior")
        as_array = Function(
            lambda theta: 2 * theta,
            input_spec={"theta": NumericArraySpec(())},
            executor="sequential",
            label="as_array",
            output_spec=OutputSpec(as_array=None),
        )
        as_record = Function(
            lambda theta: 2 * theta["theta"],
            input_spec={"theta": RecordSpec(theta=())},
            executor="sequential",
            label="as_record",
            output_spec=OutputSpec(as_record=None),
        )
        with pytest.raises(ApplicabilityError, match=r"'theta' accepts NumericArraySpec"):
            as_array(theta=law)
        result = as_record(theta=law)
        np.testing.assert_array_equal(np.asarray(result.atoms), [0.0, 2.0, 4.0])


class TestCompletedOutputDeclarations:
    @pytest.fixture
    def rows(self):
        return NumericRecordBatch(
            {"x": jnp.arange(1.0, 4.0)},
            "case",
            element=RecordSpec(x=()),
            label="inputs",
        )

    @pytest.mark.parametrize("mode", ["plain", "sweep", "broadcast"])
    def test_returned_laws_use_declaration_unification_across_paths(self, rows, mode):
        stored = Gamma("y", jnp.asarray(1.0, dtype="float32"), 1.0)
        declaration = DistributionSpec(
            OutputSpec(y=NumericArraySpec((), dtype="float64", support=real))
        )
        factory = Function(
            lambda x: stored,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="factory"),
            executor="sequential",
            max_evaluations=3,
            label="factory",
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
            assert law.event_spec.outputs["y"].dtype == np.dtype("float32")
            assert law.event_spec.outputs["y"].support == positive

    @pytest.mark.parametrize("mode", ["apply", "plain", "sweep", "broadcast"])
    def test_a_returned_law_outside_the_declared_support_is_refused(self, rows, mode):
        stored = Normal("y", 0.0, 1.0)
        factory = Function(
            lambda x: stored,
            output_spec=OutputSpec(
                factory=DistributionSpec(OutputSpec(y=NumericArraySpec((), support=positive)))
            ),
            executor="sequential",
            max_evaluations=8,
            label="factory",
        )
        operand = {
            "apply": rows[0],
            "plain": rows[0],
            "sweep": rows,
            "broadcast": Normal("x", 0.0, 1.0),
        }[mode]
        invoke = factory.apply if mode == "apply" else factory
        # A lift and a sweep check each evaluation point's output inside the point.
        error = EvaluationError if mode in ("sweep", "broadcast") else ValueError
        with (
            workflow_run(seed=0),
            pytest.raises(
                error, match=r"factory/y support real does not conform to positive"
            ) as raised,
        ):
            invoke(operand)
        if error is EvaluationError:
            assert isinstance(raised.value.__cause__, ResultSchemaError)
        assert stored.event_spec.outputs["y"].support == real

    def test_a_returned_joint_is_checked_field_by_field(self):
        stored = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        declared = RecordSpec(a=NumericArraySpec((), support=positive), b=NumericArraySpec(()))
        factory = Function(
            lambda: stored,
            output_spec=OutputSpec(factory=DistributionSpec(OutputSpec(declared))),
            label="factory",
        )
        with pytest.raises(
            ValueError, match=r"factory/a support real does not conform to positive"
        ):
            factory.apply()

    def test_a_returned_law_on_a_wider_interval_is_refused(self):
        factory = Function(
            lambda: Uniform("y", low=0.0, high=2.0),
            output_spec=OutputSpec(
                factory=DistributionSpec(OutputSpec(y=NumericArraySpec((), support=unit_interval)))
            ),
            label="factory",
        )
        with pytest.raises(
            ValueError,
            match=r"factory/y support interval\(0\.0, 2\.0\) does not conform to unit_interval",
        ):
            factory.apply()

    def test_a_returned_law_on_the_declared_interval_is_accepted(self):
        factory = Function(
            lambda: Uniform("y", low=0.0, high=1.0),
            output_spec=OutputSpec(
                factory=DistributionSpec(OutputSpec(y=NumericArraySpec((), support=unit_interval)))
            ),
            label="factory",
        )
        assert factory.apply().support == interval(0.0, 1.0)

    @pytest.mark.parametrize("container", [list, tuple])
    def test_a_returned_law_conforms_to_declared_sequence_bounds(self, container):
        stored = Beta("x", [2.0, 3.0], [3.0, 4.0])
        factory = Function(
            lambda: stored,
            output_spec=OutputSpec(
                factory=DistributionSpec(
                    OutputSpec(
                        x=NumericArraySpec(
                            (2,), support=interval(container([0.0, 0.0]), container([1.0, 1.0]))
                        )
                    )
                )
            ),
            label="factory",
        )
        assert factory.apply() is stored

    def test_a_returned_law_over_a_whole_record_is_checked_field_by_field(self):
        from probpipe import Distribution

        joint = Normal("y", 0.0, 1.0) * Normal("z", 0.0, 1.0)
        stored = Distribution(
            OutputSpec(bundle=joint.event_spec.term_spec),
            label="bundle",
        )
        declared = RecordSpec(y=NumericArraySpec((), support=positive), z=NumericArraySpec(()))
        factory = Function(
            lambda: stored,
            output_spec=OutputSpec(factory=DistributionSpec(OutputSpec(bundle=declared))),
            label="factory",
        )
        with pytest.raises(
            ValueError, match=r"factory/bundle/y support real does not conform to positive"
        ):
            factory.apply()

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD])
    def test_swept_returned_functions_enforce_the_declared_contract(self, rows, execution):
        stored = Function(
            lambda: -1.0,
            label="inner",
        )
        declaration = FunctionSpec(
            output_spec=OutputSpec(value=NumericArraySpec((), support=positive))
        )
        factory = Function(
            lambda row: stored,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="factory"),
            **execution,
            label="factory",
        )

        single = factory(rows[0])
        batch = factory(rows)
        assert batch.element_spec.term_spec == declaration
        for result in (single, *batch):
            assert result is not stored
            assert result.spec == declaration
            with pytest.raises(ValueError, match="support positive"):
                result()
        assert stored.output_spec is None
        assert factory.apply(rows[0]) is stored

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD])
    def test_swept_existing_arrays_keep_declared_support(self, rows, execution):
        stored = NumericArray(
            jnp.ones(2),
            label="stored",
        )
        declaration = NumericArraySpec((2,), support=positive)
        wrapped = Function(
            lambda row: stored,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="load"),
            **execution,
            label="load",
        )
        result = wrapped(rows)
        # The output spec leaves the dtype open, so completion takes the produced one (II.2).
        completed = NumericArraySpec((2,), dtype=stored.dtype, support=positive)
        assert result.element_spec.term_spec == wrapped(rows[0]).spec == completed
        np.testing.assert_array_equal(result.raw, np.ones((3, 2)))
        assert stored.spec.support is None

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _JAX, _THREAD])
    def test_sweep_completes_output_only_dimensions_per_call(self, rows, execution):
        declaration = RecordSpec(stats=RecordSpec(y=("width",)))
        wrapped = Function(
            lambda row, width: {"stats": {"y": jnp.full((width,), row["x"])}},
            output_spec=declaration,
            **execution,
            label="pack",
        )
        for width in (2, 4):
            result = wrapped(rows, width)
            assert result.element_spec.term_spec == RecordSpec(stats=RecordSpec(y=(width,)))
            np.testing.assert_array_equal(
                result["stats/y"], np.repeat([[1.0], [2.0], [3.0]], width, axis=1)
            )
        assert wrapped.output_spec.term_spec is declaration
        assert declaration.free_dims == {"width"}

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _JAX, _THREAD])
    @pytest.mark.parametrize("kind", ["array", "record", "hole", "record_hole"])
    def test_broadcast_completes_dimensions_and_preserves_output(self, execution, kind):
        declaration = {
            "array": OutputSpec(quantity=NumericArraySpec(("width",))),
            "record": OutputSpec(RecordSpec(quantity=("width",))),
            "hole": OutputSpec(quantity=None),
            "record_hole": OutputSpec(quantity=None),
        }[kind]

        def body(x):
            value = jnp.stack([x, x + 1])
            return {"quantity": value} if kind in ("record", "record_hole") else value

        wrapped = Function(
            body,
            output_label="result",
            output_spec=declaration,
            **execution,
            max_evaluations=8,
            label="f",
        )
        with workflow_run(seed=4):
            joint = wrapped.using(include_inputs=True)(Normal("x", 0, 1))
        assert joint.label == "result"
        assert tuple(joint.event_spec.outputs) == ("x", "quantity")
        column = joint._rows["quantity/quantity" if kind == "record_hole" else "quantity"]
        assert column.shape == (8, 2)
        np.testing.assert_allclose(column[:, 1], column[:, 0] + 1, rtol=0, atol=0)
        assert wrapped.output_spec is declaration
        if kind in ("hole", "record_hole"):
            assert declaration.term_spec is None
        else:
            assert declaration.term_spec.free_dims == {"width"}

    @pytest.mark.parametrize(
        "returned",
        [[1.0, 2.0], (1.0, 2.0), [], jnp.ones(3, dtype="float32"), np.arange(3, dtype="int32")],
        ids=["list", "tuple", "empty-list", "float32-array", "int32-array"],
    )
    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _JAX, _AUTOMATIC])
    def test_an_unspecified_output_type_completes_as_an_undeclared_return_does(
        self, returned, execution
    ):
        """An unspecified output type takes the type the kind-directed wrap gives the return.

        The dtype is included.
        """
        undeclared = Function(
            lambda row: returned,
            output_label="items",
            **execution,
            label="f",
        )
        declared = Function(
            lambda row: returned,
            output_label="items",
            output_spec=OutputSpec(items=None),
            **execution,
            label="f",
        )
        rows = NumericRecordBatch(
            {"x": jnp.arange(2.0)},
            "row",
            element=RecordSpec(x=()),
            label="rows",
        )
        jax_mapped = execution.get("executor") == "jax"
        # A plain call is one evaluation point, which jax never maps.
        cases = [(0, {"executor": None} if jax_mapped else {})]
        if not jax_mapped or not isinstance(returned, (list, tuple)):
            # The jax executor refuses a sequence row.
            cases.append((rows, {}))
        for operand, view_controls in cases:
            expected = undeclared.using(**view_controls)(operand)
            result = declared.using(**view_controls)(operand)
            assert type(result) is type(expected)
            assert result.spec == expected.spec
        assert declared.output_spec.term_spec is None
        assert declared.apply(0) is returned

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _JAX, _AUTOMATIC])
    def test_a_lift_that_includes_its_inputs_completes_the_output_declaration(self, execution):
        factory = Function(
            lambda x: jnp.stack([x, x + 1, x + 2]),
            output_label="results",
            output_spec=OutputSpec(quantity=NumericArraySpec(("width",))),
            **execution,
            max_evaluations=8,
            include_inputs=True,
            label="factory",
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        assert result.label == "results"
        assert result.event_spec.term_spec["quantity"] == NumericArraySpec((3,), dtype="float32")
        assert factory.output_spec.term_spec.free_dims == {"width"}

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _JAX, _AUTOMATIC])
    def test_an_enumerated_lift_that_includes_its_inputs_fills_an_unspecified_type(self, execution):
        factory = Function(
            lambda x: jnp.stack([x, x + 1, x + 2]),
            output_label="results",
            output_spec=OutputSpec(quantity=None),
            **execution,
            include_inputs=True,
            label="factory",
        )
        result = factory(EmpiricalDistribution(jnp.arange(3.0), output_name="x"))
        assert result.label == "results"
        assert result.event_spec.term_spec["quantity"] == NumericArraySpec((3,), dtype="float32")
        assert factory.output_spec.term_spec is None

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _JAX, _AUTOMATIC])
    @pytest.mark.parametrize("exposed", [False, True])
    def test_a_lift_keeps_the_declared_record_exposure(self, execution, exposed):
        record = RecordSpec(field=NumericArraySpec((2,), dtype="float32"))
        declaration = OutputSpec(record) if exposed else OutputSpec(bundle=record)
        factory = Function(
            lambda x: {"field": jnp.stack([x, x + 1])},
            output_label="results",
            output_spec=declaration,
            **execution,
            max_evaluations=8,
            label="factory",
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
            draws = sample(result, sample_shape=(4,))
        assert result.event_spec == declaration
        assert tuple(result.event_spec.outputs) == (("field",) if exposed else ("bundle",))
        if not exposed:
            assert result["bundle"] is result
        column = np.asarray(draws["field"])
        np.testing.assert_allclose(column[:, 1], column[:, 0] + 1, rtol=0, atol=0)
        assert factory.output_spec is declaration

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD, _AUTOMATIC])
    @pytest.mark.parametrize(
        "kind",
        [
            "function",
            "opaque",
            "record",
            "batch",
        ],
    )
    def test_a_lift_keeps_a_non_numeric_output_kind(self, execution, kind):
        stored = {
            "function": lambda: Function(
                lambda x: x + 1,
                label="inner",
            ),
            "opaque": lambda: Opaque(
                "payload",
                spec=OpaqueSpec(meta="text"),
                label="stored",
            ),
            "record": lambda: Record(
                {
                    "field": Opaque(
                        "payload",
                        label="leaf",
                    )
                },
                label="stored",
            ),
            "batch": lambda: NumericArrayBatch(
                jnp.arange(2.0),
                "row",
                element=NumericArraySpec((), dtype=jnp.float32),
                label="stored",
            ),
        }[kind]()
        declaration = OutputSpec(quantity=stored.spec)
        factory = Function(
            lambda x: stored,
            output_label="results",
            output_spec=declaration,
            **execution,
            max_evaluations=8,
            label="factory",
        )
        with workflow_run(seed=4):
            result = factory(Normal("x", 0.0, 1.0))
        assert result.event_spec == declaration
        assert result["quantity"] is result
        assert result.num_atoms == 8
        assert all(type(atom) is type(stored) for atom in result.atoms)
        if kind == "function":
            assert float(result.atoms[0].apply(2)) == 3
        assert stored.label == ("inner" if kind == "function" else "stored")

    @pytest.mark.parametrize("tracked", [False, True])
    def test_an_unspecified_output_type_accepts_a_nested_record(self, tracked):
        value = {"stats": {"mean": 2.0}}
        if tracked:
            value = Record(
                value,
                label="stored",
            )
        wrapped = Function(
            lambda: value,
            output_spec=OutputSpec(bundle=None),
            label="load",
        )
        result = wrapped()
        assert isinstance(result, Record)
        assert float(result["stats/mean"]) == 2.0
        assert wrapped.output_spec.outputs == {"bundle": None}
        assert wrapped.apply() is value

    @pytest.mark.parametrize("tracked", [False, True])
    @pytest.mark.parametrize("raw", [False, True])
    def test_returned_function_declaration_must_match_its_signature(self, tracked, raw):
        def body(x):
            return x

        returned = (
            Function(
                body,
                label="inner",
            )
            if tracked
            else body
        )
        declaration = FunctionSpec(input_spec=InputSpec(y=NumericArraySpec(())))
        factory = Function(
            lambda: returned,
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="factory"),
            label="factory",
        )
        with pytest.raises(
            ValueError,
            match=r"declares the inputs.*in input_spec, but its signature has the parameters",
        ):
            (factory.apply if raw else factory)()
        if tracked:
            assert returned.input_spec is None

    @pytest.mark.parametrize("bound", [False, True])
    @pytest.mark.parametrize("raw", [False, True])
    def test_returned_function_checks_defaults_and_bindings(self, bound, raw):
        default = jnp.ones(2)
        returned = (
            Function(
                lambda x: x,
                bind={"x": default},
                label="inner",
            )
            if bound
            else Function(
                lambda x=default: x,
                label="inner",
            )
        )
        factory = Function(
            lambda: returned,
            output_spec=OutputSpec(
                factory=FunctionSpec(input_spec=InputSpec(x=NumericArraySpec(())))
            ),
            label="factory",
        )
        with pytest.raises(ValueError, match=r"default/x|construction binding/x"):
            (factory.apply if raw else factory)()
        assert returned.input_spec is None

    def test_apply_validates_a_callable_without_invoking_or_wrapping_it(self):
        def returned(*, x=1.0):
            raise AssertionError("A return contract must not execute the callable")

        factory = Function(
            lambda: returned,
            output_spec=OutputSpec(
                factory=FunctionSpec(input_spec=InputSpec(x=NumericArraySpec(())))
            ),
            label="factory",
        )
        assert factory.apply() is returned
        assert factory().signature == inspect.signature(returned)

    @pytest.mark.parametrize("execution", [_SEQUENTIAL, _THREAD])
    def test_nested_lift_carries_completed_output_dimensions(self, rows, execution):
        declaration = RecordSpec(quantity=("width",))
        wrapped = Function(
            lambda row, x: {"quantity": jnp.full((2,), x + row["x"])},
            output_spec=declaration,
            **execution,
            max_evaluations=8,
            label="nested",
        )
        with workflow_run(seed=4):
            result = wrapped(rows, Normal("x", 0, 1))
        assert result.event_spec.term_spec.leaf_shapes == {"quantity": (2,)}
        assert result.batch_shape == (3,)
        for marginal in result:
            assert marginal.event_spec.term_spec == result.event_spec.term_spec
            values = marginal.atoms["quantity"].raw
            np.testing.assert_array_equal(values[:, 0], values[:, 1])
        assert declaration.free_dims == {"width"}

    def test_output_only_dimensions_must_agree_across_sweep_rows(self, rows):
        wrapped = Function(
            lambda row: {"value": jnp.ones(int(row["x"]))},
            output_spec=RecordSpec(value=("width",)),
            executor="sequential",
            label="ragged",
        )
        with pytest.raises(ValueError, match="already bound"):
            wrapped(rows)


@pytest.mark.parametrize("kind", ["array", "record", "batch"])
@pytest.mark.parametrize("mode", ["plain", "sweep"])
def test_a_shape_only_declaration_keeps_the_returned_terms_dtype_and_support(kind, mode):
    """Completion unifies the declared type with the produced one (II.2)."""
    from probpipe import BatchSpec, positive

    leaf = NumericArraySpec((3,), dtype="float32", support=positive)
    values = jnp.ones(3, dtype="float32")
    if kind == "array":
        stored = NumericArray(
            values,
            spec=leaf,
            label="stored",
        )
        declaration = NumericArraySpec((3,))
    elif kind == "record":
        stored = Record(
            {
                "stats": Record(
                    {
                        "x": NumericArray(
                            values,
                            spec=leaf,
                            label="x",
                        )
                    },
                    label="stats",
                )
            },
            label="stored",
        )
        declaration = RecordSpec(stats=RecordSpec(x=NumericArraySpec((3,))))
    else:
        stored = NumericArrayBatch(
            values[None, :],
            "item",
            element=leaf,
            label="stored",
        )
        declaration = BatchSpec(NumericArraySpec((3,)), item=1)
    factory = Function(
        lambda row: stored,
        output_spec=declaration
        if isinstance(declaration, OutputSpec) or declaration is None
        else OutputSpec.default(declaration, output_name="factory"),
        executor="sequential",
        label="factory",
    )
    if mode == "plain":
        actual = factory(0).spec
    else:
        rows = NumericArrayBatch(
            jnp.arange(2.0),
            "row",
            element=NumericArraySpec(()),
            label="rows",
        )
        actual = factory(rows).element_spec.term_spec
    if kind == "record":
        actual = actual["stats/x"]
    elif kind == "batch" and mode == "plain":
        actual = actual.element_spec.term_spec
    assert actual == leaf
    assert factory.output_spec.term_spec is declaration


@pytest.mark.parametrize("mode", ["plain", "sweep", "lift"])
@pytest.mark.parametrize("declared_side", [None, "input", "output"])
def test_a_returned_function_keeps_the_declarations_its_result_leaves_open(mode, declared_side):
    """A side the outer function spec leaves unspecified keeps the returned function's own."""
    from probpipe import EmpiricalDistribution, positive

    inputs = InputSpec(x=NumericArraySpec(()))
    outputs = OutputSpec(value=NumericArraySpec((), support=positive))
    inner = Function(
        lambda x: x,
        input_spec=inputs,
        output_spec=outputs,
        label="inner",
    )
    declaration = FunctionSpec(
        input_spec=inputs if declared_side == "input" else None,
        output_spec=outputs if declared_side == "output" else None,
    )
    factory = Function(
        lambda row: inner,
        output_spec=declaration
        if isinstance(declaration, OutputSpec) or declaration is None
        else OutputSpec.default(declaration, output_name="outer"),
        executor="sequential",
        label="outer",
    )
    if mode == "plain":
        returned = [factory(0)]
    elif mode == "sweep":
        rows = NumericArrayBatch(
            jnp.arange(2.0),
            "row",
            element=NumericArraySpec(()),
            label="rows",
        )
        returned = list(factory(rows))
    else:
        returned = list(factory(EmpiricalDistribution(jnp.arange(2.0), output_name="row")).atoms)
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
        jnp.arange(6.0).reshape(2, 3),
        ("chain", "row"),
        axes_per_level=(1, 1),
        element=NumericArraySpec(()),
        label="pts",
    )
    if sliced:
        stored = stored[1]
    original_label = stored.label
    factory = Function(
        lambda: stored,
        output_label="f",
        label="factory",
    )
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
    np.testing.assert_array_equal(result.raw, stored.raw)


def test_a_returned_function_is_relabeled_with_its_python_names(full_provenance_mode):
    stored = Function(
        lambda: 1,
        output_label="value",
        label="inner",
    )
    factory = Function(
        lambda: stored,
        output_label="result",
        label="factory",
    )
    result = factory()
    assert result.label == result.__name__ == result.__qualname__ == "result"
    assert result.output_label == "value"
    assert result.provenance.parents[0].parent is factory
    assert stored.label == stored.__name__ == "inner"


class TestFunctionLabels:
    def test_function_defaults_and_explicit_aliases(self):
        def predict(temperature):
            return temperature + 1

        assert Function(predict).notation == "predict(temperature)"
        assert Function(lambda temperature: temperature + 1).notation == "𝒻(temperature)"
        assert function(lambda temperature: temperature + 1).notation == "𝒻(temperature)"
        assert Function(predict)(NumericArray(2.0, label="ambient")).label == "predict(ambient)"
        assert Function(predict, output_label="prediction")(2.0).label == "prediction"

    def test_outputs_are_declared_independently_of_aliases(self):
        inferred = Function(lambda x: x, output_spec=NumericArraySpec(()))
        assert tuple(inferred.output_spec.outputs) == ("f",)
        declared = OutputSpec(prediction=NumericArraySpec(()))
        f = Function(
            lambda x: x + 1, label="predict", output_label="forecast", output_spec=declared
        )
        g = Function(f.raw, label="other", output_label="estimate", output_spec=declared)
        assert f.output_spec == g.output_spec
        assert fingerprint(f) == fingerprint(g)
        with workflow_run(seed=42):
            first = f(Normal("temperature", 0, 1))
        with workflow_run(seed=42):
            second = g(Normal("temperature", 0, 1))
        assert tuple(first.event_spec.outputs) == ("prediction",)
        assert tuple(second.event_spec.outputs) == ("prediction",)
        np.testing.assert_array_equal(first.atoms.raw, second.atoms.raw)

    @pytest.mark.parametrize("transform", [jax.jit, jax.vmap], ids=["jit", "vmap"])
    def test_a_function_applies_to_a_rebuilt_array_in_a_transform(self, transform):
        rebuilt = transform(lambda x: x)(NumericArray(jnp.arange(3.0), label="temperature"))
        assert rebuilt.label == _NO_DESCRIPTION
        increment = Function(lambda x: x + 1)
        np.testing.assert_array_equal(
            np.asarray(transform(increment)(rebuilt)), jnp.arange(3.0) + 1
        )

    def test_a_managed_result_derives_its_label_from_a_compiled_callable(self):
        @jax.jit
        def compiled(x):
            return x + 1

        first = NumericArray(1.0, label="temperature")
        wrapped = Function(compiled, label="increment")
        assert wrapped(first).label == "increment(temperature)"
        assert float(wrapped(first)) == 2.0

    def test_lifting_infers_outputs_independently_of_display_aliases(self):
        def predict(x):
            return x + 1

        inferred = Function(predict, max_evaluations=5)
        aliased = Function(predict, label="forecast", output_label="estimate", max_evaluations=5)
        assert fingerprint(inferred) == fingerprint(aliased)
        with workflow_run(seed=42):
            first = inferred(Normal("temperature", 0, 1))
        with workflow_run(seed=42):
            second = aliased.with_label("other")(Normal("temperature", 0, 1))
        assert tuple(first.event_spec.outputs) == ("predict",)
        assert tuple(second.event_spec.outputs) == ("predict",)
        np.testing.assert_array_equal(first.atoms.raw, second.atoms.raw)
        unnamed = Function(lambda x: x + 1, label="predict", max_evaluations=5)
        assert tuple(unnamed(Normal("temperature", 0, 1)).event_spec.outputs) == ("f",)
        exposed = Function(lambda x: {"prediction": x + 1}, max_evaluations=5)
        assert tuple(exposed(Normal("temperature", 0, 1)).event_spec.outputs) == ("prediction",)
