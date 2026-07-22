"""Contracts for first-class, schema-aware Function values."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import (
    Annotated,
    ArraySpec,
    Distribution,
    DistributionArray,
    EventTemplate,
    Function,
    FunctionSpec,
    Gamma,
    Normal,
    NumericRecord,
    NumericRecordArray,
    Provenance,
    ProvenanceMode,
    Record,
    Tracked,
    function,
    mean,
    positive,
    positive_definite,
    real,
    simplex,
)


class TestFunctionValueContract:
    def test_function_is_tracked_annotated_and_immutable(self):
        def increment(x):
            return x + 1

        wrapped = Function(func=increment)

        assert isinstance(wrapped, Tracked)
        assert isinstance(wrapped, Annotated)
        assert wrapped.name == "increment"
        assert wrapped.name_is_auto
        assert wrapped.provenance is None
        assert wrapped.annotations == {}
        wrapped.annotations["note"] = "append-only metadata"
        assert wrapped.annotations == {"note": "append-only metadata"}
        with pytest.raises(AttributeError, match="immutable"):
            wrapped._seed = 3

    def test_decorator_name_is_auto_but_explicit_name_is_not(self):
        @function
        def automatic(x):
            return x

        named = Function(func=lambda x: x, name="chosen")

        assert automatic.name == "automatic"
        assert automatic.name_is_auto
        assert named.name == "chosen"
        assert not named.name_is_auto

    def test_rename_synchronizes_callable_metadata_and_records_provenance(
        self, full_provenance_mode
    ):
        wrapped = Function(func=lambda x: x)

        renamed = wrapped.with_name("identity")

        assert renamed.name == renamed.__name__ == renamed.__qualname__ == "identity"
        assert inspect.signature(renamed) == wrapped.signature
        assert renamed.provenance is not None
        assert renamed.provenance.parents[0].parent is wrapped

    def test_provenance_is_write_once(self):
        wrapped = Function(func=lambda x: x)
        wrapped.with_provenance(Provenance("trained"))

        with pytest.raises(RuntimeError, match="write-once"):
            wrapped.with_provenance(Provenance("retrained"))

    def test_signature_is_captured_independently_once(self):
        def affine(x, /, scale=2, *, offset=1):
            return x * scale + offset

        wrapped = Function(func=affine)
        captured = wrapped.signature
        affine.__signature__ = inspect.Signature()  # type: ignore[attr-defined]

        assert wrapped.signature is captured
        assert inspect.signature(wrapped) is captured


class TestApplyContract:
    def test_apply_preserves_python_parameter_kinds_and_returns_raw(self):
        def affine(x, /, scale=2, *, offset=1):
            return x * scale + offset

        wrapped = Function(func=affine, output_template=EventTemplate(answer=()))

        assert wrapped.apply(3, offset=4) == 10
        result = wrapped(3, offset=4)
        assert isinstance(result, Record)
        assert result["answer"] == 10

    def test_call_records_positional_and_keyword_only_inputs(self, full_provenance_mode):
        def combine(x, /, y=2.0, *, scale=3.0):
            return (x + y) * scale

        wrapped = Function(func=combine)

        result = wrapped(1.0)

        assert result.provenance is not None
        assert tuple(result.provenance.inputs) == ("x", "y", "scale")
        assert result.provenance.inputs["x"].parent == 1.0
        assert result.provenance.inputs["y"].parent == 2.0
        assert result.provenance.inputs["scale"].parent == 3.0

    def test_variadic_apply_without_template(self):
        def collect(first, *rest, **extras):
            return first + sum(rest) + sum(extras.values())

        wrapped = Function(func=collect)

        assert wrapped.apply(1, 2, 3, bonus=4) == 10
        assert float(wrapped(1, 2, 3, bonus=4)) == 10

    def test_parameter_named_self_is_not_treated_as_a_method_receiver(self):
        wrapped = Function(
            func=lambda self: self + 1,
            input_template=EventTemplate(self=()),
            output_template=EventTemplate(result=()),
        )

        assert wrapped.apply(2) == 3
        assert float(wrapped(2)) == 3

    def test_construction_bindings_support_positional_only_and_var_keyword(self):
        def collect(x, /, **extras):
            return x + sum(extras.values())

        wrapped = Function(func=collect, bind={"x": 2}, bonus=3)

        assert wrapped.apply() == 5
        assert wrapped.apply(bonus=4) == 6

    def test_apply_validates_output_without_wrapping(self):
        wrapped = Function(
            func=lambda x: np.ones((x,)),
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(y=(3,)),
        )

        with pytest.raises(ValueError, match="output at 'y'"):
            wrapped.apply(2)

    def test_dtype_pinned_output_accepts_bare_and_inferred_record(self):
        template = EventTemplate(y=ArraySpec((), dtype="float32"))
        value = jnp.asarray(3.0, dtype=jnp.float32)
        returned = Record("returned", y=value)
        bare = Function(func=lambda: value, output_template=template)
        structured = Function(func=lambda: returned, output_template=template)

        assert bare.apply() is value
        assert structured.apply() is returned
        assert returned.event_template != template

        result = structured()

        assert result is not returned
        assert result["y"] is value
        assert result.event_template == template
        assert returned.event_template != template

    @pytest.mark.parametrize(
        ("actual_dtype", "declared_dtype"),
        [
            ("float32", "float64"),
            ("float64", "float32"),
            ("int32", "int64"),
            ("int64", "int32"),
        ],
    )
    def test_same_kind_output_dtype_conformance_is_wrapper_independent(
        self,
        actual_dtype,
        declared_dtype,
    ):
        value = np.asarray(3, dtype=actual_dtype)
        template = EventTemplate(y=ArraySpec((), dtype=declared_dtype))

        Function(func=lambda: value, output_template=template).apply()
        Function(func=lambda: Record("returned", y=value), output_template=template).apply()

    @pytest.mark.parametrize("structured", [False, True])
    def test_cross_kind_output_dtype_is_rejected(self, structured):
        value = np.asarray(3.0, dtype="float32")
        returned = Record("returned", y=value) if structured else value
        wrapped = Function(
            func=lambda: returned,
            output_template=EventTemplate(y=ArraySpec((), dtype="int32")),
        )

        with pytest.raises(ValueError, match=r"output/y.*dtype|output at 'y'"):
            wrapped.apply()

    def test_support_pinned_scalar_output_is_enforced(self):
        template = EventTemplate(y=ArraySpec((), support=positive))

        assert Function(func=lambda: jnp.asarray(3.0), output_template=template).apply() == 3.0
        with pytest.raises(ValueError, match=r"output at 'y'.*support positive"):
            Function(func=lambda: jnp.asarray(-3.0), output_template=template).apply()

    def test_support_pinned_nested_mapping_output_is_enforced(self):
        template = EventTemplate(
            stats=EventTemplate(y=ArraySpec((2,), support=positive)),
        )
        wrapped = Function(
            func=lambda value: {"stats": {"y": value}},
            output_template=template,
        )

        wrapped.apply(jnp.asarray([1.0, 2.0]))
        with pytest.raises(ValueError, match=r"output at 'stats/y'.*support positive"):
            wrapped.apply(jnp.asarray([1.0, -2.0]))

    def test_support_pinned_nested_record_output_is_enforced(self):
        template = EventTemplate(
            stats=EventTemplate(y=ArraySpec((2,), support=positive)),
        )
        valid = Record("returned", stats=Record("stats", y=jnp.asarray([1.0, 2.0])))
        invalid = Record("returned", stats=Record("stats", y=jnp.asarray([1.0, -2.0])))

        Function(func=lambda: valid, output_template=template).apply()
        with pytest.raises(ValueError, match=r"output at 'stats/y'.*support positive"):
            Function(func=lambda: invalid, output_template=template).apply()

    def test_explicit_record_support_must_conform_to_declaration(self):
        returned = Record(
            "returned",
            y=jnp.asarray(3.0),
            event_template=EventTemplate(y=ArraySpec((), support=real)),
        )
        wrapped = Function(
            func=lambda: returned,
            output_template=EventTemplate(y=ArraySpec((), support=positive)),
        )

        with pytest.raises(ValueError, match=r"output/y support real does not conform to positive"):
            wrapped.apply()

    def test_distribution_metadata_must_conform_to_declaration(self):
        normal = Normal(0, 1, name="y")
        declared = EventTemplate(
            y=ArraySpec((), dtype=normal.dtypes["y"], support=real),
        )
        wrapped = Function(func=lambda: normal, output_template=declared)

        assert wrapped.apply() is normal

        result = wrapped()

        assert result is not normal
        assert result.event_template == declared
        assert normal.event_template != declared

        with pytest.raises(ValueError, match=r"output/y support real does not conform to positive"):
            Function(
                func=lambda: normal,
                output_template=EventTemplate(y=ArraySpec((), support=positive)),
            ).apply()
        with pytest.raises(ValueError, match=r"output/y dtype .* does not conform to int32"):
            Function(
                func=lambda: normal,
                output_template=EventTemplate(y=ArraySpec((), dtype="int32")),
            ).apply()

        gamma = Gamma(1, 1, name="y")
        Function(
            func=lambda: gamma,
            output_template=EventTemplate(y=ArraySpec((), support=real)),
        ).apply()

    def test_distribution_missing_declared_metadata_is_rejected(self):
        class MetadataPoorDistribution(Distribution):
            @property
            def event_template(self):
                return EventTemplate(y=())

        returned = MetadataPoorDistribution(name="y")

        with pytest.raises(ValueError, match=r"output at 'y'.*authoritative dtype metadata"):
            Function(
                func=lambda: returned,
                output_template=EventTemplate(y=ArraySpec((), dtype="float32")),
            ).apply()
        with pytest.raises(ValueError, match=r"output at 'y'.*authoritative support metadata"):
            Function(
                func=lambda: returned,
                output_template=EventTemplate(y=ArraySpec((), support=positive)),
            ).apply()

    @pytest.mark.parametrize(
        ("support", "valid", "invalid"),
        [
            (simplex, jnp.asarray([0.25, 0.75]), jnp.asarray([0.25, 0.5])),
            (
                positive_definite,
                jnp.asarray([[2.0, 0.0], [0.0, 1.0]]),
                jnp.asarray([[1.0, 2.0], [2.0, 1.0]]),
            ),
        ],
    )
    def test_output_support_reductions_are_enforced(self, support, valid, invalid):
        template = EventTemplate(y=ArraySpec(valid.shape, support=support))

        Function(func=lambda: valid, output_template=template).apply()
        with pytest.raises(ValueError, match=r"output at 'y'.*declared support"):
            Function(func=lambda: invalid, output_template=template).apply()

    def test_support_validation_rejects_direct_jax_jit(self):
        wrapped = Function(
            func=lambda x: x + 1,
            output_template=EventTemplate(y=ArraySpec((), support=positive)),
        )

        with pytest.raises(ValueError, match="cannot run during JAX tracing"):
            jax.jit(wrapped.apply)(jnp.asarray(1.0))

    def test_input_template_support_remains_descriptive_for_lifting(self):
        class SupportAnnotatedNormal(Normal):
            @property
            def event_template(self):
                return EventTemplate(x=ArraySpec((), support=real))

        wrapped = Function(
            func=lambda x: x,
            input_template=EventTemplate(x=ArraySpec((), support=positive)),
            dispatch="sequential",
            n_broadcast_samples=5,
            seed=0,
        )

        result = wrapped(SupportAnnotatedNormal(0, 1, name="x"))

        assert result.num_atoms == 5

    def test_authoritative_nested_mapping_must_match_exactly(self):
        template = EventTemplate(
            stats=EventTemplate(copy=("obs",), total=()),
        )
        wrapped = Function(
            func=lambda x: {"stats": {"copy": x, "total": x.sum()}},
            input_template=EventTemplate(x=("obs",)),
            output_template=template,
        )

        result = wrapped(np.ones((3,)))

        assert result.event_template == EventTemplate(stats=EventTemplate(copy=(3,), total=()))
        with pytest.raises(ValueError, match="do not match template fields"):
            Function(
                func=lambda x: {"stats": {"copy": x}},
                input_template=EventTemplate(x=("obs",)),
                output_template=template,
            ).apply(np.ones((3,)))

    def test_raw_result_cannot_satisfy_multi_leaf_output_template(self):
        wrapped = Function(
            func=lambda x: x,
            output_template=EventTemplate(left=(), right=()),
        )

        with pytest.raises(ValueError, match="scalar/array for a multi-field"):
            wrapped.apply(1)

    def test_existing_record_requires_matching_authoritative_template(self):
        wrapped = Function(
            func=lambda x: Record("result", wrong=x),
            output_template=EventTemplate(expected=()),
        )

        with pytest.raises(ValueError, match=r"fields .* do not match template fields"):
            wrapped.apply(1)

    def test_existing_distribution_requires_matching_authoritative_template(self):
        matching = Normal(0, 1, name="draw")
        wrapped = Function(
            func=lambda x: matching,
            output_template=matching.event_template,
        )

        assert wrapped.apply(1) is matching

        mismatching = Normal(0, 1, name="other")
        with pytest.raises(ValueError, match=r"fields .* do not match template fields"):
            Function(
                func=lambda x: mismatching,
                output_template=matching.event_template,
            ).apply(1)

    def test_function_return_remains_event_payload_without_template(self, full_provenance_mode):
        learned = Function(func=lambda x: x + 1, name="learned")
        wrapped = Function(func=lambda: learned, name="fit_like")

        assert wrapped.apply() is learned

        result = wrapped()

        assert isinstance(result, Record)
        assert result["fit_like"] is learned
        assert result.provenance.parents[0].parent is wrapped

    def test_function_spec_return_remains_authoritative_event_payload(self, full_provenance_mode):
        learned = Function(
            func=lambda x: x + 1,
            name="learned",
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(y=()),
        )
        output_template = EventTemplate(
            learned=FunctionSpec(
                input_template=learned.input_template,
                output_template=learned.output_template,
            )
        )
        wrapped = Function(
            func=lambda: learned,
            name="fit_like",
            output_template=output_template,
        )

        assert wrapped.apply() is learned

        result = wrapped()

        assert isinstance(result, Record)
        assert result.event_template == output_template
        assert result["learned"] is learned
        assert result.provenance.parents[0].parent is wrapped


class TestTemplateDeclarationContract:
    def test_signature_and_template_match_by_name_not_order(self):
        def subtract(x, y=1):
            return x - y

        wrapped = Function(
            func=subtract,
            input_template=EventTemplate(y=(), x=()),
            output_template=EventTemplate(result=()),
        )

        assert wrapped.apply(4) == 3
        assert tuple(wrapped.input_template.children) == ("y", "x")

    @pytest.mark.parametrize(
        "template",
        [EventTemplate(x=()), EventTemplate(x=(), y=(), z=())],
    )
    def test_signature_template_requires_total_bijection(self, template):
        with pytest.raises(ValueError, match="exactly match signature parameters"):
            Function(func=lambda x, y: x + y, input_template=template)

    @pytest.mark.parametrize(
        "callable_",
        [lambda *args: args, lambda **kwargs: kwargs],
    )
    def test_authoritative_input_template_rejects_variadics(self, callable_):
        with pytest.raises(ValueError, match="variadic parameters"):
            Function(func=callable_, input_template=EventTemplate(x=()))

    def test_invalid_default_and_construction_binding_fail_at_construction(self):
        invalid_default = np.ones((2,))

        def defaulted(x=invalid_default):
            return x

        with pytest.raises(ValueError, match="default/x"):
            Function(func=defaulted, input_template=EventTemplate(x=(3,)))
        with pytest.raises(ValueError, match="construction binding/x"):
            Function(
                func=lambda x: x,
                input_template=EventTemplate(x=(3,)),
                bind={"x": np.ones((2,))},
            )

    def test_defaults_and_bindings_share_a_declaration_validation_scope(self):
        x_default = np.ones((2,))
        y_default = np.ones((3,))

        def inconsistent_defaults(x=x_default, y=y_default):
            return x, y

        with pytest.raises(ValueError, match=r"default/y.*already bound to 2"):
            Function(
                func=inconsistent_defaults,
                input_template=EventTemplate(x=("n",), y=("n",)),
            )

        with pytest.raises(ValueError, match=r"construction binding/y.*already bound to 2"):
            Function(
                func=lambda x, y: (x, y),
                input_template=EventTemplate(x=("n",), y=("n",)),
                bind={"x": np.ones((2,)), "y": np.ones((3,))},
            )

    def test_unknown_construction_binding_is_rejected(self):
        with pytest.raises(ValueError, match="invalid construction bindings"):
            Function(func=lambda x: x, missing=1)

    def test_output_symbols_must_be_declared_by_input(self):
        with pytest.raises(ValueError, match="not declared by input_template: new"):
            Function(
                func=lambda x: x,
                input_template=EventTemplate(x=("obs",)),
                output_template=EventTemplate(y=("new",)),
            )

    def test_type_errors_for_non_templates(self):
        with pytest.raises(TypeError, match="input_template"):
            Function(func=lambda x: x, input_template=("obs",))  # type: ignore[arg-type]


class TestSymbolicCalls:
    @pytest.fixture
    def regression_function(self):
        return Function(
            func=lambda X, p: X @ p,
            input_template=EventTemplate(X=("obs", "p"), p=("p",)),
            output_template=EventTemplate(y=("obs",)),
            dispatch="sequential",
        )

    def test_joint_input_output_binding_and_declaration_immutability(self, regression_function):
        declaration = regression_function.input_template

        first = regression_function(np.ones((3, 2)), np.ones((2,)))
        second = regression_function(np.ones((5, 4)), np.ones((4,)))

        assert first.event_template == EventTemplate(y=(3,))
        assert second.event_template == EventTemplate(y=(5,))
        assert regression_function.input_template is declaration
        assert declaration == EventTemplate(X=("obs", "p"), p=("p",))

    def test_template_less_distribution_array_reports_lifting_contract(self):
        def identity(x):
            return x

        wrapped = Function(
            func=identity,
            input_template=EventTemplate(x=()),
            dispatch="sequential",
        )
        values = DistributionArray(
            [
                Normal(0, 1, name="left"),
                Normal(1, 1, name="right"),
            ]
        )

        assert values.event_template is None
        with pytest.raises(
            ValueError,
            match=(
                r"Function 'identity' input 'x' does not expose an authoritative "
                r"event_template for lifting"
            ),
        ):
            wrapped(values)

    def test_repeated_input_symbol_conflict_has_function_path(self, regression_function):
        with pytest.raises(
            ValueError,
            match=r"Function '<lambda>' input/p.*'p'.*already bound",
        ):
            regression_function(np.ones((3, 2)), np.ones((4,)))

    def test_output_symbol_conflict_fails_before_publication(self):
        wrapped = Function(
            func=lambda x: x[:-1],
            input_template=EventTemplate(x=("obs",)),
            output_template=EventTemplate(y=("obs",)),
        )

        with pytest.raises(ValueError, match="output at 'y'"):
            wrapped(np.ones((4,)))

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_sweep_preserves_concrete_declared_output_template(self, dispatch):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=jnp.ones((2,)) * i) for i in range(3)]
        )
        wrapped = Function(
            func=lambda row: row["value"] + 1,
            input_template=EventTemplate(row=EventTemplate(value=("p",))),
            output_template=EventTemplate(prediction=("p",)),
            dispatch=dispatch,
        )

        result = wrapped(rows)

        assert result.event_template == EventTemplate(prediction=(2,))
        assert result.batch_shape == (3,)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_distribution_broadcast_preserves_declared_output_template(self, dispatch):
        wrapped = Function(
            func=lambda x: jnp.stack((x, x + 1)),
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(pair=(2,)),
            dispatch=dispatch,
            n_broadcast_samples=8,
            seed=4,
        )

        result = wrapped(Normal(0, 1, name="x"))

        assert result.event_template == EventTemplate(pair=(2,))
        assert result.samples["pair"].shape == (8, 2)

    def test_every_sweep_cell_is_validated_against_output_template(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", size=2), NumericRecord("row", size=3)]
        )
        wrapped = Function(
            func=lambda row: jnp.ones((int(row["size"]),)),
            input_template=EventTemplate(row=EventTemplate(size=())),
            output_template=EventTemplate(value=(2,)),
            dispatch="sequential",
        )

        with pytest.raises(ValueError, match="output at 'value'"):
            wrapped(rows)

    def test_every_sweep_cell_is_validated_against_output_support(self):
        rows = NumericRecordArray.stack(
            [
                NumericRecord("row", value=jnp.asarray(1.0)),
                NumericRecord("row", value=jnp.asarray(-1.0)),
            ]
        )
        wrapped = Function(
            func=lambda row: row["value"],
            input_template=EventTemplate(row=EventTemplate(value=())),
            output_template=EventTemplate(value=ArraySpec((), support=positive)),
            dispatch="sequential",
        )

        with pytest.raises(ValueError, match=r"output at 'value'.*support positive"):
            wrapped(rows)

    def test_support_pinned_broadcast_auto_falls_back_to_sequential(self):
        wrapped = Function(
            func=lambda x: x**2 + 1,
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(y=ArraySpec((), support=positive)),
            dispatch="auto",
            n_broadcast_samples=8,
            seed=11,
        )

        result = wrapped(Normal(0, 1, name="x"))

        assert result.provenance.metadata["dispatch"] == "sequential"
        assert result.event_template == EventTemplate(y=ArraySpec((), support=positive))
        assert bool(jnp.all(result.samples["y"] > 0))

    def test_support_pinned_broadcast_explicit_jax_reports_traceability_error(self):
        wrapped = Function(
            func=lambda x: x**2 + 1,
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(y=ArraySpec((), support=positive)),
            dispatch="jax",
            n_broadcast_samples=8,
            seed=11,
        )

        with pytest.raises(ValueError, match=r"dispatch='jax' failed while tracing"):
            wrapped(Normal(0, 1, name="x"))

    def test_support_pinned_sweep_auto_falls_back_to_sequential(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)]
        )
        template = EventTemplate(y=ArraySpec((), support=positive))
        wrapped = Function(
            func=lambda row: row["value"] + 1,
            input_template=EventTemplate(row=EventTemplate(value=())),
            output_template=template,
            dispatch="auto",
        )

        result = wrapped(rows)

        assert result.event_template == template
        np.testing.assert_allclose(result["y"], np.arange(3.0) + 1)

    def test_support_pinned_sweep_explicit_jax_reports_traceability_error(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)]
        )
        wrapped = Function(
            func=lambda row: row["value"] + 1,
            input_template=EventTemplate(row=EventTemplate(value=())),
            output_template=EventTemplate(y=ArraySpec((), support=positive)),
            dispatch="jax",
        )

        with pytest.raises(ValueError, match=r"dispatch='jax' failed while tracing"):
            wrapped(rows)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_nested_mapping_sweep_preserves_declared_structure(self, dispatch):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)]
        )
        wrapped = Function(
            func=lambda row: {
                "prediction": row["value"] + 1,
                "stats": {"doubled": row["value"] * 2},
            },
            input_template=EventTemplate(row=EventTemplate(value=())),
            output_template=EventTemplate(
                prediction=(),
                stats=EventTemplate(doubled=()),
            ),
            dispatch=dispatch,
        )

        result = wrapped(rows)

        assert result.batch_shape == (3,)
        assert result.event_template == EventTemplate(
            prediction=(),
            stats=EventTemplate(doubled=()),
        )
        np.testing.assert_allclose(result["prediction"], np.arange(3.0) + 1)
        np.testing.assert_allclose(result["stats/doubled"], np.arange(3.0) * 2)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_nested_mapping_distribution_broadcast_preserves_declared_structure(self, dispatch):
        wrapped = Function(
            func=lambda x: {"stats": {"value": x, "doubled": x * 2}},
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(
                stats=EventTemplate(value=(), doubled=()),
            ),
            dispatch=dispatch,
            n_broadcast_samples=8,
            seed=7,
        )

        result = wrapped(Normal(0, 1, name="x"))

        assert result.event_template == EventTemplate(
            stats=EventTemplate(value=(), doubled=()),
        )
        assert result.samples["stats/value"].shape == (8,)
        np.testing.assert_allclose(
            result.samples["stats/doubled"],
            result.samples["stats/value"] * 2,
        )
        averaged = mean(result)
        assert averaged.event_template == EventTemplate(
            stats=EventTemplate(value=(), doubled=()),
        )
        np.testing.assert_allclose(
            averaged["stats/doubled"],
            averaged["stats/value"] * 2,
        )

    def test_distribution_outputs_keep_declared_template_through_broadcast(self):
        wrapped = Function(
            func=lambda x: Normal(x, 1, name="y"),
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(y=()),
            dispatch="sequential",
            n_broadcast_samples=8,
            seed=3,
        )

        result = wrapped(Normal(0, 1, name="x"))

        assert result.event_template == EventTemplate(y=())

    def test_distribution_outputs_keep_declared_template_through_sweep(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)]
        )
        wrapped = Function(
            func=lambda row: Normal(row["value"], 1, name="y"),
            input_template=EventTemplate(row=EventTemplate(value=())),
            output_template=EventTemplate(y=()),
            dispatch="sequential",
        )

        result = wrapped(rows)

        assert isinstance(result, DistributionArray)
        assert result.event_template == EventTemplate(y=())

    def test_nested_broadcast_distribution_array_keeps_declared_template(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", offset=jnp.asarray(float(i))) for i in range(2)]
        )
        wrapped = Function(
            func=lambda row, noise: {"prediction": row["offset"] + noise},
            input_template=EventTemplate(
                row=EventTemplate(offset=()),
                noise=(),
            ),
            output_template=EventTemplate(prediction=()),
            dispatch="sequential",
            n_broadcast_samples=8,
            seed=5,
        )

        result = wrapped(rows, Normal(0, 1, name="noise"))

        assert isinstance(result, DistributionArray)
        assert result.event_template == EventTemplate(prediction=())


@dataclass(frozen=True)
class _AddImplementation:
    increment: float

    def invoke(self, bound_inputs, *, context):
        assert context.dimension_bindings == {}
        return bound_inputs.arguments["x"] + self.increment


@dataclass(frozen=True)
class _MultiplyImplementation:
    factor: float

    def invoke(self, bound_inputs, *, context):
        return bound_inputs.arguments["x"] * self.factor


class TestDynamicImplementation:
    def test_from_implementation_builds_normal_function_using_shared_planner(self):
        signature = inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
        wrapped = Function._from_implementation(
            _AddImplementation(2),
            signature=signature,
            name="dynamic_add",
            input_template=EventTemplate(x=()),
            output_template=EventTemplate(y=()),
            dispatch="sequential",
        )

        assert type(wrapped) is Function
        assert wrapped.apply(3) == 5
        assert float(wrapped(3)) == 5
        assert not hasattr(wrapped, "_func")

    def test_from_implementation_rejects_invalid_declaration(self):
        signature = inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )

        with pytest.raises(TypeError, match="non-empty name"):
            Function._from_implementation(
                _AddImplementation(1),
                signature=signature,
                name="",
            )
        with pytest.raises(TypeError, match="must provide an invoke"):
            Function._from_implementation(
                object(),  # type: ignore[arg-type]
                signature=signature,
                name="invalid",
            )

    def test_dynamic_fingerprint_is_declaration_level_not_artifact_identity(self):
        from probpipe.core._fingerprint import fingerprint

        signature = inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )

        def build(implementation):
            return Function._from_implementation(
                implementation,
                signature=signature,
                name="dynamic",
                input_template=EventTemplate(x=()),
                output_template=EventTemplate(y=()),
            )

        assert fingerprint(build(_AddImplementation(1))) == fingerprint(
            build(_AddImplementation(99))
        )
        assert fingerprint(build(_AddImplementation(1))) != fingerprint(
            build(_MultiplyImplementation(1))
        )

    def test_dynamic_fingerprint_with_opaque_default_is_process_stable(self):
        script = textwrap.dedent(
            """
            import inspect

            from probpipe import Function
            from probpipe.core._fingerprint import fingerprint

            class Implementation:
                def invoke(self, bound_inputs, *, context):
                    return bound_inputs.arguments["x"]

            signature = inspect.Signature([
                inspect.Parameter(
                    "x",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=object(),
                )
            ])
            function = Function._from_implementation(
                Implementation(), signature=signature, name="dynamic"
            )
            print(fingerprint(function))
            """
        )

        def run() -> str:
            return subprocess.check_output(
                [sys.executable, "-c", script],
                text=True,
            ).strip()

        assert run() == run()

    def test_dynamic_fingerprint_tracks_signature_and_template_declarations(self):
        from probpipe.core._fingerprint import fingerprint

        def build(signature, *, input_template=None, output_template=None):
            return Function._from_implementation(
                _AddImplementation(1),
                signature=signature,
                name="dynamic",
                input_template=input_template,
                output_template=output_template,
            )

        base = inspect.Signature(
            [
                inspect.Parameter(
                    "x",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=int,
                )
            ],
            return_annotation=int,
        )
        changed_kind = inspect.Signature(
            [
                inspect.Parameter(
                    "x",
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=int,
                )
            ],
            return_annotation=int,
        )
        changed_default = base.replace(parameters=[base.parameters["x"].replace(default=1)])
        changed_annotation = base.replace(
            parameters=[base.parameters["x"].replace(annotation=float)]
        )
        changed_return = base.replace(return_annotation=float)

        baseline = fingerprint(build(base))
        assert baseline != fingerprint(build(changed_kind))
        assert baseline != fingerprint(build(changed_default))
        assert baseline != fingerprint(build(changed_annotation))
        assert baseline != fingerprint(build(changed_return))
        assert fingerprint(
            build(
                base,
                input_template=EventTemplate(x=()),
                output_template=EventTemplate(y=()),
            )
        ) != fingerprint(
            build(
                base,
                input_template=EventTemplate(x=(1,)),
                output_template=EventTemplate(y=(1,)),
            )
        )


class TestReentrancyAndProvenance:
    def test_same_seed_is_repeatable_across_sequential_and_concurrent_calls(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        wrapped = Function(
            func=lambda x: x + 1,
            n_broadcast_samples=12,
            dispatch="sequential",
            seed=19,
        )
        source = Normal(0, 1, name="x")

        sequential = [wrapped(source).samples["marginal"] for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(lambda _: wrapped(source).samples["marginal"], range(2)))

        assert jnp.array_equal(sequential[0], sequential[1])
        assert all(jnp.array_equal(sequential[0], value) for value in concurrent)
        assert not hasattr(wrapped, "_key")
        assert not hasattr(wrapped, "_resolved_dispatch")

    def test_plain_result_provenance_starts_with_function_then_tracked_inputs(
        self, full_provenance_mode
    ):
        wrapped = Function(func=lambda record: record["x"] + 1)
        tracked_input = NumericRecord("input", x=1.0)

        result = wrapped(tracked_input)

        assert [parent.parent for parent in result.provenance.parents] == [
            wrapped,
            tracked_input,
        ]
        assert wrapped.provenance is None

    @pytest.mark.parametrize(
        "stored",
        [
            NumericRecord("stored", value=1.0),
            NumericRecordArray.stack([NumericRecord("stored", value=1.0)]),
            Normal(0, 1, name="stored"),
        ],
    )
    def test_preprovenanced_tracked_return_is_copied_for_each_call(
        self, stored, full_provenance_mode
    ):
        object.__setattr__(stored, "_annotations", {"owner": "callable"})
        stored.with_provenance(Provenance("inner"))
        wrapped = Function(func=lambda x: stored)

        assert wrapped.apply(1) is stored
        first = wrapped(1)
        second = wrapped(1)

        assert first is not stored
        assert second is not stored
        assert second is not first
        assert stored.provenance.operation == "inner"
        assert first.provenance.parents[0].parent is wrapped
        assert second.provenance.parents[0].parent is wrapped
        assert first.annotations == stored.annotations
        first.annotations["result"] = True
        assert "result" not in stored.annotations

    def test_off_mode_still_copies_a_tracked_return(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        stored = NumericRecord("stored", value=1.0)
        wrapped = Function(func=lambda: stored)

        result = wrapped()

        assert result is not stored
        assert result.provenance is None

    @pytest.mark.parametrize("mode", [ProvenanceMode.FULL, ProvenanceMode.LIGHTWEIGHT])
    def test_dynamic_training_lineage_is_reachable_through_function_parent(self, mode):
        probpipe.provenance_config.mode = mode
        training_data = Record("training", x=1.0)
        signature = inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
        wrapped = Function._from_implementation(
            _AddImplementation(1),
            signature=signature,
            name="fitted",
        ).with_provenance(Provenance.create("fit", parents=[training_data]))

        result = wrapped(2)
        ancestors = probpipe.provenance_ancestors(result)

        assert [ancestor.name for ancestor in ancestors] == ["fitted", "training"]

    def test_off_mode_attaches_no_call_provenance(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF

        result = Function(func=lambda x: x + 1)(2)

        assert result.provenance is None


class TestVariadicPlanning:
    def test_distribution_in_varargs_is_lifted(self):
        wrapped = Function(
            func=lambda *items: items[0] + items[1],
            dispatch="sequential",
            n_broadcast_samples=8,
            seed=11,
        )

        result = wrapped.with_options(include_inputs=True)(Normal(0, 1, name="x"), 2.0)

        assert isinstance(result, Distribution)
        assert result.num_atoms == 8
        assert tuple(result.input_samples) == ("*items[0]",)
        assert result.provenance.metadata["broadcast_args"] == ["*items[0]"]

    def test_record_array_in_varargs_is_swept(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)]
        )
        wrapped = Function(func=lambda *items: items[0]["value"] + items[1])

        result = wrapped(rows, 2.0)

        assert result.batch_shape == (3,)
        np.testing.assert_allclose(result["<lambda>"], np.arange(3.0) + 2)

    def test_tracked_varargs_are_provenance_parents(self, full_provenance_mode):
        first = NumericRecord("first", value=1.0)
        second = NumericRecord("second", value=2.0)
        wrapped = Function(func=lambda *items: items[0]["value"] + items[1]["value"])

        result = wrapped(first, second)

        assert [parent.parent for parent in result.provenance.parents] == [
            wrapped,
            first,
            second,
        ]

    def test_mixed_variadic_provenance_uses_python_call_order(self, full_provenance_mode):
        fixed = NumericRecord("fixed", value=1.0)
        item = NumericRecord("item", value=2.0)
        first_extra = NumericRecord("first_extra", value=3.0)
        second_extra = NumericRecord("second_extra", value=4.0)

        def total(head, *items, **extras):
            return (
                head["value"]
                + items[0]["value"]
                + extras["first"]["value"]
                + extras["second"]["value"]
            )

        wrapped = Function(func=total)

        result = wrapped(
            fixed,
            item,
            first=first_extra,
            second=second_extra,
        )

        assert [parent.parent for parent in result.provenance.parents] == [
            wrapped,
            fixed,
            item,
            first_extra,
            second_extra,
        ]

    def test_plain_inputs_keep_resolved_variadic_slots(self, full_provenance_mode):
        tracked = NumericRecord("tracked", value=1.0)
        shared = jnp.asarray(2.0)

        def total(head, *items, offset, bias=3.0, **extras):
            return head["value"] + items[0] + items[1] + offset + bias + extras["tail"]

        wrapped = Function(func=total, bind={"offset": 4.0})

        result = wrapped(tracked, shared, shared, tail=shared)

        assert result.provenance is not None
        assert [parent.parent for parent in result.provenance.parents] == [wrapped, tracked]
        assert tuple(result.provenance.inputs) == (
            "*items[0]",
            "*items[1]",
            "offset",
            "bias",
            "**extras['tail']",
        )
        assert result.provenance.inputs["*items[0]"].parent is shared
        assert result.provenance.inputs["*items[1]"].parent is shared
        assert result.provenance.inputs["**extras['tail']"].parent is shared

    def test_varargs_and_varkwargs_with_same_textual_name_do_not_collide(self):
        def collect(*items, **extras):
            return items[0] + extras["items"]

        wrapped = Function(func=collect)

        assert wrapped.apply(1, items=2) == 3
        assert float(wrapped(1, items=2)) == 3

    def test_distribution_annotation_applies_to_each_vararg(self):
        def count(*items: Distribution):
            return len(items)

        wrapped = Function(func=count)

        assert float(wrapped(Normal(0, 1, name="x"))) == 1

    def test_distribution_annotation_applies_to_each_varkwarg(self):
        def count(**extras: Distribution):
            return len(extras)

        wrapped = Function(func=count)

        assert float(wrapped(x=Normal(0, 1, name="x"))) == 1

    def test_construction_bound_varargs_participate_in_lifting(self):
        wrapped = Function(
            func=lambda *items: items[0] + items[1],
            bind={"items": (Normal(0, 1, name="x"), 2.0)},
            dispatch="sequential",
            n_broadcast_samples=8,
            seed=13,
        )

        result = wrapped()

        assert result.num_atoms == 8
        assert result.provenance.metadata["broadcast_args"] == ["*items[0]"]

    def test_varkw_distribution_uses_stable_planner_label(self):
        wrapped = Function(
            func=lambda **extras: extras["x"] + extras["offset"],
            dispatch="sequential",
            n_broadcast_samples=8,
            seed=17,
        )

        result = wrapped(x=Normal(0, 1, name="x"), offset=2.0)

        assert result.num_atoms == 8
        assert result.provenance.metadata["broadcast_args"] == ["**extras['x']"]
