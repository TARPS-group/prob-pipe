"""Contracts for first-class, schema-aware Function values."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import (
    Annotated,
    ApplicabilityError,
    BatchSpec,
    Distribution,
    DistributionBatch,
    DistributionSpec,
    Function,
    FunctionSpec,
    Gamma,
    InputSpec,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    OpaqueSpec,
    OutputSpec,
    Provenance,
    ProvenanceMode,
    Record,
    RecordBatch,
    RecordSpec,
    TrackedTerm,
    function,
    mean,
    positive,
    positive_definite,
    real,
    simplex,
    workflow_run,
)


class TestFunctionValueContract:
    def test_function_is_tracked_annotated_and_immutable(self):
        def increment(x):
            return x + 1

        wrapped = Function(label="increment", fn=increment)

        assert isinstance(wrapped, TrackedTerm)
        assert isinstance(wrapped, Annotated)
        assert wrapped.label == "increment"
        assert wrapped.provenance is None
        assert wrapped.annotations == {}
        wrapped.annotations["note"] = "append-only metadata"
        assert wrapped.annotations == {"note": "append-only metadata"}
        with pytest.raises(AttributeError, match="immutable"):
            wrapped._seed = 3

    def test_decorator_and_explicit_names_are_set_at_construction(self):
        @function
        def automatic(x):
            return x

        named = Function(fn=lambda x: x, label="chosen")

        assert automatic.label == "automatic"
        assert named.label == "chosen"

    def test_rename_synchronizes_callable_metadata_and_records_provenance(
        self, full_provenance_mode
    ):
        wrapped = Function(label="function", fn=lambda x: x)

        renamed = wrapped.with_label("identity")

        assert renamed.label == renamed.__name__ == renamed.__qualname__ == "identity"
        assert inspect.signature(renamed) == wrapped.signature
        assert renamed.provenance is not None
        assert renamed.provenance.parents[0].parent is wrapped

    def test_provenance_is_write_once(self):
        wrapped = Function(label="function", fn=lambda x: x)
        wrapped.with_provenance(Provenance("trained"))

        with pytest.raises(RuntimeError, match="write-once"):
            wrapped.with_provenance(Provenance("retrained"))

    def test_signature_is_captured_independently_once(self):
        def affine(x, /, scale=2, *, offset=1):
            return x * scale + offset

        wrapped = Function(label="affine", fn=affine)
        captured = wrapped.signature
        affine.__signature__ = inspect.Signature()  # type: ignore[attr-defined]

        assert wrapped.signature is captured
        assert inspect.signature(wrapped) is captured


class TestApplyContract:
    def test_apply_preserves_python_parameter_kinds_and_returns_raw(self):
        def affine(x, /, scale=2, *, offset=1):
            return x * scale + offset

        wrapped = Function(
            label="affine", fn=affine, output_spec=OutputSpec(**RecordSpec(answer=()).children)
        )

        assert wrapped.apply(3, offset=4) == 10
        result = wrapped(3, offset=4)
        assert isinstance(result, NumericArray)
        assert float(result) == 10

    def test_call_records_positional_and_keyword_only_inputs(self, full_provenance_mode):
        def combine(x, /, y=2.0, *, scale=3.0):
            return (x + y) * scale

        wrapped = Function(label="combine", fn=combine)

        result = wrapped(1.0)

        assert result.provenance is not None
        assert tuple(result.provenance.inputs) == ("x", "y", "scale")
        assert result.provenance.inputs["x"].parent == 1.0
        assert result.provenance.inputs["y"].parent == 2.0
        assert result.provenance.inputs["scale"].parent == 3.0

    def test_variadic_apply_without_template(self):
        def collect(first, *rest, **extras):
            return first + sum(rest) + sum(extras.values())

        wrapped = Function(label="collect", fn=collect)

        assert wrapped.apply(1, 2, 3, bonus=4) == 10
        assert float(wrapped(1, 2, 3, bonus=4)) == 10

    def test_parameter_named_self_is_not_treated_as_a_method_receiver(self):
        wrapped = Function(
            label="function",
            fn=lambda self: self + 1,
            input_spec=InputSpec(RecordSpec(self=()).children),
            output_spec=OutputSpec(**RecordSpec(result=()).children),
        )

        assert wrapped.apply(2) == 3
        assert float(wrapped(2)) == 3

    def test_construction_bindings_support_positional_only_and_var_keyword(self):
        def collect(x, /, **extras):
            return x + sum(extras.values())

        wrapped = Function(label="collect", fn=collect, bind={"x": 2, "bonus": 3})

        assert wrapped.apply() == 5
        assert wrapped.apply(bonus=4) == 6

    def test_apply_validates_output_without_wrapping(self):
        wrapped = Function(
            label="function",
            fn=lambda x: np.ones((x,)),
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=OutputSpec(**RecordSpec(y=(3,)).children),
        )

        with pytest.raises(ValueError, match="output/y"):
            wrapped.apply(2)

    def test_dtype_pinned_output_accepts_bare_and_inferred_record(self):
        template = RecordSpec(y=NumericArraySpec((), dtype="float32"))
        value = jnp.asarray(3.0, dtype=jnp.float32)
        returned = Record("returned", y=value)
        bare = Function(
            label="function", fn=lambda: value, output_spec=OutputSpec(**template.children)
        )
        structured = Function(label="function", fn=lambda: returned, output_spec=template)

        assert bare.apply() is value
        assert structured.apply() is returned
        assert returned.event_template != template

        result = structured()

        assert result is not returned
        assert result.raw("y") is value
        assert result.event_template == template
        assert returned.event_template != template

    def test_declared_output_accepts_record_batch_as_a_batched_event(self):
        intrinsic = RecordSpec(y=())
        declared = RecordSpec(y=NumericArraySpec((), dtype="float32"))
        returned = NumericRecordBatch(
            "batch",
            {"y": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=intrinsic,
        )
        wrapped = Function(
            label="function",
            fn=lambda: returned,
            output_spec=BatchSpec(declared, returned.axis_groups, returned.level_names),
        )

        assert wrapped.apply() is returned

        result = wrapped()

        assert result is not returned
        assert isinstance(result, NumericRecordBatch)
        assert result.batch_shape == (2,)
        assert result.event_template == declared
        assert returned.event_template is intrinsic
        np.testing.assert_allclose(result["y"], np.asarray([1.0, 2.0]))

    @pytest.mark.parametrize("batch_shape", [(2, 3), (0, 3)])
    def test_declared_output_accepts_multidimensional_and_empty_record_batches(
        self,
        batch_shape,
    ):
        template = RecordSpec(y=(2,))
        returned = NumericRecordBatch(
            "batch",
            {"y": jnp.ones((*batch_shape, 2))},
            level_names="draw",
            axes_per_level=(len(batch_shape),),
            element_spec=template,
        )
        wrapped = Function(
            label="function",
            fn=lambda: returned,
            output_spec=BatchSpec(template, returned.axis_groups, returned.level_names),
        )

        assert wrapped.apply() is returned
        result = wrapped()

        assert result.batch_shape == batch_shape
        assert result.event_template == template
        np.testing.assert_allclose(result["y"], np.ones((*batch_shape, 2)))

    def test_declared_output_rejects_a_batch_with_the_wrong_event_shape(self):
        returned = RecordBatch(
            "batch",
            {"y": jnp.ones((2, 3))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(y=(3,)),
        )
        wrapped = Function(
            label="function",
            fn=lambda: returned,
            output_spec=BatchSpec(RecordSpec(y=()), returned.axis_groups, returned.level_names),
        )

        with pytest.raises(ValueError, match=r"shape"):
            wrapped.apply()

    def test_declared_output_checks_record_batch_dtype_and_support(self):
        dtype_template = RecordSpec(y=NumericArraySpec((), dtype="int32"))
        float_array = RecordBatch(
            "batch",
            {"y": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(y=()),
        )

        with pytest.raises(ValueError, match=r"output/function/y dtype float32 does not conform"):
            Function(
                label="function",
                fn=lambda: float_array,
                output_spec=BatchSpec(
                    dtype_template, float_array.axis_groups, float_array.level_names
                ),
            ).apply()

        support_template = RecordSpec(y=NumericArraySpec((), support=positive))
        invalid_array = NumericRecordBatch(
            "batch",
            {"y": jnp.asarray([1.0, -2.0])},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(y=()),
        )

        with pytest.raises(ValueError, match=r"output/function/y.*support positive"):
            Function(
                label="function",
                fn=lambda: invalid_array,
                output_spec=BatchSpec(
                    support_template, invalid_array.axis_groups, invalid_array.level_names
                ),
            ).apply()

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
        template = RecordSpec(y=NumericArraySpec((), dtype=declared_dtype))

        Function(
            label="function", fn=lambda: value, output_spec=OutputSpec(**template.children)
        ).apply()
        Function(
            label="function", fn=lambda: Record("returned", y=value), output_spec=template
        ).apply()

    @pytest.mark.parametrize("structured", [False, True])
    def test_cross_kind_output_dtype_is_rejected(self, structured):
        value = np.asarray(3.0, dtype="float32")
        returned = Record("returned", y=value) if structured else value
        wrapped = Function(
            label="function",
            fn=lambda: returned,
            output_spec=(
                RecordSpec(y=NumericArraySpec((), dtype="int32"))
                if structured
                else OutputSpec(y=NumericArraySpec((), dtype="int32"))
            ),
        )

        with pytest.raises(ValueError, match=r"output/y.*dtype|output/y"):
            wrapped.apply()

    def test_support_pinned_scalar_output_is_enforced(self):
        template = RecordSpec(y=NumericArraySpec((), support=positive))

        assert (
            Function(
                label="function",
                fn=lambda: jnp.asarray(3.0),
                output_spec=OutputSpec(**template.children),
            ).apply()
            == 3.0
        )
        with pytest.raises(ValueError, match=r"output/y.*support positive"):
            Function(
                label="function",
                fn=lambda: jnp.asarray(-3.0),
                output_spec=OutputSpec(**template.children),
            ).apply()

    def test_support_pinned_nested_mapping_output_is_enforced(self):
        template = RecordSpec(
            stats=RecordSpec(y=NumericArraySpec((2,), support=positive)),
        )
        wrapped = Function(
            label="function",
            fn=lambda value: {"stats": {"y": value}},
            output_spec=template,
        )

        wrapped.apply(jnp.asarray([1.0, 2.0]))
        with pytest.raises(ValueError, match=r"output/stats/y.*support positive"):
            wrapped.apply(jnp.asarray([1.0, -2.0]))

    def test_support_pinned_nested_record_output_is_enforced(self):
        template = RecordSpec(
            stats=RecordSpec(y=NumericArraySpec((2,), support=positive)),
        )
        valid = Record("returned", stats=Record("stats", y=jnp.asarray([1.0, 2.0])))
        invalid = Record("returned", stats=Record("stats", y=jnp.asarray([1.0, -2.0])))

        Function(label="function", fn=lambda: valid, output_spec=template).apply()
        with pytest.raises(ValueError, match=r"output/stats/y.*support positive"):
            Function(label="function", fn=lambda: invalid, output_spec=template).apply()

    def test_explicit_record_support_must_conform_to_declaration(self):
        returned = Record(
            "returned",
            y=jnp.asarray(3.0),
            event_template=RecordSpec(y=NumericArraySpec((), support=real)),
        )
        wrapped = Function(
            label="function",
            fn=lambda: returned,
            output_spec=RecordSpec(y=NumericArraySpec((), support=positive)),
        )

        with pytest.raises(ValueError, match=r"support"):
            wrapped.apply()

    def test_a_shape_only_output_template_keeps_the_law_declaration(self):
        returned = Normal("y", 0, 1)
        declared = OutputSpec(y=NumericArraySpec(()))
        wrapped = Function(
            label="function", fn=lambda: returned, output_spec=DistributionSpec(declared)
        )

        assert wrapped.apply() is returned

        result = wrapped()

        assert result is not returned
        assert result.spec is returned.spec

    def test_schema_complete_distribution_does_not_read_parallel_metadata(self):
        class SchemaCompleteDistribution(Distribution):
            def __init__(self, record):
                super().__init__("y", record)

            @property
            def dtypes(self):
                raise AssertionError("Function must not read Distribution.dtypes")

            @property
            def supports(self):
                raise AssertionError("Function must not read Distribution.supports")

        intrinsic = RecordSpec(y=NumericArraySpec((), dtype="float32", support=positive))
        declared = RecordSpec(y=NumericArraySpec((), dtype="float32", support=positive))
        returned = SchemaCompleteDistribution(intrinsic)
        wrapped = Function(
            label="function", fn=lambda: returned, output_spec=DistributionSpec(declared)
        )

        assert intrinsic == declared
        assert intrinsic is not declared
        assert wrapped.apply() is returned

        result = wrapped()

        assert result is not returned
        assert result.event_spec.spec is intrinsic

    def test_distribution_output_uses_spec_unification_for_metadata(self):
        class _Undtyped(Distribution):
            # Declares its array's shape and nothing else.
            def __init__(self):
                super().__init__("y", NumericArraySpec(()))

        cases = [
            (
                _Undtyped(),
                OutputSpec(y=NumericArraySpec((), dtype="float32")),
            ),
            (
                Normal("y", 0, 1),
                OutputSpec(y=NumericArraySpec((), dtype="float64")),
            ),
            (
                Gamma("y", 1, 1),
                OutputSpec(y=NumericArraySpec((), support=real)),
            ),
        ]

        for returned, declared in cases:
            wrapped = Function(
                label="function",
                fn=partial(lambda value: value, returned),
                output_spec=DistributionSpec(declared),
            )
            assert wrapped.apply() is returned
            assert wrapped().spec is returned.spec

        with pytest.raises(ValueError, match=r"dtype .* does not conform to int32"):
            Function(
                "law",
                lambda: Normal("y", 0, 1),
                output_spec=DistributionSpec(OutputSpec(y=NumericArraySpec((), dtype="int32"))),
            ).apply()

    def test_returned_law_with_nested_components_matches_distribution_spec(self):
        law = (Normal("a", 0, 1) * Normal("b", 0, 1)).with_path_names(
            {"a": "params/a", "b": "params/b"}
        ) * Normal("s", 0, 1)
        nested = RecordSpec(params=RecordSpec(a=(), b=()), s=())
        assert Function("law", lambda: law, output_spec=DistributionSpec(nested)).apply() is law
        with pytest.raises(ValueError, match="does not conform"):
            Function(
                "law",
                lambda: law,
                output_spec=DistributionSpec(RecordSpec(params=(2,), s=())),
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
        template = RecordSpec(y=NumericArraySpec(valid.shape, support=support))

        Function(
            label="function", fn=lambda: valid, output_spec=OutputSpec(**template.children)
        ).apply()
        with pytest.raises(ValueError, match=r"output/y.*declared support"):
            Function(
                label="function", fn=lambda: invalid, output_spec=OutputSpec(**template.children)
            ).apply()

    def test_support_validation_is_skipped_under_a_users_jax_jit(self):
        """A traced value has no truth to test, so a user's own trace skips the support check."""
        wrapped = Function(
            label="function",
            fn=lambda x: x - 5.0,
            output_spec=OutputSpec(**RecordSpec(y=NumericArraySpec((), support=positive)).children),
        )

        assert float(jnp.asarray(jax.jit(wrapped.apply)(jnp.asarray(1.0)))) == -4.0

    def test_input_template_support_remains_descriptive_for_lifting(self):
        # A Normal declares its support as real, which the declared positive
        # support describes rather than constrains.
        wrapped = Function(
            label="function",
            fn=lambda x: x,
            input_spec=InputSpec(RecordSpec(x=NumericArraySpec((), support=positive)).children),
            dispatch="sequential",
            n_broadcast_samples=5,
        )

        with workflow_run(seed=0):
            result = wrapped(Normal("x", 0, 1))

        assert result.num_atoms == 5

    @pytest.mark.parametrize(
        "template",
        [RecordSpec(outer=RecordSpec(inner=(3,))), RecordSpec(x=(3,), empty=RecordSpec())],
        ids=["nested_leaf", "empty_sibling"],
    )
    def test_sampling_lift_does_not_flatten_record_structure(self, template):
        class StructuredNormal(Normal):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._init_declaration(template)

            def _sample(self, key, sample_shape=()):
                raise AssertionError("An incompatible schema must be rejected before sampling")

        law = StructuredNormal("x", 0, 1)
        wrapped = Function(
            label="function",
            fn=lambda v: v,
            input_spec=InputSpec(RecordSpec(v=(3,)).children),
            dispatch="sequential",
            n_broadcast_samples=5,
        )
        with pytest.raises(ApplicabilityError, match=r"accepts NumericArraySpec.*RecordSpec"):
            wrapped(v=law)

    @pytest.mark.parametrize(
        "template",
        [RecordSpec(x=(3,)), RecordSpec(outer=RecordSpec(inner=(3,)))],
        ids=["flat", "nested"],
    )
    def test_sampling_lift_preserves_explicit_record_declarations(self, template):
        from probpipe.functions._contract import _bind_planned_function_inputs

        class StructuredNormal(Normal):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._init_declaration(template)

        declared = RecordSpec(v=template)
        bound, bindings = _bind_planned_function_inputs(
            function_name="f",
            input_spec=InputSpec(declared.children),
            values={"v": StructuredNormal("x", 0, 1)},
            lifted_names={"v"},
        )
        assert bound == InputSpec(declared.children)
        assert bindings == {}

    def test_every_batch_kind_lifts_against_its_element_spec(self):
        """A batch states what one element satisfies in ``element_spec``, at every
        kind, so a declared function reads every batch the same way."""
        from probpipe import FunctionBatch, NumericArrayBatch, OpaqueBatch

        cases = [
            (
                NumericArrayBatch(
                    "rows",
                    jnp.arange(3.0),
                    "row",
                    element_spec=NumericArraySpec(()),
                ),
                RecordSpec(v=()),
            ),
            (
                OpaqueBatch(
                    "rows",
                    [object(), object()],
                    "row",
                ),
                RecordSpec(v=OpaqueSpec()),
            ),
            (
                FunctionBatch(
                    "rows",
                    [lambda z: z, lambda z: z],
                    "row",
                ),
                RecordSpec(v=FunctionSpec()),
            ),
            (
                NumericRecordBatch(
                    "rows",
                    {"x": jnp.arange(3.0)},
                    "row",
                    element_spec=RecordSpec(x=()),
                ),
                RecordSpec(v=RecordSpec(x=())),
            ),
        ]
        for operand, input_template in cases:
            wrapped = Function(
                fn=lambda v: 1.0,
                label="f",
                input_spec=InputSpec(input_template.children),
                dispatch="sequential",
            )

            result = wrapped(v=operand)

            assert isinstance(result, NumericArrayBatch), type(operand).__name__
            assert result.level_names == ("row",), type(operand).__name__

    def test_a_batch_of_records_does_not_satisfy_a_bare_array_declaration(self):
        """Reading the record-only view made a one-field element pass as its field."""
        rows = NumericRecordBatch(
            "rows",
            {"x": jnp.arange(3.0)},
            "row",
            element_spec=RecordSpec(x=()),
        )
        wrapped = Function(
            fn=lambda v: 1.0,
            label="f",
            input_spec=InputSpec(RecordSpec(v=()).children),
            dispatch="sequential",
        )

        with pytest.raises(ApplicabilityError, match=r"accepts NumericArraySpec.*RecordSpec"):
            wrapped(v=rows)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_lifted_and_raw_inputs_share_dimension_bindings(self, dispatch, reverse):
        fields = [("x", RecordSpec(value=("n",))), ("offset", NumericArraySpec(("n",)))]
        if reverse:
            fields.reverse()
        wrapped = Function(
            fn=lambda x, offset: x["value"] + offset,
            label="shift",
            input_spec=InputSpec(RecordSpec(dict(fields)).children),
            output_spec=OutputSpec(**RecordSpec(result=("n",)).children),
            dispatch=dispatch,
        )
        data = jnp.arange(6.0).reshape(2, 3)
        rows = NumericRecordBatch(
            "rows", {"value": data}, "row", element_spec=RecordSpec(value=(3,))
        )

        result = wrapped(x=rows, offset=jnp.ones(3))

        assert result.level_names == ("row",)
        assert result.batch_shape == (2,)
        assert isinstance(result, NumericArrayBatch)
        assert result.element_spec.shape == (3,)
        np.testing.assert_allclose(result.values, np.asarray(data) + 1, rtol=0, atol=0)
        assert wrapped.input_spec.free_dims == {"n"}
        with pytest.raises(ApplicabilityError, match="already bound"):
            wrapped(x=rows, offset=jnp.ones(4))

    @pytest.mark.parametrize("entrypoint", ["apply", "__call__"])
    def test_output_dimensions_are_bound_from_the_return(self, entrypoint):
        calls = []

        def evaluate(x, fn):
            calls.append(True)
            return np.zeros((2, 3, 4))

        wrapped = Function(
            fn=evaluate,
            label="unbound",
            input_spec=InputSpec(
                RecordSpec(
                    x=("n",),
                    fn=FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec(("z", "a")))),
                ).children
            ),
            output_spec=OutputSpec(**RecordSpec(y=("n", "z", "a")).children),
            dispatch="sequential",
        )
        # Output-only dimensions bind from the returned array on each call.
        result = getattr(wrapped, entrypoint)(np.zeros(2), lambda: None)
        assert result.shape == (2, 3, 4)
        assert len(calls) == 1
        assert wrapped.input_spec.free_dims == {"n", "z", "a"}
        assert wrapped.output_spec.spec.free_dims == {"n", "z", "a"}

    def test_a_mismatched_element_kind_names_both_specs(self):
        from probpipe import OpaqueBatch

        wrapped = Function(
            fn=lambda v: 1.0,
            label="f",
            input_spec=InputSpec(RecordSpec(v=()).children),
            dispatch="sequential",
        )

        with pytest.raises(ApplicabilityError, match=r"accepts NumericArraySpec.*OpaqueSpec"):
            wrapped(
                v=OpaqueBatch(
                    "rows",
                    [object()],
                    "row",
                )
            )

    def test_authoritative_nested_mapping_must_match_exactly(self):
        template = RecordSpec(
            stats=RecordSpec(copy=("obs",), total=()),
        )
        wrapped = Function(
            label="function",
            fn=lambda x: {"stats": {"copy": x, "total": x.sum()}},
            input_spec=InputSpec(RecordSpec(x=("obs",)).children),
            output_spec=template,
        )

        result = wrapped(np.ones((3,)))

        assert result.event_template == RecordSpec(stats=RecordSpec(copy=(3,), total=()))
        with pytest.raises(ValueError, match="do not match template fields"):
            Function(
                label="function",
                fn=lambda x: {"stats": {"copy": x}},
                input_spec=InputSpec(RecordSpec(x=("obs",)).children),
                output_spec=template,
            ).apply(np.ones((3,)))

    def test_raw_result_cannot_satisfy_multi_leaf_output_template(self):
        wrapped = Function(
            label="function",
            fn=lambda x: x,
            output_spec=RecordSpec(left=(), right=()),
        )

        with pytest.raises(ValueError, match="expected named fields"):
            wrapped.apply(1)

    def test_existing_record_requires_matching_authoritative_template(self):
        wrapped = Function(
            label="function",
            fn=lambda x: Record("result", wrong=x),
            output_spec=RecordSpec(expected=()),
        )

        with pytest.raises(ValueError, match=r"fields .* do not match template fields"):
            wrapped.apply(1)

    def test_existing_distribution_requires_matching_authoritative_template(self):
        matching = Normal("draw", 0, 1)
        wrapped = Function(
            label="function",
            fn=lambda x: matching,
            output_spec=matching.spec,
        )

        assert wrapped.apply(1) is matching

        mismatching = Normal("other", 0, 1)
        with pytest.raises(ValueError, match=r"declares the component 'draw'.*'other'"):
            Function(
                label="function",
                fn=lambda x: mismatching,
                output_spec=matching.spec,
            ).apply(1)

    def test_a_returned_function_keeps_its_kind(self, full_provenance_mode):
        """A term an operation returns is never buried inside another kind.

        ``apply`` hands back the implementer's object itself; the default call
        derives a result term from it — the same kind under the call's own
        provenance.
        """
        learned = Function(fn=lambda x: x + 1, label="learned")
        wrapped = Function(fn=lambda: learned, label="fit_like")

        assert wrapped.apply() is learned

        result = wrapped()

        assert isinstance(result, Function)
        assert result is not learned
        assert float(result(1.0)) == 2.0
        assert result.provenance.parents[0].parent is wrapped

    def test_function_spec_return_remains_authoritative_event_payload(self, full_provenance_mode):
        learned = Function(
            fn=lambda x: x + 1,
            label="learned",
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=OutputSpec(y=NumericArraySpec(())),
        )
        output_template = RecordSpec(
            learned=FunctionSpec(
                input_spec=learned.input_spec,
                output_spec=learned.output_spec,
            )
        )
        wrapped = Function(
            fn=lambda: learned,
            label="fit_like",
            output_spec=OutputSpec(**output_template.children),
        )

        assert wrapped.apply() is learned

        result = wrapped()

        assert isinstance(result, Function)
        assert result.spec == learned.spec
        assert result.output_label == learned.output_label
        assert result is not learned
        assert result.provenance.parents[0].parent is wrapped


class TestTemplateDeclarationContract:
    def test_signature_and_template_match_by_name_not_order(self):
        def subtract(x, y=1):
            return x - y

        wrapped = Function(
            label="subtract",
            fn=subtract,
            input_spec=InputSpec(RecordSpec(y=(), x=()).children),
            output_spec=OutputSpec(**RecordSpec(result=()).children),
        )

        assert wrapped.apply(4) == 3
        assert tuple(wrapped.input_spec) == ("y", "x")

    @pytest.mark.parametrize(
        "template",
        [RecordSpec(x=()), RecordSpec(x=(), y=(), z=())],
    )
    def test_signature_template_requires_total_bijection(self, template):
        with pytest.raises(ValueError, match="exactly match signature parameters"):
            Function(
                label="function", fn=lambda x, y: x + y, input_spec=InputSpec(template.children)
            )

    @pytest.mark.parametrize(
        "callable_",
        [lambda *args: args, lambda **kwargs: kwargs],
    )
    def test_authoritative_input_template_rejects_variadics(self, callable_):
        with pytest.raises(ValueError, match="variadic parameters"):
            Function(
                label="callable_", fn=callable_, input_spec=InputSpec(RecordSpec(x=()).children)
            )

    def test_invalid_default_and_construction_binding_fail_at_construction(self):
        invalid_default = np.ones((2,))

        def defaulted(x=invalid_default):
            return x

        with pytest.raises(ValueError, match="default/x"):
            Function(
                label="defaulted", fn=defaulted, input_spec=InputSpec(RecordSpec(x=(3,)).children)
            )
        with pytest.raises(ValueError, match="construction binding/x"):
            Function(
                label="function",
                fn=lambda x: x,
                input_spec=InputSpec(RecordSpec(x=(3,)).children),
                bind={"x": np.ones((2,))},
            )

    def test_defaults_and_bindings_share_a_declaration_validation_scope(self):
        x_default = np.ones((2,))
        y_default = np.ones((3,))

        def inconsistent_defaults(x=x_default, y=y_default):
            return x, y

        with pytest.raises(ValueError, match=r"default/y.*already bound to 2"):
            Function(
                label="inconsistent_defaults",
                fn=inconsistent_defaults,
                input_spec=InputSpec(RecordSpec(x=("n",), y=("n",)).children),
            )

        with pytest.raises(ValueError, match=r"construction binding/y.*already bound to 2"):
            Function(
                label="function",
                fn=lambda x, y: (x, y),
                input_spec=InputSpec(RecordSpec(x=("n",), y=("n",)).children),
                bind={"x": np.ones((2,)), "y": np.ones((3,))},
            )

    def test_unknown_construction_binding_is_rejected(self):
        with pytest.raises(ValueError, match="invalid construction bindings"):
            Function(label="function", fn=lambda x: x, bind={"missing": 1})

    def test_output_symbols_can_bind_independently_of_inputs(self):
        wrapped = Function(
            "identity",
            lambda x: x,
            input_spec=InputSpec(x=NumericArraySpec(("obs",))),
            output_spec=NumericArraySpec(("new",)),
        )
        result = wrapped(jnp.ones(3))
        # The declaration leaves the dtype open, so completion takes the produced one (II.2).
        assert result.spec == NumericArraySpec((3,), dtype="float32")
        assert wrapped.output_spec.spec.free_dims == {"new"}

    def test_type_errors_for_non_templates(self):
        with pytest.raises(TypeError, match="input_spec"):
            Function(label="function", fn=lambda x: x, input_spec=("obs",))  # type: ignore[arg-type]


class TestSymbolicCalls:
    @pytest.fixture
    def regression_function(self):
        return Function(
            label="function",
            fn=lambda X, p: X @ p,
            input_spec=InputSpec(RecordSpec(X=("obs", "p"), p=("p",)).children),
            output_spec=OutputSpec(**RecordSpec(y=("obs",)).children),
            dispatch="sequential",
        )

    def test_joint_input_output_binding_and_declaration_immutability(self, regression_function):
        declaration = regression_function.input_spec

        first = regression_function(np.ones((3, 2)), np.ones((2,)))
        second = regression_function(np.ones((5, 4)), np.ones((4,)))

        assert first.spec == NumericArraySpec((3,), dtype="float64")
        assert second.spec == NumericArraySpec((5,), dtype="float64")
        assert regression_function.input_spec is declaration
        assert declaration == InputSpec(RecordSpec(X=("obs", "p"), p=("p",)).children)

    def test_a_distribution_batch_of_laws_is_not_an_array_input(self):
        def identity(x):
            return x

        wrapped = Function(
            label="identity",
            fn=identity,
            input_spec=InputSpec(RecordSpec(x=()).children),
            dispatch="sequential",
        )
        values = DistributionBatch("laws", [Normal("x", 0, 1), Normal("x", 1, 1)], "law")

        # Each element is a law, which an array input does not admit.
        with pytest.raises(
            ApplicabilityError, match=r"'x' accepts NumericArraySpec.*DistributionSpec"
        ):
            wrapped(values)

    def test_repeated_input_symbol_conflict_has_function_path(self, regression_function):
        with pytest.raises(
            ApplicabilityError,
            match=r"Function 'function' input/p.*'p'.*already bound",
        ):
            regression_function(np.ones((3, 2)), np.ones((4,)))

    def test_output_symbol_conflict_fails_before_publication(self):
        wrapped = Function(
            label="function",
            fn=lambda x: x[:-1],
            input_spec=InputSpec(RecordSpec(x=("obs",)).children),
            output_spec=OutputSpec(**RecordSpec(y=("obs",)).children),
        )

        with pytest.raises(ValueError, match="output/y"):
            wrapped(np.ones((4,)))

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_sweep_preserves_concrete_declared_output_template(self, dispatch):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.ones((2,)) * i) for i in range(3)], level_name="draw"
        )
        wrapped = Function(
            label="function",
            fn=lambda row: row["value"] + 1,
            input_spec=InputSpec(RecordSpec(row=RecordSpec(value=("p",))).children),
            output_spec=OutputSpec(**RecordSpec(prediction=("p",)).children),
            dispatch=dispatch,
        )

        result = wrapped(rows)

        assert isinstance(result, NumericArrayBatch)
        assert result.element_spec.shape == (2,)
        assert result.batch_shape == (3,)
        np.testing.assert_allclose(
            result.values,
            np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        )

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_distribution_broadcast_declares_the_output_template(self, dispatch):
        wrapped = Function(
            label="function",
            fn=lambda x: jnp.stack((x, x + 1)),
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=OutputSpec(**RecordSpec(pair=(2,)).children),
            dispatch=dispatch,
            n_broadcast_samples=8,
        )

        with workflow_run(seed=4):
            result = wrapped(Normal("x", 0, 1))

        assert list(result.event_spec.components) == ["pair"]
        assert result.event_spec.spec.shape == (2,)
        assert result.num_atoms == 8
        pairs = np.asarray(result._rows)
        assert pairs.shape == (8, 2)
        np.testing.assert_allclose(pairs[:, 1], pairs[:, 0] + 1)

    def test_every_sweep_cell_is_validated_against_output_template(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", size=2), NumericRecord("row", size=3)], level_name="draw"
        )
        wrapped = Function(
            label="function",
            fn=lambda row: jnp.ones((int(row["size"]),)),
            input_spec=InputSpec(RecordSpec(row=RecordSpec(size=())).children),
            output_spec=OutputSpec(**RecordSpec(value=(2,)).children),
            dispatch="sequential",
        )

        with pytest.raises(ValueError, match="output/value"):
            wrapped(rows)

    def test_every_sweep_cell_is_validated_against_output_support(self):
        rows = NumericRecordBatch.stack(
            [
                NumericRecord("row", value=jnp.asarray(1.0)),
                NumericRecord("row", value=jnp.asarray(-1.0)),
            ],
            level_name="draw",
        )
        wrapped = Function(
            label="function",
            fn=lambda row: row["value"],
            input_spec=InputSpec(RecordSpec(row=RecordSpec(value=())).children),
            output_spec=OutputSpec(
                **RecordSpec(value=NumericArraySpec((), support=positive)).children
            ),
            dispatch="sequential",
        )

        with pytest.raises(ValueError, match=r"output/value.*support positive"):
            wrapped(rows)

    def test_support_pinned_broadcast_auto_runs_in_one_map(self):
        """A declared support is checked on the stacked results, so the lift need not run draw by draw."""
        wrapped = Function(
            label="function",
            fn=lambda x: x**2 + 1,
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=OutputSpec(**RecordSpec(y=NumericArraySpec((), support=positive)).children),
            dispatch="auto",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=11):
            result = wrapped(Normal("x", 0, 1))

        assert result.provenance.metadata["dispatch"] == "jax"
        assert list(result.event_spec.components) == ["y"]
        assert result.event_spec.spec.support == positive
        assert bool(jnp.all(result._rows > 0))

    @pytest.mark.parametrize(
        ("fn", "holds"), [(lambda x: x**2 + 1, True), (lambda x: -(x**2) - 1, False)]
    )
    def test_support_pinned_broadcast_explicit_jax_checks_the_stacked_results(self, fn, holds):
        wrapped = Function(
            label="function",
            fn=fn,
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=OutputSpec(**RecordSpec(y=NumericArraySpec((), support=positive)).children),
            dispatch="jax",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=11):
            if holds:
                assert bool(jnp.all(wrapped(Normal("x", 0, 1))._rows > 0))
            else:
                with pytest.raises(ValueError, match=r"output/y.*support positive"):
                    wrapped(Normal("x", 0, 1))

    def test_support_pinned_sweep_auto_keeps_the_declared_support(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )
        template = RecordSpec(y=NumericArraySpec((), support=positive))
        wrapped = Function(
            label="function",
            fn=lambda row: row["value"] + 1,
            input_spec=InputSpec(RecordSpec(row=RecordSpec(value=())).children),
            output_spec=OutputSpec(**template.children),
            dispatch="auto",
        )

        result = wrapped(rows)

        # The declaration leaves the dtype open, so completion takes the produced one (II.2).
        assert result.element_spec == NumericArraySpec((), dtype="float32", support=positive)
        np.testing.assert_allclose(result.values, np.arange(3.0) + 1)

    @pytest.mark.parametrize(("shift", "holds"), [(1.0, True), (-5.0, False)])
    def test_support_pinned_sweep_explicit_jax_checks_the_stacked_results(self, shift, holds):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )
        wrapped = Function(
            label="function",
            fn=lambda row: row["value"] + shift,
            input_spec=InputSpec(RecordSpec(row=RecordSpec(value=())).children),
            output_spec=OutputSpec(**RecordSpec(y=NumericArraySpec((), support=positive)).children),
            dispatch="jax",
        )

        if holds:
            np.testing.assert_allclose(wrapped(rows).values, np.arange(3.0) + shift)
        else:
            with pytest.raises(ValueError, match=r"output/y.*support positive"):
                wrapped(rows)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_nested_mapping_sweep_preserves_declared_structure(self, dispatch):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )
        wrapped = Function(
            label="function",
            fn=lambda row: {
                "prediction": row["value"] + 1,
                "stats": {"doubled": row["value"] * 2},
            },
            input_spec=InputSpec(RecordSpec(row=RecordSpec(value=())).children),
            output_spec=RecordSpec(
                prediction=(),
                stats=RecordSpec(doubled=()),
            ),
            dispatch=dispatch,
        )

        result = wrapped(rows)

        assert result.batch_shape == (3,)
        assert result.event_template == RecordSpec(
            prediction=(),
            stats=RecordSpec(doubled=()),
        )
        np.testing.assert_allclose(result["prediction"], np.arange(3.0) + 1)
        np.testing.assert_allclose(result["stats/doubled"], np.arange(3.0) * 2)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_nested_mapping_distribution_broadcast_declares_the_nested_record(self, dispatch):
        wrapped = Function(
            label="function",
            fn=lambda x: {"stats": {"value": x, "doubled": x * 2}},
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=RecordSpec(
                stats=RecordSpec(value=(), doubled=()),
            ),
            dispatch=dispatch,
            n_broadcast_samples=8,
        )

        with workflow_run(seed=7):
            result = wrapped(Normal("x", 0, 1))

        assert result.event_spec.spec.leaf_shapes == {"stats/value": (), "stats/doubled": ()}
        assert result.atoms["stats/value"].shape == (8,)
        np.testing.assert_allclose(
            result.atoms["stats/doubled"],
            result.atoms["stats/value"].raw() * 2,
        )
        averaged = mean(result)
        assert averaged.event_template == RecordSpec(
            **{"mean(stats)": RecordSpec(value=(), doubled=())},
        )
        np.testing.assert_allclose(
            averaged["mean(stats)/doubled"],
            averaged["mean(stats)/value"] * 2,
        )

    def test_distribution_outputs_keep_their_declaration_through_broadcast(self):
        wrapped = Function(
            label="function",
            fn=lambda x: Normal("y", x, 1),
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=DistributionSpec(OutputSpec(y=NumericArraySpec(()))),
            dispatch="sequential",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=3):
            joint = wrapped.with_options(include_inputs=True)(Normal("x", 0, 1))
        laws = joint._rows["function"]

        assert joint.event_spec.components["function"] == DistributionSpec(
            OutputSpec(y=NumericArraySpec((), jnp.asarray(0.0).dtype, real))
        )
        assert joint.num_atoms == 8
        np.testing.assert_allclose(jnp.stack([law.loc for law in laws]), joint._rows["x"])
        np.testing.assert_allclose(jnp.stack([law.scale for law in laws]), np.ones(8))

    def test_distribution_broadcast_rejects_cross_kind_declared_dtype(self):
        wrapped = Function(
            label="function",
            fn=lambda x: Normal("y", x, 1),
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=DistributionSpec(OutputSpec(y=NumericArraySpec((), dtype="int32"))),
            dispatch="sequential",
            n_broadcast_samples=8,
        )

        with pytest.raises(
            ValueError,
            match="does not conform",
        ):
            wrapped(Normal("x", 0, 1))

    def test_distribution_outputs_keep_their_declaration_through_sweep(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )
        wrapped = Function(
            label="function",
            fn=lambda row: Normal("y", row["value"], 1),
            input_spec=InputSpec(RecordSpec(row=RecordSpec(value=())).children),
            output_spec=DistributionSpec(OutputSpec(y=NumericArraySpec(()))),
            dispatch="sequential",
        )

        result = wrapped(rows)

        assert isinstance(result, DistributionBatch)
        assert result.event_spec == OutputSpec(y=NumericArraySpec((), jnp.asarray(0.0).dtype, real))
        assert result.batch_size == 3
        np.testing.assert_allclose(jnp.stack([law.loc for law in result]), np.arange(3.0))
        np.testing.assert_allclose(jnp.stack([law.scale for law in result]), np.ones(3))

    def test_nested_broadcast_distribution_batch_declares_the_output_record(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", offset=jnp.asarray(float(i))) for i in range(2)],
            level_name="draw",
        )
        wrapped = Function(
            label="function",
            fn=lambda row, noise: {"prediction": row["offset"] + noise},
            input_spec=InputSpec(
                RecordSpec(
                    row=RecordSpec(offset=()),
                    noise=(),
                ).children
            ),
            output_spec=RecordSpec(prediction=()),
            dispatch="sequential",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=5):
            result = wrapped(rows, Normal("noise", 0, 1))

        assert isinstance(result, DistributionBatch)
        assert result.event_spec.spec.fields == ("prediction",)


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
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=OutputSpec(**RecordSpec(y=()).children),
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

        with pytest.raises(TypeError, match="non-empty label"):
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
                input_spec=InputSpec(RecordSpec(x=()).children),
                output_spec=OutputSpec(**RecordSpec(y=()).children),
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

    def test_dynamic_fingerprint_tracks_signature_and_declarations(self):
        from probpipe.core._fingerprint import fingerprint

        def build(signature, *, input_spec=None, output_spec=None):
            return Function._from_implementation(
                _AddImplementation(1),
                signature=signature,
                name="dynamic",
                input_spec=None if input_spec is None else InputSpec(input_spec.children),
                output_spec=output_spec,
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
                input_spec=RecordSpec(x=()),
                output_spec=RecordSpec(y=()),
            )
        ) != fingerprint(
            build(
                base,
                input_spec=RecordSpec(x=(1,)),
                output_spec=RecordSpec(y=(1,)),
            )
        )


class TestReentrancyAndProvenance:
    def test_seeded_runs_are_repeatable_across_sequential_and_concurrent_calls(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        wrapped = Function(
            label="function",
            fn=lambda x: x + 1,
            n_broadcast_samples=12,
            dispatch="sequential",
        )
        source = Normal("x", 0, 1)

        def evaluate(_):
            with workflow_run(seed=19):
                return wrapped(source)._rows

        sequential = [evaluate(index) for index in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(evaluate, range(2)))

        assert jnp.array_equal(sequential[0], sequential[1])
        assert all(jnp.array_equal(sequential[0], value) for value in concurrent)
        assert not hasattr(wrapped, "_key")
        assert not hasattr(wrapped, "_resolved_dispatch")

    def test_plain_result_provenance_starts_with_function_then_tracked_inputs(
        self, full_provenance_mode
    ):
        wrapped = Function(label="function", fn=lambda record: record["x"] + 1)
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
            NumericRecordBatch.stack([NumericRecord("stored", value=1.0)], level_name="draw"),
            Normal("stored", 0, 1),
        ],
    )
    def test_preprovenanced_tracked_return_is_copied_for_each_call(
        self, stored, full_provenance_mode
    ):
        # A batch carries no annotations — its slots hold the batch's own state
        # alone — so the annotation half of the contract applies to the hosts
        # that have them.
        carries_annotations = hasattr(stored, "annotations")
        if carries_annotations:
            object.__setattr__(stored, "_annotations", {"owner": "callable"})
        stored.with_provenance(Provenance("inner"))
        wrapped = Function(label="function", fn=lambda x: stored)

        assert wrapped.apply(1) is stored
        first = wrapped(1)
        second = wrapped(1)

        assert first is not stored
        assert second is not stored
        assert second is not first
        assert stored.provenance.operation == "inner"
        assert first.provenance.parents[0].parent is wrapped
        assert second.provenance.parents[0].parent is wrapped
        if carries_annotations:
            assert first.annotations == stored.annotations
            first.annotations["result"] = True
            assert "result" not in stored.annotations

    def test_off_mode_still_copies_a_tracked_return(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        stored = NumericRecord("stored", value=1.0)
        wrapped = Function(label="function", fn=lambda: stored)

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

        assert [ancestor.label for ancestor in ancestors] == ["fitted", "training"]

    def test_off_mode_attaches_no_call_provenance(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF

        result = Function(label="function", fn=lambda x: x + 1)(2)

        assert result.provenance is None


class TestVariadicPlanning:
    def test_distribution_in_varargs_is_lifted(self):
        wrapped = Function(
            label="function",
            fn=lambda *items: items[0] + items[1],
            dispatch="sequential",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=11):
            result = wrapped.with_options(include_inputs=True)(Normal("x", 0, 1), 2.0)

        assert isinstance(result, Distribution)
        assert result.num_atoms == 8
        assert list(result.event_spec.components) == ["*items[0]", "function"]
        assert result.provenance.metadata["broadcast_args"] == ["*items[0]"]

    def test_record_batch_in_varargs_is_swept(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )
        wrapped = Function(label="function", fn=lambda *items: items[0]["value"] + items[1])

        result = wrapped(rows, 2.0)

        assert result.batch_shape == (3,)
        np.testing.assert_allclose(result.values, np.arange(3.0) + 2)

    def test_record_batch_in_any_varargs_is_swept(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )

        def double(*items: Any):
            return items[0]["value"] * 2

        result = Function(label="double", fn=double)(rows)

        assert result.batch_shape == (3,)
        np.testing.assert_allclose(result.values, np.arange(3.0) * 2)

    def test_record_batch_in_any_varkwargs_is_swept(self):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", value=jnp.asarray(float(i))) for i in range(3)], level_name="draw"
        )

        def double(**extras: Any):
            return extras["rows"]["value"] * 2

        result = Function(label="double", fn=double)(rows=rows)

        assert result.batch_shape == (3,)
        np.testing.assert_allclose(result.values, np.arange(3.0) * 2)

    def test_tracked_varargs_are_provenance_parents(self, full_provenance_mode):
        first = NumericRecord("first", value=1.0)
        second = NumericRecord("second", value=2.0)
        wrapped = Function(
            label="function", fn=lambda *items: items[0]["value"] + items[1]["value"]
        )

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

        wrapped = Function(label="total", fn=total)

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

        wrapped = Function(label="total", fn=total, bind={"offset": 4.0})

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

        wrapped = Function(label="collect", fn=collect)

        assert wrapped.apply(1, items=2) == 3
        assert float(wrapped(1, items=2)) == 3

    def test_distribution_annotation_applies_to_each_vararg(self):
        def count(*items: Distribution):
            return len(items)

        wrapped = Function(label="count", fn=count)

        assert float(wrapped(Normal("x", 0, 1))) == 1

    def test_distribution_annotation_applies_to_each_varkwarg(self):
        def count(**extras: Distribution):
            return len(extras)

        wrapped = Function(label="count", fn=count)

        assert float(wrapped(x=Normal("x", 0, 1))) == 1

    def test_construction_bound_varargs_participate_in_lifting(self):
        wrapped = Function(
            label="function",
            fn=lambda *items: items[0] + items[1],
            bind={"items": (Normal("x", 0, 1), 2.0)},
            dispatch="sequential",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=13):
            result = wrapped()

        assert result.num_atoms == 8
        assert result.provenance.metadata["broadcast_args"] == ["*items[0]"]

    def test_varkw_distribution_uses_stable_planner_label(self):
        wrapped = Function(
            label="function",
            fn=lambda **extras: extras["x"] + extras["offset"],
            dispatch="sequential",
            n_broadcast_samples=8,
        )

        with workflow_run(seed=17):
            result = wrapped(x=Normal("x", 0, 1), offset=2.0)

        assert result.num_atoms == 8
        assert result.provenance.metadata["broadcast_args"] == ["**extras['x']"]


class TestDeclaredSupportOnABatchedOutput:
    def test_every_column_is_checked_against_its_own_support(self):
        """A batch is a collection, not a named tree: walking it as one finds no
        children and asks a multi-field batch to convert to a single array."""
        from probpipe import NumericRecordBatch

        template = RecordSpec(
            a=NumericArraySpec((), support=positive), b=NumericArraySpec((), support=positive)
        )
        valid = NumericRecordBatch(
            "batch",
            {"a": jnp.ones(3), "b": jnp.ones(3) * 2},
            "draw",
            element_spec=RecordSpec(a=(), b=()),
        )

        result = Function(
            label="function",
            fn=lambda: valid,
            output_spec=BatchSpec(template, valid.axis_groups, valid.level_names),
        ).apply()

        assert result is valid

    def test_a_column_outside_its_support_is_named(self):
        from probpipe import NumericRecordBatch

        template = RecordSpec(
            a=NumericArraySpec((), support=positive), b=NumericArraySpec((), support=positive)
        )
        invalid = NumericRecordBatch(
            "batch",
            {"a": jnp.ones(3), "b": jnp.asarray([1.0, -2.0, 3.0])},
            "draw",
            element_spec=RecordSpec(a=(), b=()),
        )

        with pytest.raises(ValueError, match=r"output/function/b.*support positive"):
            Function(
                label="function",
                fn=lambda: invalid,
                output_spec=BatchSpec(template, invalid.axis_groups, invalid.level_names),
            ).apply()
