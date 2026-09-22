"""Executable declaration examples from design II.2 and shared dimension contracts."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BatchSpec,
    Distribution,
    DistributionSpec,
    FunctionSpec,
    InputSpec,
    NumericArraySpec,
    NumericRecordSpec,
    NumericSpec,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
    TermSpec,
    positive,
)


@pytest.fixture
def declared_law():
    class DeclaredLaw(Distribution):
        def __init__(self, template):
            super().__init__(name="law")
            self.event_template = template

    return DeclaredLaw


@pytest.fixture(params=["missing", "none", "type_error"])
def law_without_schema(request, declared_law):
    if request.param == "missing":
        return Distribution(name="law")
    if request.param == "none":
        return declared_law(None)

    class UndeclaredLaw(Distribution):
        @property
        def event_template(self):
            raise TypeError("draw schema is not derivable")

    return UndeclaredLaw(name="law")


class TestOutputSpec:
    def test_ii2_output_forms(self):
        beta = NumericArraySpec((3,))
        sigma = NumericArraySpec(())
        whole = OutputSpec(beta=beta)
        assert whole.spec is beta
        assert dict(whole.components) == {"beta": beta}

        hole = OutputSpec(beta=None)
        assert hole.spec is None
        assert dict(hole.components) == {"beta": None}
        assert not hole.is_concrete

        exposed = OutputSpec(RecordSpec(beta=beta))
        assert isinstance(exposed.spec, RecordSpec)
        assert exposed.spec != whole.spec
        assert dict(exposed.components) == dict(whole.components)
        assert exposed != whole

        multiple = OutputSpec(beta=beta, sigma=sigma)
        assert multiple == OutputSpec(RecordSpec(beta=beta, sigma=sigma))
        assert list(multiple.components) == ["beta", "sigma"]
        assert multiple.spec == RecordSpec(beta=beta, sigma=sigma)

        parameters = RecordSpec(beta=beta, sigma=sigma)
        wrapped = OutputSpec(parameters=parameters)
        assert wrapped.spec is parameters
        assert dict(wrapped.components) == {"parameters": parameters}
        assert wrapped != OutputSpec(parameters)

    def test_output_exposure_is_immediate_and_keeps_higher_order_kinds(self):
        nested = RecordSpec(beta=(3,), sigma=())
        law = DistributionSpec(nested)
        exposed = OutputSpec(RecordSpec(parameters=nested, law=law))
        assert list(exposed.components) == ["parameters", "law"]
        assert exposed.components["parameters"] is nested
        assert exposed.components["law"] is law
        assert list(OutputSpec(posterior=law).components) == ["posterior"]

    def test_record_none_is_opaque_not_an_output_hole(self):
        output = OutputSpec(RecordSpec(payload=None))
        assert isinstance(output.components["payload"], OpaqueSpec)
        assert output.is_concrete

    def test_output_empty_record_is_explicit(self):
        output = OutputSpec(RecordSpec())
        assert isinstance(output.spec, RecordSpec)
        assert dict(output.components) == {}
        with pytest.raises(ValueError):
            OutputSpec()

    def test_output_rejects_ambiguous_or_untyped_forms(self):
        spec = NumericArraySpec(())
        for args, kwargs in [
            ((spec,), {}),
            ((RecordSpec(x=()), RecordSpec(y=())), {}),
            ((RecordSpec(x=()),), {"y": spec}),
            ((), {"x": None, "y": spec}),
            ((), {"x": ()}),
        ]:
            with pytest.raises(TypeError):
                OutputSpec(*args, **kwargs)

    def test_output_dimension_transforms_preserve_exposure_and_holes(self):
        spec = RecordSpec(x=("n",))
        for output in (OutputSpec(parameters=spec), OutputSpec(spec)):
            renamed = output.with_dim_names(n="m").with_dims(m=3)
            assert list(renamed.components) == list(output.components)
            assert renamed.spec == RecordSpec(x=(3,))
        hole = OutputSpec(beta=None)
        assert hole.with_dims(n=3).with_dim_names(n="m") == hole


class TestDeclarationConstruction:
    @pytest.mark.parametrize(
        ("decl", "field"),
        [(InputSpec(x=NumericArraySpec(())), "_slots"), (OutputSpec(x=None), "_term_spec")],
    )
    def test_slotted_declarations_reject_attribute_mutation(self, decl, field):
        assert not hasattr(decl, "__dict__")
        original = getattr(decl, field)
        with pytest.raises(FrozenInstanceError):
            setattr(decl, field, original)
        with pytest.raises(FrozenInstanceError):
            delattr(decl, field)
        with pytest.raises(AttributeError):
            _ = decl.unknown_attribute
        # Frozen slotted dataclasses can raise TypeError on unknown writes in Python 3.12.
        with pytest.raises((AttributeError, TypeError)):
            decl.unknown_attribute = None
        with pytest.raises((AttributeError, TypeError)):
            del decl.unknown_attribute
        assert getattr(decl, field) is original
        assert not hasattr(decl, "unknown_attribute")

    @pytest.mark.parametrize("name", ["name", "spec", "components", "record_spec", "slots"])
    def test_constructor_keywords_are_component_names(self, name):
        spec = NumericArraySpec(())
        assert dict(OutputSpec(**{name: spec}).components) == {name: spec}
        assert InputSpec(**{name: spec})[name] is spec

    @pytest.mark.parametrize("name", ["a/b", "", "two words", "1x", "class"])
    def test_component_names_are_python_identifiers(self, name):
        spec = NumericArraySpec(())
        with pytest.raises(ValueError):
            InputSpec({name: spec})
        with pytest.raises(ValueError):
            OutputSpec(**{name: spec})
        if name not in ("", "a/b"):
            with pytest.raises(ValueError):
                OutputSpec(RecordSpec({name: spec}))

    @pytest.mark.parametrize("value", [None, (), {}, 1, "spec"])
    def test_input_slots_require_specs(self, value):
        with pytest.raises(TypeError):
            InputSpec(x=value)

    def test_declarations_do_not_alias_mutable_constructor_mappings(self):
        source = {"x": NumericArraySpec(())}
        inputs = InputSpec(source)
        source["y"] = OpaqueSpec()
        assert list(inputs) == ["x"]
        with pytest.raises(TypeError):
            inputs["x"] = source["x"]
        with pytest.raises(TypeError):
            InputSpec(source, y=OpaqueSpec())
        for output in (OutputSpec(x=source["x"]), OutputSpec(RecordSpec(x=()))):
            with pytest.raises(TypeError):
                output.components["x"] = OpaqueSpec()
            with pytest.raises(FrozenInstanceError):
                output._term_spec = OpaqueSpec()


class TestDeclarationRoundTrips:
    @pytest.mark.parametrize(
        "decl",
        [
            InputSpec(x=NumericArraySpec(("n",))),
            OutputSpec(x=None),
            OutputSpec(x=NumericArraySpec(("n",))),
            OutputSpec(RecordSpec(x=("n",))),
            RecordSpec(group=RecordSpec(x=("n",))),
        ],
    )
    def test_declaration_copy_and_pickle(self, decl):
        for restored in (copy.copy(decl), copy.deepcopy(decl), pickle.loads(pickle.dumps(decl))):
            assert restored == decl
            assert hash(restored) == hash(decl)
            assert type(restored) is type(decl)

    def test_declarations_can_be_static_jit_arguments(self):
        @jax.jit(static_argnames="spec")
        def total(x, spec):
            assert spec.spec.is_valid(x)
            return jnp.sum(x)

        spec = OutputSpec(x=NumericArraySpec((3,)))
        assert total(jnp.arange(3.0), spec) == 3


class TestDimensionBinding:
    @pytest.mark.parametrize(
        ("expected", "actual_spec", "actual_value", "value_error"),
        [
            pytest.param(
                NumericArraySpec((3,)),
                RecordSpec(x=(3,)),
                Record("r", x=np.zeros(3)),
                "does not conform",
                id="array-receives-record",
            ),
            pytest.param(
                RecordSpec(x=(3,)),
                NumericArraySpec((3,)),
                np.zeros(3),
                "expected named fields",
                id="record-receives-array",
            ),
        ],
    )
    @pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
    def test_spec_binding_never_unwraps_a_single_field_record(
        self, expected, actual_spec, actual_value, value_error, nested
    ):
        if nested:
            expected = RecordSpec(field=expected)
            actual_spec = RecordSpec(field=actual_spec)
            actual_value = {"field": actual_value}
        with pytest.raises(ValueError, match="does not conform"):
            expected.bind_dims_from_spec(actual_spec)
        with pytest.raises(ValueError, match=value_error):
            expected.bind_dims_from_value(actual_value)

    def test_shared_symbol_scope_across_input_slots_and_nested_batch(self):
        slots = InputSpec(
            data=NumericArraySpec(("n",)),
            batch=BatchSpec(NumericArraySpec(("n",)), [("n",)], ["draw"]),
        )
        actual = InputSpec(
            data=NumericArraySpec((3,)),
            batch=BatchSpec(NumericArraySpec((3,)), [(3,)], ["draw"]),
        )
        assert slots.bind_dims_from_spec(actual) == actual
        conflict = InputSpec(dict(actual) | {"data": NumericArraySpec((4,))})
        with pytest.raises(ValueError, match="already bound"):
            slots.bind_dims_from_spec(conflict)
        assert slots.free_dims == {"n"}

    def test_bind_from_value_and_spec_preserves_declared_metadata(self):
        expected = RecordSpec(x=NumericArraySpec(("n",), dtype="float64"), y=("n",))
        values = {"x": np.zeros(3, dtype="float32"), "y": np.ones(3)}
        bound = expected.bind_dims_from_value(values)
        assert bound == expected.with_dims(n=3)
        assert expected.bind_dims_from_spec(RecordSpec(x=(3,), y=(3,))) == bound
        assert expected.is_valid(values)
        assert not expected.is_valid(values | {"y": np.ones(4)})
        assert not expected.is_valid({"x": np.zeros(3)})
        assert expected.free_dims == {"n"}

    def test_dimension_renaming_is_simultaneous_and_crosses_all_containers(self):
        spec = RecordSpec(
            data=NumericArraySpec(("n", "m"), dtype="float32"),
            batch=BatchSpec(NumericArraySpec(("n",)), [("m",)], ["draw"]),
            law=DistributionSpec(RecordSpec(x=("n",))),
            function=FunctionSpec(RecordSpec(x=("m",)), NumericArraySpec(("n",))),
        )
        renamed = spec.with_dim_names(n="m", m="n")
        assert renamed["data"].shape == ("m", "n")
        assert renamed["data"].dtype == np.dtype("float32")
        assert renamed["batch"].axis_groups == (("n",),)
        assert renamed["batch"].element_spec.shape == ("m",)
        assert renamed["batch"].level_names == ("draw",)
        assert renamed["law"].event_spec["x"].shape == ("m",)
        assert renamed["function"].input_template["x"].shape == ("n",)
        assert renamed["function"].output_spec.shape == ("m",)
        assert spec["data"].shape == ("n", "m")
        for value in (None, "", 2):
            with pytest.raises(TypeError):
                spec.with_dim_names(n=value)
        with pytest.raises(ValueError):
            spec.with_dims(n=-1)

    def test_binding_array_values_checks_dtype_and_record_kind(self):
        spec = NumericArraySpec(("n",), dtype="int32")
        with pytest.raises(ValueError, match="does not conform"):
            spec.bind_dims_from_value(np.zeros(3, dtype="float32"))
        record = Record("record", x=np.zeros(3))
        assert not spec.is_valid(record)
        with pytest.raises(ValueError):
            spec.bind_dims_from_value(record)


class TestSpecKinds:
    def test_all_specs_use_one_kind_protocol(self):
        for spec in (NumericArraySpec(()), OpaqueSpec(), RecordSpec(x=()), FunctionSpec()):
            assert isinstance(spec, TermSpec)
        numeric = RecordSpec(group=RecordSpec(x=(2, 3)), y=())
        assert isinstance(numeric, NumericRecordSpec)
        assert isinstance(numeric, NumericSpec)
        assert numeric.vector_size == 7
        assert numeric.leaf_shapes == {"group/x": (2, 3), "y": ()}
        assert NumericArraySpec(()).vector_size == 1
        assert NumericArraySpec((0, 3)).vector_size == 0
        with pytest.raises(ValueError):
            _ = NumericArraySpec(("n",)).vector_size

    def test_record_spec_inference_keeps_stored_term_kinds(self):
        from probpipe import NumericArray, Opaque

        array = NumericArray("array", np.zeros(2), spec=NumericArraySpec((2,), dtype="float64"))
        opaque = Opaque("opaque", object(), spec=OpaqueSpec(meta="metadata"))
        record = Record("record", x=1.0)
        inferred = RecordSpec.infer_from({"array": array, "opaque": opaque, "record": record})
        assert inferred["array"] is array.spec
        assert inferred["opaque"] is opaque.spec
        assert inferred.children["record"] is record.spec

    def test_record_schema_infers_distribution_and_callable_kinds(self):
        from probpipe import Function, Normal

        law = Normal(0.0, 1.0, name="x")
        function = Function(
            func=lambda x: x, input_template=RecordSpec(x=()), output_template=RecordSpec(y=())
        )
        schema = RecordSpec.infer_from(
            {"law": law, "function": function, "raw_callable": lambda x: x}
        )
        assert schema["law"] == DistributionSpec(law.event_template)
        assert schema["function"] == FunctionSpec(RecordSpec(x=()), RecordSpec(y=()))
        assert schema["raw_callable"] == FunctionSpec()

    def test_empirical_without_event_template_remains_a_record_field(self):
        from probpipe import EmpiricalDistribution

        law = EmpiricalDistribution(["a", "b"], name="law")
        schema = RecordSpec.infer_from({"law": law})
        assert schema["law"] == OpaqueSpec()

        record = Record("r", law=law)
        assert record["law"] is law
        assert record.spec == schema
        assert schema.is_valid(record)

    def test_numeric_batch_field_keeps_its_batch_kind(self):
        from probpipe import NumericArrayBatch

        batch = NumericArrayBatch(
            "draws", np.zeros((2, 3)), "draw", element_spec=NumericArraySpec((3,))
        )
        schema = RecordSpec.infer_from({"batch": batch})
        assert schema["batch"] is batch.spec
        assert type(schema) is RecordSpec
        record = Record("container", batch=batch)
        assert record.spec == schema
        assert not NumericArraySpec((2, 3)).is_valid(batch)

    def test_numeric_record_layout_reads_each_numeric_specs_own_layout(self):
        from dataclasses import dataclass

        from probpipe import NumericRecord, NumericRecordBatch

        @dataclass(frozen=True)
        class Coordinates(NumericSpec):
            size: int

            def _vector_size(self):
                return self.size

            def is_valid(self, value):
                return np.shape(value) == (self.size,)

        schema = RecordSpec(x=Coordinates(4), nested=RecordSpec(y=(2, 3)), label=None)
        numeric = schema.numeric_subset()
        assert isinstance(numeric, NumericRecordSpec)
        assert numeric.vector_size == 10
        assert numeric.leaf_shapes == {"x": (4,), "nested/y": (2, 3)}
        with pytest.raises(TypeError, match=r"field 'x'.*requires NumericArraySpec"):
            NumericRecord.from_vector("value", numeric, np.zeros(10))
        with pytest.raises(TypeError, match=r"field 'x'.*requires NumericArraySpec"):
            NumericRecordBatch.from_vector("values", numeric, np.zeros((2, 10)), level_names="row")


class TestDistributionSchemaAvailability:
    def test_inference_keeps_a_distribution_without_schema_as_an_opaque_field(
        self, law_without_schema
    ):
        inferred = RecordSpec.infer_from({"law": law_without_schema})
        assert inferred == RecordSpec(law=OpaqueSpec())
        record = Record("r", law=law_without_schema)
        assert record["law"] is law_without_schema
        assert record.spec == inferred
        assert inferred.is_valid(record)

    def test_inference_preserves_an_available_distribution_schema(self, declared_law):
        template = RecordSpec(x=NumericArraySpec((3,), dtype="float64", support=positive))
        law = declared_law(template)
        inferred = RecordSpec.infer_from({"law": law})
        assert inferred["law"] == DistributionSpec(template)
        assert inferred["law"].event_spec is template
        record = Record("r", law=law)
        assert record["law"] is law
        assert record.spec == inferred

    @pytest.mark.parametrize("error_type", [RuntimeError, ValueError, KeyError])
    def test_unexpected_schema_getter_errors_propagate(self, error_type):
        error = error_type("broken schema getter")

        class BrokenLaw(Distribution):
            @property
            def event_template(self):
                raise error

        law = BrokenLaw(name="law")
        with pytest.raises(error_type) as caught:
            RecordSpec.infer_from({"law": law})
        assert caught.value is error
        with pytest.raises(error_type) as caught:
            DistributionSpec(RecordSpec(x=("n",))).bind_dims_from_value(law)
        assert caught.value is error


class TestInputSpec:
    def test_inputs_follow_mapping_equality_while_preserving_slot_order(self):
        array = NumericArraySpec(())
        left = InputSpec(x=array, y=OpaqueSpec())
        right = InputSpec(y=OpaqueSpec(), x=array)
        assert left == dict(left)
        assert left == right
        assert hash(left) == hash(right)
        assert list(left) == ["x", "y"]
        assert list(right) == ["y", "x"]

    def test_binding_values_preserves_slots_metadata_and_the_declaration(self):
        from probpipe.core.constraints import positive

        array = NumericArraySpec(("n",), dtype="float64", support=positive)
        inputs = InputSpec(data=array, nested=RecordSpec(x=("n",)))
        bound = inputs.bind_dims_from_value(
            {"nested": Record("r", x=np.ones(3)), "data": np.ones(3, dtype="float32")}
        )
        assert bound == inputs.with_dims(n=3)
        assert list(bound) == ["data", "nested"]
        assert bound["data"].support is positive
        assert bound["data"].dtype == np.dtype("float64")
        assert inputs.free_dims == {"n"}

    @pytest.mark.parametrize("values", [{}, {"x": np.ones(3), "extra": 1}])
    def test_binding_requires_exact_slot_names(self, values):
        inputs = InputSpec(x=NumericArraySpec(("n",)))
        with pytest.raises(ValueError, match="slots"):
            inputs.bind_dims_from_value(values)
        actual = InputSpec({name: NumericArraySpec(()) for name in values})
        with pytest.raises(ValueError, match="slots"):
            inputs.bind_dims_from_spec(actual)

    def test_empty_mapping_and_dimension_transforms(self):
        empty = InputSpec()
        assert dict(empty) == {}
        assert empty.bind_dims_from_value({}) == empty
        assert empty.bind_dims_from_spec(InputSpec()) == empty
        assert empty.is_concrete
        inputs = InputSpec(x=NumericArraySpec(("n",)), y=NumericArraySpec(("m",)))
        renamed = inputs.with_dim_names(n="m", m="n")
        assert renamed["x"].shape == ("m",)
        assert renamed["y"].shape == ("n",)
        assert renamed.with_dims(n=2, m=3) == InputSpec(
            x=NumericArraySpec((3,)), y=NumericArraySpec((2,))
        )


class TestRecordValueValidation:
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("as_record", [False, True])
    @pytest.mark.parametrize("dtype, valid", [("int32", False), ("float64", True)])
    def test_dtype_checks_actual_fields(self, nested, as_record, dtype, valid):
        spec = RecordSpec(x=NumericArraySpec(("n",), dtype=dtype))
        data = {"x": np.zeros(3, dtype="float32")}
        if nested:
            spec = RecordSpec(group=spec)
            data = {"group": data}
        value = Record("value", data) if as_record else data

        assert spec.is_valid(value) is valid
        if valid:
            assert spec.bind_dims_from_value(value) == spec.with_dims(n=3)
        else:
            with pytest.raises(ValueError, match="does not conform"):
                spec.bind_dims_from_value(value)
        assert spec.free_dims == {"n"}

    def test_a_spec_in_a_value_slot_does_not_stand_in_for_an_array(self):
        spec = RecordSpec(x=(3,))
        values = {"x": NumericArraySpec((3,))}
        assert not spec.is_valid(values)
        with pytest.raises(ValueError, match="does not conform"):
            spec.bind_dims_from_value(values)

    def test_spec_binding_uses_only_declared_dtype_information(self):
        spec = RecordSpec(x=NumericArraySpec(("n",), dtype="int32"))
        assert spec.bind_dims_from_spec(RecordSpec(x=(3,))) == spec.with_dims(n=3)

    def test_validation_and_binding_work_under_jit_and_vmap(self):
        spec = RecordSpec(x=NumericArraySpec(("n",), dtype="float32"))

        @jax.jit
        def total(x):
            record = Record("row", x=x)
            assert spec.is_valid(record)
            assert spec.bind_dims_from_value(record) == spec.with_dims(n=3)
            return jnp.sum(record["x"])

        rows = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
        np.testing.assert_allclose(jax.vmap(total)(rows), [3.0, 12.0], rtol=0, atol=0)


class TestNestedValueBinding:
    @pytest.fixture(params=["direct", "record", "nested_record", "input"])
    def wrap_binding(self, request):
        def wrap(spec, value):
            match request.param:
                case "direct":
                    return spec, value
                case "record":
                    return RecordSpec(value=spec), {"value": value}
                case "nested_record":
                    return RecordSpec(group=RecordSpec(value=spec)), {"group": {"value": value}}
                case "input":
                    return InputSpec(value=spec), {"value": value}

        return wrap

    def test_unavailable_distribution_schema_cannot_bind(self, wrap_binding, law_without_schema):
        spec = DistributionSpec(RecordSpec(x=("n",)))
        declared, value = wrap_binding(spec, law_without_schema)
        with pytest.raises(ValueError, match="exposes no schema to bind it against") as caught:
            declared.bind_dims_from_value(value)
        if isinstance(declared, RecordSpec):
            assert f"RecordSpec/{next(iter(declared))} declares" in str(caught.value)
        assert declared.free_dims == {"n"}

        concrete, value = wrap_binding(spec.with_dims(n=3), law_without_schema)
        with pytest.raises(ValueError, match="does not conform"):
            concrete.bind_dims_from_value(value)

    @pytest.mark.parametrize("kind", ["function", "distribution"])
    @pytest.mark.parametrize("size", [3, 4])
    def test_fixed_and_concretized_specs_follow_the_same_binding_rules(
        self, wrap_binding, kind, size
    ):
        from probpipe import EmpiricalDistribution, Function

        if kind == "function":
            spec_type = FunctionSpec
            reference = Function(func=lambda x: x, input_template=RecordSpec(x=(3,)))
            actual = Function(func=lambda x: x, input_template=RecordSpec(x=(size,)))
        else:
            spec_type = DistributionSpec
            reference = EmpiricalDistribution(np.zeros((2, 3)), name="x")
            actual = EmpiricalDistribution(np.zeros((2, size)), name="x")

        # Concrete distributions require exact metadata; function binding reads
        # the available declarations without checking callable compatibility.
        dtype = "float64" if kind == "function" else None
        symbolic = spec_type(RecordSpec(x=NumericArraySpec(("n",), dtype=dtype)))
        fixed = spec_type(RecordSpec(x=NumericArraySpec((3,), dtype=dtype)))
        declared, value = wrap_binding(symbolic, actual)
        expected, _ = wrap_binding(symbolic.with_dims(n=size), actual)
        bound = declared.bind_dims_from_value(value)
        assert bound == expected
        assert bound.bind_dims_from_value(value) == bound

        for spec in (fixed, symbolic.with_dims(n=3), symbolic.bind_dims_from_value(reference)):
            assert spec == fixed
            declared, value = wrap_binding(spec, actual)
            if size == 3:
                assert declared.bind_dims_from_value(value) == declared
            else:
                message = "dimension 4, expected 3" if kind == "function" else "does not conform"
                with pytest.raises(ValueError, match=message):
                    declared.bind_dims_from_value(value)
        assert symbolic.free_dims == {"n"}

    @pytest.mark.parametrize(
        "expected, actual",
        [
            (
                RecordSpec(x=NumericArraySpec((3,), dtype="float64")),
                RecordSpec(x=NumericArraySpec((3,), dtype="float32")),
            ),
            (RecordSpec(x=(3,), y=()), RecordSpec(y=(), x=(3,))),
            (
                RecordSpec(x=NumericArraySpec((3,), support=positive)),
                RecordSpec(x=(3,)),
            ),
            (
                RecordSpec(x=(3,)),
                RecordSpec(x=NumericArraySpec((3,), support=positive)),
            ),
        ],
        ids=["dtype", "field_order", "missing_support", "extra_support"],
    )
    def test_concrete_distribution_values_require_exact_schemas(
        self, wrap_binding, declared_law, expected, actual
    ):
        law = declared_law(actual)
        spec = DistributionSpec(expected)
        declared, value = wrap_binding(spec, law)
        with pytest.raises(ValueError, match="does not conform"):
            declared.bind_dims_from_value(value)
        schema = RecordSpec(law=spec)
        assert not spec.is_valid(law)
        assert not schema.is_valid({"law": law})
        assert not schema.is_valid(Record("value", law=law))
        with pytest.raises(ValueError):
            Record("value", law=law, event_template=schema)

        matching = declared_law(expected)
        declared, value = wrap_binding(spec, matching)
        assert declared.bind_dims_from_value(value) == declared
        assert schema.is_valid(Record("value", law=matching, event_template=schema))

    def test_concretized_distribution_uses_the_same_strict_metadata_check(
        self, wrap_binding, declared_law
    ):
        symbolic = DistributionSpec(RecordSpec(x=NumericArraySpec(("n",), dtype="float64")))
        law = declared_law(RecordSpec(x=(3,)))
        bound = symbolic.bind_dims_from_value(law)
        fixed = DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype="float64")))
        for spec in (bound, symbolic.with_dims(n=3), fixed):
            assert spec == fixed
            declared, value = wrap_binding(spec, law)
            with pytest.raises(ValueError, match="does not conform"):
                declared.bind_dims_from_value(value)

    def test_missing_callable_declarations_remain_unspecified(self, wrap_binding):
        symbolic = FunctionSpec(RecordSpec(x=("n",)))
        for spec in (symbolic, symbolic.with_dims(n=3)):
            declared, value = wrap_binding(spec, lambda x: x)
            assert declared.bind_dims_from_value(value) == declared
        assert symbolic.free_dims == {"n"}


class TestNestedSpecBinding:
    @pytest.fixture(params=["direct", "record", "nested_record", "input", "batch"])
    def wrap_spec(self, request):
        def wrap(spec):
            match request.param:
                case "direct":
                    return spec
                case "record":
                    return RecordSpec(value=spec)
                case "nested_record":
                    return RecordSpec(group=RecordSpec(value=spec))
                case "input":
                    return InputSpec(value=spec)
                case "batch":
                    return BatchSpec(spec, [(2,)], ["draw"])

        return wrap

    @pytest.mark.parametrize(
        "expected, actual, result",
        [
            (FunctionSpec(), FunctionSpec(output_spec=NumericArraySpec((3,))), FunctionSpec()),
            (
                FunctionSpec(RecordSpec(x=("n",))),
                FunctionSpec(),
                FunctionSpec(RecordSpec(x=("n",))),
            ),
            (
                FunctionSpec(output_spec=NumericArraySpec(("n",))),
                FunctionSpec(output_spec=NumericArraySpec((3,))),
                FunctionSpec(output_spec=NumericArraySpec((3,))),
            ),
            (
                NumericArraySpec((3,)),
                NumericArraySpec((3,), dtype="float32"),
                NumericArraySpec((3,)),
            ),
        ],
    )
    def test_binding_is_independent_of_nesting(self, wrap_spec, expected, actual, result):
        declared = wrap_spec(expected)
        assert declared.bind_dims_from_spec(wrap_spec(actual)) == wrap_spec(result)
        assert declared == wrap_spec(expected)

    @pytest.mark.parametrize(
        "expected, actual",
        [
            (FunctionSpec(), OpaqueSpec()),
            (
                FunctionSpec(output_spec=NumericArraySpec((2,))),
                FunctionSpec(output_spec=NumericArraySpec((3,))),
            ),
            (NumericArraySpec((3,), dtype="int32"), NumericArraySpec((3,), dtype="float32")),
            (NumericArraySpec((3,)), RecordSpec(x=(3,))),
            (DistributionSpec(RecordSpec(x=(3,))), OpaqueSpec()),
            (DistributionSpec(RecordSpec(x=(3,))), RecordSpec(x=(3,))),
            (DistributionSpec(RecordSpec(x=(3,))), DistributionSpec(RecordSpec(x=(4,)))),
            (
                DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype="int32"))),
                DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype="float32"))),
            ),
        ],
    )
    def test_mismatches_are_rejected_at_every_nesting_depth(self, wrap_spec, expected, actual):
        with pytest.raises(ValueError):
            wrap_spec(expected).bind_dims_from_spec(wrap_spec(actual))

    @pytest.mark.parametrize("size", [3, "n"])
    @pytest.mark.parametrize("dtype", [None, "float32"])
    def test_distribution_spec_binding_preserves_declared_order_and_metadata(
        self, wrap_spec, size, dtype
    ):
        expected = DistributionSpec(
            RecordSpec(x=NumericArraySpec((size,), dtype="float64", support=positive), y=())
        )
        actual = DistributionSpec(RecordSpec(y=(), x=NumericArraySpec((3,), dtype=dtype)))
        declared = wrap_spec(expected)
        bound = declared.bind_dims_from_spec(wrap_spec(actual))
        assert bound == wrap_spec(expected.with_dims(n=3))
        assert bound.bind_dims_from_spec(wrap_spec(actual)) == bound
        assert declared == wrap_spec(expected)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_distribution_spec_binding_shares_dimensions(self, wrap_spec, reverse):
        expected = DistributionSpec(RecordSpec(x=("n",), nested=RecordSpec(y=("n",))))
        fields = [("x", NumericArraySpec((3,))), ("nested", RecordSpec(y=(4,)))]
        if reverse:
            fields.reverse()
        with pytest.raises(ValueError, match="already bound"):
            wrap_spec(expected).bind_dims_from_spec(
                wrap_spec(DistributionSpec(RecordSpec(dict(fields))))
            )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_input_value_binding_shares_dimensions_in_either_order(self, reverse):
        fields = [("x", NumericArraySpec(("n",))), ("group", RecordSpec(y=("n",)))]
        if reverse:
            fields.reverse()
        inputs = InputSpec(dict(fields))
        with pytest.raises(ValueError, match="already bound"):
            inputs.bind_dims_from_value({"x": np.zeros(3), "group": {"y": np.zeros(4)}})
