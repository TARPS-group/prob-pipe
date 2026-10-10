"""Semantic constructor names remain independent of numerical declarations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    DistributionBatch,
    Function,
    FunctionBatch,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    Opaque,
    OutputSpec,
    Record,
    RecordBatch,
    ResultSchemaError,
    function,
    workflow_run,
)
from probpipe.core._fingerprint import fingerprint


def test_raw_values_require_names_but_named_structures_derive_them():
    with pytest.raises(TypeError, match="label"):
        NumericArray(1.0)
    with pytest.raises(TypeError, match="label"):
        Opaque(object())
    with pytest.raises(TypeError, match="label"):
        NumericArrayBatch(jnp.ones(3), "draw")
    assert Record({"temperature": 1.0}).label == "record(temperature)"
    assert RecordBatch({"temperature": jnp.ones(3)}, "draw").label == "record(temperature)"
    with pytest.raises(TypeError, match="empty record requires label"):
        Record({})
    assert Record({}, label="empty_measurements").label == "empty_measurements"


def test_record_controls_and_field_names_have_distinct_namespaces():
    value = Record({"label": "north", "name": "station"}, label="measurements")
    assert value.label == "measurements"
    assert value.raw("label") == "north"
    assert value.raw("name") == "station"
    numeric = NumericRecord.from_fields(label=1.0, name=2.0)
    assert numeric.label == "record(label,name)"
    assert tuple(numeric) == ("label", "name")


def test_coercion_requires_a_semantic_source():
    with pytest.raises(TypeError, match="unnamed value"):
        Record.ensure(jnp.ones(3))
    assert Record.ensure(jnp.ones(3), label="temperature").label == "temperature"
    assert Record.ensure({"temperature": jnp.ones(3)}).label == "record(temperature)"


def test_function_defaults_and_explicit_aliases():
    def predict(temperature):
        return temperature + 1

    assert Function(predict).notation == "predict(temperature)"
    assert Function(lambda temperature: temperature + 1).notation == "f(temperature)"
    assert function(lambda temperature: temperature + 1).notation == "f(temperature)"
    assert Function(predict)(NumericArray(2.0, label="ambient")).label == "predict(ambient)"
    assert Function(predict, output_label="prediction")(2.0).label == "prediction"


def test_output_components_are_declared_independently_of_aliases():
    with pytest.raises(TypeError, match="declared component"):
        Function(lambda x: x, output_spec=NumericArraySpec(()))
    declared = OutputSpec(prediction=NumericArraySpec(()))
    f = Function(lambda x: x + 1, label="predict", output_label="forecast", output_spec=declared)
    g = Function(f.raw(), label="other", output_label="estimate", output_spec=declared)
    assert f.output_spec == g.output_spec
    assert fingerprint(f) == fingerprint(g)
    with workflow_run(seed=42):
        first = f(Normal("temperature", 0, 1))
    with workflow_run(seed=42):
        second = g(Normal("temperature", 0, 1))
    assert tuple(first.event_spec.components) == ("prediction",)
    assert tuple(second.event_spec.components) == ("prediction",)
    np.testing.assert_array_equal(first.atoms.raw(), second.atoms.raw())


def test_generic_law_default_uses_a_declared_component():
    law = Distribution(OutputSpec(tau=NumericArraySpec(())))
    assert law.notation == "p(tau)"
    assert law.with_label("prior").event_spec == law.event_spec


def test_collection_defaults_are_bounded_and_describe_members():
    laws = [Normal("tau", 0, 1, label=f"prior{i}") for i in range(20)]
    batch = DistributionBatch(laws, "model")
    assert "prior0(tau)" in batch.label
    assert "prior7(tau)" in batch.label
    assert "prior8" not in batch.label
    assert "…" in batch.label
    assert FunctionBatch([lambda x: x], "model").label == "[f(x)]"
    with pytest.raises(TypeError, match="empty collection requires label"):
        DistributionBatch([], "model", element_spec=laws[0].spec)


@pytest.mark.parametrize("transform", [jax.jit, jax.vmap])
def test_reconstructed_terms_remain_usable_in_transforms(transform):
    value = NumericArray(jnp.arange(3.0), label="temperature")
    rebuilt = transform(lambda x: x)(value)
    assert rebuilt.label == "<no description>"
    result = transform(lambda x: x + 1)(rebuilt)
    np.testing.assert_array_equal(np.asarray(result), jnp.arange(3.0) + 1)
    increment = Function(lambda x: x + 1)
    np.testing.assert_array_equal(np.asarray(transform(increment)(rebuilt)), jnp.arange(3.0) + 1)


def test_relabeling_reuses_compilation_and_managed_results_derive_names():
    traces = []

    @jax.jit
    def compiled(x):
        traces.append(None)
        return x + 1

    first = NumericArray(1.0, label="temperature")
    compiled(first)
    compiled(first.with_label("pressure"))
    assert len(traces) == 1
    wrapped = Function(compiled, label="increment")
    assert wrapped(first).label == "increment(temperature)"
    assert float(wrapped(first)) == 2.0


def test_lifting_requires_components_from_a_declaration_or_record_fields():
    unnamed = Function(lambda x: x + 1, label="predict", n_broadcast_samples=5)
    with pytest.raises(ResultSchemaError, match="needs a named component"):
        unnamed(Normal("temperature", 0, 1))
    exposed = Function(lambda x: {"prediction": x + 1}, n_broadcast_samples=5)
    assert tuple(exposed(Normal("temperature", 0, 1)).event_spec.components) == ("prediction",)


@pytest.mark.parametrize(
    "value",
    [
        NumericRecord({"temperature": jnp.arange(3.0)}),
        NumericArrayBatch(jnp.arange(3.0), "draw", label="temperature"),
        NumericRecordBatch({"temperature": jnp.arange(3.0)}, "draw"),
    ],
)
def test_all_numeric_rebuilds_keep_specs_and_work_without_descriptions(value):
    rebuilt = jax.jit(lambda x: x)(value)
    assert rebuilt.label == "<no description>"
    assert rebuilt.spec == value.spec
    result = jax.jit(lambda x: x)(rebuilt)
    for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(value), strict=True):
        np.testing.assert_array_equal(actual, expected)
