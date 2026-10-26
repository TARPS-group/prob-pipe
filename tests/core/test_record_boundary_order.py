"""Record order is contractual; raw mappings acquire the declared order."""

from __future__ import annotations

import pytest

from probpipe import (
    Function,
    InputSpec,
    Normal,
    OutputSpec,
    Record,
    RecordSpec,
    condition_on,
    conditional_distribution,
    mean,
)
from probpipe.core._spec_base import _unify_specs


@pytest.mark.parametrize("nested", [False, True])
def test_record_conformance_and_unification_require_declared_order(nested):
    declared = RecordSpec(x=(), y=())
    reversed_spec = RecordSpec(y=(), x=())
    value = Record({"y": 2.0, "x": 1.0})
    raw = {"y": 2.0, "x": 1.0}
    if nested:
        declared = RecordSpec(g=declared)
        reversed_spec = RecordSpec(g=reversed_spec)
        value = Record({"g": value})
        raw = {"g": raw}
    assert not declared.is_valid(value)
    assert declared.is_valid(raw)
    for first, second in [(declared, reversed_spec), (reversed_spec, declared)]:
        with pytest.raises(ValueError, match="not in the declared order"):
            _unify_specs(first, second, {}, "record")
    with pytest.raises(ValueError) as error:
        declared._bind_dims_from_value(value, {}, "record")
    assert str(error.value) == (
        "fields ['y', 'x'] are not in the declared order ['x', 'y']; pass a plain mapping, "
        "which is laid out in the declared order, or build the record in that order"
    )


@pytest.mark.parametrize("method", ["apply", "__call__"])
def test_function_inputs_lay_out_raw_mappings_without_changing_the_caller(method):
    schema = RecordSpec(g=RecordSpec(x=(), y=()))
    function = Function(lambda r: next(iter(r["g"])), input_spec=InputSpec(r=schema))
    raw = {"g": {"y": 2.0, "x": 1.0}}
    result = getattr(function, method)(raw)
    assert getattr(result, "raw", result) == "x"
    assert list(raw["g"]) == ["y", "x"]
    with pytest.raises(ValueError, match="not in the declared order"):
        getattr(function, method)(Record(raw))


@pytest.mark.parametrize("method", ["apply", "__call__"])
def test_function_returns_lay_out_raw_mappings_and_refuse_reordered_records(method):
    schema = RecordSpec(g=RecordSpec(x=(), y=()))
    raw = {"g": {"y": 2.0, "x": 1.0}}
    function = Function(lambda: raw, output_spec=OutputSpec(schema))
    result = getattr(function, method)()
    assert list(result["g"]) == ["x", "y"]
    assert list(raw["g"]) == ["y", "x"]
    function = Function(lambda: Record(raw), output_spec=OutputSpec(schema))
    with pytest.raises(ValueError, match="not in the declared order"):
        getattr(function, method)()


def test_kernel_binding_lays_out_raw_mapping_and_refuses_existing_record_order():
    @conditional_distribution(given_spec={"r": RecordSpec(x=(), y=())})
    def kernel(r):
        return Normal("z", r[next(iter(r))], 1.0)

    raw = {"y": 2.0, "x": 1.0}
    result = mean(condition_on(kernel, {"r": raw}))
    assert float(result) == 1.0
    assert list(raw) == ["y", "x"]
    with pytest.raises(ValueError, match="not in the declared order"):
        condition_on(kernel, {"r": Record(raw)})


def test_removing_unknown_inputs_is_an_error_and_optionality_survives():
    inputs = InputSpec(a=(), b=()).with_optional("b")
    assert inputs.without("a") == InputSpec(b=()).with_optional("b")
    assert inputs.without() == inputs
    with pytest.raises(KeyError, match="unknown input"):
        inputs.without("missing")


@pytest.mark.parametrize(
    "names, error",
    [
        ((1,), TypeError),
        ((("a",),), TypeError),
        (("",), ValueError),
        (("a", "a"), ValueError),
        (("g", "g/b"), ValueError),
    ],
)
def test_removal_uses_the_shared_name_reader(names, error):
    with pytest.raises(error):
        InputSpec(a=(), g=RecordSpec(b=())).without(*names)
