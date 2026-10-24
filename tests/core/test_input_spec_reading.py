"""One reading of input sources, input selection, and compatible input merges."""

from __future__ import annotations

import pytest

from probpipe import (
    Function,
    InputSpec,
    Normal,
    NumericArraySpec,
    OutputSpec,
    RecordSpec,
    conditional_distribution,
)
from probpipe.distributions import ConditionalDistributionSpec

SCALAR = NumericArraySpec(())


@pytest.mark.parametrize(
    "source",
    [
        {"a": SCALAR, "g": RecordSpec(b=())},
        RecordSpec(a=(), g=RecordSpec(b=())),
        OutputSpec(RecordSpec(a=(), g=RecordSpec(b=()))),
        InputSpec(a=(), g=RecordSpec(b=())),
    ],
)
def test_four_sources_describe_top_level_inputs(source):
    result = InputSpec(source)
    assert list(result) == ["a", "g"]
    assert result["g"] == RecordSpec(b=())
    assert result.required == ("a", "g")


def test_copy_preserves_optional_flags_and_hash():
    original = InputSpec(a=(), b=("n",)).with_optional("b")
    copied = InputSpec(original)
    assert copied == original
    assert hash(copied) == hash(original)
    assert copied.optional == frozenset({"b"})


def test_size_shorthand_is_shared_by_inputs_and_keyword_outputs():
    assert InputSpec(a=())["a"] == SCALAR
    assert InputSpec({"a": (3,)})["a"] == NumericArraySpec((3,))
    assert OutputSpec(a=()).term_spec == SCALAR
    for constructor in (InputSpec, OutputSpec):
        with pytest.raises(TypeError):
            constructor(a=3)


def test_unspecified_outputs_and_nonidentifier_inputs_are_refused():
    with pytest.raises(TypeError, match="unspecified"):
        InputSpec(OutputSpec(a=None))
    with pytest.raises(ValueError, match="identifiers"):
        InputSpec(OutputSpec(**{"not-an-input": SCALAR}))


def test_input_list_selection_retains_order_and_optionality():
    inputs = InputSpec(a=(), b=(), c=()).with_optional("b")
    selected = inputs[["c", "b"]]
    assert isinstance(selected, InputSpec)
    assert list(selected) == ["b", "c"]
    assert selected.optional == frozenset({"b"})
    assert inputs[("b",)] == InputSpec(b=()).with_optional("b")
    assert inputs["a"] == SCALAR


@pytest.mark.parametrize(
    "names,exception",
    [
        ([], ValueError),
        (["a", "a"], ValueError),
        (["a", "a/b"], ValueError),
        (["missing"], KeyError),
        (["a", 0], TypeError),
        ({"a"}, TypeError),
    ],
)
def test_input_selection_uses_the_shared_reader(names, exception):
    with pytest.raises(exception):
        InputSpec(a=())[names]


@pytest.mark.parametrize("name", [["a"], ("a",), 0, None, "missing"])
def test_membership_and_get_refuse_nonnames_without_hashing_them(name):
    inputs = InputSpec(a=())
    assert name not in inputs
    assert inputs.get(name) is None
    assert inputs.get(name, SCALAR) is SCALAR


def test_merge_binds_shared_dimensions_and_keeps_optional_flags():
    left = InputSpec(a=("n",), b=("n",)).with_optional("a", "b")
    right = InputSpec(a=(3,), c=()).with_optional("c")
    merged = left & right
    assert list(merged) == ["a", "b", "c"]
    assert merged["b"] == NumericArraySpec((3,))
    assert merged.optional == frozenset({"b", "c"})
    assert left["a"] == NumericArraySpec(("n",))


def test_merge_refuses_a_conflicting_shared_dimension():
    with pytest.raises(ValueError):
        InputSpec(a=("n",), b=("n",)) & InputSpec(a=(3,), b=(4,))


def test_output_specs_need_an_explicit_conversion_before_merging():
    with pytest.raises(TypeError) as error:
        InputSpec(a=()) & OutputSpec(b=())
    assert str(error.value) == (
        "cannot merge an InputSpec with an OutputSpec. "
        "Convert the output spec first, as InputSpec(output_spec)"
    )


@pytest.mark.parametrize("source", [RecordSpec(mu=()), OutputSpec(mu=())])
def test_constructors_read_record_and_output_specs(source):
    kernel = conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec=source)
    assert list(kernel.given_spec) == ["mu"]
    assert ConditionalDistributionSpec(source, OutputSpec(y=())).given_spec == InputSpec(mu=())
    function = Function(lambda mu: mu, input_spec=source)
    assert function.input_spec == InputSpec(mu=())


@pytest.mark.parametrize(
    "factory",
    [
        lambda spec: Function(lambda mu: mu, input_spec=spec),
        lambda spec: conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec=spec),
    ],
)
def test_optional_inputs_require_parameter_defaults(factory):
    with pytest.raises(ValueError, match="has no default"):
        factory(InputSpec(mu=()).with_optional("mu"))


def test_function_defaults_mark_inputs_optional():
    function = Function(lambda mu=1.0: mu, input_spec=InputSpec(mu=()))
    assert function.input_spec.optional == frozenset({"mu"})


@pytest.mark.parametrize("source", [Normal("mu", 0.0, 1.0), Normal("mu", 0.0, 1.0).spec])
def test_laws_and_their_types_have_the_prescribed_source_message(source):
    with pytest.raises(TypeError) as error:
        conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec=source)
    kind = "Distribution" if isinstance(source, Normal) else "DistributionSpec"
    assert (
        str(error.value)
        == f"given_spec cannot be a {kind}. Pass its event_spec to declare the inputs its outputs supply"
    )


def test_an_input_the_function_does_not_take_has_the_prescribed_message():
    def fn(mu):
        return Normal("y", mu, 1.0)

    with pytest.raises(TypeError) as error:
        conditional_distribution(fn, given_spec=OutputSpec(RecordSpec(mu=(), sigma=())))
    assert (
        str(error.value)
        == "given_spec declares sigma, but fn takes only mu. Pass event_spec.without('sigma') or event_spec[['mu']]"
    )


def test_group_parts_have_the_prescribed_message():
    def fn(g):
        return Normal("y", g["a"] + g["b"], 1.0)

    with pytest.raises(TypeError) as error:
        conditional_distribution(fn, given_spec=OutputSpec(RecordSpec(a=(), b=())))
    assert (
        str(error.value)
        == "given_spec declares the fields a and b of the group g, but fn takes g itself. Pass event_spec[['g']]"
    )
