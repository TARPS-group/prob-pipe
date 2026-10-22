"""Distribution truth values, multiplication, and the refusal of coupled merges."""

from __future__ import annotations

import pytest

from probpipe import (
    Normal,
    NumericArraySpec,
    Record,
    RecordSpec,
    conditional_distribution,
    marginal,
)


def _prior():
    return Normal("mu", 0.0, 1.0, label="p")


def _likelihood():
    return conditional_distribution(
        lambda mu: Normal("y", mu, 1.0),
        given_spec={"mu": NumericArraySpec(())},
        label="k",
    )


def _root():
    return (Normal("a", 0.0, 1.0) & Normal("b", 0.0, 1.0)).with_label("d")


@pytest.mark.parametrize("left_factory", [_prior, _likelihood])
@pytest.mark.parametrize("right_factory", [_prior, _likelihood])
def test_distribution_multiplication_permanently_refuses(left_factory, right_factory):
    left, right = left_factory(), right_factory()
    with pytest.raises(TypeError) as error:
        left * right
    assert (
        str(error.value) == "cannot multiply distributions with *. To merge them, write lik & prior"
    )


@pytest.mark.parametrize("factory", [_prior, _likelihood])
def test_distribution_truth_value_permanently_refuses(factory):
    law = factory()
    for evaluate in (lambda: bool(law), lambda: law and _prior(), lambda: law or _prior()):
        with pytest.raises(TypeError) as error:
            evaluate()
        assert (
            str(error.value)
            == "a distribution has no truth value. To merge distributions, write a & b"
        )
    assert law.__mul__(2.0) is NotImplemented


def test_ordinary_records_keep_their_length_based_truth_value():
    assert not RecordSpec({})
    assert bool(Record({"a": "value"}))


def test_order_refusal_uses_the_prescribed_message():
    with pytest.raises(ValueError) as error:
        _prior() & _likelihood()
    assert str(error.value) == "cannot merge p & k: k takes mu, which p outputs. Write k & p"


def test_coupled_siblings_use_the_prescribed_message():
    root = _root()
    with pytest.raises(ValueError) as error:
        root["a"] & root["b"]
    assert str(error.value) == (
        "cannot merge d['a'] & d['b']: both are coupled marginals of d, which a merge "
        "would treat as independent. For their joint law, write d['a', 'b']. "
        "For independent laws, write marginal(d, 'a') & marginal(d, 'b')"
    )


def test_coupling_survives_a_nested_merge():
    root = _root()
    with pytest.raises(ValueError, match="both are coupled marginals of d"):
        (root["a"] & Normal("c", 0.0, 1.0)) & root["b"]


def test_coupling_is_checked_before_duplicate_output_names():
    root = _root().rename({"a": "g1/theta", "b": "g2/theta"}).with_label("d")
    with pytest.raises(ValueError, match="both are coupled marginals of d"):
        root["g1"] & root["g2"]


def test_renamed_siblings_remain_coupled():
    root = _root()
    with pytest.raises(ValueError, match="both are coupled marginals of d"):
        root["a"].rename({"a": "left"}) & root["b"].rename({"b": "right"})


def test_a_root_and_its_rename_remain_coupled():
    root = _root()
    with pytest.raises(ValueError, match="both read draws"):
        root & root.rename({"a": "left", "b": "right"})


def test_standalone_marginals_and_unrelated_laws_merge():
    root = _root()
    independent = marginal(root, "a") & marginal(root, "b")
    assert list(independent.event_spec.outputs) == ["a", "b"]
    assert list((root["a"] & Normal("c", 0.0, 1.0)).event_spec.outputs) == ["a", "c"]


def test_matching_parameters_do_not_couple_distinct_roots():
    left = Normal("a", 0.0, 1.0)
    right = Normal("a", 0.0, 1.0).rename({"a": "b"})
    assert list((left & right).event_spec.outputs) == ["a", "b"]


def test_repeated_selection_suggests_a_valid_joint_and_renaming_shared_outputs():
    root = _root()
    with pytest.raises(ValueError) as error:
        root["a"] & root["a"]
    assert "write d['a']." in str(error.value)
    assert "d['a', 'a']" not in str(error.value)
    assert "rename shared outputs" in str(error.value)
