"""The invariants of paths, inputs, and outputs, one test per invariant I1–I22.

Each test runs over the running examples: ``prior``, a law over ``{mu, tau}``;
``grouped``, a law over ``{a, g: {c, e}}``; ``hier``, a law over
``{s1: {theta}, s2: {theta}, mu}``; ``lik``, a kernel ``lik(y | mu)``; and
their draws. The part of an invariant that a later change delivers is marked
``pending`` with the change it awaits and the exception today's code raises,
so it fails loudly once delivered.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    InputSpec,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Record,
    RecordSpec,
    condition_on,
    conditional_distribution,
    marginal,
    mean,
    sample,
    standard_deviation,
    variance,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError

_REAL = NumericArraySpec(())


def _prior():
    return Normal("mu", 0.0, 5.0) * Normal("tau", 1.0, 1.0)


def _grouped():
    joint = Normal("a", 0.0, 1.0) * Normal("c", 1.0, 2.0) * Normal("e", 2.0, 3.0)
    return joint.with_path_names({"c": "g/c", "e": "g/e"})


def _hier():
    joint = Normal("t1", 0.0, 1.0) * Normal("t2", 0.0, 2.0) * Normal("mu", 0.0, 1.0)
    return joint.with_path_names({"t1": "s1/theta", "t2": "s2/theta"})


def _lik():
    return conditional_distribution(
        lambda mu: Normal("y", mu, 1.0), given_spec={"mu": _REAL}, label="lik"
    )


def _emulator():
    return Function(
        lambda theta: {"mean": theta, "var": theta * theta},
        output_spec=OutputSpec(RecordSpec(mean=_REAL, var=_REAL)),
        label="emulator",
    )


def _sample(law, seed=0, **arguments):
    with workflow_run(seed=seed):
        return sample(law, **arguments)


def _succeeds(term, key) -> bool:
    try:
        term[key]
    except (KeyError, TypeError, ValueError):
        return False
    return True


#: Paths and lists of paths of ``grouped``, and keys that are not paths of it.
_PATHS = ["a", "g", "g/c", "g/e", ("a", "g/c"), ["g/e", "a"], ("g",)]
_NOT_PATHS = ["nope", "g/x", "", "g//c", ("a", "a"), ("g", "g/c"), ("nope",)]


def test_i1_one_set_of_paths_on_a_law_its_draws_its_batches_and_its_mean():
    """I1 for records and batches: the paths of ``grouped`` index its draws, elements, and mean."""
    grouped = _grouped()
    draw, draws, grouped_mean = _sample(grouped), _sample(grouped, sample_shape=4), mean(grouped)
    for paths in _PATHS:
        assert draw[paths].spec == draws[paths].element_spec
        assert draws[2][paths] == draws[paths][2]
        assert grouped_mean[paths].spec == draw[paths].spec
        assert _succeeds(grouped, paths)


def test_i1_on_the_output_spec():
    """I1 for output specs: a path of ``grouped`` selects from its event spec the spec of its draws' node."""
    grouped = _grouped()
    draw = _sample(grouped)
    for paths in _PATHS:
        assert grouped.event_spec[paths].term_spec == draw[paths].spec


def test_i2_indexing_commutes_with_sampling_for_lists_of_paths():
    """I2 for lists: one seed draws the view at a list of paths and the parent there alike."""
    grouped = _grouped()
    for paths in [("a", "g/c"), ("g",), ("g/e", "a")]:
        view_draw = _sample(grouped[paths], seed=7)
        parent_draw = _sample(grouped, seed=7)[paths]
        assert view_draw.spec == parent_draw.spec
        for leaf in view_draw:
            np.testing.assert_allclose(view_draw[leaf].raw, parent_draw[leaf].raw)


def test_i3_indexing_commutes_with_summaries():
    """I3: exact summaries of a view equal the summaries of the parent at its paths."""
    grouped = _grouped()
    for summary in (mean, variance, standard_deviation):
        whole = summary(grouped)
        for paths in ("g/c", ("a", "g/c")):
            part = summary(grouped[paths])
            expected = whole[paths]
            if isinstance(paths, str):
                np.testing.assert_allclose(part.raw, expected.raw, rtol=1e-6)
            else:
                for leaf in expected:
                    np.testing.assert_allclose(part[leaf].raw, expected[leaf].raw, rtol=1e-6)


def test_i4_positions_and_paths_commute_on_a_batch():
    """I4: ``bt[j][s] == bt[s][j]``."""
    draws = _sample(_grouped(), sample_shape=5)
    for paths in ("g", ("a", "g/c"), ["g"]):
        assert draws[3][paths] == draws[paths][3]
    np.testing.assert_array_equal(draws[3]["g/c"].raw, draws["g/c"][3].raw)


def test_i5_paths_chain_on_records_and_batches():
    """I5 for records and batches: ``t[p][q] == t[p + "/" + q]``."""
    draw, draws = _sample(_grouped()), _sample(_grouped(), sample_shape=3)
    assert draw["g"]["c"] == draw["g/c"]
    np.testing.assert_array_equal(draws["g"]["c"].raw, draws["g/c"].raw)
    assert draw.spec["g"]["c"] == draw.spec["g/c"]


@pytest.mark.pending(reason="G1a: a view of a group offers the group's parts", raises=KeyError)
def test_i5_paths_chain_through_a_law_at_a_group():
    grouped = _grouped()
    assert grouped["g"]["c"].event_spec == grouped["g/c"].event_spec


@pytest.mark.pending(reason="G3: paths continue into a field that holds a law", raises=KeyError)
def test_i5_paths_chain_through_a_held_law():
    record_with_law = Record({"n": 3.0, "G": _prior()}, label="record_with_law")
    assert record_with_law["G/mu"].event_spec == record_with_law["G"]["mu"].event_spec


def test_i6_a_records_spec_agrees_with_its_specs_paths():
    """I6 for records: ``r[s].spec == r.spec[s]``."""
    draw = _sample(_grouped())
    for paths in ("g", "g/c", ("a", "g/c")):
        assert draw[paths].spec == draw.spec[paths]


#: Laws and the paths of each that a view's event spec reads through the parent's.
#: One path to a group is left out until views of groups have the group's
#: children as outputs (T5).
_VIEWED = [
    (_prior, ["mu", "tau", ("tau", "mu"), ("mu",)]),
    (_grouped, ["a", "g/c", ("a", "g/c"), ["g/e", "a"], ("g",)]),
    (_hier, ["mu", "s1/theta", ("s1/theta", "s2/theta"), ("s2",), ("mu", "s1")]),
]


@pytest.mark.parametrize(("law", "paths"), [(law, p) for law, ps in _VIEWED for p in ps], ids=repr)
def test_i6_a_views_output_spec_agrees_with_its_parents(law, paths):
    """I6 for laws: the event spec of ``d[s]`` and of ``marginal(d, s)`` is ``d.event_spec[s]``."""
    parent = law()
    assert parent[paths].event_spec == parent.event_spec[paths]
    assert marginal(parent, paths).event_spec == parent.event_spec[paths]


def test_i7_a_view_and_a_marginal_at_a_list_of_paths_are_one_law():
    """I7 for lists: ``marginal(d, s)`` is ``d[s]``, detached."""
    for law, paths in [(_grouped(), ["g/c", "a"]), (_hier(), ("s2/theta", "s1/theta"))]:
        view, detached = law[paths], marginal(law, paths)
        assert view.event_spec == detached.event_spec
        for leaf in view.event_spec.term_spec:
            np.testing.assert_allclose(
                variance(view)[leaf].raw, variance(detached)[leaf].raw, rtol=1e-6
            )


@pytest.mark.pending(
    reason="G4: conditioning reads a given record by its paths", raises=ResolutionError
)
def test_i8_draws_of_a_list_of_paths_condition_back():
    grouped = _grouped()
    conditioned = condition_on(grouped, _sample(grouped[["a", "g/c"]]))
    assert conditioned.event_spec.term_spec.keys() == ("g/e",)


def test_i9_one_path_gives_the_node_and_a_list_a_record_that_holds_it():
    """I9: ``t["g"]`` is the node, ``t[["g"]]`` the record ``{g: ...}``."""
    draw = _sample(_grouped())
    assert draw["g"].keys() == ("c", "e")
    assert draw[["g"]].keys() == ("g/c", "g/e")
    assert _grouped()[["g"]].event_spec.term_spec.keys() == ("g/c", "g/e")
    x = Normal("x", 0.0, 1.0)
    assert x[["x"]].event_spec == OutputSpec(RecordSpec(x=x.event_spec.term_spec))


@pytest.mark.pending(reason="S4: a batch stores its elements' output name", raises=TypeError)
def test_i9_a_batchs_output_at_the_root():
    draws = _sample(Normal("x", 0.0, 1.0), sample_shape=4)
    assert draws[["x"]].element_spec.keys() == ("x",)


@pytest.mark.parametrize("paths", [("a",), ("g/c",), ("a", "g/e"), ("g",)], ids=repr)
def test_i10_a_list_of_paths_and_without_partition_the_fields(paths):
    """I10 for records, record specs, and batches."""
    draw = _sample(_grouped())
    draws = _sample(_grouped(), sample_shape=3)
    for whole, fields in [
        (draw, lambda term: term.keys()),
        (draw.spec, lambda term: term.keys()),
        (draws, lambda term: term.element_spec.keys()),
    ]:
        chosen, rest = fields(whole[paths]), fields(whole.without(*paths))
        assert set(chosen) | set(rest) == set(fields(whole))
        assert not set(chosen) & set(rest)


@pytest.mark.parametrize("paths", [("g/e",), ("a",), ("g",), ("a", "g/c")], ids=repr)
def test_i10_on_an_output_spec(paths):
    """I10 for output specs: a list and removal partition the leaf paths."""
    spec = _grouped().event_spec
    kept = set(spec[list(paths)].term_spec.keys())
    rest = set(spec.without(*paths).term_spec.keys())
    assert kept | rest == set(spec.term_spec.keys())
    assert not kept & rest


@pytest.mark.pending(reason="S2b-in: InputSpec[] takes a list of input names", raises=TypeError)
def test_i10_on_an_input_spec():
    inputs = InputSpec(a=_REAL, b=_REAL)
    assert set(inputs[["a"]]) | set(inputs.without("a")) == {"a", "b"}


@pytest.mark.pending(reason="G4: law.without", raises=AttributeError)
def test_i10_on_a_law():
    grouped = _grouped()
    assert grouped.without("g/e").event_spec == grouped[["a", "g/c"]].event_spec


@pytest.mark.pending(
    reason="G4: the result of condition_on is the event spec without what it fixes",
    raises=AssertionError,
)
def test_i11_conditioning_removes_what_it_fixes():
    prior = _prior()
    assert condition_on(prior, {"mu": 0.1}).event_spec == prior.event_spec.without("mu")


@pytest.mark.pending(reason="G3: the accessor of the term at an output", raises=AttributeError)
def test_i12_the_term_at_an_output():
    grouped = _grouped()
    assert grouped.output_name("g").event_spec == grouped["g"].event_spec


@pytest.mark.pending(reason="G3: [] on a Function", raises=TypeError)
def test_i13_indexing_commutes_with_producing():
    emulator = _emulator()
    theta = jnp.asarray(2.0)
    np.testing.assert_allclose(emulator["mean"](theta).raw, emulator(theta)["mean"].raw)


@pytest.mark.pending(reason="G3: [] on a Function", raises=TypeError)
def test_i14_indexing_keeps_inputs():
    emulator = _emulator()
    assert emulator["mean"].input_spec == emulator.input_spec


def test_i15_a_kernels_sides_are_disjoint():
    """I15 for kernels, which holds today."""
    lik = _lik()
    assert not set(lik.given_spec) & set(lik.event_spec.outputs)


@pytest.mark.pending(reason="S3: & merges laws and kernels", raises=TypeError)
def test_i15_a_merges_sides_are_disjoint():
    model = _lik() & _prior()
    assert not set(getattr(model, "given_spec", {})) & set(model.event_spec.outputs)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (_lik, _prior),
        (_grouped, lambda: Normal("x", 0.0, 1.0)),
        (_hier, lambda: Normal("z", 0.0, 1.0)),
        (_lik, lambda: _grouped().with_path_names({"a": "mu"})),
        (lambda: Normal("x", 0.0, 1.0), _hier),
    ],
    ids=["lik*prior", "grouped*x", "hier*z", "lik*grouped", "x*hier"],
)
def test_i16_the_spec_of_a_merge_is_the_merge_of_the_specs(left, right):
    """I16 for outputs: the event spec of a merge is the merge of the operands' event specs."""
    first, second = left(), right()
    assert (first * second).event_spec == first.event_spec & second.event_spec


def test_i17_a_merge_keeps_paths():
    grouped = _grouped()
    joint = grouped * Normal("x", 0.0, 1.0)
    assert joint.event_spec["g/c"] == grouped.event_spec["g/c"]


@pytest.mark.parametrize("key", _PATHS + _NOT_PATHS + [0, None], ids=repr)
def test_i18_membership_agrees_with_indexing_on_terms_and_record_specs(key):
    """I18 for terms and record specs: ``key in t`` iff ``key`` is a str and ``t[key]`` succeeds."""
    draw = _sample(_grouped())
    for term in (draw, draw.spec, _sample(_grouped(), sample_shape=3), _grouped()):
        assert (key in term) == (isinstance(key, str) and _succeeds(term, key))


@pytest.mark.parametrize("key", _PATHS + _NOT_PATHS + [0, None], ids=repr)
def test_i18_on_an_output_spec(key):
    """I18 for output specs: ``key in O`` iff ``key`` is a str and ``O[key]`` succeeds."""
    for spec in (_grouped().event_spec, Normal("x", 0.0, 1.0).event_spec):
        assert (key in spec) == (isinstance(key, str) and _succeeds(spec, key))


@pytest.mark.pending(reason="S3: rename", raises=AttributeError)
def test_i18_rename_takes_the_same_paths():
    assert _grouped().rename({"g/c": "g/k"}).event_spec.term_spec["g/k"] == _REAL


def test_i19_a_merge_is_associative():
    """I19, which holds today for ``*``."""
    a, b, c = Normal("a", 0.0, 1.0), Normal("b", 1.0, 2.0), Normal("c", 2.0, 3.0)
    left, right = (a * b) * c, a * (b * c)
    assert left.event_spec == right.event_spec
    for name in ("a", "b", "c"):
        np.testing.assert_allclose(variance(left)[name].raw, variance(right)[name].raw)


@pytest.mark.pending(reason="S3: & merges laws", raises=TypeError)
def test_i20_sampling_commutes_with_merge():
    merged = Normal("x", 0.0, 1.0) & Normal("y", 0.0, 2.0)
    assert _sample(merged).keys() == ("x", "y")


@pytest.mark.pending(reason="S3: rename", raises=AttributeError)
def test_i21_merge_commutes_with_renaming():
    x, y = Normal("x", 0.0, 1.0), Normal("y", 0.0, 2.0)
    assert (x * y).rename(x="z").event_spec == (x.rename(x="z") * y).event_spec


def test_i22_a_merge_reads_no_label():
    """I22, which holds today: relabeling an operand changes only the result's label."""
    a, b = Normal("a", 0.0, 1.0), Normal("b", 1.0, 2.0)
    relabeled = a.with_label("first") * b
    plain = a * b
    assert relabeled.event_spec == plain.event_spec
    assert relabeled.label != plain.label
    np.testing.assert_allclose(variance(relabeled)["b"].raw, variance(plain)["b"].raw)


@pytest.mark.pending(
    reason="G1a: a view of a group offers the group's parts", raises=AssertionError
)
def test_i6_a_view_of_a_group_agrees_with_its_parents_output_spec():
    grouped = _grouped()
    assert grouped["g"].event_spec == grouped.event_spec["g"]


@pytest.mark.pending(reason="G3: lists of paths continue into a held law", raises=KeyError)
def test_i9_a_list_of_paths_through_a_held_law():
    record_with_law = Record({"n": 3.0, "G": _prior()}, label="record_with_law")
    assert record_with_law[["G/mu"]].keys() == ("G",)


@pytest.mark.pending(reason="G5: [] on a kernel", raises=TypeError)
def test_i13_indexing_commutes_with_a_kernel():
    lik = _lik()
    assert lik["y"].event_spec == lik.event_spec


@pytest.mark.pending(reason="G3: in follows paths through a held law", raises=AssertionError)
def test_i18_membership_through_a_held_law():
    record_with_law = Record({"n": 3.0, "G": _prior()}, label="record_with_law")
    assert "G/mu" in record_with_law
