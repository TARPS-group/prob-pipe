"""How records, record specs, batches, and laws read paths and lists of paths.

Every term indexed by path reads what ``[]`` takes one way: a ``str`` is one
path, and a list or a tuple of ``str`` is a list of paths, which keeps each node
at its own path in the term's order. The tests check that one reading, its
errors, and the invariants it gives across the kinds.
"""

from __future__ import annotations

import pathlib
import re

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    NumericArraySpec,
    Record,
    RecordSpec,
    marginal,
    mean,
    sample,
    variance,
    workflow_run,
)
from probpipe.core.named_tree import _PATH_SEP


def _grouped():
    """A law whose draws are records ``{a, g: {c, e}}``."""
    joint = Normal("a", 0.0, 1.0) * Normal("c", 1.0, 2.0) * Normal("e", 2.0, 3.0)
    return joint.with_path_names({"c": "g/c", "e": "g/e"})


def _hier():
    """A law whose draws are records ``{s1: {theta}, s2: {theta}, mu}``."""
    joint = Normal("t1", 0.0, 1.0) * Normal("t2", 0.0, 2.0) * Normal("mu", 0.0, 1.0)
    return joint.with_path_names({"t1": "s1/theta", "t2": "s2/theta"})


def _draw():
    return Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}}, label="draw")


def _draws():
    with workflow_run(seed=0):
        return sample(_grouped(), sample_shape=5)


def _sample(law, seed=0, **arguments):
    with workflow_run(seed=seed):
        return sample(law, **arguments)


#: One term of each kind indexed by path, over the paths a, g, g/c, g/e.
_TERMS = {
    "record": _draw,
    "record-spec": lambda: _draw().spec,
    "batch": _draws,
    "law": _grouped,
}


def _field_paths(term):
    """The paths of the fields of *term*: a record's or a spec's, a batch's elements', or a law's draws'."""
    if isinstance(term, Record | RecordSpec):
        return term.keys()
    if hasattr(term, "element_spec"):
        return term.element_spec.keys()
    return term.event_spec.spec.keys()


@pytest.fixture(params=list(_TERMS), ids=list(_TERMS))
def term(request):
    return _TERMS[request.param]()


class TestTheReaderErrors:
    """Each row of the reader's error table, on every kind."""

    @pytest.mark.parametrize(
        ("key", "error", "match"),
        [
            pytest.param("", ValueError, "has an empty name", id="empty-path"),
            pytest.param("g//c", ValueError, "has an empty name", id="empty-name"),
            pytest.param("/a", ValueError, "has an empty name", id="leading-separator"),
            pytest.param([], ValueError, "it names no paths", id="empty-list"),
            pytest.param((), ValueError, "it names no paths", id="empty-tuple"),
            pytest.param(("a", "a"), ValueError, "names the path 'a' twice", id="repeated"),
            pytest.param(("g", "g/c"), ValueError, "'g/c' is inside 'g'", id="nested"),
            pytest.param({"a"}, TypeError, "got set", id="set"),
            pytest.param(iter(["a"]), TypeError, "got list_iterator", id="iterator"),
            pytest.param(("a", 0), TypeError, None, id="path-and-position"),
        ],
    )
    def test_a_malformed_key_raises_before_any_lookup(self, term, key, error, match):
        with pytest.raises(error, match=match):
            term[key]

    def test_an_unknown_path_raises_key_error(self, term):
        with pytest.raises(KeyError):
            term["nope"]
        with pytest.raises(KeyError):
            term["a", "nope"]


class TestBatchPositions:
    """A batch also takes positions, and refuses a list or an array of them."""

    def test_a_path_and_a_position_in_one_key_raise(self):
        with pytest.raises(TypeError, match="mixes paths with positions"):
            _draws()["a", 0]

    @pytest.mark.parametrize(
        "positions", [[0, 2], np.array([0, 2]), jnp.array([0, 2])], ids=["list", "numpy", "jax"]
    )
    def test_a_list_or_an_array_of_positions_raises(self, positions):
        with pytest.raises(TypeError):
            _draws()[positions]

    def test_an_empty_tuple_never_returns_the_whole_batch(self):
        with pytest.raises(ValueError, match="it names no paths"):
            _draws()[()]

    @pytest.mark.parametrize("key", [None, True, ...], ids=["none", "bool", "ellipsis"])
    def test_none_a_bool_and_an_ellipsis_raise(self, key):
        with pytest.raises(TypeError):
            _draws()[key]


class TestOnePathOrAList:
    """One path gives the node; a list of paths keeps each node at its path (I9)."""

    def test_a_record_gives_a_group_and_a_cut_down_record(self):
        draw = _draw()
        assert draw["g"].keys() == ("c", "e")
        assert draw["g"]["c"] == draw["g/c"]
        assert draw["a", "g/c"].keys() == ("a", "g/c")
        assert draw["a", "g/c"].label == "(a, g/c)"
        assert draw[["g"]].keys() == ("g/c", "g/e")
        assert draw.keys() == ("a", "g/c", "g/e")

    def test_a_record_spec_gives_a_group_and_a_cut_down_spec(self):
        spec = _draw().spec
        assert spec["g"] == RecordSpec(c=(), e=())
        assert spec["a", "g/c"] == RecordSpec(a=(), g=RecordSpec(c=()))

    def test_a_list_keeps_the_terms_order_whatever_order_it_gives(self, term):
        assert _field_paths(term["g/e", "a", "g/c"]) == ("a", "g/c", "g/e")
        assert _field_paths(term["g/e", "a"]) == ("a", "g/e")

    def test_a_law_indexed_by_several_paths_keeps_their_paths(self):
        hier = _hier()
        view = hier["s1/theta", "s2/theta"]
        assert view.event_spec.spec == RecordSpec(
            s1=RecordSpec(theta=Normal("x", 0.0, 1.0).event_spec.spec),
            s2=RecordSpec(theta=Normal("x", 0.0, 1.0).event_spec.spec),
        )
        draw = _sample(view, seed=1)
        assert draw.keys() == ("s1/theta", "s2/theta")
        np.testing.assert_allclose(variance(view)["s2/theta"], 4.0, rtol=1e-6)

    def test_a_list_of_one_whole_term_name_keeps_todays_record(self):
        x = Normal("x", 0.0, 1.0)
        assert x[["x"]].event_spec.spec == RecordSpec(x=x.event_spec.spec)
        assert x["x"] is x


class TestMarginal:
    """``marginal(d, paths)`` takes what ``d[paths]`` takes and gives the same law (I7)."""

    @pytest.mark.parametrize("paths", [["a", "g/c"], ("g/c", "a")], ids=["list", "tuple"])
    def test_a_list_of_paths_gives_the_law_of_the_view(self, paths):
        grouped = _grouped()
        detached, view = marginal(grouped, paths), grouped[paths]
        assert detached.event_spec == view.event_spec
        assert detached.event_spec.spec.keys() == ("a", "g/c")
        np.testing.assert_allclose(variance(detached)["g/c"], 4.0, rtol=1e-6)
        np.testing.assert_allclose(mean(detached)["g/c"], mean(view)["g/c"], rtol=1e-6)

    def test_paths_with_one_last_name_keep_their_groups(self):
        detached = marginal(_hier(), ["s2/theta", "s1/theta"])
        assert detached.event_spec.spec.keys() == ("s1/theta", "s2/theta")
        np.testing.assert_allclose(variance(detached)["s1/theta"], 1.0, rtol=1e-6)


class TestMembership:
    """``p in t`` is true when ``p`` is a ``str`` and ``t[p]`` gives a node."""

    @pytest.mark.parametrize("path", ["a", "g", "g/c"])
    def test_every_node_is_a_member(self, term, path):
        assert path in term

    @pytest.mark.parametrize("key", ["nope", "g/x", "", ("a", "g/c"), 0, None])
    def test_anything_else_is_not(self, term, key):
        assert key not in term

    def test_a_batch_never_tests_its_elements(self):
        draws = _sample(Normal("x", 0.0, 1.0), sample_shape=3)
        assert float(draws.raw[0]) not in draws
        assert "x" not in draws  # x names the whole element; it is not a path inside it

    def test_a_whole_term_name_is_a_member_of_its_law(self):
        assert "x" in Normal("x", 0.0, 1.0)


class TestOnePathPerArgument:
    """``without``, ``select``, and ``at_path`` read one path per argument."""

    def test_without_refuses_a_repeated_or_nested_path_and_a_tuple(self):
        draw = _draw()
        with pytest.raises(ValueError, match="names the path 'a' twice"):
            draw.without("a", "a")
        with pytest.raises(ValueError, match="'g/c' is inside 'g'"):
            draw.without("g", "g/c")
        with pytest.raises(TypeError, match="give each path as a separate argument"):
            draw.without(("a", "g/e"))

    def test_at_path_takes_separate_segments_and_refuses_a_tuple(self):
        draw = _draw()
        assert draw.at_path("g", "c") == draw["g/c"]
        with pytest.raises(TypeError, match="give each segment as a separate argument"):
            draw.at_path(("g", "c"))

    def test_select_keys_each_path_by_its_last_name(self):
        draw = _draw()
        assert list(draw.select("a", "g/c")) == ["a", "c"]
        assert list(draw.select(x="g/c")) == ["x"]
        with pytest.raises(ValueError, match="its key 'theta' is already taken by 's1/theta'"):
            _sample(_hier()).select("s1/theta", "s2/theta")
        assert list(_draws().select("a", "g/c")) == ["a", "c"]


class TestInvariants:
    """The invariants a reading of paths gives on records, batches, and laws."""

    def test_positions_and_paths_commute_on_a_batch(self):
        """I4: ``draws[i][s] == draws[s][i]``."""
        draws = _draws()
        assert draws[3]["g"] == draws["g"][3]
        assert draws[3]["a", "g/c"] == draws["a", "g/c"][3]
        np.testing.assert_array_equal(draws[3]["g/c"].raw, draws["g/c"][3].raw)

    def test_paths_chain_on_records_and_batches(self):
        """I5: ``t[p][q] == t[p + "/" + q]``."""
        draw, draws = _draw(), _draws()
        assert draw["g"]["c"] == draw[f"g{_PATH_SEP}c"]
        np.testing.assert_array_equal(draws["g"]["c"].raw, draws["g/c"].raw)

    @pytest.mark.parametrize("paths", [("a",), ("g/c",), ("a", "g/e"), ("g",)], ids=repr)
    def test_a_list_of_paths_and_without_partition_the_fields(self, paths):
        """I10, on records, record specs, and batches."""
        for whole in (_draw(), _draw().spec):
            chosen, rest = whole[paths], whole.without(*paths)
            assert set(chosen.keys()) | set(rest.keys()) == set(whole.keys())
            assert not set(chosen.keys()) & set(rest.keys())
        draws = _draws()
        chosen, rest = draws[paths], draws.without(*paths)
        assert set(chosen.element_spec.keys()) | set(rest.element_spec.keys()) == set(
            draws.element_spec.keys()
        )

    def test_a_view_at_a_list_of_paths_draws_its_parents_draw_at_them(self):
        """I2 for lists: one key draws the parent and the view alike."""
        grouped = _grouped()
        view_draw = _sample(grouped["a", "g/c"], seed=7)
        parent_draw = _sample(grouped, seed=7)
        assert view_draw.keys() == ("a", "g/c")
        np.testing.assert_allclose(view_draw["g/c"].raw, parent_draw["g/c"].raw)

    def test_one_set_of_paths_on_a_law_its_draws_and_its_mean(self):
        """I1 and I3: the paths of a law index its draws, its batches, and its mean."""
        grouped = _grouped()
        draw, draws, grouped_mean = _sample(grouped), _draws(), mean(grouped)
        for paths in ("g", "g/c", ("a", "g/c")):
            assert draw[paths].spec == draws[paths].element_spec
        for paths in ("g", ("a", "g/c")):
            assert _field_paths(draw[paths]) == _field_paths(grouped[paths])
        assert grouped_mean["g"].keys() == ("c", "e")
        np.testing.assert_allclose(grouped_mean["g"]["e"].raw, 2.0)
        np.testing.assert_allclose(mean(grouped["a", "g/c"])["g/c"].raw, grouped_mean["g/c"].raw)

    @pytest.mark.pending(
        reason="a view of a group exposes the group's parts once laws are declared whole only "
        "over base specs",
        raises=KeyError,
    )
    def test_paths_chain_through_a_law_at_a_group(self):
        """I5 for laws: ``grouped["g"]["c"]`` is the view of ``g/c``."""
        grouped = _grouped()
        assert grouped["g"]["c"].event_spec == grouped["g/c"].event_spec


class TestOneSeparator:
    """The path separator is defined once, in ``named_tree.py``."""

    def test_no_other_module_defines_the_separator(self):
        package = pathlib.Path(__file__).parents[2] / "probpipe"
        definitions = [
            str(path.relative_to(package))
            for path in package.rglob("*.py")
            if re.search(r"^_PATH_SEP\s*=", path.read_text(), re.MULTILINE)
        ]
        assert definitions == [str(pathlib.Path("core") / "named_tree.py")]

    def test_the_separator_spells_a_path_of_a_spec_and_its_value(self):
        assert _draw().spec[_PATH_SEP.join(("g", "c"))] == NumericArraySpec(())
