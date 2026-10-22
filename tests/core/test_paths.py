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
    joint = (Normal("a", 0.0, 1.0) & Normal("c", 1.0, 2.0)) & Normal("e", 2.0, 3.0)
    return joint.rename({"c": "g/c", "e": "g/e"})


def _hier():
    """A law whose draws are records ``{s1: {theta}, s2: {theta}, mu}``."""
    joint = (Normal("t1", 0.0, 1.0) & Normal("t2", 0.0, 2.0)) & Normal("mu", 0.0, 1.0)
    return joint.rename({"t1": "s1/theta", "t2": "s2/theta"})


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
        return term.element_spec.term_spec.keys()
    return term.event_spec.term_spec.keys()


@pytest.fixture(params=list(_TERMS), ids=list(_TERMS))
def term(request):
    return _TERMS[request.param]()


class TestTheReaderErrors:
    """Each row of the reader's error table, on every kind."""

    @pytest.mark.parametrize(
        ("key", "error", "match"),
        [
            pytest.param("", ValueError, "the path is empty", id="empty-path"),
            pytest.param("g//c", ValueError, "has an empty name", id="empty-name"),
            pytest.param("/a", ValueError, "has an empty name", id="leading-separator"),
            pytest.param([], ValueError, "since it names no paths", id="empty-list"),
            pytest.param((), ValueError, "since it names no paths", id="empty-tuple"),
            pytest.param(("a", "a"), ValueError, "cannot select 'a' twice", id="repeated"),
            pytest.param(
                ("g", "g/c"),
                ValueError,
                "cannot select both 'g' and 'g/c', since 'g/c' is inside 'g'",
                id="nested",
            ),
            pytest.param({"a"}, TypeError, "got set", id="set"),
            pytest.param(iter(["a"]), TypeError, "got list_iterator", id="iterator"),
            pytest.param(
                ("a", 0),
                TypeError,
                r"cannot mix a path and a position in one \[\]",
                id="path-and-position",
            ),
        ],
    )
    def test_a_malformed_key_raises_before_any_lookup(self, term, key, error, match):
        with pytest.raises(error, match=match):
            term[key]

    def test_an_unknown_path_raises_key_error_listing_the_paths(self, term):
        with pytest.raises(
            KeyError,
            match=r"unknown path 'nope' in .*; available paths: \['a', 'g', 'g/c', 'g/e'\]",
        ):
            term["nope"]
        with pytest.raises(KeyError, match="unknown path 'nope' in"):
            term["a", "nope"]

    def test_a_tuple_that_spells_a_nested_path_says_how_to_write_it(self, term):
        with pytest.raises(
            KeyError,
            match=r"unknown path 'c' in .*\. To select the nested path, write 'g/c'",
        ):
            term["g", "c"]

    def test_a_record_is_named_by_its_label(self):
        site = Record({"habitat": {"forest": 0.7, "wetland": 0.3}, "count": 2.0}, label="site")
        with pytest.raises(
            KeyError,
            match=r"unknown path 'habitat/forst' in 'site'; available paths: \['habitat', "
            r"'habitat/forest', 'habitat/wetland', 'count'\]",
        ):
            site["habitat/forst"]

    def test_a_term_with_no_paths_raises_type_error(self):
        # A KeyError says a path is missing; a term over a single value has none to look up.
        x = Normal("x", 0.0, 1.0)
        with pytest.raises(TypeError, match="since its draws are single values with no paths"):
            x["y"]
        with pytest.raises(TypeError, match="since its elements are single values with no paths"):
            _sample(x, sample_shape=4)["y"]
        assert "y" not in x

    def test_a_path_below_a_field_of_a_record_raises_key_error(self):
        # The record has paths, and 'a/b' is not one of them, whatever the field 'a' holds.
        with pytest.raises(KeyError, match="unknown path 'a/b'"):
            _draw()["a/b"]
        with pytest.raises(KeyError, match="unknown path 'a/b'"):
            _grouped()["a/b"]


class TestBatchPositions:
    """A batch also takes positions, and refuses a list or an array of them."""

    def test_a_path_and_a_position_in_one_key_raise(self):
        with pytest.raises(TypeError, match=r"cannot mix a path and a position in one \[\]"):
            _draws()["a", 0]

    @pytest.mark.parametrize(
        "positions", [[0, 2], np.array([0, 2]), jnp.array([0, 2])], ids=["list", "numpy", "jax"]
    )
    def test_a_list_or_an_array_of_positions_raises(self, positions):
        with pytest.raises(TypeError):
            _draws()[positions]

    def test_a_list_of_positions_says_to_index_one(self):
        with pytest.raises(TypeError, match="cannot index a batch by a list of positions, got"):
            _draws()[[0, 2]]

    def test_a_path_inside_whole_elements_says_the_batch_holds_their_values(self):
        draws = _sample(Normal("mu", 0.0, 1.0), sample_shape=4)
        with pytest.raises(
            TypeError,
            match=r"since its elements are single values with no paths\. Read the values with "
            r"\.raw",
        ):
            draws["mu"]

    def test_an_empty_tuple_never_returns_the_whole_batch(self):
        with pytest.raises(ValueError, match="since it names no paths"):
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
        assert view.event_spec.term_spec == RecordSpec(
            s1=RecordSpec(theta=Normal("x", 0.0, 1.0).event_spec.term_spec),
            s2=RecordSpec(theta=Normal("x", 0.0, 1.0).event_spec.term_spec),
        )
        draw = _sample(view, seed=1)
        assert draw.keys() == ("s1/theta", "s2/theta")
        np.testing.assert_allclose(variance(view)["s2/theta"], 4.0, rtol=1e-6)

    def test_a_list_of_one_whole_term_name_keeps_todays_record(self):
        x = Normal("x", 0.0, 1.0)
        assert x[["x"]].event_spec.term_spec == RecordSpec(x=x.event_spec.term_spec)
        assert x["x"] is x


class TestLabels:
    """A term indexed by several paths is labeled by its paths, in the term's order."""

    def test_a_law_and_its_marginal_at_a_list_of_paths_read_by_their_paths(self):
        grouped = _grouped()
        view, detached = grouped["g/c", "a"], marginal(grouped, ["g/c", "a"])
        assert view.notation == detached.notation == "(Normal·Normal·Normal)(a, g/c)"
        assert _sample(view).label == "(a, g/c) ~ Normal·Normal·Normal"
        assert _sample(detached).label == "(a, g/c) ~ Normal·Normal·Normal"

    def test_one_path_still_reads_by_its_last_name(self):
        assert _grouped()["g/c"].notation == "(Normal·Normal·Normal)(c)"


class TestMarginal:
    """``marginal(d, paths)`` takes what ``d[paths]`` takes and gives the same law (I7)."""

    @pytest.mark.parametrize("paths", [["a", "g/c"], ("g/c", "a")], ids=["list", "tuple"])
    def test_a_list_of_paths_gives_the_law_of_the_view(self, paths):
        grouped = _grouped()
        detached, view = marginal(grouped, paths), grouped[paths]
        assert detached.event_spec == view.event_spec
        assert detached.event_spec.term_spec.keys() == ("a", "g/c")
        np.testing.assert_allclose(variance(detached)["g/c"], 4.0, rtol=1e-6)
        np.testing.assert_allclose(mean(detached)["g/c"], mean(view)["g/c"], rtol=1e-6)

    def test_paths_with_one_last_name_keep_their_groups(self):
        detached = marginal(_hier(), ["s2/theta", "s1/theta"])
        assert detached.event_spec.term_spec.keys() == ("s1/theta", "s2/theta")
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

    def test_a_path_of_the_elements_is_a_member_of_a_batch(self):
        assert "a" in _sample(_grouped(), sample_shape=4)
        assert "g" in _sample(_grouped())


class TestOnePathPerArgument:
    """``without``, ``select``, and ``at_path`` read one path per argument."""

    def test_without_refuses_a_repeated_or_nested_path_and_a_tuple(self):
        draw = _draw()
        with pytest.raises(ValueError, match=r"cannot call without\(\) with 'a' twice"):
            draw.without("a", "a")
        with pytest.raises(ValueError, match="'g/c' is inside 'g'"):
            draw.without("g", "g/c")
        with pytest.raises(TypeError, match="Give each path as a separate argument"):
            draw.without(("a", "g/e"))

    def test_at_path_takes_separate_segments_and_refuses_a_tuple(self):
        draw = _draw()
        assert draw.at_path("g", "c") == draw["g/c"]
        with pytest.raises(
            TypeError,
            match=r"at_path takes each segment as a separate argument, got a tuple. Write at_path\(\*segments\)",
        ):
            draw.at_path(("g", "c"))

    def test_select_keys_each_path_by_its_last_name(self):
        draw = _draw()
        assert list(draw.select("a", "g/c")) == ["a", "c"]
        assert list(draw.select(x="g/c")) == ["x"]
        with pytest.raises(
            ValueError,
            match="cannot select 's1/theta' and 's2/theta' together, since both would be named 'theta'",
        ):
            _sample(_hier()).select("s1/theta", "s2/theta")
        assert list(_draws().select("a", "g/c")) == ["a", "c"]


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
