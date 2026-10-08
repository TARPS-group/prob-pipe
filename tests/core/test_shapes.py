"""The reading of shape, level-name, and axis-count arguments in ``core/_shapes.py``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.core._shapes import (
    _as_axis_counts,
    _as_dim,
    _as_level_names,
    _as_levels,
    _as_shape,
)


class TestShape:
    """A bare ``int`` or ``str`` is one dimension, and any other iterable is one per item."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (3, (3,)),
            ("loc", ("loc",)),
            ((3,), (3,)),
            (("loc",), ("loc",)),
            ((5, 4, "loc"), (5, 4, "loc")),
            ([2, "n"], (2, "n")),
            ((), ()),
            ([], ()),
            (np.int64(3), (3,)),
            (np.array(3), (3,)),
            (np.array([2, 3]), (2, 3)),
            (jnp.array([2, 3]), (2, 3)),
            ((d for d in (1, 2)), (1, 2)),
        ],
    )
    def test_reads_each_form_as_a_tuple(self, arg, expected):
        shape = _as_shape(arg, what="shape")

        assert shape == expected
        assert all(type(d) in (int, str) for d in shape)

    def test_a_scalar_and_its_one_tuple_read_alike(self):
        assert _as_shape("loc", what="shape") == _as_shape(("loc",), what="shape")
        assert _as_shape(4, what="shape") == _as_shape((4,), what="shape")

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            (True, "shape must be an int, a str, or an iterable of them, got bool True"),
            (2.0, "got float 2.0"),
            (None, "got NoneType None"),
            (b"ab", "got bytes b'ab'"),
            ({"a": 1}, "got dict"),
            ((True,), "shape entries must be non-negative ints or dimension names, got bool True"),
            (((2,),), "got tuple"),
        ],
    )
    def test_refuses_what_is_not_a_shape(self, arg, match):
        with pytest.raises(TypeError, match=match):
            _as_shape(arg, what="shape")

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            ("", "dimension names must be Python identifiers such as 'n_obs', got ''"),
            ("n obs", "got 'n obs'"),
            (("2d",), "got '2d'"),
            (-1, "shape entries must be non-negative, got -1"),
            ((3, -1), "got -1"),
        ],
    )
    def test_refuses_a_bad_dimension(self, arg, match):
        with pytest.raises(ValueError, match=match):
            _as_shape(arg, what="shape")

    def test_a_concrete_shape_refuses_a_name(self):
        assert _as_shape(4, what="sample_shape", symbolic=False) == (4,)
        with pytest.raises(TypeError, match="sample_shape entries must be ints, got str 'S'"):
            _as_shape("S", what="sample_shape", symbolic=False)
        with pytest.raises(TypeError, match="must be an int or an iterable of ints"):
            _as_shape(2.5, what="sample_shape", symbolic=False)

    def test_names_the_caller(self):
        with pytest.raises(TypeError, match=r"^NumericArraySpec shape entries"):
            _as_dim(1.5, what="NumericArraySpec shape")


class TestLevelNames:
    """A bare ``str`` is one name."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("draw", ("draw",)),
            (("draw",), ("draw",)),
            (["chain", "draw"], ("chain", "draw")),
            ((name for name in ("a", "b")), ("a", "b")),
        ],
    )
    def test_reads_each_form_as_a_tuple(self, arg, expected):
        assert _as_level_names(arg, what="level_names") == expected

    def test_leaves_the_name_rule_to_the_spec(self):
        assert _as_level_names("", what="level_names") == ("",)

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            (3, "level_names must be a str or an iterable of str, got int 3"),
            (b"ab", "got bytes"),
            ({"draw": 1}, "got dict"),
            (("draw", 3), "level_names entries must be str, got int 3"),
        ],
    )
    def test_refuses_what_is_not_names(self, arg, match):
        with pytest.raises(TypeError, match=match):
            _as_level_names(arg, what="level_names")


class TestAxisCounts:
    """A bare ``int`` is one count, and every count is at least 1."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [(2, (2,)), ((1, 2), (1, 2)), ([3], (3,)), (np.int64(2), (2,))],
    )
    def test_reads_each_form_as_a_tuple(self, arg, expected):
        assert _as_axis_counts(arg, what="axes_per_level") == expected

    @pytest.mark.parametrize(
        ("arg", "error", "match"),
        [
            ("2", TypeError, "axes_per_level must be an int or an iterable of ints, got str '2'"),
            (True, TypeError, "got bool True"),
            ((1, True), TypeError, "axes_per_level entries must be ints, got bool True"),
            (0, ValueError, "axes_per_level entries must be at least 1, got 0"),
        ],
    )
    def test_refuses_a_bad_count(self, arg, error, match):
        with pytest.raises(error, match=match):
            _as_axis_counts(arg, what="axes_per_level")


class TestLevels:
    """Levels map each name to its shape, given as a mapping or as keywords."""

    def test_reads_each_value_as_a_shape(self):
        names, groups = _as_levels(None, {"a": (1, 2), "b": "loc", "c": (5, 4, "loc")}, what="B")

        assert names == ("a", "b", "c")
        assert groups == ((1, 2), ("loc",), (5, 4, "loc"))

    def test_the_mapping_form_takes_any_name(self):
        assert _as_levels({"my level": 4}, {}, what="B") == (("my level",), ((4,),))

    def test_refuses_both_forms(self):
        with pytest.raises(TypeError, match="B takes its levels as a mapping or as keywords"):
            _as_levels({"a": 1}, {"b": 2}, what="B")

    @pytest.mark.parametrize(
        ("levels", "keywords", "error", "match"),
        [
            (None, {}, ValueError, "B must have at least one level"),
            ({}, {}, ValueError, "B must have at least one level"),
            (None, {"a": ()}, ValueError, "B level 'a' must have at least one axis, got ()"),
            ([("a", 1)], {}, TypeError, "B levels must be a mapping from level name to shape"),
            ({1: 2}, {}, TypeError, "B level names must be str, got int 1"),
            (None, {"a": 2.0}, TypeError, "B level 'a' must be an int, a str"),
            (None, {"a": "n obs"}, ValueError, "B level 'a' dimension names must be"),
        ],
    )
    def test_refuses_malformed_levels(self, levels, keywords, error, match):
        with pytest.raises(error, match=match):
            _as_levels(levels, keywords, what="B")
