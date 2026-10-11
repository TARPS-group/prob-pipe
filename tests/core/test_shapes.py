"""The reading of shape, name-list, and axis-count arguments in ``core/_shapes.py``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.core._shapes import (
    _as_axis_counts,
    _as_dim,
    _as_levels,
    _as_names,
    _as_shape,
)

#: Iterables the readers refuse, since they are used up, unordered, or not sequences of items.
_NOT_SEQUENCES = [
    pytest.param((d for d in (1, 2)), "generator", id="generator"),
    pytest.param(iter([1, 2]), "list_iterator", id="iterator"),
    pytest.param({1, 2}, "set", id="set"),
    pytest.param(frozenset({1}), "frozenset", id="frozenset"),
    pytest.param(b"ab", "bytes", id="bytes"),
    pytest.param(memoryview(b"ab"), "memoryview", id="memoryview"),
    pytest.param({"a": 1}, "dict", id="mapping"),
    pytest.param({"a": 1}.keys(), "dict_keys", id="keys-view"),
]


class TestShape:
    """A bare ``int`` or ``str`` is one dimension, and any other sequence is one per item."""

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
            (range(2, 4), (2, 3)),
            (np.int64(3), (3,)),
            (np.array(3), (3,)),
            (np.array([2, 3]), (2, 3)),
            (jnp.array([2, 3]), (2, 3)),
            (np.str_("n"), ("n",)),
            ((np.str_("n"), np.int32(2)), ("n", 2)),
        ],
    )
    def test_reads_each_form_as_a_tuple_of_python_ints_and_strs(self, arg, expected):
        shape = _as_shape(arg, what="shape")

        assert shape == expected
        assert all(type(d) in (int, str) for d in shape)

    def test_a_scalar_and_its_one_tuple_read_alike(self):
        assert _as_shape("loc", what="shape") == _as_shape(("loc",), what="shape")
        assert _as_shape(4, what="shape") == _as_shape((4,), what="shape")

    @pytest.mark.parametrize(("arg", "type_shown"), _NOT_SEQUENCES)
    def test_refuses_an_iterable_that_is_not_a_sequence(self, arg, type_shown):
        with pytest.raises(
            TypeError, match=f"shape must be an int, a str, or a sequence of them, got {type_shown}"
        ):
            _as_shape(arg, what="shape")

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            (True, "shape must be an int, a str, or a sequence of them, got bool True"),
            (np.bool_(True), "got bool"),
            (2.0, "got float 2.0"),
            (None, "got NoneType None"),
            ((True,), "shape entry must be a non-negative int or a dimension name, got bool True"),
            (((2,),), "shape entry must be a non-negative int or a dimension name, got tuple"),
        ],
    )
    def test_refuses_what_is_not_a_shape(self, arg, match):
        with pytest.raises(TypeError, match=match):
            _as_shape(arg, what="shape")

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            ("", "shape entry must be a Python identifier such as 'n_obs', got ''"),
            ("n obs", "got 'n obs'"),
            (("2d",), "got '2d'"),
            (-1, "shape entry must be non-negative, got -1"),
            ((3, -1), "got -1"),
        ],
    )
    def test_refuses_a_bad_dimension(self, arg, match):
        with pytest.raises(ValueError, match=match):
            _as_shape(arg, what="shape")

    def test_a_shape_of_sizes_refuses_a_name(self):
        assert _as_shape(4, what="sample_shape", symbolic=False) == (4,)
        with pytest.raises(
            TypeError, match="sample_shape must be an int or a sequence of ints, got str 'S'"
        ):
            _as_shape("S", what="sample_shape", symbolic=False)
        with pytest.raises(TypeError, match="sample_shape entry must be a non-negative int"):
            _as_shape(("S",), what="sample_shape", symbolic=False)
        with pytest.raises(TypeError, match="must be an int or a sequence of ints"):
            _as_shape(2.5, what="sample_shape", symbolic=False)

    def test_names_the_caller(self):
        with pytest.raises(TypeError, match=r"^NumericArraySpec shape entry must be"):
            _as_dim(1.5, what="NumericArraySpec shape entry")


class TestNames:
    """A bare ``str`` is one name."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("draw", ("draw",)),
            (("draw",), ("draw",)),
            (["chain", "draw"], ("chain", "draw")),
            (np.array(["chain", "draw"]), ("chain", "draw")),
        ],
    )
    def test_reads_each_form_as_a_tuple_of_python_strs(self, arg, expected):
        names = _as_names(arg, what="level_names")

        assert names == expected
        assert all(type(name) is str for name in names)

    def test_leaves_the_name_rule_to_the_caller(self):
        assert _as_names("", what="level_names") == ("",)

    def test_an_empty_sequence_is_no_names(self):
        assert _as_names([], what="metrics") == ()

    @pytest.mark.parametrize(("arg", "type_shown"), _NOT_SEQUENCES)
    def test_refuses_an_iterable_that_is_not_a_sequence(self, arg, type_shown):
        with pytest.raises(
            TypeError, match=f"level_names must be a str or a sequence of str, got {type_shown}"
        ):
            _as_names(arg, what="level_names")

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            (3, "level_names must be a str or a sequence of str, got int 3"),
            (("draw", 3), "level_names entry must be a str, got int 3"),
        ],
    )
    def test_refuses_what_is_not_names(self, arg, match):
        with pytest.raises(TypeError, match=match):
            _as_names(arg, what="level_names")


class TestAxisCounts:
    """A bare ``int`` is one count, and every count is at least 1."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [(2, (2,)), ((1, 2), (1, 2)), ([3], (3,)), (np.int64(2), (2,)), (range(1, 3), (1, 2))],
    )
    def test_reads_each_form_as_a_tuple(self, arg, expected):
        assert _as_axis_counts(arg, what="axes_per_level") == expected

    @pytest.mark.parametrize(("arg", "type_shown"), _NOT_SEQUENCES)
    def test_refuses_an_iterable_that_is_not_a_sequence(self, arg, type_shown):
        with pytest.raises(
            TypeError,
            match=f"axes_per_level must be an int or a sequence of ints, got {type_shown}",
        ):
            _as_axis_counts(arg, what="axes_per_level")

    @pytest.mark.parametrize(
        ("arg", "error", "match"),
        [
            ("2", TypeError, "axes_per_level must be an int or a sequence of ints, got str '2'"),
            (True, TypeError, "got bool True"),
            ((1, True), TypeError, "axes_per_level entry must be an int, got bool True"),
            (0, ValueError, "axes_per_level must be at least 1, got 0"),
            ((1, 0), ValueError, "axes_per_level entry must be at least 1, got 0"),
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
            (None, {"a": ()}, ValueError, r"B level 'a' must have at least one axis, got \(\)"),
            ([("a", 1)], {}, TypeError, "B levels must be a mapping from level name to shape"),
            ({1: 2}, {}, TypeError, "B level names must be str, got int 1"),
            (None, {"a": 2.0}, TypeError, "B level 'a' must be an int, a str"),
            (None, {"a": "n obs"}, ValueError, "B level 'a' entry must be a Python identifier"),
        ],
    )
    def test_refuses_malformed_levels(self, levels, keywords, error, match):
        with pytest.raises(error, match=match):
            _as_levels(levels, keywords, what="B")
