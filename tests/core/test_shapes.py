"""The reading of shape, name-list, axis-count, and path arguments in ``core/_shapes.py``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.core._shapes import (
    _as_axis_counts,
    _as_dim,
    _as_joined_path,
    _as_levels,
    _as_names,
    _as_path,
    _as_path_arguments,
    _as_paths,
    _as_positions,
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


class TestPaths:
    """A ``str`` is one path, and a list or a tuple of ``str`` is a list of paths."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("a", "a"),
            ("g/c", "g/c"),
            (np.str_("g/c"), "g/c"),
            (["a", "g/c"], ("a", "g/c")),
            (("g/c", "a"), ("g/c", "a")),
            (("a",), ("a",)),
            ([np.str_("a")], ("a",)),
        ],
        ids=["name", "path", "numpy-str", "list", "tuple", "tuple-of-one", "list-of-numpy-str"],
    )
    def test_reads_one_path_or_a_list_of_paths(self, arg, expected):
        read = _as_paths(arg, action="index R by")
        assert read == expected
        assert type(read) is type(expected)
        assert all(type(path) is str for path in (read if isinstance(read, tuple) else (read,)))

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            pytest.param("", "the path is empty", id="empty"),
            pytest.param("g//c", "the path 'g//c' has an empty name", id="empty-name"),
            pytest.param("/a", "the path '/a' has an empty name", id="leading-separator"),
            pytest.param("a/", "the path 'a/' has an empty name", id="trailing-separator"),
            pytest.param([], "since it names no paths", id="empty-list"),
            pytest.param((), "since it names no paths", id="empty-tuple"),
            pytest.param(("a", "a"), "cannot select 'a' twice", id="repeated"),
            pytest.param(
                ("g", "g/c"),
                "cannot select both 'g' and 'g/c', since 'g/c' is inside 'g'",
                id="nested",
            ),
            pytest.param(
                ("g/c", "g"),
                "cannot select both 'g' and 'g/c', since 'g/c' is inside 'g'",
                id="nested-reversed",
            ),
        ],
    )
    def test_refuses_malformed_paths_with_value_error(self, arg, match):
        with pytest.raises(ValueError, match=match):
            _as_paths(arg, action="index R by")

    @pytest.mark.parametrize(
        ("arg", "match"),
        [
            pytest.param({"a"}, "got set", id="set"),
            pytest.param(iter(["a"]), "got list_iterator", id="iterator"),
            pytest.param(None, "got NoneType", id="none"),
            pytest.param(True, "got bool", id="bool"),
            pytest.param(..., "got ellipsis", id="ellipsis"),
            pytest.param(0, "got int", id="position"),
            pytest.param(
                ("a", 0),
                r"since 0 is a position and only a batch takes positions\. Give only paths",
                id="list-holding-a-position",
            ),
        ],
    )
    def test_refuses_what_is_not_paths_with_type_error(self, arg, match):
        with pytest.raises(TypeError, match=match):
            _as_paths(arg, action="index R by")

    def test_names_the_positions_for_a_caller_that_takes_them(self):
        with pytest.raises(TypeError, match=r"cannot mix a path and a position in one \[\]"):
            _as_paths(("a", 0), action="index B by", positions=True)
        with pytest.raises(TypeError, match="cannot index a batch by a list of positions"):
            _as_paths([0, 2], action="index B by", positions=True)
        with pytest.raises(TypeError, match="a path, a list of paths, or a position"):
            _as_paths({"a"}, action="index B by", positions=True)


class TestPositions:
    """An integer, a slice, or a non-empty tuple of indexers is a position of a batch."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (3, (3,)),
            (slice(0, 2), (slice(0, 2),)),
            ((1, slice(None)), (1, slice(None))),
            (np.int64(2), (np.int64(2),)),
            (None, (None,)),
            (1.5, (1.5,)),
        ],
        ids=["int", "slice", "tuple", "numpy-int", "none", "float"],
    )
    def test_reads_anything_that_is_not_paths_as_one_indexer_per_axis(self, arg, expected):
        # The batch checks each indexer against its axis, so it refuses None and 1.5.
        assert _as_positions(arg) == expected

    @pytest.mark.parametrize(
        "arg",
        ["a", ["a"], [0, 2], (), ("a", 0), {"a"}, iter([0])],
        ids=["path", "list-of-paths", "list-of-positions", "empty", "mixed", "set", "iterator"],
    )
    def test_leaves_paths_and_refused_iterables_to_the_path_reader(self, arg):
        assert _as_positions(arg) is None


class TestPathArguments:
    """Paths given one per argument follow the rules of a list of paths, apart from being empty."""

    def test_reads_each_argument_as_one_path(self):
        assert _as_path_arguments(("a", "g/c"), action="call without() with") == ("a", "g/c")
        assert _as_path_arguments((), action="call without() with") == ()

    def test_refuses_a_tuple_and_says_how_to_pass_the_paths(self):
        with pytest.raises(TypeError, match="Give each path as a separate argument"):
            _as_path_arguments((("a", "g/e"),), action="call without() with")

    @pytest.mark.parametrize(
        ("arguments", "match"),
        [
            (("a", "a"), r"cannot call without\(\) with 'a' twice"),
            (("g", "g/c"), r"cannot call without\(\) with both 'g' and 'g/c', since 'g/c' is"),
            (("g//c",), "has an empty name"),
        ],
    )
    def test_refuses_what_a_list_of_paths_refuses(self, arguments, match):
        with pytest.raises(ValueError, match=match):
            _as_path_arguments(arguments, action="call without() with")

    def test_one_path_argument_refuses_a_tuple(self):
        assert _as_path("g/c", action="rename") == "g/c"
        with pytest.raises(TypeError, match="a path must be a str, got tuple"):
            _as_path(("g", "c"), action="rename")


class TestJoinedPath:
    """Separate segments, each a name or a path, spell one path joined with ``/``."""

    @pytest.mark.parametrize("segments", [("g", "c"), ("g/c",)])
    def test_joins_the_segments(self, segments):
        assert _as_joined_path(segments, function="at_path") == "g/c"

    def test_refuses_a_tuple_of_segments(self):
        with pytest.raises(
            TypeError,
            match=r"at_path takes each segment as a separate argument, got a tuple. Write at_path\(\*segments\)",
        ):
            _as_joined_path((("g", "c"),), function="at_path")

    @pytest.mark.parametrize(
        ("segments", "match"), [((), "no segments"), (("g", ""), "empty name")]
    )
    def test_refuses_no_segment_and_an_empty_name(self, segments, match):
        with pytest.raises(ValueError, match=match):
            _as_joined_path(segments, function="at_path")
