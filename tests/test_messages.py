"""Tests for the shared message wording in ``probpipe._messages``."""

from __future__ import annotations

import pytest

from probpipe._messages import count, unknown_names


class TestCount:
    @pytest.mark.parametrize(("n", "expected"), [(0, "0 levels"), (1, "1 level"), (2, "2 levels")])
    def test_regular_plural(self, n, expected):
        assert count(n, "level") == expected

    def test_irregular_plural(self):
        assert count(1, "axis", "axes") == "1 axis"
        assert count(3, "axis", "axes") == "3 axes"


class TestUnknownNames:
    def test_one_unknown_name(self):
        message = unknown_names("level", ["test"], ["quantile"])
        assert message == "unknown level 'test'; available levels: ['quantile']"

    def test_several_unknown_names(self):
        message = unknown_names("field", ["a", "b"], ["x", "y"])
        assert message == "unknown fields ['a', 'b']; available fields: ['x', 'y']"

    def test_nothing_available(self):
        message = unknown_names("given slot", ["y"], [])
        assert message == "unknown given slot 'y'; there are no given slots"

    def test_irregular_plural(self):
        message = unknown_names("batch axis", ["k"], ["n"], plural="batch axes")
        assert message == "unknown batch axis 'k'; available batch axes: ['n']"
