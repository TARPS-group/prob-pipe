"""The design-block check: how it reads a declared default and an abstract base."""

from __future__ import annotations

import abc
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "design"))

import design_blocks


class GaussianKernel:
    """A class a declaration names as a default."""


def _check(source: str, implemented: object) -> list[str]:
    declarations = design_blocks.parse_block("X.1", source)
    findings = design_blocks.check(declarations, lambda name: implemented)
    return [f"{finding.name} {finding.problem}" for finding in findings]


class TestDefaults:
    def test_a_class_default_written_by_its_name_matches(self):
        def make(kernel=GaussianKernel): ...

        assert _check("def make(kernel=GaussianKernel): ...", make) == []

    def test_a_function_default_written_by_its_name_matches(self):
        def make(reduce=sum): ...

        assert _check("def make(reduce=sum): ...", make) == []

    def test_a_string_default_does_not_match_a_bare_name(self):
        def make(kernel="GaussianKernel"): ...

        assert _check("def make(kernel=GaussianKernel): ...", make) == [
            "make declares kernel=GaussianKernel, has 'GaussianKernel'"
        ]

    def test_a_string_default_written_quoted_matches(self):
        def make(kernel="GaussianKernel"): ...

        assert _check("def make(kernel='GaussianKernel'): ...", make) == []

    @pytest.mark.parametrize(
        ("source", "default"),
        [
            pytest.param("def make(kernel=GaussianKernel): ...", 1, id="a-number"),
            pytest.param("def make(n=3): ...", 4, id="another-number"),
        ],
    )
    def test_a_different_default_is_reported(self, source, default):
        def make(kernel=default, n=default): ...

        findings = _check(source, make)
        assert any("declares" in finding for finding in findings), findings


class TestBases:
    def test_a_class_with_an_abcmeta_metaclass_counts_as_deriving_from_abc(self):
        class Meta(abc.ABCMeta): ...

        class Term(metaclass=Meta): ...

        assert _check("class Term(ABC): ...", Term) == []

    def test_a_plain_class_does_not_derive_from_abc(self):
        class Term: ...

        assert _check("class Term(ABC): ...", Term) == ["Term does not derive from ABC"]
