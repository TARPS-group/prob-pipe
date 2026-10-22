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
        ("source", "implemented_source", "finding"),
        [
            pytest.param(
                "def make(kernel=GaussianKernel): ...",
                "def make(kernel=1): ...",
                "make declares kernel=GaussianKernel, has 1",
                id="a-class-and-a-number",
            ),
            pytest.param(
                "def make(n=3): ...",
                "def make(n=4): ...",
                "make declares n=3, has 4",
                id="two-numbers",
            ),
        ],
    )
    def test_a_different_default_is_reported(self, source, implemented_source, finding):
        namespace: dict = {}
        exec(implemented_source, namespace)
        assert _check(source, namespace["make"]) == [finding]


class TestOverloads:
    _DECLARATION = """
class Spec:
    @overload
    def __init__(self, term_spec, /) -> None: ...
    @overload
    def __init__(self, **named_output) -> None: ...
"""

    def test_an_implementation_that_accepts_every_overload_matches(self):
        class Spec:
            def __init__(self, *args, **term): ...

        assert _check(self._DECLARATION, Spec) == []

    def test_an_implementation_that_refuses_an_overload_is_reported(self):
        class Spec:
            def __init__(self, term_spec, /): ...

        findings = _check(self._DECLARATION, Spec)
        assert len(findings) == 1
        assert findings[0].startswith(
            "Spec.__init__ does not accept the overload (self, named_output)"
        )


class TestBases:
    def test_a_class_with_an_abcmeta_metaclass_counts_as_deriving_from_abc(self):
        class Meta(abc.ABCMeta): ...

        class Term(metaclass=Meta): ...

        assert _check("class Term(ABC): ...", Term) == []

    def test_a_plain_class_does_not_derive_from_abc(self):
        class Term: ...

        assert _check("class Term(ABC): ...", Term) == ["Term does not derive from ABC"]
