"""Contract tests of convert: an unchanged source under fresh identity, or a registered converter."""

from __future__ import annotations

from typing import Any

import pytest

from probpipe import ApplicabilityError
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.distributions._capabilities import SupportsSampling
from probpipe.distributions._conversion import (
    ConversionInfo,
    Converter,
    ConverterRegistry,
    converter_registry,
)
from probpipe.distributions._distribution import Distribution, DistributionSpec
from probpipe.operations._convert import convert

from ._laws import REAL, Gaussian


class _Source(Distribution):
    """A law that the suite's converters read."""

    def __init__(self, name: str) -> None:
        super().__init__(name, REAL)


class _Target(Distribution):
    """A representation that an exact and an approximate converter produce."""

    def __init__(self, name: str, component: str | None = None) -> None:
        super().__init__(name, OutputSpec(**{component or name: REAL}))


class _RoughTarget(_Target):
    """A representation that only the approximate converter produces."""


class _Unfaithful(_Target):
    """A representation whose converter changes the event declaration."""


class _SuiteConverter(Converter):
    """A converter of this suite from ``_Source`` to its targets."""

    def __init__(self, name: str, exact: bool, targets: tuple[type, ...], keep: bool = True):
        self._name, self._exact, self._targets, self._keep = name, exact, targets, keep

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int:
        return 1

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((_Source,), self._targets)

    def check(self, source: Any, target_type: type, **options: Any) -> ConversionInfo:
        return ConversionInfo(
            True, method_name=self._name, exact=self._exact, target_spec=source.spec
        )

    def execute(self, source: Any, target_type: type, **options: Any) -> Distribution:
        component = next(iter(source.event_spec.components)) if self._keep else "other"
        law = target_type(self._name, component)
        return law


@pytest.fixture
def suite_converters(monkeypatch):
    """The converters route, delegating for one test to a registry of this suite's converters.

    The converters stay out of the global converter registry, which other
    suites inspect.
    """
    registry = ConverterRegistry()
    registry.register(_SuiteConverter("operations_suite_exact_converter", True, (_Target,)))
    registry.register(
        _SuiteConverter("operations_suite_approximate_converter", False, (_Target, _RoughTarget))
    )
    registry.register(
        _SuiteConverter("operations_suite_unfaithful_converter", True, (_Unfaithful,), keep=False)
    )
    (route,) = [route for route in convert.routes if route.name == "converters"]
    monkeypatch.setattr(route, "registry", registry)
    return registry


class TestConvert:
    def test_a_source_of_the_target_class_returns_under_fresh_identity(self):
        law = Gaussian("g")
        result = convert(law, Gaussian)
        assert isinstance(result, Gaussian)
        assert result is not law
        assert result.event_spec == law.event_spec
        assert result.provenance is not None
        report = convert.check(law, Gaussian)
        assert (report.route, report.exact) == ("identity", True)

    def test_a_source_claiming_the_target_protocol_needs_no_conversion(self):
        assert convert.check(Gaussian("g"), SupportsSampling).route == "identity"

    def test_the_converted_law_carries_the_source_declaration(self):
        assert convert.check(_Source("s"), _Target).result == OutputSpec(
            convert=DistributionSpec(_Source("s").event_spec)
        )

    def test_an_exact_converter_is_selected_before_an_approximate_one(self, suite_converters):
        report = convert.check(_Source("s"), _Target)
        assert (report.route, report.method, report.exact) == (
            "converters",
            "operations_suite_exact_converter",
            True,
        )
        assert isinstance(convert(_Source("s"), _Target), _Target)

    def test_method_selects_a_converter(self, suite_converters):
        view = convert.with_options(method="operations_suite_approximate_converter")
        assert view.check(_Source("s"), _Target).exact is False

    def test_exact_only_refuses_an_approximate_converter(self, suite_converters):
        with pytest.raises(ResolutionError):
            convert.with_options(exact_only=True)(_Source("s"), _RoughTarget)

    def test_a_converter_that_changes_the_declaration_raises_value_error(self, suite_converters):
        with pytest.raises(ValueError, match="component"):
            convert(_Source("s"), _Unfaithful)

    def test_the_route_delegates_to_the_converter_registry(self):
        (route,) = [route for route in convert.routes if route.name == "converters"]
        assert route.registry is converter_registry

    def test_the_identity_route_states_why_it_rejects(self):
        report = convert.check(Gaussian("g"), _Target)
        identity = {info.method_name: info for info in report.routes}["identity"]
        assert identity.feasible is False
        assert "is not a _Target" in identity.description
        assert "already" not in identity.description

    def test_no_applicable_converter_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="identity"):
            convert(Gaussian("g"), _Target)

    def test_the_target_is_a_class(self):
        with pytest.raises(ApplicabilityError, match="distribution class or a capability protocol"):
            convert(Gaussian("g"), "Gaussian")
