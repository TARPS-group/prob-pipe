"""Contracts of cross-type conversion: the converter, its report, and its registry.

A converter is a binary dispatch method that declares the source types it reads
and the targets it produces, where a target is a distribution class or a
capability protocol. The converter registry

1. keys on the source's type and the target itself;
2. admits a converter when a target it declares is the requested one or a
   refinement of it;
3. selects among the admitted converters by exactness, then priority, then
   specificity, then registration order.

It tests the source first, returning a law that already satisfies the target
as it is. Its ``check`` reports the selected converter's promise under the
registered name and exactness, and a conversion carries the source's event
declaration and satisfies the target.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, Protocol, runtime_checkable

import pytest

from probpipe import NumericArraySpec, OutputSpec, RecordSpec
from probpipe.core._dispatch import (
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    Feasibility,
    MethodInfo,
    ResolutionError,
)
from probpipe.distributions import Distribution, DistributionSpec, NumericDistribution
from probpipe.distributions._capabilities import (
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from probpipe.distributions._conversion import (
    ConversionInfo,
    Converter,
    ConverterRegistry,
    converter_registry,
)

_SCALAR = NumericArraySpec(())

# ---------------------------------------------------------------------------
# Toy laws: the sources and targets of the conversions below
# ---------------------------------------------------------------------------


class Source(Distribution):
    """A law the converters below read, over a scalar unless given a declaration."""

    def __init__(self, label: str = "x", event_spec: OutputSpec | None = None):
        super().__init__(
            label, OutputSpec(**{label: _SCALAR}) if event_spec is None else event_spec
        )


class SourceSub(Source):
    """A more specific source."""


class Stranger(Distribution):
    """A law that no converter below reads."""

    def __init__(self, label: str = "stranger"):
        super().__init__(label, OutputSpec(**{label: _SCALAR}))


class Target(Distribution):
    """A representation converted to, built from a name and a declaration."""


class TargetSub(Target):
    """A refinement of ``Target``."""


class Elsewhere(Distribution):
    """A representation unrelated to ``Target``."""


class _Scores:
    """The two density methods, which satisfy ``SupportsLogProb`` without inheriting it."""

    def _log_prob(self, value: Any) -> Any:
        return 0.0

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return 0.0


class Scored(_Scores, Distribution):
    """A representation with a density."""


class ScoredSource(_Scores, Source):
    """A source with a density."""


class GuardedSource(_Scores, Source):
    """A source whose density carries a guard that returns what the test sets."""

    def __init__(self, guard: bool | None, label: str = "x"):
        super().__init__(label)
        object.__setattr__(self, "_guard", guard)

    def _log_prob_guard(self) -> bool | None:
        """The test's setting."""
        return self._guard


class GuardedScored(_Scores, Distribution):
    """A representation whose density carries a guard that holds by its label."""

    def _log_prob_guard(self) -> bool:
        """The label is not "rejects"."""
        return self.label != "rejects"


class Sampled(Distribution):
    """A law that samples without inheriting ``SupportsSampling``."""

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return 0.0


@runtime_checkable
class Labeled(Protocol):
    """A protocol with a data member, which ``issubclass`` cannot check."""

    label: str


# ---------------------------------------------------------------------------
# A configurable converter
# ---------------------------------------------------------------------------


class ToyConverter(Converter):
    """A converter whose declarations, report, and result each test sets.

    When feasible, ``check`` promises the source's declaration with
    ``target_class`` and ``capabilities``, and ``report`` replaces that promise.
    ``execute`` returns ``result``, or else a ``target_class`` law carrying the
    source's declaration. Each call's arguments are recorded.
    """

    def __init__(
        self,
        name: str,
        *,
        sources: tuple[type, ...] = (Source,),
        targets: tuple[type, ...] = (Target,),
        exact: bool = True,
        priority: int | None = 10,
        feasible: bool | None = True,
        pending: tuple[str, ...] = (),
        description: str = "",
        target_class: type = Target,
        capabilities: tuple[type, ...] = (),
        report: Feasibility | None = None,
        result: Any = None,
        raises: BaseException | None = None,
        supported: Any = None,
    ):
        self._name = name
        self._sources = sources
        self._targets = targets
        self._exact = exact
        self._priority = priority
        self._feasible = feasible
        self._pending = pending
        self._description = description
        self._target_class = target_class
        self._capabilities = capabilities
        self._report = report
        self._result = result
        self._raises = raises
        self._supported = supported
        self.check_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.execute_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int | None:
        return self._priority

    def supported_types(self) -> Any:
        return (self._sources, self._targets) if self._supported is None else self._supported

    def check(self, source: Any, target_type: type, **kwargs: Any) -> Feasibility:
        self.check_calls.append(((source, target_type), kwargs))
        if self._report is not None:
            return self._report
        return ConversionInfo(
            feasible=self._feasible,
            description=self._description,
            pending=self._pending,
            method_name=self._name,
            exact=self._exact,
            target_spec=source.spec if self._feasible else None,
            target_class=self._target_class,
            capabilities=self._capabilities,
        )

    def execute(self, source: Any, target_type: type, **kwargs: Any) -> Any:
        self.execute_calls.append(((source, target_type), kwargs))
        if self._raises is not None:
            raise self._raises
        if self._result is not None:
            return self._result
        return self._target_class(source.label, source.event_spec)


def _registry(*converters: Converter) -> ConverterRegistry:
    """A fresh registry holding *converters*, registered in order."""
    registry = ConverterRegistry()
    for converter in converters:
        registry.register(converter)
    return registry


# ---------------------------------------------------------------------------
# ConversionInfo: a method report that carries the converter's promise
# ---------------------------------------------------------------------------


class TestConversionInfo:
    def test_is_a_method_info(self):
        assert issubclass(ConversionInfo, MethodInfo)
        assert issubclass(ConversionInfo, Feasibility)

    def test_adds_the_promise_and_whether_the_conversion_samples(self):
        inherited = {field.name for field in fields(MethodInfo)}
        added = [field.name for field in fields(ConversionInfo) if field.name not in inherited]
        assert added == ["target_spec", "target_class", "capabilities", "samples"]

    def test_samples_must_be_a_bool(self):
        with pytest.raises(TypeError, match="samples"):
            ConversionInfo(feasible=False, samples=1)

    def test_a_report_selecting_no_converter_is_an_exact_one(self):
        """A source that already satisfies the target needs no converter, and is exact."""
        info = ConversionInfo(feasible=True, exact=True, target_spec=Source().spec)
        assert info.method_name is None and info.exact is True
        with pytest.raises(ValueError, match="must set method_name and exact"):
            ConversionInfo(feasible=True, exact=False, target_spec=Source().spec)

    def test_promises_nothing_by_default(self):
        info = ConversionInfo(feasible=False)
        assert info.target_spec is None
        assert info.target_class is None
        assert info.capabilities == ()

    def test_keeps_the_promise_and_the_method_it_is_given(self):
        spec = Source().spec
        info = ConversionInfo(
            feasible=True,
            method_name="m",
            exact=False,
            target_spec=spec,
            target_class=Target,
            capabilities=(SupportsLogProb,),
        )
        assert info.target_spec == spec
        assert info.target_class is Target
        assert info.capabilities == (SupportsLogProb,)
        assert (info.method_name, info.exact) == ("m", False)

    def test_is_frozen(self):
        info = ConversionInfo(feasible=False)
        with pytest.raises(FrozenInstanceError):
            info.target_class = Target  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [0, 1, "", "yes"], ids=repr)
    def test_feasible_must_be_a_bool_or_none(self, bad: Any):
        with pytest.raises(TypeError, match="bool or None"):
            ConversionInfo(feasible=bad)

    def test_carries_pending_requirements_exactly_when_unresolved(self):
        info = ConversionInfo(
            feasible=None, pending=("the event declaration",), method_name="m", exact=True
        )
        assert info.unresolved
        with pytest.raises(ValueError, match="pending"):
            ConversionInfo(feasible=None, method_name="m", exact=True)
        with pytest.raises(ValueError, match="only an unresolved"):
            ConversionInfo(feasible=False, pending=("the event declaration",))

    def test_sets_the_method_name_and_exactness_together(self):
        for partial in ({"method_name": "m"}, {"exact": True}):
            with pytest.raises(ValueError, match="together"):
                ConversionInfo(feasible=False, **partial)

    def test_a_feasible_or_unresolved_report_names_its_converter(self):
        """A converter fills in its own name and exactness; an infeasible report may omit both."""
        ConversionInfo(feasible=False)
        with pytest.raises(ValueError, match="must set method_name and exact"):
            ConversionInfo(feasible=True, target_spec=Source().spec)
        with pytest.raises(ValueError, match="must set method_name and exact"):
            ConversionInfo(feasible=None, pending=("the event declaration",))

    def test_a_feasible_report_promises_its_target_spec(self):
        """Planning reads the promised declaration, so a feasible conversion states it."""
        with pytest.raises(ValueError):
            ConversionInfo(feasible=True, method_name="m", exact=True, target_class=Target)


# ---------------------------------------------------------------------------
# Converter: abstract until it provides everything a registry reads
# ---------------------------------------------------------------------------

#: One implementation of each member a converter must provide.
_CONVERTER_MEMBERS: dict[str, Any] = {
    "name": property(lambda self: "complete"),
    "exact": property(lambda self: True),
    "supported_types": lambda self: ((Source,), (Target,)),
    "check": lambda self, source, target_type: ConversionInfo(
        feasible=True,
        method_name="complete",
        exact=True,
        target_spec=source.spec,
        target_class=Target,
    ),
    "execute": lambda self, source, target_type: Target(source.label, source.event_spec),
}


class TestConverter:
    def test_is_a_binary_dispatch_method(self):
        assert issubclass(Converter, BinaryDispatchMethod)

    def test_its_abstract_members_are_name_exact_supported_types_check_and_execute(self):
        assert Converter.__abstractmethods__ == frozenset(_CONVERTER_MEMBERS)

    @pytest.mark.parametrize("missing", sorted(_CONVERTER_MEMBERS))
    def test_is_abstract_while_a_member_is_missing(self, missing: str):
        members = {name: member for name, member in _CONVERTER_MEMBERS.items() if name != missing}
        partial = type("Partial", (Converter,), members)
        assert partial.__abstractmethods__ == frozenset({missing})
        with pytest.raises(TypeError, match=missing):
            partial()

    def test_is_concrete_once_every_member_is_provided(self):
        converter = type("Complete", (Converter,), dict(_CONVERTER_MEMBERS))()
        assert (converter.name, converter.exact) == ("complete", True)
        assert converter.supported_types() == ((Source,), (Target,))

    def test_is_opt_in_only_by_default(self):
        """Without a priority, auto-selection skips the converter and naming it runs it."""
        converter = type("Complete", (Converter,), dict(_CONVERTER_MEMBERS))()
        assert converter.priority is None
        registry = _registry(converter)
        with pytest.raises(ResolutionError, match="no method is registered"):
            registry.convert(Source(), Target)
        assert isinstance(registry.convert(Source(), Target, method="complete"), Target)


# ---------------------------------------------------------------------------
# Registration validates a converter's declarations before the registry changes
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.parametrize(
        "supported",
        [(Source, Target), ((Source,), ("Target",)), ((Source,),), ((Source,), Target)],
        ids=["bare-classes", "a-name-for-a-class", "one-side", "a-bare-target"],
    )
    def test_supported_types_must_be_a_pair_of_class_tuples(self, supported: Any):
        registry = ConverterRegistry()
        with pytest.raises(TypeError, match="supported_types"):
            registry.register(ToyConverter("m", supported=supported))
        assert registry.list_methods() == []

    @pytest.mark.parametrize("side", ["sources", "targets"])
    def test_the_numeric_marker_is_refused_on_either_side(self, side: str):
        """Membership follows an instance's declaration, so selection by class would miss laws."""
        registry = ConverterRegistry()
        with pytest.raises(TypeError, match="cannot dispatch on NumericDistribution"):
            registry.register(ToyConverter("m", **{side: (NumericDistribution,)}))
        assert registry.list_methods() == []

    def test_a_repeated_name_is_refused(self):
        registry = _registry(ToyConverter("m"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ToyConverter("m", targets=(Elsewhere,)))
        assert registry.list_methods() == ["m"]

    def test_a_source_protocol_with_a_data_member_is_refused(self):
        """``issubclass`` raises for such a protocol, so admitting it makes every lookup raise."""
        registry = _registry(ToyConverter("ok"))
        with pytest.raises(TypeError):
            registry.register(ToyConverter("labeled", sources=(Labeled,)))
        assert registry.list_methods() == ["ok"]
        assert registry.check(Source(), Target).method_name == "ok"


# ---------------------------------------------------------------------------
# The key: the source's type and the target itself
# ---------------------------------------------------------------------------


class TestKeying:
    def test_is_a_binary_dispatch_registry(self):
        assert issubclass(ConverterRegistry, BinaryDispatchRegistry)

    def test_the_global_instance_is_a_converter_registry(self):
        assert isinstance(converter_registry, ConverterRegistry)

    def test_a_class_target_enters_the_key_as_itself(self):
        """If the key held a target's type, every class target would share its metaclass."""
        registry = _registry(
            ToyConverter("to_target"),
            ToyConverter("to_elsewhere", targets=(Elsewhere,), target_class=Elsewhere),
        )
        source = Source()
        for _ in range(2):  # the second pass is served from the lookup cache
            assert registry.check(source, Target).method_name == "to_target"
            assert registry.check(source, Elsewhere).method_name == "to_elsewhere"
        assert type(registry.convert(source, Elsewhere)) is Elsewhere

    def test_the_source_enters_the_key_by_its_type(self):
        registry = _registry(ToyConverter("m"))
        assert registry.check(SourceSub(), Target).method_name == "m"
        with pytest.raises(ResolutionError, match=r"\(Stranger, Target\)"):
            registry.convert(Stranger(), Target)

    def test_a_declared_source_protocol_admits_a_source_that_implements_it(self):
        """A source implementing the protocol without inheriting it matches least specifically."""
        registry = _registry(
            ToyConverter("from_densities", sources=(SupportsLogProb,), priority=1),
            ToyConverter("from_sources", sources=(Source,), priority=1),
        )
        assert registry.check(ScoredSource(), Target).method_name == "from_sources"
        registry.get_method("from_sources")._feasible = False
        assert registry.check(ScoredSource(), Target).method_name == "from_densities"
        with pytest.raises(ResolutionError):
            registry.convert(Stranger(), Target)

    @pytest.mark.parametrize(
        "target",
        ["Target", Target("t", OutputSpec(t=_SCALAR)), None],
        ids=["a-class-name", "an-instance", "None"],
    )
    def test_a_target_that_is_not_a_class_raises_type_error(self, target: Any):
        converter = ToyConverter("m")
        registry = _registry(converter)
        for call in (registry.check, registry.execute, registry.convert):
            for controls in ({}, {"method": "m"}):
                with pytest.raises(TypeError, match="class or a protocol"):
                    call(Source(), target, **controls)
        assert converter.check_calls == []

    def test_fewer_than_two_positional_arguments_raise_type_error(self):
        """Naming a converter bypasses the type pre-filter, not the arity."""
        converter = ToyConverter("m")
        registry = _registry(converter)
        for arguments in ((), (Source(),)):
            for controls in ({}, {"method": "m"}):
                with pytest.raises(TypeError, match="a source and a target"):
                    registry.check(*arguments, **controls)
                with pytest.raises(TypeError, match="a source and a target"):
                    registry.execute(*arguments, **controls)
        assert converter.check_calls == []


# ---------------------------------------------------------------------------
# The objects the function engine brings into ProbPipe
# ---------------------------------------------------------------------------


class TestDistributionTypes:
    def test_a_law_is_a_distribution_type(self):
        assert ConverterRegistry().is_distribution_type(Source())

    def test_an_object_a_converter_reads_is_one(self):
        assert not ConverterRegistry().is_distribution_type(3)
        assert _registry(ToyConverter("m", sources=(int,))).is_distribution_type(3)


# ---------------------------------------------------------------------------
# Admission: a declared target is the requested one or refines it
# ---------------------------------------------------------------------------


class TestTargetAdmission:
    def test_a_declared_subclass_admits_a_request_for_its_base(self):
        """A converter producing a more specific representation serves the request."""
        registry = _registry(ToyConverter("to_sub", targets=(TargetSub,), target_class=TargetSub))
        assert registry.check(Source(), Target).method_name == "to_sub"
        assert type(registry.convert(Source(), Target)) is TargetSub

    def test_a_declared_base_does_not_admit_a_request_for_its_subclass(self):
        registry = _registry(ToyConverter("to_target"))
        info = registry.check(Source(), TargetSub)
        assert info.feasible is False and info.method_name is None
        with pytest.raises(ResolutionError, match=r"\(Source, TargetSub\)"):
            registry.convert(Source(), TargetSub)

    def test_an_unrelated_declared_target_does_not_admit(self):
        registry = _registry(
            ToyConverter("to_elsewhere", targets=(Elsewhere,), target_class=Elsewhere)
        )
        with pytest.raises(ResolutionError, match="no method is registered"):
            registry.convert(Source(), Target)

    def test_a_declared_protocol_admits_a_request_for_it(self):
        registry = _registry(
            ToyConverter(
                "to_density",
                targets=(SupportsLogProb,),
                target_class=Scored,
                capabilities=(SupportsLogProb,),
            )
        )
        assert registry.check(Source(), SupportsLogProb).method_name == "to_density"
        assert isinstance(registry.convert(Source(), SupportsLogProb), SupportsLogProb)

    def test_a_refined_protocol_admits_a_request_for_the_protocol_it_refines(self):
        """A converter promising a normalized density serves a request for an unnormalized one."""
        registry = _registry(
            ToyConverter(
                "to_density",
                targets=(SupportsLogProb,),
                target_class=Scored,
                capabilities=(SupportsLogProb,),
            )
        )
        info = registry.check(Source(), SupportsUnnormalizedLogProb)
        assert info.feasible is True and info.method_name == "to_density"
        result = registry.convert(Source(), SupportsUnnormalizedLogProb)
        assert isinstance(result, SupportsUnnormalizedLogProb)

    def test_a_protocol_does_not_admit_a_request_for_its_refinement(self):
        registry = _registry(
            ToyConverter(
                "to_unnormalized",
                targets=(SupportsUnnormalizedLogProb,),
                target_class=Scored,
                capabilities=(SupportsUnnormalizedLogProb,),
            )
        )
        with pytest.raises(ResolutionError, match=r"\(Source, SupportsLogProb\)"):
            registry.convert(Source(), SupportsLogProb)

    def test_a_declared_class_that_implements_the_protocol_admits_a_request_for_it(self):
        """``Scored`` implements the protocol's methods without inheriting it."""
        registry = _registry(
            ToyConverter(
                "to_scored",
                targets=(Scored,),
                target_class=Scored,
                capabilities=(SupportsLogProb,),
            )
        )
        assert registry.check(Source(), SupportsLogProb).method_name == "to_scored"

    def test_a_protocol_target_needs_the_capability_among_the_promised_ones(self):
        """Declaring the protocol admits the converter, and the promise must guarantee it.

        The promised class implements the protocol, but class membership does not
        establish a capability that holds for some instances only.
        """
        registry = _registry(
            ToyConverter("unpromised", targets=(SupportsLogProb,), target_class=Scored)
        )
        assert registry.check(Source(), SupportsLogProb).feasible is False
        with pytest.raises(ResolutionError):
            registry.convert(Source(), SupportsLogProb)

    def test_a_guarded_promise_is_unresolved_until_the_converted_law_exists(self):
        """Only the converted law can decide its guard, which execution then checks."""
        registry = _registry(
            ToyConverter(
                "to_guarded",
                targets=(SupportsLogProb,),
                target_class=GuardedScored,
                capabilities=(SupportsLogProb,),
            )
        )
        info = registry.check(Source(), SupportsLogProb)
        assert info.unresolved and info.method_name == "to_guarded"
        assert info.pending == ("GuardedScored._log_prob_guard of the converted law",)
        assert type(registry.convert(Source("holds"), SupportsLogProb)) is GuardedScored
        with pytest.raises(ResolutionError, match="cannot provide SupportsLogProb: log_prob"):
            registry.convert(Source("rejects"), SupportsLogProb)

    def test_a_sampling_target_admits_the_converter_declaring_it(self):
        registry = _registry(
            ToyConverter(
                "to_sampler",
                targets=(SupportsSampling,),
                target_class=Sampled,
                capabilities=(SupportsSampling,),
            )
        )
        assert registry.check(Source(), SupportsSampling).method_name == "to_sampler"

    def test_a_sampling_source_type_admits_a_law_that_samples(self):
        registry = _registry(ToyConverter("from_samplers", sources=(SupportsSampling,)))
        assert (
            registry.check(Sampled("x", OutputSpec(x=_SCALAR)), Target).method_name
            == "from_samplers"
        )


# ---------------------------------------------------------------------------
# Selection: exactness, then priority, then specificity, then registration order
# ---------------------------------------------------------------------------


class TestSelectionOrder:
    def test_an_exact_converter_precedes_any_approximate_one(self):
        registry = _registry(
            ToyConverter("moment_match", exact=False, priority=1000),
            ToyConverter("closed_form", exact=True, priority=1),
        )
        assert registry.list_methods() == ["closed_form", "moment_match"]
        assert registry.check(Source(), Target).method_name == "closed_form"

    def test_a_higher_priority_precedes_within_exactness(self):
        registry = _registry(ToyConverter("low", priority=1), ToyConverter("high", priority=2))
        assert registry.check(Source(), Target).method_name == "high"

    def test_priority_precedes_specificity(self):
        registry = _registry(
            ToyConverter("specific", sources=(SourceSub,), priority=1),
            ToyConverter("general", sources=(Distribution,), priority=2),
        )
        assert registry.check(SourceSub(), Target).method_name == "general"

    def test_the_closer_source_type_precedes_at_equal_rank(self):
        registry = _registry(
            ToyConverter("from_any_law", sources=(Distribution,), priority=1),
            ToyConverter("from_sub", sources=(SourceSub,), priority=1),
        )
        assert registry.check(SourceSub(), Target).method_name == "from_sub"
        assert registry.check(Source(), Target).method_name == "from_any_law"

    def test_the_declared_target_closest_to_the_request_precedes_at_equal_rank(self):
        """Declaring the requested class is closer than declaring a refinement of it."""
        registry = _registry(
            ToyConverter("to_sub", targets=(TargetSub,), target_class=TargetSub, priority=1),
            ToyConverter("to_target", priority=1),
        )
        assert registry.check(Source(), Target).method_name == "to_target"
        assert registry.check(Source(), TargetSub).method_name == "to_sub"

    def test_a_declared_protocol_precedes_a_class_implementing_it_at_equal_rank(self):
        """A class implementing the protocol without inheriting it is the least specific match."""
        promise = {"target_class": Scored, "capabilities": (SupportsLogProb,), "priority": 1}
        registry = _registry(
            ToyConverter("to_scored", targets=(Scored,), **promise),
            ToyConverter("to_density", targets=(SupportsLogProb,), **promise),
        )
        assert registry.check(Source(), SupportsLogProb).method_name == "to_density"

    def test_specificity_sums_the_source_and_target_distances(self):
        """For ``(SourceSub, Target)``: C at 0 + 0, then A at 1 + 0 and B at 0 + 1, then D."""
        declared = {
            "D": ((Distribution,), (TargetSub,)),  # 2 + 1
            "A": ((Source,), (Target,)),  # 1 + 0
            "B": ((SourceSub,), (TargetSub,)),  # 0 + 1, tied with A and registered after it
            "C": ((SourceSub,), (Target,)),  # 0 + 0
        }
        registry = _registry(
            *(
                ToyConverter(name, sources=sources, targets=targets, priority=1)
                for name, (sources, targets) in declared.items()
            )
        )
        for expected in ("C", "A", "B", "D"):
            assert registry.check(SourceSub(), Target).method_name == expected
            registry.get_method(expected)._feasible = False

    def test_equal_rank_and_specificity_keep_registration_order(self):
        for order in (("first", "second"), ("second", "first")):
            registry = _registry(*(ToyConverter(name, priority=1) for name in order))
            assert registry.check(Source(), Target).method_name == order[0]

    def test_the_first_feasible_candidate_is_selected(self):
        registry = _registry(
            ToyConverter("by_draws", priority=2, feasible=False, description="needs draws"),
            ToyConverter("ready", priority=1),
        )
        assert registry.check(Source(), Target).method_name == "ready"
        assert isinstance(registry.convert(Source(), Target), Target)

    def test_an_unresolved_candidate_above_a_feasible_one_is_reported_and_blocks_conversion(self):
        registry = _registry(
            ToyConverter("awaiting", priority=2, feasible=None, pending=("the event declaration",)),
            ToyConverter("ready", priority=1),
        )
        info = registry.check(Source(), Target)
        assert info.unresolved and info.method_name == "awaiting"
        assert info.pending == ("the event declaration",)
        with pytest.raises(ResolutionError, match="the event declaration"):
            registry.convert(Source(), Target)

    def test_an_opt_in_only_converter_is_skipped_by_auto_selection(self):
        registry = _registry(
            ToyConverter("opt_in", priority=None),
            ToyConverter("ranked", exact=False, priority=1),
        )
        assert registry.check(Source(), Target).method_name == "ranked"


# ---------------------------------------------------------------------------
# exact_only excludes approximate converters by their declaration
# ---------------------------------------------------------------------------


class TestExactOnly:
    def test_excludes_approximate_converters_without_consulting_them(self):
        approximate = ToyConverter("moment_match", exact=False)
        registry = _registry(approximate)
        info = registry.check(Source(), Target, exact_only=True)
        assert info.feasible is False and "exact_only" in info.description
        with pytest.raises(ResolutionError, match="exact_only"):
            registry.convert(Source(), Target, exact_only=True)
        assert approximate.check_calls == [] and approximate.execute_calls == []

    def test_leaves_exact_converters_selectable(self):
        registry = _registry(
            ToyConverter(
                "closed_form", priority=1, feasible=False, description="needs a conjugate pair"
            ),
            ToyConverter("moment_match", exact=False, priority=1),
        )
        assert registry.check(Source(), Target).method_name == "moment_match"
        info = registry.check(Source(), Target, exact_only=True)
        assert info.feasible is False
        assert "closed_form: needs a conjugate pair" in info.description
        registry.get_method("closed_form")._feasible = True
        assert registry.check(Source(), Target, exact_only=True).method_name == "closed_form"

    def test_refuses_a_named_approximate_converter(self):
        approximate = ToyConverter("moment_match", exact=False)
        registry = _registry(approximate)
        info = registry.check(Source(), Target, method="moment_match", exact_only=True)
        assert info.feasible is False
        assert (info.method_name, info.exact) == ("moment_match", False)
        with pytest.raises(ResolutionError, match="approximate"):
            registry.convert(Source(), Target, method="moment_match", exact_only=True)
        assert approximate.check_calls == []

    def test_defaults_to_no_restriction(self):
        registry = _registry(ToyConverter("moment_match", exact=False))
        assert registry.check(Source(), Target).exact is False
        assert isinstance(registry.convert(Source(), Target), Target)

    def test_the_registry_passes_neither_control_to_the_converter(self):
        converter = ToyConverter("closed_form")
        registry = _registry(converter)
        registry.check(Source(), Target, exact_only=True)
        registry.check(Source(), Target, method="closed_form", exact_only=True)
        registry.convert(Source(), Target, method="closed_form", exact_only=True)
        assert [kwargs for _, kwargs in converter.check_calls] == [{}, {}, {}]
        assert [kwargs for _, kwargs in converter.execute_calls] == [{}]


# ---------------------------------------------------------------------------
# method= names the converter to run
# ---------------------------------------------------------------------------


class TestNamedConverter:
    def test_runs_the_named_converter_over_a_higher_ranked_one(self):
        registry = _registry(
            ToyConverter("preferred", priority=100),
            ToyConverter("chosen", priority=1, target_class=TargetSub),
        )
        assert registry.check(Source(), Target, method="chosen").method_name == "chosen"
        assert type(registry.convert(Source(), Target, method="chosen")) is TargetSub

    def test_bypasses_the_type_pre_filter(self):
        """A converter declared for other types runs when named, and its own check decides."""
        converter = ToyConverter("elsewhere", sources=(Stranger,), targets=(Elsewhere,))
        registry = _registry(converter)
        source = Source()
        assert registry.check(source, Target, method="elsewhere").feasible is True
        assert isinstance(registry.convert(source, Target, method="elsewhere"), Target)
        assert converter.execute_calls == [((source, Target), {})]

    def test_an_unregistered_name_raises_resolution_error(self):
        registry = _registry(ToyConverter("m"))
        for call in (registry.check, registry.convert):
            with pytest.raises(
                ResolutionError, match=r"unknown method 'nope'; available methods: \['m'\]"
            ):
                call(Source(), Target, method="nope")

    def test_a_named_infeasible_converter_raises_resolution_error(self):
        registry = _registry(ToyConverter("m", feasible=False, description="needs finite support"))
        info = registry.check(Source(), Target, method="m")
        assert info.feasible is False and info.method_name == "m"
        with pytest.raises(ResolutionError, match="not applicable: needs finite support"):
            registry.convert(Source(), Target, method="m")

    def test_a_named_unresolved_converter_raises_resolution_error(self):
        registry = _registry(ToyConverter("m", feasible=None, pending=("the event declaration",)))
        with pytest.raises(ResolutionError, match="pending: the event declaration"):
            registry.convert(Source(), Target, method="m")


# ---------------------------------------------------------------------------
# convert runs the selected converter
# ---------------------------------------------------------------------------


class TestConvert:
    def test_returns_the_selected_converters_result(self):
        source = Source()
        result = Target("x", source.event_spec)
        converter = ToyConverter("m", result=result)
        assert _registry(converter).convert(source, Target) is result
        assert converter.check_calls == [((source, Target), {})]
        assert converter.execute_calls == [((source, Target), {})]

    def test_passes_the_requested_target_to_the_converter(self):
        """The converter receives the protocol requested, not the one it declared."""
        source = Source()
        converter = ToyConverter(
            "to_density",
            targets=(SupportsLogProb,),
            target_class=Scored,
            capabilities=(SupportsLogProb,),
        )
        _registry(converter).convert(source, SupportsUnnormalizedLogProb)
        assert converter.check_calls == [((source, SupportsUnnormalizedLogProb), {})]
        assert converter.execute_calls == [((source, SupportsUnnormalizedLogProb), {})]

    def test_check_never_executes(self):
        converter = ToyConverter("m")
        registry = _registry(converter)
        registry.check(Source(), Target)
        registry.check(Source(), Target, method="m")
        assert len(converter.check_calls) == 2
        assert converter.execute_calls == []

    def test_no_registered_converter_raises_resolution_error_naming_the_key(self):
        with pytest.raises(
            ResolutionError, match=r"no method is registered for \(Source, SupportsLogProb\)"
        ):
            ConverterRegistry().convert(Source(), SupportsLogProb)

    def test_no_feasible_converter_raises_resolution_error_naming_each_one_tried(self):
        registry = _registry(
            ToyConverter("by_draws", priority=2, feasible=False, description="needs draws"),
            ToyConverter("by_density", priority=1, feasible=False, description="needs a density"),
        )
        with pytest.raises(
            ResolutionError, match="by_draws: needs draws; by_density: needs a density"
        ):
            registry.convert(Source(), Target)

    def test_a_converter_failure_propagates_and_stops_the_search(self):
        failing = ToyConverter("failing", priority=2, raises=RuntimeError("the fit diverged"))
        fallback = ToyConverter("fallback", priority=1)
        with pytest.raises(RuntimeError, match="the fit diverged"):
            _registry(failing, fallback).convert(Source(), Target)
        assert fallback.check_calls == [] and fallback.execute_calls == []

    def test_a_source_already_of_the_target_class_is_returned_as_it_is(self):
        """The registry tests the source first, so no converter runs for it."""
        source = Target("x", OutputSpec(x=_SCALAR))
        assert ConverterRegistry().convert(source, Target) is source
        copier = ToyConverter("copy", sources=(Target,))
        registry = _registry(copier)
        assert registry.convert(source, Target, method="copy") is source
        info = registry.check(source, Target)
        assert info.feasible is True
        assert (info.method_name, info.exact, info.samples) == (None, True, False)
        assert info.target_spec == source.spec and info.target_class is Target
        assert copier.check_calls == [] and copier.execute_calls == []

    def test_a_source_claiming_the_target_protocol_is_returned_as_it_is(self):
        source = ScoredSource()
        info = ConverterRegistry().check(source, SupportsLogProb)
        assert info.feasible is True and info.method_name is None
        assert SupportsLogProb in info.capabilities
        assert ConverterRegistry().convert(source, SupportsLogProb) is source

    def test_a_source_whose_guard_rejects_is_converted(self):
        source = GuardedSource(guard=False)
        registry = _registry(
            ToyConverter(
                "to_density",
                targets=(SupportsLogProb,),
                target_class=Scored,
                capabilities=(SupportsLogProb,),
            )
        )
        assert registry.check(source, SupportsLogProb).method_name == "to_density"
        assert type(registry.convert(source, SupportsLogProb)) is Scored

    def test_a_source_whose_guard_awaits_values_is_unresolved(self):
        """Whether the source needs a conversion is not yet known, so no converter is selected."""
        source = GuardedSource(guard=None)
        registry = _registry(
            ToyConverter(
                "to_density",
                targets=(SupportsLogProb,),
                target_class=Scored,
                capabilities=(SupportsLogProb,),
            )
        )
        info = registry.check(source, SupportsLogProb)
        assert info.unresolved and info.method_name is None
        assert "log_prob() is available" in info.pending[0]
        with pytest.raises(ResolutionError, match="cannot yet tell whether"):
            registry.convert(source, SupportsLogProb)

    def test_the_converted_law_records_the_source_and_the_converter(self):
        source = Source("theta")
        result = _registry(ToyConverter("m", exact=False)).convert(source, Target)
        assert result.provenance.operation == "convert"
        assert [parent.label for parent in result.provenance.parents] == ["theta"]
        assert result.provenance.metadata == {"converter": "m", "exact": False}

    def test_a_result_that_is_not_a_law_raises_type_error(self):
        registry = _registry(ToyConverter("m", result=object()))
        with pytest.raises(TypeError, match="returns a Distribution"):
            registry.convert(Source(), Target)

    def test_a_result_that_is_not_of_the_target_class_raises_type_error(self):
        registry = _registry(ToyConverter("m", result=Elsewhere("x", OutputSpec(x=_SCALAR))))
        with pytest.raises(TypeError, match="not a Target"):
            registry.convert(Source(), Target)

    def test_check_and_execute_pass_call_keywords_to_the_converter(self):
        converter = ToyConverter("m")
        registry = _registry(converter)
        registry.check(Source(), Target, num_draws=64)
        registry.execute(Source(), Target, num_draws=64)
        assert [kwargs for _, kwargs in converter.check_calls] == [{"num_draws": 64}] * 2
        assert [kwargs for _, kwargs in converter.execute_calls] == [{"num_draws": 64}]

    def test_convert_passes_converter_options_to_the_converter(self):
        converter = ToyConverter("m")
        _registry(converter).convert(Source(), Target, num_draws=64)
        assert [kwargs for _, kwargs in converter.check_calls] == [{"num_draws": 64}]
        assert [kwargs for _, kwargs in converter.execute_calls] == [{"num_draws": 64}]

    def test_convert_takes_its_controls_by_keyword_only(self):
        registry = _registry(ToyConverter("m"))
        with pytest.raises(TypeError):
            registry.convert(Source(), Target, "m")


# ---------------------------------------------------------------------------
# The registry's check reports the promise under the registration
# ---------------------------------------------------------------------------


class TestCheckReport:
    @pytest.mark.parametrize("named", [False, True], ids=["auto-selected", "named"])
    @pytest.mark.parametrize("declared_exact", [True, False], ids=["exact", "approximate"])
    def test_keeps_the_promise_under_the_registered_name_and_exactness(
        self, named: bool, declared_exact: bool
    ):
        """A report naming another converter or exactness is corrected by the registration."""
        source = Source()
        promise = ConversionInfo(
            feasible=True,
            method_name="impostor",
            exact=not declared_exact,
            target_spec=source.spec,
            target_class=Scored,
            capabilities=(SupportsLogProb,),
        )
        registry = _registry(
            ToyConverter(
                "to_density", exact=declared_exact, targets=(SupportsLogProb,), report=promise
            )
        )
        controls = {"method": "to_density"} if named else {}
        info = registry.check(source, SupportsUnnormalizedLogProb, **controls)
        assert isinstance(info, ConversionInfo) and info.feasible is True
        assert (info.method_name, info.exact) == ("to_density", declared_exact)
        assert info.target_spec == source.spec
        assert info.target_class is Scored
        assert info.capabilities == (SupportsLogProb,)

    def test_keeps_an_unresolved_promise_and_its_pending_requirements(self):
        report = ConversionInfo(
            feasible=None,
            pending=("the event declaration",),
            method_name="m",
            exact=True,
            target_class=Target,
        )
        info = _registry(ToyConverter("m", report=report)).check(Source(), Target)
        assert isinstance(info, ConversionInfo) and info.unresolved
        assert info.pending == ("the event declaration",)
        assert info.target_spec is None and info.target_class is Target

    def test_names_an_infeasible_report_by_the_registration(self):
        report = ConversionInfo(feasible=False, description="needs finite support")
        registry = _registry(ToyConverter("m", exact=False, report=report))
        info = registry.check(Source(), Target, method="m")
        assert isinstance(info, ConversionInfo)
        assert (info.method_name, info.exact) == ("m", False)
        assert info.description == "needs finite support"

    def test_a_report_claiming_exactness_does_not_pass_exact_only(self):
        """``exact_only`` reads the declaration, so an approximate converter stays excluded."""
        source = Source()
        promise = ConversionInfo(
            feasible=True,
            method_name="moment_match",
            exact=True,
            target_spec=source.spec,
            target_class=Target,
        )
        registry = _registry(ToyConverter("moment_match", exact=False, report=promise))
        assert registry.check(source, Target).exact is False
        with pytest.raises(ResolutionError, match="exact_only"):
            registry.convert(source, Target, exact_only=True)

    def test_every_report_is_a_conversion_info(self):
        registry = _registry(
            ToyConverter("moment_match", exact=False), ToyConverter("infeasible", feasible=False)
        )
        reports = {
            "none registered": ConverterRegistry().check(Source(), Target),
            "none feasible": registry.check(Source(), Target, exact_only=True),
            "named and excluded": registry.check(
                Source(), Target, method="moment_match", exact_only=True
            ),
        }
        kinds = {case: type(report).__name__ for case, report in reports.items()}
        assert set(kinds.values()) == {"ConversionInfo"}, kinds

    def test_a_converter_report_that_is_not_a_conversion_info_raises_type_error(self):
        registry = _registry(ToyConverter("m", report=Feasibility(feasible=True)))
        with pytest.raises(TypeError):
            registry.check(Source(), Target)
        with pytest.raises(TypeError):
            registry.convert(Source(), Target)


# ---------------------------------------------------------------------------
# A conversion carries the source's event declaration
# ---------------------------------------------------------------------------


class TestEventDeclarationPreserved:
    def test_the_converted_law_carries_the_source_declaration(self):
        source = Source("theta", OutputSpec(beta=NumericArraySpec((3,))))
        result = _registry(ToyConverter("m")).convert(source, Target)
        assert result.event_spec == source.event_spec
        assert result.spec == source.spec

    @pytest.mark.parametrize(
        "declaration",
        [
            OutputSpec(y=_SCALAR),
            OutputSpec(x=NumericArraySpec((2,))),
            OutputSpec(RecordSpec(x=_SCALAR)),
        ],
        ids=["renamed-component", "reshaped", "repackaged"],
    )
    def test_a_converted_law_with_another_declaration_is_refused(self, declaration: OutputSpec):
        registry = _registry(ToyConverter("m", result=Target("x", declaration)))
        with pytest.raises(ValueError):
            registry.convert(Source("x"), Target)

    def test_a_promise_of_another_declaration_is_refused(self):
        promise = ConversionInfo(
            feasible=True,
            method_name="m",
            exact=True,
            target_spec=DistributionSpec(OutputSpec(y=_SCALAR)),
            target_class=Target,
        )
        registry = _registry(ToyConverter("m", report=promise))
        with pytest.raises(ValueError):
            registry.check(Source("x"), Target)
