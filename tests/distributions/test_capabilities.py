"""The capability protocols, their conditional twins, guards, and per-instance classes.

Every unconditional capability has a conditional twin whose methods are named
``_conditional`` followed by the unconditional name, with ``given`` prepended to
the unconditional signature. ``_capability_guard`` returns the report of a
capability's guard for one call, and a feasible report when the capability
defines no guard. A guard returns a ``bool``, ``None``, or a ``Feasibility``, and
a misspelled guard raises when its class is created. ``_capability_subclass``
builds one cached subclass of a base per set of capabilities, which keeps the
base's name and pickles and copies by reference to the base and the set.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import pickle
import typing
from typing import Any

import pytest

import probpipe.distributions as distributions
from probpipe import Normal, NumericArraySpec, OutputSpec
from probpipe.core._dispatch import Feasibility
from probpipe.distributions import ConditionalDistribution, Distribution
from probpipe.distributions._capabilities import (
    _CONDITIONAL_TWINS,
    _NORMALIZING_CAPABILITIES,
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
    _conjunction,
    _is_normalized,
    _kernel_is_normalized,
)

#: The unconditional capabilities the design declares as protocols. The two
#: conditioning capabilities are claimed by inheriting, and a kernel shares their
#: method, so neither has a twin.
_UNCONDITIONAL = (
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsRandomLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsExpectation,
    SupportsMarginals,
)

_GIVEN = ("given", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty)


def _members(protocol: type) -> frozenset[str]:
    """The names of the members *protocol* declares, its data members included."""
    return frozenset(protocol.__protocol_attrs__)


def _methods(protocol: type) -> set[str]:
    """The names of the methods *protocol* declares, its data members left out."""
    return {name for name in _members(protocol) if callable(getattr(protocol, name, None))}


def _parameters(function: Any) -> list[tuple[str, Any, Any]]:
    """Each parameter of *function* as its name, kind, and default."""
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def _noop(self: Any, *args: Any, **kwargs: Any) -> None:
    return None


def _implementing(members: typing.Iterable[str]) -> Any:
    """An instance of a plain class that defines each of *members*."""
    return type("Implementation", (), dict.fromkeys(members, _noop))()


def _cases(pending: dict[type, str] | None = None) -> list:
    """One case per unconditional capability, pending where *pending* names a reason."""
    pending = pending or {}
    return [
        pytest.param(
            protocol,
            id=protocol.__name__,
            marks=(
                pytest.mark.pending(reason=pending[protocol], raises=AssertionError)
                if protocol in pending
                else ()
            ),
        )
        for protocol in _UNCONDITIONAL
    ]


# -- Test doubles -------------------------------------------------------------


class _Kernel(SupportsConditionalLogProb):
    """A kernel with a normalized conditional density that records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        self.calls.append((given, value))
        return -1.5


class _Term:
    """A term whose mean has no guard and whose marginal has a guard returning *report*.

    Each guard call is recorded in ``guard_calls``. The term also defines a
    quantile guard without the quantile, and a variance attribute that is not a
    method.
    """

    _variance = 2.0

    def __init__(self, report: Feasibility | None = None) -> None:
        self.report = Feasibility(True) if report is None else report
        self.guard_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _mean(self) -> float:
        return 0.0

    def _marginal(self, path: str) -> None:
        return None

    def _marginal_guard(self, *arguments: Any, **keywords: Any) -> Feasibility:
        self.guard_calls.append((arguments, keywords))
        return self.report

    def _quantile_guard(self, *arguments: Any, **keywords: Any) -> Feasibility:
        self.guard_calls.append((arguments, keywords))
        return Feasibility(True)


def _host_mean(self: Any) -> float:
    return 0.5


def _host_variance(self: Any) -> float:
    return 2.0


def _host_marginal(self: Any, path: str) -> Any:
    return self


def _host_marginal_guard(self: Any, path: str) -> Feasibility:
    return Feasibility(True) if path == "h" else Feasibility(False, f"no field {path!r}")


class _Host(Distribution):
    """A scalar law whose capabilities are chosen at construction from its table."""

    _capability_table: typing.ClassVar = {
        SupportsMean: {"_mean": _host_mean},
        SupportsVariance: {"_variance": _host_variance},
        SupportsMarginals: {"_marginal": _host_marginal, "_marginal_guard": _host_marginal_guard},
    }

    def __new__(cls, label: str, protocols: typing.Iterable[type] = ()) -> _Host:
        return object.__new__(_capability_subclass(_Host, protocols))

    def __init__(self, label: str, protocols: typing.Iterable[type] = ()) -> None:
        super().__init__(
            OutputSpec(**{label: NumericArraySpec(())}),
            label=label,
        )


class _OtherHost(_Host):
    """A second base with the same table."""


class _TruncatedLaw(Distribution, SupportsMarginals):
    """A law whose marginal guard returns the answer it holds, at every path."""

    def __init__(self, label: str, answer: Any = True) -> None:
        super().__init__(
            OutputSpec(**{label: NumericArraySpec(())}),
            label=label,
        )
        self.answer = answer

    def _marginal(self, path: str) -> Any:
        return self

    def _marginal_guard(self, path: str) -> Any:
        """Exact for a path outside the truncated block.

        A second paragraph, which the reports leave out.
        """
        return self.answer


class _BareGuardLaw(Distribution, SupportsMarginals):
    """A law whose marginal guard rejects and has no docstring."""

    def __init__(self, label: str) -> None:
        super().__init__(
            OutputSpec(**{label: NumericArraySpec(())}),
            label=label,
        )

    def _marginal(self, path: str) -> Any:
        return self

    def _marginal_guard(self, path: str) -> bool:
        return False


class _GuardedDensityLaw(Distribution, SupportsLogProb):
    """A law whose density guard rejects."""

    def __init__(self, label: str) -> None:
        super().__init__(
            OutputSpec(**{label: NumericArraySpec(())}),
            label=label,
        )

    def _log_prob(self, value: Any) -> float:
        return 0.0

    def _log_prob_guard(self) -> bool:
        """Scores only once fitted."""
        return False


class _OwnUnnormalizedLaw(_GuardedDensityLaw):
    """A law that computes its unnormalized density itself, with no guard on it."""

    def _unnormalized_log_prob(self, value: Any) -> float:
        return 0.0


class _GuardedDensityKernel(ConditionalDistribution, SupportsConditionalLogProb):
    """A kernel whose conditional density guard rejects."""

    def __init__(self, label: str) -> None:
        super().__init__(
            {"s": NumericArraySpec(())},
            OutputSpec(y=NumericArraySpec(())),
            label=label,
        )

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _conditional_log_prob(self, given: Any, value: Any) -> float:
        return 0.0

    def _conditional_log_prob_guard(self) -> bool:
        """Scores only once fitted."""
        return False


# -- Tests --------------------------------------------------------------------


class TestMarginals:
    def test_a_class_defining_marginal_supports_marginals(self):
        assert isinstance(_implementing({"_marginal"}), SupportsMarginals)

    def test_a_law_without_a_marginal_does_not_support_marginals(self):
        assert not isinstance(Normal("x", 0.0, 1.0), SupportsMarginals)

    def test_the_marginal_is_the_one_member_and_its_guard_is_not_one(self):
        assert _members(SupportsMarginals) == frozenset({"_marginal"})

    def test_the_marginal_takes_one_path(self):
        assert [name for name, _, _ in _parameters(SupportsMarginals._marginal)] == ["self", "path"]

    def test_the_package_exports_the_protocol(self):
        assert distributions.SupportsMarginals is SupportsMarginals


class TestUnconditionalProtocols:
    @pytest.mark.parametrize("protocol", _cases())
    def test_a_class_defining_the_capability_methods_claims_the_capability(self, protocol):
        assert isinstance(_implementing(_methods(protocol)), protocol)


class TestConditionalMirror:
    def test_every_unconditional_capability_has_a_twin(self):
        assert set(_CONDITIONAL_TWINS) == set(_UNCONDITIONAL)

    def test_the_twins_are_the_conditional_capabilities_the_package_exports(self):
        twins = list(_CONDITIONAL_TWINS.values())
        exported = {
            getattr(distributions, name)
            for name in distributions.__all__
            if name.startswith("SupportsConditional")
        }
        assert len(set(twins)) == len(twins)
        assert set(twins) == exported

    @pytest.mark.parametrize("protocol", _cases())
    def test_a_twin_is_named_by_the_rule(self, protocol):
        expected = "SupportsConditional" + protocol.__name__.removeprefix("Supports")
        assert _CONDITIONAL_TWINS[protocol].__name__ == expected

    @pytest.mark.parametrize("protocol", _cases())
    def test_a_twin_method_is_conditional_followed_by_the_unconditional_name(self, protocol):
        expected = {f"_conditional{method}" for method in _methods(protocol)}
        assert _methods(_CONDITIONAL_TWINS[protocol]) == expected

    @pytest.mark.parametrize("protocol", _cases())
    def test_a_twin_method_prepends_given_to_the_unconditional_signature(self, protocol):
        twin = _CONDITIONAL_TWINS[protocol]
        for method in sorted(_methods(protocol)):
            receiver, *rest = _parameters(getattr(protocol, method))
            conditional = _parameters(getattr(twin, f"_conditional{method}"))
            assert conditional == [receiver, _GIVEN, *rest], method

    @pytest.mark.parametrize("protocol", _cases())
    def test_the_given_is_a_record_or_a_mapping(self, protocol):
        twin = _CONDITIONAL_TWINS[protocol]
        for method in _methods(twin):
            given = inspect.signature(getattr(twin, method)).parameters["given"]
            assert given.annotation == "Record | Mapping[str, Any]", method

    @pytest.mark.parametrize("protocol", _cases())
    def test_a_twin_and_its_capability_match_disjoint_classes(self, protocol):
        twin = _CONDITIONAL_TWINS[protocol]
        kernel = _implementing(_members(twin))
        law = _implementing(_members(protocol))
        assert isinstance(kernel, twin)
        assert not isinstance(kernel, protocol)
        assert isinstance(law, protocol)
        assert not isinstance(law, twin)

    def test_no_twin_method_shares_a_name_with_an_unconditional_method(self):
        unconditional = set().union(*map(_methods, _CONDITIONAL_TWINS))
        conditional = set().union(*map(_methods, _CONDITIONAL_TWINS.values()))
        assert not unconditional & conditional

    def test_refinement_between_capabilities_is_mirrored(self):
        for refined, base in itertools.permutations(_CONDITIONAL_TWINS, 2):
            twin_refines = _CONDITIONAL_TWINS[base] in _CONDITIONAL_TWINS[refined].__mro__
            assert (base in refined.__mro__) is twin_refines, (refined.__name__, base.__name__)


class TestConditionalLogProb:
    def test_the_normalized_twin_refines_the_unnormalized_twin(self):
        assert SupportsConditionalUnnormalizedLogProb in SupportsConditionalLogProb.__mro__

    def test_the_unnormalized_density_defaults_to_the_normalized_one(self):
        kernel = _Kernel()
        given = {"beta": 0.5}
        assert kernel._conditional_unnormalized_log_prob(given, 2.0) == -1.5
        assert kernel.calls == [(given, 2.0)]

    def test_a_normalized_kernel_claims_both_densities(self):
        kernel = _Kernel()
        assert isinstance(kernel, SupportsConditionalLogProb)
        assert isinstance(kernel, SupportsConditionalUnnormalizedLogProb)

    def test_an_unnormalized_kernel_claims_only_the_unnormalized_density(self):
        kernel = _implementing({"_conditional_unnormalized_log_prob"})
        assert isinstance(kernel, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(kernel, SupportsConditionalLogProb)


#: The capabilities whose answer presupposes a probability law (III.8).
_NORMALIZING = (
    SupportsLogProb,
    SupportsSampling,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsExpectation,
)

#: The capabilities that fix no normalizing constant when claimed alone.
_NON_NORMALIZING = (
    SupportsUnnormalizedLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsRandomLogProb,
    SupportsMarginals,
)


def _named(protocols: tuple[type, ...]) -> list:
    return [pytest.param(protocol, id=protocol.__name__) for protocol in protocols]


def _refuse(self: Any, *args: Any, **kwargs: Any) -> None:
    raise AssertionError("the classification called a capability")


class TestNormalization:
    def test_the_normalizing_capabilities_are_the_density_sampling_and_integrals(self):
        assert set(_NORMALIZING_CAPABILITIES) == set(_NORMALIZING)

    @pytest.mark.parametrize("protocol", _named(_NORMALIZING))
    def test_a_law_claiming_a_normalizing_capability_is_normalized(self, protocol):
        assert _is_normalized(_implementing(_methods(protocol)))

    @pytest.mark.parametrize("protocol", _named(_NON_NORMALIZING))
    def test_a_law_claiming_only_another_capability_is_unnormalized(self, protocol):
        assert not _is_normalized(_implementing(_methods(protocol)))

    def test_a_law_claiming_no_capability_is_unnormalized(self):
        assert not _is_normalized(_Host("h"))

    def test_a_parametric_family_is_normalized(self):
        assert _is_normalized(Normal("x", 0.0, 1.0))

    def test_the_classification_calls_no_capability_and_reads_no_guard(self):
        law = type("Refusing", (), {"_sample": _refuse, "_sample_guard": _refuse})()
        assert _is_normalized(law)
        assert _is_normalized(_GuardedDensityLaw("g"))

    @pytest.mark.parametrize("protocol", _named(_NORMALIZING))
    def test_a_kernel_claiming_a_normalizing_twin_is_normalized(self, protocol):
        assert _kernel_is_normalized(_implementing(_methods(_CONDITIONAL_TWINS[protocol])))

    @pytest.mark.parametrize("protocol", _named(_NON_NORMALIZING))
    def test_a_kernel_claiming_only_another_twin_is_unnormalized(self, protocol):
        assert not _kernel_is_normalized(_implementing(_methods(_CONDITIONAL_TWINS[protocol])))

    def test_a_kernel_whose_density_guard_rejects_is_still_normalized(self):
        assert _kernel_is_normalized(_GuardedDensityKernel("k"))

    def test_a_kernel_is_read_by_its_twins_and_a_law_by_its_capabilities(self):
        assert not _kernel_is_normalized(_implementing(_methods(SupportsSampling)))
        assert not _is_normalized(_implementing(_methods(SupportsConditionalSampling)))


class TestCapabilityGuard:
    def test_a_capability_without_a_guard_is_feasible(self):
        assert _capability_guard(_Term(), "_mean") == Feasibility(True)

    def test_a_capability_without_a_guard_is_feasible_whatever_the_arguments(self):
        assert _capability_guard(_Term(), "_mean", "a/b", exact=True) == Feasibility(True)

    @pytest.mark.parametrize(
        "report",
        [
            pytest.param(Feasibility(False, "no closed form at a/b"), id="infeasible"),
            pytest.param(Feasibility(None, pending=("the size of n",)), id="unresolved"),
            pytest.param(Feasibility(True, "exact at a/b"), id="feasible"),
        ],
    )
    def test_a_guarded_capability_returns_the_report_of_its_guard(self, report):
        assert _capability_guard(_Term(report), "_marginal", "a/b") is report

    def test_the_guard_receives_the_arguments_of_the_call(self):
        term = _Term()
        _capability_guard(term, "_marginal", "a/b", exact=True)
        assert term.guard_calls == [(("a/b",), {"exact": True})]

    def test_a_missing_capability_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="_cov"):
            _capability_guard(_Term(), "_cov")

    def test_an_attribute_that_is_not_a_method_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="_variance"):
            _capability_guard(_Term(), "_variance")

    def test_a_guard_without_its_capability_raises_before_the_guard_runs(self):
        term = _Term()
        with pytest.raises(AttributeError, match="_quantile"):
            _capability_guard(term, "_quantile", 0.5)
        assert term.guard_calls == []

    @pytest.mark.parametrize("method", ["_sample", "_log_prob", "_mean"])
    def test_a_family_capability_without_a_guard_is_feasible(self, method):
        assert _capability_guard(Normal("x", 0.0, 1.0), method) == Feasibility(True)


class TestGuardReports:
    """A guard's ``bool`` or ``None`` becomes a report naming the call and quoting the condition."""

    def test_a_guard_returning_true_is_feasible(self):
        assert _capability_guard(_TruncatedLaw("t", True), "_marginal", "a") == Feasibility(True)

    def test_a_guard_returning_false_rejects_and_quotes_its_condition(self):
        report = _capability_guard(_TruncatedLaw("t", False), "_marginal", "block/x")
        assert report == Feasibility(
            False,
            "marginal('block/x') is not available for Distribution; it requires: "
            "exact for a path outside the truncated block",
        )

    def test_a_guard_returning_none_is_unresolved_and_quotes_its_condition(self):
        report = _capability_guard(_TruncatedLaw("t", None), "_marginal", "block/x")
        assert report == Feasibility(
            None,
            pending=(
                "whether marginal('block/x') is available for Distribution depends on values "
                "not yet known; it requires: exact for a path outside the truncated block",
            ),
        )

    def test_the_report_names_the_keyword_arguments(self):
        report = _capability_guard(_TruncatedLaw("t", False), "_marginal", path="a")
        assert report.description.startswith("marginal(path='a') is not available")

    def test_a_guard_without_a_docstring_is_named_without_a_condition(self):
        report = _capability_guard(_BareGuardLaw("b"), "_marginal", "a")
        assert report == Feasibility(False, "marginal('a') is not available for Distribution")

    def test_a_feasibility_is_returned_as_the_guard_gave_it(self):
        custom = Feasibility(False, "the block is truncated")
        assert _capability_guard(_TruncatedLaw("t", custom), "_marginal", "a") is custom

    @pytest.mark.parametrize("answer", [1, 0, "", "yes"])
    def test_a_guard_returning_another_value_raises(self, answer):
        with pytest.raises(TypeError, match="a guard returns a bool, None, or a Feasibility"):
            _capability_guard(_TruncatedLaw("t", answer), "_marginal", "a")

    def test_the_default_unnormalized_density_takes_the_density_guard(self):
        law = _GuardedDensityLaw("d")
        report = _capability_guard(law, "_unnormalized_log_prob")
        assert report == _capability_guard(law, "_log_prob")
        assert report.feasible is False

    def test_an_unnormalized_density_of_its_own_is_total_without_a_guard(self):
        law = _OwnUnnormalizedLaw("d")
        assert _capability_guard(law, "_unnormalized_log_prob") == Feasibility(True)

    def test_the_default_conditional_unnormalized_density_takes_the_density_guard(self):
        kernel = _GuardedDensityKernel("k")
        report = _capability_guard(kernel, "_conditional_unnormalized_log_prob")
        assert report == _capability_guard(kernel, "_conditional_log_prob")
        assert report.feasible is False


class TestGuardNames:
    """Each guard a class body defines must guard a method the class implements."""

    def test_a_misspelled_guard_raises_when_its_class_is_created(self):
        with pytest.raises(TypeError, match=r"did you mean _marginal_guard\?"):

            class _Misspelled(Distribution, SupportsMarginals):
                def _marginal(self, path: str) -> Any: ...

                def _marginals_guard(self, path: str) -> bool:
                    return True

    def test_a_class_refused_for_its_guard_leaves_the_class_checks_intact(self):
        # The refused class stays alive through the traceback, and a protocol
        # check walks the protocol's subclasses, which include it.
        with pytest.raises(TypeError) as refused:

            class _Refused(Distribution, SupportsMarginals):
                def _marginal(self, path: str) -> Any: ...

                def _marginals_guard(self, path: str) -> bool:
                    return True

        law = Normal("x", 0.0, 1.0)
        assert not isinstance(law, SupportsMarginals)
        assert isinstance(law, Distribution)
        assert refused.value is not None

    def test_a_guard_of_a_method_the_class_lacks_raises(self):
        with pytest.raises(TypeError, match="guards _mean, which _NoMean does not implement"):

            class _NoMean(Distribution):
                def _mean_guard(self) -> bool:
                    return True

    def test_a_kernel_guard_is_checked_when_its_class_is_created(self):
        with pytest.raises(TypeError, match=r"did you mean _conditional_sample_guard\?"):

            class _Kernel(ConditionalDistribution):
                def _condition_on(self, given: Any, /, **kwargs: Any) -> Any: ...

                def _conditional_sample(self, given: Any, key: Any, sample_shape=()) -> Any: ...

                def _conditional_sampel_guard(self) -> bool:
                    return True

    def test_a_guard_that_is_not_a_method_raises(self):
        with pytest.raises(TypeError, match="must be a method that guards _mean"):

            class _DataGuard(Distribution, SupportsMean):
                _mean_guard = True

                def _mean(self) -> float:
                    return 0.0

    def test_a_guard_set_to_none_removes_the_inherited_guard(self):
        class _Total(_TruncatedLaw):
            _marginal_guard = None

        assert _capability_guard(_Total("t", False), "_marginal", "a") == Feasibility(True)

    def test_a_table_guard_is_checked_when_its_subclass_is_built(self):
        class _BadTable(Distribution):
            _capability_table: typing.ClassVar = {SupportsMean: {"_mean_guard": lambda self: True}}

        with pytest.raises(TypeError, match="guards _mean"):
            _capability_subclass(_BadTable, [SupportsMean])


class TestConjunction:
    """A call that needs several capabilities is feasible when each one is."""

    def test_no_report_is_feasible(self):
        assert _conjunction([]) == Feasibility(True)

    def test_the_first_rejection_is_the_report(self):
        first, second = Feasibility(False, "first"), Feasibility(False, "second")
        unresolved = Feasibility(None, pending=("the size of n",))
        assert _conjunction([Feasibility(True), unresolved, first, second]) is first

    def test_every_pending_entry_is_kept_when_none_rejects(self):
        reports = [
            Feasibility(None, pending=("the size of n",)),
            Feasibility(True),
            Feasibility(None, pending=("the size of m",)),
        ]
        assert _conjunction(reports) == Feasibility(
            None, pending=("the size of n", "the size of m")
        )


class TestCapabilitySubclass:
    def test_no_capability_is_the_base_itself(self):
        assert _capability_subclass(_Host, ()) is _Host
        assert type(_Host("h")) is _Host

    def test_the_subclass_claims_exactly_the_chosen_capabilities(self):
        host = _Host("h", [SupportsMean, SupportsMarginals])
        assert isinstance(host, SupportsMean)
        assert isinstance(host, SupportsMarginals)
        assert not isinstance(host, SupportsVariance)
        assert host._mean() == 0.5

    def test_the_table_installs_each_capability_with_its_guard(self):
        host = _Host("h", [SupportsMarginals])
        assert _capability_guard(host, "_marginal", "h") == Feasibility(True)
        assert _capability_guard(host, "_marginal", "g") == Feasibility(False, "no field 'g'")
        assert not hasattr(_Host("h", [SupportsMean]), "_marginal_guard")

    def test_the_subclass_derives_from_the_base_and_each_capability(self):
        subclass = _capability_subclass(_Host, [SupportsMean, SupportsVariance])
        assert issubclass(subclass, _Host)
        assert {SupportsMean, SupportsVariance} <= set(subclass.__mro__)
        assert SupportsMarginals not in subclass.__mro__

    def test_the_subclass_keeps_the_name_and_module_of_the_base(self):
        subclass = _capability_subclass(_Host, [SupportsMean])
        assert (subclass.__name__, subclass.__qualname__, subclass.__module__) == (
            _Host.__name__,
            _Host.__qualname__,
            _Host.__module__,
        )

    def test_one_subclass_is_cached_per_set(self):
        subclass = _capability_subclass(_Host, [SupportsMean, SupportsVariance])
        assert _capability_subclass(_Host, (SupportsVariance, SupportsMean)) is subclass
        assert (
            _capability_subclass(_Host, [SupportsMean, SupportsVariance, SupportsMean]) is subclass
        )
        assert _capability_subclass(_Host, iter([SupportsVariance, SupportsMean])) is subclass
        assert type(_Host("h", {SupportsVariance, SupportsMean})) is subclass

    def test_different_sets_get_different_subclasses(self):
        sets = ([SupportsMean], [SupportsVariance], [SupportsMean, SupportsVariance])
        assert len({_capability_subclass(_Host, protocols) for protocols in sets}) == len(sets)

    def test_different_bases_get_different_subclasses(self):
        other = _capability_subclass(_OtherHost, [SupportsMean])
        assert other is not _capability_subclass(_Host, [SupportsMean])
        assert issubclass(other, _OtherHost)

    @pytest.mark.parametrize(
        "protocols",
        [
            pytest.param([SupportsSampling], id="alone"),
            pytest.param([SupportsMean, SupportsCovariance], id="beside-a-listed-one"),
        ],
    )
    def test_a_capability_outside_the_table_raises_key_error(self, protocols):
        with pytest.raises(KeyError):
            _capability_subclass(_Host, protocols)

    # The term specs in a law's state pickle from protocol 2.
    @pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_restores_the_subclass_under_each_protocol_a_law_supports(self, protocol):
        host = _Host("h", [SupportsMean, SupportsMarginals]).with_label("renamed")
        restored = pickle.loads(pickle.dumps(host, protocol=protocol))
        assert type(restored) is type(host)
        assert (restored.label, restored.spec) == (host.label, host.spec)
        assert restored._mean() == 0.5

    def test_pickle_restores_the_base_as_itself(self):
        assert type(pickle.loads(pickle.dumps(_Host("h")))) is _Host

    @pytest.mark.parametrize("duplicate", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
    def test_a_copy_keeps_the_subclass(self, duplicate):
        host = _Host("h", [SupportsVariance])
        restored = duplicate(host)
        assert type(restored) is type(host)
        assert (restored.label, restored.spec) == (host.label, host.spec)

    def test_a_rename_keeps_the_subclass(self):
        host = _Host("h", [SupportsVariance])
        assert type(host.with_label("g")) is type(host)
