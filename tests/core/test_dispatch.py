"""Invariants of the dispatch registries.

Every method declares whether it is exact; selection is exact before
approximate, then priority, then type specificity, then registration order;
``set_priorities``
re-ranks without touching exactness; ``check`` probes without running;
``execute`` runs the first feasible method or raises ``ResolutionError``.
Each test is parametrized over the unary and binary registries.
"""

from __future__ import annotations

import abc
import builtins
import inspect
import itertools
import random
import re
import warnings
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from probpipe.core import _dispatch
from probpipe.core._dispatch import (
    BaseDispatchMethod,
    BaseDispatchRegistry,
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    Feasibility,
    MathematicalDomainError,
    MethodInfo,
    ResolutionError,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)

# ---------------------------------------------------------------------------
# Synthetic types and configurable fake methods
# ---------------------------------------------------------------------------


class Left:
    pass


class LeftSub(Left):
    pass


class Right:
    pass


class RightSub(Right):
    pass


class _FakeBase:
    """Shared body of the two fakes: configurable name, exactness, rank, and check.

    ``report`` replaces the ``Feasibility`` the check would build, so a test
    can return anything a method might, a ``MethodInfo`` included.
    """

    def __init__(
        self,
        name: str,
        *,
        exact: Any = True,
        priority: int | None = 10,
        feasible: bool | None = True,
        pending: tuple[str, ...] = (),
        description: str = "",
        result: Any = None,
        raises: BaseException | None = None,
        check_raises: BaseException | None = None,
        report: Feasibility | None = None,
        supported: Any = None,
    ):
        self._name = name
        self._exact = exact
        self._priority = priority
        self._feasible = feasible
        self._pending = pending
        self._description = description
        self._result = result
        self._raises = raises
        self._check_raises = check_raises
        self._report = report
        self._supported = supported
        self.check_calls = 0
        self.execute_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> Any:
        return self._exact

    @property
    def priority(self) -> int | None:
        return self._priority

    def check(self, *args: Any, **kwargs: Any) -> Feasibility:
        self.check_calls += 1
        if self._check_raises is not None:
            raise self._check_raises
        if self._report is not None:
            return self._report
        return Feasibility(
            feasible=self._feasible, description=self._description, pending=self._pending
        )

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.execute_calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result if self._result is not None else self._name


class FakeUnary(_FakeBase, UnaryDispatchMethod):
    def __init__(self, name: str, *, types: tuple[type, ...] = (object,), **kw: Any):
        super().__init__(name, **kw)
        self._types = types

    def supported_types(self) -> tuple[type, ...]:
        return self._types if self._supported is None else self._supported


class FakeBinary(_FakeBase, BinaryDispatchMethod):
    def __init__(
        self,
        name: str,
        *,
        left: tuple[type, ...] = (object,),
        right: tuple[type, ...] = (object,),
        **kw: Any,
    ):
        super().__init__(name, **kw)
        self._left = left
        self._right = right

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return (self._left, self._right) if self._supported is None else self._supported


class Arity:
    """One registry flavor: its class, its fake, the arguments of a call, and wrong shapes.

    ``bad_supported`` lists ``supported_types`` values of the other arity's
    shape or with non-class entries, which registration must reject.
    """

    def __init__(
        self,
        registry_cls: type,
        fake: type,
        args: tuple[Any, ...],
        fresh_args: tuple[Any, ...],
        other_args: tuple[Any, ...],
        general: Any,
        specific: Any,
        bad_supported: list[Any],
        exclude_args: Any,
    ):
        self.registry_cls = registry_cls
        self.fake = fake
        self.args = args
        self.fresh_args = fresh_args  # admitted by the defaults, of types no earlier call used
        self.other_args = other_args  # admitted by ``general`` only
        self.general = general  # a ``supported_types`` value admitting ``args`` at a distance
        self.specific = specific  # one admitting ``args`` exactly
        self.bad_supported = bad_supported
        self.exclude_args = exclude_args  # mutates a fake so its types no longer admit ``args``

    def registry(self) -> BaseDispatchRegistry[Any]:
        return self.registry_cls()

    def method(self, name: str, **kw: Any) -> Any:
        return self.fake(name, **kw)


@pytest.fixture(params=["unary", "binary"])
def arity(request: pytest.FixtureRequest) -> Arity:
    if request.param == "unary":
        return Arity(
            UnaryDispatchRegistry,
            FakeUnary,
            (Left(),),
            fresh_args=(LeftSub(),),
            other_args=(Right(),),
            general=(object,),
            specific=(Left,),
            bad_supported=[((Left,), (Right,)), ("Left",), Left, [Left]],
            exclude_args=lambda fake: setattr(fake, "_types", (Right,)),
        )
    return Arity(
        BinaryDispatchRegistry,
        FakeBinary,
        (Left(), Right()),
        fresh_args=(LeftSub(), RightSub()),
        other_args=(Right(), Right()),
        general=((object,), (object,)),
        specific=((Left,), (Right,)),
        bad_supported=[(Left, Right), ((Left,), ("Right",)), ((Left,),), (Left, (Right,))],
        exclude_args=lambda fake: setattr(fake, "_left", (Right,)),
    )


# ---------------------------------------------------------------------------
# A method declares whether it is exact, once, at registration
# ---------------------------------------------------------------------------


class TestExactnessDeclaration:
    def test_exact_is_abstract_on_the_base(self):
        assert "exact" in BaseDispatchMethod.__abstractmethods__

    def test_supported_types_is_abstract_on_the_base(self):
        assert "supported_types" in BaseDispatchMethod.__abstractmethods__

    @pytest.mark.parametrize("arity_base", [UnaryDispatchMethod, BinaryDispatchMethod])
    def test_arity_bases_add_no_abstract_members(self, arity_base: type):
        """The arity classes fix the shape of ``supported_types``; the base states the contract."""
        assert arity_base.__abstractmethods__ == BaseDispatchMethod.__abstractmethods__

    @pytest.mark.parametrize("bad", [None, 1, 0, "exact", "approximate"])
    def test_non_boolean_exact_is_rejected_at_register(self, arity: Arity, bad: Any):
        reg = arity.registry()
        with pytest.raises(TypeError, match="exact as a bool"):
            reg.register(arity.method("m", exact=bad))
        assert reg.list_methods() == []

    def test_set_priorities_has_no_exactness_input(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", exact=False))
        with pytest.raises(KeyError, match="No method named 'exact'"):
            reg.set_priorities(exact=True)
        assert reg.get_method("m").exact is False

    def test_default_priority_is_opt_in_only(self):
        class Bare(UnaryDispatchMethod):
            @property
            def name(self):
                return "bare"

            @property
            def exact(self):
                return True

            def supported_types(self):
                return (object,)

            def check(self, *a, **kw):
                return Feasibility(feasible=True)

            def execute(self, *a, **kw):
                return "ran"

        assert Bare().priority is None

    def test_a_feasibility_carries_no_exactness(self):
        """A method reports only what it alone knows; exactness is the registration's."""
        assert {f.name for f in fields(Feasibility)} == {"feasible", "description", "pending"}

    @pytest.mark.parametrize("declared_exact", [True, False], ids=["exact", "approximate"])
    def test_a_check_cannot_contradict_the_registration(self, arity: Arity, declared_exact: bool):
        """Even a check that returns a ``MethodInfo`` cannot override the declaration.

        Otherwise ``check`` could report an exactness that ``exact_only``,
        which reads the declaration, would not honor.
        """
        contradicting = MethodInfo(feasible=True, method_name="other", exact=not declared_exact)
        reg = arity.registry()
        reg.register(arity.method("m", exact=declared_exact, report=contradicting))
        for info in (reg.check(*arity.args), reg.check(*arity.args, method="m")):
            assert info.exact is declared_exact
            assert info.method_name == "m"
        if not declared_exact:
            with pytest.raises(ResolutionError):
                reg.execute(*arity.args, exact_only=True)


# ---------------------------------------------------------------------------
# Selection order: exact before approximate, then priority, then specificity,
# then registration
# ---------------------------------------------------------------------------


class TestSelectionOrder:
    def test_exact_precedes_approximate_for_any_ranks(self, arity: Arity):
        rng = random.Random(20260916)
        for _ in range(300):
            reg = arity.registry()
            names = [f"m{i}" for i in range(rng.randint(2, 8))]
            exactness = {n: rng.random() < 0.5 for n in names}
            for n in names:
                reg.register(arity.method(n, exact=exactness[n], priority=rng.randint(-50, 500)))
            order = reg.list_methods()
            first_approximate = next((i for i, n in enumerate(order) if not exactness[n]), None)
            last_exact = max((i for i, n in enumerate(order) if exactness[n]), default=-1)
            if first_approximate is not None:
                assert last_exact < first_approximate, order

    def test_a_feasible_exact_method_runs_over_any_approximate_one(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("approx", exact=False, priority=1000, result="approx"))
        reg.register(arity.method("exact", exact=True, priority=1, result="exact"))
        assert reg.execute(*arity.args) == "exact"

    def test_higher_rank_first_within_exactness(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("ten", priority=10, result="ten"))
        reg.register(arity.method("twenty", priority=20, result="twenty"))
        assert reg.list_methods() == ["twenty", "ten"]
        assert reg.execute(*arity.args) == "twenty"

    def test_ties_keep_registration_order(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("first", priority=5, result="first"))
        reg.register(arity.method("second", priority=5, result="second"))
        assert reg.execute(*arity.args) == "first"
        reg2 = arity.registry()
        reg2.register(arity.method("second", priority=5, result="second"))
        reg2.register(arity.method("first", priority=5, result="first"))
        assert reg2.execute(*arity.args) == "second"

    def test_first_feasible_wins_not_first_ranked(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("infeasible", priority=100, feasible=False))
        reg.register(arity.method("feasible", priority=50, result="ok"))
        assert reg.execute(*arity.args) == "ok"

    def test_opt_in_only_is_excluded_whether_exact_or_not(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("exact_opt_in", exact=True, priority=None, result="skipped"))
        reg.register(arity.method("approx", exact=False, priority=1, result="ran"))
        assert reg.execute(*arity.args) == "ran"
        assert reg.execute(*arity.args, method="exact_opt_in") == "skipped"

    def test_only_opt_in_methods_means_no_auto_method(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("opt_in", priority=None))
        with pytest.raises(ResolutionError, match="No method registered"):
            reg.execute(*arity.args)
        assert reg.execute(*arity.args, method="opt_in") == "opt_in"

    def test_zero_is_an_ordinary_rank(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("zero", priority=0, result="zero"))
        assert reg.execute(*arity.args) == "zero"

    def test_list_methods_is_selection_order_including_opt_in(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a_approx", exact=False, priority=100))
        reg.register(arity.method("e_low", exact=True, priority=1))
        reg.register(arity.method("e_high", exact=True, priority=2))
        reg.register(arity.method("e_opt", exact=True, priority=None))
        # The opt-in-only exact method keeps its position although automatic
        # selection never reaches it: the listing ranks, it does not say what runs.
        assert reg.list_methods() == ["e_high", "e_low", "e_opt", "a_approx"]


class TestSpecificity:
    """Among methods of equal rank, the closest declared type wins; the distance is per call."""

    @pytest.mark.parametrize(
        "general_first", [True, False], ids=["general-first", "specific-first"]
    )
    def test_specific_beats_general_whichever_was_registered_first(
        self, arity: Arity, general_first: bool
    ):
        reg = arity.registry()
        methods = [
            arity.method("general", priority=1, supported=arity.general, result="general"),
            arity.method("specific", priority=1, supported=arity.specific, result="specific"),
        ]
        for method in methods if general_first else reversed(methods):
            reg.register(method)
        assert reg.execute(*arity.args) == "specific"
        assert reg.check(*arity.args).method_name == "specific"

    def test_priority_beats_specificity(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("specific", priority=1, supported=arity.specific, result="s"))
        reg.register(arity.method("general", priority=2, supported=arity.general, result="g"))
        assert reg.execute(*arity.args) == "g"

    def test_exactness_beats_specificity(self, arity: Arity):
        reg = arity.registry()
        reg.register(
            arity.method(
                "specific", exact=False, priority=100, supported=arity.specific, result="s"
            )
        )
        reg.register(
            arity.method("general", exact=True, priority=1, supported=arity.general, result="g")
        )
        assert reg.execute(*arity.args) == "g"

    def test_equal_specificity_keeps_registration_order(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("first", priority=1, supported=arity.general, result="first"))
        reg.register(arity.method("second", priority=1, supported=arity.general, result="second"))
        assert reg.execute(*arity.args) == "first"

    def test_specificity_is_per_call(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("general", priority=1, supported=arity.general, result="g"))
        reg.register(arity.method("specific", priority=1, supported=arity.specific, result="s"))
        assert reg.execute(*arity.args) == "s"
        assert reg.execute(*arity.other_args) == "g"
        assert reg.list_methods() == ["general", "specific"]

    def test_registration_order_is_irrelevant_when_specificities_differ(self, arity: Arity):
        specs = {"general": arity.general, "specific": arity.specific}
        for order in itertools.permutations(specs):
            reg = arity.registry()
            for name in order:
                reg.register(arity.method(name, priority=1, supported=specs[name], result=name))
            assert reg.execute(*arity.args) == "specific", order

    def test_reranking_rebuilds_the_per_call_order(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("general", priority=1, supported=arity.general, result="g"))
        reg.register(arity.method("specific", priority=1, supported=arity.specific, result="s"))
        assert reg.execute(*arity.args) == "s"
        reg.set_priorities(general=5)
        assert reg.execute(*arity.args) == "g"
        reg.set_priorities(general=1)
        assert reg.execute(*arity.args) == "s"

    def test_a_virtual_subclass_is_least_specific(self):
        """Admitted by ``issubclass`` without appearing in the MRO.

        Such a type ranks after every real base.
        """

        class Marker(abc.ABC):
            @abc.abstractmethod
            def mark(self) -> None: ...

        Marker.register(Left)
        reg = UnaryDispatchRegistry()
        for name, types in (("marker", (Marker,)), ("object", (object,)), ("left", (Left,))):
            reg.register(FakeUnary(name, priority=1, types=types, result=name))
        assert reg.execute(Left()) == "left"
        reg.get_method("left")._feasible = False
        assert reg.execute(Left()) == "object"
        reg.get_method("object")._feasible = False
        assert reg.execute(Left()) == "marker"

    def test_binary_distance_is_the_sum_of_the_two_sides(self):
        """For ``(LeftSub, RightSub)``, C at 1 + 0 beats A at 0 + 2 and B at 1 + 1.

        A and B tie and fall back to registration order.
        """
        reg = BinaryDispatchRegistry()
        declared = {
            "D": ((object,), (object,)),  # 2 + 2
            "A": ((LeftSub,), (object,)),  # 0 + 2
            "B": ((Left,), (Right,)),  # 1 + 1
            "C": ((Left,), (RightSub,)),  # 1 + 0
        }
        for name, (left, right) in declared.items():
            reg.register(FakeBinary(name, priority=1, left=left, right=right, result=name))
        for expected in ("C", "A", "B", "D"):
            assert reg.execute(LeftSub(), RightSub()) == expected
            reg.get_method(expected)._feasible = False


# ---------------------------------------------------------------------------
# set_priorities re-ranks among methods of the same exactness only
# ---------------------------------------------------------------------------


class TestSetPriorities:
    def test_reorders_within_exactness(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=10, result="a"))
        reg.register(arity.method("b", priority=100, result="b"))
        assert reg.list_methods() == ["b", "a"]
        reg.set_priorities(a=200)
        assert reg.list_methods() == ["a", "b"]
        assert reg.execute(*arity.args) == "a"
        assert reg.check(*arity.args).method_name == "a"

    def test_no_override_sequence_lifts_approximate_above_exact(self, arity: Arity):
        rng = random.Random(7)
        for _ in range(200):
            reg = arity.registry()
            exactness = {"e1": True, "e2": True, "a1": False, "a2": False}
            for n, ex in exactness.items():
                reg.register(arity.method(n, exact=ex, priority=rng.randint(0, 100)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(rng.randint(1, 5)):
                    name = rng.choice(list(exactness))
                    value = rng.choice([None, rng.randint(-10, 10_000)])
                    reg.set_priorities({name: value})
            order = reg.list_methods()
            assert all(exactness[n] for n in order[:2]) and not any(exactness[n] for n in order[2:])

    def test_override_never_mutates_the_method(self, arity: Arity):
        reg = arity.registry()
        m = arity.method("a", priority=10)
        reg.register(m)
        reg.set_priorities(a=99)
        assert m.priority == 10

    def test_mapping_form_ranks_a_non_identifier_name(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("scipy->tfp", priority=1, result="arrow"))
        reg.register(arity.method("plain", priority=2, result="plain"))
        assert reg.execute(*arity.args) == "plain"
        reg.set_priorities({"scipy->tfp": 3})
        assert reg.execute(*arity.args) == "arrow"

    def test_mapping_and_keywords_combine(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=1))
        reg.register(arity.method("b.c", priority=2))
        reg.set_priorities({"b.c": 5}, a=10)
        assert reg.list_methods() == ["a", "b.c"]

    def test_name_in_both_forms_raises_and_applies_nothing(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=1))
        with pytest.raises(ValueError, match="both"):
            reg.set_priorities({"a": 5}, a=6)
        assert reg._effective_priority(reg._by_name["a"]) == 1

    def test_unknown_name_raises_and_applies_nothing(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=1))
        with pytest.raises(KeyError, match="No method named 'nope'"):
            reg.set_priorities(a=50, nope=1)
        assert reg._effective_priority(reg._by_name["a"]) == 1

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"a": 5, "nope": 1}, KeyError, "No method named 'nope'"),
            ({"a": 6}, ValueError, "both"),
            ({"a": "high"}, TypeError, "int or None"),
        ],
        ids=["unknown-name", "name-in-both-forms", "non-integer-value"],
    )
    def test_an_aborted_override_does_not_warn(self, arity: Arity, kwargs, exc, match):
        """Nothing is applied, so nothing crossed opt-in to warn about.

        The new value crosses opt-in, so a warning emitted before validation
        would be caught here.
        """
        reg = arity.registry()
        reg.register(arity.method("a", priority=None))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(exc, match=match):
                reg.set_priorities({"a": 5} if exc is ValueError else None, **kwargs)
        assert reg._effective_priority(reg._by_name["a"]) is None

    @pytest.mark.parametrize(
        "bad", ["high", 5.0, True, False], ids=["str", "float", "True", "False"]
    )
    def test_non_integer_override_raises_and_applies_nothing(self, arity: Arity, bad: Any):
        """Validated before anything is stored, so the registry stays usable."""
        reg = arity.registry()
        reg.register(arity.method("a", priority=1, result="a"))
        reg.register(arity.method("b", priority=2, result="b"))
        with pytest.raises(TypeError, match="priority must be an int or None"):
            reg.set_priorities(a=bad)
        assert reg.list_methods() == ["b", "a"]
        assert reg._effective_priority(reg._by_name["a"]) == 1
        assert reg.execute(*arity.args) == "b"

    def test_warns_once_on_each_crossing_of_opt_in(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=50))
        with pytest.warns(UserWarning, match="into opt-in-only") as rec:
            reg.set_priorities(a=None)
        assert len(rec) == 1
        with pytest.warns(UserWarning, match="out of opt-in-only") as rec:
            reg.set_priorities(a=42)
        assert len(rec) == 1

    def test_no_warning_without_a_crossing(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=50))
        reg.register(arity.method("b", priority=None))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            reg.set_priorities(a=0)
            reg.set_priorities(a=5)
            reg.set_priorities(a=-3)
            reg.set_priorities(b=None)

    def test_promoted_opt_in_method_joins_automatic_selection(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("opt_in", priority=None, result="ran"))
        with pytest.warns(UserWarning, match="out of opt-in-only"):
            reg.set_priorities(opt_in=10)
        assert reg.execute(*arity.args) == "ran"


# ---------------------------------------------------------------------------
# exact_only excludes approximate methods
# ---------------------------------------------------------------------------


class TestExactOnly:
    def test_excludes_the_only_feasible_approximate_method(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("approx", exact=False, priority=10))
        info = reg.check(*arity.args, exact_only=True)
        assert info.feasible is False
        assert "exact_only" in info.description
        with pytest.raises(ResolutionError, match="exact_only"):
            reg.execute(*arity.args, exact_only=True)

    def test_named_approximate_method_under_exact_only(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("approx", exact=False, priority=10))
        info = reg.check(*arity.args, method="approx", exact_only=True)
        assert info.feasible is False and info.exact is False
        with pytest.raises(ResolutionError, match="approximate"):
            reg.execute(*arity.args, method="approx", exact_only=True)

    def test_default_is_no_restriction(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("approx", exact=False, priority=10, result="ok"))
        assert reg.execute(*arity.args) == "ok"
        assert reg.check(*arity.args).exact is False

    def test_exact_methods_are_unaffected(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("approx", exact=False, priority=100, result="approx"))
        reg.register(arity.method("exact", exact=True, priority=1, result="exact"))
        assert reg.execute(*arity.args, exact_only=True) == "exact"


# ---------------------------------------------------------------------------
# check is a probe
# ---------------------------------------------------------------------------


class TestCheck:
    def test_returns_the_first_feasible_candidate_with_name_and_exactness(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", exact=False, priority=1))
        info = reg.check(*arity.args)
        assert isinstance(info, MethodInfo)
        assert info.feasible is True
        assert info.method_name == "m"
        assert info.exact is False

    def test_unresolved_candidate_above_a_feasible_one_is_reported_unresolved(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("pending", priority=20, feasible=None, pending=("output spec",)))
        reg.register(arity.method("ready", priority=10))
        info = reg.check(*arity.args)
        assert info.feasible is None and info.unresolved
        assert info.method_name == "pending"
        assert info.pending == ("output spec",)

    def test_all_infeasible_names_every_method_tried(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=2, feasible=False, description="needs a density"))
        reg.register(arity.method("b", priority=1, feasible=False, description="needs samples"))
        info = reg.check(*arity.args)
        assert info.feasible is False
        assert info.method_name is None and info.exact is None
        for fragment in ("a: needs a density", "b: needs samples"):
            assert fragment in info.description

    def test_never_executes(self, arity: Arity):
        reg = arity.registry()
        m = arity.method("m", priority=1)
        reg.register(m)
        reg.check(*arity.args)
        reg.check(*arity.args, method="m")
        assert m.check_calls == 2
        assert m.execute_calls == 0

    def test_by_name(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", priority=None))
        assert reg.check(*arity.args, method="m").feasible is True

    def test_unknown_name_is_a_resolution_error(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m"))
        with pytest.raises(ResolutionError, match="Available: m"):
            reg.check(*arity.args, method="nope")

    def test_no_arguments_is_a_type_error(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m"))
        with pytest.raises(TypeError, match="positional argument"):
            reg.check()
        with pytest.raises(TypeError, match="positional argument"):
            reg.execute()

    def test_named_dispatch_still_requires_the_arity(self, arity: Arity):
        """``method=`` bypasses the type pre-filter, not the arity."""
        reg = arity.registry()
        method = arity.method("m", priority=None)
        reg.register(method)
        for too_few in ((), arity.args[:-1]):
            with pytest.raises(TypeError, match="positional argument"):
                reg.check(*too_few, method="m")
            with pytest.raises(TypeError, match="positional argument"):
                reg.execute(*too_few, method="m")
        assert method.check_calls == 0

    def test_binary_registry_needs_two_arguments(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(TypeError, match="at least two positional arguments"):
            reg.check(Left())
        with pytest.raises(TypeError, match="at least two positional arguments"):
            reg.execute(Left())


# ---------------------------------------------------------------------------
# execute resolves or raises ResolutionError
# ---------------------------------------------------------------------------


class TestExecute:
    def test_runs_the_first_feasible_candidate(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", priority=1, result=42))
        assert reg.execute(*arity.args) == 42

    def test_unresolved_first_candidate_raises_with_pending(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("pending", priority=20, feasible=None, pending=("event spec",)))
        reg.register(arity.method("ready", priority=10))
        with pytest.raises(ResolutionError, match="event spec"):
            reg.execute(*arity.args)

    def test_no_feasible_candidate_names_the_methods_tried(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=2, feasible=False, description="needs a density"))
        reg.register(arity.method("b", priority=1, feasible=False, description="needs samples"))
        with pytest.raises(ResolutionError, match=r"a: needs a density.*b: needs samples"):
            reg.execute(*arity.args)

    def test_no_registered_candidate_names_the_key(self, arity: Arity):
        reg = arity.registry()
        with pytest.raises(ResolutionError, match="Left"):
            reg.execute(*arity.args)

    def test_named_infeasible_method_raises(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", feasible=False, description="no"))
        with pytest.raises(ResolutionError, match="not applicable"):
            reg.execute(*arity.args, method="m")

    def test_named_unresolved_method_raises(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", feasible=None, pending=("shape",)))
        with pytest.raises(ResolutionError, match="shape"):
            reg.execute(*arity.args, method="m")

    def test_unknown_name_is_a_resolution_error(self, arity: Arity):
        """Naming a method that is not registered is a dispatch that cannot resolve."""
        reg = arity.registry()
        reg.register(arity.method("m"))
        with pytest.raises(ResolutionError, match="Available: m"):
            reg.execute(*arity.args, method="nope")

    @pytest.mark.parametrize("exc", [ValueError("bad"), RuntimeError("worse")])
    def test_a_method_failure_propagates_and_stops_the_search(self, arity: Arity, exc: Exception):
        reg = arity.registry()
        failing = arity.method("failing", priority=2, raises=exc)
        fallback = arity.method("fallback", priority=1, result="never")
        reg.register(failing)
        reg.register(fallback)
        with pytest.raises(type(exc), match=str(exc)):
            reg.execute(*arity.args)
        assert fallback.execute_calls == 0

    def test_resolution_error_derives_directly_from_exception(self):
        """The base class is part of the II.7 contract: it decides what ``except`` catches."""
        assert ResolutionError.__mro__[1] is Exception

    def test_one_except_resolution_error_covers_every_dispatch_failure(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", feasible=False))
        for kwargs in ({}, {"method": "a"}, {"method": "absent"}):
            with pytest.raises(ResolutionError):
                reg.execute(*arity.args, **kwargs)


class TestMathematicalDomainError:
    def test_is_a_value_error_and_not_a_resolution_error(self):
        assert issubclass(MathematicalDomainError, ValueError)
        assert not issubclass(MathematicalDomainError, ResolutionError)
        assert not issubclass(ResolutionError, MathematicalDomainError)

    def test_propagates_from_execute_unwrapped(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("m", priority=2, raises=MathematicalDomainError("no mean")))
        reg.register(arity.method("fallback", priority=1))
        with pytest.raises(MathematicalDomainError, match="no mean"):
            reg.execute(*arity.args)

    def test_propagates_from_check_unwrapped(self, arity: Arity):
        reg = arity.registry()
        reg.register(
            arity.method("m", priority=2, check_raises=MathematicalDomainError("undefined"))
        )
        with pytest.raises(MathematicalDomainError, match="undefined"):
            reg.check(*arity.args)
        with pytest.raises(MathematicalDomainError, match="undefined"):
            reg.execute(*arity.args)


# ---------------------------------------------------------------------------
# Feasibility and MethodInfo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("report_cls", [Feasibility, MethodInfo])
class TestFeasibility:
    def test_frozen(self, report_cls: type[Feasibility]):
        report = report_cls(feasible=False)
        with pytest.raises(AttributeError):
            report.feasible = True  # type: ignore[misc]

    def test_feasible_with_pending_is_rejected(self, report_cls: type[Feasibility]):
        with pytest.raises(ValueError, match="pending"):
            report_cls(feasible=True, pending=("x",))

    def test_unresolved_without_pending_is_rejected(self, report_cls: type[Feasibility]):
        with pytest.raises(ValueError, match="pending"):
            report_cls(feasible=None)

    def test_only_unresolved_carries_pending(self, report_cls: type[Feasibility]):
        report_cls(feasible=False)
        with pytest.raises(ValueError, match="only an unresolved"):
            report_cls(feasible=False, pending=("x",))


class TestMethodInfo:
    def test_is_a_feasibility(self):
        assert issubclass(MethodInfo, Feasibility)

    def test_name_and_exactness_are_set_together(self):
        assert MethodInfo(feasible=False).method_name is None
        assert MethodInfo(feasible=True, method_name="m", exact=True).exact is True
        for partial in ({"method_name": "m"}, {"exact": True}):
            with pytest.raises(ValueError, match="together"):
                MethodInfo(feasible=True, **partial)

    def test_a_feasible_or_unresolved_report_names_its_method(self):
        """Only the infeasible aggregate that lists every method tried names none."""
        MethodInfo(feasible=False)
        with pytest.raises(ValueError, match="names its method"):
            MethodInfo(feasible=True)
        with pytest.raises(ValueError, match="names its method"):
            MethodInfo(feasible=None, pending=("x",))


# ---------------------------------------------------------------------------
# Registration, caches, and type pre-filter
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_duplicate_name_raises(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(arity.method("dup"))

    def test_empty_name_rejected(self, arity: Arity):
        reg = arity.registry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.register(arity.method(""))
        assert reg.list_methods() == []

    @pytest.mark.parametrize("name", [None, 5], ids=["None", "int"])
    def test_non_string_name_rejected(self, arity: Arity, name: Any):
        reg = arity.registry()
        with pytest.raises(TypeError, match="must be a str"):
            reg.register(arity.method(name))
        assert reg.list_methods() == []

    @pytest.mark.parametrize(
        "bad", ["5", 5.0, True, False, [1]], ids=["str", "float", "True", "False", "list"]
    )
    def test_non_integer_priority_rejected(self, arity: Arity, bad: Any):
        reg = arity.registry()
        with pytest.raises(TypeError, match="priority must be an int or None"):
            reg.register(arity.method("m", priority=bad))
        assert reg.list_methods() == []
        with pytest.raises(KeyError):
            reg.get_method("m")

    def test_a_rejected_registration_leaves_the_registry_as_it_was(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=1, result="a"))
        assert reg.execute(*arity.args) == "a"
        for bad in (
            arity.method("a", priority=2),
            arity.method("b", exact="yes"),
            arity.method("c", priority="high"),
            arity.method("d", supported=arity.bad_supported[0]),
        ):
            with pytest.raises((TypeError, ValueError)):
                reg.register(bad)
        assert reg.list_methods() == ["a"]
        assert reg.execute(*arity.args) == "a"

    def test_supported_types_must_have_the_arity_shape(self, arity: Arity):
        """Python enforces nothing about the type parameter, so registration checks the value."""
        reg = arity.registry()
        for bad in arity.bad_supported:
            with pytest.raises(TypeError, match="supported_types"):
                reg.register(arity.method("m", supported=bad))
        assert reg.list_methods() == []
        with pytest.raises(KeyError):
            reg.get_method("m")

    @pytest.mark.parametrize("name", ["scipy->tfp", "a.b"])
    def test_any_non_empty_string_is_a_name(self, arity: Arity, name: str):
        reg = arity.registry()
        reg.register(arity.method(name, priority=None, result="ok"))
        assert reg.get_method(name).name == name
        assert reg.execute(*arity.args, method=name) == "ok"

    def test_get_method(self, arity: Arity):
        reg = arity.registry()
        m = arity.method("m")
        reg.register(m)
        assert reg.get_method("m") is m
        with pytest.raises(KeyError, match="No method named"):
            reg.get_method("nope")


class TestRegistrationIsASnapshot:
    def test_mutating_a_registered_method_changes_nothing(self, arity: Arity):
        """Declarations are read once; ranking, filtering, and reporting use that reading."""
        reg = arity.registry()
        a = arity.method("a", exact=False, priority=10, result="a")
        b = arity.method("b", exact=False, priority=5, result="b")
        reg.register(a)
        reg.register(b)
        assert reg.list_methods() == ["a", "b"]

        a._name = "renamed"  # would break the name index
        a._priority = None  # would make ``a`` opt-in-only
        b._exact = True  # would move ``b`` ahead of ``a``
        arity.exclude_args(b)  # would drop ``b`` from admission

        assert reg.list_methods() == ["a", "b"]
        assert reg.get_method("a") is a
        info = reg.check(*arity.args)
        assert (info.method_name, info.exact) == ("a", False)
        assert reg.execute(*arity.args) == "a"
        # A key no call has used yet is admitted from the registration, not from ``b`` as it is now.
        a._feasible = False
        assert reg.execute(*arity.fresh_args) == "b"


class TestCaches:
    def test_register_after_a_call_is_seen(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("first", priority=10, result="first"))
        assert reg.execute(*arity.args) == "first"
        reg.register(arity.method("second", priority=200, result="second"))
        assert reg.execute(*arity.args) == "second"

    def test_override_after_a_call_is_seen(self, arity: Arity):
        reg = arity.registry()
        reg.register(arity.method("a", priority=100, result="a"))
        reg.register(arity.method("b", priority=10, result="b"))
        assert reg.execute(*arity.args) == "a"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reg.set_priorities(a=None)
        assert reg.execute(*arity.args) == "b"


class TestTypePreFilter:
    def test_unary_subclass_matches(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeUnary("m", types=(Left,), result="ok"))
        assert reg.execute(LeftSub()) == "ok"
        with pytest.raises(ResolutionError):
            reg.execute(Right())

    def test_binary_both_slots_must_match(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinary("m", left=(Left,), right=(Right,), result="ok"))
        assert reg.execute(LeftSub(), Right()) == "ok"
        assert reg.execute(Left(), RightSub()) == "ok"
        with pytest.raises(ResolutionError, match=r"\(Right, Right\)"):
            reg.execute(Right(), Right())
        with pytest.raises(ResolutionError, match=r"\(Left, Left\)"):
            reg.execute(Left(), Left())


# ---------------------------------------------------------------------------
# The design and the code agree
# ---------------------------------------------------------------------------

_DESIGN = Path(__file__).resolve().parents[2] / "design" / "02-shared-abstractions.md"


def _design_block() -> str:
    text = _DESIGN.read_text()
    start = text.index("## II.7 ")
    block = re.search(r"```python\n(.*?)```", text[start:], re.S)
    assert block is not None
    return block.group(1)


@pytest.mark.skipif(not _DESIGN.exists(), reason="design reference not checked out")
class TestDesignAgreement:
    def test_declared_classes_exist(self):
        names = re.findall(r"^class (\w+)", _design_block(), re.M)
        assert names, "no classes found in the II.7 block"
        for name in names:
            assert hasattr(_dispatch, name), f"{name} is declared in II.7 but not implemented"

    def test_declared_exception_bases_match(self):
        """Which builtin an error derives from decides which ``except`` clauses catch it."""
        declared_bases = re.findall(r"^class (\w*Error)\((\w+)\)", _design_block(), re.M)
        assert declared_bases, "no exception classes found in the II.7 block"
        for name, base in declared_bases:
            implemented_class = getattr(_dispatch, name)
            assert issubclass(implemented_class, getattr(builtins, base)), (name, base)
            assert implemented_class.__mro__[1].__name__ == base, (
                f"{name} derives from {implemented_class.__mro__[1].__name__}, II.7 says {base}"
            )

    def test_every_public_name_is_declared(self):
        """A name II.7 has dropped must not survive in ``__all__``."""
        declared_names = set(re.findall(r"^(?:class|type) (\w+)", _design_block(), re.M))
        exported_names = {name for name in _dispatch.__all__ if name[0].isupper()}
        assert exported_names <= declared_names, exported_names - declared_names

    @pytest.mark.parametrize("report_cls", [Feasibility, MethodInfo])
    def test_report_fields_match(self, report_cls: type[Feasibility]):
        """Each class declares in II.7 exactly the fields it adds to its base."""
        block = _design_block()
        header = re.search(rf"^class {report_cls.__name__}\b", block, re.M)
        assert header is not None, report_cls.__name__
        section = block[header.start() :]
        section = section[: section.index("\nclass ")]
        declared = dict(re.findall(r"^\s+(\w+):\s+([^#\n]+?)\s*(?:#.*)?$", section, re.M))
        inherited = {
            f.name for base in report_cls.__bases__ if is_dataclass(base) for f in fields(base)
        }
        implemented = {f.name: f.type for f in fields(report_cls) if f.name not in inherited}
        assert set(declared) == set(implemented)
        for name, annotation in declared.items():
            assert annotation.replace(" ", "") == str(implemented[name]).replace(" ", "")

    def test_base_method_declares_exact_priority_and_supported_types(self):
        block = _design_block()
        header = re.search(r"^class BaseDispatchMethod\b", block, re.M)
        assert header is not None
        section = block[header.start() :]
        section = section[: section.index("\nclass ")]
        for attr in ("exact", "priority"):
            assert re.search(rf"^\s+{attr}:", section, re.M), attr
            assert isinstance(getattr(BaseDispatchMethod, attr), property)
        assert re.search(r"^\s+def supported_types\(", section, re.M)
        assert "supported_types" in BaseDispatchMethod.__abstractmethods__

    def test_registry_signatures_match(self):
        block = _design_block()
        block = block[block.index("class BaseDispatchRegistry") :]
        for method_name in ("set_priorities", "execute", "check"):
            declared_signature = re.search(rf"def {method_name}\((.*?)\)\s*->", block, re.S)
            assert declared_signature is not None, method_name
            declared_entries = [
                entry.strip()
                for entry in _split_top_level(declared_signature.group(1).replace("\n", " "))
                if entry.strip()
            ]
            declared_params = [
                entry.split(":")[0].split("=")[0].strip().lstrip("*")
                for entry in declared_entries
                if entry != "/"
            ]
            signature = inspect.signature(getattr(BaseDispatchRegistry, method_name))
            implemented_params = list(signature.parameters)
            assert declared_params == implemented_params, (
                method_name,
                declared_params,
                implemented_params,
            )

            # Defaults decide what a bare call does, so they are compared too.
            for entry in declared_entries:
                if "=" not in entry or entry == "/":
                    continue
                param_name = entry.split(":")[0].split("=")[0].strip().lstrip("*")
                declared_default = entry.split("=", 1)[1].strip()
                implemented_default = signature.parameters[param_name].default
                assert declared_default in (str(implemented_default), repr(implemented_default)), (
                    method_name,
                    param_name,
                    declared_default,
                    implemented_default,
                )

            # A positional-only marker in II.7 must be one in the implementation.
            declared_positional_only = "/" in declared_entries
            implemented_positional_only = any(
                parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                for parameter in signature.parameters.values()
            )
            assert declared_positional_only == implemented_positional_only, method_name


def _split_top_level(text: str) -> list[str]:
    """Split on commas outside brackets, so an annotation like ``Mapping[str, int]`` stays whole."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts
