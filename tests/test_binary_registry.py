"""Tests for BinaryDispatchRegistry and the BaseDispatchRegistry base.

Parallels ``test_inference_registry.py`` for the unary path.  Uses a
synthetic two-argument registry so no real distributions are needed.
"""

from __future__ import annotations

import warnings

import pytest

from probpipe.core._registry import (
    BaseDispatchMethod,
    BaseDispatchRegistry,
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    MethodInfo,
    OPT_IN_ONLY_PRIORITY,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)


# ---------------------------------------------------------------------------
# Synthetic type hierarchy for dispatch tests
# ---------------------------------------------------------------------------

class Left:
    pass

class LeftSub(Left):
    pass

class Right:
    pass

class RightSub(Right):
    pass


# ---------------------------------------------------------------------------
# Stub binary method
# ---------------------------------------------------------------------------

class FakeBinaryMethod(BinaryDispatchMethod):
    """Configurable stub for binary-registry tests."""

    def __init__(
        self,
        name: str,
        left_types: tuple[type, ...] = (object,),
        right_types: tuple[type, ...] = (object,),
        priority: int = 0,
        feasible: bool = True,
        result=None,
    ):
        self._name = name
        self._left_types = left_types
        self._right_types = right_types
        self._priority = priority
        self._feasible = feasible
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return (self._left_types, self._right_types)

    @property
    def priority(self) -> int:
        return self._priority

    def check(self, *args, **kw) -> MethodInfo:
        return MethodInfo(feasible=self._feasible, method_name=self._name)

    def execute(self, *args, **kw):
        return self._result if self._result is not None else self._name


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------

class TestPublicAPI:

    def test_unary_is_subclass_of_base_method(self):
        assert issubclass(UnaryDispatchMethod, BaseDispatchMethod)

    def test_binary_is_subclass_of_base_method(self):
        assert issubclass(BinaryDispatchMethod, BaseDispatchMethod)

    def test_unary_registry_is_subclass_of_base_registry(self):
        assert issubclass(UnaryDispatchRegistry, BaseDispatchRegistry)

    def test_binary_registry_is_subclass_of_base_registry(self):
        assert issubclass(BinaryDispatchRegistry, BaseDispatchRegistry)

    def test_default_priority_is_opt_in_only(self):
        """BaseDispatchMethod.priority defaults to OPT_IN_ONLY_PRIORITY."""
        class Bare(BinaryDispatchMethod):
            @property
            def name(self): return "bare"
            def supported_types(self): return ((object,), (object,))
            def check(self, *a, **kw): return MethodInfo(feasible=True)
            def execute(self, *a, **kw): return "ran"

        assert Bare().priority == OPT_IN_ONLY_PRIORITY


# ---------------------------------------------------------------------------
# BinaryDispatchRegistry — basic registration and dispatch
# ---------------------------------------------------------------------------

class TestBinaryDispatchRegistry:

    def test_register_and_list(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("low", priority=10))
        reg.register(FakeBinaryMethod("high", priority=100))
        reg.register(FakeBinaryMethod("mid", priority=50))
        assert reg.list_methods() == ["high", "mid", "low"]

    def test_duplicate_name_raises(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("dup", priority=10))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(FakeBinaryMethod("dup", priority=20))

    def test_get_method(self):
        reg = BinaryDispatchRegistry()
        m = FakeBinaryMethod("test", priority=10)
        reg.register(m)
        assert reg.get_method("test") is m

    def test_get_method_not_found(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(KeyError, match="No method named"):
            reg.get_method("nonexistent")

    def test_execute_dispatches_on_joint_types(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod(
            "lr", left_types=(Left,), right_types=(Right,), priority=10, result="lr"
        ))
        assert reg.execute(Left(), Right()) == "lr"

    def test_execute_by_name_override(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("m", priority=10, result=42))
        assert reg.execute(object(), object(), method="m") == 42

    def test_execute_no_method_raises(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(TypeError, match="No method registered"):
            reg.execute(Left(), Right())

    def test_execute_infeasible_named_method_raises(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("m", priority=10, feasible=False))
        with pytest.raises(TypeError, match="not applicable"):
            reg.execute(object(), object(), method="m")

    def test_error_message_includes_both_type_names(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(TypeError, match=r"\(Left, Right\)"):
            reg.execute(Left(), Right())

    def test_check_returns_feasible_for_matching_types(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod(
            "m", left_types=(Left,), right_types=(Right,), priority=10
        ))
        info = reg.check(Left(), Right())
        assert info.feasible

    def test_check_returns_infeasible_when_no_match(self):
        reg = BinaryDispatchRegistry()
        info = reg.check(Left(), Right())
        assert not info.feasible

    def test_check_by_name(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("m", priority=10))
        info = reg.check(object(), object(), method="m")
        assert info.feasible


# ---------------------------------------------------------------------------
# Type pre-filter: subclass matching and both-slot requirement
# ---------------------------------------------------------------------------

class TestTypePreFilter:

    def test_subclass_matches_left(self):
        """A method supporting Left also fires for LeftSub (subclass of Left)."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod(
            "m", left_types=(Left,), right_types=(Right,), priority=10, result="ok"
        ))
        assert reg.execute(LeftSub(), Right()) == "ok"

    def test_subclass_matches_right(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod(
            "m", left_types=(Left,), right_types=(Right,), priority=10, result="ok"
        ))
        assert reg.execute(Left(), RightSub()) == "ok"

    def test_wrong_left_type_excluded(self):
        """A method does not fire when the left arg type does not match."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod(
            "m", left_types=(Left,), right_types=(Right,), priority=10
        ))
        with pytest.raises(TypeError):
            reg.execute(Right(), Right())  # Right is not a subclass of Left

    def test_wrong_right_type_excluded(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod(
            "m", left_types=(Left,), right_types=(Right,), priority=10
        ))
        with pytest.raises(TypeError):
            reg.execute(Left(), Left())  # Left is not a subclass of Right

    def test_priority_order_with_multiple_matching(self):
        """When two methods match, the higher-priority one is used."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("low", priority=10, result="low"))
        reg.register(FakeBinaryMethod("high", priority=100, result="high"))
        assert reg.execute(object(), object()) == "high"

    def test_first_feasible_wins(self):
        """The highest-priority *feasible* method wins, not just the highest."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("infeasible", priority=100, feasible=False))
        reg.register(FakeBinaryMethod("feasible", priority=50, result="ok"))
        assert reg.execute(object(), object()) == "ok"


# ---------------------------------------------------------------------------
# Opt-in-only priority (OPT_IN_ONLY_PRIORITY = 0)
# ---------------------------------------------------------------------------

class TestOptInOnlyPriority:
    """Priority 0 = opt-in only: skipped during auto-dispatch."""

    def test_priority_zero_skipped_in_auto_walk(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("opt_in", priority=0, result=10))
        with pytest.raises(TypeError, match="No method registered"):
            reg.execute(object(), object())

    def test_priority_zero_reachable_by_name(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("opt_in", priority=0, result=10))
        assert reg.execute(object(), object(), method="opt_in") == 10

    def test_default_priority_is_opt_in(self):
        """FakeBinaryMethod without priority override defaults to opt-in."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("bare"))  # priority defaults to 0
        with pytest.raises(TypeError):
            reg.execute(object(), object())
        assert reg.execute(object(), object(), method="bare") == "bare"

    def test_opt_in_alongside_positive_priority(self):
        """A priority-0 method does not block a positive-priority method."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("opt_in", priority=0, result="skipped"))
        reg.register(FakeBinaryMethod("auto", priority=10, result="ran"))
        assert reg.execute(object(), object()) == "ran"

    def test_promote_from_opt_in_via_set_priorities(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("opt_in", priority=0, result="ran"))
        with pytest.warns(UserWarning, match="out of opt-in-only"):
            reg.set_priorities(opt_in=10)
        assert reg.execute(object(), object()) == "ran"


# ---------------------------------------------------------------------------
# set_priorities
# ---------------------------------------------------------------------------

class TestSetPriorities:

    def test_reorders_methods(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("a", priority=10))
        reg.register(FakeBinaryMethod("b", priority=100))
        assert reg.list_methods() == ["b", "a"]
        reg.set_priorities(a=200)
        assert reg.list_methods() == ["a", "b"]

    def test_unknown_name_raises(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(KeyError):
            reg.set_priorities(nonexistent=100)

    def test_warn_when_demoting_to_opt_in(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("a", priority=50))
        with pytest.warns(UserWarning, match="into opt-in-only"):
            reg.set_priorities(a=0)

    def test_warn_when_promoting_from_opt_in(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("a", priority=0))
        with pytest.warns(UserWarning, match="out of opt-in-only"):
            reg.set_priorities(a=42)

    def test_no_warn_when_staying_positive(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("a", priority=50))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            reg.set_priorities(a=10)
            reg.set_priorities(a=80)

    def test_no_warn_when_staying_zero(self):
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("a", priority=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            reg.set_priorities(a=0)


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:

    def test_cache_invalidated_on_register(self):
        """A method registered after a dispatch call is still found."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("first", priority=10, result="first"))
        assert reg.execute(object(), object()) == "first"

        reg.register(FakeBinaryMethod("second", priority=200, result="second"))
        assert reg.execute(object(), object()) == "second"

    def test_cache_invalidated_on_set_priorities(self):
        """Changing priority flushes the cache and changes dispatch order."""
        reg = BinaryDispatchRegistry()
        reg.register(FakeBinaryMethod("a", priority=100, result="a"))
        reg.register(FakeBinaryMethod("b", priority=10, result="b"))
        assert reg.execute(object(), object()) == "a"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reg.set_priorities(a=0)  # demote a to opt-in only
        assert reg.execute(object(), object()) == "b"


# ---------------------------------------------------------------------------
# Argument-count guards on check / execute
# ---------------------------------------------------------------------------

class TestArgumentCountGuards:
    """No-arg and single-arg dispatch must fail with a clear error."""

    def test_check_no_args_returns_infeasible(self):
        reg = BinaryDispatchRegistry()
        info = reg.check()
        assert not info.feasible
        assert "No arguments provided" in info.description

    def test_execute_no_args_raises(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(TypeError, match="No arguments provided"):
            reg.execute()

    def test_execute_one_arg_raises_typeerror(self):
        """A single-arg execute on a binary registry must raise TypeError,
        not bare IndexError from tuple-unpacking inside _cache_key."""
        reg = BinaryDispatchRegistry()
        with pytest.raises(TypeError, match="at least two positional arguments"):
            reg.execute(Left())

    def test_check_one_arg_raises_typeerror(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(TypeError, match="at least two positional arguments"):
            reg.check(Left())


# ---------------------------------------------------------------------------
# Registration guards
# ---------------------------------------------------------------------------

class TestRegistrationGuards:

    def test_empty_name_rejected(self):
        reg = BinaryDispatchRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.register(FakeBinaryMethod("", priority=10))

    def test_none_name_rejected(self):
        """The guard is `not method.name`, so it catches falsy non-strings
        (e.g., `None` from a subclass that forgets to set `name`) and
        rejects them with the same "non-empty" error as the empty-string
        case — not a downstream TypeError or KeyError."""
        reg = BinaryDispatchRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.register(FakeBinaryMethod(None, priority=10))  # type: ignore[arg-type]
