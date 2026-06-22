"""Tests for ``probpipe.core._registry_catalog``.

Covers:

- :class:`RegistryCatalog` mechanics: register / lookup / duplicate /
  describe / repr / method-indirection.
- The :class:`SupportsRegistryCataloging` protocol behaves under
  ``isinstance`` for compliant and non-compliant objects.
- Back-compatibility of :class:`BaseDispatchRegistry` constructor — bare
  ``UnaryDispatchRegistry()`` keeps working and stays out of the catalog.
- Global-catalog population on ``import probpipe``: the built-in
  registries (``inference``, ``converters``, ``bijectors``) are present
  with the expected ``kind`` and non-zero method counts.
- Adapters expose non-empty :meth:`method_summaries` for the
  converter / bijector facades.
"""

from __future__ import annotations

from typing import Any

import pytest

import probpipe  # noqa: F401 — needed to populate the global catalog
from probpipe.core._registry import (
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from probpipe.core._registry_catalog import (
    MethodSummary,
    RegistryCatalog,
    RegistryInfo,
    SupportsRegistryCataloging,
    registry_catalog,
)
from probpipe.inference import inference_method_registry

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRegistry:
    """Hand-rolled object satisfying ``SupportsRegistryCataloging``."""

    def __init__(self, name: str, kind: str = "other", description: str = "") -> None:
        self.name = name
        self.description = description
        self.kind = kind
        self._summaries: list[MethodSummary] = []

    def method_summaries(self) -> list[MethodSummary]:
        return list(self._summaries)

    def describe_method(self, name: str) -> MethodSummary:
        for s in self._summaries:
            if s.name == name:
                return s
        raise KeyError(name)

    def add(self, s: MethodSummary) -> None:
        self._summaries.append(s)


class _StubFakeMethod(UnaryDispatchMethod):
    """Minimal :class:`UnaryDispatchMethod` for catalog-population tests."""

    def __init__(self, name: str, priority: int) -> None:
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def supported_types(self) -> tuple[type, ...]:
        return (object,)

    def check(self, *args: Any, **kw: Any) -> MethodInfo:
        return MethodInfo(feasible=True, method_name=self._name)

    def execute(self, *args: Any, **kw: Any) -> Any:
        return self._name


class _StubMethodWithDescription(UnaryDispatchMethod):
    """Stub method that carries a non-default :attr:`description`.

    Used by :class:`TestDispatchRegistryMethodSummaries` to verify that
    the description class attribute on :class:`BaseDispatchMethod`
    propagates through to :class:`MethodSummary` records.
    """

    description = "stub method for catalog tests"

    def __init__(self, name: str, priority: int) -> None:
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def supported_types(self) -> tuple[type, ...]:
        return (object,)

    def check(self, *args: Any, **kw: Any) -> MethodInfo:
        return MethodInfo(feasible=True, method_name=self._name)

    def execute(self, *args: Any, **kw: Any) -> Any:
        return self._name


# ---------------------------------------------------------------------------
# RegistryCatalog mechanics (use a *local* catalog, not the global one)
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_lookup(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a")
        b = _StubRegistry("b")
        cat.register(a)
        cat.register(b)
        assert cat["a"] is a
        assert cat["b"] is b

    def test_contains(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("a"))
        assert "a" in cat
        assert "missing" not in cat
        # Non-string operands return False, not TypeError.
        assert (object() in cat) is False

    def test_duplicate_name_raises(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("a"))
        with pytest.raises(ValueError, match="already registered"):
            cat.register(_StubRegistry("a"))

    def test_empty_name_raises(self) -> None:
        cat = RegistryCatalog()
        with pytest.raises(ValueError, match="non-empty name"):
            cat.register(_StubRegistry(""))

    def test_missing_lookup_raises(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("a"))
        with pytest.raises(KeyError, match="No registry named 'missing'"):
            cat["missing"]


class TestQuery:
    def test_names_sorted(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("z"))
        cat.register(_StubRegistry("a"))
        cat.register(_StubRegistry("m"))
        assert cat.names() == ["a", "m", "z"]

    def test_list_returns_registry_info(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch", description="A")
        a.add(MethodSummary(name="m1", priority=50))
        a.add(MethodSummary(name="m2", priority=10))
        cat.register(a)
        infos = cat.list()
        assert len(infos) == 1
        assert infos[0] == RegistryInfo(name="a", description="A", kind="dispatch", method_count=2)

    def test_describe_separates_opt_in(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch", description="A")
        a.add(MethodSummary(name="hot", priority=80))
        a.add(MethodSummary(name="cold", priority=0))  # opt-in only
        cat.register(a)
        out = cat.describe("a")
        # Both methods appear, but in different sections.
        assert "hot" in out
        assert "cold" in out
        assert "Auto-dispatched (by priority):" in out
        assert "Opt-in only" in out
        # hot must be listed under the auto section (i.e., before "Opt-in only").
        assert out.index("hot") < out.index("Opt-in only")
        assert out.index("Opt-in only") < out.index("cold")

    def test_describe_factory_uses_methods_label(self) -> None:
        cat = RegistryCatalog()
        f = _StubRegistry("f", kind="factory", description="F")
        f.add(MethodSummary(name="one", priority=None))
        f.add(MethodSummary(name="two", priority=None))
        cat.register(f)
        out = cat.describe("f")
        assert "Methods:" in out
        assert "Auto-dispatched (by priority)" not in out

    def test_describe_empty_registry(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("empty"))
        out = cat.describe("empty")
        assert "(no methods registered)" in out


class TestRepr:
    def test_empty_repr(self) -> None:
        cat = RegistryCatalog()
        assert "empty" in repr(cat)

    def test_populated_repr(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch", description="alpha")
        a.add(MethodSummary(name="m1", priority=10))
        cat.register(a)
        text = repr(cat)
        assert "a" in text
        assert "dispatch" in text
        assert "1 method" in text
        assert "alpha" in text

    def test_html_repr(self) -> None:
        cat = RegistryCatalog()
        assert "empty" in cat._repr_html_()
        a = _StubRegistry("a", kind="dispatch")
        a.add(MethodSummary(name="m1", priority=1))
        cat.register(a)
        html = cat._repr_html_()
        assert "<table>" in html
        assert "a" in html


class TestRegisterMethodIndirection:
    def test_register_method_forwards(self) -> None:
        cat = RegistryCatalog()
        # Use a real dispatch registry (with register_in_catalog=False so it
        # doesn't double-register in the global catalog).
        reg = UnaryDispatchRegistry(name="x", register_in_catalog=False)
        cat.register(reg)
        m = _StubFakeMethod("m1", priority=10)
        cat.register_method("x", m)
        assert reg.list_methods() == ["m1"]

    def test_register_method_unknown_registry_raises(self) -> None:
        cat = RegistryCatalog()
        with pytest.raises(KeyError, match="No registry named 'missing'"):
            cat.register_method("missing", _StubFakeMethod("m", 10))


class TestErrorPaths:
    """Error / edge-case paths on the catalog itself."""

    def test_describe_unknown_registry_raises(self) -> None:
        cat = RegistryCatalog()
        with pytest.raises(KeyError, match="No registry named 'missing'"):
            cat.describe("missing")

    def test_list_sorted_with_multiple_registries(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("z"))
        cat.register(_StubRegistry("a"))
        cat.register(_StubRegistry("m"))
        assert [i.name for i in cat.list()] == ["a", "m", "z"]

    def test_describe_renders_module_path_and_description(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(
            MethodSummary(
                name="m1",
                priority=50,
                description="my-desc",
                module_path="some.module.path",
            )
        )
        cat.register(a)
        out = cat.describe("a")
        assert "my-desc" in out
        assert "some.module.path" in out


# ---------------------------------------------------------------------------
# BaseDispatchRegistry constructor — back-compat + opt-out + errors
# ---------------------------------------------------------------------------


class TestConstructorBackCompat:
    """Bare ``UnaryDispatchRegistry()`` keeps working and stays out of the
    global catalog.  Regression guard for the ~60 bare-constructor sites
    in the existing test suite and the public-API smoke that no auto
    registration happens without an explicit ``name``.
    """

    def test_bare_construction_has_empty_name(self) -> None:
        reg = UnaryDispatchRegistry()
        assert reg.name == ""

    def test_bare_construction_not_in_catalog(self) -> None:
        before = set(registry_catalog.names())
        UnaryDispatchRegistry()
        after = set(registry_catalog.names())
        assert before == after  # nothing added


class TestOptOut:
    def test_named_but_opted_out_is_not_in_catalog(self) -> None:
        # Use a unique name that's unlikely to collide if the test ever
        # accidentally registers; if catalog membership leaked we'd see
        # this name appear, so the assertion catches the bug.
        before = set(registry_catalog.names())
        reg = UnaryDispatchRegistry(
            name="optout_smoke_test_x",
            description="should not be in catalog",
            register_in_catalog=False,
        )
        after = set(registry_catalog.names())
        assert reg.name == "optout_smoke_test_x"
        assert "optout_smoke_test_x" not in after
        assert before == after


class TestConstructorErrors:
    def test_register_in_catalog_without_name_raises(self) -> None:
        with pytest.raises(ValueError, match="without a name"):
            UnaryDispatchRegistry(register_in_catalog=True)

    def test_constructor_rejects_positional_args(self) -> None:
        # Constructor params are all keyword-only — passing the name
        # positionally is a programming error and should raise immediately.
        with pytest.raises(TypeError):
            UnaryDispatchRegistry("inference")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Global catalog population on `import probpipe`
# ---------------------------------------------------------------------------


class TestBuiltinPopulation:
    """The global catalog contains every built-in registry after
    ``import probpipe``.

    NOTE: when new registries land (``kl_registry`` in Stage 4, sibling
    discrepancy registries in Stage 5, ``pushforward_registry`` in
    Stage 6, third-party plugins, ...), EXTEND this test with one
    assertion per new name.  Do NOT relax to a subset check — that loses
    the disappearance signal we want here.
    """

    def test_inference_registry_present(self) -> None:
        assert "inference" in registry_catalog
        info = next(i for i in registry_catalog.list() if i.name == "inference")
        assert info.kind == "dispatch"
        assert info.method_count > 0

    def test_converters_registry_present(self) -> None:
        assert "converters" in registry_catalog
        info = next(i for i in registry_catalog.list() if i.name == "converters")
        assert info.kind == "converter"
        assert info.method_count > 0

    def test_bijectors_registry_present(self) -> None:
        assert "bijectors" in registry_catalog
        info = next(i for i in registry_catalog.list() if i.name == "bijectors")
        assert info.kind == "factory"
        assert info.method_count > 0

    def test_inference_summaries_ordering_matches_list_methods(self) -> None:
        """Catalog round-trip: the inference registry's method_summaries()
        ordering matches its (unchanged) list_methods() shape and order.

        Regression guard for the dual-API design: the names list and the
        summary list must stay in lock-step.
        """
        sums = registry_catalog["inference"].method_summaries()
        names = inference_method_registry.list_methods()
        assert [s.name for s in sums] == names

    def test_converters_identity_attrs(self) -> None:
        reg = registry_catalog["converters"]
        assert reg.name == "converters"
        assert reg.kind == "converter"
        assert reg.description  # non-empty

    def test_bijectors_identity_attrs(self) -> None:
        reg = registry_catalog["bijectors"]
        assert reg.name == "bijectors"
        assert reg.kind == "factory"
        assert reg.description  # non-empty


# ---------------------------------------------------------------------------
# Adapter introspection — non-conforming registries expose method_summaries
# ---------------------------------------------------------------------------


class TestAdapterSurfaces:
    def test_converters_method_summaries_non_empty(self) -> None:
        summaries = registry_catalog["converters"].method_summaries()
        assert len(summaries) > 0
        # Each entry should carry both source and target type tuples.
        # ``target_types`` may legitimately be empty for converters whose
        # targets are protocols rather than concrete types (e.g.
        # ``ProtocolConverter``), so we don't require non-empty here.
        for s in summaries:
            assert len(s.supported_types) == 2
            source_types, target_types = s.supported_types
            assert isinstance(source_types, tuple)
            assert isinstance(target_types, tuple)
        # At least *some* converter should have non-empty source AND target
        # type tuples — otherwise the converter registry is degenerate.
        assert any(
            len(s.supported_types[0]) > 0 and len(s.supported_types[1]) > 0 for s in summaries
        )

    def test_converters_describe_method_known(self) -> None:
        reg = registry_catalog["converters"]
        names = [s.name for s in reg.method_summaries()]
        assert names  # populated
        s = reg.describe_method(names[0])
        assert s.name == names[0]

    def test_converters_describe_method_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="No converter named"):
            registry_catalog["converters"].describe_method("DoesNotExist")

    def test_bijectors_method_summaries_non_empty(self) -> None:
        summaries = registry_catalog["bijectors"].method_summaries()
        assert len(summaries) > 0
        # Factory-style → priority is None.
        for s in summaries:
            assert s.priority is None

    def test_bijectors_describe_method_known(self) -> None:
        reg = registry_catalog["bijectors"]
        names = [s.name for s in reg.method_summaries()]
        assert "Real" in names  # registered as part of the default bijector set
        s = reg.describe_method("Real")
        assert s.name == "Real"

    def test_bijectors_describe_method_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="No bijector entry named"):
            registry_catalog["bijectors"].describe_method("DoesNotExist")

    def test_converters_summaries_have_non_empty_description(self) -> None:
        # The ConverterRegistry adapter derives ``description`` from
        # ``type(c).__doc__``; at least one built-in converter has a
        # docstring, so the propagation should produce a non-empty
        # description somewhere.  A regression that broke the
        # docstring-extraction code would silently produce all-empty
        # descriptions here.
        sums = registry_catalog["converters"].method_summaries()
        assert any(s.description for s in sums)

    def test_bijectors_summaries_have_empty_descriptions(self) -> None:
        # Factory-style: each entry is just a (constraint key → factory)
        # pair, with no per-entry description metadata.
        sums = registry_catalog["bijectors"].method_summaries()
        assert all(s.description == "" for s in sums)

    def test_bijectors_supported_types_is_one_tuple(self) -> None:
        for s in registry_catalog["bijectors"].method_summaries():
            assert len(s.supported_types) == 1

    def test_bijector_facade_renders_instance_key_via_repr(self) -> None:
        """Instance keys (rather than constraint *types*) take the
        ``repr(key)`` branch of ``_bijector_entry_name``.

        Default registrations use type keys, so this exercises a
        currently-untested branch.
        """
        from probpipe.core.constraints import _Positive
        from probpipe.distributions._bijector_dispatch import (
            _CONSTRAINT_BIJECTOR_REGISTRY,
            _BijectorRegistryFacade,
        )

        instance_key = _Positive()
        saved = dict(_CONSTRAINT_BIJECTOR_REGISTRY)
        try:
            _CONSTRAINT_BIJECTOR_REGISTRY[instance_key] = lambda c: None
            sums = _BijectorRegistryFacade().method_summaries()
            names = [s.name for s in sums]
            assert repr(instance_key) in names
        finally:
            _CONSTRAINT_BIJECTOR_REGISTRY.clear()
            _CONSTRAINT_BIJECTOR_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Protocol behaviour
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_compliant_stub_satisfies_protocol(self) -> None:
        s = _StubRegistry("anything", kind="other")
        assert isinstance(s, SupportsRegistryCataloging)

    def test_base_dispatch_registry_satisfies_protocol(self) -> None:
        reg = UnaryDispatchRegistry(register_in_catalog=False)
        assert isinstance(reg, SupportsRegistryCataloging)

    def test_object_missing_required_attrs_does_not_satisfy(self) -> None:
        # ``object()`` has none of the required attributes.
        assert not isinstance(object(), SupportsRegistryCataloging)

    def test_method_summary_is_opt_in_only(self) -> None:
        assert MethodSummary(name="x", priority=0).is_opt_in_only is True
        assert MethodSummary(name="x", priority=50).is_opt_in_only is False
        # priority=None (factory-style) is NOT opt-in-only; opt-in is
        # priority==0 specifically (the dispatch sentinel).
        assert MethodSummary(name="x", priority=None).is_opt_in_only is False


# ---------------------------------------------------------------------------
# method_summaries / describe_method on a dispatch registry directly
# ---------------------------------------------------------------------------


class TestDispatchRegistryMethodSummaries:
    """Direct test of the two new methods on ``BaseDispatchRegistry``.

    The adapter tests above cover the *non-conforming* paths (converters
    and bijectors).  These tests exercise the *conforming* path through
    a real ``UnaryDispatchRegistry`` so the ``MethodSummary`` field
    mapping, priority ordering, override reflection, and round-trip
    against ``describe_method`` are all under direct test.
    """

    def _registry_with(
        self, *names_and_priorities: tuple[str, int]
    ) -> UnaryDispatchRegistry[UnaryDispatchMethod]:
        reg = UnaryDispatchRegistry(register_in_catalog=False)
        for name, priority in names_and_priorities:
            reg.register(_StubMethodWithDescription(name, priority))
        return reg

    def test_method_summaries_priority_order_matches_list_methods(self) -> None:
        reg = self._registry_with(("low", 10), ("hi", 90), ("mid", 50))
        # list_methods is the priority-ordered names list.
        assert reg.list_methods() == ["hi", "mid", "low"]
        # method_summaries follows the same ordering.
        assert [s.name for s in reg.method_summaries()] == ["hi", "mid", "low"]

    def test_method_summaries_fields_match_method(self) -> None:
        reg = self._registry_with(("only", 42))
        [s] = reg.method_summaries()
        assert s.name == "only"
        assert s.priority == 42
        assert s.supported_types == (object,)
        assert s.description == "stub method for catalog tests"
        # ``module_path`` is the test module name.
        assert s.module_path == __name__

    def test_method_summaries_reflect_set_priorities_override(self) -> None:
        reg = self._registry_with(("a", 10), ("b", 90))
        # Before override: b > a.
        assert [s.name for s in reg.method_summaries()] == ["b", "a"]
        assert [s.priority for s in reg.method_summaries()] == [90, 10]
        # Bump ``a`` above ``b``.
        reg.set_priorities(a=100)
        assert [s.name for s in reg.method_summaries()] == ["a", "b"]
        assert [s.priority for s in reg.method_summaries()] == [100, 90]

    def test_default_description_yields_empty_string(self) -> None:
        # A method that doesn't override ``description`` inherits ``""``
        # from ``BaseDispatchMethod``; the summary reflects that.
        class _NoDesc(UnaryDispatchMethod):
            @property
            def name(self) -> str:
                return "nd"

            @property
            def priority(self) -> int:
                return 5

            def supported_types(self) -> tuple[type, ...]:
                return (object,)

            def check(self, *a: Any, **k: Any) -> MethodInfo:
                return MethodInfo(feasible=True)

            def execute(self, *a: Any, **k: Any) -> Any:
                return None

        reg = UnaryDispatchRegistry(register_in_catalog=False)
        reg.register(_NoDesc())
        [s] = reg.method_summaries()
        assert s.description == ""

    def test_describe_method_round_trips_summary(self) -> None:
        reg = self._registry_with(("x", 25))
        [s_from_list] = reg.method_summaries()
        s_from_describe = reg.describe_method("x")
        assert s_from_describe == s_from_list

    def test_describe_method_unknown_raises_with_available(self) -> None:
        reg = self._registry_with(("known", 10))
        with pytest.raises(KeyError, match="No method named 'missing'"):
            reg.describe_method("missing")
        # The error message lists what is available.
        try:
            reg.describe_method("missing")
        except KeyError as exc:
            assert "known" in str(exc)

    def test_method_summaries_on_empty_registry(self) -> None:
        reg = UnaryDispatchRegistry(register_in_catalog=False)
        assert reg.method_summaries() == []
