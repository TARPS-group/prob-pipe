"""Tests for ``probpipe.core._catalog`` and the catalog surface of the dispatch registries.

Covers:

- :class:`RegistryCatalog` mechanics: register / lookup / duplicate /
  describe / repr.
- The :class:`SupportsRegistryCataloging` protocol behaves under
  ``isinstance`` for compliant and non-compliant objects.
- Constructing a dispatch registry never catalogs it; membership takes an
  explicit :meth:`RegistryCatalog.register`.
- Global-catalog population on ``import probpipe``: the built-in
  registries (``inference``, ``converters``, ``bijectors``) are present
  with the expected ``kind`` and non-zero entry counts.
- Adapters expose non-empty :meth:`entry_summaries` for the
  converter / bijector facades.
- The catalog block of design II.7 agrees with the implementation.
"""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

import probpipe  # noqa: F401 — needed to populate the global catalog
from probpipe.core import _catalog
from probpipe.core._catalog import (
    EntrySummary,
    RegistryCatalog,
    RegistryInfo,
    SupportsRegistryCataloging,
    registry_catalog,
)
from probpipe.core._dispatch import (
    Feasibility,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
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
        self._summaries: list[EntrySummary] = []

    def entry_summaries(self) -> list[EntrySummary]:
        return list(self._summaries)

    def describe_entry(self, name: str) -> EntrySummary:
        for s in self._summaries:
            if s.name == name:
                return s
        raise KeyError(name)

    def add(self, s: EntrySummary) -> None:
        self._summaries.append(s)


class _StubMethodWithDescription(UnaryDispatchMethod):
    """Stub method that carries a non-default :attr:`description`.

    Used by :class:`TestDispatchRegistryEntrySummaries` to verify that
    the description class attribute on :class:`BaseDispatchMethod`
    propagates through to :class:`EntrySummary` records.
    """

    description = "stub method for catalog tests"

    def __init__(self, name: str, priority: int | None, exact: bool = False) -> None:
        self._name = name
        self._priority = priority
        self._exact = exact

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int | None:
        return self._priority

    def supported_types(self) -> tuple[type, ...]:
        return (object,)

    def check(self, *args: Any, **kw: Any) -> Feasibility:
        return Feasibility(feasible=True)

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
        with pytest.raises(ValueError, match="without a name"):
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
        a.add(EntrySummary(name="m1", priority=50))
        a.add(EntrySummary(name="m2", priority=10))
        cat.register(a)
        infos = cat.list()
        assert len(infos) == 1
        assert infos[0] == RegistryInfo(name="a", description="A", kind="dispatch", entry_count=2)

    def test_describe_separates_opt_in(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch", description="A")
        a.add(EntrySummary(name="hot", priority=80))
        a.add(EntrySummary(name="cold", priority=None))  # opt-in only
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

    def test_describe_factory_uses_entries_label(self) -> None:
        cat = RegistryCatalog()
        f = _StubRegistry("f", kind="factory", description="F")
        f.add(EntrySummary(name="one", priority=None))
        f.add(EntrySummary(name="two", priority=None))
        cat.register(f)
        out = cat.describe("f")
        assert "Entries:" in out
        assert "Auto-dispatched" not in out
        # A factory's None priorities are not opt-in-only in any dispatch sense.
        assert "Opt-in only" not in out

    def test_describe_shows_exactness_when_declared(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="closed_form", priority=10, exact=True))
        a.add(EntrySummary(name="monte_carlo", priority=90, exact=False))
        cat.register(a)
        out = cat.describe("a")
        assert "Auto-dispatched (exact first, then by priority):" in out
        assert re.search(r"10\s+exact\s+closed_form", out)
        assert re.search(r"90\s+approx\s+monte_carlo", out)

    def test_describe_omits_exactness_when_undeclared(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="converter")
        a.add(EntrySummary(name="c", priority=10))
        cat.register(a)
        out = cat.describe("a")
        assert "exact first" not in out
        assert re.search(r"^\s+10  c$", out, re.M)

    def test_describe_empty_registry(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("empty"))
        out = cat.describe("empty")
        assert "(no entries registered)" in out


class TestRepr:
    def test_empty_repr(self) -> None:
        cat = RegistryCatalog()
        assert "empty" in repr(cat)

    def test_populated_repr(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch", description="alpha")
        a.add(EntrySummary(name="m1", priority=10))
        cat.register(a)
        text = repr(cat)
        assert "a" in text
        assert "dispatch" in text
        assert "1 entry" in text
        assert "alpha" in text

    def test_html_repr(self) -> None:
        cat = RegistryCatalog()
        assert "empty" in cat._repr_html_()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="m1", priority=1))
        cat.register(a)
        html = cat._repr_html_()
        assert "<table>" in html
        assert "a" in html


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
            EntrySummary(
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
# BaseDispatchRegistry construction and explicit cataloging
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_bare_construction_has_empty_identity(self) -> None:
        reg = UnaryDispatchRegistry()
        assert reg.name == ""
        assert reg.description == ""
        assert reg.kind == "dispatch"

    def test_named_construction_does_not_catalog(self) -> None:
        before = set(registry_catalog.names())
        reg = UnaryDispatchRegistry(name="construction_smoke_test", description="d")
        assert reg.name == "construction_smoke_test"
        assert reg.description == "d"
        assert set(registry_catalog.names()) == before

    def test_explicit_register_catalogs_a_named_registry(self) -> None:
        cat = RegistryCatalog()
        reg = UnaryDispatchRegistry(name="local", description="a local registry")
        cat.register(reg)
        assert cat["local"] is reg
        assert cat.list() == [
            RegistryInfo(
                name="local", description="a local registry", kind="dispatch", entry_count=0
            )
        ]

    def test_unnamed_registry_cannot_be_cataloged(self) -> None:
        with pytest.raises(ValueError, match="without a name"):
            RegistryCatalog().register(UnaryDispatchRegistry())

    def test_constructor_rejects_positional_args(self) -> None:
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
        assert info.entry_count > 0

    def test_converters_registry_present(self) -> None:
        assert "converters" in registry_catalog
        info = next(i for i in registry_catalog.list() if i.name == "converters")
        assert info.kind == "converter"
        assert info.entry_count > 0

    def test_bijectors_registry_present(self) -> None:
        assert "bijectors" in registry_catalog
        info = next(i for i in registry_catalog.list() if i.name == "bijectors")
        assert info.kind == "factory"
        assert info.entry_count > 0

    def test_inference_summaries_ordering_matches_list_methods(self) -> None:
        """Catalog round-trip: the inference registry's entry_summaries()
        ordering matches its (unchanged) list_methods() shape and order.

        Regression guard for the dual-API design: the names list and the
        summary list must stay in lock-step.
        """
        sums = registry_catalog["inference"].entry_summaries()
        names = inference_method_registry.list_methods()
        assert [s.name for s in sums] == names

    def test_inference_registry_is_the_singleton(self) -> None:
        assert registry_catalog["inference"] is inference_method_registry

    def test_inference_summaries_carry_exactness(self) -> None:
        # Every built-in inference method is approximate (C1).
        assert all(s.exact is False for s in registry_catalog["inference"].entry_summaries())

    def test_inference_describe_lists_opt_in_methods_apart(self) -> None:
        out = registry_catalog.describe("inference")
        opt_in = [s.name for s in inference_method_registry.entry_summaries() if s.is_opt_in_only]
        assert opt_in, "expected at least one opt-in-only inference method"
        for name in opt_in:
            assert out.index("Opt-in only") < out.index(f"  {name}")

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
# Adapter introspection — non-conforming registries expose entry_summaries
# ---------------------------------------------------------------------------


class TestAdapterSurfaces:
    def test_converters_entry_summaries_non_empty(self) -> None:
        summaries = registry_catalog["converters"].entry_summaries()
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

    def test_converters_describe_entry_known(self) -> None:
        reg = registry_catalog["converters"]
        names = [s.name for s in reg.entry_summaries()]
        assert names  # populated
        s = reg.describe_entry(names[0])
        assert s.name == names[0]

    def test_converters_describe_entry_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="No converter named"):
            registry_catalog["converters"].describe_entry("DoesNotExist")

    def test_bijectors_entry_summaries_non_empty(self) -> None:
        summaries = registry_catalog["bijectors"].entry_summaries()
        assert len(summaries) > 0
        # Factory-style → priority is None.
        for s in summaries:
            assert s.priority is None

    def test_bijectors_describe_entry_known(self) -> None:
        reg = registry_catalog["bijectors"]
        names = [s.name for s in reg.entry_summaries()]
        assert "Real" in names  # registered as part of the default bijector set
        s = reg.describe_entry("Real")
        assert s.name == "Real"

    def test_bijectors_describe_entry_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="No bijector entry named"):
            registry_catalog["bijectors"].describe_entry("DoesNotExist")

    def test_converters_summaries_have_non_empty_description(self) -> None:
        # The ConverterRegistry adapter derives ``description`` from
        # ``type(c).__doc__``; at least one built-in converter has a
        # docstring, so the propagation should produce a non-empty
        # description somewhere.  A regression that broke the
        # docstring-extraction code would silently produce all-empty
        # descriptions here.
        sums = registry_catalog["converters"].entry_summaries()
        assert any(s.description for s in sums)

    def test_bijectors_summaries_have_empty_descriptions(self) -> None:
        # Factory-style: each entry is just a (constraint key → factory)
        # pair, with no per-entry description metadata.
        sums = registry_catalog["bijectors"].entry_summaries()
        assert all(s.description == "" for s in sums)

    def test_bijectors_supported_types_is_one_tuple(self) -> None:
        for s in registry_catalog["bijectors"].entry_summaries():
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
            sums = _BijectorRegistryFacade().entry_summaries()
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
        reg = UnaryDispatchRegistry()
        assert isinstance(reg, SupportsRegistryCataloging)

    def test_object_missing_required_attrs_does_not_satisfy(self) -> None:
        # ``object()`` has none of the required attributes.
        assert not isinstance(object(), SupportsRegistryCataloging)

    def test_entry_summary_is_opt_in_only(self) -> None:
        # None is opt-in-only, as in dispatch; 0 is an ordinary rank.
        assert EntrySummary(name="x", priority=None).is_opt_in_only is True
        assert EntrySummary(name="x", priority=0).is_opt_in_only is False
        assert EntrySummary(name="x", priority=50).is_opt_in_only is False

    def test_entry_summary_exact_defaults_to_none(self) -> None:
        assert EntrySummary(name="x", priority=1).exact is None


# ---------------------------------------------------------------------------
# entry_summaries / describe_entry on a dispatch registry directly
# ---------------------------------------------------------------------------


class TestDispatchRegistryEntrySummaries:
    """Direct test of the two new methods on ``BaseDispatchRegistry``.

    The adapter tests above cover the *non-conforming* paths (converters
    and bijectors).  These tests exercise the *conforming* path through
    a real ``UnaryDispatchRegistry`` so the ``EntrySummary`` field
    mapping, priority ordering, override reflection, and round-trip
    against ``describe_entry`` are all under direct test.
    """

    def _registry_with(
        self, *names_and_priorities: tuple[str, int | None]
    ) -> UnaryDispatchRegistry[UnaryDispatchMethod]:
        reg = UnaryDispatchRegistry()
        for name, priority in names_and_priorities:
            reg.register(_StubMethodWithDescription(name, priority))
        return reg

    def test_entry_summaries_priority_order_matches_list_methods(self) -> None:
        reg = self._registry_with(("low", 10), ("hi", 90), ("mid", 50))
        # list_methods is the priority-ordered names list.
        assert reg.list_methods() == ["hi", "mid", "low"]
        # entry_summaries follows the same ordering.
        assert [s.name for s in reg.entry_summaries()] == ["hi", "mid", "low"]

    def test_entry_summaries_fields_match_method(self) -> None:
        reg = self._registry_with(("only", 42))
        [s] = reg.entry_summaries()
        assert s.name == "only"
        assert s.priority == 42
        assert s.supported_types == (object,)
        assert s.description == "stub method for catalog tests"
        assert s.exact is False
        # ``module_path`` is the test module name.
        assert s.module_path == __name__

    def test_entry_summaries_rank_exact_first(self) -> None:
        reg = UnaryDispatchRegistry()
        reg.register(_StubMethodWithDescription("approx_hi", 90))
        reg.register(_StubMethodWithDescription("exact_lo", 10, exact=True))
        reg.register(_StubMethodWithDescription("opt_in", None, exact=True))
        summaries = reg.entry_summaries()
        assert [s.name for s in summaries] == reg.list_methods()
        # An exact method precedes an approximate one whatever their ranks.
        assert summaries.index(reg.describe_entry("exact_lo")) < summaries.index(
            reg.describe_entry("approx_hi")
        )
        assert reg.describe_entry("opt_in").is_opt_in_only

    def test_opt_in_move_is_reported(self) -> None:
        reg = self._registry_with(("m", 10))
        with pytest.warns(UserWarning, match="into opt-in-only"):
            reg.set_priorities(m=None)
        assert reg.describe_entry("m").is_opt_in_only

    def test_summary_reads_the_registration(self) -> None:
        """A method mutated after registration changes nothing the catalog reports."""
        method = _StubMethodWithDescription("m", 10)
        reg = UnaryDispatchRegistry()
        reg.register(method)
        method._priority = 99
        method.description = "changed"  # type: ignore[misc]
        s = reg.describe_entry("m")
        assert s.priority == 10
        assert s.description == "stub method for catalog tests"

    def test_non_string_description_rejected(self) -> None:
        method = _StubMethodWithDescription("m", 10)
        method.description = 3  # type: ignore[assignment,misc]
        reg = UnaryDispatchRegistry()
        with pytest.raises(TypeError, match="description must be a str"):
            reg.register(method)
        assert reg.list_methods() == []

    def test_entry_summaries_reflect_set_priorities_override(self) -> None:
        reg = self._registry_with(("a", 10), ("b", 90))
        # Before override: b > a.
        assert [s.name for s in reg.entry_summaries()] == ["b", "a"]
        assert [s.priority for s in reg.entry_summaries()] == [90, 10]
        # Bump ``a`` above ``b``.
        reg.set_priorities(a=100)
        assert [s.name for s in reg.entry_summaries()] == ["a", "b"]
        assert [s.priority for s in reg.entry_summaries()] == [100, 90]

    def test_default_description_yields_empty_string(self) -> None:
        # A method that doesn't override ``description`` inherits ``""``
        # from ``BaseDispatchMethod``; the summary reflects that.
        class _NoDesc(UnaryDispatchMethod):
            @property
            def name(self) -> str:
                return "nd"

            @property
            def exact(self) -> bool:
                return False

            @property
            def priority(self) -> int:
                return 5

            def supported_types(self) -> tuple[type, ...]:
                return (object,)

            def check(self, *a: Any, **k: Any) -> Feasibility:
                return Feasibility(feasible=True)

            def execute(self, *a: Any, **k: Any) -> Any:
                return None

        reg = UnaryDispatchRegistry()
        reg.register(_NoDesc())
        [s] = reg.entry_summaries()
        assert s.description == ""

    def test_describe_entry_round_trips_summary(self) -> None:
        reg = self._registry_with(("x", 25))
        [s_from_list] = reg.entry_summaries()
        s_from_describe = reg.describe_entry("x")
        assert s_from_describe == s_from_list

    def test_describe_entry_unknown_raises_with_available(self) -> None:
        reg = self._registry_with(("known", 10))
        with pytest.raises(KeyError, match="No method named 'missing'"):
            reg.describe_entry("missing")
        # The error message lists what is available.
        try:
            reg.describe_entry("missing")
        except KeyError as exc:
            assert "known" in str(exc)

    def test_entry_summaries_on_empty_registry(self) -> None:
        reg = UnaryDispatchRegistry()
        assert reg.entry_summaries() == []


# ---------------------------------------------------------------------------
# Agreement with design II.7
# ---------------------------------------------------------------------------

_DESIGN = Path(__file__).resolve().parents[2] / "design" / "02-shared-abstractions.md"


def _catalog_block() -> str:
    """The II.7 code block that declares the catalog (the one naming ``EntrySummary``)."""
    text = _DESIGN.read_text()
    section = text[text.index("## II.7 ") :]
    section = section[: section.index("\n### Rationale")]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    [block] = [b for b in blocks if "class EntrySummary" in b]
    return block


def _class_section(block: str, name: str) -> str:
    header = re.search(rf"^class {name}\b.*$", block, re.M)
    assert header is not None, name
    section = block[header.end() :]
    following = re.search(r"^\S", section, re.M)
    return section[: following.start()] if following else section


@pytest.mark.skipif(not _DESIGN.exists(), reason="design reference not checked out")
class TestDesignAgreement:
    def test_declared_classes_exist(self) -> None:
        names = re.findall(r"^class (\w+)", _catalog_block(), re.M)
        assert names
        for name in names:
            assert hasattr(_catalog, name), f"{name} is declared in II.7 but not implemented"

    def test_every_public_name_is_declared(self) -> None:
        block = _catalog_block()
        declared = set(re.findall(r"^class (\w+)", block, re.M))
        declared |= set(re.findall(r"^(\w+) = ", block, re.M))
        assert set(_catalog.__all__) <= declared, set(_catalog.__all__) - declared

    @pytest.mark.parametrize("record_cls", [EntrySummary, RegistryInfo])
    def test_record_fields_match(self, record_cls: type) -> None:
        section = _class_section(_catalog_block(), record_cls.__name__)
        declared = re.findall(
            r"^\s+(\w+):\s+([^=#\n]+?)\s*(?:=\s*([^#\n]+?))?\s*(?:#.*)?$", section, re.M
        )
        implemented = fields(record_cls)
        assert [name for name, _, _ in declared] == [f.name for f in implemented]
        for (name, annotation, default), field_ in zip(declared, implemented, strict=True):
            assert annotation.replace(" ", "") == str(field_.type).replace(" ", ""), name
            if default:
                assert ast.literal_eval(default) == field_.default, name

    def test_protocol_members_match(self) -> None:
        section = _class_section(_catalog_block(), "SupportsRegistryCataloging")
        attributes = re.findall(r"^\s+(\w+):", section, re.M)
        methods = re.findall(r"^\s+def (\w+)\(", section, re.M)
        for name in (*attributes, *methods):
            assert name in SupportsRegistryCataloging.__protocol_attrs__, name
        assert set(attributes) | set(methods) == set(SupportsRegistryCataloging.__protocol_attrs__)

    def test_catalog_methods_match(self) -> None:
        section = _class_section(_catalog_block(), "RegistryCatalog")
        declared = set(re.findall(r"^\s+def (\w+)\(", section, re.M))
        for name in declared:
            assert callable(getattr(RegistryCatalog, name, None)), name
