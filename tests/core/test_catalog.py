"""Tests of the registry catalog in ``probpipe.core._catalog`` and the registries' entries.

Covers:

1. the catalog: registration, lookup, listing, ``describe``, and the reprs;
2. the protocol ``SupportsRegistryCataloging`` under ``isinstance``;
3. construction: a registry joins the catalog only through ``register``;
4. the built-in registries: the bijector factory, the converters, the
   evaluation rules, the inference methods, and the operations;
5. a dispatch registry's entries: the fields, selection order, floors,
   overrides, and the registration snapshot;
6. agreement with the code blocks of design II.7, through
   ``scripts/design/design_blocks.py``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from probpipe import (
    converter_registry,
    inference_method_registry,
    operation_registry,
    register_bijector,
)
from probpipe.core import _catalog, _dispatch
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
    the ``description`` property of :class:`BaseDispatchMethod` propagates
    through to :class:`EntrySummary` records.
    """

    _description: Any = "stub method for catalog tests"

    @property
    def description(self) -> str:
        return self._description

    def __init__(
        self, name: str, priority: int | None, exact: bool = False, is_floor: bool = False
    ) -> None:
        self._name = name
        self._priority = priority
        self._exact = exact
        self._is_floor = is_floor

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int | None:
        return self._priority

    @property
    def is_floor(self) -> bool:
        return self._is_floor

    def supported_types(self) -> tuple[type, ...]:
        return (object,)

    def check(self, *args: Any, **kw: Any) -> Feasibility:
        return Feasibility(feasible=True)

    def execute(self, *args: Any, **kw: Any) -> Any:
        return self._name


@contextmanager
def _restored_bijector_registry() -> Iterator[None]:
    """Restore the bijector factory's registrations and descriptions on exit."""
    from probpipe.functions import _reparameterization as module

    tables = (module._CONSTRAINT_BIJECTOR_REGISTRY, module._BIJECTOR_DESCRIPTIONS)
    saved = [dict(table) for table in tables]
    try:
        yield
    finally:
        for table, contents in zip(tables, saved, strict=True):
            table.clear()
            table.update(contents)


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
        with pytest.raises(
            ValueError,
            match=r"^cannot catalog a second registry named 'a': the catalog already holds one",
        ):
            cat.register(_StubRegistry("a"))

    def test_empty_name_raises(self) -> None:
        cat = RegistryCatalog()
        with pytest.raises(
            ValueError, match=r'^cannot catalog a registry whose name is empty; pass name="\.\.\."'
        ):
            cat.register(_StubRegistry("", kind="dispatch"))

    def test_registering_a_cataloged_name_again_raises(self) -> None:
        """The catalog has no removal, so registering a built-in registry again raises."""
        with pytest.raises(ValueError, match="second registry named 'inference'"):
            registry_catalog.register(registry_catalog["inference"])

    def test_a_registry_class_is_refused(self) -> None:
        """A class carries the protocol's members as attributes, so it is refused by name."""
        cat = RegistryCatalog()
        with pytest.raises(TypeError, match="takes a registry instance"):
            cat.register(UnaryDispatchRegistry)  # type: ignore[arg-type]
        assert cat.names() == []

    def test_an_object_without_the_protocol_is_refused(self) -> None:
        class _NameOnly:
            name = "partial"

        cat = RegistryCatalog()
        with pytest.raises(TypeError, match="implements SupportsRegistryCataloging"):
            cat.register(_NameOnly())  # type: ignore[arg-type]
        assert cat.names() == []

    def test_a_non_string_name_is_refused(self) -> None:
        cat = RegistryCatalog()
        with pytest.raises(TypeError, match="name must be a str, got 3"):
            cat.register(_StubRegistry(3))  # type: ignore[arg-type]
        assert cat.names() == []

    def test_the_empty_name_fix_is_given_only_for_a_dispatch_registry(self) -> None:
        with pytest.raises(ValueError, match=r"^cannot catalog a registry whose name is empty$"):
            RegistryCatalog().register(_StubRegistry("", kind="factory"))

    def test_missing_lookup_raises(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("a"))
        with pytest.raises(KeyError, match="unknown registry 'missing'; available registries"):
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
        assert "Auto-dispatched, in selection order:" in out
        assert "Opt-in only" in out
        # hot must be listed under the auto section (i.e., before "Opt-in only").
        assert out.index("hot") < out.index("Opt-in only")
        assert out.index("Opt-in only") < out.index("cold")

    @pytest.mark.parametrize("kind", ["factory", "operation", "other"])
    def test_describe_lists_a_non_dispatch_registry_together(self, kind: str) -> None:
        cat = RegistryCatalog()
        f = _StubRegistry("f", kind=kind, description="F")
        f.add(EntrySummary(name="one", priority=None))
        f.add(EntrySummary(name="two", priority=None))
        cat.register(f)
        out = cat.describe("f")
        assert "Entries:" in out
        assert "Auto-dispatched" not in out
        # Only a dispatch registry ranks its entries, so only it has opt-in entries.
        assert "Opt-in only" not in out
        # No entry has a priority, so no priority column is printed.
        assert re.search(r"^    one$", out, re.M)
        assert "-" not in out

    def test_describe_lists_floors_after_the_other_entries(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="rule", priority=5, exact=False))
        a.add(EntrySummary(name="exact_floor", priority=0, exact=True, is_floor=True))
        a.add(EntrySummary(name="approx_floor", priority=0, exact=False, is_floor=True))
        a.add(EntrySummary(name="opt_in_floor", priority=None, exact=True, is_floor=True))
        cat.register(a)
        out = cat.describe("a")
        auto = out.index("Auto-dispatched, in selection order:")
        floors = out.index("Floors, after every other entry:")
        opt_in = out.index("Opt-in only")
        assert auto < out.index("rule") < floors
        assert floors < out.index("exact_floor") < out.index("approx_floor") < opt_in
        assert opt_in < out.index("opt_in_floor")

    def test_describe_has_no_fallback_section_without_floors(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="rule", priority=5))
        cat.register(a)
        assert "Floors" not in cat.describe("a")

    def test_describe_keeps_the_priority_column_when_any_entry_is_ranked(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="ranked", priority=10))
        a.add(EntrySummary(name="opt_in", priority=None))
        cat.register(a)
        out = cat.describe("a")
        assert re.search(r"^    10  ranked$", out, re.M)
        assert re.search(r"^     -  opt_in$", out, re.M)

    def test_describe_shows_exactness_when_declared(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="closed_form", priority=10, exact=True))
        a.add(EntrySummary(name="monte_carlo", priority=90, exact=False))
        cat.register(a)
        out = cat.describe("a")
        assert re.search(r"10\s+exact\s+closed_form", out)
        assert re.search(r"90\s+approx\s+monte_carlo", out)

    def test_describe_omits_exactness_when_undeclared(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="c", priority=10))
        cat.register(a)
        out = cat.describe("a")
        assert re.search(r"^\s+10  c$", out, re.M)

    def test_describe_aligns_priorities_of_any_width(self) -> None:
        cat = RegistryCatalog()
        a = _StubRegistry("a", kind="dispatch")
        a.add(EntrySummary(name="ranked", priority=5))
        a.add(EntrySummary(name="wide", priority=-(2**31)))
        cat.register(a)
        lines = [
            line for line in cat.describe("a").splitlines() if line.endswith(("ranked", "wide"))
        ]
        assert len(lines) == 2
        assert len({line.index(line.split()[-1]) for line in lines}) == 1

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

    def test_html_repr_escapes_text(self) -> None:
        cat = RegistryCatalog()
        cat.register(_StubRegistry("a<b>", kind="k&d", description="maps <x> to y & z"))
        html = cat._repr_html_()
        assert "<code>a&lt;b&gt;</code>" in html
        assert "<td>k&amp;d</td>" in html
        assert "maps &lt;x&gt; to y &amp; z" in html
        assert "<b>" not in html and "<x>" not in html


class TestErrorPaths:
    """Error / edge-case paths on the catalog itself."""

    def test_describe_unknown_registry_raises(self) -> None:
        cat = RegistryCatalog()
        with pytest.raises(KeyError, match="unknown registry 'missing'; there are no registries"):
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
        with pytest.raises(ValueError, match="name is empty"):
            RegistryCatalog().register(UnaryDispatchRegistry())

    @pytest.mark.parametrize("attribute", ["name", "description"])
    def test_the_catalog_identity_is_read_only(self, attribute: str) -> None:
        """A cataloged registry keeps the name it is cataloged under."""
        reg = UnaryDispatchRegistry(name="fixed", description="d")
        with pytest.raises(AttributeError):
            setattr(reg, attribute, "changed")
        assert (reg.name, reg.description) == ("fixed", "d")

    def test_constructor_rejects_positional_args(self) -> None:
        with pytest.raises(TypeError):
            UnaryDispatchRegistry("inference")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Global catalog population on `import probpipe`
# ---------------------------------------------------------------------------

#: Each built-in registry's catalog name and kind.
_BUILTINS = {
    "bijectors": "factory",
    "converters": "dispatch",
    "evaluation_rules": "dispatch",
    "inference": "dispatch",
    "operations": "operation",
}


class TestBuiltinPopulation:
    """The global catalog holds every built-in registry after ``import probpipe``."""

    def test_the_catalog_holds_the_builtin_registries(self) -> None:
        assert registry_catalog.names() == sorted(_BUILTINS)

    @pytest.mark.parametrize(("name", "kind"), sorted(_BUILTINS.items()))
    def test_each_builtin_registry_reports_its_identity(self, name: str, kind: str) -> None:
        registry = registry_catalog[name]
        assert isinstance(registry, SupportsRegistryCataloging)
        assert (registry.name, registry.kind) == (name, kind)
        assert registry.description
        info = next(i for i in registry_catalog.list() if i.name == name)
        assert info.entry_count == len(registry.entry_summaries()) > 0

    @pytest.mark.parametrize(
        ("name", "registry"),
        [
            ("converters", converter_registry),
            ("inference", inference_method_registry),
            ("operations", operation_registry),
        ],
    )
    def test_the_catalog_holds_the_exported_singletons(self, name: str, registry: Any) -> None:
        assert registry_catalog[name] is registry

    @pytest.mark.parametrize("name", ["converters", "evaluation_rules", "inference"])
    def test_a_dispatch_registry_lists_its_methods_in_selection_order(self, name: str) -> None:
        registry = registry_catalog[name]
        assert [s.name for s in registry.entry_summaries()] == registry.list_methods()
        assert all(isinstance(s.exact, bool) for s in registry.entry_summaries())

    def test_inference_lists_its_exact_methods_first(self) -> None:
        """Exactness ranks before priority, so no approximate method precedes an exact one."""
        exactness = [s.exact for s in inference_method_registry.entry_summaries()]
        assert True in exactness and False in exactness
        assert exactness == sorted(exactness, reverse=True)

    def test_inference_describe_lists_opt_in_methods_apart(self) -> None:
        out = registry_catalog.describe("inference")
        opt_in = [s.name for s in inference_method_registry.entry_summaries() if s.is_opt_in_only]
        assert opt_in, "expected at least one opt-in-only inference method"
        for name in opt_in:
            assert out.index("Opt-in only") < out.index(f"  {name}")

    def test_the_evaluation_floors_follow_the_other_rules(self) -> None:
        """The catalog lists the floors last and apart, as selection order ranks them."""
        summaries = registry_catalog["evaluation_rules"].entry_summaries()
        floors = [s.name for s in summaries if s.is_floor]
        assert floors == ["elementwise_sweep", "sampling_lift"]
        assert [s.name for s in summaries][-2:] == floors
        out = registry_catalog.describe("evaluation_rules")
        section = out.index("Floors, after every other entry:")
        assert out.index("empirical_enumeration") < section
        assert all(section < out.index(f"  {name}") for name in floors)
        assert str(-(2**31)) not in out

    @pytest.mark.parametrize("name", ["converters", "evaluation_rules", "inference"])
    def test_each_shipped_dispatch_method_describes_itself(self, name: str) -> None:
        for s in registry_catalog[name].entry_summaries():
            assert s.description, f"{name} method {s.name!r} declares no description"
            assert "\n" not in s.description

    def test_methods_of_one_class_describe_themselves_apart(self) -> None:
        """``blackjax_nuts`` and ``blackjax_hmc`` share a class but not a description."""
        nuts = inference_method_registry.describe_entry("blackjax_nuts")
        hmc = inference_method_registry.describe_entry("blackjax_hmc")
        assert nuts.module_path == hmc.module_path
        assert nuts.description != hmc.description


# ---------------------------------------------------------------------------
# The converters, the operations, and the bijector factory
# ---------------------------------------------------------------------------


class TestConverterEntries:
    def test_each_entry_pairs_source_and_target_types(self) -> None:
        for s in converter_registry.entry_summaries():
            source_types, target_types = s.supported_types
            assert all(isinstance(t, type) for t in (*source_types, *target_types))

    def test_the_shipped_converters_are_cataloged(self) -> None:
        summaries = converter_registry.entry_summaries()
        assert {s.name for s in summaries} >= {"tfp", "empirical", "moment_match", "kde"}

    def test_describe_entry_round_trips(self) -> None:
        [first, *_] = converter_registry.entry_summaries()
        assert converter_registry.describe_entry(first.name) == first

    def test_describe_entry_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown method 'DoesNotExist'"):
            converter_registry.describe_entry("DoesNotExist")


class TestOperationEntries:
    def test_an_operation_summary_is_an_entry_summary(self) -> None:
        summaries = operation_registry.entry_summaries()
        assert summaries
        for s in summaries:
            assert isinstance(s, EntrySummary)
            assert (s.priority, s.exact, s.is_floor) == (None, None, False)

    def test_describe_lists_the_operations_together(self) -> None:
        out = registry_catalog.describe("operations")
        assert "Entries:" in out
        assert "Opt-in only" not in out
        assert re.search(r"^    sample  — ", out, re.M)


class TestBijectorEntries:
    def test_entries_are_unranked_described_and_name_ordered(self) -> None:
        summaries = registry_catalog["bijectors"].entry_summaries()
        assert [s.name for s in summaries] == sorted(s.name for s in summaries)
        for s in summaries:
            assert (s.priority, s.exact) == (None, None)
            assert s.description, f"bijector entry {s.name!r} has no description"
            assert len(s.supported_types) == 1

    @pytest.mark.parametrize("name", ["Boolean", "IntegerInterval", "NonNegativeInteger", "Sphere"])
    def test_an_entry_without_a_bijector_says_so(self, name: str) -> None:
        description = registry_catalog["bijectors"].describe_entry(name).description
        assert description.startswith("No smooth bijector: ")

    def test_an_instance_key_is_named_by_its_repr_and_described(self) -> None:
        from probpipe.core.constraints import _Positive

        with _restored_bijector_registry():
            register_bijector(_Positive(), lambda c: None, description="an instance key")
            entry = registry_catalog["bijectors"].describe_entry(repr(_Positive()))
            assert entry.description == "an instance key"

    def test_register_bijector_rejects_a_non_string_description(self) -> None:
        from probpipe.core.constraints import _Positive

        with pytest.raises(TypeError, match="register_bijector's description must be a str, got 3"):
            register_bijector(_Positive, lambda c: None, description=3)  # type: ignore[arg-type]

    def test_describe_entry_known(self) -> None:
        assert registry_catalog["bijectors"].describe_entry("Real").name == "Real"

    def test_describe_entry_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown bijector entry 'DoesNotExist'"):
            registry_catalog["bijectors"].describe_entry("DoesNotExist")

    def test_a_shared_short_name_is_qualified_for_a_third_party_key(self) -> None:
        """Entry names stay unique: a third-party ``Positive`` is named by its module."""
        from probpipe.core.constraints import _Positive

        class Positive(_Positive):
            pass

        with _restored_bijector_registry():
            register_bijector(Positive, lambda c: None, description="third party")
            summaries = registry_catalog["bijectors"].entry_summaries()
            names = [s.name for s in summaries]
            assert len(names) == len(set(names))
            qualified = f"{Positive.__module__}.{Positive.__qualname__}"
            assert registry_catalog["bijectors"].describe_entry(qualified).description == (
                "third party"
            )
            assert registry_catalog["bijectors"].describe_entry("Positive").description == (
                "exp, onto the positive half-line."
            )

    def test_registering_a_key_again_replaces_its_description(self) -> None:
        from probpipe.core.constraints import _Positive

        with _restored_bijector_registry():
            register_bijector(_Positive, lambda c: None, description="replaced")
            entry = registry_catalog["bijectors"].describe_entry("Positive")
            assert entry.description == "replaced"


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

    def test_entry_summary_is_floor_defaults_to_false(self) -> None:
        assert EntrySummary(name="x", priority=1).is_floor is False


# ---------------------------------------------------------------------------
# entry_summaries / describe_entry on a dispatch registry directly
# ---------------------------------------------------------------------------


class TestDispatchRegistryEntrySummaries:
    """``entry_summaries`` and ``describe_entry`` on a ``UnaryDispatchRegistry``.

    Covers the ``EntrySummary`` field mapping, selection order, priority
    overrides, the registration snapshot, and the round trip through
    ``describe_entry``.
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
        assert s.is_floor is False
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

    def test_entry_summaries_list_floors_after_the_other_methods(self) -> None:
        reg = UnaryDispatchRegistry()
        reg.register(_StubMethodWithDescription("floor", 100, exact=True, is_floor=True))
        reg.register(_StubMethodWithDescription("approx", 1))
        assert [(s.name, s.is_floor) for s in reg.entry_summaries()] == [
            ("approx", False),
            ("floor", True),
        ]

    def test_description_may_differ_between_instances(self) -> None:
        first = _StubMethodWithDescription("one", 10)
        first._description = "first"
        second = _StubMethodWithDescription("two", 10)
        second._description = "second"
        reg = UnaryDispatchRegistry()
        reg.register(first)
        reg.register(second)
        assert [s.description for s in reg.entry_summaries()] == ["first", "second"]

    def test_opt_in_move_is_reported(self) -> None:
        reg = self._registry_with(("m", 10))
        with pytest.warns(UserWarning, match="now runs only when requested"):
            reg.set_priorities(m=None)
        assert reg.describe_entry("m").is_opt_in_only

    def test_summary_reads_the_registration(self) -> None:
        """A method mutated after registration changes nothing the catalog reports."""
        method = _StubMethodWithDescription("m", 10)
        reg = UnaryDispatchRegistry()
        reg.register(method)
        method._priority = 99
        method._description = "changed"
        s = reg.describe_entry("m")
        assert s.priority == 10
        assert s.description == "stub method for catalog tests"

    def test_non_string_description_rejected(self) -> None:
        method = _StubMethodWithDescription("m", 10)
        method._description = 3
        reg = UnaryDispatchRegistry()
        with pytest.raises(
            TypeError, match=r"^method 'm' must declare description as a str, got 3$"
        ):
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
        with pytest.raises(KeyError, match=r"unknown method 'missing'.*'known'"):
            reg.describe_entry("missing")

    def test_entry_summaries_on_empty_registry(self) -> None:
        reg = UnaryDispatchRegistry()
        assert reg.entry_summaries() == []


# ---------------------------------------------------------------------------
# Agreement with design II.7
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "scripts" / "design" / "design_blocks.py"

#: The modules a name II.7 declares is looked up in, first match winning.
_MODULES = (_catalog, _dispatch)


def _tool():
    if "design_blocks" in sys.modules:
        return sys.modules["design_blocks"]
    spec = importlib.util.spec_from_file_location("design_blocks", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["design_blocks"] = module
    spec.loader.exec_module(module)
    return module


def _resolve(name: str) -> Any:
    for module in _MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _declarations() -> list:
    return _tool().declarations(("II.7",)) if _TOOL.exists() else []


def _declared_name(declaration: Any) -> str:
    member = getattr(declaration, "member", None)
    return member.name if member is not None else declaration.name


@pytest.mark.skipif(
    not (_ROOT / "design").is_dir() or not _TOOL.exists(),
    reason="the design reference is not checked out",
)
class TestDesignAgreement:
    @pytest.mark.parametrize(
        "declaration",
        [pytest.param(d, id=f"{d.section}-{_declared_name(d)}") for d in _declarations()],
    )
    def test_the_declaration_matches_the_implementation(self, declaration: Any) -> None:
        findings = _tool().check([declaration], _resolve)
        assert not findings, "; ".join(f"{f.name} {f.problem}" for f in findings)

    def test_every_public_name_of_the_catalog_is_declared(self) -> None:
        declared = {_declared_name(d) for d in _declarations()}
        assert set(_catalog.__all__) <= declared, set(_catalog.__all__) - declared

    @pytest.mark.parametrize("record_cls", [EntrySummary, RegistryInfo])
    def test_record_fields_match_in_order(self, record_cls: type) -> None:
        [declaration] = [d for d in _declarations() if _declared_name(d) == record_cls.__name__]
        declared = [(f.name, f.annotation.replace(" ", "")) for f in declaration.fields]
        implemented = [(f.name, str(f.type).replace(" ", "")) for f in fields(record_cls)]
        assert declared == implemented
