"""The distribution catalog agrees with the code blocks of the design reference.

Every class, member, field, and function that Part VII declares is checked
against the ``families/`` module the package structure places it in, and, for a
class that is still defined elsewhere, against its current module: the class
exists and derives from the declared bases, and each member takes the declared
parameters, in order, with the declared kinds and defaults.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

import probpipe
import probpipe.families as families

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "scripts" / "design" / "design_blocks.py"
_CATALOG = _ROOT / "design" / "07-distribution-catalog.md"
_STRUCTURE = _ROOT / "design" / "package-structure.md"

pytestmark = pytest.mark.skipif(
    not (_ROOT / "design").is_dir() or not _TOOL.exists(),
    reason="the design reference is not checked out",
)

#: The sections of the catalog, in the order the reference gives them.
_SECTIONS = ("VII.1", "VII.2", "VII.3", "VII.4", "VII.5", "VII.6", "VII.7", "VII.8", "VII.9")

#: The ``families/`` modules that realize each section, by the package structure.
_SECTION_MODULES = {
    "VII.1": ("_backend", "_continuous", "_discrete", "_multivariate"),
    "VII.2": ("_resampling",),
    "VII.3": ("_mixture",),
    "VII.4": ("_transformed",),
    "VII.5": ("_random_functions",),
    "VII.6": ("_gaussian",),
    "VII.7": (),
    "VII.8": ("_conditional",),
    "VII.9": ("_programs",),
}

#: Where each declared class that ``families/`` does not define is today.
_CURRENT_MODULES = {
    "TFPDistribution": "probpipe.distributions._tfp_base",
    "Normal": "probpipe.distributions.continuous",
    "RandomFunction": "probpipe.core._random_functions",
    "RandomMeasure": "probpipe.core._random_measures",
    "GaussianRandomFunction": "probpipe.distributions.gaussian_random_function",
    "LinearBasisFunction": "probpipe.distributions.gaussian_random_function",
}

#: Declarations of these sections that the distribution layer owns and checks.
_OTHER_PACKAGES = frozenset({"EmpiricalDistribution"})

#: Declarations the implementation does not match yet, with the change each awaits.
_PENDING = {
    "TFPDistribution": "the adapter takes the wrapped backend distribution as backend_dist",
    "KDEDistribution": "the declaration check compares a class-valued default by the class's name",
    "GaussianRandomFunction": (
        "predict_covariance and __call__ take the stacked inputs only, without the joint flags"
    ),
    "LinearBasisFunction": (
        "the basis-function model takes basis and weights, with output_spec and event_spec"
    ),
}

#: New declarations the catalog does not define yet; each has a pending case above.
_UNBUILT: frozenset[str] = frozenset()


def _tool():
    if "design_blocks" in sys.modules:
        return sys.modules["design_blocks"]
    spec = importlib.util.spec_from_file_location("design_blocks", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["design_blocks"] = module
    spec.loader.exec_module(module)
    return module


def _declared_name(declaration) -> str:
    member = getattr(declaration, "member", None)
    return member.name if member is not None else declaration.name


def _resolver(section: str):
    """Resolve a declared name in the section's ``families/`` modules, then in its current one."""
    modules = [
        importlib.import_module(f"probpipe.families.{module}")
        for module in _SECTION_MODULES[section]
    ]

    def resolve(name: str):
        for module in modules:
            if name in getattr(module, "__all__", ()):
                return getattr(module, name)
        current = _CURRENT_MODULES.get(name)
        if current is not None:
            return getattr(importlib.import_module(current), name, None)
        return None

    return resolve


def _cases() -> list:
    if not _TOOL.exists():
        return []
    cases = []
    for declaration in _tool().declarations(_SECTIONS):
        name = _declared_name(declaration)
        if name in _OTHER_PACKAGES:
            continue
        marks = ()
        if name in _PENDING:
            marks = pytest.mark.pending(reason=_PENDING[name], raises=AssertionError)
        cases.append(pytest.param(declaration, id=f"{declaration.section}-{name}", marks=marks))
    return cases


def _new_declarations() -> dict[str, str]:
    """Each declared name that ``families/`` defines, with its section."""
    return {
        _declared_name(declaration): declaration.section
        for declaration in _tool().declarations(_SECTIONS)
        if _declared_name(declaration) not in _CURRENT_MODULES
        and _declared_name(declaration) not in _OTHER_PACKAGES
        and _declared_name(declaration) not in _UNBUILT
    }


class TestDeclarationsAreImplemented:
    @pytest.mark.parametrize("declaration", _cases())
    def test_the_declaration_matches_the_implementation(self, declaration):
        findings = _tool().check([declaration], _resolver(declaration.section))
        assert not findings, "; ".join(f"{f.name} {f.problem}" for f in findings)

    def test_each_unbuilt_declaration_is_defined(self):
        programs = importlib.import_module("probpipe.families._programs")
        for name in _UNBUILT:
            assert name in programs.__all__, name

    def test_each_new_declaration_is_defined_in_its_sections_module(self):
        for name, section in _new_declarations().items():
            owners = [
                module
                for module in _SECTION_MODULES[section]
                if name in importlib.import_module(f"probpipe.families.{module}").__all__
            ]
            assert owners, f"{name} ({section}) is not defined in {_SECTION_MODULES[section]}"
            defined = getattr(importlib.import_module(f"probpipe.families.{owners[0]}"), name)
            assert defined.__module__ == f"probpipe.families.{owners[0]}", name

    def test_a_class_defined_elsewhere_is_not_defined_again(self):
        for name in _CURRENT_MODULES:
            assert not hasattr(families, name), name
            for modules in _SECTION_MODULES.values():
                for module in modules:
                    assert (
                        name not in importlib.import_module(f"probpipe.families.{module}").__all__
                    ), (module, name)


class TestTheParametricFamilies:
    """VII.1: every family is a thin constructor taking its name first and an event_spec."""

    @staticmethod
    def _listed_families() -> list[str]:
        text = _CATALOG.read_text()
        section = text[text.index("## VII.1") : text.index("## VII.2")]
        listing = section[section.index("continuous (") : section.index("Each family derives")]
        return re.findall(r"`([A-Z]\w+)`", listing)

    def test_the_listing_names_every_family(self):
        assert len(self._listed_families()) == 24

    @pytest.mark.parametrize(
        "name",
        [
            "Normal",
            "Beta",
            "Gamma",
            "InverseGamma",
            "Exponential",
            "LogNormal",
            "StudentT",
            "Uniform",
            "Cauchy",
            "Laplace",
            "HalfNormal",
            "HalfCauchy",
            "Pareto",
            "TruncatedNormal",
            "Bernoulli",
            "Binomial",
            "Poisson",
            "Categorical",
            "NegativeBinomial",
            "MultivariateNormal",
            "Dirichlet",
            "Multinomial",
            "Wishart",
            "VonMisesFisher",
        ],
    )
    def test_a_family_takes_its_name_first_and_a_keyword_event_spec(self, name):
        assert name in self._listed_families()
        family = getattr(probpipe.distributions, name)
        assert issubclass(family, probpipe.distributions.TFPDistribution)
        parameters = list(inspect.signature(family.__init__).parameters.values())[1:]
        assert parameters[0].name == "name"
        assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        event_spec = inspect.signature(family.__init__).parameters["event_spec"]
        assert event_spec.kind is inspect.Parameter.KEYWORD_ONLY
        assert event_spec.default is None


class TestThePackage:
    def test_every_listed_module_exists(self):
        """The package structure's ``families/`` modules are the package's modules."""
        tree = _STRUCTURE.read_text()
        block = tree[tree.index("├── families/") : tree.index("├── designs/")]
        listed = set(re.findall(r"(_\w+\.py)", block))
        package = Path(families.__file__).parent
        present = {path.name for path in package.glob("_*.py") if path.name != "__init__.py"}
        assert present == listed

    def test_every_export_is_a_part_vii_declaration(self):
        """A public name the catalog does not declare is drift."""
        undeclared = set(families.__all__) - set(_new_declarations())
        assert not undeclared, sorted(undeclared)

    def test_every_new_declaration_is_exported(self):
        assert set(_new_declarations()) == set(families.__all__)

    def test_no_export_clashes_with_the_top_level_namespace(self):
        """A name both namespaces export is one object, which the top level re-exports."""
        shared = set(families.__all__) & set(probpipe.__all__)
        clashes = {
            name for name in shared if getattr(families, name) is not getattr(probpipe, name)
        }
        assert not clashes, sorted(clashes)
