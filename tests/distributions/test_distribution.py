"""Tests for the distribution base classes, ``Distribution`` and ``DistributionSpec``."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Bernoulli,
    Beta,
    Binomial,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    Categorical,
    Cauchy,
    Dirichlet,
    Distribution,
    DistributionSpec,
    EmpiricalDistribution,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    KDEDistribution,
    Laplace,
    LogNormal,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    NumericDistribution,
    NumericRecordBatch,
    NumericRecordSpec,
    OpaqueBatch,
    OutputSpec,
    Pareto,
    Poisson,
    RandomFunction,
    RandomMeasure,
    RecordBatch,
    RecordSpec,
    StudentT,
    TruncatedNormal,
    Uniform,
    VonMisesFisher,
    Wishart,
    boolean,
    expectation,
    greater_than,
    integer_interval,
    interval,
    non_negative,
    non_negative_integer,
    positive,
    positive_definite,
    real,
    sample,
    simplex,
    sphere,
    unit_interval,
)
from probpipe.core._batch import BatchSpec
from probpipe.core._opaque import OpaqueSpec
from probpipe.core._specs import NumericArraySpec
from probpipe.core.provenance import Provenance, provenance_ancestors
from probpipe.distributions._capabilities import SupportsMean
from probpipe.families import BijectorTransformedDistribution
from probpipe.functions._normalization import DISTRIBUTION_HINT_PROTOCOLS


def _make_transformed():
    import tensorflow_probability.substrates.jax.bijectors as tfb

    return BijectorTransformedDistribution(
        "transformed",
        Normal(loc=0.0, scale=1.0, name="x"),
        tfb.Exp(),
    )


# Distribution-instance factories used by ``TestNoBatchShape``. Mirrors
# the ``DISTRIBUTIONS`` table in ``tests/core/test_iteration_protocol.py`` but with
# a smaller set covering the canonical TFP-backed scalars + the most
# distinct subclasses (BijectorTransformedDistribution / KDEDistribution /
# EmpiricalDistribution).
_NO_BATCH_SHAPE_DISTS = [
    pytest.param(lambda: Normal(loc=0.0, scale=1.0, name="x"), id="Normal"),
    pytest.param(lambda: Gamma(concentration=3.0, rate=1.0, name="g"), id="Gamma"),
    pytest.param(
        lambda: MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z"),
        id="MultivariateNormal",
    ),
    pytest.param(_make_transformed, id="BijectorTransformedDistribution"),
    pytest.param(
        lambda: KDEDistribution("kde", jnp.arange(60.0).reshape(20, 3)),
        id="KDEDistribution",
    ),
    pytest.param(
        lambda: EmpiricalDistribution("x", jnp.zeros((10, 3))),
        id="EmpiricalDistribution",
    ),
]


class TestWithNameBasics:
    """Distribution.with_name() returns a new object with a new name."""

    def test_returns_new_object(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("y")
        assert n is not n2
        assert n.name == "x"  # original unchanged
        assert n2.name == "y"

    def test_is_same_type(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert type(n.with_name("y")) is type(n)

    def test_is_shallow_copy(self):
        """Underlying parameters are shared (not deep-copied)."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("y")
        assert n2._loc is n._loc  # shared array
        assert n2._scale is n._scale


class TestWithNameProvenance:
    """with_name() attaches a 'with_name' Provenance pointing to the original."""

    def test_provenance_operation(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("y")
        assert n2.provenance is not None
        assert n2.provenance.operation == "with_name"

    def test_provenance_parents(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("y")
        assert len(n2.provenance.parents) == 1
        assert n2.provenance.parents[0].name == "x"

    def test_provenance_metadata(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("y")
        assert n2.provenance.metadata["old_name"] == "x"
        assert n2.provenance.metadata["new_name"] == "y"

    def test_rename_chain_preserves_ancestry(self, full_provenance_mode):
        """a.with_name("b").with_name("c") keeps a in the ancestor DAG."""
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = a.with_name("b")
        c = b.with_name("c")
        ancestors = provenance_ancestors(c)
        assert any(anc.parent is a for anc in ancestors)
        assert any(anc.parent is b for anc in ancestors)

    def test_original_provenance_not_mutated(self):
        """Renaming does not alter the original's source."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        n.with_provenance(Provenance("construction", parents=()))
        n.with_name("y")
        assert n.provenance.operation == "construction"


class TestWithNameSampling:
    """Renamed copies behave identically under sampling/log_prob."""

    def test_sample_statistics_match(self):
        n = Normal(loc=2.0, scale=0.5, name="x")
        n2 = n.with_name("mu")
        key = jax.random.PRNGKey(0)
        s1 = n._sample(key, (2000,))
        s2 = n2._sample(key, (2000,))
        # Same key -> identical samples
        np.testing.assert_allclose(np.asarray(s1), np.asarray(s2), atol=1e-6)

    def test_log_prob_matches(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("z")
        x = jnp.asarray(1.23)
        np.testing.assert_allclose(
            float(n._log_prob(x)),
            float(n2._log_prob(x)),
            atol=1e-6,
        )

    def test_event_shape_matches(self):
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="a")
        renamed = mvn.with_name("b")
        assert renamed.event_shape == mvn.event_shape


class TestWithNameRecordSpec:
    """with_name() changes the name and keeps the event component (III.7)."""

    def test_template_field_stays_the_component(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.with_name("growth_rate")
        assert n2.name == "growth_rate"
        assert n2.event_spec is n.event_spec
        assert tuple(n2.event_spec.components) == ("x",)

    def test_template_shape_preserved(self):
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="a")
        b = mvn.with_name("b")
        assert tuple(b.event_spec.components) == ("a",)
        assert b.event_spec.spec.shape == mvn.event_spec.spec.shape == (3,)


class TestNoBatchShape:
    """``Distribution`` has no ``batch_shape`` attribute. Pins the
    absence across the public Distribution family so a future
    subclass can't silently reintroduce it as a defensive default.
    Container types such as ``DistributionBatch`` and ``RecordBatch`` keep
    their own ``batch_shape``, which is a different concept.
    """

    def test_no_batch_shape_on_base_class(self):
        from probpipe import Distribution

        assert not hasattr(Distribution, "batch_shape")

    @pytest.mark.parametrize(
        "make_dist",
        _NO_BATCH_SHAPE_DISTS,
    )
    def test_no_batch_shape_on_instance(self, make_dist):
        dist = make_dist()
        assert not hasattr(dist, "batch_shape"), (
            f"{type(dist).__name__} unexpectedly exposes a batch_shape attribute."
        )


class TestAnnotationsDiagnosticsAccessor:
    def test_annotations_defaults_to_none(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")
        assert dist.annotations is None
        assert dist.diagnostics is None

    def test_diagnostics_none_when_annotations_has_no_diagnostics_group(self):
        import xarray as xr

        dist = Normal(loc=0.0, scale=1.0, name="x")
        dist._annotations = xr.DataTree.from_dict({"arviz": xr.Dataset()})
        assert dist.annotations is dist._annotations
        assert dist.diagnostics is None

    def test_diagnostics_none_when_annotations_has_no_children_attr(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")
        dist._annotations = object()

        assert dist.diagnostics is None

    def test_diagnostics_returns_view_for_diagnostics_group(self):
        import xarray as xr

        from probpipe.diagnostics.views import DiagnosticsView

        dist = Normal(loc=0.0, scale=1.0, name="x")
        dist._annotations = xr.DataTree.from_dict(
            {"diagnostics": xr.Dataset(attrs={"warnings": "[]"})}
        )
        view = dist.diagnostics
        assert view is not None
        assert isinstance(view, DiagnosticsView)
        assert view.warnings == []


class TestDistributionRepr:
    def test_base_repr_includes_class_and_name(self):
        from probpipe import Distribution

        class _NamedDist(Distribution):
            def __init__(self):
                super().__init__("x", OpaqueSpec())

        assert repr(_NamedDist()) == "_NamedDist(name='x')"


class TestConstructorNameCheck:
    """``Distribution.__init__`` rejects a name that is not a non-empty string."""

    @pytest.mark.parametrize("name", ["", 123, None])
    def test_invalid_name_raises(self, name):
        from probpipe import Distribution

        class _Dist(Distribution):
            def __init__(self, name):
                super().__init__(name, OpaqueSpec())

        with pytest.raises(TypeError, match="requires a non-empty name"):
            _Dist(name)


class TestMetaclassEnforcement:
    """The ``_TrackedTermMeta`` metaclass enforces a non-empty ``name``
    on every Distribution subclass instance, even when the subclass
    bypasses ``super().__init__``.
    """

    def test_subclass_without_name_raises_at_construction(self):
        """A subclass whose ``__init__`` doesn't set ``_name`` cannot be
        constructed — the metaclass post-init check fires before the
        instance escapes."""
        from probpipe import Distribution

        class _NoNameDist(Distribution):
            def __init__(self):
                # Deliberately omit calling super().__init__ and
                # setting self._name.
                pass

        with pytest.raises(TypeError, match="non-empty name"):
            _NoNameDist()

    def test_subclass_with_empty_string_name_raises(self):
        """An empty-string ``_name`` is also rejected — the check
        insists on a truthy string."""
        from probpipe import Distribution

        class _EmptyNameDist(Distribution):
            def __init__(self):
                self._name = ""

        with pytest.raises(TypeError, match="non-empty name"):
            _EmptyNameDist()

    def test_subclass_with_non_string_name_raises(self):
        """The metaclass requires the final ``_name`` to be a string."""
        from probpipe import Distribution

        class _NonStringNameDist(Distribution):
            def __init__(self):
                self._name = 123

        with pytest.raises(TypeError, match="non-empty name"):
            _NonStringNameDist()

    def test_subclass_setting_name_directly_succeeds(self):
        """Bypassing ``super().__init__`` is fine as long as
        ``self._name`` ends up set to a non-empty string and the event is
        declared."""
        from probpipe import Distribution

        class _DirectNameDist(Distribution):
            def __init__(self):
                # Skip super().__init__ deliberately.
                self._name = "direct"
                self._init_declaration(OpaqueSpec())

        dist = _DirectNameDist()
        assert dist.name == "direct"
        assert dist.event_spec == OutputSpec(direct=OpaqueSpec())

    @pytest.mark.parametrize("base", [Distribution, NumericDistribution])
    def test_a_law_that_leaves_its_event_undeclared_raises(self, base):
        """Construction checks the declaration after ``__init__``, naming
        the class, whichever base it bypasses."""

        class _NoDeclaration(base):
            def __init__(self):
                self._name = "undeclared"

        with pytest.raises(TypeError, match=r"_NoDeclaration\.__init__ left the event undeclared"):
            _NoDeclaration()


class TestWithNameTemplateRoundtrip:
    """``with_name`` keeps a declared law's event component; explicit and
    multi-field templates are preserved.
    """

    def test_with_name_keeps_the_declared_component(self):
        """A single-array law captures its component at construction, so
        the clone draws under the original component and the original is
        untouched."""
        from probpipe import Normal

        original = Normal(loc=0.0, scale=1.0, name="x")
        clone = original.with_name("y")
        assert clone.name == "y"
        assert tuple(clone.event_spec.components) == ("x",)
        assert original.name == "x"
        assert tuple(original.event_spec.components) == ("x",)

    def test_with_name_preserves_multi_field_template(self):
        """A multi-field joint's components are independent of the
        distribution's name, so renaming leaves them."""
        import jax.numpy as jnp

        jg = MultivariateNormal("x", jnp.zeros(1), cov=jnp.eye(1)) * MultivariateNormal(
            "y", jnp.zeros(1), cov=jnp.eye(1)
        )
        original_fields = tuple(jg.event_spec.components)
        clone = jg.with_name("renamed_jg")
        assert tuple(clone.event_spec.components) == original_fields == ("x", "y")

    def test_with_name_preserves_a_non_numeric_declaration(self):
        """An empirical law over records with an opaque field declares its atoms'
        record, not the distribution's name, so renaming leaves the declaration
        intact."""
        import numpy as np

        rows = RecordBatch(
            "rows",
            {"labels": np.array(["a", "b", "c"], dtype=object), "ids": np.array([0, 1, 2])},
            "row",
            element_spec=RecordSpec(labels=None, ids=()),
        )
        law = EmpiricalDistribution("rows", rows)
        original_fields = tuple(law.event_spec.components)
        clone = law.with_name("renamed")
        assert tuple(clone.event_spec.components) == original_fields == ("labels", "ids")


class TestDistributionSpecIsValid:
    def test_matching_distribution_valid(self):
        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert DistributionSpec(dist.event_spec).is_valid(dist)

    def test_packaging_mismatch_invalid(self):
        # A whole term x and a one-field record exposing x are different draws.
        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert not DistributionSpec(RecordSpec(x=())).is_valid(dist)

    def test_template_mismatch_invalid(self):
        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert not DistributionSpec(event_spec=RecordSpec(y=())).is_valid(dist)

    def test_non_distribution_invalid(self):
        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(42)
        assert not spec.is_valid(RecordSpec(x=()))

    def test_an_opaque_law_does_not_match_a_record_declaration(self):
        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_DeclaredLaw("d", OpaqueSpec()))


class TestNoTypeParameter:
    """The distribution kinds and their capabilities are not generic.

    A draw's type follows from the event declaration, so a static parameter
    could record only the declaration's kind.
    """

    @pytest.mark.parametrize(
        "cls",
        [
            Distribution,
            EmpiricalDistribution,
            BootstrapReplicateDistribution,
            RandomFunction,
            RandomMeasure,
            *DISTRIBUTION_HINT_PROTOCOLS,
        ],
        ids=lambda cls: cls.__name__,
    )
    def test_class_takes_no_type_parameter(self, cls):
        assert cls.__type_params__ == ()
        with pytest.raises(TypeError):
            cls[Any]


def _public_distribution_classes() -> list[type]:
    """Every public distribution class in the package, found by a subclass walk."""
    import probpipe

    for module in pkgutil.walk_packages(probpipe.__path__, "probpipe."):
        try:
            importlib.import_module(module.name)
        except ImportError as exc:
            # An optional backend that is not installed is skipped.
            if (exc.name or "").partition(".")[0] == "probpipe":
                raise
    found: list[type] = []
    stack = [Distribution]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub.__module__.startswith("probpipe") and sub not in found:
                found.append(sub)
                stack.append(sub)
    return [cls for cls in found if not cls.__name__.startswith("_")]


# The classes the design retires keep a keyword name until they are removed.
_RETIRING = {
    "ApproximateDistribution",
}
_PUBLIC_CLASSES = _public_distribution_classes()

# Laws that indexing constructs from a parent, so no caller names them.
_CONSTRUCTED_BY_INDEXING = {"FieldView"}


class TestNameFirstSignature:
    """Every constructor the design keeps takes ``name`` first, required."""

    @pytest.mark.parametrize(
        "cls",
        [
            cls
            for cls in _PUBLIC_CLASSES
            if cls.__name__ not in _RETIRING | _CONSTRUCTED_BY_INDEXING
        ],
        ids=lambda cls: cls.__name__,
    )
    def test_name_is_the_required_first_parameter(self, cls):
        first = next(iter(inspect.signature(cls.__init__).parameters.values()))
        if first.name == "self":
            first = list(inspect.signature(cls.__init__).parameters.values())[1]
        assert first.name == "name"
        assert first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert first.default is inspect.Parameter.empty

    def test_retiring_list_names_existing_classes(self):
        assert {cls.__name__ for cls in _PUBLIC_CLASSES} >= _RETIRING


class TestNameBinding:
    def test_positional_name_binds(self):
        assert Normal("x", 0.0, 1.0).name == "x"

    def test_keyword_name_binds(self):
        assert Normal(loc=0.0, scale=1.0, name="x").name == "x"

    def test_name_given_both_ways_raises(self):
        with pytest.raises(TypeError, match="multiple values for argument 'name'"):
            Normal("x", 0.0, 1.0, name="y")

    # The empirical law chooses its capabilities from its atoms, which it reads by
    # keyword as well as by position.
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda s: EmpiricalDistribution(name="x", atoms=s), id="all-keywords"),
            pytest.param(lambda s: EmpiricalDistribution("x", atoms=s), id="atoms-keyword"),
        ],
    )
    def test_keyword_atoms_reach_the_numeric_capabilities(self, make):
        law = make(jnp.arange(4.0))
        assert isinstance(law, SupportsMean)
        assert law.name == "x"

    def test_a_keyword_source_reaches_the_bootstrap(self):
        source = EmpiricalDistribution("r", jnp.arange(4.0))
        law = BootstrapReplicateDistribution("b", source=source)
        assert law.replicate_size == 4
        assert law.name == "b"


class TestDerivedNames:
    """A law that ``expectation`` constructs is named for the operation."""

    @pytest.mark.parametrize(
        ("make_operand", "f"),
        [
            pytest.param(lambda: Normal("law", 0.0, 1.0), lambda x: x, id="monte-carlo"),
            pytest.param(
                lambda: EmpiricalDistribution(
                    "law", OpaqueBatch("labels", ["a", "b", "c", "d"], "atom")
                ),
                lambda x: jnp.asarray(1.0),
                id="generic-empirical",
            ),
            pytest.param(
                lambda: EmpiricalDistribution("law", jnp.arange(10.0)),
                lambda x: x,
                id="array-empirical",
            ),
            pytest.param(
                lambda: BootstrapReplicateDistribution(
                    "law", EmpiricalDistribution("data", jnp.arange(5.0))
                ),
                jnp.mean,
                id="bootstrap-replicate",
            ),
            pytest.param(
                lambda: (Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0))["a"],
                lambda x: x,
                id="field-view",
            ),
        ],
    )
    def test_expectation_bootstrap_is_named_for_the_operation(self, make_operand, f):
        result = expectation(make_operand(), f, num_evaluations=3, key=jax.random.PRNGKey(0))
        assert result.name == "expectation"


class TestPublicImportPaths:
    """Both public namespaces export the distribution base classes."""

    def test_distributions_package_exports_the_base(self):
        import probpipe
        import probpipe.distributions as distributions

        assert distributions.Distribution is probpipe.Distribution
        assert distributions.DistributionSpec is probpipe.DistributionSpec
        assert {"Distribution", "DistributionSpec"} <= set(distributions.__all__)

    def test_core_distribution_module_is_removed(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("probpipe.core." + "distribution")


class _DeclaredLaw(Distribution):
    """A test-only law that declares whatever event it is given."""

    def __init__(self, name, event_spec):
        super().__init__(name, event_spec)


class TestEventDeclaration:
    """A law stores one declaration, completed from what its constructor supplies."""

    def test_a_bare_array_spec_is_a_whole_term_under_the_name(self):
        law = _DeclaredLaw("x", NumericArraySpec((3,)))
        assert law.event_spec == OutputSpec(x=NumericArraySpec((3,)))
        assert law.event_spec is law.spec.event_spec

    def test_a_record_spec_exposes_its_fields_even_with_one(self):
        one = _DeclaredLaw("law", RecordSpec(x=()))
        two = _DeclaredLaw("law", RecordSpec(x=(), y=(2,)))
        assert one.event_spec == OutputSpec(RecordSpec(x=()))
        assert tuple(two.event_spec.components) == ("x", "y")

    def test_an_output_spec_is_stored_as_given(self):
        declaration = OutputSpec(beta=NumericArraySpec((2,)))
        assert _DeclaredLaw("law", declaration).event_spec is declaration

    def test_a_type_hole_raises(self):
        with pytest.raises(ValueError, match="type hole"):
            _DeclaredLaw("x", OutputSpec(x=None))

    def test_a_value_that_is_not_a_spec_raises(self):
        with pytest.raises(TypeError, match="must be an OutputSpec or a TermSpec"):
            _DeclaredLaw("x", (3,))

    def test_with_name_keeps_the_component(self):
        law = _DeclaredLaw("x", NumericArraySpec(()))
        renamed = law.with_name("y")
        assert renamed.name == "y"
        assert renamed.event_spec == law.event_spec
        # A label need not be a component name at all.
        assert list(law.with_name("a-b").event_spec.components) == ["x"]

    @pytest.mark.parametrize("name", ["my-param", "class", "post-1", "product(a,b)"])
    def test_a_whole_term_component_is_any_field_name(self, name):
        assert list(_DeclaredLaw(name, NumericArraySpec(())).event_spec.components) == [name]

    def test_a_label_with_a_slash_cannot_name_a_whole_term(self):
        with pytest.raises(ValueError, match="component names must be non-empty"):
            _DeclaredLaw("a/b", NumericArraySpec(()))


class TestComponentAccess:
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: _DeclaredLaw("x", NumericArraySpec(())), id="declared"),
            pytest.param(lambda: Normal("x", 0.0, 1.0), id="family"),
        ],
    )
    def test_a_whole_term_is_itself_under_its_component(self, make):
        law = make()
        assert law["x"] is law
        with pytest.raises(KeyError):
            law["y"]

    def test_a_tuple_selects_its_paths_as_an_exposed_record(self):
        from probpipe.distributions import FieldView

        law = _DeclaredLaw("x", NumericArraySpec(()))
        selection = law[("x",)]
        assert isinstance(selection, FieldView)
        assert selection.parent is law
        assert selection.event_spec == OutputSpec(RecordSpec(x=NumericArraySpec(())))

    def test_the_component_addresses_a_renamed_law(self):
        renamed = _DeclaredLaw("x", NumericArraySpec(())).with_name("y")
        assert renamed["x"] is renamed
        with pytest.raises(KeyError):
            renamed["y"]

    def test_a_joint_field_is_a_view_declaring_the_field(self):
        product = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        assert product["a"].event_spec.spec == product.event_spec.spec["a"]

    def test_indexing_does_not_make_a_law_iterable(self):
        with pytest.raises(TypeError):
            iter(_DeclaredLaw("x", NumericArraySpec(())))


class TestNumericMembership:
    def test_membership_follows_the_declaration(self):
        assert isinstance(_DeclaredLaw("x", NumericArraySpec(())), NumericDistribution)
        assert not isinstance(_DeclaredLaw("x", OpaqueSpec()), NumericDistribution)

    def test_a_class_claiming_the_marker_must_declare_a_numeric_event(self):
        class _Claims(NumericDistribution):
            def __init__(self, name, event_spec):
                super().__init__(name, event_spec)

        assert issubclass(_Claims, NumericDistribution)
        assert isinstance(_Claims("x", NumericArraySpec(())), NumericDistribution)
        with pytest.raises(TypeError, match="must declare a numeric event"):
            _Claims("x", OpaqueSpec())

    def test_the_claim_checked_is_the_constructed_class(self):
        # A factory __new__ may construct a subclass that claims the marker.
        class _Factory(Distribution):
            def __new__(cls, *args, **kwargs):
                return object.__new__(_Claiming if cls is _Factory else cls)

            def __init__(self, name, event_spec):
                super().__init__(name, event_spec)

        class _Claiming(_Factory, NumericDistribution):
            pass

        assert type(_Factory("x", NumericArraySpec(()))) is _Claiming
        with pytest.raises(TypeError, match="_Claiming inherits NumericDistribution"):
            _Factory("x", OpaqueSpec())


class TestSchemaViews:
    def test_event_shape_is_the_declared_array_shape(self):
        assert _DeclaredLaw("x", NumericArraySpec((3, 2))).event_shape == (3, 2)

    def test_event_shape_is_undefined_for_a_record_draw(self):
        law = _DeclaredLaw("law", RecordSpec(x=()))
        with pytest.raises(AttributeError, match="does not draw a single array"):
            _ = law.event_shape
        assert not hasattr(law, "event_shape")
        assert getattr(law, "event_shape", None) is None

    def test_event_shape_needs_bound_dimensions(self):
        with pytest.raises(ValueError, match="unbound dimensions"):
            _ = _DeclaredLaw("x", NumericArraySpec(("n",))).event_shape

    def test_dtypes_and_supports_are_keyed_by_leaf_path(self):
        from probpipe import positive, real

        law = _DeclaredLaw(
            "law",
            RecordSpec(
                a=NumericArraySpec((), dtype="float32", support=positive),
                b=RecordSpec(c=NumericArraySpec((2,), dtype="int32", support=real)),
            ),
        )
        assert law.dtypes == {"a": np.dtype("float32"), "b/c": np.dtype("int32")}
        assert law.supports == {"a": positive, "b/c": real}
        assert law.dtype is None
        assert law.support is None

    def test_dtype_and_support_are_shared_by_every_leaf(self):
        from probpipe import positive

        law = _DeclaredLaw(
            "law",
            RecordSpec(
                a=NumericArraySpec((), dtype="float32", support=positive),
                b=RecordSpec(c=NumericArraySpec((2,), dtype="float32", support=positive)),
            ),
        )
        assert law.dtype == np.dtype("float32")
        assert law.support == positive

    def test_support_under_jit_reads_no_traced_value(self):
        from probpipe import interval

        seen = []

        def build(bound):
            law = _DeclaredLaw(
                "law",
                RecordSpec(
                    a=NumericArraySpec((), support=interval(0.0, bound)),
                    b=NumericArraySpec((), support=interval(0.0, bound)),
                ),
            )
            # Two traced intervals cannot be shown equal, so no support is shared.
            seen.append(law.support)
            return jnp.asarray(0.0)

        jax.jit(build)(2.0)
        assert seen == [None]

    def test_one_array_leaf_gives_its_dtype_and_support(self):
        from probpipe import positive

        law = _DeclaredLaw("x", NumericArraySpec((), dtype="float64", support=positive))
        assert law.dtypes == {"x": np.dtype("float64")}
        assert law.dtype == np.dtype("float64")
        assert law.support is positive

    @pytest.mark.parametrize("view", ["dtypes", "supports", "dtype", "support"])
    def test_the_numeric_views_belong_to_numeric_laws(self, view):
        # A numeric law has them whatever its class; any other law has none.
        numeric = _DeclaredLaw("x", NumericArraySpec((), dtype="float32"))
        opaque = _DeclaredLaw("law", RecordSpec(a=NumericArraySpec(()), o=OpaqueSpec()))
        assert not hasattr(Distribution, view)
        assert hasattr(NumericDistribution, view)
        assert hasattr(numeric, view)
        with pytest.raises(AttributeError, match=f"non-numeric event, and {view} belongs"):
            getattr(opaque, view)

    def test_an_attribute_error_of_a_property_is_kept(self):
        class _Raising(Distribution):
            @property
            def broken(self):
                raise AttributeError("the property's own message")

        with pytest.raises(AttributeError, match="the property's own message"):
            _ = _Raising("x", NumericArraySpec(())).broken


# One construction per TFP family, which passes its keywords to the family, with
# the event shape, dtype, and support it declares. A dtype of None is the default
# float.
_FAMILY_SCHEMAS = [
    pytest.param(lambda **kw: Normal("x", loc=0.0, scale=1.0, **kw), (), None, real, id="Normal"),
    pytest.param(
        lambda **kw: Beta("x", alpha=2.0, beta=3.0, **kw), (), None, unit_interval, id="Beta"
    ),
    pytest.param(
        lambda **kw: Gamma("x", concentration=2.0, rate=1.0, **kw), (), None, positive, id="Gamma"
    ),
    pytest.param(
        lambda **kw: InverseGamma("x", concentration=2.0, scale=1.0, **kw),
        (),
        None,
        positive,
        id="InverseGamma",
    ),
    pytest.param(
        lambda **kw: Exponential("x", rate=1.0, **kw), (), None, positive, id="Exponential"
    ),
    pytest.param(
        lambda **kw: LogNormal("x", loc=0.0, scale=1.0, **kw), (), None, positive, id="LogNormal"
    ),
    pytest.param(
        lambda **kw: StudentT("x", df=3.0, loc=0.0, scale=1.0, **kw), (), None, real, id="StudentT"
    ),
    pytest.param(
        lambda **kw: Uniform("x", low=-1.0, high=2.0, **kw),
        (),
        None,
        interval(-1.0, 2.0),
        id="Uniform",
    ),
    pytest.param(lambda **kw: Cauchy("x", loc=0.0, scale=1.0, **kw), (), None, real, id="Cauchy"),
    pytest.param(lambda **kw: Laplace("x", loc=0.0, scale=1.0, **kw), (), None, real, id="Laplace"),
    pytest.param(
        lambda **kw: HalfNormal("x", scale=1.0, **kw), (), None, non_negative, id="HalfNormal"
    ),
    pytest.param(
        lambda **kw: HalfCauchy("x", loc=0.5, scale=1.0, **kw),
        (),
        None,
        greater_than(0.5),
        id="HalfCauchy",
    ),
    pytest.param(
        lambda **kw: Pareto("x", concentration=2.0, scale=1.5, **kw),
        (),
        None,
        greater_than(1.5),
        id="Pareto",
    ),
    pytest.param(
        lambda **kw: TruncatedNormal("x", loc=0.0, scale=1.0, low=-1.0, high=1.0, **kw),
        (),
        None,
        interval(-1.0, 1.0),
        id="TruncatedNormal",
    ),
    pytest.param(
        lambda **kw: Bernoulli("x", probs=0.3, **kw), (), "int32", boolean, id="Bernoulli"
    ),
    pytest.param(
        lambda **kw: Binomial("x", total_count=5, probs=0.3, **kw),
        (),
        None,
        integer_interval(0, 5),
        id="Binomial",
    ),
    pytest.param(
        lambda **kw: Poisson("x", rate=2.0, **kw), (), None, non_negative_integer, id="Poisson"
    ),
    pytest.param(
        lambda **kw: Categorical("x", probs=[0.2, 0.3, 0.5], **kw),
        (),
        "int32",
        integer_interval(0, 2),
        id="Categorical",
    ),
    pytest.param(
        lambda **kw: NegativeBinomial("x", total_count=5.0, probs=0.3, **kw),
        (),
        None,
        non_negative_integer,
        id="NegativeBinomial",
    ),
    pytest.param(
        lambda **kw: MultivariateNormal("x", loc=jnp.zeros(3), cov=jnp.eye(3), **kw),
        (3,),
        None,
        real,
        id="MultivariateNormal",
    ),
    pytest.param(
        lambda **kw: Dirichlet("x", concentration=jnp.ones(3), **kw),
        (3,),
        None,
        simplex,
        id="Dirichlet",
    ),
    pytest.param(
        lambda **kw: Multinomial("x", total_count=4.0, probs=jnp.array([0.2, 0.3, 0.5]), **kw),
        (3,),
        None,
        non_negative_integer,
        id="Multinomial",
    ),
    pytest.param(
        lambda **kw: Wishart("x", df=4.0, scale_tril=jnp.eye(2), **kw),
        (2, 2),
        None,
        positive_definite,
        id="Wishart",
    ),
    pytest.param(
        lambda **kw: VonMisesFisher(
            "x", mean_direction=jnp.array([0.0, 1.0]), concentration=2.0, **kw
        ),
        (2,),
        None,
        sphere,
        id="VonMisesFisher",
    ),
]


class TestFamilyDeclarations:
    """A TFP family declares one draw as a whole-term array whose component defaults to its name."""

    @pytest.mark.parametrize(("make", "shape", "dtype", "support"), _FAMILY_SCHEMAS)
    def test_the_schema_views_read_the_declaration(self, make, shape, dtype, support):
        law = make()
        dtype = np.dtype(dtype) if dtype is not None else jnp.asarray(0.0).dtype
        assert law.event_spec is law.spec.event_spec
        assert law.event_spec == OutputSpec(x=NumericArraySpec(shape, dtype, support))
        assert law.event_shape == shape
        assert law.dtypes == {"x": dtype}
        assert law.dtype == dtype
        assert law.supports == {"x": support}
        assert law.support == support
        assert issubclass(type(law), NumericDistribution)
        assert isinstance(law, NumericDistribution)

    @pytest.mark.parametrize(("make", "shape", "dtype", "support"), _FAMILY_SCHEMAS)
    def test_event_spec_names_the_component(self, make, shape, dtype, support):
        law = make(event_spec=OutputSpec(theta=None))
        dtype = np.dtype(dtype) if dtype is not None else jnp.asarray(0.0).dtype
        assert law.name == "x"
        assert law.event_spec == OutputSpec(theta=NumericArraySpec(shape, dtype, support))
        assert law["theta"] is law
        with pytest.raises(KeyError):
            law["x"]

    def test_a_declared_type_is_checked_against_the_draw(self):
        dtype = jnp.asarray(0.0).dtype
        law = MultivariateNormal(
            "x", jnp.zeros(3), cov=jnp.eye(3), event_spec=OutputSpec(theta=NumericArraySpec(("d",)))
        )
        assert law.event_spec == OutputSpec(theta=NumericArraySpec((3,), dtype, real))
        with pytest.raises(ValueError, match="has dimension 3, expected 2"):
            MultivariateNormal(
                "x",
                jnp.zeros(3),
                cov=jnp.eye(3),
                event_spec=OutputSpec(theta=NumericArraySpec((2,))),
            )
        with pytest.raises(ValueError, match="does not conform"):
            Normal("x", 0.0, 1.0, event_spec=OutputSpec(theta=NumericArraySpec((), "int32")))

    def test_event_spec_declares_a_whole_array(self):
        record = OutputSpec(RecordSpec(theta=NumericArraySpec(())))
        with pytest.raises(TypeError, match="needs a RecordSpec"):
            Normal("x", 0.0, 1.0, event_spec=record)
        with pytest.raises(TypeError, match="must be an OutputSpec"):
            Normal("x", 0.0, 1.0, event_spec=NumericArraySpec(()))


class TestJointDeclarations:
    """A joint declares an exposed record of its components' declared terms."""

    def test_a_joint_keeps_each_component_dtype_and_support(self):
        joint = Normal("a", 0.0, 1.0) * Gamma("c", 2.0, 1.0)
        dtype = jnp.asarray(0.0).dtype
        assert joint.event_spec == OutputSpec(
            RecordSpec(
                a=NumericArraySpec((), dtype, real),
                c=NumericArraySpec((), dtype, positive),
            )
        )
        assert joint.dtypes == {"a": dtype, "c": dtype}
        assert joint.supports == {"a": real, "c": positive}
        assert tuple(joint.event_spec.components) == ("a", "c")
        with pytest.raises(AttributeError, match="does not draw a single array"):
            _ = joint.event_shape

    def test_a_declared_component_is_keyed_by_the_joint(self):
        from probpipe.distributions import FactoredDistribution

        growth = Normal("x", 0.0, 1.0, event_spec=OutputSpec(growth=None))
        joint = FactoredDistribution("p", [growth])
        assert tuple(joint.event_spec.components) == ("growth",)

    def test_a_gaussian_joint_declares_its_blocks(self):
        joint = MultivariateNormal("x", jnp.zeros(1), cov=jnp.eye(1)) * MultivariateNormal(
            "y", jnp.zeros(2), cov=jnp.eye(2)
        )
        assert joint.event_spec.spec == RecordSpec(
            x=NumericArraySpec((1,), jnp.asarray(0.0).dtype, real),
            y=NumericArraySpec((2,), jnp.asarray(0.0).dtype, real),
        )

    def test_an_empirical_law_over_records_declares_each_row(self):
        rows = RecordBatch(
            "rows",
            {"labels": np.array(["a", "b"], dtype=object), "ids": np.array([0, 1], dtype=np.int32)},
            "row",
            element_spec=RecordSpec(labels=OpaqueSpec(), ids=NumericArraySpec((), np.int32)),
        )
        joint = EmpiricalDistribution("rows", rows)
        assert joint.event_spec.spec == RecordSpec(
            labels=OpaqueSpec(), ids=NumericArraySpec((), np.int32)
        )
        # An opaque field makes the draw non-numeric, so it has no dtypes.
        assert not hasattr(joint, "dtypes")


class TestEmpiricalDeclarations:
    """An empirical or bootstrap law declares what one draw is, read off its atoms or source."""

    @pytest.mark.pending(
        reason="the exported sample wraps a batch-valued draw as a record, not as its declared batch",
        raises=AssertionError,
    )
    def test_a_replicate_of_a_record_valued_law_declares_a_batch_of_records(self):
        source = (Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)).with_name("p")
        replicate = BootstrapReplicateDistribution("rep", source, replicate_size=3, level="row")
        spec = replicate.event_spec.spec
        assert isinstance(spec, BatchSpec)
        assert spec.batch_shape == (3,)
        assert tuple(spec.element_spec.fields) == ("a", "b")
        assert spec.is_valid(sample(replicate, key=jax.random.PRNGKey(0)))

    def test_a_replicate_of_a_nested_posterior_keeps_its_groups(self):
        from probpipe.inference._approximate_distribution import ApproximateDistribution

        posterior = ApproximateDistribution(
            [jnp.ones((10, 4))],
            name="post",
            event_spec=RecordSpec(a=RecordSpec(b=(2,), c=()), d=()),
        )
        replicate = BootstrapReplicateDistribution("rep", posterior, level="draw")
        spec = replicate.event_spec.spec
        assert spec.batch_shape == (10,)
        assert spec.element_spec == posterior.event_spec.spec
        assert set(spec.element_spec.children["a"].children) == {"b", "c"}

    def test_opaque_atoms_are_a_whole_term(self):
        law = EmpiricalDistribution("law", OpaqueBatch("labels", ["a", "b"], "atom"))
        assert law.event_spec == OutputSpec(law=OpaqueSpec())

    def test_array_atoms_are_a_whole_term(self):
        law = EmpiricalDistribution("x", jnp.zeros((5, 2)))
        dtype = jnp.asarray(0.0).dtype
        assert law.event_spec == OutputSpec(x=NumericArraySpec((2,), dtype))
        assert law.dtypes == {"x": dtype}
        assert law.supports == {"x": None}
        assert law.event_shape == (2,)

    def test_record_atoms_expose_their_leaves(self):
        atoms = NumericRecordBatch(
            "r",
            {"a": jnp.zeros(4), "b/c": jnp.zeros((4, 3))},
            "atom",
            element_spec=NumericRecordSpec(a=(), b=NumericRecordSpec(c=(3,))),
        )
        law = EmpiricalDistribution("r", atoms)
        assert set(law.dtypes) == {"a", "b/c"}
        assert law.event_spec.spec["b/c"].shape == (3,)

    def test_a_replicate_of_an_empirical_law_is_a_batch_on_its_level(self):
        law = BootstrapReplicateDistribution(
            "x", EmpiricalDistribution("x", jnp.zeros((5, 2))), replicate_size=3
        )
        spec = law.event_spec.spec
        assert (spec.batch_shape, spec.level_names) == ((3,), ("x",))
        assert spec.element_spec.shape == (2,)

    def test_a_replicate_of_an_array_law_is_a_batch_of_its_term(self):
        law = BootstrapReplicateDistribution("reps", Normal("x", 0.0, 1.0), replicate_size=4)
        assert law.event_spec == OutputSpec(
            reps=BatchSpec(NumericArraySpec((), jnp.asarray(0.0).dtype, real), ((4,),), ("x",))
        )

    def test_a_replicate_needs_a_law_that_samples(self):
        class _Sampler:
            # Implements SupportsSampling without being a Distribution.

            def _sample(self, key, sample_shape=()):
                return jax.random.normal(key, (*sample_shape, 2))

        with pytest.raises(TypeError, match="samples"):
            BootstrapReplicateDistribution("reps", _Sampler(), replicate_size=5)

    def test_a_bootstrap_measure_draws_laws_of_its_sources_event(self):
        source = Normal("x", 0.0, 1.0)
        law = BootstrapDistribution("measure", source, 3)
        assert law.event_spec == OutputSpec(measure=DistributionSpec(source.event_spec))

    def test_a_numeric_record_empirical_declares_its_atoms(self):
        atoms = NumericRecordBatch(
            "rows",
            {"u": jnp.ones((4, 2)), "v": jnp.zeros(4)},
            "row",
            element_spec=NumericRecordSpec(
                u=NumericArraySpec((2,), None, real), v=NumericArraySpec((), None, real)
            ),
        )
        law = EmpiricalDistribution("rows", atoms)
        assert law.supports == {"u": real, "v": real}
        assert law.event_spec.spec["u"].shape == (2,)


class TestDerivedDeclarations:
    """Transformed laws, random functions and measures, and minibatch laws declare
    what one draw is."""

    def test_a_transformed_law_declares_the_image(self):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        law = BijectorTransformedDistribution("t", Normal("x", 0.0, 1.0), tfb.Exp())
        assert law.event_spec == OutputSpec(
            t=NumericArraySpec((), jnp.asarray(0.0).dtype, positive)
        )
        over_atoms = BijectorTransformedDistribution(
            "u", EmpiricalDistribution("e", jnp.ones((4, 2))), tfb.Exp()
        )
        assert over_atoms.event_shape == (2,)

    def test_a_random_function_draws_a_callable_with_a_named_output(self):
        from probpipe import FunctionSpec, LinearBasisFunction

        weights = MultivariateNormal("w", loc=jnp.zeros(2), cov=jnp.eye(2))
        f = LinearBasisFunction("f", lambda X: jnp.concatenate([X, X**2], -1), weights)
        assert f.event_spec == OutputSpec(f=FunctionSpec(output_spec=OutputSpec(f=None)))
        # A derived function keeps its base's component; its label is not one.
        shifted = f + 1.0
        assert shifted.name == "shift(f)"
        assert shifted.event_spec is f.event_spec

    def test_a_minibatched_measure_draws_laws_over_the_prior_parameters(self):
        from probpipe import MinibatchedDistribution
        from probpipe.families import BernoulliFamily, glm_likelihood

        X = jnp.eye(4)
        prior = MultivariateNormal("beta", loc=jnp.zeros(4), cov=jnp.eye(4))
        measure = MinibatchedDistribution(
            "measure",
            prior,
            glm_likelihood("y", BernoulliFamily(), X=X),
            jnp.array([1.0, 0.0, 1.0, 0.0]),
            batch_size=2,
        )
        assert measure.event_spec == OutputSpec(measure=DistributionSpec(prior.event_spec))
        draw = measure._draw_one(jax.random.PRNGKey(0))
        assert draw.event_spec is prior.event_spec
        assert measure.event_spec.spec.is_valid(draw)
        at_point = measure._random_unnormalized_log_prob()(jnp.zeros(4))
        assert at_point.event_spec == OutputSpec(log_prob=NumericArraySpec(()))


class TestViewAndWrapperDeclarations:
    """Views declare the term they select."""

    def test_a_variadic_input_label_names_its_marginal(self):
        from probpipe import function

        @function
        def double(*args):
            return args[0] * 2.0

        out = double.with_options(include_inputs=True)(Normal("a", 0.0, 1.0))
        assert list(out.event_spec.components) == ["*args[0]", "double"]
        assert list(out["*args[0]"].event_spec.components) == ["*args[0]"]

    def test_a_nested_view_joins_a_path_into_its_group(self):
        law = _DeclaredLaw("p", RecordSpec(a=RecordSpec(b=RecordSpec(c=NumericArraySpec(())))))
        group = law["a"]
        assert group["a/b/c"].event_spec == law["a/b/c"].event_spec
        assert group["a"] is group
        with pytest.raises(KeyError):
            group["missing"]
        with pytest.raises(KeyError):
            group["b/c"]

    def test_a_field_view_is_a_whole_term_under_its_last_segment(self):
        dtype = jnp.asarray(0.0).dtype
        a, c = NumericArraySpec((), dtype, real), NumericArraySpec((), dtype, positive)
        law = _DeclaredLaw("p", RecordSpec(a=a, b=RecordSpec(c=c)))
        assert law["a"].event_spec == OutputSpec(a=a)
        nested = law["b"]["b/c"]
        assert nested.name == "b/c"
        assert nested.event_spec == OutputSpec(c=c)

    def test_a_slash_path_selects_the_field_it_names(self):
        law = _DeclaredLaw("p", RecordSpec(a=(), b=RecordSpec(c=())))
        view = law["b/c"]
        assert view.name == "b/c"
        assert view.event_spec == law["b"]["b/c"].event_spec


class TestModelDeclarations:
    """Posteriors declare what their targets or stored draws are."""

    def test_a_posterior_declares_its_targets_event(self):
        from probpipe.inference._approximate_distribution import make_posterior

        prior = MultivariateNormal("z", loc=jnp.zeros(2), cov=jnp.eye(2))
        post = make_posterior(
            [jnp.zeros((10, 2))],
            parents=(prior,),
            algorithm="test",
            event_spec=RecordSpec(a=(), b=()),
        )
        assert post.event_spec == OutputSpec(RecordSpec(a=(), b=()))
        positive_target = RecordSpec(a=NumericArraySpec((), None, positive), b=())
        post = make_posterior(
            [jnp.ones((10, 2))], parents=(prior,), algorithm="test", event_spec=positive_target
        )
        assert post.event_spec == OutputSpec(positive_target)


class TestDimensionTransforms:
    def test_with_dim_sizes_binds_a_free_dimension(self):
        law = _DeclaredLaw("x", NumericArraySpec(("n",)))
        bound = law.with_dim_sizes(n=3)
        assert type(bound) is type(law)
        assert bound.name == "x"
        assert bound.event_spec == OutputSpec(x=NumericArraySpec((3,)))
        assert law.event_spec == OutputSpec(x=NumericArraySpec(("n",)))

    def test_with_dim_sizes_refuses_a_name_that_is_not_free(self):
        bound = _DeclaredLaw("x", NumericArraySpec(("n",))).with_dim_sizes(n=3)
        with pytest.raises(ValueError, match=r"no free dimensions \['n'\]"):
            bound.with_dim_sizes(n=4)

    def test_with_dim_names_renames_simultaneously(self):
        law = _DeclaredLaw("x", NumericArraySpec(("n", "m")))
        assert law.with_dim_names(n="m", m="n").event_spec.spec.shape == ("m", "n")

    def test_with_dim_names_ignores_a_name_that_is_not_free(self):
        law = _DeclaredLaw("x", NumericArraySpec(("n",)))
        assert law.with_dim_names(k="j").event_spec == law.event_spec

    @pytest.mark.parametrize(
        ("transform", "arguments"),
        [("with_dim_sizes", {"n": 3}), ("with_dim_names", {"n": "m"})],
    )
    def test_a_transform_records_its_provenance(self, transform, arguments):
        law = _DeclaredLaw("x", NumericArraySpec(("n",)))
        result = getattr(law, transform)(**arguments)
        assert result.provenance.operation == transform
        assert result.provenance.metadata == arguments

    def test_with_dim_sizes_refuses_a_bad_size(self):
        law = _DeclaredLaw("x", NumericArraySpec(("n",)))
        with pytest.raises(ValueError, match="must be non-negative"):
            law.with_dim_sizes(n=-1)
        with pytest.raises(TypeError, match="must be an integer"):
            law.with_dim_sizes(n=2.5)

    def test_a_library_law_transforms_its_declaration(self):
        law = Normal("x", 0.0, 1.0)
        with pytest.raises(ValueError, match="no free dimensions"):
            law.with_dim_sizes(n=3)
        renamed = law.with_dim_names(n="m")
        assert type(renamed) is Normal
        assert renamed.event_spec == law.event_spec


class TestDistributionSpecMatching:
    def test_a_packaging_mismatch_is_named(self):
        spec = DistributionSpec(RecordSpec(x=()))
        with pytest.raises(ValueError, match="declares an exposed record, but the law declares"):
            spec.bind_dims_from_value(_DeclaredLaw("x", NumericArraySpec(())))

    def test_a_whole_term_under_another_component_is_named(self):
        spec = DistributionSpec(OutputSpec(y=NumericArraySpec(())))
        with pytest.raises(
            ValueError, match="declares the component 'y', but the law declares 'x'"
        ):
            spec.bind_dims_from_value(_DeclaredLaw("x", NumericArraySpec(())))


class TestDistributionSpecFingerprint:
    def test_the_two_packagings_fingerprint_apart(self):
        from probpipe.core._fingerprint import fingerprint

        whole = DistributionSpec(OutputSpec(x=NumericArraySpec(())))
        exposed = DistributionSpec(RecordSpec(x=()))
        assert fingerprint(whole) != fingerprint(exposed)
        assert fingerprint(whole) == fingerprint(
            DistributionSpec(OutputSpec(x=NumericArraySpec(())))
        )
