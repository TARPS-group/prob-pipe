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
    BootstrapReplicateDistribution,
    Categorical,
    Cauchy,
    Dirichlet,
    Distribution,
    DistributionArray,
    DistributionSpec,
    EmpiricalDistribution,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    LogNormal,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    NumericDistribution,
    NumericRecordSpec,
    OutputSpec,
    Pareto,
    Poisson,
    RandomFunction,
    RandomMeasure,
    RecordEmpiricalDistribution,
    RecordSpec,
    StudentT,
    TransformedDistribution,
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
    simplex,
    sphere,
    unit_interval,
)
from probpipe.core._opaque import OpaqueSpec
from probpipe.core._specs import NumericArraySpec
from probpipe.core._workflow_distribution_normalization import DISTRIBUTION_HINT_PROTOCOLS
from probpipe.core.provenance import Provenance, provenance_ancestors
from probpipe.distributions.kde import KDEDistribution


def _make_transformed():
    import tensorflow_probability.substrates.jax.bijectors as tfb

    return TransformedDistribution(
        "transformed",
        Normal(loc=0.0, scale=1.0, name="x"),
        tfb.Exp(),
    )


# Distribution-instance factories used by ``TestNoBatchShape``. Mirrors
# the ``DISTRIBUTIONS`` table in ``tests/core/test_iteration_protocol.py`` but with
# a smaller set covering the canonical TFP-backed scalars + the most
# distinct subclasses (TransformedDistribution / KDEDistribution /
# RecordEmpiricalDistribution).
_NO_BATCH_SHAPE_DISTS = [
    pytest.param(lambda: Normal(loc=0.0, scale=1.0, name="x"), id="Normal"),
    pytest.param(lambda: Gamma(concentration=3.0, rate=1.0, name="g"), id="Gamma"),
    pytest.param(
        lambda: MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z"),
        id="MultivariateNormal",
    ),
    pytest.param(_make_transformed, id="TransformedDistribution"),
    pytest.param(
        lambda: KDEDistribution("kde", jnp.zeros((20, 3))),
        id="KDEDistribution",
    ),
    pytest.param(
        lambda: RecordEmpiricalDistribution("x", jnp.zeros((10, 3))),
        id="RecordEmpiricalDistribution",
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
        assert n2.event_template.fields == ("x",)

    def test_template_shape_preserved(self):
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="a")
        assert mvn.event_template["a"] == NumericArraySpec((3,))
        b = mvn.with_name("b")
        assert b.event_template["a"] == NumericArraySpec((3,))


class TestNoBatchShape:
    """``Distribution`` has no ``batch_shape`` attribute. Pins the
    absence across the public Distribution family so a future
    subclass can't silently reintroduce it as a defensive default.
    Container types (``DistributionArray``, ``RecordBatch``) keep
    their own ``batch_shape`` — that's a different concept and is
    asserted separately at the bottom.
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

    def test_distribution_array_keeps_batch_shape(self):
        """Container types are unaffected: ``DistributionArray``
        retains its own ``batch_shape`` (the array's outer shape)."""
        from probpipe import DistributionArray

        da = DistributionArray.from_batched_params(
            Normal,
            loc=jnp.zeros(5),
            scale=1.0,
            name="x",
        )
        assert da.batch_shape == (5,)


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
                super().__init__(name="x")

        assert repr(_NamedDist()) == "_NamedDist(name='x')"


class TestConstructorNameCheck:
    """``Distribution.__init__`` rejects a name that is not a non-empty string."""

    @pytest.mark.parametrize("name", ["", 123, None])
    def test_invalid_name_raises(self, name):
        from probpipe import Distribution

        class _Dist(Distribution):
            def __init__(self, name):
                super().__init__(name=name)

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
        ``self._name`` ends up set to a non-empty string."""
        from probpipe import Distribution

        class _DirectNameDist(Distribution):
            def __init__(self):
                # Skip super().__init__ deliberately.
                self._name = "direct"

        dist = _DirectNameDist()
        assert dist.name == "direct"

    def test_record_distribution_without_template_raises(self):
        """The ``_RecordDistributionMeta`` adds a record-template check
        on top of the name check. A RecordDistribution subclass whose
        ``__init__`` neither sets ``_event_template`` nor leaves
        ``name + event_shape`` derivable can't be constructed."""
        from probpipe import RecordDistribution

        class _NoTemplate(RecordDistribution):
            def __init__(self):
                self._name = "no_template"
                # No _event_template; no event_shape declared.

        with pytest.raises(TypeError, match="event_template"):
            _NoTemplate()


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
        assert clone.event_template.fields == ("x",)
        assert original.name == "x"
        assert original.event_template.fields == ("x",)

    def test_with_name_preserves_multi_field_template(self):
        """Multi-field joints have explicit templates whose field
        names are independent of the distribution's name — renaming
        must not touch the template."""
        import jax.numpy as jnp

        from probpipe import JointGaussian

        jg = JointGaussian(
            mean=jnp.zeros(2),
            cov=jnp.eye(2),
            x=1,
            y=1,
        )
        original_fields = jg.event_template.fields
        clone = jg.with_name("renamed_jg")
        assert clone.event_template.fields == original_fields

    def test_with_name_preserves_non_numeric_event_template(self):
        """``JointEmpirical`` (non-NRD ``RecordDistribution``) builds its
        template from the stored samples, not from the distribution's
        name — renaming must leave the template intact (otherwise the
        metaclass invariant would be violated, since the non-numeric
        base has no auto-rebuild path)."""
        import numpy as np

        from probpipe import JointEmpirical

        je = JointEmpirical(
            labels=np.array(["a", "b", "c"], dtype=object),
            ids=np.array([0, 1, 2]),
        )
        original_fields = je.event_template.fields
        clone = je.with_name("renamed_je")
        assert clone.event_template is not None
        assert clone.event_template.fields == original_fields


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

    def test_distribution_without_template_invalid(self):
        # A distribution always carries the schema of its draws; one that
        # exposes no event template cannot satisfy any DistributionSpec.
        from probpipe import Distribution

        class _NoTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_NoTemplate())

    def test_distribution_with_none_template_invalid(self):
        from probpipe import Distribution

        class _NoneTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                return None

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_NoneTemplate())

    def test_type_error_template_is_not_a_match(self):
        # TypeError is the documented "template not derivable" signal (e.g. an
        # un-named auto-deriving distribution): a non-match, so is_valid
        # returns False.
        from probpipe import Distribution

        class _NotDerivable(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                raise TypeError("template not derivable")

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_NotDerivable())

    @pytest.mark.parametrize("error", [RuntimeError, ValueError, KeyError])
    def test_unexpected_template_error_propagates(self, error):
        # An unexpected error from event_template is a malfunctioning
        # distribution, not a clean non-match — is_valid must not mask it as
        # invalid; it propagates so the bug surfaces.
        from probpipe import Distribution

        class _Broken(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                raise error("boom")

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        with pytest.raises(error):
            spec.is_valid(_Broken())


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
            DistributionArray,
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
    "BayesFlowModel",
    "BroadcastDistribution",
    "DistributionArray",
    "FlattenedDistributionView",
    "JointEmpirical",
    "JointGaussian",
    "NumericJointEmpirical",
    "NumericRecordDistributionView",
    "ProductDistribution",
    "SequentialJointDistribution",
    "SimpleGenerativeModel",
    "SimpleModel",
    "TFPProductDistribution",
}
_PUBLIC_CLASSES = _public_distribution_classes()


class TestNameFirstSignature:
    """Every constructor the design keeps takes ``name`` first, required."""

    @pytest.mark.parametrize(
        "cls",
        [cls for cls in _PUBLIC_CLASSES if cls.__name__ not in _RETIRING],
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


class TestDerivedNames:
    """A law that ``expectation`` constructs is named for the operation."""

    @pytest.mark.parametrize(
        ("make_operand", "f"),
        [
            pytest.param(lambda: Normal("law", 0.0, 1.0), lambda x: x, id="monte-carlo"),
            pytest.param(
                lambda: EmpiricalDistribution("law", ["a", "b", "c", "d"]),
                lambda x: jnp.asarray(1.0),
                id="generic-empirical",
            ),
            pytest.param(
                lambda: RecordEmpiricalDistribution("law", jnp.arange(10.0)),
                lambda x: x,
                id="record-empirical",
            ),
            pytest.param(
                lambda: BootstrapReplicateDistribution(
                    "law", EmpiricalDistribution("data", jnp.arange(5.0))
                ),
                jnp.mean,
                id="bootstrap-replicate",
            ),
            pytest.param(
                lambda: MultivariateNormal(
                    "law", jnp.zeros(2), cov=jnp.eye(2)
                ).as_record_distribution(template=NumericRecordSpec(a=(), b=())),
                lambda x: x["a"],
                id="record-view",
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


class TestComponentAccess:
    def test_a_whole_term_is_itself_under_its_component(self):
        law = _DeclaredLaw("x", NumericArraySpec(()))
        assert law["x"] is law
        with pytest.raises(KeyError):
            law["y"]


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


class TestSchemaViews:
    def test_event_shape_is_the_declared_array_shape(self):
        assert _DeclaredLaw("x", NumericArraySpec((3, 2))).event_shape == (3, 2)

    def test_event_shape_is_undefined_for_a_record_draw(self):
        with pytest.raises(TypeError, match="does not draw a single array"):
            _ = _DeclaredLaw("law", RecordSpec(x=())).event_shape

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


# One construction per TFP family, with the event shape, dtype, and support each
# reported before it stored its declaration. A dtype of None is the default float.
_FAMILY_SCHEMAS = [
    pytest.param(lambda: Normal("x", loc=0.0, scale=1.0), (), None, real, id="Normal"),
    pytest.param(lambda: Beta("x", alpha=2.0, beta=3.0), (), None, unit_interval, id="Beta"),
    pytest.param(lambda: Gamma("x", concentration=2.0, rate=1.0), (), None, positive, id="Gamma"),
    pytest.param(
        lambda: InverseGamma("x", concentration=2.0, scale=1.0),
        (),
        None,
        positive,
        id="InverseGamma",
    ),
    pytest.param(lambda: Exponential("x", rate=1.0), (), None, positive, id="Exponential"),
    pytest.param(lambda: LogNormal("x", loc=0.0, scale=1.0), (), None, positive, id="LogNormal"),
    pytest.param(lambda: StudentT("x", df=3.0, loc=0.0, scale=1.0), (), None, real, id="StudentT"),
    pytest.param(
        lambda: Uniform("x", low=-1.0, high=2.0), (), None, interval(-1.0, 2.0), id="Uniform"
    ),
    pytest.param(lambda: Cauchy("x", loc=0.0, scale=1.0), (), None, real, id="Cauchy"),
    pytest.param(lambda: Laplace("x", loc=0.0, scale=1.0), (), None, real, id="Laplace"),
    pytest.param(lambda: HalfNormal("x", scale=1.0), (), None, non_negative, id="HalfNormal"),
    pytest.param(
        lambda: HalfCauchy("x", loc=0.5, scale=1.0),
        (),
        None,
        greater_than(0.5),
        id="HalfCauchy",
    ),
    pytest.param(
        lambda: Pareto("x", concentration=2.0, scale=1.5),
        (),
        None,
        greater_than(1.5),
        id="Pareto",
    ),
    pytest.param(
        lambda: TruncatedNormal("x", loc=0.0, scale=1.0, low=-1.0, high=1.0),
        (),
        None,
        interval(-1.0, 1.0),
        id="TruncatedNormal",
    ),
    pytest.param(lambda: Bernoulli("x", probs=0.3), (), "int32", boolean, id="Bernoulli"),
    pytest.param(
        lambda: Binomial("x", total_count=5, probs=0.3),
        (),
        None,
        integer_interval(0, 5),
        id="Binomial",
    ),
    pytest.param(lambda: Poisson("x", rate=2.0), (), None, non_negative_integer, id="Poisson"),
    pytest.param(
        lambda: Categorical("x", probs=[0.2, 0.3, 0.5]),
        (),
        "int32",
        integer_interval(0, 2),
        id="Categorical",
    ),
    pytest.param(
        lambda: NegativeBinomial("x", total_count=5.0, probs=0.3),
        (),
        None,
        non_negative_integer,
        id="NegativeBinomial",
    ),
    pytest.param(
        lambda: MultivariateNormal("x", loc=jnp.zeros(3), cov=jnp.eye(3)),
        (3,),
        None,
        real,
        id="MultivariateNormal",
    ),
    pytest.param(
        lambda: Dirichlet("x", concentration=jnp.ones(3)), (3,), None, simplex, id="Dirichlet"
    ),
    pytest.param(
        lambda: Multinomial("x", total_count=4.0, probs=jnp.array([0.2, 0.3, 0.5])),
        (3,),
        None,
        non_negative_integer,
        id="Multinomial",
    ),
    pytest.param(
        lambda: Wishart("x", df=4.0, scale_tril=jnp.eye(2)),
        (2, 2),
        None,
        positive_definite,
        id="Wishart",
    ),
    pytest.param(
        lambda: VonMisesFisher("x", mean_direction=jnp.array([0.0, 1.0]), concentration=2.0),
        (2,),
        None,
        sphere,
        id="VonMisesFisher",
    ),
]


class TestFamilyDeclarations:
    """A TFP family declares one draw as a whole-term array under its name."""

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

    def test_a_batched_backend_declares_one_cell(self):
        arr = DistributionArray.from_batched_params(Normal, loc=jnp.zeros(4), scale=1.0, name="arr")
        assert arr._backend._batched_dist.event_spec.spec.shape == ()


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


class TestDistributionSpecFingerprint:
    def test_the_two_packagings_fingerprint_apart(self):
        from probpipe.core._fingerprint import fingerprint

        whole = DistributionSpec(OutputSpec(x=NumericArraySpec(())))
        exposed = DistributionSpec(RecordSpec(x=()))
        assert fingerprint(whole) != fingerprint(exposed)
        assert fingerprint(whole) == fingerprint(
            DistributionSpec(OutputSpec(x=NumericArraySpec(())))
        )
