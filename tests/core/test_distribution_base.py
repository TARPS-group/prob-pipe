"""Tests for Distribution base-class machinery (name, renamed, provenance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    Gamma,
    Record,
    MultivariateNormal,
    RecordEmpiricalDistribution,
    TransformedDistribution,
)
from probpipe.core.provenance import Provenance, provenance_ancestors
from probpipe.distributions.kde import KDEDistribution


def _make_transformed():
    import tensorflow_probability.substrates.jax.bijectors as tfb
    return TransformedDistribution(
        Normal(loc=0.0, scale=1.0, name="x"), tfb.Exp(),
    )


# Distribution-instance factories used by ``TestNoBatchShape``. Mirrors
# the ``DISTRIBUTIONS`` table in ``test_iteration_protocol.py`` but with
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
        lambda: KDEDistribution(jnp.zeros((20, 3)), name="kde"),
        id="KDEDistribution",
    ),
    pytest.param(
        lambda: RecordEmpiricalDistribution(jnp.zeros((10, 3)), name="x"),
        id="RecordEmpiricalDistribution",
    ),
]


class TestRenamedBasics:
    """Distribution.renamed() returns a new object with a new name."""

    def test_returns_new_object(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.renamed("y")
        assert n is not n2
        assert n.name == "x"  # original unchanged
        assert n2.name == "y"

    def test_is_same_type(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert type(n.renamed("y")) is type(n)

    def test_is_shallow_copy(self):
        """Underlying parameters are shared (not deep-copied)."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.renamed("y")
        assert n2._loc is n._loc  # shared array
        assert n2._scale is n._scale


class TestRenamedProvenance:
    """renamed() attaches a 'renamed' Provenance pointing to the original."""

    def test_source_operation(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.renamed("y")
        assert n2.source is not None
        assert n2.source.operation == "renamed"

    def test_source_parents(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.renamed("y")
        assert n2.source.parents == (n,)

    def test_source_metadata(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.renamed("y")
        assert n2.source.metadata["old_name"] == "x"
        assert n2.source.metadata["new_name"] == "y"

    def test_rename_chain_preserves_ancestry(self):
        """a.renamed("b").renamed("c") keeps a in the ancestor DAG."""
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = a.renamed("b")
        c = b.renamed("c")
        ancestors = provenance_ancestors(c)
        assert a in ancestors
        assert b in ancestors

    def test_original_source_not_mutated(self):
        """Renaming does not alter the original's source."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        n.with_source(Provenance("construction", parents=()))
        n.renamed("y")
        assert n.source.operation == "construction"


class TestRenamedSampling:
    """Renamed copies behave identically under sampling/log_prob."""

    def test_sample_statistics_match(self):
        n = Normal(loc=2.0, scale=0.5, name="x")
        n2 = n.renamed("mu")
        key = jax.random.PRNGKey(0)
        s1 = n._sample(key, (2000,))
        s2 = n2._sample(key, (2000,))
        # Same key -> identical samples
        np.testing.assert_allclose(np.asarray(s1), np.asarray(s2), atol=1e-6)

    def test_log_prob_matches(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n2 = n.renamed("z")
        x = jnp.asarray(1.23)
        np.testing.assert_allclose(
            float(n._log_prob(x)), float(n2._log_prob(x)), atol=1e-6,
        )

    def test_event_shape_matches(self):
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="a")
        renamed = mvn.renamed("b")
        assert renamed.event_shape == mvn.event_shape


class TestRenamedRecordTemplate:
    """renamed() regenerates the cached record_template with the new name."""

    def test_template_field_name_updates(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        # Touch the template on the original so it's cached
        assert n.record_template.fields == ("x",)
        n2 = n.renamed("growth_rate")
        assert n2.record_template.fields == ("growth_rate",)

    def test_template_shape_preserved(self):
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="a")
        assert mvn.record_template["a"] == (3,)
        b = mvn.renamed("b")
        assert b.record_template["b"] == (3,)


class TestNoBatchShape:
    """``Distribution`` has no ``batch_shape`` attribute. Pins the
    absence across the public Distribution family so a future
    subclass can't silently reintroduce it as a defensive default.
    Container types (``DistributionArray``, ``RecordArray``) keep
    their own ``batch_shape`` — that's a different concept and is
    asserted separately at the bottom.
    """

    def test_no_batch_shape_on_base_class(self):
        from probpipe import Distribution
        assert not hasattr(Distribution, "batch_shape")

    @pytest.mark.parametrize(
        "make_dist", _NO_BATCH_SHAPE_DISTS,
    )
    def test_no_batch_shape_on_instance(self, make_dist):
        dist = make_dist()
        assert not hasattr(dist, "batch_shape"), (
            f"{type(dist).__name__} unexpectedly exposes a "
            f"batch_shape attribute."
        )

    def test_distribution_array_keeps_batch_shape(self):
        """Container types are unaffected: ``DistributionArray``
        retains its own ``batch_shape`` (the array's outer shape)."""
        from probpipe import DistributionArray
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.zeros(5), scale=1.0, name="x",
        )
        assert da.batch_shape == (5,)


class TestAuxiliaryDiagnosticsAccessor:
    def test_auxiliary_defaults_to_none(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")
        assert dist.auxiliary is None
        assert dist.diagnostics is None

    def test_diagnostics_none_when_auxiliary_has_no_diagnostics_group(self):
        import xarray as xr

        dist = Normal(loc=0.0, scale=1.0, name="x")
        dist._auxiliary = xr.DataTree.from_dict({"arviz": xr.Dataset()})
        assert dist.auxiliary is dist._auxiliary
        assert dist.diagnostics is None

    def test_diagnostics_returns_view_for_diagnostics_group(self):
        import xarray as xr
        from probpipe.diagnostics.views import DiagnosticsView

        dist = Normal(loc=0.0, scale=1.0, name="x")
        dist._auxiliary = xr.DataTree.from_dict(
            {"diagnostics": xr.Dataset(attrs={"warnings": "[]"})}
        )
        view = dist.diagnostics
        assert view is not None
        assert isinstance(view, DiagnosticsView)
        assert view.warnings == []


class TestMetaclassEnforcement:
    """The ``_DistributionMeta`` metaclass enforces a non-empty ``name``
    on every Distribution subclass instance, even when the subclass
    bypasses ``super().__init__``.
    """

    def test_subclass_without_name_raises_at_construction(self):
        """A subclass whose ``__init__`` doesn't set ``_name`` cannot be
        constructed — the metaclass post-init check fires before the
        instance escapes."""
        from probpipe.core.distribution import Distribution

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
        from probpipe.core.distribution import Distribution

        class _EmptyNameDist(Distribution):
            def __init__(self):
                self._name = ""

        with pytest.raises(TypeError, match="non-empty name"):
            _EmptyNameDist()

    def test_subclass_with_non_string_name_raises(self):
        """The metaclass requires the final ``_name`` to be a string."""
        from probpipe.core.distribution import Distribution

        class _NonStringNameDist(Distribution):
            def __init__(self):
                self._name = 123

        with pytest.raises(TypeError, match="non-empty name"):
            _NonStringNameDist()

    def test_subclass_setting_name_directly_succeeds(self):
        """Bypassing ``super().__init__`` is fine as long as
        ``self._name`` ends up set to a non-empty string."""
        from probpipe.core.distribution import Distribution

        class _DirectNameDist(Distribution):
            def __init__(self):
                # Skip super().__init__ deliberately.
                self._name = "direct"

        dist = _DirectNameDist()
        assert dist.name == "direct"

    def test_record_distribution_without_template_raises(self):
        """The ``_RecordDistributionMeta`` adds a record-template check
        on top of the name check. A RecordDistribution subclass whose
        ``__init__`` neither sets ``_record_template`` nor leaves
        ``name + event_shape`` derivable can't be constructed."""
        from probpipe.core.distribution import RecordDistribution

        class _NoTemplate(RecordDistribution):
            def __init__(self):
                self._name = "no_template"
                # No _record_template; no event_shape declared.

        with pytest.raises(TypeError, match="record_template"):
            _NoTemplate()


class TestRenamedTemplateRoundtrip:
    """``NumericRecordDistribution.renamed`` regenerates an auto-built
    template under the new name; explicit and multi-field templates
    are preserved.
    """

    def test_renamed_rebuilds_single_field_auto_template(self):
        """Single-field auto-built template: the clone's
        ``record_template`` has the new field name (matches
        ``new_name``)."""
        from probpipe import Normal

        original = Normal(loc=0.0, scale=1.0, name="x")
        # Trigger the auto-build so ``_record_template`` is cached.
        assert original.record_template.fields == ("x",)

        clone = original.renamed("y")
        assert clone.name == "y"
        # The rebuilt template uses the new name as the field key.
        assert clone.record_template.fields == ("y",)
        # The original is untouched (renamed returns a copy).
        assert original.record_template.fields == ("x",)

    def test_renamed_preserves_multi_field_template(self):
        """Multi-field joints have explicit templates whose field
        names are independent of the distribution's name — renaming
        must not touch the template."""
        from probpipe import JointGaussian
        import jax.numpy as jnp

        jg = JointGaussian(
            mean=jnp.zeros(2), cov=jnp.eye(2), x=1, y=1,
        )
        original_fields = jg.record_template.fields
        clone = jg.renamed("renamed_jg")
        assert clone.record_template.fields == original_fields

    def test_renamed_preserves_non_numeric_record_template(self):
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
        original_fields = je.record_template.fields
        clone = je.renamed("renamed_je")
        assert clone.record_template is not None
        assert clone.record_template.fields == original_fields
