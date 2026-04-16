"""Tests for Distribution base-class machinery (name, renamed, provenance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Normal, Gamma, Record, MultivariateNormal
from probpipe.core.provenance import Provenance, provenance_ancestors


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
