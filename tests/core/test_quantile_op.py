"""The ``quantile`` op + ``RecordEmpiricalDistribution._quantile`` (issue #301, PR 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EmpiricalDistribution, Normal, quantile
from probpipe.core.record import Record


class TestQuantileOp:
    def test_uniform_matches_jnp_quantile(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (1000,))
        emp = EmpiricalDistribution(samples, name="x")
        q = jnp.array([0.1, 0.5, 0.9])
        # Uniform weights → exactly jnp.quantile along the sample axis.
        np.testing.assert_allclose(
            np.asarray(quantile(emp, q)), np.asarray(jnp.quantile(samples, q)), atol=1e-5
        )

    def test_scalar_q_returns_scalar_shape(self):
        samples = jax.random.normal(jax.random.PRNGKey(1), (1000,))
        emp = EmpiricalDistribution(samples, name="x")
        med = np.asarray(quantile(emp, 0.5))
        assert med.shape == ()
        assert float(med) == pytest.approx(float(jnp.median(samples)), abs=1e-5)

    def test_weighted_quantile_shifts_toward_upweighted_values(self):
        # Linear weights favoring larger values pull the weighted median above 0.5.
        samples = jnp.linspace(0.0, 1.0, 101)
        emp = EmpiricalDistribution(samples, weights=samples + 1e-3, name="x")
        wq = float(np.asarray(quantile(emp, 0.5)))
        assert 0.6 < wq < 0.8  # closed-form ≈ 1/√2 ≈ 0.707

    def test_multifield_per_field_quantiles(self):
        key = jax.random.PRNGKey(2)
        a = jax.random.normal(key, (1000,))
        b = jax.random.normal(jax.random.PRNGKey(3), (1000,)) + 5.0
        emp = EmpiricalDistribution(Record(a=a, b=b))
        res = quantile(emp, 0.5)
        assert float(np.asarray(res["a"])) == pytest.approx(float(jnp.median(a)), abs=1e-5)
        assert float(np.asarray(res["b"])) == pytest.approx(float(jnp.median(b)), abs=1e-5)

    def test_vector_q_on_vector_event(self):
        # (n, 2) samples, q a 3-vector → per-field quantile shape (3, 2).
        samples = jax.random.normal(jax.random.PRNGKey(4), (1000, 2))
        emp = EmpiricalDistribution(samples, name="z")
        out = np.asarray(quantile(emp, jnp.array([0.25, 0.5, 0.75])))
        assert out.shape == (3, 2)
        np.testing.assert_allclose(
            out, np.asarray(jnp.quantile(samples, jnp.array([0.25, 0.5, 0.75]), axis=0)), atol=1e-5
        )

    def test_raises_on_unsupported_distribution(self):
        with pytest.raises(TypeError, match="quantile"):
            quantile(Normal(loc=0.0, scale=1.0, name="x"), 0.5)
