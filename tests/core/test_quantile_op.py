"""The ``quantile`` op + ``RecordEmpiricalDistribution._quantile``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EmpiricalDistribution, Normal, NumericRecord, SupportsQuantile, quantile
from probpipe.core.record import Record


def _np_weighted_quantile(values, weights, qs):
    """Independent midpoint-CDF (Hazen) weighted quantile, per column (NumPy)."""
    n = values.shape[0]
    event = values.shape[1:]
    flat = values.reshape(n, -1)
    out = np.empty((len(qs), flat.shape[1]))
    for j in range(flat.shape[1]):
        order = np.argsort(flat[:, j])
        v, w = flat[order, j], weights[order]
        cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
        out[:, j] = np.interp(qs, cdf, v)
    return out.reshape(len(qs), *event)


class TestQuantileOp:
    def test_uniform_matches_numpy_hazen(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (1000,))
        emp = EmpiricalDistribution(samples, name="x")
        q = np.array([0.1, 0.5, 0.9])
        # Uniform weights use the same type-5 (Hazen) convention as the weighted
        # path — not jnp.quantile's type-7 — so the two paths stay consistent.
        np.testing.assert_allclose(
            np.asarray(quantile(emp, jnp.asarray(q))),
            np.quantile(np.asarray(samples), q, method="hazen"),
            atol=1e-5,
        )

    def test_scalar_q_returns_scalar_shape(self):
        samples = jax.random.normal(jax.random.PRNGKey(1), (1000,))
        emp = EmpiricalDistribution(samples, name="x")
        med = np.asarray(quantile(emp, 0.5))
        assert med.shape == ()
        assert float(med) == pytest.approx(float(jnp.median(samples)), abs=1e-5)

    def test_weighted_quantile_matches_analytic_cdf(self):
        # Weights ∝ value give density f(x) ∝ x on [0, 1], so F(x) = x² and the
        # q-quantile is √q — an independent analytic baseline for the weighted path.
        samples = jnp.linspace(0.0, 1.0, 101)
        emp = EmpiricalDistribution(samples, weights=samples, name="x")
        q = jnp.array([0.1, 0.5, 0.9])
        wq = np.asarray(quantile(emp, q))
        # Hazen weighted quantile; discretization error ≤ 0.005 on 101 points.
        np.testing.assert_allclose(wq, np.sqrt(np.asarray(q)), atol=0.02)

    def test_weighted_quantile_vector_event(self):
        # Non-uniform weights + a 2-D event exercise the reshape/vmap in
        # _weighted_quantile, checked against an independent NumPy midpoint-CDF
        # weighted quantile.
        samples = jax.random.normal(jax.random.PRNGKey(7), (500, 2))
        weights = jnp.arange(1.0, 501.0)
        emp = EmpiricalDistribution(samples, weights=weights, name="z")
        qs = [0.25, 0.75]
        out = np.asarray(quantile(emp, jnp.array(qs)))
        assert out.shape == (2, 2)
        ref = _np_weighted_quantile(np.asarray(samples), np.asarray(weights), qs)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_multifield_per_field_quantiles(self):
        key = jax.random.PRNGKey(2)
        a = jax.random.normal(key, (1000,))
        b = jax.random.normal(jax.random.PRNGKey(3), (1000,)) + 5.0
        emp = EmpiricalDistribution(Record("r", a=a, b=b))
        res = quantile(emp, 0.5)
        assert float(np.asarray(res["a"])) == pytest.approx(float(jnp.median(a)), abs=1e-5)
        assert float(np.asarray(res["b"])) == pytest.approx(float(jnp.median(b)), abs=1e-5)

    def test_vector_q_on_vector_event(self):
        # (n, 2) samples, q a 3-vector → per-field quantile shape (3, 2).
        samples = jax.random.normal(jax.random.PRNGKey(4), (1000, 2))
        emp = EmpiricalDistribution(samples, name="z")
        q = np.array([0.25, 0.5, 0.75])
        out = np.asarray(quantile(emp, jnp.asarray(q)))
        assert out.shape == (3, 2)
        np.testing.assert_allclose(
            out, np.quantile(np.asarray(samples), q, axis=0, method="hazen"), atol=1e-5
        )

    def test_raises_on_unsupported_distribution(self):
        with pytest.raises(TypeError, match="quantile"):
            quantile(Normal(loc=0.0, scale=1.0, name="x"), 0.5)

    def test_raises_on_out_of_range_q(self):
        emp = EmpiricalDistribution(jnp.arange(10.0), name="x")
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            quantile(emp, 1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            quantile(emp, jnp.array([0.2, -0.1]))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            quantile(emp, jnp.nan)

    def test_single_field_result_is_numeric_record(self):
        # A single-field empirical returns a NumericRecord that the shim coerces
        # to a bare scalar — pin the type so the unwrap contract can't drift.
        emp = EmpiricalDistribution(jax.random.normal(jax.random.PRNGKey(8), (200,)), name="x")
        res = quantile(emp, 0.5)
        assert isinstance(res, NumericRecord)
        assert np.asarray(res).shape == ()

    def test_empirical_satisfies_supports_quantile(self):
        emp = EmpiricalDistribution(jnp.arange(10.0), name="x")
        assert isinstance(emp, SupportsQuantile)
