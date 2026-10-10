"""The ``quantile`` op over empirical laws, the generalized inverse of the weighted CDF."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    EmpiricalDistribution,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    OutputSpec,
    Poisson,
    RecordSpec,
    ResolutionError,
    SupportsQuantile,
    quantile,
    workflow_run,
)


def _np_weighted_quantile(values, weights, qs):
    """Independent weighted quantile ``inf{x : F(x) >= q}``, per column (NumPy)."""
    n = values.shape[0]
    event = values.shape[1:]
    flat = values.reshape(n, -1)
    out = np.empty((len(qs), flat.shape[1]))
    for j in range(flat.shape[1]):
        order = np.argsort(flat[:, j])
        v, w = flat[order, j], weights[order]
        cdf = np.cumsum(w) / w.sum()
        out[:, j] = v[np.searchsorted(cdf, qs, side="left")]
    return out.reshape(len(qs), *event)


class TestQuantileOp:
    def test_uniform_matches_numpy_inverted_cdf(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (1000,))
        emp = EmpiricalDistribution(samples, component="x")
        q = np.array([0.1, 0.5, 0.9])
        # The generalized inverse CDF is NumPy's type-1 ``inverted_cdf`` method.
        np.testing.assert_allclose(
            np.asarray(quantile(emp, jnp.asarray(q))),
            np.quantile(np.asarray(samples), q, method="inverted_cdf"),
            atol=1e-5,
        )

    def test_scalar_q_returns_scalar_shape(self):
        samples = jax.random.normal(jax.random.PRNGKey(1), (1000,))
        emp = EmpiricalDistribution(samples, component="x")
        med = np.asarray(quantile(emp, 0.5))
        assert med.shape == ()
        expected = np.quantile(np.asarray(samples), 0.5, method="inverted_cdf")
        assert float(med) == pytest.approx(float(expected), abs=1e-5)

    def test_weighted_quantile_matches_analytic_cdf(self):
        # Weights ∝ value give density f(x) ∝ x on [0, 1], so F(x) = x² and the
        # q-quantile is √q — an independent analytic baseline for the weighted path.
        samples = jnp.linspace(0.0, 1.0, 101)
        emp = EmpiricalDistribution(samples, weights=samples, component="x")
        q = jnp.array([0.1, 0.5, 0.9])
        wq = np.asarray(quantile(emp, q))
        # The discretization error is at most 0.01 on 101 points.
        np.testing.assert_allclose(wq, np.sqrt(np.asarray(q)), atol=0.02)

    def test_weighted_quantile_vector_event(self):
        # Non-uniform weights and a 2-D event, checked against an independent
        # NumPy weighted quantile.
        samples = jax.random.normal(jax.random.PRNGKey(7), (500, 2))
        weights = jnp.arange(1.0, 501.0)
        emp = EmpiricalDistribution(samples, weights=weights, component="z")
        qs = [0.25, 0.75]
        out = np.asarray(quantile(emp, jnp.array(qs)))
        assert out.shape == (2, 2)
        ref = _np_weighted_quantile(np.asarray(samples), np.asarray(weights), qs)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_multifield_per_field_quantiles(self):
        key = jax.random.PRNGKey(2)
        a = jax.random.normal(key, (1000,))
        b = jax.random.normal(jax.random.PRNGKey(3), (1000,)) + 5.0
        atoms = NumericRecordBatch(
            {"a": a, "b": b},
            "row",
            element_spec=RecordSpec(a=(), b=()),
            label="rows",
        )
        emp = EmpiricalDistribution(atoms, label="emp")
        res = quantile(emp, 0.5)
        for field, values in (("a", a), ("b", b)):
            expected = np.quantile(np.asarray(values), 0.5, method="inverted_cdf")
            estimate = float(np.asarray(res[f"quantile({field})"]))
            assert estimate == pytest.approx(float(expected), abs=1e-5)

    def test_vector_q_on_vector_event(self):
        # (n, 2) samples, q a 3-vector → per-field quantile shape (3, 2).
        samples = jax.random.normal(jax.random.PRNGKey(4), (1000, 2))
        emp = EmpiricalDistribution(samples, component="z")
        q = np.array([0.25, 0.5, 0.75])
        out = np.asarray(quantile(emp, jnp.asarray(q)))
        assert out.shape == (3, 2)
        np.testing.assert_allclose(
            out, np.quantile(np.asarray(samples), q, axis=0, method="inverted_cdf"), atol=1e-5
        )

    def test_raises_on_unsupported_distribution(self):
        """A law without quantiles that does not sample has no converter to quantiles."""

        class Bare(Distribution):
            pass

        with pytest.raises(ResolutionError, match="does not implement SupportsQuantile"):
            quantile(
                Bare(
                    OutputSpec(x=NumericArraySpec(())),
                    label="x",
                ),
                0.5,
            )

    def test_a_law_without_quantiles_that_samples_converts_to_its_empirical_law(self):
        """The Poisson family has no closed-form quantile, so its draws' quantile is returned."""
        with workflow_run(seed=0):
            median = quantile(Poisson("x", 2.0), 0.5)
        assert float(jnp.asarray(median)) in (1.0, 2.0, 3.0)

    def test_raises_on_out_of_range_q(self):
        emp = EmpiricalDistribution(jnp.arange(10.0), component="x")
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            quantile(emp, 1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            quantile(emp, jnp.array([0.2, -0.1]))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            quantile(emp, jnp.nan)

    def test_one_level_of_an_array_law_is_an_array(self):
        emp = EmpiricalDistribution(jax.random.normal(jax.random.PRNGKey(8), (200,)), component="x")
        res = quantile(emp, 0.5)
        assert isinstance(res, NumericArray)
        assert np.asarray(res).shape == ()

    def test_empirical_satisfies_supports_quantile(self):
        emp = EmpiricalDistribution(jnp.arange(10.0), component="x")
        assert isinstance(emp, SupportsQuantile)


class TestRawQuantilesAtTheirLevels:
    """A law's raw quantiles lead each leaf with the level axes, and the op wraps them."""

    @staticmethod
    def _record_law():
        atoms = NumericRecordBatch(
            {"b": jnp.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0]]), "a": jnp.array([2.0, 1.0, 3.0])},
            "row",
            element_spec=RecordSpec(b=(2,), a=()),
            label="rows",
        )
        return EmpiricalDistribution(atoms, label="post")

    def test_several_levels_of_an_array_law_are_a_batch_on_the_level_quantile(self):
        result = quantile(
            EmpiricalDistribution(jnp.array([2.0, 4.0, 1.0, 3.0]), component="x"),
            jnp.array([0.0, 1.0]),
        )
        assert isinstance(result, NumericArrayBatch)
        assert (result.level_names, result.batch_shape) == (("quantile",), (2,))
        np.testing.assert_allclose(np.asarray(result.values), [1.0, 4.0])

    def test_one_level_of_a_record_law_is_a_record(self):
        result = quantile(self._record_law(), 1.0)
        assert isinstance(result, NumericRecord)
        assert result.fields == ("quantile(b)", "quantile(a)")
        np.testing.assert_allclose(np.asarray(result["quantile(b)"]), [2.0, 3.0])

    def test_several_levels_of_a_record_law_are_a_batch_of_records(self):
        result = quantile(self._record_law(), jnp.array([0.0, 1.0]))
        assert isinstance(result, NumericRecordBatch)
        assert (result.level_names, result.batch_shape) == (("quantile",), (2,))
        np.testing.assert_allclose(np.asarray(result["quantile(a)"]), [1.0, 3.0])
        np.testing.assert_allclose(np.asarray(result["quantile(b)"]), [[0.0, 1.0], [2.0, 3.0]])
