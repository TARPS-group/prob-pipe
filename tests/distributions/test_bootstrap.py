

import numpy as np
import pytest

from probpipe.distributions.continuous import BootstrapDistribution, Normal1D
from probpipe.distributions.multivariate import Multivariate, MvNormal


# ------------------------------ Helpers ------------------------------

def _pop_cov(X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted population covariance: sum w (x - m)(x - m)^T."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    if w is None:
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        w = np.asarray(w, dtype=float).reshape(-1)
        w = w / w.sum()
    m = (w[:, None] * X).sum(axis=0)
    diff = X - m
    return diff.T @ (diff * w[:, None])


# ------------------------------- Basics --------------------------------

def test_init_rejects_bad_weights_shape_and_values():
    Theta = np.array([[0.0], [1.0], [2.0]], dtype=float)

    # Wrong shape
    with pytest.raises(ValueError):
        BootstrapDistribution(Theta, weights=np.array([0.2, 0.8]))

    # Negative weights
    with pytest.raises(ValueError):
        BootstrapDistribution(Theta, weights=np.array([0.5, -0.2, 0.7]))

    # Sum to zero
    with pytest.raises(ValueError):
        BootstrapDistribution(Theta, weights=np.array([0.0, 0.0, 0.0]))


def test_properties_and_summaries_univariate():
    Theta = np.array([1.0, 3.0, 5.0], dtype=float)  # (B,) -> (B,1)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    boot = BootstrapDistribution(Theta, weights=w, rng=np.random.default_rng(0))

    assert boot.n == 3
    assert boot.d == 1
    assert boot.replicates.shape == (3, 1)
    assert boot.weights.shape == (3,)

    m_expected = (w * Theta).sum()
    cov_expected = _pop_cov(Theta, w)

    assert np.allclose(boot.mean()[0], m_expected, atol=1e-12)
    assert np.allclose(boot.cov()[0, 0], cov_expected[0, 0], atol=1e-12)
    assert np.allclose(boot.var()[0], cov_expected[0, 0], atol=1e-12)
    assert np.allclose(boot.std()[0], np.sqrt(cov_expected[0, 0]), atol=1e-12)


def test_mean_and_cov_weighted_multivariate():
    Theta = np.array([[0.0, 0.0],
                      [2.0, 1.0],
                      [4.0, 3.0]], dtype=float)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    boot = BootstrapDistribution(Theta, weights=w, rng=np.random.default_rng(1))

    m_expected = (w[:, None] * Theta).sum(axis=0)
    cov_expected = _pop_cov(Theta, w)

    assert np.allclose(boot.mean(), m_expected, atol=1e-12)
    assert np.allclose(boot.cov(), cov_expected, atol=1e-12)


# ------------------------------- Sampling -------------------------------

def test_sample_shape_and_weight_bias():
    # Strongly biased weights: 0.9 on (-1, 0), 0.1 on (+1, 0)
    Theta = np.array([[-1.0, 0.0],
                      [ 1.0, 0.0]], dtype=float)
    w = np.array([0.9, 0.1], dtype=float)
    boot = BootstrapDistribution(Theta, weights=w, rng=np.random.default_rng(42))

    n = 40_000
    draws = boot.sample(n)  # (n, 2)
    prop_pos = (draws[:, 0] > 0).mean()  # should be ~0.1

    assert draws.shape == (n, 2)
    assert abs(prop_pos - 0.1) < 0.02  # allow sampling noise


def test_sample_without_replacement_guard():
    Theta = np.array([[0.0], [1.0], [2.0]], dtype=float)
    boot = BootstrapDistribution(Theta, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        boot.sample(4, replace=False)  # B=3, cannot sample 4 without replacement


def test_rvs_alias_matches_sample():
    Theta = np.array([[10.0], [20.0], [30.0]], dtype=float)
    boot = BootstrapDistribution(Theta, rng=np.random.default_rng(123))
    out1 = boot.sample(5)
    out2 = boot.rvs(5)
    assert out1.shape == out2.shape == (5, 1)


# ----------------------------- expectation() -----------------------------

def test_expectation_scalar_returns_Normal1D_and_values():
    Theta = np.array([[0.0, 1.0],
                      [2.0, 3.0],
                      [4.0, 5.0]], dtype=float)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    boot = BootstrapDistribution(Theta, weights=w, rng=np.random.default_rng(7))

    # f(Theta*) = first coordinate
    def f_first(thetas):
        return thetas[:, 0]

    n_mc = 4096
    out = boot.expectation(f_first, n_mc=n_mc)
    assert isinstance(out, Normal1D)

    y = Theta[:, 0]
    m = float((w * y).sum())
    var = float((w * (y - m) ** 2).sum())
    se = np.sqrt(max(var, 0.0)) / np.sqrt(n_mc)

    assert np.isclose(out.mu, m, atol=1e-12)
    assert np.isclose(out.sigma, max(se, 1e-12), atol=1e-12)
    assert out.sigma >= 0.0


def test_expectation_vector_returns_MvNormal_and_scales_with_nmc():
    Theta = np.array([[0.0, 1.0],
                      [2.0, 3.0],
                      [4.0, 5.0]], dtype=float)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    boot = BootstrapDistribution(Theta, weights=w, rng=np.random.default_rng(9))

    # f(Theta*) = identity (vector-valued)
    def f_id(thetas):
        return thetas  # shape (B, 2)

    out1 = boot.expectation(f_id, n_mc=100)   # cov_mean ~ Cov(Theta)/100
    out2 = boot.expectation(f_id, n_mc=1000)  # cov_mean ~ Cov(Theta)/1000

    assert isinstance(out1, MvNormal)
    assert isinstance(out2, MvNormal)

    m_expected = (w[:, None] * Theta).sum(axis=0)
    cov_pop = _pop_cov(Theta, w)

    # Means match
    assert np.allclose(out1.mean(), m_expected, atol=1e-12)
    assert np.allclose(out2.mean(), m_expected, atol=1e-12)

    # Covariance-of-mean scales like 1/n_mc
    assert np.allclose(out1.cov(), cov_pop / 100.0, atol=1e-12)
    assert np.allclose(out2.cov(), cov_pop / 1000.0, atol=1e-12)
    assert np.linalg.norm(out2.cov()) < np.linalg.norm(out1.cov()) / 5.0


# --------------------------- from_data constructor ---------------------------

def test_from_data_univariate_mean_matches_sample_mean():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=2.5, scale=1.0, size=200)  # i.i.d. sample
    sample_mean = float(np.mean(data))

    def stat_mean(x):
        # x arrives with samples on axis 0
        return np.mean(x, axis=0)

    B = 5000
    boot = BootstrapDistribution.from_data(data, stat_fn=stat_mean, B=B, axis=0, rng=np.random.default_rng(999))

    # Shape and rough agreement: mean of bootstrap means should be close to sample mean
    assert boot.replicates.shape == (B, 1)
    assert abs(boot.mean()[0] - sample_mean) < 0.05  # loose tolerance to avoid flakiness


def test_from_data_vector_stat_shapes_and_means():
    rng = np.random.default_rng(321)
    data = rng.normal(loc=0.0, scale=1.0, size=(300, 2))  # 2D observations

    def stat_vec(x):
        # Return 2-dim statistic: per-dimension means
        return np.mean(x, axis=0)  # shape (2,)

    B = 4000
    boot = BootstrapDistribution.from_data(data, stat_fn=stat_vec, B=B, axis=0, rng=np.random.default_rng(2024))

    # Shapes
    assert boot.replicates.shape == (B, 2)
    # Mean of bootstrap statistic ~ statistic on original data
    sample_stat = np.mean(data, axis=0)
    assert np.allclose(boot.mean(), sample_stat, atol=0.05)