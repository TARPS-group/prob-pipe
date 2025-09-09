# from probpipe.distributions.continuous import EmpiricalDistribution, Normal1D
# from probpipe.distributions.multivariate import MvNormal

import numpy as np
import pytest


# ------------------------------- Basics --------------------------------

def test_init_rejects_bad_weights_shape_and_values():
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)

    # Wrong shape
    with pytest.raises(ValueError):
        EmpiricalDistribution(X, weights=np.array([0.2, 0.8]))  # length 2 vs n=3

    # Negative weights
    with pytest.raises(ValueError):
        EmpiricalDistribution(X, weights=np.array([0.5, -0.2, 0.7]))

    # Sum to zero
    with pytest.raises(ValueError):
        EmpiricalDistribution(X, weights=np.array([0.0, 0.0, 0.0]))


def test_properties_and_shapes_univariate():
    X = np.array([1.0, 3.0, 5.0], dtype=float)  # 1D; will be reshaped to (n,1)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    emp = EmpiricalDistribution(X, weights=w, rng=np.random.default_rng(0))

    assert emp.n == 3
    assert emp.d == 1
    assert emp.samples.shape == (3, 1)
    assert emp.weights.shape == (3,)

    # Weighted moments (population)
    m_expected = (w * X).sum()
    v_expected = ((w * (X - m_expected) ** 2).sum())

    assert emp.mean().shape == (1,)
    assert np.allclose(emp.mean()[0], m_expected, rtol=0, atol=1e-12)
    assert emp.cov().shape == (1, 1)
    assert np.allclose(emp.cov()[0, 0], v_expected, rtol=0, atol=1e-12)
    assert np.allclose(emp.var()[0], v_expected, rtol=0, atol=1e-12)
    assert np.allclose(emp.std()[0], np.sqrt(v_expected), rtol=0, atol=1e-12)


def test_mean_and_cov_weighted_multivariate():
    X = np.array([[0.0, 0.0],
                  [2.0, 1.0],
                  [4.0, 3.0]], dtype=float)
    w = np.array([0.2, 0.3, 0.5], dtype=float)

    emp = EmpiricalDistribution(X, weights=w, rng=np.random.default_rng(1))

    m_expected = (w[:, None] * X).sum(axis=0)
    diff = X - m_expected
    cov_expected = diff.T @ (diff * w[:, None])  # weighted population covariance

    assert np.allclose(emp.mean(), m_expected, atol=1e-12)
    assert np.allclose(emp.cov(), cov_expected, atol=1e-12)


# ------------------------------- Sampling -------------------------------

def test_sample_shape_and_weight_bias():
    # Strongly biased weights: 0.9 on -1, 0.1 on +1
    X = np.array([[-1.0], [1.0]], dtype=float)
    w = np.array([0.9, 0.1], dtype=float)
    emp = EmpiricalDistribution(X, weights=w, rng=np.random.default_rng(42))

    n = 50_000
    draws = emp.sample(n)  # (n,1)
    prop_pos = (draws[:, 0] > 0).mean()  # should be ~0.1

    assert draws.shape == (n, 1)
    assert abs(prop_pos - 0.1) < 0.02  # allow sampling noise


def test_sample_without_replacement_guard():
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    emp = EmpiricalDistribution(X, rng=np.random.default_rng(0))

    with pytest.raises(ValueError):
        emp.sample(4, replace=False)  # n=3, cannot sample 4 without replacement


def test_rvs_alias_matches_sample():
    X = np.array([[10.0], [20.0], [30.0]], dtype=float)
    emp = EmpiricalDistribution(X, rng=np.random.default_rng(123))
    out1 = emp.sample(5)
    out2 = emp.rvs(5)
    assert out1.shape == out2.shape == (5, 1)


# ----------------------------- expectation() -----------------------------

def test_expectation_scalar_returns_Normal1D_and_values():
    X = np.array([[0.0, 1.0],
                  [2.0, 3.0],
                  [4.0, 5.0]], dtype=float)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    emp = EmpiricalDistribution(X, weights=w, rng=np.random.default_rng(7))

    # f(X) = first coordinate
    def f_first(xs):
        return xs[:, 0]

    n_mc = 4096
    out = emp.expectation(f_first, n_mc=n_mc)
    assert isinstance(out, Normal1D)

    # Expected mean & SE from weighted population moments of f
    y = X[:, 0]
    m = float((w * y).sum())
    var = float((w * (y - m) ** 2).sum())
    se = np.sqrt(max(var, 0.0)) / np.sqrt(n_mc)

    assert np.isclose(out.mu, m, atol=1e-12)
    assert np.isclose(out.sigma, max(se, 1e-12), atol=1e-12)
    assert out.sigma >= 0.0


def test_expectation_vector_returns_MvNormal_and_values_and_scales_with_nmc():
    X = np.array([[0.0, 1.0],
                  [2.0, 3.0],
                  [4.0, 5.0]], dtype=float)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    emp = EmpiricalDistribution(X, weights=w, rng=np.random.default_rng(9))

    # f(X) = identity (vector-valued)
    def f_id(xs):
        return xs  # shape (n, 2)

    out1 = emp.expectation(f_id, n_mc=100)   # cov_mean ~ Cov(Y)/100
    out2 = emp.expectation(f_id, n_mc=1000)  # cov_mean ~ Cov(Y)/1000

    assert isinstance(out1, MvNormal)
    assert isinstance(out2, MvNormal)

    # Mean check
    m_expected = (w[:, None] * X).sum(axis=0)
    assert np.allclose(out1.mean(), m_expected, atol=1e-12)
    assert np.allclose(out2.mean(), m_expected, atol=1e-12)

    # Covariance of the mean should scale ~ 1/n_mc
    diff = X - m_expected
    cov_pop = diff.T @ (diff * w[:, None])  # population Cov(Y)
    cov1 = out1.cov()
    cov2 = out2.cov()

    assert np.allclose(cov1, cov_pop / 100.0, atol=1e-12)
    assert np.allclose(cov2, cov_pop / 1000.0, atol=1e-12)
    # And cov2 should be 10x smaller (Frobenius norm) than cov1
    assert np.linalg.norm(cov2) < np.linalg.norm(cov1) / 5.0