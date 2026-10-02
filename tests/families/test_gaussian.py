"""The Gaussian algebra (VII.6): the Gaussian random functions and the factored joint."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    FunctionSpec,
    GaussianRandomFunction,
    LinearBasisFunction,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OutputSpec,
    RandomFunction,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
    mean,
    sample,
    variance,
)
from probpipe.distributions import FactoredNumericDistribution
from probpipe.families import (
    FactoredMultivariateGaussian,
    GaussianProcess,
    LinearGaussianConditional,
)
from probpipe.linalg import DenseLinOp, LinOp

# ---------------------------------------------------------------------------
# Dense ground truth
# ---------------------------------------------------------------------------
#
# For a LinearBasisFunction with basis Φ and weights w ~ N(m, C), f(x) = Φ(x) w:
#
#   scalar output, Φ(X) of shape (n, d_w):
#     mean(X) = Φ(X) m, Cov(X) = Φ(X) C Φ(X)ᵀ, Var(X) = diag Cov(X)
#   output vectors of size d_out, Φ(X) of shape (n, d_out, d_w):
#     the joint covariance over the flattened values (point index slowest) is
#     Φ_flat C Φ_flatᵀ with Φ_flat = Φ(X) reshaped to (n d_out, d_w)
#
# After a linear map h(x) = A g(x), the basis of h is A Φ_g(x), and the same
# formulas apply.


def _dense_joint_cov(phi_X, w_cov):
    """The joint covariance of a basis-function model's flattened values."""
    phi_flat = phi_X.reshape(-1, phi_X.shape[-1])
    return phi_flat @ w_cov @ phi_flat.T


def _dense(cov):
    assert isinstance(cov, LinOp)
    return np.asarray(cov.to_dense())


def _assert_the_joint_law(law, phi_X, w_mean, w_cov, key):
    """*law* is the joint law of the flattened values ``Φ(X) w`` for ``w ~ N(m, C)``.

    Its moments are the dense ground truth, and its draws are finite with that covariance,
    although the covariance is singular when there are more values than weights.
    """
    phi_flat = np.asarray(phi_X).reshape(-1, phi_X.shape[-1])
    joint_cov = phi_flat @ np.asarray(w_cov) @ phi_flat.T
    np.testing.assert_allclose(mean(law), phi_flat @ np.asarray(w_mean), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(variance(law), np.diag(joint_cov), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(_dense(law._cov()), joint_cov, rtol=1e-5, atol=1e-7)
    draws = np.asarray(law._sample(key, (100_000,)))
    assert np.isfinite(draws).all()
    np.testing.assert_allclose(np.cov(draws, rowvar=False), joint_cov, atol=2e-3)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Members written directly against the interface
# ---------------------------------------------------------------------------


def _rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """The squared-exponential kernel between two stacks of points."""
    sq_dist = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


class _ScalarGP(GaussianRandomFunction):
    """A scalar-output process with a squared-exponential kernel and a nugget."""

    def __init__(self, lengthscale=1.0, variance=1.0, noise=0.01, name="gp"):
        super().__init__(name)
        self._ls = lengthscale
        self._var = variance
        self._noise = noise

    def predict_mean(self, X):
        return jnp.zeros(X.shape[0])

    def predict_variance(self, X):
        return jnp.full(X.shape[0], self._var + self._noise)

    def predict_covariance(self, X):
        K = _rbf_kernel(X, X, self._ls, self._var)
        return DenseLinOp(K + self._noise * jnp.eye(X.shape[0]))


class _MultiOutputGRF(GaussianRandomFunction):
    """A two-output member with independent values at every point."""

    def __init__(self):
        super().__init__("grf")

    def predict_mean(self, X):
        s = jnp.sum(X, axis=-1)
        return jnp.stack([s, 2 * s], axis=-1)

    def predict_variance(self, X):
        return jnp.ones((X.shape[0], 2))

    def predict_covariance(self, X):
        return DenseLinOp(jnp.eye(2 * X.shape[0]))


class _MarginalOnlyGRF(GaussianRandomFunction):
    """A member that gives only the marginal variance at each point."""

    def __init__(self, name="marginal_only"):
        super().__init__(name)

    def predict_mean(self, X):
        return jnp.zeros(X.shape[0])

    def predict_variance(self, X):
        return jnp.ones(X.shape[0])


class TestGaussianRandomFunction:
    def test_isinstance_hierarchy(self):
        grf = _ScalarGP()
        assert isinstance(grf, GaussianRandomFunction)
        assert isinstance(grf, RandomFunction)
        assert isinstance(grf, Distribution)
        assert isinstance(grf, SupportsMean)
        assert isinstance(grf, SupportsVariance)

    def test_stacked_points_give_a_multivariate_normal(self):
        dist = _ScalarGP()(jnp.ones((5, 2)))
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (5,)

    def test_a_multi_output_member_flattens_its_values(self):
        dist = _MultiOutputGRF()(jnp.ones((5, 2)))
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)
        np.testing.assert_allclose(dist._mean(), jnp.tile(jnp.array([2.0, 4.0]), 5), atol=1e-6)

    def test_one_value_is_a_normal(self):
        dist = _ScalarGP()(jnp.ones((1, 2)))
        assert isinstance(dist, Normal)
        assert dist.event_shape == (1,)

    def test_a_member_without_a_covariance_gives_independent_normals(self):
        dist = _MarginalOnlyGRF()(jnp.ones((5, 2)))
        assert isinstance(dist, Normal)
        assert dist.event_shape == (5,)
        np.testing.assert_allclose(dist._variance(), jnp.ones(5), atol=1e-6)

    def test_predict_covariance_raises_by_default(self):
        with pytest.raises(NotImplementedError, match="jointly"):
            _MarginalOnlyGRF().predict_covariance(jnp.ones((5, 2)))

    def test_a_scalar_input_stacks_no_points(self):
        with pytest.raises(ValueError, match="leading axis"):
            _ScalarGP()(jnp.asarray(1.0))

    def test_the_law_carries_the_label_and_the_output_component(self):
        grf = _ScalarGP(name="gp")
        dist = grf(jnp.ones((3, 2)))
        assert dist.name == "gp"
        assert list(dist.event_spec.components) == ["gp"]

    def test_the_mean_and_variance_are_functions(self):
        grf = _ScalarGP(variance=2.0, noise=0.0)
        X = jnp.ones((4, 2))
        np.testing.assert_allclose(np.asarray(mean(grf)(X)), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(np.asarray(variance(grf)(X)), np.full(4, 2.0), atol=1e-6)


class TestDeclarations:
    def test_both_components_default_to_the_label(self):
        grf = _MarginalOnlyGRF("f")
        assert grf.event_spec == OutputSpec(f=FunctionSpec(output_spec=OutputSpec(f=None)))

    def test_output_spec_names_the_evaluated_component(self):
        process = GaussianProcess(
            "f", lambda X: jnp.zeros(X.shape[0]), _rbf_kernel, output_spec=OutputSpec(y=None)
        )
        law = process(jnp.ones((3, 2)))
        assert law.name == "f"
        assert list(law.event_spec.components) == ["y"]
        assert list(process.event_spec.components) == ["f"]

    def test_event_spec_names_the_event_and_its_hole_is_the_function(self):
        process = GaussianProcess(
            "f", lambda X: jnp.zeros(X.shape[0]), _rbf_kernel, event_spec=OutputSpec(g=None)
        )
        assert process.event_spec == OutputSpec(g=FunctionSpec(output_spec=OutputSpec(f=None)))

    def test_an_event_that_is_not_a_function_raises(self):
        with pytest.raises(TypeError, match="FunctionSpec"):
            GaussianProcess(
                "f",
                lambda X: jnp.zeros(X.shape[0]),
                _rbf_kernel,
                event_spec=OutputSpec(g=NumericArraySpec(())),
            )

    def test_an_event_function_naming_another_output_raises(self):
        with pytest.raises(ValueError, match="names the output"):
            GaussianProcess(
                "f",
                lambda X: jnp.zeros(X.shape[0]),
                _rbf_kernel,
                output_spec=OutputSpec(y=None),
                event_spec=OutputSpec(g=FunctionSpec(output_spec=OutputSpec(z=None))),
            )

    def test_an_event_function_naming_the_output_is_kept(self):
        event = OutputSpec(g=FunctionSpec(output_spec=OutputSpec(y=None)))
        process = GaussianProcess(
            "f",
            lambda X: jnp.zeros(X.shape[0]),
            _rbf_kernel,
            output_spec=OutputSpec(y=None),
            event_spec=event,
        )
        assert process.event_spec == event

    def test_an_event_function_without_an_output_takes_the_output_declaration(self):
        process = GaussianProcess(
            "f",
            lambda X: jnp.zeros(X.shape[0]),
            _rbf_kernel,
            output_spec=OutputSpec(y=None),
            event_spec=OutputSpec(g=FunctionSpec()),
        )
        assert process.event_spec == OutputSpec(g=FunctionSpec(output_spec=OutputSpec(y=None)))

    def test_a_record_output_declaration_raises(self):
        from probpipe import NumericRecordSpec

        with pytest.raises(TypeError, match="one component"):
            GaussianProcess(
                "f",
                lambda X: jnp.zeros(X.shape[0]),
                _rbf_kernel,
                output_spec=OutputSpec(NumericRecordSpec(a=(), b=())),
            )


# ---------------------------------------------------------------------------
# The Gaussian process
# ---------------------------------------------------------------------------


def _rbf(x, y):
    return jnp.exp(-0.5 * (x[:, None, 0] - y[None, :, 0]) ** 2)


class TestTheGaussianProcess:
    def test_evaluation_at_stacked_points_is_a_multivariate_normal(self):
        process = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf)
        X = jnp.array([[0.0], [0.5], [2.0]])
        law = process(X)
        assert isinstance(law, MultivariateNormal)
        np.testing.assert_allclose(law._mean(), jnp.zeros(3), atol=1e-6)
        np.testing.assert_allclose(law._cov().to_dense(), _rbf(X, X), rtol=1e-5)

    def test_evaluation_at_one_point_is_a_normal(self):
        process = GaussianProcess("f", lambda X: jnp.ones(X.shape[0]), _rbf)
        assert isinstance(process(jnp.array([[0.0]])), Normal)

    def test_the_components_default_to_the_label(self):
        process = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf)
        assert list(process.event_spec.components) == ["f"]

    def test_the_variance_is_the_kernel_diagonal(self):
        process = GaussianProcess(
            "f", lambda X: jnp.zeros(X.shape[0]), lambda X, Y: _rbf_kernel(X, Y, variance=2.5)
        )
        X = jnp.array([[0.0, 1.0], [2.0, 3.0], [-1.0, 0.5]])
        np.testing.assert_allclose(
            process.predict_variance(X), np.diag(_dense(process.predict_covariance(X))), atol=1e-6
        )
        np.testing.assert_allclose(process.predict_variance(X), np.full(3, 2.5), atol=1e-6)

    def test_the_mean_is_the_mean_function(self):
        process = GaussianProcess("f", lambda X: jnp.sum(X, axis=-1), _rbf_kernel)
        X = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_allclose(process.predict_mean(X), [1.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(np.asarray(mean(process)(X)), [1.0, 5.0], atol=1e-6)

    def test_a_process_does_not_sample_whole_functions(self):
        process = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf)
        assert not isinstance(process, SupportsSampling)

    def test_its_parts_are_callables(self):
        with pytest.raises(TypeError, match="callables"):
            GaussianProcess("f", jnp.zeros(3), _rbf)

    def test_a_sum_of_processes_adds_the_kernels(self):
        first = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf_kernel)
        second = GaussianProcess(
            "g", lambda X: jnp.ones(X.shape[0]), lambda X, Y: _rbf_kernel(X, Y, lengthscale=0.5)
        )
        X = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]])
        law = (first + second)(X)
        expected = _rbf_kernel(X, X) + _rbf_kernel(X, X, lengthscale=0.5)
        np.testing.assert_allclose(law._cov().to_dense(), expected, rtol=1e-5)
        np.testing.assert_allclose(law._mean(), jnp.ones(3), atol=1e-6)


# ---------------------------------------------------------------------------
# The basis-function model
# ---------------------------------------------------------------------------


def _polynomial_basis(X):
    """The features [1, x, x²] of a scalar input."""
    return jnp.stack([jnp.ones_like(X[..., 0]), X[..., 0], X[..., 0] ** 2], axis=-1)


def _multi_output_basis(X):
    """The features of a two-output model, of shape (n, 2, 2)."""
    x = X[..., 0]
    phi = jnp.stack([jnp.ones_like(x), x], axis=-1)
    return jnp.stack([phi, 0.5 * phi], axis=-2)


@pytest.fixture
def scalar_lbf():
    weights = MultivariateNormal("weights", jnp.array([1.0, 0.5, 0.1]), cov=0.01 * jnp.eye(3))
    return LinearBasisFunction("f", _polynomial_basis, weights)


@pytest.fixture
def multi_output_lbf():
    weights = MultivariateNormal("weights", jnp.array([1.0, 0.5]), cov=0.01 * jnp.eye(2))
    return LinearBasisFunction("f", _multi_output_basis, weights)


class TestLinearBasisFunction:
    def test_isinstance_hierarchy(self, scalar_lbf):
        assert isinstance(scalar_lbf, LinearBasisFunction)
        assert isinstance(scalar_lbf, GaussianRandomFunction)
        assert isinstance(scalar_lbf, RandomFunction)
        assert isinstance(scalar_lbf, SupportsSampling)

    def test_evaluation_is_the_joint_law(self, scalar_lbf, key):
        X = jnp.linspace(-1, 1, 10).reshape(-1, 1)
        dist = scalar_lbf(X)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)
        weights = scalar_lbf._weights
        _assert_the_joint_law(dist, _polynomial_basis(X), weights.loc, weights.cov, key)

    def test_a_multi_output_evaluation_is_the_flattened_joint_law(self, multi_output_lbf, key):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = multi_output_lbf(X)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)
        weights = multi_output_lbf._weights
        _assert_the_joint_law(dist, _multi_output_basis(X), weights.loc, weights.cov, key)

    def test_the_mean_value(self, scalar_lbf):
        dist = scalar_lbf(jnp.array([[0.0], [1.0]]))
        # At x=0 the features are [1, 0, 0], and at x=1 they are [1, 1, 1].
        np.testing.assert_allclose(mean(dist), [1.0, 1.6], atol=1e-5)

    def test_the_covariance_shapes(self, scalar_lbf, multi_output_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        assert _dense(scalar_lbf.predict_covariance(X)).shape == (5, 5)
        assert _dense(multi_output_lbf.predict_covariance(X)).shape == (10, 10)

    def test_a_basis_of_the_wrong_width_raises(self):
        weights = MultivariateNormal("weights", jnp.zeros(2), cov=jnp.eye(2))
        lbf = LinearBasisFunction("f", _polynomial_basis, weights)
        with pytest.raises(ValueError, match="weights' dimension 2"):
            lbf.predict_mean(jnp.ones((3, 1)))

    # -- Drawing functions -------------------------------------------------

    def test_sample_single(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key)
        assert callable(f)
        assert f(jnp.linspace(-1, 1, 10).reshape(-1, 1)).shape == (10,)

    def test_sample_batched(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key, sample_shape=(7,))
        assert callable(f)
        assert f(jnp.linspace(-1, 1, 10).reshape(-1, 1)).shape == (7, 10)

    def test_sample_multi_dim_shape(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key, sample_shape=(3, 4))
        assert f(jnp.linspace(-1, 1, 5).reshape(-1, 1)).shape == (3, 4, 5)

    def test_sample_consistency(self, key, scalar_lbf):
        """One drawn function takes one value at a point, whatever else it is evaluated at."""
        f = sample(scalar_lbf, key=key)
        y1 = f(jnp.array([[0.0], [1.0]]))
        y2 = f(jnp.array([[0.0], [2.0]]))
        np.testing.assert_allclose(y1[0], y2[0], atol=1e-6)

    def test_sample_batched_consistency(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key, sample_shape=(5,))
        np.testing.assert_allclose(f(jnp.array([[0.0]])), f(jnp.array([[0.0]])), atol=1e-6)

    def test_sample_multi_output(self, key, multi_output_lbf):
        f = sample(multi_output_lbf, key=key)
        assert f(jnp.linspace(-1, 1, 8).reshape(-1, 1)).shape == (8, 2)

    def test_sample_batched_multi_output(self, key, multi_output_lbf):
        f = sample(multi_output_lbf, key=key, sample_shape=(5,))
        assert f(jnp.linspace(-1, 1, 8).reshape(-1, 1)).shape == (5, 8, 2)

    # -- Validation --------------------------------------------------------

    def test_invalid_weights_type(self):
        with pytest.raises(TypeError, match="MultivariateNormal"):
            LinearBasisFunction("f", _polynomial_basis, "not_a_distribution")

    def test_the_basis_is_a_callable(self):
        weights = MultivariateNormal("weights", jnp.zeros(2), cov=jnp.eye(2))
        with pytest.raises(TypeError, match="callable"):
            LinearBasisFunction("f", jnp.ones((2, 2)), weights)


# ---------------------------------------------------------------------------
# The closed-form algebra
# ---------------------------------------------------------------------------


def _weight_basis(X):
    """The features of a three-output model with two basis functions, of shape (n, 3, 2)."""
    x = X[..., 0]
    features = jnp.stack([jnp.ones_like(x), x], axis=-1)
    return jnp.stack([features, 0.5 * features, 2.0 * features], axis=-2)


@pytest.fixture
def weight_grf():
    """A member whose value at each point is a vector of size 3."""
    weights = MultivariateNormal("weights", jnp.array([1.0, 0.5]), cov=0.01 * jnp.eye(2))
    return LinearBasisFunction("f", _weight_basis, weights)


class TestLinearMap:
    def test_isinstance_hierarchy(self, weight_grf):
        h = jnp.eye(3, 3) @ weight_grf
        assert isinstance(h, GaussianRandomFunction)
        assert isinstance(h, RandomFunction)

    def test_the_value_shape(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        assert h.predict_mean(X).shape == (5, 2)
        assert h.predict_variance(X).shape == (5, 2)

    def test_evaluation_is_the_flattened_joint_law(self, weight_grf, key):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = (A @ weight_grf)(X)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)
        # The basis of A g is A Φ_g(x).
        phi = jnp.einsum("od,ndw->now", A, _weight_basis(X))
        weights = weight_grf._weights
        _assert_the_joint_law(dist, phi, weights.loc, weights.cov, key)

    def test_mean_value(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.array([[0.5]])
        expected = jnp.einsum("ow,...w->...o", A, weight_grf.predict_mean(X))
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_the_variance_is_the_joint_diagonal(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 3).reshape(-1, 1)
        np.testing.assert_allclose(
            np.asarray(h.predict_variance(X)).reshape(-1),
            np.diag(_dense(h.predict_covariance(X))),
            atol=1e-6,
        )

    def test_a_scalar_output_raises(self, scalar_lbf):
        h = jnp.eye(2) @ scalar_lbf
        with pytest.raises(ValueError, match="output vector of size 2"):
            h.predict_mean(jnp.ones((3, 1)))

    def test_a_size_mismatch_raises(self, weight_grf):
        h = jnp.eye(2) @ weight_grf
        with pytest.raises(ValueError, match=r"has shape \(3,\)"):
            h(jnp.ones((3, 1)))

    def test_the_map_is_a_matrix(self, weight_grf):
        with pytest.raises(ValueError, match="2-D"):
            jnp.ones(3) @ weight_grf

    def test_a_base_without_a_covariance_has_independent_outputs(self):
        class _MarginalVector(GaussianRandomFunction):
            def predict_mean(self, X):
                return jnp.zeros((X.shape[0], 2))

            def predict_variance(self, X):
                return jnp.stack([jnp.ones(X.shape[0]), 4.0 * jnp.ones(X.shape[0])], -1)

        h = jnp.array([[1.0, 1.0]]) @ _MarginalVector("v")
        X = jnp.ones((3, 1))
        np.testing.assert_allclose(h.predict_variance(X), np.full((3, 1), 5.0), atol=1e-6)
        assert isinstance(h(X), Normal)


class TestShift:
    def test_mean_shifted(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        b = jnp.float32(10.0)
        h = scalar_lbf + b
        np.testing.assert_allclose(h.predict_mean(X), scalar_lbf.predict_mean(X) + b, atol=1e-5)

    def test_variance_unchanged(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = scalar_lbf + 10.0
        np.testing.assert_allclose(h.predict_variance(X), scalar_lbf.predict_variance(X), atol=1e-6)

    def test_covariance_unchanged(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        h = scalar_lbf + 10.0
        np.testing.assert_allclose(
            _dense(h.predict_covariance(X)), _dense(scalar_lbf.predict_covariance(X)), atol=1e-6
        )

    def test_the_declaration_is_the_base_declaration(self, scalar_lbf):
        h = scalar_lbf + 1.0
        assert h.event_spec is scalar_lbf.event_spec

    def test_radd(self, scalar_lbf):
        X = jnp.array([[0.0]])
        h = 5.0 + scalar_lbf
        np.testing.assert_allclose(h.predict_mean(X), scalar_lbf.predict_mean(X) + 5.0, atol=1e-5)

    def test_sub(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = scalar_lbf - 3.0
        np.testing.assert_allclose(h.predict_mean(X), scalar_lbf.predict_mean(X) - 3.0, atol=1e-5)


class TestScale:
    def test_mean_scaled(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = 3.0 * scalar_lbf
        np.testing.assert_allclose(h.predict_mean(X), 3.0 * scalar_lbf.predict_mean(X), atol=1e-5)

    def test_variance_scaled_squared(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = 3.0 * scalar_lbf
        np.testing.assert_allclose(
            h.predict_variance(X), 9.0 * scalar_lbf.predict_variance(X), atol=1e-5
        )

    def test_covariance_scaled_squared(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        h = 2.0 * scalar_lbf
        np.testing.assert_allclose(
            _dense(h.predict_covariance(X)),
            4.0 * _dense(scalar_lbf.predict_covariance(X)),
            atol=1e-5,
        )

    def test_rmul(self, scalar_lbf):
        X = jnp.array([[0.0]])
        h = scalar_lbf * 2.0
        np.testing.assert_allclose(h.predict_mean(X), 2.0 * scalar_lbf.predict_mean(X), atol=1e-5)

    def test_neg(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = -scalar_lbf
        np.testing.assert_allclose(h.predict_mean(X), -scalar_lbf.predict_mean(X), atol=1e-5)
        np.testing.assert_allclose(h.predict_variance(X), scalar_lbf.predict_variance(X), atol=1e-6)

    def test_a_scaling_is_by_a_scalar(self, weight_grf):
        with pytest.raises(ValueError, match="scalar"):
            jnp.array([1.0, 2.0, 3.0]) * weight_grf

    def test_a_traced_scalar_differentiates_and_compiles(self):
        process = GaussianProcess("g", lambda X: jnp.zeros(X.shape[0]), _rbf_kernel)
        X = jnp.array([[0.0], [0.5], [1.0]])
        kernel_sum = float(jnp.sum(_rbf_kernel(X, X)))

        def total(alpha):
            return jnp.sum((alpha * process).predict_covariance(X).to_dense())

        # d/dα of Σ α² K is 2 α Σ K.
        for differentiate in (jax.grad(total), jax.jit(jax.grad(total))):
            assert float(differentiate(2.0)) == pytest.approx(4.0 * kernel_sum, rel=1e-5)
        assert float(jax.jit(total)(2.0)) == pytest.approx(4.0 * kernel_sum, rel=1e-5)

    def test_the_law_of_a_traced_scaling_differentiates(self):
        process = GaussianProcess("g", lambda X: jnp.zeros(X.shape[0]), _rbf_kernel)
        X = jnp.array([[0.0], [0.5], [1.0]])
        y = jnp.array([0.1, 0.2, -0.1])

        def log_density(alpha):
            return (alpha * process)(X)._log_prob(y)

        # log N(y; 0, α² K) has the derivative yᵀ K⁻¹ y / α³ - n / α.
        quadratic = float(y @ jnp.linalg.solve(_rbf_kernel(X, X), y))
        expected = quadratic / 8.0 - 3.0 / 2.0
        for differentiate in (jax.grad(log_density), jax.jit(jax.grad(log_density))):
            assert float(differentiate(2.0)) == pytest.approx(expected, rel=1e-3)


class TestIndependentSum:
    def test_mean_is_sum(self):
        gp1, gp2 = _ScalarGP(1.0, 1.0, name="a"), _ScalarGP(0.5, 0.5, name="b")
        h = gp1 + gp2
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_mean(X), gp1.predict_mean(X) + gp2.predict_mean(X), atol=1e-6
        )

    def test_variance_is_sum(self):
        gp1, gp2 = _ScalarGP(1.0, 1.0, name="a"), _ScalarGP(0.5, 0.5, name="b")
        h = gp1 + gp2
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_variance(X), gp1.predict_variance(X) + gp2.predict_variance(X), atol=1e-6
        )

    def test_covariance_is_sum(self):
        gp1, gp2 = _ScalarGP(1.0, 1.0, name="a"), _ScalarGP(0.5, 0.5, name="b")
        h = gp1 + gp2
        X = jnp.stack([jnp.linspace(-1, 1, 5), jnp.zeros(5)], axis=-1)
        np.testing.assert_allclose(
            _dense(h.predict_covariance(X)),
            _dense(gp1.predict_covariance(X)) + _dense(gp2.predict_covariance(X)),
            atol=1e-5,
        )

    def test_a_sum_with_a_marginal_member_gives_independent_normals(self):
        h = _ScalarGP() + _MarginalOnlyGRF()
        assert isinstance(h(jnp.ones((5, 2))), Normal)

    def test_same_object_raises(self):
        gp = _ScalarGP()
        with pytest.raises(ValueError, match="itself"):
            gp + gp

    def test_a_shape_mismatch_raises(self):
        h = _ScalarGP() + _MultiOutputGRF()
        with pytest.raises(ValueError, match="one shape"):
            h.predict_mean(jnp.ones((3, 2)))

    def test_a_shape_mismatch_raises_for_the_variance_as_for_the_mean(self):
        # At two points the scalar member's variance, (2,), broadcasts against (2, 2).
        h = _ScalarGP() + _MultiOutputGRF()
        X = jnp.ones((2, 2))
        with pytest.raises(ValueError, match="one shape"):
            h.predict_mean(X)
        with pytest.raises(ValueError, match="one shape"):
            h.predict_variance(X)

    def test_sub_grfs(self):
        gp1, gp2 = _ScalarGP(1.0, 1.0, name="a"), _ScalarGP(0.5, 0.5, name="b")
        h = gp1 - gp2
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_mean(X), gp1.predict_mean(X) - gp2.predict_mean(X), atol=1e-6
        )


class TestAlgebraComposition:
    def test_affine_transform(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        b = jnp.array([0.1, -0.2])
        h = A @ weight_grf + b
        X = jnp.array([[0.5]])
        expected = jnp.einsum("ow,...w->...o", A, weight_grf.predict_mean(X)) + b
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_scale_then_shift(self, scalar_lbf):
        h = 2.0 * scalar_lbf + 5.0
        X = jnp.array([[1.0]])
        np.testing.assert_allclose(
            h.predict_mean(X), 2.0 * scalar_lbf.predict_mean(X) + 5.0, atol=1e-5
        )

    def test_scale_sum(self):
        gp1, gp2 = _ScalarGP(1.0, 1.0, name="a"), _ScalarGP(0.5, 0.5, name="b")
        h = 3.0 * (gp1 + gp2)
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_mean(X), 3.0 * (gp1.predict_mean(X) + gp2.predict_mean(X)), atol=1e-5
        )
        np.testing.assert_allclose(
            h.predict_variance(X),
            9.0 * (gp1.predict_variance(X) + gp2.predict_variance(X)),
            atol=1e-5,
        )


def _named_weight_grf(name):
    weights = MultivariateNormal("weights", jnp.array([1.0, 0.5]), cov=0.01 * jnp.eye(2))
    return LinearBasisFunction(name, _weight_basis, weights)


class TestAlgebraNames:
    """A result of the algebra is named from its operands."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(lambda f, g: jnp.eye(3) @ f, "linear_map(f)", id="linear-map"),
            pytest.param(lambda f, g: f + 1.0, "shift(f)", id="shift"),
            pytest.param(lambda f, g: 2.0 * f, "scale(f)", id="scale"),
            pytest.param(lambda f, g: f + g, "sum(f,g)", id="sum"),
            pytest.param(lambda f, g: (f + g) + f, "sum(sum(f,g),f)", id="nested"),
        ],
    )
    def test_a_result_is_named_from_its_operands(self, build, expected):
        assert build(_named_weight_grf("f"), _named_weight_grf("g")).name == expected


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
#
# Each test builds a basis-function model with a known basis and weight law,
# evaluates the basis at the test points, computes the moments densely, and
# compares them with the member's predictions. The Monte Carlo tests compare
# the predictions with the moments of drawn functions.


@pytest.fixture
def correctness_X():
    return jnp.array([[-1.0], [0.0], [0.5], [1.0]])


@pytest.fixture
def correctness_w_mean():
    return jnp.array([2.0, -1.0])


@pytest.fixture
def correctness_w_cov():
    """A weight covariance with cross-correlations."""
    return jnp.array([[1.0, 0.3], [0.3, 0.5]])


def _simple_multi_output_features(X):
    """The features [1, x], [x, x²], and [1, -x] of three outputs, of shape (n, 3, 2).

    The outputs are correlated through the shared weights.
    """
    x = X[..., 0]
    ones = jnp.ones_like(x)
    return jnp.stack(
        [
            jnp.stack([ones, x], axis=-1),
            jnp.stack([x, x**2], axis=-1),
            jnp.stack([ones, -x], axis=-1),
        ],
        axis=-2,
    )


@pytest.fixture
def correlated_lbf(correctness_w_mean, correctness_w_cov):
    weights = MultivariateNormal("weights", correctness_w_mean, cov=correctness_w_cov)
    return LinearBasisFunction("f", _simple_multi_output_features, weights)


def _simple_scalar_features(X):
    """The features [1, x, x²] of a scalar input."""
    x = X[..., 0]
    return jnp.stack([jnp.ones_like(x), x, x**2], axis=-1)


_SCALAR_W_MEAN = np.array([1.0, -0.5, 0.2])
_SCALAR_W_COV = np.array([[1.0, 0.2, -0.1], [0.2, 0.8, 0.05], [-0.1, 0.05, 0.3]])


@pytest.fixture
def scalar_correctness_lbf():
    weights = MultivariateNormal(
        "weights", jnp.asarray(_SCALAR_W_MEAN), cov=jnp.asarray(_SCALAR_W_COV)
    )
    return LinearBasisFunction("f", _simple_scalar_features, weights)


def _blocks(cov, n, d_out):
    """The per-point output blocks of a joint covariance over n points of d_out values."""
    return np.einsum("ioip->iop", np.asarray(cov).reshape(n, d_out, n, d_out))


class TestLinearBasisFunctionCorrectness:
    def test_mean_scalar(self, scalar_correctness_lbf, correctness_X):
        phi = np.array(_simple_scalar_features(correctness_X))
        np.testing.assert_allclose(
            scalar_correctness_lbf.predict_mean(correctness_X), phi @ _SCALAR_W_MEAN, atol=1e-5
        )

    def test_variance_equals_cov_diag_scalar(self, scalar_correctness_lbf, correctness_X):
        var = scalar_correctness_lbf.predict_variance(correctness_X)
        cov = _dense(scalar_correctness_lbf.predict_covariance(correctness_X))
        np.testing.assert_allclose(var, np.diag(cov), atol=1e-5)

    def test_joint_cov_scalar(self, scalar_correctness_lbf, correctness_X):
        phi = np.array(_simple_scalar_features(correctness_X))
        np.testing.assert_allclose(
            _dense(scalar_correctness_lbf.predict_covariance(correctness_X)),
            phi @ _SCALAR_W_COV @ phi.T,
            atol=1e-5,
        )

    def test_the_law_at_the_points_has_the_dense_moments(
        self, scalar_correctness_lbf, correctness_X
    ):
        phi = np.array(_simple_scalar_features(correctness_X))
        law = scalar_correctness_lbf(correctness_X)
        np.testing.assert_allclose(law._mean(), phi @ _SCALAR_W_MEAN, atol=1e-5)
        np.testing.assert_allclose(law._cov().to_dense(), phi @ _SCALAR_W_COV @ phi.T, atol=1e-5)

    def test_mean_multi_output(self, correlated_lbf, correctness_X, correctness_w_mean):
        phi = np.array(_simple_multi_output_features(correctness_X))
        expected = np.einsum("now,w->no", phi, np.array(correctness_w_mean))
        np.testing.assert_allclose(correlated_lbf.predict_mean(correctness_X), expected, atol=1e-5)

    def test_full_joint_cov_multi_output(self, correlated_lbf, correctness_X, correctness_w_cov):
        phi = np.array(_simple_multi_output_features(correctness_X))
        expected = _dense_joint_cov(phi, np.array(correctness_w_cov))
        np.testing.assert_allclose(
            _dense(correlated_lbf.predict_covariance(correctness_X)), expected, atol=1e-5
        )

    def test_variance_equals_diag_of_per_point_blocks(self, correlated_lbf, correctness_X):
        var = correlated_lbf.predict_variance(correctness_X)
        blocks = _blocks(_dense(correlated_lbf.predict_covariance(correctness_X)), 4, 3)
        for i in range(4):
            np.testing.assert_allclose(var[i], np.diag(blocks[i]), atol=1e-5)


class TestLinearMapCorrectness:
    """``h(x) = A g(x)`` for a multi-output basis-function model ``g``."""

    A = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])

    def test_mean_ground_truth(self, correlated_lbf, correctness_X, correctness_w_mean):
        h = jnp.array(self.A) @ correlated_lbf
        phi_g = np.array(_simple_multi_output_features(correctness_X))
        g_mean = np.einsum("now,w->no", phi_g, np.array(correctness_w_mean))
        expected = np.einsum("oh,...h->...o", self.A, g_mean)
        np.testing.assert_allclose(h.predict_mean(correctness_X), expected, atol=1e-5)

    def test_per_point_cov_ground_truth(self, correlated_lbf, correctness_X, correctness_w_cov):
        """The output covariance at a point is A Φ(x) C Φ(x)ᵀ Aᵀ."""
        h = jnp.array(self.A) @ correlated_lbf
        phi_g = np.array(_simple_multi_output_features(correctness_X))
        C = np.array(correctness_w_cov)
        blocks = _blocks(_dense(h.predict_covariance(correctness_X)), 4, 2)
        var = h.predict_variance(correctness_X)
        for i in range(4):
            phi_h_i = self.A @ phi_g[i]
            expected_i = phi_h_i @ C @ phi_h_i.T
            np.testing.assert_allclose(blocks[i], expected_i, atol=1e-5)
            np.testing.assert_allclose(var[i], np.diag(expected_i), atol=1e-5)

    def test_full_joint_cov_ground_truth(self, correlated_lbf, correctness_X, correctness_w_cov):
        h = jnp.array(self.A) @ correlated_lbf
        phi_g = np.array(_simple_multi_output_features(correctness_X))
        phi_h = np.einsum("oh,nhw->now", self.A, phi_g)
        expected = _dense_joint_cov(phi_h, np.array(correctness_w_cov))
        np.testing.assert_allclose(_dense(h.predict_covariance(correctness_X)), expected, atol=1e-5)

    def test_cross_output_correlation_enters_the_covariance(self, correctness_w_cov):
        """Outputs 0 and 2 are correlated through the weights, so A = [1, 0, 1] needs it."""
        weights = MultivariateNormal("weights", jnp.array([2.0, -1.0]), cov=correctness_w_cov)
        base = LinearBasisFunction("base", _simple_multi_output_features, weights)
        A_np = np.array([[1.0, 0.0, 1.0]])
        h = jnp.array(A_np) @ base
        X = jnp.array([[-1.0], [1.0]])
        phi_g = np.array(_simple_multi_output_features(X))
        C = np.array(correctness_w_cov)
        correct = _dense_joint_cov(np.einsum("oh,nhw->now", A_np, phi_g), C)
        # Ignoring the cross-output covariance keeps only each output's own block.
        per_output = np.stack([phi_g[:, o, :] @ C @ phi_g[:, o, :].T for o in range(3)])
        ignored = np.einsum("ow,wij->oij", A_np**2, per_output)[0]
        assert not np.allclose(correct, ignored, atol=1e-3)
        np.testing.assert_allclose(_dense(h.predict_covariance(X)), correct, atol=1e-5)


class TestScaleCorrectness:
    def test_variance_ground_truth(self, scalar_correctness_lbf, correctness_X):
        alpha = 3.0
        h = alpha * scalar_correctness_lbf
        phi = np.array(_simple_scalar_features(correctness_X))
        expected = alpha**2 * np.diag(phi @ _SCALAR_W_COV @ phi.T)
        np.testing.assert_allclose(h.predict_variance(correctness_X), expected, atol=1e-5)

    def test_full_joint_cov_ground_truth(self, scalar_correctness_lbf, correctness_X):
        alpha = -2.5
        h = alpha * scalar_correctness_lbf
        phi = np.array(_simple_scalar_features(correctness_X))
        expected = alpha**2 * (phi @ _SCALAR_W_COV @ phi.T)
        np.testing.assert_allclose(_dense(h.predict_covariance(correctness_X)), expected, atol=1e-5)


class TestShiftCorrectness:
    def test_mean_ground_truth(self, scalar_correctness_lbf, correctness_X):
        h = scalar_correctness_lbf + 42.0
        phi = np.array(_simple_scalar_features(correctness_X))
        np.testing.assert_allclose(
            h.predict_mean(correctness_X), phi @ _SCALAR_W_MEAN + 42.0, atol=1e-5
        )

    def test_cov_unchanged_ground_truth(self, scalar_correctness_lbf, correctness_X):
        h = scalar_correctness_lbf + 999.0
        phi = np.array(_simple_scalar_features(correctness_X))
        np.testing.assert_allclose(
            _dense(h.predict_covariance(correctness_X)), phi @ _SCALAR_W_COV @ phi.T, atol=1e-5
        )


def _scalar_pair(C1, C2, m1=None, m2=None):
    w1 = MultivariateNormal(
        "w1", jnp.zeros(3) if m1 is None else jnp.asarray(m1), cov=jnp.asarray(C1)
    )
    w2 = MultivariateNormal(
        "w2", jnp.zeros(3) if m2 is None else jnp.asarray(m2), cov=jnp.asarray(C2)
    )
    return (
        LinearBasisFunction("lbf1", _simple_scalar_features, w1),
        LinearBasisFunction("lbf2", _simple_scalar_features, w2),
    )


_C1 = np.array([[1.0, 0.2, 0.0], [0.2, 0.5, 0.0], [0.0, 0.0, 0.3]])
_C2 = np.array([[0.5, 0.0, 0.1], [0.0, 0.8, 0.0], [0.1, 0.0, 0.2]])


class TestIndependentSumCorrectness:
    def test_mean_with_nonzero_means(self, correctness_X):
        m1, m2 = np.array([1.0, -0.5, 0.2]), np.array([-0.3, 0.8, 0.0])
        lbf1, lbf2 = _scalar_pair(0.1 * np.eye(3), 0.2 * np.eye(3), m1, m2)
        phi = np.array(_simple_scalar_features(correctness_X))
        np.testing.assert_allclose(
            (lbf1 + lbf2).predict_mean(correctness_X), phi @ m1 + phi @ m2, atol=1e-5
        )

    def test_variance_ground_truth(self, correctness_X):
        lbf1, lbf2 = _scalar_pair(_C1, _C2)
        phi = np.array(_simple_scalar_features(correctness_X))
        expected = np.diag(phi @ _C1 @ phi.T) + np.diag(phi @ _C2 @ phi.T)
        np.testing.assert_allclose(
            (lbf1 + lbf2).predict_variance(correctness_X), expected, atol=1e-5
        )

    def test_covariance_ground_truth(self, correctness_X):
        lbf1, lbf2 = _scalar_pair(_C1, _C2)
        phi = np.array(_simple_scalar_features(correctness_X))
        expected = phi @ _C1 @ phi.T + phi @ _C2 @ phi.T
        np.testing.assert_allclose(
            _dense((lbf1 + lbf2).predict_covariance(correctness_X)), expected, atol=1e-5
        )


class TestCompositionCorrectness:
    A = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])

    def test_affine_full_joint_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_mean, correctness_w_cov
    ):
        """``A g + b``: the shift moves the mean and leaves the covariance."""
        b = jnp.array([10.0, -5.0])
        h = jnp.array(self.A) @ correlated_lbf + b
        phi_g = np.array(_simple_multi_output_features(correctness_X))
        g_mean = np.einsum("now,w->no", phi_g, np.array(correctness_w_mean))
        expected_mean = np.einsum("oh,...h->...o", self.A, g_mean) + np.array(b)
        np.testing.assert_allclose(h.predict_mean(correctness_X), expected_mean, atol=1e-5)
        phi_h = np.einsum("oh,nhw->now", self.A, phi_g)
        expected_cov = _dense_joint_cov(phi_h, np.array(correctness_w_cov))
        np.testing.assert_allclose(
            _dense(h.predict_covariance(correctness_X)), expected_cov, atol=1e-5
        )

    def test_scale_of_linear_map_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        alpha = 2.5
        h = alpha * (jnp.array(self.A) @ correlated_lbf)
        phi_g = np.array(_simple_multi_output_features(correctness_X))
        phi_h = np.einsum("oh,nhw->now", self.A, phi_g)
        expected = alpha**2 * _dense_joint_cov(phi_h, np.array(correctness_w_cov))
        np.testing.assert_allclose(_dense(h.predict_covariance(correctness_X)), expected, atol=1e-4)


class TestMonteCarlo:
    """The moments of drawn functions match the predictions."""

    N_SAMPLES = 20_000
    MC_ATOL_MEAN = 0.05
    MC_ATOL_VAR = 0.25

    def _sample_outputs(self, lbf, X, key, n_samples):
        f = sample(lbf, key=key, sample_shape=(n_samples,))
        return np.array(f(X))

    def test_linear_basis_function_moments(self, scalar_correctness_lbf, correctness_X):
        Y = self._sample_outputs(
            scalar_correctness_lbf, correctness_X, jax.random.PRNGKey(123), self.N_SAMPLES
        )
        np.testing.assert_allclose(
            Y.mean(axis=0),
            np.array(scalar_correctness_lbf.predict_mean(correctness_X)),
            atol=self.MC_ATOL_MEAN,
        )
        np.testing.assert_allclose(
            np.cov(Y, rowvar=False),
            _dense(scalar_correctness_lbf.predict_covariance(correctness_X)),
            atol=self.MC_ATOL_VAR,
        )

    def test_linear_map_moments(self, correlated_lbf, correctness_X):
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        h = jnp.array(A_np) @ correlated_lbf
        Y_base = self._sample_outputs(
            correlated_lbf, correctness_X, jax.random.PRNGKey(456), self.N_SAMPLES
        )
        Y_h = np.einsum("oh,...h->...o", A_np, Y_base)
        np.testing.assert_allclose(
            Y_h.mean(axis=0), np.array(h.predict_mean(correctness_X)), atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(
            Y_h.var(axis=0), np.array(h.predict_variance(correctness_X)), atol=self.MC_ATOL_VAR
        )

    def test_linear_map_full_joint_cov(self, correlated_lbf, correctness_X):
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        h = jnp.array(A_np) @ correlated_lbf
        Y_base = self._sample_outputs(
            correlated_lbf, correctness_X, jax.random.PRNGKey(789), self.N_SAMPLES
        )
        Y_flat = np.einsum("oh,...h->...o", A_np, Y_base).reshape(self.N_SAMPLES, -1)
        np.testing.assert_allclose(
            np.cov(Y_flat, rowvar=False),
            _dense(h.predict_covariance(correctness_X)),
            atol=self.MC_ATOL_VAR,
        )

    def test_scale_moments(self, scalar_correctness_lbf, correctness_X):
        alpha = 3.0
        h = alpha * scalar_correctness_lbf
        Y_h = alpha * self._sample_outputs(
            scalar_correctness_lbf, correctness_X, jax.random.PRNGKey(101), self.N_SAMPLES
        )
        np.testing.assert_allclose(
            Y_h.mean(axis=0), np.array(h.predict_mean(correctness_X)), atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(
            Y_h.var(axis=0), np.array(h.predict_variance(correctness_X)), atol=self.MC_ATOL_VAR
        )

    def test_shift_moments(self, scalar_correctness_lbf, correctness_X):
        h = scalar_correctness_lbf + 42.0
        Y_h = 42.0 + self._sample_outputs(
            scalar_correctness_lbf, correctness_X, jax.random.PRNGKey(202), self.N_SAMPLES
        )
        np.testing.assert_allclose(
            Y_h.mean(axis=0), np.array(h.predict_mean(correctness_X)), atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(
            Y_h.var(axis=0), np.array(h.predict_variance(correctness_X)), atol=self.MC_ATOL_VAR
        )

    def test_independent_sum_moments(self, correctness_X):
        lbf1, lbf2 = _scalar_pair(
            0.5 * np.eye(3), 0.3 * np.eye(3), [1.0, -0.5, 0.2], [-0.3, 0.8, 0.0]
        )
        h = lbf1 + lbf2
        key1, key2 = jax.random.split(jax.random.PRNGKey(303))
        Y_h = self._sample_outputs(lbf1, correctness_X, key1, self.N_SAMPLES) + (
            self._sample_outputs(lbf2, correctness_X, key2, self.N_SAMPLES)
        )
        np.testing.assert_allclose(
            Y_h.mean(axis=0), np.array(h.predict_mean(correctness_X)), atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(
            np.cov(Y_h, rowvar=False),
            _dense(h.predict_covariance(correctness_X)),
            atol=self.MC_ATOL_VAR,
        )


class TestRBFKernelBaseline:
    """The test kernel is the squared exponential ``var · exp(-‖x − y‖² / (2 ℓ²))``."""

    def test_rbf_matches_scipy_unit(self):
        from scipy.spatial.distance import cdist

        X = np.random.default_rng(0).standard_normal((7, 2)).astype(np.float32)
        K = np.asarray(_rbf_kernel(jnp.asarray(X), jnp.asarray(X)))
        np.testing.assert_allclose(K, np.exp(-0.5 * cdist(X, X, "sqeuclidean")), atol=1e-5)

    def test_rbf_matches_scipy_with_hyperparams(self):
        from scipy.spatial.distance import cdist

        X = np.random.default_rng(1).standard_normal((5, 3)).astype(np.float32)
        K = np.asarray(_rbf_kernel(jnp.asarray(X), jnp.asarray(X), lengthscale=2.0, variance=3.5))
        expected = 3.5 * np.exp(-0.5 * cdist(X, X, "sqeuclidean") / 4.0)
        np.testing.assert_allclose(K, expected, atol=1e-5)

    def test_rbf_diagonal_equals_variance(self):
        X = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
        K = _rbf_kernel(X, X, variance=2.5)
        np.testing.assert_allclose(np.diag(np.asarray(K)), [2.5, 2.5], atol=1e-6)


# ---------------------------------------------------------------------------
# The factored Gaussian joint
# ---------------------------------------------------------------------------


def test_the_factored_gaussian_is_a_numeric_factored_joint():
    assert issubclass(FactoredMultivariateGaussian, FactoredNumericDistribution)


@pytest.mark.pending(
    reason="the linear-Gaussian conditional, the algebra's conditional member",
    raises=NotImplementedError,
)
def test_composition_derives_the_factored_gaussian_of_a_linear_gaussian_observation():
    prior = MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
    observation = LinearGaussianConditional(
        "y", DenseLinOp(jnp.array([[1.0, 0.0], [1.0, 1.0]])), jnp.zeros(2), DenseLinOp(jnp.eye(2))
    )
    assert isinstance(observation * prior, FactoredMultivariateGaussian)


@pytest.mark.pending(
    reason="the linear-Gaussian conditional, the algebra's conditional member",
    raises=NotImplementedError,
)
def test_conditioning_a_linear_gaussian_joint_on_its_observation_is_exact():
    prior = MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
    observation = LinearGaussianConditional(
        "y", DenseLinOp(jnp.eye(2)), jnp.zeros(2), DenseLinOp(jnp.eye(2))
    )
    posterior = (observation * prior)._condition_on({"y": jnp.array([1.0, -1.0])})
    np.testing.assert_allclose(np.asarray(posterior._mean()["beta"]), [0.5, -0.5], rtol=1e-6)


def _gaussian_joint() -> FactoredMultivariateGaussian:
    """``a ~ N(1, 2²)`` and ``b ~ N((0, 3), diag(1, 4))``, independent."""
    return Normal("a", 1.0, 2.0) * MultivariateNormal(
        "b", jnp.array([0.0, 3.0]), cov=jnp.diag(jnp.array([1.0, 4.0]))
    )


class TestTheFactoredGaussian:
    def test_composition_of_gaussian_factors_derives_it(self):
        joint = _gaussian_joint()
        assert isinstance(joint, FactoredMultivariateGaussian)
        assert joint.name == "a·b"
        assert list(joint.event_spec.components) == ["a", "b"]

    def test_constructing_the_factored_law_refines_to_it(self):
        from probpipe.distributions import FactoredDistribution

        joint = FactoredDistribution("j", [Normal("a", 0.0, 1.0), Normal("b", 0.0, 1.0)])
        assert isinstance(joint, FactoredMultivariateGaussian)

    def test_a_factor_that_is_not_gaussian_keeps_the_factored_law(self):
        from probpipe import Gamma
        from probpipe.distributions import FactoredDistribution

        joint = Normal("a", 0.0, 1.0) * Gamma("g", 2.0, 1.0)
        assert isinstance(joint, FactoredDistribution)
        assert not isinstance(joint, FactoredMultivariateGaussian)

    def test_direct_construction_refuses_a_factor_that_is_not_gaussian(self):
        from probpipe import Gamma

        with pytest.raises(TypeError, match="jointly Gaussian"):
            FactoredMultivariateGaussian("j", [Normal("a", 0.0, 1.0), Gamma("g", 2.0, 1.0)])

    def test_its_moments_are_the_factors_in_closed_form(self):
        joint = _gaussian_joint()
        means = joint._mean()
        np.testing.assert_allclose(np.asarray(means["a"]), 1.0)
        np.testing.assert_allclose(np.asarray(means["b"]), [0.0, 3.0])
        np.testing.assert_allclose(_dense(joint._cov()), np.diag([4.0, 1.0, 4.0]))

    def test_its_log_density_is_the_sum_of_the_factors(self):
        joint = _gaussian_joint()
        a, b = jnp.asarray(0.5), jnp.array([1.0, 2.0])
        expected = joint.factors[0]._log_prob(a) + joint.factors[1]._log_prob(b)
        np.testing.assert_allclose(joint._log_prob({"a": a, "b": b}), expected, rtol=1e-6)

    def test_conditioning_on_a_component_is_exact_and_keeps_the_others(self):
        from probpipe import SupportsExactConditioning

        joint = _gaussian_joint()
        assert isinstance(joint, SupportsExactConditioning)
        conditioned = joint._condition_on({"a": 3.0})
        assert isinstance(conditioned, FactoredMultivariateGaussian)
        assert list(conditioned.event_spec.components) == ["b"]
        assert conditioned.factors == (joint.factors[1],)
        assert conditioned.name == joint.name

    def test_the_conditioning_guard_needs_components_and_a_remainder(self):
        from probpipe.distributions._capabilities import _capability_guard

        joint = _gaussian_joint()
        assert _capability_guard(joint, "_condition_on", ("b",)).feasible is True
        assert _capability_guard(joint, "_condition_on", ("a", "b")).feasible is False
        assert _capability_guard(joint, "_condition_on", ("c",)).feasible is False

    def test_conditioning_on_every_component_raises(self):
        with pytest.raises(ValueError, match="covers every component"):
            _gaussian_joint()._condition_on({"a": 0.0, "b": jnp.zeros(2)})

    def test_conditioning_on_a_name_that_is_not_a_component_raises(self):
        with pytest.raises(KeyError, match="not components"):
            _gaussian_joint()._condition_on({"c": 0.0})

    def test_the_marginal_at_a_component_is_its_factor(self):
        joint = _gaussian_joint()
        assert joint._marginal("b").name == joint.name
        np.testing.assert_allclose(
            np.asarray(joint._marginal("b")._mean()), np.asarray(joint.factors[1]._mean())
        )

    def test_a_rebuilt_joint_stays_the_factored_gaussian(self):
        symbolic = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        assert isinstance(symbolic.with_dim_names(n="m"), FactoredMultivariateGaussian)
        assert isinstance(symbolic.with_path_names(a="c"), FactoredMultivariateGaussian)
