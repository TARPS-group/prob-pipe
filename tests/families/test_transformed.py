"""Contracts of the evaluation-result families (VII.4)."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Function,
    MultivariateNormal,
    Normal,
    NumericArrayBatch,
    NumericDistribution,
    ResolutionError,
    bijector_for,
    greater_than,
    interval,
    log_prob,
    mean,
    positive,
    real,
    replay_run,
    sample,
    simplex,
    unit_interval,
    workflow_run,
)
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.families import BijectorTransformedDistribution, LinearPushforwardDistribution
from probpipe.linalg import DenseLinOp, LinOp
from tests.functions._replay_fixtures import replayable_difference


class _Exp(Function):
    """The exponential map as a bijector that claims its inverse and its log-Jacobian."""

    def __init__(self) -> None:
        super().__init__("exp", jnp.exp)

    def _inverse(self, y):
        """The preimage ``log(y)`` of *y*."""
        return jnp.log(y)

    def _log_det_jacobian(self, x):
        """``log |d exp(x) / dx|``, which is *x*."""
        return x


@pytest.fixture
def base():
    return MultivariateNormal("x", jnp.array([1.0, -1.0]), cov=jnp.array([[2.0, 0.5], [0.5, 1.0]]))


@pytest.fixture
def op():
    return DenseLinOp(jnp.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]]))


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def standard():
    return Normal("x", 0.0, 1.0)


class TestTheLinearPushforward:
    @pytest.mark.pending(reason="the pushforward's event is the operator's output type")
    def test_the_event_is_the_operator_output_under_its_own_component(self, base, op):
        pushed = LinearPushforwardDistribution("y", base, op)
        assert list(pushed.event_spec.components) == ["y"]
        assert pushed.event_spec.spec.shape == (3,)

    @pytest.mark.pending(reason="the pushforward's moments delegate exactly")
    def test_the_moments_delegate_through_the_operator(self, base, op):
        pushed = LinearPushforwardDistribution("y", base, op)
        A = op.to_dense()
        np.testing.assert_allclose(pushed._mean(), A @ base._mean(), rtol=1e-6)
        covariance = pushed._cov()
        assert isinstance(covariance, LinOp)
        np.testing.assert_allclose(
            covariance.to_dense(), A @ base._cov().to_dense() @ A.T, rtol=1e-5
        )

    @pytest.mark.pending(reason="the pushforward samples by pushing base draws through op")
    def test_sampling_pushes_the_base_draws(self, base, op):
        pushed = LinearPushforwardDistribution("y", base, op)
        key = jax.random.PRNGKey(0)
        np.testing.assert_allclose(
            pushed._sample(key), op.to_dense() @ base._sample(key), rtol=1e-6
        )


class TestTheBijectorTransform:
    def test_a_map_without_an_inverse_raises(self, standard):
        with pytest.raises(ResolutionError, match="SupportsInverse"):
            BijectorTransformedDistribution("y", standard, Function("exp", jnp.exp))

    def test_a_base_drawing_records_raises(self, standard):
        joint = standard * Normal("z", 0.0, 1.0)
        with pytest.raises(TypeError, match="draws are arrays"):
            BijectorTransformedDistribution("y", joint, tfb.Exp())

    def test_the_change_of_variables_bijector_claims_its_inverse_and_log_jacobian(self):
        bijector = _Exp()
        x = jnp.array([-1.0, 0.5, 2.0])
        np.testing.assert_allclose(bijector._inverse(bijector.apply(x)), x, rtol=1e-6)
        derivative = jax.vmap(jax.grad(bijector.apply))(x)
        np.testing.assert_allclose(bijector._log_det_jacobian(x), jnp.log(derivative), rtol=1e-6)

    def test_the_log_density_is_the_change_of_variables(self, standard):
        transformed = BijectorTransformedDistribution("y", standard, _Exp())
        y = jnp.asarray(2.0)
        expected = standard._log_prob(jnp.log(y)) - jnp.log(y)
        np.testing.assert_allclose(transformed._log_prob(y), expected, rtol=1e-6)

    def test_a_backend_bijector_enters_as_a_function(self, standard):
        transformed = BijectorTransformedDistribution("y", standard, tfb.Exp())
        assert isinstance(transformed.bijector, Function)

    def test_the_log_density_matches_the_backend_transform(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Exp())
        reference = tfd.TransformedDistribution(distribution=standard.raw(), bijector=tfb.Exp())
        ys = jnp.array([0.1, 1.0, 5.0])
        np.testing.assert_allclose(
            transformed._log_prob(ys), reference.log_prob(ys), rtol=1e-5, atol=1e-6
        )

    def test_the_label_names_the_event_and_the_base_is_its_parent(self, standard):
        transformed = BijectorTransformedDistribution("log_normal", standard, tfb.Exp())
        assert transformed.label == "log_normal"
        assert list(transformed.event_spec.components) == ["log_normal"]
        assert transformed.base is standard
        assert transformed.provenance.operation == "transform"
        assert transformed.provenance.parents[0].label == "x"

    def test_the_repr_names_the_class_the_base_and_the_bijector(self, standard):
        r = repr(BijectorTransformedDistribution("td", standard, tfb.Exp()))
        assert "BijectorTransformedDistribution" in r
        assert "Normal" in r
        assert "exp" in r

    def test_it_is_a_numeric_law_of_the_base_dtype(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Exp())
        assert isinstance(transformed, NumericDistribution)
        assert transformed.dtype == jnp.zeros((), dtype=float).dtype
        assert not hasattr(transformed, "batch_shape")

    def test_it_pickles(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Exp())
        restored = pickle.loads(pickle.dumps(transformed))
        np.testing.assert_allclose(restored._log_prob(2.0), transformed._log_prob(2.0))


class TestSampling:
    def test_exp_samples_are_positive(self, standard, key):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Exp())
        draws = jnp.asarray(sample(transformed, sample_shape=(100,)))
        assert draws.shape == (100,)
        assert jnp.all(draws > 0)
        assert transformed.event_shape == ()

    def test_the_log_density_keeps_the_leading_axes(self, standard, key):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Exp())
        draws = sample(transformed, sample_shape=(5,))
        densities = log_prob(transformed, draws)
        assert densities.shape == (5,)
        assert jnp.all(jnp.isfinite(densities))

    @pytest.mark.parametrize(
        ("bijector", "low", "high"),
        [(tfb.Sigmoid(), 0.0, 1.0), (tfb.Softplus(), 0.0, jnp.inf)],
        ids=["sigmoid", "softplus"],
    )
    def test_draws_lie_in_the_image(self, standard, key, bijector, low, high):
        transformed = BijectorTransformedDistribution("td", standard, bijector)
        draws = jnp.asarray(sample(transformed, sample_shape=(100,)))
        assert jnp.all(draws >= low) and jnp.all(draws <= high)

    def test_a_multivariate_base(self, key):
        base = MultivariateNormal("z", jnp.zeros(3), cov=jnp.eye(3))
        transformed = BijectorTransformedDistribution("td", base, tfb.Exp())
        draws = jnp.asarray(sample(transformed, sample_shape=(10,)))
        assert draws.shape == (10, 3)
        assert jnp.all(draws > 0)

    def test_a_dimension_changing_bijector(self, key):
        base = MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2))
        transformed = BijectorTransformedDistribution("p", base, bijector_for(simplex))
        assert transformed.event_shape == (3,)
        draw = transformed._sample(key)
        assert float(jnp.sum(draw)) == pytest.approx(1.0, rel=1e-5)
        assert jnp.isfinite(transformed._log_prob(jnp.array([0.2, 0.3, 0.5])))

    def test_a_density_onto_the_simplex_integrates_to_one(self):
        """The density is stated in the simplex's first two coordinates, as a Dirichlet's is."""
        base = MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2))
        transformed = BijectorTransformedDistribution("p", base, bijector_for(simplex))
        cells = 300
        grid = (np.arange(cells) + 0.5) / cells
        first, second = np.meshgrid(grid, grid, indexing="ij")
        inside = first + second < 1
        points = np.stack(
            [first[inside], second[inside], 1 - first[inside] - second[inside]], axis=-1
        )
        density = jnp.exp(jax.vmap(transformed._log_prob)(jnp.asarray(points, jnp.float32)))
        assert float(density.sum()) / cells**2 == pytest.approx(1.0, abs=0.01)

    def test_an_empirical_base(self, key):
        atoms = jax.random.normal(key, (50, 2))
        transformed = BijectorTransformedDistribution(
            "td", EmpiricalDistribution("x", atoms), tfb.Exp()
        )
        draws = jnp.asarray(transformed._sample(key, (10,)))
        assert draws.shape == (10, 2)
        assert jnp.all(draws > 0)

    def test_a_chain_of_bijectors(self, standard, key):
        chain = tfb.Chain([tfb.Exp(), tfb.Shift(jnp.array(1.0)), tfb.Scale(jnp.array(2.0))])
        transformed = BijectorTransformedDistribution("td", standard, chain)
        drawn = sample(transformed, sample_shape=(10,))
        draws = jnp.asarray(drawn)
        assert draws.shape == (10,)
        assert jnp.all(draws > 0)
        assert jnp.all(jnp.isfinite(jnp.asarray(log_prob(transformed, drawn))))

    def test_the_identity_keeps_draws_and_densities(self, key):
        base = Normal("x", 2.0, 0.5)
        transformed = BijectorTransformedDistribution("td", base, tfb.Identity())
        np.testing.assert_allclose(
            np.asarray(transformed._sample(key, (100,))),
            np.asarray(base._sample(key, (100,))),
            atol=1e-6,
        )
        xs = NumericArrayBatch("x", jnp.array([-1.0, 0.0, 1.0, 2.5]), "point")
        np.testing.assert_allclose(
            np.asarray(log_prob(transformed, xs)), np.asarray(log_prob(base, xs)), atol=1e-5
        )


def _exp_law(bijector_kind: str) -> BijectorTransformedDistribution:
    bijector = tfb.Exp() if bijector_kind == "backend" else _Exp()
    return BijectorTransformedDistribution("y", Normal("x", 0.0, 1.0), bijector)


class TestReplay:
    """A transformed law builds, samples, and lifts inside ``replay_run`` as when recorded."""

    @pytest.mark.parametrize("bijector_kind", ["backend", "function"])
    def test_a_draw_replays_identically(self, bijector_kind):
        law = _exp_law(bijector_kind)
        with workflow_run(seed=4):
            original = sample(law, sample_shape=(3,))
        with replay_run(original.provenance):
            replayed = sample(law, sample_shape=(3,))
        np.testing.assert_array_equal(np.asarray(replayed), np.asarray(original))

    @pytest.mark.parametrize("bijector_kind", ["backend", "function"])
    def test_a_law_built_inside_the_replay_draws_the_recorded_values(self, bijector_kind):
        with workflow_run(seed=4):
            original = sample(_exp_law(bijector_kind))
        with replay_run(original.provenance):
            replayed = sample(_exp_law(bijector_kind))
        np.testing.assert_array_equal(np.asarray(replayed), np.asarray(original))

    def test_a_lift_replays_identically(self):
        difference = Function(
            label="replayable_difference",
            fn=replayable_difference,
            n_broadcast_samples=8,
            dispatch="sequential",
        )

        def operands():
            root = Normal("root", 0.0, 1.0)
            return {
                "left": root,
                "right": BijectorTransformedDistribution("right", root, tfb.Exp()),
            }

        with workflow_run(seed=53):
            original = difference(**operands())
        with replay_run(original.provenance):
            replayed = difference(**operands())
        np.testing.assert_array_equal(
            np.asarray(replayed.atoms.values), np.asarray(original.atoms.values)
        )


class TestTheCapabilities:
    def test_a_density_base_gives_a_density(self, standard):
        assert isinstance(
            BijectorTransformedDistribution("td", standard, tfb.Exp()), SupportsLogProb
        )

    def test_a_base_without_a_density_gives_none(self):
        empirical = EmpiricalDistribution("x", jnp.array([1.0, 2.0, 3.0]))
        transformed = BijectorTransformedDistribution("td", empirical, tfb.Exp())
        assert isinstance(transformed, SupportsSampling)
        assert not isinstance(transformed, SupportsLogProb)

    def test_a_nonlinear_map_claims_no_moment(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Exp())
        for protocol in (SupportsMean, SupportsVariance, SupportsCovariance):
            assert not isinstance(transformed, protocol)

    def test_the_mean_of_a_nonlinear_map_is_estimated(self, key):
        atoms = jax.random.normal(key, (50, 2))
        transformed = BijectorTransformedDistribution(
            "td", EmpiricalDistribution("x", atoms), tfb.Exp()
        )
        assert jnp.all(jnp.isfinite(mean(transformed)))

    def test_a_shift_moves_the_mean(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Shift(jnp.array(5.0)))
        assert isinstance(transformed, SupportsMean)
        assert float(transformed._mean()) == pytest.approx(5.0, abs=1e-6)

    def test_a_scale_scales_the_variance(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Scale(jnp.array(2.0)))
        assert float(transformed._variance()) == pytest.approx(4.0, abs=1e-6)

    def test_the_identity_keeps_the_moments(self):
        base = Normal("x", 2.0, 0.5)
        transformed = BijectorTransformedDistribution("td", base, tfb.Identity())
        assert float(transformed._mean()) == pytest.approx(2.0, abs=1e-6)
        assert float(transformed._variance()) == pytest.approx(0.25, abs=1e-6)

    def test_an_affine_map_pushes_the_covariance(self, base):
        matrix = jnp.array([[2.0, 0.0], [1.0, 1.0]])
        shift = jnp.array([1.0, -2.0])
        affine = tfb.Chain([tfb.Shift(shift), tfb.ScaleMatvecTriL(matrix)])
        transformed = BijectorTransformedDistribution("y", base, affine)
        np.testing.assert_allclose(transformed._mean(), matrix @ base._mean() + shift, rtol=1e-6)
        covariance = transformed._cov()
        assert isinstance(covariance, LinOp)
        np.testing.assert_allclose(
            covariance.to_dense(), matrix @ base._cov().to_dense() @ matrix.T, rtol=1e-5
        )


class TestTheSupport:
    """The support is the image of the bijector, when it is known."""

    @pytest.mark.parametrize(
        ("bijector", "support"),
        [
            (tfb.Exp(), positive),
            (tfb.Sigmoid(), unit_interval),
            (tfb.Softplus(), positive),
            (tfb.Chain([tfb.Exp(), tfb.Shift(1.0)]), positive),
            (tfb.Chain([tfb.Shift(1.0), tfb.Scale(2.0)]), real),
            (tfb.Shift(1.0), real),
        ],
        ids=["exp", "sigmoid", "softplus", "chain", "affine-chain", "shift"],
    )
    def test_a_backend_bijector_declares_its_image(self, standard, bijector, support):
        transformed = BijectorTransformedDistribution("td", standard, bijector)
        assert transformed.support == support

    @pytest.mark.parametrize(
        "bijector",
        [tfb.Chain([tfb.Shift(1.0), tfb.Exp()]), tfb.Chain([tfb.Scale(2.0), tfb.Sigmoid()])],
        ids=["shift-after-exp", "scale-after-sigmoid"],
    )
    def test_a_chain_with_an_inner_image_short_of_the_line_leaves_the_support_undeclared(
        self, standard, bijector
    ):
        # The images, (1, ∞) and (0, 2), are not the outermost bijector's image, the line.
        transformed = BijectorTransformedDistribution("td", standard, bijector)
        assert transformed.support is None

    @pytest.mark.parametrize(
        "constraint",
        [positive, unit_interval, interval(2.0, 5.0), greater_than(3.0)],
        ids=repr,
    )
    def test_the_factory_bijector_declares_its_constraint(self, standard, constraint):
        transformed = BijectorTransformedDistribution("td", standard, bijector_for(constraint))
        assert transformed.support == constraint

    def test_an_unknown_image_leaves_the_support_undeclared(self, standard):
        transformed = BijectorTransformedDistribution("td", standard, tfb.Tanh())
        assert transformed.support is None
