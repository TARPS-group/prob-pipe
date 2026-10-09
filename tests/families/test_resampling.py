"""Contracts of the resampling families (VII.2): the smoothing kernels, the KDE, and the bootstrap.

A bootstrap replicate is ``replicate_size`` iid draws of a source that samples,
in the event's batch form on one level, and the bootstrap distribution is the
random measure whose draw is the empirical measure of one replicate. A KDE
smooths weighted atoms with a smoothing kernel whose bank of placed copies one
uniform constructor builds; its density, draws, and moments are exact for the
smoothed law.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from probpipe import (
    Function,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    OutputSpec,
    RandomMeasure,
    Record,
    RecordSpec,
    workflow_run,
)
from probpipe.core._batch import BatchSpec
from probpipe.core._numeric_record import NumericRecord
from probpipe.core.constraints import positive, real
from probpipe.distributions import DistributionSpec
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from probpipe.families import (
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    EpanechnikovKernel,
    GaussianKernel,
    KDEDistribution,
    SmoothingKernel,
)
from tests._ops import EmpiricalDistribution, sample

_KERNELS = [GaussianKernel, EpanechnikovKernel]


@pytest.fixture
def centers():
    return jnp.array([[0.0, 1.0], [2.0, -1.0], [4.0, 0.5]])


@pytest.fixture
def record_centers(centers):
    return NumericRecordBatch(
        "atoms",
        {"a": centers[:, 0], "b": centers[:, 1]},
        "atom",
        element_spec=NumericRecordSpec(a=(), b=()),
    )


# ---------------------------------------------------------------------------
# The smoothing kernels
# ---------------------------------------------------------------------------


class TestTheUniformConstructor:
    def test_the_base_is_abstract(self, centers):
        with pytest.raises(TypeError):
            SmoothingKernel(centers, 1.0)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_build_kernels_places_one_copy_per_center(self, kernel, centers):
        bank = kernel.build_kernels(centers, 0.5)
        assert isinstance(bank, kernel)
        assert bank._log_density(jnp.zeros(2)).shape == (3,)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_scales_broadcast_over_centers_and_coordinates(self, kernel, centers):
        x = jnp.array([1.0, 0.25])
        scalar = kernel.build_kernels(centers, 0.8)._log_density(x)
        per_coordinate = kernel.build_kernels(centers, jnp.array([0.8, 0.8]))._log_density(x)
        per_copy = kernel.build_kernels(centers, jnp.full((3, 2), 0.8))._log_density(x)
        np.testing.assert_allclose(scalar, per_coordinate, rtol=1e-6)
        np.testing.assert_allclose(scalar, per_copy, rtol=1e-6)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_record_centers_and_scales_flatten_to_their_coordinates(
        self, kernel, centers, record_centers
    ):
        scales = NumericRecord("h", {"a": 0.5, "b": 1.5})
        from_records = kernel.build_kernels(record_centers, scales)
        from_arrays = kernel.build_kernels(centers, jnp.array([0.5, 1.5]))
        x = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(
            from_records._log_density(x), from_arrays._log_density(x), rtol=1e-6
        )

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_record_scales_match_the_centers_fields_by_path(self, kernel, centers, record_centers):
        # The point is inside the first copy only when the scales are matched by field.
        scales = NumericRecord("h", {"b": 0.8, "a": 2.5})
        from_records = kernel.build_kernels(record_centers, scales)
        from_arrays = kernel.build_kernels(centers, jnp.array([2.5, 0.8]))
        x = jnp.array([1.0, 0.5])
        assert np.isfinite(from_arrays._log_density(x)[0])
        np.testing.assert_allclose(
            from_records._log_density(x), from_arrays._log_density(x), rtol=1e-6
        )

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_nested_record_scales_match_the_centers_leaf_paths(self, kernel):
        points = jnp.array([[0.0, 1.0, -1.0, 2.0], [2.0, 0.5, 0.0, -1.0]])
        batch = NumericRecordBatch(
            "atoms",
            {"x/p": points[:, 0], "x/q": points[:, 1:3], "y": points[:, 3]},
            "atom",
            element_spec=NumericRecordSpec(x=NumericRecordSpec(p=(), q=(2,)), y=()),
        )
        scales = NumericRecord("h", {"y": 3.0, "x": {"q": jnp.array([2.0, 1.5]), "p": 1.2}})
        from_records = kernel.build_kernels(batch, scales)
        from_arrays = kernel.build_kernels(points, jnp.array([1.2, 2.0, 1.5, 3.0]))
        x = jnp.array([0.5, 0.0, -0.5, 1.0])
        np.testing.assert_allclose(
            from_records._log_density(x), from_arrays._log_density(x), rtol=1e-6
        )

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_a_fields_scale_broadcasts_over_the_fields_coordinates(self, kernel):
        points = jnp.array([[0.0, 1.0, -1.0], [2.0, 0.5, 0.0]])
        batch = NumericRecordBatch(
            "atoms",
            {"a": points[:, 0], "b": points[:, 1:]},
            "atom",
            element_spec=NumericRecordSpec(a=(), b=(2,)),
        )
        from_records = kernel.build_kernels(batch, NumericRecord("h", {"b": 2.0, "a": 0.5}))
        from_arrays = kernel.build_kernels(points, jnp.array([0.5, 2.0, 2.0]))
        x = jnp.array([0.2, 1.5, -0.5])
        np.testing.assert_allclose(
            from_records._log_density(x), from_arrays._log_density(x), rtol=1e-6
        )

    @pytest.mark.parametrize("kernel", _KERNELS)
    @pytest.mark.parametrize(
        "fields",
        [{"a": 0.5, "c": 1.5}, {"a": 0.5}, {"a": 0.5, "b": 1.5, "c": 1.0}],
        ids=["another-field", "a-missing-field", "an-extra-field"],
    )
    def test_record_scales_over_other_fields_raise(self, kernel, record_centers, fields):
        with pytest.raises(ValueError, match="fields"):
            kernel.build_kernels(record_centers, NumericRecord("h", fields))

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_a_fields_scale_that_does_not_broadcast_over_the_field_raises(
        self, kernel, record_centers
    ):
        scales = NumericRecord("h", {"a": 0.5, "b": jnp.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="'b'"):
            kernel.build_kernels(record_centers, scales)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_record_scales_for_array_centers_raise(self, kernel, centers):
        with pytest.raises(ValueError, match="record"):
            kernel.build_kernels(centers, NumericRecord("h", {"a": 0.5, "b": 1.5}))

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_scales_that_do_not_broadcast_raise(self, kernel, centers):
        with pytest.raises(ValueError, match="broadcast"):
            kernel.build_kernels(centers, jnp.ones(3))

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_a_nonpositive_scale_raises(self, kernel, centers):
        with pytest.raises(ValueError, match="positive"):
            kernel.build_kernels(centers, jnp.array([1.0, 0.0]))

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_centers_without_an_atom_axis_raise(self, kernel):
        with pytest.raises(ValueError, match="centers must have a leading axis"):
            kernel.build_kernels(jnp.asarray(1.0), 1.0)


class TestTheCopies:
    def test_a_gaussian_copy_is_the_normal_density_with_its_scale_jacobian(self, centers):
        scales = jnp.array([0.5, 2.0])
        bank = GaussianKernel.build_kernels(centers, scales)
        x = jnp.array([1.0, 0.25])
        expected = stats.norm.logpdf(np.asarray(x), np.asarray(centers), np.asarray(scales)).sum(-1)
        np.testing.assert_allclose(bank._log_density(x), expected, rtol=1e-5)

    def test_an_epanechnikov_copy_is_the_scaled_product_kernel(self, centers):
        scales = jnp.array([1.5, 2.0])
        bank = EpanechnikovKernel.build_kernels(centers, scales)
        x = jnp.array([0.5, 0.0])
        u = (np.asarray(x) - np.asarray(centers)) / np.asarray(scales)
        density = np.prod(np.where(np.abs(u) < 1, 0.75 * (1 - u**2), 0.0), axis=-1) / np.prod(
            np.asarray(scales)
        )
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(bank._log_density(x), np.log(density), rtol=1e-5)

    def test_an_epanechnikov_copy_vanishes_beyond_one_scale(self):
        bank = EpanechnikovKernel.build_kernels(jnp.array([0.0]), 1.0)
        assert bank._log_density(jnp.array([1.5]))[0] == -jnp.inf

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_a_copy_is_a_mean_zero_density_with_the_declared_variance(self, kernel):
        h = 0.7
        bank = kernel.build_kernels(jnp.array([[0.0]]), h)
        grid = np.linspace(-8.0, 8.0, 40001)
        density = np.exp(np.asarray(bank._log_density(jnp.asarray(grid)[:, None])[:, 0], float))
        assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-4)
        assert np.trapezoid(grid * density, grid) == pytest.approx(0.0, abs=1e-4)
        variance = np.trapezoid(grid**2 * density, grid)
        assert variance == pytest.approx(h**2 * kernel.variance, rel=1e-3)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_log_density_scores_a_batch_of_points_under_every_copy(self, kernel, centers):
        bank = kernel.build_kernels(centers, 3.0)
        points = jnp.zeros((4, 5, 2))
        assert bank._log_density(points).shape == (4, 5, 3)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_sample_draws_from_each_indexed_copy(self, kernel, centers):
        scales = jnp.array([0.5, 0.25])
        bank = kernel.build_kernels(centers, scales)
        index = jnp.full((20000,), 1)
        draws = bank._sample(jax.random.PRNGKey(0), index)
        assert draws.shape == (20000, 2)
        np.testing.assert_allclose(draws.mean(axis=0), centers[1], atol=0.02)
        np.testing.assert_allclose(draws.var(axis=0), scales**2 * kernel.variance, rtol=0.05)

    @pytest.mark.parametrize("kernel", _KERNELS)
    def test_sample_keeps_the_index_shape(self, kernel, centers):
        bank = kernel.build_kernels(centers, 1.0)
        draws = bank._sample(jax.random.PRNGKey(1), jnp.array([[0, 1], [2, 0]]))
        assert draws.shape == (2, 2, 2)

    def test_epanechnikov_draws_stay_within_one_scale_of_their_center(self):
        bank = EpanechnikovKernel.build_kernels(jnp.array([[3.0]]), 0.5)
        draws = bank._sample(jax.random.PRNGKey(2), jnp.zeros((5000,), dtype=int))
        assert float(jnp.max(jnp.abs(draws - 3.0))) <= 0.5

    def test_scalar_atoms_are_scored_and_drawn_as_scalars(self):
        bank = GaussianKernel.build_kernels(jnp.array([0.0, 1.0, 2.0]), 0.5)
        assert bank._log_density(jnp.asarray(1.0)).shape == (3,)
        assert bank._sample(jax.random.PRNGKey(3), jnp.array([0, 2])).shape == (2,)


class TestTheKDELaw:
    """VII.2: the KDE law is the weighted mixture of the placed copies."""

    def test_the_gaussian_bank_reproduces_the_gaussian_kde_density(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        bandwidth = jnp.array([0.4, 0.6])
        kde = KDEDistribution("k", centers, weights=weights, bandwidth=bandwidth)
        bank = GaussianKernel.build_kernels(centers, bandwidth)
        x = jnp.array([1.5, 0.0])
        mixture = jax.scipy.special.logsumexp(jnp.log(weights) + bank._log_density(x))
        np.testing.assert_allclose(kde._log_prob(x), mixture, rtol=1e-5)

    def test_the_density_is_the_scipy_mixture(self, centers):
        weights = np.array([0.2, 0.3, 0.5])
        kde = KDEDistribution("k", centers, jnp.array([0.4, 0.6]), jnp.asarray(weights))
        x = np.array([1.5, 0.0])
        density = sum(
            w * stats.multivariate_normal(c, np.diag([0.16, 0.36])).pdf(x)
            for w, c in zip(weights, np.asarray(centers))
        )
        np.testing.assert_allclose(kde._log_prob(jnp.asarray(x)), np.log(density), rtol=1e-5)

    def test_the_density_scores_a_batch_of_points(self, centers):
        kde = KDEDistribution("k", centers, 0.5)
        assert kde._log_prob(jnp.zeros((4, 5, 2))).shape == (4, 5)

    def test_the_mean_is_the_weighted_atom_mean(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        kde = KDEDistribution("k", centers, weights=weights, bandwidth=0.5)
        np.testing.assert_allclose(kde._mean(), weights @ centers, rtol=1e-5)

    def test_the_variance_adds_the_kernel_variance(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        kde = KDEDistribution("k", centers, weights=weights, bandwidth=0.5)
        atom_mean = weights @ centers
        atom_variance = weights @ (centers - atom_mean) ** 2
        np.testing.assert_allclose(
            kde._variance(), atom_variance + 0.5**2 * GaussianKernel.variance, rtol=1e-5
        )

    def test_the_covariance_adds_the_kernel_variance_on_the_diagonal(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        kde = KDEDistribution("k", centers, jnp.array([0.5, 1.0]), weights)
        diff = np.asarray(centers - weights @ centers)
        expected = (np.asarray(weights)[:, None] * diff).T @ diff + np.diag([0.25, 1.0])
        np.testing.assert_allclose(kde._cov().to_dense(), expected, rtol=1e-5)

    def test_per_atom_scales_enter_the_variance_by_weight(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        scales = jnp.array([[0.5], [1.0], [2.0]])
        kde = KDEDistribution("k", centers, scales, weights)
        atom_variance = weights @ (centers - weights @ centers) ** 2
        np.testing.assert_allclose(
            kde._variance(), atom_variance + weights @ (scales**2)[:, 0], rtol=1e-5
        )

    def test_the_kde_takes_a_kernel_class(self, centers):
        kde = KDEDistribution("k", centers, 0.5, kernel=EpanechnikovKernel)
        atom_variance = jnp.var(centers, axis=0)
        np.testing.assert_allclose(
            kde._variance(), atom_variance + 0.25 * EpanechnikovKernel.variance, rtol=1e-5
        )

    def test_the_kernel_is_a_smoothing_kernel_class(self, centers):
        with pytest.raises(TypeError, match="SmoothingKernel"):
            KDEDistribution("k", centers, 0.5, kernel=GaussianKernel.build_kernels(centers, 1.0))

    def test_draws_follow_the_smoothed_law(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        kde = KDEDistribution("k", centers, jnp.array([0.4, 0.6]), weights)
        draws = kde._sample(jax.random.PRNGKey(4), (40000,))
        assert draws.shape == (40000, 2)
        np.testing.assert_allclose(draws.mean(axis=0), kde._mean(), atol=0.03)
        np.testing.assert_allclose(draws.var(axis=0), kde._variance(), rtol=0.03)

    def test_one_draw_has_the_event_shape(self, centers):
        assert KDEDistribution("k", centers, 0.5)._sample(jax.random.PRNGKey(5)).shape == (2,)

    def test_it_claims_the_exact_capabilities(self, centers):
        kde = KDEDistribution("k", centers, 0.5)
        for protocol in (
            SupportsSampling,
            SupportsLogProb,
            SupportsMean,
            SupportsVariance,
            SupportsCovariance,
        ):
            assert isinstance(kde, protocol), protocol.__name__


class TestTheBandwidthRules:
    @staticmethod
    def _spread(centers, weights):
        mean = weights @ centers
        return np.sqrt(np.asarray(weights @ (centers - mean) ** 2))

    def test_scott_is_the_default(self, centers):
        default = KDEDistribution("k", centers)
        scott = KDEDistribution("k", centers, "scott")
        np.testing.assert_allclose(default._variance(), scott._variance(), rtol=1e-6)

    @pytest.mark.parametrize("rule", ["scott", "silverman"])
    def test_bandwidth_accepts_a_selection_rule(self, rule, centers):
        kde = KDEDistribution("k", centers, rule)
        assert kde.event_spec.components == {"k": kde.event_spec.spec}

    def test_scott_scales_the_spread_by_the_effective_sample_size(self, centers):
        weights = jnp.array([0.2, 0.3, 0.5])
        n_eff = 1.0 / float(jnp.sum(weights**2))
        h = n_eff ** (-1.0 / 6.0) * self._spread(centers, weights)
        kde = KDEDistribution("k", centers, "scott", weights)
        atom_variance = np.asarray(weights @ (centers - weights @ centers) ** 2)
        np.testing.assert_allclose(kde._variance(), atom_variance + h**2, rtol=1e-5)

    @pytest.mark.parametrize("d", [1, 3])
    def test_silverman_adds_its_constant(self, d):
        # Silverman's constant (4/(d+2))^(1/(d+4)) is one at d = 2, where the rules agree.
        atoms = jnp.array([[0.0, 1.0, -2.0], [2.0, -1.0, 0.5], [4.0, 0.5, 1.0]])[:, :d]
        weights = jnp.array([0.2, 0.3, 0.5])
        n_eff = 1.0 / float(jnp.sum(weights**2))
        constant = (4.0 / (d + 2)) ** (1.0 / (d + 4))
        h = constant * n_eff ** (-1.0 / (d + 4)) * self._spread(atoms, weights)
        silverman = KDEDistribution("k", atoms, "silverman", weights)
        scott = KDEDistribution("k", atoms, "scott", weights)
        atom_variance = np.asarray(weights @ (atoms - weights @ atoms) ** 2)
        np.testing.assert_allclose(silverman._variance(), atom_variance + h**2, rtol=1e-5)
        assert not np.allclose(silverman._variance(), scott._variance(), rtol=1e-3)

    def test_uniform_weights_count_every_atom(self):
        atoms = jnp.arange(8.0)
        h = 8.0 ** (-1.0 / 5.0) * float(jnp.std(atoms))
        kde = KDEDistribution("k", atoms)
        np.testing.assert_allclose(kde._variance(), jnp.var(atoms) + h**2, rtol=1e-5)

    def test_an_unknown_rule_raises(self, centers):
        with pytest.raises(ValueError, match="unknown bandwidth rule 'rule-of-thumb'"):
            KDEDistribution("k", centers, "rule-of-thumb")

    def test_a_bad_bandwidth_is_reported_in_the_kdes_terms(self, centers):
        with pytest.raises(ValueError, match=r"bandwidth has shape \(3,\).*atoms of shape"):
            KDEDistribution("k", centers, jnp.ones(3))
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KDEDistribution("k", centers, jnp.array([1.0, -1.0]))

    def test_atoms_that_are_not_an_array_raise(self):
        with pytest.raises(TypeError, match=r"atoms must be an array .* got list"):
            KDEDistribution("k", [1.0, 2.0, 3.0])

    def test_a_rule_refuses_atoms_without_spread(self):
        with pytest.raises(ValueError, match="atoms do not vary"):
            KDEDistribution("k", jnp.ones((4, 2)))


class TestTheKDEDeclaration:
    def test_array_atoms_form_a_whole_term_under_the_law_name(self, centers):
        kde = KDEDistribution("post", centers, 0.5)
        assert kde.event_spec == OutputSpec(post=NumericArraySpec((2,), jnp.float32, real))
        assert kde.event_shape == (2,)

    def test_scalar_atoms_draw_scalars(self):
        kde = KDEDistribution("k", jnp.arange(5.0), 0.5)
        assert kde.event_shape == ()
        assert kde._sample(jax.random.PRNGKey(0), (3,)).shape == (3,)

    def test_integer_atoms_are_smoothed_as_floats(self):
        kde = KDEDistribution("k", jnp.arange(5), 0.5)
        assert jnp.issubdtype(kde.event_spec.spec.dtype, jnp.floating)

    def test_a_type_hole_takes_the_atoms_term(self, centers):
        kde = KDEDistribution("post", centers, 0.5, event_spec=OutputSpec(theta=None))
        assert kde.event_spec == OutputSpec(theta=NumericArraySpec((2,), jnp.float32, real))

    def test_a_declaration_the_atoms_do_not_match_is_refused(self, centers):
        with pytest.raises(ValueError, match="theta"):
            KDEDistribution(
                "post", centers, 0.5, event_spec=OutputSpec(theta=NumericArraySpec((3,)))
            )

    def test_an_exposed_record_needs_record_atoms(self, centers):
        with pytest.raises(TypeError, match="RecordSpec"):
            KDEDistribution("post", centers, 0.5, event_spec=OutputSpec(RecordSpec(a=(2,))))

    def test_record_atoms_expose_their_fields_on_the_real_line(self, record_centers):
        kde = KDEDistribution("post", record_centers, 0.5)
        assert kde.event_spec == OutputSpec(
            RecordSpec(
                a=NumericArraySpec((), jnp.float32, real), b=NumericArraySpec((), jnp.float32, real)
            )
        )

    def test_a_declared_support_of_the_atoms_is_widened_to_the_real_line(self):
        atoms = NumericRecordBatch(
            "atoms",
            {"s": jnp.array([1.0, 2.0, 3.0])},
            "atom",
            element_spec=NumericRecordSpec(s=NumericArraySpec((), jnp.float32, positive)),
        )
        kde = KDEDistribution("post", atoms, 0.5)
        assert kde.event_spec.spec["s"].support is real

    @pytest.mark.parametrize(
        "atoms",
        [
            pytest.param(Record("r", a=jnp.zeros(3)), id="one-record"),
            pytest.param(["a", "b"], id="list"),
            pytest.param(np.array(["a", "b"], dtype=object), id="object-array"),
        ],
    )
    def test_atoms_are_numeric(self, atoms):
        with pytest.raises(TypeError):
            KDEDistribution("k", atoms, 1.0)


class TestARecordKDE:
    def test_draws_are_the_raw_form_of_the_record(self, record_centers):
        kde = KDEDistribution("post", record_centers, 0.5)
        one = kde._sample(jax.random.PRNGKey(0))
        many = kde._sample(jax.random.PRNGKey(1), (8,))
        assert set(one) == {"a", "b"} and np.shape(one["a"]) == ()
        assert np.shape(many["a"]) == (8,) and np.shape(many["b"]) == (8,)

    def test_the_sample_operation_returns_records(self, record_centers):
        kde = KDEDistribution("post", record_centers, 0.5)
        with workflow_run(seed=2):
            draws = sample(kde, sample_shape=(5,))
        assert isinstance(draws, NumericRecordBatch)
        assert draws.batch_shape == (5,)

    def test_the_density_reads_a_record_and_a_batch_of_records(self, centers, record_centers):
        kde = KDEDistribution("post", record_centers, NumericRecord("h", {"a": 0.5, "b": 1.5}))
        flat = KDEDistribution("flat", centers, jnp.array([0.5, 1.5]))
        point = NumericRecord("v", {"a": 0.5, "b": -0.3})
        np.testing.assert_allclose(kde._log_prob(point), flat._log_prob(jnp.array([0.5, -0.3])))
        np.testing.assert_allclose(kde._log_prob({"a": 0.5, "b": -0.3}), kde._log_prob(point))
        batch = NumericRecordBatch(
            "values",
            {"a": jnp.array([0.5, 0.6, 0.7]), "b": jnp.array([-0.3, -0.4, -0.5])},
            "value",
            element_spec=NumericRecordSpec(a=(), b=()),
        )
        expected = flat._log_prob(jnp.array([[0.5, -0.3], [0.6, -0.4], [0.7, -0.5]]))
        np.testing.assert_allclose(kde._log_prob(batch), expected, rtol=1e-6)

    def test_a_batch_of_records_is_read_by_the_kdes_leaf_paths(self):
        atoms = NumericRecordBatch(
            "atoms",
            {
                "a": jnp.array([[0.0, 1.0], [2.0, -1.0], [4.0, 0.5]]),
                "b": jnp.array([1.0, 3.0, 2.0]),
            },
            "atom",
            element_spec=NumericRecordSpec(a=(2,), b=()),
        )
        kde = KDEDistribution("post", atoms, NumericRecord("h", {"a": 0.5, "b": 1.5}))
        a, b = jnp.array([[0.5, 0.2], [2.5, -0.5]]), jnp.array([1.2, 2.8])
        in_order = NumericRecordBatch(
            "values", {"a": a, "b": b}, "value", element_spec=NumericRecordSpec(a=(2,), b=())
        )
        reordered = NumericRecordBatch(
            "values", {"b": b, "a": a}, "value", element_spec=NumericRecordSpec(b=(), a=(2,))
        )
        expected = kde._log_prob({"a": a, "b": b})
        np.testing.assert_allclose(kde._log_prob(in_order), expected, rtol=1e-6)
        np.testing.assert_allclose(kde._log_prob(reordered), expected, rtol=1e-6)

    def test_the_density_refuses_a_bare_array_for_a_record(self, record_centers):
        kde = KDEDistribution("post", record_centers, 0.5)
        with pytest.raises(TypeError, match="record"):
            kde._log_prob(jnp.zeros(2))

    def test_the_moments_are_the_raw_form_of_the_record(self, centers, record_centers):
        kde = KDEDistribution("post", record_centers, 0.5)
        flat = KDEDistribution("flat", centers, 0.5)
        mean, variance = kde._mean(), kde._variance()
        np.testing.assert_allclose([mean["a"], mean["b"]], flat._mean(), rtol=1e-6)
        np.testing.assert_allclose([variance["a"], variance["b"]], flat._variance(), rtol=1e-6)


# ---------------------------------------------------------------------------
# The bootstrap replicate
# ---------------------------------------------------------------------------


def _rows() -> NumericRecordBatch:
    return NumericRecordBatch(
        "xy",
        {"x": jnp.arange(4.0), "y": 10.0 * jnp.arange(4.0)},
        "row",
        element_spec=NumericRecordSpec(x=(), y=()),
    )


class TestTheBootstrapReplicate:
    def test_a_draw_is_replicate_size_draws_of_a_sampling_source(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), replicate_size=5)
        assert replicate._sample(jax.random.PRNGKey(0)).shape == (5,)
        assert replicate._sample(jax.random.PRNGKey(0), (3,)).shape == (3, 5)
        assert replicate._sample(jax.random.PRNGKey(0), (2, 3)).shape == (2, 3, 5)

    def test_replicate_size_is_required_for_a_source_without_atoms(self):
        with pytest.raises(ValueError, match="replicate_size"):
            BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0))

    def test_replicate_size_defaults_to_the_source_atom_count(self):
        source = EmpiricalDistribution("data", jnp.arange(6.0))
        assert BootstrapReplicateDistribution("b", source).replicate_size == 6

    @pytest.mark.parametrize(
        ("size", "error"),
        [
            pytest.param(0, ValueError, id="zero"),
            pytest.param(-2, ValueError, id="negative"),
            pytest.param(2.5, TypeError, id="float"),
            pytest.param(True, TypeError, id="bool"),
        ],
    )
    def test_an_invalid_replicate_size_raises(self, size, error):
        with pytest.raises(error, match="replicate_size"):
            BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), size)

    @pytest.mark.parametrize("source", [jnp.arange(3.0), Record("r", a=jnp.zeros(3))])
    def test_the_source_is_a_law_that_samples(self, source):
        with pytest.raises(TypeError, match="must be a Distribution that supports sampling"):
            BootstrapReplicateDistribution("b", source, 3)

    def test_the_outer_component_defaults_to_the_label_for_an_array_source(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), replicate_size=5)
        assert list(replicate.event_spec.components) == ["b"]
        assert replicate.event_spec.spec.batch_shape == (5,)
        assert replicate.event_spec.spec.element_spec == Normal("x", 0.0, 1.0).event_spec.spec

    def test_the_outer_component_defaults_to_the_label_for_an_empirical_source(self):
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("data", jnp.ones(6)))
        assert not replicate.event_spec.exposes_record
        assert list(replicate.event_spec.components) == ["b"]

    def test_a_replicate_of_records_is_a_record_batch(self):
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("xy", _rows()), 3)
        raw = replicate._sample(jax.random.PRNGKey(0))
        assert set(raw) == {"x", "y"} and np.shape(raw["x"]) == (3,)
        with workflow_run(seed=3):
            draw = sample(replicate)
        assert isinstance(draw, NumericRecordBatch)
        assert (draw.batch_shape, draw.level_names) == ((3,), ("row",))

    def test_record_rows_are_resampled_jointly(self):
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("xy", _rows()), 50)
        raw = replicate._sample(jax.random.PRNGKey(6))
        np.testing.assert_allclose(raw["y"], 10.0 * raw["x"])

    def test_replicate_size_is_positional(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), 4)
        assert replicate.replicate_size == 4

    def test_event_spec_names_the_outer_component(self):
        replicate = BootstrapReplicateDistribution(
            "b", Normal("x", 0.0, 1.0), 4, event_spec=OutputSpec(dataset=None)
        )
        assert list(replicate.event_spec.components) == ["dataset"]

    def test_an_exposed_record_declaration_is_refused(self):
        with pytest.raises(TypeError, match="RecordSpec"):
            BootstrapReplicateDistribution(
                "b", Normal("x", 0.0, 1.0), 4, event_spec=OutputSpec(RecordSpec(a=()))
            )

    def test_the_draws_are_atoms_of_an_empirical_source(self):
        data = jnp.array([1.0, 5.0, 9.0])
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("y", data), 20)
        assert set(np.asarray(replicate._sample(jax.random.PRNGKey(7))).tolist()) <= {1.0, 5.0, 9.0}

    def test_an_atom_of_zero_weight_is_never_drawn(self):
        source = EmpiricalDistribution("y", jnp.array([1.0, 5.0, 9.0]), jnp.array([1.0, 0.0, 1.0]))
        draws = BootstrapReplicateDistribution("b", source, 400)._sample(jax.random.PRNGKey(8))
        assert 5.0 not in set(np.asarray(draws).tolist())

    def test_it_samples_and_claims_no_moment(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), 3)
        assert isinstance(replicate, SupportsSampling)
        assert not isinstance(replicate, SupportsMean)

    def test_the_bootstrap_distribution_of_a_statistic_is_its_lift(self):
        """The replicate means of a dataset concentrate at its mean, with the standard error's spread."""
        data = jax.random.normal(jax.random.PRNGKey(9), (200,))
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("y", data))
        with workflow_run(seed=4):
            means = Function("stat", lambda b: jnp.mean(b)).with_options(n_broadcast_samples=400)(
                b=replicate
            )
        np.testing.assert_allclose(means._mean(), jnp.mean(data), atol=0.03)
        np.testing.assert_allclose(
            jnp.sqrt(means._variance()), jnp.std(data) / jnp.sqrt(200.0), rtol=0.2
        )


class TestTheReplicateLevel:
    def test_it_defaults_to_an_empirical_sources_one_level(self):
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("xy", _rows()))
        assert replicate.event_spec.spec.level_names == ("row",)

    def test_it_defaults_to_the_component_of_a_whole_term_source(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), 3)
        assert replicate.event_spec.spec.level_names == ("x",)

    def test_a_source_with_several_levels_takes_its_component(self):
        atoms = NumericArrayBatch(
            "draws", jnp.zeros((2, 3)), ("chain", "draw"), element_spec=NumericArraySpec(())
        )
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("theta", atoms))
        assert replicate.event_spec.spec.level_names == ("theta",)

    def test_it_is_required_for_a_source_exposing_several_components(self):
        atoms = NumericRecordBatch(
            "draws",
            {"x": jnp.zeros((2, 3)), "y": jnp.zeros((2, 3))},
            ("chain", "draw"),
            element_spec=NumericRecordSpec(x=(), y=()),
            axes_per_level=(1, 1),
        )
        source = EmpiricalDistribution("post", atoms)
        with pytest.raises(ValueError, match=r"level is required .* \['x', 'y'\]; pass level="):
            BootstrapReplicateDistribution("b", source)
        named = BootstrapReplicateDistribution("b", source, level="row")
        assert named.event_spec.spec.level_names == ("row",)

    def test_level_names_the_replicates_level(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), 3, level="obs")
        assert replicate.event_spec.spec == BatchSpec(Normal("x", 0.0, 1.0).event_spec.spec, obs=3)

    def test_a_level_follows_the_rule_for_component_names(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), 3, level="my level")
        assert replicate.event_spec.spec.level_names == ("my level",)

    @pytest.mark.parametrize("law", [BootstrapReplicateDistribution, BootstrapDistribution])
    @pytest.mark.parametrize("level", ["", "a/b"])
    def test_an_invalid_level_raises_at_construction(self, law, level):
        with pytest.raises(ValueError, match="level names"):
            law("b", Normal("x", 0.0, 1.0), 3, level=level)


# ---------------------------------------------------------------------------
# The bootstrap random measure
# ---------------------------------------------------------------------------


class TestTheBootstrapMeasure:
    def test_the_bootstrap_distribution_is_a_random_measure(self):
        assert issubclass(BootstrapDistribution, RandomMeasure)

    def test_a_draw_is_the_empirical_measure_of_one_replicate(self):
        measure = BootstrapDistribution("B", Normal("x", 0.0, 1.0), 7)
        draw = measure._sample(jax.random.PRNGKey(0))
        assert isinstance(draw, EmpiricalDistribution)
        assert draw.num_atoms == 7

    def test_a_drawn_measure_carries_the_sources_declaration(self):
        source = EmpiricalDistribution("xy", _rows())
        draw = BootstrapDistribution("B", source)._sample(jax.random.PRNGKey(1))
        assert draw.event_spec == source.event_spec
        assert draw.atoms.level_names == ("row",)
        np.testing.assert_allclose(draw.weights, 0.25)

    def test_the_outer_declaration_is_a_law_of_the_sources_event(self):
        source = Normal("x", 0.0, 1.0)
        measure = BootstrapDistribution("B", source, 4)
        assert measure.event_spec == OutputSpec(B=DistributionSpec(source.event_spec))
        named = BootstrapDistribution("B", source, 4, event_spec=OutputSpec(measure=None))
        assert list(named.event_spec.components) == ["measure"]

    def test_draws_under_a_sample_shape_are_an_array_of_measures(self):
        measure = BootstrapDistribution("B", Normal("x", 0.0, 1.0), 5)
        draws = measure._sample(jax.random.PRNGKey(2), (2, 3))
        assert draws.shape == (2, 3)
        assert all(isinstance(draw, EmpiricalDistribution) for draw in draws.flat)
        assert draws[0, 0].num_atoms == 5

    def test_the_mean_is_the_source(self):
        source = EmpiricalDistribution("y", jnp.array([1.0, 2.0, 4.0]))
        measure = BootstrapDistribution("B", source)
        assert isinstance(measure, SupportsMean)
        assert measure._mean() is source

    def test_the_drawn_atoms_are_the_sources(self):
        source = EmpiricalDistribution("y", jnp.array([1.0, 5.0, 9.0]))
        draw = BootstrapDistribution("B", source, 30)._sample(jax.random.PRNGKey(3))
        assert set(np.asarray(draw.atoms.values).tolist()) <= {1.0, 5.0, 9.0}
