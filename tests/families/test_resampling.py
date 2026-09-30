"""Contracts of the resampling families (VII.2): the smoothing kernels, the KDE, and the bootstrap.

The smoothing kernels are new in ``families/``; the bootstrap replicate, the
bootstrap distribution, and the KDE are checked where they are defined today,
and the contracts they do not meet yet are pending.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from probpipe import (
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    EmpiricalDistribution,
    KDEDistribution,
    Normal,
    NumericRecordBatch,
    NumericRecordSpec,
    OutputSpec,
    RandomMeasure,
    Record,
)
from probpipe.core._numeric_record import NumericRecord
from probpipe.families import EpanechnikovKernel, GaussianKernel, SmoothingKernel

_KERNELS = [GaussianKernel, EpanechnikovKernel]


@pytest.fixture
def centers():
    return jnp.array([[0.0, 1.0], [2.0, -1.0], [4.0, 0.5]])


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
    def test_record_centers_and_scales_flatten_to_their_coordinates(self, kernel, centers):
        batch = NumericRecordBatch(
            "atoms",
            {"a": centers[:, 0], "b": centers[:, 1]},
            "atom",
            element_spec=NumericRecordSpec(a=(), b=()),
        )
        scales = NumericRecord("h", {"a": 0.5, "b": 1.5})
        from_records = kernel.build_kernels(batch, scales)
        from_arrays = kernel.build_kernels(centers, jnp.array([0.5, 1.5]))
        x = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(
            from_records._log_density(x), from_arrays._log_density(x), rtol=1e-6
        )

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
        with pytest.raises(ValueError, match="atoms"):
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

    @pytest.mark.pending(
        reason="the KDE takes a SmoothingKernel class and builds its bank", raises=TypeError
    )
    def test_the_kde_takes_a_kernel_class(self, centers):
        kde = KDEDistribution("k", centers, 0.5, kernel=EpanechnikovKernel)
        atom_variance = jnp.var(centers, axis=0)
        np.testing.assert_allclose(
            kde._variance(), atom_variance + 0.25 * EpanechnikovKernel.variance, rtol=1e-5
        )

    @pytest.mark.pending(
        reason="bandwidth accepts the name of a selection rule", raises=(TypeError, ValueError)
    )
    @pytest.mark.parametrize("rule", ["scott", "silverman"])
    def test_bandwidth_accepts_a_selection_rule(self, rule, centers):
        kde = KDEDistribution("k", centers, rule)
        assert kde.event_spec.components == {"k": kde.event_spec.spec}


# ---------------------------------------------------------------------------
# The bootstrap replicate
# ---------------------------------------------------------------------------


class TestTheBootstrapReplicate:
    def test_a_draw_is_replicate_size_draws_of_a_sampling_source(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), replicate_size=5)
        assert replicate._sample(jax.random.PRNGKey(0)).shape == (5,)
        assert replicate._sample(jax.random.PRNGKey(0), (3,)).shape == (3, 5)

    def test_replicate_size_is_required_for_a_source_without_atoms(self):
        with pytest.raises(ValueError, match="replicate_size"):
            BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0))

    def test_replicate_size_defaults_to_the_source_atom_count(self):
        source = EmpiricalDistribution("data", jnp.arange(6.0))
        assert BootstrapReplicateDistribution("b", source).replicate_size == 6

    def test_the_outer_component_defaults_to_the_label_for_an_array_source(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), replicate_size=5)
        assert list(replicate.event_spec.components) == ["b"]
        assert replicate.event_spec.spec.shape == (5,)

    @pytest.mark.pending(
        reason="a replicate's outer declaration is its own, under the law's name",
        raises=AssertionError,
    )
    def test_the_outer_component_defaults_to_the_label_for_an_empirical_source(self):
        replicate = BootstrapReplicateDistribution("b", EmpiricalDistribution("data", jnp.ones(6)))
        assert not replicate.event_spec.exposes_record
        assert list(replicate.event_spec.components) == ["b"]

    @pytest.mark.pending(
        reason="a replicate of record draws is a batch of records", raises=AssertionError
    )
    def test_a_replicate_of_records_is_a_record_batch(self):
        source = EmpiricalDistribution("xy", Record("xy", x=jnp.arange(4.0), y=jnp.ones(4)))
        replicate = BootstrapReplicateDistribution("b", source, replicate_size=3)
        assert isinstance(replicate._sample(jax.random.PRNGKey(0)), NumericRecordBatch)

    @pytest.mark.pending(reason="replicate_size is positional-or-keyword", raises=TypeError)
    def test_replicate_size_is_positional(self):
        replicate = BootstrapReplicateDistribution("b", Normal("x", 0.0, 1.0), 4)
        assert replicate.replicate_size == 4

    @pytest.mark.pending(reason="event_spec names the replicate's component", raises=TypeError)
    def test_event_spec_names_the_outer_component(self):
        replicate = BootstrapReplicateDistribution(
            "b", Normal("x", 0.0, 1.0), 4, event_spec=OutputSpec(dataset=None)
        )
        assert list(replicate.event_spec.components) == ["dataset"]


# ---------------------------------------------------------------------------
# The bootstrap random measure
# ---------------------------------------------------------------------------


class TestTheBootstrapMeasure:
    @pytest.mark.pending(
        reason="BootstrapDistribution is the bootstrap random measure", raises=AssertionError
    )
    def test_the_bootstrap_distribution_is_a_random_measure(self):
        assert issubclass(BootstrapDistribution, RandomMeasure)

    @pytest.mark.pending(
        reason="a bootstrap draw is the empirical measure of one replicate",
        raises=(TypeError, ValueError),
    )
    def test_a_draw_is_the_empirical_measure_of_one_replicate(self):
        measure = BootstrapDistribution("B", Normal("x", 0.0, 1.0), 7)
        draw = measure._sample(jax.random.PRNGKey(0))
        assert isinstance(draw, EmpiricalDistribution)
        assert draw.num_atoms == 7
