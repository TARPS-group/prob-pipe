import numpy as np
import pytest
from numpy.typing import NDArray

from probpipe import EmpiricalDistribution, Gaussian, Distribution
from probpipe.core.node import Workflow, wf


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestBroadcastingSingleArg:
    """Broadcasting a single Distribution argument."""

    def test_returns_empirical_distribution(self, rng):
        def double_it(x: NDArray) -> NDArray:
            return x * 2

        w = Workflow(func=double_it, n_broadcast_samples=50)
        g = Gaussian(mean=np.array([1.0, 2.0]), cov=np.eye(2), rng=rng)
        result = w(x=g)

        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 50
        assert result.dim == 2

    def test_output_values_are_correct(self, rng):
        def add_one(x: NDArray) -> NDArray:
            return x + 1.0

        w = Workflow(func=add_one, n_broadcast_samples=500)
        g = Gaussian(mean=np.array([0.0]), cov=np.eye(1), rng=rng)
        result = w(x=g)

        assert result.n == 500
        assert result.dim == 1
        # add_one shifts the mean by 1: Gaussian(0,1) + 1 → mean ≈ 1.0
        assert abs(float(result.mean()) - 1.0) < 0.3, (
            f"Expected mean ≈ 1.0, got {float(result.mean()):.3f}"
        )

    def test_scalar_return(self, rng):
        def norm(x: NDArray) -> float:
            return float(np.linalg.norm(x))

        w = Workflow(func=norm, n_broadcast_samples=25)
        g = Gaussian(mean=np.zeros(3), cov=np.eye(3), rng=rng)
        result = w(x=g)

        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 25
        assert result.dim == 1


class TestBroadcastingMultipleArgs:
    """Broadcasting multiple independent Distribution arguments."""

    def test_two_distributions(self, rng):
        def add_them(a: NDArray, b: NDArray) -> NDArray:
            return a + b

        w = Workflow(func=add_them, n_broadcast_samples=30)
        g1 = Gaussian(mean=np.array([1.0]), cov=np.eye(1), rng=rng)
        g2 = Gaussian(mean=np.array([2.0]), cov=np.eye(1), rng=rng)
        result = w(a=g1, b=g2)

        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 30
        assert result.dim == 1


class TestBroadcastingMixedArgs:
    """Mix of Distribution and concrete arguments."""

    def test_one_dist_one_concrete(self, rng):
        def scale(x: NDArray, factor: float) -> NDArray:
            return x * factor

        w = Workflow(func=scale, n_broadcast_samples=20)
        g = Gaussian(mean=np.array([5.0]), cov=np.eye(1), rng=rng)
        result = w(x=g, factor=3.0)

        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 20
        assert result.dim == 1


class TestBroadcastingNSamples:
    """n_broadcast_samples configuration."""

    def test_default_n_broadcast_samples(self, rng):
        """Default n_broadcast_samples should match the class constant."""
        def identity(x: NDArray) -> NDArray:
            return x

        w = Workflow(func=identity)
        g = Gaussian(mean=np.array([0.0]), cov=np.eye(1), rng=rng)
        result = w(x=g)

        assert result.n == Workflow.DEFAULT_N_BROADCAST_SAMPLES

    def test_call_time_override(self, rng):
        def identity(x: NDArray) -> NDArray:
            return x

        w = Workflow(func=identity, n_broadcast_samples=100)
        g = Gaussian(mean=np.array([0.0]), cov=np.eye(1), rng=rng)
        result = w(x=g, n_broadcast_samples=10)

        assert result.n == 10

    def test_n_broadcast_samples_not_stolen_from_function(self, rng):
        """If the function itself has an n_samples parameter, don't intercept it."""

        def sample_wrapper(x: NDArray, n_samples: int) -> NDArray:
            return np.tile(x, (n_samples, 1))

        w = Workflow(func=sample_wrapper, n_broadcast_samples=5)
        g = Gaussian(mean=np.array([1.0]), cov=np.eye(1), rng=rng)
        # n_samples=3 should go to the function, not be intercepted
        result = w(x=g, n_samples=3)

        assert isinstance(result, EmpiricalDistribution)
        # Broadcasting uses the construction default (5), n_samples=3 goes to function
        assert result.n == 5


class TestNoBroadcasting:
    """Cases where broadcasting should NOT happen."""

    def test_distribution_hint_uses_conversion(self, rng):
        """When type hint IS a Distribution subclass, convert instead of broadcast."""

        def takes_dist(x: Distribution) -> NDArray:
            return x.sample(1)

        w = Workflow(func=takes_dist)
        g = Gaussian(mean=np.array([0.0]), cov=np.eye(1), rng=rng)
        result = w(x=g)

        # Should call function directly (no broadcasting), passing the Distribution
        assert isinstance(result, np.ndarray)

    def test_concrete_args_pass_through(self):
        """No Distribution arguments means no broadcasting."""

        def add(a: float, b: float) -> float:
            return a + b

        w = Workflow(func=add)
        result = w(a=1.0, b=2.0)

        assert result == 3.0


class TestBroadcastingEnumeration:
    """EmpiricalDistribution args should enumerate samples and propagate weights."""

    def test_single_empirical_uses_all_samples(self, rng):
        def identity(x: NDArray) -> NDArray:
            return x

        samples = np.array([[1.0], [2.0], [3.0]])
        weights = np.array([0.2, 0.3, 0.5])
        ed = EmpiricalDistribution(samples, weights, rng=rng)

        w = Workflow(func=identity, n_broadcast_samples=100)
        result = w(x=ed)

        assert isinstance(result, EmpiricalDistribution)
        # Should enumerate all 3 samples, not resample 100
        assert result.n == 3
        np.testing.assert_allclose(result.weights, weights / weights.sum())
        np.testing.assert_allclose(result.samples, samples)

    def test_two_empiricals_cartesian_product(self, rng):
        def add_them(a: NDArray, b: NDArray) -> NDArray:
            return a + b

        ed1 = EmpiricalDistribution(np.array([[1.0], [2.0]]), rng=rng)
        ed2 = EmpiricalDistribution(np.array([[10.0], [20.0], [30.0]]), rng=rng)

        w = Workflow(func=add_them, n_broadcast_samples=100)
        result = w(a=ed1, b=ed2)

        assert isinstance(result, EmpiricalDistribution)
        # Cartesian product: 2 * 3 = 6 evaluations
        assert result.n == 6
        # Uniform weights on both -> uniform product weights
        np.testing.assert_allclose(result.weights, np.ones(6) / 6)

    def test_large_empirical_falls_back_to_sampling(self, rng):
        def identity(x: NDArray) -> NDArray:
            return x

        # 200 samples > n_broadcast_samples=50, so should sample
        big_ed = EmpiricalDistribution(np.random.default_rng(0).standard_normal((200, 2)), rng=rng)

        w = Workflow(func=identity, n_broadcast_samples=50)
        result = w(x=big_ed)

        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 50

    def test_product_exceeds_n_broadcast_samples_falls_back(self, rng):
        def add_them(a: NDArray, b: NDArray) -> NDArray:
            return a + b

        ed1 = EmpiricalDistribution(np.arange(10).reshape(-1, 1).astype(float), rng=rng)
        ed2 = EmpiricalDistribution(np.arange(10).reshape(-1, 1).astype(float), rng=rng)

        # Product would be 100, but n_broadcast_samples=50 -> falls back to sampling
        w = Workflow(func=add_them, n_broadcast_samples=50)
        result = w(a=ed1, b=ed2)

        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 50

    def test_mixed_empirical_and_other(self, rng):
        """One EmpiricalDistribution + one Gaussian: enumerate empirical, sample Gaussian."""
        call_count = 0

        def add_them(a: NDArray, b: NDArray) -> NDArray:
            nonlocal call_count
            call_count += 1
            return a + b

        ed = EmpiricalDistribution(np.array([[1.0], [2.0], [3.0]]), rng=rng)
        g = Gaussian(mean=np.array([0.0]), cov=np.eye(1), rng=rng)

        w = Workflow(func=add_them, n_broadcast_samples=100)
        result = w(a=ed, b=g)

        assert isinstance(result, EmpiricalDistribution)
        # Enumerates 3 empirical combos × 33 reps each (floor(100/3)=33) = 99
        reps_per_combo = 100 // 3  # 33
        expected_n = 3 * reps_per_combo  # 99
        assert result.n == expected_n
        assert call_count == expected_n


class TestBroadcastingNonNumericResults:
    """When results can't be stacked into a numpy array, return a list."""

    def test_string_results(self, rng):
        def describe(x: NDArray) -> str:
            return f"mean={x.mean():.2f}"

        w = Workflow(func=describe, n_broadcast_samples=5)
        g = Gaussian(mean=np.array([1.0]), cov=np.eye(1), rng=rng)
        result = w(x=g)

        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(r, str) for r in result)

    def test_dict_results(self, rng):
        def summarize(x: NDArray) -> dict:
            return {"sum": float(x.sum()), "len": len(x)}

        w = Workflow(func=summarize, n_broadcast_samples=3)
        g = Gaussian(mean=np.zeros(2), cov=np.eye(2), rng=rng)
        result = w(x=g)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, dict) for r in result)
