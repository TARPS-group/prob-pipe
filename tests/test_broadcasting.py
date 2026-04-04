"""Broadcasting tests for JAX-based distribution API."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import BroadcastDistribution, EmpiricalDistribution, MultivariateNormal, Normal, ArrayDistribution
from probpipe.core.node import WorkflowFunction


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Basic broadcasting (loop backend) — default returns marginal
# ---------------------------------------------------------------------------

class TestBroadcastingBasic:
    def test_returns_marginal_by_default(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=50, vectorize="loop", seed=0)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g)
        assert not isinstance(result, BroadcastDistribution)
        assert hasattr(result, 'samples')
        assert result.n == 50

    def test_output_values_correct(self):
        def add_one(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1.0

        w = WorkflowFunction(func=add_one, n_broadcast_samples=200, vectorize="loop", seed=1)
        g = Normal(loc=0.0, scale=0.1)
        result = w(x=g)
        # Mean should be ~1.0 (0 + 1)
        assert abs(float(jnp.mean(result.samples)) - 1.0) < 0.1

    def test_scalar_return(self):
        def compute_norm(x: jnp.ndarray) -> float:
            return float(jnp.linalg.norm(x))

        w = WorkflowFunction(func=compute_norm, n_broadcast_samples=20, vectorize="loop", seed=2)
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))
        result = w(x=mvn)
        assert not isinstance(result, BroadcastDistribution)
        assert result.dim == 1

    def test_positional_args(self):
        """WorkflowFunction accepts positional arguments."""
        from probpipe.core.node import workflow_function

        @workflow_function(n_broadcast_samples=30, vectorize="loop", seed=5)
        def add(x, y):
            return x + y

        # Both positional
        result = add(jnp.array(1.0), jnp.array(2.0))
        np.testing.assert_allclose(float(result), 3.0)

        # Mixed: first positional, second keyword
        result = add(jnp.array(1.0), y=jnp.array(2.0))
        np.testing.assert_allclose(float(result), 3.0)

        # Positional with distribution triggers broadcasting
        g = Normal(loc=0.0, scale=0.1)
        result = add(g, y=jnp.array(1.0))
        assert hasattr(result, 'samples')
        assert result.n == 30

    def test_no_input_samples_by_default(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=20, vectorize="loop", seed=0)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g)
        assert not hasattr(result, 'input_samples')


class TestBroadcastingMultipleArgs:
    def test_two_distributions(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = WorkflowFunction(func=add_them, n_broadcast_samples=100, vectorize="loop", seed=3)
        g1 = Normal(loc=1.0, scale=0.1)
        g2 = Normal(loc=2.0, scale=0.1)
        result = w(a=g1, b=g2)
        assert result.n == 100
        assert abs(float(jnp.mean(result.samples)) - 3.0) < 0.2


class TestBroadcastingMixedArgs:
    def test_one_dist_one_concrete(self):
        def scale(x: jnp.ndarray, factor: float) -> jnp.ndarray:
            return x * factor

        w = WorkflowFunction(func=scale, n_broadcast_samples=50, vectorize="loop", seed=4)
        g = Normal(loc=5.0, scale=0.1)
        result = w(x=g, factor=3.0)
        assert result.n == 50
        assert abs(float(jnp.mean(result.samples)) - 15.0) < 1.0


class TestBroadcastingNSamples:
    def test_default(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        w = WorkflowFunction(func=identity, vectorize="loop", seed=5)
        g = Normal(loc=0.0, scale=1.0)
        result = w(x=g)
        assert result.n == WorkflowFunction.DEFAULT_N_BROADCAST_SAMPLES

    def test_call_time_override(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        w = WorkflowFunction(func=identity, n_broadcast_samples=100, vectorize="loop", seed=6)
        g = Normal(loc=0.0, scale=1.0)
        result = w(x=g, n_broadcast_samples=10)
        assert result.n == 10


class TestReservedParameterNames:
    def test_n_broadcast_samples_forbidden(self):
        def bad_fn(x: jnp.ndarray, n_broadcast_samples: int = 10) -> jnp.ndarray:
            return x

        with pytest.raises(ValueError, match="reserved"):
            WorkflowFunction(func=bad_fn, vectorize="loop")

    def test_seed_forbidden(self):
        def bad_fn(x: jnp.ndarray, seed: int = 0) -> jnp.ndarray:
            return x

        with pytest.raises(ValueError, match="reserved"):
            WorkflowFunction(func=bad_fn, vectorize="loop")

    def test_include_inputs_forbidden(self):
        def bad_fn(x: jnp.ndarray, include_inputs: bool = False) -> jnp.ndarray:
            return x

        with pytest.raises(ValueError, match="reserved"):
            WorkflowFunction(func=bad_fn, vectorize="loop")


class TestNoBroadcasting:
    """Cases where broadcasting should NOT happen."""

    def test_concrete_args_pass_through(self):
        def add(a: float, b: float) -> float:
            return a + b

        w = WorkflowFunction(func=add, vectorize="loop")
        result = w(a=1.0, b=2.0)
        assert result == 3.0


# ---------------------------------------------------------------------------
# Empirical enumeration
# ---------------------------------------------------------------------------

class TestBroadcastingEnumeration:
    def test_single_empirical(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        samples = jnp.array([[1.0], [2.0], [3.0]])
        weights = jnp.array([0.2, 0.3, 0.5])
        ed = EmpiricalDistribution(samples, weights)

        w = WorkflowFunction(func=identity, n_broadcast_samples=100, vectorize="loop", seed=7)
        result = w(x=ed)
        assert result.n == 3
        np.testing.assert_allclose(result.weights, weights, atol=1e-5)

    def test_two_empiricals_cartesian(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        ed1 = EmpiricalDistribution(jnp.array([[1.0], [2.0]]))
        ed2 = EmpiricalDistribution(jnp.array([[10.0], [20.0], [30.0]]))

        w = WorkflowFunction(func=add_them, n_broadcast_samples=100, vectorize="loop", seed=8)
        result = w(a=ed1, b=ed2)
        assert result.n == 6  # 2 x 3

    def test_greedy_cutoff(self):
        """When product exceeds budget, largest empiricals are sampled instead."""
        def sum_three(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
            return a + b + c

        ed_small = EmpiricalDistribution(jnp.array([[1.0], [2.0]]))          # n=2
        ed_medium = EmpiricalDistribution(jnp.arange(5).reshape(-1, 1).astype(jnp.float32))  # n=5
        ed_large = EmpiricalDistribution(jnp.arange(20).reshape(-1, 1).astype(jnp.float32))  # n=20

        w = WorkflowFunction(func=sum_three, n_broadcast_samples=50, vectorize="loop", seed=9)
        result = w(a=ed_small, b=ed_medium, c=ed_large)
        # 2*5=10 enumerated, 50//10=5 reps from ed_large per combo → 50 total
        assert result.n == 50

    def test_mixed_empirical_and_other(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        ed = EmpiricalDistribution(jnp.array([[1.0], [2.0], [3.0]]))
        g = Normal(loc=0.0, scale=1.0)

        w = WorkflowFunction(func=add_them, n_broadcast_samples=30, vectorize="loop", seed=10)
        result = w(a=ed, b=g)
        # 3 empirical combos, 30//3=10 reps each → 30 total
        assert result.n == 30

    def test_enumeration_input_samples_aligned(self):
        """Input samples should be aligned with output samples when include_inputs=True."""
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        samples = jnp.array([[1.0], [2.0], [3.0]])
        ed = EmpiricalDistribution(samples)

        w = WorkflowFunction(func=identity, n_broadcast_samples=100, vectorize="loop", seed=7)
        result = w(x=ed, include_inputs=True)
        assert isinstance(result, BroadcastDistribution)
        assert "x" in result.input_samples
        assert result.input_samples["x"].shape == (3, 1)
        # Output should match input (identity function)
        marginal = result.marginalize()
        np.testing.assert_allclose(result.input_samples["x"], marginal.samples, atol=1e-5)


# ---------------------------------------------------------------------------
# Non-numeric results
# ---------------------------------------------------------------------------

class TestBroadcastingNonNumeric:
    def test_string_results(self):
        def describe(x: jnp.ndarray) -> str:
            return f"val={float(x):.2f}"

        w = WorkflowFunction(func=describe, n_broadcast_samples=5, vectorize="loop", seed=11)
        g = Normal(loc=0.0, scale=1.0)
        result = w(x=g)
        # Non-numeric results still return a marginal (ListMarginal)
        assert not isinstance(result, BroadcastDistribution)
        assert len(result.items) == 5
        assert all(isinstance(r, str) for r in result.items)


# ---------------------------------------------------------------------------
# JAX vmap backend
# ---------------------------------------------------------------------------

class TestBroadcastingJAX:
    def test_basic_vmap(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=50, vectorize="jax", seed=20)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g)
        assert not isinstance(result, BroadcastDistribution)
        assert result.n == 50

    def test_vmap_values_correct(self):
        def add_one(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1.0

        w = WorkflowFunction(func=add_one, n_broadcast_samples=200, vectorize="jax", seed=21)
        g = Normal(loc=0.0, scale=0.1)
        result = w(x=g)
        assert abs(float(jnp.mean(result.samples)) - 1.0) < 0.1

    def test_vmap_multiple_args(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = WorkflowFunction(func=add_them, n_broadcast_samples=100, vectorize="jax", seed=22)
        g1 = Normal(loc=1.0, scale=0.1)
        g2 = Normal(loc=2.0, scale=0.1)
        result = w(a=g1, b=g2)
        assert result.n == 100
        assert abs(float(jnp.mean(result.samples)) - 3.0) < 0.2

    def test_vmap_mixed_dist_and_concrete(self):
        def scale(x: jnp.ndarray, factor: float) -> jnp.ndarray:
            return x * factor

        w = WorkflowFunction(func=scale, n_broadcast_samples=50, vectorize="jax", seed=23)
        g = Normal(loc=5.0, scale=0.1)
        result = w(x=g, factor=3.0)
        assert result.n == 50
        assert abs(float(jnp.mean(result.samples)) - 15.0) < 1.0

    def test_vmap_multivariate(self):
        def halve(x: jnp.ndarray) -> jnp.ndarray:
            return x / 2.0

        w = WorkflowFunction(func=halve, n_broadcast_samples=30, vectorize="jax", seed=24)
        mvn = MultivariateNormal(loc=jnp.array([4.0, 6.0]), cov=0.01 * jnp.eye(2))
        result = w(x=mvn)
        assert result.n == 30
        assert result.dim == 2
        mean = jnp.mean(result.samples, axis=0)
        np.testing.assert_allclose(mean, jnp.array([2.0, 3.0]), atol=0.2)

    def test_vmap_input_samples(self):
        """include_inputs=True preserves input–output alignment for vmap."""
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=30, vectorize="jax", seed=20)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g, include_inputs=True)
        assert isinstance(result, BroadcastDistribution)
        assert "x" in result.input_samples
        assert result.input_samples["x"].shape[0] == 30
        # Output should be 2x input
        marginal = result.marginalize()
        np.testing.assert_allclose(
            marginal.samples, result.input_samples["x"] * 2, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Auto backend detection
# ---------------------------------------------------------------------------

class TestAutoBackend:
    def test_auto_selects_jax_for_traceable(self):
        def pure_jax(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(x)

        w = WorkflowFunction(func=pure_jax, n_broadcast_samples=20, vectorize="auto", seed=30)
        g = Normal(loc=0.0, scale=1.0)
        result = w(x=g)
        assert not isinstance(result, BroadcastDistribution)
        assert w._resolved_vectorize == "jax"

    def test_auto_falls_back_to_loop_for_non_traceable(self):
        import scipy.special

        def scipy_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(scipy.special.gamma(np.asarray(x)))

        w = WorkflowFunction(func=scipy_fn, n_broadcast_samples=20, vectorize="auto", seed=31)
        g = Normal(loc=2.0, scale=0.1)
        result = w(x=g)
        assert not isinstance(result, BroadcastDistribution)
        assert w._resolved_vectorize == "loop"


# ---------------------------------------------------------------------------
# Seed / key management
# ---------------------------------------------------------------------------

class TestSeedManagement:
    def test_different_seeds_give_different_results(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        g = Normal(loc=0.0, scale=1.0)

        w1 = WorkflowFunction(func=identity, n_broadcast_samples=20, vectorize="loop", seed=0)
        r1 = w1(x=g)

        w2 = WorkflowFunction(func=identity, n_broadcast_samples=20, vectorize="loop", seed=99)
        r2 = w2(x=g)

        assert not jnp.allclose(r1.samples, r2.samples)

    def test_seed_override_at_call_time(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        g = Normal(loc=0.0, scale=1.0)
        w = WorkflowFunction(func=identity, n_broadcast_samples=20, vectorize="loop", seed=0)

        r1 = w(x=g, seed=42)
        # Reset and call with same seed
        r2 = w(x=g, seed=42)
        np.testing.assert_allclose(r1.samples, r2.samples, atol=1e-5)


# ---------------------------------------------------------------------------
# include_inputs argument
# ---------------------------------------------------------------------------

class TestIncludeInputsArgument:
    def test_include_inputs_at_construction(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=20, vectorize="loop",
                             seed=0, include_inputs=True)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g)
        assert isinstance(result, BroadcastDistribution)
        assert "x" in result.input_samples
        assert result.n == 20

    def test_include_inputs_at_call_time(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=20, vectorize="loop", seed=0)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g, include_inputs=True)
        assert isinstance(result, BroadcastDistribution)
        assert "x" in result.input_samples

    def test_default_no_input_samples(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=20, vectorize="loop", seed=0)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g)
        assert not isinstance(result, BroadcastDistribution)
        assert not hasattr(result, 'input_samples')

    def test_include_inputs_has_named_components(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = WorkflowFunction(func=add_them, n_broadcast_samples=20, vectorize="loop", seed=0)
        g1 = Normal(loc=1.0, scale=0.1)
        g2 = Normal(loc=2.0, scale=0.1)
        result = w(a=g1, b=g2, include_inputs=True)
        assert isinstance(result, BroadcastDistribution)
        assert "a" in result.component_names
        assert "b" in result.component_names
        assert "_output" in result.component_names


# ---------------------------------------------------------------------------
# Named components (require include_inputs=True)
# ---------------------------------------------------------------------------

class TestNamedComponents:
    def test_component_names(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = WorkflowFunction(func=add_them, n_broadcast_samples=20, vectorize="loop", seed=0)
        g1 = Normal(loc=1.0, scale=0.1)
        g2 = Normal(loc=2.0, scale=0.1)
        result = w(a=g1, b=g2, include_inputs=True)
        assert "a" in result.component_names
        assert "b" in result.component_names
        assert "_output" in result.component_names

    def test_getitem_input(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=20, vectorize="loop", seed=0)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g, include_inputs=True)
        x_marginal = result["x"]
        assert isinstance(x_marginal, EmpiricalDistribution)
        assert x_marginal.n == 20

    def test_getitem_output(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = WorkflowFunction(func=double_it, n_broadcast_samples=20, vectorize="loop", seed=0)
        g = Normal(loc=1.0, scale=0.5)
        result = w(x=g, include_inputs=True)
        out = result["_output"]
        assert hasattr(out, 'samples')
