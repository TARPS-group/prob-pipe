"""Broadcasting tests for JAX-based distribution API."""

from __future__ import annotations

from contextlib import suppress

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    MultivariateNormal,
    Normal,
    workflow_run,
)
from probpipe.values._function_base import Function


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Basic broadcasting (loop backend) - default returns marginal
# ---------------------------------------------------------------------------


class TestBroadcastingBasic:
    def test_returns_marginal_by_default(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=50, dispatch="sequential")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert hasattr(result, "atoms")
        assert result.num_atoms == 50

    def test_output_values_correct(self):
        def add_one(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1.0

        w = Function(label="add_one", fn=add_one, n_broadcast_samples=200, dispatch="sequential")
        g = Normal(loc=0.0, scale=0.1, label="x")
        with workflow_run(seed=1):
            result = w(x=g)
        # Mean should be ~1.0 (0 + 1)
        assert abs(float(jnp.mean(np.asarray(result.atoms))) - 1.0) < 0.1

    def test_scalar_return(self):
        def compute_norm(x: jnp.ndarray) -> float:
            return float(jnp.linalg.norm(x))

        w = Function(
            label="compute_norm", fn=compute_norm, n_broadcast_samples=20, dispatch="sequential"
        )
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="x")
        with workflow_run(seed=2):
            result = w(x=mvn)
        assert isinstance(result, EmpiricalDistribution)
        assert result.event_spec.spec.vector_size == 1

    def test_positional_args(self):
        """Function accepts positional arguments."""
        from probpipe.functions import function

        @function(
            n_broadcast_samples=30,
            dispatch="sequential",
        )
        def add(x, y):
            return x + y

        # Both positional
        result = add(jnp.array(1.0), jnp.array(2.0))
        np.testing.assert_allclose(float(result), 3.0)

        # Mixed: first positional, second keyword
        result = add(jnp.array(1.0), y=jnp.array(2.0))
        np.testing.assert_allclose(float(result), 3.0)

        # Positional with distribution triggers broadcasting
        g = Normal(loc=0.0, scale=0.1, label="x")
        with workflow_run(seed=5):
            result = add(g, y=jnp.array(1.0))
        assert hasattr(result, "atoms")
        assert result.num_atoms == 30

    def test_no_input_samples_by_default(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=20, dispatch="sequential")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w(x=g)
        assert list(result.event_spec.components) == ["double_it"]


class TestBroadcastingMultipleArgs:
    def test_two_distributions(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = Function(label="add_them", fn=add_them, n_broadcast_samples=100, dispatch="sequential")
        g1 = Normal(loc=1.0, scale=0.1, label="a")
        g2 = Normal(loc=2.0, scale=0.1, label="b")
        with workflow_run(seed=3):
            result = w(a=g1, b=g2)
        assert result.num_atoms == 100
        assert abs(float(jnp.mean(np.asarray(result.atoms))) - 3.0) < 0.2


class TestBroadcastingMixedArgs:
    def test_one_dist_one_concrete(self):
        def scale(x: jnp.ndarray, factor: float) -> jnp.ndarray:
            return x * factor

        w = Function(label="scale", fn=scale, n_broadcast_samples=50, dispatch="sequential")
        g = Normal(loc=5.0, scale=0.1, label="x")
        with workflow_run(seed=4):
            result = w(x=g, factor=3.0)
        assert result.num_atoms == 50
        assert abs(float(jnp.mean(np.asarray(result.atoms))) - 15.0) < 1.0


class TestBroadcastingNSamples:
    def test_default(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        w = Function(label="identity", fn=identity, dispatch="sequential")
        g = Normal(loc=0.0, scale=1.0, label="x")
        with workflow_run(seed=5):
            result = w(x=g)
        assert result.num_atoms == Function.DEFAULT_N_BROADCAST_SAMPLES

    def test_call_time_override(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        w = Function(label="identity", fn=identity, n_broadcast_samples=100, dispatch="sequential")
        g = Normal(loc=0.0, scale=1.0, label="x")
        with workflow_run(seed=6):
            result = w.with_options(n_broadcast_samples=10)(x=g)
        assert result.num_atoms == 10

    def test_call_time_override_rejects_non_integer(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        w = Function(label="identity", fn=identity, dispatch="sequential")
        g = Normal(loc=0.0, scale=1.0, label="x")

        with pytest.raises(TypeError, match="n_broadcast_samples must be an integer"):
            w.with_options(n_broadcast_samples=2.5)(x=g)

    @pytest.mark.parametrize("n_broadcast_samples", [0, -1])
    def test_call_time_override_rejects_non_positive(self, n_broadcast_samples):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        w = Function(label="identity", fn=identity, dispatch="sequential")
        g = Normal(loc=0.0, scale=1.0, label="x")

        with pytest.raises(ValueError, match="n_broadcast_samples must be a positive integer"):
            w.with_options(n_broadcast_samples=n_broadcast_samples)(x=g)


class TestWorkflowControlParameterNames:
    def test_n_broadcast_samples_allowed(self):
        def func(x: jnp.ndarray, n_broadcast_samples: int = 10) -> float:
            return float(n_broadcast_samples)

        wf = Function(label="func", fn=func, dispatch="sequential")

        assert float(wf(x=jnp.asarray(1.0), n_broadcast_samples=3)) == 3.0

    def test_seed_allowed(self):
        def func(x: jnp.ndarray, seed: int = 0) -> float:
            return float(seed)

        wf = Function(label="func", fn=func, dispatch="sequential")

        assert float(wf(x=jnp.asarray(1.0), seed=5)) == 5.0

    def test_include_inputs_allowed(self):
        def func(x: jnp.ndarray, include_inputs: bool = False) -> float:
            return 1.0 if include_inputs else 0.0

        wf = Function(label="func", fn=func, dispatch="sequential")

        assert float(wf(x=jnp.asarray(1.0), include_inputs=True)) == 1.0


class TestFunctionCallResolution:
    def test_unexpected_keyword_raises(self):
        def identity(x):
            return x

        w = Function(label="identity", fn=identity, dispatch="sequential")

        with pytest.raises(TypeError, match="unexpected keyword argument 'typo'"):
            w(x=1.0, typo=2.0)

    def test_var_keyword_name_is_not_unpacked(self):
        seen = []

        def identity(x, **kwargs):
            seen.append(kwargs)
            return x

        w = Function(label="identity", fn=identity, dispatch="sequential")
        out = w(x=1.0, kwargs={"scale": 2.0})

        assert float(out) == 1.0
        assert seen == [{"kwargs": {"scale": 2.0}}]

    def test_extra_keywords_still_reach_var_keyword(self):
        seen = []

        def identity(x, **kwargs):
            seen.append(kwargs)
            return x

        w = Function(label="identity", fn=identity, dispatch="sequential")
        out = w(x=1.0, scale=2.0)

        assert float(out) == 1.0
        assert seen == [{"scale": 2.0}]


class TestNoBroadcasting:
    """Cases where broadcasting should NOT happen."""

    def test_concrete_args_pass_through(self):
        def add(a: float, b: float) -> float:
            return a + b

        w = Function(label="add", fn=add, dispatch="sequential")
        result = w(a=1.0, b=2.0)
        # Concrete args return the Function's auto-wrapped scalar:
        # ``NumericRecord({"add": 3.0})``. The ``__float__`` shim unwraps it.
        assert float(result) == 3.0


# ---------------------------------------------------------------------------
# Empirical enumeration
# ---------------------------------------------------------------------------


class TestBroadcastingEnumeration:
    def test_single_empirical(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        samples = jnp.array([[1.0], [2.0], [3.0]])
        weights = jnp.array([0.2, 0.3, 0.5])
        ed = EmpiricalDistribution("x", samples, weights)

        w = Function(label="identity", fn=identity, n_broadcast_samples=100, dispatch="sequential")
        result = w(x=ed)
        assert result.num_atoms == 3
        np.testing.assert_allclose(result.weights, weights, atol=1e-5)

    def test_two_empiricals_cartesian(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        ed1 = EmpiricalDistribution("x", jnp.array([[1.0], [2.0]]))
        ed2 = EmpiricalDistribution("x", jnp.array([[10.0], [20.0], [30.0]]))

        w = Function(label="add_them", fn=add_them, n_broadcast_samples=100, dispatch="sequential")
        result = w(a=ed1, b=ed2)
        assert result.num_atoms == 6  # 2 x 3

    def test_greedy_cutoff(self):
        """When product exceeds budget, largest empiricals are sampled instead."""

        def sum_three(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
            return a + b + c

        ed_small = EmpiricalDistribution("x", jnp.array([[1.0], [2.0]]))  # n=2
        ed_medium = EmpiricalDistribution(
            "x", jnp.arange(5).reshape(-1, 1).astype(jnp.float32)
        )  # n=5
        ed_large = EmpiricalDistribution(
            "x", jnp.arange(20).reshape(-1, 1).astype(jnp.float32)
        )  # n=20

        w = Function(label="sum_three", fn=sum_three, n_broadcast_samples=50, dispatch="sequential")
        with workflow_run(seed=9):
            result = w(a=ed_small, b=ed_medium, c=ed_large)
        # 2*5=10 enumerated, 50//10=5 reps from ed_large per combo → 50 total
        assert result.num_atoms == 50

    def test_mixed_empirical_and_other(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        ed = EmpiricalDistribution("x", jnp.array([[1.0], [2.0], [3.0]]))
        g = Normal(loc=0.0, scale=1.0, label="b")

        w = Function(label="add_them", fn=add_them, n_broadcast_samples=30, dispatch="sequential")
        with workflow_run(seed=10):
            result = w(a=ed, b=g)
        # 3 empirical combos, 30//3=10 reps each → 30 total
        assert result.num_atoms == 30

    def test_enumeration_input_samples_aligned(self):
        """Input samples should be aligned with output samples when include_inputs=True."""

        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        samples = jnp.array([[1.0], [2.0], [3.0]])
        ed = EmpiricalDistribution("x", samples)

        w = Function(label="identity", fn=identity, n_broadcast_samples=100, dispatch="sequential")
        result = w.with_options(include_inputs=True)(x=ed)
        assert isinstance(result, EmpiricalDistribution)
        assert "x" in result.event_spec.components
        # Each draw of the array law is an array, and stays one in the joint.
        assert result._rows["x"].shape == (3, 1)
        # Output should match input (identity function)
        np.testing.assert_allclose(result._rows["x"], result._rows["identity"], atol=1e-5)


# ---------------------------------------------------------------------------
# Non-numeric results
# ---------------------------------------------------------------------------


class TestBroadcastingNonNumeric:
    def test_string_results(self):
        def describe(x: jnp.ndarray) -> str:
            return f"val={float(x):.2f}"

        w = Function(label="describe", fn=describe, n_broadcast_samples=5, dispatch="sequential")
        g = Normal(loc=0.0, scale=1.0, label="x")
        with workflow_run(seed=11):
            result = w(x=g)
        # Non-numeric results are the atoms of an empirical law over opaque values.
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 5
        assert all(isinstance(r, str) for r in result.atoms._store)


# ---------------------------------------------------------------------------
# JAX vmap backend
# ---------------------------------------------------------------------------


class TestBroadcastingJAX:
    def test_basic_vmap(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=50, dispatch="jax")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=20):
            result = w(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 50

    def test_vmap_values_correct(self):
        def add_one(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1.0

        w = Function(label="add_one", fn=add_one, n_broadcast_samples=200, dispatch="jax")
        g = Normal(loc=0.0, scale=0.1, label="x")
        with workflow_run(seed=21):
            result = w(x=g)
        assert abs(float(jnp.mean(np.asarray(result.atoms))) - 1.0) < 0.1

    def test_vmap_multiple_args(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = Function(label="add_them", fn=add_them, n_broadcast_samples=100, dispatch="jax")
        g1 = Normal(loc=1.0, scale=0.1, label="a")
        g2 = Normal(loc=2.0, scale=0.1, label="b")
        with workflow_run(seed=22):
            result = w(a=g1, b=g2)
        assert result.num_atoms == 100
        assert abs(float(jnp.mean(np.asarray(result.atoms))) - 3.0) < 0.2

    def test_vmap_mixed_dist_and_concrete(self):
        def scale(x: jnp.ndarray, factor: float) -> jnp.ndarray:
            return x * factor

        w = Function(label="scale", fn=scale, n_broadcast_samples=50, dispatch="jax")
        g = Normal(loc=5.0, scale=0.1, label="x")
        with workflow_run(seed=23):
            result = w(x=g, factor=3.0)
        assert result.num_atoms == 50
        assert abs(float(jnp.mean(np.asarray(result.atoms))) - 15.0) < 1.0

    def test_vmap_multivariate(self):
        def halve(x: jnp.ndarray) -> jnp.ndarray:
            return x / 2.0

        w = Function(label="halve", fn=halve, n_broadcast_samples=30, dispatch="jax")
        mvn = MultivariateNormal(loc=jnp.array([4.0, 6.0]), cov=0.01 * jnp.eye(2), label="x")
        with workflow_run(seed=24):
            result = w(x=mvn)
        assert result.num_atoms == 30
        assert result.event_spec.spec.vector_size == 2
        mean = jnp.mean(np.asarray(result.atoms), axis=0)
        np.testing.assert_allclose(mean, jnp.array([2.0, 3.0]), atol=0.2)

    def test_vmap_input_samples(self):
        """include_inputs=True preserves input-output alignment for vmap."""

        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=30, dispatch="jax")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=20):
            result = w.with_options(include_inputs=True)(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert "x" in result.event_spec.components
        assert result._rows["x"].shape[0] == 30
        # Output should be 2x input
        np.testing.assert_allclose(result._rows["double_it"], result._rows["x"] * 2, atol=1e-5)


# ---------------------------------------------------------------------------
# Auto dispatch detection
# ---------------------------------------------------------------------------


class TestAutoDispatch:
    def test_auto_selects_jax_for_traceable(self):
        def pure_jax(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(x)

        w = Function(label="pure_jax", fn=pure_jax, n_broadcast_samples=20, dispatch="auto")
        g = Normal(loc=0.0, scale=1.0, label="x")
        with workflow_run(seed=30):
            result = w(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert not hasattr(w, "_resolved_dispatch")

    def test_auto_falls_back_to_sequential_for_non_traceable(self):
        import scipy.special

        def scipy_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(scipy.special.gamma(np.asarray(x)))

        w = Function(label="scipy_fn", fn=scipy_fn, n_broadcast_samples=20, dispatch="auto")
        g = Normal(loc=2.0, scale=0.1, label="x")
        with workflow_run(seed=31):
            result = w(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert not hasattr(w, "_resolved_dispatch")

    def test_jax_dispatch_rejects_non_traceable_function_with_clear_error(self):
        import scipy.special

        def scipy_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(scipy.special.gamma(np.asarray(x)))

        w = Function(label="scipy_fn", fn=scipy_fn, n_broadcast_samples=20, dispatch="jax")
        g = Normal(loc=2.0, scale=0.1, label="x")

        with pytest.raises(ValueError, match="failed while tracing"):
            w(x=g)

    def test_auto_falls_back_to_row_wise_for_multi_field_joint(self):
        """A multi-field joint broadcast argument
        can't be probed with a single ``event_shape`` (the property
        raises ``NotImplementedError`` / ``TypeError`` on multi-leaf
        instances). The auto-detect path should catch that and
        fall back to row-wise dispatch rather than crash.
        """

        def consume(joint) -> jnp.ndarray:
            # The function works on a Record / dict; the probe never
            # actually calls it under JAX tracing for this case.
            return jnp.asarray(joint["x"] + joint["y"])

        w = Function(
            label="consume",
            fn=consume,
            n_broadcast_samples=10,
            dispatch="auto",
        )
        joint = Normal(loc=0.0, scale=1.0, label="x") * Normal(loc=0.0, scale=1.0, label="y")
        # Probing fails gracefully (NotImplementedError caught inside
        # ``_resolve_dispatch``) and the call-local planner falls back.
        with workflow_run(seed=32), suppress(Exception):
            w(joint=joint)
        assert not hasattr(w, "_resolved_dispatch")


# ---------------------------------------------------------------------------
# Workflow RNG management
# ---------------------------------------------------------------------------


class TestWorkflowRngManagement:
    def test_distinct_bare_calls_give_different_results(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        g = Normal(loc=0.0, scale=1.0, label="x")

        w1 = Function(label="identity", fn=identity, n_broadcast_samples=20, dispatch="sequential")
        r1 = w1(x=g)

        w2 = Function(label="identity", fn=identity, n_broadcast_samples=20, dispatch="sequential")
        r2 = w2(x=g)

        assert not jnp.allclose(np.asarray(r1.atoms), np.asarray(r2.atoms))

    def test_same_workflow_seed_reproduces_a_call(self):
        def identity(x: jnp.ndarray) -> jnp.ndarray:
            return x

        g = Normal(loc=0.0, scale=1.0, label="x")
        w = Function(label="identity", fn=identity, n_broadcast_samples=20, dispatch="sequential")

        with workflow_run(seed=42):
            r1 = w(x=g)
        with workflow_run(seed=42):
            r2 = w(x=g)
        np.testing.assert_allclose(np.asarray(r1.atoms), np.asarray(r2.atoms), atol=1e-5)


# ---------------------------------------------------------------------------
# include_inputs argument
# ---------------------------------------------------------------------------


class TestIncludeInputsArgument:
    def test_include_inputs_at_construction(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(
            label="double_it",
            fn=double_it,
            n_broadcast_samples=20,
            dispatch="sequential",
            include_inputs=True,
        )
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert "x" in result.event_spec.components
        assert result.num_atoms == 20

    def test_include_inputs_at_call_time(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=20, dispatch="sequential")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w.with_options(include_inputs=True)(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert "x" in result.event_spec.components

    def test_default_no_input_samples(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=20, dispatch="sequential")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w(x=g)
        assert isinstance(result, EmpiricalDistribution)
        assert "x" not in result.event_spec.components

    def test_include_inputs_has_named_components(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = Function(label="add_them", fn=add_them, n_broadcast_samples=20, dispatch="sequential")
        g1 = Normal(loc=1.0, scale=0.1, label="a")
        g2 = Normal(loc=2.0, scale=0.1, label="b")
        with workflow_run(seed=0):
            result = w.with_options(include_inputs=True)(a=g1, b=g2)
        assert isinstance(result, EmpiricalDistribution)
        assert list(result.event_spec.components) == ["a", "b", "add_them"]


# ---------------------------------------------------------------------------
# Named components (require include_inputs=True)
# ---------------------------------------------------------------------------


class TestNamedComponents:
    def test_fields(self):
        def add_them(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return a + b

        w = Function(label="add_them", fn=add_them, n_broadcast_samples=20, dispatch="sequential")
        g1 = Normal(loc=1.0, scale=0.1, label="a")
        g2 = Normal(loc=2.0, scale=0.1, label="b")
        with workflow_run(seed=0):
            result = w.with_options(include_inputs=True)(a=g1, b=g2)
        assert list(result.event_spec.components) == ["a", "b", "add_them"]

    def test_getitem_input(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=20, dispatch="sequential")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w.with_options(include_inputs=True)(x=g)
        x_marginal = result._marginal("x")
        assert isinstance(x_marginal, EmpiricalDistribution)
        assert x_marginal.num_atoms == 20

    def test_getitem_output(self):
        def double_it(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        w = Function(label="double_it", fn=double_it, n_broadcast_samples=20, dispatch="sequential")
        g = Normal(loc=1.0, scale=0.5, label="x")
        with workflow_run(seed=0):
            result = w.with_options(include_inputs=True)(x=g)
        out = result._marginal("double_it")
        assert isinstance(out, EmpiricalDistribution)
        assert out.num_atoms == 20


# ---------------------------------------------------------------------------
# Cross-dispatch consistency: sequential / thread / auto must agree
# ---------------------------------------------------------------------------
#
# Empirical enumeration semantics (cartesian product of small
# empiricals, weighted) do not depend on the dispatch mode; under
# ``dispatch="jax"`` the enumeration runs in one ``jax.vmap``.
# ---------------------------------------------------------------------------


class TestDispatchConsistency:
    """Empirical enumeration and count semantics match across
    row-wise dispatch modes."""

    ROWWISE_DISPATCH_MODES = ("sequential", "thread", "auto")
    SAMPLE_DISPATCH_MODES = ("sequential", "thread", "auto", "jax")

    def _run(self, mode, func, **kwargs):
        w = Function(
            label="func",
            fn=func,
            n_broadcast_samples=kwargs.pop("n_broadcast_samples", 100),
            dispatch=mode,
        )
        return w(**kwargs)

    def test_two_empiricals_cartesian_all_modes(self):
        """Cartesian enumeration of two small empiricals must give the
        exact same samples and weights in every backend."""

        def add_them(a, b):
            return a + b

        ed1 = EmpiricalDistribution("x", jnp.array([[1.0], [2.0]]))
        ed2 = EmpiricalDistribution("x", jnp.array([[10.0], [20.0], [30.0]]))

        results = {m: self._run(m, add_them, a=ed1, b=ed2) for m in self.ROWWISE_DISPATCH_MODES}
        # Same size (2 x 3 = 6) in every mode - regression guard.
        for mode, r in results.items():
            assert r.num_atoms == 6, f"{mode}: expected n=6, got {r.num_atoms}"

        # Same sample set (order may differ; compare sorted).
        def _samples_array(d):
            return np.asarray(d.atoms)

        ref = sorted(_samples_array(results["sequential"]).ravel().tolist())
        for mode in ("auto", "thread"):
            got = sorted(_samples_array(results[mode]).ravel().tolist())
            np.testing.assert_allclose(
                got,
                ref,
                err_msg=f"{mode} samples diverged from sequential: {got} vs {ref}",
            )
        # Weights: uniform 1/6 in every mode (inputs unweighted).
        for mode, r in results.items():
            np.testing.assert_allclose(
                r.weights, jnp.full(6, 1 / 6), atol=1e-5, err_msg=f"{mode} weights diverged"
            )

    def test_weighted_empiricals_preserve_weights_all_modes(self):
        """Exact empirical weights survive the product in every backend."""

        def add_them(a, b):
            return a + b

        ed1 = EmpiricalDistribution("x", jnp.array([[1.0], [2.0]]), weights=jnp.array([0.8, 0.2]))
        ed2 = EmpiricalDistribution(
            "x", jnp.array([[10.0], [20.0]]), weights=jnp.array([0.25, 0.75])
        )
        expected_weights = sorted(
            [
                0.8 * 0.25,
                0.8 * 0.75,
                0.2 * 0.25,
                0.2 * 0.75,
            ]
        )
        for mode in self.ROWWISE_DISPATCH_MODES:
            r = self._run(mode, add_them, a=ed1, b=ed2)
            assert r.num_atoms == 4
            got = sorted(np.asarray(r.weights).tolist())
            np.testing.assert_allclose(
                got,
                expected_weights,
                atol=1e-5,
                err_msg=f"{mode} weights diverged: {got} vs {expected_weights}",
            )

    def test_mixed_empirical_and_parametric_count_all_modes(self):
        """Mixed empirical and continuous inputs preserve row-wise results."""

        def add_them(a, b):
            return a + b

        ed = EmpiricalDistribution("x", jnp.array([[1.0], [2.0], [3.0]]))
        g = Normal(loc=0.0, scale=1.0, label="b")

        samples = []
        for mode in self.ROWWISE_DISPATCH_MODES:
            with workflow_run(seed=0):
                r = self._run(mode, add_them, a=ed, b=g, n_broadcast_samples=30)
            # 3 empirical combos x 10 reps each = 30 evaluations.
            assert r.num_atoms == 30, f"{mode}: expected n=30, got {r.num_atoms}"
            np.testing.assert_allclose(float(r.weights.sum()), 1.0, atol=1e-5)
            samples.append(np.asarray(r.atoms))

        for sample_values in samples[1:]:
            np.testing.assert_array_equal(sample_values, samples[0])

    def test_over_budget_empirical_falls_to_sampling_all_modes(self):
        """When a single empirical exceeds the sample budget, every
        backend falls back to resampling and returns exactly
        ``n_broadcast_samples`` evaluations."""

        def identity(x):
            return x

        big = EmpiricalDistribution("x", jnp.arange(200).reshape(-1, 1).astype(jnp.float32))
        for mode in self.SAMPLE_DISPATCH_MODES:
            with workflow_run(seed=0):
                r = self._run(mode, identity, x=big, n_broadcast_samples=20)
            assert r.num_atoms == 20, f"{mode}: expected n=20, got {r.num_atoms}"

    def test_no_empiricals_all_modes_same_count(self):
        """Without empirical inputs every backend preserves sampled rows."""

        def add_them(a, b):
            return a + b

        n1 = Normal(loc=0.0, scale=1.0, label="a")
        n2 = Normal(loc=5.0, scale=1.0, label="b")
        samples = []
        for mode in self.SAMPLE_DISPATCH_MODES:
            with workflow_run(seed=0):
                r = self._run(mode, add_them, a=n1, b=n2, n_broadcast_samples=50)
            assert r.num_atoms == 50, f"{mode}: expected n=50, got {r.num_atoms}"
            samples.append(np.asarray(r.atoms))

        for sample_values in samples[1:]:
            np.testing.assert_array_equal(sample_values, samples[0])

    def test_jax_dispatch_maps_exact_empirical_enumeration(self):
        def add_them(a, b):
            return a + b

        ed1 = EmpiricalDistribution("x", jnp.array([[1.0], [2.0]]), weights=jnp.array([0.8, 0.2]))
        ed2 = EmpiricalDistribution(
            "x", jnp.array([[10.0], [20.0]]), weights=jnp.array([0.25, 0.75])
        )
        mapped = self._run("jax", add_them, a=ed1, b=ed2, n_broadcast_samples=20)
        sequential = self._run("sequential", add_them, a=ed1, b=ed2, n_broadcast_samples=20)
        assert mapped.provenance.metadata["dispatch"] == "jax"
        np.testing.assert_array_equal(np.asarray(mapped.atoms), np.asarray(sequential.atoms))
        np.testing.assert_array_equal(np.asarray(mapped.weights), np.asarray(sequential.weights))
