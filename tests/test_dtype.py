"""Regression tests for dtype handling under JAX's x64 mode.

Each test runs in a subprocess so that ``jax.config.update("jax_enable_x64",
True)`` is set before any JAX/TFP/probpipe code runs. Setting the flag
mid-process is unreliable because JAX caches a default dtype on import.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest


def _run_x64(snippet: str) -> str:
    """Execute *snippet* in a fresh Python process with x64 enabled.

    Returns the captured stdout, stripped. Stderr is included only on
    failure to surface useful tracebacks.
    """
    # Disable Prefect for the subprocess too. The session-scoped autouse
    # fixture in tests/conftest.py only affects the parent process; this
    # subprocess starts fresh and would otherwise pick up whatever
    # ``WorkflowKind.DEFAULT`` auto-resolves to (TASK if Prefect is
    # importable, which then fails for any user whose ``PREFECT_API_URL``
    # points at an unreachable server).
    program = textwrap.dedent(
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from probpipe import WorkflowKind, prefect_config
        prefect_config.workflow_kind = WorkflowKind.OFF
        """
    ) + textwrap.dedent(snippet)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"x64 subprocess failed:\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# x32 default — distributions follow JAX's default float dtype
# ---------------------------------------------------------------------------


def test_x32_default_normal_is_float32():
    from probpipe.distributions.continuous import Normal
    import probpipe.core.ops as ops

    n = Normal(loc=0.0, scale=1.0, name="n")
    assert n.dtype == jnp.float32
    assert ops.log_prob(n, 0.5).dtype == jnp.float32
    assert ops.sample(n, key=jax.random.key(0)).dtype == jnp.float32


def test_x32_explicit_float32_array_preserved():
    from probpipe.distributions.continuous import Normal

    loc = jnp.array(0.0, dtype=jnp.float32)
    scale = jnp.array(1.0, dtype=jnp.float32)
    assert Normal(loc=loc, scale=scale, name="n").dtype == jnp.float32


def test_x32_int_inputs_promote_to_float32():
    from probpipe.distributions.continuous import Normal

    # Integer inputs should not silently produce an int distribution;
    # they are promoted to JAX's default float dtype.
    n = Normal(loc=0, scale=1, name="n")
    assert n.dtype == jnp.float32


# ---------------------------------------------------------------------------
# x64 mode — distributions, log_prob, sample, and mean all stay float64
# ---------------------------------------------------------------------------


def test_x64_normal_full_pipeline():
    out = _run_x64(
        """
        from probpipe.distributions.continuous import Normal
        import probpipe.core.ops as ops
        n = Normal(loc=0.0, scale=1.0, name='n')
        assert n.dtype == jnp.float64, n.dtype
        assert ops.log_prob(n, 0.5).dtype == jnp.float64
        assert ops.sample(n, key=__import__('jax').random.key(0)).dtype == jnp.float64
        assert ops.mean(n).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_uniform_beta_gamma():
    out = _run_x64(
        """
        from probpipe.distributions.continuous import Uniform, Beta, Gamma
        import probpipe.core.ops as ops
        import jax

        for cls, kwargs in [
            (Uniform, dict(low=0.0, high=1.0)),
            (Beta, dict(alpha=2.0, beta=3.0)),
            (Gamma, dict(concentration=2.0, rate=1.0)),
        ]:
            d = cls(**kwargs, name='d')
            assert d.dtype == jnp.float64, (cls.__name__, d.dtype)
            assert ops.sample(d, key=jax.random.key(0)).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_multivariate_normal_does_not_raise():
    """The original failure mode: log_prob raised TypeError under x64."""
    out = _run_x64(
        """
        from probpipe.distributions.multivariate import MultivariateNormal
        import probpipe.core.ops as ops
        import jax

        mvn = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name='m')
        assert mvn.dtype == jnp.float64, mvn.dtype
        assert ops.log_prob(mvn, jnp.asarray([0.5, -0.5])).dtype == jnp.float64
        assert ops.sample(mvn, key=jax.random.key(0)).dtype == jnp.float64
        assert ops.mean(mvn).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_explicit_float32_input_preserved_under_x64():
    """User passing float32 explicitly under x64 should keep float32."""
    out = _run_x64(
        """
        from probpipe.distributions.continuous import Normal
        loc = jnp.array(0.0, dtype=jnp.float32)
        scale = jnp.array(1.0, dtype=jnp.float32)
        n = Normal(loc=loc, scale=scale, name='n')
        assert n.dtype == jnp.float32, n.dtype
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_promotion_mixed_dtype():
    """Mixed float32 / float64 inputs promote to the wider dtype."""
    out = _run_x64(
        """
        from probpipe.distributions.continuous import Normal
        loc32 = jnp.array(0.0, dtype=jnp.float32)
        scale64 = jnp.array(1.0, dtype=jnp.float64)
        n = Normal(loc=loc32, scale=scale64, name='n')
        assert n.dtype == jnp.float64, n.dtype
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_empirical_distribution_preserves_dtype():
    out = _run_x64(
        """
        from probpipe import EmpiricalDistribution
        samples = jnp.array([[1.0], [2.0], [3.0]])  # float64 under x64
        d = EmpiricalDistribution(samples, name="x")
        assert d.flat_samples.dtype == jnp.float64, d.flat_samples.dtype
        assert d.dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_transformed_distribution_preserves_dtype():
    out = _run_x64(
        """
        import tensorflow_probability.substrates.jax.bijectors as tfb
        from probpipe.distributions.continuous import Normal
        from probpipe.distributions.transformed import TransformedDistribution
        import probpipe.core.ops as ops
        import jax

        base = Normal(loc=0.0, scale=1.0, name='base')
        td = TransformedDistribution(base, tfb.Exp())
        assert td.dtype == jnp.float64, td.dtype
        assert ops.log_prob(td, 1.0).dtype == jnp.float64
        assert ops.sample(td, key=jax.random.key(0)).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_joint_gaussian_preserves_dtype():
    out = _run_x64(
        """
        from probpipe.distributions.joint import JointGaussian
        import probpipe.core.ops as ops
        import jax

        jg = JointGaussian(
            mean=jnp.zeros(3),
            cov=jnp.eye(3),
            x=1, y=2,
        )
        sample = ops.sample(jg, key=jax.random.key(0))
        assert sample['x'].dtype == jnp.float64, sample['x'].dtype
        assert sample['y'].dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_weights_preserves_dtype():
    out = _run_x64(
        """
        from probpipe import Weights

        w = Weights(weights=jnp.array([0.2, 0.3, 0.5]))
        assert w.dtype == jnp.float64, w.dtype
        assert w.normalized.dtype == jnp.float64

        # Uniform weights default to JAX's default float
        u = Weights.uniform(5)
        assert u.dtype == jnp.float64
        assert u.normalized.dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x32_weights_default():
    """Without x64, Weights still defaults to float32."""
    from probpipe import Weights

    w = Weights(weights=jnp.array([0.2, 0.3, 0.5]))
    assert w.dtype == jnp.float32
    u = Weights.uniform(5)
    assert u.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Discrete distributions, KDE, GRF, and joint conditioning under x64
# ---------------------------------------------------------------------------


def test_x64_discrete_distributions():
    out = _run_x64(
        """
        from probpipe.distributions.discrete import (
            Bernoulli, Binomial, Poisson, Categorical, NegativeBinomial,
        )
        import probpipe.core.ops as ops
        import jax

        cases = [
            Bernoulli(probs=0.5, name='b'),
            Binomial(total_count=10, probs=0.3, name='bn'),
            Poisson(rate=2.0, name='p'),
            Categorical(probs=jnp.array([0.2, 0.3, 0.5]), name='c'),
            NegativeBinomial(total_count=5.0, probs=0.4, name='nb'),
        ]
        for d in cases:
            # log_prob on int-valued support should not raise
            sample = ops.sample(d, key=jax.random.key(0))
            ops.log_prob(d, sample)
        # The float-valued probs/rate/logits should produce float64 internals
        b = Bernoulli(probs=0.5, name='b2')
        assert b._probs.dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_kde_distribution():
    out = _run_x64(
        """
        from probpipe.distributions.kde import KDEDistribution
        import probpipe.core.ops as ops
        import jax

        samples = jnp.linspace(-2.0, 2.0, 50)
        kde = KDEDistribution(samples, name='kde')
        assert kde.dtype == jnp.float64
        assert ops.sample(kde, key=jax.random.key(0)).dtype == jnp.float64
        assert ops.log_prob(kde, 0.5).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_gaussian_random_function():
    out = _run_x64(
        """
        from probpipe.distributions import MultivariateNormal
        from probpipe.distributions.gaussian_random_function import LinearBasisFunction
        import jax

        weights = MultivariateNormal(
            loc=jnp.zeros(3), cov=jnp.eye(3), name='w',
        )
        feature_map = lambda X: jnp.stack([jnp.ones_like(X[..., 0]),
                                            X[..., 0],
                                            X[..., 0] ** 2], axis=-1)
        grf = LinearBasisFunction(
            feature_map, weights, input_shape=(1,), output_shape=(),
        )
        X = jnp.linspace(-1.0, 1.0, 5)[:, None]
        d = grf(X)
        assert d.dtype == jnp.float64, d.dtype
        # bias defaults to zeros at the weight dtype
        assert grf._bias.dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_product_distribution_promotes():
    out = _run_x64(
        """
        from probpipe.distributions import ProductDistribution
        from probpipe.distributions.continuous import Normal
        import probpipe.core.ops as ops
        import jax

        prod = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name='a'),
            b=Normal(loc=0.0, scale=1.0, name='b'),
        )
        sample = ops.sample(prod, key=jax.random.key(0))
        assert sample['a'].dtype == jnp.float64
        assert sample['b'].dtype == jnp.float64
        assert ops.log_prob(prod, sample).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_joint_gaussian_conditioning():
    """Conditioning on observed data must respect the parent's dtype under x64."""
    out = _run_x64(
        """
        from probpipe.distributions.joint import JointGaussian
        from probpipe.core.ops import condition_on, sample
        import jax

        jg = JointGaussian(
            mean=jnp.zeros(3), cov=jnp.eye(3), x=1, y=2,
        )
        # Conditioning on a float32 observed value must not corrupt the
        # parent's float64 dtype.
        cond = condition_on(jg, x=jnp.array([1.0], dtype=jnp.float32))
        sample_y = sample(cond, key=jax.random.key(0))
        assert sample_y['y'].dtype == jnp.float64, sample_y['y'].dtype
        print('OK')
        """
    )
    assert out == "OK"


def test_x64_bootstrap_distribution():
    out = _run_x64(
        """
        from probpipe.core._numeric_record_distribution import BootstrapDistribution
        import probpipe.core.ops as ops
        import jax

        evals = jnp.linspace(0.0, 1.0, 10)
        b = BootstrapDistribution(evals, name='b')
        assert b.dtype == jnp.float64
        assert ops.mean(b).dtype == jnp.float64
        assert ops.sample(b, key=jax.random.key(0)).dtype == jnp.float64
        print('OK')
        """
    )
    assert out == "OK"
