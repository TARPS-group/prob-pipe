"""Tests that pin the values a seeded workflow run gives (V.9).

A seeded run gives the same draws under every executor, so the values below
hold for each Python executor. A change to how keys are derived inside
evaluation points changes them, which every other test of the workflow's
randomness compares only against another run.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Function, Normal, NumericArrayBatch, sample, workflow_run


def _shifted(x):
    return np.asarray(x) + 1.0


def _with_noise(x):
    return np.asarray(x) + np.asarray(sample(Normal("noise", np.float32(0), np.float32(1))))


@pytest.mark.parametrize("executor", ["sequential", "thread", "process"])
def test_a_seeded_run_gives_its_pinned_values(executor):
    lift = Function(_shifted, label="shifted", traceable=False, max_evaluations=4)
    sweep = Function(_with_noise, label="with_noise", traceable=False)
    nested = Function(_with_noise, label="with_noise", traceable=False, max_evaluations=3)
    grid = NumericArrayBatch(jnp.arange(3, dtype=jnp.float32), "x", label="grid")
    with workflow_run(seed=0):
        lifted = lift.using(executor=executor)(x=Normal("x", np.float32(0), np.float32(1)))
        swept = sweep.using(executor=executor)(x=grid)
        nested_lift = nested.using(executor=executor)(x=Normal("x", np.float32(0), np.float32(1)))
        after_the_calls = sample(Normal("z", np.float32(0), np.float32(1)))
    # Each value is a float32 that JAX's normal sampler computes, which can
    # differ in its last bit across CPU architectures.
    np.testing.assert_allclose(
        [float(atom.raw) for atom in lifted.atoms],
        [2.4226012, 0.25604874, 1.0734022, -0.7593237],
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(swept.raw), [-0.3635109, 2.4527502, 3.4476619], rtol=1e-6)
    np.testing.assert_allclose(
        [float(atom.raw) for atom in nested_lift.atoms],
        [0.026247676, 0.89417195, -2.0279312],
        rtol=1e-6,
    )
    np.testing.assert_allclose(float(after_the_calls), -0.48050687, rtol=1e-6)
