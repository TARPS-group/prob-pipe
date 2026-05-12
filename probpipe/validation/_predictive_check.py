"""Predictive checking for model validation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp

from ..core.distribution import RecordEmpiricalDistribution
from ..core.node import workflow_function
from ..core.protocols import SupportsSampling
from ..custom_types import PRNGKey
from .._utils import _auto_key
from ..modeling._likelihood import GenerativeLikelihood  # needed for type hint resolution

__all__ = ["predictive_check"]


@workflow_function
def predictive_check[P, D](
    distribution: SupportsSampling,
    generative_likelihood: GenerativeLikelihood[P, D],
    test_fn: Callable[[D], float],
    observed_data: D | None = None,
    *,
    n_samples: int | None = None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> dict:
    """Predictive check — works as both prior and posterior check.

    Draws parameter samples from *distribution*, generates replicated
    data via *generative_likelihood*, and computes *test_fn* on each
    replicate.

    When *observed_data* is provided, also computes *test_fn* on the
    observed data and returns a calibration p-value, making this a
    posterior predictive check.  Without *observed_data*, this is a
    prior predictive check — useful for understanding the implications
    of the prior.

    When ``generate_data`` accepts a ``key`` keyword argument, all
    replications are generated in a single vectorized call (by passing
    a batch of parameter vectors), giving a large speedup.  The test
    function is then applied via ``jax.vmap`` when possible, with an
    automatic fallback to a Python loop.

    Parameters
    ----------
    distribution : Distribution[P]
        Prior or posterior to sample parameters from.
    generative_likelihood : GenerativeLikelihood[P, D]
        Must have ``generate_data(params: P, n_samples: int, *, key: PRNGKey | None = None) -> D``.
        If ``generate_data`` also accepts a ``key`` keyword, the
        vectorized fast path is used.
    test_fn : Callable[[D], float]
        Test statistic mapping data to a scalar.
    observed_data : D or None, optional
        If provided, compute the observed test statistic and p-value.
    n_samples : int, optional
        Number of observations per replicated dataset.  Required if
        *observed_data* is not provided; otherwise defaults to
        ``len(observed_data)``.
    n_replications : int
        Number of replicated datasets to generate.
    key : PRNGKey, optional
        JAX PRNG key.  Auto-generated if ``None``.

    Returns
    -------
    dict
        Always contains:

        - ``"replicated_statistics"`` — ``RecordEmpiricalDistribution``
          over the test statistic values from replicated data.

        When *observed_data* is provided, also contains:

        - ``"observed_statistic"`` — ``test_fn(observed_data)``
        - ``"p_value"`` — fraction of replicates where the test
          statistic is at least as extreme as the observed value.
    """
    if n_samples is None:
        if observed_data is None:
            raise ValueError(
                "n_samples is required when observed_data is not provided"
            )
        n_samples = len(observed_data)

    if key is None:
        key = _auto_key()

    # -- Fast path: batched generation + vmap test_fn -----------------------
    if _supports_key_arg(generative_likelihood):
        stats_array = _predictive_check_batched(
            distribution, generative_likelihood, test_fn,
            n_samples, n_replications, key,
        )
    else:
        stats_array = _predictive_check_loop(
            distribution, generative_likelihood, test_fn,
            n_samples, n_replications, key,
        )

    replicated_dist = RecordEmpiricalDistribution(
        stats_array, name="replicated_statistics",
    )

    test_fn_name = getattr(test_fn, "__name__", repr(test_fn))
    result = {
        "replicated_statistics": replicated_dist,
        "test_fn_name": test_fn_name,
    }

    if observed_data is not None:
        obs_stat = float(test_fn(observed_data))
        p_value = float(np.mean(stats_array >= obs_stat))
        result["observed_statistic"] = obs_stat
        result["p_value"] = p_value

    # Attach to the distribution for easy access
    if hasattr(distribution, "validation_results"):
        distribution.validation_results.append(result)

    return result


def _supports_key_arg(generative_likelihood: Any) -> bool:
    """Check whether generate_data accepts a ``key`` keyword argument."""
    import inspect

    try:
        sig = inspect.signature(generative_likelihood.generate_data)
        return "key" in sig.parameters
    except (ValueError, TypeError):
        return False


def _predictive_check_batched(
    distribution: SupportsSampling,
    generative_likelihood: Any,
    test_fn: Callable,
    n_samples: int,
    n_replications: int,
    key: PRNGKey,
) -> np.ndarray:
    """Vectorized predictive check using batched data generation."""
    key_params, key_data = jax.random.split(key)

    # Draw all parameter samples at once: (n_replications, *event_shape)
    params_batch = distribution._sample(key_params, (n_replications,))

    # Generate all replicated datasets in one call
    y_rep_batch = generative_likelihood.generate_data(
        params_batch, n_samples, key=key_data,
    )

    # Apply test_fn to each replicate — try vmap, fall back to loop
    try:
        stats = jax.vmap(test_fn)(y_rep_batch)
        return np.asarray(stats, dtype=np.float64)
    except Exception:
        # test_fn may not be JAX-traceable (e.g., uses Python control flow)
        return np.array(
            [float(test_fn(y_rep_batch[i])) for i in range(n_replications)],
            dtype=np.float64,
        )


def _predictive_check_loop(
    distribution: SupportsSampling,
    generative_likelihood: Any,
    test_fn: Callable,
    n_samples: int,
    n_replications: int,
    key: PRNGKey,
) -> np.ndarray:
    """Fallback: sequential predictive check in a Python loop."""
    stats = []
    for i in range(n_replications):
        key, subkey = jax.random.split(key)
        params_i = distribution._sample(subkey, ())
        y_rep = generative_likelihood.generate_data(params_i, n_samples)
        stats.append(float(test_fn(y_rep)))
    return np.array(stats, dtype=np.float64)
