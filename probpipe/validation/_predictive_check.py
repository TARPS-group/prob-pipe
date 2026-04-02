"""Predictive checking for model validation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

import jax

from ..core.distribution import ArrayEmpiricalDistribution
from ..core.node import workflow_function
from ..core.protocols import SupportsSampling
from ..custom_types import PRNGKey
from .._utils import _auto_key
from ..modeling._likelihood import GenerativeLikelihood

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

    Parameters
    ----------
    distribution : Distribution[P]
        Prior or posterior to sample parameters from.
    generative_likelihood : GenerativeLikelihood[P, D]
        Must have ``generate_data(params: P, n_samples: int) -> D``.
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

        - ``"replicated_statistics"`` — ``ArrayEmpiricalDistribution``
          over the test statistic values from replicated data.

        When *observed_data* is provided, also contains:

        - ``"observed_statistic"`` — ``test_fn(observed_data)``
        - ``"p_value"`` — fraction of replicates where the test
          statistic is at least as extreme as the observed value.
    """
    if not isinstance(distribution, SupportsSampling):
        raise TypeError(
            f"{type(distribution).__name__} does not support sampling "
            f"(does not implement SupportsSampling)"
        )
    if not isinstance(generative_likelihood, GenerativeLikelihood):
        raise TypeError(
            f"{type(generative_likelihood).__name__} does not implement "
            f"GenerativeLikelihood (missing generate_data method)"
        )

    if n_samples is None:
        if observed_data is None:
            raise ValueError(
                "n_samples is required when observed_data is not provided"
            )
        n_samples = len(observed_data)

    if key is None:
        key = _auto_key()

    # Draw one parameter sample per replication and compute test statistics
    stats = []
    for i in range(n_replications):
        key, subkey = jax.random.split(key)
        params_i = distribution._sample(subkey, ())
        y_rep = generative_likelihood.generate_data(params_i, n_samples)
        stats.append(float(test_fn(y_rep)))

    stats_array = np.array(stats, dtype=np.float64)
    replicated_dist = ArrayEmpiricalDistribution(
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
