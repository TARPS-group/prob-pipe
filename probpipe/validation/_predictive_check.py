"""Predictive checking for model validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from .._utils import _auto_key
from ..core.distribution import RecordEmpiricalDistribution
from ..core.node import workflow_function
from ..core.protocols import SupportsSampling
from ..custom_types import PRNGKey
from ..modeling._likelihood import GenerativeLikelihood  # needed for type hint resolution

__all__ = ["predictive_check"]


@workflow_function
def predictive_check[P, D](
    distribution: SupportsSampling,
    generative_likelihood: GenerativeLikelihood[P, D],
    test_fn: Callable[[D], float],
    observed_data: D | None = None,
    *,
    num_observations: int | None = None,
    num_replications: int = 500,
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
        Must have ``generate_data(params: P, num_observations: int, *,
        key: PRNGKey | None = None) -> D``. If ``generate_data`` also
        accepts a ``key`` keyword, the vectorized fast path is used.
    test_fn : Callable[[D], float]
        Test statistic mapping data to a scalar.
    observed_data : D or None, optional
        If provided, compute the observed test statistic and p-value.
    num_observations : int, optional
        Number of observations per replicated dataset.  Required if
        *observed_data* is not provided; otherwise defaults to
        ``len(observed_data)``.
    num_replications : int
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
    if num_observations is None:
        if observed_data is None:
            raise ValueError("num_observations is required when observed_data is not provided")
        num_observations = len(observed_data)

    if key is None:
        key = _auto_key()

    # -- Fast path: batched generation + vmap test_fn -----------------------
    if _supports_key_arg(generative_likelihood):
        stats_array = _predictive_check_batched(
            distribution,
            generative_likelihood,
            test_fn,
            num_observations,
            num_replications,
            key,
        )
    else:
        stats_array = _predictive_check_loop(
            distribution,
            generative_likelihood,
            test_fn,
            num_observations,
            num_replications,
            key,
        )

    replicated_dist = RecordEmpiricalDistribution(
        stats_array,
        name="replicated_statistics",
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

    # Attach to the distribution's ``auxiliary`` DataTree under a
    # ``predictive_check`` group; each invocation appends a numbered
    # child Dataset (``check_0``, ``check_1``, …). This keeps the
    # validation history alongside the distribution without crowding
    # the public API surface with a separate ``validation_results``
    # property — and future validation functions (LOO, WAIC, …)
    # land under their own named groups in the same DataTree.
    _record_check_in_auxiliary(distribution, stats_array, result)

    return result


def _record_check_in_auxiliary(
    distribution: Any,
    stats_array: Any,
    result: dict[str, Any],
) -> None:
    """Append a per-invocation result Dataset under
    ``distribution.auxiliary["predictive_check/check_N"]``.

    Mutates ``distribution._auxiliary`` in place. This is the
    documented exception to ``Distribution`` immutability (see
    :attr:`Distribution.auxiliary` and ``CONTRIBUTING.md`` §"Design
    principles" §1) — diagnostic ops attach results under named
    groups rather than returning renamed clones, which would break
    source/identity tracking.

    Encoding:

    - ``replicated_statistics`` becomes a ``DataArray`` of dims
      ``("replication",)``.
    - ``test_fn_name`` + optional ``observed_statistic`` /
      ``p_value`` become Dataset attrs.

    Frozen/slotted distributions (where ``_auxiliary`` can't be set
    via ``object.__setattr__``) skip the attachment silently — the
    caller still gets the ``result`` dict via the public return.
    """
    try:
        import xarray as xr
        from xarray import DataTree
    except ImportError:
        # xarray isn't available — skip the attachment silently. The
        # caller still gets the ``result`` dict via the return value.
        return

    attrs = {"test_fn_name": result["test_fn_name"]}
    if "observed_statistic" in result:
        attrs["observed_statistic"] = result["observed_statistic"]
        attrs["p_value"] = result["p_value"]
    ds = xr.Dataset(
        {"replicated_statistics": (("replication",), np.asarray(stats_array))},
        attrs=attrs,
    )

    aux = getattr(distribution, "_auxiliary", None)
    if aux is None:
        aux = DataTree()
        try:
            object.__setattr__(distribution, "_auxiliary", aux)
        except (AttributeError, TypeError):
            # Frozen/immutable distribution — give up silently.
            return
    group = aux.get("predictive_check")
    if group is None:
        aux["predictive_check"] = DataTree()
        group = aux["predictive_check"]
    n_existing = len(list(group.children))
    aux[f"predictive_check/check_{n_existing}"] = DataTree(dataset=ds)


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
    num_observations: int,
    num_replications: int,
    key: PRNGKey,
) -> np.ndarray:
    """Vectorized predictive check using batched data generation."""
    key_params, key_data = jax.random.split(key)

    # Draw all parameter samples at once: (num_replications, *event_shape)
    params_batch = distribution._sample(key_params, (num_replications,))

    # Generate all replicated datasets in one call
    y_rep_batch = generative_likelihood.generate_data(
        params_batch,
        num_observations,
        key=key_data,
    )

    # Apply test_fn to each replicate — try vmap, fall back to loop
    try:
        stats = jax.vmap(test_fn)(y_rep_batch)
        return np.asarray(stats, dtype=np.float64)
    except Exception:
        # test_fn may not be JAX-traceable (e.g., uses Python control flow)
        return np.array(
            [float(test_fn(y_rep_batch[i])) for i in range(num_replications)],
            dtype=np.float64,
        )


def _predictive_check_loop(
    distribution: SupportsSampling,
    generative_likelihood: Any,
    test_fn: Callable,
    num_observations: int,
    num_replications: int,
    key: PRNGKey,
) -> np.ndarray:
    """Fallback: sequential predictive check in a Python loop."""
    stats = []
    for i in range(num_replications):
        key, subkey = jax.random.split(key)
        params_i = distribution._sample(subkey, ())
        y_rep = generative_likelihood.generate_data(params_i, num_observations)
        stats.append(float(test_fn(y_rep)))
    return np.array(stats, dtype=np.float64)
