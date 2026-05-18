"""Unified interface for Posterior/Prior/Sequential Predictive Checks.

Bridges the existing JAX-native engine in
``probpipe.validation._predictive_check`` with the diagnostics workflow.

The generative likelihood is auto-detected from the distribution when
not explicitly provided — works out of the box with
:class:`~probpipe.modeling.SimpleGenerativeModel`.

Functions
---------
run_ppc   : one or more test functions on a single posterior/prior
run_spc   : sequential predictive check across a list of posteriors
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..core.node import workflow_function
from ..custom_types import PRNGKey
from ..validation._predictive_check import predictive_check
from ._utils import _resolve_generative_likelihood

__all__ = ["run_ppc", "run_spc"]


@workflow_function
def run_ppc(
    distribution,
    test_fns: Callable | Sequence[Callable],
    observed_data=None,
    *,
    generative_likelihood=None,
    n_samples: int | None = None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> dict:
    """Run one or more posterior/prior predictive checks.

    Auto-detects the generative likelihood from ``distribution`` when
    not explicitly provided, so no extra arguments are needed for
    :class:`~probpipe.modeling.SimpleGenerativeModel`.

    Passing ``observed_data`` makes this a **posterior** predictive
    check (p-value computed).  Omitting it makes this a **prior**
    predictive check.

    Parameters
    ----------
    distribution : SupportsSampling
        Prior or posterior to sample parameters from.
    test_fns : callable or list of callable
        One or more test statistics mapping data → scalar float.
    observed_data : optional
        If provided → posterior predictive check (p-value included).
        If ``None`` → prior predictive check.
    generative_likelihood : GenerativeLikelihood or None
        Auto-detected from ``distribution`` when not provided.
    n_samples : int or None
        Observations per replicated dataset.
        Defaults to ``len(observed_data)`` when ``observed_data`` is given.
    n_replications : int
        Number of replicated datasets. Default 500.
    key : PRNGKey or None
        JAX PRNG key. Auto-generated if ``None``.

    Returns
    -------
    dict
        Keys are test function names; values are individual
        ``predictive_check`` result dicts, each containing:

        - ``"replicated_statistics"`` — ``ArrayEmpiricalDistribution``
        - ``"observed_statistic"``    — float  (if ``observed_data`` given)
        - ``"p_value"``               — float  (if ``observed_data`` given)

    Examples
    --------
    ::

        # Prior predictive check — no observed_data
        result = run_ppc(prior, test_fns=var_mean_ratio)

        # Posterior predictive check — with observed_data
        result = run_ppc(posterior, test_fns=[var_mean_ratio, zero_prop],
                         observed_data=y)
        result["var_mean_ratio"]["p_value"]
    """
    gl = _resolve_generative_likelihood(distribution, generative_likelihood)

    if callable(test_fns):
        test_fns = [test_fns]

    results = {}
    for fn in test_fns:
        name = getattr(fn, "__name__", repr(fn))
        results[name] = predictive_check(
            distribution=distribution,
            generative_likelihood=gl,
            test_fn=fn,
            observed_data=observed_data,
            n_samples=n_samples,
            n_replications=n_replications,
            key=key,
        )
    return results


@workflow_function
def run_spc(
    distributions: Sequence,
    test_fns: Callable | Sequence[Callable],
    observed_data_sequence: Sequence,
    *,
    generative_likelihood=None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> dict:
    """Sequential Predictive Check across a sequence of posteriors.

    At each time step *t*, draws from ``distributions[t]`` and compares
    against ``observed_data_sequence[t]``.  The generative likelihood is
    auto-detected from ``distributions[0]``.

    Typical use: pass the output of
    :func:`~probpipe.core.transition.iterate` as ``distributions``.

    Parameters
    ----------
    distributions : sequence of SupportsSampling
        Posteriors at each time step.
    test_fns : callable or list of callable
        One or more test statistics.
    observed_data_sequence : sequence
        Observed data at each time step.
        Must be the same length as ``distributions``.
    generative_likelihood : GenerativeLikelihood or None
        Auto-detected from ``distributions[0]`` when not provided.
    n_replications : int
        Replications per step. Default 500.
    key : PRNGKey or None
        JAX PRNG key. Auto-generated if ``None``.

    Returns
    -------
    dict
        - ``"steps"``    — list of per-step :func:`run_ppc` result dicts
        - ``"p_values"`` — ``{fn_name: np.ndarray}`` across time steps
        - ``"n_steps"``  — int

    Raises
    ------
    ValueError
        If ``distributions`` and ``observed_data_sequence`` differ in length.

    Examples
    --------
    ::

        posteriors = iterate(step, prior, data_batches)
        result = run_spc(posteriors, test_fns=var_mean_ratio,
                         observed_data_sequence=data_batches)
        result["p_values"]["var_mean_ratio"]   # shape (n_steps,)
    """
    if len(distributions) != len(observed_data_sequence):
        raise ValueError(
            f"distributions and observed_data_sequence must be the same "
            f"length: got {len(distributions)} and "
            f"{len(observed_data_sequence)}."
        )

    # Auto-resolve once from the first distribution
    gl = _resolve_generative_likelihood(
        distributions[0], generative_likelihood
    )

    if callable(test_fns):
        test_fns = list([test_fns])
    else:
        test_fns = list(test_fns)

    step_results = []
    for dist, obs in zip(distributions, observed_data_sequence):
        step_results.append(
            run_ppc(
                distribution=dist,
                test_fns=test_fns,
                observed_data=obs,
                generative_likelihood=gl,
                n_replications=n_replications,
                key=key,
            )
        )

    # Collect p-values per test_fn across all time steps
    fn_names = [getattr(fn, "__name__", repr(fn)) for fn in test_fns]
    p_values = {
        name: np.array([
            step[name]["p_value"]
            for step in step_results
            if name in step and "p_value" in step[name]
        ])
        for name in fn_names
    }

    return {
        "steps":   step_results,
        "p_values": p_values,
        "n_steps":  len(step_results),
    }