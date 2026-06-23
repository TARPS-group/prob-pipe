"""Shared helpers for the BayesFlow surrogate tests.

Kept out of the ``test_*`` modules so the two BayesFlow test files
(``test_bayesflow_likelihoods.py`` / ``test_bayesflow_posteriors.py``) share a
single implementation rather than duplicating it.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from probpipe.custom_types import Array


def theta_vec(params: Any) -> Array:
    """Coerce a per-draw ``params`` object to its 1-D parameter vector.

    Mirrors ``BayesFlowLikelihood._theta_row``: structured records serialize via
    :meth:`~probpipe.NumericRecord.to_vector`; raw array-likes are ravelled.

    Parameters
    ----------
    params : Any
        A per-draw ``NumericRecord`` / ``NumericRecordArray`` (training and
        predictive paths) or a flat array-like (the gradient-MCMC path).

    Returns
    -------
    Array
        The 1-D parameter vector.
    """
    return params.to_vector() if hasattr(params, "to_vector") else jnp.ravel(jnp.asarray(params))
