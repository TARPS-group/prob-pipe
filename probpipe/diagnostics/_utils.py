"""General utilities for probpipe.diagnostics.

Private module — do not import directly.
All public symbols are re-exported via probpipe.diagnostics.__init__.
"""
from __future__ import annotations

from typing import Any

__all__ = ["_resolve_generative_likelihood"]


def _resolve_generative_likelihood(
    distribution: Any,
    generative_likelihood: Any = None,
) -> Any:
    """Auto-detect generative likelihood from a posterior distribution.

    Resolution order:

    1. Explicitly passed ``generative_likelihood`` argument.
    2. ``distribution["data"]`` — works for
       :class:`~probpipe.modeling.SimpleGenerativeModel` where
       ``__getitem__("data")`` returns a
       :class:`~probpipe.modeling.GenerativeLikelihood`.
    3. ``distribution._likelihood`` — direct attribute fallback.
    4. ``distribution.generative_likelihood`` — future-proofing.
    5. Raise a descriptive :class:`ValueError`.

    Parameters
    ----------
    distribution : Distribution
        Posterior or prior — typically a ``SimpleGenerativeModel``
        or a conditioned posterior.
    generative_likelihood : optional
        Explicitly supplied likelihood; returned as-is if not ``None``.

    Returns
    -------
    GenerativeLikelihood
        Object with a ``generate_data(params, n_samples, *, key)`` method.

    Raises
    ------
    ValueError
        If no generative likelihood can be found.
    """
    # 1. Explicit argument — highest priority
    if generative_likelihood is not None:
        return generative_likelihood

    # 2. distribution["data"] — SimpleGenerativeModel path
    try:
        candidate = distribution["data"]
        if hasattr(candidate, "generate_data"):
            return candidate
    except (KeyError, TypeError):
        pass

    # 3. distribution._likelihood — direct attribute
    candidate = getattr(distribution, "_likelihood", None)
    if candidate is not None and hasattr(candidate, "generate_data"):
        return candidate

    # 4. distribution.generative_likelihood — future-proofing
    candidate = getattr(distribution, "generative_likelihood", None)
    if candidate is not None and hasattr(candidate, "generate_data"):
        return candidate

    # 5. Nothing found
    raise ValueError(
        "Could not auto-detect a generative likelihood from the distribution.\n"
        "Either:\n"
        "  (a) pass `generative_likelihood` explicitly, or\n"
        "  (b) use a SimpleGenerativeModel whose 'data' component "
        "has a `generate_data()` method."
    )