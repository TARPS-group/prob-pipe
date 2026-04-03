"""Inference method registry for condition_on dispatch.

Provides a pluggable system where inference methods (MCMC, VI, etc.)
register themselves with requirements, priority, and a unique name.
The registry auto-selects the best method, or the user overrides via
``method="tfp_nuts"``.
"""

from __future__ import annotations

from ..core._registry import Method, MethodInfo, MethodRegistry


class InferenceMethod(Method):
    """Abstract base for an inference method.

    Subclasses implement ``check`` to probe feasibility (cheap) and
    ``execute`` to run inference (expensive).
    """
    pass


class InferenceMethodRegistry(MethodRegistry[InferenceMethod]):
    """Registry of inference methods for ``condition_on``.

    Methods are tried in descending priority order.  The first method
    whose ``check()`` returns ``feasible=True`` wins.  Pass
    ``method="name"`` to select a specific method.

    Examples
    --------
    >>> from probpipe.inference import inference_method_registry
    >>> inference_method_registry.list_methods()
    ['tfp_nuts', 'tfp_hmc', 'nutpie_nuts', ...]
    >>> posterior = inference_method_registry.execute(model, data)
    >>> posterior = inference_method_registry.execute(model, data, method="tfp_rwmh")
    """
    pass


# Module-level singleton
inference_method_registry = InferenceMethodRegistry()
