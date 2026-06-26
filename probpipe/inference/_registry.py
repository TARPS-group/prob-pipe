"""Inference method registry for condition_on dispatch.

Provides a pluggable system where inference methods (MCMC, VI, etc.)
register themselves with requirements, priority, and a unique name.
The registry auto-selects the best method, or the user overrides via
``method="tfp_nuts"``.
"""

from __future__ import annotations

from ..core._registry import (  # noqa: F401 (re-export)
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)

__all__ = ["InferenceMethod", "inference_method_registry"]

# Re-export UnaryDispatchMethod as InferenceMethod for clarity in type
# annotations on inference-method classes.
InferenceMethod = UnaryDispatchMethod

# The singleton registry — a plain UnaryDispatchRegistry, no subclass
# needed.
inference_method_registry: UnaryDispatchRegistry[UnaryDispatchMethod] = UnaryDispatchRegistry()
