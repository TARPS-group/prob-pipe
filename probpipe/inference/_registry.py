"""Inference method registry for condition_on dispatch.

Provides a pluggable system where inference methods (MCMC, VI, etc.)
register themselves with requirements, priority, and a unique name.
The registry auto-selects the best method, or the user overrides via
``method="tfp_nuts"``.
"""

from __future__ import annotations

from ..core._registry import Method, MethodInfo, MethodRegistry  # noqa: F401 (re-export)

__all__ = ["InferenceMethod", "inference_method_registry"]

# Re-export Method as InferenceMethod for clarity in type annotations.
InferenceMethod = Method

# The singleton registry — plain MethodRegistry, no subclass needed.
inference_method_registry: MethodRegistry[Method] = MethodRegistry()
