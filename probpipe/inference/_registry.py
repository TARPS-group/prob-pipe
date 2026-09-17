"""Inference method registry for condition_on dispatch.

Inference methods (MCMC, VI, ABC, ...) register themselves with a unique
name, the types they apply to, and a rank. The registry auto-selects the
first feasible method in selection order, or the user names one through
``method="blackjax_nuts"``.
"""

from __future__ import annotations

from ..core._dispatch import (  # noqa: F401 (re-export)
    Feasibility,
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)

__all__ = ["InferenceMethod", "inference_method_registry"]


class InferenceMethod(UnaryDispatchMethod):
    """Base for the registered inference methods.

    Every inference method is approximate: a finite MCMC, SG-MCMC, slice,
    ABC, or variational output stands in for the conditional law, whatever
    its invariant target or asymptotic guarantee. Those guarantees are the
    method's own documentation, not its exactness. A method that returns a
    representation of the conditional law itself overrides ``exact``.
    """

    @property
    def exact(self) -> bool:
        return False


# The singleton registry — a plain UnaryDispatchRegistry, no subclass
# needed.
inference_method_registry: UnaryDispatchRegistry[UnaryDispatchMethod] = UnaryDispatchRegistry()
