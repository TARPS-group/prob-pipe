"""Inference method registry for condition_on dispatch.

Provides a pluggable system where inference methods (MCMC, VI, etc.)
register themselves with requirements, priority, and a unique name.
The registry auto-selects the best method, or the user overrides via
``method="tfp_nuts"``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..core._registry import Method, MethodInfo, MethodRegistry


class InferenceMethod(Method):
    """Abstract base for an inference method.

    Subclasses implement ``check`` to probe feasibility (cheap) and
    ``condition`` to run the actual inference (expensive).
    """

    @abstractmethod
    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        """Probe whether this method can condition *dist* on *observed*.

        Must be cheap — no sampling or heavy computation.
        """
        ...

    @abstractmethod
    def condition(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        """Run inference: condition *dist* on *observed*, return posterior."""
        ...


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
    >>> posterior = inference_method_registry.condition(model, data)
    >>> posterior = inference_method_registry.condition(model, data, method="tfp_rwmh")
    """

    def check(
        self, dist: Any, observed: Any, *, method: str | None = None, **kwargs: Any
    ) -> MethodInfo:
        """Check feasibility for conditioning *dist* on *observed*.

        Parameters
        ----------
        dist : Any
            Distribution or model to condition.
        observed : Any
            Observed data.
        method : str or None
            If provided, check only the named method.
        """
        if method is not None:
            m = self.get_method(method)
            return m.check(dist, observed, **kwargs)

        for m in self._find_methods(type(dist)):
            info = m.check(dist, observed, **kwargs)
            if info.feasible:
                return info

        return MethodInfo(
            feasible=False,
            description=f"No inference method for {type(dist).__name__}",
        )

    def condition(
        self, dist: Any, observed: Any, *, method: str | None = None, **kwargs: Any
    ) -> Any:
        """Condition *dist* on *observed* using the best (or named) method.

        Parameters
        ----------
        dist : Any
            Distribution or model to condition.
        observed : Any
            Observed data.
        method : str or None
            If provided, use only the named method.
        **kwargs
            Passed to the method's ``condition()``.

        Returns
        -------
        Distribution
            The posterior distribution.

        Raises
        ------
        TypeError
            If no registered method can handle this combination.
        KeyError
            If *method* is specified but not registered.
        """
        if method is not None:
            m = self.get_method(method)
            info = m.check(dist, observed, **kwargs)
            if not info.feasible:
                raise TypeError(
                    f"Method {method!r} cannot condition "
                    f"{type(dist).__name__}: {info.description}"
                )
            return m.condition(dist, observed, **kwargs)

        for m in self._find_methods(type(dist)):
            info = m.check(dist, observed, **kwargs)
            if info.feasible:
                return m.condition(dist, observed, **kwargs)

        raise TypeError(
            f"No inference method registered for {type(dist).__name__}. "
            f"Available methods: {self.list_methods()}"
        )


# Module-level singleton
inference_method_registry = InferenceMethodRegistry()
