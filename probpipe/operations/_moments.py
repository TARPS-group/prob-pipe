"""The expectation operation and the registry of methods that realize it.

``expectation(d, f)`` returns ``E[f(X)]`` for ``X ~ d``. The methods that can
compute it form a dispatch registry keyed on the distribution's type, and the
registry's selection order decides which runs:

1. ``exact``: the distribution's own ``_expectation``, for a law that claims
   :class:`~probpipe.distributions._capabilities.SupportsExpectation`, which in
   practice means finite support.
2. ``monte_carlo``: the average of ``f`` over ``num_evaluations`` independent
   draws, for any law that samples. It is the default approximate method.

Exact methods rank before approximate ones, so an exact route is taken whenever
one applies. Another method, such as quasi-Monte Carlo or quadrature, joins by
registering with ``expectation_method_registry.register``. A caller selects it
with ``method=``, and ``set_priorities`` makes it the default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from ..core._dispatch import Feasibility, UnaryDispatchMethod, UnaryDispatchRegistry
from ..custom_types import PRNGKey
from ..distributions import _distribution as _base
from ..distributions._capabilities import SupportsExpectation, SupportsSampling, _capability_guard
from ..distributions._distribution import Distribution
from ..functions import _broker, _descendants, function

__all__ = ["ExpectationMethod", "expectation", "expectation_method_registry"]


class ExpectationMethod(UnaryDispatchMethod):
    """A method that computes ``E[f(X)]`` for ``X ~ d``.

    A subclass declares ``name``, ``exact``, a ``priority``, and the
    distribution types it admits, and implements ``check`` and ``execute``
    over ``(d, f, **options)``. The options are ``num_evaluations`` and ``key``,
    which a method that does not sample ignores.
    """

    def supported_types(self) -> tuple[type, ...]:
        """Every distribution; ``check`` decides which it applies to."""
        return (Distribution,)


class _ExactExpectation(ExpectationMethod):
    """The distribution's own exact ``_expectation``."""

    @property
    def name(self) -> str:
        return "exact"

    @property
    def exact(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return 100

    def check(self, d: Any, f: Callable[[Any], Any], /, **options: Any) -> Feasibility:
        """Feasible when *d* claims ``SupportsExpectation`` and its guard admits the call."""
        if not isinstance(d, SupportsExpectation):
            return Feasibility(False, f"{type(d).__name__} has no exact expectation")
        return _capability_guard(d, "_expectation")

    def execute(self, d: Any, f: Callable[[Any], Any], /, **options: Any) -> Any:
        """``d._expectation(f)``."""
        return d._expectation(f)


class _MonteCarloExpectation(ExpectationMethod):
    """The average of ``f`` over independent draws of the distribution.

    ``num_evaluations`` sets the number of draws, defaulting to
    ``probpipe.distributions._distribution.DEFAULT_NUM_EVALUATIONS``. Without a
    ``key``, the draws are workflow-owned random events, so a surrounding
    workflow's seed and replay govern them.
    """

    @property
    def name(self) -> str:
        return "monte_carlo"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return 50

    def check(self, d: Any, f: Callable[[Any], Any], /, **options: Any) -> Feasibility:
        """Feasible when *d* samples."""
        if not isinstance(d, SupportsSampling):
            return Feasibility(False, f"{type(d).__name__} does not sample")
        return _capability_guard(d, "_sample")

    def execute(
        self,
        d: Any,
        f: Callable[[Any], Any],
        /,
        *,
        num_evaluations: int | None = None,
        key: PRNGKey | None = None,
        **options: Any,
    ) -> Any:
        """The mean of ``f`` over ``num_evaluations`` draws of *d*.

        Raises
        ------
        TypeError
            If ``num_evaluations`` is not an integer.
        ValueError
            If ``num_evaluations`` is not positive.
        """
        n = _base.DEFAULT_NUM_EVALUATIONS if num_evaluations is None else num_evaluations
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"num_evaluations must be an integer; got {n!r}")
        if n <= 0:
            raise ValueError(f"num_evaluations must be positive; got {n!r}")
        if key is None:
            captured = _descendants.capture_stochastic_consumer(d)
            key = _broker._resolve_automatic_key(
                None,
                _broker._singleton_effect_plan(
                    operation_kind="expectation",
                    execution_mode="monte_carlo",
                    sample_shape=(n,),
                    record_path=captured.record_path,
                    descendant_descriptor=captured.descendant_descriptor,
                ),
            )
            draws = _descendants.sample_captured_consumer(captured, key, (n,))
        else:
            draws = d._sample(key, sample_shape=(n,))
        values = jax.vmap(f)(draws)
        return jax.tree.map(lambda v: jnp.mean(v, axis=0), values)


expectation_method_registry: UnaryDispatchRegistry[ExpectationMethod] = UnaryDispatchRegistry()
"""The methods that compute an expectation, in selection order."""

expectation_method_registry.register(_ExactExpectation())
expectation_method_registry.register(_MonteCarloExpectation())


@function
def expectation(
    dist: Distribution,
    f: Any,
    *,
    method: str | None = None,
    exact_only: bool = False,
    num_evaluations: int | None = None,
    key: PRNGKey | None = None,
) -> Any:
    """Compute ``E[f(X)]`` for ``X ~ dist``.

    The exact method runs when *dist* has one, and the Monte Carlo method
    otherwise; see :data:`expectation_method_registry` for the methods and their
    order.

    Parameters
    ----------
    dist : Distribution
        The law to integrate against.
    f : callable
        Maps one draw to an array or a pytree of arrays.
    method : str, optional
        The name of a registered method to run instead of selecting one.
    exact_only : bool
        If ``True``, only exact methods are considered.
    num_evaluations : int, optional
        The number of draws a sampling method takes.
    key : PRNGKey, optional
        The key a sampling method draws with; workflow-owned when omitted.

    Returns
    -------
    Array or pytree of arrays
        ``E[f(X)]``, shaped as the output of *f*.

    Raises
    ------
    ResolutionError
        If no method applies under the controls, or *method* names one that is
        not registered or does not apply.
    """
    return expectation_method_registry.execute(
        dist,
        f,
        method=method,
        exact_only=exact_only,
        num_evaluations=num_evaluations,
        key=key,
    )
