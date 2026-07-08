"""Kullback-Leibler divergence via a binary dispatch registry.

``KL(p ‖ q) = E_p[log p(x) − log q(x)]`` has several implementations with
different tradeoffs, and the right one depends on the *joint* type of
``(p, q)``.  This module registers them on a
:class:`~probpipe.core._registry.BinaryDispatchRegistry` (the ``"kl"``
registry) so :func:`~probpipe.kl_divergence` auto-selects the best
feasible method by priority:

======  ==================  ====================================================
Prio    Method              Applies when
======  ==================  ====================================================
90      ``kl_normal_normal``  Both arguments are :class:`~probpipe.Normal`
                              (exact, closed-form).
70      ``kl_tfp``            Both wrap a TFP distribution for which TFP has a
                              registered closed-form KL (exact).
5       ``kl_mc``             ``p`` supports sampling + ``log_prob`` and ``q``
                              supports ``log_prob`` (inexact Monte Carlo
                              fallback).
======  ==================  ====================================================

The Monte Carlo fallback sits at a *low positive* priority rather than
``OPT_IN_ONLY_PRIORITY`` so that ``kl_divergence(p, q)`` returns a
numerical answer whenever one is computable; ``method="kl_mc"`` forces it.

The public op lives in :mod:`probpipe.core.ops`.  See
``docs/contributor/dispatch-conventions.md`` for the conventions this
module follows.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .._utils import _auto_key
from ..core._registry import BinaryDispatchMethod, BinaryDispatchRegistry, MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb, SupportsSampling
from ..custom_types import Array
from ..distributions._tfp_base import TFPDistribution
from ..distributions.continuous import Normal

__all__ = ["kl_registry"]


kl_registry: BinaryDispatchRegistry[BinaryDispatchMethod] = BinaryDispatchRegistry(
    name="kl",
    description="Kullback-Leibler divergence KL(p || q).",
)


class GaussianKL(BinaryDispatchMethod):
    """Closed-form ``KL(p ‖ q)`` for two univariate :class:`~probpipe.Normal`.

    Uses the analytic Gaussian formula, preferred over the TFP path
    (:class:`TFPKL`) so ProbPipe controls the computation for its own
    Normal type.
    """

    description = "Closed-form KL for two Normal distributions."

    @property
    def name(self) -> str:
        return "kl_normal_normal"

    @property
    def priority(self) -> int:
        return 90  # exact tier (> 50); analytic

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Normal,), (Normal,))

    def check(self, p: Any, q: Any, **_: Any) -> MethodInfo:
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, p: Any, q: Any, **_: Any) -> Array:
        var_ratio = jnp.square(p.scale / q.scale)
        t1 = jnp.square((p.loc - q.loc) / q.scale)
        return 0.5 * (var_ratio + t1 - 1.0 - jnp.log(var_ratio))


class TFPKL(BinaryDispatchMethod):
    """Closed-form ``KL(p ‖ q)`` via TFP's own KL registry.

    Feasible only when both arguments wrap a TFP distribution *and* TFP
    has a registered closed-form for the pair; :meth:`check` probes TFP's
    registry without running the computation.
    """

    description = "Closed-form KL via TFP for a registered distribution pair."

    @property
    def name(self) -> str:
        return "kl_tfp"

    @property
    def priority(self) -> int:
        return 70  # exact tier; below the ProbPipe analytic Normal path

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((TFPDistribution,), (TFPDistribution,))

    def check(self, p: Any, q: Any, **_: Any) -> MethodInfo:
        from tensorflow_probability.substrates.jax.distributions import kullback_leibler

        registered = (
            kullback_leibler._registered_kl(type(p._tfp_dist), type(q._tfp_dist)) is not None
        )
        if not registered:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description=(
                    f"TFP has no registered closed-form KL for "
                    f"({type(p._tfp_dist).__name__}, {type(q._tfp_dist).__name__})"
                ),
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, p: Any, q: Any, **_: Any) -> Array:
        import tensorflow_probability.substrates.jax.distributions as tfd

        return tfd.kl_divergence(p._tfp_dist, q._tfp_dist)


class MCKL(BinaryDispatchMethod):
    """Monte Carlo ``KL(p ‖ q) = E_p[log p(x) − log q(x)]``.

    Universal fallback: draws ``n_samples`` from *p* and averages the
    log-density difference.  Requires *p* to support sampling and
    ``log_prob`` and *q* to support ``log_prob``.
    """

    description = "Monte Carlo KL estimate (universal fallback)."

    @property
    def name(self) -> str:
        return "kl_mc"

    @property
    def priority(self) -> int:
        return 5  # inexact tier; last-resort auto-dispatch fallback

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Distribution,), (Distribution,))

    def check(self, p: Any, q: Any, **_: Any) -> MethodInfo:
        if not (isinstance(p, SupportsSampling) and isinstance(p, SupportsLogProb)):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="p must support sampling and log_prob",
            )
        if not isinstance(q, SupportsLogProb):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="q must support log_prob",
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(
        self,
        p: Any,
        q: Any,
        *,
        random_seed: int | None = None,
        n_samples: int = 10_000,
        **_: Any,
    ) -> Array:
        key = jax.random.PRNGKey(random_seed) if random_seed is not None else _auto_key()
        xs = p._sample(key, sample_shape=(n_samples,))
        return jnp.mean(p._log_prob(xs) - q._log_prob(xs))


kl_registry.register(GaussianKL())
kl_registry.register(TFPKL())
kl_registry.register(MCKL())
