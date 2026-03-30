"""Built-in closed-form pushforward rules."""

from __future__ import annotations

import tensorflow_probability.substrates.jax.bijectors as tfb

from .bijector import TFPBijector
from .registry import (
    PushforwardRule,
    PushforwardInfo,
    PushforwardMethod,
    pushforward_registry,
)


class ExpNormalRule(PushforwardRule):
    """Closed-form: Exp pushforward of Normal → LogNormal.

    Applies when the transport map is a ``TFPBijector`` wrapping
    ``tfb.Exp()`` and the distribution is ``Normal``.
    """

    def map_types(self):
        return (TFPBijector,)

    def dist_types(self):
        from ..distributions.continuous import Normal

        return (Normal,)

    def check(self, transport_map, dist):
        if isinstance(transport_map._tfp_bijector, tfb.Exp):
            return PushforwardInfo(
                feasible=True,
                method=PushforwardMethod.CLOSED_FORM,
                description="Exp(Normal) → LogNormal",
            )
        return PushforwardInfo(feasible=False)

    def apply(self, transport_map, dist, *, return_joint=False, **kwargs):
        from ..distributions.continuous import LogNormal
        from ..core.distribution import (
            BroadcastDistribution,
            Provenance,
            _auto_key,
        )
        from ..core.node import WorkflowFunction

        import jax

        result = LogNormal(loc=dist.loc, scale=dist.scale)
        result.with_source(
            Provenance(
                "pushforward",
                parents=(dist,),
                metadata={
                    "map": repr(transport_map),
                    "method": "closed_form",
                    "rule": "Exp(Normal) → LogNormal",
                },
            )
        )

        if not return_joint:
            return result

        # return_joint=True: sample paired data, store exact marginal
        key = kwargs.get("key")
        num_samples = kwargs.get("num_samples")
        if key is None:
            key = _auto_key()
        n = num_samples if num_samples is not None else WorkflowFunction.DEFAULT_N_BROADCAST_SAMPLES

        input_samples = dist._sample(key, sample_shape=(n,))
        output_samples = jax.vmap(transport_map.forward)(input_samples)

        bd = BroadcastDistribution(
            input_samples={"input": input_samples},
            output_samples=output_samples,
            weights=None,
            broadcast_args=["input"],
        )
        bd._exact_output_marginal = result
        bd.with_source(
            Provenance(
                "pushforward",
                parents=(dist,),
                metadata={
                    "map": repr(transport_map),
                    "method": "closed_form",
                    "rule": "Exp(Normal) → LogNormal",
                    "return_joint": True,
                },
            )
        )
        return bd

    @property
    def priority(self):
        return 10  # above CoV (0) and sampling (-100)


# Register on import
pushforward_registry.register(ExpNormalRule())
