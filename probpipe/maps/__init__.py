"""Transport maps and pushforward dispatch for prob-pipe."""

from .transport_map import TransportMap
from .bijector import Bijector, TFPBijector
from .registry import (
    PushforwardRule,
    PushforwardRegistry,
    PushforwardMethod,
    PushforwardInfo,
    pushforward_registry,
)
from .transformed_distribution import BijectorTransformedDistribution

# Import rules to trigger registration of built-in closed-form rules
from . import rules as _rules  # noqa: F401

__all__ = [
    "TransportMap",
    "Bijector",
    "TFPBijector",
    "BijectorTransformedDistribution",
    "PushforwardRule",
    "PushforwardRegistry",
    "PushforwardMethod",
    "PushforwardInfo",
    "pushforward_registry",
]
