"""Distribution type conversion registry.

Provides a global ``converter_registry`` that handles bidirectional
conversion between ProbPipe distributions, raw TFP distributions, and
(optionally) scipy.stats distributions.  Also supports **protocol-based
conversion** – passing a ``@runtime_checkable`` protocol (e.g.,
``SupportsLogProb``) as the target to ``converter_registry.convert()``.
"""

from ._registry import ConverterRegistry, ConversionInfo, ConversionMethod, converter_registry
from ._protocol import Converter

# -- register built-in converters -------------------------------------------

from ._probpipe import ProbPipeConverter
from ._tfp import TFPConverter

converter_registry.register(ProbPipeConverter())
converter_registry.register(TFPConverter())

# Optionally register scipy converter
try:
    from ._scipy import ScipyConverter, _HAS_SCIPY
    if _HAS_SCIPY:
        converter_registry.register(ScipyConverter())
except ImportError:
    pass

# -- register protocol converter with built-in resolvers --------------------

from ._protocol_converter import (
    ProtocolConverter,
    _resolve_target_for_log_prob,
    _resolve_target_for_sampling,
)
from ..core.protocols import SupportsLogProb, SupportsSampling

_protocol_converter = ProtocolConverter(converter_registry)
_protocol_converter.register_protocol_target(SupportsLogProb, _resolve_target_for_log_prob)
_protocol_converter.register_protocol_target(SupportsSampling, _resolve_target_for_sampling)
converter_registry.register(_protocol_converter)

__all__ = [
    "ConverterRegistry",
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
    "ProtocolConverter",
    "converter_registry",
]
