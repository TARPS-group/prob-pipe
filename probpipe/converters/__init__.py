"""Distribution type conversion registry.

Provides a global ``converter_registry`` that handles bidirectional
conversion between ProbPipe distributions, raw TFP distributions, and
(optionally) scipy.stats distributions.  Also supports **protocol-based
conversion** – passing a ``@runtime_checkable`` protocol (e.g.,
``SupportsLogProb``) as the target to ``converter_registry.convert()``.
"""

# -- register built-in converters -------------------------------------------
from ._probpipe import ProbPipeConverter
from ._registry import (
    ConversionInfo,
    ConversionMethod,
    Converter,
    ConverterRegistry,
    converter_registry,
)
from ._tfp import TFPConverter

converter_registry.register(ProbPipeConverter())
converter_registry.register(TFPConverter())

# Optionally register scipy converter
try:
    from ._scipy import _HAS_SCIPY, ScipyConverter

    if _HAS_SCIPY:
        converter_registry.register(ScipyConverter())
except ImportError:
    pass

# -- register protocol converter with built-in resolvers --------------------

from ..core.protocols import SupportsLogProb
from ._protocol import ProtocolConverter, _resolve_target_for_log_prob

_protocol_converter = ProtocolConverter(converter_registry)
_protocol_converter.register_protocol_target(SupportsLogProb, _resolve_target_for_log_prob)
converter_registry.register(_protocol_converter)

__all__ = [
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
    "ConverterRegistry",
    "ProtocolConverter",
    "converter_registry",
]
