"""Distribution type conversion registry.

Provides a global ``converter_registry`` that handles bidirectional
conversion between ProbPipe distributions, raw TFP distributions, and
(optionally) scipy.stats distributions.
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

__all__ = [
    "ConverterRegistry",
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
    "converter_registry",
]
