# Conversion and interop

ProbPipe's `from_distribution` op dispatches via a **converter registry** of
priority-ordered `Converter` classes. Built-ins cover ProbPipe-internal
conversions, bidirectional TFP, and bidirectional scipy.stats; user code
can register more.

The op itself is documented under
[Operations → Conversion](operations.md#conversion):

```python
from probpipe import from_distribution
target = from_distribution(source, TargetClass)
```

For Record-side conversion of xarray / pandas metadata (which is metadata,
not a distribution), see
[Records and data → Auxiliary-metadata registry](records.md#auxiliary-metadata-registry).

## Registry

The shipped registry instance is `converter_registry`. It is rarely
imported directly — `from_distribution` does the lookup — but is the entry
point for registering new converters.

::: probpipe.converter_registry
    options:
      show_root_heading: true

::: probpipe.converters.ConverterRegistry
    options:
      members:
        - register
        - check
        - convert
        - is_distribution_type

## Converter classes

::: probpipe.Converter

::: probpipe.ConversionInfo

::: probpipe.ConversionMethod

## Built-in priorities

| Priority | Converter | Role |
|----------|-----------|------|
| 200 | `ProtocolConverter` | Intercepts protocol targets (e.g., `SupportsLogProb`), resolves to a concrete type, and delegates back to the registry |
| 100 | `ProbPipeConverter` | ProbPipe-to-ProbPipe conversions (same-class passthrough or cross-family moment-matching) |
| 50 | `TFPConverter` | Bidirectional TFP ↔ ProbPipe conversions |
| 25 | `ScipyConverter` | Bidirectional scipy.stats ↔ ProbPipe conversions (optional) |

Higher priority is tried first. Protocol-level converters should be above
concrete-type converters.
