# Conversion and interop

[`from_distribution`](operations.md#conversion) dispatches through the
converter registry, trying registered `Converter` classes in descending
priority order. Built-ins cover ProbPipe-to-ProbPipe, TFP, and
scipy.stats. Backend-specific `Record` metadata (xarray dims, pandas
index) round-trips through the
[auxiliary-metadata registry](records.md#auxiliary-metadata-registry).

## Registry

::: probpipe.converter_registry

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
| 200 | `ProtocolConverter` | Resolves protocol targets (e.g., `SupportsLogProb`) to a concrete type and delegates back to the registry |
| 100 | `ProbPipeConverter` | ProbPipe-to-ProbPipe (same-class passthrough or cross-family moment-matching) |
| 50 | `TFPConverter` | Bidirectional TFP ↔ ProbPipe |
| 25 | `ScipyConverter` | Bidirectional scipy.stats ↔ ProbPipe (optional) |

Higher priority is tried first.
