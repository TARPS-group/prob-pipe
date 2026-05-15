# Conversion and interop

ProbPipe converts between distribution representations through a registry
of `Converter` classes. The [`from_distribution`](operations.md#conversion)
op tries registered converters in descending priority order and runs the
first whose `check()` reports a feasible conversion. Built-ins cover:

- ProbPipe-to-ProbPipe (same-class passthrough; cross-family
  moment-matching)
- TFP ↔ ProbPipe (bidirectional)
- scipy.stats ↔ ProbPipe (bidirectional, optional dependency)

For `Record`-side metadata interop — preserving xarray `dims` / `coords`
or pandas `index` / `columns` through a `NumericRecord` round-trip — see
the [auxiliary-metadata registry](records.md#auxiliary-metadata-registry)
instead. That's a separate, simpler registry; this page is only about
distribution-to-distribution conversion.

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
