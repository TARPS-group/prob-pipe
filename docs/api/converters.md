# Conversion and interop

ProbPipe converts between distribution representations through a registry
of `Converter` classes. The [`from_distribution`](operations.md#conversion)
op tries registered converters in descending priority order and runs the
first whose `check()` reports a feasible conversion. Built-ins cover:

- ProbPipe-to-ProbPipe (same-class passthrough; cross-family
  moment-matching)
- TFP ↔ ProbPipe (bidirectional)
- scipy.stats ↔ ProbPipe (bidirectional, optional dependency)

For `Record`-side metadata interop — xarray `dims` / `coords` or pandas
`index` / `columns` leaves held natively by a `NumericRecord` — see
the [array-backend registry](records.md#array-backend-registry)
instead. That's a separate, simpler registry; this page is only about
distribution-to-distribution conversion.

Sampled conversions require `num_samples` to be a non-boolean positive
integer, whether the converter uses workflow-owned randomness or an explicit
`key=`. Supplying a key changes RNG ownership; it does not bypass conversion
parameter validation.

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
