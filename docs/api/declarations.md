> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Declarations

A declaration states the kind and structure of a value without holding the value: a term spec declares one value, and an input or output declaration declares a side of a function or of a draw.
A constraint declares the support of a numeric value.
The values themselves are on [Values and records](values.md).

## Term specs

::: probpipe.TermSpec

::: probpipe.NumericSpec

::: probpipe.NumericArraySpec

::: probpipe.OpaqueSpec

::: probpipe.RecordSpec

::: probpipe.NumericRecordSpec

::: probpipe.BatchSpec

::: probpipe.FunctionSpec

::: probpipe.DistributionSpec

::: probpipe.distributions.ConditionalDistributionSpec
    options:
      show_root_full_path: true

## Input and output declarations

::: probpipe.InputSpec

::: probpipe.OutputSpec

## Constraints

::: probpipe.Constraint

::: probpipe.real

::: probpipe.positive

::: probpipe.non_negative

::: probpipe.non_negative_integer

::: probpipe.boolean

::: probpipe.unit_interval

::: probpipe.simplex

::: probpipe.positive_definite

::: probpipe.sphere

::: probpipe.interval

::: probpipe.greater_than

::: probpipe.integer_interval

## Bijectors onto a support

`bijector_for` returns the canonical bijector from an unconstrained space onto a constraint's support.
`register_bijector`, on [Registries for extensions](extending.md), registers the bijector for a constraint.

::: probpipe.bijector_for
