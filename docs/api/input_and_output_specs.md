> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Input and output specs

A spec states the kind and structure of a value without holding the value: a term spec declares one value, an input spec declares the inputs of a function or of a conditional distribution, and an output spec declares the outputs of a function or of a draw.
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

## Input and output specs

An `InputSpec` reads another input spec, a mapping, a `RecordSpec`'s top-level
fields, or an `OutputSpec`'s outputs. Copying an input spec preserves its optional
inputs. Tuple shapes are shorthand for `NumericArraySpec`, in input declarations
and keyword output declarations alike:

```python
from probpipe import InputSpec, OutputSpec, RecordSpec

inputs = InputSpec(mu=(), observations=(100,))
event = OutputSpec(RecordSpec(mu=(), sigma=()))
mu_input = InputSpec(event[["mu"]])
```

Pass a law's `event_spec` directly as `given_spec` when a kernel takes those
outputs as inputs. Use `event_spec[["mu"]]` to retain one output, or
`event_spec.without("sigma")` to remove an output. To pass a whole record as one
input, retain its named group with list selection. These declarations name
inputs; passing the law itself does not describe their types.

Input list selection retains declaration order and optional flags. Merging
input specs with `left & right` retains the first type of a shared input after
checking compatibility. A shared input is optional only when both sides mark it
optional. A Python parameter's default marks its declared input optional;
declaring an optional input for a parameter with no default is an error.

Record field order is part of a declaration. An existing record or record spec
in another order does not conform, including inside a nested record. At function
input and return boundaries, and when binding a kernel's conditioning values,
a raw mapping may arrive in any order: the body and returned value receive its
fields in declaration order. This leaves the caller's mapping unchanged.
`InputSpec.without` raises `KeyError` for a name it does not declare.

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
