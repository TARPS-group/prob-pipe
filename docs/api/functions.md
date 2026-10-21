> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Functions

A `Function` is an immutable callable with a frozen signature, and calling it with a distribution or a batch as an argument lifts the call over that argument.
This page also documents the capabilities a function claims and the errors a call raises.
The operations, which are functions with several implementations, are on [Operations](operations.md), and the workflow scopes that key a call's random draws are on [Workflows and reproducibility](workflows.md).

## Functions

::: probpipe.Function

::: probpipe.function

::: probpipe.FunctionBatch

## Binding arguments

Pass an ordinary bound method to `Function` when a computation uses an object's
state. Use `bind=` for arguments supplied at construction; call arguments
replace those bindings, and omitted arguments use the callable's defaults.

```python
from probpipe import Function


class Affine:
    def __init__(self, offset):
        self.offset = offset

    def evaluate(self, x, scale=1.0):
        return (x + self.offset) * scale


compute = Function(Affine(3.0).evaluate, bind={"scale": 2.0})
assert float(compute(1.0)) == 8.0
assert float(compute(1.0, scale=3.0)) == 12.0
```

## Differentiability and invertibility

::: probpipe.SupportsDifferentiation

::: probpipe.is_differentiable

::: probpipe.SupportsInverse

::: probpipe.is_invertible

::: probpipe.SupportsLogDetJacobian

## Checking a call

::: probpipe.CallReport

::: probpipe.ApplicabilityError

::: probpipe.ResultKindError

::: probpipe.ResultSchemaError

## Failed evaluations

A call that evaluates a function many times, such as a call lifted over a distribution, runs every evaluation and keeps the outputs that succeeded.
When some evaluations fail, the call then raises `EvaluationError`, which names each failed one and is chained to the first failure's exception.

::: probpipe.EvaluationError
