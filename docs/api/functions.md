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

## Opaque functions inside JAX

Wrap code that uses NumPy or an external executable in a `Function` with
`traceable=False` and a concrete numeric `output_spec`, including every output
leaf's shape and dtype. JAX traces the surrounding computation without running
the opaque body. At execution, a host callback sends its evaluation points to
an executor and checks the returned values against the output spec. A missing
shape, a missing dtype, or a host-only return value is refused during tracing.
For example:

```python
import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Function, NumericArraySpec

solver = Function(
    lambda x: np.asarray(x, dtype="float32") + np.float32(1),
    traceable=False,
    output_spec=NumericArraySpec((), dtype="float32"),
)
assert float(jax.jit(lambda x: solver(x))(jnp.float32(2))) == 3
```

An embedded evaluation cannot draw workflow randomness. Gradients through an
opaque function without a derivative are refused, naming that function.
A JAX program opened directly by the user uses the default host context; a
named executor emits a warning in that case. ProbPipe's stored programs carry
the running call's context through a dynamic token.

A failed embedded evaluation stops the compiled program and raises
`EvaluationError`; a host-side executor refusal retains its own exception type
when the program runs through ProbPipe. A directly opened JAX program also
raises on failure, although JAX wraps the callback's exception.

A declared `float64` output requires `jax_enable_x64=True` when the call opens,
for both ordinary calls and `apply`. Enable it before calling, or declare a
`float32` output. An undeclared output still records the dtype actually stored.

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
