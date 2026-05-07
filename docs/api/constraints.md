# Constraints

::: probpipe.Constraint

## Built-in Constraints

The following constraint singletons are available:

- `probpipe.real` -- any real number
- `probpipe.positive` -- strictly positive
- `probpipe.non_negative` -- non-negative
- `probpipe.non_negative_integer` -- non-negative integers
- `probpipe.boolean` -- 0 or 1
- `probpipe.unit_interval` -- values in [0, 1]
- `probpipe.simplex` -- vectors summing to 1 with non-negative entries
- `probpipe.positive_definite` -- positive-definite matrices
- `probpipe.sphere` -- unit-norm vectors

## Constraint Factories

::: probpipe.interval

::: probpipe.greater_than

::: probpipe.integer_interval

## Bijectors for Unconstrained Reparameterization

Given a `Constraint`, `probpipe.bijector_for` returns a canonical
TFP bijector mapping unconstrained ℝⁿ to values satisfying the
constraint. This is useful for unconstrained continuous optimization
(e.g., BFGS over a box-constrained acquisition function), MAP, and
reparameterized MCMC / VI.

Defaults follow Pyro / NumPyro conventions:

| Constraint | Bijector |
|---|---|
| `real` | `tfb.Identity()` |
| `positive` | `tfb.Exp()` |
| `non_negative` | `tfb.Softplus()` |
| `unit_interval` | `tfb.Sigmoid()` |
| `interval(low, high)` | `tfb.Sigmoid(low, high)` |
| `greater_than(b)` | `tfb.Chain([tfb.Shift(b), tfb.Exp()])` |
| `simplex` | `tfb.SoftmaxCentered()` |
| `positive_definite` | `tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])` |

Constraints with no canonical smooth bijector (`sphere`, `boolean`,
`non_negative_integer`, `integer_interval`) raise
`NotImplementedError` with a specific reason.

`register_bijector` is the extension point for custom Constraint
subclasses or for overriding defaults (e.g., preferring `Softplus`
over `Exp` for `positive`). Instance registrations take precedence
over type registrations.

::: probpipe.bijector_for

::: probpipe.register_bijector
