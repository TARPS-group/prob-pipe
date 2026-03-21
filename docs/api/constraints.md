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
