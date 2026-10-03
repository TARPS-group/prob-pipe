> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Values and records

This page documents the value kinds, such as `Record` and `NumericArray`, their batches, and the linear operators.
The specs that declare these values are on [Declarations](declarations.md), and the callable kind `Function` is on [Functions](functions.md).

## Records

::: probpipe.Record

::: probpipe.NumericRecord

## Numeric arrays

::: probpipe.NumericArray

::: probpipe.Numeric

## Opaque values

::: probpipe.Opaque

## Batches

::: probpipe.Batch

::: probpipe.RecordBatch

::: probpipe.NumericRecordBatch

::: probpipe.NumericArrayBatch

::: probpipe.OpaqueBatch

## Designs

::: probpipe.Design

::: probpipe.FullFactorialDesign

## Weights

::: probpipe.Weights

## Named trees

::: probpipe.NamedTree

## Linear operators

`MultivariateNormal` accepts a linear operator as its `cov`, and the `_cov` method of a family that claims `SupportsCovariance` returns one.

::: probpipe.linalg.LinOp
    options:
      show_root_full_path: true

::: probpipe.linalg.DenseLinOp
    options:
      show_root_full_path: true

::: probpipe.linalg.DiagonalLinOp
    options:
      show_root_full_path: true

::: probpipe.linalg.TriangularLinOp
    options:
      show_root_full_path: true

::: probpipe.linalg.CholeskyLinOp
    options:
      show_root_full_path: true

::: probpipe.linalg.RootLinOp
    options:
      show_root_full_path: true

::: probpipe.linalg.DiagonalRootLinOp
    options:
      show_root_full_path: true
