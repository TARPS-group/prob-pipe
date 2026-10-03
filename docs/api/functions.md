> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Functions

A `Function` is an immutable callable with a frozen signature, and calling it with a distribution or a batch as an argument lifts the call over that argument.
This page also documents the capabilities a function claims and the errors a call raises.
The operations, which are functions with several implementations, are on [Operations](operations.md), and the workflow scopes that key a call's random draws are on [Workflows and reproducibility](workflows.md).

## Functions

::: probpipe.Function

::: probpipe.function

::: probpipe.FunctionBatch

::: probpipe.core.node.Node
    options:
      show_root_full_path: true

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

## Modules

`Module` and the decorators of this section are experimental.

::: probpipe.Module

::: probpipe.AbstractModule

::: probpipe.workflow_method

::: probpipe.abstract_workflow_method

::: probpipe.core.node.InputFrozenError
    options:
      show_root_full_path: true
