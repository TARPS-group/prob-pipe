# Protocols

Protocols define capabilities that distributions may support. Each protocol is
`@runtime_checkable`, so compliance is checked via `isinstance` at dispatch time.
External distribution types can satisfy protocols through structural subtyping
without inheriting from ProbPipe base classes.

Protocol methods use an underscore prefix (`_sample`, `_log_prob`, `_mean`, ...)
to distinguish the primitive implementation from the public
[operations](operations.md) API.

## Sampling and Expectations

::: probpipe.core.protocols.SupportsSampling

::: probpipe.core.protocols.SupportsExpectation

## Density Evaluation

::: probpipe.core.protocols.SupportsLogProb

::: probpipe.core.protocols.SupportsUnnormalizedLogProb

## Moments

::: probpipe.core.protocols.SupportsMean

::: probpipe.core.protocols.SupportsVariance

::: probpipe.core.protocols.SupportsCovariance

## Conditioning

::: probpipe.core.protocols.SupportsConditioning

## Named Components

::: probpipe.core.protocols.SupportsNamedComponents

::: probpipe.core.protocols.SupportsConditionableComponents
