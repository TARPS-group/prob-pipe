# ProbPipe Consolidated Design Reference

This document consolidates the design intent spread across the active design threads that reconfigure the core value, distribution, and collection abstractions. Its purpose is coordination: to surface the principles those threads share, to make overlapping concerns explicit, and to limit re-litigation by fixing what is settled at the level of *principles*. It is meant to stand on its own and to be the reference from which more granular documentation is later derived.

### Mathematical scope

ProbPipe is built around five kinds of mathematical objects and the operations that connect them. Every operation returns another ProbPipe object, so the system is closed under all of them.

| Object | Mathematics | ProbPipe |
|---|---|---|
| value | a point `x` in a structured space `X` | `Record` |
| probability measure | `μ ∈ P(X)` | `Distribution` |
| probability kernel | `K : S → P(T)` | `ConditionalDistribution` |
| function | `f : X → Y` | workflow function |
| linear operator | `A : ℝⁿ → ℝᵐ` | `LinOp` |

Each object also has an indexed-collection form (a batch), and every operation lifts to batches elementwise. Values, distributions, and conditional distributions additionally have **numeric** specializations (`NumericRecord`, `NumericDistribution`, …) covering the all-array case: they identify the event space with a flat vector space, where `LinOp` acts and differentiation applies.

| Operation | Mathematics |
|---|---|
| evaluation | `f(x)`, `K(s, ·)`, `Ax` |
| sampling | `x ~ μ` |
| density evaluation | `(dμ/dν)(x)` |
| expectation and moments | `E[f(X)]` for `X ~ μ`; mean, variance, covariance, quantiles |
| pushforward | `f♯μ`, the law of `f(X)` for `X ~ μ` |
| composition | `p(x \| y) · p(y)`, `f ∘ g`, `A B` |
| conditioning | `μ(· \| y = b)` for a field `y`, from exact currying to Bayesian inversion |
| marginalization | the law of a named field of `X ~ μ` |
| prediction | `μK = ∫ K(s, ·) μ(ds)` |

### Contents 

The document has six parts plus one more planned:

- **[Part I — Design Principles](01-design-principles.md)** — the high-level commitments that drive every downstream design decision. They are deliberately stated without reference to any specific class, type, or API.
- **[Part II — Shared Abstractions](02-shared-abstractions.md)** — the generic, type-agnostic machinery the rest of the library is built on: the named-tree abstraction, identity, provenance, metadata, batching, and the dispatch registries.
- **[Part III — Values and Distributions](03-values-and-distributions.md)** — the probability domain in dependency order: schemas, values, linear operators, distributions, conditional distributions, composition, and the hierarchy of distribution kinds, each with a precise contract that must align with the design principles.
- **[Part IV — Workflow Functions](04-workflow-functions.md)** — how an ordinary function is lifted into ProbPipe: broadcasting over distributions, dispatch, orchestration, and provenance. This is the layer the operations build on.
- **[Part V — Operations](05-operations.md)** — precise contracts for the core operations: moments, sampling, density evaluation, conditioning, composition, pushforward, and batched operations.
- **[Part VI — The Distribution Catalog](06-distribution-catalog.md)** — the concrete families placed on the hierarchy's axes: parametric, empirical, mixtures, pushforward results, the Gaussian algebra, random functions and measures, inference-produced distributions, and the conditional families, including GLM likelihoods.
- **Part VII — Agentic Interface (planned)** — A higher-level agentic interface to help guide the process of designing, building, and auditing a ProbPipe workflow. 

### Conventions

#### Structure 

Every section in Parts II through VI leads with a **Contract** subsection, which describes what the abstraction or operation is, and its precise public interface, in plain language and typing. Next, the **Rationale** subsection describes the reasoning for the design and how it aligns with the Part I design principles. **Notes** and **Open points** subsections appear only where necessary, and should be used sparingly.
#### Formatting

Class and method names are set in code font.  Design principles are cited only in the **Rationale** subsections, by identifier and short name, drawing on the *core principles* (the C-series) and *derived principles* (the D-series) of Part I. For example, the fourth core principle would be cited as `C4 – Function lifting via pushforward`. An abstraction is referred to by its class name rather than by the underlying mathematical concept, except in mathematical statements. For example, `ConditionalDistribution` is used throughout, with *kernel* reserved for mathematical statements such as its definition as a probability kernel `K : S → P(T)`. 
