# ProbPipe Consolidated Design Reference

This reference is the single source of truth for the design intent of ProbPipe, a Python framework for building probabilistic pipelines with automated uncertainty quantification. It is meant to stand on its own and to be the reference from which more granular documentation is later derived. It always describes the *target state*, so where it disagrees with the current code, the design should be treated as ahead of implementation. The repository's contributor documentation should describe the code as it stands.

### Mathematical scope

ProbPipe is built around a small number of mathematical objects, their specializations, and the operations that connect them. Every operation returns another ProbPipe object, so the system is closed under all of them.

| Object | Mathematics | ProbPipe |
|---|---|---|
| numeric array | a multidimensional array `x` in a subset of a finite-dimensional real space | `NumericArray` |
| non-numeric value | a value `x` in an arbitrary space `X`, treated as atomic | `Opaque` |
| structured value | a point `x` in a named product space `X = X₁ × ⋯ × Xₖ`, possibly nested | `Record` |
| probability measure | `μ ∈ P(X)` | `Distribution` |
| probability kernel | `K : S → P(T)` | `ConditionalDistribution` |
| function | `f : X → Y` | `Function` |
| linear operator | `A : ℝⁿ → ℝᵐ` | `LinOp`, the linear `Function` subtype |

Each object also has an indexed-collection form (a *batch*), and every function lifts to batches elementwise. Structured values, distributions, and conditional distributions additionally have **numeric** specializations (`NumericRecord`, `NumericDistribution`, …) covering the all-array case: they identify the event space with a flat vector space, where `LinOp` acts and differentiation applies.

Some important mathematical operations supported by ProbPipe include the following:

| Operation | Mathematics |
|---|---|
| evaluation | `f(x)`, `K(s, ·)`, `Ax`, and the pushforward `f♯μ` (i.e., the law of `f(X)` for `X ~ μ`) |
| sampling | `x ~ μ` |
| density evaluation | `(dμ/dν)(x)` |
| expectation and summaries | `E[f(X)]` for `X ~ μ`; mean, variance, covariance, quantiles |
| composition | `p(x \| y) · p(y)`, `f ∘ g`, `A B` |
| conditioning | `μ(· \| y = b)` for a field `y` |
| marginalization | the law of a named field of `X ~ μ` |
| mixture | `μK = ∫ K(s, ·) μ(ds)`, the law of `T` for `S ~ μ` and `T ~ K(S, ·)` |

### Contents

The document has six parts, a package-structure companion, and one more part planned:

- **[Part I — Design Principles](01-design-principles.md)** — the high-level commitments that drive every downstream design decision. They are deliberately stated without reference to any specific class, type, or API.
- **[Part II — Shared Abstractions](02-shared-abstractions.md)** — the generic, type-agnostic machinery the rest of the library is built on: the term-specification layer with the input/output declarations and the `Numeric` interface, identity, provenance, and metadata, batching, the named-tree abstraction, and the dispatch registries.
- **[Part III — Values and Distributions](03-values-and-distributions.md)** — the term kinds and the probability domain in dependency order: the base value kinds, functions and linear operators, records and record batches, distributions, conditional distributions, composition, and the classification of distribution kinds, each with a precise contract that must align with the design principles.
- **[Part IV — Functions](04-functions.md)** — how an ordinary Python callable is lifted into ProbPipe: broadcasting over distributions, randomness, dispatch, orchestration, and provenance. This is the layer the operations build on.
- **[Part V — Operations](05-operations.md)** — precise contracts for the core operations, functions before distributions: function evaluation, inversion, sampling, density evaluation, distribution functionals, conditioning, joints, marginals and factors, mixtures, and batched operations.
- **[Part VI — The Distribution Catalog](06-distribution-catalog.md)** — the concrete families placed on the hierarchy's axes: parametric, empirical, mixtures, evaluation results, random functions and measures, the Gaussian algebra, inference-produced distributions, and the conditional families, including GLM likelihoods.
- **Part VII — Agentic Interface (planned)** — A higher-level agentic interface to help guide the process of designing, building, and auditing a ProbPipe workflow.
- **[Package Structure](package-structure.md)** — the target package and module layout realizing the parts: the layered import graph, upward registration, and the public-API conventions.

### Conventions

#### Structure

Every numbered section in Parts II through VI leads with a **Contract** subsection, which describes what the abstraction or operation is, and its precise public interface, in plain language and typed signatures. Next, the **Rationale** subsection describes the reasoning for the design and how it aligns with the Part I design principles. **Notes** and **Open points** subsections appear only where necessary.

#### Formatting

Class and method names are set in code font. Design principles are cited only in the **Rationale** subsections, by identifier and short name, drawing on the *core principles* (the C-series), *derived principles* (the D-series), and *boundary principles* (the B-series) of Part I. For example, the fourth core principle would be cited as `C4 – Function lifting`.  An abstraction is referred to by its class name rather than by the underlying mathematical concept, except in mathematical statements. For example, `ConditionalDistribution` is used throughout, with *kernel* reserved for mathematical statements such as its definition as a probability kernel `K : S → P(T)`.

A code comment states what the signature does not — a default, a constraint, a unit — and never the sentence above the block. A **Rationale** names the principle and the reason; it never restates the contract.
