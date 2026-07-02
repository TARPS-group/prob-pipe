# ProbPipe Consolidated Design Reference

This document consolidates the design intent spread across the active design threads that reconfigure the core value, distribution, and collection abstractions. Its purpose is coordination: to surface the principles those threads share, to make overlapping concerns explicit, and to limit re-litigation by fixing what is settled at the level of *principles*. It is meant to stand on its own and to be the reference from which more granular documentation is later derived.

### Contents 

The document has five parts plus one more planned:

- **[Part I — Design Principles](01-design-principles.md)** — the high-level commitments that drive every downstream design decision. They are deliberately stated without reference to any specific class, type, or API.
- **[Part II — Infrastructure](02-infrastructure.md)** — the generic, type-agnostic machinery the rest of the library is built on: the named-tree abstraction, identity, provenance, metadata, batching, and the dispatch registries.
- **[Part III — Core Abstractions](03-core-abstractions.md)** — the probability domain in dependency order: schemas, values, distributions, conditional distributions, composition, and the hierarchy of distribution kinds, each with a precise contract that must align with the design principles.
- **[Part IV — Workflow Functions](04-workflow-functions.md)** — how an ordinary function is lifted into ProbPipe: broadcasting over distributions, dispatch, orchestration, and provenance. This is the substrate the operations build on.
- **[Part V — Operation Contracts](05-operation-contracts.md)** — precise contracts for the core operations: moments, sampling, density evaluation, conditioning, composition, and batched operations.
- **Part VI — Agentic Interface (planned)** — A higher-level agentic interface to help guide the process of designing, building, and auditing a ProbPipe workflow. 

### Conventions

#### Structure 

Every section in Parts II through V leads with a **Contract** subsection, which describes what the abstraction or operation is, and its precise public interface, in plain language and typing. Next, the **Rationale** subsection describes the reasoning for the design and how it aligns with the Part I design principles. **Notes** and **Open points** subsections appear only where necessary, and should be used sparingly.
#### Formatting

Class and method names are set in code font.  Design principles are cited only in the **Rationale** subsections, by identifier and short name, drawing on the *core principles* (the C-series) and *derived principles* (the D-series) of Part I. For example, the fourth core principle would be cited as `C4 – Function lifting via pushforward`. An abstraction is referred to by its class name rather than by the underlying mathematical concept, except in mathematical statements. For example, `ConditionalDistribution` is used throughout, with *kernel* reserved for mathematical statements such as its definition as a probability kernel `K : S → P(T)`. 
