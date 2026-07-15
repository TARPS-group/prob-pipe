# Domain Context Format

Use this profile only when the repository already uses `CONTEXT.md`/`CONTEXT-MAP.md` or the user explicitly approves creating that form. Prefer the repository's native design-document format otherwise. Do not create or modify context files without write authorization.

## Structure

The following is just an example.

```md
# Probabilistic Workflow Semantics

This context defines the mathematical objects that cross ProbPipe's public operation boundary and the language used to distinguish their structure, randomness, and multiplicity.

## Language

**Event**:
One value drawn from a probability distribution, with structure described by an EventTemplate.
_Avoid_: Sample, record

**Conditional Distribution**:
A probability kernel that maps a given value to a Distribution over its event.
_Avoid_: Model, callable distribution

**Batch**:
A collection of objects along named multiplicity axes, distinct from one structured or joint event.
_Avoid_: Event shape, joint distribution

## Relationships

- A **Distribution** has exactly one **EventTemplate** describing each draw
- A **Conditional Distribution** produces a **Distribution** when all given fields are bound
- A **Computation** lifts over a **Batch** or **Distribution** when its declared parameter expects one element or value

## Example dialogue (optional)

> **Contributor:** "If a Computation receives a Distribution where it expects a value, does the result become a Conditional Distribution?"
> **Maintainer:** "No. The Computation samples and lifts to a pushforward Distribution; a Conditional Distribution is specifically a kernel with an unbound given side."

## Flagged ambiguities

- "batch" was used for both a collection of separate laws and one joint array-valued event — resolved: **Batch** describes multiplicity; `event_template` describes one event.
```

## Rules

- **Be opinionated.** When multiple words exist for the same concept, pick the best one and list the others as aliases to avoid.
- **Flag conflicts explicitly.** If a term is used ambiguously, call it out in "Flagged ambiguities" with a clear resolution.
- **Keep definitions tight.** One sentence max. Define what it IS, not what it does.
- **Show relationships.** Use bold term names and express cardinality where obvious.
- **Only include terms specific to this project's domain or mathematical language.** General programming concepts (timeouts, error types, utility patterns) do not belong merely because the project uses them.
- **Group terms under subheadings** when natural clusters emerge. If all terms belong to a single cohesive area, a flat list is fine.
- **Use an example dialogue only when it clarifies a real ambiguity.** Do not force business-style dialogue into mathematical, scientific, or library documentation.
- **Record decision status outside definitions.** A term definition should describe the accepted language; provisional terminology belongs in flagged ambiguities or a working draft.

## Single vs multi-context repos

**Single context (most repos):** One `CONTEXT.md` at the repo root.

**Multiple contexts:** A `CONTEXT-MAP.md` at the repo root lists the contexts, where they live, and how they relate to each other:

```md
# Context Map

## Contexts

- [Core Value Model](./probpipe/core/CONTEXT.md) — defines EventTemplate, Record, Batch, identity, and provenance
- [Distribution Algebra](./probpipe/distributions/CONTEXT.md) — defines probability measures, kernels, capabilities, and composition
- [Inference Backends](./probpipe/inference/CONTEXT.md) — adapts inference methods to produce ordinary ProbPipe distributions

## Relationships

- **Core Value Model → Distribution Algebra**: distributions use EventTemplate for event schemas and Batch for collections of laws
- **Distribution Algebra → Inference Backends**: inference methods consume capability-bearing distributions and return distributions with provenance
- **Inference Backends → Core Value Model**: backend results cross the public boundary as tracked Records, Batches, or distribution-like terms
```

When this profile is selected, infer the existing structure:

- If `CONTEXT-MAP.md` exists, read it to find contexts
- If only a root `CONTEXT.md` exists, single context
- If neither exists, do not create one unless the user explicitly approves this profile and location

When multiple contexts exist, infer which one the current topic relates to. If unclear, ask.
