# Part I — Design Principles

ProbPipe's overarching aim is *simplification via abstraction*: complexity is absorbed into a few general, mathematically-grounded abstractions, so the interface a user interacts with stays small even as the space of supported methods and representations grows. The principles in this part make that aim concrete.

## Core Design Principles

**C1 — Uniform interface to functions, distributions, and values.** ProbPipe provides a uniform interface to operate on mathematical objects such as functions, distributions, and values.

**C2 — Functional interface over immutable objects.** Operations are functional, so have no semantically relevant side effects, and objects are never modified in place. Hence, an operation's result is determined entirely by its inputs, and that result is a new object.

**C3 — Computational detail hidden by default, available on demand.** Computational and algorithmic details are hidden whenever possible, while keeping them reachable for users who need precise control. The algorithm that realizes an operation and the representation of a given mathematical object are computational rather than mathematical concerns. Therefore, by default, they are handled automatically: canonical default algorithms are used and representations are converted as needed whenever possible.

**C4 — Function lifting.** A function defined on values remains well-defined when one or more of its arguments are replaced by distributions or batches over their respective types, the result being a distribution (respectively batch) over the function's output type. In the distributional case the result is the pushforward of the replaced arguments' joint distribution through the function; in the batching case the function is broadcast over the elements.

**C5 — Naming for unambiguous meaning.** Each object and any of its components must carry semantically meaningful names, so a user can determine, from an object alone, what it and its components represent.

**C6 — Traceable and reproducible workflows.** Every result can be traced to the operations and inputs that produced it, and every workflow can be re-run to reproduce a result. The record of how a result was computed is inspectable, so a computation can be audited after the fact.

## Derived Design Principles

**D1 — Mathematical fidelity.** Every abstraction denotes a well-defined mathematical object, and every operation a well-defined mathematical operation on such objects. Distinctions that are real in the mathematics are real in the interface; conversely, a distinction that is *not* mathematical should never be reified.

**D2 — Generality first.** The generality of the mathematical abstraction is a primary objective. A specific construction is accommodated as a special case that *refines* the general contract (for example, adding capabilities or efficiency).

**D3 — Capability-based operations.** A small, fixed vocabulary of operations applies to every object that mathematically supports them, independent of the object's concrete encoding or computational backend.

**D4 — Closed system of objects under operations.** Every operation returns another first-class object of the library, so any result can itself be operated on or composed further.

**D5 — Explicit, carried structure.** The structure of a value — its named parts and their kinds — is represented explicitly and travels with the value and with the objects that produce and consume it. Structure is propagated forward from the producer that knows it.

**D6 — Single source of truth.** Each quantity has one authoritative place of storage; summaries, alternate encodings, exports, and all other views are derived from it as needed rather than stored separately.

## The Three Layers

The principles above imply a three-layer separation of concerns that the rest of the design is built around:
1. `C3 – Computational detail hidden by default, available on demand` distinguishes what a user means (semantics) from how it is computed (representation).
2. `D6 – Single source of truth` requires that each quantity be stored once.
3. `C6 – Traceable and reproducible workflows` requires a record of how a result was produced.

These distinct responsibilities are organized into layers:

| Layer | What it holds | Examples |
|---|---|---|
| **Semantic** | the mathematical objects and the operations on them | the object kinds and their type specifications; capabilities; the operation vocabulary |
| **Representation** | how a semantic object is concretely stored and used in computations | backing arrays and array backends, columnar and object-array storage, flat-vector layouts, operation dispatch registries and representation converters |
| **Workflow** | how a computation is *run* and *recorded* | the call engine, lifting, provenance tracking, randomness and replay, compute dispatch and orchestration, caching |

Throughout, **raw** refers to the representation layer accessed from the semantic one: a term's raw form is the object that represents it, detached from the workflow — no lineage or anything the workflow would carry forward. A **tracked term** is a semantic object with a representation to compute with and the workflow record of where it came from. A user operates in the semantic layer by default and accesses the representation layer through a small number of explicit boundary points.
