# Part I — Design Principles

ProbPipe's overarching aim is *simplification via abstraction*: complexity is absorbed into a few general, mathematically-grounded abstractions, so the interface a user interacts with stays small even as the space of supported methods and representations grows. The principles in this part make that aim concrete.

## Core Design Principles

**C1 — Uniform interface to distributions and values.** ProbPipe provides a single, mathematically-oriented interface to probability distributions and to the values that arise from them. Distributions and values are first-class objects, related by the operations between them.

**C2 — Functional interface over immutable objects.** The interface is functional: operations have no side effects, and objects are never modified in place. An operation's result is determined entirely by its inputs, and any transformation yields a new object rather than mutating an existing one.

**C3 — Computational detail hidden by default, available on demand.** Computational and algorithmic details are hidden whenever possible, while keeping them reachable for users who need precise control. The algorithm that realizes an operation and the representation of a given mathematical object are computational rather than mathematical concerns; by default they are handled automatically — sensible default algorithms are used and representations are converted as needed, when possible. 

**C4 — Function lifting via pushforward.** A function defined on values remains well-defined when one or more of its arguments are replaced by distributions over their respective types, the result being a distribution over the function's output type. That result is the pushforward of the replaced arguments' joint distribution through the function.

**C5 — Naming for unambiguous meaning.** Readability is a first-class goal: a user must be able to determine, from an object alone, what it and its parts represent. Names carry semantic meaning and help to minimize ambiguity.

**C6 — Traceable and reproducible workflows.** Every result can be traced to the operations and inputs that produced it, and every workflow can be re-run to reproduce a result. The record of how a result was computed is inspectable, so a computation can be audited after the fact. 

## Derived Design Principles

**D1 — Mathematical fidelity.** Every abstraction denotes a well-defined mathematical object, and every operation a well-defined mathematical operation on such objects. Distinctions that are real in the mathematics are real in the interface — a measure versus a family of measures indexed by a conditioning value, conditioning versus marginalizing, a single object versus a collection of objects. Conversely, a distinction that is *not* mathematical is not reified.

**D2 — Generality first.** The generality of the mathematical abstraction is the primary design driver. A specific construction is accommodated as a special case that *refines* the general contract — adding capabilities or efficiency — and never as an exception that narrows or contradicts it. When a convenience for one construction conflicts with the generality of the abstraction, the abstraction is kept general and the construction is expressed within it.

**D3 — Capability-based operations.** A small, fixed vocabulary of operations applies to every object that mathematically supports them, independent of the object's concrete encoding or computational backend. 

**D4 — Closed system of objects under operations.** Every operation returns another first-class object of the library, so any result can itself be operated on or composed further. 

**D5 — Explicit, carried structure.** The structure of a value — its named parts and their kinds — is represented explicitly and travels with the value and with the objects that produce and consume it. Structure is propagated forward from the producer that knows it.

**D6 — Differentiability where possible.** An operation is differentiable end-to-end when the objects it acts on are backed by a differentiable, array-native representation, so gradients propagate through an entire workflow; when they are not, the operation remains correct but forgoes differentiability. Support for differentiation is encouraged but not required.

**D7 — Single source of truth.** Each quantity is stored once, in one authoritative place; summaries, alternate encodings, exports, and any other view are derived from it on demand rather than stored separately. 
