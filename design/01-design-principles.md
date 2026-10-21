# Part I — Design Principles

ProbPipe's overarching aim is *simplification via abstraction*: complexity is absorbed into a few general, mathematically-grounded abstractions, so the interface a user interacts with stays small even as the space of supported methods and representations grows. That aim motivates the following principles that guide the design.

## Core Design Principles

**C1 — Uniform interface to functions, distributions, and values.** ProbPipe provides a uniform interface to operate on mathematical objects such as functions, distributions, and values.

**C2 — Functional interface over immutable objects.** Operations are functional, so have no semantically relevant side effects, and objects are never modified in place. Hence, an operation's result is determined entirely by its inputs, and that result is a new object. Every callable a user passes to ProbPipe is pure in the same sense: its result depends only on its arguments and its random key, so ProbPipe may evaluate it once and reuse the result, evaluate it again, or skip an evaluation whose result is unused.

**C3 — Computational detail hidden by default, available on demand.** Computational and algorithmic details are hidden whenever possible, while keeping them accessible for users who need precise control. The algorithm that realizes an operation and the representation of a given mathematical object are computational rather than mathematical concerns. Therefore, by default, they are handled automatically: canonical default algorithms are used and representations are converted as needed whenever possible.

**C4 — Function lifting.** A function defined on values remains well-defined when one or more of its arguments are replaced by distributions or batches over their respective types, the result being a distribution (respectively batch) over the function's output type. In the distributional case the result is the pushforward of the replaced arguments' joint distribution through the function; in the batching case the function is broadcast over the elements.

**C5 — Naming for unambiguous meaning.** Each object and the quantities inside it must carry semantically meaningful names, so a user can determine, from an object alone, what it and each of those quantities represent. An object's **label** is its name, which has no mathematical meaning. A name is required for each mathematically distinguishable quantity of an object: its outputs, and the nodes at its paths.

**C6 — Traceable and reproducible workflows.** Every result can be traced to the operations and inputs that produced it, and every workflow can be re-run to reproduce a result. The record of how a result was computed is inspectable, so a computation can be audited after the fact.

## Derived Design Principles

**D1 — Mathematical fidelity.** Every abstraction denotes a well-defined mathematical object, and every operation a well-defined mathematical operation on such objects. Distinctions that are real in the mathematics are real in the interface; conversely, a distinction that is *not* mathematical should never be reified.

**D2 — Generality first.** The generality of the mathematical abstraction is a primary objective. A specific construction is accommodated as a special case that *refines* the general contract (for example, adding capabilities or efficiency).

**D3 — Capability-based operations.** A small, fixed vocabulary of operations applies to every object that mathematically supports them, independent of the object's concrete encoding or computational backend.

**D4 — Closed system of objects under operations.** Every operation returns another first-class object of the library, so any result can itself be operated on or composed further.

**D5 — Explicit, carried structure.** The structure of a value — its named parts and their kinds — is represented explicitly and travels with the value and with the objects that produce and consume it. Structure is propagated forward from the producer that knows it.

**D6 — Single source of truth.** Each quantity has one authoritative place of storage; summaries, alternate encodings, exports, and all other views are derived from it as needed rather than stored separately.

## Execution Principles

These principles refine `C2 – Functional interface over immutable objects`, `C3 – Computational detail hidden by default, available on demand`, and `C6 – Traceable and reproducible workflows` for every part of ProbPipe that runs a computation. An **evaluation point** is one evaluation of a callable at one set of arguments, the evaluation points of one callable that one request asks for form a **set of evaluation points**, and an **executor** is a named way to run a set of evaluation points (V.9).

**P1 — Meaning is settled before execution.** Before an evaluation point runs, ProbPipe has fixed which callable it evaluates, at which arguments, and with which random key. An adaptive algorithm fixes one set of evaluation points, runs it, and may build the next set from the results.

**P2 — An executor decides only how evaluation points run.** It decides where and when they run, and whether they are compiled. It never reads an evaluation point's arguments or derives a key, except that an executor that compiles evaluation points reads their arguments to compile them.

**P3 — Results do not depend on the executor.** Every executor gives each evaluation point the same value, up to the last bit of rounding. An algorithm whose decisions can amplify one unit of rounding, such as a Markov chain's accept step, reproduces its draws bitwise only under the same compiled program and device configuration, which provenance records.

**P4 — A callable states what it allows, and ProbPipe reads the statement.** The statements are fixed when the callable is wrapped. A callable that states nothing is probed at most once per structure of its inputs, and the outcome is stored, so a feasibility check and the call it checks always agree.

**P5 — The compiled contract is opt-in, per callable.** A callable the array backend can trace is compiled, vectorized, and differentiated. Any other callable runs on concrete values through the same interface and gives the same results, more slowly, so a user never rewrites working code to use ProbPipe.

**P6 — Every set of evaluation points goes through one registry of executors,** whether it is built in Python or arrives from inside a compiled program. An executor a user registers therefore serves every part of ProbPipe that evaluates a callable many times.

**P7 — Opaque code joins compiled code at one boundary.** The outputs of code the backend cannot trace enter compiled programs as data, and a gradient crosses that code only through a derivative the callable supplies.

**P8 — Choosing an executor works like every other choice in ProbPipe:** a registry with priorities, feasibility checks, opt-in entries, and selection by name.

**P9 — A failure is reported per evaluation point.** An executor reports every evaluation point's outcome, and the code that asked for the evaluation points decides what a failure means. No executor drops an evaluation point or replaces its value.

**P10 — Nothing changes silently.** How a call runs, such as the number of its evaluation points, its executor, and any fallback, is reported before the call and recorded in its provenance. A fallback that the report cannot foresee warns when it happens.

**P11 — Per-call values are data at every boundary.** Random keys, the identity of the running call, and parameters enter compiled programs, callbacks, and worker processes as arguments, never as captured constants or as state of one thread.

**The backend's global settings.** The array backend keeps process-wide settings that change results, such as its floating-point precision and the implementation of its random generator. Five rules govern them:
1. **They belong to the user.** ProbPipe never changes one, at import or during a call.
2. **A call holds them fixed.** The outermost call takes a snapshot of them when it opens, and provenance records them.
3. **A change during a call raises,** naming the setting and both values, because the call's compiled programs were built under the old value.
4. **Output specs are checked against them.** A declared output precision that the settings do not provide is refused when the call opens, and never silently cast.
5. **Work elsewhere runs under the same settings.** Work sent to another process applies them before it reads any of its data.

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
| **Workflow** | how a computation is *run* and *recorded* | the call engine, lifting, provenance tracking, randomness and replay, executors and orchestration, caching |

Every value has two presentations:

1. The **tracked** form is the semantic layer's presentation.
2. The **raw** form is the representation layer's presentation.

The tracked form carries the raw form together with the workflow record. A user operates in the semantic layer by default and crosses into the representation layer only at explicit boundary points, whose contract is given by four further **boundary principles**:

**B1 — Either presentation in.** Wherever a value or declaration is accepted — a constructor, an operation, a method argument — it is accepted in tracked or raw form. Therefore, a caller never needs to convert manually.

**B2 — Representations only inside.** An implementation behind the boundary receives the raw form and it may return either form. Therefore, the representation layer never needs to depend on the workflow layer. One exception holds: an implementation that evaluates a user's callable many times runs those evaluation points through the workflow layer's executors. The executor of the enclosing call decides how they run and never the value of an evaluation point, and its failure rule decides which outputs the requester keeps.

**B3 — Tracked forms out by default.** A result is always returned to the user in tracked form. The raw form must be explicitly requested by the user.

**B4 — No copying at boundaries.** A wrap holds the representation it is given and raw access returns that same representation without copying. A view refers into its source rather than copying out of it, so the two presentations are backed by one store. Only a computation, or a construction that combines several values, produces new storage.
