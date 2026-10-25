# Glossary

The terms of art of the reference, in alphabetical order. Each entry defines its term and names the section that owns its rules, where those rules are stated.

- **alias**: a label the caller gives explicitly, through a constructor's `label=`, a function's `output_label=`, or `with_label`, which replaces a term's description for display and supplies no output, field, level, or identity (II.4).
- **argument normalization**: step 3 of a call, which wraps each argument into its kind, plans any conversion it needs, and admits the result against its parameter (V.4).
- **atom**: one point of an empirical law's finite support, held with its weight in the event type's batch form (VII.2).
- **automatic selection**: the selection order of the executor registry applied to a set of evaluation points, which tries the executors from the highest priority down, skips the opt-in ones, and runs the first that can run the set (V.9).
- **base spec**: a `NumericArraySpec` or an `OpaqueSpec`, the spec of a base value, which has no named entries; only a base value takes a name stated outside it, as the output at the root (II.2).
- **batch**: an indexed collection of separate objects of one element type, with its axes grouped into named levels (II.5). A kind's batch form is its batch class, such as `RecordBatch` for records (II.1).
- **broadcast**: the lift of a function over a distribution argument, whose result is the pushforward law of the function's output (V.5).
- **call context**: the per-call state that the code a call runs reads wherever it runs, such as the active `config.using` blocks, the controls a derived route passes down, and the call's resolved controls. Each call of a `Function` opens a child of the call context in effect, and a `config` block derives a changed copy (V.2, V.9).
- **call token**: a `numpy.uint32` that names one open call context in a thread-safe table of active calls, passed as an argument to every JAX program ProbPipe opens to run user or route code, so that an embedded call's host side finds its call's settings; token 0 names no call (V.13).
- **capability**: an underscore implementation of an operation, such as `_sample` or `_inverse`, that a class claims for its instances by implementing the capability's protocol (III.8, III.3).
- **chunk**: a contiguous slice of a set of evaluation points that an executor runs as one unit, such as one cluster job; an `EvaluationChunk` pickles (V.9).
- **column**: one array of a batch's columnar storage, a field's raw batch form, which `raw` returns (III.5), or one column of a displayed table (II.4). The batch that `b[path]` returns is the *marginal batch*, not a column.
- **component**: one of the component laws of a `MixtureDistribution`, which its constructor takes as `components=` (VII.3).
- **conforms**: a value conforms to a spec when it has the structure the spec describes, and a spec conforms to another when every value that conforms to the first conforms to the second, within one dimension scope; conformance is the one-way check of what an input, an output spec, or a field receives (II.1). Completing an output spec checks that the produced value conforms (II.2).
- **conversion**: a change of a distribution's representation that preserves its law, up to the recorded fidelity, and its event spec; a converter is the registered method that carries one out (IV.3).
- **coupled**: two laws are coupled when they have one root ancestor: a law and its coupled marginals, two coupled marginals of one law, and a law and a rename of it. A lift draws coupled laws together (V.5), and a merge refuses them (IV.2).
- **coupled marginal**: the marginal `d[path]` of a law, or `k[path]` of a kernel, which keeps its parent: a draw of it is the part at `path` of a draw of its parent, so it keeps the dependence between the parts of one law. `CoupledMarginal` is its class for laws and `ConditionalCoupledMarginal` for kernels (III.7, III.9).
- **declaration**: static information an object states before any computation, such as a spec, a claimed capability, or a method's exactness (II.7). A term's inputs and outputs are declared by its input spec and output spec (II.2).
- **derived operation**: an operation defined by a formula over other operations, such as `prob = exp ∘ log_prob` or `expectation(d, f) = mean(pushforward(d, f))` (VI.0, VI.4, VI.5).
- **derived route**: the route that computes a derived operation by its formula: by default the body route, which runs the declaration's body, or a composite route that runs a route of each constituent operation in turn. It passes `exact_only` and the two counts to the calls its formula makes (VI.0).
- **description**: the expression a term derives at construction, from its declared fields, its callable's name, or its operands, which its label renders when the caller gives no alias. A term that JAX rebuilds from its leaves has none and displays `<no description>` (II.4). A registry and each of its entries carry a different `description`, the one line the registry catalog shows (II.7).
- **element**: one object of a batch, at one position of its batch axes (II.5). The scalar values of an array are its entries.
- **embeddable and host-only**: an opaque `Function` is embeddable when its output spec describes a value a host callback can return, a numeric array or a record of them, with dtypes, and shapes bound by the declared inputs, so it can run inside a JAX program as an embedded call; it is host-only otherwise (V.13).
- **embedded call**: a call of an opaque `Function` with at least one traced argument, inside a JAX program or under an eager JAX transformation, which ProbPipe makes a host callback whose host side runs the batch it receives as one set of evaluation points (V.13).
- **evaluation context**: the call context that a call's evaluation points run under, wherever they run: the call's context with automatic selection in place of its executor choice, so a call made inside an evaluation point chooses its executor afresh, and with the session's values as they were when the outermost call opened (V.9).
- **evaluation point**: one evaluation of one callable at one set of arguments, with its random key when the callable draws (V.9). An atom of an empirical law is a point of its support, which is not an evaluation point.
- **evaluation point identity**: an evaluation point's unique name within its call, made of the callable, the ordinal of its set of evaluation points among the sets the call requested, and its point index, which reports and errors use (V.9).
- **evaluation rule**: a method of the evaluation-rule registry, which realizes a map applied to a distribution or a batch, such as a closed-form rule, an integration rule, or the sampling lift (V.7).
- **event**: one value of a distribution's space, such as a draw or a stored datum, typed by the law's event spec (III.5, III.7). A workflow-owned random event is one draw's occurrence in a workflow scope (V.8).
- **event spec**: the output spec of one draw of a `Distribution` or a `ConditionalDistribution`, its `event_spec`, whose `term_spec` is the spec of a draw and whose outputs are the law's (II.2, III.7).
- **exact and approximate**: a method or route is exact when its returned representation denotes the requested mathematical result, and approximate when it returns a stand-in for that result (II.7).
- **execution claim**: a statement that a `Function` makes when it is built about how it may be run: `traceable`, `vectorized`, or `differentiable`, which an opaque `Function` can claim only with a derivative it supplies. A claim is a declaration, fixed for the `Function`'s life (V.13).
- **executor**: an entry of the executor registry, which names one way to run a set of evaluation points, such as `sequential`, `thread`, or a user's cluster executor. It decides how the evaluation points run and never what they compute (V.9).
- **executor registry**: the dispatch registry of executors, `executor_registry`, which the registry catalog lists (V.9, II.7).
- **expression**: the immutable tree a tracked term carries, which states what the term is and which its label and its notation render (II.4). An operation builds its result's expression from its operands' expressions, and no operation reads one.
- **factor**: a constituent `Distribution` or `ConditionalDistribution` that a factored joint was built from (IV.1); `factor(d, output_name)` returns the one that gives an output (VI.8). Only a merge and a rename keep a law's factors (IV.2).
- **field**: an entry of a named tree that is not a tree, such as an array, an opaque value, or a law that a record holds, addressed by its path (II.6). A tree's `keys()` are its field paths.
- **floor**: a generic fallback with the lowest rank on its stated domain, such as the sampling lift on samplable distributions and the elementwise sweep on batches (V.7).
- **formula**: the equation over other operations that defines a derived operation, such as `prob = exp ∘ log_prob`, which the operation's declaration states as its body (VI.0).
- **group**: an entry of a named tree that is itself a tree, addressed by its path as a field is; `[]` at a group's path returns it (II.6).
- **guard**: a predicate on a capability's instance and a call's arguments that decides whether the capability supports the call, as a `LinOp`'s inverse guard rejects a non-square operator (III.8).
- **inert setting**: a `config` setting that changes no computed value, such as how a label renders or whether provenance is recorded, so no compiled JAX program depends on it; every setting of `notation`, `provenance`, and `prefect` (V.2).
- **inference target**: the law an inference method normalizes, an `InferenceTarget`: a model's unobserved quantities at its observed values, whose unnormalized log-density is the model's there; a method reads the model, the observed values, and the model's prior and likelihood through it, and the target's density as any law's (VI.6).
- **input**: a named place where a function or a conditional distribution receives one value: a function's parameter or what a conditional distribution is conditioned on. An input has a spec, to which every value supplied there conforms, and is required or optional; an optional input that a binding omits takes its default (II.2, III.9).
- **input spec**: an `InputSpec`, which lists a term's inputs in order, each with its spec and whether it is optional, and holds no defaults: a function's `input_spec` and a conditional distribution's `given_spec` (II.2).
- **input structure**: the tree structure that ProbPipe's canonical flatten gives for a call's arguments, with their shapes and dtypes and the values of its non-numeric hashable arguments, by which a probe result is stored (V.13).
- **JAX program**: a function that JAX traces and compiles, through `jax.jit` alone or with `vmap` or `pmap` inside it; one ProbPipe opens runs user or route code and takes a call token, and one the user opens takes none (V.13). A JAX program with embedded calls is one whose trace staged at least one embedded call.
- **key**: the random-number key, a PRNG key, that a workflow scope derives for each draw (V.8); a mapping's keys keep their Python sense, so a named tree's keys are its field paths (II.6).
- **kind**: a sort of tracked term, such as `NumericArray`, `Record`, or `Distribution`, identified by the class of its term spec; the kind table records each kind's tracked class and batch form (II.1).
- **label**: the string that names a tracked term for a reader, its `label` attribute, which has no mathematical meaning and on which no computation depends (C5): an operation reads its operands' labels only to label its result. It is the name part of the rendering of the term's expression. Constructors take it as keyword `label=`, requiring it for raw values without semantic structure and otherwise deriving it from fields, callable names, or described operands. II.4 owns how a label is set, defaulted, derived, and grouped, and III.7 and VI.8 own the labels of a coupled and of a standalone marginal. A law, a kernel, or a function displays its label in its notation.
- **level**: a named group of contiguous batch axes; a batch's `axis_groups` tiles its batch shape into levels, outermost first (II.5).
- **lift**: the application of a function to a distribution or a batch where it expects a value, which is a broadcast for a distribution and a sweep for a batch (C4, V.5). The sampling lift is the generic evaluation rule that realizes a broadcast from draws (V.7).
- **marginal**: the selection of a law, a kernel, a function, or a batch at a path or a list of paths, which replaces each of its values by the value's nodes there: the marginal law, kernel, function, or batch. A law's or a kernel's marginal is either *coupled*, as `[]` returns it, or *standalone*, as `marginal` returns it, and `_marginal` is a law's capability (VI.8, III.8). A record's node is not called a marginal.
- **marginal batch**: the batch, over a batch's levels, of each element's node at one path, which `b[path]` returns as a view; a batch's outputs are its marginal batches. It is the batch analog of the coupled marginal `d[path]`: for samples of `d`, the marginal batch at a path holds samples of `d[path]` (II.5).
- **merge**: what `&` does: `a & b` combines two terms of one kind by name, each output of `b` supplying the input of `a` of the same name, and returns the outputs of both, `a`'s first; for laws and kernels it is the joint, whose density is the product of the operands' (IV.2). Records merge by top-level name (II.6), functions into the function returning both results (III.3), and batches element by element (VI.11). Composition is `f ∘ g` and `A B`, never `&`.
- **method**: a dispatch method, which is one named implementation that a dispatch registry selects by type and feasibility (II.7). Elsewhere the word keeps its Python sense of a function defined on a class, as `with_label` is.
- **name**: a non-empty string without `/` that identifies an input, an output, a field, a group, or a level (II.2, II.5, II.6); an input's name is also a Python identifier. A method's or a route's name and an operation's key in `operation_registry` identify registry entries. A tracked term's own identifier is its label.
- **node**: a named tree itself, its root, or an entry at any depth below it, either a field or a group (II.6).
- **normalized and unnormalized**: a law is normalized when it claims a capability defined only for a probability law, such as sampling or a normalized density, and unnormalized otherwise (III.8). Step 3 of a call is argument normalization (V.4).
- **notation**: how a law, a kernel, or a function displays, which is the whole rendering of its expression: its label followed by its signature in parentheses, as `prior(mu)`, `glm(y | beta)`, or `predict(x, y)`, which the property `notation` and `str()` return (II.4). An unlabeled product's notation joins its factors' notations with `·` unless the product holds fixed paths (IV.2).
- **operation**: a `Function` that fronts many implementations with one call, such as `mean(d)`, by adding operand roles, a result rule with its applicability checks, and a set of routes (VI.0).
- **outcome**: an evaluation point's result, its output or a failure that says why it has none (V.9).
- **output**: a name under which other terms receive one value from a term, which a merge matches to an input of the same name (II.2, IV.2). The outputs follow from the term's value type: a record type has one per child, a base spec has one, the **output at the root**, whose name the output spec states, and a law or kernel type as a value has none.
- **output spec**: an `OutputSpec`, which describes a term's outputs by its value type, `term_spec`, and the name of an output at the root, in the keyword form `OutputSpec(y=...)` over a base spec or the positional form `OutputSpec(T)` over any other; a function's `output_spec`, a law's `event_spec`, and a batch's `element_spec` (II.2). A spec left unspecified, `None`, is completed by `with_term_spec`.
- **path**: the names from the root of a named tree down to one of its nodes, joined by `/`, as `g/b` (II.6); a term's paths are those of its value type, and they continue into the term a field holds. The occurrence path of a random event is its call's position in the workflow (V.8).
- **point index**: an evaluation point's position within its set of evaluation points (V.9).
- **probe**: a trace made only to learn whether a callable traces, run at most once per `Function` and input structure (V.13).
- **probe result**: the stored outcome of a probe, kept per `Function` and input structure and shared by the copies that `using` makes (V.13).
- **program store**: the compiled JAX programs that an owner, such as a kernel a joint maps, keeps for the bodies it runs, each compiled once for each set of argument shapes and dtypes and each value of the call's static settings, with what their traces noted on the running call (V.9).
- **provenance**: the record of how a tracked term was produced, which holds its operation, its tracked parents, its resolved controls, and its plain inputs (II.4).
- **pushforward**: the law `f♯d` of `f(X)` for `X ~ d`, which the operation `pushforward(d, f)` computes by passing the map draws of `d` whatever its annotation; the direct call `f(d)` is a pushforward when the map's parameter takes draws (V.5, VI.1).
- **raw form**: the representation layer's presentation of a value, detached from the workflow, such as a backing array or a wrapped callable; it is a term's `raw` (Part I, II.4).
- **record of the outputs**: the record with one field per output of a term, holding its output value, which reads a term's outputs as one value; it exists exactly when the term has outputs, and a term whose value type is a distribution's or a conditional distribution's spec has none and must be wrapped under a name (II.2).
- **registry**: an extensible set of registered entries, discoverable through the registry catalog; a dispatch registry selects one of its methods for a call (II.7).
- **requester**: the code that asks for a set of evaluation points, such as a lift, a joint, or an inference method, and that decides what a failed evaluation point means (V.9).
- **route**: one implementation of an operation, with a `check`, an `execute`, and an exact flag, from a structural, capability, registry, or fallback source (VI.0).
- **schema**: a `RecordSpec` read as the structure of one structured value, such as a draw or a stored datum (III.5).
- **selection order**: the ranking by which a registry tries the matching methods for a call, the same in every registry (II.7).
- **set of evaluation points**: the evaluation points of one callable that one request asks for, held in one `EvaluationPoints`, which an executor runs (V.9).
- **signature**: what a law, a kernel, or a function is over, read from its input and output specs: a law's outputs, a kernel's outputs with `|` and its inputs, or a function's parameters, and a law or a kernel lists after `;` the paths it holds fixed at given values (II.4). A function's Python call signature keeps its Python name, `Function.signature`, which binding reads (III.3, V.3).
- **standalone marginal**: the marginal `marginal(d, paths)` returns: a new law, drawn independently of `d` and of its coupled marginals, computed once, and exact when `d` knows its marginal at `paths`, an empirical approximation from draws of `d` otherwise (VI.8).
- **static setting**: a `config` setting that can change what a compiled JAX program computes, because ProbPipe code reads it while JAX traces the program, so a JAX program ProbPipe opens is compiled for its value; `sampling.max_evaluations` and `sampling.num_samples` (V.2, V.9).
- **sweep**: the lift of a function over a batch argument, which maps the function over the batch's elements; the elementwise sweep is its generic evaluation rule (V.5, V.7).
- **term spec**: the typing information of a term, held in a `TermSpec`; it validates whether an object satisfies it and carries the symbolic-dimension protocol (II.1).
- **traceable and opaque**: a callable is traceable for its input shapes when `jax.jit` and `jax.vmap` both succeed on it, and opaque otherwise, such as one that runs a subprocess; traceable says nothing about gradients (V.13).
- **tracked term**: a value, function, distribution, or batch in its tracked form, which carries a label, a spec, a provenance, and annotations through the `TrackedTerm` mixin (II.4).
- **unify**: two specs unify when the rules of II.1 give a spec, their unification, that conforms to both; it is the check where one value must satisfy two declared specs, as an input that both operands of a merge take (II.1, IV.2).
- **unresolved**: the outcome of a feasibility check whose required declarations are not yet available (II.7).
- **value type**: the spec from which a term's paths and outputs are read: a law's or kernel's `event_spec.term_spec`, a function's `output_spec.term_spec`, a batch's `element_spec.term_spec`, or a record's or base value's own `spec`; it has no attribute of its own (II.2).
- **view**: a tracked term that refers into a container's storage instead of copying it, such as a record field, a batch element, a sub-batch, or a marginal batch `b[path]` (B4, II.4). A law's `d[paths]` is a coupled marginal, not a view.

## Canonical names

The canonical name of a recurring concept is the name that its parameters, attributes, and variables take in the code.

| Concept | Name |
|---|---|
| a draw or a value of a term | `value` |
| a 1-D numeric serialization | `vec` |
| the batch dimensions | `batch_shape` |
| the batch axes tiled into levels | `axis_groups`, as reported; construction takes `axes_per_level` |
| one name per level of a batch | `level_names` |
| a batch's level names mapped to their axis sizes | `levels`, which `BatchSpec` reports and its construction takes |
| the output spec of one element of a batch, whose `term_spec` is the element's spec | `element_spec` |
| the renames a `rename` call gives, by position and as keywords | `renames`, `keyword_renames` |
| what a batch is declared over, given to `BatchSpec` or a batch constructor: the elements' spec, or for `BatchSpec` also an output spec | `element` |
| the objects a batch is built from | `elements` |
| the independent-draw shape prefix of `sample` | `sample_shape` |
| the maximum number of evaluations of a function lifted over laws | `max_evaluations` |
| the number of Monte Carlo draws of a summary or an approximate conversion | `num_samples` |
| the set of evaluation points an executor runs | `evaluation_points` |
| the options of an executor, kept apart from the arguments | `executor_options` |
| the number of evaluation points in one chunk | `chunk_size` |
| the acceptance probability that an MCMC method's step-size adaptation targets, a method option | `target_accept_prob` |
| the maximum depth of a NUTS tree, a method option | `max_tree_depth` |
| a distribution's event spec, an `OutputSpec` | `event_spec` |
| a PRNG key | `key` |
| a function from which `distribution` builds a law | the operation it realizes: `sample`, `log_prob`, or `unnormalized_log_prob` |
| a tracked term's identity for a reader: the `label=` keyword of a constructor | `label` |
| the name of an output at the root: the first argument of a family's constructor, and a keyword of `EmpiricalDistribution`, `distribution`, `factor`, and `OutputSpec.default` | `output_name` |
| a path within a tree | `path`, and `paths` for several |
| the name assigned to a field | `field_name` |
| the attributes an immutable class keeps out of its state round-trip, such as a memo | `_transient_state` |
| the attributes an immutable class restores into a container of their own, such as a store written in place | `_decoupled_state` |
| whether executing a planned step claims at least one workflow-owned random event, so that its result depends on the workflow seed (V.8) | `claims_random_events` |
| the one line the registry catalog shows for a registry or an entry, such as a dispatch method (II.7) | `description` |
