# Glossary

The terms of art of the reference, in alphabetical order. Each entry defines its term and names the section that owns its rules, where those rules are stated.

- **argument normalization**: step 3 of a call, which wraps each argument into its kind, plans any conversion it needs, and admits the result against its parameter (V.4).
- **atom**: one point of an empirical law's finite support, held with its weight in the event type's batch form (VII.2).
- **batch**: an indexed collection of separate objects of one element type, with its axes grouped into named levels (II.5). A kind's batch form is its batch class, such as `RecordBatch` for records (II.1).
- **broadcast**: the lift of a function over a distribution argument, whose result is the pushforward law of the function's output (V.5).
- **capability**: an underscore implementation of an operation, such as `_sample` or `_inverse`, that a class claims for its instances by implementing the capability's protocol (III.8, III.3).
- **component**: a named part of the interface that an `OutputSpec` declares for one produced term, which is the whole term under one name or each field of an exposed record (II.2). The components of a `MixtureDistribution` are its component laws (VII.3).
- **conversion**: a change of a distribution's representation that preserves its law, up to the recorded fidelity, and its event declaration; a converter is the registered method that carries one out (IV.3).
- **declaration**: static information an object states before any computation, such as a spec, a claimed capability, or a method's exactness (II.7). In particular, an `InputSpec` declares a map-like kind's input slots, an `OutputSpec` declares the components of one produced term, and a law's event declaration is the `OutputSpec` of a draw (II.2, III.7).
- **derived operation**: an operation defined by an identity over other operations, such as `prob = exp ∘ log_prob` or `expectation(d, f) = mean(evaluate(f, d))` (VI.0, VI.4, VI.5).
- **element**: one object of a batch, at one position of its batch axes (II.5). The scalar values of an array are its entries.
- **evaluation rule**: a method of the evaluation-rule registry, which realizes a map applied to a distribution or a batch, such as a closed-form rule, an integration rule, or the sampling lift (V.7).
- **event**: one value of a distribution's space, such as a draw or a stored datum, typed by the law's event declaration (III.5, III.7). A workflow-owned random event is one draw's occurrence in a workflow scope (V.8).
- **exact and approximate**: a method or route is exact when its returned representation denotes the requested mathematical result, and approximate when it returns a stand-in for that result (II.7).
- **factor**: a constituent `Distribution` or `ConditionalDistribution` that a factored joint was built from (IV.1); `factor(d, component_name)` returns the one that produces a component (VI.8).
- **field**: one named leaf of a named tree, such as a record's leaf value or a leaf spec of a draw's schema, addressed by its key (II.6). An interior path names a group of fields (IV.1).
- **floor**: a generic fallback with the lowest rank on its stated domain, such as the sampling lift on samplable distributions and the elementwise sweep on batches (V.7).
- **given slot**: a named input of a `ConditionalDistribution`, declared in its `given_spec`, which `condition_on` binds (III.9, VI.6).
- **guard**: a predicate on a capability's instance and a call's arguments that decides whether the capability supports the call, as a `LinOp`'s inverse guard rejects a non-square operator (III.8).
- **key**: a path that addresses a field, so the keys of a named tree are its field paths (II.6). A PRNG key is the random-number key that a workflow scope derives for each draw (V.8).
- **kind**: a sort of tracked term, such as `NumericArray`, `Record`, or `Distribution`, identified by the class of its term spec; the kind table records each kind's tracked class and batch form (II.1).
- **label**: the string that identifies a tracked term to a reader, its `label` attribute, which has no mathematical meaning (C5). II.4 owns how a label is set and derived, and III.7 and III.8 own the labels of a field view and of a marginal.
- **level**: a named group of contiguous batch axes; a batch's `axis_groups` tiles its batch shape into levels, outermost first (II.5).
- **lift**: the application of a function to a distribution or a batch where it expects a value, which is a broadcast for a distribution and a sweep for a batch (C4, V.5). The sampling lift is the generic evaluation rule that realizes a broadcast from draws (V.7).
- **marginal**: the law of the field or field group at a path of a distribution, detached from that distribution; `marginal(d, path)` returns it, and `_marginal` is its capability (VI.8, III.8).
- **method**: a dispatch method, which is one named implementation that a dispatch registry selects by type and feasibility (II.7). Elsewhere the word keeps its Python sense of a function defined on a class, as `with_name` is.
- **name**: the identifier of a matched part or a registry entry: a component name, a field name, or a level name (II.2, II.5, II.6), a method's or a route's name, and an operation's key in `operation_registry`. A tracked term's own identifier is its label.
- **normalized and unnormalized**: a law is normalized when it claims a capability defined only for a probability law, such as sampling or a normalized density, and unnormalized otherwise (III.8). Step 3 of a call is argument normalization (V.4).
- **operation**: a `Function` that fronts many implementations with one call, such as `mean(d)`, by adding operand roles, a result rule with its applicability conditions, and a set of routes (VI.0).
- **optional slot**: an input slot that a binding may omit, which then takes its default; a parameter with a default of a kernel's function is one, and in a joint a factor that produces a component of its name meets it (II.2, IV.2, IV.4).
- **packaging**: whether a declaration exposes a record's fields as its components or names one whole term, which the two forms of `OutputSpec` decide (II.2).
- **path**: a `/`-joined sequence of names that addresses one node of a named tree, either a field or an interior node (II.6); II.2 fixes the paths of a declaration. The occurrence path of a random event is its call's position in the workflow (V.8).
- **provenance**: the record of how a tracked term was produced, which holds its operation, its tracked parents, its resolved controls, and its plain inputs (II.4).
- **raw form**: the representation layer's presentation of a value, detached from the workflow, such as a backing array or a wrapped callable; `raw()` returns it (Part I, II.4).
- **registry**: an extensible set of registered entries, discoverable through the registry catalog; a dispatch registry selects one of its methods for a call (II.7).
- **route**: one implementation of an operation, with a `check`, an `execute`, and an exact flag, from a structural, capability, registry, or fallback source (VI.0).
- **schema**: a `RecordSpec` read as the structure of one structured value, such as a draw or a stored datum (III.5).
- **selection**: the part of an object that one indexing call picks out, either positions of a batch, from which the view takes its name (II.5), or several paths of a law, which one field view addresses jointly (III.7).
- **selection order**: the ranking by which a registry tries the matching methods for a call, the same in every registry (II.7).
- **sweep**: the lift of a function over a batch argument, which maps the function over the batch's elements; the elementwise sweep is its generic evaluation rule (V.5, V.7).
- **term spec**: the typing information of a term, held in a `TermSpec`; it validates whether an object satisfies it and carries the symbolic-dimension protocol (II.1).
- **tracked term**: a value, function, distribution, or batch in its tracked form, which carries a name, a spec, a provenance, and annotations through the `TrackedTerm` mixin (II.4).
- **type hole**: a component of an `OutputSpec` whose term spec is pending, written `None`, which a producer fills with `with_spec` (II.2). `None` means only a pending type, so an opaque field is declared as `OpaqueSpec()` (III.2, III.5).
- **unresolved**: the outcome of a feasibility check whose required declarations are not yet available (II.7).
- **view**: a tracked term that refers into its source, such as a record field, a batch element or sub-batch, or a distribution's field view `d[path]` (B4, II.4, III.7).

## Canonical names

The canonical name of a recurring concept is the name that its parameters, attributes, and variables take in the code.

| Concept | Name |
|---|---|
| a draw or a value of a term | `value` |
| a 1-D numeric serialization | `vec` |
| the batch dimensions | `batch_shape` |
| the batch axes tiled into levels | `axis_groups`, as reported; construction takes `axes_per_level` |
| one name per level of a batch | `level_names` |
| the spec every element of a batch satisfies | `element_spec` |
| the objects a batch is built from | `elements` |
| the independent-draw shape prefix of `sample` | `sample_shape` |
| a distribution's event declaration, an `OutputSpec` | `event_spec` |
| a PRNG key | `key` |
| a function from which `distribution` builds a law | the operation it realizes: `sample`, `log_prob`, or `unnormalized_log_prob` |
| a tracked term's identity, the required first argument of `Record` and of a distribution | `label` |
| a field key within a tree, or the name assigned to a field | `field_name` or `key` |
| the attributes an immutable class keeps out of its state round-trip, such as a memo | `_transient_state` |
| the attributes an immutable class restores into a container of their own, such as a store written in place | `_decoupled_state` |
