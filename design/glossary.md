# Glossary

The terms of art of the reference, in alphabetical order. Each entry defines its term and names the section that owns its rules, where those rules are stated.

- **alias**: a label the caller gives explicitly, through a constructor's `label=`, a function's `output_label=`, or `with_label`, which replaces a term's description for display and supplies no component, field, level, or identity (II.4).
- **argument normalization**: step 3 of a call, which wraps each argument into its kind, plans any conversion it needs, and admits the result against its parameter (V.4).
- **atom**: one point of an empirical law's finite support, held with its weight in the event type's batch form (VII.2).
- **base spec**: a `NumericArraySpec` or an `OpaqueSpec`: the type of a term with no named parts, which is a base value; the keyword form of `OutputSpec` names only a base spec (II.2).
- **batch**: an indexed collection of separate objects of one element type, with its axes grouped into named levels (II.5). A kind's batch form is its batch class, such as `RecordBatch` for records (II.1).
- **broadcast**: the lift of a function over a distribution argument, whose result is the pushforward law of the function's output (V.5).
- **capability**: an underscore implementation of an operation, such as `_sample` or `_inverse`, that a class claims for its instances by implementing the capability's protocol (III.8, III.3).
- **column**: the column of a batch at a path is the batch, over the same levels, of each element's node at that path; a batch of records offers its columns as its components (II.5).
- **component**: a name under which a term offers what it produces, declared by its `OutputSpec`: the one component at the root of the keyword form, which names the whole produced term, or each child of a positional record, each column of a batch, or each component function of a function; a produced law or kernel offers none (II.2). `t.component(c)` reads the term at a component (II.6). The components of a `MixtureDistribution` are its component laws (VII.3).
- **conforms**: a term spec `A` conforms to a term spec `B` when every value of type `A` is one that `B` admits, in one dimension scope; a fed component conforms to the slot it feeds, and completion checks that a produced value conforms (II.1, IV.2).
- **conversion**: a change of a distribution's representation that preserves its law, up to the recorded fidelity, and its event declaration; a converter is the registered method that carries one out (IV.3).
- **declaration**: static information an object states before any computation, such as a spec, a claimed capability, or a method's exactness (II.7). In particular, an `InputSpec` declares a term's slots, an `OutputSpec` declares a term's produced term and its components, and a law's event declaration is the `OutputSpec` of a draw (II.2, III.7).
- **derived operation**: an operation defined by an identity over other operations, such as `prob = exp ∘ log_prob` or `expectation(d, f) = mean(pushforward(d, f))` (VI.0, VI.4, VI.5).
- **description**: the expression a term derives at construction, from its declared fields, its callable's name, or its operands, which its label renders when the caller gives no alias. A term that JAX rebuilds from its leaves has none and displays `<no description>` (II.4). A registry and each of its entries carry a different `description`, the one line the registry catalog shows (II.7).
- **element**: one object of a batch, at one position of its batch axes; a batch's element spec is the output declaration of one element, and `element` is the default component of the elements of a batch built from raw base values (II.5). The scalar values of an array are its entries.
- **evaluation rule**: a method of the evaluation-rule registry, which realizes a map applied to a distribution or a batch, such as a closed-form rule, an integration rule, or the sampling lift (V.7).
- **event**: one value of a distribution's space, such as a draw or a stored datum, typed by the law's event declaration (III.5, III.7). A workflow-owned random event is one draw's occurrence in a workflow scope (V.8).
- **exact and approximate**: a method or route is exact when its returned representation denotes the requested mathematical result, and approximate when it returns a stand-in for that result (II.7).
- **expression**: the immutable tree a tracked term carries, which states what the term is and which its label and its notation render (II.4). An operation builds its result's expression from its operands' expressions, and no operation reads one.
- **factor**: a constituent `Distribution` or `ConditionalDistribution` that a factored joint was built from (IV.1); `factor(d, component_name)` returns the one that offers a component (VI.8). A grouped sub-joint is the one factor that a rename gathering components under one node makes of their factors (IV.1).
- **feeds**: when `a & b` is formed, a component of `b` feeds the slot of the same name of `a`, or the unfed slot of that name of one of `a`'s factors, so the value at the component is the value the slot receives (IV.2).
- **field**: a node of a record, named through records only, that is not itself a record, such as a record's leaf value or a leaf spec of a draw's schema; a field that holds a law, a kernel, a function, or a batch is a held field, whose held term has paths of its own, and `keys()` lists a record's fields (II.6).
- **floor**: a generic fallback with the lowest rank on its stated domain, such as the sampling lift on samplable distributions and the elementwise sweep on batches (V.7).
- **given slot**: a slot of a `ConditionalDistribution`, declared in its `given_spec`, which `condition_on` binds (III.9, VI.6).
- **group**: a node of a record, other than the root, that is itself a record; `t["g"]` is the group's contents, and `t[["g"]]` the record that holds it under its name (II.6).
- **guard**: a predicate on a capability's instance and a call's arguments that decides whether the capability supports the call, as a `LinOp`'s inverse guard rejects a non-square operator (III.8).
- **key**: a PRNG key, the random-number key that a workflow scope derives for each draw (V.8). The word keeps its Python sense of a mapping's keys, as in `keys()`, which lists a record's fields, and the keys of a rename mapping; a place in a tree is named by its path (II.6).
- **kind**: a sort of tracked term, such as `NumericArray`, `Record`, or `Distribution`, identified by the class of its term spec; the kind table records each kind's tracked class and batch form (II.1).
- **label**: the string that names a tracked term for a reader, its `label` attribute, which has no mathematical meaning and on which no computation depends (C5): an operation reads its operands' labels only to label its result. It is the name part of the rendering of the term's expression. Constructors take it as keyword `label=`, requiring it for raw values without semantic structure and otherwise deriving it from fields, callable names, or described operands. II.4 owns how a label is set, defaulted, derived, and grouped, and III.7 and VI.8 own the labels of a view and of a marginal. A law, a kernel, or a function displays its label in its notation.
- **level**: a named group of contiguous batch axes; a batch's `axis_groups` tiles its batch shape into levels, outermost first (II.5).
- **lift**: the application of a function to a distribution or a batch where it expects a value, which is a broadcast for a distribution and a sweep for a batch (C4, V.5). The sampling lift is the generic evaluation rule that realizes a broadcast from draws (V.7).
- **marginal**: the law of the node at a path, or of the nodes at a list of paths, of a distribution, detached from that distribution; `marginal(d, paths)` returns it for a path or a list of paths, and `_marginal` is its capability (VI.8, III.8).
- **merge**: what `&` does, for every kind: it puts the components of both operands side by side, makes a slot that both operands take one slot, and lets each component of the right operand feed the slot of the same name of the left operand, as `lik & prior` builds a joint law and `rec1 & rec2` a record (IV.2). Composition keeps its meaning of `f ∘ g` and of a product of linear operators `A B`.
- **method**: a dispatch method, which is one named implementation that a dispatch registry selects by type and feasibility (II.7). Elsewhere the word keeps its Python sense of a function defined on a class, as `with_label` is.
- **name**: the identifier of a matched part or a registry entry: the name of a slot, a component, a field, or a level (II.2, II.5, II.6), a method's or a route's name, and an operation's key in `operation_registry`. A tracked term's own identifier is its label.
- **node**: a place in a term that a path names; the root is the term itself, which has no path (II.6).
- **normalized and unnormalized**: a law is normalized when it claims a capability defined only for a probability law, such as sampling or a normalized density, and unnormalized otherwise (III.8). Step 3 of a call is argument normalization (V.4).
- **notation**: how a law, a kernel, or a function displays, which is the whole rendering of its expression: its label followed by its signature in parentheses, as `prior(mu)`, `glm(y | beta)`, or `predict(x, y)`, which the property `notation` and `str()` return (II.4). An unlabeled product's notation joins its factors' notations with `·` unless the product holds fixed paths (IV.2).
- **operation**: a `Function` that fronts many implementations with one call, such as `mean(d)`, by adding operand roles, a result rule with its applicability conditions, and a set of routes (VI.0).
- **optional slot**: a slot that a binding may omit, which then takes the default of the term that owns it; a parameter with a default of a kernel's function or of a `Function` is one, and in a joint a factor that offers a component of its name feeds it (II.2, IV.2, IV.4).
- **path**: the `/`-joined names from the top of a term down to one node, always a `str`; a path goes through a held law, kernel, function, or batch into its own paths, and a term's paths are its output declaration's (II.6, II.2). `[]` takes a path or a list of paths. The occurrence path of a random event is its call's position in the workflow (V.8).
- **produced term**: what one evaluation of a term yields: a value, a record, or a batch itself, a draw of a law, a draw of a kernel at each value of its slots, or a function's return value at each value of its slots (II.2).
- **provenance**: the record of how a tracked term was produced, which holds its operation, its tracked parents, its resolved controls, and its plain inputs (II.4).
- **pushforward**: the law `f♯d` of `f(X)` for `X ~ d`, which the operation `pushforward(d, f)` computes by passing the map draws of `d` whatever its annotation; the direct call `f(d)` is a pushforward when the map's parameter takes draws (V.5, VI.1).
- **raw form**: the representation layer's presentation of a value, detached from the workflow, such as a backing array or a wrapped callable; it is a term's `raw` (Part I, II.4).
- **registry**: an extensible set of registered entries, discoverable through the registry catalog; a dispatch registry selects one of its methods for a call (II.7).
- **route**: one implementation of an operation, with a `check`, an `execute`, and an exact flag, from a structural, capability, registry, or fallback source (VI.0).
- **schema**: a `RecordSpec` read as the structure of one structured value, such as a draw or a stored datum (III.5).
- **selection order**: the ranking by which a registry tries the matching methods for a call, the same in every registry (II.7).
- **signature**: what a law, a kernel, or a function is over, read from its declaration: a law's components, a kernel's components with `|` and its slots, or a function's parameters, and a law or a kernel lists after `;` the paths it holds fixed at given values (II.4). A function's Python call signature keeps its Python name, `Function.signature`, which binding reads (III.3, V.3).
- **slot**: a named place where a term receives one value, with a term spec and a flag that makes it required or optional; a `Function`'s slots are its parameters, a `ConditionalDistribution`'s are what it conditions on, and an `InputSpec` declares them (II.2).
- **spec with parts**: a `RecordSpec` with at least one child, a `DistributionSpec`, a `ConditionalDistributionSpec`, a `FunctionSpec`, or a `BatchSpec`, which an output declaration offers by its components in the positional form (II.2).
- **sweep**: the lift of a function over a batch argument, which maps the function over the batch's elements; the elementwise sweep is its generic evaluation rule (V.5, V.7).
- **term spec**: the typing information of a term, held in a `TermSpec`; it validates whether an object satisfies it and carries the symbolic-dimension protocol (II.1).
- **tracked term**: a value, function, distribution, or batch in its tracked form, which carries a label, a spec, a provenance, and annotations through the `TrackedTerm` mixin (II.4).
- **type hole**: the keyword form of an `OutputSpec` with its term spec pending, written `None`, which a producer fills with `with_term_spec` (II.2). `None` means only a pending type, so an opaque field is declared as `OpaqueSpec()` (III.2, III.5).
- **unfed**: said of a slot that a joint's factors take and no factor feeds; `unfed(t)` is the input declaration of an operand's unfed slots, which a merge reads (IV.2).
- **unify**: two term specs unify when their unification, the spec of the values both admit, exists in one dimension scope; a slot that two operands of a merge share carries the unification of its specs (II.1, IV.2).
- **unresolved**: the outcome of a feasibility check whose required declarations are not yet available (II.7).
- **view**: a tracked term that refers into its source, such as a record field, a batch element or sub-batch, or a distribution's view `d[paths]`, a `MarginalView` (B4, II.4, III.7).

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
| the output declaration of one element of a batch, whose `term_spec` is the elements' type | `element_spec` |
| the objects a batch is built from | `elements` |
| the independent-draw shape prefix of `sample` | `sample_shape` |
| a distribution's event declaration, an `OutputSpec` | `event_spec` |
| a PRNG key | `key` |
| a function from which `distribution` builds a law | the operation it realizes: `sample`, `log_prob`, or `unnormalized_log_prob` |
| a tracked term's identity for a reader: the `label=` keyword of a constructor | `label` |
| the component at the root of a law's event, the first argument of a family's constructor and a keyword of `EmpiricalDistribution` and `distribution` | `component` |
| a path within a tree | `path`, and `paths` for several |
| the name assigned to a field | `field_name` |
| the attributes an immutable class keeps out of its state round-trip, such as a memo | `_transient_state` |
| the attributes an immutable class restores into a container of their own, such as a store written in place | `_decoupled_state` |
| the one line the registry catalog shows for a registry or an entry, such as a dispatch method (II.7) | `description` |
