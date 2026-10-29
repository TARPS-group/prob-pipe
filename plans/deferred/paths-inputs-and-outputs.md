# Plan: one contract for paths, inputs, and outputs

> **Status: proposed; not approved.**
> Written on 2026-10-28 against `main` at `086b4ae4`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Paths, inputs, and outputs".
> It revisits the concepts of `plans/declarations-and-addressing.md` (Part I §§ 2 to 4, Part II §§ 9 to 12) and replaces that plan's unmerged steps S4b and G1a to G5; § 7 lists the edits that plan and the other deferred plans need.
> Once approved, its rules move into `design/` (II.1, II.2, II.5, II.6, III.3, III.5, III.7, III.9, III.10, and the glossary) and its steps into the plans of the work after the workshop.

**Settled with the maintainer on 2026-10-28:**

1. A record's paths are the paths of its own tree and end at its leaves; any other object has paths exactly when its value spec is a record (§ 4.1, § 4.2). This reverses the decision of 2026-10-16 that paths go through an object a record holds.
2. `in` is part of the path interface (§ 4.1).
3. A list of paths keeps each node at its full path (§ 4.1).
4. `select` is not part of the path interface, and this plan leaves it unchanged (§ 4.1).

**Decisions still open** (each explained where it arises, and gathered in § 9):

1. Two cases for outputs instead of II.2's five, so that every named object has outputs (§ 4.3).
2. The output name is stored on the object, not inside its spec; `OutputSpec` is removed, and an object's output spec is a `RecordSpec` (§ 4.3).
3. One public conformance relation, `conforms_to`, with inputs compared contravariantly (§ 4.5).
4. The names `value_spec`, `return_spec`, `output_name`, `outputs`, and `output_spec` (§ 5).

---

## 1. Why this plan exists

ProbPipe connects objects by name.
A *path* such as `g/b` picks out a part of an object, an *output* names a part that another object can receive, and an *input* names what an object receives.
Three sources state these ideas, and they disagree: the maintainer's conceptual model, `design/` (II.1, II.2, II.6), and the code.
The code also implements each idea once per kind:

```python
d = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)
r = Record({"x": 3.0, "g": d})

d["mu"]                      # a CoupledMarginal, read through OutputSpec and _read_paths
r["g"]                       # the law d, read through NamedTree
r["g/mu"]                    # KeyError: unknown path 'g/mu'; design II.6 says it is d["mu"]
Normal("x", 0.0, 1.0)["x"]   # the law itself; design II.2 says TypeError
```

The goal is one definition of each idea, one implementation that every kind shares, and a precise statement of when one spec conforms to another.

## 2. The current state

### 2.1 The conceptual model

The maintainer's note "Core Conceptual Model" (Obsidian vault, `work/projects/probpipe/design/`) builds the ideas in three layers.

1. **Path indexing is defined for records.** A record is a tree with a name at every node, and `R[p]` is the node at the path `p`. The *path indexing interface* is `[]`, `in`, `without`, and `rename`. Its rules:
   - indexing an object without paths is a `TypeError`, and a missing path is a `KeyError`;
   - `x[p1/p2] == x[p1][p2]`;
   - `x[p, q] == x[[p, q]]`, and both, like `without`, return an object of the kind of `x`;
   - `paths(x[~Q]) == paths(x.without(Q))` up to order, so `without` and list indexing need one source of truth;
   - `S(x)[P] == S(x[P])`, where `S(x)` is the static structure of `x`;
   - `p in x` exactly when `x[p]` exists.
2. **Other kinds inherit the interface.** A law, a kernel, a function, and a batch support paths when their event, output, or element is a record, and `x[p]` is the *marginal* of that kind.
3. **Outputs are defined from paths.** An object with paths has the outputs `top(x)`, and the output `c` is `x[c]`. An object without paths is one output, the whole object, under a name assigned to it. That name gives no path: for an array `alpha` named `a`, `alpha["a"]` is a `TypeError`, while `outputs(alpha)["a"]` is `alpha`.

Inputs are named slots, each with a structure and an optional default.
*Binding* connects an output to the input of the same name when the output's structure conforms to the input's.
The note lists three gaps: the definition of conformance, the definition of the dimension scope, and its definition of `[[ ]]`, which is to keep full paths as its examples already do.

### 2.2 What `design/` states

`design/` follows `plans/declarations-and-addressing.md` Part I, and it agrees with the conceptual model on most points:

| Point | Conceptual model | `design/` |
|---|---|---|
| Paths of a record | its tree | its tree, continued into the objects its fields hold (II.6) |
| Which other objects have paths | event, output, or element is a record | the value type has paths, at any depth (II.2) |
| `x[p1/p2] == x[p1][p2]` | yes | yes |
| `[]` at a list, and `without`, keep the kind | yes | yes |
| `R[["b/c"]]` | `{b: {c: …}}`, by full path | `{b: {c: …}}`, by full path (II.6) |
| Outputs of an object with a record value | its top-level names | its top-level names |
| Outputs of an object without a record value | one, under a stated name | one for a base value; the outputs of the element or return value for a batch or function value; none for a law value (II.2, cases 2 to 5) |
| Where a stated name is stored | not said | inside the spec, in `OutputSpec` (II.2) |
| Conformance | open | defined by kind (II.1) |

Three differences matter.

**The design continues paths past a record's leaves.**
For a law `d` over `{mu, sigma}`, II.6 gives the record `{x: 3.0, g: d}` the paths `g/mu` and `g/sigma`.
II.2 gives paths to every object whose value type holds a record at any depth, such as a batch of laws over a record.
The conceptual model ends a record's paths at its leaves, and gives other kinds paths only when their event, output, or element is a record.

**The design states outputs in more cases.**
II.2 reads outputs from the *value type*: the event type of a law or kernel, the return type of a function, the element type of a batch, and a value's own type.
It then needs five cases and two exceptions, one of which leaves a function that returns a law without outputs.
The conceptual model needs two cases, and every named object has outputs.

**The design stores the stated name in the spec.**
`OutputSpec` pairs a term spec with an optional name, and `DistributionSpec.event_spec`, `FunctionSpec.output_spec`, and `BatchSpec.element_spec` each hold one.
II.2 § Equality then needs three rules to keep the name out of a term's equality, hash, fingerprint, and JAX structure while the spec's own `==` compares it.

### 2.3 What the code does

The overhaul merged steps S1, S2a, S2b-out, S2b-in, S3, and S4a of the declarations plan; S4b and G1a to G5 did not merge.

**Path indexing is implemented once per kind.**

| Kind | `[p]` | `[[ps]]` | `in` | `without` | `rename` | Implemented by |
|---|---|---|---|---|---|---|
| `Record`, `RecordSpec` | ✓ | ✓ | ✓ | ✓ | ✓ | `NamedTree` |
| `RecordBatch` | ✓ | ✓ | ✓ | ✓ | ✓ | `_at_paths` over `event_template` |
| `OutputSpec` | ✓ | ✓ | ✓ | ✓ | ✓ | `_read_paths` and `_path_record` |
| `Distribution` | ✓ | ✓ | ✓ | – | ✓ | `_read_event_paths` and `CoupledMarginal` |
| `ConditionalDistribution` | – | – | – | – | ✓ | its own `_check_rename_source` |
| `Function`, and batches of non-records | – | – | – | – | – | – |
| `DistributionSpec`, `FunctionSpec`, `BatchSpec` | – | – | – | – | – | – |

The shared parts are the reader of the argument, `_as_paths` in `core/_shapes.py`, and the separator `_PATH_SEP` in `core/named_tree.py`.
Each kind repeats the rest: whether a path exists, the spec at a path, canonical order, the error for an object with no paths, and the cut to a list.
`distributions/_views.py` alone holds more than a dozen helpers that restate `NamedTree` on an `OutputSpec` (Appendix A).
Every kind ends a record's paths at its leaves, as the conceptual model does:

```python
r = Record({"x": 3.0, "g": d})
r.spec["g/mu"]   # KeyError: unknown path 'g/mu' in this RecordSpec; available paths: ['x', 'g']
```

**The name stored in the spec causes most special cases.**

```python
x, y = Normal("x", 0.0, 1.0), Normal("y", 0.0, 1.0)
x.spec              # DistributionSpec(event_spec=OutputSpec(x=NumericArraySpec(shape=(), ...)))
x.spec == y.spec    # False, and the hashes differ
x.spec.is_valid(y)  # False: conformance compares the names, and II.1 says it ignores them
x["x"] is x         # True: the name is read as a path, and II.2 says TypeError
RecordSpec(G=x.spec)  # holds the name "x" inside a field named G
```

Because the name is in the spec, `OutputSpec` must know every kind of spec it can wrap.
`_exposed_kind`, `_EXPOSED_SPECS`, `_VALUES_WITHOUT_RECORD`, `_ROOT_NAMES`, `_renamed_exposed`, and `_unnamed_exposed` each dispatch on the class of the wrapped spec.
Every read of a law's event takes an extra step, `event_spec.term_spec`, and `.term_spec` appears 271 times in `probpipe/`.
The helpers that read the form, `_root_output_name`, `_has_record_outputs`, `_output_record`, and `_exposed_kind`, have 130 further references.
The deferred fingerprint bug ("the fingerprint of a `DistributionSpec` includes the name") and G1a's interim whole form have the same cause.

**Conformance has no single definition.**
`TermSpec` offers `is_valid(value)` and `bind_dims_from_spec(other)`.
The private `_bind_dims_from_spec` of each class is its conformance check, and `_unify_specs` falls back to `==` when a class returns `False`.
Unification is three more functions: `_unification` completes an output spec, `_unify_input_types` serves `InputSpec.__and__`, and `_unify_declarations` compares event specs.
Five results disagree with II.1:

```python
narrow = FunctionSpec(InputSpec(a=NumericArraySpec((), dtype=np.int32)), OutputSpec(out=()))
wide   = FunctionSpec(InputSpec(a=NumericArraySpec((), dtype=np.float32)), OutputSpec(out=()))
narrow.bind_dims_from_spec(wide)   # ValueError, although wide accepts every int32
wide.bind_dims_from_spec(narrow)   # passes, although narrow refuses a float32

OpaqueSpec(type=int).bind_dims_from_spec(OpaqueSpec(type=bool))   # ValueError, though bool subclasses int
FunctionSpec(InputSpec(a=())).bind_dims_from_spec(FunctionSpec())  # passes with unknown inputs
narrow.is_valid(len)                                               # True for any callable

InputSpec(t=NumericArraySpec((), dtype=np.float32)) & InputSpec(t=NumericArraySpec((), dtype=np.int32))
# InputSpec(t=NumericArraySpec(shape=(), dtype=float32)); II.1 unifies to int32
```

The first pair is the variance of inputs: the code compares them covariantly, and II.1 contravariantly.

### 2.4 The three questions, assessed

1. **Outputs as a wrapper spec.** The approach does not hold up. An output name says how an object connects to others, and it is not part of the object's structure. `OutputSpec` stores it inside the structure of every law, kernel, function, and batch, so each operation that reads structure strips the name again or treats it as a special case. § 4.3 computes outputs from the spec and a name stored on the object, and an output spec becomes a `RecordSpec`.
2. **One source of truth for paths.** The reader of the argument is shared. What a path means is implemented once per kind, and three kinds lack it. § 4.1 and § 4.2 define paths once, on specs, and § 5.2 gives every kind one implementation.
3. **Spec conformance.** II.1 defines conformance and unification by kind. The code has no public relation, its private one disagrees with II.1 in five places, and the conceptual model leaves the notion open. § 4.5 defines three relations on specs, and § 4.6 defines when an output feeds an input in terms of them.

## 3. Principles

1. **Structure and names are separate.** A spec states what an object is. A name says how the object connects to others, and it is stored on the object.
2. **Records are the only source of names inside an object.** Every other kind's paths and outputs are read from its value spec when that spec is a record.
3. **One rule per idea, applied to specs first.** Each operation is defined on specs, and an object's operation is checked against it: `x[p].spec == x.spec[p]`.
4. **Two relations between specs.** Conformance checks what is received against what is declared, and unification combines two declarations of one value.

## 4. The proposed design

Each layer uses only the layers before it.

### 4.1 Structure, and the paths of a record

Every object has a spec, its static structure, which the conceptual model writes `S(x)`.

> [!IMPORTANT]
> **Definition (spec, value spec).**
> 1. The *spec* of an array is its shape, dtype, and support; of an opaque value, its Python type and metadata; of a record, its tree with a spec at each leaf.
> 2. The spec of a law is its *event spec*; of a kernel, its *input spec* and event spec; of a function, its input spec and its *return spec*; of a batch, its levels and its *element spec*. Each of these is itself a spec, so specs nest.
> 3. The *value spec* of an object is the spec of one of its values: a law's or kernel's event spec, a function's return spec, a batch's element spec, and a record's or base value's own spec.
>
> A spec holds no name for the object or for its value.

The value spec is the design's "value type" (II.2), given an attribute in § 5.1.
A law over `{mu, sigma}`, its draws, its mean, and a batch of its draws share one value spec.

Records are the only objects with a name at each node.

> [!IMPORTANT]
> **Definition (paths of a record).** A record's tree has a *group* at each nested record and a *leaf* at each object of any other kind.
> A *path* is a sequence of names joined by `/`, as `g/c`, that leads from the top of the tree to a group or a leaf.
> The paths of a record are the paths of its tree, and `R[p]` is the node at `p`: the object at a leaf, or the record at a group.

A path ends at a leaf.
For a law `d` over `{mu, sigma}` held at the leaf `g`, `R["g"]` is `d`, and `R["g"]["mu"]` is `d["mu"]`.

The **path interface** is five operations:

| Operation | Result for a record `R` |
|---|---|
| `R[p]` | the node at `p`: a leaf's object, or a group as a record |
| `R[[p, q, …]]`, also written `R[p, q, …]` | the record of the chosen nodes, each at its full path, in `R`'s order |
| `p in R` | whether `p` is a path of `R` |
| `R.without(p, q, …)` | `R` with those nodes removed, every other node at its path |
| `R.rename({old: new})` | `R` with nodes renamed or moved, all at once, by II.6's rules |

> [!IMPORTANT]
> **Rules (path interface).** For an object `x` with paths, a path `p`, and a list of its paths `Q`:
> 1. one reader refuses a malformed argument (II.6's table); a missing path is a `KeyError`; any path given to an object without paths is a `TypeError`;
> 2. `x[p1/p2] == x[p1][p2]` for each path `p1/p2` of `x`;
> 3. `x[p]` is of the kind of the node at `p`, and `x[Q]` and `x.without(*Q)` are of the kind of `x`;
> 4. `x.without(*Q) == x[complement(Q)]`, where `complement(Q)` lists the outermost nodes that are neither in `Q` nor above or below a path of `Q`, so selection and removal partition the leaf paths;
> 5. `x.spec[p] == x[p].spec`, and likewise for a list, `without`, and `rename`;
> 6. removing every node is a `ValueError`;
> 7. `p in x` exactly when `x[p]` raises no `KeyError`; a malformed path, and any path given to an object without paths, raise as they do for `[]`.

By rule 4, removal is derived from selection: an implementation provides the cut to a list, and `without` is the cut to the complement.

```python
R = Record({"a": 1.0, "b": {"c": 2.0, "d": 3.0}})
R["b/c"]               # 2.0
R[["b/c"]]             # {b: {c: 2.0}}
R["a", "b/c"]          # {a: 1.0, b: {c: 2.0}}
R.without("b/d")       # {a: 1.0, b: {c: 2.0}}, the same record as R["a", "b/c"]
"b/c" in R             # True
```

`Record.select` and `RecordBatch.select` are not part of the path interface, and this plan leaves them unchanged.
Each returns a `dict` keyed by last names, for `**`-splatting.

### 4.2 Which objects have paths

> [!IMPORTANT]
> **Definition (paths of an object).** An object has paths exactly when its value spec is a record spec, and its paths are the paths of that record spec (§ 4.1).

A record's value spec is its own spec, so for a record the definition gives the paths of § 4.1.
For the other kinds it is the conceptual model's rule: a law or kernel has paths when its event is a record, a function when its return value is, and a batch when its element is.

> [!IMPORTANT]
> **Definition (marginal).** For an object `x` that is not a record, `x[p]` is the object of `x`'s kind whose spec is `x.spec[p]`, and which gives `v[p]` wherever `x` gives a value `v`:
> 1. for a law, the coupled marginal, whose draw is the node at `p` of a draw of the law;
> 2. for a kernel, the kernel whose law at each input value is the coupled marginal at `p` of the law the kernel gives there;
> 3. for a function, the function with the same inputs given by `f[p](w) == f(w)[p]`;
> 4. for a batch, the batch over the same levels of `b[i][p]`, its marginal batch.
>
> On specs the rule is one rule for all four kinds: `S[p]` is `S` with its value spec replaced by the value spec's selection at `p`.

Consequences of the two definitions:
1. A law, its draws, its mean, and a batch of its draws have one set of paths, and `sample(d[p])` is distributed as `sample(d)[p]` at every path.
2. `x.spec[p] == x[p].spec` holds for every kind, by construction (rule 5).
3. An object whose value spec is not a record has no paths, whatever its value holds. A law over an array, a batch of laws over `{mu, sigma}`, and a function that returns such a law have none; for the batch `b`, `b[i]["mu"]` reads a part of one element.
4. An object has paths exactly when its outputs are the top-level names of its value spec (§ 4.3).

### 4.3 Outputs

When two objects combine, an input of one receives what the other gives under the same name.
That name must name a whole part, so that the receiver gets one object.
The parts named must together determine the giver, so that the combination loses nothing.

A record value spec already names its children, so its children's names serve as the outputs.
Any other value spec has no names of its own, so the object is one output, under a name it states.

> [!IMPORTANT]
> **Definition (output name, outputs).**
> 1. An object whose value spec is not a record may state an *output name*, a name without `/`. The object stores it; its spec does not.
> 2. The *outputs* of an object whose value spec is a record are the top-level names of that record, and the output `c` is the part `x[c]`.
> 3. The outputs of an object whose value spec is not a record are its output name alone, and that output is `x` itself. An object without an output name has no outputs.
> 4. `x.outputs` is the record of the outputs: `{c: x[c] for each output c}` in case 2, and `{name: x}` in case 3.

Consequences of the definition:
1. **A stated name gives no path.** `Normal("x", 0, 1)["x"]` is a `TypeError`, and `Normal("x", 0, 1).outputs["x"]` is the law, as the conceptual model requires.
2. **Outputs split only at a record.** II.2 splits the outputs of an object whose value is a function or a batch (its cases 3 and 4), and gives none to one whose value is a law (its case 5). Here each of these is one output under its stated name.
3. **A function that returns a law has one output**, the function. II.2 asks for it to be wrapped in a record under a name; the output name gives the same result without changing the object's value spec.
4. **The parts that are outputs determine the object.** A record is rebuilt from its children, a function and a batch from their marginals, and a law from its coupled marginals, which share its random variable. An object whose value is a law keeps its coupled marginals together only as a whole, so it is not split.
5. **Spec equality ignores names with no rule to say so.** `Normal("x", 0, 1).spec == Normal("y", 0, 1).spec`, since neither spec holds a name, and the fingerprint and JAX structure read the spec.

| Object | Value spec | Paths | Outputs |
|---|---|---|---|
| an array with output name `a` | `()` | none | `a`, whose value is the array |
| a record `{a, b: {c}}` | `{a, b: {c}}` | `a`, `b`, `b/c` | `a`, `b` |
| a function from `{u, v}` to an array, output name `f` | `()` | none | `f`, whose value is the function |
| a function from an array to `{a, b}` | `{a, b}` | `a`, `b` | `a` and `b`, whose values are marginal functions |
| a function returning a law over `{a, b}`, output name `h` | a law spec | none | `h`, whose value is the function |
| a law over `{a, b}` | `{a, b}` | `a`, `b` | `a` and `b`, whose values are coupled marginals |
| a batch of laws over `{a, b}`, output name `fits` | a law spec | none | `fits`, whose value is the batch |
| a record `{x, g}` whose leaf `g` holds a law over `{mu}` | `{x, g}` | `x`, `g` | `x`, `g` |

A check made before anything runs, such as whether two objects can merge, needs each output's name and spec.

> [!IMPORTANT]
> **Definition (output spec).** The *output spec* of an object is a record spec with one field per output: the value spec itself when it is a record, and `{name: value spec}` otherwise.

Each field holds the spec of one value of its output.
For a law over `{mu, sigma}` the output spec is `{mu: (), sigma: ()}`, and for a function `h` that returns a law it is `{h: DistributionSpec(...)}`.
The object's kind says how a value passes: as one draw, the result of one call, or one element (§ 4.6).
An output spec is a `RecordSpec`, so the operations II.2 lists for output specs are record operations that `NamedTree` already implements: `O["g/b"]`, `O[["a"]]`, `O.without("g")`, `O.rename(...)`, and `O & P`.

**Decision 1: two cases for outputs.**
- **Recommendation.** The definition above, which is the conceptual model's.
- **Alternative.** II.2's five cases, which leave an object whose value is a law without outputs until it is wrapped.

**Decision 2: the output name is stored on the object.**
- **Recommendation.** Specs hold structure only, an object stores its output name, and the output spec is computed from both. `OutputSpec` is removed.
- **Alternative.** `OutputSpec` stays as a computed view, built by `x.output_spec` and never stored inside another spec. It keeps the class and its repr, at the cost of a class that repeats `RecordSpec`.

### 4.4 Inputs

Inputs keep II.2's definition.

> [!IMPORTANT]
> **Definition (input).** The *inputs* of a kernel or a function are named, and each has a spec and a flag for whether it is optional. An optional input has a default, which the object holds. An `InputSpec` lists the inputs without their defaults, and two input specs compare by name, not by order.

A path continues into an input whose spec is a record, so `i["g/b"]` and `i.without("g/b")` follow the path interface over the record of the inputs.
One difference applies: an input that a path cuts becomes required, since its default is one whole value.
`InputSpec(O)` for an output spec `O` declares the inputs that `O`'s outputs can feed, and it is the reading of an `InputSpec` from a `RecordSpec`, which II.2 already defines.

### 4.5 Conformance and unification

Conformance decides whether something can be given where a spec is declared.
Four places need it: a value against a declared spec, a term's spec against a declared spec (a law passed to an input that takes laws), an output against an input (§ 4.6), and a function's result against its declared return spec.

> [!IMPORTANT]
> **Definition (conforms, unify, refine).** Within one dimension scope:
> 1. a value *conforms to* a spec `B` when it has the structure `B` describes, and a spec `A` *conforms to* `B` when every value that conforms to `A` conforms to `B`;
> 2. `A` and `B` *unify* when some spec conforms to both and a most general such spec `U` exists, so that a spec conforms to `U` exactly when it conforms to both; `U` is their *unification*;
> 3. a declared spec `B` is *refined* by a spec `A` that conforms to it: the result is `B` with its dimension names bound from `A`, and with each dtype or support that `B` leaves unset taken from `A`.

The rules by kind are II.1's.
The table states each, and notes where the code departs:

| Kind | `A` conforms to `B` when | Code today |
|---|---|---|
| array | equal rank, each axis agreeing in the scope, and `A`'s dtype casts to `B`'s by same-kind casting when both set one | agrees |
| opaque | `B`'s type is unset or a superclass of `A`'s, and `B`'s metadata is unset or equal to `A`'s | requires equal types, in both directions |
| record | the same child names in the same order, child by child | agrees |
| law | `A`'s event spec conforms to `B`'s | also compares output names |
| kernel, function | the same input names and optional inputs; each of `B`'s input specs conforms to `A`'s; `A`'s event or return spec conforms to `B`'s; a side `B` leaves unknown accepts any `A`, and a side only `A` leaves unknown does not conform | inputs compared covariantly; optional inputs not compared; a side only `A` leaves unknown passes; output names compared |
| batch | the same levels, and `A`'s element spec conforms to `B`'s | agrees, apart from the element name |

A value that carries a spec, such as a law, a function, or a batch, conforms when its spec does.
So `FunctionSpec.is_valid(len)` is false, since `len` carries no spec, and `is_valid` reports conformance for every kind.
No spec holds a name, so names take no part in any of these relations.

**The dimension scope.** II.1 defines it, the conceptual model lists it as a gap, and the code passes a `dict` of bindings by hand.
A `DimensionScope` holds the size each dimension name is bound to.
It binds a name the first time the name is compared with a size, and refuses a second size for it and two unbound names at one axis.
Every relation takes one, and a caller that checks several specs together, such as the inputs of one call or the operands of one merge, passes the same scope to each.

**Decision 3: one public relation.**
- **Recommendation.** `A.conforms_to(B, scope=None)`, `unify(A, B, scope=None)`, and `B.refine(A)` are the entry points. `bind_dims_from_spec(A)` is kept as `refine`, and `bind_dims_from_value(v)` as refinement by the value's spec.
- **Alternative.** Keep today's private names, and fix only the departures in the table.

### 4.6 How an output feeds an input

The conceptual model asks what it means for an output to conform to an input, and whether an input can feed another input.

> [!IMPORTANT]
> **Definition (feeds).** In a combination of `x` and `y`, the output `c` of `x` *feeds* the input `c` of `y` when `x.output_spec[c]` conforms to `y.input_spec[c]` in the combination's dimension scope.

The output spec holds the spec of one value of each output, so the check is one rule for every kind of giver.
The giver's kind decides only how the values pass:
1. a law passes its random variable, so the combination is a joint law;
2. a function passes its result at each call;
3. a batch passes its elements, position by position;
4. a record passes its value.

An input that both operands of a merge take is the case of an input feeding an input.
The merged object takes that input once, and its spec is the unification of the two operands' specs.
A merge therefore needs two relations: conformance, from an output to an input, and unification, from an input to an input.

## 5. In code: one source of truth

### 5.1 Specs hold structure only

```python
DistributionSpec(event_spec: TermSpec)
ConditionalDistributionSpec(given_spec: InputSpec, event_spec: TermSpec)
FunctionSpec(input_spec: InputSpec | None, return_spec: TermSpec | None)
BatchSpec(element_spec: TermSpec, **levels)
```

`TermSpec.value_spec` is the value spec of § 4.1, and `TermSpec.with_value_spec(spec)` replaces it.
These two methods are all that a spec of a law, kernel, function, or batch needs for paths.
The extra step `.term_spec` goes from the 271 places that take it.

The output name moves to the object:

```python
x = Normal("x", 0.0, 1.0)
x.spec          # DistributionSpec(event_spec=NumericArraySpec(shape=(), ...))
x.output_name   # "x"
x.output_spec   # RecordSpec(x=NumericArraySpec(shape=(), ...))

f = Function(predict, input_spec=InputSpec(w=()), return_spec=(), output_name="y")
f.output_spec   # RecordSpec(y=NumericArraySpec(shape=()))

g = Function(fit, input_spec=InputSpec(data=...), return_spec=RecordSpec(mu=(), sigma=()))
g.output_spec   # RecordSpec(mu=..., sigma=...); passing output_name= raises ValueError
```

A function whose return spec is not known yet has `return_spec=None` and keeps its output name, as `OutputSpec(mean=None)` does today.
Its first call refines the return spec.
Until then, reading its `output_spec` raises `TypeError`, as II.2 says for an unspecified output.
Constructors default the output name as today: a family's first argument, a function's callable name, and `"element"` for a batch built from bare values.

**Decision 4: names.**
- **Recommendation.** `value_spec`, `with_value_spec`, `return_spec` (today's `FunctionSpec.output_spec`, which no longer names an output spec), `output_name`, `outputs`, and `output_spec`. `output_name` is already the canonical name of `design/glossary.md`.
- **Alternative.** `result_spec` for the return spec, which avoids "return" as a noun.

### 5.2 The path interface, implemented once

One module, `core/_paths.py`, holds the path syntax that three modules split today: `_PATH_SEP`, the reader `_as_paths`, `_last_name`, and the prefix helpers that `_views.py` restates.

One mixin, `PathIndexed`, implements `[]`, `in`, `without`, and `rename` for every kind:

```python
class PathIndexed:
    # provided by each kind
    def _path_spec(self) -> RecordSpec | None: ...            # the record spec whose paths are this object's
    def _part(self, path: str) -> Any: ...                    # the kind's part at one path
    def _cut(self, paths: tuple[str, ...]) -> Self: ...       # the kind's part at a list of paths
    def _renamed(self, moves: Mapping[str, str]) -> Self: ...  # the kind's rename, keys already resolved

    # written once
    def __getitem__(self, paths): ...          # read, check against _path_spec, order, then _part or _cut
    def __contains__(self, path): ...
    def without(self, *paths): ...             # self._cut(complement(paths)), rule 4
    def rename(self, renames=None, /, **keyword_renames): ...
```

The mixin writes once the reading, the check, the ordering, the complement, the `TypeError` for an object without paths, and every message.
Each kind provides four methods, and every spec class derives them from `value_spec` and `with_value_spec`:

| Class | `_part(p)` | `_cut(ps)` |
|---|---|---|
| `Record` | the node at `p` | the cut tree |
| `RecordSpec` | the spec at `p` | the cut tree |
| `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, `BatchSpec` | `with_value_spec(value_spec[p])` | `with_value_spec(value_spec[ps])` |
| `Distribution` | `CoupledMarginal(d, p)` | `CoupledMarginal(d, ps)` |
| `ConditionalDistribution` | the conditional coupled marginal at `p` | the same at a list |
| `Function` | `f` followed by the selection at `p` | the same at a list |
| `RecordBatch` | the column, or the batch of a group's records | the cut columns |

A batch of any other kind has no paths, so the mixin refuses its paths with the `TypeError` of rule 1.

### 5.3 Outputs, implemented once

Every tracked term gains three properties, each written once on the mixin; a spec holds no name, so specs have none of them:

```python
x.output_name   # str | None, stored on the object; refused when the value spec is a record
x.output_spec   # RecordSpec: the value spec if a record, else {output_name: value spec}
x.outputs       # Record: {c: x[c]} or {output_name: x}; x.output(c) is x.outputs[c]
```

The keys of `rename` are paths and the output name.
An output name that is also a top-level path of the object is refused, so a key never names both.

### 5.4 Conformance, implemented once

One module, `core/_conformance.py`, holds `DimensionScope`, `conforms_to`, `unify`, and `refine`.
Each spec class provides two private methods, `_conform(other, scope, path)` and `_unify(other, scope, path)`, and nothing else for conformance.
`_unify_specs`, `_unification`, `_unify_input_types`, `_unify_declarations`, `ConditionalDistributionSpec._unify_with`, and the conformance part of each `_bind_dims_from_spec` go.

## 6. Tests

1. One test, parametrized over every class of § 5.2, asserts rules 2 to 7 of § 4.1 and consequence 2 of § 4.2 on a table of objects, records holding laws at leaves included, and asserts the `TypeError` for each kind of § 4.2's consequence 3.
2. One table test of `conforms_to` and `unify` per kind, including each case of § 2.3.
3. One test that `x.output_spec` and `x.outputs` follow § 4.3 for each row of its table.
4. One test that no module outside `core/_paths.py` splits or joins a path with `/` or `_PATH_SEP`.

## 7. Edits to the other plans

| Plan | Its section | Edit |
|---|---|---|
| `declarations-and-addressing.md` | Part I § 3.2, § 3.5, § 3.7 | Outputs by the two cases of § 4.3; the output name stored on the object; no object without outputs because of its value's kind |
| | Part I § 2.3, § 4.1, § 4.2, and the figure of a held field | A record's paths end at its leaves, and an object has paths only when its value type is a record |
| | § 11, and the technical notes of § 3.5 | `OutputSpec` is removed, and an output spec is a `RecordSpec` |
| | B.6 G1a, G1b, G2 | Not needed: the interim whole form exists because names are stored in specs |
| | B.6 S4b, G3, G5, and G4's `law.without` | Replaced by steps P2 to P4 of § 8; G3's paths through held objects, and G5's `[]` on batches of laws and of functions, are dropped |
| | B.4, rows T30 and I10 | `without` through a law held at a leaf, and on a batch of laws, is dropped |
| `batch-contract.md` | § 4.3, step B1 | The one check is this plan's `conforms_to`, from step P1 |
| | § 4.5, § 5.1 | `element=OutputSpec(x=())` becomes `element=(), output_name="x"` |
| | § 4.8 | The lifted-operation rule does not cover `[]` at a path, `without`, or `rename`: a batch whose element is not a record has no paths |
| | § 5.4 rule 5, § 5.5 item 3, step B6 | A batch of functions or of laws has no paths, so `fb["a"]`, `laws["mu"]`, and `laws.without("mu")` are refused |
| | § 4.9 | The name is out of spec equality with no canonical spec |
| `deferred.md` | the entries for steps G3 and G5, and for `without` on every kind | Narrowed as for B.6 and row T30 above |
| `numeric-contract.md` | § 1 and its examples | `event_spec=OutputSpec(trajectory=event)` becomes `event_spec=event, output_name="trajectory"` |
| | § 4.2 | `VectorLayout` reads leaf paths from the one enumeration of § 4.2 |
| `vector-interface.md`, `inference-interface.md` | throughout | `event_spec.term_spec` becomes `event_spec` |
| `decisions.md` | the entry of 2026-10-16 (paths go through a law, a function, or a batch held in a field) | Superseded by the settled rule for paths |
| | the entries of 2026-10-20 (the name out of a term's equality) and 2026-10-18 (O3, the element name stored in `BatchSpec`) | Superseded by Decision 2 once it is approved |

`design/` drops paths through held objects from three sections:
- II.2, where an object's value type has paths at any depth;
- II.6, where `{x: 3.0, g: d}["g/mu"]` is `d["mu"]` and `without` reaches into a held term;
- III.3, where `record[path]` reads through a held term.

The conceptual model needs one edit: its definition of `R[[p]]` keeps the full path, as its examples already do.

## 8. Steps

Each step is one PR.

1. **P1, conformance:** `core/_conformance.py` with `DimensionScope`, `conforms_to`, `unify`, and `refine`; contravariant inputs, directional opaque types, and unknown sides by § 4.5; `is_valid` through conformance; `InputSpec.__and__` through `unify`. It adds names and fixes the departures, and renames nothing. `batch-contract.md` step B1 builds on it.
2. **P2, names out of specs:** the four specs of § 5.1 hold term specs; `output_name` is stored on objects; `output_spec` is computed; `OutputSpec` is removed; fingerprints, equality, and JAX static data read the spec as it is. It is one PR with two commits: a scripted rewrite of the `.term_spec` reads, then the behaviour.
3. **P3, one path interface:** `core/_paths.py` and `PathIndexed`; `value_spec` and `with_value_spec`; `Record`, `RecordSpec`, `RecordBatch`, `Distribution`, and every spec class on the mixin; `without` on laws; the helpers of Appendix A deleted.
4. **P4, paths on every kind:** the path interface on kernels and functions whose value spec is a record.
5. **P5, outputs and feeding:** `outputs` and `output(c)`; merge's check through § 4.6.

**What users see.**
- **Breaking:** `OutputSpec` is removed; `event_spec`, `element_spec`, and `return_spec` (today's `FunctionSpec.output_spec`) are bare specs; a law's stated name is `output_name`; `x["x"]` on a law over an array named `x` raises `TypeError`; a function whose inputs accept less than declared no longer conforms.
- **Added:** paths on kernels and functions whose value spec is a record, `without` on laws, `outputs`, `conforms_to`, and `unify`.
- **Fixed:** the stated name leaves spec equality and fingerprints.

## 9. Questions for the maintainer

1. **Outputs.** Recommended: the conceptual model's two cases (§ 4.3).
2. **Where the output name is stored.** Recommended: on the object, with `OutputSpec` removed (§ 4.3, § 5.1).
3. **Conformance.** Recommended: the public `conforms_to`, `unify`, and `refine`, with II.1's rules (§ 4.5).
4. **Names.** Recommended: `value_spec`, `with_value_spec`, `return_spec`, `output_name`, `outputs`, and `output_spec` (§ 5.1).

## Appendix A. Path helpers that the path interface replaces

| Helper | What it restates |
|---|---|
| `distributions/_views.py`: `_node_at`, `_view_declaration` | the spec at a path, `RecordSpec[p]` |
| `distributions/_views.py`: `_draw_segments`, `_draw_path` | removing the output name from a path, needed only because the name is in the spec |
| `distributions/_views.py`: `_leaf_paths`, `_leaves_at`, `_node_leaves` | `RecordSpec.keys()` and the nodes above each leaf |
| `distributions/_views.py`: `_is_within`, `_below` | the prefix test and relative path |
| `distributions/_views.py`: `_projector`; `distributions/_factored.py`: `_extract` | `Record[p]` and `Record[ps]` on raw values |
| `distributions/_views.py`: `_fields_of` | the flat leaf mapping, the inverse of `_unflatten_paths` |
| `distributions/_views.py`: `_ordered_paths_of`; `distributions/_distribution.py`: `_read_event_paths`; `core/_specs.py`: `_read_paths` | reading and ordering a list, `NamedTree._ordered_paths` |
| `core/_specs.py`: `_has_path`, `_path_record`, `_paths_error`, `_output_spec_at` | `NamedTree.__contains__` and its messages, with the output name as a special case |
| `core/_batch.py`: `_has_path`, `_at_paths`; `core/_record_batch.py`: `_has_path`, `_check_path` | the check of a path and the error for an object without paths |
| `distributions/_empirical.py`: `_projection`, `_atoms_at_paths` | the cut of a batch of atoms, with its own removal of the output name |
| `distributions/_conditional.py`: `_check_rename_source`, `_is_input_path` | the nodes of the input record, and the check of a rename key |

## Appendix B. Readings this plan adopts

1. **"Binding".** The conceptual model calls connecting an output to an input "binding". In `design/`, binding supplies values to inputs at a call, so this plan says "feeds", as the declarations plan does (its A.1).
3. **A bare value.** An array or an opaque value standing alone may state an output name, as the conceptual model's `outputs(alpha) = {a: alpha}` does. Without one it has no outputs, as II.2 says today.
4. **Promotion to a numeric record.** The conceptual model asks whether a list that leaves only numeric fields gives a numeric record. It does today, since `RecordSpec.__new__` returns a `NumericRecordSpec` when every field is numeric. `numeric-contract.md` § 4.1 makes numeric-ness a property of the spec, so the question has the same answer without a class change.
5. **Coupled and standalone marginals.** `x[p]` gives the coupled marginal, and the operation `marginal(d, p)` the standalone one (VI.8). Both follow the paths of § 4.2, and this plan changes neither.
6. **An input cut by a path.** An input whose record spec a path cuts becomes required (§ 4.4), as II.2 states.
