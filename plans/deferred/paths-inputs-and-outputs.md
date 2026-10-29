# Plan: one contract for paths, inputs, and outputs

> **Status: proposed; not approved.**
> Written on 2026-10-28 against `main` at `086b4ae4`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Paths, inputs, and outputs".
> It revisits the concepts of `plans/declarations-and-addressing.md` (Part I §§ 2 to 4, Part II §§ 9 to 12) and replaces that plan's unmerged steps S4b and G1a to G5; § 7 lists the edits that plan and the other deferred plans need.
> Once approved, its rules move into `design/` (II.1, II.2, II.5, II.6, III.3, III.5, III.7, III.9, III.10, and the glossary) and its steps into the plans of the work after the workshop.

**Settled with the maintainer on 2026-10-28:**

1. A record's paths are the paths of its own tree and end at its fields; any other object has paths exactly when its value spec is a record (§ 4.1, § 4.2). This reverses the decision of 2026-10-16 that paths go through an object a record holds.
2. `in` is part of the path interface, and it raises as `[]` does for a malformed path or an object without paths (§ 4.1).
3. A list of paths keeps each node at its full path, as II.6 states; the conceptual model's definition of `[[ ]]` changes to match (§ 4.1).
4. `select` is not part of the path interface, and this plan leaves it unchanged (§ 4.1).
5. Outputs follow the conceptual model's two cases instead of II.2's five, so that every named object has outputs (§ 4.3).
6. The output name is stored on the object, not inside its spec; `OutputSpec` is removed, and an object's output spec is a `RecordSpec` (§ 4.3, § 5.5).
7. Inputs and outputs are named by single names, never by paths (§ 4.3, § 4.4).

**Decisions still open** (each explained where it arises, and gathered in § 9):

1. One public conformance relation, `conforms_to`, with inputs compared contravariantly (§ 4.5).
2. The names `value_spec`, `return_spec`, `output_name`, `outputs`, and `output_spec` (§ 5).

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

**The design continues paths past a record's fields.**
For a law `d` over `{mu, sigma}`, II.6 gives the record `{x: 3.0, g: d}` the paths `g/mu` and `g/sigma`.
II.2 gives paths to every object whose value type holds a record at any depth, such as a batch of laws over a record.
The conceptual model ends a record's paths at its fields, and gives other kinds paths only when their event, output, or element is a record.

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
Every kind ends a record's paths at its fields, as the conceptual model does:

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
3. **Spec conformance.** II.1 defines conformance and unification by kind. The code has no public relation, its private one disagrees with II.1 in five places, and the conceptual model leaves the notion open. § 4.5 keeps II.1's two relations, adds a third, refinement, and lists where the code departs; § 4.6 keeps IV.2's rule for when an output feeds an input.

## 3. Principles

1. **Structure and names are separate.** A spec states what an object is. A name says how the object connects to others, and it is stored on the object.
2. **Records are the only source of names inside an object.** Every other kind's paths and outputs are read from its value spec when that spec is a record.
3. **One rule per idea, applied to specs first.** Each operation is defined on specs, and an object's operation is checked against it: `x[p].spec == x.spec[p]`.
4. **Two relations between specs.** Conformance checks what is received against what is declared, and unification combines two declarations of one value.

## 4. The proposed design

Each subsection names the sections of `design/` it keeps, and states only what it changes or adds.
This plan calls II.2's *value type* the **value spec**: the spec of one value of an object, which is a law's or kernel's event spec, a function's return spec, a batch's element spec, or a record's or base value's own spec.
§ 5.1 gives it an attribute, and Decision 2 settles the name.

### 4.1 The paths of a record

**Kept from II.6:**
- the parts of a tree: names, nodes, children, fields, groups, paths, and canonical order;
- the reader of `[]`'s argument, and its table of errors;
- selection by one path and by a list of paths, each node at its full path in canonical order;
- chaining, `x["g"]["b"] == x["g/b"]`;
- `without`, its partition of the fields with a list, and the `ValueError` for removing every node;
- `rename` and its rules.

**Changed in II.6:**
1. **A path ends at a field.** A path leads from the root to a group or a field, and never continues into the object a field holds. II.6's selection through a held term, as `{x: 3.0, g: d}["g/mu"]`, and its cut of a held term by a list or by `without`, are removed. For a law `d` over `{mu, sigma}` at the field `g`, `R["g"]["mu"]` is `d["mu"]`, and `R["g/mu"]` raises `KeyError`.
2. **`in` refuses as `[]` does.** `p in x` is true exactly when `x[p]` raises no `KeyError`, and a malformed path, or any path given to an object without paths, raises as it does for `[]`. II.6 has `p in x` false for an object without paths, never raising.

**New:** a rule that ties every kind's path operations to its spec.

> [!IMPORTANT]
> **Rule (spec agreement).** For every object `x` with paths, `x.spec[p] == x[p].spec`, and likewise for a list of paths, `without`, and `rename`.

```python
R = Record({"a": 1.0, "b": {"c": 2.0, "d": 3.0}})
R["b/c"]               # 2.0
R[["b/c"]]             # {b: {c: 2.0}}
R["a", "b/c"]          # {a: 1.0, b: {c: 2.0}}
R.without("b/d")       # {a: 1.0, b: {c: 2.0}}, the same record as R["a", "b/c"]
"b/c" in R             # True
```

`Record.select` and `RecordBatch.select` are not part of the path interface, and this plan leaves them unchanged.

### 4.2 Which objects have paths

**Changed in II.2 and II.6:** which objects have paths.

> [!IMPORTANT]
> **Definition (paths of an object).** An object has paths exactly when its value spec is a record spec, and its paths are the paths of that record spec (§ 4.1).

II.2 and II.6 give paths to every object whose value type holds a record at any depth.
Under this definition, a law over an array, a batch of laws over `{mu, sigma}`, and a function that returns such a law have no paths; for the batch `b`, `b[i]["mu"]` reads a part of one element.
An object has paths exactly when its outputs are the top-level names of its value spec (§ 4.3).

**Kept:** the part of each kind at a path, which the conceptual model calls its *marginal*: the coupled marginal of a law (III.7), the coupled marginal of a kernel (III.9), the selected function `f[s]` (III.3), and the marginal batch (II.5).
A law, its draws, its mean, and a batch of its draws share one set of paths (II.2).

**New:** one rule gives the spec of every kind's part, so the rule of § 4.1 holds by construction.

> [!IMPORTANT]
> **Rule (the spec of a part).** For a spec `S` of a law, a kernel, a function, or a batch, `S[p]` is `S` with its value spec replaced by the value spec's selection at `p`.

### 4.3 Outputs

**Kept from II.2:** an output is one name for one whole value, never a path; a record value type has one output per child; and a stated name gives no path, so `Normal("x", 0, 1)["x"]` is a `TypeError`.

**Changed in II.2:**
1. **Two cases instead of five.** An object whose value spec is not a record has one output, the object itself, under its output name. This replaces II.2's cases 2 to 5: an object whose value is a function, a batch, or a law is one output, and no object lacks outputs because of its value's kind. An object without an output name has no outputs.
2. **The output name is stored on the object.** No spec holds it, and an object whose value spec is a record refuses one. II.2 stores it in `OutputSpec`.
3. **The output spec is II.2's record of the outputs,** as a plain `RecordSpec`: the value spec itself when it is a record, and `{name: value spec}` otherwise. It is empty for an object without outputs, and `None` while the value spec is not known, as for a function before its first call, so `None` means "not known yet" for every spec accessor. `OutputSpec` is removed. Its operations, `O["g/b"]`, `O[["a"]]`, `O.without("g")`, `O.rename(...)`, and `O & P`, are those of a record (II.6).

| Object | Value spec | Paths | Outputs |
|---|---|---|---|
| an array with output name `a` | `()` | none | `a`, whose value is the array |
| a record `{a, b: {c}}` | `{a, b: {c}}` | `a`, `b`, `b/c` | `a`, `b` |
| a function from `{u, v}` to an array, output name `f` | `()` | none | `f`, whose value is the function |
| a function from an array to `{a, b}` | `{a, b}` | `a`, `b` | `a` and `b`, whose values are selected functions |
| a function returning a law over `{a, b}`, output name `h` | a law spec | none | `h`, whose value is the function |
| a law over `{a, b}` | `{a, b}` | `a`, `b` | `a` and `b`, whose values are coupled marginals |
| a batch of laws over `{a, b}`, output name `fits` | a law spec | none | `fits`, whose value is the batch |
| a record `{x, g}` whose field `g` holds a law over `{mu}` | `{x, g}` | `x`, `g` | `x`, `g` |

**New:** `x.outputs`, the record that maps each output name to its output, which is the conceptual model's `outputs(x)`.
`design/` has only `x.output(c)`, for one output, and it becomes `x.outputs[c]`.

§ 5.5 gives the implementation.

### 4.4 Inputs

**Kept from II.2:** an input and `InputSpec`, the four sources an input spec is read from, `&`, and `rename`. The source that is an output spec is now a `RecordSpec`, which II.2 already reads.

**Changed in II.2:** inputs, like outputs, are named by single names, never by paths.
An input spec's `[]` and `without` take input names only, so `i["g/b"]` raises `KeyError` even when the spec of `g` is a record, and no input becomes required because a path cut it.

**Changed in II.2:** `None` is not a source of inputs.
`InputSpec()` has no inputs, and `InputSpec(None)` raises `TypeError`, since `None` stands for a spec not known yet, as in `FunctionSpec(input_spec=None)` and an unknown `output_spec`.
So `InputSpec(f.output_spec)` fails for a function whose return spec is not known, rather than giving an input spec with no inputs.

```python
prior = Normal("mu", 0.0, 1.0) & HalfNormal("sigma", 1.0)
InputSpec(prior.output_spec)   # InputSpec(mu=NumericArraySpec(()), sigma=NumericArraySpec(()))
```

### 4.5 Conformance and unification

**Kept from II.1:** the definitions of conformance and unification, the dimension scope, and the rules by kind, including contravariant inputs and the superclass rule for opaque types.
The conceptual model lists conformance and the dimension scope as gaps, and II.1's definitions fill them.

**The code departs from II.1 in five places,** which step P1 fixes:

| Kind | Code today |
|---|---|
| opaque | requires equal types, in both directions |
| law | also compares output names |
| kernel, function | inputs compared covariantly; optional inputs not compared; output names compared |
| function | a function spec with unknown inputs conforms to one with declared inputs |
| function | `FunctionSpec.is_valid` accepts any callable, though II.1 checks a value by its spec |

**New:**
1. **A side that only `A` leaves unknown does not conform.** II.1 says that a side `B` leaves undeclared accepts any `A`, and is silent on the reverse; this follows from II.1's definition of conformance.
2. **Refinement is one named relation.** `B.refine(A)`, for an `A` that conforms to `B`, is `B` with its dimension names bound from `A` (II.1's binding) and with each dtype or support that `B` leaves unset taken from `A` (II.2's completion). `bind_dims_from_spec` becomes `refine`, so it also fills an unset dtype or support.
3. **`DimensionScope` is a class.** It holds II.1's scope in place of the `dict` the code passes by hand, and every relation takes one.

**Decision 1: one public relation.**
- **Recommendation.** `A.conforms_to(B, scope=None)`, `unify(A, B, scope=None)`, and `B.refine(A)` are the entry points, and `bind_dims_from_value(v)` is refinement by the value's spec.
- **Alternative.** Keep today's private names, and fix only the departures in the table.

### 4.6 How an output feeds an input

**Kept from IV.2, III.3, and II.5:** an output feeds the input of the same name, and its spec must conform to the input's (IV.2, condition 6); an input that both operands take is one input, whose spec is the unification of the two (II.2's `InputSpec &`, IV.2, III.3).
These rules answer the conceptual model's two open questions about binding: an output binds to an input by conformance, and an input meets another input by unification.

**New:** the check reads the output spec, `x.output_spec[c]` conforms to `y.input_spec[c]`, so one rule serves every kind of giver.

## 5. In code: one source of truth

### 5.1 Specs hold structure only

```python
DistributionSpec(event_spec: TermSpec)
ConditionalDistributionSpec(given_spec: InputSpec, event_spec: TermSpec)
FunctionSpec(input_spec: InputSpec | None, return_spec: TermSpec | None)
BatchSpec(element_spec: TermSpec, **levels)
```

`value_spec` is the value spec of § 4, a read-only property of every object and every spec: a law's or kernel's `event_spec`, a function's `return_spec`, a batch's `element_spec`, and otherwise `spec`.
No constructor takes it, and nothing sets it.
The private `TermSpec._with_value_spec(spec)` replaces it, to build the spec of a part.
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
Until then, its `output_spec` is `None`, as its `return_spec` is.
Constructors default the output name as today: a family's first argument, a function's callable name, and `"element"` for a batch built from bare values.

**Decision 2: names.**
- **Recommendation.** `value_spec`, `return_spec` (today's `FunctionSpec.output_spec`, which no longer names an output spec), `output_name`, `outputs`, and `output_spec`. `output_name` is already the canonical name of `design/glossary.md`.
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
    def without(self, *paths): ...             # self._cut(complement(paths)), by II.6's partition
    def rename(self, renames=None, /, **keyword_renames): ...
```

The mixin writes once the reading, the check, the ordering, the complement, the `TypeError` for an object without paths, and every message.
Each kind provides four methods, and every spec class derives them from `value_spec` and `_with_value_spec`:

| Class | `_part(p)` | `_cut(ps)` |
|---|---|---|
| `Record` | the node at `p` | the cut tree |
| `RecordSpec` | the spec at `p` | the cut tree |
| `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, `BatchSpec` | `_with_value_spec(value_spec[p])` | `_with_value_spec(value_spec[ps])` |
| `Distribution` | `CoupledMarginal(d, p)` | `CoupledMarginal(d, ps)` |
| `ConditionalDistribution` | the conditional coupled marginal at `p` | the same at a list |
| `Function` | `f` followed by the selection at `p` | the same at a list |
| `RecordBatch` | the column, or the batch of a group's records | the cut columns |

A batch of any other kind has no paths, so the mixin refuses its paths with the `TypeError` of II.6's reader.

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

### 5.5 The output name on the object (step P2)

This section lists every change step P2 makes to move the output name out of specs.

**Storage.** The name is a second identity field beside the label:

```python
class TrackedTerm:
    _output_name: str | None                                  # a new slot, beside _label

    @property
    def output_name(self) -> str | None: ...
    def with_output_name(self, name: str | None) -> Self: ...  # a copy, as with_label makes
    @property
    def output_spec(self) -> RecordSpec | None: ...            # output_spec_of(self.value_spec, self.output_name)
```

1. `_init_tracked(label, *, output_name=None, provenance=None)` stores the name, and every host class declares `_output_name` in its `__slots__`.
2. `_TrackedTermMeta` checks the name once `__init__` returns, beside its check of the label. A name is a non-empty `str` without `/`, and an object whose value spec is a record refuses one with `ValueError`.
3. `_shallow_copy`, `with_label`, `with_provenance`, and `_with_expression` keep the name.
4. A constructor that requires an output name today still requires one. An array and an opaque value have none unless one is given.

**Specs.** The four classes of § 5.1 hold term specs, and so does every spec nested in another, such as a law's spec in a record field or a function's spec in an input.
`FunctionSpec`'s second argument is renamed `return_spec`, and `BatchSpec(element=...)` takes a term spec.
`value_spec` and `_with_value_spec` arrive in this step, since the reads of `.term_spec` move to them.

**One function computes an output spec.** `core/_outputs.py` holds it:

```python
def output_spec_of(value_spec: TermSpec | None, output_name: str | None) -> RecordSpec | None:
    # value_spec itself when it is a record; {output_name: value_spec} when a name is given;
    # RecordSpec() when there is no name; None when value_spec is None (not known yet)
```

A consumer that needs the spec refuses `None` for its own reason, with a message that names the object and says its return spec is known after its first call.
A merge refuses it because its conformance check needs a spec, and `InputSpec` refuses it because `None` is not a source of inputs.

**`InputSpec`'s default.** Its positional argument defaults to a private sentinel, `_NO_INPUTS`, so the constructor tells an explicit `None` from no argument:

```python
_NO_INPUTS = object()   # InputSpec() was given no source: no inputs, as _UNSET in core/tracked.py

def __init__(self, inputs: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec = _NO_INPUTS, /,
             **inputs_by_name: TermSpec | tuple) -> None: ...
```

`InputSpec(None)` raises `TypeError`: "cannot read inputs from None, which stands for a spec not known yet. Write InputSpec() for no inputs".
No code, test, or document in the repository writes `InputSpec(None)` today.

`x.output_spec` calls it, and so does code that holds only specs, such as an operation's `result=` rule.

**Constructors.**

| Constructor | Today | After P2 |
|---|---|---|
| every family whose first argument is the output name, as `Normal("x", 0, 1)` | `event_spec=OutputSpec(x=spec)` | `event_spec=spec`; the name stays the first argument |
| `Distribution`, `ConditionalDistribution` | `event_spec=OutputSpec(x=spec)`, or a `RecordSpec` | `event_spec=spec, output_name="x"` |
| `distribution`, `EmpiricalDistribution`, `KDEDistribution` | `event_spec=` an `OutputSpec`, or a spec with `output_name=` | `event_spec=spec, output_name="x"` |
| `conditional_distribution(fn, ...)` | the name of the law `fn` returns, read from its `OutputSpec` | `output_name=`, by default the `output_name` of the law `fn` returns |
| `GaussianRandomFunction`, `GaussianProcess`, `LinearBasisFunction` | `output_spec=OutputSpec(y=None)` for the drawn functions | `return_spec=`; a drawn function takes the law's output name |
| `Function`, `function`, `EvaluationPoints` | `output_spec=OutputSpec(y=spec)` | `return_spec=spec, output_name="y"`; by default the callable's name, as `_callable_output_name` gives it |
| `BatchSpec` and every batch class | `element=OutputSpec(x=spec)` on `BatchSpec`; the batch classes name the element `"element"` | `element=spec, output_name="x"`; by default `"element"` for an element that is not a record |
| `InputSpec(O)` | `O` an `OutputSpec` | `O` a `RecordSpec`, such as `x.output_spec` |

A function whose return spec is not known keeps its name until its first call.
If that call returns a record, a default name is dropped and a given name raises `TypeError`, as `OutputSpec(y=None).with_term_spec` raises today.
The function stores whether its name was given.

**The name of a result.** Each operation names the term it returns by one of five rules:

| Rule | The result's output name | Where it applies |
|---|---|---|
| keep | the operand's | `mean`, `variance`, `standard_deviation`, `quantile`; `expectation` (the integrand's); `convert`; `mixture` (the kernel's); `sample`, for each draw and for the batch of draws; `condition_on`, when the posterior's value spec is not a record; `make_posterior` (the target's); the converters and the resampling laws (the source's) |
| part | the last name of the path, when the node is not a record | `x[p]` on every kind; `marginal(d, p)`; the column batches of a `RecordBatch`; `EmpiricalDistribution._marginal` |
| own | a fixed name | `cov`; each density, under the operation's name; `log_det_jacobian`; the Monte Carlo law, `"draws"`; `MinibatchedDistribution`, `"target"` and `"parameters"`; the BayesFlow likelihood, `_observation_input(prior)`; a posterior with no target name, `"posterior"` |
| call | the function's | the value a `Function` call returns; `pushforward` (the map's); the batch of a sweep or a broadcast, and the law of a lifted call |
| none | none, since the result's value spec is a record | `&` on every kind; `condition_on` when several outputs remain |

`x.rename({x.output_name: "z"})` is `x.with_output_name("z")`, and every other key of `rename` is a path.
Three results change: the batch of draws, the column batches of a `RecordBatch`, and the batch of a sweep name their element `"element"` today, and after P2 they carry the name their rule gives.

**The `result=` rule of an operation.** The rule of `@operation(result=rule)` returns `Declared(value_spec, output_name)` or `None`.
`Declared` is a private `NamedTuple` in `operations/_operation.py`.
An output spec cannot replace it, since `{y: ()}` is the output spec both of a record `{y}` and of an array named `y`.

**Readers.** Each read of the name moves from the spec to the object:

| Today | After P2 |
|---|---|
| `_root_output_name(x.event_spec)`, `f._output_name` | `x.output_name` |
| `x.event_spec.outputs` | `x.output_spec.children` |
| `_has_record_outputs(x.event_spec)` | `isinstance(x.value_spec, RecordSpec)` |
| `d.event_spec.term_spec`, `f.output_spec.term_spec`, `b.element_spec.term_spec` | `d.event_spec`, `f.return_spec`, `b.element_spec`, or `x.value_spec` |
| `OutputSpec.default(spec, output_name=n)` | `spec`, with `output_name=n` passed to the constructor |
| `declared.with_term_spec(produced)` | `declared.refine(produced)`, from step P1 |
| `_unnamed_declaration(f.output_spec)` | `f.return_spec` |

Four groups of readers change behaviour:
1. **The name is no longer a path.** `Distribution.__getitem__`, the factored marginal, `CoupledMarginal._parent_request`, `_draw_segments`, `_requested_nodes`, and `_requested_paths` drop the case of the root name, so `Normal("x", 0, 1)["x"]` is a `TypeError`.
2. **A spec's name is no longer compared with an object's.** The checks in `_unify_declarations`, `FunctionSpec._bind_dims_from_spec`, `_whole_term_event`, `glm_likelihood`, the Gaussian random function's `_declarations`, and `_agreed_event_spec` go. The checks that compare two objects' names stay: the components of a mixture, `_packaged_alike` in `condition_on`, and `_target_atoms` in reweighting.
3. **Factored laws lose the packaged form.** `FactoredDistribution(..., _output_name=)` and `FactoredConditionalDistribution(..., _output_name=)` go: a gathered factor is renamed into its group, so its value spec is `{g: {...}}`. `_SoleField` gives the law at one field that field's name as its output name.
4. **Labels and levels read the object.** The array atoms of `EmpiricalDistribution`, the array backend of a TFP family, the default level of a bootstrap, the atom level of a broadcast, and the label of a posterior's atoms read `x.output_name`.

`_law_repr` reads `x.output_name`, so `Normal('x', ...)` prints as today, and a spec's repr shows no name.

**Equality, fingerprints, and JAX.**
1. A spec's equality and hash compare structure only. II.2's three rules for the name in equality become one: no spec holds a name.
2. A term's equality and hash ignore its output name, as they ignore its label.
3. The fingerprint drops the `OutputSpec` branch and the root name of a `DistributionSpec`, so `Normal("x", 0, 1)` and `Normal("y", 0, 1)` share a fingerprint. A result reused by fingerprint is named again by its rule, so it never carries another operand's name.
4. The static data of a `NumericArray` and a `NumericArrayBatch` is `(spec, output_name)`, so a transform keeps the name, and two values that differ only in their names trace separately. A record and a batch of records have no name.

**Deleted.**
- `OutputSpec` and its export from `probpipe`.
- In `core/_specs.py`: `_EXPOSED_SPECS`, `_EXPOSED_OF`, `_exposed_kind`, `_exposed_outputs`, `_root_path`, `_ROOT_NAMES`, `_unification`, `_renamed_exposed`, `_root_output_name`, `_has_record_outputs`, `_VALUES_WITHOUT_RECORD`, `_no_output_record`, `_type_not_known`, `_output_record`, `_unnamed_declaration`, and `_unnamed_exposed`.
- `_complete_event_spec` and `_whole_term_event` in `distributions/_distribution.py`, `_complete_output_spec` in `values/_function_base.py`, the `OutputSpec` cases of `_as_element_spec` in `core/_batch.py`, and `_form_keywords` in `distributions/_factored.py`.

`_merged_output_record` stays, and merges the output specs of its operands.
Five path helpers read an `OutputSpec`: `_read_paths`, `_path_record`, `_has_path`, `_paths_error`, and `_output_spec_at`.
P2 rewrites them over the value spec, and P3 deletes them.

**Commits.** P2 is one PR of four commits, and the suite passes after each:
1. **Accessors, with no change in behaviour:** `output_name`, `output_spec`, `value_spec`, `_with_value_spec`, and `output_spec_of`, still read from the `OutputSpec` in the spec. Every reader of the tables above moves to them.
2. **Bare specs:** the four spec classes, the constructors, the `result=` rules, the five naming rules, equality, fingerprints, and JAX static data. A script rewrites the `.term_spec` reads, 279 in `probpipe/` and 443 in `tests/`.
3. **Deletion:** `OutputSpec`, its export, and the helpers above.
4. **Documents:** `docs/api/input_and_output_specs.md`, `docs/api/extending.md`, the notebooks that build an `OutputSpec` (user guide 01, 03, and 09, and tutorial 06), the `design/` sections of § 7, and the CHANGELOG.

## 6. Tests

1. One test, parametrized over every class of § 5.2, asserts II.6's selection, chaining, partition, and errors, the two changes of § 4.1, and the spec agreement rule on a table of objects, records holding laws in fields included, and asserts the `TypeError` for each object of § 4.2 that has no paths.
2. One table test of `conforms_to` and `unify` per kind, including each case of § 2.3.
3. One test that `x.output_spec` and `x.outputs` follow § 4.3 for each row of its table.
4. One test that no module outside `core/_paths.py` splits or joins a path with `/` or `_PATH_SEP`.
5. One table test of each constructor of § 5.5: its default output name, a given name, and its output spec.
6. One table test of the five naming rules of § 5.5, with a row for each operation it lists.
7. Two laws that differ only in their names have equal specs and equal fingerprints, and a `jax.jit` round trip keeps the name of a `NumericArray` and of a `NumericArrayBatch`.
8. `with_output_name` on an object whose value spec is a record raises `ValueError`, and `probpipe.OutputSpec` no longer exists.
9. A function whose return spec is not known has `output_spec` `None`; `InputSpec()` has no inputs; `InputSpec(None)` and `InputSpec(f.output_spec)` for such a function raise `TypeError`; and a merge with such a function refuses it with a message that names the function.

## 7. Edits to the other plans

| Plan | Its section | Edit |
|---|---|---|
| `declarations-and-addressing.md` | Part I § 3.2, § 3.5, § 3.7 | Outputs by the two cases of § 4.3; the output name stored on the object; no object without outputs because of its value's kind |
| | Part I § 2.3, § 4.1, § 4.2, and the figure of a held field | A record's paths end at its fields, and an object has paths only when its value type is a record |
| | § 11, and the technical notes of § 3.5 | `OutputSpec` is removed, and an output spec is a `RecordSpec` |
| | B.6 G1a, G1b, G2 | Not needed: the interim whole form exists because names are stored in specs |
| | B.6 S4b, G3, G5, and G4's `law.without` | Replaced by steps P2 to P4 of § 8; G3's paths through held objects, G4's paths into a record input, and G5's `[]` on batches of laws and of functions are dropped |
| | B.4, rows T30 and I10 | `without` through a law held in a field, and on a batch of laws, is dropped |
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
| | the entries of 2026-10-20 (the name out of a term's equality) and 2026-10-18 (O3, the element name stored in `BatchSpec`) | Superseded by the settled rule for output names |

`design/` drops paths through held objects from three sections:
- II.2, where an object's value type has paths at any depth;
- II.6, where `{x: 3.0, g: d}["g/mu"]` is `d["mu"]` and `without` removes a node inside a held term;
- III.3, where `record[path]` reads through a held term.

II.2 also drops paths into a record input: an input spec's `[]` and `without` take input names only, and no input becomes required because a path cut it.
In II.2, `InputSpec`'s signature drops `| None = None` for the sentinel `_NO_INPUTS`, and `None` is refused as a source.

The conceptual model needs one edit: its definition of `R[[p]]` keeps the full path, as its examples already do.

## 8. Steps

Each step is one PR.

1. **P1, conformance:** `core/_conformance.py` with `DimensionScope`, `conforms_to`, `unify`, and `refine`; contravariant inputs, directional opaque types, and unknown sides by § 4.5; `is_valid` through conformance; `InputSpec.__and__` through `unify`. It adds names and fixes the departures, and renames nothing. `batch-contract.md` step B1 builds on it.
2. **P2, names out of specs:** § 5.5, in its four commits.
3. **P3, one path interface:** `core/_paths.py` and `PathIndexed`; `Record`, `RecordSpec`, `RecordBatch`, `Distribution`, and every spec class on the mixin; `without` on laws; the helpers of Appendix A deleted.
4. **P4, paths on every kind:** the path interface on kernels and functions whose value spec is a record.
5. **P5, outputs and feeding:** `outputs` and `output(c)`; merge's check through § 4.6.

**What users see.**
- **Breaking:** `OutputSpec` is removed; `event_spec`, `element_spec`, and `return_spec` (today's `FunctionSpec.output_spec`) are bare specs; a law's stated name is `output_name`; `x["x"]` on a law over an array named `x` raises `TypeError`; a function whose inputs accept less than declared no longer conforms; `InputSpec(None)` raises `TypeError`, and `InputSpec()` is the input spec with no inputs.
- **Added:** paths on kernels and functions whose value spec is a record, `without` on laws, `outputs`, `conforms_to`, and `unify`.
- **Fixed:** the stated name leaves spec equality and fingerprints.

## 9. Questions for the maintainer

1. **Conformance.** Recommended: the public `conforms_to`, `unify`, and `refine`, with II.1's rules (§ 4.5).
2. **Names.** Recommended: `value_spec`, `return_spec`, `output_name`, `outputs`, and `output_spec` (§ 5.1).

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
| `distributions/_conditional.py`: `_check_rename_source`, `_is_input_path` | the check of a rename key against the input names and the event's paths |

## Appendix B. Readings this plan adopts

1. **"Binding".** The conceptual model calls connecting an output to an input "binding". In `design/`, binding supplies values to inputs at a call, so this plan says "feeds", as the declarations plan does (its A.1).
3. **A bare value.** An array or an opaque value standing alone may state an output name, as the conceptual model's `outputs(alpha) = {a: alpha}` does. Without one it has no outputs, as II.2 says today.
4. **Promotion to a numeric record.** The conceptual model asks whether a list that leaves only numeric fields gives a numeric record. It does today, since `RecordSpec.__new__` returns a `NumericRecordSpec` when every field is numeric. `numeric-contract.md` § 4.1 makes numeric-ness a property of the spec, so the question has the same answer without a class change.
5. **Coupled and standalone marginals.** `x[p]` gives the coupled marginal, and the operation `marginal(d, p)` the standalone one (VI.8). Both follow the paths of § 4.2, and this plan changes neither.
