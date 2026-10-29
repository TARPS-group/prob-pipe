# Plan: one contract for paths, inputs, and outputs

> **Status: proposed; not approved.**
> Written on 2026-10-28 against `main` at `086b4ae4`, and revised on 2026-10-29.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Paths, inputs, and outputs".
> It revisits the concepts of `plans/declarations-and-addressing.md` (Part I §§ 2 to 4, Part II §§ 9 to 12) and replaces that plan's unmerged steps S4b and G1a to G5; § 8 lists the edits that plan and the other deferred plans need.
> Once approved, its rules move into `design/` (II.1, II.2, II.5, II.6, III.3, III.5, III.7, III.9, III.10, and the glossary) and its steps into the plans of the work after the workshop.

**Settled with the maintainer on 2026-10-28:**

1. A record's paths are the paths of its own tree and end at its fields; any other object has paths exactly when its value spec is a `RecordSpec` (§ 4.1, § 4.2). This reverses the decision of 2026-10-16 that paths go through an object a record holds.
2. `in` is part of the path interface, and it raises as `[]` does for a malformed path or an object without paths (§ 4.1).
3. A list of paths keeps each node at its full path, as II.6 states (§ 4.1).
4. `select` is not part of the path interface, and this plan leaves it unchanged (§ 4.1).
5. Outputs follow two cases instead of II.2's five, so that every named object has outputs (§ 4.3).
6. The output name is stored on the object, not inside its spec; `OutputSpec` is removed, and an object's output spec is a `RecordSpec` (§ 4.3, § 6).
7. Inputs and outputs are named by single names, never by paths (§ 4.3, § 4.4).
8. `x.output_names` is the one public read of an object's output names; the stored name is private, a constructor states it with `output_name=`, and `rename` changes it (§ 4.3).
9. `None` means "not known yet" for every spec accessor, `x.output_spec` included, and `InputSpec(None)` raises (§ 4.3, § 4.4).
10. No name is both an input name and an output name of one object, for every kind; one function checks it (§ 4.7).
11. A rename never crosses between inputs and outputs, and an input is never renamed to a path. Each name is renamed where it is stored: `TermSpec.rename` renames a spec's inputs and paths, and `TrackedTerm.rename` renames an object's output name and passes every other key to its spec; each kind supplies only the mechanics (§ 4.7).
12. Functions and kernels both rename their inputs, by one flat name each; the kernel's grouping and splitting of inputs by path is removed (§ 4.7).

**Decisions still open** (each explained where it arises, and gathered in § 10):

1. One public conformance relation, `conforms_to`, with inputs compared contravariantly (§ 4.5).
2. The names `value_spec`, `return_spec`, `output_names`, `outputs`, and `output_spec` (§ 5).

**How to read this plan.**
§ 4 states each rule, citing the section of `design/` it keeps and stating only what it changes or adds.
§ 5 gives the full API of every new or changed class, function, and method, with its docstring and type hints.
§ 6 gives step P2, the move of the output name out of specs, site by site.
Every example is ProbPipe code.
Throughout, `d` is the law

```python
d = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)
d.event_spec   # RecordSpec(mu=NumericArraySpec(()), sigma=NumericArraySpec(()))
```

---

## 1. Why this plan exists

ProbPipe connects objects by name.
A *path* such as `"g/b"` picks out a part of an object, an *output* names a part that another object can receive, and an *input* names what an object receives.
`design/` (II.1, II.2, II.6) and the code state these ideas, and they disagree.
The code also implements each idea once per kind:

```python
r = Record({"x": 3.0, "g": d})

d["mu"]                      # a CoupledMarginal, read through OutputSpec and _read_paths
r["g"]                       # the law d, read through NamedTree
r["g/mu"]                    # KeyError: unknown path 'g/mu'; design II.6 says it is d["mu"]
Normal("x", 0.0, 1.0)["x"]   # the law itself; design II.2 says TypeError
```

The goal is one definition of each idea, one implementation that every kind shares, and a precise statement of when one spec conforms to another.

## 2. The current state

### 2.1 What `design/` states, and what this plan changes

| Point | `design/` today | This plan |
|---|---|---|
| The paths of a record | its tree, continued into the objects its fields hold (II.6) | its tree, ending at its fields |
| Which other objects have paths | every object whose value type holds a `RecordSpec` at any depth (II.2, II.6) | every object whose value spec is a `RecordSpec` |
| `p in x` for an object without paths | `False` (II.6) | `TypeError` |
| `[]` on an input spec | input names, and paths into a record input (II.2) | input names only |
| The outputs of an object whose value is not a record | five cases, one of which gives no outputs (II.2) | one output, the object, under its output name |
| Where an output name is stored | in an `OutputSpec`, inside the spec (II.2) | on the object |
| Conformance and unification | defined by kind (II.1) | kept; the code's departures fixed |
| A name that is both an input and an output | refused for a kernel (III.9); allowed for a function, whose `rename` renames both (III.3) | refused for every kind |
| Renaming a kernel's inputs | by path, so inputs may be grouped into a record input or split out of one (III.9) | by name only |

### 2.2 What the code does

The overhaul merged steps S1, S2a, S2b-out, S2b-in, S3, and S4a of the declarations plan; S4b and G1a to G5 did not merge.

**Path indexing is implemented once per kind.**

| Kind | `[p]` | `[[p, q]]` | `in` | `without` | `rename` | Implemented by |
|---|---|---|---|---|---|---|
| `Record`, `RecordSpec` | ✓ | ✓ | ✓ | ✓ | ✓ | `NamedTree` |
| `RecordBatch` | ✓ | ✓ | ✓ | ✓ | ✓ | `_at_paths` over `event_template` |
| `OutputSpec` | ✓ | ✓ | ✓ | ✓ | ✓ | `_read_paths` and `_path_record` |
| `Distribution` | ✓ | ✓ | ✓ | – | ✓ | `_read_event_paths` and `CoupledMarginal` |
| `ConditionalDistribution` | – | – | – | – | ✓ | its own `_check_rename_source` |
| `Function`, and batches whose element is not a record | – | – | – | – | – | – |
| `DistributionSpec`, `FunctionSpec`, `BatchSpec` | – | – | – | – | – | – |

The shared parts are the reader of the argument, `_as_paths` in `core/_shapes.py`, and the separator `_PATH_SEP` in `core/named_tree.py`.
Each kind repeats the rest: whether a path exists, the spec at a path, canonical order, the error for an object with no paths, and the cut to a list.
`distributions/_views.py` alone holds more than a dozen helpers that restate `NamedTree` on an `OutputSpec` (Appendix A).
Every kind already ends a record's paths at its fields:

```python
Record({"x": 3.0, "g": d}).spec["g/mu"]
# KeyError: unknown path 'g/mu' in this RecordSpec; available paths: ['x', 'g']
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
Every read of a law's event takes an extra step, `event_spec.term_spec`, and `.term_spec` appears 279 times in `probpipe/` and 443 times in `tests/`.
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

### 2.3 The three questions, assessed

1. **Outputs as a wrapper spec.** The approach does not hold up. An output name says how an object connects to others, and it is not part of the object's structure. `OutputSpec` stores it inside the structure of every law, kernel, function, and batch, so each operation that reads structure strips the name again or treats it as a special case. § 4.3 computes outputs from the spec and a name stored on the object, and an output spec becomes a `RecordSpec`.
2. **One source of truth for paths.** The reader of the argument is shared. What a path means is implemented once per kind, and three kinds lack it. § 4.1 and § 4.2 define paths once, on specs, and § 5.2 gives every kind one implementation.
3. **Spec conformance.** II.1 defines conformance and unification by kind. The code has no public relation, and its private one disagrees with II.1 in five places. § 4.5 keeps II.1's two relations, adds a third, refinement, and lists where the code departs; § 4.6 keeps IV.2's rule for when an output feeds an input.

## 3. Principles

1. **Structure and names are separate.** A spec states what an object is. A name says how the object connects to others, and it is stored on the object.
2. **Records are the only source of names inside an object.** Every other kind's paths and outputs are read from its value spec when that spec is a `RecordSpec`.
3. **One rule per idea, applied to specs first.** Each operation is defined on specs, and an object's operation is checked against it: `x[p].spec == x.spec[p]`.
4. **Two relations between specs.** Conformance checks what is received against what is declared, and unification combines two declarations of one value.

## 4. The proposed design

Each subsection names the sections of `design/` it keeps, and states only what it changes or adds.
This plan calls II.2's *value type* the **value spec**: the spec of one value of an object.
It is a law's or kernel's `event_spec`, a function's `return_spec`, a batch's `element_spec`, and a record's or base value's own `spec`, and § 5.5 gives it the read-only property `value_spec`.

### 4.1 The paths of a record

**Kept from II.6:**
- the parts of a tree: names, nodes, children, fields, groups, paths, and canonical order;
- the reader of `[]`'s argument, and its table of errors;
- selection by one path and by a list of paths, each node at its full path in canonical order;
- chaining, `r["b"]["c"] == r["b/c"]`;
- `without`, its partition of the fields with a list, and the `ValueError` for removing every node;
- `rename` and its rules.

**Changed in II.6:**
1. **A path ends at a field.** A path leads from the root to a group or a field, and never continues into the object a field holds. II.6's selection through a held term, as `Record({"x": 3.0, "g": d})["g/mu"]`, and its cut of a held term by a list or by `without`, are removed:

   ```python
   r = Record({"x": 3.0, "g": d})
   r["g"]["mu"]   # d["mu"], a CoupledMarginal
   r["g/mu"]      # KeyError: unknown path 'g/mu'; available paths: ['x', 'g']
   ```

2. **`in` refuses as `[]` does.** `p in x` is true exactly when `x[p]` raises no `KeyError`, and a malformed path, or any path given to an object without paths, raises as it does for `[]`. II.6 has `p in x` false for an object without paths, never raising.

**New:** a rule that ties every kind's path operations to its spec.

> [!IMPORTANT]
> **Rule (spec agreement).** For every object `x` with paths and every path `p` of `x`, `x.spec[p] == x[p].spec`, and likewise for a list of paths, `without`, and `rename`.

```python
r = Record({"a": 1.0, "b": {"c": 2.0, "d": 3.0}})
r["b/c"]               # NumericArray 2.0
r[["b/c"]]             # Record({"b": {"c": 2.0}})
r["a", "b/c"]          # Record({"a": 1.0, "b": {"c": 2.0}})
r.without("b/d")       # Record({"a": 1.0, "b": {"c": 2.0}}), equal to r["a", "b/c"]
"b/c" in r             # True
```

`Record.select` and `RecordBatch.select` are not part of the path interface, and this plan leaves them unchanged.

### 4.2 Which objects have paths

**Changed in II.2 and II.6:** which objects have paths.

> [!IMPORTANT]
> **Definition (paths of an object).** An object has paths exactly when its value spec is a `RecordSpec`, and its paths are the paths of that `RecordSpec` (§ 4.1).

II.2 and II.6 give paths to every object whose value type holds a `RecordSpec` at any depth.
Under this definition, three objects that hold a record further down have no paths:

```python
Normal("x", 0.0, 1.0)["x"]       # TypeError: its value spec is a NumericArraySpec
laws = DistributionBatch([d, d], "dataset")
laws["mu"]                        # TypeError: its value spec is d.spec, a DistributionSpec
laws[0]["mu"]                     # the coupled marginal of one element
f = Function(fit, return_spec=d.spec, output_name="posterior")
f["mu"]                           # TypeError: its value spec is a DistributionSpec
```

**Kept:** the part of each kind at a path: the coupled marginal of a law (III.7), the coupled marginal of a kernel (III.9), the selected function `f[s]` (III.3), and the marginal batch (II.5).
A law, its draws, its mean, and a batch of its draws share one set of paths (II.2).

**New:** one rule gives the spec of every kind's part, so the rule of § 4.1 holds by construction.

> [!IMPORTANT]
> **Rule (the spec of a part).** For a `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, or `BatchSpec` `S` whose value spec is a `RecordSpec`, `S[p]` is `S` with its value spec replaced by `S.value_spec[p]`, and `S[[p, q]]` is `S` with its value spec replaced by `S.value_spec[[p, q]]`.

```python
d.spec["mu"]              # DistributionSpec(event_spec=NumericArraySpec(()))
d.spec["mu"] == d["mu"].spec   # True
```

### 4.3 Outputs

**Kept from II.2:** an output is one name for one whole value, never a path; an object whose value spec is a `RecordSpec` has one output per top-level child; and an output name gives no path, so `Normal("x", 0.0, 1.0)["x"]` is a `TypeError`.

**Changed in II.2:**
1. **Two cases instead of five.** An object whose value spec is not a `RecordSpec` has one output, the object itself, under its output name. This replaces II.2's cases 2 to 5: an object whose value is a function, a batch, or a law is one output, and no object lacks outputs because of its value's kind. An object without an output name has no outputs.
2. **The output name is stored on the object.** No spec holds it, and an object whose value spec is a `RecordSpec` refuses one with `ValueError`. II.2 stores it in `OutputSpec`.
3. **The output spec is II.2's record of the outputs,** as a plain `RecordSpec`: the value spec itself when it is a `RecordSpec`, and `RecordSpec({output_name: value_spec})` otherwise. It is `RecordSpec()` for an object without outputs, and `None` while the value spec is not known, as for a function before its first call. `OutputSpec` is removed. The operations on an output spec, `O["g/b"]`, `O[["a"]]`, `O.without("g")`, `O.rename(...)`, and `O & P`, are those of a `RecordSpec` (II.6).

**New:** two public reads of the outputs, `x.output_names` and `x.outputs` (§ 5.4).
The stored name is private, so `output_names` is the one way to read the names.
A constructor states the name with its `output_name=` keyword, and `rename` changes it.
`design/` has `x.output(c)`, for one output; it becomes `x.outputs[c]`.

In this table, `fit` is a callable of one argument `data`, and `h` is a callable of two arguments `u` and `v`:

| Object `x` | `x.value_spec` | Paths of `x` | `x.output_names` | `x.outputs[c]` |
|---|---|---|---|---|
| `NumericArray(1.0, label="a", output_name="a")` | `NumericArraySpec(())` | none | `("a",)` | `x` |
| `Record({"a": 1.0, "b": {"c": 2.0}})` | `RecordSpec(a=NumericArraySpec(()), b=RecordSpec(c=NumericArraySpec(())))` | `"a"`, `"b"`, `"b/c"` | `("a", "b")` | `x[c]` |
| `Function(h, return_spec=NumericArraySpec(()), output_name="y")` | `NumericArraySpec(())` | none | `("y",)` | `x` |
| `Function(fit, return_spec=RecordSpec(mu=(), sigma=()))` | `RecordSpec(mu=..., sigma=...)` | `"mu"`, `"sigma"` | `("mu", "sigma")` | `x[c]`, a selected function |
| `Function(fit, return_spec=d.spec, output_name="posterior")` | `d.spec` | none | `("posterior",)` | `x` |
| `d` | `d.event_spec` | `"mu"`, `"sigma"` | `("mu", "sigma")` | `x[c]`, a coupled marginal |
| `DistributionBatch([d, d], "dataset", output_name="fits")` | `d.spec` | none | `("fits",)` | `x` |
| `Record({"x": 3.0, "g": d})` | `RecordSpec(x=NumericArraySpec(()), g=d.spec)` | `"x"`, `"g"` | `("x", "g")` | `x[c]` |
| `Function(fit)`, before its first call | `None` | not known | `None` | not known |

### 4.4 Inputs

**Kept from II.2:** an input and `InputSpec`, the sources an input spec is read from, `&`, and `rename`. The source that is an output spec is now a `RecordSpec`, which II.2 already reads.

**Changed in II.2:**
1. **Inputs are named by single names, never by paths.** An input spec's `[]` and `without` take input names only, so `i["g/b"]` raises `KeyError` even when the spec of `g` is a `RecordSpec`, and no input becomes required because a path cut it.
2. **`None` is not a source of inputs.** `InputSpec()` has no inputs, and `InputSpec(None)` raises `TypeError`, since `None` stands for a spec not known yet, as in `FunctionSpec(input_spec=None)` and an unknown `output_spec`. So `InputSpec(f.output_spec)` fails for a function whose return spec is not known, rather than giving an input spec with no inputs.

```python
InputSpec(d.output_spec)   # InputSpec(mu=NumericArraySpec(()), sigma=NumericArraySpec(()))
InputSpec()                # no inputs
InputSpec(None)            # TypeError
```

### 4.5 Conformance and unification

**Kept from II.1:** the definitions of conformance and unification, the dimension scope, and the rules by kind, including contravariant inputs and the superclass rule for opaque types.

**The code departs from II.1 in five places,** which step P1 fixes:

| Kind | Code today |
|---|---|
| opaque | requires equal types, in both directions |
| law | also compares output names |
| kernel, function | inputs compared covariantly; optional inputs not compared; output names compared |
| function | a `FunctionSpec` with unknown inputs conforms to one with declared inputs |
| function | `FunctionSpec.is_valid` accepts any callable, though II.1 checks a value by its spec |

**New:**
1. **A side that only `A` leaves unknown does not conform.** II.1 says that a side `B` leaves undeclared accepts any `A`, and is silent on the reverse; this follows from II.1's definition of conformance.
2. **Refinement is one named relation.** `B.refine(A)`, for an `A` that conforms to `B`, is `B` with its dimension names bound from `A` (II.1's binding) and with each dtype, support, type, or metadata that `B` leaves unset taken from `A` (II.2's completion). It replaces `bind_dims_from_spec`, which bound dimension names only.
3. **`DimensionScope` is a class.** It holds II.1's scope in place of the `dict` the code passes by hand, and every relation takes one. A relation that fails leaves its scope unchanged.

**Decision 1: one public relation.**
- **Recommendation.** `A.conforms_to(B, scope=None)`, `unify(A, B, scope=None)`, and `B.refine(A, scope=None)` are the entry points, `bind_dims_from_spec` is replaced by `refine`, and `bind_dims_from_value(v)` is refinement by the spec of `v` (§ 5.6).
- **Alternative.** Keep today's private names, and fix only the departures in the table.

### 4.6 How an output feeds an input

**Kept from IV.2, III.3, and II.2:** an output feeds the input of the same name, and its spec must conform to the input's (IV.2, condition 6); an input that both operands take is one input, whose spec is the unification of the two (II.2's `InputSpec.__and__`, IV.2, III.3).

**New:** the check reads the output spec, so one rule serves every kind of giver: the output `c` of `x` feeds the input `c` of `y` when `x.output_spec[c].conforms_to(y.input_spec[c], scope)`, in the dimension scope of the merge.

### 4.7 Renaming

**Kept from II.6:** the rules for renaming and moving nodes by path, all at once.

**Changed in III.3 and III.9:**
1. **Input names and output names are disjoint, for every kind.** No name is both an input name and an output name of one object. A kernel already refuses one at construction (III.9); a function now refuses one too, where III.3 allows it. One function, `_check_disjoint_names` (§ 5.3), makes the check for every kind once construction ends, and again when a function's first call makes its output names known.
2. **A rename never crosses between inputs and outputs.** Each key of `rename` is exactly one of three things: an input name, the output name, or a path. A key without `/` matches at most one input name, or else the output name or a top-level path, since the two sets are disjoint; a key with `/` is a path. An input's new name is a single name, so renaming an input to a path raises `ValueError`.
3. **Inputs are renamed by name only.** III.9's grouping of a kernel's inputs into a record input by path, and its splitting of one, are removed. A kernel renames its inputs as a function does, one name for one name.
4. **Each name is renamed where it is stored.** An object holds three kinds of names, in two places:

   | Name | Stored in | Renamed by |
   |---|---|---|
   | input names | the spec: `FunctionSpec.input_spec`, `ConditionalDistributionSpec.given_spec`, and a `BatchSpec`'s element's | `TermSpec.rename`, through `InputSpec.rename` |
   | paths, and so the output names of an object whose value spec is a `RecordSpec` | the spec: its value spec | `TermSpec.rename`, through `NamedTree.rename` |
   | the output name of any other object | the object: `_output_name` | `TrackedTerm.rename`, through `_with_output_name` |

   `TermSpec.rename` (§ 5.5) is the one place that sorts a key into an input name or a path, checks it, and builds the renamed spec.
   `TrackedTerm.rename` (§ 5.4) handles the output name itself and passes every other key to `TermSpec`, so `x.rename(m).spec == x.spec.rename(m)` for every key but the output name.
   Each kind supplies only the mechanics, in `_with_renamed_spec`: carrying its values, its callable, or its parent over to the new names.
   A spec's constructor checks that its input names and top-level names are disjoint, and an object checks its inputs against its output name, both with `_check_disjoint_names`, so a rename that gives an input and an output one name raises.
   `PathIndexed` (§ 5.2) has no part in renaming: it selects by path.

```python
k = conditional_distribution(lambda loc: Normal("y", loc, 1.0), given_spec={"loc": NumericArraySpec(())})
k.rename({"loc": "mu"})          # a kernel with the input mu and the output y
k.rename({"loc": "y"})           # ValueError: 'y' would be both an input and an output
k.rename({"loc": "theta/loc"})   # ValueError: an input's new name cannot be a path

def step(state):
    return state + 1.0

Function(step, return_spec=(), output_name="state")   # ValueError: 'state' is both an input and an output
Function(step, return_spec=(), output_name="next_state").rename({"state": "s"})   # input s, output next_state
```

## 5. The API

This section gives every new or changed class, function, and method, with the docstring as it is to be written in the code.
A section headed "new" is a new module or class; one headed "changed" lists only what changes in an existing one.
Docstrings follow `STYLE_GUIDE.md` § 3, and messages follow § 9.3.
A docstring states its contract in full and cites no section of `design/`.

### 5.1 `core/_paths.py` (new): path syntax

The module holds the path syntax that `core/_shapes.py`, `core/named_tree.py`, and `distributions/_views.py` split today.
It moves these unchanged: `_PATH_SEP` and `_last_name` from `core/named_tree.py`; `PathsLike`, `_as_paths`, `_as_path`, `_as_path_arguments`, `_as_joined_path`, `_missing_path`, `_checked_path`, and `_check_distinct_paths` from `core/_shapes.py`.
The old modules import them from here, and nothing else splits or joins a path.
It adds four functions:

```python
"""Path syntax: reading the paths a caller gives, and the relations between paths.

A path is a sequence of names joined by ``/``, as ``"g/b"``. It leads from the
root of a tree to a node: ``"g"`` to the top-level node ``g``, and ``"g/b"`` to
the node ``b`` inside it. Every module that reads, splits, joins, or compares
paths uses this module, so the separator and the reader have one definition.
"""


def _parent_path(path: str) -> str:
    """The path of the parent of the node at *path*, or ``""`` for a top-level node.

    Parameters
    ----------
    path : str
        A well-formed path.

    Returns
    -------
    str
        *path* without its last name.

    Examples
    --------
    >>> _parent_path("g/b/c")
    'g/b'
    >>> _parent_path("g")
    ''
    """


def _is_subpath(path: str, *, base: str) -> bool:
    """Whether the node at *path* is the node at *base* or lies inside it.

    The comparison is by whole names, so ``"gb"`` is not a subpath of ``"g"``.

    Parameters
    ----------
    path : str
        A well-formed path.
    base : str
        A well-formed path.

    Returns
    -------
    bool
        True when *path* equals *base*, or starts with *base* followed by ``/``.

    Examples
    --------
    >>> _is_subpath("g/b", base="g")
    True
    >>> _is_subpath("g", base="g")
    True
    >>> _is_subpath("gb", base="g")
    False
    """


def _path_inside(path: str, *, base: str) -> str:
    """The path of a node as seen from inside the subtree at *base*.

    For a record ``r``, ``r[path]`` is ``r[base][_path_inside(path, base=base)]``.
    The result is *path* with the names of *base* removed from its front.

    Parameters
    ----------
    path : str
        The path of a node strictly inside the node at *base*.
    base : str
        The path of the subtree's root.

    Returns
    -------
    str
        The path from the node at *base* down to the node at *path*.

    Raises
    ------
    ValueError
        If the node at *path* is not strictly inside the node at *base*.

    Examples
    --------
    >>> _path_inside("g/b/c", base="g")
    'b/c'
    >>> _path_inside("g/b/c", base="g/b")
    'c'
    """


def _complement(selected_paths: tuple[str, ...], tree_paths: Iterable[str]) -> tuple[str, ...]:
    """The paths that select what ``without(*selected_paths)`` keeps.

    For a tree ``x``, ``x.without(*selected_paths)`` is
    ``x[list(_complement(selected_paths, tree_paths))]``.

    A node of the tree is in the result when it is not a selected node, does
    not contain a selected node, does not lie inside a selected node, and is
    top-level or has a parent that contains a selected node. So each kept part
    of the tree is named by the path of its largest node. The selected paths
    and the result then split the fields: every field lies at or inside exactly
    one path of the two.

    Parameters
    ----------
    selected_paths : tuple of str
        Paths of nodes of the tree, none inside another.
    tree_paths : iterable of str
        The path of every node of the tree, groups and fields, in canonical
        order.

    Returns
    -------
    tuple of str
        The kept paths, in canonical order; empty when the selected paths
        cover every field.

    Examples
    --------
    For ``Record({"a": 1.0, "b": {"c": 2.0, "d": 3.0}})``, whose node paths are
    ``"a"``, ``"b"``, ``"b/c"``, and ``"b/d"``:

    >>> tree_paths = ("a", "b", "b/c", "b/d")
    >>> _complement(("b/c",), tree_paths)
    ('a', 'b/d')
    >>> _complement(("b",), tree_paths)
    ('a',)
    >>> _complement(("a", "b/c", "b/d"), tree_paths)
    ()
    """
```

### 5.2 `core/_path_indexed.py` (new): the path interface

```python
class PathIndexed:
    """The path interface: ``[]``, ``in``, and ``without``.

    An object or a spec has paths exactly when its value spec is a
    ``RecordSpec``, and they are the paths of that ``RecordSpec``. One
    path selects the part there, of the node's kind; a list of paths selects an
    object of this object's kind restricted to those paths, each node at its
    full path, in canonical order. ``without`` restricts to the complement, so
    a list and ``without`` partition the fields.

    For every path ``p``, ``self.spec[p] == self[p].spec`` on an object, and
    likewise for a list of paths and ``without``.

    A class mixes this in by providing three methods: ``_path_spec``,
    ``_part_at``, and ``_restrict_to``. Renaming is not part of this mixin:
    ``TermSpec.rename`` and ``TrackedTerm.rename`` rename paths together with
    the other names an object holds.

    Examples
    --------
    >>> d = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)
    >>> d["mu"].event_spec
    NumericArraySpec(shape=())
    >>> d.without("sigma").event_spec == d.event_spec[["mu"]]
    True
    >>> "mu" in d
    True
    """

    __slots__ = ()

    # Provided by each class.

    def _path_spec(self) -> RecordSpec | None:
        """The ``RecordSpec`` whose paths are this object's, or None when it has no paths.

        Returns
        -------
        RecordSpec or None
            ``self.value_spec`` when it is a ``RecordSpec``, else None.
        """

    def _part_at(self, path: str) -> Any:
        """The part of this object at *path*, a path of ``_path_spec()``, of the node's kind.

        Parameters
        ----------
        path : str
            A path of this object, already read and checked.

        Returns
        -------
        Any
            The part. On an object, the part's spec is ``self.spec[path]``; on
            a spec, the part is this spec with its value spec replaced by
            ``value_spec[path]``.
        """

    def _restrict_to(self, paths: tuple[str, ...]) -> Self:
        """This object restricted to *paths*, of this object's kind.

        Parameters
        ----------
        paths : tuple of str
            Paths of this object in canonical order, none inside another,
            already read and checked.

        Returns
        -------
        Self
            The object whose value spec is ``self._path_spec()[list(paths)]``.
        """

    # Written once.

    def __getitem__(self, paths: PathsLike, /) -> Any:
        """The part at one path, or this object restricted to a list of paths.

        Parameters
        ----------
        paths : str, or list or tuple of str
            One path, which gives the part there, of the node's kind; or a list
            of paths, which gives an object of this object's kind holding each
            of those nodes at its full path, in canonical order.

        Returns
        -------
        Any
            The part at the path, or the restricted object.

        Raises
        ------
        TypeError
            If this object has no paths, or *paths* is neither a string nor a
            list or tuple of strings.
        KeyError
            If a path is not a path of this object.
        ValueError
            If a path is empty or has an empty name, or a list of paths is
            empty, names a path twice, or holds a path inside another.
        """

    def _select(self, read: str | tuple[str, ...], given: Any) -> Any:
        """``[]`` after the reader: check *read* against ``_path_spec()``, then select.

        ``Batch.__getitem__`` reads positions first and passes a path or
        a list of paths here.

        Parameters
        ----------
        read : str or tuple of str
            The path or paths as ``_as_paths`` returned them.
        given : Any
            The argument as the caller gave it, which messages quote.

        Returns
        -------
        Any
            ``self._part_at(read)`` for one path, else
            ``self._restrict_to(ordered)`` for the list in canonical order.

        Raises
        ------
        TypeError
            If this object has no paths.
        KeyError
            If a path is not a path of this object.
        """

    def __contains__(self, path: object, /) -> bool:
        """Whether *path* is a path of this object.

        Parameters
        ----------
        path : object
            One path.

        Returns
        -------
        bool
            True exactly when ``self[path]`` raises no ``KeyError``.

        Raises
        ------
        TypeError
            If this object has no paths, or *path* is not a string.
        ValueError
            If *path* is empty or has an empty name.
        """

    def without(self, *paths: str) -> Self:
        """This object without the nodes at *paths*, every other node at its path.

        A group left empty goes too. ``self.without(*paths)`` equals
        ``self[_complement(paths, ...)]``.

        Parameters
        ----------
        *paths : str
            Paths of this object, none inside another.

        Returns
        -------
        Self
            The restricted object, of this object's kind.

        Raises
        ------
        TypeError
            If this object has no paths, or a path is not a string.
        KeyError
            If a path is not a path of this object.
        ValueError
            If a path is malformed or repeated, a path is inside another, or the
            paths cover every field.
        """
```

Each class provides the three methods as follows; every spec class derives them from `value_spec` and `_with_value_spec` (§ 5.5):

| Class | `_path_spec()` | `_part_at(p)` | `_restrict_to(ps)` |
|---|---|---|---|
| `NamedTree`, so `Record` and `RecordSpec` | `self.spec`, or `self` for a `RecordSpec` | the node at `p` | today's `_cut_to(ps)` |
| `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, `BatchSpec` | `self.value_spec` if a `RecordSpec` | `self._with_value_spec(self.value_spec[p])` | `self._with_value_spec(self.value_spec[list(ps)])` |
| `Distribution` | `self.event_spec` if a `RecordSpec` | `CoupledMarginal(self, p)` | `CoupledMarginal(self, ps)` |
| `ConditionalDistribution` | `self.event_spec` if a `RecordSpec` | `ConditionalCoupledMarginal(self, p)` (III.9) | the same at `ps` |
| `Function` | `self.return_spec` if a `RecordSpec` | the function with the same inputs that returns `self(w)[p]` | the same at `ps` |
| `RecordBatch` | `self.element_spec` | the column at a field, the batch of a group's records at a group | the cut columns |
| every other batch | `None` | never called | never called |

`NumericArray` and `Opaque` do not mix it in; their `[]` keeps its array and opaque meaning.

### 5.3 `core/_outputs.py` (new): the output spec

```python
def output_spec_of(value_spec: TermSpec | None, output_name: str | None) -> RecordSpec | None:
    """The output spec of an object with this value spec and output name.

    The output spec has one field per output: the value spec itself when it is
    a ``RecordSpec``, and ``RecordSpec({output_name: value_spec})`` otherwise.
    ``x.output_spec`` calls this, and so does code that holds only specs, such
    as an operation's ``result=`` rule.

    Parameters
    ----------
    value_spec : TermSpec or None
        The spec of one value of the object; None while it is not known.
    output_name : str or None
        The object's output name, or None when it has none.

    Returns
    -------
    RecordSpec or None
        *value_spec* when it is a ``RecordSpec``; ``RecordSpec({output_name:
        value_spec})`` when *output_name* is given; ``RecordSpec()`` when
        neither; None when *value_spec* is None.

    Raises
    ------
    ValueError
        If *value_spec* is a ``RecordSpec`` and *output_name* is given, since a
        record's children name its outputs.

    Examples
    --------
    >>> output_spec_of(NumericArraySpec(()), "x")
    RecordSpec(x=NumericArraySpec(shape=()))
    >>> output_spec_of(RecordSpec(mu=(), sigma=()), None)
    RecordSpec(mu=NumericArraySpec(shape=()), sigma=NumericArraySpec(shape=()))
    """


def _check_disjoint_names(
    input_names: Iterable[str], output_names: Iterable[str] | None, *, owner: str
) -> None:
    """Raise ``ValueError`` if a name is both an input name and an output name of one object.

    The one check of disjointness. ``_TrackedTermMeta`` calls it for every
    object with inputs once its constructor returns, and a function calls it
    again when its first call makes its output names known.

    Parameters
    ----------
    input_names : iterable of str
        The object's input names.
    output_names : iterable of str or None
        The object's output names, or None while they are not known, which
        passes.
    owner : str
        How the message names the object: its public class name and label,
        such as ``"Function 'step'"``.

    Raises
    ------
    ValueError
        If a name is in both: "{owner} names [...] both as an input and as an
        output. Rename one of them".

    Examples
    --------
    >>> _check_disjoint_names(("state",), ("next_state",), owner="Function 'step'")
    >>> _check_disjoint_names(("state",), ("state",), owner="Function 'step'")
    Traceback (most recent call last):
    ValueError: Function 'step' names ['state'] both as an input and as an output. Rename one of them
    """
```

### 5.4 `core/tracked.py` (changed): the output name on every tracked term

```python
class TrackedTerm(Immutable, metaclass=_TrackedTermMeta):
    _label: str
    _label_collapse: Collapse | None
    _expression: Expression
    _provenance: Provenance | None
    _output_name: str | None          # new: the output name, or None
    __slots__ = ()

    def _init_tracked(
        self,
        label: str,
        *,
        output_name: str | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        """Initialize the identity state (constructor helper for host classes).

        Assigns the expression of *label* alone, the label, the output name,
        and ``_provenance`` via ``object.__setattr__`` so immutable hosts can
        call it from their constructor. The metaclass checks the output name
        once the constructor returns.

        Parameters
        ----------
        label : str
            The term's label.
        output_name : str or None, optional
            The term's output name, None by default.
        provenance : Provenance or None, optional
            The term's provenance, unset by default.
        """

    @property
    def value_spec(self) -> TermSpec | None:
        """The spec of one value of this term: ``self.spec.value_spec``.

        A law's or kernel's ``event_spec``, a function's ``return_spec``, a
        batch's ``element_spec``, and a record's or base value's own ``spec``.
        None for a function whose return spec is not known yet.
        """

    @property
    def output_spec(self) -> RecordSpec | None:
        """The spec of each of this term's outputs, as one ``RecordSpec``.

        ``output_spec_of(self.value_spec, self._output_name)``: the value spec
        itself when it is a ``RecordSpec``; ``RecordSpec({name: value_spec})``
        for a term with an output name; ``RecordSpec()`` for a term without
        outputs; None while the value spec is not known.
        """

    @property
    def output_names(self) -> tuple[str, ...] | None:
        """The names of this term's outputs, in order.

        ``tuple(self.output_spec.keys())`` at the top level, so ``("x",)`` for
        ``Normal("x", 0.0, 1.0)``, ``("mu", "sigma")`` for a law over
        ``RecordSpec(mu=..., sigma=...)``, ``()`` for a term without outputs,
        and None while ``output_spec`` is None.
        """

    @property
    def outputs(self) -> Record:
        """This term's outputs as a record: each output name maps to its output.

        For a term whose value spec is a ``RecordSpec``, the output ``c`` is
        ``self[c]``; for any other term, its one output is the term itself.
        The record is labeled by this term's label and is empty for a term
        without outputs.

        Raises
        ------
        TypeError
            If ``output_spec`` is None, saying that the outputs of a function
            are known after its first call.
        """

    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Self:
        """This term with its output name, inputs, or paths renamed, all at once.

        The term renames the one name it stores itself, its output name, and
        passes every other key to its spec, which holds the inputs and the
        paths:

        - a key equal to the term's output name renames it, as
          ``_with_output_name`` does;
        - every other key goes to ``self.spec``, whose ``_split_renames`` sorts
          it into an input rename or a path rename and checks it, and whose
          ``_apply_renames`` builds the renamed spec;
        - the term's kind then rebuilds the term over the renamed spec, through
          ``_with_renamed_spec``.

        So for every key but the output name, ``self.rename(m).spec ==
        self.spec.rename(m)``. A top-level path of a term whose value spec is a
        ``RecordSpec`` is also an output name, and renaming it renames that
        output. The result's input names and output names are checked to be
        disjoint.

        Parameters
        ----------
        renames : mapping of str to str, optional
            Old name or path to new name or path.
        **keyword_renames : str
            More pairs, each keyword a top-level name.

        Returns
        -------
        Self
            The renamed term, of this term's kind; this term is unchanged.

        Raises
        ------
        TypeError
            If a key contains ``/`` and this term has no paths.
        KeyError
            If a key is neither the output name, an input name, nor a path of
            this term.
        ValueError
            If a new output name or input name is empty or contains ``/``; if
            the renamed term would have a name that is both an input and an
            output; or if a path rename breaks a rule of ``NamedTree.rename``.

        Examples
        --------
        >>> Normal("x", 0.0, 1.0).rename(x="z").output_names
        ('z',)
        >>> d = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)
        >>> d.rename(mu="loc").output_names
        ('loc', 'sigma')
        >>> d.rename(mu="loc").spec == d.spec.rename(mu="loc")
        True
        """

    def _with_renamed_spec(
        self,
        spec: TermSpec,
        *,
        input_moves: Mapping[str, str],
        path_moves: Mapping[str, str],
    ) -> Self:
        """This term rebuilt over *spec*, its spec with *input_moves* and *path_moves* applied.

        ``rename`` calls it after the spec has checked every move, so it only
        carries the term's values, callable, or parent over to the new names.
        With both mappings empty it returns this term. A kind with inputs or
        paths overrides it; a kind with neither never receives a move.

        Parameters
        ----------
        spec : TermSpec
            The renamed spec, ``self.spec._apply_renames(input_moves,
            path_moves)``.
        input_moves : mapping of str to str
            Each renamed input's old name to its new name.
        path_moves : mapping of str to str
            Each renamed node's old path to its new path, applied all at once as
            ``NamedTree.rename`` applies them.

        Returns
        -------
        Self
            The rebuilt term, whose spec is *spec*.

        Raises
        ------
        NotImplementedError
            If a move is given to a kind that does not override this method.
        """

    def _with_output_name(self, name: str | None) -> Self:
        """A copy of this term with output name *name*, as ``with_label`` copies.

        Parameters
        ----------
        name : str or None
            The new output name, or None for none.

        Returns
        -------
        Self
            The copy.

        Raises
        ------
        ValueError
            If *name* is empty or contains ``/``, or this term's value spec is a
            ``RecordSpec``.
        """
```

`_TrackedTermMeta` checks the output name once `__init__` returns, beside its check of the label:

```python
# in _TrackedTermMeta.__call__, after the label check
name = getattr(instance, "_output_name", None)
if name is not None:
    _check_output_name(name, owner=instance)            # str, non-empty, no "/"; else ValueError
    if isinstance(instance.value_spec, RecordSpec):
        raise ValueError(
            f"{public_class_name(type(instance))} {instance.label!r} has a record value, whose "
            f"children name its outputs, so it takes no output_name. Rename a child instead, "
            f"as rename({{'a': 'b'}})"
        )
inputs = instance.spec._inputs
if inputs is not None and name is not None:
    _check_disjoint_names(
        inputs, (name,), owner=f"{public_class_name(type(instance))} {instance.label!r}"
    )
```

The spec's own constructor has already checked its inputs against its top-level paths (§ 5.5), so the term checks only its output name.
`rename` makes the same check on its result, since `_with_output_name` copies without the constructor.

Each kind's `_with_renamed_spec`:

| Kind | `_with_renamed_spec(spec, *, input_moves, path_moves)` |
|---|---|
| `Record`, `NumericRecord` | not used: `NamedTree.rename` renames a record's paths directly, since a record has no inputs and no output name |
| `RecordBatch`, `NumericRecordBatch` | the batch over *spec* with its columns moved by *path_moves* |
| `Distribution` | today's law rename by *path_moves*, which keeps the family where it can (`Distribution.rename`) |
| `FactoredDistribution` | the joint of its factors, each renamed by the moves it holds |
| `FactoredConditionalDistribution` | the joint of its factors, each factor that takes or gives a moved name renamed |
| every other `ConditionalDistribution` | `_RenamedConditionalDistribution(self, input_moves=input_moves, event_moves=path_moves)` (§ 5.11) |
| `Function` | the function over `_RenamedCallable(self.raw, input_moves)` (§ 5.10), returning `self(w).rename(path_moves)` when *path_moves* is not empty |
| `FunctionBatch`, `DistributionBatch`, `ConditionalDistributionBatch` | the batch of its elements, each renamed by *input_moves* |
| `NumericArray`, `Opaque`, `NumericArrayBatch`, `OpaqueBatch` | the default: these have no inputs and no paths |

`_shallow_copy`, `with_label`, `with_provenance`, and `_with_expression` keep `_output_name`.

### 5.5 `core/_spec_base.py` (changed): the value spec, the inputs, and the renaming of a spec

```python
class TermSpec(ABC):
    @property
    def value_spec(self) -> TermSpec | None:
        """The spec of one value this spec describes.

        A ``DistributionSpec``'s or ``ConditionalDistributionSpec``'s
        ``event_spec``, a ``FunctionSpec``'s ``return_spec``, a ``BatchSpec``'s
        ``element_spec``, and for every other spec the spec itself. None for a
        ``FunctionSpec`` whose return spec is not known yet. Read-only.
        """
        return self

    def _with_value_spec(self, value_spec: TermSpec) -> Self:
        """This spec with its value spec replaced by *value_spec*, of this spec's class.

        Builds the spec of a part: ``S[p]`` is
        ``S._with_value_spec(S.value_spec[p])``. Each of the four specs whose value
        spec is another spec overrides it; for every other spec, the value spec
        is the spec itself, so the result is *value_spec*.

        Parameters
        ----------
        value_spec : TermSpec
            The new value spec.

        Returns
        -------
        Self
            The new spec.
        """
        return value_spec

    @property
    def _inputs(self) -> InputSpec | None:
        """The input spec this spec holds, or None.

        A ``FunctionSpec``'s ``input_spec``, a ``ConditionalDistributionSpec``'s
        ``given_spec``, and a ``BatchSpec``'s element's; None for every other
        spec, and for a ``FunctionSpec`` whose inputs are not known.
        """
        return None

    def _with_inputs(self, inputs: InputSpec) -> Self:
        """This spec with its input spec replaced by *inputs*, of this spec's class.

        Parameters
        ----------
        inputs : InputSpec
            The new input spec.

        Returns
        -------
        Self
            The new spec, whose constructor checks that its input names and its
            top-level names are disjoint.

        Raises
        ------
        TypeError
            If this spec holds no input spec.
        """
        raise TypeError(f"a {type(self).__name__} has no inputs to replace")

    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Self:
        """This spec with inputs or paths renamed, all at once.

        A spec holds two kinds of names: the names of its inputs, in
        ``_inputs``, and the paths of its value spec, when that is a
        ``RecordSpec``. Each key is exactly one of the two. A key without ``/``
        is an input name when it names an input, and a top-level path
        otherwise; the two are disjoint, so no key is both. A key with ``/`` is
        a path. An input is renamed to a single name, and a path is renamed or
        moved as ``NamedTree.rename`` does. A ``RecordSpec`` has no inputs, and
        its ``rename`` is ``NamedTree.rename``.

        Parameters
        ----------
        renames : mapping of str to str, optional
            Old input name or path to new input name or path.
        **keyword_renames : str
            More pairs, each keyword a top-level name.

        Returns
        -------
        Self
            ``self._apply_renames(*self._split_renames(pairs))`` for the pairs
            given; this spec is unchanged.

        Raises
        ------
        TypeError
            If a key contains ``/`` and this spec has no paths.
        KeyError
            If a key is neither an input name nor a path of this spec.
        ValueError
            If an input is renamed to a path or to an empty name; if the renamed
            spec would have a name that is both an input and a top-level path;
            or if a path rename breaks a rule of ``NamedTree.rename``.

        Examples
        --------
        >>> spec = FunctionSpec(InputSpec(loc=()), return_spec=RecordSpec(y=()))
        >>> spec.rename(loc="mu", y="z")
        FunctionSpec(input_spec=InputSpec(mu=NumericArraySpec(shape=())), return_spec=RecordSpec(z=NumericArraySpec(shape=())))
        >>> spec.rename(loc="g/mu")
        Traceback (most recent call last):
        ValueError: cannot rename the input 'loc' to the path 'g/mu'. An input is renamed to a single name, such as 'mu'
        """

    def _split_renames(
        self, pairs: Mapping[str, str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Sort and check rename pairs into input renames and path renames.

        *pairs* is what ``_checked_path_renames`` returns for the caller's
        ``renames`` and ``keyword_renames``. A key that names an input of
        ``_inputs`` is an input rename, and every other key is a path of
        ``_path_spec()``. ``TrackedTerm.rename`` calls this after removing the
        term's output name from *pairs*.

        Parameters
        ----------
        pairs : mapping of str to str
            Old name or path to new name or path, in the order given.

        Returns
        -------
        tuple of (dict of str to str, dict of str to str)
            The input renames, then the path renames, each in the order given.

        Raises
        ------
        TypeError
            If a key contains ``/`` and this spec has no paths.
        KeyError
            If a key is neither an input name nor a path of this spec.
        ValueError
            If an input is renamed to a path or to an empty name.
        """

    def _apply_renames(
        self, input_moves: Mapping[str, str], path_moves: Mapping[str, str]
    ) -> Self:
        """This spec with checked input renames and path renames applied.

        The inputs are renamed by ``InputSpec.rename``, and the value spec by
        ``NamedTree.rename``, then the spec is rebuilt with ``_with_inputs`` and
        ``_with_value_spec``. The rebuilt spec's constructor checks that its
        input names and its top-level names are disjoint.

        Parameters
        ----------
        input_moves : mapping of str to str
            Each renamed input's old name to its new name.
        path_moves : mapping of str to str
            Each renamed node's old path to its new path.

        Returns
        -------
        Self
            The renamed spec.

        Raises
        ------
        ValueError
            If the renamed spec would have a name that is both an input and a
            top-level path, or a path rename breaks a rule of
            ``NamedTree.rename``.
        """
```

`FunctionSpec`, `ConditionalDistributionSpec`, and `BatchSpec` override `_inputs` and `_with_inputs`.
The constructors of `FunctionSpec` and `ConditionalDistributionSpec` call `_check_disjoint_names(inputs, top_level_names, owner=...)`, where the top-level names are those of the value spec when it is a `RecordSpec`; `BatchSpec` inherits the check from its element spec.

### 5.6 `core/_conformance.py` (new) and `TermSpec` (changed): conformance

```python
"""Conformance, unification, and refinement of specs, within one dimension scope.

A spec ``A`` conforms to a spec ``B`` when every value that conforms to ``A``
conforms to ``B``. The check is one-way, from what is received to what is
declared where it is received. Two specs unify when some spec ``U`` conforms to
both and a value conforms to ``U`` exactly when it conforms to both; ``U`` is
their unification. Refinement takes a declared spec and a produced spec that
conforms to it, and returns the declared spec with its dimension names bound and
its unset fields filled from the produced spec.

The rules by kind, for ``A`` conforming to ``B``:

- arrays: equal ranks, each axis agreeing in the scope; when both set a dtype,
  ``A``'s casts to ``B``'s by NumPy's same-kind rule; a support is never
  checked;
- opaque values: ``B``'s type is unset or a superclass of ``A``'s, and ``B``'s
  metadata is unset or equal to ``A``'s;
- records: the same child names in the same order, each child of ``A``
  conforming to the child of ``B``;
- laws: ``A``'s event spec conforms to ``B``'s;
- kernels and functions: the same input names and the same optional inputs;
  each input spec of ``B`` conforms to ``A``'s spec for that input, since a map
  that accepts more conforms; ``A``'s event or return spec conforms to
  ``B``'s; a side that ``B`` leaves unknown accepts any ``A``, and a side that
  only ``A`` leaves unknown does not conform;
- batches: the same levels, and ``A``'s element spec conforms to ``B``'s.

Unification follows the same rules in both directions, with three differences:
two arrays take the narrower dtype, so ``int32`` and ``float32`` unify to
``int32``, and keep a support only when both set the same one; two opaque specs
unify when their types are equal or one is unset, and likewise their metadata;
two kernels or functions unify only when their input specs are equal.

No spec holds a name, so names take no part in any relation.
"""


class DimensionScope:
    """The sizes bound to dimension names where specs meet.

    A name binds to a size the first time it meets one, and a later size that
    disagrees raises. The same name on both sides of an axis is one dimension
    and may stay unbound; two different unbound names at one axis raise. Every
    relation takes a scope, and a caller that checks several specs together,
    such as the inputs of one call or the operands of one merge, passes the same
    scope to each. A relation that raises leaves its scope unchanged.

    Parameters
    ----------
    sizes : mapping of str to int, optional
        Sizes already bound, by dimension name.

    Raises
    ------
    TypeError
        If a name is not a string or a size is not an integer.
    ValueError
        If a name is not an identifier or a size is negative.

    Examples
    --------
    >>> scope = DimensionScope()
    >>> NumericArraySpec((3,)).conforms_to(NumericArraySpec(("n",)), scope)
    True
    >>> scope.sizes
    mappingproxy({'n': 3})
    """

    __slots__ = ("_sizes",)

    def __init__(self, sizes: Mapping[str, int] | None = None) -> None: ...

    @property
    def sizes(self) -> Mapping[str, int]:
        """The bound sizes, by dimension name, as a read-only mapping."""

    def agree(self, first: int | str, second: int | str, *, path: str) -> int | str:
        """The one size or name that two dimensions meeting at one axis agree on.

        A name already bound stands for its size. A name meeting a size binds
        to it. Two equal names stay one unbound name.

        Parameters
        ----------
        first : int or str
            The dimension on one side: a size or a name.
        second : int or str
            The dimension on the other side.
        path : str
            The location of the axis, which messages name.

        Returns
        -------
        int or str
            The agreed size, or the shared unbound name.

        Raises
        ------
        ValueError
            If two sizes differ, a bound name meets another size, or two
            different unbound names meet.
        """

    def copy(self) -> DimensionScope:
        """A scope with the same bound sizes, which binds independently of this one."""

    def _adopt(self, other: DimensionScope) -> None:
        """Take *other*'s bound sizes, after a relation checked on *other* succeeds."""


def unify(first: TermSpec, second: TermSpec, scope: DimensionScope | None = None) -> TermSpec:
    """The unification of two specs: the spec a value conforms to exactly when it conforms to both.

    Parameters
    ----------
    first : TermSpec
        One declaration.
    second : TermSpec
        The other declaration.
    scope : DimensionScope, optional
        The scope the two share, which receives the sizes the unification
        binds; a new scope by default.

    Returns
    -------
    TermSpec
        The unification, with the dimensions it binds substituted.

    Raises
    ------
    TypeError
        If either argument is not a ``TermSpec``.
    ValueError
        If the two do not unify by the rules of this module, naming the first path where
        they differ.

    Examples
    --------
    >>> unify(NumericArraySpec((), dtype=np.float32), NumericArraySpec((), dtype=np.int32))
    NumericArraySpec(shape=(), dtype=int32)
    """


def _check_conforms(actual: TermSpec, declared: TermSpec, scope: DimensionScope, path: str) -> None:
    """Raise ``ValueError`` naming *path* and the difference, unless *actual* conforms to *declared*.

    The one check behind ``conforms_to``, ``refine``, ``is_valid``, a merge's
    condition 6, and a function's check of its result. It checks on a copy of
    *scope* and adopts the copy's sizes only on success.

    Parameters
    ----------
    actual : TermSpec
        The spec of what is received.
    declared : TermSpec
        The spec declared where it is received.
    scope : DimensionScope
        The shared scope.
    path : str
        The location of *actual*, which messages name.

    Raises
    ------
    ValueError
        If *actual* does not conform to *declared* by the rules of this module.
    """
```

`TermSpec` gains three public methods and two private methods, which every spec class implements:

```python
class TermSpec(ABC):
    def conforms_to(self, declared: TermSpec, scope: DimensionScope | None = None) -> bool:
        """Whether every value that conforms to this spec conforms to *declared*.

        Parameters
        ----------
        declared : TermSpec
            The spec declared where a value of this spec is received.
        scope : DimensionScope, optional
            The shared scope, which receives the sizes the check binds when it
            succeeds; a new scope by default.

        Returns
        -------
        bool
            True when this spec conforms to *declared*.

        Raises
        ------
        TypeError
            If *declared* is not a ``TermSpec``.
        """

    def refine(self, produced: TermSpec, scope: DimensionScope | None = None) -> Self:
        """This declared spec refined by a produced spec that conforms to it.

        The result is this spec with its dimension names bound from *produced*
        and with each dtype, support, type, or metadata it leaves unset taken
        from *produced*. A function's first call refines its return spec this
        way.

        Parameters
        ----------
        produced : TermSpec
            The spec of what was produced.
        scope : DimensionScope, optional
            The shared scope; a new scope by default.

        Returns
        -------
        Self
            The refined spec.

        Raises
        ------
        TypeError
            If *produced* is not a ``TermSpec``.
        ValueError
            If *produced* does not conform to this spec, naming the first path
            where it differs.
        """

    def bind_dims_from_value(self, value: Any) -> Self:
        """This spec refined by the spec of *value*: ``self.refine(spec_of(value))``.

        Parameters
        ----------
        value : Any
            A tracked term, whose spec is read, or a raw value, whose spec is
            inferred as the kind's constructor infers it.

        Returns
        -------
        Self
            The refined spec.

        Raises
        ------
        ValueError
            If *value* does not conform to this spec.
        """

    def _conform(self, declared: TermSpec, scope: DimensionScope, path: str) -> None:
        """Raise ``ValueError`` naming *path* unless this spec conforms to *declared*.

        Each class implements its kind's rule from the module docstring of
        ``core/_conformance.py``, comparing *declared*'s
        class first; dimensions meet through ``scope.agree``.

        Parameters
        ----------
        declared : TermSpec
            The declared spec.
        scope : DimensionScope
            The scope, which this method binds into.
        path : str
            The location of this spec, which messages name.
        """

    def _unify(self, other: TermSpec, scope: DimensionScope, path: str) -> TermSpec:
        """The unification of this spec with *other*, by its kind's rule in ``core/_conformance.py``.

        Parameters
        ----------
        other : TermSpec
            The other declaration.
        scope : DimensionScope
            The scope, which this method binds into.
        path : str
            The location, which messages name.

        Returns
        -------
        TermSpec
            The unification.

        Raises
        ------
        ValueError
            If the two do not unify.
        """
```

`is_valid(value)` becomes conformance: a tracked term conforms when its spec does, and a raw value is checked as today.
`FunctionSpec.is_valid(len)` is therefore false, since a raw callable carries no spec.
`bind_dims_from_spec` is removed, and its callers call `refine`.
`_bind_dims_from_spec`, `_unify_specs`, `_unify_array_shape`, `_unification`, `_unify_input_types`, `_unify_declarations`, and `ConditionalDistributionSpec._unify_with` are deleted.
`probpipe` exports `DimensionScope` and `unify`.

### 5.7 The spec classes (changed)

```python
class DistributionSpec(TermSpec):
    """The spec of a law: the spec of one draw.

    Parameters
    ----------
    event_spec : TermSpec or tuple
        The spec of one draw. A tuple is a ``NumericArraySpec`` of that shape.

    Raises
    ------
    TypeError
        If *event_spec* is not a ``TermSpec`` or a tuple.

    Examples
    --------
    >>> Normal("x", 0.0, 1.0).spec
    DistributionSpec(event_spec=NumericArraySpec(shape=()))
    """

    event_spec: TermSpec


class ConditionalDistributionSpec(TermSpec):
    """The spec of a kernel: its inputs and the spec of one draw at each input value.

    Parameters
    ----------
    given_spec : InputSpec, Mapping, or RecordSpec
        The inputs, read as ``InputSpec`` reads them.
    event_spec : TermSpec or tuple
        The spec of one draw. A tuple is a ``NumericArraySpec`` of that shape.

    Raises
    ------
    TypeError
        If *given_spec* is not a source of inputs, or *event_spec* is not a
        ``TermSpec`` or a tuple.
    """

    given_spec: InputSpec
    event_spec: TermSpec


class FunctionSpec(TermSpec):
    """The spec of a function: its inputs and the spec of what it returns.

    Parameters
    ----------
    input_spec : InputSpec or None, optional
        The inputs. None leaves them unknown.
    return_spec : TermSpec, tuple, or None, optional
        The spec of the returned value. A tuple is a ``NumericArraySpec`` of
        that shape. None leaves it unknown until the function's first call.

    Raises
    ------
    TypeError
        If *input_spec* is not an ``InputSpec`` or None, or *return_spec* is
        not a ``TermSpec``, a tuple, or None.
    """

    input_spec: InputSpec | None
    return_spec: TermSpec | None


class BatchSpec(TermSpec):
    """The spec of a batch: the spec of one element, and the named levels.

    Parameters
    ----------
    element : TermSpec or tuple
        The spec of one element. A tuple is a ``NumericArraySpec`` of that shape.
    levels : mapping of str to shape, optional
        The levels, by name, positionally.
    **level_shapes : shape
        The levels by keyword.

    Raises
    ------
    TypeError
        If *element* is not a ``TermSpec`` or a tuple, or the levels are
        malformed.
    """

    element_spec: TermSpec
```

Each of the four overrides `value_spec` and `_with_value_spec`, and mixes in `PathIndexed` (§ 5.2).
Their equality, hash, and repr compare and show these fields only, so no spec shows or compares a name.

`InputSpec.__init__` changes its default and refuses `None`:

```python
_NO_INPUTS = object()   # InputSpec() was given no source: the input spec with no inputs


class InputSpec(Mapping[str, TermSpec]):
    """An immutable, flat mapping of input names to term specs.

    Parameters
    ----------
    inputs : InputSpec, Mapping, or RecordSpec, optional
        Inputs in their source's order. A ``RecordSpec`` supplies one input per
        top-level child, such as ``x.output_spec``; another ``InputSpec``
        preserves its optional flags. With no argument, there are no inputs.
    **inputs_by_name : TermSpec or tuple
        Alternatively, the inputs by keyword.

    Raises
    ------
    TypeError
        If *inputs* is None: "cannot read inputs from None, which stands for a
        spec not known yet. Write InputSpec() for no inputs". Also if the
        source is unsupported, or positional and keyword forms are combined.
    ValueError
        If an input name is not a non-keyword Python identifier.
    """

    def __init__(
        self,
        inputs: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec = _NO_INPUTS,
        /,
        **inputs_by_name: TermSpec | tuple,
    ) -> None: ...
```

`Function(input_spec=None)` keeps its meaning, "read the inputs from the signature", and never passes `None` to `InputSpec`.

### 5.8 Constructors (changed)

Every constructor that took an `OutputSpec` takes a bare spec and an `output_name`.
The table gives each signature change; the docstring entries follow it.

| Constructor | Parameters before | Parameters after |
|---|---|---|
| `Distribution.__init__` | `event_spec: OutputSpec \| TermSpec` | `event_spec: TermSpec, *, output_name: str \| None = None` |
| `ConditionalDistribution.__init__` | `given_spec: ... \| OutputSpec, event_spec: OutputSpec \| TermSpec` | `given_spec: InputSpec \| Mapping[str, TermSpec \| tuple] \| RecordSpec, event_spec: TermSpec, *, output_name: str \| None = None` |
| each TFP family, as `Normal(output_name, loc, scale, *, label=None, event_spec=None)` | `event_spec: OutputSpec \| None` | `event_spec: TermSpec \| None` |
| `distribution(...)` | `event_spec: OutputSpec \| TermSpec, output_name: str \| None` | `event_spec: TermSpec, output_name: str \| None` |
| `conditional_distribution(fn, ...)` | `given_spec: ... \| OutputSpec, event_spec: OutputSpec \| TermSpec \| None` | `given_spec: InputSpec \| Mapping[str, TermSpec \| tuple] \| RecordSpec \| None, event_spec: TermSpec \| None, output_name: str \| None = None` |
| `EmpiricalDistribution.__init__`, `KDEDistribution.__init__` | `output_name: str \| None, event_spec: OutputSpec \| None` | `output_name: str \| None, event_spec: TermSpec \| None` |
| `BootstrapReplicateDistribution`, `BootstrapDistribution`, `RandomFunction`, `RandomMeasure`, `glm_likelihood`, `GLMFamily.build` | `event_spec: OutputSpec \| None` | `event_spec: TermSpec \| None` |
| `GaussianRandomFunction`, `GaussianProcess`, `LinearBasisFunction` | `output_spec: OutputSpec \| None, event_spec: OutputSpec \| None` | `return_spec: TermSpec \| None, event_spec: TermSpec \| None` |
| `Function.__init__`, `function(...)` | `input_spec: ... \| OutputSpec, output_spec: OutputSpec \| TermSpec \| None` | `input_spec: InputSpec \| Mapping[str, TermSpec \| tuple] \| RecordSpec \| None, return_spec: TermSpec \| None, output_name: str \| None = None` |
| `EvaluationPoints.__init__` | `output_spec: OutputSpec \| None` | `return_spec: TermSpec \| None, output_name: str \| None = None` |
| `NumericArray.__init__`, `Opaque.__init__` | (none) | `output_name: str \| None = None`, keyword-only |
| `NumericArrayBatch`, `OpaqueBatch`, `FunctionBatch`, `DistributionBatch`, `ConditionalDistributionBatch` | (none) | `output_name: str = "element"`, keyword-only |
| `RecordBatch`, `NumericRecordBatch` | unchanged | unchanged: a record element has no output name |

The new and changed docstring entries:

```python
# Distribution.__init__, ConditionalDistribution.__init__, distribution, EmpiricalDistribution,
# KDEDistribution
"""
event_spec : TermSpec
    The spec of one draw. A ``RecordSpec`` names the outputs by its children.
output_name : str, optional
    The name of the law's one output, required when *event_spec* is not a
    ``RecordSpec`` and refused when it is.
"""

# each TFP family, as Normal
"""
event_spec : TermSpec, optional
    A declared spec of one draw, which the family's own spec refines, such as
    ``NumericArraySpec((3,), dtype=np.float32)``. By default, the family's spec.
"""

# conditional_distribution
"""
output_name : str, optional
    The name of the kernel's one output when its event is not a record. By
    default, the output name of the law the function returns.
"""

# Function.__init__, function, EvaluationPoints
"""
return_spec : TermSpec, tuple, or None, optional
    The spec of what the function returns. A tuple is a ``NumericArraySpec`` of
    that shape. None leaves it unknown until the first call, which refines it.
output_name : str, optional
    The name of the function's one output when its return value is not a
    record. By default, the callable's name, or ``"result"`` when that is not an
    identifier. A given name with a record return value raises ``ValueError``;
    the default name is dropped.
"""

# GaussianRandomFunction, GaussianProcess, LinearBasisFunction
"""
return_spec : TermSpec, optional
    The spec of what each drawn function returns. Each drawn function takes the
    law's output name.
"""

# NumericArray.__init__, Opaque.__init__
"""
output_name : str, optional
    The value's output name, under which a merge or an input receives it. None
    by default, so the value has no outputs.
"""

# NumericArrayBatch, OpaqueBatch, FunctionBatch, DistributionBatch, ConditionalDistributionBatch
"""
output_name : str
    The batch's output name, ``"element"`` by default.
"""
```

### 5.9 `operations/_operation.py` (changed): the result of an operation

```python
class Declared(NamedTuple):
    """The value spec and output name an operation's ``result=`` rule plans for its result.

    Attributes
    ----------
    value_spec : TermSpec
        The spec of one value of the result.
    output_name : str or None
        The result's output name, None when the value spec is a ``RecordSpec``
        or the result has no name.
    """

    value_spec: TermSpec
    output_name: str | None
```

The rule of `@operation(result=rule)` takes the arguments it takes today and returns a `Declared` or `None`, where it returns an `OutputSpec` or `None` today.
The operation builds its result with the planned name, and checks the result's spec against `value_spec` with `_check_conforms`.
An output spec cannot replace `Declared`, since `RecordSpec(y=NumericArraySpec(()))` is the output spec both of `Record({"y": 1.0})` and of `NumericArray(1.0, label="y", output_name="y")`.

**Decision 2: names.**
- **Recommendation.** `value_spec`, `return_spec` (today's `FunctionSpec.output_spec`, which no longer names an output spec), `output_names`, `outputs`, and `output_spec`. The constructor keyword stays `output_name=`, the canonical name of `design/glossary.md`.
- **Alternative.** `result_spec` for the return spec, which avoids "return" as a noun.

### 5.10 `values/_function_base.py` (changed): renaming a function's inputs

A `Function` binds every call through `self.signature`, the Python signature of its callable, at every binding site (`functions/_call.py`, `functions/_boundary.py`, `functions/_rules.py`, `functions/_function.py`, and `values_to_bound_arguments`).
So a function with renamed inputs is a new `Function` over a callable whose signature has the new names, and no binding site changes.

```python
class _RenamedCallable:
    """A callable with renamed keyword parameters, which calls *wrapped* under its own names.

    Its signature is *wrapped*'s with each parameter in *moves* renamed, and
    each parameter keeps its kind, default, and annotation, so a ``Function``
    built on it binds the new names. A call passes positional arguments through
    unchanged, maps each keyword argument back to its old name, and calls
    *wrapped*. It holds only *wrapped* and *moves*, so it traces under JAX
    whenever *wrapped* does, and it pickles whenever *wrapped* does.

    Parameters
    ----------
    wrapped : Callable
        The callable whose parameters are renamed. A ``_RenamedCallable`` is
        unwrapped first and its moves composed, so renames never nest.
    moves : mapping of str to str
        Old parameter name to new parameter name.

    Raises
    ------
    ValueError
        If a key is not a named parameter of *wrapped*, a new name is not a
        Python identifier, or a new name is the name of another parameter.

    Examples
    --------
    >>> def scale(x, s=2.0):
    ...     return x * s
    >>> renamed = _RenamedCallable(scale, {"s": "factor"})
    >>> inspect.signature(renamed)
    <Signature (x, factor=2.0)>
    >>> renamed(3.0, factor=4.0)
    12.0
    >>> _RenamedCallable(renamed, {"factor": "k"}).__wrapped__ is scale
    True
    """

    __slots__ = ("__name__", "__signature__", "__wrapped__", "_moves")

    def __init__(self, wrapped: Callable[..., Any], moves: Mapping[str, str]) -> None: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    @property
    def moves(self) -> Mapping[str, str]:
        """Each original parameter name to its current name, composed over every rename."""
```

```python
class Function:
    def _with_renamed_spec(
        self,
        spec: FunctionSpec,
        *,
        input_moves: Mapping[str, str],
        path_moves: Mapping[str, str],
    ) -> Function:
        """This function rebuilt over *spec*, its spec with the checked moves applied.

        The result is a new ``Function`` whose spec is *spec*, with the same
        label, output name, ``traceable``, ``vectorized``, and
        ``differentiable``. Its callable is ``_RenamedCallable(self.raw,
        input_moves)`` when *input_moves* is not empty, so it binds the new
        input names; when *path_moves* is not empty, the callable's result is
        renamed by ``rename(path_moves)`` before it is returned. Its ``bind=``
        values are keyed by the new input names, its controls are carried over
        unchanged, and its provenance records the rename.

        Parameters
        ----------
        spec : FunctionSpec
            The renamed spec.
        input_moves : mapping of str to str
            Each renamed input's old name to its new name.
        path_moves : mapping of str to str
            Each renamed path of the return value's old path to its new path.

        Returns
        -------
        Function
            The renamed function; this function is unchanged.
        """
```

The fingerprint of a `_RenamedCallable` is that of `__wrapped__` together with `moves`, so two functions that differ only in their input names have different fingerprints.
`core/_fingerprint.py` gains that one case.

### 5.11 `distributions/_views.py` (changed): the renamed kernel

A kernel that is not factored renames its inputs or its event paths through one view, which translates at its boundary.
Today's view also groups and splits inputs by path; with flat input names, it reduces to a translation of keys.

```python
class _RenamedConditionalDistribution(ConditionalDistribution):
    """A kernel with renamed inputs or event paths, which translates to *parent* at its boundary.

    Binding or conditioning on given values maps each given key from its new
    name back to the parent's input name, calls the parent, and renames the law
    or kernel the parent returns by *event_moves*. Each conditional capability
    of the parent is claimed, translated the same way, and carries the parent's
    guard. Its input spec is the parent's with *input_moves* applied, and its
    event spec is the parent's renamed by *event_moves*. Renaming it again
    composes the new moves onto *parent*, so views never nest.

    Parameters
    ----------
    parent : ConditionalDistribution
        The kernel whose names are renamed.
    input_moves : mapping of str to str
        The parent's input name to this kernel's input name, for each renamed
        input.
    event_moves : mapping of str to str
        The parent's event path to this kernel's event path, applied all at once
        as ``NamedTree.rename`` applies them.

    Raises
    ------
    ValueError
        If the renamed kernel would have a name that is both an input and an
        output.
    """

    def __init__(
        self,
        parent: ConditionalDistribution,
        *,
        input_moves: Mapping[str, str],
        event_moves: Mapping[str, str],
    ) -> None: ...
```

Its code for grouped and split inputs is deleted: `_leaf_values`, the `origins` mapping of each leaf to the parent's leaf, the bindings held until the rest of a split input is bound, and, in `distributions/_conditional.py`, `_moved_inputs`, `_is_input_path`, `_check_rename_source`, and `_check_disjoint_sides`.

## 6. Step P2 in detail: the output name out of specs

This section lists every site that step P2 changes, beyond the API of § 5.

**The name of a result.** Each operation names the term it returns by one of five rules:

| Rule | The result's output name | Where it applies |
|---|---|---|
| keep | the operand's | `mean`, `variance`, `standard_deviation`, `quantile`; `expectation` (the integrand's); `convert`; `mixture` (the kernel's); `sample`, for each draw and for the batch of draws; `condition_on`, when the posterior's value spec is not a `RecordSpec`; `make_posterior` (the target's); the converters and the resampling laws (the source's) |
| part | the last name of the path, when the node is not a `RecordSpec` | `x[p]` on every kind; `marginal(d, p)`; the column batches of a `RecordBatch`; `EmpiricalDistribution._marginal` |
| own | a fixed name | `cov`, `"cov"`; each density, the operation's name; `log_det_jacobian`, `"log_det_jacobian"`; the Monte Carlo law, `"draws"`; `MinibatchedDistribution`, `"target"`, and its parameters, `"parameters"`; the BayesFlow likelihood, `_observation_input(prior)`; a posterior with no target name, `"posterior"` |
| call | the function's | the value a `Function` call returns; `pushforward` (the map's); the batch of a sweep or a broadcast, and the law of a lifted call |
| none | none, since the result's value spec is a `RecordSpec` | `&` on every kind; `condition_on` when several outputs remain |

Three results change: the batch of draws, the column batches of a `RecordBatch`, and the batch of a sweep name their element `"element"` today, and after P2 they carry the name their rule gives.

A function whose return spec is not known keeps its output name until its first call.
If that call returns a record, a default name is dropped and a given name raises `TypeError`, as `OutputSpec(y=None).with_term_spec` raises today.
The function stores whether its name was given, in a private `_output_name_given: bool`.

**Readers.** Each read of the name moves from the spec to the object:

| Today | After P2 |
|---|---|
| `_root_output_name(x.event_spec)`, `f._output_name` | `x._output_name` |
| `x.event_spec.outputs` | `x.output_spec.children` |
| `_has_record_outputs(x.event_spec)` | `isinstance(x.value_spec, RecordSpec)` |
| `d.event_spec.term_spec`, `f.output_spec.term_spec`, `b.element_spec.term_spec` | `d.event_spec`, `f.return_spec`, `b.element_spec`, or `x.value_spec` |
| `OutputSpec.default(spec, output_name=n)` | `spec`, with `output_name=n` passed to the constructor |
| `declared.with_term_spec(produced)` | `declared.refine(produced)`, from step P1 |
| `_unnamed_declaration(f.output_spec)` | `f.return_spec` |

Four groups of readers change behaviour:
1. **The name is no longer a path.** `Distribution.__getitem__`, the factored marginal, `CoupledMarginal._parent_request`, `_draw_segments`, `_requested_nodes`, and `_requested_paths` drop the case of the root name, so `Normal("x", 0.0, 1.0)["x"]` is a `TypeError`.
2. **A spec's name is no longer compared with an object's.** The checks in `_unify_declarations`, `FunctionSpec._bind_dims_from_spec`, `_whole_term_event`, `glm_likelihood`, the Gaussian random function's `_declarations`, and `_agreed_event_spec` go. The checks that compare two objects' names stay: the components of a mixture, `_packaged_alike` in `condition_on`, and `_target_atoms` in reweighting.
3. **Factored laws lose the packaged form.** `FactoredDistribution(..., _output_name=)` and `FactoredConditionalDistribution(..., _output_name=)` go. A gathered factor is renamed into its group instead, so its value spec is `RecordSpec(g=RecordSpec(...))`. `_SoleField` gives the law at one field that field's name as its output name.
4. **Labels and levels read the object.** The array atoms of `EmpiricalDistribution`, the array backend of a TFP family, the default level of a bootstrap, the atom level of a broadcast, and the label of a posterior's atoms read `x._output_name`.

`_law_repr` reads `x._output_name`, so `Normal('x', ...)` prints as today, and a spec's repr shows no name.

**A consumer that needs a known output spec.** It refuses `None` for its own reason, with a message that names the object and says its return spec is known after its first call.
A merge refuses it because its conformance check needs a spec, and `InputSpec` refuses it because `None` is not a source of inputs.

**Equality, fingerprints, and JAX.**
1. A spec's equality and hash compare structure only. II.2's three rules for the name in equality become one: no spec holds a name.
2. A term's equality and hash ignore its output name, as they ignore its label.
3. The fingerprint drops the `OutputSpec` branch and the root name of a `DistributionSpec`, so `Normal("x", 0.0, 1.0)` and `Normal("y", 0.0, 1.0)` share a fingerprint. A result reused by fingerprint is named again by its rule, so it never carries another operand's name.
4. The static data of a `NumericArray` and a `NumericArrayBatch` is `(spec, output_name)`, so a transform keeps the name, and two values that differ only in their names trace separately. A record and a batch of records have no name.

**Deleted.**
- `OutputSpec` and its export from `probpipe`.
- In `core/_specs.py`: `_EXPOSED_SPECS`, `_EXPOSED_OF`, `_exposed_kind`, `_exposed_outputs`, `_root_path`, `_ROOT_NAMES`, `_unification`, `_renamed_exposed`, `_root_output_name`, `_has_record_outputs`, `_VALUES_WITHOUT_RECORD`, `_no_output_record`, `_type_not_known`, `_output_record`, `_unnamed_declaration`, and `_unnamed_exposed`.
- `_complete_event_spec` and `_whole_term_event` in `distributions/_distribution.py`, `_complete_output_spec` in `values/_function_base.py`, the `OutputSpec` cases of `_as_element_spec` in `core/_batch.py`, and `_form_keywords` in `distributions/_factored.py`.

`_merged_output_record` stays, and merges the output specs of its operands.
Five path helpers read an `OutputSpec`: `_read_paths`, `_path_record`, `_has_path`, `_paths_error`, and `_output_spec_at`.
P2 rewrites them over the value spec, and P3 deletes them.

**Commits.** P2 is one PR of four commits, and the suite passes after each:
1. **Accessors, with no change in behaviour:** `_output_name`, `output_names`, `output_spec`, `value_spec`, `_with_value_spec`, and `output_spec_of`, still read from the `OutputSpec` in the spec. Every reader of the tables above moves to them.
2. **Bare specs:** the spec classes of § 5.7, the constructors of § 5.8, `Declared`, the five naming rules, equality, fingerprints, and JAX static data. A script rewrites the `.term_spec` reads.
3. **Deletion:** `OutputSpec`, its export, and the helpers above.
4. **Documents:** `docs/api/input_and_output_specs.md`, `docs/api/extending.md`, the notebooks that build an `OutputSpec` (user guide 01, 03, and 09, and tutorial 06), the `design/` sections of § 8, and the CHANGELOG.

## 7. Tests

1. One test, parametrized over every class of § 5.2, asserts II.6's selection, chaining, partition, and errors, the two changes of § 4.1, and the spec agreement rule on a table of objects, records holding laws in fields included, and asserts the `TypeError` for each object of § 4.2 that has no paths.
2. One table test of `conforms_to`, `unify`, and `refine` per kind, including each case of § 2.2, and a test that a failed relation leaves its `DimensionScope` unchanged.
3. One test that `x.output_spec`, `x.output_names`, and `x.outputs` follow § 4.3 for each row of its table, with `x.output_names == tuple(x.output_spec.keys())`.
4. One test that no module outside `core/_paths.py` splits or joins a path with `/` or `_PATH_SEP`.
5. One table test of each constructor of § 5.8: its default output name, a given name, and its output spec.
6. One table test of the five naming rules of § 6, with a row for each operation it lists.
7. Two laws that differ only in their names have equal specs and equal fingerprints, and a `jax.jit` round trip keeps the name of a `NumericArray` and of a `NumericArrayBatch`.
8. Stating an output name on an object whose value spec is a `RecordSpec` raises `ValueError`, from a constructor or from `rename`, and `probpipe.OutputSpec` no longer exists.
9. A function whose return spec is not known has `output_spec` `None`; `InputSpec()` has no inputs; `InputSpec(None)` and `InputSpec(f.output_spec)` for such a function raise `TypeError`; and a merge with such a function refuses it with a message that names the function.
10. One table test of `rename` over every kind with inputs: each key classified as an input name, the output name, or a path; an input renamed to a path raises `ValueError`; a rename that makes a name both an input and an output raises `ValueError`; and a function or kernel constructed with a name on both sides raises `ValueError`, as does a function whose first call returns a record with a child named like an input.
11. A function's input rename keeps each parameter's default, the function's `traceable` flag and its `jax.jit` result, its `bind=` values under the new names, and its controls; renaming twice gives one `_RenamedCallable` over the original callable; and two functions that differ only in input names have different fingerprints.
12. For every kind and every rename whose keys exclude the output name, `x.rename(m).spec == x.spec.rename(m)`; `TermSpec.rename` on each spec class renames inputs and paths and refuses an input renamed to a path.
13. A renamed kernel binds and conditions under its new input names, renames the law it returns by its event moves, and renamed again wraps the original kernel, not the first view.

## 8. Edits to the other plans and to `design/`

| Plan | Its section | Edit |
|---|---|---|
| `declarations-and-addressing.md` | Part I § 3.2, § 3.5, § 3.7 | Outputs by the two cases of § 4.3; the output name stored on the object; no object without outputs because of its value's kind |
| | Part I § 2.3, § 4.1, § 4.2, and the figure of a held field | A record's paths end at its fields, and an object has paths only when its value type is a `RecordSpec` |
| | § 11, and the technical notes of § 3.5 | `OutputSpec` is removed, and an output spec is a `RecordSpec` |
| | B.6 G1a, G1b, G2 | Not needed: the interim whole form exists because names are stored in specs |
| | B.6 S4b, G3, G5, and G4's `law.without` | Replaced by steps P2 to P4 of § 9; G3's paths through held objects, G4's paths into a record input, and G5's `[]` on batches of laws and of functions are dropped |
| | B.4, rows T30 and I10 | `without` through a law held in a field, and on a batch of laws, is dropped |
| `batch-contract.md` | § 4.3, step B1 | The one check is this plan's `_check_conforms`, from step P1 |
| | § 4.5, § 5.1 | `element=OutputSpec(x=())` becomes `element=(), output_name="x"` |
| | § 4.8 | The lifted-operation rule does not cover `[]` at a path, `without`, or `rename` by path: a batch whose element is not a record has no paths |
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
- II.6, where `Record({"x": 3.0, "g": d})["g/mu"]` is `d["mu"]` and `without` removes a node inside a held term;
- III.3, where `record[path]` reads through a held term.

II.2 also drops paths into a record input: an input spec's `[]` and `without` take input names only, and no input becomes required because a path cut it.
In II.2, `InputSpec`'s signature drops `| None = None` for the sentinel `_NO_INPUTS`, and `None` is refused as a source.
II.6 changes `in` as § 4.1 states, and adds the spec agreement rule.
III.3 drops "a key that names both an input and an output of a function renames both", and states that a function's input names and output names are disjoint, as III.9 states for a kernel.
III.9 drops the grouping and splitting of a kernel's inputs by path: its `rename` renames each input to one name.
IV.2's condition 5, that no name of an operand is both an input and an output, now holds for every operand by construction, so a merge never raises it.

## 9. Steps

Each step is one PR.

1. **P1, conformance:** § 5.6, with contravariant inputs, directional opaque types, and unknown sides by § 4.5; `is_valid` through conformance; `InputSpec.__and__` through `unify`. It fixes the departures of § 4.5, and its only rename is `bind_dims_from_spec` to `refine`. `batch-contract.md` step B1 builds on it.
2. **P2, names out of specs:** § 5.3 to § 5.5 and § 5.7 to § 5.9, by § 6, in its four commits.
3. **P3, one path interface:** § 5.1 and § 5.2 for `Record`, `RecordSpec`, `RecordBatch`, `Distribution`, and the spec classes; `without` on laws; the helpers of Appendix A deleted.
4. **P4, paths and renames on every kind:** § 5.2 for `ConditionalDistribution` and `Function`; § 4.7's renaming, with `TermSpec.rename` and its helpers (§ 5.5), `TrackedTerm.rename` and each kind's `_with_renamed_spec` (§ 5.4), `_check_disjoint_names` (§ 5.3), `_RenamedCallable` (§ 5.10), and the reduced `_RenamedConditionalDistribution` (§ 5.11). Every other `rename` method (on `Distribution`, `ConditionalDistribution`, `RecordBatch`, and the spec classes) is deleted, except `NamedTree.rename` and `InputSpec.rename`, which the two call.
5. **P5, outputs and feeding:** `outputs`, and merge's check through § 4.6.

**What users see.**
- **Breaking:** `OutputSpec` is removed; `event_spec`, `element_spec`, and `return_spec` (today's `FunctionSpec.output_spec`) are bare specs; an object's output names are read with `output_names`, and the stored name is private; `x["x"]` on a law over an array named `x` raises `TypeError`; a function whose inputs accept less than declared no longer conforms; `bind_dims_from_spec` becomes `refine`; `InputSpec(None)` raises `TypeError`, and `InputSpec()` is the input spec with no inputs; a function with a name that is both an input and an output raises at construction; a kernel's inputs can no longer be grouped or split by `rename`, and an input renamed to a path raises.
- **Added:** renaming a function's inputs; paths on kernels and functions whose value spec is a `RecordSpec`, `without` on laws, `output_names`, `outputs`, `conforms_to`, `unify`, `refine`, `DimensionScope`, and `output_name=` on `NumericArray` and `Opaque`.
- **Fixed:** the output name leaves spec equality and fingerprints.

## 10. Questions for the maintainer

1. **Conformance.** Recommended: the public `conforms_to`, `unify`, and `refine`, with II.1's rules (§ 4.5).
2. **Names.** Recommended: `value_spec`, `return_spec`, `output_names`, `outputs`, and `output_spec`, with the constructor keyword `output_name=` (§ 5.9).

## Appendix A. Path helpers that the path interface replaces

| Helper | What it restates |
|---|---|
| `distributions/_views.py`: `_node_at`, `_view_declaration` | the spec at a path, `RecordSpec[p]` |
| `distributions/_views.py`: `_draw_segments`, `_draw_path` | removing the output name from a path, needed only because the name is in the spec |
| `distributions/_views.py`: `_leaf_paths`, `_leaves_at`, `_node_leaves` | `RecordSpec.keys()` and the nodes above each leaf |
| `distributions/_views.py`: `_is_within`, `_below` | `_is_subpath` and `_path_inside` of § 5.1 |
| `distributions/_views.py`: `_projector`; `distributions/_factored.py`: `_extract` | `Record[p]` and `Record[ps]` on raw values |
| `distributions/_views.py`: `_fields_of` | the flat leaf mapping, the inverse of `_unflatten_paths` |
| `distributions/_views.py`: `_ordered_paths_of`; `distributions/_distribution.py`: `_read_event_paths`; `core/_specs.py`: `_read_paths` | reading and ordering a list, `NamedTree._ordered_paths` |
| `core/_specs.py`: `_has_path`, `_path_record`, `_paths_error`, `_output_spec_at` | `NamedTree.__contains__` and its messages, with the output name as a special case |
| `core/_batch.py`: `_has_path`, `_at_paths`; `core/_record_batch.py`: `_has_path`, `_check_path` | the check of a path and the error for an object without paths |
| `distributions/_empirical.py`: `_projection`, `_atoms_at_paths` | the restriction of a batch of atoms, with its own removal of the output name |
| `distributions/_conditional.py`: `_check_rename_source`, `_is_input_path`, `_moved_inputs`, `_check_disjoint_sides` | the classification and check of a rename key, and the disjointness check, which `TermSpec._split_renames` and `_check_disjoint_names` make once |

## Appendix B. Readings this plan adopts

1. **"Feeds", not "binds".** Connecting an output to an input is *feeding* (IV.2). In `design/`, binding supplies values to inputs at a call.
2. **A bare value.** An array or an opaque value standing alone may state an output name with `output_name=` (§ 5.8). Without one it has no outputs, as II.2 says today.
3. **Promotion to a numeric record.** A list of paths that leaves only numeric fields gives a numeric record, since `RecordSpec.__new__` returns a `NumericRecordSpec` when every field is numeric. `numeric-contract.md` § 4.1 makes numeric-ness a property of the spec, so the answer stays the same without a class change.
4. **Coupled and standalone marginals.** `x[p]` gives the coupled marginal, and the operation `marginal(d, p)` the standalone one (VI.8). Both follow the paths of § 4.2, and this plan changes neither.
