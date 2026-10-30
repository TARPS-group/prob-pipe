# Plan: one contract for every batch

> **Status: proposed; not approved.**
> Written on 2026-10-27 against `main` at `97891b7d`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Batches".
> Once approved, its rules move into `design/` (II.1, II.5, III.3, III.10, VI.11) and its steps into the plans of the work after the workshop.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 7):

1. How a batch infers its element spec when none is given (§ 4.4).
2. Whether a raw callable's Python name is its output name in a batch (§ 5.4).
3. Whether a batch admits functions whose output spec is unspecified (§ 5.4).
4. Whether `Batch.stack` becomes public API for every kind (§ 4.6).
5. Whether every batch compares by value (§ 4.9).

---

## 1. Why this plan exists

A batch is ProbPipe's collection type: many values, functions, or laws of one type, laid out along named levels.
Six classes implement it, one per kind of element, and each was written with its own idea of what "of one type" means.

The draws of a law show the intended contract.
`sample(d, sample_shape=100)` returns 100 records over the fields of `d`'s event, so the batch states one element spec and reads its outputs, such as `mu` and `sigma`, from it.

The six classes keep this promise to different degrees.
A batch of laws refuses a law whose event spec differs from the batch's, while a batch of functions accepts any callable:

```python
DistributionBatch([Normal("x", 0.0, 1.0), Normal("y", 0.0, 1.0)], "law")
# TypeError: element 1 of the DistributionBatch does not match the batch's event spec:
#   the batch declares the outputs ['x'] but the law ['y']

FunctionBatch([f, g, h], "variant", label="fs")
# f returns a record, g an array, and h takes two inputs; all three are accepted:
# FunctionBatch('fs', levels={'variant': 3}, element=FunctionSpec())
```

The goal is one contract that every batch meets.
It is stated once in `design/`, implemented once on the base class `Batch`, and tested once over every kind.

## 2. The design today

### 2.1 What `design/` states

II.5 gives the contract of every batch:

1. **One spec.** A batch holds a `BatchSpec`: the element spec, which is the output spec of one element, and the levels. A batch with elements is concrete, and construction refuses free dimensions.
2. **Checked elements.** "Construction checks every element against `element_spec.term_spec` and reports the position that failed."
3. **Inference.** A constructor given no `element=` "infers the spec from the elements, as the element kind's constructor infers a term's spec from its value."
4. **Provenance of the element spec.** An operation stores its maker's output spec of one element, bare values take the raw default, and a marginal batch stores the output spec at its path.
5. **Shared operations.** `Batch` declares `[]` by position and by path, `in`, `output`, `without`, `rename`, `&`, `at_levels`, `with_level_names`, and `raw`.
6. **Outputs.** A batch of records has its elements' children as outputs, a batch of base values one output at the root, a batch of functions its elements' outputs, and a batch of laws or kernels none.

Each kind's section adds its storage and what its elements share:

| Kind | Section | What the elements share |
|---|---|---|
| `NumericArrayBatch` | III.1 | one `NumericArraySpec` |
| `OpaqueBatch` | III.2 | the type every element has, or `None` when the types differ |
| `FunctionBatch` | III.3 | nothing; "an element is a `Function` whatever callable its position holds" |
| `RecordBatch` | III.6 | one `RecordSpec` |
| `DistributionBatch` | III.10 | "one event spec" |
| `ConditionalDistributionBatch` | III.10 | "one `given_spec` and one event spec" |
| `LinOpBatch` | III.4 | "both schemas" |

VI.11 lifts every operation to a batch element by element, and merges two batches over equal levels.

### 2.2 What the design leaves open

The design says that construction checks elements, but four parts of that check are unstated or stated differently by kind.

1. **The relation.** II.1 defines two relations between specs: conformance, which is one-way, and unification, which is symmetric. II.5 does not say which one admits an element.
2. **The scope.** II.1 puts a batch and its element in one dimension scope. It says nothing about two elements of one batch, so whether a symbolic size `n` takes one value across the batch is open.
3. **Inference.** III.2 widens to the shared type. III.10 says "sharing one event spec" without saying how that spec is found. III.3 says nothing.
4. **Functions.** III.3 never requires the elements of a `FunctionBatch` to share a spec. II.5 says a batch of functions "has its elements' outputs", which presumes that they do.

## 3. The code today

### 3.1 The classes

| Class | Storage | How an element is checked | Element spec when `element=` is omitted |
|---|---|---|---|
| `NumericArrayBatch` | one array | dtype cast and trailing shape | read off the array |
| `RecordBatch`, `NumericRecordBatch` | one column per leaf path | each column against its field | read off the columns |
| `OpaqueBatch` | object array | `OpaqueSpec.is_valid` | the shared type, else `OpaqueSpec()` |
| `FunctionBatch` | object array | `callable(element)` | `FunctionSpec()`, which admits every callable |
| `DistributionBatch` | object array | event specs unify, in a new scope per element | the first law's spec |
| `ConditionalDistributionBatch` | object array | specs unify, in a new scope per element | the first kernel's spec |

`LinOpBatch` is not built.
The four object-array classes share `_ObjectBatch` (`core/_object_batch.py`), which checks each element with `element_term_spec.is_valid(element)`.

### 3.2 Where the code agrees with the design

- Levels, `axis_groups`, `at_levels`, `with_level_names`, and the derived labels of views follow II.5 for every class, through the base `Batch` in `core/_batch.py`.
- Positional `[]` and iteration return views for every class.
- `RecordBatch` implements nearly all of II.5 and III.6: marginal batches by path and by list of paths, `rename`, `without`, `&`, `select`, `map`, and equality by value.
- `DistributionBatch` and `ConditionalDistributionBatch` check their elements against the batch's spec, and a sweep of a law-valued function returns a `DistributionBatch`.
- `NumericArrayBatch` refuses a path with the message II.5 asks for: its elements are single values with no paths.

### 3.3 Where the code departs from the design

**D1. A function batch admits functions of any spec, even under a declared one.**
`FunctionSpec.is_valid` checks `callable(value)` alone (`values/_function_base.py:213`), so a declared `element=` is never checked against a stored `Function`:

```python
F = Function(lambda x: {"a": x}, input_spec=InputSpec(x=()),
             output_spec=OutputSpec(RecordSpec(a=())))
G = Function(lambda x: 2 * x, input_spec=InputSpec(x=()),
             output_spec=OutputSpec(result=(3,)))

fb = FunctionBatch([F, G], "variant", label="fs", element=F.spec)   # accepted
fb.element_spec.term_spec.output_spec   # OutputSpec(NumericRecordSpec(a=()))
fb[1].output_spec                       # OutputSpec(result=NumericArraySpec(shape=(3,)))
```

The batch states that every element returns a record `{a}`, while element 1 returns an array named `result`.
The check that catches this exists already and is not called:

```python
F.spec._bind_dims_from_spec(G.spec, {}, "element 1")
# ValueError: element 1 has the outputs ['result'], but the expected FunctionSpec
#   has the outputs ['a']. Make the output names match
```

A raw callable's signature is checked only when its element is read:

```python
b = FunctionBatch([lambda a: a], "variant", label="fs",
                  element=FunctionSpec(input_spec=InputSpec(x=())))   # accepted
b[0]
# ValueError: Function 'fs[variant=0]' declares the inputs ['x'] in input_spec,
#   but its signature has the parameters ['a']. They must match
```

**D2. The spec classes admit values by different rules.**
Each spec class has two checks: `is_valid(value)`, and `_bind_dims_from_spec(spec, bindings, path)`, which checks a spec against a spec in a shared dimension scope.
They disagree for two of the four classes that a batch's elements can carry:

| Spec | `is_valid` | `_bind_dims_from_spec` |
|---|---|---|
| `DistributionSpec` | unification of the event specs | unification of the event specs |
| `ConditionalDistributionSpec` | unification of both specs | unification of both specs |
| `FunctionSpec` | `callable(value)` | input and output names, then unification |
| `BatchSpec` | equality, apart from an opaque type (`_admits`) | levels, then unification |

II.1 says that a distribution, a function, or a batch conforms "when its own spec does", so `is_valid` is conformance for every kind.
`BatchSpec.is_valid` refuses a batch that conforms:

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=4)
x.spec
# BatchSpec(NumericArraySpec(shape=(), dtype=float32, support=real), sample=4)
BatchSpec(NumericArraySpec(()), sample=4).is_valid(x)
# False
```

II.1 says that a spec with no dtype and no support admits every spec of its shape, so the answer should be `True`.

**D3. A batch of laws takes the first law's spec, so its spec depends on the order of its elements.**

```python
DistributionBatch([Normal("x", 0.0, 1.0), HalfNormal("x", 1.0)], "law").event_spec
# OutputSpec(x=NumericArraySpec(shape=(), dtype=float32, support=real))

DistributionBatch([HalfNormal("x", 1.0), Normal("x", 0.0, 1.0)], "law").event_spec
# OutputSpec(x=NumericArraySpec(shape=(), dtype=float32, support=non_negative))
```

The second batch states that every element is supported on the non-negative reals, which is false for its `Normal`.
Unification admits the `Normal` because it never compares supports (II.1), while the stored spec still states one.

**D4. Each element is checked in its own dimension scope.**
`_check_elements` and `_check_declarations` start a new `bindings` for each element, so a symbolic size never binds across the batch:

```python
spec = DistributionSpec(OutputSpec(x=NumericArraySpec(("n",))))
laws = [Normal("x", jnp.zeros(2), 1.0), Normal("x", jnp.zeros(3), 1.0)]
db = DistributionBatch(laws, "law", element=spec)   # accepted
db.spec.free_dims                                   # frozenset({'n'})
```

II.5 says construction refuses a spec with free dimensions.
`NumericArrayBatch` refuses one, and the object batches do not.

**D5. A sweep chooses its result's class by a chain of `isinstance` checks, and loses kinds.**
`_make_stack` (`functions/_result.py:596`) chooses the class of a sweep's result one kind at a time.
Three results depart from the design:

```python
mus = NumericArrayBatch(jnp.array([0.0, 1.0]), "row", label="mus")

@function
def kernel_of(mu):
    return conditional_distribution(lambda s: Normal("y", mu, s), given_spec=InputSpec(s=()))

kernel_of(mus).element_spec
# OutputSpec(element=OpaqueSpec(type=_FunctionKernel)): an OpaqueBatch, not a ConditionalDistributionBatch

@function
def fields_by_row(mu):
    return {"a": mu} if mu < 0.5 else {"b": mu}

fields_by_row(mus)
# TypeError: _make_stack: Records in list have inconsistent fields.
```

The second message names a private function and gives no fix, against `STYLE_GUIDE.md` §9.3.
Third, the sweep passes its declared output spec as `element=` to `FunctionBatch` and `OpaqueBatch`, but not to `DistributionBatch`, which infers its spec from the first law instead.

**D6. Of the operations II.5 gives every batch, only `RecordBatch` has most.**

| Operation | Array | Opaque | Record | Function | Law | Kernel |
|---|---|---|---|---|---|---|
| `output` | – | – | – | – | – | – |
| `rename` | – | – | ✓ | – | – | – |
| `without` | – | – | ✓ | – | – | – |
| `&` | – | – | ✓ | – | – | – |
| `[path]` | refused, as II.5 says | refused | ✓ | "not supported yet" | "not supported yet" | "not supported yet" |
| `==` | identity | identity | by value | identity | identity | identity |

`deferred.md` postpones parts of this table in four separate entries: step S4b (cut before the workshop), step G5, "`&` on batches other than `RecordBatch`", and "`without` on a batch of laws".
Each entry plans its own change for its own kinds.

**D7. The element spec of a batch of functions is in the keyword form.**
`_as_element_spec` (`core/_batch.py:1606`) puts every spec other than a record, a law, a kernel, or a batch of these under the name `element`:

```python
FunctionBatch([F, F], "variant", element=F.spec).element_spec
# OutputSpec(element=FunctionSpec(input_spec=..., output_spec=OutputSpec(NumericRecordSpec(a=()))))
```

II.5 gives every spec except a base spec the positional form.
Under it, this batch's output would be `a`, its elements' output, and not `element`.
The same code names a batch of samples `element` rather than after its law, as `x.spec` in D2 shows; that part is step S4b's work.

**D8. Code outside the batch classes reads their storage by class.**
Fourteen modules outside the batch modules hold 41 `isinstance` checks against concrete batch classes.
Each check serves one of three operations, and each operation is implemented several times:

| Operation | Implementations |
|---|---|
| the columns of a batch, by leaf path | `RecordBatch._raw_columns`; `_batch_columns` and `_draw_columns` in `functions/_broadcast.py`; `mapped_storage` and `_mapped_row` in `functions/_sweep.py`; `_flat_rows` in `distributions/_empirical.py` |
| a batch over checked columns, without a copy | `NumericArrayBatch._over_store`; `_ObjectBatch._over_store`; `RecordBatch._view_over`; `_batch_over` in `core/_record_batch.py` |
| a batch from a list of elements | `RecordBatch.stack`; `_make_stack` and its helpers in `functions/_result.py`; `_stacked_batches` in `core/_record_batch.py`; the object-batch constructors |

A new batch kind must be added at each site, and a missed site fails at run time, as the kernels of D5 do.

**D9. The constructors differ in what they require.**

- `NumericArrayBatch` and `OpaqueBatch` require `label=`, and the other four derive a label.
- Every constructor's `element=` takes a bare spec, so a caller cannot name the elements of a batch of arrays, although `BatchSpec` takes an `OutputSpec`.
- Given no elements and no `element=`, `FunctionBatch` and `OpaqueBatch` take the widest spec, and `DistributionBatch` raises.

## 4. The proposed design

### 4.1 Principles

1. **One homogeneity rule.** Every batch holds elements of one element spec, with one set of output names. A batch of functions keeps the promise a batch of laws keeps.
2. **One check.** One function decides whether an element belongs to a batch: II.1's conformance, in one dimension scope for the whole batch.
3. **One implementation.** What every batch shares is written once, on `Batch`.
4. **Kind-specific code is storage.** A concrete class says how it stores its elements and how its element kind is checked, and nothing the base can derive.

### 4.2 Homogeneity

> **Definition (homogeneous batch).** A batch `b` is *homogeneous* when `b.spec` is concrete and every element `b[i]` conforms to `b.element_spec.term_spec` (II.1), with every element checked in one dimension scope that also holds the batch's levels.
> An element that carries a spec of its own, such as a law, a kernel, a function, or a batch, conforms when its own spec does. Its output names and output form are therefore those of the batch's element spec.

Construction makes every batch homogeneous, and every operation that returns a batch returns a homogeneous one.
Three consequences follow from the definition:

1. Every batch has one set of outputs, read from its element spec (II.5), whatever its kind.
2. A function batch whose elements return different outputs, or take different inputs, is refused at construction, as a law batch is today (D1).
3. A symbolic size binds once across the batch, so the batch of laws of D4 is refused.

### 4.3 The one check

`is_valid` becomes II.1's conformance for every spec class, and the batch's construction raises with the reason an element fails:

```python
def _check_conforms(declared: TermSpec, value: Any, bindings: dict[str, int], path: str) -> None:
    """Raise ValueError naming *path* and the difference, unless *value* conforms to *declared*."""

class TermSpec:
    def is_valid(self, value: Any) -> bool:   # defined once, on the base
        try:
            _check_conforms(self, value, {}, "the value")
        except ValueError:
            return False
        return True
```

1. A value that carries its own spec is checked through `_bind_dims_from_spec`, which each spec class already implements.
2. `FunctionSpec.is_valid` stops accepting every callable, and `BatchSpec.is_valid` stops requiring equality (D2).
3. `_check_declarations` in `distributions/_batches.py` is deleted. Its message comes from the check, so a law, a kernel, and a function that do not match report the difference alike.

Step B1 checks each class's `_bind_dims_from_spec` against II.1. Where one unifies rather than conforms, it is changed to conform, since II.1 makes the declared side check what it receives.

### 4.4 Inference: the common spec

> **Definition (common spec).** The *common spec* of a set of specs is the narrowest spec to which every one of them conforms. It exists when the specs agree in structure, and it differs from each of them only in facts that are not structure.

| Part of a spec | Must agree | The common spec takes |
|---|---|---|
| kind; field names and order; output form and names; input names and optional inputs; ranks and sizes; levels | yes | the shared value |
| an array's dtype | no | the promoted dtype, `jnp.result_type` of all |
| an array's support | no | the shared support, or none when they differ |
| an opaque spec's `type` and `meta` | no | the shared value, or `None` when they differ |
| the spec of one input of a function or a kernel | no | the unification of the elements' specs for that input, since one value is passed to every element |

A constructor given no `element=` stores the common spec of its elements.
One function computes it, `_common_spec(specs, *, where)`, and its error names the first position whose structure differs:

```python
DistributionBatch([Normal("x", 0.0, 1.0), HalfNormal("x", 1.0)], "law").event_spec
# OutputSpec(x=NumericArraySpec(shape=(), dtype=float32)), in either order

FunctionBatch([F, G], "variant")
# TypeError: element 1 of the FunctionBatch returns the outputs ['result'], and element 0
#   returns ['a']. Give every function one output spec, or pass element= with a spec they
#   all conform to
```

The common spec is true of every element and independent of their order, which fixes D3.
For opaque values it is III.2's existing rule, and for arrays it is the dtype promotion that sweeps already apply.
**Decision 1.**

A batch with no elements has no specs to read, so it requires `element=`, unless its storage states the spec, as an empty array states its shape and dtype.

### 4.5 Declared specs and raw elements

A constructor's `element=` takes a bare spec or an `OutputSpec`, as `BatchSpec` does, so a caller can name the elements:

```python
NumericArrayBatch(jnp.arange(4.0), "draw", element=OutputSpec(x=())).element_spec
# OutputSpec(x=NumericArraySpec(shape=()))
```

Every element must conform to a declared spec, and two rules give each element view its spec:

1. A stored element with no declaration of its own, such as a raw array, an opaque value, or a raw callable, takes the declared spec, names included.
2. A stored term, such as a `Function` or a law, keeps its own spec, which conforms to the batch's.

So `b[i]` always conforms to `b.element_spec`, and equals it for a raw element.

### 4.6 One way to build a batch from elements

```python
class Batch:
    @classmethod
    def stack(cls, elements: Sequence[Any], *, level_name: str,
              element: OutputSpec | TermSpec | None = None,
              label: str | None = None) -> Batch: ...
    # On Batch, the class comes from the kind table: batch_class_for_spec of the
    # element spec. RecordBatch.stack is the same method, refusing other kinds.
```

Every producer that holds a list of elements calls `Batch.stack`, then reshapes the level it minted to its own geometry.
The producers are the sweep's `_make_stack`, `RecordBatch.stack`, `_stacked_batches`, and the `DistributionBatch` built in `core/transition.py`.

The class of a sweep's result then follows from the kind table, so a sweep of kernels returns a `ConditionalDistributionBatch` (D5), and a new kind needs one `register_kind` call.
A producer with a declared output spec always passes it to `stack` as `element=`.
**Decision 4.**

### 4.7 One storage interface

Every concrete batch implements two private methods, and code outside the batch classes reads a batch's storage only through them:

```python
def _raw_columns(self) -> Mapping[str, Any]: ...
    # One stacked array or object array per leaf path, batch axes leading.
    # The key "" holds the whole element of a batch whose elements have no paths.

@classmethod
def _over_raw_columns(cls, columns: Mapping[str, Any], *, spec: BatchSpec, label: str) -> Self: ...
    # The batch over columns already checked, sharing them without a copy.
```

`_raw_columns` generalizes `RecordBatch._raw_columns`.
`_over_raw_columns` replaces `_over_store`, `_view_over`, and `_batch_over`.
The sites of D8 then read `batch._raw_columns()` without checking the class, and a test asserts that no module outside the batch classes checks for a concrete batch class.

### 4.8 The common operations

> **Rule (lifted operation).** For an operation `op` on one element, `b.op(...)` is the batch over `b`'s levels whose element spec is `b.element_spec.op(...)` and whose element at each position `i` is `b[i].op(...)`.

The base `Batch` implements `output`, `rename`, `without`, and `&` by this rule over object-array storage.
`RecordBatch` and `NumericArrayBatch` compute them on their columns, and a shared test checks each against the rule:

```python
renamed = b.rename(x="y")
assert renamed.element_spec == b.element_spec.rename(x="y")
for i in range(len(b)):
    assert renamed[i] == b[i].rename(x="y")
```

An operation that the element kind lacks, such as `output` on a batch of laws, raises the error one element would raise, as VI.11 states for capabilities.

The rule replaces the batch parts of four `deferred.md` entries: S4b, G5, "`&` on batches other than `RecordBatch`", and "`without` on a batch of laws".
Each becomes the base implementation plus the element kind's own operation.

### 4.9 Equality

Two batches are equal when they are of one class, their specs are equal, and their elements are equal at every position under the element kind's `==`.
Once S4b's name-free spec exists, the name of the output at the root stays out of the comparison, as decided on 2026-10-20.
Every batch is unhashable, as `RecordBatch` is today, since a hash by value would read every element.
**Decision 5.**

### 4.10 Constructors

Every constructor takes one signature after its storage argument:

```python
(storage, /, level_names, *, element=None, axes_per_level=None, label=None, provenance=None)
```

One rule on the base derives an omitted label: the elements' output names when they have any, else the list of members that an object batch shows today.

## 5. Kind by kind

### 5.1 Arrays: `NumericArrayBatch`

Its storage already states its spec, and its constructor already refuses free dimensions.
It gains three things:

1. `element=` as an `OutputSpec`, so a batch of arrays carries a name (§ 4.5);
2. `output`, `rename`, `&`, and the list form `[["x"]]`, by § 4.8, which is the batch part of S4b;
3. equality by value (§ 4.9).

```python
x = sample(Normal("x", 0.0, 1.0), sample_shape=4)
y = sample(Normal("y", 0.0, 1.0), sample_shape=4)
x.output("x") is x              # True
x & y                           # a RecordBatch of 4 records {x, y}
x.rename(x="z").element_spec    # OutputSpec(z=NumericArraySpec(shape=(), dtype=float32, support=real))
```

### 5.2 Opaque values: `OpaqueBatch`

III.2's inference rule is the common spec's rule for opaque values, so it stays.
`label=` becomes optional (§ 4.10), and the batch compares by value (§ 4.9).

### 5.3 Records: `RecordBatch` and `NumericRecordBatch`

`RecordBatch` is the reference implementation, and most of its code is kept.

1. `RecordBatch.stack` becomes `Batch.stack` restricted to records, and `_shared_opaque_types` becomes a use of `_common_spec`.
2. `output` is added by § 4.8.
3. Stacking a record whose field has another shape raises numpy's "Cannot stack arrays with different numbers of dimensions", and a sweep of records with other fields raises D5's message. Both become `_common_spec`'s message, which names the position and the field.

### 5.4 Functions: `FunctionBatch`

A batch of functions follows § 4.2, by five rules:

1. **Inputs.** Every element takes the same input names and the same optional inputs. The batch's spec for each input is the unification of the elements' specs for it (§ 4.4), so one value can be passed to every element.
2. **Outputs.** Every element's output spec has the same form and the same output names. The batch's output spec is the common spec of the elements' output specs.
3. **Raw callables.** A raw callable is read as `Function(c)` reads it: its input names from its signature, and its output name from its Python name (III.3). Its signature is checked against the batch's inputs at construction, rather than when its element is read (D1).
4. **An unspecified output spec.** A `FunctionSpec` whose `output_spec` is `None` admits functions whose return values are not yet known. The output name is still fixed by rule 3, and each element's return value is checked when it is called, as for one `Function` (III.3).
5. **Element spec form.** The element spec takes the positional form of II.5, so a batch of functions that return `{a, b}` has the outputs `a` and `b`, and `fb["a"]` is the batch of the functions `f["a"]` (D7, G5). This needs `OutputSpec` to take a `FunctionSpec` positionally, which III.3's rule 4 already requires of a function that returns a function.

```python
def f(x): return x
def g(x): return 2 * x

FunctionBatch([lambda x: x, lambda x: 2 * x], "variant")   # accepted: both output names are "f"
FunctionBatch([f, g], "variant")
# TypeError: element 1 of the FunctionBatch has the output name 'g', and element 0 has 'f'.
#   Pass element=FunctionSpec(output_spec=OutputSpec(y=None)) to give them one name
FunctionBatch([f, g], "variant",
              element=FunctionSpec(output_spec=OutputSpec(y=None)))   # accepted: both return y
```

**Decision 2** is whether rule 3 reads a raw callable's Python name as its output name.
**Decision 3** is whether rule 4 admits functions with an unspecified output spec.

Calling a batch of functions, as `fb(1.0)`, is outside this plan, since neither II.5 nor V.5 defines it.

### 5.5 Laws: `DistributionBatch`

1. `_common_spec` and the one check replace `_first_element_term_spec` and `_check_declarations`, which fixes D3 and D4.
2. A sweep passes its declared output spec as `element=` (§ 4.6).
3. `laws["mu"]` is the batch of the coupled marginals `law["mu"]`, and `laws.without("mu")` is the batch of `law.without("mu")`. Both follow from § 4.8 and are the batch parts of G5 and of row T30.
4. `output` raises `KeyError`, since a batch of laws has no outputs (II.5).

### 5.6 Kernels: `ConditionalDistributionBatch`

Kernels follow § 5.5, and their inputs follow § 5.4, rule 1.
Two changes are their own:

1. A kernel that does not match names the difference, as a law does. Today the message is "elements must be a ConditionalDistribution whose input and event specs match the batch's", which names none.
2. A sweep of a kernel-valued function returns a `ConditionalDistributionBatch` (D5).

### 5.7 Linear operators: `LinOpBatch`

III.4 designs `LinOpBatch`, and nothing builds it yet.
When it is built, it registers its kind and implements § 4.7, and every rule above applies to it unchanged.

## 6. Steps

| Step | Delivers | Depends on | Changes user-visible behaviour |
|---|---|---|---|
| B1 | The one check (§ 4.3), with a raw callable's signature checked at construction | — | yes: a `FunctionBatch` of mismatched functions is refused |
| B2 | Homogeneity in one scope and the common spec (§ 4.2, § 4.4); `element=` as an `OutputSpec` (§ 4.5); one constructor signature (§ 4.10) | B1 | yes: a law batch's support may change, and laws of two shapes under one symbolic size are refused |
| B3 | The storage interface (§ 4.7), with the sites of D8 rewritten through it | — | no |
| B4 | `Batch.stack`, and the sweep's assembly through it (§ 4.6) | B2, B3 | yes: a sweep of kernels returns a `ConditionalDistributionBatch` |
| B5 | The lifted operations and equality (§ 4.8, § 4.9), with S4b's batch part | B3 | yes: batches compare by value |
| B6 | The positional element spec of a function batch, and its paths (§ 5.4, rule 5) | B5 | yes: a function batch's outputs change from `element` to its elements' outputs |

Each step adds its cases to one parametrized test module, `tests/core/test_batch_contract.py`.
It builds a batch of every kind and asserts the shared contract:

1. homogeneity, and the refusal of each structural mismatch with its position;
2. a common spec that does not depend on the order of the elements;
3. the lifted-operation rule for every operation a kind supports;
4. equality by value;
5. no `isinstance` check against a concrete batch class outside the batch modules.

## 7. Decisions

1. **The inferred element spec** (§ 4.4).
   - (a) The common spec of the elements.
   - (b) The first element's spec, to which every other element must conform.
   - (c) Equality of every element's spec.

   *Recommendation:* (a), the only option whose spec is true of every element and independent of their order. Option (b) is the behaviour that D3 shows, and (c) refuses a batch of a `Normal` and a `HalfNormal`.
2. **A raw callable's output name in a batch** (§ 5.4, rule 3).
   - (a) Its Python name, so `[f, g]` needs `element=`.
   - (b) A name the batch gives, the first callable's name by default.

   *Recommendation:* (a), which reads a raw callable as a single `Function` reads it, and refuses what a batch of laws refuses.
3. **Functions with an unspecified output spec** (§ 5.4, rule 4).
   - (a) Admitted, with the output name fixed and each return value checked at its call.
   - (b) Refused, so every function batch declares its return value.

   *Recommendation:* (a), since a single `Function` admits an unspecified output spec, and a sweep of a function that returns functions rarely declares their return values.
4. **`Batch.stack` as public API** (§ 4.6).
   - (a) Public on `Batch`, with `RecordBatch.stack` as its record case.
   - (b) Private, with `RecordBatch.stack` the only public stack.

   *Recommendation:* (a), which gives a batch of laws or of functions the constructor from elements that a batch of records has.
5. **Equality by value for every batch** (§ 4.9).
   - (a) Every batch compares by value.
   - (b) Only batches with columnar storage compare by value, and object batches by identity.

   *Recommendation:* (a), since II.5 and § 18.3 of `declarations-and-addressing.md` state the equality of batches for every kind.
