# Plan: one contract for numeric terms

> **Status: proposed; not approved.**
> Written on 2026-10-27 against `main` at `97891b7d`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Numeric terms".
> It is more fundamental than `plans/deferred/vector-interface.md`, which builds on it; § 6 lists the edits that plan needs.
> Once approved, its rules move into `design/` (II.1, II.3, III.1, III.5 to III.10, VI.5) and its steps into the plans of the work after the workshop.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 8):

1. Two levels of "numeric": a numeric *value* has a vector, and a numeric *term* of any kind has numeric values at its value positions (§ 4.1).
2. Whether a batch at a value position, such as a batch-valued event, counts as numeric (§ 4.1, § 5.4).
3. Whether `to_vector()` returns a tracked term on every kind, which changes its return type on values (§ 4.3).
4. The names of the vector forms' inputs and outputs: `vec`, `given_vec`, and `input_vec` (§ 4.4).
5. Whether only the required inputs of a kernel or function decide its numeric-ness and enter its vector (§ 5.5).
6. Which marker classes stay (§ 4.6).
7. Whether `from_vector` becomes one module-level function (§ 4.3).

---

## 1. Why this plan exists

"Numeric" is used in ProbPipe for three different things, and no section of `design/` defines it once:

1. a value that can be laid out as one vector of numbers, such as a `NumericRecord` (II.3);
2. a law whose draws are such values, such as a `NumericDistribution` (III.7);
3. one side of a kernel, through three positional markers (III.9).

Functions and batches of laws have no numeric notion at all.
The code decides numeric-ness in ten private predicates, which disagree.
One law shows the result:

```python
event = BatchSpec(NumericArraySpec(()), time=3)
atoms = NumericArrayBatch(values, ("trajectory", "time"), element=NumericArraySpec(()),
                          axes_per_level=(1, 1), label="t")
law = EmpiricalDistribution(atoms, [0.5, 0.5], event_spec=OutputSpec(trajectory=event))

isinstance(law, NumericDistribution)   # False
cov(law).shape                         # (3, 3): cov treats the event as numeric
law.dtypes
# AttributeError: dtypes is only available for a distribution with a numeric event,
#   but EmpiricalDistribution 'ℙ' has a non-numeric event
```

The goal is one definition of "numeric" for every kind, one vector layout that every vector conversion uses, and one rule for what each kind's vector form is.
Symbolic dimensions are included, because a layout needs concrete sizes.

## 2. The design today

### 2.1 What `design/` states

| Section | What it states |
|---|---|
| II.1 | Every `TermSpec` has the dimension protocol: `free_dims`, `with_dim_sizes`, `with_dim_names`, and binding. Dimension names occur in array shapes and in batch levels. |
| II.3 | The `Numeric` interface of values: `to_vector()` returns one flat `Array` in canonical order, `vector_size` its length, `from_vector(spec, vec, *, label)` the inverse. The coordinate protocols `__array__` and `__jax_array__` present `to_vector()`. `NumericSpec` is the spec-side mixin, with `vector_size`. |
| III.1 | `NumericArray`'s vector is its array raveled row-major. Its coordinate protocols present the array itself. `NumericArrayBatch` is given no vector interface. |
| III.5 | A `RecordSpec` whose leaves are all `NumericSpec` is promoted to `NumericRecordSpec`, and a record to `NumericRecord`. The layout is depth-first in declared field order. |
| III.6 | `NumericRecordBatch.to_vector()` has shape `(*batch_shape, vector_size)`, and its `__jax_array__` presents it. |
| III.7 | `NumericDistribution` is a marker whose membership is read from the event spec. It has the views `dtypes`, `supports`, `dtype`, and `support`. The **flat-vector law** `pushforward(d, to_vector)` is described. |
| III.9 | Three kernel markers: `NumericConditionalDistribution` (given numeric), `ConditionalNumericDistribution` (event numeric), and `FullyNumericConditionalDistribution` (both). |
| IV.1 | Four more markers for factored joints, one per numeric side. |
| III.4 | A `LinOp` is a linear map between flat numeric spaces, whose `shape` is the two sides' `vector_size`. |
| VI.5 | "A batch-valued event is numeric when its elements are", and `cov` flattens such a draw over its levels and then the element. |

### 2.2 What the design leaves open

1. **Two meanings, no definition.** II.3 calls a *value* numeric when it has a vector. III.7 calls a *law* numeric when its draws have one. No section states the general rule that covers both, so functions, kernels, and batches have none.
2. **Batches.** III.6 gives a batch of records a vector interface, and III.1 gives a batch of arrays none. VI.5 calls a batch-valued event numeric, and III.7's `NumericDistribution` requires a `NumericSpec` event, which a `BatchSpec` is not.
3. **Vector forms.** III.7 describes the vector form of a law. Nothing describes the vector form of a kernel, a function, or a batch of laws.
4. **Identity under vectorization.** `to_vector()` returns a bare array, so the label, the provenance, the output names, and each leaf's dtype and support are dropped. No section says which of them a vector form keeps.
5. **Dimensions.** II.1 puts the dimension protocol on every spec. III.7 lists `with_dim_sizes` and `with_dim_names` on a law, III.9 lists only `with_dim_names` on a kernel, and III.3 lists neither on a function.

## 3. The code today

### 3.1 Where the code agrees with the design

- `NumericArray` and `NumericRecord` implement `to_vector`, `vector_size`, and `from_vector` in row-major, depth-first, declared order.
- `NumericSpec.vector_size` refuses a symbolic dimension and names `with_dim_sizes` in its message.
- `Record`, `RecordSpec`, and `RecordBatch` are promoted to their numeric classes as III.5 and III.6 state.
- `NumericDistribution` membership follows the event spec, and dispatch refuses a method registered on the marker.
- `NumericRecordBatch.to_vector()` returns `(*batch_shape, vector_size)`.

### 3.2 Where the code departs from the design

**C1. A record's coordinate protocols present its only field.**
II.3 says they present `to_vector()`.

```python
r = Record({"a": jnp.array([1.0, 2.0]), "b": jnp.array(3)}, label="r")
r.to_vector()      # Array([1., 2., 3.], dtype=float32)
jnp.sum(r)
# TypeError: NumericRecord with 2 fields is not scalar-like; access a specific field
#   with record['field_name'] first.
```

A `NumericRecordBatch` raises the same way from `jnp.asarray`, where III.6 says it presents its batched vector.

**C2. A batch of arrays has no vector interface.**

```python
b = NumericArrayBatch(jnp.ones((3, 2)), "draw", label="b")
b.to_vector()   # AttributeError: 'NumericArrayBatch' object has no attribute 'to_vector'
```

**C3. Batch-valued events.** The example of § 1: the marker says "not numeric" while `mean` and `cov` treat the event as numeric.

**C4. Ten predicates decide what is numeric.** They are listed in Appendix A.
They give different answers:

- `operations/_moments._is_numeric` counts a batch of numeric elements, and `distributions/_distribution._declares_numeric_event` does not;
- `core/_array_backend._is_numeric_dtype` and `core/_execution_claims._is_numeric_dtype` are two copies of one dtype test;
- `inference/_approximate_distribution._first_non_numeric` accepts array leaves only, so it refuses a nested `NumericRecordSpec`.

Seventeen sites test `isinstance(spec, NumericSpec)` directly, in eleven modules.

**C5. Markers without readers.**
There are eight marker classes.
Only `NumericDistribution` is read in `probpipe/`, at three sites, one of which converts an argument for a parameter annotated with it (`functions/_normalization.py`).
The other seven are read only by their own definitions and by tests, and `FactoredNumericDistribution` is also inherited by `FactoredMultivariateGaussian`.
Membership by declaration cannot give methods to a law whose class does not inherit the marker, so `Distribution.__getattr__` resolves the four views through a table, `_NUMERIC_VIEWS`.
The kernel markers have no such table:

```python
k = conditional_distribution(lambda mu: Normal("y", mu, 1.0),
                             given_spec={"mu": NumericArraySpec(())}, label="lik")
isinstance(k, ConditionalNumericDistribution)   # True
k.dtypes   # AttributeError: '_FunctionKernel' object has no attribute 'dtypes'
```

**C6. `conditional_distribution` confuses numeric with traceable.**
It evaluates the function at stand-ins of its inputs when it is built, so it refuses every input it cannot make a stand-in for:

```python
conditional_distribution(lambda d: Normal("y", 0.0, 1.0),
                         given_spec={"d": OpaqueSpec(str)}, label="k")
# TypeError: input 'd' of 'k' must be declared as an array or a record of arrays, got OpaqueSpec

conditional_distribution(lambda mu: Normal("y", mu, 1.0),
                         given_spec={"mu": NumericArraySpec(("n",))}, label="k")
# TypeError: input 'mu' of 'k' has unbound dimensions ['n'], but conditional_distribution
#   needs a fixed shape for each required input. ...
```

The kernel spec admits both inputs; only this constructor refuses them.

**C7. A record's inferred spec drops dtypes, so a round trip changes them.**
A `NumericArray`'s spec keeps its dtype, and a `Record`'s inferred spec does not:

```python
NumericArray(jnp.array([True, False]), label="x").spec
# NumericArraySpec(shape=(2,), dtype=bool)

r = Record({"a": jnp.array([1.5]), "n": jnp.array(3)}, label="r")
r.spec                                               # NumericRecordSpec(a=(1,), n=())
NumericRecord.from_vector(r.spec, r.to_vector())["n"].raw   # 3.0, a float; it was the int 3
```

**C8. Functions have no numeric or dimension interface.**
`FunctionSpec` has `free_dims`, and `Function` has neither `to_vector`, `with_dim_sizes`, nor `with_dim_names`.
A call binds the dimensions from its arguments:

```python
g = Function(lambda x: 2 * x, input_spec=InputSpec(x=("n",)), output_spec=OutputSpec(y=("n",)))
g.spec.free_dims          # frozenset({'n'})
g(jnp.ones(4))            # NumericArray of shape (4,): the call binds n = 4
g.with_dim_sizes(n=3)     # AttributeError
```

A kernel has `with_dim_sizes` in the code, which III.9 does not list.

**C9. Vector conversion is reimplemented.**
`plans/deferred/vector-interface.md` § 1.2 counts about twenty private helpers that lay out values or laws, and four modules outside `core/` import the private `_reconstruct_from_vector`.

## 4. The proposed design: what every numeric term shares

### 4.1 What "numeric" means

A **value position** of a term is a place where the term holds or produces a value: a field of a record, the event of a law, the given side and event of a kernel, the inputs and output of a function, and the element of a batch.
Numeric-ness is defined in two levels, values first.

> [!IMPORTANT]
> **Definition (numeric value).** A *numeric value* is a value whose entries are numbers laid out in arrays:
> 1. an array of a numeric dtype (bool, integer, float, or complex) is a numeric value, and a `NumericArraySpec` that sets no dtype declares one;
> 2. a record is a numeric value when every leaf is a numeric value.
>
> Each numeric value of a concrete spec has a *vector*, the one-dimensional array of § 4.2.

> [!IMPORTANT]
> **Definition (numeric term).** A term of any kind is *numeric* when each of its value positions holds a numeric value, or a batch of numeric values:
> 1. an array or a record: when it is a numeric value;
> 2. a batch: when its element is numeric;
> 3. a law: when its event is;
> 4. a kernel: when its required inputs and its event are;
> 5. a function: when its required inputs and its output are, and its output is declared;
> 6. an opaque value: never.

Consequences:

1. **Numeric-ness is read from the spec.** It depends on declarations alone, so `is_numeric` is a property of a spec, and a term is numeric when its spec is.
2. **Numeric is not a claim about execution.** A numeric function may be opaque, untraceable, or not differentiable; those are its execution claims (V.13) and its differentiability claim (V.11). A Fortran solver wrapped as a `Function` with array inputs and output is numeric.
3. **Laws inside records do not count.** A record with a field that holds a numeric law is not a numeric value, since a law has no vector. The two levels exist for this case: a recursion through every kind would call that record numeric and then could not lay it out.
4. **Supports do not count.** A discrete or constrained support leaves a value numeric. Only the law-level reparameterization of V.12 refuses some supports (`vector-interface.md` § 4.4).

**Decision 1: two levels.**
- **Recommendation.** The two definitions above.
- **Alternative.** One recursion through every kind, in which a record of numeric laws is numeric. It needs a second word for "has a vector", which is what "numeric value" is.

**Decision 2: batches at value positions.**
- **Recommendation.** A batch of numeric values at a value position counts, as VI.5 already says for events. § 5.4 states its vector form.
- **Alternative.** Only arrays and records count at a value position, and the moments' predicate is renamed to say what it tests.

### 4.2 The layout: one order for every vector

> [!IMPORTANT]
> **Definition (layout).** The *layout* of a concrete numeric value spec assigns each entry of a value one index of its vector:
> 1. the leaves are taken depth-first in declared field order, an array being its own one leaf;
> 2. each leaf's entries are taken in row-major order;
> 3. each leaf occupies one contiguous segment, and the vector's size is the sum of the leaves' sizes;
> 4. the vector's dtype is the result type of the leaves' dtypes.
>
> The *batched layout* of a batch puts the batch axes first: `(*batch_shape, vector_size)`.

Consequences:

1. **The layout is a function of the spec.** Two values of one spec have the same layout, which is why III.5 makes field order part of a `RecordSpec`.
2. **A symbolic dimension has no layout.** `vector_size` and every vector form raise for a polymorphic spec, with today's message naming `with_dim_sizes`.
3. **Round trips need dtypes.** `from_vector` casts each leaf back to its declared dtype, so a spec inferred from a value records each leaf's dtype, which fixes C7. A declared spec may leave a dtype unset, and the leaf then takes the vector's dtype.
4. **The vector's support** is the leaves' common support when that support is entrywise, such as `positive`, and is unset otherwise. A raveled simplex is not a simplex, so a leaf with a simplex support gives an unset support. The leaves' own supports stay in the source spec.
5. **An empty record** has a vector of size 0 and the default float dtype.

The layout is one object, `VectorLayout`, which a numeric value spec returns as `spec.vector_layout`:

```python
s = RecordSpec(beta=NumericArraySpec((2,), jnp.float32),
               g=RecordSpec(n=NumericArraySpec((), jnp.int32),
                            sigma=NumericArraySpec((), jnp.float32, positive)))
s.vector_layout
# VectorLayout(size=4, dtype=float32, leaves=(
#     Leaf(path="beta",    shape=(2,), dtype=float32, support=None,     segment=slice(0, 2)),
#     Leaf(path="g/n",     shape=(),   dtype=int32,   support=None,     segment=slice(2, 3)),
#     Leaf(path="g/sigma", shape=(),   dtype=float32, support=positive, segment=slice(3, 4))))
```

`VectorLayout.ravel(raw, batch_ndim=0)` and `VectorLayout.unravel(vec)` convert between a raw value and its vector, with any leading batch axes.
They are pure JAX functions, so they run inside `jit`, `vmap`, and `grad`.
Every other vector conversion in ProbPipe calls these two, and no other code computes an order.

### 4.3 The vector form of a term

The vector form of a numeric term is the same term with every value position read through its layout.

> [!IMPORTANT]
> **Definition (vector form).** For a numeric term `x` of concrete spec, `x.to_vector()` is the term of the same kind whose value positions hold vectors:
> 1. an array or a record gives a `NumericArray` of shape `(vector_size,)`;
> 2. a batch gives a batch on the same levels whose elements are the elements' vector forms;
> 3. a law gives the law of the vector of a draw;
> 4. a kernel gives the kernel from the vector of a given value to the law of the vector of a draw;
> 5. a function gives the function from the vector of its inputs to the vector of its output.
>
> `from_vector(spec, v)` is its inverse: for `v` the vector form of a term of spec `spec`, it returns that term again, equal by value for values and batches of values, and the same law, kernel, or function for the other kinds.
>
> An `InputSpec` is laid out as the record of its required inputs, in declared order, and `from_vector` on an `InputSpec` returns that record.

Consequences, each a property the tests assert for every kind:

1. **Round trip:** `from_vector(x.spec, x.to_vector()) == x`.
2. **Draws:** `sample(d.to_vector())` equals `sample(d).to_vector()` under one seed.
3. **Densities:** `log_prob(d.to_vector(), x.to_vector()) == log_prob(d, x)`, since the layout only reorders entries and its log-Jacobian is 0.
4. **Moments:** `mean(d.to_vector()) == mean(d).to_vector()`, and `cov(d.to_vector())` equals `cov(d)`, which VI.5 already defines over the layout.
5. **Kernels:** `condition_on(k.to_vector(), {"given_vec": v})` equals `condition_on(k, from_vector(k.given_spec, v)).to_vector()`.
6. **Functions:** `f.to_vector()(v)` equals `f(**from_vector(f.input_spec, v).select_all()).to_vector()`.
7. **Batches:** `b.to_vector()[i] == b[i].to_vector()`.

All five constructions use two `Function`s per spec: `vector_map(spec)`, from a value to its vector, and its inverse.
`vector_map` calls `VectorLayout.ravel`, claims its inverse and a zero log-Jacobian (V.7), and claims `traceable`, `vectorized`, and `differentiable`.
A law's vector form is `pushforward(d, vector_map(spec))`, and a function's and a kernel's are compositions with the two maps on each side.
So the kind-specific code is one composition per kind.

**Decision 3: `to_vector()` returns a tracked term.**
- **Recommendation.** `to_vector()` returns the vector form on every kind, a `NumericArray` for a value. Raw coordinates are `x.to_vector().raw`, or `jnp.asarray(x)` through the coordinate protocols. Four call sites in `probpipe/` outside `core/_numeric.py` read the bare array today.
- **Alternative.** `to_vector()` keeps returning a bare array on values, and the other kinds get a differently named method. The same name would then mean two things.

**Decision 7: `from_vector` as one function.**
- **Recommendation.** One module-level `from_vector(spec, vec, *, label=None)`, where the kind of `spec` decides what is rebuilt: a value from a `NumericArraySpec` or `NumericRecordSpec`, a batch from a `BatchSpec`, a law from a `DistributionSpec`, and so on. The classmethods `NumericArray.from_vector`, `NumericRecord.from_vector`, and `NumericRecordBatch.from_vector` are removed, and `Numeric` no longer declares one.
- **Alternative.** A classmethod on every class, each delegating to the module. The class adds nothing that the spec does not already decide.

### 4.4 What a vector form keeps: labels, names, provenance, annotations

| Part of the term | In `x.to_vector()` | In `from_vector(spec, v)` |
|---|---|---|
| description | `vec(x)`, as any derived term's (II.4) | `from_vector(v)` |
| label | `label=` if given | `label=` if given |
| output name | `vec`, the glossary's name for a 1-D numeric serialization | the names in `spec` |
| per-leaf dtypes, supports, paths, nesting | only in the source spec | restored from `spec` |
| batch levels and level names | kept | taken from `spec` |
| provenance | one step, `to_vector`, with `x` as its parent | one step, `from_vector`, with `v` as its parent |
| annotations | none, as for any derived term (II.4) | none |

The names of the inputs follow the kind:

| Kind | Vector form's inputs | Vector form's output or event |
|---|---|---|
| value | none | `vec` |
| law | none | `vec` |
| kernel | `given_vec` | `vec` |
| function | `input_vec` | `vec` |

A kernel needs two names because its given and event names must differ (III.9).

```python
d = Normal("mu", 0.0, 1.0) & HalfNormal("sigma", 1.0)
v = d.to_vector()
v.event_spec      # OutputSpec(vec=NumericArraySpec(shape=(2,), dtype=float32))
# v's provenance is one step, to_vector, whose parent is d;
# from_vector(d.spec, v) is a law over {mu, sigma} again, equal to d in law
```

Under `jit`, identity is attached at the boundary (II.4), so inside a trace a vector form is its raw vector and nothing else.

**Decision 4: names.** Recommended: `vec`, `given_vec`, and `input_vec`. The alternative keeps a value's own output name for an array, so `Normal("mu", ...).to_vector()` has the output `mu`, and uses `vec` only for records.

### 4.5 Symbolic dimensions

Dimension names occur in two places: array shapes, and batch levels (II.5).
A batch level may be symbolic whatever its element is, so `BatchSpec(OpaqueSpec(), sample="S")` is polymorphic and not numeric.
The dimension protocol therefore belongs to every spec, as II.1 states, and what is specific to numeric terms is that a layout needs concrete sizes.

> [!IMPORTANT]
> **Definition (the dimension contract).** Every kind whose spec may be polymorphic, which is a law, a kernel, and a function, has the members `free_dims`, `with_dim_sizes`, and `with_dim_names`, read from and applied to its spec.
> A value, and a batch that holds elements, is always concrete.
> Each term has one dimension scope across all its value positions.

Kind by kind:

1. **Law.** As today: `with_dim_sizes` binds, and a merge binds across operands (IV.2).
2. **Kernel.** One scope across given and event. `condition_on` binds the given side's dimensions from the given value, and the event's sizes follow. A constructor accepts a polymorphic given (§ 5.5).
3. **Function.** One scope across inputs and output. A call binds from its arguments, as today. `with_dim_sizes` fixes sizes for every later call (new).
4. **Vector forms.** `to_vector()` on a polymorphic term raises and names `with_dim_sizes`.

```python
g = Function(lambda x: 2 * x, input_spec=InputSpec(x=("n",)), output_spec=OutputSpec(y=("n",)))
g.free_dims                               # frozenset({'n'})
g.to_vector()                             # ValueError: ... set the sizes with with_dim_sizes(n=...)
g.with_dim_sizes(n=3).to_vector().spec    # input_vec: (3,) -> vec: (3,)
```

Two gaps stay outside this plan.
- **Symbolic vector sizes.** The vector of a `("n",)` array and a `(2,)` array has `n + 2` entries, which needs arithmetic on dimension names. II.1 has none.
- **Polymorphic shapes in execution.** A constructor that evaluates a Python function abstractly, as `conditional_distribution` does, needs a size for each dimension, and how a symbolic shape crosses a JAX trace is undesigned. JAX's symbolic shapes (`jax.export.symbolic_shape`) are one candidate. This belongs with the execution layer and is recorded in `deferred.md`.

### 4.6 Where the numeric contract is implemented

1. **The spec decides.** `TermSpec.is_numeric` is a property that each spec class defines by the rules of § 4.1, recursively, and `InputSpec` and `OutputSpec` define it over their inputs and outputs. It replaces the ten predicates of Appendix A and the seventeen direct `isinstance(spec, NumericSpec)` tests.
2. **The layout is one module.** `core/_vector_layout.py` holds `VectorLayout`, `ravel`, and `unravel`. `NumericSpec` still marks the value specs that have a layout, and it gains `vector_layout`.
3. **Methods are on the base kinds.** `Distribution`, `ConditionalDistribution`, `Function`, and `Batch` each define `to_vector()`. On a term that is not numeric it raises `TypeError` and names the first value position that does not hold a numeric value. `event_shape` follows the same pattern today: every law has it, and it raises unless the law is over one array.
4. **The views come from the layout.** `dtypes`, `supports`, `dtype`, and `support` are read from `vector_layout` and defined on every numeric law and kernel. The `_NUMERIC_VIEWS` table and its `__getattr__` route go.
5. **Markers are queries.** A marker whose membership follows the spec cannot give methods to an instance whose class does not inherit it, so the methods of item 3 are on the base kinds instead. A marker holds no behaviour, and `isinstance(x, M)` holds exactly when `x.spec.is_numeric`.
6. **Values keep their classes.** `NumericArray`, `NumericRecord`, `NumericArrayBatch`, and `NumericRecordBatch` stay real classes, since each has its own storage and promotion chooses the class when a value is built. `Numeric` stays the base of the two value classes, with `vector_size`, `to_vector()`, and the coordinate protocols. A record's protocols present its raw vector, which fixes C1. `NumericArray`'s present its array, whose row-major order is the vector's.

**Decision 6: the markers.**
- **Recommendation.** Three markers stay, all queries:
  - `NumericDistribution`, as today;
  - `NumericConditionalDistribution`, which now means that both sides are numeric;
  - `NumericFunction`, new.

  The other six are removed: `ConditionalNumericDistribution`, `FullyNumericConditionalDistribution`, and the four `Factored*Numeric*` markers. A question about one side reads the spec, as `k.event_spec.is_numeric`. Batches of laws, kernels, and functions get no markers.
- **Alternative.** No markers, only `x.spec.is_numeric`. The recommendation keeps three because `NumericDistribution` is public and a parameter annotated with it already converts its argument (`functions/_normalization.py`).

## 5. Kind by kind

### 5.1 Arrays

```python
x = NumericArray(jnp.arange(4.0).reshape(2, 2), label="x")
v = x.to_vector()     # NumericArray 'vec(x)', shape (4,), the entries 0., 1., 2., 3.
from_vector(x.spec, v) == x        # True
jnp.asarray(x).shape               # (2, 2): an array presents itself
```

The vector's spec is `NumericArraySpec((4,), x.dtype, support)`, with `x`'s support when it is entrywise.

### 5.2 Records

```python
r = Record({"a": jnp.array([1.0, 2.0]), "n": jnp.array(3)}, label="r")
r.spec                       # NumericRecordSpec(a=NumericArraySpec((2,), float32), n=NumericArraySpec((), int32))
r.to_vector().raw            # Array([1., 2., 3.], dtype=float32)
from_vector(r.spec, r.to_vector())["n"].raw    # Array(3, dtype=int32)
jnp.sum(r)                   # Array(6., dtype=float32): the protocols present the vector
```

A field that holds a stored `NumericRecord` term is a numeric leaf, as III.5 states.
A field that holds a law makes the record not numeric (§ 4.1, consequence 3).

### 5.3 Batches of values

`NumericArrayBatch` gains the vector form, as `NumericRecordBatch` has it.
Both return a `NumericArrayBatch` on the same levels, and both present the batched layout through `__jax_array__`.

```python
draws = sample(Normal("mu", 0.0, 1.0) & HalfNormal("sigma", 1.0), sample_shape=100)
draws.to_vector()        # NumericArrayBatch 'vec(...)', levels {'sample': 100}, element (2,)
jnp.asarray(draws).shape # (100, 2)
```

### 5.4 Laws

A law's vector form is `pushforward(d, vector_map(spec))`: the flat-vector law of III.7, which `vector-interface.md` § 4.3 calls `vector_law(d)`.
Its event is `OutputSpec(vec=NumericArraySpec((vector_size,), dtype, support))`.
It has a density where `d` has one, samples where `d` samples, and has `d`'s moments in the layout (§ 4.3, consequences 2 to 4).

A law over a batch-valued event, such as the trajectories of § 1, is numeric under decision 2.
Its vector form follows the batch rule: it is a law over a batch of vectors on the same levels.
Where one vector of the whole draw is needed, as `cov` needs it, that vector is the batched layout raveled row-major, which is the order VI.5 already uses.

### 5.5 Kernels

```python
lik = conditional_distribution(lambda mu, tau: Normal("y", mu, tau),
                               given_spec={"mu": (), "tau": ()}, label="lik")
k = lik.to_vector()
k.given_spec      # InputSpec(given_vec=NumericArraySpec((2,)))
k.event_spec      # OutputSpec(vec=NumericArraySpec((1,)))
condition_on(k, {"given_vec": jnp.array([0.0, 1.0])})   # the law of [y] for mu = 0, tau = 1
```

- **The given's layout.** The required inputs are laid out as a record of the inputs, in declared input order.
- **Optional inputs.** An optional input stays at its default in the vector form, so a kernel whose optional inputs are opaque, such as a method name, is numeric when its required inputs are.
- **Polymorphic givens.** The spec admits them, and `condition_on` binds them. `conditional_distribution`'s refusal of opaque and polymorphic givens (C6) comes from its stand-in evaluation, which is a question of traceability. It is fixed with the execution gap of § 4.5, and this plan only states the contract the constructor must meet.

**Decision 5: required inputs.**
- **Recommendation.** Only required inputs decide numeric-ness and enter the vector; optional inputs keep their defaults. A caller who wants an optional input in the vector declares it required.
- **Alternative.** Every input counts and enters the vector. Then `f(x, method="lbfgs")` is not numeric.

### 5.6 Functions

```python
f = Function(lambda x, y: {"s": x + y, "p": x * y},
             input_spec=InputSpec(x=(2,), y=()),
             output_spec=OutputSpec(RecordSpec(s=(2,), p=(2,))))
g = f.to_vector()
g.input_spec      # InputSpec(input_vec=NumericArraySpec((3,)))
g.output_spec     # OutputSpec(vec=NumericArraySpec((4,)))
g(jnp.array([1.0, 2.0, 3.0])).raw   # Array([4., 5., 3., 6.]): s then p
```

- **Execution claims carry over.** The layout maps are traceable, vectorized, and differentiable, so `f.to_vector()` claims what `f` claims. An opaque `f` gives an opaque vector form.
- **An undeclared output.** A function whose `output_spec` leaves its type unspecified is not numeric, since numeric-ness is a declaration. `to_vector()` raises and asks for an `output_spec`.
- **`LinOp`.** A `LinOp` is numeric, and its matrix already acts on vectors (III.4). Its vector form is the same operator with its two sides replaced by their vector specs.

### 5.7 Batches of laws, kernels, and functions

```python
laws = DistributionBatch([Normal("x", 0.0, 1.0), Normal("x", 1.0, 2.0)], "law")
laws.to_vector()     # DistributionBatch on the level 'law' of laws over vec: (1,)
```

The vector form lifts element by element (VI.11).
All elements share one layout because they share one element spec.
For a batch of functions that guarantee arrives with `batch-contract.md` D1, which makes a `FunctionBatch` check its elements' specs; until then `FunctionBatch.to_vector()` refuses a batch declared with `FunctionSpec()`.

## 6. Edits to the other deferred plans

`vector-interface.md` needs these edits, all of which keep its design:

| Its section | Edit |
|---|---|
| § 1.2, § 2 principle 1, § 4.1 | The value-level contract is no longer fixed as it stands: `to_vector()` returns a tracked term (decision 3), the layout is `VectorLayout`, and `NumericArrayBatch`'s vector form moves to this plan's step N2. |
| § 3, § 4.2, § 8.1 | `to_vector_map(spec)` is this plan's `vector_map(spec)`, the one map every vector form uses. Its output name `vec` stays. |
| § 4.3, § 8.2 | `vector_law(d)` is `d.to_vector()`, so the separate function and the `VectorLaw` class are not needed. |
| § 4.4 | `vector_bijector` reads the leaves' supports from `spec.vector_layout`. Its leaf policy is unchanged. |
| § 4.5, § 8.5 | `log_prob_function` of a kernel takes the vector form's `given_vec` when given a kernel's vector form. |
| § 7 | Its table of deleted helpers gains this plan's ten predicates. |

`batch-contract.md` needs no edit.
This plan uses its common element spec (§ 5.7) and adds the batch's vector form, which its lifted-operation rule covers.

## 7. Steps

Each step is one PR.

1. **N1, specs:** `TermSpec.is_numeric`, `VectorLayout` and `spec.vector_layout`, and record specs inferred with dtypes. The ten predicates and the seventeen direct tests move onto `is_numeric`.
2. **N2, values:** the tracked `to_vector()` on arrays, records, and their batches; the module-level `from_vector`; `NumericArrayBatch`'s vector form; the coordinate protocols of records and record batches. The four call sites that read a bare array, and the four imports of `_reconstruct_from_vector`, move onto it.
3. **N3, dimensions:** `free_dims`, `with_dim_sizes`, and `with_dim_names` on `Function`, and III.9 updated to list the kernel's `with_dim_sizes`.
4. **N4, markers and views:** `NumericFunction`, the new meaning of `NumericConditionalDistribution`, the removal of six markers and of `_NUMERIC_VIEWS`, and the views on every numeric law and kernel.
5. **N5, vector forms:** `vector_map`, and `to_vector()` on laws, kernels, functions, and batches of them. This replaces V1 and V2 of `vector-interface.md`, whose V3 and V4 then follow.

**Tests.**
- One conformance test, parametrized over every kind, asserts the seven consequences of § 4.3.
- One test asserts that `is_numeric` agrees with the definition of § 4.1 on a table of specs, batch-valued events and records holding laws included.
- Round trips keep dtypes for bool, integer, float, and complex leaves (C7).

**What users see.**
- **Breaking:** `x.to_vector()` on a value returns a `NumericArray`, and raw code reads `.raw`. A record's inferred spec records dtypes, so it no longer equals a spec written without them, as `RecordSpec(a=(1,))`. `from_vector` becomes a function, and its classmethods go. Six marker classes go, and `NumericConditionalDistribution` changes its meaning from "given numeric" to "both sides numeric".
- **Fixed:** `jnp.sum(record)` and `jnp.asarray(record_batch)` work. A record's round trip keeps dtypes. A law over a batch-valued event is numeric.
- **Added:** `to_vector()` on every numeric kind, `NumericFunction`, and the dimension methods on `Function`.

## 8. Questions for the maintainer

1. **Two levels.** Recommended: numeric values (arrays, records) and numeric terms of any kind (§ 4.1).
2. **Batches at value positions.** Recommended: they count, and a law's vector form keeps the levels (§ 4.1, § 5.4).
3. **The tracked `to_vector()`.** Recommended: it returns the vector form on every kind (§ 4.3).
4. **Names.** Recommended: `vec`, `given_vec`, and `input_vec` (§ 4.4).
5. **Required inputs.** Recommended: only required inputs decide numeric-ness and enter the vector (§ 5.5).
6. **Markers.** Recommended: `NumericDistribution`, `NumericConditionalDistribution` for both sides, and a new `NumericFunction`, all as queries; six removed (§ 4.6).
7. **`from_vector`.** Recommended: one module-level function, keyed by the spec's kind (§ 4.3).

## Appendix A. The predicates that decide numeric-ness today

| Predicate | What it accepts |
|---|---|
| `core/_record_spec._is_numeric_spec`, `_all_numeric` | a `NumericSpec`, a numeric `RecordSpec`, or a mapping of numeric specs or shape tuples |
| `RecordSpec.is_numeric` | every leaf a `NumericSpec` |
| `core/record._is_numeric_field_value` | a numeric scalar, array, or backend container, or a `NumericRecord` |
| `core/_array_backend._is_numeric_leaf`, `_is_numeric_dtype` | a registered backend's answer, or a dtype of kind `b`, `i`, `u`, `f`, or `c` |
| `core/_execution_claims._is_numeric_dtype` | the same dtype kinds, and PRNG keys |
| `_array_utils._is_numeric_array` | any `jax.Array`, or a NumPy array of a dtype other than `object` |
| `operations/_moments._is_numeric`, `_numeric_event` | a `NumericSpec`, or a batch of one |
| `distributions/_distribution._declares_numeric_event` | a law whose event is a `NumericSpec`; a batch-valued event is refused |
| `distributions/_conditional._given_is_numeric`, `_event_is_numeric`; `distributions/_factored._numeric_givens` | every input a `NumericSpec`; the event a `NumericSpec`; the bound inputs a `NumericSpec` |
| `distributions/_empirical._numeric_atoms` | a batch whose element is a `NumericSpec`, or a numeric array |
| `inference/_approximate_distribution._first_non_numeric` | array leaves only |

The leaf-level predicates (`_is_numeric_leaf`, `_is_numeric_dtype`) decide what a raw value is and stay. The spec-level ones become `is_numeric`.

## Appendix B. Readings this plan adopts

1. **III.1 and II.3 on the coordinate protocols.** II.3 says they present `to_vector()`, and III.1 says an array presents itself. The plan keeps both: the two agree on the order of entries and differ only in shape.
2. **III.9 lists `with_dim_names` alone on a kernel.** The code also has `with_dim_sizes`, and the plan reads the omission as a gap in the text.
3. **"Numeric dtype" includes bool and complex,** as `_is_numeric_dtype` does today. The law-level interfaces of `vector-interface.md` that need real coordinates refuse the others (its § 4.4, rule 3).
4. **A `Function` without an `output_spec` is not numeric,** although its calls may return numeric values. Numeric-ness is a declaration, so an undeclared output is unknown.
