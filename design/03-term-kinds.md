# Part III — Term Kinds

Part III introduces the **term kinds**: the values, functions, and distributions a user constructs and operates on.

| §      | Category                       | Contents                                                                                              | Role                                                                                                            |
| ------ | --------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| III.1  | Values                      | `NumericArray` / `NumericArrayBatch`                                                                  | The base numeric-array kind                      |
| III.2  | Values                      | `Opaque` / `OpaqueBatch`                                                                              | The fallback kind                                            |
| III.3  | Functions                      | `Function`, `FunctionBatch`                                                | The base function kind |
| III.4  | Functions                      | `LinOp`                                                                                               | `Function` subtype for linear maps between numeric kinds                          |
| III.5  | Structured Values                      | `RecordSpec` / `Record` / `NumericRecord`                                                             | The base kind for structured values  |
| III.6  | Structured Values                      | `RecordBatch` / `NumericRecordBatch`                                                                  | A batch of records                               |
| III.7  | Distributions               | `Distribution`                                                                                        | The base kind for a probability measure (i.e., distribution)                           |
| III.8  | Distributions               | Distribution capabilities                                                                             | The protocols a distribution can implement, such as sampling, density evaluation, and conditioning               |
| III.9  | Distributions   | `ConditionalDistribution`                                                                             | The base kind for a probability kernel  |
| III.10  |Distributions | `DistributionBatch` / `ConditionalDistributionBatch`                                                  | Batches of distributions and conditional distributions             |

## III.1 — `NumericArray`

### Contract

`NumericArray` is one array value with identity: a `TrackedTerm`, holding a single array whose `shape` is the **event** shape, with no batch axes; its `raw()` is that array. `NumericArraySpec` is its kind's term spec:

```python
class NumericArraySpec(NumericSpec):  # the numeric-array kind's spec, a NumericSpec (II.3)
    shape: tuple[int | str, ...]   # a str names a symbolic dimension (II.1)
    dtype: DType
    support: Constraint            # the support (II.3)
```

The constructor reads `shape` as a shape argument (II.1), so `NumericArraySpec("n")` is `NumericArraySpec(("n",))`.

It carries the full set of array operators, for example arithmetic and comparison, and the coordinate protocols. An operator returns a tracked term under a deterministically derived, evaluation-order name, such as `x + 1` or `(x + y) * x`, with identity attached as for any operation (II.4). Each operand is grouped by the rules of II.4, so it reads as one unit in the name whatever the precedence, as in `(2 * x) + 1`, `(-x) ** 2`, and `x + [other effect]`. The result declares its value's shape, and its value's dtype when every tracked operand declares a dtype. Indexing and iteration return bare arrays.

`NumericArray` implements the `Numeric` interface of II.3. Its vector is the array raveled in row-major order, and its coordinate protocols present the array itself, so NumPy and JAX functions see its shape:

```python
class NumericArray(TrackedTerm, Numeric):
    @property
    def vector_size(self) -> int: ...     # the number of entries
    def to_vector(self) -> Array: ...     # the array raveled in row-major order
    @classmethod
    def from_vector(cls, spec: NumericArraySpec, vec: Array, *, label: str) -> NumericArray: ...
```

`NumericArrayBatch` is the kind's batch form: a `Batch` whose `element_spec` is the `NumericArraySpec` and whose storage is one array with the batch axes leading — the same split `RecordBatch` uses, with one column instead of many. An array with leading axes is just an array; the batch form is what carries the level names, the shared spec, and provenance.

### Rationale

The full set of array operators is safe here and only here: with no fields, an operator applied to one array has exactly one meaning (`D1 – Mathematical fidelity`).

## III.2 — `Opaque`

### Contract

`Opaque` adds identity alone, and its `raw()` is the wrapped value. `OpaqueSpec` is the fallback spec: it admits a value that no other kind admits, so a collection such as a list, a tuple, or a set is opaque:

```python
class OpaqueSpec(TermSpec):        # the fallback spec; is_valid accepts a value no other kind's spec class admits
    type: type | None = None       # the Python type of the admitted values; None admits any non-mapping value
    meta: Hashable = None          # free-form metadata, never checked or inferred
```

**`type` and `meta`.** `type` is the Python type of the values the spec admits, and `is_valid` checks `isinstance(value, type)`. `None` admits every value that is not a mapping, since a mapping is a subtree (III.5). Construction from a value infers the type: an `Opaque` built from a string and a string leaf of a record built from values both have `OpaqueSpec(type=str)`. An `OpaqueBatch` infers the type its elements share exactly, and `None` when they differ. `meta` is free-form hashable metadata, such as units or a tag. It is part of the spec's equality and hash, and it is never checked against a value or inferred. Two opaque specs unify when their types are equal or one of them is `None` and their `meta` are equal or one of them is `None`, and the unification takes the known type and the set `meta`, so a declared `meta` unifies with the spec a value infers, which carries none. Completion keeps the declared type and checks the produced value (II.2).

`OpaqueBatch` is its batch form. It **stores** each element outright. Its `raw()` is an object array of the stored raw values.

### Rationale

The kind exists so that closure under operations holds for every return value (`D4 – Closed system of objects under operations`), and it adds identity alone because a richer interface would promise structure the value does not declare (`D1 – Mathematical fidelity`).

## III.3 — `Function`

### Contract

The function kind's base type is `Function`. A `Function` is a tracked term that wraps exactly one Python callable as its representation and carries a `FunctionSpec`, whose sides it exposes as the `input_spec` and `output_spec` views; either side is optional, as in the spec. A `Function` also carries a frozen `inspect.Signature`, which is authoritative for Python argument binding, since parameter kinds, defaults, and variadic parameters are not expressible in a value schema; the `input_spec` is authoritative for the value schema. Construction validates their one-for-one correspondence, so binding an argument binds a slot by name. Its `raw()` is the wrapped callable.

A `Function` accepts an optional `output_label` as a result alias. Without an alias, its managed calls display the application expression; named callables default to their Python name and lambdas to `f`. Output components default to the callable's original name, with `f` for a lambda and `result` for a callable without a name or whose name is not a Python identifier. Record results expose their fields. An optional `OutputSpec`, such as `OutputSpec(mean=None)`, overrides this default or declares the result's type. A bare term spec uses the same default packaging. Relabeling a function or result never changes these declarations.

```python
@function(label="predict", output_label="prediction",
          output_spec=OutputSpec(mean=None))
def predict_impl(theta, x):
    return x @ theta

prediction = predict_impl(theta, x)
# predict_impl.label == "predict"; prediction.label == "prediction"
# output component: mean; the inferred return is an array, not a record
```

A `Function` is invoked two ways. `apply` evaluates the wrapped callable at a point: given values that conform to `input_spec`, it returns one conforming to `output_spec`, with no tracking or lifting — the raw map that operations such as change of variables build on. `__call__` runs the **call handler**, which is the base's one extension point: on the base it is plain evaluation, and the engine layer (Part V) replaces it once, at import. The base also carries its **controls** (V.2), set at construction and revised functionally by `with_options`; the engine alone interprets them, at call time.

A `Function` is constructed directly or using the `@function` decorator. `FunctionSpec`, which is the function kind's term spec, admits any callable.

```python
class FunctionSpec(TermSpec):      # the function kind's spec; is_valid accepts any callable
    input_spec: InputSpec | None   # None: that side's structure unspecified
    output_spec: OutputSpec | None
```

```python
class Function(TrackedTerm):
    def __init__(self, fn: Callable, /, *, label: str | None = None,
                 input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
                 output_spec: OutputSpec | TermSpec | None = None,
                 output_label: str | None = None,
                 differentiable: NumericSpec = ...) -> None: ...
                 # optional differentiability claim (V.11)
    @property
    def spec(self) -> FunctionSpec: ...
    @property
    def input_spec(self) -> InputSpec | None: ...                   # view on spec
    @property
    def output_spec(self) -> OutputSpec | None: ...                 # view on spec
    @property
    def output_label(self) -> str: ...                             # result label, outside spec
    @property
    def options(self) -> Mapping[str, Any]: ...          # the controls; opaque to the base
    def with_options(self, **controls) -> Self: ...      # functional update
    def apply(self, *args, **kwargs) -> Any: ...
    # evaluate the wrapped callable at a point (input_spec -> output_spec),
    # with no tracking or lifting; the raw map operations build on
    def __call__(self, *args, **kwargs) -> Any: ...
    # run the call handler: plain evaluation on the base, the Part V engine after import
    @property
    def notation(self) -> str: ...   # the label and the parameters, as predict(x, y); str() returns it (II.4)

def install_call_engine(engine: Callable[..., Any]) -> None: ...
    # replaces the call handler, once, at import time; until then calls evaluate plainly.
    # The engine reads the controls the Function carries and must agree with
    # plain evaluation on concrete values.
```

`FunctionBatch` is the function kind's batch form, storing its elements exactly as `OpaqueBatch` does (III.2); an element is a `Function` whatever callable the slot holds, and `raw()` is an object array of the stored callables.

**Capabilities.** As a distribution declares the operations it supports (III.8), a `Function` may claim capabilities beyond evaluation. A `Function` claims `SupportsInverse` by providing the inverse map, its own `apply` serving as the forward. In addition, `SupportsLogDetJacobian` provides the log-determinant of the Jacobian, which exists only for a differentiable map. Both are typed over the `Numeric` interface (II.3).

```python
@runtime_checkable
class SupportsInverse(Protocol):            # an invertible map; the forward is the claiming Function's apply
    def _inverse(self, y: Numeric) -> Numeric: ...

@runtime_checkable
class SupportsLogDetJacobian(Protocol):     # a map with a tractable Jacobian determinant
    def _log_det_jacobian(self, x: Numeric) -> Array: ...

def is_invertible(f: Any) -> bool: ...      # SupportsInverse membership together with its guard
```

`is_invertible(f)` is `True` when `f` claims `SupportsInverse` and its guard accepts `f`, and `False` otherwise. An unresolved guard (II.7) therefore gives `False`, and a slot that needs the inverse raises `ResolutionError` (V.12).

### Rationale

Defining the base in the value layer keeps the layering strict: the representation is fixed here, the call engine arrives by upward registration (`D2 – Generality first`), and `LinOp` and the specs reference `Function` downward — the split the package structure realizes as `values/_function_base.py` and `functions/`. Invertibility as a capability is `D3 – Capability-based operations`: an invertible map is an ordinary `Function` that additionally claims `SupportsInverse`, so it evaluates, composes, and pushes forward like any other, with *bijector* reserved for the mathematical statement. The Jacobian determinant is a separate capability for the same reason, since a map can be invertible without a tractable determinant, and change of variables asks for exactly the pair.

## III.4 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces and the linear subtype of `Function` (III.3). It therefore applies, composes, and evaluates like any map. Its action is the map the base carries: `apply` evaluates the operator at a `Numeric` conforming to its input schema and returns the matching form, with the operator's parameters as private state. `matvec`, `matmat`, `rmatvec`, and `rmatmat` are the linear-algebra names for the action and its transpose, and `matmat` is the operator's registered batched rule. Its output declaration names its component; a constructor given only a codomain shape declares the output as a whole term under the fixed component `result`, the component of a nameless callable, which no label changes (III.3). Its domain is the `NumericSpec` (II.3) of its single input slot, and its codomain is `output_spec.spec`; an exposed record output may have several components while remaining one numeric value. The operator reads its spaces from these declarations alone. It therefore maps whatever `Numeric` its sides declare, for example a bare array under a `NumericArraySpec` side, so an operator over a scalar law's draws takes them as bare arrays. The two sides coincide for an endomorphism such as a covariance or Hessian, which the operator algebra reads as the fact that operands compose or act on the same space.

Its schemas are always concrete, and construction from a schema with unbound dimensions raises. A consumer whose sizes are not yet known holds the operator as a recipe, the operator class and its size-free parameters, and mints the instance once the sizes are bound. The base fixes the action and the square-only queries, and every query raises `LinAlgError` where it is undefined:

```python
class LinOp(Function, ABC):        # the linear subtype of the III.3 base
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...    # (output schema's vector_size, input schema's vector_size)
    @property
    @abstractmethod
    def dtype(self) -> DType: ...
    @abstractmethod
    def to_dense(self) -> Array: ...

    def matvec(self, x: Numeric) -> Numeric: ...
    # shorthand for apply: A x, with a Numeric flattened through
    # the input schema and the result matching the argument's form
    def matmat(self, X: Array) -> Array: ...
    # A X on stacked columns, the operator's registered batched rule;
    # rmatvec / rmatmat apply the transpose

    # square-only queries
    def solve(self, b: Array) -> Array: ...
    def cholesky(self) -> LinOp: ...           # a lower-triangular L with A = L Lᵀ
    def diag(self) -> Array: ...
    def logdet(self) -> Array: ...   # scalar Arrays rather than floats, keeping the queries differentiable
    def trace(self) -> Array: ...

    @property
    def flags(self) -> frozenset[str]: ...      # structure metadata, e.g. "symmetric", "positive_definite"
    def with_flag(self, flag: str) -> Self: ... # functional; construction otherwise fixes the flags
```

A `LinOp` claims `SupportsInverse` and `SupportsLogDetJacobian` (III.3) with a guard that rejects a non-square operator; its inverse comes from the operator algebra, and its `logdet` is the log-Jacobian. Singularity is decided at call time and raised as `LinAlgError`, as for `solve`.

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators that defer to their parts: `ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view. The algebra checks and propagates the schemas: `A @ B` requires `B`'s output schema to equal `A`'s input schema and declares `B`'s input schema and `A`'s output schema as its own sides, `A + B` requires both pairs to match, and `A.T` exchanges the term specs of the two sides: its one input slot accepts the original output's packaging, and its output is the original input, offered whole under that slot's name. Each side keeps its own declaration type, since an `InputSpec` and an `OutputSpec` are different contracts (II.2). Composite operators are tracked terms like any other, with names derived from their operands.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant. A constructor from arrays derives the output declaration from the matrix shape as a whole term under the fixed component `result`, and accepts an `output_spec` that names the component otherwise or fills a type hole; a consumer that knows the event declaration, such as covariance construction (VII.6), passes it. Each also fixes the kind's `raw()` (II.4) as its stored parameterization:
- `DenseLinOp`: the matrix;
- `DiagonalLinOp`: the diagonal;
- `TriangularLinOp`: the triangular matrix, whose flags name the triangle;
- `CholeskyLinOp`: the triangular factor;
- `RootLinOp` and `DiagonalRootLinOp`: the root operator `S` of `A = S Sᵀ`.

A composite's `raw()` is its operand tuple in its constructor's order, such as `(A, B)` for `A @ B` and `(A, c)` for `c * A`, since laziness is its representation.

**The batch form.** `LinOpBatch` is the element batch over operators, a thin `Batch[LinOp]` whose elements share both schemas. It is what a batched `cov` returns. Application is elementwise: a single operator maps over a batch's elements, and a `LinOpBatch` zips with a broadcast-compatible batch of numeric values, element by element in both cases. The queries lift the same way, elementwise to batched results.

### Rationale

Operations mint linear operators, covariances above all, so the kind exists to keep those results first-class (`D4 – Closed system of objects under operations`). The structured subclasses exploit their form automatically behind one interface (`C3 – Computational detail hidden by default, available on demand`), the algebra returns lazy views rather than materialized matrices (`D6 – Single source of truth`), and typing both sides with numeric schemas makes closure concrete: the operator `cov` returns accepts the very draws its distribution produces (`D5 – Explicit, carried structure`).

### Open points

- *Structure-exploiting solves.* Exploiting structure in both operands of `A⁻¹B`, possibly through a dedicated `SolveLinOp`, is open.
- *Flag semantics.* Whether flags only describe structure or also steer which implementation a query selects is open.
- *Batched matrix action.* `matmat` against a batched operand, where a batch axis would meet the operator's matrix axis, and any richer `LinOpBatch` alignment are deferred until a concrete consumer exists.

## III.5 — `RecordSpec`, `Record`, and `NumericRecord`

### Contract

A `RecordSpec` is a `NamedTree` whose leaves are term specifications: the record kind's spec and the **schema** of one structured value — the structure of one event, such as a draw or a stored datum. One class serves both readings because they denote the same space. A record-shaped position inside a schema is a subtree.

When every leaf is a `NumericSpec`, the schema is fully numeric and construction auto-promotes it to a `NumericRecordSpec`. A stored numeric-record term counts as a numeric leaf, since its spec is one, so flattening spans nested numeric structure. The promotion is re-derived whenever a transform constructs a new schema, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it. Beyond the inherited `NamedTree` interface (with `L = TermSpec`), `RecordSpec` adds construction shorthand, lossy inference from a value, and the numeric projection:

```python
class RecordSpec(NamedTree[TermSpec], TermSpec):
    def __init__(self, field_specs: Mapping[str, Any] | None = None, /,
                 **fields: TermSpec | Mapping | tuple[int, ...]) -> None: ...
    # shorthand: a bare shape tuple means NumericArraySpec(shape); a field given None raises
    # TypeError, since None is a pending type (II.2), and an opaque field is OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> RecordSpec: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    def numeric_subset(self) -> NumericRecordSpec: ...   # remove non-NumericSpec leaves
```

`infer_from` types a term-valued field at its own kind, for example a `DistributionSpec` for a `Distribution`-valued field, and nested structure as nested structure.

`NumericRecordSpec` further provides a flat (vectorized) layout of the leaves:

```python
class NumericRecordSpec(NamedTree[NumericSpec], NumericSpec, RecordSpec):
    # a RecordSpec whose leaf type narrows to NumericSpec, itself a NumericSpec:
    # vector_size sums over the leaves, and the flat layout is the tree's
    # canonical order over the leaves' own layouts
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
```

Within one schema a symbolic name refers to one dimension: fields `X: ("obs", "features")` and `coefficients: ("features",)` share the dimension `features`, which no pair of concrete integers can express (II.1). Validation therefore runs one unification over every occurrence of a name across all fields, so data of shapes `(100, 5)` and `(5,)` bind both dimensions consistently while `(100, 5)` and `(7,)` raise. A leaf's `is_valid` checks its own rank and dtype; sizes belong to the one pass, since only it sees every occurrence of a name. A nested spec's schema lies inside the scope, so a name declared within a `DistributionSpec` is the same dimension as that name beside it, binding once whatever the declaration order.

Two rules govern record-shaped positions, symmetric in what arrives. Mapping data materializes into the record's own structure under derived identity, whereas a supplied tracked term is stored with its identity intact as the field's **source**; access is described below. Both conform to the same spec, so structure and identity agree on what a field is.

A `Record` is a `NamedTree` that is a `TrackedTerm` with leaves that are *values*, its structure conforming to its authoritative `RecordSpec`. `NumericRecord` is the specialization in which every leaf's value implements `Numeric`, so it carries a `NumericRecordSpec`.

Since the structure of `Record` matches that of its schema, the following invariants must hold:
1. *matching keys:* `record.keys() == record.spec.keys()`.
2. *valid values:* for any valid key `p`, the value stored at `p` satisfies `record.spec[p].is_valid`.
3. *matching sub-schemas:* for any valid non-key path `p`, `record.at_path(p).spec == record.spec.at_path(p)`.

Construction binds the schema, so a `Record` always carries the concrete, bound form, and the data and its schema agree.

Two records are equal when they share a class, a `RecordSpec`, and field-by-field equal data. A stored array, opaque value, or function counts by its raw value, so a record built from the views of another record equals it. Because the schema is carried, an identity transform that threads it through compares equal to its input. A transform that instead rebuilds the schema by inference matches only when that inference recovers the original, for instance when the original schema was itself produced by `infer_from`.

```python
class Record(NamedTree[Any], TrackedTerm):
    def __init__(self, fields: Mapping[str, Any], /, *,
                 label: str | None = None,
                 event_template: RecordSpec | None = None) -> None: ...
    @classmethod
    def from_fields(cls, **fields: Any) -> Record: ...
        # Field names such as `label` and `name` remain ordinary data fields.
        # Mapping-valued fields are subtrees; the root label defaults to the fields.

    @property
    def spec(self) -> RecordSpec: ...
    def to_numeric(self) -> NumericRecord: ...  # requires every leaf to be numeric
    def raw(self, path: str | tuple[str, ...] | None = None) -> Any: ...
    # the whole record as the nested mapping of raw leaves, or one node's at path:
    # a field's raw value, or a subtree's nested mapping

    @classmethod
    def from_field_values(cls, spec: RecordSpec, values: Sequence[Any], *,
                          label: str | None = None) -> Record: ...
    # reconstruct from values in the schema's canonical order; ValueError on count/shape mismatch

    def select(self, *fields: str, **mapping: str) -> dict[str, Any]: ...
    # fields into a plain dict for **-splatting into a `Function` call;
    # keywords remap: select(x="r") == {"x": self["r"]}
    def select_all(self) -> dict[str, Any]: ...   # every top-level field, ready to splat
```

`select` resolves each argument with `at_path`, so a key selects a leaf and a partial path a subtree view, and returns a plain `dict` of tracked values carrying no schema; its purpose is `**`-splatting a value's parts into a `Function` call, with `select_all` the whole-record form over the top-level children.

**Storage and access are separate contracts.** Storage retains the representation and the source: leaves are held in native form, so a supplied `NumericArray`'s array is stored as that array, and a supplied term's identity is held as a reference or a descriptor per the provenance mode (II.4). Access returns views. `record[key]`, `values()`, and `items()` give each field as a view (II.4) of the field's kind, named by its key:
1. an array field as a `NumericArray`;
2. an opaque field as an `Opaque`;
3. a callable field as a `Function`;
4. a stored term of another kind, such as a law, as a copy of that term.

`record.at_path(path)` gives the field's view at a key and a sub-`Record` view at an interior path (II.6). `record.raw(path)` returns the stored representation, and `record.raw()` the whole record's nested mapping of raw leaves — the record kind's raw host.

When every leaf is numeric, a `Record` is a `NumericRecord`. Leaves are stored in native form, for example a bare array or an `xarray` container, and convert to `jax.Array` only at the compute boundary, which is the pytree flatten that `grad`, `vmap`, and `jit` traverse and `to_vector`; each leaf converts at most once. A `Record` is promoted exactly as its schema is (above): when every leaf is numeric and no explicit non-numeric schema vetoes it, re-derived by every transform. Flat vectorization reads its layout from the schema: `leaf_shapes`, `vector_size`, and the canonical order. Flattening is numeric-only, which is why `NamedTree` itself has no `flatten`.

```python
class NumericRecord(Record, Numeric):
    @property
    def vector_size(self) -> int: ...
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, spec: NumericRecordSpec, vec: Array, *,
                    label: str | None = None) -> NumericRecord: ...
```

**Vector-space arithmetic.** `NumericRecord` implements the `Numeric` interface of II.3, so functions act on it in the two ways stated there. ProbPipe's own operators preserve structure and return tracked terms. They are the vector-space set, which is `+` and `-` between records sharing a schema and scalar `*` and `/`, and `map(f)` for entrywise maps, so `record.map(jnp.cos)` is the tracked form of `jnp.cos(record)`. Array-shaped behavior such as broadcasting and positional indexing stays with arrays, and `__array_ufunc__` is left undefined, so NumPy and JAX functions behave alike on the same object.

### Rationale

A `Record` is the *values* half of `C1 – Uniform interface to functions, distributions, and values`: a distribution's draw is a `Record` (or a `RecordBatch` for many), and a function over named values consumes one. One class serves the schema and the kind because they denote the same space: a second tag class would be a distinction without mathematical content, converted at every construction site (`D1 – Mathematical fidelity`, `D6 – Single source of truth`). Carrying the schema forward from its producer rather than re-inferring it downstream is `D5 – Explicit, carried structure` made concrete.

### Notes

- *Pytrees.* `Record` and `NumericRecord` are registered as JAX pytrees for advanced use, and the native `NamedTree` methods are the supported interface. JAX traversal follows the pytree registration, which can disagree with ProbPipe on what is a leaf, so users applying raw JAX functions are responsible for the documented behavior. Record equality is structural value equality, which is weaker than treedef equality. The registration's children are the field arrays, with a `NumericRecord`'s native leaves converting at this boundary, and its static aux data is the schema alone, since identity is boundary-attached (II.4); native container types therefore never enter a trace either. A round-trip returns bare-array leaves and keeps a `Record` a `Record`.

- *Single-field presentation.* A `Record` is a container and presents as one, whatever its field count: a field is reached by indexing, and the array operators are the vector-space operations above.
- *Construction validation.* Construction checks each leaf against its spec's `is_valid`, which validates structure only; for a `NumericArraySpec` that is shape and dtype, with dtype checked by `numpy.can_cast` same-kind, so a widening promotion or a within-kind narrowing passes and a cross-kind conversion raises. A `NumericArraySpec`'s `support` is descriptive metadata: checking it is data-dependent and entrywise, and it reduces to a Python `bool`, so it cannot run under `jax.jit` tracing, where construction also happens because pytree unflatten reconstructs a value inside the trace. Leaf validation is skipped during pytree unflattening, where a leaf's shape is transform-relative.

## III.6 — `RecordBatch` and `NumericRecordBatch`

### Contract

A `RecordBatch` is a batch of `Record`s that all conform to one shared `RecordSpec`. It is the batched value a `Function` produces and consumes, such as the many draws a `sample` yields. It is a *collection* of records. `NumericRecordBatch` is the all-array specialization. Indexing addresses both axes and stays unambiguous by dispatching on the type of its argument:

```python
class RecordBatch(Batch[Record]):
    def raw(self, path: str | tuple[str, ...] | None = None) -> Any: ...
    # the storage view: the nested mapping of raw columns, each field's raw batch form (II.5);
    # at a path, a field's column or the columns beneath an interior node, as Record.raw(path)

    def __getitem__(self, key: int | slice | tuple[int, ...] | str | tuple[str, ...]) -> Record | RecordBatch | Batch: ...
    # int / slice (or a tuple of ints) -> an element Record or a sub-batch, indexing the batch axes
    # field path (str or tuple of strs) -> the field's tracked batch column:
    #   a NumericArrayBatch for an array field, the matching element batch otherwise;
    #   a sub-RecordBatch if nested
```

Storage is columnar: per-field columns in each field's batch form. A field column is therefore a direct tracked view, and an element `Record` is assembled on demand from the columns. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `element_spec`. The field transforms of II.6 apply columnwise: `with_path_names`, `without`, `merge`, `replace`, and `map` act on every element's fields at once and return a batch on the same levels, and `select` and `select_all` splat fields as they do for a `Record` (III.5). A `NumericRecordBatch` additionally presents the coordinates view through `__jax_array__`, which is its batched `to_vector`, so the tracked batch, the raw columns, and the flat coordinates are three presentations of one store.

When every element is a `NumericRecord`, the batch is a `NumericRecordBatch`: a pytree of arrays whose leading dimensions are the `batch_shape`, bound to one shared `NumericRecordSpec`. Its columns are the leaves `vmap` / `grad` / `jit` traverse. The batch is *rebuilt* on the way out only where the level identity of what arrives is recoverable — a transform that preserves every batch axis, or one that removes all of them, which yields a single `NumericRecord`. Mapping one level of several is refused: the pytree unflatten is not told which axis the transform consumed, and no shape records it, so a rebuilt `BatchSpec` could name the wrong level. An operation that knows which level it consumes carries that knowledge itself; the workflow sweep does, mapping raw columns and building each row explicitly. It also adds batched flat vectorization, where `to_vector` stacks one flat vector per element into a `(*batch_shape, vector_size)` array:

```python
class NumericRecordBatch(RecordBatch):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, spec: NumericRecordSpec, vec: Array, *,
                    level_names: str | Iterable[str],
                    axes_per_level: int | Iterable[int] | None = None,
                    label: str | None = None) -> NumericRecordBatch: ...
    # vec has shape (*batch_shape, vector_size): the last axis is the flat dimension
```

Its elements implement `Numeric` (II.3), and the batch's `to_vector` stacks their vectors. A `RecordBatch` is promoted as a `Record` is (III.5): `RecordBatch(...)` and `RecordBatch.stack` return a `NumericRecordBatch` when every column is numeric and no explicit non-numeric `element_spec` vetoes it, and a view over numeric fields, such as a field selected from a batch that also holds opaque fields, is a `NumericRecordBatch` too.

A constructor that mints a level takes the name to give it (II.5), so both constructions here require one: `from_vector` names the levels it reconstructs, which is what lets a multi-level batch round-trip, and `stack` names the single level it introduces.

```python
class RecordBatch(Batch[Record]):
    @classmethod
    def stack(cls, records: list[Record], *, level_name: str,
              element_spec: RecordSpec | None = None,
              label: str | None = None) -> RecordBatch: ...
    # one level of (len(records),); the element spec is taken from the first record
    # when omitted, and every record's fields must be exactly its fields.
    # An omitted label is derived
    # from the first record's -- a batch of `draw` records is
    # about `draw`, so no caller has to invent a label for it.
```

### Rationale

It indexes the batch axes and omits the leaf-keyed `Mapping` contract, so a batch of `N` records reads as `N` records and never as one record of `N` fields, which is `D1 – Mathematical fidelity` at the point where the two would otherwise be conflated.

## III.7 — `Distribution`

### Contract

A `Distribution` is a probability measure over the values its event declaration describes. Its `DistributionSpec` carries the draw's `OutputSpec`, which the property `event_spec` returns. The declaration determines the kind of a draw and the components it exposes (II.2). It is the same declaration type a `Function` carries as `output_spec`, and the attribute names `event_spec` and `output_spec` distinguish a draw from a function's return value. A bare `RecordSpec` is accepted and completed at construction to an exposed record, whose fields are the components. A whole-term event is declared under its component, which a constructor requires: a family takes it first, as `Normal("mu", 0.0, 1.0)` does, and an `OutputSpec`, such as the `event_spec` a family constructor also takes to declare the event's type, names that same component. A constructor fills the declaration's type hole from its parameters with `with_spec`, so the stored declaration is complete. The label is the optional keyword `label=`, which defaults to the family's class name, as `Normal`, and to `p` for any other law (II.4), so `Normal("mu", 0.0, 1.0)` is labeled `Normal` and displays as `Normal(mu)`.

It declares the operations it supports as **capabilities** (III.8), so operational support is decoupled from the class. Its `raw()` is the law detached (II.4). The `raw()` of a field view `d[p]` is the representation of the detached marginal, `d._marginal(p).raw()` (III.8): the backend object where the marginal has one, such as the TFP distribution of a parametric family (VII.1), and the detached law otherwise. That `raw()` raises `ResolutionError` where `d` has no exact marginal at `p`. A draw is a tracked term of the kind the event declaration names.

**Components and fields.** The law's produced slots are exactly `event_spec.components`, and its event paths are the paths of its declaration, each starting with a component (II.2). `OutputSpec(beta=beta_spec)` and `OutputSpec(RecordSpec(beta=beta_spec))` both export `beta`, but the former draws an array and the latter a record. `d[path]` and `marginal` address event paths: for an exposed record, `d["beta"]` is the marginal law of that field under the component `beta`, and for a whole term named `beta` it is `d` itself, so a consumer addresses a law by component whatever its packaging. A projection onto one path returns the leaf or subtree whole, under a component named by the path's final segment; a selection of several paths returns an exposed record of those fields. A whole record declared as `OutputSpec(parameters=RecordSpec(beta=...))` has the output slot `parameters` and the event path `parameters/beta`, which addresses the field `beta` of each draw; composition extracts and reconstructs it as II.2 specifies.

`with_path_names` returns the same law with `OutputSpec.with_path_names` (II.2) applied to its event declaration. A factored joint renames through its factors, and a rename that gathers its components under one node regroups their factors into a packaged sub-joint (IV.1). `with_label` changes only the object label. A polymorphic law binds its dimensions as II.1 specifies. Dimension transforms commute with renames: `d.with_path_names(m).with_dim_sizes(n=3)` is `d.with_dim_sizes(n=3).with_path_names(m)`, and likewise for `with_dim_names` and for a kernel (III.9). A renamed law conditions as the original does on the given translated to the original paths, with each value's fields in the original node's declared order whatever order the caller wrote (VI.6). Its marginal at a node that gathers fields of several of the original's nodes is the original's marginal at the selection of those fields, repackaged under the node, since a transform that preserves the event exposes what the law it wraps supports (III.8).

```python
class Distribution(TrackedTerm):
    def __init__(self, event_spec: OutputSpec | RecordSpec, *, label: str | None = None) -> None: ...
        # a subclass passes the label its caller gave, or its default;
        # a bare RecordSpec completes to the exposed record, OutputSpec(event_spec) (II.2)

    @property
    def spec(self) -> DistributionSpec: ...
    @property
    def event_spec(self) -> OutputSpec: ...     # the event declaration, read from spec
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename or move nodes by event path, keeping the packaging (II.2); the law is unchanged
    def with_dim_sizes(self, **sizes: int) -> Self: ...
    # bind named symbolic dimensions (II.1); a name that is not a free dimension raises
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions before composing (IV.2)
    def __getitem__(self, key: str | tuple[str, ...]) -> Distribution: ...
    # the law itself at a whole term's component, or the field view at another event path;
    # any other argument raises
    @property
    def notation(self) -> str: ...   # the label and the signature, as prior(mu); str() returns it (II.4)
```

**Numeric distributions.** A `NumericDistribution` is a `Distribution` whose `event_spec.spec` is a `NumericSpec` (II.3), so its draws implement `Numeric` and the flat-vector interface applies; a scalar `Normal`'s `NumericArraySpec` event qualifies as a record event does. Membership is read from the declaration, so `isinstance(d, NumericDistribution)` holds if and only if the declaration of `d` is numeric, whatever its class. A class whose every instance is numeric may inherit the marker, and construction checks that each instance is numeric. Exactly the numeric laws have the marker's properties, which, like `event_shape`, are final and computed from the declaration:

- `dtypes` and `supports`: the declared dtype and support of each array leaf, keyed by the leaf's event path.
- `dtype` and `support`: the dtype and the support every array leaf shares, each `None` when the leaves differ.

```python
class NumericDistribution(Distribution):   # the event spec is a NumericSpec
    @property
    def dtypes(self) -> Mapping[str, np.dtype | None]: ...       # each array leaf's dtype, by path
    @property
    def supports(self) -> Mapping[str, Constraint | None]: ...   # each array leaf's support, by path
    @property
    def dtype(self) -> np.dtype | None: ...          # the dtype every array leaf shares, else None
    @property
    def support(self) -> Constraint | None: ...      # the support every array leaf shares, else None
```

**Field views.** `d[path]` returns a `FieldView`: a `Distribution` over the field or field group at `path` that holds a reference to its parent. A selection of several paths is one `FieldView`, whose `path` is the tuple of the selected paths. A view displays as the detached marginal at its path does: it is labeled as `marginal` labels that marginal (VI.8), its signature is its own components (II.4), and it holds the paths its parent holds fixed. So `prior["beta"]` is labeled `prior` and exports the component `beta`, `model["y"]` displays as `model(y)`, and `model["mu"]` at the factor `prior` displays as `prior(mu)`. A view of the whole events of several factors displays factor by factor, as `a(a)·b(b)`, and a view at one factor holds the paths that factor holds fixed before its parent's. The view `post["mu"]` of a posterior `post = condition_on(model, {"y": data})` over `mu` and `tau` displays as `model(mu; y)`. The repr names the class `FieldView` and the path, which distinguishes a view from the detached marginal. A view of a view and the marginal of a view are labeled by the parent's factors whose whole events their paths select, and otherwise by the view's label. The capabilities a view offers are derived from its parent's (III.8). Two selected paths with the same final segment raise `ValueError`, as a colliding rename does (II.6), and so does an empty selection, from indexing and from `_marginal` alike. On a view, `with_dim_sizes` and `with_dim_names` apply to the parent and return the view of the result at the same path, since the view's declaration is the parent's schema at that path and a schema is one dimension scope (II.1).

**The flat-vector law.** A numeric law's law over its coordinates is `evaluate(to_vector, d)`, with the map specialized to `d.event_spec.spec` and carrying an explicitly named array output declaration. Its inverse reconstructs that original event, including singleton and nested record packaging. The map claims the inverse and unit-Jacobian capabilities, so the change-of-variables rule preserves an available density (V.7). This changes the event space by a declared isomorphism; an ordinary representation conversion preserves the event declaration (IV.3). An inference method that works on ℝᵈ also applies the reparameterization of V.12.

```python
class FieldView(Distribution):
    # constructed by Distribution.__getitem__
    @property
    def parent(self) -> Distribution: ...
    @property
    def path(self) -> str | tuple[str, ...]: ...
    # the declaration is the parent's schema at path; a view (II.4)
```

**The distribution term specification.** `DistributionSpec` is the distribution kind's term spec. As a leaf, it types a field holding a matching `Distribution`: one whose own event declaration unifies with the declared one, so the packaging and component names agree and the components bind in one dimension scope (II.1). As an event declaration, it declares a random measure: a distribution whose draws are themselves `Distribution`s.

```python
class DistributionSpec(TermSpec):  # a Distribution; is_valid accepts a matching Distribution
    event_spec: OutputSpec         # the output declaration of one draw (II.2)
```

### Rationale

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to functions, distributions, and values`. A field view is `B4 – No copying at boundaries` at a field, and deriving its capabilities from its parent's ensures a view advertises only what it can compute (`D3 – Capability-based operations`). A view is labeled as the marginal at its path because the two are one law, and its component names that field (`C5 – Naming for unambiguous meaning`). A `Distribution` takes no type parameter, because the type of its draws is a function of the stored declaration and a static parameter could record only the declaration's kind (`D6 – Single source of truth`). Requiring the component and defaulting the label serves `C5 – Naming for unambiguous meaning` on both counts: the component names what a draw holds, which composition and indexing match, and the label names the law for a reader, so neither stands in for the other and the label never enters the mathematics.

### Open points

- *Structuring an atomic event.* Demoting a term-drawing law's output name into a group (`with_path_names({"x": "group/x"})`) is mathematically well-defined, being the canonical isomorphism with the one-field product, but changes the draw's kind, so the packaging rule of II.2 excludes it. A producer that wants the structure declares an exposed one-field record output (II.2) instead; revisit only if a concrete consumer appears.

## III.8 — Distribution capabilities

### Contract

For each operation it supports, a distribution supplies a **capability**: an underscore implementation such as `_sample` or `_mean` that operates on raw forms (II.4). A class **claims** a capability for its instances by implementing the capability's protocol. Where support depends on the instance or the call, the capability carries a **guard**: a predicate on the instance and the call's arguments that decides whether the capability supports this call, as a `LinOp`'s inverse guard rejects a non-square operator (III.4). The matching operation calls the capability through its route (VI.0): protocol membership establishes that the implementation exists, and the guard establishes support for the requested call. A capability called on a well-formed request that its guard rejects raises `ResolutionError` (II.7), and a malformed request raises its own argument error, such as `KeyError` for an unknown path.

```python
@runtime_checkable
class SupportsSampling(Protocol):
    def _sample(self, key: Key, sample_shape: tuple[int, ...] = ()) -> Any: ...
    # one draw for sample_shape=(); a non-empty shape prepends batch axes

@runtime_checkable
class SupportsUnnormalizedLogProb(Protocol):
    def _unnormalized_log_prob(self, value: Any) -> Array: ...   # log-density up to an additive constant

@runtime_checkable
class SupportsLogProb(SupportsUnnormalizedLogProb, Protocol):
    def _log_prob(self, value: Any) -> Array: ...                # the *normalized* log-density (refines the above)

@runtime_checkable
class SupportsRandomUnnormalizedLogProb(Protocol):
    def _random_unnormalized_log_prob(self) -> Distribution: ...
    # for a random measure M: the law of x ↦ log D̃(x) with D ~ M, itself a random function

@runtime_checkable
class SupportsRandomLogProb(Protocol):
    def _random_log_prob(self) -> Distribution: ...   # likewise, with the normalized log-density of a draw

@runtime_checkable
class SupportsMean(Protocol):
    def _mean(self) -> Any: ...       # event-typed: a value shaped like a draw

@runtime_checkable
class SupportsVariance(Protocol):
    def _variance(self) -> Any: ...   # event-typed, like _mean

@runtime_checkable
class SupportsCovariance(Protocol):
    def _cov(self) -> LinOp: ...    # a (d, d) operator over the flat numeric event

@runtime_checkable
class SupportsQuantile(Protocol):
    def _quantile(self, q: ArrayLike) -> Array: ...   # numeric draws: one value per probability in q, per coordinate

@runtime_checkable
class SupportsExpectation(Protocol):
    def _expectation(self, f: Callable[[Any], Array]) -> Array: ...   # exact E[f(X)] for arbitrary f

class SupportsExactConditioning(ABC):        # claimed by inheriting, not structurally
    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution: ...   # the conditional law given fixed values
class SupportsApproximateConditioning(ABC):  # same primitive, returning a stand-in for that law
    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution: ...

@runtime_checkable
class SupportsMarginals(Protocol):
    def _marginal(self, path: str | tuple[str, ...]) -> Distribution: ...   # the detached marginal of a field or field group
```

Here `Key` is a PRNG key and `ArrayLike` an array-or-scalar input. `_expectation` integrates an *arbitrary* function exactly, which in practice means finite support: its argument is an opaque callable, so a guard has only the law to inspect, and only a law exact for every integrand claims the capability. The exact expectation of a structured map is computed through `evaluate`, which dispatches on the map's type (VI.5). A law of finite support, such as an empirical law, computes `_expectation(f)` by evaluating `f` at every atom, in one vectorized call when `f` traces, as `auto` dispatch does (V.9), and one atom at a time otherwise.

`d._marginal(p)` returns a law labeled as `marginal(d, p)` is (VI.8) and declared as the view `d[p]` is (III.7), and at a selection its fields follow the order of the paths.

**Normalization.** A law is **normalized** when it claims one of the following capabilities, and **unnormalized** otherwise:

- `SupportsLogProb`: a density that integrates to one;
- `SupportsSampling`: draws from the law;
- a moment, quantile, or expectation capability: a functional of the law.

Each of these capabilities is defined only for a probability law: a sampler draws from one, and each of the others is defined against one. A kernel's laws are normalized when the kernel claims the conditional twin of one of these (III.9). The classification reads protocol membership alone, so a route decides it without evaluating a body (VI.0), and `condition_on` uses it to return a normalized law (VI.6).

**View derivation.** A `FieldView` derives each capability from its parent's. For a parent `d` and a view `v = d[p]`, with π the extraction of field `p` from an event and `c` the view's component, which is the final segment of `p`:

| capability on `v` | derivation | available when |
|---|---|---|
| `_sample` | co-sample: draw `X ~ d` and return `π(X)` | parent `SupportsSampling` |
| `_mean` | projection: `mean(d)[p]`, since `E[πX] = π E[X]` | parent `SupportsMean` |
| `_variance` | restriction of `variance(d)` to the coordinates of `p` | parent `SupportsVariance` |
| `_cov` | the sub-block `P Σ Pᵀ`, with `P` the coordinate-selection `LinOp`, built lazily through the operator algebra | parent `SupportsCovariance`, numeric field |
| `_quantile` | restriction of the parent's per-coordinate quantiles to `p` | parent `SupportsQuantile`, numeric field |
| `_expectation` | composition: `d._expectation(f ∘ π)` | parent `SupportsExpectation` |
| `_log_prob` / `_unnormalized_log_prob` | via the detached marginal `d._marginal(p)` | parent `SupportsMarginals`, exact at `p`, and the marginal at `p` reports the density |
| `_mean`, `_variance`, `_cov`, `_quantile` of a parent that claims none | the moment of the detached marginal `d._marginal(p)` | parent `SupportsMarginals`, exact at `p`, and the marginal at `p` reports the moment; a numeric field for `_cov` and `_quantile` |
| `_marginal` at `q = c/r` | path composition: `d._marginal(p/r)` | parent `SupportsMarginals`, exact at `p/r` |
| `_condition_on` a proper sub-field `s ⊂ p` | conditioning commutes with marginalization: `d.condition_on(s)[p ∖ s]`, both sides the law of `p ∖ s` given `s` | parent conditioning available for `s` |

The projection rows are exact whenever the parent's answer is, and the rows through the marginal are exact per path. Each row needs only the parent capability it names, so a view of a parent that does not sample still carries its projected moments.

Each derived capability carries the parent's guard for the call that its derivation makes, such as the parent's quantile guard at the same probabilities for the quantile row. A moment row through the marginal carries the parent's marginal guard at `p` and then the marginal's guard of the moment, so the view returns or raises as the marginal does. The sample row passes its key and sample shape to the parent unchanged, so a view's draw at a key is the projection at `p` of its parent's draw at that key. A given that covers every field of the view is malformed, since no law remains, so the conditioning row's guard rejects it and `_condition_on` raises `ValueError`. A selection drops each node that the given covers and keeps the rest.

**The marginal's capabilities.** A law claiming `SupportsMarginals` may define the companion `_marginal_capabilities(path)`, which returns the capabilities its exact marginal at `path` claims, read from its declarations without building the marginal. The marginals of a law that defines none claim the law's own capabilities. A factored joint reports the capabilities of the factors a marginal keeps, and an empirical law reports sampling and its moments but no density, so a view of `Normal("a", 0.0, 1.0) * EmpiricalDistribution(atoms, component="b")` at `a` claims a density and one at `b` does not. A view reads the report once, at construction, since its path is fixed. It offers the density rows when the report includes the density, and a moment row its parent does not claim when the report includes the moment, so a view of a dependent joint at a root factor carries the factor's exact moments, which the joint does not claim. The projection rows derive from the parent's own capabilities, since each computes from the parent's answer.

### Rationale

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Structural protocol membership identifies an implementation without requiring inheritance; its guard determines the calls it supports (VI.0). A transform that preserves the event exposes exactly the capabilities of whatever it wraps, and a field view offers those its parent's capabilities can derive, so advertised support matches actual support in both cases.

## III.9 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `Distribution` is the empty-given case, whose law exists without a conditioning value, so the unconditional operations apply to it, and a kernel with a non-empty given has a law only once its given is bound. The two are distinct tracked types that both subclass `TrackedTerm`. A `ConditionalDistribution` and its spec always carry a non-empty `given_spec`, since binding the last required slot returns a `Distribution` directly; the empty-given case is `DistributionSpec`'s.

A `ConditionalDistribution` carries a `given_spec`, which is the `InputSpec` of independently bindable slots it conditions on (II.2), and an `event_spec`, which is the output declaration of one produced draw and is read as for a `Distribution` (III.7); both are read from its stored `ConditionalDistributionSpec`. Its `raw()` is the kernel detached (II.4), as a law's is. A kernel's given and event are distinct *roles*, which are the value conditioned on and the law produced, so their given-slot and produced-component names stay disjoint even when the two spaces coincide. A Markov kernel with `S = T` uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. Symbolic dimensions are scoped over the two sides jointly, so a name shared between given and event fields is one dimension, bound by `with_dim_sizes` or, in the fused conditional calls, from the given value at call time. `with_path_names` renames or moves names across both sides, returning the same kernel: the event side behaves as a `Distribution`'s, and on the given side a path-valued target may split or group slots, since the kernel's `given_spec` alone fixes its slots. The slots that a split makes bind independently: after `{"theta/a": "a"}` splits `theta`, binding `a` alone curries the kernel, and the original kernel is evaluated once the rest of `theta` is bound. An optional slot that a rename moves whole stays optional, and one that a move groups into a structured slot is required, since the structured slot's value holds it. A `Function`'s signature fixes its input slots (III.3), so restructuring them makes a new signature, obtained by wrapping the callable in one that takes the parameters wanted.

`condition_on(K, s)` binds the given slots and evaluates the kernel to a `Distribution`. The evaluation is exact unless the kernel claims `SupportsApproximateConditioning`, as an amortized posterior does, whose evaluation stands in for the posterior it was trained to approximate (VII.7). `condition_on` normalizes a result that is unnormalized (VI.6). `sample(K, given=s)`, `log_prob(K, y, given=s)`, and `mean(K, given=s)` are the **fused conditional calls**, with the invariant `op(K, given=s) == op(condition_on(K, s))`: the same law for exact realizations, and equal in law for their random draws. Equal draws need a shared workflow scope and, within it, the same sampling realization, random-event identity, and key derivation (V.8). A fused call served by an approximate route records the route and its assumptions in place of that invariant. Binding a subset of the given slots *curries* to a smaller `ConditionalDistribution` (VI.6).

```python
class ConditionalDistribution(TrackedTerm):
    def __init__(self, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | RecordSpec, *, label: str | None = None) -> None: ...
        # a subclass passes the label its caller gave, or its default; the event is read as a law's (III.7)
        # given before event, as in FunctionSpec
    @property
    def spec(self) -> ConditionalDistributionSpec: ...
    @property
    def given_spec(self) -> InputSpec: ...               # read from spec
    @property
    def event_spec(self) -> OutputSpec: ...              # the event declaration, read from spec
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions on both sides (II.1)
    @property
    def notation(self) -> str: ...   # the label and the signature, as glm(y | beta); str() returns it (II.4)
    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution | ConditionalDistribution: ...
    # the required primitive: the law K(given, ·), or a curried kernel for a partial given; every given value arrives in given, and the keyword options configure the kernel or the method, such as a budget

@runtime_checkable
class SupportsConditionalSampling(Protocol):
    def _conditional_sample(self, given: Record | Mapping[str, Any], key: Key, sample_shape: tuple[int, ...] = ()) -> Any: ...
@runtime_checkable
class SupportsConditionalLogProb(Protocol):
    def _conditional_log_prob(self, given: Record | Mapping[str, Any], value: Any) -> Array: ...
@runtime_checkable
class SupportsConditionalMean(Protocol):
    def _conditional_mean(self, given: Record | Mapping[str, Any]) -> Any: ...
# … and likewise SupportsConditionalVariance (_conditional_variance(given) -> Any),
#   SupportsConditionalCovariance (_conditional_cov(given) -> LinOp),
#   SupportsConditionalExpectation (_conditional_expectation(given, f, …) -> Array),
#   SupportsConditionalMarginals (_conditional_marginal(given, path) -> Distribution).
```

The conditional vocabulary is closed by one rule: every unconditional capability has a conditional counterpart whose method prepends the given to the unconditional signature. The two vocabularies stay mirrored by construction, and a capability added on the unconditional side names its conditional twin automatically.

A conditional capability receives one given value, as its signature declares, so a caller with a batch of givens maps the capability over them. A joint samples a dependent factor this way, with one given and one key per draw of the factor's producers, vectorized for numeric givens and one draw at a time otherwise.

`conditional_distribution` builds a kernel from a function of its given values that returns a law (IV.4).

**The numeric special cases.** A `ConditionalDistribution` has *two* sides, and either can be numeric, so the single `Numeric` prefix becomes positional: `Numeric` before `Conditional` marks the **given** side numeric, `Numeric` before `Distribution` marks the **event** side numeric, and `FullyNumeric*` marks both. Each is a marker only, as `NumericDistribution` is.

```python
class ConditionalNumericDistribution(ConditionalDistribution): ...   # event numeric: every K(s, ·) is a NumericDistribution
class NumericConditionalDistribution(ConditionalDistribution): ...   # given numeric: the conditioning value is a numeric vector
class FullyNumericConditionalDistribution(
        NumericConditionalDistribution, ConditionalNumericDistribution): ...   # both sides numeric
```

**The conditional distribution term specification.** `ConditionalDistributionSpec` is the conditional-distribution kind's term spec. As a leaf, it types a field holding a matching `ConditionalDistribution`: one whose given slots are exactly the declared names, in any order, and whose slot specs each accept the declared ones in the scope that the given side shares with the event side (II.1). The given side is checked contravariantly, so every value the declared slot admits is one the kernel accepts. The event side is a declaration, checked as for `DistributionSpec`, and the given side is always an `InputSpec`:

```python
class ConditionalDistributionSpec(TermSpec):  # a ConditionalDistribution; is_valid accepts a match
    given_spec: InputSpec          # the named slots it conditions on, non-empty
    event_spec: OutputSpec         # the output declaration, as for DistributionSpec
```

### Rationale

Applying a `ConditionalDistribution` to a conditioning value returns a `Distribution`, which ensures `D4 – Closed system of objects under operations` is satisfied. A `ConditionalDistribution`'s capabilities are the `Distribution` capabilities shifted by one conditioning argument (`D3 – Capability-based operations`), so a single operation vocabulary applies to conditional distributions too, under the rule that *`Distribution` and `ConditionalDistribution` behave as similarly as possible*. The capabilities use distinct `_conditional_*` method names because a `@runtime_checkable` check matches on method name alone, so reusing `_sample` / `_log_prob` would corrupt the unconditional capability checks. `_condition_on` is the exception: fixing given fields means the same thing on both types, so a `ConditionalDistribution` claiming a conditioning capability is intended rather than a collision, and the names stay distinct where the meanings differ. The same rule is why the two conditioning capabilities are claimed by inheriting rather than structurally: whether `_condition_on` returns the conditional law or a stand-in for it is a claim about the result, which no check on method names can read, so two structural protocols declaring it would match the same classes.

## III.10 — `DistributionBatch` and `ConditionalDistributionBatch`

### Contract

A `DistributionBatch` is a `Batch` of `Distribution`s: `N` separate distributions sharing one event declaration, indexed along a batch axis. A `ConditionalDistributionBatch` is the same construction over `ConditionalDistribution`s: `N` separate conditional distributions sharing one `given_spec` and one event declaration. The shared event declaration is the elements' `event_spec`, so a batch of random measures declares term-valued draws as its elements do. They are the native batch forms of `DistributionSpec`- and `ConditionalDistributionSpec`-valued draws. Each stores its elements as `OpaqueBatch` does (III.2), so its `raw()` is the object array of the stored laws or kernels.

```python
class DistributionBatch(Batch[Distribution]):
    @property
    def event_spec(self) -> OutputSpec: ...   # the shared declaration, read from spec

class ConditionalDistributionBatch(Batch[ConditionalDistribution]):
    @property
    def given_spec(self) -> InputSpec: ...               # the shared declarations, read from spec
    @property
    def event_spec(self) -> OutputSpec: ...
```

### Rationale

This is `D1 – Mathematical fidelity` on the distribution layer: a `DistributionBatch` of `N` laws is a *collection of separate measures*, distinct from one *joint* law over a product space, as a `RecordBatch` of `N` draws is distinct from one `Record` of `N` fields. It is the natural result of a vectorized operation that yields many distributions: sweeping a parameter batch through a `ConditionalDistribution` produces a `DistributionBatch` of conditioned laws.
