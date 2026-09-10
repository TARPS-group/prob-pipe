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

It carries the full set of array operators, for example arithmetic and comparison, and the coordinate protocols. Its arithmetic returns tracked terms under a deterministically derived, evaluation-order name, with identity attached as for any operation (II.4).

`NumericArrayBatch` is the kind's batch form: a `Batch` whose `element_spec` is the `NumericArraySpec` and whose storage is one array with the batch axes leading — the same split `RecordBatch` uses, with one column instead of many. An array with leading axes is just an array; the batch form is what carries the level names, the shared spec, and provenance.

### Rationale

The full set of array operators is safe here and only here: with no fields, an expression on one array has exactly one meaning (`D1 – Mathematical fidelity`).

## III.2 — `Opaque`

### Contract

`Opaque` adds identity and nothing else, and its `raw()` is the wrapped value. `OpaqueSpec` is the fallback spec: it admits a value that no other kind admits, so a collection such as a list, a tuple, or a set is opaque:

```python
class OpaqueSpec(TermSpec):        # the fallback spec; is_valid accepts a value no other kind's spec class admits
    meta: Hashable
```

`OpaqueBatch` is its batch form. It **stores** its elements rather than materializing them. Its `raw()` is an object array of the stored raw values.

### Rationale

The kind exists so that closure under operations holds for every return value (`D4 – Closed system of objects under operations`), and it adds identity and nothing else because a richer interface would promise structure the value does not declare (`D1 – Mathematical fidelity`).

## III.3 — `Function`

### Contract

The function kind's base type is `Function`. A `Function` is a tracked term that wraps exactly one Python callable as its representation and carries a `FunctionSpec`, whose sides it exposes as the `input_spec` and `output_spec` views; either side is optional, as in the spec. A `Function` also carries a frozen `inspect.Signature`, which is authoritative for Python argument binding, since parameter kinds, defaults, and variadic parameters are not expressible in a value schema; the `input_spec` is authoritative for the value schema. Construction validates their one-for-one correspondence, so binding an argument binds a slot by name. Its `raw()` is the wrapped callable.

A `Function` also carries an `output_name`, which is the label its results receive and is separate from its own `name` and from its `output_spec`. Under `@function`, `name` defaults to the callable's `__name__` and `output_name` to the initial `name`, captured once at construction, so `with_name` changes only the function's label, and renaming a result changes only that result's label. Neither label enters spec equality, but a whole-term result's component defaults to `output_name`, captured once at construction; `OutputSpec(mean=None)` names it otherwise. A function an operation derives, such as `inverse(f)`, fixes its result label at construction, and its output declaration follows the operation's result rule (VI.0).

```python
@function(name="predict", output_name="prediction",
          output_spec=OutputSpec(mean=None))
def predict_impl(theta, x):
    return x @ theta

prediction = predict_impl(theta, x)
# predict_impl.name == "predict"; prediction.name == "prediction"
# output component: mean; the inferred return is an array, not a record
```

A `Function` is invoked two ways. `apply` evaluates the wrapped callable at a point: given values that conform to `input_spec`, it returns one conforming to `output_spec`, with no tracking or lifting — the raw map that operations such as change of variables build on. `__call__` runs the **call path**, which is the base's one extension point: the base fills it with plain evaluation, and the engine layer (Part V) replaces it once, at import. The base also carries its **controls** (V.2), set at construction and revised functionally by `with_options`; it gives them no meaning, and the engine reads them at call time.

A `Function` is constructed directly or using the `@function` decorator. `FunctionSpec`, which is the function kind's term spec, admits any callable.

```python
class FunctionSpec(TermSpec):      # the function kind's spec; is_valid accepts any callable
    input_spec: InputSpec | None   # None: that side's structure unspecified
    output_spec: OutputSpec | None
```

```python
class Function(TrackedTerm):
    def __init__(self, name: str, fn: Callable, *,
                 input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
                 output_spec: OutputSpec | TermSpec | None = None,
                 output_name: str | None = None,
                 differentiable: NumericSpec = ...) -> None: ...
                 # optional differentiability claim (V.11)
    @property
    def spec(self) -> FunctionSpec: ...
    @property
    def input_spec(self) -> InputSpec | None: ...                   # view on spec
    @property
    def output_spec(self) -> OutputSpec | None: ...                 # view on spec
    @property
    def output_name(self) -> str: ...                              # result label, outside spec
    @property
    def options(self) -> Mapping[str, Any]: ...          # the controls; opaque to the base
    def with_options(self, **controls) -> Self: ...      # functional update
    def apply(self, *args, **kwargs) -> Any: ...
    # evaluate the wrapped callable at a point (input_spec -> output_spec),
    # with no tracking or lifting; the raw map operations build on
    def __call__(self, *args, **kwargs) -> Any: ...
    # run the call path: plain evaluation on the base, the Part V engine after import

def install_call_engine(engine: Callable[..., Any]) -> None: ...
    # replaces the call path, once, at import time; until then calls evaluate plainly.
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

def is_invertible(f: Any) -> bool: ...      # the claim together with its instance guard
```

### Rationale

Defining the base in the value layer keeps the layering strict: the representation is fixed here, the call engine arrives by upward registration (`D2 – Generality first`), and `LinOp` and the specs reference `Function` downward — the split the package structure realizes as `values/_function_base.py` and `functions/`. Invertibility as a capability is `D3 – Capability-based operations`: an invertible map is an ordinary `Function` that additionally claims `SupportsInverse`, so it evaluates, composes, and pushes forward like any other, with *bijector* reserved for the mathematical statement. The Jacobian determinant is a separate claim for the same reason, since a map can be invertible without a tractable determinant, and change of variables asks for exactly the pair.

## III.4 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces and the linear subtype of `Function` (III.3). It therefore applies, composes, and evaluates like any map; the operator algebra and the structured queries below are what linearity adds. Its action is the map the base carries: `apply` evaluates the operator at a `Numeric` conforming to its input schema and returns the matching form, with the operator's parameters as private state. `matvec`, `matmat`, `rmatvec`, and `rmatmat` are the linear-algebra names for the action and its transpose, and `matmat` is the operator's registered batched rule. Its output declaration names its component; a constructor given only a codomain shape declares the output whole under the operator's `output_name` (III.3). Its domain is the `NumericSpec` (II.3) of its single input slot, and its codomain is `output_spec.spec`; an exposed record output may have several components while remaining one numeric value. There is no operator-specific accessor beside these declarations. It therefore maps whatever `Numeric` its sides declare, for example a bare array under a `NumericArraySpec` side, so an operator over a scalar law's draws needs no single-field placeholder. The two sides coincide for an endomorphism such as a covariance or Hessian, which the operator algebra reads as the fact that operands compose or act on the same space.

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
    def cholesky(self) -> LinOp: ...           # a triangular factor L with A = L Lᵀ
    def diag(self) -> Array: ...
    def logdet(self) -> Array: ...   # scalar Arrays rather than floats, keeping the queries differentiable
    def trace(self) -> Array: ...

    @property
    def flags(self) -> frozenset[str]: ...      # structure metadata, e.g. "symmetric", "positive_definite"
    def with_flag(self, flag: str) -> Self: ... # functional; construction otherwise fixes the flags
```

A `LinOp` claims `SupportsInverse` and `SupportsLogDetJacobian` (III.3) only when they apply: the claim is guarded per instance by squareness, its inverse comes from the operator algebra, and its `logdet` is the log-Jacobian. Singularity, which no construction-time check decides, is raised at call time as `LinAlgError`, as for `solve`.

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators that defer to their parts: `ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view. The algebra checks and propagates the schemas: `A @ B` requires `B`'s output schema to equal `A`'s input schema and declares `B`'s input schema and `A`'s output schema as its own sides, `A + B` requires both pairs to match, and `A.T` exchanges the term specs of the two sides: its one input slot accepts the original output's packaging, and its output is the original input, offered whole under that slot's name. The component mappings themselves are not swapped, since an `InputSpec` and an `OutputSpec` are different contracts (II.2). Composite operators are tracked terms like any other, with names derived from their operands.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant. A constructor from arrays derives the output declaration from the matrix shape, whole under the operator's `output_name`, and accepts an `output_spec` that names the component otherwise or fills a type hole; a consumer that knows the event declaration, such as covariance construction (VII.6), passes it. Each also fixes the kind's `raw()` (II.4) as its stored parameterization, for example the matrix for `DenseLinOp` or the diagonal for `DiagonalLinOp`; a composite's `raw()` is its operand tuple, since laziness is its representation.

**The batch form.** `LinOpBatch` is the element batch over operators, a thin `Batch[LinOp]` whose elements share both schemas. It is what a batched `cov` returns. Application is elementwise: a single operator maps over a batch's elements, and a `LinOpBatch` zips with a broadcast-compatible batch of numeric values, element by element in both cases. The queries lift the same way, elementwise to batched results.

### Rationale

Operations mint linear operators, covariances above all, so the kind exists to keep those results first-class (`D4 – Closed system of objects under operations`). The structured subclasses exploit their form automatically behind one interface (`C3 – Computational detail hidden by default, available on demand`), the algebra returns lazy views rather than materialized matrices (`D6 – Single source of truth`), and typing both sides with numeric schemas makes closure concrete: the operator `cov` returns accepts the very draws its distribution produces (`D5 – Explicit, carried structure`).

### Open points

- *Structure-exploiting solves.* Exploiting structure in both operands of `A⁻¹B`, possibly through a dedicated `SolveLinOp`, is open.
- *Flag semantics.* Whether flags only describe structure or also steer which implementation a query selects is open.
- *Batched matrix action.* `matmat` against a batched operand, where a batch axis would meet the operator's matrix axis, and any richer `LinOpBatch` alignment are deferred until a concrete consumer exists.

## III.5 — `RecordSpec`, `Record`, and `NumericRecord`

### Contract

A `RecordSpec` is a `NamedTree` whose leaves are term specifications: the record kind's spec and the **schema** of one structured value — the structure of one event, such as a draw or a stored datum. One class serves both readings because they denote the same space. A record-shaped position inside a schema is a subtree, not a second spec class.

When every leaf is a `NumericSpec`, the schema is fully numeric and construction auto-promotes it to a `NumericRecordSpec`. A stored numeric-record term counts as a numeric leaf, since its spec is one, so flattening spans nested numeric structure. The promotion is re-derived whenever a transform constructs a new schema, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it. Beyond the inherited `NamedTree` interface (with `L = TermSpec`), `RecordSpec` adds construction shorthand, lossy inference from a value, and the numeric projection:

```python
class RecordSpec(NamedTree[TermSpec], TermSpec):
    def __init__(self, field_specs: Mapping[str, Any] | None = None, /,
                 **fields: TermSpec | Mapping | tuple[int, ...] | None) -> None: ...
    # shorthand: a bare shape tuple means NumericArraySpec(shape) and None means OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> RecordSpec: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    def numeric_subset(self) -> NumericRecordSpec: ...   # remove non-NumericSpec leaves
```

`infer_from` types a term-valued field at its own kind, for example a `DistributionSpec` for a `Distribution`-valued field, and nested structure as nested structure, so inference never mistypes a term as the raw value it resembles.

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

Two rules govern record-shaped positions, symmetric in what arrives. Mapping data materializes into the record's own structure under derived identity, whereas a supplied tracked term is stored with its identity intact as the field's **source**; access is described below. Both conform to the same spec, so structure and identity never disagree about what a field is.

A `Record` is a `NamedTree` that is a `TrackedTerm` with leaves that are *values*, its structure conforming to its authoritative `RecordSpec`. `NumericRecord` is the specialization in which every leaf's value implements `Numeric`, so it carries a `NumericRecordSpec`.

Since the structure of `Record` matches that of its schema, the following invariants must hold:
1. *matching keys:* `record.keys() == record.spec.keys()`.
2. *valid values:* for any valid key `p`, the value stored at `p` satisfies `record.spec[p].is_valid`.
3. *matching sub-schemas:* for any valid non-key path `p`, `record.at_path(p).spec == record.spec.at_path(p)`.

Construction binds the schema, so a `Record` always carries the concrete, bound form and never an unbound dimension, and the data and its schema cannot disagree.

Two records are equal when they share a class, a `RecordSpec`, and field-by-field equal data. Because the schema is carried rather than re-inferred, an identity transform that threads it through compares equal to its input. A transform that instead rebuilds the schema by inference matches only when that inference recovers the original, for instance when the original schema was itself produced by `infer_from`.

```python
class Record(NamedTree[Any], TrackedTerm):
    def __init__(self, name: str, fields: Mapping[str, Any] | None = None, /, *,
                 spec: RecordSpec | Mapping | None = None,
                 name_is_auto: bool = False,
                 **kw_fields: Any) -> None: ...
        # name is the required first argument (semantic identity)
        # name_is_auto (II.4): user constructions leave it False
        # a mapping-valued field is a subtree, never a leaf (II.6)
        # Binds to the declaration if given (structural validation); nested mapping
        # data is normalized to a RecordSpec.
        # Otherwise, infers it once via RecordSpec.infer_from.

    @property
    def spec(self) -> RecordSpec: ...
    def to_numeric(self) -> NumericRecord: ...  # requires every leaf to be numeric
    def raw(self, path: str | tuple[str, ...] | None = None) -> Any: ...
    # the stored representation: a field's raw value at path, or the whole
    # record as the nested mapping of raw leaves

    @classmethod
    def from_field_values(cls, name: str, spec: RecordSpec, values: Sequence[Any]) -> Record: ...
    # reconstruct from values in the schema's canonical order; ValueError on count/shape mismatch

    def select(self, *fields: str, **mapping: str) -> dict[str, Any]: ...
    # fields into a plain dict for **-splatting into a `Function` call;
    # keywords remap: select(x="r") == {"x": self["r"]}
    def select_all(self) -> dict[str, Any]: ...   # every top-level field, ready to splat
```

`select` resolves each argument with `at_path`, so a key selects a leaf and a partial path a subtree view, and returns a plain `dict` of tracked values carrying no schema; its purpose is `**`-splatting a value's parts into a `Function` call, with `select_all` the whole-record form over the top-level children.

**Storage and access are separate contracts.** Storage retains the representation and the source: leaves are held in native form, so a supplied `NumericArray`'s array is stored as that array, and a supplied term's identity is held as a reference or a descriptor per the provenance mode (II.4). Access never returns the stored source itself: `record[path]` returns a view (II.4) of the field's kind. An interior path yields a sub-`Record` view. `record.raw(path)` returns the stored representation, and `record.raw()` the whole record's nested mapping of raw leaves — the record kind's raw host.

When every leaf is numeric, a `Record` is a `NumericRecord`. Leaves are stored in native form, for example a bare array or an `xarray` container, and convert to `jax.Array` only at the compute boundary, which is the pytree flatten that `grad`, `vmap`, and `jit` traverse and `to_vector`; each leaf converts at most once. A `Record` is promoted exactly as its schema is (above): when every leaf is numeric and no explicit non-numeric schema vetoes it, re-derived by every transform. Flat vectorization reads its layout from the schema: `leaf_shapes`, `vector_size`, and the canonical order. Flattening is numeric-only, which is why `NamedTree` itself has no `flatten`.

```python
class NumericRecord(Record):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, spec: NumericRecordSpec, vec: Array) -> NumericRecord: ...
```

**Vector-space arithmetic.** `NumericRecord` implements the `Numeric` interface of II.3, and its arithmetic follows the two routes stated there. ProbPipe's own operators preserve structure and return tracked terms. They are the vector-space set, which is `+` and `-` between records sharing a schema and scalar `*` and `/`, and `map(f)` for elementwise maps, so `record.map(jnp.cos)` is the tracked form of `jnp.cos(record)`. Array-shaped operations such as broadcasting and positional indexing stay with arrays, and no `__array_ufunc__` is defined, so NumPy and JAX functions never behave differently on the same object.

### Rationale

A `Record` is the *values* half of `C1 – Uniform interface to functions, distributions, and values`: a distribution's draw is a `Record` (or a `RecordBatch` for many), and a function over named values consumes one. One class serves the schema and the kind because they denote the same space: a second tag class would be a distinction without mathematical content, converted at every construction site (`D1 – Mathematical fidelity`, `D6 – Single source of truth`). Carrying the schema forward from its producer rather than re-inferring it downstream is `D5 – Explicit, carried structure` made concrete.

### Notes

- *Pytrees.* `Record` and `NumericRecord` are registered as JAX pytrees for advanced use, and the native `NamedTree` methods are the supported interface. JAX traversal follows the pytree registration, which does not always agree with ProbPipe on what is a leaf, so users applying raw JAX functions are responsible for the documented behavior. Record equality is structural value equality, which is weaker than treedef equality. The registration's children are the field arrays, with a `NumericRecord`'s native leaves converting at this boundary, and its static aux data is the schema alone, since identity is boundary-attached (II.4); native container types therefore never enter a trace either. A round-trip returns bare-array leaves and never promotes a `Record` to a `RecordBatch`.

- *Single-field presentation.* A `Record` is a container and presents as one, whatever its field count: no coercion, no forwarding, and no array operators beyond the vector-space operations above.
- *Construction validation.* Construction checks each leaf against its spec's `is_valid`, which validates structure only; for a `NumericArraySpec` that is shape and dtype, with dtype checked by `numpy.can_cast` same-kind, so a widening promotion or a within-kind narrowing passes and a cross-kind conversion raises. A `NumericArraySpec`'s `support` is **not** part of `is_valid`: it is a data-dependent, element-wise check that reduces to a Python `bool` and so cannot run under `jax.jit` tracing, where construction also happens because pytree unflatten reconstructs a value inside the trace. `support` is therefore descriptive metadata, and invariant 2 (`is_valid`) covers shape and dtype. Leaf validation is skipped on the unflatten path, where a leaf's shape is transform-relative.

## III.6 — `RecordBatch` and `NumericRecordBatch`

### Contract

A `RecordBatch` is a batch of `Record`s that all conform to one shared `RecordSpec`. It is the batched value a `Function` produces and consumes, such as the many draws a `sample` yields. It is a *collection* of records rather than itself a named tree. `NumericRecordBatch` is the all-array specialization. Indexing addresses both axes and stays unambiguous by dispatching on the key's type:

```python
class RecordBatch(Batch[Record]):
    def raw(self) -> Mapping[str, Any]: ...
    # the storage view: the nested mapping of raw columns, each field's raw batch form (II.5)

    def __getitem__(self, key: int | slice | tuple[int, ...] | str | tuple[str, ...]) -> Record | RecordBatch | Batch: ...
    # int / slice (or a tuple of ints) -> an element Record or a sub-batch, indexing the batch axes
    # field path (str or tuple of strs) -> the field's tracked batch column:
    #   a NumericArrayBatch for an array field, the matching element batch otherwise;
    #   a sub-RecordBatch if nested
```

Storage is columnar: per-field columns in each field's batch form. A field column is therefore a direct tracked view, and an element `Record` is assembled on demand from the columns rather than stored a second time. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `element_spec`. The field transforms of II.6 apply columnwise: `with_path_names`, `without`, `merge`, `replace`, and `map` act on every element's fields at once and return a batch on the same levels, and `select` and `select_all` splat fields as they do for a `Record` (III.5). A `NumericRecordBatch` additionally presents the coordinates view through `__jax_array__`, which is its batched `to_vector`, so the tracked batch, the raw columns, and the flat coordinates are three presentations of one store.

When every element is a `NumericRecord`, the batch is a `NumericRecordBatch`: a pytree of arrays whose leading dimensions are the `batch_shape`, bound to one shared `NumericRecordSpec`. Its columns are the leaves `vmap` / `grad` / `jit` traverse. The batch is *rebuilt* on the way out only where the level identity of what arrives is recoverable — a transform that preserves every batch axis, or one that removes all of them, which yields a single `NumericRecord`. Mapping one level of several is refused: the pytree unflatten is not told which axis the transform consumed, and no shape records it, so a rebuilt `BatchSpec` could name the wrong level. An operation that knows which level it consumes carries that knowledge itself; the workflow sweep does, mapping raw columns and building each row explicitly. It also adds batched flat vectorization, where `to_vector` stacks one flat vector per element into a `(*batch_shape, vector_size)` array:

```python
class NumericRecordBatch(RecordBatch):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, spec: NumericRecordSpec, vec: Array, *,
                    level_names: str | Iterable[str],
                    axes_per_level: Iterable[int] | None = None) -> NumericRecordBatch: ...
    # vec has shape (*batch_shape, vector_size): the last axis is the flat dimension
```

A constructor that mints a level takes the name to give it (II.5), so both constructions here require one: `from_vector` names the levels it reconstructs, which is what lets a multi-level batch round-trip, and `stack` names the single level it introduces.

```python
class RecordBatch(Batch[Record]):
    @classmethod
    def stack(cls, records: list[Record], *, level_name: str,
              element_spec: RecordSpec | None = None,
              name: str | None = None) -> RecordBatch: ...
    # one level of (len(records),); the element spec is taken from the first record
    # when omitted, and every record's fields must be exactly its fields.
    # `name` is the one place a batch's name may be omitted: it is then derived
    # from the first record's, and marked auto -- a batch of `draw` records is
    # about `draw`, so no caller has to invent a name for it.
```

### Rationale

It claims only the batch axis and never the leaf-keyed `Mapping` contract, so a batch of `N` records can never read as one record of `N` fields, which is `D1 – Mathematical fidelity` at the point where the two would otherwise be conflated.

## III.7 — `Distribution`

### Contract

A `Distribution[T]` is a probability measure over values of type `T`, where `T` is the implementer-side draw type fixed below. Its `DistributionSpec` carries the draw's `OutputSpec` as `event_spec`, exposed as a view. The declaration determines both the returned kind and its component interface (II.2). It is the same declaration type a `Function` carries as `output_spec`; the names distinguish a draw from a function's return. A bare term spec is accepted and completed at construction: a `RecordSpec` exposes its fields, and any other spec is a whole-term event whose component is the law's `name`, captured once so that `with_name` never moves it. An `OutputSpec`, or a family constructor's `component_name`, names the component otherwise; a constructor fills a type hole from its parameters and stores only the complete declaration.

It declares the operations it supports as **capabilities** (III.8), so operational support is decoupled from the class. Its `raw()` is the law detached (II.4), so a field view's `raw()` is the detached marginal rather than a reference into its parent. A draw is a tracked term of the kind the event declaration names, never wrapped in another kind to make draws uniform.

**Components and fields.** The law's produced slots are exactly `event_spec.components` (II.2); its object name never participates in matching. A field is a path within a record-valued draw, and only a record-valued draw has that field interface. Thus `OutputSpec(beta=beta_spec)` and `OutputSpec(RecordSpec(beta=beta_spec))` both export `beta`, but the former draws an array and the latter a record. `d[name]` and `marginal` address components and, within a record-valued draw, paths: for an exposed record, `d["beta"]` is the marginal law of that field under the component `beta`, and for a whole term named `beta` it is `d` itself, so a consumer addresses a law by component whatever its packaging. A projection onto one path returns the leaf or subtree whole, under a component named by the path's final segment; a selection of several paths returns an exposed record of those fields. Neither reads the object label. A record exposed as `OutputSpec(parameters=RecordSpec(beta=...))` has output slot `parameters` and event field `beta`; composition extracts and reconstructs it using II.2.

`with_path_names` renames or moves event fields under the rules of II.6 and renames a whole-term output component by its declared name. On an exposed record, its component names are derived from the renamed immediate children. On a named whole record, renaming the outer component leaves the record's own fields unchanged. An unqualified name that could address both is ambiguous and raises; the caller disambiguates with a full event path where available. Restructuring never silently changes the event kind or the declaration's exposure form: a path-valued target for a whole-term component is refused. `with_name` changes only the object label. A polymorphic law is legal and binds as II.1 specifies.

**The draw type `T`.** `T` is the implementer-side draw type, derived from the event spec's kind rather than declared independently: the spec is the source, and the bracket is typing documentation. Writing the tracked kind (`Distribution[NumericArray]`) or its raw host (`Distribution[Array]`) names the same array kind: either notation is read at the kind level. Per kind, the implementer type is the kind's raw host, except where the host cannot carry the structure the mathematics needs:

| event spec's kind | implementer draw type `T` |
|---|---|
| `NumericArraySpec` | `Array` |
| `OpaqueSpec` | the wrapped object |
| `FunctionSpec` | a callable |
| `RecordSpec` | `Record` — the flat mapping loses schema and layout |
| `DistributionSpec` | `Distribution` — a draw's own raw form |
| `ConditionalDistributionSpec` | `ConditionalDistribution` |

A `NumericDistribution` is a `Distribution` whose `event_spec.spec` is a `NumericSpec` (II.3), so its draws implement `Numeric` and the flat-vector interface applies; a scalar `Normal`'s `NumericArraySpec` event qualifies as a record event does.

```python
class Distribution[T](TrackedTerm):
    def __init__(self, name: str, event_spec: OutputSpec | TermSpec) -> None: ...
        # event declaration completion follows II.2; labels never supply components

    @property
    def spec(self) -> DistributionSpec: ...
    @property
    def event_spec(self) -> OutputSpec: ...     # view on spec: the event declaration
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename output components or event fields; ambiguous names raise;
    # keys and path-valued targets resolve as for NamedTree.with_path_names (II.6),
    # and the law is unchanged
    def with_dims(self, **sizes: int) -> Self: ...
    # bind named symbolic dimensions (II.1); a conflict with an existing binding raises
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions before composing (IV.2)
    def __getitem__(self, key: str | tuple[str, ...]) -> Distribution: ...
    # the field view at a leaf or group path; raises on a term-drawing law, which has no fields

class NumericDistribution(Distribution): ...   # marker: the event spec is a NumericSpec
```

**Field views.** `d[path]` returns a `FieldView`: a `Distribution` over the field or field group at `path`, holding a reference to its parent rather than a detached law. Sibling views co-sample from one parent draw, so correlation between them is preserved. The capabilities a view offers are derived from its parent's, one by one (III.8).

**The flat view.** A numeric law's law over its coordinates is `evaluate(to_vector, d)`, with the map specialized to `d.event_spec.spec` and carrying an explicitly named array output declaration. Its inverse reconstructs that original event, including singleton and nested record packaging. The map claims the inverse and unit-Jacobian capabilities, so the change-of-variables rule preserves an available density (V.7). This changes the event space by a declared isomorphism; an ordinary representation conversion preserves the event declaration (IV.3). An inference method that works on ℝᵈ also applies the reparameterization of V.12.

```python
class FieldView(Distribution):
    # constructed by Distribution.__getitem__, never by hand
    @property
    def parent(self) -> Distribution: ...
    @property
    def path(self) -> str: ...
    # the declaration is the parent's schema at path; a view (II.4)
```

**The distribution term specification.** `DistributionSpec` is the distribution kind's term spec. As a leaf, it types a field holding a matching `Distribution`. As an event declaration, it declares a random measure: a distribution whose draws are themselves `Distribution`s.

```python
class DistributionSpec(TermSpec):  # a Distribution; is_valid accepts a matching Distribution
    event_spec: OutputSpec         # the output declaration of one draw (II.2)
```

### Rationale

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to functions, distributions, and values`. A field view is `B4 – No copying at boundaries` at a field, and deriving its capabilities from its parent's ensures a view advertises only what it can compute (`D3 – Capability-based operations`). The draw-type table is `B2 – Representations only inside` per kind: an implementer writes over `T` and never sees a tracked draw. Defaulting a whole-term component to the label and capturing it once serves `C5 – Naming for unambiguous meaning` on both counts: a draw is addressable by a meaningful component without a second name in the common case, and the label never enters the mathematics afterward.

### Open points

- *Structuring an atomic event.* Demoting a term-drawing law's output name into a group (`with_path_names({"x": "group/x"})`) is mathematically well-defined, being the canonical isomorphism with the one-field product, but changes the draw's kind, so the rule above excludes it. A producer that wants the structure declares an exposed one-field record output (II.2) instead; revisit only if a concrete consumer appears.

## III.8 — Distribution capabilities

### Contract

For each operation it supports, a distribution supplies a **capability**: an underscore implementation such as `_sample` or `_mean` over `T` (III.7). Where support is partial the capability carries a **guard**, the per-instance predicate that narrows the claim, as squareness narrows a `LinOp`'s invertibility (V.12). The matching operation calls the capability through its route (VI.0): protocol membership establishes that the implementation exists, and the guard establishes support for the requested call.

```python
@runtime_checkable
class SupportsSampling[T](Protocol):
    def _sample(self, key: Key, sample_shape: tuple[int, ...] = ()) -> T: ...
    # one draw for sample_shape=(); a non-empty shape prepends batch axes

@runtime_checkable
class SupportsUnnormalizedLogProb[T](Protocol):
    def _unnormalized_log_prob(self, value: T) -> Array: ...   # log-density up to an additive constant

@runtime_checkable
class SupportsLogProb[T](SupportsUnnormalizedLogProb[T], Protocol):
    def _log_prob(self, value: T) -> Array: ...                # the *normalized* log-density (refines the above)

@runtime_checkable
class SupportsRandomUnnormalizedLogProb(Protocol):
    def _random_unnormalized_log_prob(self) -> Distribution: ...
    # for a random measure M: the law of x ↦ log D̃(x) with D ~ M, itself a random function

@runtime_checkable
class SupportsRandomLogProb(Protocol):
    def _random_log_prob(self) -> Distribution: ...   # likewise, with the normalized log-density of a draw

@runtime_checkable
class SupportsMean[T](Protocol):
    def _mean(self) -> T: ...       # event-typed: a value shaped like a draw

@runtime_checkable
class SupportsVariance[T](Protocol):
    def _variance(self) -> T: ...   # event-typed, like _mean

@runtime_checkable
class SupportsCovariance(Protocol):
    def _cov(self) -> LinOp: ...    # a (d, d) operator over the flat numeric event

@runtime_checkable
class SupportsQuantile[T](Protocol):
    def _quantile(self, q: ArrayLike) -> Array: ...   # numeric draws: one value per level in q, per coordinate

@runtime_checkable
class SupportsExpectation[T](Protocol):
    def _expectation(self, f: Callable[[T], Array]) -> Array: ...   # exact E[f(X)] for arbitrary f

@runtime_checkable
class SupportsConditioning(Protocol):
    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution: ...   # the conditional law given fixed values

@runtime_checkable
class SupportsMarginals(Protocol):
    def _marginal(self, path: str | tuple[str, ...]) -> Distribution: ...   # the detached marginal of a field or field group
```

Here `Key` is a PRNG key and `ArrayLike` an array-or-scalar input. `_expectation` must integrate an *arbitrary* function exactly, which in practice means finite support: its argument is an opaque callable, so a per-call feasibility check has nothing to inspect, and a law that is exact only for special maps must not advertise the capability. Exact moments of structured maps are instead computed by `evaluate`, which dispatches on the map's type.

**View derivation.** A `FieldView` derives each capability from its parent's, so what a view supports is read off the parent. For a parent `d` and a view `v = d[p]`, with π the extraction of field `p` from an event:

| capability on `v` | derivation | available when |
|---|---|---|
| `_sample` | co-sample: draw `X ~ d` and return `π(X)` | parent `SupportsSampling` |
| `_mean` | projection: `mean(d)[p]`, since `E[πX] = π E[X]` | parent `SupportsMean` |
| `_variance` | restriction of `variance(d)` to the coordinates of `p` | parent `SupportsVariance` |
| `_cov` | the sub-block `P Σ Pᵀ`, with `P` the coordinate-selection `LinOp`, built lazily through the operator algebra | parent `SupportsCovariance`, numeric field |
| `_quantile` | restriction of the parent's per-coordinate quantiles to `p` | parent `SupportsQuantile`, numeric field |
| `_expectation` | composition: `d._expectation(f ∘ π)` | parent `SupportsExpectation` |
| `_log_prob` / `_unnormalized_log_prob` | via the detached marginal `d._marginal(p)` | parent `SupportsMarginals`, exact at `p`, and the marginal scores |
| `_marginal` at a sub-path `q` | path composition: `d._marginal(p/q)` | parent `SupportsMarginals`, exact at `p/q` |
| `_condition_on` a sub-field `s ⊂ p` | conditioning commutes with marginalization: `d.condition_on(s)[p ∖ s]`, both sides the law of `p ∖ s` given `s` | parent conditioning available for `s` |

The projection rows are exact whenever the parent's answer is, and the density rows are exact per path. Only sampling requires the parent to sample, so a view on a non-sampling parent still carries its projected moments.

### Rationale

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Structural protocol membership identifies an implementation without requiring inheritance; its guard determines the calls it supports (VI.0). A transform that preserves the event exposes exactly the capabilities of whatever it wraps, and a field view offers those its parent's capabilities can derive, so advertised support matches actual support in both cases.

## III.9 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution[S, T]` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `Distribution` is the empty-given case, a kernel with nothing to condition on, so its marginal law exists and the unconditional operations apply; a kernel with a non-empty given has none. The two are distinct tracked types, and neither inherits from the other. A `ConditionalDistribution` and its spec always carry a non-empty `given_spec`, since binding the last given field returns a `Distribution` directly; the empty-given case is `DistributionSpec`'s.

A `ConditionalDistribution` carries a `given_spec`, which is the `InputSpec` of independently bindable slots it conditions on (II.2), and an `event_spec`, which is the output declaration of one produced draw `T` and is read as for a `Distribution` (III.7); both are views on its stored `ConditionalDistributionSpec`. Unlike a function's domain and codomain, a kernel's given and event are distinct *roles*, the value conditioned on and the law produced, so their given-slot and produced-component names stay disjoint even when the two spaces coincide. A Markov kernel with `S = T` uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. Symbolic dimensions are scoped over the two sides jointly, so a name shared between given and event fields is one dimension, bound by `with_dims` or, in the fused conditional paths, from the given value at call time. `with_path_names` renames or moves names across both sides, returning the same kernel: the event side behaves exactly as a `Distribution`'s, and on the given side a path-valued target may split or group slots, since a kernel carries no signature to fix its top level. A `Function`'s input slots are fixed by its signature instead (III.3), so restructuring across its top level is not a rename but a new signature, obtained by wrapping the callable in one that takes the parameters wanted.

Users never call a method on the `ConditionalDistribution`. Instead, they use the existing operations. `condition_on(K, s)` binds the given fields and evaluates the kernel to a `Distribution` with no inference. `sample(K, given=s)`, `log_prob(K, y, given=s)`, and `mean(K, given=s)` are the **fused conditional paths**, with the invariant `op(K, given=s) == op(condition_on(K, s))`: the same law for exact realizations, and equal in law for their random draws. Equality draw for draw needs the same sampling realization, random-event identity, and key derivation as well, which sharing a workflow scope alone does not provide (V.8). An approximate path records its route and assumptions; it does not promise equality in law merely because it targets the same conditional. Binding a subset of the given slots *curries* to a smaller `ConditionalDistribution` (VI.6).

```python
class ConditionalDistribution[S, T](TrackedTerm):
    def __init__(self, name: str, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | TermSpec) -> None: ...
        # given before event, as in FunctionSpec
    @property
    def spec(self) -> ConditionalDistributionSpec: ...
    @property
    def given_spec(self) -> InputSpec: ...               # view on spec
    @property
    def event_spec(self) -> OutputSpec: ...              # view on spec: the event declaration
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions on both sides (II.1)
    def _condition_on(self, given: S, /, **kwargs) -> Distribution[T] | ConditionalDistribution: ...
    # the required primitive: the law K(given, ·), or a curried kernel for a partial given

@runtime_checkable
class SupportsConditionalSampling[S, T](Protocol):
    def _conditional_sample(self, given: S, key: Key, sample_shape: tuple[int, ...] = ()) -> T: ...
@runtime_checkable
class SupportsConditionalLogProb[S, T](Protocol):
    def _conditional_log_prob(self, given: S, value: T) -> Array: ...
@runtime_checkable
class SupportsConditionalMean[S, T](Protocol):
    def _conditional_mean(self, given: S) -> T: ...
# … and likewise SupportsConditionalVariance (_conditional_variance(given) -> T),
#   SupportsConditionalCovariance (_conditional_cov(given) -> LinOp),
#   SupportsConditionalExpectation (_conditional_expectation(given, f, …) -> Array),
#   SupportsConditionalMarginals (_conditional_marginal(given, path) -> Distribution).
```

The conditional vocabulary is closed by one rule: every unconditional capability has a conditional counterpart whose method prepends the given to the unconditional signature. The two vocabularies stay mirrored by construction, and a capability added on the unconditional side names its conditional twin automatically.

**The numeric special cases.** A `ConditionalDistribution` has *two* sides, and either can be numeric, so the single `Numeric` prefix becomes positional: `Numeric` before `Conditional` marks the **given** side numeric, `Numeric` before `Distribution` marks the **event** side numeric, and `FullyNumeric*` marks both. Each is a marker only and adds no operations of its own, as `NumericDistribution` does not.

```python
class ConditionalNumericDistribution(ConditionalDistribution): ...   # event numeric: every K(s, ·) is a NumericDistribution
class NumericConditionalDistribution(ConditionalDistribution): ...   # given numeric: the conditioning value is a numeric vector
class FullyNumericConditionalDistribution(
        NumericConditionalDistribution, ConditionalNumericDistribution): ...   # both sides numeric
```

**The conditional distribution term specification.** `ConditionalDistributionSpec` is the conditional-distribution kind's term spec. As a leaf, it types a field holding a matching `ConditionalDistribution`. Its event side is a declaration, exactly as for `DistributionSpec`; the given side is always an `InputSpec`:

```python
class ConditionalDistributionSpec(TermSpec):  # a ConditionalDistribution; is_valid accepts a match
    given_spec: InputSpec          # the named slots it conditions on, non-empty
    event_spec: OutputSpec         # the output declaration, as for DistributionSpec
```

### Rationale

Applying a `ConditionalDistribution` to a conditioning value returns a `Distribution`, which ensures `D4 – Closed system of objects under operations` is satisfied. A `ConditionalDistribution`'s capabilities are the `Distribution` capabilities shifted by one conditioning argument (`D3 – Capability-based operations`), so a single operation vocabulary applies to conditional distributions too, under the rule that *`Distribution` and `ConditionalDistribution` behave as similarly as possible*. The capabilities use distinct `_conditional_*` method names because a `@runtime_checkable` check matches on method name alone, so reusing `_sample` / `_log_prob` would corrupt the unconditional capability checks. `_condition_on` is the exception: fixing given fields means the same thing on both types, so a `ConditionalDistribution` satisfying `SupportsConditioning` is intended rather than a collision, and the names stay distinct where the meanings differ.

## III.10 — `DistributionBatch` and `ConditionalDistributionBatch`

### Contract

A `DistributionBatch` is a `Batch` of `Distribution`s: `N` separate distributions sharing one event declaration, indexed along a batch axis. A `ConditionalDistributionBatch` is the same construction over `ConditionalDistribution`s: `N` separate conditional distributions sharing one `given_spec` and one event declaration. The shared event declaration is the elements' `event_spec`, so a batch of random measures declares term-valued draws exactly as its elements do. They are the native batch forms of `DistributionSpec`- and `ConditionalDistributionSpec`-valued draws.

```python
class DistributionBatch(Batch[Distribution]):
    @property
    def event_spec(self) -> OutputSpec: ...   # the shared declaration, a view on spec

class ConditionalDistributionBatch(Batch[ConditionalDistribution]):
    @property
    def given_spec(self) -> InputSpec: ...               # the shared declarations, views on spec
    @property
    def event_spec(self) -> OutputSpec: ...
```

### Rationale

This is `D1 – Mathematical fidelity` on the distribution layer: a `DistributionBatch` of `N` laws is a *collection of separate measures*, kept firmly distinct from one *joint* law over a product space, exactly as a `RecordBatch` of `N` draws is distinct from one `Record` of `N` fields. It is the natural result of a vectorized operation that yields many distributions: sweeping a parameter batch through a `ConditionalDistribution` produces a `DistributionBatch` of conditioned laws.
