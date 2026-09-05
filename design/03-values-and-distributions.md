# Part III — Values and Distributions

Part III introduces the value and distribution objects a user constructs and operates on. Each is built on the shared abstractions of Part II and introduced in dependency order, and each section states only what its kind adds — everything else is Part II's contract. The final two sections cover the registries that act across these objects: cross-type conversion and constraint reparameterization.

| §      | Category                       | Contents                                                                                              | Role                                                                                                            |
| ------ | --------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| III.1  | Values                      | `NumericArray` / `NumericArrayBatch`                                                                  | The numeric-array kind: its spec, tracked class, and batch form, with tracked arithmetic.                       |
| III.2  | Values                      | `Opaque` / `OpaqueBatch`                                                                              | The fallback kind: identity for values the library cannot introspect.                                           |
| III.3  | Functions                      | `Function`, `FunctionBatch`                                                | The function kind's base — declared sides, identity, plain evaluation — and its batch form. |
| III.4  | Functions                      | `LinOp`                                                                                               | A lazy structured linear map — the linear `Function` subtype (III.3) — typed by numeric schemas, and the representation of covariances.                          |
| III.5  | Structured Values                      | `RecordSpec` / `Record` / `NumericRecord`                                                             | The record kind: its spec — the schema of one structured value — and the `NamedTree` of values bound to it.  |
| III.6  | Structured Values                      | `RecordBatch` / `NumericRecordBatch`                                                                  | A batch of records — what `sample` returns for many draws of a record-valued law.                               |
| III.7  | Distributions               | `Distribution`                                                                                        | A probability measure over one value type that carries an event declaration (`event_spec`) for its draws.                           |
| III.8  | Distributions               | Distribution capabilities                                                                             | The `Supports*` protocols — sampling, density, moments, conditioning — a distribution implements.               |
| III.9  | Distributions   | `ConditionalDistribution`                                                                             | A probability kernel: a family of distributions indexed by a conditioning value, and a sibling of `Distribution`.   |
| III.10  |Distributions | `DistributionBatch` / `ConditionalDistributionBatch`                                                  | A batch of distributions (or conditional distributions): `N` separate laws, distinct from one joint distribution.                 |
| III.11  | Distributions | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from named sub-distributions, with the factor and field access interfaces.                       |
| III.12  | Distributions | the `*` operator                                                                                      | Builds a joint from parts, with the result kind derived from the operands.                                        |
| III.13 | Distributions | `Distribution` hierarchy                                                                              | The classification of kinds — atomic, structured, and joint — assembled once composition exists.                        |
| III.14 | Registries | cross-type conversion (`converter_registry`) | Moving a distribution between representations, at a recorded fidelity. |
| III.15 | Registries | constraint reparameterization (`bijector_for`, `SupportsInverse`, `SupportsLogDetJacobian`) | Mapping a `Constraint` to a bijector for unconstrained inference, with the invertibility and Jacobian claims. |

## III.1 — `NumericArray`

### Contract

`NumericArray` is one array value with identity: a `TrackedTerm`, holding a single array whose `shape` is the **event** shape, with no batch axes; its `raw()` is that array. `NumericArraySpec` is its kind's term spec:

```python
class NumericArraySpec(NumericSpec):  # the numeric-array kind's spec, a NumericSpec (II.3)
    shape: tuple[int | str, ...]   # a str names a symbolic dimension (II.1)
    dtype: DType
    support: Constraint            # the support (II.3)
```

It carries the full set of array operators — arithmetic, comparison, and the coordinate protocols — and its arithmetic returns tracked terms under a deterministically derived, evaluation-order name, with identity attached as for any operation (II.4).

`NumericArrayBatch` is the kind's batch form: a `Batch` whose `element_spec` is the `NumericArraySpec` and whose storage is one array with the batch axes leading — the same split `RecordBatch` uses, with one column instead of many. An array with leading axes is just an array; the batch form is what carries the level names, the shared spec, and provenance.

### Rationale

The full set of array operators is safe here and only here: with no fields, an expression on one array has exactly one meaning (`D1 – Mathematical fidelity`).

## III.2 — `Opaque`

### Contract

`Opaque` adds identity and nothing else: no attribute forwarding, no `__call__`, and `raw` as the one explicit accessor for the wrapped value. `OpaqueSpec` is the fallback spec, admitting any non-mapping value:

```python
class OpaqueSpec(TermSpec):        # the fallback spec; is_valid accepts any non-mapping value
    meta: Hashable
```

`OpaqueBatch` is its batch form. It **stores** its elements rather than materializing them. Its `raw()` is an object array of the stored raw values.

### Rationale

The kind exists so that closure under operations holds for every return value (`D4 – Closed system of objects under operations`), and it adds identity and nothing else because a richer interface would promise structure the value does not declare (`D1 – Mathematical fidelity`).

## III.3 — `Function`

### Contract

The function kind's base type is `Function`: a tracked term wrapping exactly one Python callable, its representation, together with its `FunctionSpec`, whose sides it exposes as the `input_spec` and `output_spec` views, either side optional exactly as in the spec. A `Function` also carries a frozen `inspect.Signature`, authoritative for Python argument binding — parameter kinds, defaults, and variadic parameters, which a value schema cannot express — while the `input_spec` is authoritative for the value schema; construction validates their one-for-one correspondence, so binding an argument binds a slot by name. The wrapped callable and its state stay private: no attribute forwarding, and no backend object escapes through an operation.

A `Function` is invoked two ways. `apply` evaluates the wrapped callable at a point: given values that conform to `input_spec`, it returns one conforming to `output_spec`, with no tracking or lifting — the raw map that operations such as change of variables build on. `__call__` runs the **call path**, the base's one extension point: the base fills it with plain evaluation, and the engine layer (Part IV) replaces it once, at import. The base also carries its **controls** (IV.4), set at construction and revised functionally by `with_options`; it gives them no meaning, and the engine reads them at call time.

A `Function` is authored with the `@function` decorator or produced by an operation; both use the same call path. Three capability protocols accompany the base: `SupportsDifferentiation`, whose contract is given with the engine's differentiability claims in IV.1, and `SupportsInverse` and `SupportsLogDetJacobian`, whose contracts are given with constraint reparameterization in III.15. All are claims declared at construction and checked by protocol membership, except that a claim with an instance guard is read through its predicate — `is_differentiable`, `is_invertible`. The base is the tracked *wrapper*, not a restriction on what may be wrapped: `FunctionSpec`, the function kind's term spec, admits any callable, a `Function` being one such rather than the required type, and a `FunctionBatch` holds a collection of them. Its two sides are the declarations of II.2, either side optional — a bare `FunctionSpec()` describes any callable — and validity is callability alone: the sides document the schema, enforced at the call boundary rather than by `is_valid`.

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
                 differentiable: NumericSpec = ...) -> None: ...
                 # optional differentiability claim; its contract is the engine's (IV.1)
    @property
    def spec(self) -> FunctionSpec: ...
    @property
    def input_spec(self) -> InputSpec | None: ...                   # view on spec
    @property
    def output_spec(self) -> OutputSpec | None: ...                 # view on spec
    @property
    def options(self) -> Mapping[str, Any]: ...          # the controls; opaque to the base
    def with_options(self, **controls) -> Self: ...      # functional update
    def apply(self, *args, **kwargs) -> Any: ...
    # evaluate the wrapped callable at a point (input_spec -> output_spec),
    # with no tracking or lifting; the raw map operations build on
    def __call__(self, *args, **kwargs) -> Any: ...
    # run the call path: plain evaluation on the base, the Part IV engine after import

def install_call_engine(engine: Callable[..., Any]) -> None: ...
    # replaces the call path, once, at import time; until then calls evaluate plainly.
    # The engine reads the controls the Function carries and must agree with
    # plain evaluation on concrete values.
```

 `FunctionBatch` is the function kind's batch form, storing its elements exactly as `OpaqueBatch` does (III.2); an element is a `Function` whatever callable the slot holds, and `raw()` is an object array of the stored callables.

### Rationale

Defining the base in the value layer keeps the layering strict: the representation is fixed here, the call engine arrives by upward registration (`D2 – Generality first`), and `LinOp` and the specs reference `Function` downward — the split the package structure realizes as `values/_function_base.py` and `functions/`.

## III.4 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces: the linear subtype of `Function` (III.3), so it applies, composes, and evaluates like any map, with the operator algebra and the structured queries below as what linearity adds. Its action is the map the base carries — `apply` evaluates the operator at a `Numeric` conforming to its input schema and returns the matching form, with the operator's parameters as the private state behind it — and `matvec` / `matmat` / `rmatvec` / `rmatmat` are its linear-algebra names, `matmat` the operator's registered batched rule. Its domain and codomain schemas are the `NumericSpec`s (II.3) of the single slot each inherited side declares, with no operator-specific accessor beside them, so it maps whatever `Numeric` its sides declare — a bare array under a `NumericArraySpec` side, a named tree of them under the numeric record spec — and an operator over a scalar law's draws needs no single-field placeholder. The two sides coincide exactly for an endomorphism such as a covariance or Hessian, which the operator algebra reads as the fact that operands compose or act on the same space.

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
    # syntactic sugar for apply: A x, with a Numeric flattened through
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

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators (`ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view) that defer to their parts. The algebra checks and propagates the schemas: `A @ B` requires `B`'s output schema to equal `A`'s input schema and declares `B`'s input schema and `A`'s output schema as its own sides, `A + B` requires both pairs to match, and `A.T` swaps them. Composite operators are tracked terms like any other, with names derived from their operands.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant. Each also fixes the kind's `raw()` (II.4) as its stored parameterization, detached — the matrix for `DenseLinOp`, the diagonal for `DiagonalLinOp` — and a composite's is its operand tuple, laziness being the representation.

**The batch form.** `LinOpBatch` is the element batch over operators, a thin `Batch[LinOp]` whose elements share both schemas. It is what a batched `cov` returns. Application is elementwise: a single operator maps over a batch's elements, and a `LinOpBatch` zips with a broadcast-compatible batch of numeric values, element by element in both cases. The queries lift the same way, elementwise to batched results.

### Rationale

Operations mint linear operators, covariances above all, so the kind exists to keep those results first-class (`D4 – Closed system of objects under operations`). The structured subclasses exploit their form automatically behind one interface (`C3 – Computational detail hidden by default, available on demand`), the algebra returns lazy views rather than materialized matrices (`D6 – Single source of truth`), and typing both sides with numeric schemas makes closure concrete: the operator `cov` returns accepts the very draws its distribution produces (`D5 – Explicit, carried structure`).

### Open points

- *Structure-exploiting solves.* Exploiting structure in both operands of `A⁻¹B`, possibly through a dedicated `SolveLinOp`, is open.
- *Flag semantics.* Whether flags only describe structure or also steer which implementation a query selects is open.
- *Batched matrix action.* `matmat` against a batched operand, where a batch axis would meet the operator's matrix axis, and any richer `LinOpBatch` alignment are deferred until a concrete consumer exists.

## III.5 — `RecordSpec`, `Record`, and `NumericRecord`

### Contract

A `RecordSpec` is a `NamedTree` whose leaves are term specifications: the record kind's spec and the **schema** of one structured value — the structure of one event, such as a draw or a stored datum. One class serves both readings because they denote the same space. Nesting is just nesting: a record-shaped position inside a schema is a subtree, not a second spec class.

When every leaf is a `NumericSpec`, the schema is fully numeric and construction auto-promotes it to a `NumericRecordSpec` — a stored numeric-record term's leaf counts, since its spec is one, which is what lets flattening span nested numeric structure. The promotion is re-derived whenever a transform constructs a new schema, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it: the numeric axis is an invariant of the current leaves, not of the object's history. Beyond the inherited `NamedTree` interface (with `L = TermSpec`), `RecordSpec` adds construction shorthand, lossy inference from a value, and the numeric projection:

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

`infer_from` types a term-valued field at its own kind — a `Distribution`-valued field infers a `DistributionSpec`, a callable a `FunctionSpec` — and nested structure as nested structure, so inference never mistypes a term as the raw value it happens to resemble.

`NumericRecordSpec` further provides a flat (vectorized) layout of the leaves:

```python
class NumericRecordSpec(NamedTree[NumericSpec], NumericSpec, RecordSpec):
    # a RecordSpec whose leaf type narrows to NumericSpec, itself a NumericSpec:
    # vector_size sums over the leaves, and the flat layout is the tree's
    # canonical order over the leaves' own layouts
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
```

Within one schema a symbolic name refers to one dimension: fields `X: ("obs", "features")` and `coefficients: ("features",)` share the dimension `features`, an equality no pair of concrete integers can express (II.1), so validation runs one unification over every occurrence of a name across all fields — data of shapes `(100, 5)` and `(5,)` bind both dimensions consistently, while `(100, 5)` and `(7,)` raise. A leaf's `is_valid` checks its own rank and dtype; sizes belong to the one pass, since only it sees every occurrence of a name. A nested spec's schema lies inside the scope, so a name declared within a `DistributionSpec` is the same dimension as that name beside it, binding once whatever the declaration order.

Two rules govern record-shaped positions, symmetric in what arrives. Mapping data materializes into the record's own structure, under derived identity; a supplied tracked term is stored and keeps its identity (name, provenance, capabilities) as the field's **source** (access below). Both conform to the same spec, so structure and identity never disagree about what a field is.

A `Record` is a `NamedTree` that is a `TrackedTerm` with leaves that are *values*, its structure conforming to its authoritative `RecordSpec`. `NumericRecord` is the specialization in which every leaf is numeric — its value implements `Numeric` — and hence carries a `NumericRecordSpec`.

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

**Storage and access are separate contracts.** Storage retains the representation and the source: leaves are held in native form — a supplied `NumericArray`'s array is stored natively — and a supplied term's identity is held as a reference or a descriptor per the provenance mode (II.4). Access never returns the stored source itself: `record[path]` returns a view (II.4) of the field's kind. An interior path yields a sub-`Record` view. `record.raw(path)` returns the stored representation, and `record.raw()` the whole record's nested mapping of raw leaves — the record kind's raw host.

When every leaf is numeric, a `Record` is a `NumericRecord`. Leaves are stored in native form — a bare array, an `xarray` / `pandas` container, or any registered array backend — and convert to `jax.Array` only at the compute boundary (the pytree flatten that `grad` / `vmap` / `jit` traverse, and `to_vector`), each leaf at most once. A `Record` is promoted exactly as its schema is (above): when every leaf is numeric and no explicit non-numeric schema vetoes it, re-derived by every transform. Flat vectorization reads its layout (`leaf_shapes`, `vector_size`, canonical order) from the schema. Flattening is deliberately numeric-only, which is why `NamedTree` itself has no `flatten`.

```python
class NumericRecord(Record):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, spec: NumericRecordSpec, vec: Array) -> NumericRecord: ...
```

**Vector-space arithmetic.** `NumericRecord` implements the `Numeric` interface of II.3, and its arithmetic follows the two routes stated there. ProbPipe's own operators preserve structure and return tracked terms: the vector-space set — `+` and `-` between records sharing a schema, and scalar `*` and `/` — and `map(f)` for elementwise maps, `record.map(jnp.cos)` being the tracked form of `jnp.cos(record)`; array-shaped operations (broadcasting, positional indexing) stay with arrays, and no `__array_ufunc__` is defined, so NumPy and JAX functions never behave differently on the same object.

### Rationale

A `Record` is the *values* half of `C1 – Uniform interface to functions, distributions, and values`: a distribution's draw is a `Record` (or a `RecordBatch` for many), and a function over named values consumes one. One class serves the schema and the kind because they denote the same space: a second tag class would be a distinction without mathematical content, converted at every construction site (`D1 – Mathematical fidelity`, `D6 – Single source of truth`). Carrying the schema forward from its producer rather than re-inferring it downstream is `D5 – Explicit, carried structure` made concrete.

### Notes

- *Pytrees.* `Record` and `NumericRecord` are registered as JAX pytrees for advanced use, and the native `NamedTree` methods are the supported interface. JAX traversal follows the pytree registration, which does not always agree with ProbPipe on what is a leaf, so users applying raw JAX functions are responsible for the documented behavior. Record equality is structural value equality, which is weaker than treedef equality. The registration's children are the field arrays (a `NumericRecord`'s native leaves convert at this boundary) and its static aux data is the schema alone, identity being boundary-attached (II.4); native container types therefore never enter a trace either. A round-trip returns bare-array leaves and never promotes a `Record` to a `RecordBatch`.

- *Single-field presentation.* A `Record` is a container and presents as one, whatever its field count: no coercion, no forwarding, and no array operators beyond the vector-space operations above.
- *Construction validation.* Construction checks each leaf against its spec's `is_valid`, which validates structure only — for a `NumericArraySpec`, shape and dtype (dtype by `numpy.can_cast` same-kind: a widening promotion or a within-kind narrowing passes, a cross-kind conversion raises). A `NumericArraySpec`'s `support` is **not** part of `is_valid`: it is a data-dependent, element-wise check that reduces to a Python `bool` and so cannot run under `jax.jit` tracing, where construction also happens (pytree unflatten reconstructs a value inside the trace). `support` is therefore descriptive metadata, and invariant 2 (`is_valid`) covers shape and dtype. Leaf validation is skipped on the unflatten path, where a leaf's shape is transform-relative.

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

Storage is columnar: per-field columns in each field's batch form. A field column is therefore a direct tracked view, and an element `Record` is assembled on demand from the columns rather than stored a second time. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `element_spec`. A `NumericRecordBatch` additionally presents the coordinates view, `__jax_array__` being its batched `to_vector`, so the tracked batch, the raw columns, and the flat coordinates are three presentations of one store.

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

It claims only the batch axis and never the leaf-keyed `Mapping` contract, so a batch of `N` records can never read as one record of `N` fields — `D1 – Mathematical fidelity` at exactly the point where the two would otherwise be conflated.

## III.7 — `Distribution`

### Contract

A `Distribution[T]` is a single random law: a probability measure over values of type `T`, the implementer-side draw type fixed below. Its `DistributionSpec` carries `event_spec`, the output declaration of one draw (an `OutputSpec`, II.2), exposed as a view. That is the same declaration type a `Function` carries as its `output_spec`, and the two are named apart deliberately: a function's output is what one call *returns*, while a law's event is the space its draws inhabit, a standing property of the law rather than a per-call result. Construction normalizes the event declaration to an `OutputSpec` whose name defaults from the law's own constructor `name`, and the stored spec's class fixes the draw kind.

It declares the operations it supports as **capabilities** (III.8), so operational support is decoupled from the class. Its `raw()` is the law detached (II.4), so a field view's `raw()` is the detached marginal rather than a reference into its parent. A draw is a tracked term of the kind the event declaration names, never wrapped in another kind to make draws uniform.

**A slot is not a field.** Every law has exactly one **produced slot**, the name its `OutputSpec` declares: it is what composition matches on, what `include_inputs` labels, and what `with_path_names` renames by bare name. A **field** is a named part of one draw, a path in the event schema, and only a record-drawing law has any — its top-level fields are what `d[path]`, `marginal`, and the field views address. So a record-drawing law has both, while a term-drawing law has a slot and no fields: `Normal("x", 0, 1)` composes and renames under `x`, but offers no field interface, since projecting an atomic draw is the draw. The two never merge, and neither is derived from the other.

Fields are renamed or moved with `with_path_names` — a path-valued target restructures the event (II.6) — returning the same law under the canonical relabeling of its event space; on a factored joint the result is a relabeling view over the stored factors. `with_path_names` never changes the event's kind: an atomic event stays atomic, a record event stays a record, so a path-valued target on the output name raises. A distribution whose declaration is polymorphic is legal, binding exactly as II.1 fixes — by value, or explicitly through `with_dims`.

**The draw type `T`.** `T` is the implementer-side draw type, derived from the event spec's kind rather than declared independently: the spec is the source, and the bracket is typing documentation. Writing the tracked kind (`Distribution[NumericArray]`) or its raw host (`Distribution[Array]`) names the same array kind: either notation is read at the kind level. Per kind, the implementer type is the kind's raw host, except where the host cannot carry the structure the mathematics needs:

| event spec's kind | implementer draw type `T` |
|---|---|
| `NumericArraySpec` | `Array` |
| `OpaqueSpec` | the wrapped object |
| `FunctionSpec` | a callable |
| `RecordSpec` | `Record` — the flat mapping loses schema and layout |
| `DistributionSpec` | `Distribution` — a draw's own raw form |
| `ConditionalDistributionSpec` | `ConditionalDistribution` |

A `NumericDistribution` is a `Distribution` whose event spec is a `NumericSpec` (II.3), so its draws implement `Numeric` and the flat-vector interface applies — a scalar `Normal`'s `NumericArraySpec` event qualifies exactly as a record event does.

```python
class Distribution[T](TrackedTerm):
    def __init__(self, name: str, event_spec: OutputSpec | TermSpec | Mapping) -> None: ...
        # the event declaration, normalized to an OutputSpec (name defaulted from
        # the law's own name)

    @property
    def spec(self) -> DistributionSpec: ...
    @property
    def event_spec(self) -> OutputSpec: ...     # view on spec: the event declaration
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename the produced slot by bare name, or rename and move event fields;
    # keys and path-valued targets resolve as for NamedTree.with_path_names (II.6),
    # and the law is unchanged
    def with_dims(self, **sizes: int) -> Self: ...
    # bind named symbolic dimensions (II.1); a conflict with an existing binding raises
    def __getitem__(self, path: str | tuple[str, ...]) -> FieldView: ...
    # the field view at a leaf or group path; raises on a term-drawing law, which has no fields

class NumericDistribution(Distribution): ...   # marker: the event spec is a NumericSpec
```

**Field views.** `d[path]` returns a `FieldView`: a `Distribution` over the field or field group at `path`, holding a reference to its parent rather than a detached law. Sibling views co-sample from one parent draw, so correlation between them is preserved. The capabilities a view offers are derived from its parent's, one by one (III.8).

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

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to functions, distributions, and values`. A field view is `B4 – No copying at boundaries` at a field, and deriving its capabilities from its parent's ensures a view advertises only what it can compute (`D3 – Capability-based operations`). The draw-type table is `B2 – Representations only inside` per kind: an implementer writes over `T` and never sees a tracked draw.

### Open points

- *Structuring an atomic event.* Demoting a term-drawing law's output name into a group (`with_path_names({"x": "group/x"})`) is mathematically well-defined — the canonical isomorphism with the one-field product — but changes the draw's kind, so the rule above excludes it. A producer that wants the structure declares a one-field `RecordSpec` instead; revisit only if a concrete consumer appears.

## III.8 — Distribution capabilities

### Contract

Each operation on a distribution is a **capability**: a distribution implements an underscore method (`_sample`, `_log_prob`, `_mean`, …) over `T` (III.7) for each operation it supports, and the matching operation (`sample`, `log_prob`, `mean`, …) calls it through a capability route (V.0).

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

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Because support is structural (tested by `isinstance(dist, SupportsX)`, not subclassing), a distribution gains an operation just by implementing its method. A transform that preserves the event exposes exactly the capabilities of whatever it wraps, and a field view offers those its parent's capabilities can derive, so advertised support matches actual support in both cases.

## III.9 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution[S, T]` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `Distribution` is the empty-given case, a kernel with nothing to condition on, so its marginal law exists and the unconditional operations apply; a kernel with a non-empty given has none. The two are distinct tracked types, neither inheriting from the other, and a `ConditionalDistribution` always carries a non-empty `given_spec` — binding its last given field returns a `Distribution` directly, so no kernel is ever in the empty-given state — as does its spec, the empty-given case being `DistributionSpec`'s.

A `ConditionalDistribution` carries a `given_spec`, the `InputSpec` of independently bindable slots it conditions on (II.2), and an `event_spec`, the output declaration of one produced draw `T`, read exactly as for a `Distribution` (III.7); both are views on its stored `ConditionalDistributionSpec`. Unlike a function's domain and codomain, a kernel's given and event are distinct *roles* — the value conditioned on versus the law produced — so their field names stay disjoint even when the two spaces coincide: a Markov kernel (where `S = T`) uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. Symbolic dimensions are scoped over the two sides jointly, so a name shared between given and event fields is one dimension, bound by `with_dims` or, in the fused conditional paths, from the given value at call time. `with_path_names` renames or moves names across both sides, returning the same kernel: the event side behaves exactly as a `Distribution`'s, and on the given side a path-valued target may split or group slots, since a kernel carries no signature to fix its top level. A `Function`'s input slots are fixed by its signature instead (III.3), so restructuring across its top level is not a rename but a new signature, reached by wrapping the callable in one that takes the parameters wanted.

Users never call a method on the `ConditionalDistribution`. Instead, they use the existing ops: `condition_on(K, s)` binds the given fields, evaluating it to a `Distribution` exactly and with no inference, and `sample(K, given=s)` / `log_prob(K, y, given=s)` / `mean(K, given=s)` are the fused conditional paths, with the invariant `op(K, given=s) == op(condition_on(K, s))`, bitwise under a shared PRNG key in the exact cases and in law when inference is involved. Binding a subset of the given slots *curries* to a smaller `ConditionalDistribution` (V.6).

```python
class ConditionalDistribution[S, T](TrackedTerm):
    def __init__(self, name: str, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | TermSpec | Mapping) -> None: ...
        # given before event, as in FunctionSpec
    @property
    def spec(self) -> ConditionalDistributionSpec: ...
    @property
    def given_spec(self) -> InputSpec: ...               # view on spec
    @property
    def event_spec(self) -> OutputSpec: ...              # view on spec: the event declaration
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

**The numeric special cases.** A `ConditionalDistribution` has *two* sides, and either can be numeric, so the single `Numeric` prefix becomes positional: `Numeric` before `Conditional` marks the **given** side numeric, `Numeric` before `Distribution` marks the **event** side numeric, and `FullyNumeric*` marks both. Each is a marker only, adding no operations of its own (mirroring `NumericDistribution`).

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

Applying a `ConditionalDistribution` to a conditioning value returns a `Distribution`, which ensures `D4 – Closed system of objects under operations` is satisfied. A `ConditionalDistribution`'s capabilities are the `Distribution` capabilities shifted by one conditioning argument (`D3 – Capability-based operations`), so a single operation vocabulary applies to conditional distributions too, under the rule that *`Distribution` and `ConditionalDistribution` behave as similarly as possible*. The capabilities use distinct `_conditional_*` method names because a `@runtime_checkable` check matches on method name alone, so reusing `_sample` / `_log_prob` would corrupt the unconditional capability checks. `_condition_on` is the deliberate exception: fixing given fields means the same thing on both types, so a `ConditionalDistribution` satisfying `SupportsConditioning` is intended rather than a collision, while the names stay distinct exactly where the meanings differ.

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

## III.11 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from named sub-distributions**: beyond being an ordinary distribution, it carries an explicit factorization into its parts, marked by the capability `SupportsFactors`, which `FactoredDistribution` and `FactoredConditionalDistribution` implement generically.

Both carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's given fields against the fields produced by earlier factors, rather than stored. The joint's event declaration is the disjoint union of the factors' produced slots (III.12) — a record-drawing factor contributes its top-level fields, internal structure preserved, and any other factor the single field its `OutputSpec` names — and factor names are unique across the list. Conditioning a `FactoredConditionalDistribution` on all of its given fields yields a `FactoredDistribution`. Sampling and the log-prob capabilities are the intersection of the factors'. The moment capabilities are decided at construction, present exactly when the joint's structure makes the moment derivable: an edge-free joint derives its moments componentwise, so it has a moment exactly when every factor does; jointly Gaussian factors are exact (VI.6); any other dependent joint carries no moment capability, since factor-wise conditional moments do not compose into a closed form — with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `E[y] = e^{1/2}` is not reachable from the factors' means.

```python
@runtime_checkable
class SupportsFactors(Protocol):
    @property
    def factors(self) -> tuple[Distribution | ConditionalDistribution, ...]: ...   # ordered; the graph is derived, not stored

class FactoredDistribution(Distribution, SupportsFactors): ...
class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors): ...

# numeric markers, by which of the (given, event) sides is numeric. Numeric is positional,
# as for the ConditionalDistribution markers: before Distribution marks the event numeric, before Conditional the given.
class FactoredNumericDistribution(FactoredDistribution): ...                                # unconditional joint, event numeric
class FactoredConditionalNumericDistribution(FactoredConditionalDistribution): ...          # conditional joint, event numeric
class FactoredNumericConditionalDistribution(FactoredConditionalDistribution): ...          # conditional joint, given numeric
class FactoredFullyNumericConditionalDistribution(
        FactoredNumericConditionalDistribution, FactoredConditionalNumericDistribution): ... # conditional joint, both numeric
```

**Field versus factor.** A **field** is a named part of a draw, that is, a path in the event schema. A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint exposes up to two clearly separated interfaces, never through the same operator.
- The **field interface** is available on every distribution: `d["intercept"]` is the field view of III.7, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, "coeffs")` returns a building-block factor, keyed by factor name, which is a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. There need be no factor for a given field, and no field for a given factor.

**Marginals of a joint.** Whether a marginal is exactly available depends on the factors' own marginal support and on where the target sits in the dependence graph, so the factored classes resolve `_marginal` per path rather than wholesale. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out for free. What remains is integrating the extra ancestor fields back out, which is exact in three cases: there are none (the target is ancestrally closed, as for a root factor or an edge-free group), the reduction lies within a single factor and delegates to that factor's own `SupportsMarginals`, or the affected factors admit closed-form integration, as when they are jointly Gaussian. On any other path `_marginal` raises.

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, instead of sitting in a parallel class tower. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two are genuinely different in the mathematics. Deciding moment presence at construction follows `D3 – Capability-based operations`: a capability is advertised exactly when the object can compute it.

### Notes

- *Group views.* The field interface also accepts an interior path, which names a group of fields rather than a single field. For example, when the event declaration nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns the marginal over the whole group. Like a single-field view, it is a view onto the parent joint, not a detached distribution.

## III.12 — Composition

### Contract

Composition builds a factored distribution from parts, written as an *expression*: a single binary operator `*` combines `Distribution`s and `ConditionalDistribution`s into one joint. The *kind* of the result is **derived** from the operands and never chosen by hand, and every result is itself a `Distribution` or a `ConditionalDistribution`. The base objects expose `*` as a thin `__mul__` that delegates to the operator.

**The `*` operator.** `A * B` composes two operands into a joint. It is **conditional-first**: the left operand may condition on the right, so `lik * prior` reads as the density p(y | β) · p(β), while the reverse (a consumer before its producer) is an error. Characterize each operand by its **produced slots** `F` — the top-level fields of a record-drawing operand, or the single name its `OutputSpec` declares otherwise — and its **unmet given slots** `G`, with `G = ∅` for a `Distribution` and `G` the given slots for a `ConditionalDistribution`. The composition is then fixed by the name sets alone:

```
bound  = G_A ∩ F_B            # dependency edges: left A conditions on a name that right B produces
unmet  = (G_A − F_B) ∪ G_B    # residual exogenous givens — met by no factor
require  F_A ∩ F_B = ∅         # each name is produced exactly once
     and G_B ∩ F_A = ∅         # B must not consume a name A produces — else reorder (producer on the right)
law:  p(F_A, F_B | unmet) = p_A(F_A | bound ∪ (G_A − F_B)) · p_B(F_B | G_B)      # reads left → right
```

The result has two mathematical degrees of freedom, *conditional?* (`unmet ≠ ∅`) and *dependent?* (`bound ≠ ∅`), but only the first is a **class** distinction:

| `unmet` | result |
|---|---|
| `∅` | `FactoredDistribution` — a joint `Distribution` |
| `≠ ∅` | `FactoredConditionalDistribution` — a joint `ConditionalDistribution`, its `given_spec` exactly `unmet` |

`*` returns the **most specific** class, recomputed from the *flattened* factor graph at each step. `A * B * C` builds one flat N-factor joint, with independent factors commuting and dependent ones kept in conditional-first order. Same-named unmet givens unify into one slot of the joint: their specs must unify, a disagreement raising at composition, and binding the slot feeds every factor that names it — two givens that are genuinely different quantities are renamed apart first, the discipline fields and levels already follow. Symbolic dimensions never unify by name across operands, since two factors may both call something `"obs"` and mean different dimensions. Each operand's dimensions instead enter the joint under a deterministic factor-qualified renaming, shared fields and shared unmet givens contribute the only identifications, and the joint stores the factors so refined. The renaming is canonical, so derived names and fingerprints stay deterministic.

**Naming the result.** A joint is *derived*, not created by the user, so `*` **auto-derives** its `name` deterministically from its factors. The factors are listed in **canonical order** (the conditional-first topological order of the flattened factor graph, with factors incomparable in the derived graph ordered lexicographically by the fields they produce), and their names are joined by `·`. So `lik * prior` is named `lik·prior`, and because neither association nor the ordering of independent factors changes the canonical list, `A * B * C`, `(A * B) * C`, and `A * (B * C)` produce the same joint distribution.

Re-composition reads `name_is_auto`. An auto-named operand is **flattened**: its factors enter the new joint directly, its old name is discarded, and a fresh name is derived from the full factor list. An operand whose name the user has set with `with_name` is **not** flattened. It enters as a single factor under that name, and that name appears as one token in the parent's derived name. So `(lik * prior).with_name("posterior")` both labels the joint and, in any later composition, keeps it as the single factor `posterior`.

```python
def __mul__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product (`bound = ∅`) is just an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. Deriving the name deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`), while `with_name` lets the user impose grouping where it carries intent. The conditional-first order is already a valid topological listing, so composition is associative and acyclicity is automatic, with no separate graph inference. Associativity rests on the `G_B ∩ F_A = ∅` requirement, under which the validity of `(A * B) * C` and `A * (B * C)` coincide. Composition is written as an expression so that a model is *built* rather than declared (`C2 – Functional interface over immutable objects`), and every result is a first-class joint that composes further (`D4 – Closed system of objects under operations`).

### Notes

- *Operator coexistence.* `*` also denotes scalar scaling on some objects, such as a random function or a linear operator. The two coexist by operand-type dispatch: `Distribution` and `ConditionalDistribution` operands compose, while scalar operands scale.

## III.13 — The `Distribution` hierarchy

### Contract

With the base `Distribution`, the `ConditionalDistribution`, and the factored distributions and their composition in place, the distribution *kinds* can be classified. Two **independent** questions classify any distribution:
1. **The type axis** — what does a draw look like, and is a factorization exposed? This fixes *which interfaces apply*.
2. **The family axis** — how is the law realized? This fixes *which capabilities* the object implements, and how.

The axes are orthogonal, so they combine freely: a `Normal` is *atomic* (type) and *parametric* (family); a posterior over `{μ, σ}` reconstructed from samples is *atomic-structured* and *empirical*; a joint's factors may themselves come from any family.

**The type axis — atomic vs. joint.** The structural classification:

| type | a draw is | factorization? | interface beyond the capabilities |
|---|---|---|---|
| **atomic, array-valued** | one `Array` (scalar or vector) | none | — (a single-field draw) |
| **atomic, structured** | a multi-field `Record` | none | the *field* interface — `d["x"]`, `marginal` |
| **joint** | a `Record` | `SupportsFactors` | the *field* interface **and** the *factor* interface — `factor` |

The line between the last two is **factorization, not field count**: a multi-field empirical distribution or an amortized posterior has fields but no factors, so it is *atomic-structured*, and only a distribution built from named sub-distributions is *joint*. Hence there is **no `RecordDistribution`**: draw structure is declared in `event_spec` and factor access by `SupportsFactors`.

**The family axis — how the law is realized.** Each family is an ordinary `Distribution` (a `NumericDistribution` when its event is numeric), differing only in which capabilities it implements and how — refinement by capability, not a parallel class tower. The families themselves — their events, the capabilities each implements, and how each arises — are Part VI's catalog.

**Conditional distributions and batches stratify identically.** A `ConditionalDistribution` repeats both axes (atomic / structured / the `FactoredConditionalDistribution` joint, crossed with parametric / amortized / empirical / …), and a `DistributionBatch` is `N` of any of these. The catalog is one classification, reused across the conditional and multiplicity layers.

### Rationale

The hierarchy embodies `D2 – Generality first`: one base refined by *optional capabilities* (`SupportsFactors`, the numeric marker, each `SupportsX`) rather than a rigid class tower, so a new family slots in by implementing the capabilities it supports rather than by widening the base. The atomic-versus-joint split is a `D1 – Mathematical fidelity` distinction, since a joint genuinely offers its factors as distributions while an atomic-structured law does not, and it is carried by a capability rather than by the draw's type. Because composition is closed (`D4 – Closed system of objects under operations`), a joint re-enters the catalog as an ordinary `Distribution` for the next operation.

## III.14 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current type to a requested target type, resolved by the **converter registry** whose methods are *converters*. The registry is an ordinary binary dispatch registry keyed on `(type(source), target)`, the target being a type already.

Conversion is also the entry route for raw distributions: a backend distribution supplied at a distribution-shaped position — an argument, a given, a record field — is converted on entry through the registry, exactly as a bare array is wrapped at an array-spec position.

A converter declares the source types it converts *from* and the target types it converts *to* in the binary method's two slots. A conversion is rarely unique, so each carries a **fidelity** on the shared scale of II.7: `exact` for an equivalent representation, `approximate` where something is lost — moment matching, which preserves only low-order moments, is the common case — and `sample` for a Monte Carlo stand-in. A caller may set `min_fidelity`, a feasibility floor `check` enforces.

```python
@dataclass(frozen=True)
class ConversionInfo(MethodInfo):   # the feasibility probe's result; fidelity comes from MethodInfo
    ...

class Converter(BinaryDispatchMethod):
    name: str
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (source, target) types
    def check(self, source, target_type: type, *,
              min_fidelity: Fidelity | None = None) -> ConversionInfo: ...
    def execute(self, source, target_type: type) -> Distribution: ...             # the conversion itself

class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    # keyed on (type(source), target): the second key is the target type itself
    def convert(self, source, target_type: type,
                method: str | None = None, min_fidelity: Fidelity | None = None) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity makes the approximation explicit, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while an `approximate` or `sample` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`). Realizing the registry as a subclass of the shared dispatch registry gives conversion registration, feasibility probing, prioritized selection, and cataloging without duplicating any of them (`D6 – Single source of truth`). Entry conversion is how `B1 – Either presentation in` is realized for distributions: the registry is the kind-directed wrap at a distribution-shaped position.

## III.15 — Constraint reparameterization

### Contract

Many inference algorithms operate on an unconstrained space ℝᵈ — gradient-based optimization and Hamiltonian Monte Carlo among them — so a constrained support must be reparameterized. The **constraint-to-bijector factory** maps a `Constraint` (the support a `NumericArraySpec` carries) to a *bijector*: a `Function` that takes `ℝⁿ` onto that support and claims the two capabilities below. `bijector_for(constraint)` returns the canonical one, and `register_bijector` plugs in a factory for a constraint type or a specific instance, instance registrations taking precedence.

Invertibility and the Jacobian determinant are two separate capabilities. A `Function` claims `SupportsInverse` by providing the inverse map, its own `apply` serving as the forward; `SupportsLogDetJacobian` provides the log-determinant of the Jacobian, which exists only for a differentiable map, and a map can be invertible without it. Change of variables requires exactly the pair, and both are typed over the `Numeric` interface (II.3). A `LinOp` claims them only when they apply — guarded per instance by squareness, with its inverse from the operator algebra and its `logdet` the log-Jacobian — and `is_invertible` reads the claim together with its guard, while singularity, which no construction-time check decides, is raised at call time as `LinAlgError`, exactly as for `solve`. A slot checks the claims it needs at construction and raises a capability error otherwise: the bijector of a transformed distribution requires both, the link of a GLM likelihood `is_invertible` alone.

```python
@runtime_checkable
class SupportsInverse(Protocol):            # an invertible map; the forward is the claiming Function's apply
    def _inverse(self, y: Numeric) -> Numeric: ...

@runtime_checkable
class SupportsLogDetJacobian(Protocol):     # a map with a tractable Jacobian determinant
    def _log_det_jacobian(self, x: Numeric) -> Array: ...

def is_invertible(f: Any) -> bool: ...      # the claim together with its instance guard

def bijector_for(constraint: Constraint) -> Function: ...   # the canonical map ℝⁿ → support(constraint)
def register_bijector(key: type[Constraint] | Constraint,
                      factory: Callable[[Constraint], Function]) -> None: ...
```

The factory keys on constraint instances and types rather than dispatching on argument types alone, so it is not a dispatch registry; it still satisfies `SupportsRegistryCataloging` and appears in the registry catalog alongside the dispatch registries.

### Rationale

A bijector for every constraint lets inference run in an unconstrained space while a model stays stated in its natural, constrained one, which is `C3 – Computational detail hidden by default, available on demand`: the reparameterization an algorithm needs is supplied for it rather than written into the model. Invertibility as a capability is `D3 – Capability-based operations`: an invertible map is an ordinary `Function` that additionally claims `SupportsInverse`, so it evaluates, composes, and pushes forward like any other, with *bijector* reserved for the mathematical statement. The Jacobian determinant is a separate claim for the same reason: a map can be invertible without a tractable determinant, so the two claims are separate and change of variables asks for exactly the pair. Keeping the factory open through `register_bijector` is `D2 – Generality first`: a new constrained support becomes inference-ready by registering its reparameterization, without touching the distributions that use it.

### Open points

- *Round-trip fidelity.* The forward map (bijector to support) and this inverse map (support to bijector) are not strict inverses for every constraint, so a reparameterized support can drift to a coarser one. Whether to unify the two is unsettled.
