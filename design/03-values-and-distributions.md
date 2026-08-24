# Part III — Values and Distributions

Part III introduces the value and distribution objects a user constructs and operates on, and the machinery specific to them. Each is built on the shared abstractions of Part II and introduced in dependency order. The final two sections cover the registries that act across these objects: cross-type conversion and constraint reparameterization.

| §      | Layer                       | Contents                                                                                              | Role                                                                                                            |
| ------ | --------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| III.1  | Values                      | `NumericArray` / `NumericArrayBatch`                                                                  | The numeric-array kind: its spec, tracked class, and batch form, with tracked arithmetic.                       |
| III.2  | Values                      | `Opaque` / `OpaqueBatch`                                                                              | The fallback kind: identity for values the library cannot introspect.                                           |
| III.3  | Values                      | `Function`, `FunctionBatch`                                                | The function kind's base — declared sides, identity, plain evaluation — and its batch form. |
| III.4  | Values                      | `RecordSpec` / `Record` / `NumericRecord`                                                             | The record kind: its spec — the schema of one structured value — and the `NamedTree` of values bound to it.  |
| III.5  | Values                      | `RecordBatch` / `NumericRecordBatch`                                                                  | A batch of records — what `sample` returns for many draws of a record-valued law.                               |
| III.6  | Values                      | `LinOp`                                                                                               | A lazy structured linear map — the linear `Function` subtype (III.3) — typed by numeric schemas, and the carrier of covariances.                          |
| III.7  | Distributions               | `Distribution`                                                                                        | A probability measure over one value type that carries an event declaration (`event_spec`) for its draws.                           |
| III.8  | Distributions               | Distribution capabilities                                                                             | The `Supports*` protocols — sampling, density, moments, conditioning — a distribution implements.               |
| III.9  | Conditional Distributions   | `ConditionalDistribution`                                                                             | A probability kernel: a family of distributions indexed by a conditioning value, and a sibling of `Distribution`.   |
| III.10  | (Conditional) Distributions | `DistributionBatch` / `ConditionalDistributionBatch`                                                  | A batch of distributions (or conditional distributions): `N` separate laws, distinct from one joint distribution.                 |
| III.11  | (Conditional) Distributions | factored distributions (`SupportsFactors`, `FactoredDistribution`, `FactoredConditionalDistribution`) | A distribution built from named sub-distributions, with the factor and field access interfaces.                       |
| III.12  | (Conditional) Distributions | the `*` operator                                                                                      | Builds a joint from parts, with the result kind derived from the operands.                                        |
| III.13 | (Conditional) Distributions | `Distribution` hierarchy                                                                              | The catalog of kinds — basic, structured, and joint — assembled once composition exists.                        |
| III.14 | Registries | cross-type conversion (`converter_registry`) | Moving a distribution between representations, at a recorded fidelity. |
| III.15 | Registries | constraint reparameterization (`bijector_for`, `SupportsInverse`, `SupportsLogDetJacobian`) | Mapping a `Constraint` to a bijector for unconstrained inference, with the invertibility and Jacobian claims. |

## III.1 — `NumericArray`

### Contract

`NumericArray` is one array value with identity: `TrackedTerm` and `Annotated`, holding a single array whose `shape` is the **event** shape, with no batch axes. `NumericArraySpec` is its kind's term spec:

```python
class NumericArraySpec(TermSpec):  # the numeric-array kind's spec
    shape: tuple[int | str, ...]   # a str names a symbolic dimension (II.2)
    dtype: DType
    support: Constraint            # descriptive membership metadata (II.3)
```

It names the numeric-array kind alongside `NumericRecord`, whose leaves are values of that kind; `Array` stays the type alias for a bare backend array, so nothing is renamed to make room. It carries the full array surface — arithmetic, comparison, and the conversion hooks — because with no fields and no field count `arr + 1` has exactly one meaning. **Arithmetic returns tracked terms.** `arr + 1` is a `NumericArray` under a deterministically derived, evaluation-order name marked `name_is_auto`, with identity attached as for any operation (II.4). `Record` carries the vector-space subset only and otherwise stays a container.

`NumericArrayBatch` is the kind's batch form: a `Batch` whose `element_spec` is the `NumericArraySpec` and whose storage is one array with the batch axes leading — the same split `RecordBatch` uses, with one column instead of many. Native storage is not identity: an array with leading axes is just an array, and the batch form is what carries the level names, the shared spec, and provenance, so a `draw` level has somewhere to live for an array-valued law. `NumericArrayBatch[i]` materializes a `NumericArray` under the derived name with its own lineage (II.6), and `raw()` is the backing array itself.

### Rationale

The full operator surface is safe here and only here: with no fields, an expression on one array has exactly one meaning (`D1 – Mathematical fidelity`). Everything else — the stored spec, tracked results, boundary-attached identity, the batch form — is Part II's contract instantiated at the array kind.

## III.2 — `Opaque`

### Contract

`Opaque` adds identity and nothing else: no attribute forwarding, no `__call__`, and `raw` as the one explicit accessor for the wrapped value. `OpaqueSpec` is the fallback spec, admitting any non-mapping value:

```python
class OpaqueSpec(TermSpec):        # the fallback spec; is_valid accepts any non-mapping value
    meta: Hashable
```

`OpaqueBatch` is its batch form. It **stores** its elements rather than materializing them, and indexing is tracked all the same: a stored tracked term is handed back untouched, under its own name and lineage, while a stored bare value materializes as the tracked term of its kind under the derived name (II.6), and a *sub-batch* takes the derived name as any view does. Its `raw()` is an object array of the stored raw values. And every element is checked against the shared `element_spec` at construction, reporting the position that failed, since the batch asserts that spec of all of them — one element that fails it would make the batch's own spec a false statement.

```python
class OpaqueBatch(Batch[Any]):
    @property
    def spec(self) -> BatchSpec: ...      # the batch's own type
    @property
    def element_spec(self) -> OpaqueSpec: ...     # view on spec
```

### Rationale

The kind exists so that closure under operations holds for every return value (`D4 – Closed system of objects under operations`), and it adds identity and nothing else because a richer interface would promise structure the value does not declare (`D1 – Mathematical fidelity`).

## III.3 — `Function`

### Contract

The function kind's base type is `Function`: a tracked term wrapping exactly one Python callable. It carries the representation of a map: a `name`, `provenance`, and its `FunctionSpec` — the single stored source of its type, whose sides it exposes as the `input_spec` and `output_spec` views, either side optional exactly as in the spec. Construction fixes all of it. A `Function` carries a frozen `inspect.Signature`, authoritative for Python argument binding — parameter kinds, defaults, and variadic parameters, which a value schema cannot express — while the `input_spec` is authoritative for the value schema. Construction validates their total correspondence, the signature's parameters matching the slots one for one, so binding an argument binds a slot by name. The wrapped callable and its state stay private, reached only through the `Function`'s methods, so a stateful map exposes no backend object.

A `Function` is invoked two ways. `apply` evaluates the wrapped callable at a point: given values that conform to `input_spec`, it returns one conforming to `output_spec`, with no tracking or lifting. `__call__` runs the **call path**, the base's one extension point: the base fills it with plain evaluation, and the engine layer (Part IV) replaces it once, at import, adding lifting, tracking, and provenance to every `Function`. So `apply` is the raw map that operations such as change-of-variables build on, and `__call__` is the tracked call a user makes. The base also carries its **controls** — the execution defaults (sample count, dispatch, orchestration) — set at construction and revised functionally by `with_options`; the base gives them no meaning, and the engine reads them at call time.

A `Function` is authored with the `@function` decorator or produced by an operation; both use the same call path, and a produced `Function` carries its provenance like any other tracked term. Three capability protocols accompany the base: `SupportsDifferentiation`, whose contract is given with the engine's differentiability claims in IV.1, and `SupportsInverse` and `SupportsLogDetJacobian`, whose contracts are given with constraint reparameterization in III.15 — the protocols themselves sit beside the base in the layout. All are claims a `Function` carries, declared at construction and checked by protocol membership, except that a claim with an instance guard is read through its predicate: `SupportsDifferentiation` declares *which* values differentiate, read through `is_differentiable`, and invertibility is read through `is_invertible`. The `Function` base is the tracked *wrapper*, not a restriction on what may be wrapped: the value-layer specs stay **callable-generic**. A `FunctionSpec` admits any callable — a plain lambda, a NumPy function, a `Function` — the `Function` being one such, not the required type, and a `FunctionBatch` holds a collection of them. No operation branches on whether a callable arrived bare or wrapped. `FunctionSpec` is the function kind's term spec, its two sides the declarations of II.2, either side optional — a bare `FunctionSpec()` describes any callable. Validity is callability alone: the sides document the schema, enforced at the call boundary rather than by `is_valid`, and it is the spec's identity as a `FunctionSpec`, not `is_valid`, that tells the wrap boundary to wrap a raw callable result into a `Function`.

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
                 differentiable: NumericRecordSpec = ...) -> None: ...
                 # optional differentiability claim; its contract is the engine's (IV.1)
    @property
    def spec(self) -> FunctionSpec: ...                  # the single stored source of the type
    @property
    def input_spec(self) -> InputSpec | None: ...                   # view on spec
    @property
    def output_spec(self) -> OutputSpec | None: ...                 # view on spec
    @property
    def options(self) -> Mapping[str, Any]: ...          # the controls; opaque to the base
    def with_options(self, **controls) -> Self: ...      # functional update (C2)
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

 `FunctionBatch` is the function kind's batch form, storing its elements exactly as `OpaqueBatch` does (III.2): indexing hands a stored `Function` back untouched under its own identity, materializes a stored bare callable as a `Function` under the derived name (II.6), and `raw()` is an object array of the stored callables, with every element checked against the shared `element_spec` at construction.

```python
class FunctionBatch(Batch[Callable]):
    @property
    def spec(self) -> BatchSpec: ...      # the batch's own type
    @property
    def element_spec(self) -> FunctionSpec: ...   # view on spec: what every element satisfies
```

### Rationale

Defining the base in the value layer keeps the layering strict: the representation is fixed here, the call engine arrives by upward registration (`D2 – Generality first`), and `LinOp` and the specs reference `Function` downward — the split the package structure realizes as `values/_function_base.py` and `functions/`.

## III.4 — `RecordSpec`, `Record`, and `NumericRecord`

### Contract

A `RecordSpec` is a `NamedTree` whose leaves are term specifications: the record kind's spec and the **schema** of one structured value — the shape of one event, such as a draw or a stored datum. One class serves both readings because they denote the same space. Nesting is just nesting: a record-shaped position inside a schema is a subtree, and whether the value arriving there materializes as structure or is stored as a term is decided by the value, per the symmetric rule below, not by a second spec class.

When every leaf is a `NumericArraySpec`, the schema is fully numeric and construction auto-promotes it to a `NumericRecordSpec`. The promotion is re-derived whenever a transform constructs a new schema, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it: the numeric axis is an invariant of the current leaves, not of the object's history. Beyond the inherited `NamedTree` interface (with `L = TermSpec`), `RecordSpec` adds construction sugar, lossy inference from a value, and the numeric projection:

```python
class RecordSpec(NamedTree[TermSpec], TermSpec):
    def __init__(self, field_specs: Mapping[str, Any] | None = None, /,
                 **fields: TermSpec | Mapping | tuple[int, ...] | None) -> None: ...
    # sugar: a bare shape tuple means NumericArraySpec(shape) and None means OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> RecordSpec: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    def numeric_subset(self) -> NumericRecordSpec: ...   # remove non-NumericArraySpec leaves
```

`infer_from` types a term-valued field at its own kind — a `Distribution`-valued field infers a `DistributionSpec`, a callable a `FunctionSpec` — and nested structure as nested structure, so inference never mistypes a term as the raw value it happens to resemble.

`NumericRecordSpec` further provides a flat (vectorized) layout of the leaves:

```python
class NumericRecordSpec(RecordSpec):
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
    @property
    def vector_size(self) -> int: ...                          # total flat dimension; defined only when concrete
```

Within one schema a symbolic name refers to one dimension: fields `X: ("obs", "features")` and `coefficients: ("features",)` share the dimension `features`, an equality no pair of concrete integers can express (II.2). Checking a value against a polymorphic schema therefore cannot be done leaf by leaf, because a symbolic name constrains several leaves at once: validation runs a single pass over all fields, resolving each name to one size — a name is bound the first time it is seen, every later occurrence must agree, a disagreement raises, and bound names never rebind. The work splits accordingly: a leaf's `is_valid` checks its own rank and dtype, and nothing else — a `NumericArraySpec`'s `support` is descriptive metadata, unchecked — while sizes belong to the one pass, since only it sees every occurrence of a name. A nested spec's schema lies inside the scope, so a name declared within a `DistributionSpec` is the same dimension as that name beside it, binding once whatever the declaration order.

Two rules govern record-shaped positions, symmetric in what arrives. Mapping data materializes into the record's own structure, under derived identity; a supplied tracked term is stored and keeps its identity (name, provenance, capabilities). Both conform to the same spec, so structure and identity never disagree about what a field is.

A `Record` is a  `NamedTree` that is `TrackedTerm` and `Annotated` with leaves that are *values*. Its structure conforms to its authoritative `RecordSpec`. Records provide a uniform representation for all types of values, including the data a function consumes and the draws a distribution produces. `NumericRecord` is the specialization in which every leaf is a numeric array and hence carries a `NumericRecordSpec`.

Since the structure of `Record` matches that of its schema, the following invariants must hold:
1. *matching keys:* `record.keys() == record.spec.keys()`.
2. *valid values:* for any valid key `p`, the value stored at `p` satisfies `record.spec[p].is_valid`.
3. *matching sub-schemas:* for any valid non-key path `p`, `record.at_path(p).spec == record.spec.at_path(p)`.

Against a polymorphic schema, the invariants are checked by one joint unification across all fields: with `X: ("obs", "features")` and `coefficients: ("features",)`, data of shapes `(100, 5)` and `(5,)` bind both dimensions consistently, while `(100, 5)` and `(7,)` raise. Construction binds the schema, so a `Record` always carries the concrete, bound form and never an unbound dimension, and the data and its schema cannot disagree.

Two records are equal when they share a class, a `RecordSpec`, and field-by-field equal data. Because the schema is carried rather than re-inferred, an identity transform that threads it through compares equal to its input. A transform that instead rebuilds the schema by inference matches only when that inference recovers the original, for instance when the original schema was itself produced by `infer_from`.

```python
class Record(NamedTree[Any], TrackedTerm, Annotated):
    def __init__(self, name: str, fields: Mapping[str, Any] | None = None, /, *,
                 spec: RecordSpec | Mapping | None = None,
                 name_is_auto: bool = False,
                 **kw_fields: Any) -> None: ...
        # name is the required first argument (semantic identity)
        # name_is_auto marks an operation-derived name (II.4); user constructions leave it False
        # a nested sub-record's name is its field key; a mapping-valued field is a subtree, never a leaf.
        # Binds to the declaration if given (structural validation); nested mapping
        # data is normalized to a RecordSpec.
        # Otherwise, infers it once via RecordSpec.infer_from.

    @property
    def spec(self) -> RecordSpec: ...                    # the single stored source of the type
    def to_numeric(self) -> NumericRecord: ...  # requires every leaf to be an array
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

`select` resolves each argument with `at_path`, so a key reaches a leaf and a partial path a subtree view, and returns a plain `dict` of tracked values carrying no schema; its purpose is `**`-splatting a value's parts into a `Function` call, with `select_all` the whole-record form over the top-level children.

**Access is tracked.** `record[path]` returns the field as the tracked term of its kind — a `NumericArray` for an array field, the stored term itself for a term-valued field — under the field key as its name, marked `name_is_auto`, with view provenance; an interior path yields a sub-`Record` view as before. `record.raw(path)` returns the stored representation, and `record.raw()` the whole record's nested mapping of raw leaves — the record kind's raw host.

When every leaf is numeric, a `Record` is a `NumericRecord`. Leaves are stored in native form — a bare array, an `xarray` / `pandas` container, or any registered array backend — and convert to `jax.Array` only at the compute boundary (the pytree flatten that `grad` / `vmap` / `jit` traverse, and `to_vector`), each leaf at most once. Because promotion changes no data, construction auto-promotes exactly when every leaf is numeric and no explicit non-numeric schema vetoes it, and every transform re-derives the promotion from the current leaves — removing the last non-numeric leaf promotes, introducing one demotes — exactly as for its schema (above). Flat vectorization reads its layout (`leaf_shapes`, `vector_size`, canonical order) from the schema. At the boundary a `NumericRecord` presents a bare array pytree, so it passes through `grad` / `vmap` / `jit` unchanged and a JAX round-trip returns bare-array leaves; passing through means leaf transport only — a transform never promotes a `Record` to a `RecordBatch`. Flattening is deliberately numeric-only, which is why `NamedTree` itself has no `flatten`.

```python
class NumericRecord(Record):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, spec: NumericRecordSpec, vec: Array) -> NumericRecord: ...
```

**Vector-space arithmetic and the two routes.** `NumericRecord` implements the `Numeric` interface of II.3, and its arithmetic follows the two routes stated there. ProbPipe's own surface preserves structure and returns tracked terms: the vector-space set — `+` and `-` between records sharing a schema, and scalar `*` and `/` — and `map(f)` for elementwise maps, while array-shaped operations (broadcasting, positional indexing) stay with arrays. Foreign functions see the coordinates: `jnp.cos(record)` works and returns the bare coordinate vector, and `record.map(jnp.cos)` is the tracked spelling. No `__array_ufunc__` is defined, so NumPy and JAX functions never behave differently on the same object.

### Rationale

A `Record` is the *values* half of `C1 – Uniform interface to distributions and values`: a distribution's draw is a `Record` (or a `RecordBatch` for many), and a function over named values consumes one. One class serves the schema and the kind because they denote the same space: a second tag class would be a distinction without mathematical content, converted at every construction site (`D1 – Mathematical fidelity`, `D7 – Single source of truth`). Carrying the schema forward from its producer rather than re-inferring it downstream is `D5 – Explicit, carried structure` made concrete.

### Notes

- *Pytrees.* `Record` and `NumericRecord` are registered as JAX pytrees for advanced use, and the native `NamedTree` methods are the supported interface. JAX traversal follows the pytree registration, which does not always agree with ProbPipe on what is a leaf, so users applying raw JAX functions are responsible for the documented behavior. Record equality is structural value equality, which is weaker than treedef equality. The registration's children are the field arrays (a `NumericRecord`'s native leaves convert at this boundary) and its static aux data is the schema alone: identity is boundary-attached (II.4), so name, provenance, annotations, and native container types never enter a trace, and a name can never affect compilation-cache identity. Lineage instead rides on the function layer, which records the transform itself.

- *Single-field presentation.* A `Record` is a container and presents as one, whatever its field count: no coercion, no forwarding, and no array surface beyond the vector-space operations above.
- *Construction validation.* Construction checks each leaf against its spec's `is_valid`, which validates structure only — for a `NumericArraySpec`, shape and dtype (dtype by `numpy.can_cast` same-kind: a widening promotion or a within-kind narrowing passes, a cross-kind conversion raises). A `NumericArraySpec`'s `support` is **not** part of `is_valid`: it is a data-dependent, element-wise check that reduces to a Python `bool` and so cannot run under `jax.jit` tracing, where construction also happens (pytree unflatten reconstructs a value inside the trace). `support` is therefore descriptive metadata, and invariant 2 (`is_valid`) covers shape and dtype. Leaf validation is skipped on the unflatten path, where a leaf's shape is transform-relative.

## III.5 — `RecordBatch` and `NumericRecordBatch`

### Contract

A `RecordBatch` is a batch of `Record`s that all conform to one shared `RecordSpec`. It is the batched value a `Function` produces and consumes, such as the many draws a `sample` yields. It is a *collection* of records rather than itself a named tree. `NumericRecordBatch` is the all-array specialization. Indexing reaches both axes and stays unambiguous by dispatching on the key's type:

```python
class RecordBatch(Batch[Record]):
    @property
    def spec(self) -> BatchSpec: ...                # the batch's own type
    @property
    def element_spec(self) -> RecordSpec: ...       # view on spec
    def raw(self) -> Mapping[str, Any]: ...
    # the storage view: the nested mapping of raw columns, each field's raw batch form (II.6)

    def __getitem__(self, key: int | slice | tuple[int, ...] | str | tuple[str, ...]) -> Record | RecordBatch | Array | Batch: ...
    # int / slice (or a tuple of ints) -> an element Record or a sub-batch, indexing the batch axes
    # field path (str or tuple of strs) -> the field's column in its native batch form:
    #   an array for an array field, the matching element batch otherwise; a sub-RecordBatch if nested
```

Storage is columnar: per-field columns in each field's batch form. A field column is therefore a direct view, and an element `Record` is assembled on demand from the columns rather than stored a second time. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `element_spec`. Its `raw()` is that columnar store itself, the nested mapping of raw columns; a `NumericRecordBatch` additionally presents the coordinates view, `__jax_array__` being its batched `to_vector`, so the tracked batch, the raw columns, and the flat coordinates are three presentations of one store.

When every element is a `NumericRecord`, the batch is a `NumericRecordBatch`: a pytree of arrays whose leading dimensions are the `batch_shape`, bound to one shared `NumericRecordSpec`. Its columns are the leaves `vmap` / `grad` / `jit` traverse. The batch is *rebuilt* on the way out only where the level identity of what arrives is recoverable — a transform that preserves every batch axis, or one that removes all of them, which yields a single `NumericRecord`. Mapping one level of several is refused: the pytree hook is not told which axis the transform consumed, and no shape records it, so a rebuilt `BatchSpec` could name the wrong level. An operation that knows which level it consumes carries that knowledge itself; the workflow sweep does, mapping raw columns and building each row explicitly. It also adds batched flat vectorization, where `to_vector` stacks one flat vector per element into a `(*batch_shape, vector_size)` array:

```python
class NumericRecordBatch(RecordBatch):
    def to_vector(self) -> Array: ...
    @classmethod
    def from_vector(cls, name: str, spec: NumericRecordSpec, vec: Array, *,
                    level_names: str | Iterable[str],
                    axis_groups: Iterable[Iterable[int]] | None = None) -> NumericRecordBatch: ...
    # vec has shape (*batch_shape, vector_size): the last axis is the flat dimension
```

An operation that mints a level takes the name to give it (II.6), so both of the
constructions that mint one require it: `from_vector` names the levels it
reconstructs, which is what lets a multi-level batch round-trip, and `stack`
names the single level it introduces.

```python
class RecordBatch(Batch[Record]):
    @classmethod
    def stack(cls, records: list[Record], *, level_name: str,
              element_spec: RecordSpec | None = None) -> RecordBatch: ...
    # one level of (len(records),); the element spec is taken from the first record
    # when omitted, and every record's fields must be exactly its fields
```

### Rationale

It claims only the batch axis and never the leaf-keyed `Mapping` contract, so a batch of `N` records can never read as one record of `N` fields — `D1 – Mathematical fidelity` at exactly the point where the two would otherwise be conflated.

## III.6 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces. It is the linear subtype of `Function` (III.3), so it is `TrackedTerm` and applies, composes, and evaluates like any map; the operator algebra and the structured queries below are what linearity adds. Its action is the map the base carries: `apply` evaluates the operator at a value, a `Numeric` conforming to its input schema, returning the matching form, with the operator's parameters as the private state behind it. `matvec` is syntactic sugar for `apply`, its linear-algebra name; `matmat` applies the action to stacked columns in one routine and is the operator's registered batched rule, with `rmatvec` and `rmatmat` for the transpose. It is how ProbPipe represents structured matrices, above all covariances, without materializing them. It carries an input and an output `NumericRecordSpec` — its domain and codomain schemas — reached through the declared sides it inherits as a `Function`, with no operator-specific accessor beside them: the `FunctionSpec` declares a single structured slot on each side, so the input schema is that slot's spec in `input_spec` and the output schema is `output_spec`'s spec. It therefore maps numeric records and not just anonymous vectors, and a bare matrix is given names explicitly rather than defaulting to a single-field placeholder. The two sides coincide exactly when the operator maps a space to *itself* (an endomorphism such as a covariance or Hessian): then the two schemas are equal, which the operator algebra reads as the structural fact that operands compose or act on the same space.

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
    # syntactic sugar for apply: A x, with a NumericRecord flattened through
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

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators (`ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view) that defer to their parts. The scalar `*` coexists with distribution composition by operand type. The algebra checks and propagates the schemas: `A @ B` requires `B`'s output schema to equal `A`'s input schema and declares `B`'s input schema and `A`'s output schema as its own sides, `A + B` requires both pairs to match, and `A.T` swaps them. Composite operators are tracked terms like any other, with names auto-derived from their operands and marked `name_is_auto`.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant.

**The batch form.** `LinOpBatch` is the element batch over operators, a thin `Batch[LinOp]` whose elements share both schemas. It is what a batched `cov` returns. Application is elementwise: a single operator maps over a `RecordBatch`'s elements, and a `LinOpBatch` zips with a broadcast-compatible `RecordBatch`, element by element in both cases. The queries lift the same way, elementwise to batched results.

### Rationale

Operations mint linear operators, covariances above all, so the kind exists to keep those results first-class (`D4 – Closed system of objects under operations`). The structured subclasses exploit their form automatically behind one interface (`C3 – Computational detail hidden by default, available on demand`), the algebra returns lazy views rather than materialized matrices (`D7 – Single source of truth`), and typing both sides with numeric schemas makes closure concrete: the operator `cov` returns accepts the very draws its distribution produces (`D5 – Explicit, carried structure`).

### Open points

- *Structure-exploiting solves.* Exploiting structure in both operands of `A⁻¹B`, possibly through a dedicated `SolveLinOp`, is open.
- *Flag semantics.* Whether flags only describe structure or also steer which implementation a query selects is open.
- *Batched matrix action.* `matmat` against a batched operand, where a batch axis would meet the operator's matrix axis, and any richer `LinOpBatch` alignment are deferred until a concrete consumer exists.

## III.7 — `Distribution`

### Contract

A `Distribution[T]` is a single random law: a probability measure over values of type `T`. The type `T` is the natural raw form of a draw (an `Array` for a scalar law, or a `Record` for a multi-field one), which implementer code uses directly. Its single stored source of type is its `DistributionSpec`, whose `event_spec`, the output declaration of one draw (an `OutputSpec`, II.2), it exposes as a view. Construction accepts a bare term spec or nested mapping data as a convenience, normalizing to an `OutputSpec` whose name defaults from the law's own constructor `name`, and the stored spec's class fixes the draw kind. It is `TrackedTerm` and `Annotated`. It declares the operations it supports as **capabilities**, so operational support is decoupled from the class. The draw a *user* sees is the tracked term of the declared kind: a `Record` for a record-valued law, and a `NumericArray` for a scalar one (III.1). It is not wrapped in another kind to make it uniform, so a draw's type follows the law's declaration rather than its field count. Fields are renamed or moved with `with_path_names` — a path-valued target restructures the event (II.1) — returning the same law under the canonical relabeling of its event space; on a factored joint the result is a relabeling view over the stored factors. A distribution whose declaration is polymorphic is legal, binding exactly as II.2 fixes — by value, or explicitly through `with_dims`.

A `NumericDistribution` is a `Distribution` whose draws are numeric, so it carries a `NumericRecordSpec` and can use the flat-vector machinery.

```python
class Distribution[T](TrackedTerm, Annotated):
    def __init__(self, name: str, event_spec: OutputSpec | TermSpec | Mapping) -> None: ...
        # the event declaration, normalized to an OutputSpec (name defaulted from
        # the law's own name); its stored spec's class fixes the draw kind

    @property
    def spec(self) -> DistributionSpec: ...     # the single stored source of the type
    @property
    def event_spec(self) -> OutputSpec: ...     # view on spec: the event declaration
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename or move event fields; keys and path-valued targets resolve as for
    # NamedTree.with_path_names (II.1), and the law is unchanged
    def with_dims(self, **sizes: int) -> Self: ...
    # bind named symbolic dimensions (II.2); a conflict with an existing binding raises
    def __getitem__(self, path: str | tuple[str, ...]) -> FieldView: ...
    # the field view at a leaf or group path; available on every distribution

class NumericDistribution(Distribution): ...   # marker: numeric draws, carries a NumericRecordSpec
```

**Field views.** `d[path]` returns a `FieldView`: a `Distribution` over the field or field group at `path`, holding a reference to its parent rather than a detached law. Sibling views co-sample from one parent draw, so correlation between them is preserved. Every record-drawing law's declaration has at least one named field, so views exist on every record law, however it was constructed and whatever it supports. A term-drawing law declares no fields, so it offers no field interface. The capabilities a view offers are derived from its parent's, one by one, fixed with the capability protocols.

```python
class FieldView(Distribution):
    # constructed by Distribution.__getitem__, never by hand
    @property
    def parent(self) -> Distribution: ...
    @property
    def path(self) -> str: ...
    # the declaration is the parent's schema at path; name == the field key,
    # marked name_is_auto; provenance records the view and its parent
```

**The distribution term specification.** `DistributionSpec` is the distribution kind's term spec. As a leaf, it types a field holding a matching `Distribution`. As an event declaration, it declares a random measure: a distribution whose draws are themselves `Distribution`s. The draw kind comes from the declaration, never from what `_sample` happens to return.

```python
class DistributionSpec(TermSpec):  # a Distribution; is_valid accepts a matching Distribution
    event_spec: OutputSpec         # the output declaration of one draw (II.2)
```

### Rationale

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to distributions and values`. A field view is a reference rather than a copy (`D7 – Single source of truth`), and deriving its capabilities from its parent's keeps advertised support honest (`D3 – Capability-based operations`); the rest — the stored spec, purity, tracked results — is Part II's contract at the distribution kind.

## III.8 — Distribution capabilities

### Contract

Each operation on a distribution is a **capability**: a distribution implements an underscore method (`_sample`, `_log_prob`, `_mean`, …) on the raw type `T` for each operation it supports, and the matching user-facing op (`sample`, `log_prob`, `mean`, …) wraps the result at the boundary when appropriate.

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
    def _quantile(self, q: ArrayLike) -> Array: ...   # one value per level in q; per-coordinate for multivariate

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

Here `Key` is a PRNG key and `ArrayLike` an array-or-scalar input. `_cov` and `_quantile` require a numeric draw, with `_cov` ranging over the *flattened* draw, while `_mean` and `_variance` are event-typed and open to any event type that supports them. For a random measure, a draw's log-density is itself random, so the `SupportsRandom*LogProb` capabilities take no value and return the law of the log-density function, the random function `x ↦ log D(x)` for `D ~ M`. `_expectation` must integrate an *arbitrary* function exactly, which in practice means finite support: its argument is an opaque callable, so a per-call feasibility check has nothing to inspect, and a law that is exact only for special maps must not advertise the capability. Exact moments of structured maps are instead the business of `evaluate`, which dispatches on the map's type.

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

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Because support is structural (tested by `isinstance(dist, SupportsX)`, not subclassing), a distribution gains an operation just by implementing its method. A transform that preserves the event exposes exactly the capabilities of whatever it wraps, and a field view offers those its parent's capabilities can derive, so advertised support stays honest in both cases.

## III.9 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution[S, T]` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `Distribution` is the empty-given corner of this picture, a kernel with nothing to condition on, so its marginal law exists and `sample` / `log_prob` / `mean` apply unconditionally, whereas a kernel with a non-empty given has none. A distribution is *isomorphic* to a kernel with an empty given but not identical to one, and the two stay distinct tracked types, neither inheriting from the other. ProbPipe represents only the distribution at that corner: a `ConditionalDistribution` carries a non-empty `given_spec`, so there is no empty given and no rule that turns an empty-given kernel into a `Distribution` — a kernel is never in that state, since binding its last given field returns the `Distribution` directly — and a `ConditionalDistributionSpec` likewise carries a non-empty given, the empty corner being `DistributionSpec`'s. They are siblings sharing a capability vocabulary, each unconditional capability mirrored by a conditional twin that prepends the given.

A `ConditionalDistribution` carries a `given_spec`, the `InputSpec` of named slots it conditions on, and an `event_spec`, the output declaration of one produced draw `T`. The event side is read exactly as for a `Distribution`: normalized to an `OutputSpec` at construction, its stored spec's class fixing the draw kind. The given side is always an `InputSpec` of independently bindable slots (II.2). Both sides are views on its stored `ConditionalDistributionSpec`, the single source of its type. Unlike a function's domain and codomain, a kernel's given and event are distinct *roles* — the value conditioned on versus the law produced — so their field names stay disjoint even when the two spaces coincide. For example, a Markov kernel (where `S = T`) uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. Symbolic dimensions are scoped over the two sides jointly, so a name shared between given and event fields is one dimension, and the fused conditional paths bind dimensions from the given value at call time. `with_path_names` renames or moves names across both sides, returning the same kernel: the event side behaves exactly as a `Distribution`'s, and on the given side a path-valued target may split or group slots, since a kernel carries no signature to pin its top level — unlike a `Function`, whose input slots the signature fixes (III.3). `with_dims` binds symbolic dimensions across both.

Users never call a method on the `ConditionalDistribution`. Instead, they use the existing ops: `condition_on(K, s)` binds the given fields, evaluating it to a `Distribution` exactly and with no inference, and `sample(K, given=s)` / `log_prob(K, y, given=s)` / `mean(K, given=s)` are the fused conditional paths, with the invariant `op(K, given=s) == op(condition_on(K, s))`, bitwise under a shared PRNG key in the exact cases and in law when inference is involved. Conditioning on only a subset of given slots *curries* to a smaller `ConditionalDistribution` view, and a path key that binds part of a structured slot is restructure-then-bind, leaving the residual slot. A value supplied for a given field is always bound, whatever its type; the mixture `∫ K(s, ·) μ(ds)` over a mixing distribution is obtained through the separate `mixture` operation, not by conditioning on a distribution.

```python
class ConditionalDistribution[S, T](TrackedTerm, Annotated):
    def __init__(self, name: str, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | TermSpec | Mapping) -> None: ...
        # given before event, as in FunctionSpec; the stored event spec's class fixes the draw kind
    @property
    def spec(self) -> ConditionalDistributionSpec: ...   # the single stored source of the type
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

| class | given | event |
|---|---|---|
| `NumericConditionalDistribution` | numeric | any |
| `ConditionalNumericDistribution` | any | numeric |
| `FullyNumericConditionalDistribution` | numeric | numeric |

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

A `DistributionBatch` is a `Batch` of `Distribution`s: `N` separate distributions sharing one event declaration, indexed along a batch axis. A `ConditionalDistributionBatch` is the same construction over `ConditionalDistribution`s: `N` separate conditional distributions sharing one `given_spec` and one event declaration. The shared event declaration is the elements' `event_spec`, so a batch of random measures declares term-valued draws exactly as its elements do. They are grouped because they are the identical multiplicity wrapper over the two distribution-like base types, and the conditional one merely adds the `given_spec`. They are also the native batch forms of `DistributionSpec`- and `ConditionalDistributionSpec`-valued draws.

```python
class DistributionBatch(Batch[Distribution]):
    @property
    def spec(self) -> BatchSpec: ...          # the batch's own type
    @property
    def element_spec(self) -> DistributionSpec: ...   # view on spec
    @property
    def event_spec(self) -> OutputSpec: ...   # view on spec
    def __getitem__(self, index: int | slice) -> Distribution | DistributionBatch: ...

class ConditionalDistributionBatch(Batch[ConditionalDistribution]):
    @property
    def spec(self) -> BatchSpec: ...                     # the batch's own type
    @property
    def element_spec(self) -> ConditionalDistributionSpec: ...   # view on spec
    @property
    def given_spec(self) -> InputSpec: ...               # view on spec
    @property
    def event_spec(self) -> OutputSpec: ...              # view on spec
    def __getitem__(self, index: int | slice) -> ConditionalDistribution | ConditionalDistributionBatch: ...
```

### Rationale

This is `D1 – Mathematical fidelity` on the distribution layer: a `DistributionBatch` of `N` laws is a *collection of separate measures*, kept firmly distinct from one *joint* law over a product space, exactly as a `RecordBatch` of `N` draws is distinct from one `Record` of `N` fields. It is the natural result of a vectorized operation that yields many distributions: sweeping a parameter batch through a `ConditionalDistribution` produces a `DistributionBatch` of conditioned laws.

## III.11 — Factored distributions

### Contract

A *factored distribution* is a distribution **built from named sub-distributions**. Beyond being an ordinary distribution, it carries an explicit factorization into its parts. The capability that marks a factored distribution is `SupportsFactors`. The `FactoredDistribution` and `FactoredConditionalDistribution` classes generically implement `SupportsFactors` for distributions and conditional distributions. As another example, the `FactoredMultivariateGaussian` is a factored distribution in which the factors are jointly Gaussian, so conditioning and marginalization are exact.

The generic factored (conditional) distributions `FactoredDistribution` and `FactoredConditionalDistribution`  carry an ordered list of factors, each a `Distribution` or a `ConditionalDistribution`. The dependence graph is *derived* by matching each factor's given fields against the fields produced by earlier factors, rather than stored. The joint distribution's event declaration is the structural, disjoint union of the factors' **produced slots**: a record-drawing factor contributes its top-level fields, internal structure preserved and no additional nesting introduced, and any other factor contributes the single field its `OutputSpec` names. Factor names are unique across the list, and a duplicate is an error.
In the case of the `FactoredConditionalDistribution`, conditioning values for all given fields results in a `FactoredDistribution`. Sampling and the log-prob capabilities follow the intersection of the factors': if all factors implement `SupportsLogProb` then so does the factored distribution. The moment capabilities are decided at construction, present exactly when the joint's structure makes the moment derivable, so a capability check stays honest. An edge-free joint derives its moments componentwise, so it has a given moment exactly when every factor does, the same intersection rule sampling and log-prob follow. Jointly Gaussian factors compose into a `FactoredMultivariateGaussian`, whose moments are exact. Any other dependent joint carries no moment capability, and the Monte Carlo fallback answers, since factor-wise conditional moments do not compose into a closed form. For example, with `x ~ Normal(0, 1)` and `y | x ~ Normal(exp(x), 1)`, `E[y] = e^{1/2}` is not reachable from the factors' means. As with other generic distributions, there are also numeric specializations.

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

**Field versus factor.** A **field** is a named part of a draw, that is, a path in the event declaration. A **factor** is a constituent distribution the joint was built from. The two coincide only for an independent joint of single-field factors and differ in general. A correlated `MultivariateNormal` presented as `{intercept, slope}` is one factor with two fields. Conversely, the same draw `{x, y}` can arise from a single bivariate normal (no factors), from two independent factors (no edges), or from a chain p(y | x) · p(x) (two factors, one edge). The fields are identical but the factorization differs.

**The two access interfaces.** A joint exposes up to two clearly separated interfaces, never through the same operator.
- The **field interface** is available on every distribution. `d["intercept"]` returns a **view**: the field's marginal carrying a reference to its parent, so that sibling views co-sample from one parent draw and preserve correlation under broadcast. A view carries each capability its parent's capabilities can derive, its density routes through the parent's `SupportsMarginals`, and `marginal(d, "intercept")` returns that same marginal **detached** from the parent.
- The **factor interface** is available only with `SupportsFactors`. `factor(d, "coeffs")` returns a building-block factor, keyed by factor name, which is a `Distribution` or, for a dependent edge, a `ConditionalDistribution`. There need be no factor for a given field, and no field for a given factor.

**Marginals of a joint.** Whether a marginal is exactly available depends on the factors' own marginal support and on where the target sits in the dependence graph, so the factored classes resolve `_marginal` per path rather than wholesale. The graph reduction is always exact: the target's ancestor closure yields a sub-joint of whole factors, and everything outside it integrates out for free. What remains is integrating the extra ancestor fields back out, which is exact in three cases: there are none (the target is ancestrally closed, as for a root factor or an edge-free group), the reduction lies within a single factor and delegates to that factor's own `SupportsMarginals`, or the affected factors admit closed-form integration, as when they are jointly Gaussian. On any other path `_marginal` raises, and the `marginal` operation falls back to its Monte Carlo route.

### Rationale

Factorization is an *optional capability*, `SupportsFactors`, rather than a base class. This is `D2 – Generality first`: a joint is an ordinary distribution that gains factor access by carrying the capability, instead of sitting in a parallel class tower. Keeping the field interface (part of a draw) and the factor interface (part of the construction) separate serves `D1 – Mathematical fidelity`, since the two are genuinely different in the mathematics. Independence is likewise a property of the derived graph rather than a class, so dropping any dedicated product class loses no behavior: an edge-free joint still samples its factors in parallel and conditions exactly on a field. For sampling and densities a joint's capabilities are the intersection of its factors': ancestral `sample` requires every factor to sample, and a summed `log_prob` requires every factor to score. Deciding moment presence at construction keeps `D3 – Capability-based operations` honest, since a capability is advertised exactly when the object can answer it.

### Notes

- *Group views.* The field interface also accepts an interior path, which names a group of fields rather than a single field. For example, when the event declaration nests `coeffs/intercept` and `coeffs/slope` under `coeffs`, `d["coeffs"]` returns the marginal over the whole group. Like a single-field view, it is a view onto the parent joint, not a detached distribution, so co-sampling through the parent preserves correlation.

## III.12 — Composition

### Contract

Composition builds a factored distribution from parts, written as an *expression*: a single binary operator `*` combines `Distribution`s and `ConditionalDistribution`s into one joint. The *kind* of the result is **derived** from the operands and never chosen by hand, and every result is itself a `Distribution` or a `ConditionalDistribution`. The operator is defined after the factored-distribution classes it returns. The base objects carry no composition logic. Each merely exposes `*` as a thin `__mul__` that delegates to the operator, which keeps the dependency one-directional.

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

**Naming the result.** A joint is *derived*, not created by the user, so `*` **auto-derives** its `name` deterministically from its factors. The factors are listed in **canonical order** (the conditional-first topological order of the flattened factor graph, with factors incomparable in the derived graph ordered lexicographically by the fields they produce), and their names are joined by `·`. So `lik * prior` is named `lik·prior`, and because neither association nor the ordering of independent factors changes the canonical list, `A * B * C`, `(A * B) * C`, and `A * (B * C)` produce the same joint distribution. The derived name is marked `name_is_auto`.

Re-composition reads `name_is_auto`. An auto-named operand is **flattened**: its factors enter the new joint directly, its old name is discarded, and a fresh name is derived from the full factor list. An operand whose name the user has pinned with `with_name` is **not** flattened. It enters as a single factor under that name, and that name appears as one token in the parent's derived name. So `(lik * prior).with_name("posterior")` both labels the joint and, in any later composition, keeps it as the single factor `posterior`.

```python
def __mul__(self, other: Distribution | ConditionalDistribution) -> FactoredDistribution | FactoredConditionalDistribution: ...
```

### Rationale

Reifying both degrees of freedom would force a 2×2 of joint classes. By `D2 – Generality first`, an independent product (`bound = ∅`) is just an edge-free joint, so *dependent?* is a runtime property of the derived graph and only *conditional?* names a class, giving two classes rather than four. Deriving the name deterministically keeps a joint's meaning clear without forcing the user to label every intermediate output (`C5 – Naming for unambiguous meaning`), while `with_name` lets the user impose grouping where it carries intent. The conditional-first order is already a valid topological listing, so composition is associative and acyclicity is automatic, with no separate graph inference. Associativity rests on the `G_B ∩ F_A = ∅` requirement, under which the validity of `(A * B) * C` and `A * (B * C)` coincide.

### Notes

- *Operator coexistence.* `*` also denotes scalar scaling on some objects, such as a random function or a linear operator. The two coexist by operand-type dispatch: `Distribution` and `ConditionalDistribution` operands compose, while scalar operands scale.
- *The realigning `joint` form.* `joint(A, B, **align)` is `*` plus field renaming — path-valued targets included (II.1), so realignment can also promote a nested field to a slot-matchable name — for factors whose names do not line up: it is `A * B.with_path_names(**align)`.

## III.13 — The `Distribution` hierarchy

### Contract

With the base `Distribution`, the `ConditionalDistribution`, and the factored distributions and their composition in place, the distribution *kinds* can be cataloged. Every class named here is defined earlier, and this section only organizes them. Two **independent** questions classify any distribution:
1. **The type axis** — what does a draw look like, and is a factorization exposed? This fixes *which interfaces apply*.
2. **The family axis** — how is the law realized? This fixes *which capabilities* the object implements, and how.

The axes are orthogonal, so they combine freely: a `Normal` is *atomic* (type) and *parametric* (family); a posterior over `{μ, σ}` reconstructed from samples is *atomic-structured* and *empirical*; a joint's factors may themselves come from any family.

**The type axis — atomic vs. joint.** The structural classification:

| type | a draw is | factorization? | interface beyond the capabilities |
|---|---|---|---|
| **atomic, array-valued** | one `Array` (scalar or vector) | none | — (a single-field draw) |
| **atomic, structured** | a multi-field `Record` | none | the *field* interface — `d["x"]`, `marginal` |
| **joint** | a `Record` | `SupportsFactors` | the *field* interface **and** the *factor* interface — `factor` |

The line between the last two is **factorization, not field count**: a multi-field empirical distribution or an amortized posterior has fields but no factors, so it is *atomic-structured*, and only a distribution built from named sub-distributions is *joint*. Hence there is **no `RecordDistribution`**: draw structure lives in `event_spec` and factor access in `SupportsFactors`.

**The family axis — how the law is realized.** Each family is an ordinary `Distribution` (a `NumericDistribution` when its event is numeric), differing only in which capabilities it implements and how — refinement by capability, not a parallel class tower:

| family | the law is realized as | capabilities, and how |
|---|---|---|
| **parametric (closed-form)** | the standard tensor-library families — continuous (`Normal`, `Gamma`, `Beta`, `Exponential`), discrete (`Bernoulli`, `Categorical`, `Poisson`), multivariate (`MultivariateNormal`, `Dirichlet`) — the bulk of the atomic, array-valued row | exact `sample` / `log_prob` / moments over a constrained-support event declaration; the multivariate families implement `SupportsMarginals` exactly |
| **empirical** | a finite, possibly weighted, sample set — scalar or structured | `sample` resamples, moments are sample estimates, and marginals are again empirical; no density |
| **transformed (pushforward)** | a base distribution pushed through a map with recognizable structure | an invertible map keeps an exact `log_prob` by change of variables, and a linear map exact first and second moments; a general map's pushforward proceeds by sampling, landing in the empirical family |
| **mixture** | a convex combination of finitely many component distributions — the form a dependent joint's detached `marginal` takes when the mixing parent is finite | what its components jointly support, combined componentwise |
| **random function** | a distribution over functions: a `FunctionSpec` event, so a draw is a callable — a Gaussian process is the canonical case | `mean` returns the mean function |
| **random measure** | a distribution *over distributions*: a `DistributionSpec` event, so a draw is itself a `Distribution` | `mean` returns the marginalized law |

Over a continuous mixing parent the true marginal is a continuous mixture, which no finite-component family can represent, so the `marginal` operation returns its Monte Carlo fallback, an `EmpiricalDistribution` of projected draws, unless an exact route applies, as when the factors are jointly Gaussian.

Any family can arise as the approximation of a target: what makes a result approximate (the target, the method, and the fit) is recorded in its `provenance`.

**Conditional distributions and batches stratify identically.** A `ConditionalDistribution` repeats both axes (atomic / structured / the `FactoredConditionalDistribution` joint, crossed with parametric / amortized / empirical / …), and a `DistributionBatch` is `N` of any of these. The catalog is one classification, reused across the conditional and multiplicity layers.

### Rationale

The hierarchy embodies `D2 – Generality first`: one base refined by *optional capabilities* (`SupportsFactors`, the numeric marker, each `SupportsX`) rather than a rigid class tower, so a new family slots in by implementing the capabilities it supports rather than by widening the base. The atomic-versus-joint split is a `D1 – Mathematical fidelity` distinction, since a joint genuinely offers its factors as distributions while an atomic-structured law does not, and it is carried by a capability rather than by the draw's type. Because composition is closed (`D4 – Closed system of objects under operations`), a joint re-enters the catalog as an ordinary `Distribution` for the next operation.

## III.14 — Cross-type conversion

### Contract

A distribution may have more than one representation, and an operation or backend sometimes needs a different one than the user holds. **Conversion** moves a distribution from its current type to a requested target type, resolved by the **converter registry** whose methods are *converters*. The registry is an ordinary binary dispatch registry keyed on `(type(source), target)`: the target is already a type, so the second key is the argument itself rather than its type, supplied through the key extraction the dispatch base leaves to its subclasses.

Conversion is also the entry route for raw distributions: under the two-presentation rule (II.2), a backend distribution supplied at a distribution-shaped position — an argument, a given, a record field — is accepted in its raw form and converted on entry through the registry, exactly as a bare array is wrapped at an array-spec position.

A converter declares the source types it converts *from* and the target types it converts *to* in the binary method's two slots, a cheap `check` that reports feasibility without converting, and the conversion as its `execute`. A conversion is rarely unique, so each carries a **fidelity**: `exact` for an equivalent representation, `moment_match` when only low-order moments are preserved, and `sample` for a Monte Carlo stand-in. The fidelities are totally ordered, `EXACT > MOMENT_MATCH > SAMPLE`. Priority remains the registry's sole selection order, with converter priorities assigned from the fidelity tiers, `EXACT` in the exact tier and the inexact fidelities in descending bands below it, so higher fidelity is preferred through the existing mechanism rather than a second ordering. A caller can name a converter or set `min_fidelity`, a feasibility floor `check` enforces.

```python
class ConversionMethod(Enum):
    EXACT = "exact"
    MOMENT_MATCH = "moment_match"
    SAMPLE = "sample"

@dataclass(frozen=True)
class ConversionInfo(MethodInfo):   # the feasibility probe's result, extended with fidelity
    method: ConversionMethod | None = None

class Converter(BinaryDispatchMethod):
    name: str                            # unique within the registry; convert(..., method=name) selects it
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (source, target) types
    def check(self, source, target_type: type, *,
              min_fidelity: ConversionMethod | None = None) -> ConversionInfo: ...
    def execute(self, source, target_type: type) -> Distribution: ...             # the conversion itself

class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    # keyed on (type(source), target): the second key is the target type itself
    def convert(self, source, target_type: type,
                method: str | None = None, min_fidelity: ConversionMethod | None = None) -> Distribution: ...

converter_registry: ConverterRegistry   # the global instance
```

### Rationale

Conversion makes `C3 – Computational detail hidden by default, available on demand` concrete on the distribution layer: a representation is a computational choice, so the library converts as needed and the user rarely converts by hand. Recording each conversion's fidelity keeps the approximation honest, which is `D1 – Mathematical fidelity`, since an `exact` conversion loses nothing while a `moment_match` or `sample` conversion is a stated approximation the caller can see and control. New representations interoperate by registering converters, so the set of convertible pairs grows without changing the distributions themselves (`D2 – Generality first`). Realizing the registry as a subclass of the shared dispatch machinery gives conversion registration, feasibility probing, prioritized selection, and cataloging without duplicating any of them (`D7 – Single source of truth`).

## III.15 — Constraint reparameterization

### Contract

Many inference algorithms are defined to operate on an unconstrained space ℝᵈ, among them gradient-based optimization and Hamiltonian Monte Carlo, so a constrained support must be reparameterized. The **constraint-to-bijector factory** maps a `Constraint` (the support a `NumericArraySpec` carries) to a *bijector*: a `Function` that takes `ℝⁿ` onto that support and claims the two capabilities below. Invertibility is one capability: a `Function` claims `SupportsInverse` by providing the inverse map, its own `apply` (III.3) serving as the forward, so the protocol stays minimal and the forward map comes from the `Function` that claims it. The Jacobian determinant is a second, separate capability: `SupportsLogDetJacobian` provides the log-determinant of the Jacobian, which exists only for a differentiable map, and a map can be invertible without it. Change of variables requires exactly the pair, and both are typed over the `Numeric` interface (II.3), structured records and bare arrays alike. A `LinOp` claims them only when they apply: the claim is guarded per instance by squareness (equal input and output schemas), with its inverse from the operator algebra and its `logdet` the log-Jacobian, and a rectangular operator makes no claim. `is_invertible` reads the claim together with its guard — unconditional for a declared bijector, squareness for a linear operator — while singularity, which no construction-time check decides, surfaces at call time as `LinAlgError`, exactly as for `solve`. A slot checks the claims it needs at construction and raises a capability error otherwise: the bijector of a transformed distribution requires `is_invertible` and the Jacobian claim, the link of a GLM likelihood `is_invertible` alone. `bijector_for(constraint)` returns the canonical such `Function`, and `register_bijector` plugs in a factory for a constraint type or for a specific constraint instance, with instance registrations taking precedence over type registrations.

```python
@runtime_checkable
class SupportsInverse(Protocol):            # an invertible map
    # minimal by design: the forward map is the claiming Function's own apply (III.3);
    # typed over the Numeric interface (II.3), not fixed to arrays
    def _inverse(self, y: Numeric) -> Numeric: ...

@runtime_checkable
class SupportsLogDetJacobian(Protocol):     # a map with a tractable Jacobian determinant
    def _log_det_jacobian(self, x: Numeric) -> Array: ...
    # the log-determinant of the Jacobian at x, defined only for a differentiable map

def is_invertible(f: Any) -> bool: ...
# True when f claims SupportsInverse and the instance guard passes: unconditionally
# for a declared bijector, squareness for a linear operator

def bijector_for(constraint: Constraint) -> Function: ...
# the canonical map ℝⁿ → support(constraint), a Function claiming both capabilities
def register_bijector(
    key: type[Constraint] | Constraint,
    factory: Callable[[Constraint], Function],   # each factory output claims both capabilities
) -> None: ...
```

The factory keys on constraint instances and types rather than dispatching on argument types alone, so it is not a dispatch registry; it still satisfies `SupportsRegistryCataloging` and appears in the registry catalog alongside the dispatch registries.

### Rationale

A bijector for every constraint lets inference run in an unconstrained space while a model stays stated in its natural, constrained one, which serves `D6 – Differentiability as a capability`. Invertibility as a capability is `D3 – Capability-based operations`: an invertible map is an ordinary `Function` that additionally claims `SupportsInverse`, so it evaluates, composes, and pushes forward like any other, with *bijector* reserved for the mathematical statement. The Jacobian determinant is a separate claim for the same reason: a map can be invertible without a tractable determinant, so each claim stays honest on its own and change of variables asks for exactly the pair. Keeping the factory open through `register_bijector` is `D2 – Generality first`: a new constrained support becomes inference-ready by registering its reparameterization, without touching the distributions that use it.

### Open points

- *Round-trip fidelity.* The forward map (bijector to support) and this inverse map (support to bijector) are not strict inverses for every constraint, so a reparameterized support can drift to a coarser one. Whether to unify the two is unsettled.
