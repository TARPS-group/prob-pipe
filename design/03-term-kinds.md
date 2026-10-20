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

`NumericArray` is one array value with identity: a `TrackedTerm`, holding a single array whose `shape` is the **event** shape, with no batch axes; its `raw` is that array. `NumericArraySpec` is its kind's term spec:

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

`NumericArrayBatch` is the kind's batch form: a `Batch` whose element spec's `term_spec` is the `NumericArraySpec` and whose storage is one array with the batch axes leading — the same split `RecordBatch` uses, with one column instead of many. An array with leading axes is just an array; the batch form is what carries the level names, the shared spec, and provenance.

### Rationale

The full set of array operators is safe here and only here: with no fields, an operator applied to one array has exactly one meaning (`D1 – Mathematical fidelity`).

## III.2 — `Opaque`

### Contract

`Opaque` adds identity alone, and its `raw` is the wrapped value. `OpaqueSpec` is the fallback spec: it admits a value that no other kind admits, so a collection such as a list, a tuple, or a set is opaque:

```python
class OpaqueSpec(TermSpec):        # the fallback spec; is_valid accepts a value no other kind's spec class admits
    type: type | None = None       # the Python type of the admitted values; None admits any non-mapping value
    meta: Hashable = None          # free-form metadata, never checked or inferred
```

**`type` and `meta`.** `type` is the Python type of the values the spec admits, and `is_valid` checks `isinstance(value, type)`. `None` admits every value that is not a mapping, since a mapping is a group (III.5). Construction from a value infers the type: an `Opaque` built from a string and a string leaf of a record built from values both have `OpaqueSpec(type=str)`. An `OpaqueBatch` infers the type its elements share exactly, and `None` when they differ. `meta` is free-form hashable metadata, such as units or a tag. It is part of the spec's equality and hash, and it is never checked against a value or inferred. Two opaque specs unify when their types are equal or one of them is `None` and their `meta` are equal or one of them is `None`, and the unification takes the known type and the set `meta`, so a declared `meta` is kept when a declared spec is completed with the spec a value infers, which carries none. Completion keeps the declared type and checks that the produced value conforms (II.2).

`OpaqueBatch` is its batch form. It **stores** each element outright. Its `raw` is an object array of the stored raw values.

### Rationale

The kind exists so that closure under operations holds for every return value (`D4 – Closed system of objects under operations`), and it adds identity alone because a richer interface would promise structure the value does not declare (`D1 – Mathematical fidelity`).

## III.3 — `Function`

### Contract

The function kind's base type is `Function`. A `Function` is a tracked term that wraps exactly one Python callable as its representation and carries a `FunctionSpec`, whose two specs it presents as the `input_spec` and `output_spec` views; either side is optional, as in the spec. A `Function` also carries a frozen `inspect.Signature`, which is authoritative for Python argument binding, since parameter kinds, defaults, and variadic parameters are not expressible in a value schema; the `input_spec` is authoritative for the value schema. Construction validates their one-for-one correspondence, so binding an argument binds an input by name. Its `raw` is the wrapped callable.

A `Function` accepts an optional `output_label` as a result alias. Without an alias, a value its managed calls return displays the application expression, and a law or a kernel they return takes the function's label and displays its own signature, as `predict(y, mu)` (IV.2); named callables default to their Python name and lambdas to `𝒻`.

**Inputs and outputs.** A `Function`'s inputs are its parameters, and `input_spec` is read through the `InputSpec` constructor from any of its four sources (II.2): an `InputSpec`, a mapping, a `RecordSpec`, or an `OutputSpec`. A parameter with a default is an optional input, and an `input_spec` that marks optional a parameter without one raises `ValueError`. A `Function`'s outputs are those of its return value (II.2). Its **output name** is the callable's original name, with `f` for a lambda and `result` for a callable without a name or whose name is not a Python identifier or is a keyword. An `OutputSpec`, such as `OutputSpec(mean=None)`, states the outputs, or the return value's spec. A bare term spec given as `output_spec` reads as `OutputSpec.default(spec, output_name=...)` under the output name, and so does the spec of the value a call returns when no output spec is given:
1. a base spec is the keyword form under the output name;
2. a record is positional, its children the outputs;
3. a batch whose elements carry only the default name `element` is positional, its elements named by the output name;
4. a distribution, a conditional distribution, a function, or any other batch is positional, so a returned distribution has no outputs, and a returned function or batch has its own.

The output name comes from the callable and never from the label, so relabeling a function or a result never changes its outputs. The body builds any record itself, since ProbPipe never wraps the value a call returns (II.2): to pass a returned distribution on by name, the body returns `{"params": law}` under `OutputSpec(RecordSpec(params=law_spec))`.

**Selection and merge.** A `Function` is selected by the paths of its return value (II.6). `f[s]` is the function with the same inputs that returns `f(w)[s]`, so `f["y"](1.0)` is `f(1.0)["y"]`, and without a declared return value its paths are checked when the selected function is called. `f.output(c)` is `f[c]`, or `f` itself when it returns a base value named `c`. `f.without(*paths)` returns `f(w)` without those paths, and `rename` renames inputs and outputs alike, so a key that names both an input and an output of a function renames both. `f & g` is the function that returns both results: each output of `g` supplies the input of `f` of the same name, every input left over, optional ones included, is an input of the result, and an input both take is one input, under the conditions of IV.2. Both operands must declare their inputs and a return value whose spec is specified.

```python
@function(label="predict", output_label="prediction",
          output_spec=OutputSpec(mean=None))
def predict_impl(theta, x):
    return x @ theta

prediction = predict_impl(theta, x)
# predict_impl.label == "predict"; prediction.label == "prediction"
# the output mean; the inferred return is an array, not a record
```

A `Function` is invoked two ways. `apply` evaluates the wrapped callable at a point: given values that conform to `input_spec`, it returns one conforming to `output_spec`, with no tracking or lifting — the raw map that operations such as change of variables build on. `__call__` runs the **call handler**, which is the base's one extension point: on the base it is plain evaluation, and the engine layer (Part V) replaces it once, at import. The base also carries its **controls** (V.2), set at construction and revised functionally by `using`; the engine alone interprets them, at call time.

A `Function` is constructed directly or using the `@function` decorator. `FunctionSpec`, which is the function kind's term spec, admits any callable. Construction also takes the `Function`'s **execution claims** (V.13): `traceable` states whether JAX can trace the callable, `True`, `False`, or `None` for no statement, and `vectorized` states whether it takes a stack of evaluation points in one call. Unlike controls, claims are fixed for the `Function`'s life, and every copy `using` makes keeps them.

```python
class FunctionSpec(TermSpec):      # the function kind's spec; is_valid accepts any callable
    input_spec: InputSpec | None   # None: that side's structure unspecified
    output_spec: OutputSpec | None
```

```python
class Function(TrackedTerm):
    def __init__(self, fn: Callable, /, *, label: str | None = None,
                 input_spec: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None,
                 output_spec: OutputSpec | TermSpec | None = None,
                 output_label: str | None = None,
                 traceable: bool | None = None,
                 vectorized: bool = False,
                 differentiable: NumericSpec | None = None,
                 bind: Mapping[str, Any] | None = None,
                 **controls) -> None: ...
                 # traceable, vectorized: execution claims, fixed for the Function's life (V.13)
                 # differentiable: an optional differentiability claim (V.11)
                 # bind: arguments of the wrapped callable fixed at construction (V.3)
                 # controls: the controls' construction values (V.2)
    @property
    def spec(self) -> FunctionSpec: ...
    @property
    def input_spec(self) -> InputSpec | None: ...                   # view on spec
    @property
    def output_spec(self) -> OutputSpec | None: ...                 # view on spec
    @property
    def output_label(self) -> str: ...                             # the results' alias, else the label; outside spec
    @property
    def options(self) -> Mapping[str, Any]: ...          # the controls; opaque to the base
    def using(self, **controls) -> Self: ...             # functional update
    def apply(self, *args, **kwargs) -> Any: ...
    # evaluate the wrapped callable at a point (input_spec -> output_spec),
    # with no tracking or lifting; the raw map operations build on
    def __call__(self, *args, **kwargs) -> Any: ...
    # run the call handler: plain evaluation on the base, the Part V engine after import
    @property
    def notation(self) -> str: ...   # the label and the parameters, as predict(x, y); str() returns it (II.4)
    def __getitem__(self, paths: str | Sequence[str], /) -> Function: ...   # w -> self(w)[paths], with the same inputs
    def output(self, output_name: str, /) -> Function: ...
    def without(self, *paths: str) -> Function: ...
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Function: ...
    def __and__(self, other: Function, /) -> Function: ...   # the function returning both results

def install_call_engine(engine: Callable[..., Any]) -> None: ...
    # replaces the call handler, once, at import time; until then calls evaluate plainly.
    # The engine reads the controls the Function carries and must agree with
    # plain evaluation on concrete values.
```

`FunctionBatch` is the function kind's batch form, storing its elements exactly as `OpaqueBatch` does (III.2); an element is a `Function` whatever callable its position holds, and `raw` is an object array of the stored callables.

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

`is_invertible(f)` is `True` when `f` claims `SupportsInverse` and its guard accepts `f`, and `False` otherwise. An unresolved guard (II.7) therefore gives `False`, and an input that needs the inverse raises `ResolutionError` (V.12).

### Rationale

Defining the base in the value layer keeps the layering strict: the representation is fixed here, the call engine arrives by upward registration (`D2 – Generality first`), and `LinOp` and the specs reference `Function` downward — the split the package structure realizes as `values/_function_base.py` and `functions/`. Invertibility as a capability is `D3 – Capability-based operations`: an invertible map is an ordinary `Function` that additionally claims `SupportsInverse`, so it evaluates, composes, and pushes forward like any other, with *bijector* reserved for the mathematical statement. The Jacobian determinant is a separate capability for the same reason, since a map can be invertible without a tractable determinant, and change of variables asks for exactly the pair.

## III.4 — `LinOp`

### Contract

A `LinOp` is a lazy linear map `A : ℝⁿ → ℝᵐ` between flat numeric spaces and the linear subtype of `Function` (III.3). It therefore applies, composes, and evaluates like any map. Its action is the map the base carries: `apply` evaluates the operator at a `Numeric` conforming to its input schema and returns the matching form, with the operator's parameters as private state. `matvec`, `matmat`, `rmatvec`, and `rmatmat` are the linear-algebra names for the action and its transpose, and `matmat` is the operator's registered batched rule. Its output spec names its output: a constructor given only a codomain shape declares an output at the root under the fixed name `result`, the output name of a nameless callable, which no label changes (III.3). Its domain is the `NumericSpec` (II.3) of its single input, and its codomain is `output_spec.term_spec`; a record output may have several outputs while remaining one numeric value. The operator reads its spaces from these declarations alone. It therefore maps whatever `Numeric` its sides declare, for example a bare array under a `NumericArraySpec` side, so an operator over a scalar law's draws takes them as bare arrays. The two sides coincide for an endomorphism such as a covariance or Hessian, which the operator algebra reads as the fact that operands compose or act on the same space.

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

**The operator algebra.** `A @ B`, `A + B`, `c * A`, and `A.T` return lazy composite operators that defer to their parts: `ProductLinOp`, `SumLinOp`, `ScaledLinOp`, and a transpose view. The algebra checks and propagates the schemas: `A @ B` requires `B`'s output schema to equal `A`'s input schema and declares `B`'s input schema and `A`'s output schema as its own sides, `A + B` requires both pairs to match, and `A.T` exchanges the term specs of the two sides: its one input takes the original output's spec, and its output takes the original input's spec, named by that input when the spec is a base spec, as the keyword form requires, and positional otherwise (II.2). Each side keeps its own spec class, since an `InputSpec` and an `OutputSpec` are different contracts (II.2). Composite operators are tracked terms like any other, with names derived from their operands.

**Structured subclasses.** `DenseLinOp`, `DiagonalLinOp`, `TriangularLinOp`, `CholeskyLinOp`, `RootLinOp`, and `DiagonalRootLinOp` each override the queries their structure accelerates, such as a triangular solve or a diagonal log-determinant. A constructor from arrays derives the output spec from the matrix shape, as an output at the root under the fixed name `result`, and accepts an `output_spec` that names the output otherwise or states its spec; a consumer that knows the event spec, such as covariance construction (VII.6), passes it. Each also fixes the kind's `raw` (II.4) as its stored parameterization:
- `DenseLinOp`: the matrix;
- `DiagonalLinOp`: the diagonal;
- `TriangularLinOp`: the triangular matrix, whose flags name the triangle;
- `CholeskyLinOp`: the triangular factor;
- `RootLinOp` and `DiagonalRootLinOp`: the root operator `S` of `A = S Sᵀ`.

A composite's `raw` is its operand tuple in its constructor's order, such as `(A, B)` for `A @ B` and `(A, c)` for `c * A`, since laziness is its representation.

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
                 **fields: TermSpec | Mapping | tuple[int | str, ...]) -> None: ...
    # a tuple of sizes is the size shorthand of II.1; a field given None raises
    # TypeError, since None is an unspecified spec (II.2), and an opaque field is OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> RecordSpec: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    def numeric_subset(self) -> NumericRecordSpec: ...   # remove non-NumericSpec leaves
```

**Field order.** A field's order is part of the spec: `RecordSpec(x=(), y=())` and `RecordSpec(y=(), x=())` are unequal, and neither conforms to nor unifies with the other (II.1). Wherever a value meets a `RecordSpec`, such as a function's return value, a call's argument, a kernel's conditioning value, or `Record(..., event_template=...)`, a raw mapping is laid out in the declared order at each node and then checked, so the key order of a user's mapping never matters: a function declared to return `RecordSpec(x=(), y=())` that returns `{"y": 0.0, "x": 1.0}` returns the record `{x, y}`. A `Record` already built in another order is refused with `ValueError`, as is a `RecordSpec` in another order where one must conform, and the message names both orders. A record that conforms therefore flattens in the declared order.

**Selection.** A `RecordSpec` is indexed by its paths (II.6): `s["g/b"]` is the spec there, a group's being a `RecordSpec`, and `s["a", "g/b"]` is the `RecordSpec` of the chosen nodes, each at its own path. `keys()` lists the field paths, `without` removes nodes, and `s & t` merges two record specs by top-level name.

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

Within one schema a symbolic name refers to one dimension: fields `X: ("obs", "features")` and `coefficients: ("features",)` share the dimension `features`, which no pair of concrete integers can express (II.1). Validation therefore runs one conformance check over every occurrence of a name across all fields, so data of shapes `(100, 5)` and `(5,)` bind both dimensions consistently while `(100, 5)` and `(7,)` raise. A leaf's `is_valid` checks its own rank and dtype; sizes belong to the one pass, since only it sees every occurrence of a name. A nested spec's schema lies inside the scope, so a name declared within a `DistributionSpec` is the same dimension as that name beside it, binding once whatever the order of the fields.

Two rules govern record-shaped positions, symmetric in what arrives. Mapping data materializes into the record's own structure under derived identity, whereas a supplied tracked term is stored with its identity intact as the field's **source**; access is described below. Both conform to the same spec, so structure and identity agree on what a field is.

A `Record` is a `NamedTree` that is a `TrackedTerm` with leaves that are *values*, its structure conforming to its authoritative `RecordSpec`. `NumericRecord` is the specialization in which every leaf's value implements `Numeric`, so it carries a `NumericRecordSpec`.

Since the structure of `Record` matches that of its schema, the following invariants must hold:
1. *matching fields:* `record.keys() == record.spec.keys()`.
2. *valid values:* for any field path `p`, the value stored at `p` satisfies `record.spec[p].is_valid`.
3. *matching groups:* for any path `p` of a group, `record[p].spec == record.spec[p]`.

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
    @property
    def raw(self) -> dict[str, Any]: ...
    # the whole record as the nested mapping of raw leaves; a node's is record.raw["g"]["x"]

    @classmethod
    def from_field_values(cls, template: RecordSpec, values: Iterable[Any], *,
                          label: str | None = None) -> Record: ...
    # reconstruct from values in the schema's canonical order; ValueError on count/shape mismatch

    def output(self, output_name: str, /) -> Any: ...   # the top-level field or group output_name
    def select(self, *paths: str, **remap: str) -> dict[str, Any]: ...
    # nodes into a plain dict for **-splatting into a `Function` call, each keyed by its
    # path's last name; keywords remap: select(x="g/b") == {"x": self["g/b"]}
    def select_all(self) -> dict[str, Any]: ...   # every top-level field, ready to splat
```

`select` resolves each path with `[]`, so a field path selects a field and a group's path a sub-record view, and keys each by its path's last name, so `select("a", "g/b")` is `{"a": ..., "b": ...}`. Two paths that share a last name, as `g1/theta` and `g2/theta`, raise `ValueError`, and a keyword names one of them instead. It returns a plain `dict` of tracked values carrying no schema; its purpose is `**`-splatting a value's parts into a `Function` call, with `select_all` the whole-record form over the top-level children.

**Outputs and merge.** A record's value type is its own spec, so its outputs are its top-level children, fields and groups alike (II.2), its output spec `OutputSpec(record.spec)` is derived, and `output(c)` returns the child `c`. `x & y` merges two records by top-level name (II.6), and a shared top-level name raises, a group included.

**Storage and access are separate contracts.** Storage retains the representation and the source: leaves are held in native form, so a supplied `NumericArray`'s array is stored as that array, and a supplied term's identity is held as a reference or a descriptor per the provenance mode (II.4). Access returns views. `record[path]` gives the node at a path (II.6), a group as a sub-`Record` view and a path through a held term that term's selection, as `{x: 3.0, g: d}["g/mu"]` is `d["mu"]`. `record[path]` at a field, `values()`, and `items()` give each field as a view (II.4) of the field's kind, labeled by its path:
1. an array field as a `NumericArray`;
2. an opaque field as an `Opaque`;
3. a callable field as a `Function`;
4. a stored term of another kind, such as a law, as a copy of that term.

`record.at_path(*segments)` is `record["/".join(segments)]`, and `[]` is preferred. `record.raw` is the whole record's nested mapping of raw leaves, the record kind's raw host, and a node's stored form is an entry of that mapping, as `record.raw["g"]["x"]`, inside a trace as well. A field's view has its own `raw`, which presents the leaf at the view's kind, so a Python scalar stored in a record reads as an array through `record["x"].raw`.

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

**Vector-space arithmetic.** `NumericRecord` implements the `Numeric` interface of II.3, so functions act on it in the two ways stated there. ProbPipe's own operators preserve structure and return tracked terms. They are the vector-space set, which is `+` and `-` between records sharing a schema and scalar `*` and `/`, and `map(f)` for entrywise maps, while `&` merges records by name and is not among them, so `record.map(jnp.cos)` is the tracked form of `jnp.cos(record)`. Array-shaped behavior such as broadcasting and positional indexing stays with arrays, and `__array_ufunc__` is left undefined, so NumPy and JAX functions behave alike on the same object.

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
    @property
    def raw(self) -> dict[str, Any]: ...
    # the storage view: the nested mapping of raw columns, each field's raw batch form (II.5);
    # a node's columns are batch.raw["g"]["x"]

    def __getitem__(self, key: int | slice | tuple[int | slice, ...] | str | Sequence[str]) -> Record | RecordBatch | Batch: ...
    # a position -> an element Record or a sub-batch, indexing the batch axes
    # a path -> its tracked batch column: a NumericArrayBatch for an array field,
    #   the matching element batch for another field, a sub-RecordBatch for a group
    # a list of paths -> the RecordBatch of the chosen nodes, each at its own path (II.5)
```

Storage is columnar: per-field columns in each field's batch form. A field column is therefore a direct tracked view, and an element `Record` is assembled on demand from the columns. A `RecordBatch` omits the field-keyed `Mapping` protocol (`keys()` / `values()` / `children`), so `len` and `iter` unambiguously range over the batch, and the field structure is read from `element_spec.term_spec`. Paths are read as II.6 states and II.5 applies to batches. The tree transforms of II.6 apply columnwise: `rename`, `without`, `replace`, and `map` act on every element's fields at once and return a batch on the same levels, and two batches of records over equal levels merge element by element with `&`. `select` and `select_all` splat fields as they do for a `Record` (III.5), keying each path by its last name, and each entry they return is a batch with one column. A `NumericRecordBatch` additionally presents the coordinates view through `__jax_array__`, which is its batched `to_vector`, so the tracked batch, the raw columns, and the flat coordinates are three presentations of one store.

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

Its elements implement `Numeric` (II.3), and the batch's `to_vector` stacks their vectors. A `RecordBatch` is promoted as a `Record` is (III.5): `RecordBatch(...)` and `RecordBatch.stack` return a `NumericRecordBatch` when every column is numeric and no explicit non-numeric `element_spec.term_spec` vetoes it, and a view over numeric fields, such as a field selected from a batch that also holds opaque fields, is a `NumericRecordBatch` too.

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

A `Distribution` is a probability measure over the values its event spec describes. Its `DistributionSpec` holds the **event spec**, the output spec of one draw, which the property `event_spec` returns (II.2): `event_spec.term_spec` is the spec of a draw, the law's value type, and the law's outputs and paths are those of its event spec. It is the same spec class a `Function` holds as `output_spec`, and the attribute names `event_spec` and `output_spec` distinguish a draw from a function's return value. A constructor reads `event_spec=` by one rule: a bare `RecordSpec` is the positional form, whose top-level names are the law's outputs, and an event over a base spec needs its name, in the keyword form. A family takes the name of its output at the root first, as `Normal("mu", 0.0, 1.0)` does, and an `event_spec` it also takes to declare the event's spec names that same output. A family over distributions, functions, or batches wraps each value it is over in a record under that name, so a random measure constructed under `G` is a law over `{G: law}` (VII.5). A constructor completes the event spec from its parameters with `with_term_spec` (II.2), so the stored event spec is complete. The label is the optional keyword `label=`, which defaults to the family's class name, as `Normal`, and to `ℙ` for any other law (II.4), so `Normal("mu", 0.0, 1.0)` is labeled `Normal` and displays as `Normal(mu)`.

It declares the operations it supports as **capabilities** (III.8), so operational support is decoupled from the class. Its `raw` is the law detached (II.4). The `raw` of a view `d[p]` is the representation of the detached marginal, `d._marginal(p).raw` (III.8): the backend object where the marginal has one, such as the TFP distribution of a parametric family (VII.1), and the detached law otherwise. That `raw` raises `ResolutionError` where `d` has no exact marginal at `p`. A draw is a tracked term of the kind its event spec's `term_spec` names.

**Outputs and paths.** A law's outputs are its event spec's outputs, and its paths are the paths of its event spec (II.2, II.6), which every draw, its mean, and a batch of its draws share. A law over `{mu, sigma}` has the outputs `mu` and `sigma`, and `Normal("x", 0.0, 1.0)` has the one output `x`, at the root, and no paths. `d[s]`, for a path or a list of paths read as II.6 states, is the marginal law there, a view whose event spec is `d.event_spec[s]`:
1. at a path to a base node, a law over an array whose output at the root is named by the path's last name, so `d["g/b"]` has the output `b`;
2. at a group, a law over the group's record, whose children are its outputs, so `d["g"]["b"]` is `d["g/b"]`;
3. at a list of paths, a law over the record of the chosen nodes, each at its own path, in the law's order, so `d["a", "g/b"]` is a law over `{a, g: {b}}`;
4. at or through a field that holds a law, a kernel, a function, or a batch, a law over that field's selection, in the positional form, so the view at a law-valued field is a law over laws, which has no outputs.

The name of an output at the root alone is not a path, so `Normal("x", 0.0, 1.0)["x"]` raises `KeyError`, and `output("x")` returns the law itself; a list may name it, so `Normal("x", 0.0, 1.0)[["x"]]` is the law over `{x}`. `d.output(c)` is `d[c]` for an output `c` of a record event. `p in d` holds exactly when `d[p]` succeeds. `d.without(*paths)` is the marginal law of every node except those at `paths`, so `d.without("g/c")` is the same law as `d["a", "g/b"]`. Two views of one law merge as independent laws (IV.2), so the joint law of two groups is `d["g1", "g2"]`.

`rename` returns the same law with its event spec renamed by `OutputSpec.rename` (II.2): nodes renamed or moved by path, or the output at the root renamed. A factored joint renames through its factors, and a rename that gathers outputs of several factors under one group makes them one factor, a grouped sub-joint (IV.1). `with_label` changes only the object label. A polymorphic law binds its dimensions as II.1 specifies. Dimension transforms commute with renames: `d.rename(m).with_dim_sizes(n=3)` is `d.with_dim_sizes(n=3).rename(m)`, and likewise for `with_dim_names` and for a kernel (III.9). A renamed law conditions as the original does on the given translated to the original paths, with each value's fields in the original node's declared order whatever order the caller wrote (VI.6). Its marginal at a node that gathers fields of several of the original's nodes is the original's marginal at the selection of those fields, placed under the node, since a transform that preserves the event exposes what the law it wraps supports (III.8).

```python
class Distribution(TrackedTerm):
    def __init__(self, event_spec: OutputSpec | RecordSpec, *, label: str | None = None) -> None: ...
        # a subclass passes the label its caller gave, or its default;
        # a bare RecordSpec is the positional form, OutputSpec(event_spec) (II.2)

    @property
    def spec(self) -> DistributionSpec: ...
    @property
    def event_spec(self) -> OutputSpec: ...     # the event spec, read from spec
    @property
    def event_shape(self) -> tuple[int, ...]: ...    # defined only when a draw is a single array

    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Self: ...
    # rename or move nodes by path, or rename the output at the root (II.2); the law is unchanged
    def with_dim_sizes(self, **sizes: int) -> Self: ...
    # bind named symbolic dimensions (II.1); a name that is not a free dimension raises
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions before merging (IV.2)
    def __getitem__(self, paths: str | Sequence[str], /) -> MarginalView: ...   # the marginal law at a path or a list of paths
    def __contains__(self, path: object, /) -> bool: ...  # whether self[path] succeeds
    def output(self, output_name: str, /) -> Distribution: ...   # self[output_name], or self at the output at the root
    def without(self, *paths: str) -> MarginalView: ...   # the marginal law of every node except those at paths
    @property
    def notation(self) -> str: ...   # the label and the signature, as prior(mu); str() returns it (II.4)
```

**Numeric distributions.** A `NumericDistribution` is a `Distribution` whose `event_spec.term_spec` is a `NumericSpec` (II.3), so its draws implement `Numeric` and the flat-vector interface applies; a scalar `Normal`'s `NumericArraySpec` event qualifies as a record event does. Membership is read from the event spec, so `isinstance(d, NumericDistribution)` holds if and only if the event spec of `d` is numeric, whatever its class. A class whose every instance is numeric may inherit the marker, and construction checks that each instance is numeric. Exactly the numeric laws have the marker's properties, which, like `event_shape`, are final and computed from the event spec:

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

**Marginal views.** `d[s]` returns a `MarginalView`: a `Distribution` over the nodes at `s` that holds a reference to its parent, computes from it, and conditions and scores through it. It is the same law as `marginal(d, s)` (VI.8), which has no reference to `d`. A view at a list of paths is one `MarginalView`, whose `path` is the tuple of the selected paths, and a view that `without` returns records the paths of the nodes it keeps. A view displays as the detached marginal at its path does: it is labeled as `marginal` labels that marginal (VI.8), its signature is its own outputs (II.4), and it holds the paths its parent holds fixed. So `prior["beta"]` is labeled `prior` and has the output `beta`, `model["y"]` displays as `model(y)`, and `model["mu"]` at the factor `prior` displays as `prior(mu)`. A view of the whole events of several factors displays factor by factor, as `a(a)·b(b)`, and a view at one factor holds the paths that factor holds fixed before its parent's. The view `post["mu"]` of a posterior `post = condition_on(model, {"y": data})` over `mu` and `tau` displays as `model(mu; y)`. The repr names the class `MarginalView` and the path, which distinguishes a view from the detached marginal. A view of a view and the marginal of a view are labeled by the parent's factors whose whole events their paths select, and otherwise by the view's label. The capabilities a view offers are derived from its parent's (III.8). An empty list, a repeated path, and a path inside another raise as II.6 states, from indexing and from `_marginal` alike. On a view, `with_dim_sizes` and `with_dim_names` apply to the parent and return the view of the result at the same path, since the view's event spec is the parent's at that path and a schema is one dimension scope (II.1).

**The flat-vector law.** A numeric law's law over its coordinates is `pushforward(d, to_vector)`, with the map specialized to `d.event_spec.term_spec` and carrying an output spec in the keyword form. Its inverse reconstructs that original event, including a record's nesting. The map claims the inverse and unit-Jacobian capabilities, so the change-of-variables rule preserves an available density (V.7). This changes the event space by a declared isomorphism; an ordinary representation conversion preserves the event spec (IV.3). An inference method that works on ℝᵈ also applies the reparameterization of V.12.

```python
class MarginalView(Distribution):
    # constructed by Distribution.__getitem__ and without
    @property
    def parent(self) -> Distribution: ...
    @property
    def path(self) -> str | tuple[str, ...]: ...
    # the event spec is the parent's event_spec[path]; a view (II.4)
```

**The distribution term specification.** `DistributionSpec` is the distribution kind's term spec. As a leaf, it types a field holding a matching `Distribution`: one whose own event spec conforms to the declared one, so its outputs agree and bind in one dimension scope (II.1). As the `term_spec` of an event spec, it describes a law over laws, such as a random measure, whose draws are themselves `Distribution`s and which has no outputs unless it wraps them under a name (II.2). Its own equality compares the event spec, the name of an output at the root included, while a law's fingerprint leaves the name out (II.2).

```python
class DistributionSpec(TermSpec):  # a Distribution; is_valid accepts a matching Distribution
    event_spec: OutputSpec         # the event spec: the output spec of one draw (II.2)
```

### Rationale

Including a `Distribution` class is necessary to satisfy `C1 – Uniform interface to functions, distributions, and values`. A marginal view is `B4 – No copying at boundaries` at a path, and deriving its capabilities from its parent's ensures a view advertises only what it can compute (`D3 – Capability-based operations`). A view is labeled as the marginal at its path because the two are one law, and its event spec is the parent's at that path, so a view and a draw of it are indexed alike (`C5 – Naming for unambiguous meaning`). Taking a law's paths from its event spec makes selecting commute with sampling: a draw of `d[p]` is distributed as a draw of `d` at `p`, so the mean of `d` at `p` and the mean of `d[p]` are the mean of one quantity (`D1 – Mathematical fidelity`). A `Distribution` takes no type parameter, because the type of its draws is a function of the stored event spec and a static parameter could record only the spec's kind (`D6 – Single source of truth`). Requiring the name of an output at the root and defaulting the label serves `C5 – Naming for unambiguous meaning` on both counts: the output name names what a draw holds, which a merge matches, and the label names the law for a reader, so neither stands in for the other and the label never enters the mathematics.

### Open points

- *Structuring an atomic event.* Moving a law's output at the root into a group (`rename({"x": "group/x"})`) is mathematically well-defined, being the canonical isomorphism with the one-field product, but changes the draw's kind, so II.2 refuses it. A law that wants the structure declares a one-field record event, `OutputSpec(RecordSpec(x=...))`, instead; revisit only if a concrete consumer appears.

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
    def _expectation(self, f: Callable[[Any], Array]) -> Array: ...   # exact 𝔼[f(X)] for arbitrary f

class SupportsExactConditioning(ABC):        # claimed by inheriting, not structurally
    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution: ...   # the conditional law given fixed values
class SupportsApproximateConditioning(ABC):  # same primitive, returning a stand-in for that law
    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution: ...

@runtime_checkable
class SupportsMarginals(Protocol):
    def _marginal(self, path: str | tuple[str, ...]) -> Distribution: ...   # the detached marginal at a path or a list of paths
```

Here `Key` is a PRNG key and `ArrayLike` an array-or-scalar input. `_expectation` integrates an *arbitrary* function exactly, which in practice means finite support: its argument is an opaque callable, so a guard has only the law to inspect, and only a law exact for every integrand claims the capability. The exact expectation of a structured map is computed through the evaluation rules of `pushforward`, which dispatch on the map's type (VI.5). A law of finite support, such as an empirical law, computes `_expectation(f)` by evaluating `f` at every atom, in one vectorized call when `f` traces, as `auto` dispatch does (V.9), and one atom at a time otherwise.

`d._marginal(p)` returns a law labeled as `marginal(d, p)` is (VI.8), with the event spec of the view `d[p]` (III.7), so at a list of paths its nodes keep their paths, in the law's order.

**Normalization.** A law is **normalized** when it claims one of the following capabilities, and **unnormalized** otherwise:

- `SupportsLogProb`: a density that integrates to one;
- `SupportsSampling`: draws from the law;
- a moment, quantile, or expectation capability: a functional of the law.

Each of these capabilities is defined only for a probability law: a sampler draws from one, and each of the others is defined against one. A kernel's laws are normalized when the kernel claims the conditional twin of one of these (III.9). The classification reads protocol membership alone, so a route decides it without evaluating a body (VI.0), and `condition_on` uses it to return a normalized law (VI.6).

**View derivation.** A `MarginalView` derives each capability from its parent's. For a parent `d` and a view `v = d[p]`, with π the selection at `p` from an event:

| capability on `v` | derivation | available when |
|---|---|---|
| `_sample` | co-sample: draw `X ~ d` and return `π(X)` | parent `SupportsSampling` |
| `_mean` | projection: `mean(d)[p]`, since `𝔼[πX] = π 𝔼[X]` | parent `SupportsMean` |
| `_variance` | restriction of `variance(d)` to the coordinates of `p` | parent `SupportsVariance` |
| `_cov` | the sub-block `P Σ Pᵀ`, with `P` the coordinate-selection `LinOp`, built lazily through the operator algebra | parent `SupportsCovariance`, numeric field |
| `_quantile` | restriction of the parent's per-coordinate quantiles to `p` | parent `SupportsQuantile`, numeric field |
| `_expectation` | composition: `d._expectation(f ∘ π)` | parent `SupportsExpectation` |
| `_log_prob` / `_unnormalized_log_prob` | via the detached marginal `d._marginal(p)` | parent `SupportsMarginals`, exact at `p`, and the marginal at `p` reports the density |
| `_mean`, `_variance`, `_cov`, `_quantile` of a parent that claims none | the moment of the detached marginal `d._marginal(p)` | parent `SupportsMarginals`, exact at `p`, and the marginal at `p` reports the moment; a numeric field for `_cov` and `_quantile` |
| `_marginal` at a path `r` of `v` | path composition: `d._marginal(p/r)` | parent `SupportsMarginals`, exact at `p/r` |
| `_condition_on` a proper sub-field `s ⊂ p` | conditioning commutes with marginalization: `d.condition_on(s)[p ∖ s]`, both sides the law of `p ∖ s` given `s` | parent conditioning available for `s` |

The projection rows are exact whenever the parent's answer is, and the rows through the marginal are exact per path. Each row needs only the parent capability it names, so a view of a parent that does not sample still carries its projected moments.

Each derived capability carries the parent's guard for the call that its derivation makes, such as the parent's quantile guard at the same probabilities for the quantile row. A moment row through the marginal carries the parent's marginal guard at `p` and then the marginal's guard of the moment, so the view returns or raises as the marginal does. The sample row passes its key and sample shape to the parent unchanged, so a view's draw at a key is the projection at `p` of its parent's draw at that key. A given that covers every field of the view is malformed, since no law remains, so the conditioning row's guard rejects it and `_condition_on` raises `ValueError`. A selection drops each node that the given covers and keeps the rest.

**The marginal's capabilities.** A law claiming `SupportsMarginals` may define the companion `_marginal_capabilities(path)`, which returns the capabilities its exact marginal at `path` claims, read from its declarations without building the marginal. The marginals of a law that defines none claim the law's own capabilities. A factored joint reports the capabilities of the factors a marginal keeps, and an empirical law reports sampling and its moments but no density, so a view of `Normal("a", 0.0, 1.0) & EmpiricalDistribution(atoms, output_name="b")` at `a` claims a density and one at `b` does not. A view reads the report once, at construction, since its path is fixed. It offers the density rows when the report includes the density, and a moment row its parent does not claim when the report includes the moment, so a view of a dependent joint at a root factor carries the factor's exact moments, which the joint does not claim. The `by_output` route of `mean`, `variance`, and `quantile` (VI.5) computes those outputs of the joint's summary through their views, so those outputs of the summary are exact. The projection rows derive from the parent's own capabilities, since each computes from the parent's answer.

### Rationale

Making each operation a *capability* rather than a base-class method follows `D3 – Capability-based operations`. Structural protocol membership identifies an implementation without requiring inheritance; its guard determines the calls it supports (VI.0). A transform that preserves the event exposes exactly the capabilities of whatever it wraps, and a marginal view offers those its parent's capabilities can derive, so advertised support matches actual support in both cases.

## III.9 — `ConditionalDistribution`

### Contract

A `ConditionalDistribution` is a *probability kernel* `K : S → P(T)` — a family of distributions p(· | s) indexed by a *conditioning value* `s : S`. Supply a value for what it conditions on and it yields an ordinary `Distribution` over what it produces. A `Distribution` is the empty-given case, whose law exists without a conditioning value, so the unconditional operations apply to it, and a kernel with a non-empty given has a law only once its given is bound. The two are distinct tracked types that both subclass `TrackedTerm`. A `ConditionalDistribution` and its spec always carry a non-empty `given_spec`, since binding the last required input returns a `Distribution` directly; the empty-given case is `DistributionSpec`'s.

A `ConditionalDistribution` carries a `given_spec`, the input spec of the inputs it conditions on, each bound independently (II.2), and an `event_spec`, the output spec of one draw, read as for a `Distribution` (III.7); both are read from its stored `ConditionalDistributionSpec`. `given_spec` is named for what the kernel is conditioned on, and it is read through the `InputSpec` constructor from any of its four sources, so `given_spec=d.event_spec` declares the inputs that a law `d`'s outputs supply, cut down with `[...]`, `without`, and `&` (II.2). Its outputs and paths are those of its event spec. Its `raw` is the kernel detached (II.4), as a law's is. A kernel's given and event are distinct *roles*, which are the value conditioned on and the law produced, so their input and output names stay disjoint even when the two spaces coincide, and a kernel that would have a name as both raises at construction. A Markov kernel with `S = T` uses names like `state → next_state` rather than `state → state`, for the same reason we write `K(x, dy)` rather than `K(x, dx)`. Symbolic dimensions are scoped over the two sides jointly, so a name shared between given and event fields is one dimension, bound by `with_dim_sizes` or, in the fused conditional calls, from the given value at call time. `rename` renames or moves names across both sides alike, returning the same kernel: the event side behaves as a `Distribution`'s, and on the given side a path-valued target may split or group inputs, since the kernel's `given_spec` alone fixes its inputs. The inputs that a split makes bind independently: after `{"theta/a": "a"}` splits `theta`, binding `a` alone leaves a kernel over the rest of `theta`, and the original kernel is evaluated once the rest of `theta` is bound. An optional input that a rename moves whole stays optional, and one that a move groups into a record input is required, since the record input's value holds it. A `Function`'s signature fixes its inputs (III.3), so restructuring them makes a new signature, obtained by wrapping the callable in one that takes the parameters wanted.

**Selection.** A kernel is selected by the paths of its event (II.6). `k[s]` is the kernel with the same inputs whose law at each value of its inputs is the marginal there at `s`, so `k[s].given_spec == k.given_spec`. `k.output(c)` is `k[c]`, or `k` itself at its output at the root, and `k.without(*paths)` is the kernel of every node except those at `paths`.

`condition_on(K, s)` binds the inputs and evaluates the kernel to a `Distribution`. The evaluation is exact unless the kernel claims `SupportsApproximateConditioning`, as an amortized posterior does, whose evaluation stands in for the posterior it was trained to approximate (VII.7). `condition_on` normalizes a result that is unnormalized (VI.6). `sample(K, given=s)`, `log_prob(K, y, given=s)`, and `mean(K, given=s)` are the **fused conditional calls**, with the invariant `op(K, given=s) == op(condition_on(K, s))`: the same law for exact realizations, and equal in law for their random draws. Equal draws need a shared workflow scope and, within it, the same sampling realization, random-event identity, and key derivation (V.8). A fused call served by an approximate route records the route and its assumptions in place of that invariant. Binding a subset of the inputs returns a smaller `ConditionalDistribution` over the inputs left unbound (VI.6).

```python
class ConditionalDistribution(TrackedTerm):
    def __init__(self, given_spec: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec, event_spec: OutputSpec | RecordSpec, *, label: str | None = None) -> None: ...
        # a subclass passes the label its caller gave, or its default; the event is read as a law's (III.7)
        # given before event, as in FunctionSpec
    @property
    def spec(self) -> ConditionalDistributionSpec: ...
    @property
    def given_spec(self) -> InputSpec: ...               # read from spec
    @property
    def event_spec(self) -> OutputSpec: ...              # the event spec, read from spec
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions on both sides (II.1)
    def __getitem__(self, paths: str | Sequence[str], /) -> ConditionalDistribution: ...   # the marginal at paths, at each input value
    def output(self, output_name: str, /) -> ConditionalDistribution: ...
    def without(self, *paths: str) -> ConditionalDistribution: ...
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Self: ...   # inputs and outputs alike
    @property
    def notation(self) -> str: ...   # the label and the signature, as glm(y | beta); str() returns it (II.4)
    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution | ConditionalDistribution: ...
    # the required primitive: the law K(given, ·), or the kernel over the unbound inputs for a partial given; every given value arrives in given, and the keyword options configure the kernel or the method, such as a budget

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

**The conditional distribution term specification.** `ConditionalDistributionSpec` is the conditional-distribution kind's term spec. As a leaf, it types a field holding a matching `ConditionalDistribution`: one whose inputs are exactly the declared names, in any order, and whose inputs' specs each accept the declared ones in the scope that the given side shares with the event side (II.1). The given side is checked contravariantly, so every value the declared input admits is one the kernel accepts. The event side is an output spec, checked as for `DistributionSpec`, and the given side is always an `InputSpec`, read through its constructor (II.2):

```python
class ConditionalDistributionSpec(TermSpec):  # a ConditionalDistribution; is_valid accepts a match
    given_spec: InputSpec          # the inputs it conditions on, non-empty
    event_spec: OutputSpec         # the event spec, as for DistributionSpec
```

### Rationale

Applying a `ConditionalDistribution` to a conditioning value returns a `Distribution`, which ensures `D4 – Closed system of objects under operations` is satisfied. A `ConditionalDistribution`'s capabilities are the `Distribution` capabilities shifted by one conditioning argument (`D3 – Capability-based operations`), so a single operation vocabulary applies to conditional distributions too, under the rule that *`Distribution` and `ConditionalDistribution` behave as similarly as possible*. The capabilities use distinct `_conditional_*` method names because a `@runtime_checkable` check matches on method name alone, so reusing `_sample` / `_log_prob` would corrupt the unconditional capability checks. `_condition_on` is the exception: fixing given fields means the same thing on both types, so a `ConditionalDistribution` claiming a conditioning capability is intended rather than a collision, and the names stay distinct where the meanings differ. The same rule is why the two conditioning capabilities are claimed by inheriting rather than structurally: whether `_condition_on` returns the conditional law or a stand-in for it is a claim about the result, which no check on method names can read, so two structural protocols declaring it would match the same classes.

## III.10 — `DistributionBatch` and `ConditionalDistributionBatch`

### Contract

A `DistributionBatch` is a `Batch` of `Distribution`s: `N` separate distributions sharing one event spec, indexed along a batch axis. A `ConditionalDistributionBatch` is the same construction over `ConditionalDistribution`s: `N` separate conditional distributions sharing one `given_spec` and one event spec. The batch's element spec is the positional form over the elements' spec, so its `event_spec` is `element_spec.term_spec.event_spec`, and a batch of random measures describes its draws as its elements do. A batch of laws or of kernels has no outputs (II.5), and it is indexed by the paths of its elements' events: `laws[s]` is the batch of the views `law[s]`, over the same levels, and a batch of kernels gives the batch of the kernels `k[s]`. They are the native batch forms of `DistributionSpec`- and `ConditionalDistributionSpec`-valued draws. Each stores its elements as `OpaqueBatch` does (III.2), so its `raw` is the object array of the stored laws or kernels.

```python
class DistributionBatch(Batch[Distribution]):
    @property
    def event_spec(self) -> OutputSpec: ...   # the shared event spec: element_spec.term_spec.event_spec

class ConditionalDistributionBatch(Batch[ConditionalDistribution]):
    @property
    def given_spec(self) -> InputSpec: ...               # the shared input spec, read from spec
    @property
    def event_spec(self) -> OutputSpec: ...
```

### Rationale

This is `D1 – Mathematical fidelity` on the distribution layer: a `DistributionBatch` of `N` laws is a *collection of separate measures*, distinct from one *joint* law over a product space, as a `RecordBatch` of `N` draws is distinct from one `Record` of `N` fields. It is the natural result of a vectorized operation that yields many distributions: sweeping a parameter batch through a `ConditionalDistribution` produces a `DistributionBatch` of conditioned laws.
