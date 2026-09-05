# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on.

| § | Category | Abstraction | Role |
|---|---|---|---|
| II.1 | Typing | `TermSpec` | The term-specification base every field and declaration is typed by, with the symbolic-dimension protocol. Each kind's concrete spec is defined beside the kind it describes. |
| II.2 | Declarations | `InputSpec` / `OutputSpec` | The input and output declarations of the map-like kinds: named slots in, one named produced term out. |
| II.3 | Numeric values | `Numeric` / `NumericSpec` / `Constraint` | The flat-vector interface the numeric kinds share, its spec-side mixin, and the elementwise support constraint. |
| II.4 | Identity | `TrackedTerm` / `Provenance` | The name, type (spec), lineage, and annotations an object carries beyond its raw representation, and `raw()`, its access to that representation. |
| II.5 | Multiplicity | `Batch` | The generic multiplicity axis: an indexed collection of *separate* objects, distinct from one object over a structured space, with its `BatchSpec`. |
| II.6 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.7 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved, and the catalog that makes every registry discoverable. The shared mechanism behind the operations, converters, inference selection, and bijector factories. |

## II.1 — Term specifications: `TermSpec`

### Contract

A **term specification** ("term spec") describes the typing information available for a mathematical term and validates whether an object satisfies it.

The specs partition into the **base kinds** and the **batch kinds**. Every base kind has exactly one term spec, one **base form**, and one **batch form**. Since a batch of batches is a batch, the base form and batch form of a `Batch` are identical.

**Symbolic dimensions.** A dimension size for a numeric value may be an integer or a **named symbolic dimension**: a name that fixes a dimension's identity while deferring its size. A spec with any symbolic dimensions is **polymorphic**; one with none is **concrete**. A spec can *report* the names still unbound, *substitute* explicit sizes for names, and *bind* names by unification against a value, reading the sizes off that value's spec. Binding is one unification over everything bound together: a name takes its size from its first occurrence, every later occurrence must agree, a disagreement raises, and a bound name never rebinds.

The base API is validation plus the dimension protocol:

```python
class TermSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...      # structural validity such as kind, rank, dtype

    # the symbolic-dimension protocol: report, substitute, bind
    @property
    def free_dims(self) -> frozenset[str]: ...       # report: the unbound symbolic dimensions
    @property
    def is_concrete(self) -> bool: ...               # True when free_dims is empty
    def with_dims(self, **sizes: int) -> Self: ...   # substitute explicit sizes
    def bind_dims_from_value(self, value: Any) -> Self: ...   # bind by unification against a value
```

### Rationale

One `is_valid` contract across the kinds keeps validation uniform (`C1 – Uniform interface to functions, distributions, and values`), and defining each concrete spec beside the kind it describes keeps this layer generic (`D2 – Generality first`). Naming the base for the terms it types is `C5 – Naming for unambiguous meaning` applied to the library's own vocabulary: every spec types a tracked term, and *value* stays reserved for the mathematical kind. The kind rule is `D2 – Generality first`: every result can be tracked and every collection of draws stacked, so nothing an operation produces falls outside the system. A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it; hence cross-field equalities travel with the term, and sizes bind when their producer appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`).

## II.2 — Input and output declarations: `InputSpec`, `OutputSpec`

### Contract

An `InputSpec` is a flat mapping from names to term specs: the independently bindable slots a map-like term takes in. An `OutputSpec` is a term spec plus a required name for the produced term.

```python
class InputSpec(Mapping[str, TermSpec]): ...   # named slots; keys are Python identifiers

class OutputSpec:
    name: str        # an identifier naming the produced term
    spec: TermSpec
```

### Rationale

Requiring a name on every output declaration serves `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows` together: the produced term's name is fixed where the producer is declared.

## II.3 — Numeric values: `Numeric`, `NumericSpec`, `Constraint`

### Contract

**The `Numeric` interface.** The numeric kinds share one flat-vector interface: `to_vector` lays a value out as one flat vector in canonical order, `vector_size` is that vector's length, and `from_vector` rebuilds a value from it. The coordinate protocols expose the same layout to foreign libraries, so `np.*` and `jnp.*` functions apply to the numeric kinds at the coordinates and return bare arrays. A bare array passed where a `Numeric` value is expected is promoted to the appropriate numeric type. Two routes therefore meet at a numeric value: a foreign function sees the coordinates and returns a bare array, while ProbPipe's own operators and elementwise `map` preserve structure and return tracked terms.

```python
class Numeric(ABC):                         # the flat-vector interface of the numeric kinds
    @property
    @abstractmethod
    def vector_size(self) -> int: ...       # total flat dimension; defined only when concrete
    @abstractmethod
    def to_vector(self) -> Array: ...       # the coordinates: one flat vector, canonical order

    # supplied once, so no kind restates them: numpy and JAX see to_vector
    def __array__(self) -> np.ndarray: ...  # and therefore return bare arrays
    def __jax_array__(self) -> Array: ...
```

**The `NumericSpec` mixin.** The spec-side counterpart to `Numeric` marks the specs whose values implement it. A `NumericSpec` carries `vector_size` and the shapes `from_vector` unflattens into; the layout itself is the kind's, fixed once by `Numeric`. Construction from coordinates stays with the value types, since a spec describes and never builds. The mixin is abstract and specifies no kind of its own.

```python
class NumericSpec(TermSpec, ABC):   # mixin: the specs whose values implement Numeric
    @property
    def vector_size(self) -> int: ...   # total flat dimension; defined only when concrete
    # the flat dimension and shapes the kind's to_vector / from_vector read; the layout is the kind's
```

A numeric kind may also specify the **support** of its values with a `Constraint`, which compares and hashes by value.

```python
class Constraint(ABC):
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # elementwise membership
```

### Rationale

One flat-vector interface over the numeric kinds is `D2 – Generality first`: everything that consumes flat numeric values types against it once, and the coordinate protocols keep foreign array functions usable with no ProbPipe-specific code (`C3 – Computational detail hidden by default, available on demand`). The spec-side mixin is the same generality at the type level, whether the event is one array or a named tree of them. Both are abstract bases rather than protocols, which keeps the pair symmetric and follows the rule the library uses throughout: an interface a closed set of ProbPipe kinds implements is a base, while an open claim any object may make is a structural protocol. The base also holds the shared coordinate protocols once rather than four times (`D6 – Single source of truth`). A constraint is data, not behavior: comparing and hashing by value lets a support key a registry, so the bijector factories select by the mathematics rather than by class identity (`D3 – Capability-based operations`).

## II.4 — Identity, type & metadata: `TrackedTerm`, `Provenance`

### Contract

`TrackedTerm` is the one mixin that carries identity, type, and metadata, so every tracked term carries the same four things and no kind opts in or out:
1. a **name**, what the object is called;
2. a **spec**, the declaration of its type (II.1);
3. a **provenance**, how it was produced;
4. free-form **annotations**, auxiliary information supplied by the user or an algorithm.

Every first-class object, the kind an operation consumes and produces, is a `TrackedTerm`; structural helpers such as specs are not.

A tracked term's name is supplied by the user at explicit construction, as the required first argument, and derived deterministically from the inputs when an operation produces the object. The `name_is_auto` flag records which, since the two behave differently: a structure-changing transform re-derives an auto-derived name from its result, and composition may rename it again, while a user-given name is preserved. `with_name` renames the object itself. A name is a label, not an identity: nothing resolves an object by name, derived names need no escaping scheme, and two objects may share a name unless field uniqueness forces them apart.

The `spec` slot is the term's type, stored once. Each kind narrows it to its own spec class and exposes convenience accessors for its properties.

Every tracked term exposes `raw()`, the single access point to the representation layer. It returns the term **detached** from the workflow: without provenance, annotations, or any reference to a container or parent it was viewed from, but keeping its spec, its name, and `name_is_auto`. A kind with a **raw host**, a representation that is not itself a ProbPipe object, returns it (the backing array, the nested mapping of raw leaves, the wrapped callable, the wrapped object, an operator's stored parameterization, or a batch's storage view); a kind whose representation is itself a ProbPipe object, such as a distribution or a kernel, returns that object detached.

Reaching into a container (a record field, a batch element, a distribution's field) returns a **view**: a tracked term named from the accessor, the field key or the selected levels, marked `name_is_auto`, whose provenance records the container and the source term where one was supplied.

**A tracked term is immutable.** Assignment and deletion raise, naming the class the caller touched, and every transformation, each `with_*` method included, returns a new term sharing the representation; nothing is relabelled in place. `TrackedTerm` carries the guard itself, so no subclass can report a different rule than its base.

Identity is **boundary-attached** under compiled execution. Inside a `jit` or `vmap` trace a term presents as its raw representation with only its spec as static data, so name, provenance, and annotations never enter a trace and a name can never affect compilation-cache identity; the tracked result is minted at the enclosing call boundary.

```python
class TrackedTerm:
    name:         str
    name_is_auto: bool
    spec:         TermSpec                       # the single stored source of the term's type (II.1)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    annotations:  Mapping[str, Any] | None       # free-form; the one store written after construction
    def with_name(self, name: str) -> Self: ...  # sets name_is_auto = False
    def with_provenance(self, p: Provenance) -> Self: ...
    def raw(self) -> Any: ...                    # the representation, detached from the workflow
    # immutable: __setattr__ / __delattr__ raise; state round-trips through the
    # attributes the term holds, so copy and pickle need nothing from the class

class Provenance:
    operation: str                       # the operation that produced the object
    parents:   tuple[ParentInfo, ...]    # descriptors of the tracked inputs
    controls:  Mapping[str, Any]         # the resolved controls: PRNG key, sample count, selected method, ...
    inputs:    Mapping[str, ParentInfo]  # plain (untracked) arguments, keyed by parameter name

class ParentInfo:
    type_name:   str
    name:        str
    fingerprint: str            # best-effort content hash
    fingerprint_is_weak: bool   # True when the fingerprint is only object identity
    parent:      Any | None     # optional reference to original parent
```

The **annotations** store is the one exception to immutability: it can be written after construction, so a diagnostic can attach its result to the object it examined, and by convention it is append-only. Annotations are otherwise inert: no operation reads them to decide behavior, they do not propagate to results (lineage is recorded in `provenance`), and they never enter a compiled trace.

Fingerprints are best-effort and tiered: a content hash, the qualified name and code hash of a closure-free callable, or object identity, with `fingerprint_is_weak` marking the weakest. **The provenance mode, not the operation, sets the tier.** The default lightweight mode records identity-tier descriptors for every operation's parents with no content hashing; the full mode retains parent references and content-verifiable fingerprints, computed lazily at export since the objects are held; off records nothing. Every operation behaves identically under a given mode, so provenance cost is one user-visible setting rather than a property of what ran.

### Rationale

`TrackedTerm` serves the two non-mathematical principles, `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D6 – Single source of truth` for a term's type: one slot, declared once, that every kind's accessors are views on. Recording the resolved controls, not just the parents, turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`), and boundary attachment keeps names inert in computation, so a name never decides what gets compiled. Immutability is `C2 – Functional interface over immutable objects` embodied; confining the one writable store to a container that no operation reads keeps that contract intact in substance, and carrying annotations on the base makes every tracked term annotatable, a batch of draws as much as a record. `raw()` is `B3 – Tracked forms out by default` for a term already in hand: the representation is one explicit call away and never the default.

### Notes

- *Reconstruction.* `pickle` and `copy` restore a term by assigning its state back rather than through its constructor, so a schema fixed by an explicit declaration, a class decided at construction, and a field written afterward all survive the round-trip. A term declares any memo as transient and any store written in place (annotations) as decoupled, so a copy takes its own container.

## II.5 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd-array of shape `batch_shape`, holding objects of a common element type. `batch_shape` is nonempty. A concrete batch specifies how it stores its elements and adds whatever its element type affords, including indexing into the elements' fields. Since a batch is a *collection* of its elements, `len`, `iter`, `batch_shape`, and `batch_size` operate on the batch axes only. `BatchSpec` is the batch kind's term spec (II.1), carrying the spec the elements satisfy together with the named multiplicity:

```python
class BatchSpec(TermSpec):         # the batch kind's spec; is_valid accepts a matching batch
    element_spec: TermSpec                          # the spec every element satisfies
    axis_groups: tuple[tuple[int | str, ...], ...]  # the multiplicity, tiled into levels below;
                                                    #   a str names a symbolic dimension (II.1)
    level_names: tuple[str, ...]
```

Construction checks every element against `element_spec` and reports the position that failed, since the batch asserts that spec of all of them.

**`[]` dispatch.** A key is either a **position** or a **name**; the two namespaces never collide, since an axis has no name and a field no position. A position (an integer, a slice, or a tuple of those) addresses the batch axes, which `Batch` itself handles. A name (a string, or a tuple of strings for a path) addresses a field within every element, and applies only to a batch whose elements have fields. A tuple mixing the two is invalid.

```python
class Batch[E](TrackedTerm):
    @property
    def spec(self) -> BatchSpec: ...
    @property
    def element_spec(self) -> TermSpec: ...              # view on spec: what every element satisfies
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_size(self) -> int: ...                    # total element count, prod(batch_shape)
    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]: ...   # batch_shape tiled into levels, outermost first
    @property
    def level_names(self) -> tuple[str, ...]: ...       # one name per level, aligned with axis_groups
    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename levels old -> new; shapes and elements unchanged, as with_path_names is for fields
    def __len__(self) -> int: ...                       # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...       # over the leading batch axis
    def __repr__(self) -> str: ...                      # the class, the name, and each level with its sizes
    def __getitem__(self, key: Any) -> Any: ...         # a position indexes the axes, a name the elements' fields
    def at_levels(self, /, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
    def raw(self) -> Any: ...
    # the storage view: the elements' raw values in their native stacked layout
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**: `axis_groups` tiles `batch_shape` into contiguous groups, outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, flat vectorization above all, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and positional `[]` address the leading **axis**, `batch_shape[0]`, not the leading level; the two coincide when the outermost level holds one axis, so iterating `(N,)` of `(S,)` runs over the `N` and yields each inner batch of `S` as a view. When the outermost level spans several axes, indexing drops the leading axis and leaves the level in place, one axis shorter.

**The batch's own type.** A batch stores its `BatchSpec`; `element_spec`, `axis_groups`, and `level_names` are views on it. `spec` is the batch's own type, not its element's, which keeps `OpaqueBatch` well typed although its elements carry no structure.

**The raw view.** `raw()` returns the **storage view**, the elements' raw values in their native stacked layout: one array for array-valued elements, an object array for callable- or opaque-valued ones, and the nested mapping of raw columns for record-valued ones.

**A polymorphic multiplicity.** An axis size may be a symbolic dimension name instead of an integer, as a `NumericArraySpec` shape entry may, so a *declaration* can fix the number of levels while deferring how many elements each holds: "returns a batch of `S` draws" before `S` is known. The names share one scope with the element's schema, so a batch of `("n",)` over arrays of shape `("n",)` is square by declaration. A live `Batch` may not be polymorphic, since it holds elements at positions; construction refuses a spec with free dimensions, and `batch_size` is undefined until they are bound.

**Level names.** Each level carries a name, listed in order by `level_names`. Names are unique within a batch and are identifiers, since `at_levels` addresses a level by keyword. An operation names the level it mints after itself, and a constructor such as `stack` takes the name to give it. A name already in use raises, as does a rename onto one, so the caller renames first or supplies another. `with_level_names` renames levels, shapes and elements unchanged, re-reading its own derived name under the new ones so that it and any view taken from it agree; renaming a *view* is refused when the new name belongs to a level the view derives its name from but no longer carries, and `with_name` renames it instead. Level names are the key by which operations align batched operands (V.10), and they are independent of the field names within an element.

**View identity.** A view of a batch, an element or a sub-batch, derives its name from the batch it was taken from and the positions it selects, naming the level each selection addresses. Take a batch named `posterior`, with a `chain` level of `(4,)` over a `draw` level of `(1000,)`:

```python
posterior.at_levels(chain=0).name           # "posterior[chain=0]"          — a sub-batch of draws
posterior.at_levels(chain=0, draw=7).name   # "posterior[chain=0, draw=7]"  — an element
posterior.at_levels(draw=slice(1, 3)).name  # "posterior[draw=1:3]"         — both levels kept
```

The derived name states what was selected rather than how it was reached. Levels selected whole are left out, so selecting all of a batch derives the batch's own name, and the levels that appear are listed in the batch's own order rather than the order they were indexed in; hence two routes to one selection read alike, and two different selections never do. Whether a batch *materializes* an element (a row of columnar storage that does not exist until it is built) or *stores* it outright, indexing returns a view (II.4); storage is invisible to access. A *sub-batch* is a view in the same way, being the batch's own selection rather than an object a caller put there.

**Selecting by level.** `at_levels(**levels)` indexes a batch along its named levels, the by-name counterpart of positional `[]`. (It is not `select`, the field-splatting selector on `Record`, and its name does not promise elements, which it returns only when the selection indexes down to one.) Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order: an integer drops its axis, and a slice or `None` keeps it. `None` stands for the whole axis here and only here, since a keyword cannot take a `:` literal and positional `[]` refuses `None`. A shorter tuple fills the leading axes and leaves the rest whole, so `draw=i` on a two-axis `draw` level means `draw=(i, None)`. A level whose axes are all dropped is removed, yielding the inner batch or element as positional indexing does; a level left unnamed is kept whole.

### Rationale

`Batch` satisfies `D1 – Mathematical fidelity` by keeping how many objects there are separate from what one object contains. Levels extend the same fidelity to collections of collections: how many objects sit at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis: a user can say which multiplicity is which, and operations align batches by meaning rather than by position. A suffixing rule for a taken name (`draw2`) was rejected on the same ground, since it would state the order the levels were added in, a fact about the computation path rather than about meaning. The `batch_*` names are kept rather than a numpy-style `.shape` / `size`, which could cover the batch axes, the per-element content, or both. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). Indexing and iteration yield views, the levels of a multi-level batch included, to satisfy `D6 – Single source of truth`; the obligation lands on the concrete batch, since only it knows its storage, and its sub-batch method shares the store wherever the storage affords it (`B4 – No copying at boundaries`). Hence selecting all of a batch needs no special case: it calls the same method as any other selection, and a view costs nothing there.

### Notes

- *Relation to xarray.* `at_levels` parallels `isel` in indexing by name and by position, with `level_names` in the role of dimension names; there is no label-based counterpart, since batch levels carry no coordinate labels, and a tuple here addresses a level's axes in order, where `isel` reads one as several positions of a single dimension.

## II.6 — `NamedTree`

### Contract

Structured values in ProbPipe are represented as named, ordered trees. The following terminology refers to components of these trees:
- A **field** is one named leaf — a single object in the collection.
- A **path** is a `/`-joined sequence which can address either a field or an interior node.
- A **key** is a path that specifically addresses a *field*. Every key is a path but a path for an interior node is not a key.
- A **child** of a node is an entry directly under that node.
- The **canonical order** of a tree is a depth-first walk visiting children in insertion order.

Naming, addressing, traversal, and structure-preserving transforms are defined once, in `NamedTree`. A `NamedTree` implements the `Mapping` interface with keys exactly its leaf paths, a leaf path written equivalently as a string (`"a/b"`) or a tuple (`("a", "b")`). Since interior nodes are not keys, `[]` raises on an interior path; interior nodes are reached through the one-level `children` view or through `at_path`, which reaches any leaf or subtree. So the invariants `x.children["a"].children["b"] == x.at_path("a", "b") == x.at_path("a/b")` hold, and navigation returns views. Sibling names are distinct, so every path identifies at most one node; distinct subtrees may reuse a name, as in `a/c` and `b/c`.

```python
class NamedTree[L]:
    # mapping interface — keyed by FIELD path
    def __getitem__(self, key: str | tuple[str, ...]) -> L: ...
    def __contains__(self, key: str | tuple[str, ...]) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...  # field paths
    def __len__(self) -> int: ...             # field count
    def keys(self) -> Iterable[str]: ...      # field paths, canonical order
    def values(self) -> Iterable[L]: ...
    def items(self) -> Iterable[tuple[str, L]]: ...

    # tree navigation — ranges over ALL paths
    def at_path(self, *path: str) -> L | Self: ...    # at_path("a", "b") == at_path("a/b")
    @property
    def children(self) -> Mapping[str, L | Self]: ...
    def is_field(self, path: str | tuple[str, ...]) -> bool: ...
    @property
    def is_multi_field(self) -> bool: ...   # True when the tree holds more than one field

    # structure-preserving transforms — return the same family
    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename or move nodes, old -> new; keys are paths, or bare names when unambiguous;
    # a new name may itself be a path, which moves the node there
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str | tuple[str, ...], leaf: L | Self) -> Self: ...
    def merge(self, other: Self) -> Self: ...              # a key collision raises
    def without(self, path: str | tuple[str, ...]) -> Self: ...

    # serialization (export; the constructor reads a nested mapping back)
    def to_nested_dict(self) -> dict[str, Any]: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type: every navigator and transform returns the same family
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

The parameter `L` declares the leaf type, the one axis on which families differ and what `values()`, `[]`, and `map` accept and return; interior nodes are always the family's own class. Implementations should check leaves against the declared leaf type at construction.

`with_path_names` renames the fields *within* a tree, which distinguishes it from `with_name`, the rename of the object itself (II.4). `at_path` has a level analogue in `Batch.at_levels` (II.5), and the two are alike: a path addresses a position and returns a leaf or a subtree, and named level indexers address positions and return an element or a sub-batch.

`with_path_names` renames or moves nodes by `old="new"` pairs. A key may be a bare name, resolving to the unique node so named and raising on ambiguity; keyword pairs cover the common case, and the positional mapping form addresses any path. A target may itself be a path: `with_path_names({"group/mu": "mu"})` promotes the field to the top level and `{"mu": "group/mu"}` demotes it under `group`, creating intermediate nodes as needed and dissolving an interior node a move empties, since a tree holds no empty subtrees. All substitutions apply simultaneously, so sources resolve against the original tree and swaps and simultaneous ancestor–descendant moves are well-defined. Every target is checked against the result: a collision raises, as a rename onto an existing sibling does, and so do a move into the moved node's own subtree and two targets where one is a prefix of the other. Ordering is deterministic: an in-place rename keeps its position, and a moved node appends at the end of its new parent's children.

### Rationale

Named paths satisfy `C5 – Naming for unambiguous meaning`. Housing the collection contract in one shared class ensures the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved (`C1 – Uniform interface to functions, distributions, and values`).

### Notes

- *Mappings are never leaves.* Construction should materialize a mapping-valued field into a subtree. The need for this mapping check is why `NamedTree` implements the `Mapping` interface without registering as a `collections.abc.Mapping`, ensuring a nested mapping and its `to_nested_dict` export round-trip through the constructor.

## II.7 — Dispatch and registries

### Contract

Some operations have many possible implementations, and which one applies depends on the *types* of the objects involved rather than on an object's own class. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **dispatch method** declares:
1. a unique `name`;
2. the types it applies to, via `supported_types`;
3. a `check` function that probes feasibility without significant computation and reports, as a `MethodInfo`, whether the call is feasible and at what fidelity;
4. an `execute` function that performs it;
5. a **fidelity** on one scale shared across the registries (exact, approximate, or sampled), declared where the method is registered and fixed for the method's life;
6. a **priority**, an integer rank *within* a fidelity tier and the one thing about a method a deployment may change at runtime.

Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in **selection order** and runs the first whose `check` reports feasible. Selection order is the same in every registry: fidelity first, exact above approximate above sampled, so exactness is never silently traded for anything below it; then priority within the tier, higher first; then specificity, the method whose declared types sit closest to the argument's class in method-resolution order; then registration order. A method whose priority is `None` is **opt-in-only**, skipped by auto-selection and reachable only by name. That is the default, so registering a method never silently changes what runs until a contributor ranks it. `set_priorities` re-ranks at runtime within a tier, never across fidelities, and warns when a method moves into or out of opt-in-only. A caller can bypass auto-selection with `method="..."`. New methods are added by registration at import, by whichever layer owns the implementation, so a registry gains its providers without importing them.

```python
class Fidelity(Enum):     # how exact an answer is; totally ordered, EXACT the highest
    EXACT       = "exact"
    APPROXIMATE = "approximate"
    SAMPLE      = "sample"

class BaseDispatchMethod(ABC):
    name: str
    fidelity: Fidelity            # declared at registration, fixed for the method's life
    priority: int | None = None   # rank within the tier, higher first; None is opt-in-only

    @abstractmethod
    def check(self, *args, **kwargs) -> MethodInfo: ...
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any: ...

class UnaryDispatchMethod(BaseDispatchMethod):    # still abstract
    @abstractmethod
    def supported_types(self) -> tuple[type, ...]: ...
class BinaryDispatchMethod(BaseDispatchMethod):   # still abstract
    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (left, right) types

class MethodInfo:
    feasible:    bool
    method_name: str
    description: str
    fidelity:    Fidelity | None   # what this call would achieve; None when infeasible

class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    # the public interface is concrete; arity subclasses supply key extraction and matching
    def register(self, method: M) -> None: ...
    def set_priorities(self, **priorities: int | None) -> None: ...   # within a tier only; warns on a move into or out of opt-in-only
    def execute(self, *args, method: str | None = None, **kwargs) -> Any: ...   # auto-select, or run the named method
    def check(self, *args, method: str | None = None, **kwargs) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                           # names, in selection order

class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]): ...  # keys on the first two
```

A single **catalog** makes every registry discoverable: it lists the registries, their entries with their priorities, and a one-line description each, so a user can see which entries exist and how a call will resolve. An **entry** is one registered item within a registry; the term is generic because the catalog spans registries whose items are not all type-dispatched methods. A registry can be cataloged if it implements `SupportsRegistryCataloging`; satisfying the protocol is structural, and membership requires an explicit `register`. The operation vocabulary is cataloged the same way, so what ProbPipe can do and how a given call resolves are answered from one place.

```python
@dataclass(frozen=True)
class EntrySummary:
    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    @property
    def is_opt_in_only(self) -> bool: ...

@dataclass(frozen=True)
class RegistryInfo:            # the catalog's per-registry record
    name: str                  # unique within the catalog
    description: str           # one line
    kind: str                  # e.g., "dispatch", "factory", "converter"
    entry_count: int

@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    name: str
    description: str
    kind: str
    def entry_summaries(self) -> list[EntrySummary]: ...
    def describe_entry(self, name: str) -> EntrySummary: ...

class RegistryCatalog:
    def register(self, registry: SupportsRegistryCataloging) -> None: ...   # empty or duplicate names raise
    def names(self) -> list[str]: ...
    def list(self) -> list[RegistryInfo]: ...
    def __getitem__(self, name: str) -> SupportsRegistryCataloging: ...     # get the registry
    def __contains__(self, name: object) -> bool: ...                       # check if a registry exists
    def describe(self, name: str) -> str: ...                               # a readable summary of one registry
registry_catalog = RegistryCatalog()  # the global instance
```

### Rationale

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` apply to operations whose implementation cannot be chosen from a single object alone. The `check` probe keeps auto-selection safe, and `method="..."` leaves the choice in the user's hands. New implementations join by registration, so the supported set grows without touching the call sites (`D2 – Generality first`). One catalog over every registry is `D6 – Single source of truth`: one place to see which implementations exist and how a call resolves.
