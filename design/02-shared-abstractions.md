# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on: generic, type-agnostic machinery, one piece at a time, in dependency order.

## II.0 — Overview: the shared abstractions

The shared abstractions, in dependency order. Each is generic and type-agnostic, defined once and reused throughout the library:

| § | Layer | Abstraction | Role |
|---|---|---|---|
| II.1 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.2 | Identity | `Tracked` / `Annotated` / `Provenance` | The name, lineage, and annotations an object carries beyond its mathematical content. |
| II.3 | Multiplicity | `Batch` | The generic multiplicity axis: an indexed collection of *separate* objects, distinct from one object over a structured space. |
| II.4 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved. The shared mechanism behind converters, inference selection, and bijector factories. |

## II.1 — `NamedTree`

### Contract

Values in ProbPipe are represented as named, ordered trees.  ProbPipe uses the following standardized **terminology** to refer to components of these trees:
- A **field** is one named leaf — a single object in the collection.
- A **path** is a `/`-joined sequence which can address either a field or an interior node.
- A **key** is a path that specifically addresses a *field*. Every key is a path but a path for an interior node is not a key.
- A **child** of a node is an entry directly under that node.
- The **canonical order** of a tree is a depth-first walk visiting children in insertion order.

Naming, addressing, traversal, and structure-preserving transforms are defined once in the `NamedTree` class. Specially, a `NamedTree` is a `Mapping` whose keys are exactly its leaf paths, in canonical order, so `keys()`, `values()`, `items()`, `len`, `in`, and `[]` all agree on that one key set, as for a plain `dict`. A leaf path may be equivalently written as a string (`"a/b"`) or a tuple (`("a", "b")`). Since interior nodes are *not* keys, `[]` raises on an interior path. Interior nodes are instead accessed via either the one-level `children` view or `at_path`, which can access any leaf or subtree. So, for example, the invariants  `x.children["a"].children["b"] == x.at_path("a", "b") == x.at_path("a/b")` hold. Sibling names are distinct, so every path identifies at most one node. Distinct subtrees may reuse a name, as in `a/c` and `b/c`, and a bare name is then ambiguous on its own.

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
    # rename nodes, old -> new; keys are paths, or bare names when unambiguous
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str | tuple[str, ...], leaf: L | Self) -> Self: ...
    def merge(self, other: Self) -> Self: ...              # a key collision raises
    def without(self, path: str | tuple[str, ...]) -> Self: ...

    # (de)serialization
    def to_nested_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_nested_dict(cls, d: Mapping[str, Any]) -> Self: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

### Rationale

Using named paths is necessary to satisfy `C5 – Naming for unambiguous meaning`. Housing the collection contract in one shared class ensures `D7 – Single source of truth` is satisfied — the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved.
### Notes

- *Same-family closure.* Every navigator and transform returns the same family, enforced by `_node_type()`.
- *Leaf type versus node type.* The parameter `L` declares the leaf type, the only axis on which tree families differ: it is what `values()`, `[]`, and `map` traffic in. The node type is not a second parameter, since interior nodes are always instances of the family's own class, which is what `_node_type()` reports. The runtime partition therefore uses the node type alone: a field value is an interior node when it is an instance of `_node_type()`, and a leaf otherwise. Validation is handled once in `NamedTree` rather than by each family: construction checks every leaf against the family's declared leaf type, reported by `_leaf_type()`, so a malformed tree fails at construction rather than at first navigation. A family whose leaves are arbitrary values declares `object`, making the check vacuous.
- *Navigation yields views.* `children`, `at_path`, and `[]` return a subtree or leaf that is a *view* into the same underlying store, derived on demand rather than copied out.
- *Mappings are never leaves.* Construction rejects a mapping-valued leaf, and `from_nested_dict` reads every mapping as a subtree, so the two agree and serialization round-trips faithfully.
- *Bare-name reference.* `with_path_names` renames by `old="new"` pairs, where a key may be a bare name instead of a full path: a bare name resolves to the unique node so named and raises when the tree contains it more than once. Keyword pairs therefore cover the common case, while the positional mapping form addresses any path, as in `with_path_names({"group1/mu": "loc"})`. The `Mapping` interface itself is untouched, with `[]` and `keys()` keyed by full path.

## II.2 — Identity & metadata: `Tracked`, `Annotated`, `Provenance`

### Contract

Identity & metadata is the cross-cutting layer that lets any object carry, alongside its mathematical content, three things: a **name** (what the object is called), a **provenance** (how it was produced), and free-form **annotations** (auxiliary information supplied by the user or an algorithm). The structure for identity and metadata is provided by two mixins: `Tracked` (name + provenance) and `Annotated` (annotations). Every first-class object, the kind an operation consumes and produces, must be `Tracked`, while structural helpers such as templates and specs are not. We call any such object a **tracked term**: a value, distribution, conditional distribution, linear operator, or batch that carries a name and provenance.

Annotation metadata is a free-form mapping:

```python
class Annotated:
    annotations: Mapping[str, Any] | None
```

A tracked term's name must be provided by the user when constructed explicitly (as the required first argument to the constructor). When an operation produces an object, it must provide a meaningful, deterministic name derived from its inputs. The `name_is_auto` flag records which, because the two behave differently: an auto-derived name may need to be updated when the object is combined into a larger one, while a user-given name is preserved. A *nested* object (i.e., a sub-object of a `NamedTree`) takes its name from the field key it sits under. For example, a sub-object reached at `parameters` is itself named `parameters`. `with_name` renames the object itself, unlike `NamedTree.with_path_names`, which renames the fields within it. A name is a human label rather than an identity: nothing resolves an object by name, derived names need no escaping scheme, and two objects may share a name wherever field uniqueness does not force distinctness. The provenance of a tracked term stores pointers to descriptors of the parent objects that created it, along with the operation. Optionally, it can also provide references to the parents themselves.

```python
class Tracked:
    name:         str
    name_is_auto: bool
    provenance:   Provenance | None              # write-once via with_provenance(...)
    def with_name(self, name: str) -> Self: ...  # shallow copy with name_is_auto = False
    def with_provenance(self, p: Provenance) -> Self: ...

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

A provenance records everything its operation resolved: the tracked parents, the plain arguments, and the **controls** the run actually used, the PRNG key, sample counts, and any selected method among them, so a result can be reproduced from its record. Fingerprints are best-effort, in tiers: a content hash for arrays and records, the qualified name and code hash for a closure-free callable, and object identity otherwise, with `fingerprint_is_weak` marking that last tier.

### Rationale

`Tracked` serves the two non-mathematical principles: `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. The guarantee behind `C6 – Traceable and reproducible workflows` is a single rule: **every object a ProbPipe operation natively returns is a tracked term** (whether or not it is also `Annotated`), so the provenance chain is never broken. Recording the resolved controls, not just the parents, is what turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`). Because identity and metadata are orthogonal to *what* an object is mathematically, they are defined uniformly across classes.

## II.3 — `Batch`

### Contract

A `Batch` is the generic `Tracked` nd array of shape `batch_shape` that holds objects of a common element type. `batch_shape` is nonempty: a batch has at least one batch axis. It could also be `Annotated` if applications for it arise. A concrete batch implementation must specify how to store the elements. Since a batch is a *collection* of its elements, `len` / `iter` / `batch_shape` / `batch_size` operate only on the batch axes. The `batch_*` names are kept deliberately rather than a numpy-style `.shape` / `size`, which could ambiguously cover both the batch axes and the per-element content. A concrete batch implementation adds whatever its element type affords in that element's own section — including, where useful, indexing into the elements' fields, since `[]` dispatches unambiguously on the key type.

```python
class Batch[E](Tracked):
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_size(self) -> int: ...                    # total element count, prod(batch_shape)
    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]: ...   # batch_shape tiled into levels, outermost first
    @property
    def level_names(self) -> tuple[str, ...]: ...       # one name per level, aligned with axis_groups
    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # shallow copy renaming levels old -> new; shapes and elements unchanged, as with_path_names is for fields
    def __len__(self) -> int: ...                       # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...       # over the leading batch axis
    def __getitem__(self, index: Any) -> E | Self: ...  # returns a view of either an element or a sub-batch
    def select(self, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**. `axis_groups` tiles `batch_shape` into contiguous groups, outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, flat vectorization above all, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and indexing operate on the outermost level, and an element of a multi-level batch is the inner-level batch, as a view. Nesting needs no dedicated classes: a batch is itself a tracked term, so a batch whose elements are batches is already admitted, and grouped storage presents the levels as views into one store.

**Level names.** Each level carries a name, listed in order by `level_names`. A name is auto-derived by the operation that produces the level, and `with_level_names` repins it, the batch-level counterpart of `with_path_names`: a shallow copy changing only the names. Names are unique within a batch. An operation that would mint a duplicate appends the smallest free integer suffix, so nested sampling yields `draw`, then `draw2`, while a `with_level_names` rename onto an existing name raises, as field and factor renames do. The names are what operations align on: levels match by name rather than by position, and two levels meant to correspond under different names are lined up by renaming one first, exactly as fields are realigned for composition. Level names are independent of the field names within each element, so a `RecordBatch` may carry a `draw` level over records with fields `mu` and `sigma`.

**Element identity.** An element view of a batch derives its name as `name[i]` from the batch's own name, marked `name_is_auto`, with provenance recording the indexing, and the elements of nested levels compose the scheme, as in `name[i][j]`. A batch whose elements are bare values yields bare elements, which carry no identity to derive.

**Selecting by level.** `select(**levels)` indexes a batch along its named levels and returns a view, the by-name counterpart of positional `[]`. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order, where an integer drops its axis and a slice or `None` keeps it, `None` meaning the whole axis as `:` does. A level spanning several axes takes one indexer per axis, and a shorter indexer fills the leading axes and leaves the rest whole, so a scalar `draw=i` on a two-axis `draw` level means `draw=(i, None)`. Selecting an entire single-axis level by an integer removes it, yielding the inner batch or element just as positional indexing and iteration do, while a level left unnamed is kept whole. This parallels xarray's `isel`, with `level_names` in the role of xarray dimension names; there is no label-based counterpart, since batch levels carry no coordinate labels.

### Rationale

`Batch` is necessary to satisfy `D1 – Mathematical fidelity` by ensuring how many objects there are stays separate from what one object contains. The level structure extends the same fidelity to collections of collections: how many objects sit at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis, letting a user say which multiplicity is which and letting operations align batches by meaning rather than by position. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). When those elements are array-backed the mapping should be vectorized and differentiable (`D6 – Differentiability where possible`). To satisfy `D7 – Single source of truth`, indexing or iterating yields a *view*, the levels of a multi-level batch included.

## II.4 — Dispatch and registries

### Contract

Some operations have many possible implementations, and which one applies depends on the *types* of the objects involved rather than on an object's own class. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **dispatch method** declares a unique `name`, the types it applies to via `supported_types`, a `check` function that probes feasibility without doing significant computation, an `execute` function that performs it, and a `priority` that orders auto-selection. Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in priority order and runs the first whose `check` reports feasible. Within one priority, the method whose declared types sit closest to the argument's class in method-resolution order wins, and any remaining tie falls to registration order. A caller can bypass auto-selection and name a method with `method="..."`. New methods are added by registration.

```python
class BaseDispatchMethod(ABC):
    name: str
    priority: int   # defaults to opt-in-only (0)

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

class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    # the public interface is concrete; arity subclasses supply key extraction and matching
    def register(self, method: M) -> None: ...
    def execute(self, *args, method: str | None = None, **kwargs) -> Any: ...   # auto-select, or run the named method
    def check(self, *args, method: str | None = None, **kwargs) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                           # names, in selection order

class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]): ...  # keys on the first two
```

A single **catalog** makes every registry discoverable. It provides a list of registries, their entries with their priorities, and a one-line description, so a user can see which entries exist and how a given call will resolve. An **entry** is one registered item within a registry; the catalog uses this generic term rather than *method* because it spans registries whose items are not all type-dispatched methods. A registry can be added to the catalog if it implements `SupportsRegistryCataloging`. Satisfying the protocol is structural, while membership requires an explicit `register`.

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

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` reach operations whose implementation cannot be chosen from a single object alone. The `check` probe keeps auto-selection safe, while `method="..."` leaves the choice in the user's hands. New implementations join by registering a method, so the supported set grows without touching the call sites that use it, which is `D2 – Generality first`. Gathering every registry under one catalog serves `D7 – Single source of truth`: there is one place to see which implementations exist and how a call resolves.
