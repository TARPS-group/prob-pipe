# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on.

| § | Category | Abstraction | Role |
|---|---|---|---|
| II.1 | Typing | `TermSpec` | The term-specification base every field, input, and output is typed by, with the symbolic-dimension protocol, conformance, and unification. |
| II.2 | Inputs and outputs | `InputSpec` / `OutputSpec` | The input specs and output specs of every kind: the inputs a term receives values at, and the outputs other terms receive values from. |
| II.3 | Numeric values | `Numeric` / `NumericSpec` / `Constraint` | The flat-vector interface the numeric kinds share, its spec-side mixin, and the elementwise support constraint. |
| II.4 | Identity | `TrackedTerm` / `Provenance` | The label, type (spec), lineage, and annotations an object carries beyond its raw representation, with `raw` as its access to that representation. |
| II.5 | Multiplicity | `Batch` | An indexed collection of *separate* objects, with its `BatchSpec`. |
| II.6 | Structure | `NamedTree` | Abstract named, ordered tree addressed by path with leaf-keyed mapping contract and navigation. |
| II.7 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved, and the catalog that makes every registry discoverable. |

## II.1 — Term specifications: `TermSpec`

### Contract

A **term specification** ("term spec") describes the typing information available for a mathematical term and validates whether an object satisfies it.

**Base and batch kinds.** The specs partition into the **base kinds** and the **batch kinds**. Every base kind has exactly one term spec, one **base form**, and one **batch form**. Since a batch of batches is a batch, the base form and batch form of a batch are identical. The correspondence is recorded in the **kind table**: `register_kind` records the tracked class and the batch form of a spec class, and `term_class_for_spec` and `batch_class_for_spec` look them up through the spec's class and its bases, so a spec subclass inherits its base's kind unless it registers its own. A second registration that disagrees raises.

**Symbolic dimensions.** A dimension size for a numeric value spec may be an integer or a **named symbolic dimension**. A dimension name is a Python identifier, such as `n_obs`, so that a call such as `with_dim_sizes(n_obs=100)` can name it; a reserved word such as `class` is an identifier too, and is named in a mapping, as `with_dim_sizes(**{"class": 100})`. A spec with any symbolic dimensions is **polymorphic**; one with none is **concrete**. A spec can *report* the names still unbound, *substitute* explicit sizes for names, *rename* a dimension, and *bind* names by checking that a value, or another spec, conforms to it (below), reading the sizes off the value's spec. In binding, a name takes its size from its first occurrence, and a later occurrence that disagrees raises. Two different unbound names compared at one axis are refused, since nothing says whether their sizes are equal. Names form one **dimension scope** wherever specs meet: within a schema (III.5), between a batch and its element (II.5), across the inputs of one map (V.6), and across the operands of a merge (IV.2). Two dimensions that are different quantities are renamed apart before they meet.

**Conformance and unification.** Two relations between specs decide whether a value may be supplied where a spec is declared, each within one dimension scope:
1. a value **conforms** to a spec when it has the structure the spec describes, and a spec `A` conforms to a spec `B` when every value that conforms to `A` conforms to `B`;
2. two specs **unify** when the rules below give a spec, their **unification**, that conforms to both; a value conforms to the unification exactly when it conforms to both specs.

Conformance is the one-way check a declared spec makes of what it receives, such as an input of the value supplied to it or an output spec of a returned value. Unification is the symmetric check where one value must satisfy two declared specs at once, as an input that both operands of a merge take must (IV.2). The rules, by kind:
1. **arrays:** equal ranks, each axis agreeing by the dimension scope; for conformance, `A`'s dtype casts into `B`'s by NumPy's same-kind rule when both set one, and for unification the narrower dtype is taken, so `int32` and `float32` unify to `int32`; a support describes the values and is never checked, and a unification keeps a support only when both set the same one;
2. **opaque values:** `B`'s Python type is unset or a superclass of `A`'s, and `B`'s `meta` is unset or equal to `A`'s; two opaque specs unify when their types are equal or one is unset, and likewise their `meta` (III.2);
3. **records:** the same child names in the same order, child by child; a field's order is part of a `RecordSpec`, so `RecordSpec(x=(), y=())` and `RecordSpec(y=(), x=())` neither conform to nor unify with each other (III.5);
4. **distributions:** their event specs (II.2);
5. **conditional distributions and functions:** the same input names and optional inputs; for conformance, each input's spec in `B` conforms to that input's spec in `A`, since a map that accepts more conforms, and `A`'s event or return spec conforms to `B`'s; a unification requires equal input specs; a `FunctionSpec` whose inputs or return value `B` leaves undeclared accepts any `A` in that respect;
6. **batches:** the same levels, and their element specs.

A spec that sets no dtype or support conforms to and unifies with any spec of its shape. A value is checked as its spec is, so a distribution, a function, or a batch conforms when its own spec does; a raw mapping, such as one a function returns or one given for an input, is first laid out in the declared order and then checked (III.5).

The base API is validation plus the dimension protocol:

```python
class TermSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...      # structural validity such as kind, rank, dtype

    # the symbolic-dimension protocol: report, substitute, rename, bind
    @property
    def free_dims(self) -> frozenset[str]: ...       # report: the unbound symbolic dimensions
    @property
    def is_concrete(self) -> bool: ...               # True when free_dims is empty
    def with_dim_sizes(self, **sizes: int) -> Self: ...   # substitute explicit sizes
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions, old=new
    def bind_dims_from_value(self, value: Any) -> Self: ...   # bind by checking that value conforms
    def bind_dims_from_spec(self, other: TermSpec) -> Self: ...   # bind by checking that other conforms

def register_kind(spec_type: type[TermSpec], *, term_class: type, batch_class: type) -> None: ...
def term_class_for_spec(spec: TermSpec) -> type: ...    # the tracked class of the spec's kind
def batch_class_for_spec(spec: TermSpec) -> type: ...   # its batch form
```

**Shape arguments.** An argument that takes a shape, such as `NumericArraySpec`'s `shape`, a level of a `BatchSpec` (II.5), or `sample_shape` (VI.3), takes a single int or str as one dimension and a sequence, such as a tuple, a list, a `range`, or a 1-D array, as one dimension per item, and it stores the tuple. So `3` is `(3,)` and `"n"` is `("n",)`, as NumPy reads `np.zeros(3)`. An iterator such as a generator, a set, `bytes`, a `memoryview`, and a mapping are refused: an iterator is used up by one reading, a set has no order, and the others iterate into byte values or keys. An integer of another type, such as `numpy.int64`, is stored as a Python `int`, a name of a `str` subclass is stored as a `str`, and a `bool` is refused. A shape that sizes draws or an array, such as `sample_shape`, takes integers only. Two arguments that hold sizes are not shape arguments: `with_dim_sizes` takes one non-negative int per name, and the size shorthand below takes a tuple only.

**The size shorthand.** Wherever an input of an `InputSpec`, a field of a `RecordSpec`, or the keyword form of an `OutputSpec` (II.2) takes a spec, a tuple of sizes is the `NumericArraySpec` of that shape, with no dtype or support: `InputSpec(x=())`, `RecordSpec(x=("n",))`, and `OutputSpec(y=(3,))`. A bare int, as `x=3`, is refused, since it would read as a size where a spec is expected.

`with_dim_sizes` substitutes supplied sizes and leaves other dimensions symbolic. `with_dim_names` renames simultaneously. Binding reads concrete sizes in one shared scope and rejects disagreements; unobserved dimensions remain symbolic.

### Rationale

One `is_valid` contract across the kinds keeps validation uniform (`C1 – Uniform interface to functions, distributions, and values`), and defining each concrete spec beside the kind it describes keeps this layer generic (`D2 – Generality first`). Naming the base for the terms it types is `C5 – Naming for unambiguous meaning` applied to the library's own vocabulary: every spec types a tracked term, and *value* stays reserved for the mathematical kind. The kind rule is `D2 – Generality first`: every result can be tracked and every collection of draws stacked, so everything an operation produces lies within the system. A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it; hence cross-field equalities travel with the term, and sizes bind when the value that fixes them appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`). Two relations, one for receiving a value and one for a value received twice, keep each check to its own question: conformance asks whether what arrives fits what was declared, and unification asks whether one value can fit two declarations (`D1 – Mathematical fidelity`). Field order counts in both because it fixes how a numeric record is laid out as one vector (II.3), so a record that conforms always flattens in the declared order. Declaring the kind correspondence once, where each kind is defined, is `D6 – Single source of truth`: the wrap, the declaration enforcement, and the batch constructors read one table, so widening a kind widens all of them.

## II.2 — Input and output specs: `InputSpec`, `OutputSpec`

### Contract

A term receives values at its **inputs**, and other terms receive values from it at its **outputs**. An `InputSpec`, an **input spec**, states a term's inputs, and an `OutputSpec`, an **output spec**, states its outputs. Each kind holds them under its own attributes:

| Kind | Input spec | Output spec |
|---|---|---|
| `Function` | `input_spec`, its parameters | `output_spec`, its return value |
| `ConditionalDistribution` | `given_spec`, what it is conditioned on (III.9) | `event_spec`, its **event spec** |
| `Distribution` | none | `event_spec` |
| `Batch` | its elements' input spec | `element_spec`, the output spec of one element (II.5) |

A record and a base value have no input spec, and a record's output spec is derived from its own spec, as `OutputSpec(record.spec)`.

**Inputs.** An input is a named place where a function or a conditional distribution receives one value. Each input has:
1. a name, which is a Python identifier, since binding matches it to a parameter;
2. a term spec, to which every value supplied there must conform (II.1);
3. whether it is optional: a binding may omit an optional input, and the term then uses its default, while a required input must be supplied.

An `InputSpec` lists a term's inputs in order. It holds no defaults, which belong to the term, and two input specs are equal when they have the same inputs, specs, and optional inputs, in any order, since inputs bind by name. An input whose spec is a `RecordSpec` is one input, which receives one whole record.

```python
class InputSpec(Mapping[str, TermSpec]):   # inputs in order; names are Python identifiers
    def __init__(self, inputs: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None, /,
                 **inputs_by_name: TermSpec | tuple) -> None: ...
    @property
    def required(self) -> tuple[str, ...]: ...   # the inputs a binding supplies, in order
    @property
    def optional(self) -> frozenset[str]: ...    # the inputs a binding may omit
    def with_optional(self, *names: str) -> InputSpec: ...
    def __getitem__(self, names: str | Sequence[str], /) -> TermSpec | InputSpec: ...
    def __contains__(self, name: object, /) -> bool: ...   # whether self[name] succeeds; False for a list
    def get(self, name: str, default: TermSpec | None = None, /) -> TermSpec | None: ...
    def without(self, *names: str) -> InputSpec: ...       # a name that is not an input raises KeyError
    def __and__(self, other: InputSpec, /) -> InputSpec: ...
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> InputSpec: ...
```

**Reading inputs.** The constructor reads its argument from one of four sources, and a `RecordSpec` and an `OutputSpec` are recognized before a plain mapping:
1. an `InputSpec`, taken as it is;
2. a mapping, one input per entry;
3. a `RecordSpec`, one input per top-level name, so a group is one input;
4. an `OutputSpec`, one input per output, typed by the output's spec, so `InputSpec(d.event_spec)` declares the inputs that `d`'s outputs can supply.

Every input read from a mapping, a `RecordSpec`, or an `OutputSpec` is required, and `with_optional` marks optional inputs. Keywords give the inputs by name instead, with the size shorthand of II.1, as `InputSpec(mu=(), sigma=())`. An input name that is not an identifier raises `ValueError`. A source of another class, an `OutputSpec` whose spec is unspecified (below), and an argument together with keywords raise `TypeError`. `conditional_distribution(fn, given_spec=...)`, `ConditionalDistributionSpec(given_spec, ...)`, and `Function(fn, input_spec=...)` read their argument through this constructor. They then mark optional each input whose parameter has a default, and refuse with `ValueError` an optional input whose parameter has none. Only named parameters are inputs, never `*args` or `**kwargs`.

**Operations on input specs.** Inputs are names, and a path continues only into an input whose spec is a `RecordSpec`, so `[]` on an input spec gives a spec:
1. `i["x"]` is the spec of the input `x`, and `i["g/b"]` the spec at the field `b` of the record input `g`;
2. `i[["x", "z"]]` is the `InputSpec` of those inputs, in this spec's order, optional ones still optional; a path in the list keeps its input with only the named fields, and that input becomes required, since a default is one whole value;
3. `name in i` holds exactly when `i[name]` succeeds, and `get` returns a default instead of raising.

`without(*names)` removes inputs, or the fields of a record input that paths name, under the same rule, and it may leave no inputs. `i & j` merges two input specs: this spec's inputs, then those of `j` not already present, and an input both take is one input, whose spec is the unification of its two specs (II.1) and which is optional only if optional in both. The dimension sizes the unification binds are applied and not kept. `&` refuses an operand that is not an `InputSpec`, as a mixed `InputSpec & OutputSpec` is refused with a message that names `InputSpec(output_spec)`, and raises `ValueError` when the two specs of a shared input do not unify or a dimension name would take two sizes. `rename` renames inputs by II.6's rules restricted to names, keeping their order. The selection, the removal, and the reader of paths are those of II.6.

```python
InputSpec(mu=(), sigma=()).with_optional("sigma")     # mu required, sigma optional
InputSpec(RecordSpec(g=RecordSpec(a=(), b=()), x=()))  # two inputs: g, a record, and x
d = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)
InputSpec(d.event_spec)                                # InputSpec(mu=..., sigma=...)
InputSpec(theta=("n",), x=()) & InputSpec(theta=(3,), y=())
# InputSpec(theta=NumericArraySpec((3,)), x=..., y=...): theta unifies to (3,)
```

**The value type.** A term's outputs, and its paths (II.6), are read from one spec, its **value type**:
1. a `Distribution`'s or a `ConditionalDistribution`'s event spec's `term_spec`, the spec of the values it is over;
2. a `Function`'s output spec's `term_spec`, the spec of what it returns;
3. a `Batch`'s element spec's `term_spec`, the spec of one element;
4. a record's or a base value's own `spec`.

The value type has no attribute of its own, since these attributes give it. A function whose return value is undeclared has none, so its outputs and paths are not known before it is called. A distribution, its draws, its mean, and a batch of its draws share one value type, so a path names one quantity in all of them (II.6).

**Outputs.** An output is a name under which other terms receive one value from a term: when two terms merge (IV.2), an output of one supplies the input of the same name of the other. An output is therefore one name for one whole value, as an input is, and never a nested path. The outputs follow from the value type, by the class of its spec:
1. a `RecordSpec` has one output per child, a field or a group, named and typed by that child;
2. a base spec, a `NumericArraySpec` or an `OpaqueSpec`, has one output, the **output at the root**, typed by the base spec, whose name the output spec states, as `Normal("x", 0.0, 1.0)` states `x`;
3. a `FunctionSpec` has the outputs of its return value, each one a function with the same inputs;
4. a `BatchSpec` has the outputs of its element spec, each typed by its marginal batch's `BatchSpec`, the batch's levels over the element's output (II.5);
5. a `DistributionSpec` or a `ConditionalDistributionSpec`, which is the value type of a function that returns a distribution or of a batch of distributions, has no outputs.

A base value standing alone has no outputs, since nothing states a name for it, and its label is not such a name. Only a base value takes a name stated outside it, since a record's children already have names: a record named from outside would give each child two paths, `mu` in a draw and `params/mu` in the distribution. A distribution as a value has no outputs because its marginals do not determine it, so passing them on separately would lose their dependence. To pass a distribution on by name, wrap it in a record under that name, `OutputSpec(RecordSpec(params=d.spec))`.

Outputs are not chosen: the value type fixes them, and the one name it does not fix is that of an output at the root. A term with an output at the root has no paths, so the name of an output at the root never collides with a path. A distribution over `{a, g: {c, e}}` has the outputs `a` and `g`, while `g/c` is a path of it and not an output; renaming `g/c` to `c` makes it one.

```python
class OutputSpec:
    @overload
    def __init__(self, term_spec: TermSpec, /) -> None: ...   # the positional form, for any spec but a base spec
    @overload
    def __init__(self, **named_output: TermSpec | tuple | None) -> None: ...   # the keyword form: one name over a base spec
    @property
    def term_spec(self) -> TermSpec | None: ...   # the value type; None while it is unspecified
    @property
    def outputs(self) -> Mapping[str, TermSpec | None]: ...   # each output's name and spec, in order
    def __getitem__(self, paths: str | Sequence[str], /) -> OutputSpec: ...
    def __contains__(self, path: object, /) -> bool: ...   # whether self[path] succeeds
    def __iter__(self) -> NoReturn: ...   # raises TypeError: an output spec is indexed by path; .outputs lists the outputs
    def without(self, *paths: str) -> OutputSpec: ...
    def __and__(self, other: OutputSpec, /) -> OutputSpec: ...
    def with_term_spec(self, term_spec: TermSpec, /) -> OutputSpec: ...
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> OutputSpec: ...
    @classmethod
    def default(cls, term_spec: TermSpec, *, output_name: str) -> OutputSpec: ...
```

**The two forms.** The value type decides the form, so an output spec carries no flag:
1. the **keyword form** `OutputSpec(y=spec)` names the output at the root of a base spec, with the size shorthand of II.1, as `OutputSpec(y=(3,))`;
2. the **positional form** `OutputSpec(term_spec)` takes any other spec.

The positional form refuses a base spec with `TypeError`, since its output needs a name, and refuses an empty `RecordSpec` with `ValueError`, since it would have no outputs to name. The keyword form refuses any other spec with `TypeError`, as for `OutputSpec(params=RecordSpec(mu=(), sigma=()))`: "cannot use the keyword params for a RecordSpec: a keyword of OutputSpec must name a single array or opaque value. Write OutputSpec(RecordSpec(mu=..., sigma=...)) for the outputs mu and sigma, or OutputSpec(RecordSpec(params=RecordSpec(mu=..., sigma=...))) for one output params". For a distribution's spec it reads "cannot use the keyword G for a distribution's spec: a keyword of OutputSpec must name a single array or opaque value. Write OutputSpec(RecordSpec(G=law_spec)) to pass the distribution on under G". A name is any non-empty string without `/`, since a merge makes it the name of a field; an output whose name is not an identifier supplies no input.

**Operations on output specs.** Paths lead into the value type, so `[]` on an output spec gives the output spec there:
1. `outputs` maps each output's name to its spec, in order;
2. `O["g/b"]` is the output spec of the node at that path: the keyword form named by the path's last name at a base node, as `OutputSpec(b=...)`; the positional form over a group, whose children are then its outputs; and the positional form at or below a field that holds a law, a kernel, a function, or a batch, so a law-valued field's output spec has no outputs;
3. `O[["a", "g/b"]]` is the positional form over the record of the chosen nodes, each at its own path, in this spec's order; a list may name the output at the root, so `OutputSpec(x=())[["x"]]` is `OutputSpec(RecordSpec(x=...))`;
4. `path in O` holds exactly when `O[path]` succeeds.

The name of an output at the root alone is not a path, so `OutputSpec(x=())["x"]` raises `KeyError`: `[]` keeps one meaning, and a term's `output("x")` returns the term itself (II.5, III.7). An output spec is not a mapping, and iterating it raises `TypeError`: "cannot iterate over an OutputSpec. Iterate over its outputs instead, as list(spec.outputs)". `without(*paths)` removes the nodes at paths, and a group left empty goes too; removing every node raises `ValueError`. `O & P` merges two output specs: the outputs of `O`, then those of `P`, as one record, where an output at the root `y` becomes the field `y`. It refuses an operand whose spec is unspecified or is not a record or a base spec, and an output name in both, and a dimension name that would take two sizes. `rename` renames or moves nodes by path under II.6's rules, or renames the output at the root, as `{"x": "y"}`; a root name cannot become a nested path, since a base value has no record to hold it. Selection, removal, and the reader of paths are those of II.6, and dimension names keep their scope through selection, so binding a size commutes with selecting.

```python
O = OutputSpec(RecordSpec(a=(), g=RecordSpec(b=(), c=())))
list(O.outputs)          # ["a", "g"]
O["g/b"]                 # OutputSpec(b=NumericArraySpec(()))
O["g"]                   # OutputSpec(RecordSpec(b=..., c=...)): the outputs b and c
O[["g"]]                 # OutputSpec(RecordSpec(g=RecordSpec(b=..., c=...))): the output g
O.without("g/c")         # OutputSpec(RecordSpec(a=..., g=RecordSpec(b=...)))
OutputSpec(x=()) & O     # OutputSpec(RecordSpec(x=..., a=..., g=...))
OutputSpec(x=())["x"]    # KeyError: x names the value at the root
```

**Completion.** A term that produces a value knows the spec of what it produced, and `with_term_spec` **completes** the declared output spec with it. The declared spec is kept with its dtype and support. Its dimension names are bound from the produced spec, and the produced value must conform to it (II.1); a produced spec that does not conform raises. A function given no output spec uses `OutputSpec.default` under its callable's name (III.3), and a distribution's constructor reads its event spec by III.7's rule.

**An output spec whose spec is unspecified.** The keyword form alone may leave the spec unspecified, written `None`: `OutputSpec(mean=None)` names the output `mean` before its spec is known, as a function's output is before its first call. `with_term_spec` completes it once the produced value is known:
1. with a base spec, the output keeps the name `mean`;
2. with a batch whose elements carry only the default name `element` (II.5), the elements take the name `mean`;
3. with any other spec, it raises `TypeError`, asking for a record to be declared instead.

Until it is completed, only a spec that could complete it conforms to it or unifies with it, which is a base spec or a batch of base values. `InputSpec` refuses it as a source, and `&` refuses it, since neither has a spec to read. `None` means only that the spec is unspecified, so an opaque field of a `RecordSpec` is declared as `OpaqueSpec()` (III.5).

**Defaults.** `OutputSpec.default(term_spec, output_name=...)` is the output spec ProbPipe uses where none is declared:
1. for a base spec, the keyword form under `output_name`;
2. for a batch whose elements carry only the default name `element`, the batch with its elements named `output_name`;
3. for any other spec, the positional form, so a distribution has no outputs and a function or a batch has its own.

The body of a function builds any record itself: ProbPipe never wraps the value a call returns, and a body that returns a distribution alone where `{params: law}` is declared fails the output spec's check.

**Equality.** Two output specs are equal when they have equal value types, field order included (II.1), and the same name for an output at the root; an output spec equals only another output spec, and its hash agrees with its equality. The name of an output at the root belongs to the output spec, not to the term's type, so the same rule holds for every kind that states one, a distribution's or a conditional distribution's event name, a function's output name, and a batch's element name:
1. conformance and unification (II.1) ignore it;
2. a term's equality, its hash, its fingerprint (II.4), and the static data of its JAX pytree ignore it, through a canonical spec without the name;
3. `OutputSpec` itself, and the specs that hold one (`DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, and `BatchSpec`), compare it in their own equality.

So `Normal("x", 0.0, 1.0)` and `Normal("y", 0.0, 1.0)` are the same law, whose specs differ, and they merge differently, as two batches of the same values named `x` and `y` do (II.5).

### Rationale

Naming inputs and outputs, apart from the term's label, is `C5 – Naming for unambiguous meaning`: a model's label and the variables it gives answer different questions, and a merge matches the variables. Reading the outputs from the value type is `D6 – Single source of truth`: no flag restates what the spec already says, and every kind derives its outputs, its paths, and the output spec of a part from one spec, so they cannot disagree. Naming only a base value from outside keeps one path per node, so a draw and its distribution are indexed alike (`C5 – Naming for unambiguous meaning`). A distribution as a value has no outputs because its marginals do not determine it, and offering them would let a later step lose their dependence (`D1 – Mathematical fidelity`). One reading of inputs from four sources serves every constructor that declares inputs (`C1 – Uniform interface to functions, distributions, and values`), and it lets a model declare a kernel's inputs from the outputs of the terms that will supply them. Keeping the root name out of a term's equality, fingerprint, and JAX structure lets two terms that compute the same thing share a compilation and a cache entry, while the specs that state the name compare it, so no cached structure holds a name that compares equal without it (`C3 – Computational detail hidden by default, available on demand`, `C6 – Traceable and reproducible workflows`).

### Notes

- *The keyword form over a record, a distribution, a function, or a batch.* The families of laws over laws, functions, or batches still name their event whole in the keyword form, which then has its name as one output with paths below it, until each wraps the values it is over under its constructor's name (VII.2, VII.5, VII.6). Once none uses it, the keyword form refuses such a spec, as the contract states.

## II.3 — Numeric values: `Numeric`, `NumericSpec`, `Constraint`

### Contract

**The `Numeric` interface.** The numeric kinds share the `to_vector` interface, which lays a value out as one flat vector in canonical order. In addition, `vector_size` gives that vector's length, and `from_vector` rebuilds a value from a vector. The coordinate protocols expose the same layout to foreign libraries, so `np.*` and `jnp.*` functions apply to the numeric kinds (returning bare arrays). A bare array passed where a `Numeric` value is expected is promoted to the appropriate numeric type. Functions therefore act on a numeric value in two ways: a foreign function sees the coordinates and returns a bare array, while ProbPipe's own operators and entrywise `map` preserve structure and return tracked terms.

```python
class Numeric(ABC):                         # the flat-vector interface of the numeric kinds
    @property
    @abstractmethod
    def vector_size(self) -> int: ...       # total flat dimension; defined only when concrete
    @abstractmethod
    def to_vector(self) -> Array: ...       # the coordinates: one flat vector, canonical order
    @classmethod
    @abstractmethod
    def from_vector(cls, spec: NumericSpec, vec: Array, *,
                    label: str) -> Self: ...  # the inverse of to_vector

    # the coordinate protocols: NumPy and JAX read the value as to_vector(),
    # so their functions return bare arrays
    def __array__(self) -> np.ndarray: ...
    def __jax_array__(self) -> Array: ...
```

**The array-backend registry.** A numeric host that duck typing cannot read, such as an xarray `DataArray` or a pandas object, is recognized through a registry keyed on its type. A registered backend reports whether an instance holds numeric data, its event shape and dtype without converting it, its conversion to the backend array at the compute boundary, and the metadata it carries, which the wrap of V.4 reads. Backends register at import (II.7); NumPy and JAX arrays need none.

**The `NumericSpec` mixin.** A term spec includes `NumericSpec` when every value it admits implements `Numeric`. The mixin serves two purposes. As a type, it identifies numeric structure from declarations alone. As an interface, it adds `vector_size`, since the flat dimension is known from the spec's shapes alone. A spec that includes it keeps its own kind.

```python
class NumericSpec(TermSpec, ABC):   # mixin: the specs whose values implement Numeric
    @property
    def vector_size(self) -> int: ...   # total flat dimension; raises unless concrete
    @abstractmethod
    def _vector_size(self) -> int: ...  # the count itself; called only for a concrete spec
```

A numeric kind may also specify the **support** of its values with a `Constraint`, which compares and hashes by value.

```python
class Constraint(ABC):
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # entrywise membership
```

A support that is a subset of the reals contains a complex entry only where its imaginary part is zero and its real part is in the support.

### Rationale

One flat-vector interface over the numeric kinds is `D2 – Generality first`: everything that consumes flat numeric values types against it once, and the coordinate protocols keep foreign array functions usable with no ProbPipe-specific code (`C3 – Computational detail hidden by default, available on demand`). The spec-side mixin is the same generality at the type level, whether the event is one array or a named tree of them. Both are abstract bases rather than protocols, which keeps the pair symmetric and follows the rule the library uses throughout: an interface a closed set of ProbPipe kinds implements is a base, while an interface any object may implement is a structural protocol. The base also holds the shared coordinate protocols once rather than in each kind (`D6 – Single source of truth`). A constraint is data: comparing and hashing by value lets a support key a registry, so the bijector factories select by the mathematics rather than by class identity (`D3 – Capability-based operations`).

## II.4 — Identity, type & metadata: `TrackedTerm`, `Provenance`

### Contract

Every tracked term carries four things through the one mixin `TrackedTerm`:
1. a **label**: what a reader calls the object;
2. a **spec**: its type (II.1);
3. a **provenance**: how it was produced;
4. **annotations**: free-form auxiliary information supplied by the user or an algorithm.

Every term a constructor builds carries a description, the expression its label renders unless the caller gives an alias. Constructors take mathematical data first and the alias as the keyword `label=`. A label is required when the data cannot identify what the value represents: a raw `NumericArray`, `Opaque`, `NumericArrayBatch`, or `OpaqueBatch` value, and a function or a function batch member whose callable has no name, such as a `functools.partial`. Otherwise, construction derives a description from declared structure or described operands:

1. records and record batches use their ordered field names, except that a batch stacked from records takes the first record's label (III.6); an empty record requires a label;
2. functions use the callable's name, with `𝒻` for a lambda; lambda kernels use `ℙ`;
3. catalog laws use their family name and their outputs; generic laws and kernels use `ℙ` and their input and event specs;
4. products, mixtures, and minibatched laws describe their operands, a mixture as `mixture([a(x), b(x)])` and a minibatched law as `minibatch(prior(beta), lik(y | beta), batch_size=40)`; object batches describe their members in row-major order over all their batch axes, showing at most eight and then `…`, and empty object batches require a label.

A managed function call derives an application expression from the function and arguments unless an explicit `output_label` supplies an alias. Unnamed array arguments appear by their parameter names. An operation derives its result's expression from its operands as its own rule specifies. Computational names belong to specs: labels never supply outputs, fields, levels, fingerprints, or random-stream identity.

JAX pytree rebuilding carries numerical leaves and the specs needed to interpret them, without descriptions or provenance. A rebuilt term displays `<no description>` and remains fully usable, including inside `jit` and `vmap`, and the managed result boundary supplies its result expression.

The result of a user-defined function called on values carries its application description unless an explicit `output_label` provides an alias (III.3). A law or a kernel that the function returns is labeled by the function's label, or by the alias, and displays its own signature, as `predict(y, mu)` (IV.2). An operation's result carries the expression that the operation's rule builds from its operands' expressions, and an operation without a rule gives its result the expression of its primary operand, the first in its signature. A merge (IV.2), `condition_on` (VI.6), and `marginal` and `factor` (VI.8) return a part or a conditional of a law, so their rules label the result by what it is: a product of factors joins the factors' labels, and a conditional keeps the label of the law it conditions and records the paths it fixes, as `condition_on(schools, data)` is labeled `schools` and displays as `schools(theta; y)`. When conditioning leaves one factor of a product alone, the result is that factor's conditional and keeps the factor's label, even when the product carries an alias from `with_label`, so `condition_on(lik & prior, {"mu": 0.5})` displays as `lik(y; mu)` whatever the product's label.

A label is set once, at construction, and representation changes preserve it: a record with renamed fields, a realigned factor, or a converted law keeps the label it had, and `with_label` replaces it explicitly. A law converted from a law constructed without `label=` takes its own default label instead, since the source's default names only the source's class, so `convert(Laplace("g", 9.0, 1.0), Normal)` displays as `Normal(g)` (VI.10). `with_label` keeps the paths a law or a kernel holds fixed, so a posterior relabeled `fit` displays as `fit(mu; y)`. No lookup resolves an object by its label, and derived labels need no escaping scheme. A derived label may hold `~`, a space, `;`, or `/`, so no operation uses a label as an output name or a level name: a law an operation builds for its own computation takes a fixed name or its operand's outputs.

**Label, signature, and notation.** A label names an object for a reader, and each other mechanism that names or identifies something has a responsibility of its own:

| Mechanism | Responsibility | Section |
|---|---|---|
| label | names an object for a reader | II.4 |
| signature | states what a law, a kernel, or a function is over, read from its input and output specs | II.4 |
| output name | names an output, which a merge matches to an input of the same name | II.2 |
| path | names a node of a value type, which indexing selects | II.6 |
| input name | names where an argument or a given value binds | II.2 |
| source identity, random-event identity, fingerprints, replay, and caching | identify objects, content, and random events, and decide reuse | V.5, V.8, II.4 |

The **signature** of a term is read from its input and output specs:
1. a law: its outputs in order, as `mu, tau`;
2. a kernel: its outputs, then `|` and its inputs, as `y | beta`;
3. a function: its parameters, as `x, y`.

An input that has a default reads `name=value`, as `counts(y | K, r, n0=50.0)` or `predict(x, scale=1.0)`. A default that is a number, a string, `None`, or an array of no axes shows its value, and any other default, or one the kernel cannot read, shows `name=…`. An input with a default stays an input of each law and kernel that conditioning makes until a given binds it (VI.6), so `counts` at `K` and `r` reads `counts(y | n0=50.0; K, r)`. The call of a lifted function (below) shows the arguments of the call, so a parameter left at its default is not among them.

A law or a kernel that holds paths fixed at given values lists them after `;`, as `mu; y` for the posterior of `mu` given `y` and `y | sigma; beta` for a kernel applied at `beta`. A `rename` of outputs or inputs changes the signature, and `with_label` changes the label. The **notation** is the label followed by the signature in parentheses, as `prior(mu)`, `glm(y | beta)`, or `predict(x, y)`, and the property `notation` returns it. `str()` shows the notation, and the repr shows the label and the name of the output at the root as the repr paragraph below states. The signature has no public attribute of its own, since `Function.signature` is the Python call signature that binding reads (III.3). No operation reads a label or a signature to decide anything, and two objects may share either.

**The expression.** Every tracked term carries one immutable **expression**, a tree that states what the term is, and its label, its notation, `str()`, and every label derived from it are renderings of that tree. A term constructed directly carries its label alone. The operation that makes a term builds the term's expression from its operands' expressions, so no operation builds a label of its own, and no operation reads an expression. A node of the tree is one of these:
1. a label, with the signature of a law, a kernel, or a function;
2. a product of factors without a label;
3. a law or a kernel conditioned at given values of some paths;
4. a law selected at some of its paths;
5. a draw from a law;
6. a function applied to draws of laws and to values;
7. a summary of a law or a draw: an expectation, a variance, a covariance, a quantile, a score, or a density;
8. an operator applied to values;
9. a batch at some of its positions.

A term's label is the name part of the rendering:
1. a conditioned or a selected law keeps its base's label;
2. a product without a label joins its factors' labels;
3. an applied function takes the function's label;
4. a value renders in full, as `mu ~ prior` or `𝔼[mu ~ prior]`.

The notation is the whole rendering, grouped by the rules below. A law's signature is read from its event spec, and the paths it holds fixed and whether a product was given a label are read from its expression (IV.2). `with_label` replaces the expression with the new label, so a user's label hides the derivation, which provenance still records. Provenance records how a term was computed, and the expression records what the term is, for a reader. Fingerprints and replay omit the expression, as they omit the label, and copies and pickles keep it.

A rendering shows at most `config.notation.max_depth` nested levels (V.2), eight by default and at most 64, and a part nested deeper renders as its label, the name of a law or a function, or as `…` for a value. A term's label is rendered once, when the term is built, at the setting in effect then, and the term keeps it. A notation is rendered each time it is shown, at the current setting. A stored expression keeps at most 64 levels, so a long derivation, such as a loop that adds to a value, never builds a tree too deep to copy or pickle. A node built over a part that nests 64 levels stores the whole part as its label or `…`, together with its signature and the paths it holds fixed, and it renders that way at any setting, which is why the setting is at most 64. A long derivation is therefore truncated whole each time it reaches 64 levels, and the label of the operation that truncates it shows that operation alone, as `… + 1`. Showing a term warns: `str()`, `repr()`, and `notation` warn with a `UserWarning` when what they show left a part out, and the warning names the setting where raising it shows more. A part that storage truncated warns only where its text shows less than the part in full, and that warning says that no setting shows the part.

**Values computed from a law.** A value computed from a law `d` is labeled by the value written in probability notation over `d`, and its outputs follow the result rule of VI.0:
1. **A draw** (VI.3) is labeled by its outputs, `~`, and `d`'s label, then `;` and the paths `d` holds fixed when it holds any, as `mu ~ prior`, `(y, mu) ~ model`, or `mu ~ model; y`, and it keeps `d`'s outputs.
2. **A score** (VI.4) is labeled `log` followed by `d`'s notation, as `log prior(mu)` or `log model(mu; y)`, and a density is labeled by `d`'s notation, as `prior(mu)`, since that notation denotes the density.
3. **A moment** (VI.5) is labeled `𝔼[draw]`, `𝕍[draw]`, or `ℂ[draw]`, and a quantile `ℚ[draw]`, as `𝔼[(y, mu) ~ model]`.
4. **An expectation** of `f` is labeled `𝔼[...]` around the label that a call of `f` on a draw takes, as `𝔼[f(mu ~ prior)]`, with `𝒻` for a lambda, so `expectation(d, f)` and `mean(f(d))` read alike.

A batch of draws or of scores has the label of one draw or one score and a level of its own, and its element is grouped before the positions, as `(mu ~ prior)[sample=0]`. A batch of laws reads as one law under its label, so a value computed from a batch `schools` of laws over `effect` is labeled `effect ~ schools`, `𝔼[effect ~ schools]`, or `log schools(effect)`, and `str()` of the batch shows that notation, `over`, and its levels, as `schools(effect) over school`. The atoms of an empirical law that `condition_on` or an inference method produces are labeled by the law's outputs, as `beta` or `(K, r, phi)`, and the atoms of an empirical law a user constructs keep the label they were given. A node of a value takes its path, so `sample(model)["y"]` is labeled `y`, and a value selected at a list of paths is labeled by its paths, as `(a, g/b)`, and an operator on values is labeled by its expression, as `2 * effect` or `2 * 𝔼[mu ~ prior]`.

**Lifted functions.** A law that lifting a function over laws produces (V.5) is labeled by the function's `output_label`, and it displays as the function applied to draws of its inputs, as `challenger_damage_probability(beta ~ oring_model; damage)`, since it is the law of that random quantity. Arguments that the lift draws together from one law share one draw, as `f((a, b) ~ model)`. A law that is not lifted appears by its notation, as `log_prob(g(g), q ~ q)`, any other tracked argument by its label, and any other argument that the caller passed by its value when it is a scalar, as `f(beta ~ model; y, 2.0)`, and by its parameter's name otherwise, as `f(beta ~ model; y, X)`. A batch of such laws, which a lift over a law and a sweep over a batch give together, carries the same expression, so its mean is labeled `𝔼[f(mu ~ prior, tau)]`. Its element keeps the label of its position, as `f[tau=3]`, and displays as its row's call, which shows a scalar of the swept batch by its value, as `f(mu ~ prior, 4.0)`; any other positions display as the batch's call and the selected levels, as `f(mu ~ prior, tau)[tau=1:3]`. A value computed from such a law takes the law's notation in place of a draw, as `𝔼[challenger_damage_probability(beta ~ oring_model; damage)]`.

**Grouping.** A label is grouped where another label is built from it, so it reads as one operand:
1. a **compound** label is parenthesized, and a label is compound when it has one of these forms:
   - an operator at its top level, as `effect + 1.0` or the draw `mu ~ prior`;
   - a product of labels, as `lik·prior`;
   - a unary operator or `log` at its start, as `-x` or the score `log prior(mu)`;
2. any other label of several words is bracketed, as `[other effect]`;
3. a label of one word is used as it is, and a call such as `prior(mu)` is one word.

So positions of a batch labeled `x·y` is labeled `(x·y)[sample=0:2]`, and a law derived from an unlabeled product displays as `(lik·prior)(y)`. A draw and a score are grouped before positions, as in `(mu ~ prior)[sample=0]` and `(log prior(mu))[sample=0]`, and the notation of a product without a label is grouped after `log`, as in `log (lik(y | mu)·prior(mu))`. Two rules join labels without grouping them. The right side of `~` is not grouped, since `~` binds most loosely, as in `(y, mu) ~ lik·prior`. Labels join associatively, so a product joins a further factor's label as it is, and `(lik & prior) & d` is labeled `lik·prior·d` (IV.2).

The `spec` attribute is the term's type, stored once. Each kind narrows it to its own spec class and exposes convenience accessors for its properties.

Every tracked term exposes the property `raw` as the single access point to the representation layer. It is the term **detached** from the workflow. Detachment removes provenance, annotations, and any reference to a container or parent, and it keeps the spec and the label. A kind represented by an object from outside ProbPipe has a **raw host**, which is its `raw`, for example a backing array object or a wrapped callable. The `raw` of a kind whose representation is a ProbPipe object, such as a factored joint, is that object detached, and the `raw` of a coupled marginal of a law is the raw form of its parent's marginal at its path (III.7). **Every read of `raw` gives the same raw form.** A term keeps the detached copy it builds on the first read, and the copy refuses provenance, since every reader shares it, while a relabeled or unpickled copy of it is a term of its own. A record keeps its resolved leaves and gives a new `dict` over them on each read, which the caller may change (III.5).

Accessing a container returns a **view**, for example a record field or a batch element. A container's view is a tracked term labeled from the accessor, which is the path for a record and the selected levels for a batch, and a batch element is a relabeled copy, so the term the batch stores keeps its own label; its provenance records the container and the source term where one was supplied. A batch's label is grouped before the positions, as in `(lik·prior)[dataset=0]`, so the positions read as applying to the whole label.

**The repr.** A law's or kernel's repr reads as a call of its public constructor, with any paths it holds fixed as its last argument. Other terms show diagnostic summaries of their label and structure, which are not constructor calls, so `NumericArray('x', shape=(3,), dtype=float32)` shows the label first although the constructor takes it as a keyword:
1. **the label**: a value, a record, or a batch displays it first in its diagnostic summary, as in `RecordBatch('schools', ...)`. A law or a kernel takes it as the keyword `label=` after its other arguments, and leaves it out exactly when the constructor, given the arguments the call shows and no label, would give that label again: its default, the class name or `ℙ`, and a label it derives from its arguments, such as a product's or a mixture's. Any other label is shown, including one an operation carries to another class, so `convert` of a mixture to `Normal` reads `Normal('x', ..., label='mixture')`. A label that a function's name gives is shown, as in `ConditionalDistribution('y', given=('mu',), label='ricker_counts')`, since the class does not state it;
2. **a batch's levels**: a mapping of level name to size, as in `levels={'chain': 4, 'draw': 500}`, with the tuple of its sizes for a level of several axes;
3. **the element's structure**, named for what it is:
   - a record: its field paths, as in `fields=('data/effect', 'data/se', 'label')`, which a numeric record lists on the lines below its summary instead;
   - an array: its shape and dtype, as in `NumericArray('x', shape=(3,), dtype=float32)`;
   - a batch of arrays or laws: its element spec;
   - a law or a kernel: the name of an output at the root and its other mathematical arguments, as its constructor takes them, and its event spec as `event_spec=` where that differs from the default of III.7. So `Normal("mu", 0.0, 1.0)` reads `Normal('mu', loc=0.0, scale=1.0)`, and `Normal("mu", 0.0, 1.0, label="prior")` reads `Normal('mu', loc=0.0, scale=1.0, label='prior')`. An atoms-based law follows its own constructor's order, as `EmpiricalDistribution(atoms=..., output_name='theta')`.

A law or a kernel that holds paths fixed shows them as the call's last argument, `fixed=(...)`. A posterior of `oring_model` given `damage` therefore reads `EmpiricalDistribution(atoms=..., label='oring_model', fixed=('damage',))` and differs in its repr from a law of the same atoms that holds nothing fixed, and a kernel applied at `beta` reads `ConditionalDistribution('y', given=('sigma',), label='glm', fixed=('beta',))`. The repr of a family that holds nothing fixed, and whose parameters print in full, evaluates to a law of the same class, with equal parameters, under the same label. A parameter prints in full when it is a number, an array of at most eight entries, a law whose own repr evaluates, or a sequence of at most eight such laws, as a mixture's components are. A function, such as a bijector, prints by its label and parameters, so a law that holds one does not evaluate.

A spec reads as its own constructor call, with the attributes it sets, and so does a report, such as a `MethodInfo` or a `CallReport` (V.1). A name that is not a Python identifier, such as the output name `my model`, cannot be a keyword, so a call with one writes its names in order inside one `**{...}` mapping, as in `OutputSpec(**{'my model': NumericArraySpec(shape=())})`. A private class reads as its public class or kind, so a law that renames another reads as the class of the law it renames. A repr longer than about 100 characters shows one argument per line. In a notebook a `CallReport` also displays as a table, with a row for each route and the selected route marked.

**A numeric term shows its values.** A numeric term, which is a `NumericArray`, a `NumericRecord`, or a batch of either, shows its values on the lines below the summary, with each line indented by two spaces. `str()` shows the values alone, at the margin. NumPy's printing formats each value, so `np.set_printoptions` sets the precision and the size past which an array is shortened with `...`. Only the entries shown are read from the device. A record and a batch of records show one line for each field path, such as `population/mu`, followed by its value, and a value shortened on such a line ends with its shape. The summary above these lines leaves out `fields=`, since the lines list the paths. A batch shows its stored array or its columns, whose leading axes are its levels. A summary shown alone keeps `fields=`, and a term shows its summary alone in three cases: inside another repr, as the atoms in a law's repr do; inside a JAX trace, where the values are tracers; and when no entry can be read. Its `str()` is then the summary too. In a notebook a record and a batch of records also display as a two-column table of paths and values.

**A record of summaries displays as a table.** A numeric record displays as a table under its label when its top-level fields are records over the same paths, with the same shape at each path, and at least one of them records how it was computed. The table has one row for each path, or for each element of an array path, as `y[0]`, and one column for each top-level field, with the table's lines indented by two spaces as a record's lines are. NumPy's print options format each column's entries together and shorten the rows of a large array path to its first and last elements. The line below the table states how each column was computed: `exact` for an exact route, `Monte Carlo` with the number of draws for a route that records its draws, and `approximate` for another approximate route, as each column's provenance records it. A record whose provenance records a description of each column under `computations`, as the record that `summarize` returns does (VI.5), takes those descriptions instead. `str()` gives the table without its label and that line, and in a notebook the table is an HTML table. The record that `summarize` returns displays this way, and so does one built by hand, as `Record({"mean": mean(d), "sd": standard_deviation(d)})`. A record of plain values, or the draw or the mean of a law whose groups share their field names, as `a/mu` and `b/mu`, records no computation and shows one line for each path, as does every record when provenance tracking is off.

A batch of records displays one table for each element when its provenance describes each element's columns, as the batch that `summarize` returns for a `DistributionBatch` does. Each table is titled by its element's levels, as `law=0`, with each title indented by two spaces and each table by four. The line that states how the columns were computed follows each table, or follows the last when every element's line is the same. The elements are shortened to their first and last ones when their rows number more than NumPy's `threshold`. In a notebook each table is an HTML table.

**A tracked term is immutable.** `TrackedTerm` carries an immutability guard automatically, so assignment and deletion raise an error. Immutability requires that every transformation, including each `with_*` method, returns a new term that shares the representation.

Identity is **boundary-attached** under compiled execution. Inside a `jit` or `vmap` trace a term presents as its raw representation with only its spec as static data, so label, provenance, and annotations never enter a trace and a label can never affect compilation-cache identity; the tracked result is minted at the enclosing call boundary. A value, a record, and a batch of either are pytrees whose static data is their spec alone, without the name of an output at the root (II.2), so two terms that differ only in their labels or their expressions have equal treedefs and share a compilation. A term rebuilt from its leaves displays `<no description>` until a managed result boundary gives it the call's expression or explicit output alias. Rebuilt terms remain usable during JAX tracing and direct rebuilding.

```python
class TrackedTerm(ABC):
    label:        str
    spec:         TermSpec                       # the single stored source of the term's type (II.1)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    annotations:  Mapping[str, Any] | None       # free-form; the one store written after construction
    def with_label(self, label: str) -> Self: ...  # the one way a label changes
    def with_provenance(self, p: Provenance) -> Self: ...
    @property
    @abstractmethod
    def raw(self) -> Any: ...
    # the representation, detached from the workflow; each kind defines it, as its section states
    # immutable: __setattr__ / __delattr__ raise; state round-trips through the
    # attributes the term holds, so copy and pickle need nothing from the class

class Provenance:
    operation: str                       # the operation that produced the object
    parents:   tuple[ParentInfo, ...]    # descriptors of the tracked inputs
    controls:  Mapping[str, Any]         # the resolved controls, and the derived key of each draw: the counts, selected method, ...
    inputs:    Mapping[str, ParentInfo]  # plain (untracked) arguments, keyed by parameter name

class ParentInfo:
    type_name:   str
    name:        str
    fingerprint: str            # best-effort content hash
    fingerprint_is_weak: bool   # True when the fingerprint is only object identity
    parent:      Any | None     # optional reference to original parent

def provenance_ancestors(term: TrackedTerm) -> list[ParentInfo]: ...   # every ancestor reachable through parents
def provenance_dag(term: TrackedTerm) -> Any: ...                       # the chain rendered as a Graphviz graph

class NotationConfig:                  # how labels and notations render
    @property
    def max_depth(self) -> int: ...    # the nested levels a rendering shows: 8 by default, from 1 to 64
    def reset(self) -> None: ...       # the defaults, re-reading PROBPIPE_NOTATION_MAX_DEPTH

# the settings in effect are the section config.notation (V.2)
```

The **annotations** store is the one exception to immutability: it can be written after construction, so a diagnostic can attach its result to the object it examined, and by convention it is append-only. Annotations are otherwise inert: no operation reads them to decide behavior, they do not propagate to results because lineage is recorded in `provenance`, and they never enter a compiled trace.

Fingerprints are best-effort and tiered, from a content hash, through the code hash of a closure-free callable, down to object identity, with `fingerprint_is_weak` marking the weakest tier. A fingerprint records what a term computes, so a label, which names the term for display, is no part of it: a relabeled law or function keeps its fingerprint. Nor is the name of an output at the root, which names the term's output rather than what it computes (II.2), so renaming it keeps the fingerprint. **The provenance mode alone sets the tier.** The default `lightweight` mode records identity-tier descriptors for every operation's parents with no content hashing; the `full` mode retains parent references and content-verifiable fingerprints, computed lazily at export; `off` mode records nothing. Every operation behaves identically under a given mode. Two functions can access provenance information when available: `provenance_ancestors` returns every ancestor reachable through the parents while `provenance_dag` shows it as a graph.

### Rationale

`TrackedTerm` serves the two non-mathematical principles, `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D6 – Single source of truth` for a term's type: one attribute, declared once, that every kind's accessors are views on. Recording the resolved controls, not just the parents, turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived labels keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`), and boundary attachment keeps labels inert in computation, so a label never decides what gets compiled. Immutability is `C2 – Functional interface over immutable objects` embodied, and confining the one writable store to a container that no operation reads keeps that contract intact in substance. Carrying annotations on the base makes every tracked term annotatable, including a batch of draws. `raw` is `B3 – Tracked forms out by default` for a term already in hand: the representation is one attribute away. The repr serves `C5 – Naming for unambiguous meaning`, since it names the term's label, its public class, and each part by what it is, so a reader learns what a term is and holds from its repr alone. A numeric term's repr also shows its values, so a reader sees a result in the tracked form that `B3 – Tracked forms out by default` returns without reading `raw`. Reading only the entries shown keeps the display of a large array cheap. Reading the signature from the input and output specs is `D6 – Single source of truth`: the notation states what a term is over from the one place that fixes it, so a label chosen by the user cannot misstate it.

### Notes

- *Reconstruction.* `pickle` and `copy` restore a term by assigning its state back rather than through its constructor, so a schema fixed by an explicit declaration, a class decided at construction, and a field written afterward all survive the round-trip. A term declares any memo as transient and any store written in place (annotations) as decoupled, so a copy takes its own container.

Automatic summaries use these symbols:

| Summary | Symbol |
|---|---|
| Mean and expectation | `𝔼` |
| Variance | `𝕍` |
| Standard deviation | `√𝕍` |
| Covariance | `ℂ` |
| Quantile | `ℚ` |

Explicit labels remain literal, including ASCII `p`, `f`, `E`, and `Q`. Generated output names retain their ASCII spelling, and caller-supplied names are unchanged. An anonymous function displays as `𝒻` while its default output name is `f`.

## II.5 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd-array of shape `batch_shape`, holding objects of a common element type. `batch_shape` is nonempty. A concrete batch specifies how it stores its elements and adds whatever its element type affords, including indexing by the paths of its elements. Since a batch is a *collection* of its elements, `len`, `iter`, `batch_shape`, and `batch_size` operate on the batch axes only. `BatchSpec` is the batch kind's term spec (II.1), carrying the element spec, the output spec of one element (II.2), together with the named multiplicity:

```python
class BatchSpec(TermSpec):         # the batch kind's spec; is_valid accepts a matching batch
    element_spec: OutputSpec                        # one element: its spec, element_spec.term_spec, and its outputs
    axis_groups: tuple[tuple[int | str, ...], ...]  # the multiplicity, tiled into levels below;
                                                    #   a str names a symbolic dimension (II.1)
    level_names: tuple[str, ...]
    def __init__(self, element: OutputSpec | TermSpec,
                 levels: Mapping[str, int | str | Iterable[int | str]] | None = None, /,
                 **level_shapes: int | str | Iterable[int | str]) -> None: ...
    # one keyword per level, outermost first, giving the shape of its axes
    @property
    def levels(self) -> Mapping[str, tuple[int | str, ...]]: ...   # level name -> its axis sizes
```

**Constructing a `BatchSpec`.** `BatchSpec(element, **levels)` names each level and gives the shape of its axes, outermost first, as in `BatchSpec(NumericArraySpec(()), chain=4, draw="S")`. Each level's shape is a shape argument (II.1), so `draw=4` is one axis of size 4 and `grid=(3, 4)` is one level of two axes. A level name that no keyword spells is given in a mapping passed positionally, as `BatchSpec(spec, {"my level": 2})`, and the two forms are not combined. `levels` returns that mapping, so `BatchSpec(other_spec, batch.spec.levels)` declares another element type over a batch's levels. A batch constructor takes `level_names`, where a single str names one level, and `axes_per_level`, where a single int is one level's count, since a live batch reads its sizes off its elements.

`BatchSpec`'s first argument, `element`, takes an output spec or a bare spec. A bare spec takes the **raw default**: a base spec names the elements `element`, in the keyword form, and any other spec is in the positional form (II.2). Construction checks every element against `element_spec.term_spec` and reports the position that failed, since the batch asserts that spec of all of them. A constructor given no element spec infers the spec from the elements, as the element kind's constructor infers a term's spec from its value, and takes the raw default. The axes past those the levels hold are an array's event shape, so `NumericArrayBatch(jnp.arange(4.0), 'draw', label='draws')` holds four scalars, and its element spec is `OutputSpec(element=NumericArraySpec(()))`, whose name the repr leaves out. A batch of records infers each field's spec from its column in the same way. The batch's label never supplies an output name.

**Where an element spec comes from.** Every batch holds an element spec:
1. a batch an operation produces stores its maker's output spec of one element: a law's event spec for draws (VI.3), and a function's completed output spec for a sweep (V.6), which evaluates a function over a grid of inputs;
2. a batch built from bare values takes the raw default;
3. a marginal batch of a batch of records stores the output spec at its path (II.2), so `b["a"]` stores `OutputSpec(a=...)`, and the marginal batch at a group stores the positional form over the group's record.

**`[]` dispatch.** `[]` takes either a **position** or what II.6's reader reads as a path or a list of paths:
1. a position is an integer, a slice, or a tuple of these, and it addresses the batch axes, giving an element or a sub-batch;
2. a path, such as `"g/b"`, gives the **marginal batch** at that path, the batch over the same levels of each element's node there, as a view (II.4) labeled by the path; a group's marginal batch is the batch of the group's records. It is to a batch what the coupled marginal `d[p]` is to a law: for samples of `d`, the marginal batch at `p` holds samples of `d[p]`;
3. a list of paths gives the batch over the same levels of each element's record of the chosen nodes, each at its own path, in the element's order, so on a batch of records it is a `RecordBatch`.

A list may name the output at the root of base elements, so for samples `x` from `Normal("x", 0.0, 1.0)`, `x[["x"]]` is the batch of records `{x}`. The name of that output alone is not a path, so `x["x"]` raises `KeyError`, whose message says that `x` names the elements as a whole and points to `output`. `b[()]` and `b[[]]` raise `ValueError`, since they name nothing; a list of positions, as `b[[0, 2]]`, raises `TypeError`; and a path and a position mixed in one `[]` raise `TypeError`. Positions and paths commute: `b[3]["g"]` is `b["g"][3]`. `p in b` tests paths, never elements, and holds exactly when `b[p]` succeeds. A batch of laws is indexed by the paths of its elements' events, giving the batch of their views (III.10).

**Outputs.** A batch's value type is its element's spec (II.2), so its outputs are its element spec's outputs, each typed by its marginal batch's `BatchSpec`, the batch's levels over that output's element-level spec:
1. a batch of records has its elements' children;
2. a batch of base values has one output at the root, its **element name**;
3. a batch of functions has its elements' outputs;
4. a batch of laws or of conditional distributions has none.

`b.output(c)` returns the marginal batch at `c`, or the batch itself when its elements are base values named `c`. `b & c` merges two batches over equal levels element by element, so a batch of 3 values named `x` and a batch of 3 values named `y` merge into a batch of 3 records `{x, y}`; batches over other levels are refused (VI.11). `without(*paths)` removes nodes from every element of a batch of records or of laws. `rename` renames paths in every element, and renaming the element name renames a batch of base values' one output.

**The element name.** The element name belongs to the element spec, not to the batch's type, so II.2's rule for the name of an output at the root applies to it, recursively through records, laws, and functions that hold a batch. Two batches of the same 4 values, one naming them `x` and one `y`, are therefore equal, while their specs differ, and they merge differently. A base element's name does not survive a raw JAX transform, so `jax.jit(lambda b: b)(x)` returns a batch whose output is `element`, and ProbPipe's own operations restore names at their results. A record element's names are in its `RecordSpec`, so they survive.

```python
class Batch[E](TrackedTerm):
    @property
    def spec(self) -> BatchSpec: ...
    @property
    def element_spec(self) -> OutputSpec: ...            # view on spec: one element's spec and outputs
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_size(self) -> int: ...                    # total element count, prod(batch_shape)
    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]: ...   # batch_shape tiled into levels, outermost first
    @property
    def level_names(self) -> tuple[str, ...]: ...       # one name per level, aligned with axis_groups
    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename levels old -> new; shapes and elements unchanged, as rename is for paths
    def __len__(self) -> int: ...                       # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...       # over the leading batch axis
    def __repr__(self) -> str: ...                      # the label, the levels, the elements' structure, and a numeric batch's values (II.4)
    def __getitem__(self, key: Any) -> Any: ...         # a position indexes the axes; a path or a list of paths gives marginal batches
    def __contains__(self, path: object) -> bool: ...   # whether self[path] succeeds; never tests elements
    def output(self, output_name: str, /) -> Self: ...  # the marginal batch at an output, or self at the element name
    def without(self, *paths: str) -> Self: ...
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Self: ...
    def __and__(self, other: Batch, /) -> Batch: ...    # element by element, over equal levels
    def at_levels(self, /, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
    @property
    def raw(self) -> Any: ...
    # the storage view: the elements' raw values in their native stacked layout
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**: `axis_groups` tiles `batch_shape` into contiguous groups with the outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, such as flat vectorization, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and positional `[]` address the leading **axis**, which is `batch_shape[0]`, rather than the leading level; the two coincide when the outermost level holds one axis, so iterating `(N,)` of `(S,)` runs over the `N` and yields each inner batch of `S` as a view. When the outermost level spans several axes, indexing drops the leading axis and leaves the level in place with one axis fewer.

**The batch's own type.** A batch stores its `BatchSpec`; `element_spec`, `axis_groups`, and `level_names` are views on it. `spec` is the batch's own type, which keeps `OpaqueBatch` well typed although its elements carry no structure.

**The raw view.** `raw` is the **storage view**, which holds the elements' raw values in their native stacked layout — for example, one array for array-valued elements and an object array for opaque-valued ones.

**A polymorphic multiplicity.** An axis size may be a symbolic dimension name instead of an integer, as a `NumericArraySpec` shape entry may, so a *spec* can fix the number of levels while deferring how many elements each holds: "returns a batch of `S` draws" before `S` is known. The names share one scope with the element's schema, so a batch of `("n",)` over arrays of shape `("n",)` is square by declaration. A live `Batch` is concrete, since it holds elements at positions; construction refuses a spec with free dimensions, and `batch_size` is defined once they are bound.

**Level names.** Each level carries a name, listed in order by `level_names`. Names are unique within a batch and follow the rule for output names, so a level name is any non-empty string without `/` (II.2). `at_levels` takes a level's name as a keyword, and a name that is not a Python identifier is passed in a mapping, as `at_levels(**{"my model": 0})`. An operation names the level it mints after itself, and a constructor such as `stack` takes the name to give it. A name already in use raises, as does a rename onto one, so the caller renames first or supplies another. `with_level_names` renames levels while preserving the object's label, shapes, and elements; subsequent views use the renamed levels. Renaming a *view* is refused when the new name collides with a level in its root selection that it no longer carries; `with_label` gives it a new label and view root. Operations align batched operands by their level names (VI.11), which are independent of the field names within an element.

**View identity.** A view of a batch, whether an element or a sub-batch, derives its label from the batch it was taken from and the positions it selects, naming the level each selection addresses. Take a batch labeled `posterior`, with a `chain` level of `(4,)` over a `draw` level of `(1000,)`:

```python
posterior.at_levels(chain=0).label          # "posterior[chain=0]"          — a sub-batch of draws
posterior.at_levels(chain=0, draw=7).label  # "posterior[chain=0, draw=7]"  — an element
posterior.at_levels(draw=slice(1, 3)).label # "posterior[draw=1:3]"         — both levels kept
```

The derived label states what was selected. Levels selected whole are left out, so selecting all of a batch derives the batch's own label, and the levels that appear are listed in the batch's own order; hence two ways of indexing one selection read alike, and two different selections read differently. Whether a batch *stores* an element outright or *materializes* it on demand, as columnar storage builds a row, indexing returns a view (II.4); storage is invisible to access. A *sub-batch* is a view in the same way, being the batch's own selection.

**Selecting by level.** `at_levels(**levels)` indexes a batch along its named levels as the by-name counterpart of positional `[]`. It is distinct from `select`, which splats a `Record`'s fields, and it returns an element only when the selection indexes down to one. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order: an integer drops its axis, and a slice or `None` keeps it. `None` stands for the whole axis here and only here, since a keyword cannot take a `:` literal and positional `[]` refuses `None`. A shorter tuple fills the leading axes and leaves the rest whole, so `draw=i` on a two-axis `draw` level means `draw=(i, None)`. A level whose axes are all dropped is removed, yielding the inner batch or element as positional indexing does; a level left unnamed is kept whole.

### Rationale

`Batch` satisfies `D1 – Mathematical fidelity` by keeping how many objects there are separate from what one object contains. Levels extend the same fidelity to collections of collections: how many objects are at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis: a user can say which multiplicity is which, and operations align batches by meaning rather than by position. A suffixing rule for a taken name, such as `draw2`, was rejected on the same ground: it would state the order the levels were added in, which is a fact about the computation rather than about meaning. The `batch_*` names are kept rather than a numpy-style `.shape` / `size`, which could cover the batch axes, the per-element content, or both. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). Indexing and iteration yield views at every level, to satisfy `D6 – Single source of truth`. The obligation falls on the concrete batch, since only it knows its storage, and its sub-batch method shares the store wherever the storage affords it (`B4 – No copying at boundaries`). Hence selecting all of a batch needs no special case: it calls the same method as any other selection, and a view costs nothing there. Indexing a batch by its elements' paths gives, at each path, the marginal batch of the quantity that path names in a draw, so a batch of draws, a draw, and their law share one set of paths (`C5 – Naming for unambiguous meaning`). The element name stays out of the batch's JAX structure because JAX compares that structure to decide whether two values share a compiled program: with the name in it, batches of draws named `x` and `y` would recompile apart and could not be combined by a JAX transform, so the name is kept where a merge reads it, in the element spec (`C3 – Computational detail hidden by default, available on demand`).

### Notes

- *Relation to xarray.* `at_levels` parallels `isel` in indexing by name and by position, with `level_names` in the role of dimension names; there is no label-based counterpart, since batch levels carry no coordinate labels, and a tuple here addresses a level's axes in order, where `isel` reads one as several positions of a single dimension.

## II.6 — `NamedTree`

### Contract

Structured values in ProbPipe are represented as named, ordered trees. A tree is made of these parts:
1. a **name** is a non-empty string without `/`;
2. a **node** is either the tree itself, its **root**, or an entry at any depth below it, and each entry has a name, unique among its siblings;
3. a **child** of a node is an entry directly under it;
4. a **field** is an entry that is not a tree, such as an array, an opaque value, or a law, and a **group** is an entry that is a tree;
5. the **path** of a node is the names from the root down to it, joined by `/`, as `g/b`, and the root has no path; a path of one name is **top-level**, and one of several names is nested;
6. the **canonical order** of the nodes is depth first, each node before its children, and siblings in insertion order.

Sibling names are distinct, so a path names at most one node, while distinct groups may reuse a name, as in `g1/theta` and `g2/theta`. Paths continue into the term a field holds: for a law `d` over `{mu, sigma}`, the record `{x: 3.0, g: d}` has the paths `x`, `g`, `g/mu`, and `g/sigma`, and its fields are `x` and `g` alone. A tree's paths are therefore the paths of its spec, which every term shares with its value type (II.2): a law, its draws, its mean, and a batch of its draws have one set of paths.

Naming, addressing, traversal, and structure-preserving transforms are defined once, in `NamedTree`. A `NamedTree` implements the `Mapping` interface with keys exactly its field paths, so `keys()`, `len`, iteration, `values()`, and `items()` range over the fields in canonical order, while `[]` and `in` range over every path.

```python
class NamedTree[L]:
    # mapping interface: keys are FIELD paths; [] and in take every path
    def __getitem__(self, paths: str | Sequence[str], /) -> L | Self: ...   # a path gives the node; a list of paths, the tree of the chosen nodes
    def __contains__(self, path: object, /) -> bool: ...   # whether path is a str and self[path] succeeds, groups included
    def __iter__(self) -> Iterator[str]: ...  # field paths
    def __len__(self) -> int: ...             # field count
    def keys(self) -> Iterable[str]: ...      # field paths, canonical order
    def values(self) -> Iterable[L]: ...
    def items(self) -> Iterable[tuple[str, L]]: ...

    # tree navigation
    def at_path(self, *segments: str) -> L | Self: ...   # self["/".join(segments)]; [] is preferred
    @property
    def children(self) -> Mapping[str, L | Self]: ...
    def is_field(self, path: str) -> bool: ...
    @property
    def is_multi_field(self) -> bool: ...   # True when the tree holds more than one field

    # structure-preserving transforms — return the same family
    def rename(self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str) -> Self: ...
    # rename or move nodes, old -> new, where old is the path of a node;
    # a new name may itself be a path, which moves the node there
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str, leaf: L | Self) -> Self: ...
    def __and__(self, other: Self, /) -> Self: ...         # this tree's children, then other's; a top-level name in both raises
    def without(self, *paths: str) -> Self: ...

    # serialization (export; the constructor reads a nested mapping back)
    def to_nested_dict(self) -> dict[str, Any]: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type: every navigator and transform returns the same family
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

The parameter `L` declares the leaf type, which is what `values()`, `[]` at a field, and `map` accept and return; groups are always the family's own class. Implementations should check leaves against the declared leaf type at construction.

**How `[]` reads its argument.** One reader parses what `[]` takes, for every kind that `[]` indexes by path (II.2, II.5, III.3, III.5, III.6, III.7, III.9, III.10):
1. a `str` is one path, and a `str` subclass is read as a `str`;
2. a list or a tuple of `str` is a **list of paths**, so `x["a", "g/b"]` and `x[["a", "g/b"]]` agree;
3. an `int`, an object with `__index__`, a `slice`, or a tuple of these is a position, which only a batch takes (II.5).

`bool`, `None`, and `...` raise `TypeError`. Every argument that takes one path, such as each path of `without(*paths)`, `at_path(*segments)`, and `is_field(path)`, is checked by the same reader, so `at_path(("g", "b"))` raises `TypeError`: "at_path takes each segment as a separate argument, got a tuple. Write at_path(*segments)". The reader refuses a malformed argument before anything is looked up:

| Argument | Error | Why |
|---|---|---|
| `x[""]`, `x["g//b"]`, `x["/a"]` | `ValueError` | an empty path or an empty name within one |
| `x[[]]`, `x[()]` | `ValueError` | no paths, on a batch too |
| `x["a", "a"]` | `ValueError` | the same path twice |
| `x["g", "g/b"]` | `ValueError` | "cannot select both 'g' and 'g/b', since 'g/b' is inside 'g'. Keep one of them" |
| `x["a", 0]` | `TypeError` | "cannot mix a path and a position in one [], got ('a', 0). Index the position and the path separately, as b[0]['a']" |
| `b[[0, 2]]`, `b[np.array([0, 2])]` | `TypeError` | "cannot index a batch by a list of positions, got [0, 2]. Index one position or a slice" |
| `x[{"a"}]`, `x[iter(["a"])]` | `TypeError` | a set has no order, and an iterator is read once |
| `x["z"]` | `KeyError` | "unknown path 'z' in 'x'; available paths: [...]" |

When a missing path comes from a tuple whose `/`-joined form is a path, the message adds "To select the nested path, write 'g/b'".

**Selection.** One path gives the node there: a field's value, a group as a tree of the same family, or, for a path that passes through a field holding a term, that term's selection at the rest of the path, so `{x: 3.0, g: d}["g/mu"]` is `d["mu"]`. A list of paths gives the tree of the chosen nodes, each at its own path, in the tree's canonical order whatever the order of the list, so `x["g/c", "a"]` is `{a, g: {c}}`. One path and a list of one path differ: `x["g"]` is the group, and `x[["g"]]` is the tree that holds it. A path in a list that passes through a field holding a term cuts that term down to the rest of the path, in its own kind. Paths chain, so `x["g"]["b"]` is `x["g/b"]`. `p in x` holds exactly when `p` is a `str` and `x[p]` succeeds, so `"g" in x` is true for a group `g`, although `"g"` is not among `x.keys()`.

**Removal.** `without(*paths)` removes the nodes at those paths and keeps every other node at its path. A group left empty goes too, and so does a held term from which every path is removed. Removing every node raises `ValueError`, and so do a repeated path and a path inside another. A list of paths and `without` partition the fields: each field path of `x` is in exactly one of `x[ps]` and `x.without(*ps)`.

**Merge.** `x & y` merges two trees of one family by top-level name: `x`'s children, then `y`'s, so `{a} & {b, g: {c}}` is `{a, b, g: {c}}`. A top-level name in both raises `ValueError`, a group included, whatever the two groups hold; to merge two trees that share `g`, merge the groups first, as `(x.without("g") & y.without("g")) & {g: x["g"] & y["g"]}`. An operand of another family returns `NotImplemented`, so Python raises `TypeError`.

`rename` renames the nodes *within* a tree, whereas `with_label` relabels the object itself (II.4). `at_path` has a level analogue in `Batch.at_levels` (II.5), and the two are alike: a path addresses a position and returns a field or a group, and named level indexers address positions and return an element or a sub-batch.

`rename` renames nodes by `old="new"` pairs. A **rename** gives the node at path `old` the path `new`, and the node keeps its children. In each pair, `old` is the path of a node, read by the reader above, so keyword pairs address top-level nodes and the positional mapping form addresses any node.

A target may itself be a path, so a rename can **move** a node to another parent: `rename({"group/mu": "mu"})` promotes the field to the top level, and `{"mu": "group/mu"}` demotes it under `group`, creating intermediate nodes as needed. A rename is *in place* when its target has the node's parent path and no ancestor of the node is itself renamed, so the node keeps its parent; every other rename moves the node. A group that the result leaves empty dissolves, since a tree holds no empty groups, and a target may then take its path.

All renames apply simultaneously: each `old` resolves against the original tree, so swaps such as `{"a": "b", "b": "a"}` and simultaneous moves of a node and its ancestor are well-defined.

Every target is checked against the result, and three cases raise:
1. a collision, as when a node is renamed onto an existing sibling;
2. a move into the moved node's own subtree;
3. two targets of which one is a prefix of the other.

Ordering is deterministic. An in-place rename keeps the node's position, and a moved node is appended to its new parent's children in the order the renames are given, with the mapping's entries before the keywords. A group that one move empties and another refills keeps its position.

For example, in `{"g": "h", "g/mu": "g/m"}` the original `g` becomes `h`. The target `g/m` keeps the parent path of `mu`, but `g` itself is renamed, so `mu` moves and is appended as `m` to a new group `g`.

### Rationale

Named paths satisfy `C5 – Naming for unambiguous meaning`, and keying a rename by path gives every node one address, fixed by its position in the tree. One reader of paths, shared by every kind, gives `[]` one meaning: a string is one path and a list is several, so a law, its draws, and a batch of its draws read the same argument alike (`C1 – Uniform interface to functions, distributions, and values`). A list keeps each node at its own path, so the result can be merged with terms that use those paths, and two nodes that share a last name in different groups are both kept (`D1 – Mathematical fidelity`). A merge by top-level name keeps each output whole, since a group is one output (II.2), and a deeper merge would let one operand supply only part of another's output. Housing the collection contract in one shared class ensures the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved (`C1 – Uniform interface to functions, distributions, and values`).

### Notes

- *Mappings are subtrees.* Construction should materialize a mapping value into a group. The need for this mapping check is why `NamedTree` implements the `Mapping` interface without registering as a `collections.abc.Mapping`, ensuring a nested mapping and its `to_nested_dict` export round-trip through the constructor.

## II.7 — Dispatch and registries

### Contract

Some operations have many possible implementations, held outside the classes they apply to, and which one applies depends on the *types* of the objects involved. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **dispatch method** declares:
1. a unique `name`;
2. the types it applies to, via `supported_types`;
3. a `check` function that probes feasibility without significant computation and reports, as a `Feasibility`, whether the call is feasible, infeasible, or **unresolved**, which means the declarations the probe needs are not yet available;
4. an `execute` function that performs it;
5. whether it is **exact**: a method either returns a representation of the requested result or a stand-in for it, declared where the method is registered and fixed for the method's life;
6. whether it is a **floor**, `is_floor`: a fallback on its supported types, which ranks after every method that is not a floor, declared where the method is registered and fixed for the method's life, and `False` by default;
7. a **priority**: an integer rank among the methods of the same floor status and exactness, and the one thing about a method a deployment may change at runtime;
8. optionally, a one-line `description`, which the registry catalog shows.

Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in **selection order** and runs the first whose `check` establishes feasibility. While a higher-ranked candidate is unresolved, the registry's `check` reports the call unresolved (V.1), and execution resolves prerequisite plans first or reports unavailable requirements (V.9). Selection order is the same in every registry:
1. every method that is not a floor before every floor, so auto-selection tries a floor only after every other candidate;
2. exact methods before approximate ones, so auto-selection runs an exact method whenever one that is not a floor is feasible;
3. priority among methods of the same floor status and exactness, higher first;
4. specificity, favoring the method whose declared types are closest to the arguments' classes in method-resolution order: a type that admits a class without appearing in its method-resolution order, such as a registered virtual subclass of an abstract base class, counts as least specific, and a binary registry sums the distances of its two arguments;
5. registration order.

The first criterion sets the floors apart as a tier of their own: a floor ranks after every other method a call admits, whatever their exactness and priority, as the sampling lift and the elementwise sweep do among the evaluation rules (V.7). The other four criteria order the methods within each tier in the same way.

**Priority.** A method whose priority is `None` is **opt-in-only**: auto-selection skips it, and a caller selects it by name. `None` is the default, so a method registered without a priority leaves auto-selection unchanged until a contributor ranks it. `set_priorities` re-ranks methods at runtime, by keyword or by a mapping, which accepts every method name. It keeps each method's declared exactness and floor status, so it moves a method only within its tier and its exactness, and it warns when a method moves into or out of opt-in-only.

**Listing.** `list_methods` returns the method names ranked by floor status, exactness, priority, and registration order. A call also ranks its candidates by specificity, before registration order, since specificity is measured against that call's arguments.

**Naming a method.** `method="..."` selects the named method in place of auto-selection. Naming skips the type pre-filter alone: the named method's `check` still decides feasibility, and a call with fewer positional arguments than the registry's arity raises `TypeError`, as it does without a name. A call with no feasible method raises `ResolutionError`, naming the methods tried and what each was missing, and an infeasible named method or an unregistered name raises the same. The registry's `check` and `execute` are the dispatch interface, so an unregistered name is a dispatch that cannot resolve, whereas `get_method` and `set_priorities` look a name up and raise `KeyError` for an unregistered one.

**Registration.** Methods are added by registration at import, by whichever layer owns the implementation, so a registry gains its providers without importing them. A registry reads a method's declarations once, at registration, and validates them before adding the method; thereafter it ranks and reports the declarations it read.

```python
type UnarySupportedTypes = tuple[type, ...]
type BinarySupportedTypes = tuple[tuple[type, ...], tuple[type, ...]]   # (left, right) types

class BaseDispatchMethod[SupportedTypesT](ABC):
    name: str
    exact: bool                   # declared at registration, fixed for the method's life
    is_floor: bool = False        # a fallback, ranked after every method that is not one; fixed for the method's life
    priority: int | None = None   # rank among methods of the same floor status and exactness, higher first; None is opt-in-only
    description: str = ""         # one line, shown by the registry catalog

    @abstractmethod
    def supported_types(self) -> SupportedTypesT: ...   # admitted by the registry's structural pre-filter
    @abstractmethod
    def check(self, *args, **kwargs) -> Feasibility: ...
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any: ...

class UnaryDispatchMethod(BaseDispatchMethod[UnarySupportedTypes]): ...     # the first argument determines admission
class BinaryDispatchMethod(BaseDispatchMethod[BinarySupportedTypes]): ...   # the first two arguments do

class Feasibility:                # what a method's check reports
    feasible:    bool | None   # None when required declarations are not yet available
    description: str           # why not, when infeasible
    pending:     tuple[str, ...]   # the missing declarations; non-empty exactly when feasible is None
    actionable:  bool          # infeasible on a detail of the call the caller can fix; a listing leads with it

class MethodInfo(Feasibility):    # what a registry's check reports
    method_name: str | None    # from the registration; both None exactly when no method was selected
    exact:       bool | None   # the selected method's declared exactness, never the check's

class ResolutionError(Exception): ...   # no available implementation under the requested controls
class MathematicalDomainError(ValueError): ...  # the mathematical operation is known to be undefined

class BaseDispatchRegistry[M: BaseDispatchMethod[Any]](ABC):
    # the public interface is concrete; arity subclasses supply key extraction and matching
    kind: str = "dispatch"        # with name and description, the SupportsRegistryCataloging identity
    def __init__(self, *, name: str = "", description: str = "") -> None: ...   # constructing does not catalog
    def register(self, method: M) -> None: ...
    def set_priorities(self, priorities: Mapping[str, int | None] | None = None, /,
                       **kwargs: int | None) -> None: ...   # ranks only; warns on a move into or out of opt-in-only
    def execute(self, *args, method: str | None = None, exact_only: bool = False, **kwargs) -> Any: ...   # auto-select, or run the named method
    def check(self, *args, method: str | None = None, exact_only: bool = False, **kwargs) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                           # names, ranked by floor status, exactness, priority, and registration order
    def entry_summaries(self) -> list[EntrySummary]: ...               # one per method, in list_methods order
    def describe_entry(self, name: str) -> EntrySummary: ...           # KeyError for an unknown name

class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]): ...  # keys on the first two
```

**Fidelity and validity.** Exact means that the returned representation denotes the requested mathematical result; for a draw-producing operation it means a draw exact in law, so two exact algorithms may draw differently. Approximate means a stand-in for that result. A sampling method may be either: a finite empirical posterior or pushforward is approximate, while sampling from that empirical law may be exact for the law it is. Approximate methods state their assumptions, such as integrability for a Monte Carlo mean. Finite MCMC output is approximate even when the chain has the desired invariant law, and asymptotic targets and convergence assumptions are recorded as method guarantees. Exactness is therefore a fact about a returned representation, so it is one boolean, `exact`, and `exact_only=True` excludes approximate methods.

Provenance records whether each step was exact and its target, preserving upstream approximation history. An earlier approximation stays recorded relative to its upstream target through exact downstream work, and a later exact calculation on the returned law stays locally exact. Derived routes and registry routes report the exactness of their selected implementation chain; a wrapper cannot upgrade it.

**Two failures.** Known mathematical undefinedness alone raises `MathematicalDomainError`, for example the mean of a `Cauchy`, whereas a moment that diverges is the extended real ±∞ (VII.1). Computational unavailability, including a missing capability and a failure to establish existence, raises `ResolutionError`, and a numerical execution failure propagates unchanged. Structural admission errors and return-contract defects are specified at their engine steps (V.4, V.10).

A single **catalog** makes every registry discoverable: it lists the registries and their entries, each entry with its exactness, its floor status, its priority, and a one-line description, so a user can see which entries exist and how a call will resolve. An **entry** is one registered item within a registry; the term is generic because the catalog spans registries of every `kind`, such as the dispatch registries and the bijector factory. A registry can be cataloged if it implements `SupportsRegistryCataloging`; satisfying the protocol is structural, and membership requires an explicit `register`. The operation vocabulary is cataloged the same way, so what ProbPipe can do and how a given call resolves are answered from one place.

```python
@dataclass(frozen=True)
class EntrySummary:
    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    exact: bool | None = None   # None when the registry declares no exactness per entry
    is_floor: bool = False      # a fallback, ranked after every entry that is not one
    @property
    def is_opt_in_only(self) -> bool: ...   # priority is None; meaningful only in a registry that ranks its entries

@dataclass(frozen=True)
class RegistryInfo:            # the catalog's per-registry record
    name: str                  # unique within the catalog
    description: str           # one line
    kind: str                  # e.g., "dispatch", "operation", "factory"
    entry_count: int

@runtime_checkable
class SupportsRegistryCataloging(Protocol):   # the identity members are read-only
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def kind(self) -> str: ...
    def entry_summaries(self) -> list[EntrySummary]: ...
    def describe_entry(self, name: str) -> EntrySummary: ...

class RegistryCatalog:
    def register(self, registry: SupportsRegistryCataloging) -> None: ...   # empty or duplicate names raise
    def names(self) -> list[str]: ...
    def list(self) -> list[RegistryInfo]: ...
    def __getitem__(self, name: str) -> SupportsRegistryCataloging: ...     # get the registry
    def __contains__(self, name: object) -> bool: ...                       # check if a registry exists
    def describe(self, name: str) -> str: ...                               # a readable summary of one registry
registry_catalog: RegistryCatalog   # the global instance
```

### Rationale

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` apply to operations whose implementation cannot be chosen from a single object alone. The guarded `check` probe keeps auto-selection safe, and `method="..."` leaves the choice in the user's hands. New implementations join by registration, so the supported set grows without changing the call sites (`D2 – Generality first`). Keeping undefinedness distinct from unavailability is `D1 – Mathematical fidelity`, and recording fidelity with its target serves `C6 – Traceable and reproducible workflows`. One catalog over every registry is `D6 – Single source of truth`: one place to see which implementations exist and how a call resolves.
