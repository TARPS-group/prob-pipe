# Plan: one contract for arrays

> **Status: proposed; not approved.**
> Written on 2026-10-31 against `main` at `98c55c87`. It builds on `paths-inputs-and-outputs.md`, which #689 adds.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Arrays".
> It overlaps `numeric-contract.md`, `batch-contract.md`, and `paths-inputs-and-outputs.md`; § 9 lists the edits each needs.
> Once approved, its rules move into `design/` (Part I B4, II.1, II.3, II.5, III.1, III.5, III.6, V.4, VI.3, VI.11, and the glossary) and its steps into the plans of the work after the workshop.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 11):

1. Whether array axes carry dimension names, optional and never generated (§ 5.2).
2. Whether a dimension name is the identity of a dimension everywhere, or a variable in the declarations of functions and kernels (§ 5.2).
3. Whether an operator on arrays aligns axes by position and checks their names, or aligns them by name as xarray does (§ 5.6).
4. Whether coordinates are held by terms and bound in the dimension scope, with no automatic alignment (§ 5.4).
5. Whether every numeric array is stored as a JAX array, converted once when it is wrapped, which changes B4 (§ 5.5).
6. Whether precision follows the user's JAX settings silently at the wrap (§ 5.5).
7. Whether a batch level is one axis, so that `axis_groups` goes (§ 5.7).
8. Whether `rename_dims` replaces both `with_dim_names` and `with_level_names` (§ 5.10).
9. Whether a pandas `DataFrame` is wrapped as a record of its columns (§ 5.12).
10. Whether dimension names are part of an array's JAX structure (§ 5.11).

---

## 1. Why this plan exists

An array is the most common value in ProbPipe: every parameter, every observation, and every draw of a numeric law is one.
No section of `design/` states, in one place, what an array is, how it is stored, how its axes are named, or how it enters JAX.
The rules are scattered over II.1, II.3, II.5, III.1, III.5, and V.4, and the code follows a different rule in each module.

One small example shows the result.
The same function gives a different answer depending on the library that holds its argument:

```python
g = Function(lambda x, y: x * y,
             input_spec=InputSpec(x=("n",), y=("n",)), output_spec=OutputSpec(z=("n",)))

g(jnp.arange(3.0), jnp.arange(3.0))      # NumericArray of shape (3,)

a = xr.DataArray(np.arange(3.0), dims=("obs",))
c = xr.DataArray(np.arange(3.0), dims=("time",))
g(a, c)
# ResultSchemaError: Function '𝒻' output/z has rank 2, expected rank 1 from shape ('n',)
```

The body received the two `DataArray`s, and xarray multiplied them by dimension name into a 3 × 3 array.
ProbPipe never read the names `obs` and `time`, so it could neither warn that they differ nor keep them in the result.

This plan gives arrays one contract:

1. what an array, an axis, and a dimension are;
2. how dimensions are named, sized, and labeled, for single arrays and for batches alike;
3. how an array is stored, and how it crosses into JAX and back out to NumPy, xarray, and pandas;
4. how a user selects along a dimension, by a separate interface from paths;
5. the classes and functions that implement it, each written once.

## 2. The design today

### 2.1 What `design/` states

| Section | What it states |
|---|---|
| Part I, B4 | "A wrap holds the representation it is given and raw access returns that same representation without copying." |
| II.1 | A shape entry is an integer or a **symbolic dimension**, a name such as `n_obs`. A name stands for one size within a **dimension scope**: a schema, a batch and its element, the inputs of one map, or the operands of a merge. `with_dim_sizes` substitutes sizes, so the name is gone afterwards. Two different unbound names at one axis are refused. |
| II.3 | The **array-backend registry** recognizes a numeric host, such as an xarray `DataArray` or a pandas object. It reports the host's shape and dtype without converting it, converts it "at the compute boundary", and reports its metadata. |
| II.5 | A batch's axes are tiled into named **levels**, and a level may hold several axes. `at_levels` selects by level name. Its note says that "batch levels carry no coordinate labels". |
| III.1 | `NumericArray` holds one array, and its `raw` is that array. Its operators return tracked terms. |
| III.5 | A record's leaves are "stored in native form, for example a bare array or an `xarray` container, and convert to `jax.Array` only at the compute boundary"; "each leaf converts at most once". |
| V.4 | A host's "named dimensions bind the spec's symbolic dimensions", and "its remaining metadata, such as coordinates and attributes, is carried as annotations". |
| V.5 | A parameter annotated with a raw class, such as `pd.Series`, receives a stored host of that class. |
| VI.11 | A binary operation on batches aligns their levels **by name**, with size-1 broadcasting. |
| P5, the backend's global settings | The array backend is JAX. Its precision setting belongs to the user, and ProbPipe never changes it. |

### 2.2 What the design leaves open

1. **Axes have no names.** An array's axes are positional. A symbolic name sizes an axis until a size is substituted, and then it is lost.
2. **Two kinds of name for one idea.** A level name says what a batch axis ranges over. A symbolic dimension name only says that two axes have one size. The design never relates them, so a batch's level `draw` and an element's axis `n` are in separate namespaces.
3. **Where conversion happens.** "At the compute boundary" names no point in the code, and "at most once" requires either a second, cached copy, which D6 forbids, or a conversion at every use.
4. **What the operators mean.** III.1 gives `NumericArray` the array operators without saying whether they follow NumPy, JAX, or the host's library.
5. **Coordinates.** V.4 keeps a host's coordinates as annotations, which no operation reads or carries forward.
6. **Export.** No section says how an array, a record, or a batch leaves ProbPipe as an xarray or pandas object.

## 3. The code today

### 3.1 Where the code agrees with the design

- Symbolic dimensions are parsed and bound as II.1 states, in one scope per check (`core/_shapes.py`, `core/_spec_base.py:_unify_array_shape`).
- Hosts are stored in their native type, and a registry (`core/_array_backend.py`) reads xarray and pandas hosts' shapes, dtypes, and metadata.
- Levels have names, `at_levels` selects by them, and VI.11's alignment by level name is implemented for batch operations.

### 3.2 Where the code departs from the design, or from itself

**R1. An operator follows the host's library.**
III.1's operators call the stored host's own operator, so the result depends on which library holds the array:

```python
a = NumericArray(xr.DataArray(np.arange(3.0), dims=("obs",)), label="a")
b = NumericArray(xr.DataArray(np.arange(4.0), dims=("time",)), label="b")
(a + b).spec        # NumericArraySpec(shape=(3, 4), dtype=float64): xarray broadcast by name
jnp.add(a, b)       # TypeError: add got incompatible shapes for broadcasting: (3,), (4,)

s = NumericArray(pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2]), label="s")
t = NumericArray(pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3]), label="t")
(s + t).raw.values  # array([nan, 3., 5., nan]): pandas aligned the two indexes
```

II.3 says that a foreign function and ProbPipe's own operator see one array, and here they disagree.
A function body receives the same hosts, so it disagrees too (§ 1).

**R2. The declared dtype is not the computed dtype.**

```python
w = NumericArray(np.ones(2), label="w")
w.spec              # NumericArraySpec(shape=(2,), dtype=float64)
w.as_jax().dtype    # float32, under JAX's default settings
```

The spec states a precision that no computation uses.
`as_jax()` caches its result and does not refresh it when the settings change (`core/_numeric_array.py:205`).

**R3. Wrapping changes the caller's array.**

```python
x = np.ones(3)
NumericArray(x, label="x")
x[0] = 5.0          # ValueError: assignment destination is read-only
```

`_read_only` (`core/_array_backend.py:361`) sets the flag on the caller's own array, which is a side effect against C2.

**R4. Names and coordinates are read and then ignored.**
The registry reads a `DataArray`'s dims and coords, but no spec, level, or operation uses them:

```python
da = xr.DataArray(np.zeros((2, 3)), dims=("chain", "obs"))
NumericArrayBatch(da, "draw", label="b").spec
# BatchSpec(NumericArraySpec(shape=(3,), dtype=float64), draw=2): the level is "draw", not "chain"
```

V.4's binding of symbolic dimensions from a host's names is not implemented.

**R5. A JAX transformation drops the host.**
Pytree flattening converts to a JAX array, so `jax.jit(lambda t: t)(a)` returns a `NumericArray` over a bare JAX array, and the dims are gone.
`_MappedBatchStore.of` keeps the native host for a batch and converts a single array (`core/_numeric_array_batch.py:474` and `:487`).

**R6. Conversion is written many times.**

| Job | Implementations |
|---|---|
| to a JAX array | `_to_jax_array` (registry-aware, 8 calls), `_array_utils._as_array`, `_dtype._as_float_array`, and 177 direct calls of `jnp.asarray` |
| to a NumPy array | `_to_numpy_array`, `Record.to_numpy` (its own dispatch, `core/record.py:1120`), `families/_programs._to_numpy` |
| is this a numeric array | `_is_numeric_leaf`, `_array_utils._is_numeric_array` (which accepts string arrays), two functions named `_is_numeric_dtype` with different rules |
| read a shape | `_event_shape_of`, and `getattr(x, "shape", None)` at 11 sites |
| broadcast batch shapes | inline in `functions/_boundary.py`, `functions/_sweep.py`, `functions/_plan.py`, and `families/_discrete.py` |

`NumericRecordBatch.__jax_array__` and `to_vector` call `jnp.asarray` directly and skip the registry (`core/_numeric_record_batch.py:224`).
Stacking a batch converts its elements with `jnp.stack` and loses their hosts (`core/_record_batch.py:1575`), so storage is native for some batches and JAX for others.

**R7. Three exports name dimensions three ways.**

| Export | Event dimension names | Path separator | Coordinates |
|---|---|---|---|
| `diagnostics/_arviz_bridge.py:152` | `dim_{i}` | `/` | none; values cast to `float` |
| `diagnostics/_datatree_store.py:150` | `{field}_dim_{i}` | `/` | none |
| `inference/_approximate_distribution.py:479` | `{name}_dim_{i}` | `.` | `chain` and `draw` |

**R8. "Array backend" names two things.**
`ArrayBackend` is the registry of hosts, and `SupportsArrayBackend` is a distribution's batched storage (`core/protocols.py:28`, `families/_backend.py:560`).

**R9. One sampling call mints one level of two axes.**

```python
sample(Normal("x", jnp.zeros(3), 1.0), sample_shape=(4, 5)).spec
# BatchSpec(NumericArraySpec(shape=(3,), ...), sample=(4, 5))
```

The two axes have one name between them, so neither can be selected or aligned by name.

## 4. Principles

1. **An axis may name what it ranges over.** A name is optional, and it means the same thing on a single array, on a batch's level, and in a spec. (C5)
2. **Positions compute, names check.** Arithmetic and every function body see positional JAX arrays, as NumPy and JAX define them. ProbPipe reads the names to check that axes match, and never to reorder them silently. (P10, II.3)
3. **ProbPipe aligns only what ProbPipe creates.** A batch level is created by ProbPipe when it lifts, samples, or sweeps, so ProbPipe aligns levels by name (VI.11). An event axis is computed on by user code, so ProbPipe aligns it by position.
4. **Structure in specs, labels in values.** A dimension's name and size are structure, so specs hold them. A dimension's coordinate labels are data, so terms hold them, and no spec does. (D5, D6)
5. **One storage.** A numeric array is stored as one JAX array. Every other library's array is converted when it is wrapped, and its names, labels, and attributes are read into ProbPipe's own structures. An export to that library is derived from the stored array. (D6, C2)
6. **Dimensions are addressed apart from paths.** A path names a part of a record-valued term. A dimension name names an axis. The two are selected by different methods and renamed by different methods. (C5)

## 5. The proposed design

### 5.1 Arrays, axes, and dimensions

An array holds numbers at the positions of a grid.
Each **axis** of the array runs along one direction of the grid.
What an axis ranges over is often a meaningful set: the 85 counties of a survey, the 1000 draws of a sampler, the 3 coefficients of a regression.
Naming that set lets a user, and ProbPipe, say which axis is which.

> [!IMPORTANT]
> **Definition (dimension).** A *dimension* is a named, finite set of positions that axes range over, such as the observations of a data set.
> 1. Each axis of an array either ranges over a dimension, named by the dimension's *name*, or is *anonymous*.
> 2. Every axis has a *size*, its number of positions, which a spec may leave unknown for a named axis.
> 3. A dimension may have a *coordinate*: one label for each position, such as the county names.
>
> Within one dimension scope (II.1), a name stands for one dimension, so every axis with that name has one size and one coordinate.

Consequences:

1. **A name is the identity of a set, not of an axis.** Two axes of one array may range over one dimension. A covariance matrix over counties has the dimensions `("county", "county")`, and both axes have the county names as labels.
2. **Anonymous axes are NumPy's axes.** An array with no names behaves as it does today, and every rule of this plan reduces to today's rule on it.
3. **A batch level is a dimension.** The draws of a sampler and the observations of a data set are both sets of positions. § 5.7 makes each level one named dimension, so batches and arrays share one mechanism.

### 5.2 Naming dimensions

**Names are optional.**
A constructor never requires a name, and ProbPipe never invents one for an event axis:

```python
x = NumericArray(jnp.zeros((100, 3)), label="x")
x.dims        # (Dim(None, 100), Dim(None, 3)): two anonymous axes

X = NumericArray(jnp.zeros((100, 3)), dims=("obs", "feature"), label="X")
X.dims        # (Dim('obs', 100), Dim('feature', 3))
```

A name is generated in two places only.
A level must have a name, because VI.11 aligns levels by name, so an operation that mints a level names it after itself, as II.5 states today (§ 5.7).
An export to a library that requires names, such as xarray, names each anonymous axis at the export, and stores nothing (§ 5.12).

**Decision 1.**
- **Recommendation.** Optional names, never generated for an event axis. A generated name such as `dim_0` says nothing about what the axis ranges over, so storing it would make a distinction that is not real (D1). It would also make two anonymous arrays of one shape disagree whenever their generated names differ.
- **Alternative.** Every axis has a name, generated as `dim_0`, `dim_1`, … when none is given, as xarray does. Every rule then compares names, and arrays from NumPy code carry names that a user never chose.

**A name is the dimension's identity.**
Two axes that meet in one scope, such as an argument and the input it is passed to, must range over the same dimension:

> [!IMPORTANT]
> **Rule (two axes at one position).** When two axes meet, as the axes of an argument and of the input declared for it:
> 1. if both are named, the names must be equal;
> 2. if one is anonymous, it takes the other's name;
> 3. if both are anonymous, the result is anonymous.
>
> The sizes meet as II.1 states. A mismatch of names raises `ValueError`, naming both dimensions and the fix.

This extends II.1's existing rule, which refuses two different unbound names at one axis, to names whose sizes are known.

```python
f = Function(lambda X, beta: X @ beta,
             input_spec=InputSpec(X=("obs", "feature"), beta=("feature",)),
             output_spec=OutputSpec(mu=("obs",)))

f(jnp.ones((100, 3)), jnp.ones(3)).dims          # (Dim('obs', 100),): anonymous axes take the declared names

data = xr.DataArray(np.ones((100, 3)), dims=("obs", "covariate"))
f(data, jnp.ones(3))
# ValueError: axis 1 of the argument 'X' ranges over 'covariate', but 'f' declares 'feature'
#   for it. Rename one of them, as data.rename_dims(covariate="feature") or
#   f.rename_dims(feature="covariate")
```

**Decision 2.**
- **Recommendation.** A name is the identity of a dimension everywhere: in values, in laws' events, and in the declarations of functions and kernels. A function written for any dimension is renamed onto the caller's dimension with `rename_dims`. IV.2 already shares names across the operands of a merge, which is the same rule.
- **Alternative.** In the declaration of a function or a kernel, a name is a *variable*, as a type variable is: it binds to whatever dimension the argument has, so `f` above would accept `covariate` and return an array over `obs`. A generic function then needs no rename, at the cost of two meanings for one name: a variable in a declaration and an identity everywhere else. The library's own declarations, such as the GLM likelihoods' `("obs", "features")` (`families/_conditional.py:579`), would accept any names under this alternative and would need renaming under the recommendation.

**Where names are stated.**

| Place | How |
|---|---|
| a spec | a shape entry: `NumericArraySpec(("obs", 3))`, or `Dim("obs", 100)` for a known size (§ 7.1) |
| an array | the `dims=` argument of `NumericArray` or `NumericArrayBatch`, or the names of a wrapped host (§ 5.12) |
| a law | its event spec, or `rename_dims` on the law; a family's constructor reads the dims of its array parameters (§ 5.9) |
| a batch's level | the level name (§ 5.7) |

### 5.3 Sizes, and symbolic dimensions

A *symbolic dimension* (II.1) is a named dimension whose size a spec leaves unknown.
With names on axes, binding a size no longer erases the name:

```python
s = NumericArraySpec(("n",))
s.free_dims                  # frozenset({'n'})
s.with_dim_sizes(n=3)        # NumericArraySpec((Dim('n', 3),)): the name stays
```

Consequences:

1. **`free_dims` is the set of names whose size is unknown**, as today, and a spec is *concrete* when that set is empty.
2. **An anonymous axis always has a size.** A spec cannot say "some size, unrelated to every other axis". Today's design has no such axis either.
3. **No arithmetic on sizes.** A size is an integer or unknown. `n + 2` stays out of scope, as `numeric-contract.md` § 4.5 records.
4. **A binding records the dimension.** The scope of a call binds `obs` to its size and, when one is given, its coordinate (§ 5.4). The private `_DimensionSizes` of `paths-inputs-and-outputs.md` § 5.6 becomes `_DimensionScope`, which holds both (§ 7.3).

### 5.4 Coordinates

A coordinate gives each position of a dimension a label, such as a county's name or an observation's date.
Labels let a user select by meaning, `x.sel(county="Aitkin")`, and let an export carry them to ArviZ.

> [!IMPORTANT]
> **Definition (coordinate).** A *coordinate* is a one-dimensional array of distinct labels, one per position of a named dimension.
> 1. A term holds a mapping `coords` from dimension names to coordinates, for some of the named dimensions of its spec.
> 2. A spec holds no coordinate.
> 3. Within one dimension scope, a name has at most one coordinate. A first coordinate binds, and a later one must be equal; a term without one takes the bound one.

Consequences:

1. **Labels are never computed on.** A coordinate is stored as a read-only NumPy array, which may hold strings, dates, or numbers. It never enters a JAX trace, as a label never does (II.4).
2. **One rule carries coordinates forward.** A result of any operation takes, for each named dimension of its spec, the coordinate bound to that name in the operation's scope. The rule covers calls, operators, merges, records, batches, sampling, and views without a case for each.
3. **No automatic alignment.** Two different coordinates for one name raise `ValueError`. xarray would align them by an inner join and drop the positions that one side lacks, which changes the data silently (P10). A user who wants alignment does it in xarray before wrapping.
4. **Selection keeps labels.** `x.isel(obs=slice(0, 10))` keeps the first ten labels of `obs`. A dimension that a selection removes takes its coordinate with it.
5. **Specs stay cheap to compare.** Equality, hashing, and JAX structure read specs, so they never compare label arrays (§ 5.11).

```python
y = NumericArray(jnp.arange(3.0), dims=("county",),
                 coords={"county": ["Aitkin", "Anoka", "Becker"]}, label="y")
y.sel(county="Anoka")              # NumericArray 'y[county=Anoka]', shape ()
(y + 1).coords["county"]           # Coordinate(['Aitkin', 'Anoka', 'Becker'])

z = NumericArray(jnp.ones(3), dims=("county",), coords={"county": ["A", "B", "C"]}, label="z")
y + z
# ValueError: the dimension 'county' has the coordinate ['Aitkin', 'Anoka', 'Becker'] in 'y' and
#   ['A', 'B', 'C'] in 'z'. Give both the same labels, or drop one with z.with_coords(county=None)
```

**Decision 4.**
- **Recommendation.** Coordinates held by terms and bound in the dimension scope, with equality required and no alignment.
- **Alternatives.** (a) Coordinates in the spec, which makes every spec comparison read label arrays and every distinct label set a new compiled program. (b) No coordinates, with a host's labels kept as annotations, as V.4 says today, so no operation carries them and every export loses them.

### 5.5 Storage: one JAX array

ProbPipe computes with JAX (P5), so every numeric computation needs a JAX array.
Today the stored array may be a NumPy, JAX, xarray, or pandas object, and each use converts it or computes with the host's own library (R1, R2, R5, R6).

> [!IMPORTANT]
> **Rule (storage).** Every numeric array that ProbPipe stores is one immutable `jax.Array`.
> 1. A `jax.Array` is stored as it is given, without a copy.
> 2. Any other numeric host, such as a NumPy array, an xarray `DataArray`, or a pandas `Series`, is converted once, when it is wrapped. Its dimension names, coordinates, and attributes are read into the term's dims, coords, and annotations (§ 5.12).
> 3. The stored dtype is the converted dtype, under the user's JAX settings, and the spec states that dtype.
> 4. `raw` returns the stored `jax.Array`. A NumPy, xarray, or pandas form is an export, built on request (§ 5.12).

Consequences:

1. **One meaning for operators and bodies.** A function body and an operator see the same JAX array, so R1 and § 1 cannot occur.
2. **The spec states the truth.** A spec's dtype is the dtype every computation uses (R2).
3. **Immutability without side effects.** JAX arrays are immutable, so ProbPipe never marks a caller's array read-only (R3), and a caller's later change to its NumPy array cannot change a term.
4. **One conversion point.** Every conversion runs through the array-host registry at the wrap. `as_jax()`, its cache, `_take_at`, and `_read_only` are removed (§ 8).
5. **Non-numeric arrays are opaque.** An array of strings, dates, or Python objects is not numeric, so it is an `Opaque` value, which keeps its host. Coordinates are the one place labels of those types appear in a numeric term.
6. **Out-of-core arrays are opaque.** A Dask- or Zarr-backed host that should not be loaded is wrapped as an `Opaque` value, and a function that loads it declares so in its body.

This changes Part I's B4, whose rule "a wrap holds the representation it is given" holds today for every host.
The new B4 reads: *a wrap of a `jax.Array` holds it without copying; a wrap of another numeric host converts it once; a view refers into its source and never copies.*
III.5's "convert to `jax.Array` only at the compute boundary" becomes "convert when wrapped".

A function that needs a host of another library, such as a body written for pandas, declares it with a raw-class annotation (V.5), and the call builds that host by the registry's export (§ 5.12):

```python
@function
def smooth(y: pd.Series) -> pd.Series:
    return y.rolling(7).mean()

smooth(NumericArray(jnp.arange(30.0), dims=("day",), label="y"))
# the body receives pd.Series(..., index=pd.RangeIndex(30, name="day"));
# the result is wrapped back, with the dimension 'day'
```

**Decision 5.**
- **Recommendation.** One JAX storage, as stated, with B4 and III.5 changed.
- **Alternative.** Native storage, as II.3 and III.5 state today, with one conversion function that every compute site calls. The host's operators still differ from JAX's (R1), and "converts at most once" still needs a cached second copy.

**Decision 6.**
- **Recommendation.** A float64 host becomes float32 at the wrap when JAX's 64-bit mode is off, without a warning, as `jnp.asarray` does. The spec then states float32, and a declared float64 spec is refused when the call opens (Part I, the backend's settings, rule 4).
- **Alternative.** Warn once per process when a wrap narrows a dtype.

### 5.6 Computation: positions compute, names check

User code sees positional JAX arrays.
ProbPipe's operators therefore compute by position too, and use the names only to check.

> [!IMPORTANT]
> **Rule (operators on arrays).** A binary operator on `NumericArray`s, such as `x + y`, broadcasts their axes by position, from the right, as NumPy does.
> 1. Each pair of aligned axes meets by the rule of § 5.2, so two different names at one position raise.
> 2. An axis that only one operand has keeps its name.
> 3. The result's coordinates follow § 5.4.
>
> The entries of `x + y` equal those of `jnp.add(x.raw, y.raw)` whenever the operator does not raise.

```python
X = NumericArray(jnp.ones((100, 3)), dims=("obs", "feature"), label="X")
b = NumericArray(jnp.ones(3), dims=("feature",), label="b")
(X * b).dims          # (Dim('obs', 100), Dim('feature', 3))

m = NumericArray(jnp.ones(100), dims=("obs",), label="m")
X * m
# ValueError: in 'X * m', axis 1 of 'X' ranges over 'feature' and axis 0 of 'm' over 'obs'.
#   Line up the axes first, as X.transpose("feature", "obs")
```

NumPy also fails here, because the sizes 3 and 100 differ.
With 3 observations and 3 features, NumPy would multiply along the features without an error, and the name check still raises.

`transpose(*dims)` reorders axes by name, and `expand_dims(name)` adds an axis of size 1, so a user lines up axes by name explicitly before an operator.

A NumPy or JAX function applied to a term, such as `jnp.sum(X, axis=0)`, receives the raw array and returns a bare array (II.3), so it keeps no names.
ProbPipe's tracked reductions, if added, would take dimension names (`X.sum("obs")`); this plan does not add them.

**Decision 3.**
- **Recommendation.** Positional broadcasting with a name check, as stated. It is the rule of PyTorch's named tensors, and it keeps `x + y` equal to the computation a function body performs.
- **Alternative.** Broadcasting by name, as xarray does: `X * m` would align `m` with `obs`. An operator would then differ from `jnp.multiply` on the same terms, which is R1 again.

**Levels are aligned by name.**
VI.11 aligns batch levels by name, and this plan keeps that rule.
The two rules agree with principle 3: a level exists because ProbPipe lifted, sampled, or swept, and ProbPipe performs the broadcast over it, so ProbPipe can align it by name without user code noticing.
An event axis is indexed by the user's code, so ProbPipe leaves its position alone.

### 5.7 Batches: a level is a dimension

A batch's level and an array's named axis now mean the same thing: a set of positions with a name, a size, and possibly labels.
This section makes them one mechanism.

> [!IMPORTANT]
> **Rule (levels).** Each level of a batch is one axis, which ranges over a named dimension.
> 1. `batch_dims` is the batch's tuple of `Dim`s, outermost first; `level_names` and `batch_shape` are the names and sizes it holds.
> 2. A level may have a coordinate, held in the batch's `coords`.
> 3. An operation that mints one level names it after itself, as `sample`. An operation that mints several from a tuple names them after itself and their positions, as `sample_0` and `sample_1`, and a mapping names them explicitly.
> 4. A `NumericArrayBatch`'s `dims` are its `batch_dims` followed by its element's dims, which is the tuple of its stored array.

```python
draws = sample(Normal("x", jnp.zeros(3), 1.0), sample_shape={"chain": 4, "draw": 1000})
draws.batch_dims      # (Dim('chain', 4), Dim('draw', 1000))
draws.dims            # (Dim('chain', 4), Dim('draw', 1000), Dim(None, 3))
draws.isel(chain=0)   # NumericArrayBatch 'x[chain=0]', levels {'draw': 1000}

sample(Normal("x", 0.0, 1.0), sample_shape=(4, 1000)).level_names   # ('sample_0', 'sample_1')
```

Consequences:

1. **`axis_groups` and `axes_per_level` are removed.** A level is one axis, so the tiling of II.5 and the tuple indexers of `at_levels` go. `len`, iteration, and positional `[]` address the leading level, which is now always the leading axis.
2. **One selection method for batches and arrays.** `isel` and `sel` (§ 5.10) replace `at_levels`, and select level dims and element dims alike on a `NumericArrayBatch`.
3. **One namespace.** A batch over `obs` of arrays over `obs` ranges over one dimension twice, as the matrix of § 5.1 does. A leave-one-out batch is an example: the level `obs` holds the left-out observation, and each element is an array over all of `obs`.
4. **A level's size has no second name.** A level of unknown size is written `draw=None`. A level and an element axis have one size exactly when they range over one dimension, as in `BatchSpec(NumericArraySpec(("n",)), n=None)` (Appendix B, item 5).
5. **Export needs no level table.** A `NumericArrayBatch` exports to a `DataArray` with `dims` as its dimensions, and a batch of records to a `DataTree` whose variables share the level dims (§ 5.12). R7's three conventions become one.

**Decision 7.**
- **Recommendation.** One axis per level. A `sample_shape` tuple and the tests build multi-axis levels, while the library's own batches use one axis per level, as `inference/_approximate_distribution.py` does for `chain` and `draw`.
- **Alternative.** Keep multi-axis levels, and give each of a level's axes its own `Dim` inside the level. A level then has a name and its axes have others, which is two names for one set of positions.

### 5.8 Records: one scope across fields

III.5 already makes a record's schema one dimension scope.
With names, a record over named arrays is what xarray calls a `Dataset`: its fields share dimensions, and a dimension has one size and one coordinate across them.

```python
data = Record({"X": X, "y": NumericArray(jnp.ones(100), dims=("obs",), label="y")}, label="data")
data.dim_sizes                   # {'obs': 100, 'feature': 3}
data.isel(obs=slice(0, 10))      # Record 'data[obs=0:10]': X of shape (10, 3) and y of shape (10,)
```

`isel` and `sel` on a record select the dimension in every field that has it, and leave the other fields whole.
A record's `coords` is the scope's coordinates.

### 5.9 Laws, kernels, and functions

These kinds hold dimensions in their specs, and they may hold coordinates for the named dimensions of those specs.

1. **Laws.** A law's event dims are in its event spec. A family's constructor reads the dims of its array parameters and broadcasts them by § 5.6, so `Normal("theta", mu, 1.0)` with `mu` over `county` is a law over `county`, with `mu`'s coordinate. Its draws carry the law's dims and coords, and the levels the sampling mints.
2. **Kernels.** The given side and the event share one scope (`numeric-contract.md` § 4.5). `condition_on` binds the given side's dims and coords from the given value, and the event's follow.
3. **Functions.** Inputs and outputs share one scope. A call binds dims and coords from its arguments, and the output takes them (§ 5.4, consequence 2). A function may hold coordinates for an output dimension that no input binds, such as the levels of a quantile.
4. **Selecting along a dimension of a law** is a marginal over positions, which this plan does not add. Laws, kernels, and functions get `rename_dims`, `with_dim_sizes`, and `with_coords`, and no `isel` or `sel`.

```python
mu = NumericArray(jnp.zeros(3), dims=("county",),
                  coords={"county": ["Aitkin", "Anoka", "Becker"]}, label="mu")
theta = Normal("theta", mu, 1.0)
theta.event_spec          # RecordSpec(theta=NumericArraySpec((Dim('county', 3),), float32, real))
sample(theta, sample_shape=500).coords["county"]   # Coordinate(['Aitkin', 'Anoka', 'Becker'])
```

### 5.10 Addressing by dimension, apart from paths

A path names a node of a record-valued term (`paths-inputs-and-outputs.md` § 4.1).
A dimension name names an axis.
The two never mix: `x["obs"]` is a path, and it raises on an array, as it does today.

| Method | On | What it does |
|---|---|---|
| `isel(**{dim: indexer})` | arrays, records, batches | selects positions along named dimensions: an integer removes the axis, and a slice or a sequence of integers keeps it |
| `sel(**{dim: label})` | the same | selects by coordinate labels, then calls `isel` |
| `rename_dims(**{old: new})` | every term and spec | renames dimensions in the spec, the coordinates, and the levels |
| `with_coords(**{dim: labels})` | every term | sets or drops (`None`) a dimension's coordinate |
| `transpose(*dims)`, `expand_dims(dim)` | arrays | reorders axes by name, or adds an axis of size 1 |
| `dims`, `batch_dims`, `dim_sizes`, `coords` | as § 7 states | reads the dimensions |

`rename_dims` replaces `with_dim_names` on specs, laws, and kernels, and `with_level_names` on batches, which `deferred.md` already proposed to rename (`rename_dims` and `rename_levels`).
Levels are dimensions, so one method serves both.
`rename` keeps its meaning for paths, inputs, and output names (`paths-inputs-and-outputs.md` § 4.7), and never renames a dimension.

A selection's derived label names the dimension and the position, as `at_levels` does today: `posterior.isel(chain=0)` is `posterior[chain=0]`, and `y.sel(county="Anoka")` is `y[county=Anoka]`.

**Decision 8.**
- **Recommendation.** One `rename_dims` for dimensions and levels.
- **Alternative.** `rename_dims` for event dimensions and `rename_levels` for levels, which keeps two names for one operation on one kind of object.

### 5.11 Crossing into JAX

Three rules govern dimensions inside JAX programs.

1. **Sizes are concrete in every trace.** A call binds every size before it traces (P1), so no symbolic size enters JAX. A compiled program is reused for every call with the same sizes, as JAX already caches by shape. A constructor that evaluates a callable abstractly, as `conditional_distribution` does (`numeric-contract.md` C6), evaluates it at the first call whose sizes are bound and keeps the result for those sizes. This settles `deferred.md`'s entry on symbolic shapes in a JAX trace; JAX's own symbolic shapes (`jax.export.symbolic_shape`) are not needed for it.
2. **Names are structure, coordinates are not.** A `NumericArray`'s pytree flattens to its stored array, with its spec, names included, as static data. A raw JAX transformation, such as `jax.jit(lambda t: t)(x)`, returns an array with the same dims and no coordinates. ProbPipe's own calls restore coordinates from the call's scope, as they restore labels (II.4).
3. **Vectors ignore names.** A vector layout ravels the stored array in row-major order (`numeric-contract.md` § 4.2), and `from_vector` restores the dims from the spec.

**Decision 10.**
- **Recommendation.** Names in the static data, as a record's field names are today. Two arrays that differ only in their names compile twice, which is rare and cheap.
- **Alternative.** Names out of the static data, as II.5 keeps a batch's element name out. A raw transformation then returns anonymous arrays, and ProbPipe's calls restore the names.

### 5.12 Hosts: reading and exporting other libraries' arrays

The **array-host registry** replaces the array-backend registry (R8).
An entry describes one library's array type in both directions:

1. **reading**, at the wrap: the data as a JAX array, the dimension names, the coordinates, and the attributes;
2. **exporting**, on request: a host of that type built from a stored array, its dims, and its coords.

| Host | Read as | Dimension names | Coordinates | Export |
|---|---|---|---|---|
| `jax.Array` | itself | none | none | `raw` |
| `numpy.ndarray` | a copy on the default device | none | none | `to_numpy()` |
| `xarray.DataArray` | its values | its dims | its 1-D index coordinates | `to_xarray()` |
| `xarray.Dataset` | a record of its variables | each variable's dims | the shared coordinates | `to_xarray()` on a record |
| `pandas.Series` | its values | the index's name, if it has one | the index, unless it is the default `RangeIndex` | the raw-class annotation (§ 5.5) |
| `pandas.DataFrame` | a record of its columns | the index's name, shared by every column | the index | the same |

Rules of the table:

1. **A non-index coordinate is an annotation.** xarray allows a coordinate that is not one label per position of one dimension, such as a 2-D coordinate. It is kept as an annotation (II.4), as V.4 states today.
2. **Attributes are annotations,** as today.
3. **An export names each anonymous axis** `"{output}_dim_{i}"`, the ArviZ convention, where `output` is the term's output name and `i` the axis's position among the element's axes. Nothing is stored.
4. **A record exports to a `DataTree`** whose groups are the record's groups and whose leaves are its fields, so paths map to DataTree paths and no separator is invented (R7). The ArviZ export of a posterior is this export of its draws, with `chain` and `draw` as levels.

```python
da = xr.DataArray(np.random.rand(4, 1000, 3), dims=("chain", "draw", "county"),
                  coords={"county": ["Aitkin", "Anoka", "Becker"]})
b = NumericArrayBatch(da, ("chain", "draw"), label="theta")
b.batch_dims              # (Dim('chain', 4), Dim('draw', 1000))
b.element_spec            # NumericArraySpec((Dim('county', 3),), float32)
b.to_xarray().dims        # ('chain', 'draw', 'county'), with the county coordinate

NumericArrayBatch(da, ("sample", "draw"), label="theta")
# ValueError: the level names ('sample', 'draw') disagree with the DataArray's leading
#   dimensions ('chain', 'draw'). Pass level_names=("chain", "draw"), or rename the DataArray's dims
```

**Decision 9.**
- **Recommendation.** A `DataFrame` wraps as a record of its columns, which keeps each column's name and dtype. `NumericArray(df)` still builds the 2-D array, with the column labels as the coordinate of a second dimension, named by `df.columns.name` or left anonymous.
- **Alternative.** A `DataFrame` wraps as a 2-D array, as the registry reads it today, which promotes all columns to one dtype and turns column names into labels.

## 6. An example from data to export

This example uses every rule above once, in the order a user meets them.

```python
radon = xr.Dataset(
    {"log_radon": ("obs", np.random.rand(919)),
     "county_idx": ("obs", np.random.randint(0, 85, 919))},
    coords={"obs": np.arange(919)})
data = Record(radon, label="data")                       # § 5.12: a record over the dimension obs
data.dim_sizes                                           # {'obs': 919}

county_names = [f"county_{i}" for i in range(85)]
alpha = Normal("alpha", NumericArray(jnp.zeros(85), dims=("county",),
                                     coords={"county": county_names}, label="mu"), 1.0)  # § 5.9

county_idx = data["county_idx"].raw                      # § 5.5: a JAX array
lik = conditional_distribution(
    lambda alpha: Normal("log_radon", alpha[county_idx], 0.5),
    given_spec={"alpha": ("county",)})                   # the body sees JAX arrays

posterior = condition_on(alpha & lik, data[["log_radon"]])   # the anonymous axis of 919 takes 'obs'
draws = sample(posterior, sample_shape={"chain": 4, "draw": 1000})   # § 5.7
draws["alpha"].sel(county="county_3")                    # § 5.10: a batch of 4 × 1000 scalars
draws.to_xarray()                                        # § 5.12: alpha over (chain, draw, county)
```

## 7. The API

Each block gives the new or changed public objects, with their docstrings and type hints.
A changed class shows only its changed members.

### 7.1 `core/_dims.py` (new): dimensions and shape arguments

```python
"""Dimensions: what an axis ranges over, and its size.

An axis of an array may name the dimension it ranges over, such as ``obs``
for the observations of a data set, and has a size. A shape argument, such
as the ``shape`` of ``NumericArraySpec`` or a ``sample_shape``, is read into a
tuple of ``Dim`` by ``_as_dims``, which every constructor calls.
"""


@dataclass(frozen=True, slots=True)
class Dim:
    """One axis: the name of the dimension it ranges over, and its size.

    A dimension is a set of positions that one or more axes range over, such
    as the observations of a data set or the draws of a sampler. An axis
    states its dimension by name, its size, or both:

    - ``Dim("obs", 100)``: an axis over the dimension ``obs``, of 100 positions;
    - ``Dim("obs")``: an axis over ``obs`` whose size is not known yet, which
      a call or ``with_dim_sizes`` binds later;
    - ``Dim(None, 3)``: an anonymous axis of size 3, as a NumPy array's axes are.

    Two axes with one name range over one dimension, so they have one size.

    Parameters
    ----------
    name : str or None
        The dimension's name: a non-empty string without ``/``. ``None`` makes
        the axis anonymous.
    size : int or None, default None
        The number of positions, or ``None`` when it is not known yet.

    Raises
    ------
    TypeError
        If *name* is not a string or ``None``, or *size* is not an integer or
        ``None``.
    ValueError
        If *name* and *size* are both ``None``, *name* is empty or contains
        ``/``, or *size* is negative.

    Examples
    --------
    >>> Dim("obs", 100)
    Dim('obs', 100)
    >>> Dim("obs").is_concrete
    False
    >>> Dim(None, None)
    Traceback (most recent call last):
    ValueError: a Dim needs a name or a size, and got neither. Pass Dim("obs") for a named axis of unknown size, or Dim(None, 3) for an anonymous axis of size 3
    """

    name: str | None
    size: int | None = None

    @property
    def is_concrete(self) -> bool:
        """Whether the size is known."""

    def with_size(self, size: int) -> Dim:
        """This dimension with its size set to *size*.

        Raises
        ------
        ValueError
            If the size is known and differs from *size*.
        """

    def with_name(self, name: str | None) -> Dim:
        """This axis with its dimension renamed to *name*, or made anonymous by ``None``."""


type ShapeArgument = (
    int | str | Dim | Sequence[int | str | Dim] | Mapping[str, int | None]
)
"""What a shape argument accepts.

An int is one anonymous axis, a str one named axis of unknown size, and a
``Dim`` one axis as it states. A sequence gives one axis per item. A mapping
gives one named axis per key, in order, with its size or ``None``.
"""


def _as_dims(shape: ShapeArgument, *, argument_name: str) -> tuple[Dim, ...]:
    """Read a shape argument into one ``Dim`` per axis.

    Replaces ``_shapes._as_shape`` and ``_shapes._as_dim``, and keeps their
    refusals: an iterator, a set, ``bytes``, and a ``bool`` are refused, and an
    integer of another type, such as ``numpy.int64``, is stored as an ``int``.

    Parameters
    ----------
    shape : ShapeArgument
        The argument as the caller gave it.
    argument_name : str
        The caller's parameter name, which messages name.

    Returns
    -------
    tuple of Dim
        One per axis, outermost first.

    Raises
    ------
    TypeError
        If *shape* or an item of it has a type a shape argument refuses.
    ValueError
        If a size is negative or a name is empty or contains ``/``.

    Examples
    --------
    >>> _as_dims(3, argument_name="shape")
    (Dim(None, 3),)
    >>> _as_dims(("obs", 3), argument_name="shape")
    (Dim('obs', None), Dim(None, 3))
    >>> _as_dims({"chain": 4, "draw": 1000}, argument_name="sample_shape")
    (Dim('chain', 4), Dim('draw', 1000))
    """


def _broadcast_dims(
    first: tuple[Dim, ...], second: tuple[Dim, ...], scope: _DimensionScope, *, path: str
) -> tuple[Dim, ...]:
    """The dims of two operands broadcast by position from the right, as NumPy broadcasts shapes.

    Each pair of aligned axes meets in *scope* (``_DimensionScope.meet``), so
    two different names at one position raise. An axis of size 1 broadcasts
    against any size and takes the other axis's name. It replaces the four
    inline uses of ``np.broadcast_shapes`` that broadcast batch shapes.

    Raises
    ------
    ValueError
        If two aligned axes have different names, or sizes that do not
        broadcast, naming *path* and both axes.

    Examples
    --------
    >>> _broadcast_dims((Dim("obs", 5), Dim("feature", 3)), (Dim("feature", 3),),
    ...                 _DimensionScope(), path="X * b")
    (Dim('obs', 5), Dim('feature', 3))
    """
```

### 7.2 `core/_coordinates.py` (new): coordinates

```python
@final
class Coordinate:
    """The labels of one dimension's positions, such as county names or dates.

    A coordinate holds one distinct label per position, in order. Its labels
    are stored as a read-only NumPy array and are never computed on, so they
    may be strings, dates, or numbers. Two coordinates are equal when their
    labels are equal in order.

    Parameters
    ----------
    labels : array-like
        One label per position: a sequence, a 1-D NumPy array, or a pandas
        ``Index``. The array is copied, so later changes to it do not change
        the coordinate.

    Raises
    ------
    ValueError
        If *labels* is not one-dimensional, or two labels are equal.

    Examples
    --------
    >>> county = Coordinate(["Aitkin", "Anoka", "Becker"])
    >>> len(county)
    3
    >>> county.positions_of("Anoka")
    1
    >>> county.positions_of(["Becker", "Aitkin"])
    array([2, 0])
    """

    def __init__(self, labels: ArrayLike, /) -> None: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

    @property
    def labels(self) -> np.ndarray:
        """The labels, as a read-only 1-D NumPy array."""

    def positions_of(self, labels: Any) -> int | slice | np.ndarray:
        """The positions of *labels*: one label, a sequence of labels, or a slice of labels.

        A slice of labels includes both ends, as pandas and xarray do.

        Raises
        ------
        KeyError
            If a label is not in the coordinate, naming it and the first few
            labels.
        """

    def take(self, positions: int | slice | Sequence[int]) -> Coordinate | None:
        """The coordinate of the positions an ``isel`` keeps, or ``None`` when an integer removes the axis."""
```

### 7.3 `core/_dimension_scope.py` (new): the dimension scope

This replaces `_DimensionSizes` of `paths-inputs-and-outputs.md` § 5.6, and keeps its interface for sizes.

```python
class _DimensionScope:
    """The dimensions bound so far in one dimension scope: each name's size and coordinate.

    A scope is the set of specs and terms one check or one operation relates,
    such as the arguments of one call or the operands of one merge. In it,
    each dimension name stands for one dimension: the first size and the first
    coordinate it meets bind, and a later one must be equal. The private forms
    of conformance, unification, and refinement bind into one scope as they
    run, and an operation reads the bound coordinates for its result.

    Parameters
    ----------
    sizes : mapping of str to int, optional
        Sizes already bound, by dimension name.

    Examples
    --------
    >>> scope = _DimensionScope()
    >>> scope.meet(Dim("obs", 3), Dim(None, 3), path="x")
    Dim('obs', 3)
    >>> scope.meet(Dim("obs"), Dim(None, 4), path="y")
    Traceback (most recent call last):
    ValueError: the dimension 'obs' at 'y' has size 4, but it was bound to 3. Rename one of the two dimensions, or give both the same size
    >>> scope.meet(Dim("obs", 3), Dim("time", 3), path="z")
    Traceback (most recent call last):
    ValueError: at 'z', one axis ranges over 'obs' and the other over 'time'. Rename one of them with rename_dims
    """

    def __init__(self, sizes: Mapping[str, int] | None = None, /) -> None: ...

    @property
    def sizes(self) -> Mapping[str, int]:
        """The bound sizes, by dimension name."""

    @property
    def coordinates(self) -> Mapping[str, Coordinate]:
        """The bound coordinates, by dimension name."""

    def meet(self, first: Dim, second: Dim, *, path: str) -> Dim:
        """The axis two axes at one position form, binding what it fixes.

        The names follow the rule of two axes at one position: equal names
        stay, an anonymous axis takes the other's name, and two different names
        raise. The sizes follow II.1: a known size binds an unknown one, and
        two known sizes must be equal. A failed meeting binds nothing.

        Raises
        ------
        ValueError
            If the names differ or the sizes disagree, naming *path*.
        """

    def bind_coordinate(self, dim_name: str, coordinate: Coordinate, *, path: str) -> None:
        """Bind *coordinate* to *dim_name*, or check it against the one bound.

        Raises
        ------
        ValueError
            If another coordinate is bound to *dim_name*, or the coordinate's
            length differs from the bound size, naming *path* and both.
        """

    def coordinates_for(self, dims: Iterable[Dim]) -> dict[str, Coordinate]:
        """The bound coordinates of the named dims among *dims*, for a result's ``coords``."""
```

### 7.4 `core/_array_hosts.py` (replaces `core/_array_backend.py`): the array-host registry

```python
class HostedArray(NamedTuple):
    """What the registry reads from a host: its data and what names it."""

    data: jax.Array                            # the converted entries
    dim_names: tuple[str | None, ...]          # one per axis; None for an anonymous axis
    coords: Mapping[str, Coordinate]           # by dimension name
    attributes: Mapping[str, Any]              # kept as annotations


@dataclass(frozen=True, slots=True, kw_only=True)
class ArrayHost:
    """How ProbPipe reads, and exports, the arrays of one library.

    A *host* is an array of a library other than JAX, such as an xarray
    ``DataArray`` or a pandas ``Series``. When a host is wrapped, ProbPipe calls
    ``read`` once and stores the converted array, so later computations use
    JAX alone. ``export`` builds a host of the same library from a stored
    array, for ``to_xarray``, ``to_numpy``, or a body that asks for that type.

    Parameters
    ----------
    host_type : type
        The library's array type.
    is_numeric : callable
        ``is_numeric(host) -> bool``: whether the host holds numbers. A
        non-numeric host is wrapped as an ``Opaque`` value.
    read : callable
        ``read(host) -> HostedArray``: the host's entries as a JAX array, under
        the user's JAX settings, with its dimension names, coordinates, and
        attributes.
    export : callable or None, default None
        ``export(data, dims, coords) -> host``: a host built from a stored
        NumPy array, its ``Dim``s, and its coordinates. ``None`` when the
        library is read-only to ProbPipe.

    Examples
    --------
    >>> host = array_host_for(xr.DataArray(np.zeros(3), dims=("obs",)))
    >>> host.read(xr.DataArray(np.zeros(3), dims=("obs",))).dim_names
    ('obs',)
    """

    host_type: type
    is_numeric: Callable[[Any], bool]
    read: Callable[[Any], HostedArray]
    export: Callable[[np.ndarray, tuple[Dim, ...], Mapping[str, Coordinate]], Any] | None = None


def register_array_host(host: ArrayHost) -> None:
    """Register how ProbPipe reads arrays of ``host.host_type``.

    Raises
    ------
    ValueError
        If another entry is registered for the same type.
    """


def array_host_for(value: Any) -> ArrayHost | None:
    """The entry for *value*'s type or its nearest registered base class, or ``None``.

    NumPy arrays and JAX arrays have built-in entries, so ``None`` means that
    *value* is not an array ProbPipe can read.
    """
```

`_is_numeric_leaf`, `_event_shape_of`, `_numpy_dtype_of`, `_to_jax_array`, `_to_numpy_array`, `_take_at`, `_read_only`, and `_metadata_of` are removed: a wrap calls `read` once, and every later reader uses the stored JAX array and the term's `dims` and `coords`.

### 7.5 `NumericArraySpec` (changed)

```python
@dataclass(frozen=True, slots=True, init=False)
class NumericArraySpec(NumericSpec):
    """The spec of one numeric array: its axes, its dtype, and its support.

    Parameters
    ----------
    shape : ShapeArgument, default ()
        One entry per axis: an int is an anonymous axis of that size, a str a
        named axis of unknown size, and a ``Dim`` an axis as it states. A
        mapping gives one named axis per key.
    dtype : dtype-like, optional
        The entries' dtype, or ``None`` to admit any.
    support : Constraint, optional
        The set the entries lie in. It describes the values and is never
        checked.

    Examples
    --------
    >>> NumericArraySpec(("obs", 3))
    NumericArraySpec(('obs', 3))
    >>> NumericArraySpec(("obs", 3)).with_dim_sizes(obs=100)
    NumericArraySpec((Dim('obs', 100), 3))
    >>> NumericArraySpec({"county": 85}).dim_names
    ('county',)
    """

    dims: tuple[Dim, ...]
    dtype: np.dtype | None
    support: Constraint | None

    def __init__(self, shape: ShapeArgument = (), dtype: DTypeLike | None = None,
                 support: Constraint | None = None) -> None: ...

    @property
    def shape(self) -> tuple[int | str, ...]:
        """One entry per axis: its size, or its name when the size is unknown.

        For a concrete spec this is the NumPy shape.
        """

    @property
    def dim_names(self) -> tuple[str | None, ...]:
        """The name of each axis's dimension, or ``None`` for an anonymous axis."""
```

`TermSpec` gains `rename_dims` in place of `with_dim_names`, with the same semantics extended to names whose size is known, and to levels:

```python
class TermSpec:
    def rename_dims(self, renames: Mapping[str, str] | None = None, /,
                    **keyword_renames: str) -> Self:
        """This spec with dimensions renamed, old name to new, all at once.

        Renames every axis and every batch level that ranges over an old
        name, whether its size is known or not. Two old names may swap.

        Raises
        ------
        ValueError
            If an old name is not a dimension of this spec, or a new name
            would give two dimensions one name.

        Examples
        --------
        >>> NumericArraySpec(("n", 3)).rename_dims(n="obs")
        NumericArraySpec(('obs', 3))
        """
```

`with_dim_sizes` keeps its signature and stops erasing names (§ 5.3).

### 7.6 `DimensionIndexed` (new): selection by dimension

One mixin implements selection by dimension for arrays, records, and batches, so the rules are written once.

```python
class DimensionIndexed(ABC):
    """Selection, renaming, and labeling along named dimensions.

    A term with named dimensions, such as an array over ``obs`` or a batch
    over ``chain`` and ``draw``, is selected along them by name with ``isel``,
    by labels with ``sel``, and relabeled with ``rename_dims`` and
    ``with_coords``. These methods never read paths: ``x["obs"]`` is a path,
    and ``x.isel(obs=0)`` selects a position.

    A subclass implements ``_dim_axes`` and ``_take_positions``; every public
    method is defined here.
    """

    @property
    def dim_sizes(self) -> Mapping[str, int]:
        """The size of each named dimension of this term, in the order the dimensions first occur.

        An anonymous axis is not listed. A dimension that several axes or
        fields range over is listed once.

        Examples
        --------
        >>> NumericArray(jnp.zeros((4, 4)), dims=("county", "county"), label="cov").dim_sizes
        {'county': 4}
        """

    @property
    def coords(self) -> Mapping[str, Coordinate]:
        """The coordinates of this term's named dimensions that have one, by name."""

    def isel(self, indexers: Mapping[str, Indexer] | None = None, /,
             **keyword_indexers: Indexer) -> Any:
        """Select positions along named dimensions, as a view.

        Each indexer is an integer, which removes the dimension's axes, or a
        slice or a sequence of integers, which keeps them. A dimension that
        several axes of one array range over cannot be selected by name;
        transpose or rename one of them first. A batch whose levels are all
        removed by integers gives its element. The result's label names the
        selection, as ``posterior[chain=0]``.

        Raises
        ------
        KeyError
            If a name is not a dimension of this term, naming the dimensions
            it has.
        ValueError
            If a name is the dimension of several axes of one array.
        IndexError
            If a position is out of range.

        Examples
        --------
        >>> X = NumericArray(jnp.zeros((100, 3)), dims=("obs", "feature"), label="X")
        >>> X.isel(obs=slice(0, 10)).dims
        (Dim('obs', 10), Dim('feature', 3))
        >>> X.isel(feature=0).label
        'X[feature=0]'
        """

    def sel(self, labels: Mapping[str, Any] | None = None, /, **keyword_labels: Any) -> Any:
        """Select along named dimensions by coordinate labels, as a view.

        Each value is one label, which removes the axis, or a sequence or a
        slice of labels, which keeps it; a slice includes both ends. The
        labels are turned into positions and passed to ``isel``.

        Raises
        ------
        KeyError
            If a dimension has no coordinate, or a label is not in it.

        Examples
        --------
        >>> y = NumericArray(jnp.arange(3.0), dims=("county",),
        ...                  coords={"county": ["Aitkin", "Anoka", "Becker"]}, label="y")
        >>> y.sel(county="Anoka").raw
        Array(1., dtype=float32)
        """

    def rename_dims(self, renames: Mapping[str, str] | None = None, /,
                    **keyword_renames: str) -> Self:
        """This term with dimensions renamed, in its spec, its levels, and its coordinates.

        Returns the same values under new dimension names, so a law renamed
        this way draws together with the law it is made from (V.5).
        """

    def with_coords(self, coords: Mapping[str, ArrayLike | Coordinate | None] | None = None, /,
                    **keyword_coords: ArrayLike | Coordinate | None) -> Self:
        """This term with the coordinates of the named dimensions set, or dropped by ``None``.

        Raises
        ------
        KeyError
            If a name is not a dimension of this term.
        ValueError
            If a coordinate's length differs from its dimension's size.
        """

    @abstractmethod
    def _dim_axes(self) -> Mapping[str, tuple[DimAxis, ...]]:
        """Where each named dimension occurs: for each name, the axes that range over it.

        A ``DimAxis`` names the stored array and the axis within it: the one
        array of a ``NumericArray`` or a ``NumericArrayBatch``, or a field's
        column of a record or a batch of records.
        """

    @abstractmethod
    def _take_positions(self, positions: Mapping[str, int | slice | np.ndarray],
                        *, label: str) -> Any:
        """The view selecting *positions* along each named dimension, under *label*."""
```

`Batch` implements it over its levels, `NumericArray` and `NumericArrayBatch` over all their axes, and `Record` and `RecordBatch` over their fields' axes and their levels.
`at_levels` and `with_level_names` are removed.
Laws, kernels, and functions do not implement the mixin (§ 5.9): they define `rename_dims`, `with_dim_sizes`, and `with_coords` with the docstrings above, each applied to their specs and stored coordinates.

### 7.7 `NumericArray` (changed)

```python
class NumericArray(TrackedTerm, Numeric, DimensionIndexed):
    """One numeric array, with its dimensions and their labels.

    Parameters
    ----------
    value : array-like
        The entries: a JAX array, which is stored as it is, or a NumPy array,
        an xarray ``DataArray``, a pandas ``Series``, or another registered
        host, which is converted to a JAX array once. A host's dimension
        names and coordinates become this array's, unless *dims* or *coords*
        is given.
    dims : sequence of str or None, optional
        The dimension of each axis, by name or ``None`` for an anonymous axis;
        the sizes are read from *value*. Defaults to the host's names, or to
        anonymous axes.
    coords : mapping of str to array-like, optional
        Labels for named dimensions.
    label : str
        The array's label.
    spec : NumericArraySpec, optional
        A declared spec, which the array must conform to; anonymous axes
        take its names.
    provenance : Provenance, optional

    Raises
    ------
    TypeError
        If *value* is not a numeric array, naming its type.
    ValueError
        If *dims* has the wrong number of axes, a coordinate has the wrong
        length, or the array does not conform to *spec*.

    Examples
    --------
    >>> X = NumericArray(np.zeros((100, 3)), dims=("obs", None), label="X")
    >>> X.dims
    (Dim('obs', 100), Dim(None, 3))
    >>> isinstance(X.raw, jax.Array)
    True
    """

    def __init__(self, value: ArrayLike, /, dims: Sequence[str | None] | None = None, *,
                 coords: Mapping[str, ArrayLike | Coordinate] | None = None,
                 label: str, spec: NumericArraySpec | None = None,
                 provenance: Provenance | None = None) -> None: ...

    @property
    def raw(self) -> jax.Array:
        """The stored JAX array."""

    @property
    def dims(self) -> tuple[Dim, ...]:
        """One ``Dim`` per axis, with its size."""

    def transpose(self, *dim_names: str) -> NumericArray:
        """This array with its named axes in the order *dim_names* gives.

        Anonymous axes keep their positions after the named ones. A dimension
        of several axes cannot be named here.
        """

    def expand_dims(self, dim_name: str, /, *, axis: int = 0) -> NumericArray:
        """This array with a new axis of size 1, ranging over *dim_name*, at *axis*."""

    def to_numpy(self) -> np.ndarray:
        """A NumPy copy of the entries."""

    def to_xarray(self) -> xr.DataArray:
        """A ``DataArray`` with this array's dims and coords.

        Each anonymous axis is named ``"{output}_dim_{i}"`` in the export.
        """
```

`as_jax()` is removed, since `raw` is the JAX array.
Indexing with `[]` by position and iteration keep their meaning from III.1 and return bare arrays.

### 7.8 `BatchSpec` and `Batch` (changed)

```python
class BatchSpec(TermSpec):
    """The spec of a batch: one element's spec, and one named dimension per level.

    Parameters
    ----------
    element : TermSpec
        The spec of one element.
    levels : mapping of str to int or None, optional
        The levels, outermost first, each name to its size or ``None``.
    **level_sizes : int or None
        The levels as keywords, each a size or ``None`` when it is not known
        yet. A level whose name is also a dimension of the element ranges
        over that dimension, so the two have one size.

    Examples
    --------
    >>> BatchSpec(NumericArraySpec(("obs",)), chain=4, draw=1000).batch_dims
    (Dim('chain', 4), Dim('draw', 1000))
    """

    element_spec: TermSpec
    batch_dims: tuple[Dim, ...]

    @property
    def level_names(self) -> tuple[str, ...]: ...
    @property
    def batch_shape(self) -> tuple[int | str, ...]: ...
```

`Batch` gains `batch_dims` and `coords`, implements `DimensionIndexed`, and loses `axis_groups`, `at_levels`, and `with_level_names`.
Its constructors take `level_names` and `coords`, and lose `axes_per_level`, so `batch-contract.md` § 4.10's signature becomes:

```python
(storage, /, level_names, *, element=None, coords=None, label=None, provenance=None)
```

### 7.9 Exports (new)

```python
class NumericArrayBatch:
    def to_xarray(self) -> xr.DataArray:
        """A ``DataArray`` over the levels and the element's axes, with every coordinate."""

class Record:
    def to_xarray(self) -> xr.DataTree:
        """A ``DataTree`` with one node per group and one variable per field, with every coordinate."""

class RecordBatch:
    def to_xarray(self) -> xr.DataTree:
        """A ``DataTree`` whose variables range over the levels, then each field's own axes."""

def to_inference_data(draws: RecordBatch) -> xr.DataTree:
    """The ArviZ form of a batch of draws: ``draws.to_xarray()`` placed in a ``posterior`` group.

    Requires the levels ``chain`` and ``draw``.
    """
```

`to_inference_data` replaces the three exports of R7.

## 8. What this removes

| Removed | Replaced by |
|---|---|
| `ArrayBackend`, `register_array_backend`, `array_backend_for` | `ArrayHost`, `register_array_host`, `array_host_for` (§ 7.4) |
| `NumericArray.as_jax`, `NumericArrayBatch.as_jax`, and their caches | `raw`, which is the JAX array |
| `_to_jax_array`, `_array_utils._as_array`, `_dtype._as_float_array`, and the `jnp.asarray` calls that convert stored values | the wrap's one `read` |
| `_to_numpy_array`, `Record.to_numpy`'s dispatch, `families/_programs._to_numpy` | `to_numpy()` |
| `_take_at`, `_read_only`, `_metadata_of`, `_event_shape_of`, `_numpy_dtype_of`, and the 11 `getattr(x, "shape", None)` reads | the stored JAX array, `dims`, and `coords` |
| `_shapes._as_shape`, `_as_dim`, `_as_levels`, `_axis_groups_for` | `_as_dims` |
| the four inline broadcasts of batch shapes | `_broadcast_dims` |
| `axis_groups`, `axes_per_level`, `at_levels`, `with_level_names`, `with_dim_names` | `batch_dims`, `isel`, `rename_dims` |
| `_array_utils._is_array`, which has no caller | nothing |
| the three exports of R7 | `to_xarray` and `to_inference_data` |

The numeric predicates of R6 are `numeric-contract.md` step N1's, and the stacking helpers are `batch-contract.md` step B4's.

## 9. Edits to the other plans and to `design/`

| Document | Edit |
|---|---|
| `paths-inputs-and-outputs.md` § 5.6 | `_DimensionSizes` becomes `_DimensionScope` (§ 7.3), and `bind_shared_dim` becomes `meet` on `Dim`s. The array rule of conformance adds the name rule of § 5.2. |
| `paths-inputs-and-outputs.md` § 4.7 | `rename` never renames a dimension; `rename_dims` does (§ 5.10). |
| `numeric-contract.md` § 4.2 | A layout's `Leaf` holds `dims` in place of `shape`, and `from_vector` restores them. |
| `numeric-contract.md` § 4.5 | The dimension contract's `with_dim_names` is `rename_dims`. Its gap on symbolic shapes in a JAX trace is settled by § 5.11, rule 1. |
| `numeric-contract.md` C7 | A record's inferred spec records the stored dtype, which § 5.5 makes the computed dtype. |
| `batch-contract.md` § 4.4 | The common spec requires equal dimension names; coordinates are not part of it. |
| `batch-contract.md` § 4.6, § 4.10 | `Batch.stack` takes `level_name`; constructors lose `axes_per_level` and take `coords`. |
| `batch-contract.md` § 4.7 | `_raw_columns` returns JAX arrays only, as § 5.5 stores them. |
| `deferred.md` | "Two different dimension names at one axis become one dimension" is answered by § 5.2: they stay refused. "Rename `with_level_names` and `with_dim_names`" is replaced by `rename_dims`. "Aligning the levels of two batches by name" is unchanged, and now covers coordinates. |
| `design/` | Part I B4; II.1 (dimensions, the name rule, `rename_dims`); II.3 (the array-host registry); II.5 (levels as dimensions, `isel`, `sel`, coordinates); III.1 (storage, `dims`, the operator rule); III.5 and III.6 (storage, a record as one scope); V.4 (the wrap reads names and coordinates); VI.3 (`sample_shape` names); VI.11 (coordinates in alignment); the glossary gains *anonymous axis*, *coordinate*, *dimension*, and *dimension name*, and *level* becomes "a named dimension of a batch's axes". |

## 10. Steps

Each step is one PR.

| Step | Delivers | Depends on | Changes user-visible behaviour |
|---|---|---|---|
| A1 | One JAX storage (§ 5.5): the array-host registry, conversion at the wrap, `raw` as the JAX array, and the removals of § 8 that concern conversion | — | yes: `raw` is a JAX array; a body receives JAX arrays unless it annotates a host type; a float64 NumPy array is stored as float32 when 64-bit mode is off |
| A2 | `Dim`, `_as_dims`, names in specs, the name rule in conformance, `rename_dims` on specs (§ 5.2, § 5.3, § 7.1, § 7.5) | the conformance step of `paths-inputs-and-outputs.md` | yes: `with_dim_sizes` keeps names, and two different names at one axis are refused |
| A3 | Names on arrays and records: `dims=`, the wrap's names, the operator rule, `DimensionIndexed.isel`, `transpose` (§ 5.6, § 5.8, § 7.6, § 7.7) | A1, A2 | yes: operators check names |
| A4 | Levels as dimensions: `batch_dims`, one axis per level, `isel` in place of `at_levels`, `rename_dims` in place of `with_level_names`, `sample_shape` names (§ 5.7, § 7.8) | A2, `batch-contract.md` B3 | yes: multi-axis levels go, and two methods are renamed |
| A5 | Coordinates: `Coordinate`, `_DimensionScope`'s coordinates, propagation, `sel`, `with_coords` (§ 5.4, § 7.2, § 7.3) | A3, A4 | yes: results carry coordinates, and two different coordinates for one name are refused |
| A6 | Exports and hosts in full: `to_xarray`, `to_numpy`, `to_inference_data`, `Dataset` and `DataFrame` wraps (§ 5.12, § 7.9) | A5 | yes: the ArviZ export's names and coordinates change |

**Tests.**
One module, `tests/core/test_array_contract.py`, asserts:

1. for every registered host, that a wrap stores a JAX array, reads the names and coordinates, and leaves the caller's host unchanged and writable;
2. that `x + y` equals `jnp.add(x.raw, y.raw)` on every pair of operands that the name rule admits, and raises on a table of mismatches;
3. that names survive `with_dim_sizes`, a call, sampling, a raw `jax.jit`, and `to_vector`/`from_vector`;
4. that coordinates follow § 5.4 through a call, a merge, a record, sampling, and `isel`, and are dropped by a raw JAX transformation;
5. that `isel` on a `NumericArrayBatch` commutes for level and element dims: `b.isel(chain=0).isel(obs=1)` equals `b.isel(obs=1).isel(chain=0)`;
6. that `NumericArray(da).to_xarray()` is identical to `da` for a float32 `DataArray` with index coordinates.

## 11. Questions for the maintainer

1. **Named dimensions** (§ 5.2). Recommended: optional names, never generated for an event axis.
2. **A name's meaning in declarations** (§ 5.2). Recommended: the identity of a dimension everywhere, with `rename_dims` for generic functions; the alternative makes declared names variables.
3. **Operators** (§ 5.6). Recommended: positional broadcasting with a name check; the alternative aligns by name, as xarray does.
4. **Coordinates** (§ 5.4). Recommended: held by terms, bound in the scope, equal or refused.
5. **Storage** (§ 5.5). Recommended: one JAX storage, converted at the wrap, with B4 and III.5 changed.
6. **Precision at the wrap** (§ 5.5). Recommended: follow JAX's settings without a warning.
7. **Levels** (§ 5.7). Recommended: one axis per level, with generated names for a tuple `sample_shape`.
8. **`rename_dims`** (§ 5.10). Recommended: one method for dimensions and levels.
9. **`DataFrame`** (§ 5.12). Recommended: a record of its columns.
10. **Names in JAX structure** (§ 5.11). Recommended: in the static data.

## Appendix A. Other designs considered

| Design | What it does | Why this plan departs from it |
|---|---|---|
| xarray | Every axis is named, generated as `dim_0` when not given. Arithmetic aligns by name and joins coordinates. | Name alignment makes operators differ from the JAX code in function bodies (§ 5.6). Joins drop data silently (§ 5.4). |
| PyTorch named tensors | Names are optional. Operators align by position and check that names unify, with an anonymous axis matching any name. | This plan adopts the rule (§ 5.6). |
| Haliax and Penzai (JAX) | Every axis is a named object with a size, and operations take axes by name. | This plan adopts the `Dim` of a name and a size (§ 7.1). It keeps names optional so that NumPy code runs unchanged. |
| PyMC | A model holds `coords`, and each variable names its `dims`; sizes follow from the coordinates. | This plan holds coordinates on terms rather than on one model object, since ProbPipe has no global model; the scope plays the model's role (§ 5.4). |
| jaxtyping | Names in annotations are size variables, bound per call. | This is the alternative of decision 2. |

## Appendix B. Readings this plan adopts

1. **Dimension names follow the rule for level names.** II.1 requires a dimension name to be an identifier, and II.5 lets a level name be any non-empty string without `/`. Since a level is now a dimension, the plan takes II.5's rule, and a name that is not an identifier is passed in a mapping, as `isel(**{"my level": 0})`.
2. **V.4's binding of symbolic dimensions from a host's names** is read as the name rule of § 5.2: a host's dim `obs` meets a declared `obs` and binds its size, and meets a declared `n` and raises.
3. **II.3's coordinate protocols** present the stored array, so `np.asarray(x)` and `jnp.asarray(x)` keep their meaning and return bare arrays without names.
4. **A NumPy array is copied at the wrap,** as `jnp.array(x, copy=True)` copies it, so a later change by the caller never changes a term, on any device.
5. **A level's size is never named apart from the level.** II.5's `BatchSpec(spec, draw="S")` names the level `draw` and its size `S`. A level is now a dimension, so its name is the only name its size has, and the spec is `BatchSpec(spec, draw=None)`. II.5's batch "square by declaration" is a batch whose level and element axis range over one dimension: `BatchSpec(NumericArraySpec(("n",)), n=None)`.
