# Inputs, outputs, paths, and merge

ProbPipe builds statistical workflows from values, laws, kernels, functions, and batches of these.
This document designs how they fit together:
1. **Names:** how the parts of a value are named, so that a law, its draws, and its summaries share one set of names.
2. **Inputs and outputs:** what each object receives and what it offers.
3. **Selecting:** how a part of any object is picked out, removed, or renamed.
4. **Merging:** how two objects combine into one by matching names, as a likelihood and a prior combine into a joint model.

Part I states the ideas without code; the design can be judged from it alone.
Part II gives the code, class by class.
The appendices hold the open questions, the implementation plan, the differences from today's code, the relation to earlier plans, the alternatives weighed, and the status.
Citations such as II.6 and D1 name sections and principles of [`design/`](../../design/README.md).

## Contents

**[Part I: The conceptual model](#part-i-the-conceptual-model)**

1. [Values and the places inside them](#1-values-and-the-places-inside-them)
   1. [Base values and records](#11-base-values-and-records)
   2. [Paths, nodes, fields, and groups](#12-paths-nodes-fields-and-groups)
   3. [The parts of a value](#13-the-parts-of-a-value)
2. [Objects, kinds, and types](#2-objects-kinds-and-types)
   1. [Objects and their kinds](#21-objects-and-their-kinds)
   2. [What an object produces](#22-what-an-object-produces)
   3. [The type of an object](#23-the-type-of-an-object)
   4. [The paths of every object](#24-the-paths-of-every-object)
   5. [When one type admits another](#25-when-one-type-admits-another)
3. [Inputs and outputs](#3-inputs-and-outputs)
   1. [Inputs](#31-inputs)
   2. [Outputs](#32-outputs)
   3. [How outputs are named](#33-how-outputs-are-named)
   4. [Input specs and output specs](#34-input-specs-and-output-specs)
   5. [Signatures and their figures](#35-signatures-and-their-figures)
   6. [Inputs and outputs compared](#36-inputs-and-outputs-compared)
4. [Selecting, removing, and renaming parts](#4-selecting-removing-and-renaming-parts)
   1. [Selecting one part](#41-selecting-one-part)
   2. [Selecting several parts](#42-selecting-several-parts)
   3. [Removing parts](#43-removing-parts)
   4. [Selecting from a type](#44-selecting-from-a-type)
   5. [Renaming and moving parts](#45-renaming-and-moving-parts)
   6. [The object at an output](#46-the-object-at-an-output)
5. [Merging objects by name](#5-merging-objects-by-name)
   1. [Why merge](#51-why-merge)
   2. [The definition](#52-the-definition)
   3. [Optional inputs](#53-optional-inputs)
   4. [The value of a merge](#54-the-value-of-a-merge)
   5. [Refused merges](#55-refused-merges)
   6. [Properties](#56-properties)
   7. [What merge is not](#57-what-merge-is-not)
6. [The kinds, one by one](#6-the-kinds-one-by-one)
   1. [Base values and records](#61-base-values-and-records)
   2. [Laws](#62-laws)
   3. [Kernels](#63-kernels)
   4. [Functions](#64-functions)
   5. [Batches](#65-batches)
7. [Invariants](#7-invariants)
   1. [The invariants](#71-the-invariants)
   2. [Exceptions on purpose](#72-exceptions-on-purpose)

**[Part II: The design in code](#part-ii-the-design-in-code)**

8. [Reading Part II](#8-reading-part-ii)
9. [Term specs](#9-term-specs)
10. [`InputSpec`](#10-inputspec)
11. [`OutputSpec`](#11-outputspec)
12. [Indexing in code](#12-indexing-in-code)
13. [Merging in code](#13-merging-in-code)
14. [Records: `Record` and `RecordSpec`](#14-records-record-and-recordspec)
15. [Laws](#15-laws)
16. [Kernels](#16-kernels)
17. [Functions](#17-functions)
18. [Batches](#18-batches)
19. [Errors and messages](#19-errors-and-messages)
20. [Symbolic dimensions in code](#20-symbolic-dimensions-in-code)

**Appendices**

- [A. Open questions and issues](#a-open-questions-and-issues)
- [B. Implementation plan and sequencing](#b-implementation-plan-and-sequencing)
- [C. Differences from today's code](#c-differences-from-todays-code)
- [D. Relation to previous plans](#d-relation-to-previous-plans)
- [E. Alternatives considered](#e-alternatives-considered)
- [F. Status and history](#f-status-and-history)

---

# Part I: The conceptual model

Part I uses no code.
Records are written `{a: 1.0, g: {b: 2.0}}`, laws by their family, as `mu ~ Normal(0, 1)`; other notation is defined where it first appears.

## 1. Values and the places inside them

### 1.1 Base values and records

> [!IMPORTANT]
> **Definition (base value).** A *base value* is an array, or an *opaque value*: any other value ProbPipe does not look inside, such as a string.

> [!IMPORTANT]
> **Definition (name).** A *name* is a non-empty string without `/`.

> [!IMPORTANT]
> **Definition (record).** A *record* is a tree of named values: each entry has a name, unique among its siblings, and holds a value or another record.

`2.5` and `"forest"` are base values; `{a: 1.0, g: {b: 2.0, c: 3.0}}` is a record.

### 1.2 Paths, nodes, fields, and groups

> [!IMPORTANT]
> **Definition (node, root, path).** A *node* is a place in a record; the *root* is the record itself.
> The *path* of a node is the names from the root down to it, joined by `/`, as `g/b`; the root has no path.

> [!IMPORTANT]
> **Definition (child, canonical order).** A *child* of a node is an entry directly under it.
> The *canonical order* visits children depth first, in the order they were inserted.

> [!IMPORTANT]
> **Definition (field, group).** A *field* is a node that is not a record.
> A *group* is a node, other than the root, that is a record.
> A *leaf path* is a path with no path below it.

![The record r drawn as a tree: the field a, and the group g holding the fields b and c.](../figures/record-tree.svg)
*Figure: the record `{a: 1.0, g: {b: 2.0, c: 3.0}}` has the paths `a`, `g`, `g/b`, `g/c`, the fields `a`, `g/b`, `g/c`, and the group `g`.*

> [!NOTE]
> Sibling names are distinct, so a path names at most one node; two groups may reuse a name, as `g1/theta` and `g2/theta`.

### 1.3 The parts of a value

> [!IMPORTANT]
> **Definition (part).** A *part* of a value is a node below its root.

> [!NOTE]
> A base value has no parts, so it has no paths: an array's entries are indexed by position, which is not a path.

## 2. Objects, kinds, and types

### 2.1 Objects and their kinds

> [!IMPORTANT]
> **Definition (object, kind).** An *object* is anything a workflow is built from: this law, this record.
> Its *kind* is one of six:
> 1. **base value** (§ 1.1);
> 2. **record** (§ 1.1);
> 3. **law:** a probability distribution;
> 4. **kernel:** a law for each value of what it conditions on, $K : S \to P(T)$;
> 5. **function:** a callable whose calls ProbPipe tracks;
> 6. **batch:** many objects of one kind, its *elements*, stacked along named axes, its *levels*; a *position* indexes the levels.
>
> Base values and records are *values*.

> [!IMPORTANT]
> **Definition (label).** An object's *label* is a display name. No operation reads it.

Examples: the law `mu ~ Normal(0, 1)`; the kernel that draws `y ~ Normal(mu, 1)` for each `mu`; a function from `x` to `{y, z}`; a batch of 100 draws of a law, over the level `sample`.

### 2.2 What an object produces

> [!IMPORTANT]
> **Definition (draw, event).** A *draw* is one value sampled from a law; the *event* of a law is what one draw is.

> [!IMPORTANT]
> **Definition (produced object).** What one evaluation of an object yields:
>
> | Kind | Produced object |
> | --- | --- |
> | base value, record | itself |
> | law | a draw |
> | kernel | at each value of what it conditions on, a draw |
> | function | at each value of its parameters, its return value |
> | batch | itself |

> [!NOTE]
> 1. A law's or kernel's produced object is random, so a statement about it is a statement about the law of a draw.
> 2. A law produces a draw, not itself; a joint law is a law over joint draws.

A law over `{mu, sigma}` produces records `{mu, sigma}`; a function from `x` to `{y, z}` produces, at each `x`, a record `{y, z}`.

### 2.3 The type of an object

> [!IMPORTANT]
> **Definition (type).** The *type* of an object states what all objects of that type share:
> 1. an *array type*: a shape, and optionally a dtype and a support;
> 2. an *opaque type*: optionally a Python type;
> 3. a *record type*: the record's tree, with a type at each field;
> 4. a *law type*: the *event type*, the type of one draw;
> 5. a *kernel type*: the types of what it conditions on, and its event type;
> 6. a *function type*: the types of its parameters and of its return value, either of which may be unknown;
> 7. a *batch type*: its element type and its levels.
>
> Array and opaque types are *base types*.

> [!IMPORTANT]
> **Definition (dimension name, dimension scope).** An array size may be a *dimension name*, such as `n`.
> Within one *dimension scope*, a name stands for one size: its first occurrence binds it, a later occurrence that disagrees is refused, and two different unbound names at one axis are refused (II.1).

> [!NOTE]
> Every law has an event type, whether or not it can be sampled; § 2.4 defines its paths from it.

The record type `{x: (n, m), w: (m,)}` is one dimension scope, so `m` is one size in both fields.

### 2.4 The paths of every object

> [!IMPORTANT]
> **Definition (paths of a type).** The *paths* of a type are:
> 1. for a record type, the paths of its tree, continuing below a field of law, kernel, function, or batch type into that type's paths;
> 2. for a law or kernel type, the paths of its event type;
> 3. for a function type, the paths of its return type;
> 4. for a batch type, the paths of its element type;
> 5. for a base type, none.
>
> The paths of an object are the paths of its type.

> [!IMPORTANT]
> **Definition (column).** The node of a batch at a path `p` is its *column* at `p`: the batch, over the same levels, of each element's node at `p`.

> [!NOTE]
> 1. A record's fields are only its own: a path through a field into a law is a path of the record, not a field.
> 2. A law over a single array has no path below its root.
> 3. A function whose return type is unknown has no paths known before it is called.

![The record r with the field x and the field g, which holds a law d over mu and sigma.](../figures/held-field.svg)
*Figure: for a law `d` over `{mu, sigma}`, the record `{x: 3.0, g: d}` has the paths `x`, `g`, `g/mu`, `g/sigma`; its fields are `x` and `g`.*

### 2.5 When one type admits another

> [!IMPORTANT]
> **Definition (conforms).** A type `A` *conforms* to a type `B` when every value of type `A` is admitted by `B`, within one dimension scope.

> [!IMPORTANT]
> **Definition (unify).** Two types *unify* when some type admits exactly the values both admit, within one dimension scope; that type is their *unification*.

> [!NOTE]
> 1. A type conforms to the unification of `A` and `B` exactly when it conforms to both, provided no two different unbound dimension names meet at one axis among the three.
> 2. Unification is symmetric, and associative where it exists.
> 3. A type given by sizes alone sets no dtype or support, so it conforms to and unifies with any type of its shape.

| `A` | `B` | Conforms | Unification |
| --- | --- | --- | --- |
| `(3,)`, `float32` | `(n,)` | yes, binding `n` to 3 | `(3,)`, `float32` |
| `()`, `int32` | `()`, `float32` | yes, a same-kind cast | `()`, `int32` |
| `()`, `float32` | `()`, `float64` | yes | `()`, `float32` |
| `(4,)` | `(3,)` | no | none |
| `(m,)` | `(n,)` | no: two unbound names | none |

<details>
<summary>Technical note: conformance by kind</summary>

`A` conforms to `B` when, in one dimension scope:
1. **arrays:** the ranks are equal, each axis agrees by the scope rule, and when both set a dtype, `A`'s casts into `B`'s by NumPy's same-kind rule; supports are not compared, since a support is descriptive (III.5);
2. **opaque values:** `B`'s Python type is unset or a superclass of `A`'s, and `B`'s metadata is unset or equal to `A`'s (III.2);
3. **records:** the same child names, each child of `A` conforming to `B`'s;
4. **laws:** `A`'s event type conforms to `B`'s;
5. **kernels and functions:** compared by their inputs and their output specs, which § 3.4 defines; its technical note states the rule;
6. **batches:** the same levels, and `A`'s element type conforms to `B`'s; a base element's name is not compared.

</details>

<details>
<summary>Technical note: unification by kind</summary>

1. **arrays:** equal ranks, each axis agreeing; a dtype only one sets, or, when both set one, the dtype of the type that conforms to the other, and the narrower by NumPy's safe cast when each conforms to the other; a support only when both set the same one;
2. **opaque values:** equal Python types, or one unset, and likewise the metadata (III.2);
3. **records:** the same child names, child by child;
4. **laws:** the unification of the event types;
5. **kernels and functions:** by their inputs and output specs, as the technical note of § 3.4 states;
6. **batches:** the same levels, and the unification of the element types.

Any other pair, such as an array type with a record type, does not unify.

</details>

## 3. Inputs and outputs

An object receives values at its inputs and offers what it produces at its outputs.

### 3.1 Inputs

> [!IMPORTANT]
> **Definition (input).** An *input* is a named place where an object receives one value.
> It has a name, which is a Python identifier; a type (§ 2.3); and a flag, *required* or *optional*.
> An *optional* input has a default, held by the object that owns the input.

> [!IMPORTANT]
> **Definition (binding).** *Binding* supplies values to some of an object's inputs, by name.
> An omitted optional input takes its default.

> [!NOTE]
> 1. Kernels have an input for each value they condition on, and functions one for each parameter. A batch of kernels or functions has its elements' inputs. Base values, records, and laws have none.
> 2. Binding a kernel's last required input gives a law.
> 3. An input of record type is one input, supplied with one record.

The kernel `a(y | mu, sigma)` draws `y ~ Normal(mu, sigma)`, with `sigma` optional and default 1.
Binding `mu = 0.3` gives the law `Normal(0.3, 1)`; binding also `sigma = 2` gives `Normal(0.3, 2)`.
The values of a binding are written as a record, `{mu: 0.3, sigma: 2.0}`.

<details>
<summary>Technical note: binding part of a record input</summary>

A binding may supply part of a record input by a path: for an input `g` of type `{b, c}`, it may supply `g/b` alone.
The input `g` then takes the rest, `{c}`, and becomes required, since a default is a whole value (III.9).

</details>

### 3.2 Outputs

> [!IMPORTANT]
> **Definition (output).** An *output* is a name under which an object offers part or all of what it produces (§ 2.2).
> The *value at an output* is that part, or the whole.

### 3.3 How outputs are named

A value with parts that also had a name outside would have two sets of paths.
If a law whose draws are `{mu, sigma}` were also named `params` from outside, its draws would be indexed by `mu` and the law by `params/mu`.
The naming rule gives each name one place.

> [!IMPORTANT]
> **Definition (naming rule).** A produced object offers:
> 1. **a base value:** itself, whole, under one name, stated by the object that produces it; this output is *at the root*;
> 2. **a record:** its children, under their names;
> 3. **a function:** the outputs of its return value, each as the function that returns the value there;
> 4. **a batch:** the outputs of its elements, each as the batch of the elements' values there;
> 5. **a law or a kernel:** nothing.
>
> An object's outputs are those of its produced object, so only a base value is ever named from outside.

> [!IMPORTANT]
> **Definition (wrap).** To offer a value that is not a base value under one name `n`, the object that produces it *wraps* it: it produces the record `{n: value}` instead.

A law's produced object is a draw, so a law offers its draws' outputs; rule 5 concerns a produced law, such as one a function returns.

![Three objects. A kernel k whose draw is an array offers it whole under the name y, written outside the box. A law d whose draws are records offers mu and sigma. The same law wrapped offers one output, params, holding mu and sigma.](../figures/output-naming.svg)
*Figure: a base value is offered whole under an outside name (rule 1); a record offers its children (rule 2); a wrap offers the record under one name.*

| Object | Offers | By rules |
| --- | --- | --- |
| kernel `k(y \| mu)`, whose draw is an array | `y`, at the root | 1 |
| law `d` with draws `{mu, sigma}` | `mu`, `sigma` | 2 |
| law with draws `{params: {mu, sigma}}` | `params`; its draws and the law have the path `params/mu` | 2 |
| function `f` from `x` to `{y, z}` | `y`, `z`; the value at `y` maps `x` to `f(x)`'s `y` | 3, 2 |
| function `h` returning a law | nothing; to offer it, `h` returns `{params: law}` | 3, 5 |
| 100 draws of `d` | `mu`, `sigma`, each a batch of 100 values | 4, 2 |
| 4 draws of a law over an array `x` | `x`, the whole batch | 4, 1 |

> [!NOTE]
> 1. An object offers either one output at the root or outputs that are all one-name paths, never both.
> 2. The name of an output at the root is not a path, since nothing is below it; every other output's name is its own path. So an object's outputs, its parts, and its draws' paths use one set of names.
> 3. A base value that no object produced, such as a lone array, has no outputs: nothing states its name, and a label is not one.

<details>
<summary>Technical note: why a function into records offers outputs and a law over records does not</summary>

A function `f` into `{y, z}` is determined by its output functions, since `f(x)` is `{y: f_y(x), z: f_z(x)}`, and a batch of records by its columns.
A law over `{mu, sigma}` is not determined by its marginals, since many joint laws share them.
Offering the marginals would let a later step receive `mu` and `sigma` as separate laws and lose their dependence (D1), so a produced law offers nothing, and so does a batch of laws.

</details>

<details>
<summary>Technical note: names of outputs and of inputs</summary>

An output's name is any name (§ 1.1), since it becomes a field; an input's name is also a Python identifier.
An output named `"n-obs"` therefore supplies no input.

</details>

### 3.4 Input specs and output specs

> [!IMPORTANT]
> **Definition (input spec).** An object's *input spec* lists its inputs in order: names, types, and which are optional.
> It holds no defaults. Two input specs are equal when they list the same inputs, types, and optional inputs, in any order.

> [!IMPORTANT]
> **Definition (output spec).** An object's *output spec* states the type of its produced object and, when that type is a base type, the name of the output at the root.
> A law's or kernel's output spec is its *event spec*.

> [!IMPORTANT]
> **Definition (type of an output).** The type of the value at that output.

> [!IMPORTANT]
> **Definition (type hole).** An output spec may name the output at the root before its type is known; the pending type is a *type hole*, filled once the object knows it (§ 11).

> [!IMPORTANT]
> **Definition (record of outputs).** A produced base value `v` offered under `n` is read as the record `{n: v}`; a produced record is its own record of outputs.

> [!NOTE]
> 1. The type alone decides whether an output spec names an output at the root, so no flag records it.
> 2. Each output's type is read from the output spec: the produced object's type at the root, the child's type for a record, a function type for a function's output, a batch type for a batch's output.
> 3. A produced base value or record is rebuilt from its record of outputs. A produced function, batch, or law has none.

The kernel `a(y | mu)` has an output spec stating an array type of shape `()` and the name `y`; its draw `0.3` is the record of outputs `{y: 0.3}`.
A law over `{mu, sigma}` has an event spec stating the record type `{mu, sigma}` and no name.

<details>
<summary>Technical note: equality and order</summary>

Output order is the field order of the record of outputs, which fixes a numeric record's flat layout, so output specs listing the same outputs in two orders are unequal.
An output spec naming a base value `n` and one whose type is `{n: …}` offer the same output and are unequal.

</details>

<details>
<summary>Technical note: conformance and unification of output specs, kernels, and functions</summary>

This completes the rules of § 2.5.
- **Output specs:** `A` conforms to `B` when they state the same root name, if any, and `A`'s type conforms to `B`'s; a type hole in `B` admits any type. Two output specs unify likewise, and a type hole takes the other's type.
- **Kernels and functions:** `A` conforms to `B` when they have the same input names; each of `B`'s input types conforms to `A`'s, and an input optional in `B` is optional in `A`; `A`'s output spec conforms to `B`'s; a side that `B` leaves unknown admits any, and a side only `A` leaves unknown does not conform. They unify when they have the same inputs, with equal types and optional flags, and output specs that unify; a side one leaves unknown takes the other's.

</details>

### 3.5 Signatures and their figures

> [!IMPORTANT]
> **Definition (signature).** An object's *signature* is `name(outputs | inputs)`, as `a(y | mu, sigma)`; with no inputs, `name(outputs)`, as `d(mu, sigma)`.

Figures read from right to left, as a density $p(y \mid \mu)$ does: inputs enter on the right edge, outputs leave on the left.

![The legend for objects: data flows right to left; an object is a rounded box with a kind badge; outputs are ports on the left edge; an output at the root has its name outside the box; inputs are ports on the right edge, hollow when optional; a value's parts are a tree inside the box; a batch is a stack of boxes.](../figures/legend-objects.svg)
*Figure: how a figure draws an object.*

![The kernel k with the output y outside its left edge, the input mu, and the optional input sigma as a hollow port.](../figures/signature.svg)
*Figure: the kernel `k(y | mu, sigma)`; its draw is an array offered under `y`, and `sigma` is optional.*

![The record r with the outputs a and g; g holds the fields b and c.](../figures/record-parts.svg)
*Figure: the record `{a: 1.0, g: {b: 2.0, c: 3.0}}` offers its children `a` and `g`.*

### 3.6 Inputs and outputs compared

| | Inputs | Outputs | Why |
| --- | --- | --- | --- |
| Naming | every input is named | only a base value is named from outside | values are supplied to inputs by name; a produced value with parts carries its own names |
| Names allowed | identifiers | any name | an input matches a parameter; an output becomes a field |
| Structure | flat | one root output, or top-level paths with paths below | an input receives one value; an output is a place in a produced object |
| Presence | may be optional | always present | a produced object has a value at every output |
| Type unknown | never | type hole | a binding is checked against the input's type |
| Equality | ignores order | keeps order | binding is by name; output order fixes layout |
| None at all | allowed | never an empty record | an empty record describes nothing produced |
| Batches | the elements' inputs | the elements' outputs, as batches | one binding supplies every element |

## 4. Selecting, removing, and renaming parts

### 4.1 Selecting one part

> [!IMPORTANT]
> **Definition (selection).** For an object `x` and a path `p` of `x`, the *selection* `x[p]` is:
> - for a record, the node at `p` itself;
> - for a law, a kernel, a function, or a batch, the object of the same kind whose produced object is the node at `p` of `x`'s produced object.
>
> A batch is also selected by a position, giving an element or a smaller batch.

> [!IMPORTANT]
> **Definition (marginal law).** The selection `d[p]` of a law `d` is its *marginal law* at `p`: the law of a draw's node at `p`.

So a kernel's selection is the kernel whose law at each conditioning value is the law of a draw's node at `p`; a function's is the function mapping each argument `w` to the node at `p` of its return value; a batch's is its column at `p` (§ 2.4).

> [!NOTE]
> 1. **Selecting commutes with producing:** a draw of `d[p]` is distributed as a draw of `d` at `p`, at every path, including paths through fields that hold objects; `f[p](w)` is `f(w)` at `p`.
> 2. **Paths chain:** `x[p][q]` is `x[p/q]`, through a field that holds a law, a function, or a batch.
> 3. **Inputs are kept:** selecting never changes what an object conditions on or takes.
> 4. **Positions and paths commute** on a batch: the element at position `j` of `b[p]` is the element at `j` of `b`, at `p`.
> 5. **A selection at a base node** offers it under the path's last name; **a selection at a node with parts** offers its parts.

![Four laws: d over {a, g: {b, c}}; d[g], offering b and c; d[[g]], offering g; d[g/b], offering b at the root.](../figures/select-one.svg)
*Figure: a law `d` over `{a, g: {b, c}}`, its selections `d[g]` and `d[g/b]`, and its selection `d[[g]]` at a list of one path (§ 4.2).*

For a function `f` from `x` to `{y, z}`, `f[y]` maps `x` to the field `y` of `f(x)`.
For the record `r = {x: 3.0, g: d}`, `r[g/mu]` is `d[mu]`, the marginal law of `mu`.

![The function f from x to {y, z}, and f[y], from x to an array y.](../figures/function-outputs.svg)
*Figure: selecting a function selects in its return value; the input is unchanged.*

### 4.2 Selecting several parts

> [!IMPORTANT]
> **Definition (selection at a list of paths).** `x[[p, q, …]]` is the record of the chosen nodes, each at its own path, in `x`'s canonical order; for a law, a kernel, a function, or a batch, it is the object whose produced object is that record.
> An empty list, a repeated path, and a path inside another are refused.

> [!NOTE]
> 1. One path gives the node; a list of one path gives a record that holds it: `d[g/b]` draws arrays, while `d[[g/b]]` draws `{g: {b}}`.
> 2. Same-named nodes in two groups are both kept: for a law over `{g1: {theta}, g2: {theta}}`, `d[[g1/theta, g2/theta]]` is the law over `{g1: {theta}, g2: {theta}}`.
> 3. The order is `x`'s, whatever the order of the list: `d[[g/c, a, g/b]]` is over `{a, g: {b, c}}`.
> 4. A value selected at a list is labelled by its paths, as `(a, g/b)`.

<details>
<summary>Technical note: a list through a field that holds an object</summary>

A path in a list that enters a field holding a law, a function, or a batch cuts that object down to the rest of the path, in its own kind.
For `r = {x: 3.0, g: d}`, `r[[x, g/mu]]` is the record `{x, g}` whose `g` holds `d[[mu]]`.

</details>

<details>
<summary>Technical note: the name of the output at the root in a list</summary>

A list may name the output at the root (§ 3.3), at the top level only, and then gives the record `{n: …}` or the object producing it: for a law `d` whose draws are arrays named `x`, `d[[x]]` is the law over `{x}`.
The name alone is not a path, so `d[x]` is refused, and § 4.6 reads the object at that output.

</details>

### 4.3 Removing parts

> [!IMPORTANT]
> **Definition (removal).** `x without p, q, …` is `x` in its own kind with the nodes at the given paths removed and every other node at its path.
> A group, or a field holding an object, all of whose leaf paths are removed goes too; a record stays a record however little is left; removing everything is refused; inputs are unchanged.

> [!NOTE]
> Selection and removal partition the leaf paths: each leaf path of `x` is in exactly one of `x[[ps]]` and `x without ps`.

For a law `d` over `{a, g: {b, c}}`: `d without g/c` is over `{a, g: {b}}`, the complement of `d[[a, g/b]]`, and `d without g/b, g/c` is over `{a}`.

### 4.4 Selecting from a type

The output spec of `x` (§ 3.4) determines the output spec of every selection of `x`.

> [!IMPORTANT]
> **Definition (selection from an output spec).** For an output spec `O` and a path or list `s`, `O[s]` is:
> 1. at a path to a base node, that node's type, under the path's last name;
> 2. at a path to a node with parts, that node's type, offering its parts;
> 3. at a path at or below a field holding a law, kernel, function, or batch, the type of an object of that kind whose produced object is the node there;
> 4. at a list of paths, the record type of the chosen nodes at their paths, and at a list that names the output at the root, `{n: type}`.
>
> The name of the output at the root alone is not a path, and is refused.

> [!NOTE]
> 1. The output spec of `x[s]` is `O[s]`; for a record, the type of `x[s]` is the type of `x` at `s`.
> 2. `O[s]` holds no name that `O` does not hold.
> 3. Dimension names are kept, in the scope of `O`; binding a size commutes with selecting.
> 4. Removal from an output spec follows removal from objects (§ 4.3).

For a law over `{a, g: {b, c}}`: `O[g/b]` is an array type named `b`; `O[g]` is the record type `{b, c}`, offering `b` and `c`; `O[[g]]` is the record type `{g: {b, c}}`.

<details>
<summary>Technical note: selecting from a law, kernel, function, or batch type</summary>

For a law type with event spec `E`, the type at a path `p` is the law type whose event spec is `E[p]`; the same holds for a kernel type and a function type, which keep their inputs, and for a batch type, which keeps its levels.
So the type at `p` of a record type whose field `g` holds a law type is, at `g/p`, the law type whose event spec is that law's event spec at `p`.
A list `ps` keeps each child that a path names whole, cuts each child a path enters down to the rest of its paths, and inside a type that a field holds cuts that type's event spec the same way.
A path through a field holding a law whose draws are arrays named `x`, such as `g/x`, is no path: `x` names the draw at the root.

</details>

### 4.5 Renaming and moving parts

> [!IMPORTANT]
> **Definition (rename).** A *rename* gives the node at a path a new path; a new path in another group moves the node there.
> Its keys are the object's paths and the name of its output at the root. All keys apply at once, by II.6's rules, and a collision is refused.

> [!NOTE]
> 1. No key is both a name and a path, since the output at the root has no path below its name and every other output's name is its path.
> 2. The name of an output at the root cannot become a nested path, since a base value has no record to hold it.
> 3. On a batch of base values, renaming the elements' root name renames the batch's output.

For a law over `{a, g: {b, c}}`, the rename `g/b → g/z` renames in place and `g/b → b` moves `b` to the top level; for a law whose draws are arrays named `x`, `x → y` renames the output, while `x → g/y` is refused.

<details>
<summary>Technical note: how renames apply</summary>

II.6 states the rules: every key is resolved against the original tree, so swaps are well defined; a group a move empties dissolves; a collision, a move into the moved node's own subtree, and two targets of which one is a prefix of the other are refused; an in-place rename keeps the node's position, and a moved node is appended to its new parent.

</details>

### 4.6 The object at an output

> [!IMPORTANT]
> **Definition (object at an output).** The *object at* an output `c` of `x` is `x[c]` when `c` is a path, and `x` itself when `c` is at the root.

> [!NOTE]
> 1. The type of output `c` is the type of the produced object of the object at `c`; for a record or a batch, which produce themselves, it is that object's own type, so for `{x: 3.0, g: d}` the output `g` has `d`'s type.
> 2. An object whose produced object is a law offers no outputs, so it has no object at an output.

For a law whose draws are arrays named `x`, the object at `x` is the law; for a law over `{a, g: {b, c}}`, the object at `g` is `d[g]`; for 4 draws of the first law, the object at `x` is the batch.

## 5. Merging objects by name

### 5.1 Why merge

A model is often a product of densities joined by shared names:

$$p(y, \mu, \tau) = p(y \mid \mu)\, p(\mu, \tau)$$

Here the kernel `lik(y | mu)` draws `y ~ Normal(mu, 1)`, and the law `prior(mu, tau)` draws `mu ~ Normal(0, 5)` and `tau ~ Normal(1, 1)`.
The merge `lik & prior` is the joint law over `{y, mu, tau}`, built by matching the name `mu`.
Merge also puts records side by side, combines functions by name, and combines input specs or output specs.
It reads right to left, as the density does: in `a & b`, `b` may supply inputs of `a`, never the reverse.

### 5.2 The definition

> [!IMPORTANT]
> **Definition (family).** Two objects are of one *family* when they are two records; two laws or kernels, in any mix; two functions; or two batches over equal levels (the same level names, order, and sizes).
> Two types are of one family when they are two input specs, two output specs, two record types, two law or kernel types, two function types, or two batch types.

> [!IMPORTANT]
> **Definition (feeds, shared).** In `a & b`, an output `c` of `b` *feeds* the input `c` of `a`: `a` receives at `c` the value `b` offers there.
> An input of both `a` and `b` that `b` does not feed is *shared*.

> [!IMPORTANT]
> **Definition (merge).** The *merge* `a & b` of two objects of one family has:
> 1. **outputs:** `a`'s outputs, then `b`'s;
> 2. **inputs:** `a`'s inputs that `b` does not feed, then `b`'s inputs that are not shared; a shared input is one input, typed by the unification of its two types (§ 2.5), and optional only if optional in both;
> 3. **kind:** records give a record; laws and kernels give a kernel if a required input remains and a law otherwise; functions give a function; batches give a batch over the same levels;
> 4. **produced object:** the record of all outputs, with `b`'s part produced first and `a`'s next, from its inputs and the values `b` feeds (§ 5.4).

> [!IMPORTANT]
> **Definition (valid merge).** `a & b` is refused unless:
> 1. **one family:** `a` and `b` are of one family;
> 2. **recordable produced objects:** each operand produces a base value or a record, of known type;
> 3. **one source per output:** `outputs(a) ∩ outputs(b) = ∅`;
> 4. **order:** `inputs(b) ∩ outputs(a) = ∅`;
> 5. **separate sides:** within each operand, no name is both an input and an output;
> 6. **fed types conform:** each feeding output's type conforms to its input's type (§ 2.5);
> 7. **shared types unify:** the two types of each shared input unify;
> 8. **one dimension scope:** no dimension name stands for two sizes, and no two unbound names meet at one axis (§ 2.3).

![The legend for merges: a wire runs from an output of b to the input of a with the same name; the result is a dashed boundary with its ports on its edges; a port fanning out to both operands is a shared input; a refused merge is red with a cross.](../figures/legend-merge.svg)
*Figure: how a figure draws a merge.*

![The merge a & b: b's output x feeds a's input x; w fans out to both operands; the result offers u, v, x, z and takes w and q.](../figures/merge-general.svg)
*Figure: `a(u, v | x, w) & b(x, z | w, q)`. `x` is fed, and stays an output; `w` is shared; the result is the kernel `(u, v, x, z | w, q)`.*

![The merge lik & prior: prior's output mu feeds lik's input mu; the result is a law offering y, mu, tau.](../figures/merge-kernel-law.svg)
*Figure: `lik & prior`. `mu` is fed, no input remains, so the result is a law; `lik`'s root output `y` becomes the field `y` of the joint draw.*

![The merge a & b of two kernels sharing theta: one input theta fans out to both.](../figures/merge-shared-input.svg)
*Figure: `a(x | theta) & b(y | theta)`. `theta` is shared; types `("n",)` and `(3,)` would unify to `(3,)`.*

> [!WARNING]
> A merge reads a law or kernel through its factors, so a law built by merging can still have an optional input fed later (§ 5.3).

<details>
<summary>Technical note: factors, and the inputs a law's factors still take</summary>

> [!IMPORTANT]
> **Definition (factor).** A merge of laws and kernels keeps its operands' factors. A law or kernel built by merging has the factors it kept; any other has itself as its one factor. A *grouped sub-joint*, which a rename creates by gathering outputs of several factors under one group (§ 4.5), is one factor.

> [!IMPORTANT]
> **Definition (remaining inputs of the factors).** The inputs, required or optional, that the factors take and no factor feeds, in order of first appearance: for a kernel, its inputs; for a law of a family such as `Normal`, none; for a law built by merging, the optional inputs its factors left at their defaults; for a grouped sub-joint, its own inputs if it is a kernel, none if a law.

For laws and kernels, "inputs" in the definitions above means these remaining inputs, so condition 4 covers optional ones.
For records, functions, and types, the remaining inputs are the inputs; a type, and an element in a merge of batches, is one factor.

</details>

<details>
<summary>Technical note: merging types</summary>

- **Output specs:** the record type of the first's outputs, then the second's; a root output `n` becomes the child `n`, and every other path keeps its type. Conditions 2, 3, and 8 apply.
- **Input specs:** inputs in order of first appearance; a shared input is unified, and optional only if optional in both; dimension bindings are not kept.
- **Record types:** both sets of children; a shared top-level name is refused.
- **Law, kernel, function, and batch types:** the type of the merged object.

An input spec and an output spec do not merge. The inputs an output spec's outputs would feed form the input spec with one required input per output, of its type.

</details>

<details>
<summary>Technical note: objects of two families</summary>

A record could be read as a law giving itself with probability one, and a function as a kernel of point masses. A law with a point-mass factor has no joint density, and a joint's factors are laws and kernels (IV.1), so objects of two families are refused.

</details>

### 5.3 Optional inputs

An optional input that the other operand does not feed stays an optional input of the result; one that it feeds is fed like any other.

![The merge a & b where a takes mu and an optional sigma, and b produces mu from theta: the result takes sigma, hollow, and theta.](../figures/merge-optional-open.svg)
*Figure: `a(y | mu, sigma) & b(mu | theta)`. `sigma` is not fed, so it stays an optional input of the kernel result.*

![The merge a & b where b is a law offering mu and sigma: both inputs of a are fed and the result is a law.](../figures/merge-optional-fed.svg)
*Figure: `a(y | mu, sigma) & b(mu, sigma)`. `b` feeds the optional `sigma` too; no input remains, so the result is a law.*

> [!WARNING]
> A law built by merging keeps optional inputs its factors left at their defaults, and a later merge can feed them.
> For a law `c(mu)`, `a & c` is a law that takes no inputs, with `sigma` at 1; merging it with a law `e(sigma)`, where `sigma ~ Normal(100, 0.001)`, feeds `sigma`, so `y`'s scale becomes about 100.

<details>
<summary>Technical note: a shared optional input that a binding omits</summary>

Each factor then takes its own default, since defaults belong to objects and an input spec holds none; the law at the defaults need not equal the kernel at any one value of the input.

</details>

<details>
<summary>Technical note: a merge that leaves only optional inputs</summary>

It gives a law, sampled at the defaults, while a kernel whose inputs are all optional stays a kernel (IV.2).

</details>

### 5.4 The value of a merge

**Laws and kernels.**
For values `u` of the result's inputs, let `u_a` be those `a` takes and `b` does not feed, and `u_b` those `b` takes.

> [!IMPORTANT]
> **Definition (law of a merge).** A draw `z` of `a & b` at `u`: draw `z_b` from `b` at `u_b`; draw `z_a` from `a` at `u_a` and the values `z_b` feeds; `z` is `z_a`'s record of outputs followed by `z_b`'s.

With densities, where `z_fed` is the fed values:

$$p_{a \& b}(z \mid u) = p_a(z_a \mid u_a, z_{\mathrm{fed}})\; p_b(z_b \mid u_b)$$

For `lik & prior`, this is the product of § 5.1.

**Records.** No inputs, so nothing is fed: `a & b` is `a`'s children followed by `b`'s.

> [!NOTE]
> Records merge by top-level name: two records that both have a group `g` are refused by condition 3, whatever the groups hold.

![The merge of r1 = {a} and r2 = {b, g: {c}}: the result offers a, b, g.](../figures/merge-records.svg)
*Figure: `{a: 1.0} & {b: 2.0, g: {c: 3.0}}` is `{a, b, g: {c}}`.*

> [!TIP]
> To merge two records that share `g`, merge the groups first: `(x without g) & (y without g) & {g: x[g] & y[g]}`.

**Functions.** A function is a kernel of point masses, so `(a & b)(u)` is `a`'s record of outputs at `u_a` and the fed values, followed by `b`'s at `u_b`.
For `a(mu | data)` returning the mean and `b(sigma | data)` the standard deviation, `a & b` takes `data` once and returns `{mu, sigma}`.

**Batches.** Over equal levels, element by element; a base element is read as its record of outputs, so 3 draws of `x` and 3 draws of `y` merge into 3 records `{x, y}`.

<details>
<summary>Technical note: batches merge element by element</summary>

Each element is one operand, typed by the elements' type, and the conditions are checked once on that type; an element that is a law is one factor, so its factors' hidden optional inputs are not fed.

</details>

### 5.5 Refused merges

| Condition | Why |
| --- | --- |
| 1. One family | each family has its own produced object (§ 5.4) |
| 2. Recordable produced objects | each operand's produced object is rebuilt from fields of the result |
| 3. One source per output | one name cannot hold two values |
| 4. Order | sides stay separate, and each operand follows those that feed it |
| 5. Separate sides | fed and shared are then decided by names alone |
| 6. Fed types conform | the input must admit every value fed to it |
| 7. Shared types unify | one value supplies both inputs |
| 8. One dimension scope | a dimension name is one size (II.1) |

![A refused merge where a and b both output x.](../figures/merge-invalid-output.svg)
*Figure: refused by condition 3: `x` is an output of both `a` and `b`.*

![A refused merge where b takes x, which a outputs.](../figures/merge-invalid-order.svg)
*Figure: refused by condition 4: `b` takes `x`, which `a` outputs; written `b & a`, it is valid.*

![A refused merge where b's output x of shape (4,) feeds a's input x of shape (3,).](../figures/merge-invalid-conform.svg)
*Figure: refused by condition 6: the output `x` of shape `(4,)` does not conform to the input of shape `(3,)`.*

![A refused merge where the shared input theta has shapes (2,) and (3,).](../figures/merge-invalid-unify.svg)
*Figure: refused by condition 7: the shared `theta` has shapes `(2,)` and `(3,)`, which do not unify.*

![A refused merge where the function b returns a law.](../figures/merge-invalid-law.svg)
*Figure: refused by condition 2: `b` returns a law, which a record cannot hold unnamed; `b` should return `{params: law}`.*

<details>
<summary>Technical note: why outputs feed only from right to left</summary>

If a factor of `b` took an input that `a` outputs, nothing would feed it, so the result would have the name on both sides.
With the order rule, in `a & b & c` each operand is fed only from its right, and the density reads left to right.
The rule does not make associativity hold, and it does not let every acyclic graph be built one module at a time: for kernels `c(z | y)`, `a(y | x)` and a law `b(x)`, both `a & (c & b)` and `(c & b) & a` are refused, while `c & a & b` is valid.

</details>

<details>
<summary>Technical note: a merge never lifts</summary>

An output feeds an input only when its type conforms. A merge does not apply the receiver to each draw of a law fed in place of a value, which would ignore the law's dependence on the receiver's other inputs.

</details>

<details>
<summary>Technical note: two selections of one law</summary>

For a law `d` over `{g1: {theta}, g2: {theta}}`, `d[[g1]] & d[[g2]]` is the product of the two marginals, since a selection is one factor; the joint is `d[[g1, g2]]`.
`d[g1] & d[g2]` is refused, both offering `theta`, and its message points to `d[[g1, g2]]` (§ 19).

</details>

### 5.6 Properties

> [!NOTE]
> 1. **Separate sides:** no name is both an input and an output of `a & b`.
> 2. **Paths kept:** every path of an operand is a path of `a & b` with the same type, once dimension names are bound; a root output `n` becomes the path `n`.
> 3. **Specs:** the output spec of `a & b` is the merge of the operands' output specs, and its input spec the merge of `a`'s without the fed inputs and `b`'s.
> 4. **Associative when both groupings are valid**, for laws, kernels, records, record types, output specs, and batches of records or base values.
> 5. **Commutative up to order** when neither operand feeds the other: same law or value, outputs in another order.
> 6. **Sampling commutes** for laws that do not feed each other and draw records: a draw of `a & b` is distributed as a draw of `a` merged with a draw of `b`.
> 7. **Renaming commutes** under the conditions below.
> 8. **Labels are not read;** the result's label joins the operands' with `·`.

<details>
<summary>Technical note: when two groupings differ in validity</summary>

`(a & b) & c` and `a & (b & c)` check different pairs: a name `a` and `b` take and `c` outputs, whose types in `a` and `b` do not unify while `c`'s conforms to each; or two unbound dimension names at one axis that one grouping binds first.
Input specs and functions keep no dimension scope, so for inputs `x: ("n",)`, then `x: (3,)` with `y: ("n",)`, then `z: ("n",)`, the left grouping leaves `z: ("n",)` and the right gives `z: (3,)`.
Batches of laws can fail associativity, since each element is one factor.

</details>

<details>
<summary>Technical note: when renaming commutes</summary>

For a one-to-one rename `m` of the names of `a` and `b`, applied to inputs and outputs alike, renaming `a & b` by `m` equals merging `a` and `b` each renamed by its part of `m`, for laws, kernels, records, and functions, provided `m` renames no remaining input of a law's factors.
A rename that moves a node into a group does not commute.

</details>

<details>
<summary>Technical note: optional inputs across merges</summary>

For `M = a & c`, with `a(y | mu, sigma)` and a law `c(mu)`:
1. `M & e` feeds `sigma` for a law `e(sigma)`, and equals `a & (c & e)`.
2. For a kernel `k(w | sigma)`, `M & k` shares `sigma`, now required; condition 7 checks a type no spec of `M` shows.
3. For an unrelated kernel `k(w | v)`, `M & k` gains the optional input `sigma`.
4. `M` and `a1 & c`, for a kernel `a1(y | mu)` equal to `a` at `sigma = 1`, are equal laws that merge differently with `e`.
5. A result keeps `sigma` exactly when it keeps `a` as a factor: a selection of `M` does not, the marginal computed as a merge of factors does, a rename does, conditioning on observed `y` does not.
6. The merge of types can be valid where the merge of objects is refused: `e`'s type and `M`'s type merge, while `e & M` breaks condition 4.

</details>

<details>
<summary>Technical note: dimension scopes across merges</summary>

A merged law or kernel keeps its dimension scope; merged specs and functions apply it and keep nothing.
A kept scope can refuse a merge over a name no type shows: if `m1` bound `n = 3` and `m2` bound `n = 4`, `m1 & m2` breaks condition 8 although neither event spec contains `n`.

</details>

### 5.7 What merge is not

1. **Not composition:** `f ∘ g` passes `g`'s result to `f` by position and returns `f`'s; a merge matches names and returns both operands' outputs.
2. **Not conditioning:** conditioning `lik` on `mu = 0.5` removes `mu`; in `lik & prior`, `mu` is fed and stays an output.
3. **Not renaming:** rename an operand first (§ 4.5), then merge.

## 6. The kinds, one by one

Each kind follows §§ 1 to 5; this section states only what is specific to it.

### 6.1 Base values and records

> [!NOTE]
> 1. **What it is:** a base value is an array or an opaque value; a record is a tree of named values (§ 1.1).
> 2. **Produces:** itself.
> 3. **Parts:** a record's nodes, continuing into objects its fields hold (§ 2.4); a base value has none.
> 4. **Inputs:** none.
> 5. **Outputs:** a record offers its children; a base value standing alone offers nothing (§ 3.3).
> 6. **Merge:** two records merge side by side, by top-level name (§ 5.4); base values do not merge.

![The record r with the outputs a and g; g holds b and c.](../figures/record-parts.svg)
*Figure: the record `{a: 1.0, g: {b: 2.0, c: 3.0}}` offers `a` and `g`.*

For `r = {a: 1.0, g: {b: 2.0, c: 3.0}}`: `r[g]` is the record `{b, c}`, `r[[a, g/b]]` is `{a, g: {b}}`, and `r & {x: 0.5}` is `{a, g: {b, c}, x}`.

### 6.2 Laws

> [!IMPORTANT]
> **Definition (conditioning).** *Conditioning* a law on values at some of its paths gives the law of the rest of a draw given those values.

> [!IMPORTANT]
> **Definition (summary).** A *summary* of a law is its mean, variance, standard deviation, or a quantile.

> [!NOTE]
> 1. **Produces** a draw; **parts** and **outputs** are those of its event type (§ 2.4, § 3.3); **inputs:** none.
> 2. **Selecting commutes with sampling** for every law that can be sampled, since its paths come from its event type (§ 4.1).
> 3. **A selection and the marginal law are one law**, whether the selection keeps its link to `d` or the marginal is computed apart.
> 4. **Conditioning removes what it fixes:** the event spec of `d` conditioned at paths `ps` is `d`'s event spec without `ps`, whatever computes it; conditioning `d` on a draw of `d[[ps]]` fixes exactly the nodes at `ps`.
> 5. **A summary keeps the law's paths** when it has one value per field, so the mean of `d` at `p` and the mean of `d[p]` estimate one quantity; a summary is accepted wherever a draw of its type is.
> 6. **A law whose draws are laws, functions, or batches** is allowed and offers nothing (§ 3.3, rule 5); to name its draws, it wraps them, and a family names its own draws under the name it is given.
> 7. **Merge:** laws and kernels merge, read through their factors (§ 5.2).

![A law d whose draws are laws, offering nothing, and the same law wrapped, offering params.](../figures/law-of-laws.svg)
*Figure: a law whose draws are laws offers nothing; wrapped, it draws `{params: law}` and offers `params`.*

For `d` over `{mu, sigma}` with `mu ~ Normal(0, 1)` and `sigma ~ Normal(1, 1)`: conditioning on `mu = 0.1` gives a law over `{sigma}`; the mean of `d` is the record `{mu, sigma}` of means; `d[mu]` is the marginal law of `mu`.

<details>
<summary>Technical note: conditioning at nested paths and on records</summary>

Given values are a mapping from paths to values, or a record read by its paths.
Conditioning a law over `{a, g: {b, c}}` at `g/b` is conditioning at a nested path.
To condition on the whole group `g`, name it: give `{g: v}` for a draw `v` of `d[g]`, or a draw of `d[[g]]`; a bare `{b, c}` would be read at the top level, where `b` and `c` are not paths.
A method that cannot condition at a nested path refuses with a message naming the method, not the path.

</details>

<details>
<summary>Technical note: the type of a summary</summary>

A summary's type is derived from the event type: a mean is floating, with the hull of the event's support, so a mean of a law over integers is refused where an integer value is required.
A result that is not one value per field, such as a covariance, is one base value named by the operation.

</details>

### 6.3 Kernels

> [!NOTE]
> 1. **What it is:** a law for each value of what it conditions on; it has at least one input.
> 2. **Produces** a draw at each value of its inputs; **parts** and **outputs** are its event type's.
> 3. **Inputs:** one per value it conditions on. A kernel whose inputs are all optional is still a kernel; binding its last required input gives a law, with optional inputs at their defaults.
> 4. **Separate sides:** no name is both an input and an output.
> 5. **Selecting** `k[p]` keeps its inputs; renaming a kernel renames inputs and outputs alike (III.9).
> 6. **Merge:** with laws and kernels; it feeds and is fed (§ 5.2).

![The kernel k with the output y outside its left edge, the input mu, and the optional input sigma.](../figures/signature.svg)
*Figure: the kernel `k(y | mu, sigma)`.*

<details>
<summary>Technical note: a kernel and the function that returns its laws</summary>

A kernel and the function returning the law at each value are one map $S \to P(T)$, but their produced objects differ: a draw for the kernel, a law for the function.
So the kernel offers its draws' outputs, and the function offers nothing (§ 3.3, rule 5).

</details>

### 6.4 Functions

> [!NOTE]
> 1. **Produces** its return value at each value of its parameters; **parts** and **outputs** are its return value's (§ 3.3, rule 3).
> 2. **Inputs:** its parameters; a parameter with a default is optional.
> 3. **A function into records** is the record of its output functions, so it offers them.
> 4. **A function that returns a law** offers nothing; to name the law, it returns `{params: law}`.
> 5. **A function that returns a function** offers that function's outputs.
> 6. **Selecting** `f[p]` maps `w` to `f(w)` at `p`, with the same inputs.
> 7. **Merge:** functions merge as kernels whose laws are point masses, and must state their inputs and return type (§ 5.2).

For `f` from `x` to `{y, z}`: it offers `y` and `z`; `f[y]` maps `x` to `f(x)`'s `y` (figure in § 4.1).
For `h` from `data` to a law over `{mu, sigma}`: it offers nothing, `h[mu]` maps `data` to the law's marginal at `mu`, and `h` cannot be merged (§ 5.5).

<details>
<summary>Technical note: a function whose return type is unknown</summary>

It has no paths known before it is called, so `f[p]` is checked when the selected function is called, against what the call returns.

</details>

### 6.5 Batches

> [!IMPORTANT]
> **Definition (element type).** A batch's *element type* is the output spec of one element: the element's type and, for a base element, its name.
> A batch built from bare base values names them `element`.

> [!NOTE]
> 1. **Produces** itself; **parts** are its elements' paths, each node a column (§ 2.4); positions index its levels.
> 2. **Inputs:** its elements' inputs, supplied to every element by one binding.
> 3. **Outputs:** its elements' outputs, as columns; a batch of base values offers one output at the root, a batch of laws or kernels nothing.
> 4. **Merge:** two batches over equal levels merge element by element (§ 5.4).
> 5. **Names are not data:** two batches that differ only in a base element's name are equal.

![A batch b of draws over {mu, sigma}, and its column b[mu], which offers mu at the root.](../figures/batch-columns.svg)
*Figure: a batch of 100 draws of a law over `{mu, sigma}`, and its column at `mu`.*

For 4 draws `b` of a law whose draws are arrays named `x`: `b` offers `x` at the root, `b[x]` is refused, the object at `x` is `b`, and `b[[x]]` is a batch of records `{x}`.

<details>
<summary>Technical note: why names stay out of a batch's equality</summary>

A batch's base-element name is a description of the batch, as a label is, not part of its values; keeping it out of equality lets draws of `x` and of `y` be combined entry by entry.
§ 18 states how the name is stored and what this costs.

</details>

## 7. Invariants

### 7.1 The invariants

> [!IMPORTANT]
> **Definition (equal).** Two values are *equal* when they have the same type and equal entries; two laws or functions when they have equal output specs and the same distribution or map.
> Merge reads factors, which equality does not compare.

`d` is a law, `k` a kernel, `f` a function, `b` a batch, `x` any object with parts; `s` is a path or a list of paths.

| # | Invariant | Follows from, or chosen |
| --- | --- | --- |
| I1 | **One set of paths:** a path or list valid on `d` is valid on its draws, on each element of a batch of its draws, on its mean, and on its event spec | § 2.4 |
| I2 | **Selecting commutes with sampling:** a draw of `d[s]` is distributed as a draw of `d` at `s`, at every path | § 4.1 |
| I3 | **Selecting commutes with summaries:** the mean of `d` at `s` and the mean of `d[s]` estimate one quantity, and are equal when computed in closed form; likewise variance, standard deviation, and quantiles | I2, § 6.2 |
| I4 | **Positions and paths commute** on a batch | § 2.4, § 4.1 |
| I5 | **Paths chain:** `x[p][q]` is `x[p/q]`, through fields that hold objects | § 2.4 |
| I6 | **Output specs agree:** the output spec of `x[s]` is `x`'s output spec at `s`; for a record, the type | § 4.4 |
| I7 | **A selection and the marginal law are one law** | chosen (VI.8) |
| I8 | **Draws of a list condition back:** conditioning `d` on a draw of `d[[ps]]` fixes the nodes at `ps` and nothing else | chosen |
| I9 | **One path gives the node, a list a record holding it**, the output at the root included | chosen |
| I10 | **A list and removal partition** the leaf paths of an object, and the inputs of an input spec | § 4.2, § 4.3 |
| I11 | **Conditioning removes what it fixes**, whatever computes it; a kernel's bound inputs leave its input spec | chosen |
| I12 | **The object at an output:** output `c`'s type is the type of the produced object of the object at `c` | § 3.4, § 4.6 |
| I13 | **Selecting commutes with producing:** `f[s](w)` is `f(w)` at `s`; `k[s]` at `w` is the law of a draw's node at `s` | § 4.1 |
| I14 | **Selecting keeps inputs** | § 4.1 |
| I15 | **Sides are separate** for every kernel, law, and merge | chosen for kernels; conditions 4 and 5 for merges |
| I16 | **The specs of a merge are the merges of the specs**, after binding dimension names, when no operand's factors take an input its input spec does not state | § 5.6 |
| I17 | **A merge keeps paths**; an output at the root moves to the path of its name | § 5.6 |
| I18 | **Paths agree with selection:** a string is a path of `x` exactly when `x[s]` is defined, and rename keys are those paths and the root name | chosen |
| I19 | **Merge is associative when both groupings are valid**, under § 5.6's conditions | § 5.6 |
| I20 | **Sampling commutes with merge** for laws that do not feed each other and draw records | § 5.4 |
| I21 | **Renaming commutes with merge**, under § 5.6's conditions | § 5.6 |
| I22 | **Merge reads no label** | § 5.2 |

<details>
<summary>Technical note: what the proofs rest on</summary>

I2, I4, I5, I6, and I13 hold by construction, since they restate a definition; their tests check the implementation against it.
I2, I5, I6, and I9 hold with no exception at fields that hold objects, since no selection wraps.
jhuggins's Rocq model proves I1 for laws, draws, and batches, and I2, I4, I5, and I10 for every kind it models, with the facts behind I17 and I21; it does not model types, dimension names, or optional inputs, so I19's conditions are this design's.

</details>

### 7.2 Exceptions on purpose

<details>
<summary>Technical note: the cases outside the invariants</summary>

| Case | What happens | Why |
| --- | --- | --- |
| Covariance | the covariance of `d` is one base value, so it has no paths | it is not one value per field; the block for `s` is the covariance of `d[s]` |
| Monte Carlo summaries | the mean of `d` at `s` and the mean of `d[s]` differ | each takes its own draws |
| Summaries and conditioning at a field holding a law | a mean of drawn laws is undefined, and conditioning cannot fix a drawn law | I1, I3, and I8 cover paths that end at values |
| Labels | `d[[a, g/b]]`'s draws and `d`'s draws at `[a, g/b]` differ in label | no operation reads a label |
| Groups and fields | `g` is a path of `{a, g: {b, c}}` but not a field | selection names groups, while the fields are the leaves |
| The list form at the root | `d[[x]]` is defined although `x` is not a path, and `x` cannot be removed | an extra rule at the top level (§ 4.2) |
| A shared optional input omitted | each factor takes its own default | defaults belong to objects (§ 5.3) |
| A kernel whose inputs are all optional | merged with a law, it gives a law at the defaults, though it asks for its inputs alone | the kind follows the required inputs left (§ 5.3) |
| Two selections of one law | `d[[g1]] & d[[g2]]` is the product of the marginals | a selection is one factor (§ 5.5) |
| A selection and a marginal of one joint (I7) | they can merge differently when a factor has an optional input left at its default | merge reads factors (§ 5.6) |
| Hidden optional inputs | a later merge can feed or share them; equal laws can merge differently | merge reads factors (§ 5.6) |
| Merges of types and of objects (I16) | a type merge can be valid where the object merge is refused | a type states no inputs of factors (§ 5.6) |
| Associativity of validity (I19) | one grouping can be refused and the other valid | they check different pairs (§ 5.6) |
| Batches of laws (I19) | the two groupings of three batches can give different laws | each element is one factor (§ 5.4) |
| Input specs and functions (I19) | two valid groupings can differ in a dimension size | they keep no dimension scope (§ 5.6) |
| A kept dimension scope | a merge can be refused over a name no type shows | merged laws keep their scope (§ 5.6) |

</details>

---

## A. Open questions and issues

The rest of this appendix is drafted in [`appendices.md`](appendices.md); this running list is merged into it.

### A.2 Open questions and issues found while writing

Each item gives the reading the text adopts.

1. **When the library stops building the named-whole form.** Source § 6.8 says once G1b merges; T19, § 5.5, and the probe allowance let selections at fields holding objects build it until G3, and the source marks T19 "Inconsistent". *Reading:* until G3.
2. **"Has parts" without parts.** The source calls every law and batch an object "with parts", including a law over one array and a batch of arrays, which have no paths. *Reading:* a part is a node below the root (§ 1.3), and the naming rule says "only a base value is named from outside" (§ 3.3); behaviour is unchanged.
3. **Paths of a function whose return type is unknown.** `f[p]` is accepted and checked at the call (§ 6.4), while its type has no paths (§ 2.4), and I18 ties paths to selection. *Reading:* none; both rules are stated, and the question is which governs I18.
4. **Removal on batches of functions or kernels** is not defined (source § 4.5), though such batches are selected by path. *Reading:* not defined.
5. **Paths of an opaque value.** I18 covers "every object other than an array", while an opaque value has no selection. *Reading:* no string is a path of an opaque value.
6. **A record node inside a law that a field holds.** The source defines a field as "named through records only" and gives a group no such clause. *Reading:* such a node is neither a field nor a group of the outer record; removal drops a field holding an object when all its leaf paths are removed (§ 4.3).
7. **The word for IV.2's "unmet".** The source has S0 write "unfed" into `design/`; this text says "remaining inputs". *Reading:* Appendix B keeps the source's instruction, pending this question.
8. **Whether a law type and a kernel type merge.** Source § 2.16 says "two of one of these". *Reading:* they merge, as laws and kernels do (§ 5.2).
9. **Renaming a law's output onto an input its factors still take.** A law's rename does not rename the remaining inputs of its factors (§ 5.6), so renaming an output of `a & c` to `sigma` would put `sigma` on both sides, against I15; the source does not say what happens. *Reading:* refused, since I15 keeps sides separate.
10. **Unifying kernel and function types.** Unification requires their input types to be equal (§ 3.4, technical note), while a shared input elsewhere takes the unification of its types; the source gives no reason for the difference. *Reading:* equal, as stated.
