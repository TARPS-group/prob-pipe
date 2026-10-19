# Inputs, outputs, paths, and merge

> [!NOTE]
> **Style sample**, for approval before the full document is written: the front matter and §§ 3, 5, and 14 in final form.
> Links to other sections do not resolve yet; [`outline.md`](outline.md) lists every section.

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

---

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

---

## 14. Records: `Record` and `RecordSpec`

A record is represented by the tracked term `Record`, and a record type by the term spec `RecordSpec` (§ 8).
Both share the tree interface of `NamedTree` (II.6); only members this design defines or changes are shown.

### 14.1 API

```python
class RecordSpec(NamedTree[TermSpec], TermSpec):
    """The type of a record: a tree of names whose fields hold term specs.

    Its paths are those of the records it types (§ 1.2); indexing it gives the
    type of the same part of those records.
    """

    def __init__(self, field_specs: Mapping[str, TermSpec | Mapping | tuple] | None = None, /,
                 **fields: TermSpec | Mapping | tuple) -> None:
        """Build the type from its children.

        A tuple of sizes is the array type of that shape, with no dtype or support;
        a mapping is a group; the positional mapping also takes paths as keys.

        Raises:
            TypeError: A field is None, a pending type; an opaque field is OpaqueSpec().
        """

    def __getitem__(self, paths: str | Sequence[str], /) -> TermSpec:
        """The type at a path, or the RecordSpec of the nodes at a list of paths.

        A list keeps each node at its own path, in this spec's order.

        Raises:
            KeyError: A path is not a path of this spec.
            ValueError: The path reader refuses the argument (§ 12).
        """

    def __contains__(self, path: object, /) -> bool:
        """Whether path is a str and self[path] succeeds."""

    def keys(self) -> tuple[str, ...]:
        """The field paths in canonical order; groups are not listed."""

    def without(self, *paths: str) -> RecordSpec:
        """This type without the nodes at paths; an emptied group goes too.

        Raises:
            KeyError: A path is not a path of this spec.
            ValueError: Nothing would remain, a path repeats, or one path is inside another.
        """

    def rename(self, renames: Mapping[str, str] | None = None, /,
               **keyword_renames: str) -> RecordSpec:
        """Rename or move nodes by path, by II.6's rules (§ 4.5)."""

    def __and__(self, other: RecordSpec, /) -> RecordSpec:
        """Both types' children, this one's first (§ 5.2).

        Raises:
            TypeError: other is not a RecordSpec.
            ValueError: A top-level name is in both, or a dimension name binds two sizes.
        """


class Record(NamedTree[Any], TrackedTerm):
    """A tree of named values.

    A record is its own produced object, so its outputs are its children; its
    output spec, OutputSpec(record.spec), is derived, and an empty record has none.
    """

    def __init__(self, fields: Mapping[str, Any], /, *, label: str | None = None) -> None:
        """Build a record from a nested mapping, in which a mapping is a group.

        A field may hold any tracked term, such as a law; paths continue into it (§ 2.4).
        """

    @property
    def spec(self) -> RecordSpec:
        """The record's type, with every dimension name bound."""

    def __getitem__(self, paths: str | Sequence[str], /) -> Any:
        """The node at a path, or the record of the nodes at a list of paths.

        One path gives the node: a field's value as a tracked term, a group as a
        Record, or the term a field holds; a path continues into that term, so
        r["g/mu"] is r["g"]["mu"]. A list keeps each node at its own path, in
        this record's order, labelled by its paths, as "(a, g/b)".

        Raises:
            KeyError: A path is not a path of this record; the message lists its paths.
            ValueError: The path reader refuses the argument (§ 12).
        """

    def __contains__(self, path: object, /) -> bool:
        """Whether path is a str and self[path] succeeds, groups included."""

    def keys(self) -> tuple[str, ...]:
        """The field paths in canonical order; groups are not listed."""

    def at_path(self, *segments: str) -> Any:
        """self["/".join(segments)]; [] is preferred."""

    def output(self, output_name: str, /) -> Any:
        """The child output_name (§ 4.6).

        Raises:
            KeyError: output_name is not a child.
        """

    def without(self, *paths: str) -> Record:
        """This record without the nodes at paths; an emptied group, or a held term
        whose every leaf path is removed, goes too.

        Raises:
            KeyError: A path is not a path of this record.
            ValueError: Nothing would remain, a path repeats, or one path is inside another.
        """

    def rename(self, renames: Mapping[str, str] | None = None, /,
               **keyword_renames: str) -> Record:
        """Rename or move nodes by path, by II.6's rules (§ 4.5)."""

    def select(self, *paths: str, **remap: str) -> dict[str, Any]:
        """The nodes at paths, keyed by last name, for keyword arguments; select(x="g/b") is {"x": ...}.

        Raises:
            ValueError: Two paths share a last name.
        """

    def __and__(self, other: Record, /) -> Record:
        """Both records' fields, this one's first (§ 5.4).

        Raises:
            TypeError: other is not a Record.
            ValueError: A top-level name is in both, a group included.
        """
```

### 14.2 Examples

**Parts.** Paths name every node; `keys()` lists only fields.

```python
from probpipe import Record

r = Record({"a": 1.0, "g": {"b": 2.0, "c": 3.0}}, label="r")
r.keys()          # ("a", "g/b", "g/c")
r["g"]            # Record {b: 2.0, c: 3.0}
r["g/b"]          # NumericArray 2.0, labelled "g/b"
r["g"]["b"]       # the same node
"g" in r          # True
```

**Outputs and spec.**

```python
from probpipe import OutputSpec, Record

r = Record({"a": 1.0, "g": {"b": 2.0, "c": 3.0}})
r.spec                        # RecordSpec(a=NumericArraySpec(()), g=RecordSpec(b=..., c=...))
OutputSpec(r.spec).outputs    # {a: NumericArraySpec(()), g: RecordSpec(b=..., c=...)}
r.output("g")                 # r["g"]
```

![The record r with the outputs a and g; g holds b and c.](../figures/record-parts.svg)
*Figure: the record `r` offers its children `a` and `g`.*

**Several paths, and removal.**

```python
from probpipe import Record

r = Record({"a": 1.0, "g": {"b": 2.0, "c": 3.0}})
r["a", "g/b"]           # Record {a, g: {b}}, labelled "(a, g/b)"
r["g/c", "a"]           # Record {a, g: {c}}: the record's order
r[["g"]]                # Record {g: {b, c}}
r.without("g/c")        # Record {a, g: {b}}, the complement of r["a", "g/b"]
r.without("g/b", "g/c") # Record {a}
```

**A record type.**

```python
from probpipe import RecordSpec

s = RecordSpec(a=(), g=RecordSpec(b=(), c=()))
s["g/b"]              # NumericArraySpec(())
s[["a", "g/b"]]       # RecordSpec(a=..., g=RecordSpec(b=...))
s.without("g/b", "g/c")   # RecordSpec(a=...)
```

**A field holding a law.**

```python
from probpipe import Normal, Record

d = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)   # a law over {mu, sigma}
r = Record({"x": 3.0, "g": d})
r.keys()          # ("x", "g")
r["g/mu"]         # d["mu"], the law of mu
r["x", "g/mu"]    # Record {x, g}, whose g holds d[["mu"]]
```

**Renaming.**

```python
from probpipe import Record

r = Record({"a": 1.0, "g": {"b": 2.0, "c": 3.0}})
r.rename({"g/b": "g/z"})   # Record {a, g: {z, c}}
r.rename({"g/b": "b"})     # Record {a, g: {c}, b}: moved, appended last
```

**Merging.**

```python
from probpipe import Record

x = Record({"a": 1.0})
y = Record({"b": 2.0, "g": {"c": 3.0}})
x & y        # Record {a, b, g: {c}}
y & x        # Record {b, g: {c}, a}

Record({"g": {"b": 1.0}}) & Record({"g": {"c": 2.0}})
# ValueError: g is an output of both records; rename one, or merge the groups first
```

![The merge of r1 = {a} and r2 = {b, g: {c}}.](../figures/merge-records.svg)
*Figure: two records merge side by side.*

**Keyword arguments.**

```python
from probpipe import Record

r = Record({"g1": {"theta": 0.1}, "g2": {"theta": 0.2}, "a": 1.0})
r.select("a", "g1/theta")      # {"a": ..., "theta": ...}
r.select(t2="g2/theta")        # {"t2": ...}
r.select("g1/theta", "g2/theta")
# ValueError: g1/theta and g2/theta share the last name theta; name one with a keyword
```

<details>
<summary>Technical note: the errors of records</summary>

For `r = Record({"a": 1.0, "g": {"b": 2.0, "c": 3.0}}, label="r")`:

| Call | Error |
| --- | --- |
| `r["z"]` | `KeyError`: "'z' is not a path of 'r'; its paths: ['a', 'g', 'g/b', 'g/c']" |
| `r["g", "g/b"]` | `ValueError`: "'g/b' is inside 'g'; pick one" |
| `r["a", "a"]`, `r[[]]` | `ValueError` |
| `r["a", 0]` | `TypeError`: "a path and a position cannot be mixed in one []" |
| `r.without(("a", "g/c"))` | `TypeError`: pass the paths as separate arguments |
| `r.without("a", "g")` | `ValueError`: nothing would remain |
| `r.rename({"a": "g"})` | `ValueError`: collision (II.6) |
| `r & {"z": 2.0}` | `TypeError`: not a record |

</details>
