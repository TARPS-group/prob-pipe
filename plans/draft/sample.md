# Inputs, outputs, paths, and merge

> [!NOTE]
> **This is a style sample**, for approval before the full document is written.
> It holds the front matter and three sections in their final form: § 3 (inputs and outputs), § 5 (merge), and § 14 (records).
> Links to other sections do not resolve yet.
> [`outline.md`](outline.md) lists every section.

ProbPipe builds statistical workflows from a few kinds of pieces: values, probability laws, kernels, functions, and batches of any of these.
This document is the design of how those pieces describe themselves and fit together.
It answers four questions:
1. **Names:** how the parts of a value are named, so that a law, its draws, and its summaries are all indexed by the same names.
2. **Inputs and outputs:** what each piece receives, what it offers, and how it says so.
3. **Selecting:** how a part of any piece is picked out, removed, or renamed.
4. **Merging:** how two pieces are combined into one by matching the names of what one offers with what the other receives, as a likelihood and a prior are combined into a joint model.

Part I explains the ideas without code, from the most general to the most specific, so that the design can be judged from Part I alone.
Part II gives the code that realizes each idea, class by class, with its API and short examples.
The appendices hold everything that is not the design itself:
- the open questions;
- the plan for implementing the design;
- how it differs from today's code;
- how it relates to earlier plans;
- the alternatives that were weighed;
- its status and history.

The document is written for ProbPipe's developers, and assumes no part in the earlier discussions of the design.
Citations such as II.6 and D1 name sections and principles of the design reference in [`design/`](../../design/README.md).

## Contents

**[Part I: The conceptual model](#part-i-the-conceptual-model)**

1. [Values and the places inside them](#1-values-and-the-places-inside-them)
   1. [Base values and records](#11-base-values-and-records)
   2. [Paths, nodes, fields, and groups](#12-paths-nodes-fields-and-groups)
   3. [The parts of a value](#13-the-parts-of-a-value)
2. [Terms, kinds, and types](#2-terms-kinds-and-types)
   1. [Terms and their kinds](#21-terms-and-their-kinds)
   2. [What a term produces](#22-what-a-term-produces)
   3. [The type of a term](#23-the-type-of-a-term)
   4. [The paths of every term](#24-the-paths-of-every-term)
   5. [When one type admits another: conformance and unification](#25-when-one-type-admits-another-conformance-and-unification)
3. [Inputs and outputs](#3-inputs-and-outputs)
   1. [Inputs: where a term receives values](#31-inputs-where-a-term-receives-values)
   2. [Outputs: what a term offers](#32-outputs-what-a-term-offers)
   3. [How outputs are named](#33-how-outputs-are-named)
   4. [The input spec and the output spec](#34-the-input-spec-and-the-output-spec)
   5. [A term's signature, and how a figure draws it](#35-a-terms-signature-and-how-a-figure-draws-it)
   6. [Inputs and outputs compared](#36-inputs-and-outputs-compared)
4. [Selecting, removing, and renaming parts](#4-selecting-removing-and-renaming-parts)
   1. [Selecting one part](#41-selecting-one-part)
   2. [Selecting several parts at once](#42-selecting-several-parts-at-once)
   3. [Removing parts](#43-removing-parts)
   4. [Selecting from a type](#44-selecting-from-a-type)
   5. [Renaming and moving parts](#45-renaming-and-moving-parts)
   6. [The term at an output](#46-the-term-at-an-output)
5. [Merging terms by name](#5-merging-terms-by-name)
   1. [Why merge](#51-why-merge)
   2. [The definition of merge](#52-the-definition-of-merge)
   3. [The value of a merge](#53-the-value-of-a-merge)
   4. [Why each condition holds](#54-why-each-condition-holds)
   5. [Properties of merge](#55-properties-of-merge)
   6. [What merge is not](#56-what-merge-is-not)
6. [The kinds, one by one](#6-the-kinds-one-by-one)
   1. [Records and base values](#61-records-and-base-values)
   2. [Laws](#62-laws)
   3. [Kernels](#63-kernels)
   4. [Functions](#64-functions)
   5. [Batches](#65-batches)
7. [Invariants](#7-invariants)
   1. [The invariants](#71-the-invariants)
   2. [Where an invariant does not hold, on purpose](#72-where-an-invariant-does-not-hold-on-purpose)

**[Part II: The design in code](#part-ii-the-design-in-code)**

8. [Reading Part II](#8-reading-part-ii)
9. [Term specs](#9-term-specs)
10. [`InputSpec`](#10-inputspec)
11. [`OutputSpec`](#11-outputspec)
12. [Indexing in code: `[]`, `in`, `without`, and `rename`](#12-indexing-in-code--in-without-and-rename)
13. [Merging in code: `&`](#13-merging-in-code-)
14. [Records: `Record` and `RecordSpec`](#14-records-record-and-recordspec)
15. [Laws: `Distribution` and `MarginalView`](#15-laws-distribution-and-marginalview)
16. [Kernels: `ConditionalDistribution`](#16-kernels-conditionaldistribution)
17. [Functions: `Function`](#17-functions-function)
18. [Batches: `Batch` and `BatchSpec`](#18-batches-batch-and-batchspec)
19. [Errors and their messages](#19-errors-and-their-messages)
20. [Symbolic dimensions in code](#20-symbolic-dimensions-in-code)

**Appendices**

- [A. Open questions](#a-open-questions)
- [B. Implementation plan and sequencing](#b-implementation-plan-and-sequencing)
- [C. Differences from today's code](#c-differences-from-todays-code)
- [D. Relation to previous plans](#d-relation-to-previous-plans)
- [E. Alternatives considered](#e-alternatives-considered)
- [F. Status and history](#f-status-and-history)

---

## 3. Inputs and outputs

A term meets the rest of a workflow in two ways.
It receives values at its inputs, and it offers what it produces at its outputs.
This section defines both, states the rule that decides how outputs are named, and shows how a figure draws a term with its inputs and outputs.

### 3.1 Inputs: where a term receives values

> [!IMPORTANT]
> **Definition (input).** An *input* of a term is a named place where the term receives one value.
> An input has three attributes:
> - a *name*, which is also a Python identifier, since a value is supplied to an input by its name;
> - a *type*, which states the values the input admits (§ 2.3);
> - a flag, which makes the input *required* or *optional*.

> [!IMPORTANT]
> **Definition (optional input).** An *optional input* has a default value.
> The term that owns the input holds the default.

A kernel has one input for each value it conditions on, and a function has one input for each of its parameters.
A batch of kernels or of functions has the inputs of its elements, since one value bound to an input goes to every element.
A value and a law have no inputs.

> [!IMPORTANT]
> **Definition (binding).** *Binding* supplies values to some of a term's inputs, by name.
> A binding may omit an optional input, and the term then uses its own default for it.

Calling a function binds its parameters, and conditioning a kernel on values binds the inputs it conditions on.
Binding a kernel's last required input gives a law, in which every optional input that the binding omits takes its default.
Where a later section needs to write the values of a binding, it writes them as a record with one field for each input supplied, such as `{mu: 0.3, s: 2.0}`.

**Example.**
Take the kernel `lik_opt`, which for each pair of values `mu` and `s` is the law of `y ~ Normal(mu, s)`, and whose `s` has the default 1.
- It has two inputs, each an array of shape `()`: `mu`, which is required, and `s`, which is optional.
- Binding `mu` to 0.3 gives the law `Normal(0.3, 1)`, with `s` at its default.
- Binding `mu` to 0.3 and `s` to 2 gives the law `Normal(0.3, 2)`.

> [!NOTE]
> **Consequence.** An input whose type is a record type is one input, and a binding supplies it with one record.

<details>
<summary>Technical note: binding part of a record input</summary>

A binding may supply part of a record input through a path into it.
For an input `g` of the record type `{c, e}`, a binding may supply `g/c` alone.
The input `g` then takes only the rest of the record, `{e}`, and it is required from then on, since a default is a whole value (III.9).

</details>

### 3.2 Outputs: what a term offers

> [!IMPORTANT]
> **Definition (output).** An *output* of a term is a name under which the term offers part or all of what it produces.
> The *value at an output* is that part, or the whole.

Which parts a term offers, and under which names, depends only on what it produces (§ 2.2).
The next subsection states the rule.

### 3.3 How outputs are named

A name can be attached to a value in two places.
It can stand outside the value, as one name for the whole, or inside it, as the names of the value's parts.
A value with parts that also carried a name outside would have two sets of paths.
Suppose a law whose draws are records `{beta, sigma}` were also named `params` from outside.
Its draws would be indexed by `beta`, while everything that sees the law from outside, such as a rename, would index it by `params/beta`.
The naming rule gives each name one place.

> [!IMPORTANT]
> **Definition (the naming rule).** A produced term offers its outputs as follows:
> 1. **A base value** is offered whole, under one name that stands outside it, which the term that produces it states. This output is *at the root*.
> 2. **A record** offers its children: one output for each child, under the child's name.
> 3. **A function** offers the outputs of its return value. The value at an output `c` is the function that maps each value of its inputs to the value at `c` of its return value.
> 4. **A batch** offers the outputs of its elements. The value at an output `c` is the batch, over the same levels, of the elements' values at `c`.
> 5. **A law or a kernel** offers no outputs.
>
> The outputs of a term are the outputs that its produced term offers.
> So no value other than a base value is ever named from outside.

Rules 3 and 4 apply to what a function returns and to what a batch holds, whatever its kind.
Rule 5 concerns a law that is itself produced, such as a law that a function returns or that a random measure draws.
A law as a term offers the outputs of its draws, since its produced term is a draw: a law whose draws are records `{mu, tau}` offers `mu` and `tau`, and the value at `mu` is the field `mu` of a draw.

> [!IMPORTANT]
> **Definition (wrap).** To offer a value that is not a base value under one name, the code that produces it *wraps* it: it produces the one-field record `{name: value}` instead.
> The record's one child is then the one output, and `name` is a path of every value produced.

![Three terms. A kernel lik whose draw is an array offers it whole under the name y, drawn outside the box. A law params_prior whose draws are records offers its fields beta and sigma. The same law wrapped offers one output, params, which holds beta and sigma.](../figures/output-naming.svg)

The figure shows the first two rules and a wrap.
Each example below builds the term it uses.
- The kernel `lik` draws `y ~ Normal(mu, 1)` for each value of `mu`. Its draw is an array, so by rule 1 it offers the array whole, under the name `y`, at the root.
- A law `params_prior` whose draws are records `{beta, sigma}` offers `beta` and `sigma`, by rule 2.
- A law whose draws are records `{params: {beta, sigma}}` offers one output, `params`, by rule 2. Its draws are indexed by `params/beta`, and so is the law.
- A function `emulator` that maps `theta` to a record `{mean, var}` offers `mean` and `var`, by rules 3 and 2. The value at `mean` is the function that maps `theta` to the field `mean` of `emulator(theta)`.
- A function `fit` that returns a posterior law offers no outputs, by rules 3 and 5. To offer the posterior under a name, `fit` returns the record `{posterior: law}` instead.
- A batch of 100 draws of a law over `{mu, tau}` offers `mu` and `tau`, by rules 4 and 2. The value at `mu` is the batch of the 100 values of `mu`.
- A batch of 4 draws of a law whose draws are arrays named `x` offers `x`, by rules 4 and 1. The value at `x` is the whole batch.

> [!NOTE]
> **Consequence (one set of paths).** A term offers either one output at the root, or outputs that are all paths of one name, never both.
> The name of an output at the root is not a path, since nothing is below it.
> The name of every other output is its own path.

The reason is that rule 1 gives an output at the root only to a base value, which has no parts, and rules 2 to 4 give outputs that are top-level parts, of a record or of every return value or element.
So the names that pick out a term's parts, the names its outputs are offered under, and the paths of its draws all agree.

> [!NOTE]
> **Consequence.** A base value that no term produced, such as an array on its own, has no outputs: nothing states a name for it, and its label is not a name (§ 2.1).

<details>
<summary>Technical note: why a function into records offers its outputs, and a law over records does not</summary>

A function `f` into records `{mean, var}` is determined by its two output functions: `f(x)` is the record `{mean: f_mean(x), var: f_var(x)}`, where `f_mean` maps `x` to the field `mean` of `f(x)`.
So offering `f` by its outputs loses nothing.
A batch of records is determined by its columns in the same way.

A law over records `{mu, tau}` is not determined by its two marginal laws, since many joint laws have the same marginals.
If a produced law offered its marginals as outputs, a later step could receive `mu` and `tau` as two separate laws and lose the dependence between them.
So a produced law offers no outputs (D1), and a batch of laws offers none either, by rules 4 and 5.

</details>

<details>
<summary>Technical note: the names of outputs and of inputs</summary>

The name of an output is any name (§ 1.1), since it becomes the name of a field.
The name of an input is also a Python identifier.
So an output whose name is not an identifier, such as `"n-obs"`, cannot supply an input of the same name.

</details>

### 3.4 The input spec and the output spec

A term's inputs and its outputs are each described apart from the term: the inputs by an input spec, and what the term produces by an output spec.

> [!IMPORTANT]
> **Definition (input spec).** The *input spec* of a term lists its inputs, in order: each input's name and type, and which inputs are optional.
> It holds no defaults, since each default belongs to the term.
> Two input specs are equal when they list the same inputs, with equal types and the same optional inputs, in any order.

A term with no inputs has the empty input spec.

> [!IMPORTANT]
> **Definition (output spec).** The *output spec* of a term describes its produced term by two things:
> 1. the type of the produced term;
> 2. when that type is a base type, the name of the output at the root.
>
> The output spec of a law or of a kernel is also called its *event spec*, since it describes one draw.

> [!NOTE]
> **Consequence.** The type alone decides whether an output spec names an output at the root: it names one exactly when the type is a base type.
> So no separate flag records how a term offers its outputs.

> [!IMPORTANT]
> **Definition (type of an output).** The *type of an output* is the type of the value at that output.

> [!NOTE]
> **Consequence.** The type of every output can be read from the output spec alone:
> - an output at the root has the type of the produced term;
> - an output of a record has the type of its child;
> - an output `c` of a function has the type of a function with the same inputs that returns the value at `c` of the return value;
> - an output `c` of a batch has the type of a batch, over the same levels, of the elements' values at `c`.

**Example.**
- The kernel `lik`, which draws `y ~ Normal(mu, 1)`, has an output spec that states the array type of shape `()` and the name `y`. Its one output, `y`, has that array type.
- A law over `{mu, tau}`, both scalars, has an event spec that states the record type `{mu, tau}` and no name. Its outputs `mu` and `tau` each have the array type of shape `()`.
- A batch of 100 draws of that law, over the level `sample`, has the outputs `mu` and `tau`, each of the type "a batch over `sample` of arrays of shape `()`".

> [!IMPORTANT]
> **Definition (type hole).** An output spec may state the name of an output at the root before its type is known.
> The pending type is a *type hole*, which the term fills once it knows the type, as a function can once it has run.

§ 11 states how a hole is filled.

> [!IMPORTANT]
> **Definition (record of outputs).** A produced term that is a base value or a record can be read as the *record of its outputs*:
> - a record is its own record of outputs;
> - a base value `v` offered under the name `n` is the record `{n: v}`.

> [!NOTE]
> **Consequence.** A produced term is rebuilt from its record of outputs: a record is the record itself, and a base value is the field `n` of `{n: v}`.

For example, a draw `0.3` of `lik` is the record of outputs `{y: 0.3}`, and a draw `{mu: 0.1, tau: 1.2}` of a law over `{mu, tau}` is its own record of outputs.
A produced function, batch, or law has no record of outputs in this sense, since it is not a record and carries no name.

<details>
<summary>Technical note: equality and order</summary>

The order of a term's outputs is the order of the fields of its record of outputs, which fixes the layout of a numeric record as a flat vector.
So two output specs that list the same outputs in two orders are unequal.
Input specs are compared without order, since a binding reads inputs by name.

An output spec that names a base value `n` and an output spec whose type is the record `{n: ...}` offer the same output and describe different produced terms, so they are unequal.

</details>

### 3.5 A term's signature, and how a figure draws it

> [!IMPORTANT]
> **Definition (signature).** The *signature* of a term is its outputs and its inputs, written `name(outputs | inputs)`, as in `lik_opt(y | mu, s)`.
> A term with no inputs is written without the bar, as in `prior(mu, tau)`.

This document draws a term by its signature.
Data flows from right to left in every figure, as a density `p(y | mu)` is written, with what is produced before the bar and what is received after it: inputs enter a term on its right edge, and outputs leave on its left edge.

![The legend for terms. Data flows from right to left. A term is a rounded box whose badge names its kind: value, law, kernel, function, or batch. An output is a port on the left edge, labelled inside the box. An output at the root has its name outside the box. An input is a port on the right edge, and a hollow port is an optional input. The parts of a value are a tree inside the box. A batch is a stack of boxes with its levels at the bottom.](../figures/legend-terms.svg)

The kernel `lik_opt` of § 3.1 draws `y ~ Normal(mu, s)`, with the default `s = 1`.
Its draw is an array, offered under `y`, so `y` is an output at the root, drawn outside the box.
It takes `mu`, and `s` is optional, so the port of `s` is hollow.

![The kernel lik_opt: the output y on its left edge, with its name outside the box; the input mu on its right edge; and the optional input s, a hollow port below it.](../figures/signature.svg)

A record is its own produced term, so it offers its children, by rule 2 of the naming rule.
The record `{a: 1.0, g: {c: 2.0, e: 3.0}}` offers `a` and `g`, and the group `g` holds the fields `c` and `e`.

![The record r with the outputs a and g on its left edge, and the fields c and e drawn as a tree below g.](../figures/record-parts.svg)

### 3.6 Inputs and outputs compared

The two sides have the same structure, apart from the differences below.
Each follows from the definitions of this section.

| | Inputs | Outputs | Why they differ |
| --- | --- | --- | --- |
| Where a name stands | every input is named | only a base value is named from outside; a record carries its names inside | a value is always supplied to an input by name, while a produced value with parts carries its own names |
| What a name may be | a Python identifier | any name | an input matches a parameter, while an output becomes a field |
| Structure | flat names | one output at the root, or top-level paths, with paths below them | an input receives one value, while an output is a place inside a produced term |
| Presence | an input may be optional, with the term's default | an output is always present | a produced term has a value at every output |
| A type not yet known | never | a type hole | a binding is checked against an input's type, while a term may learn its output's type when it runs |
| Equality | ignores order | keeps order | a binding reads inputs by name, while the order of outputs fixes the layout of a record |
| None at all | a term may have no inputs | an output spec never describes an empty record | a term with no inputs is a law or a value, while an empty record describes nothing produced |
| Batches | a batch's inputs are its elements' | a batch's outputs are its elements' outputs, as batches | one binding supplies every element, while a produced batch is offered output by output |

---

## 5. Merging terms by name

### 5.1 Why merge

A statistical model is often written as a product of densities, joined by the names they share.
For data `y` with a location `mu` and a scale `tau`:

```math
p(y, \mu, \tau) = p(y \mid \mu)\; p(\mu, \tau)
```

The first factor is a kernel, `lik(y | mu)`, which draws `y ~ Normal(mu, 1)` for each value of `mu`.
The second is a law, `prior(mu, tau)`, with `mu ~ Normal(0, 5)` and `tau ~ Normal(1, 1)`.
The product joins the two by the name `mu`: the value of `mu` that the prior draws is the value the likelihood conditions on.

*Merging* builds a joint term from two terms by matching names in this way.
`lik & prior` is the law over `{y, mu, tau}` whose density is the product above.
The same operation puts two records side by side, combines two functions by name, and combines the input specs or output specs of several terms.

A merge is written from right to left, as the density is.
In `a & b`, the right operand `b` may supply inputs of the left operand `a`, and never the reverse.

### 5.2 The definition of merge

A merge combines two terms of one family.

> [!IMPORTANT]
> **Definition (family).** Two terms are of one *family* when they are:
> - two records;
> - two laws, two kernels, or a law and a kernel, in either order;
> - two functions;
> - two batches over equal levels, which means the same level names, in the same order, with the same sizes.
>
> Two types are of one family when they are two input specs, two output specs, two record types, two types of laws or kernels, two function types, or two batch types.

> [!IMPORTANT]
> **Definition (feeds).** In `a & b`, an output `c` of `b` *feeds* the input `c` of `a`, when `a` has an input of that name: the value `b` offers at `c` becomes the value `a` receives at `c`.

> [!IMPORTANT]
> **Definition (shared input).** In `a & b`, an input that `a` and `b` both have, and that `b` does not feed, is *shared*.

> [!IMPORTANT]
> **Definition (merge).** The *merge* `a & b` of two terms of one family is the term with:
> 1. **Outputs:** the outputs of `a`, followed by the outputs of `b`.
> 2. **Inputs:** the inputs of `a` that `b` does not feed, followed by the inputs of `b` that are not shared. A shared input is one input of the result, which supplies both operands. Its type is the unification of its two types (§ 2.5), and it is optional only when it is optional in both operands.
> 3. **Kind:** two records give a record; two laws or kernels give a kernel when a required input remains, and a law when none does; two functions give a function; two batches give a batch over the same levels.
> 4. **Produced term:** the record of all its outputs. The part from `b` is produced first, from the values of the inputs `b` takes. The part from `a` is produced next, from the values of the inputs `a` takes, among them the values that `b` feeds. § 5.3 states this for each family.

> [!IMPORTANT]
> **Definition (valid merge).** The merge `a & b` exists only when all of these conditions hold, and otherwise it is refused:
> 1. **One family:** `a` and `b` are of one family.
> 2. **Produced terms that a record can hold:** each operand's produced term is a base value or a record, and its type is known. So no output has a type hole, and a function states its inputs and its return type.
> 3. **One source for each output:** no name is an output of both `a` and `b`.
> 4. **The order rule:** no input of `b` is an output of `a`.
> 5. **Separate sides:** within each operand, no name is both an input and an output.
> 6. **Fed types conform:** the type of each output of `b` that feeds an input of `a` conforms to the type of that input (§ 2.5).
> 7. **Shared types unify:** the two types of each shared input unify (§ 2.5).
> 8. **One dimension scope:** all the types of both operands form one dimension scope (§ 2.3), so no dimension name stands for two sizes, and no two different unbound names meet at one axis.

§ 5.4 gives the reason for each condition.

A figure draws a merge with three more conventions.

![The legend for merges. In a & b, a wire runs from an output of b to the input of a with the same name. The result of a merge is a dashed boundary with the result's ports on its edges, and one port that fans out to both operands is a shared input.](../figures/legend-merge.svg)

The figure below shows every part of the definition at once, for two kernels `a(u, v | x, w)` and `b(x, z | w, q)`.

![The merge a & b of two kernels. b's output x feeds a's input x. The input w of both operands is one port of the result, which fans out to both. The result offers u, v, x, and z on its left edge, and takes w and q on its right edge.](../figures/merge-general.svg)

- `b`'s output `x` feeds `a`'s input `x`, so `x` is not an input of the result.
- `w` is an input of both operands, and `b` does not feed it, so it is shared: the result has one input `w`, whose port fans out to both operands.
- The result offers `u` and `v`, from `a`, then `x` and `z`, from `b`. Feeding `x` to `a` does not remove it from the outputs.
- The result takes `w`, then `q`. Required inputs remain, so `a & b` is a kernel.

**Example: a likelihood and a prior.**
Take the kernel `lik(y | mu)`, which draws `y ~ Normal(mu, 1)`, and the law `prior(mu, tau)`, with `mu ~ Normal(0, 5)` and `tau ~ Normal(1, 1)`.
In `lik & prior`, `prior`'s output `mu` feeds `lik`'s input `mu`.
No input remains, so the result is a law, with the outputs `y`, `mu`, and `tau`.
`lik` offers its array draw at the root under `y`, so in the joint draw `{y, mu, tau}` the same value is the field `y`.

![The merge lik & prior. The output mu of prior feeds the input mu of lik. The result is a law offering y, mu, and tau, with no inputs.](../figures/merge-kernel-law.svg)

`prior & lik` is refused by the order rule: `lik`, on the right, takes `mu`, which `prior`, on the left, outputs.
Written `lik & prior`, the term that produces `mu` stands on the right, as in the density `p(y | mu) p(mu, tau)`.

**Example: two kernels that share an input.**
Take two kernels of one parameter `theta`: `lik_nee(nee | theta)` and `lik_lai(lai | theta)`.
Neither outputs `theta`, so in `lik_nee & lik_lai` the input `theta` is shared.
The result is a kernel with the one input `theta`, and a value bound to `theta` supplies both operands.
If `lik_nee` declares `theta` as an array of shape `("p",)` and `lik_lai` as an array of shape `(3,)`, the result's `theta` is their unification, an array of shape `(3,)`.

![The merge lik_nee & lik_lai. The result has one input theta, which fans out to the input theta of both operands. The result offers nee and lai.](../figures/merge-shared-input.svg)

> [!WARNING]
> **A later merge can feed or share an optional input of a law built by merging.**
> Take `lik_opt(y | mu, s)`, which draws `y ~ Normal(mu, s)` with the default `s = 1`, and the prior above.
> `lik_opt & prior` is a law: `mu` is fed, and `s` keeps its default, so the law takes no inputs.
> Yet a later merge still sees `s`.
> Let `s_prior` be a law with `s ~ Normal(100, 0.001)`: merging `lik_opt & prior` with `s_prior` feeds `s`, and `y`'s scale then follows `s`, near 100.
> The reason is that a merge reads a law or a kernel through its factors, as the note below states.
> § 5.5 lists the consequences.

<details>
<summary>Technical note: factors, and the inputs that a law's factors still take</summary>

> [!IMPORTANT]
> **Definition (factor).** A merge of laws and kernels keeps its operands' factors.
> The *factors* of a law or a kernel `t` are:
> - for a law or a kernel built by merging, the factors it keeps;
> - for any other law or kernel, `t` itself.
>
> A *grouped sub-joint*, the factor that a rename creates when it gathers outputs of several factors under one group (§ 4.5), stays one factor.

> [!IMPORTANT]
> **Definition (remaining inputs of the factors).** The *remaining inputs of the factors* of a law or a kernel `t` are the inputs, required or optional, that `t`'s factors take and that no factor of `t` feeds, in order of first appearance:
> - for a kernel, its inputs;
> - for a law of a distribution family, such as `Normal`, none;
> - for a law built by merging, the optional inputs that its factors left at their defaults;
> - for a grouped sub-joint, its own inputs when it is a kernel, and none when it is a law, so the optional inputs inside it are not read.

For a law or a kernel operand, every use of "the inputs of `a`" or "the inputs of `b`" in the definitions of this section means the remaining inputs of its factors.
The order rule then reads: no remaining input of `b`'s factors, required or optional, is an output of `a`.
The result keeps `a`'s factors, followed by `b`'s.

For a record, a function, and a type, the remaining inputs are its inputs.
A type, and an element of a batch in a merge of batches, is read as one factor.

For example, `lik_opt & prior` is a law with no inputs, and the remaining inputs of its factors are `s`, optional.
So `(lik_opt & prior) & s_prior` feeds `s`, and `s_prior & (lik_opt & prior)` is refused by the order rule.

</details>

<details>
<summary>Technical note: merging types</summary>

A merge of two types describes the merge of two terms of those types:
- **Two output specs** give the output spec of two terms side by side, with nothing fed, since an output spec has no inputs. Its type is the record type of the first output spec's outputs, followed by the second's. An output at the root named `n` becomes the child `n`, so its value moves from the root to the path `n`. A record contributes its children unchanged, so every path of either operand is a path of the result, with the same type. Conditions 2, 3, and 8 apply.
- **Two input specs** give the input spec of two terms that take their inputs side by side, with nothing fed, since an input spec has no outputs. The inputs appear in order of first appearance. A shared input has the unification of its two types, and it is optional only when it is optional in both. The two operands form one dimension scope, whose bindings the result does not keep.
- **Two record types** give the record type of both sets of children. A top-level name in both is refused by condition 3.
- **Two types of laws or kernels, two function types, or two batch types** give the type of the merged term.

An input spec and an output spec are of two families, so they do not merge.
The inputs that an output spec's outputs would feed are the input spec with one required input for each output, of the output's type.
Merging those input specs describes the inputs that several terms' outputs would supply together.

</details>

<details>
<summary>Technical note: terms of two families do not merge</summary>

A record could be read as the law that gives the record itself with probability one, and a function as the kernel whose law at each input is a point mass at its return value.
Under that reading, a kernel and a record, such as `lik & {mu: 0.3}`, would have a meaning.
A law with a point-mass factor has no joint density, and the factors of a joint law are laws and kernels only (IV.1).
So terms of two families are refused by condition 1.

</details>

### 5.3 The value of a merge

The definition of merge states the result's signature and the order in which its produced term is made.
This subsection states the produced term for each family.

**Laws and kernels.**
Let `u` be values of the result's inputs, written as a record (§ 3.1), and write:
- `u_a` for the values in `u` of the inputs that `a` takes and `b` does not feed;
- `u_b` for the values in `u` of the inputs that `b` takes.

> [!IMPORTANT]
> **Definition (the law of a merge).** A draw `z` of `a & b` at `u` is made by drawing the right operand first:
> 1. draw `z_b` from `b` at `u_b`;
> 2. draw `z_a` from `a` at `u_a`, together with the values of `z_b` at the names that `b` feeds;
> 3. `z` is the record of outputs of `z_a` (§ 3.4), followed by the record of outputs of `z_b`.

When both operands have densities, the same law has the density

```math
p_{a \mathbin{\&} b}(z \mid u) \;=\; p_a\!\left(z_a \mid u_a,\ z_{\text{fed}}\right)\; p_b\!\left(z_b \mid u_b\right)
```

Here `z_a` and `z_b` are the two operands' draws, rebuilt from their outputs in `z` (§ 3.4), and `z_fed` is the values in `z` of the names that `b` feeds.

For `lik & prior`, the prior is drawn first, giving `{mu, tau}`.
Then `lik` is drawn at that `mu`, giving an array `y`.
The joint draw is `{y, mu, tau}`, and the density is the product of § 5.1:

```math
p(y, \mu, \tau) = p_{\text{lik}}(y \mid \mu)\; p_{\text{prior}}(\mu, \tau)
```

**Records.**
A record takes no inputs, so nothing is fed, and `a & b` is the record of `a`'s children followed by `b`'s children.

> [!NOTE]
> **Consequence (records merge by top-level name).** A group is one child, so two records that both have a group `g` are refused by condition 3, even when the two groups hold different fields.

For example, `{a: 1.0} & {b: 2.0, g: {c: 3.0}}` is the record `{a, b, g: {c}}`, while `{g: {c: 1.0}} & {g: {e: 2.0}}` is refused.

![The merge of the records r1 = {a} and r2 = {b, g: {c}}: the result offers a, then b and g.](../figures/merge-records.svg)

> [!TIP]
> To merge two records that share the group `g`, merge the groups first and put the result back: with removal and selection of § 4, that is `(r1 without g) & (r2 without g) & {g: r1[g] & r2[g]}`.

**Functions.**
A function is the kernel whose law at each value of its inputs is a point mass at its return value.
So two functions merge as two kernels do, with each draw replaced by a return value: `(a & b)(u)` is the record of outputs of `a`'s return value at `u_a` and the fed values, followed by the record of outputs of `b`'s return value at `u_b`.

For example, take two functions of an array `y` of shape `("n",)`: `center(center | y)`, which returns the mean of `y`, and `spread(spread | y)`, which returns its standard deviation.
`center & spread` takes `y` once, as a shared input, and returns the record `{center, spread}`.
For `standardize(z | y, center, spread)`, which returns `(y - center) / spread`, the merge `standardize & (center & spread)` feeds `center` and `spread`, shares `y`, and returns `{z, center, spread}`.

**Batches.**
Two batches over equal levels merge element by element: the element of `a & b` at each position is the merge of the operands' elements at that position.
An element that is a base value is read as its record of outputs, so it merges as a record.
For example, a batch of 3 draws of a law over an array `x` and a batch of 3 draws of a law over an array `y` merge into a batch of 3 records `{x, y}`.

<details>
<summary>Technical note: a shared optional input that a binding omits</summary>

When a binding supplies a shared input, one value supplies every operand that takes it.
When a binding omits a shared optional input, each factor takes its own default.
A default belongs to the term that owns the input, and an input spec holds none.
So one input can then hold two values, one in each factor, and the law at the defaults need not be the kernel at any single value of that input.

</details>

<details>
<summary>Technical note: a merge that leaves only optional inputs</summary>

The kind of a merge of laws and kernels depends on the required inputs that remain.
A kernel whose inputs are all optional is a kernel on its own, while a merge that leaves only optional inputs is a law, which samples and scores its factors at their defaults.
So for a kernel `k_opt` whose one input `s` is optional, `k_opt & z_law`, for a law `z_law` with `z ~ Normal(0, 1)`, is a law, sampled at the default of `s`, although sampling `k_opt` alone asks for `s` (IV.2).

</details>

<details>
<summary>Technical note: batches merge element by element, each element as one factor</summary>

In a merge of batches, each element is read as one operand, whose input spec and output spec are those of the elements' type.
The conditions of § 5.2 are checked once, on those types.
An element that is a law is read as one factor, so the optional inputs that its own factors left at their defaults are not fed in a merge of batches.

</details>

### 5.4 Why each condition holds

| Condition | Why |
| --- | --- |
| 1. One family | Each family has its own rule for the produced term (§ 5.3) |
| 2. Produced terms that a record can hold | The result holds each operand's outputs as fields, and each operand's produced term must be rebuilt from them |
| 3. One source for each output | An output offers a value, and two operands offering one name would give it two values |
| 4. The order rule | It keeps the result's sides separate, and makes the written order one in which each operand is produced after the operands that feed it |
| 5. Separate sides | Whether a name is fed or shared is then decided by the names alone |
| 6. Fed types conform | The fed values are given, so the input must admit each of them |
| 7. Shared types unify | One value supplies both inputs, so neither type is preferred |
| 8. One dimension scope | A dimension name stands for one size wherever types meet (II.1) |

<details>
<summary>Technical note: why produced terms must be values that a record can hold</summary>

The produced term of `a & b` holds the value at each output of both operands as a field.
Each operand's own produced term must then be rebuilt from those fields, since a joint law scores each factor's density on that factor's own draw (IV.2).
A base value and a record are rebuilt from their record of outputs without loss (§ 3.4).
A produced law, kernel, function, or batch is not a record and carries no name, so the merge would have to split it into its outputs or invent a name for it.
The code that produces it wraps it instead (§ 3.3), keeping it whole under a name it chooses.

This condition rests on rebuilding, not on the rule that a produced law offers no outputs.

</details>

<details>
<summary>Technical note: why outputs feed only from right to left</summary>

If a factor of `b` took an input that `a` outputs, nothing in the merge would supply it, since only `b` feeds `a`, so the result would keep that input while it also outputs the name: one name on both sides.
With the order rule, each operand in a chain `a & b & c` is fed only from its right, so the written order lists every factor after the factors that feed it, and the density reads from left to right.

The order rule does not make associativity hold: a chain of kernels has the same inputs, outputs, and validity in every grouping with or without it.
Nor does it let a merge build every acyclic graph one module at a time.
Take the kernels `c(z | y)` and `a(y | x)`, a law `b` over `x`, and `cb = c & b`.
Both `a & cb` and `cb & a` are refused, while `c & a & b` builds the same graph from its factors.

</details>

<details>
<summary>Technical note: a merge never lifts</summary>

An output feeds an input only when its type conforms to the input's type.
A merge does not apply the receiving term to each draw of a law that it is fed instead of a value.
Doing so would ignore the dependence between that law and everything else the receiving term is given, which only the merge itself can see.

</details>

<details>
<summary>Technical note: two selections of one law merge as independent factors</summary>

For a hierarchical law `hier` over `{s1: {theta}, s2: {theta}, mu}`, the selections `hier[[s1]]` and `hier[[s2]]` (§ 4.2) merge into the product of their two marginal laws.
A merge reads names and factors, and a selection of a law is one factor, so the merge cannot tell two selections of one law from two independent laws.
The joint law of both groups, which keeps their dependence, is the selection `hier[[s1, s2]]`.

The selections `hier[s1]` and `hier[s2]` both offer `theta`, so their merge is refused by condition 3.
Wrapping the two selections would silence that refusal and drop the dependence between the groups, so the refusal's message points to `hier[[s1, s2]]` (§ 19).

</details>

### 5.5 Properties of merge

Each property below follows from the definitions of § 5.2 and § 5.3.
Appendix D says where each is proved.

> [!NOTE]
> **Property (separate sides).** No name is both an input and an output of `a & b`.

> [!NOTE]
> **Property (paths are kept).** Every path of an operand is a path of `a & b`, with the same type there once dimension names are bound.
> An output of an operand at the root, named `n`, becomes the path `n` of the result.

For example, merging a law over an array `x` with a law over `{a, g: {c, e}}` gives a law over `{x, a, g: {c, e}}`.
The paths `a`, `g`, `g/c`, and `g/e` are unchanged, and `x`, at the root in its operand, becomes the path `x`.

> [!NOTE]
> **Property (the specs of a merge).** The output spec of `a & b` is the merge of the operands' output specs.
> The input spec of `a & b` is the merge of `a`'s input spec without the inputs that `b` feeds and `b`'s input spec.
> Both hold once dimension names are bound.

> [!NOTE]
> **Property (associative when both groupings are valid).** When `(a & b) & c` and `a & (b & c)` are both valid, they have the same factors, the same outputs in the same order, the same inputs with the same types and optional flags, and the same value.
> This holds for laws, kernels, records, record types, output specs, and batches of records or of base values.

> [!NOTE]
> **Property (commutative up to order).** When neither operand feeds the other, `a & b` and `b & a` are valid together and have the same law or value.
> Their outputs come in two orders, so their output specs are unequal, and their input specs are equal.

> [!NOTE]
> **Property (sampling commutes with merge).** For two laws `a` and `b` that do not feed each other and whose draws are records, when `a & b` is valid, a draw of `a & b` is distributed as the merge of a draw of `a` and a draw of `b`.

> [!NOTE]
> **Property (renaming commutes with merge).** Renaming the names of `a & b` gives the merge of the operands renamed, under the conditions of the note below.

> [!NOTE]
> **Property (labels are not read).** Relabeling an operand changes only the result's label, which joins the operands' labels with `·`, as `lik·prior`.

<details>
<summary>Technical note: when the validity of two groupings differs</summary>

The two groupings of `a & b & c` check different pairs of types, so one can be refused while the other is valid, in two cases:
1. a name that `a` and `b` both take and `c` outputs, whose types in `a` and `b` do not unify while `c`'s type conforms to each;
2. two different unbound dimension names at one axis in one grouping, which the other grouping binds first.

Input specs and functions keep no dimension scope after a merge, so for them the two groupings also agree only when no dimension name that one grouping binds is unbound in the operand it is merged with later.
For three input specs `J1` with `u` of shape `("n",)`, `J2` with `u` of shape `(3,)` and `q` of shape `("n",)`, and `J3` with `w` of shape `("n",)`, the grouping `(J1 & J2) & J3` binds `n` to 3 in its first merge and forgets it, so `w` keeps the shape `("n",)`, while `J1 & (J2 & J3)` gives `w` the shape `(3,)`.

For batches of laws or of kernels, associativity can fail, since a merge of batches reads each element as one factor and so drops the element's factors and dimension scope.

</details>

<details>
<summary>Technical note: the conditions under which renaming commutes with merge</summary>

The rename `m` maps names to names, one to one over the names of `a` and `b`, and renames inputs and outputs alike.
`m_a` and `m_b` are its restrictions to each operand's names.
Then renaming `a & b` by `m` gives `a` renamed by `m_a` merged with `b` renamed by `m_b`.
This holds for laws, kernels, records, and functions, provided `m` renames no remaining input of a law operand's factors.
A law's rename does not rename such an input, while the merged kernel's rename may, so the two sides can differ there.

A rename that moves a node to a path does not commute with merge: moving `c` and `e` under `g` in two operands before merging them is refused at `g` by condition 3.

</details>

<details>
<summary>Technical note: optional inputs across merges, and their consequences</summary>

A merge reads its operands' factors (§ 5.2), so an optional input that a factor left at its default can be fed or shared by a later merge.
For `M = lik_opt & mu_law`, where `mu_law` is a law with `mu ~ Normal(0, 0.001)`, `M` is a law in which `y`'s scale is near 1:
1. **A later merge can feed it.** `(lik_opt & prior) & s_prior` feeds `s`, so `y`'s scale is near 100, the same law as `lik_opt & (prior & s_prior)`. This keeps associativity.
2. **A later merge can share it.** For a kernel `k2(w | s)` that neither feeds nor is fed by `M`, `M & k2` is a kernel whose input `s` is required, and binding `s` to 30 gives `y` a scale near 30. Condition 7 also checks the types of such a shared input, so a merge can be refused over an input that no type of `M` shows.
3. **It appears with no sharing.** For an unrelated kernel `kw(v | w)`, `M & kw` is a kernel with two inputs: `s`, which is optional, and `w`. A law, which has no inputs, gains the input `s` through a merge with a term that does not mention it.
4. **Equal laws merge differently.** `lik_opt & prior` and `lik & prior` are equal laws, since `lik` is `lik_opt` at `s = 1`, yet merging each with `s_prior` gives `y` a scale near 100 in the first and near 1 in the second.
5. **Whether a result keeps the factor depends on the operation.** A result keeps an optional input that a factor left at its default exactly when it keeps that factor unchanged:

   | Law built from `M` | Keeps `lik_opt` as a factor | Merged with `s_prior`, `y`'s scale is near |
   | --- | --- | --- |
   | the selection `M[[y, mu]]` | no: a selection is one factor | 1 |
   | the marginal law of `M` at `[y, mu]`, as a merge of factors | yes | 100 |
   | `M` with `y` renamed to `w/y` | yes | 100 |
   | `lik_opt & mu_law & tau_law`, for a law `tau_law` over `tau`, conditioned on `tau`, on its exact route | yes | 100 |
   | `M` conditioned on observed `y` | no: a law of stored draws | — |
   | `lik_opt & z_law`, for a law `z_law` over `z`, conditioned on `mu` | no: `lik_opt` is bound | 1 |
   | a law regrouped so that `lik_opt` is inside a grouped sub-joint | no: the sub-joint is one factor | 1 |

   So a selection and a marginal law of one joint, which § 4.1 defines as one law, can merge differently.
6. **A merge of types can be valid where the merge of terms is refused.** `s_prior`'s type merged with the type of `lik_opt & prior` reads a law's type, which states no remaining inputs of factors, so it feeds nothing and is valid, while `s_prior & (lik_opt & prior)` is refused by the order rule. So the type of a merge equals the merge of the types only when neither operand's factors take a remaining input that its input spec does not state.
7. **A merge of batches reads each element as one factor**, so an element's hidden optional inputs are not fed.

</details>

<details>
<summary>Technical note: dimension scopes across merges</summary>

A merged law or kernel keeps its dimension scope: the sizes it bound to dimension names.
A merged input spec, output spec, record type, or function applies those sizes to its types and keeps nothing more.
Two consequences follow:
1. **Merges of specs and of functions are associative only under a condition**, which the note on validity states.
2. **A kept scope can refuse a merge over a name that no type shows.** Let `m1 = a & b3`, for a kernel `a(ya | u)` whose `ya` and `u` both have shape `("n",)` and a law `b3` over `u` of shape `(3,)`: its event is `{ya, u}`, both of shape `(3,)`, and it keeps `n = 3`. Let `m2 = k & lv`, for a kernel `k(w | v)` with both of shape `("n",)` and a law `lv` over `v` of shape `(4,)`, which keeps `n = 4`. Then `m1 & m2` is refused by condition 8, and renaming `n` in `m1` first does not help, since neither event spec contains `n`. The merge of a draw of `m1` and a draw of `m2` is still a valid merge of records, so sampling commutes with merge only when `a & b` is valid.

Keeping the scope is what makes the merge of laws and kernels associative; forgetting it is what keeps merges of specs simple.

</details>

### 5.6 What merge is not

- **Merge is not composition.** The composition `f ∘ g` passes `g`'s whole result to `f` by position and returns only `f`'s result, and the product of linear operators `A B` is a composition too. A merge matches by name and returns the outputs of both operands.
- **Merge is not conditioning.** Conditioning `lik` on `mu = 0.5` fixes `mu`, and the result does not produce `mu`. In `lik & prior`, `mu` is fed by a law and stays an output.
- **Merge is not renaming.** To merge under new names, rename an operand first (§ 4.5), then merge.

---

## 14. Records: `Record` and `RecordSpec`

A record is a tree of named values (§ 1.1), and its type is a record type (§ 2.3).
`Record` is the class of records, and `RecordSpec` is the class of record types.
A record is its own produced term, so its outputs are its children (§ 3.3, rule 2), and records merge side by side (§ 5.3).

### 14.1 The API

Only the members that this design defines or changes are shown.
`RecordSpec` and `Record` share the tree interface of `NamedTree` (II.6), and a `RecordSpec` is the record kind's `TermSpec`.

```python
class RecordSpec(NamedTree[TermSpec], TermSpec):
    """The type of a record: a tree of names whose fields hold term specs.

    A record spec is the record kind's term spec, and the schema of one
    structured value, such as a draw. Its paths are the paths of the records it
    types (§ 1.2), and indexing it gives the type of the same part of those
    records.
    """

    def __init__(self, field_specs: Mapping[str, TermSpec | Mapping | tuple] | None = None, /,
                 **fields: TermSpec | Mapping | tuple) -> None:
        """Build the type from its children.

        A tuple of sizes is the array type of that shape, with no dtype and no
        support. A mapping is a group. The positional mapping also accepts
        paths as keys, such as "g/c".

        Raises:
            TypeError: A field is given None, which means a pending type; an
                opaque field is OpaqueSpec().
        """

    def __getitem__(self, paths: str | Sequence[str], /) -> TermSpec:
        """The type at a path, or the record type of the nodes at a list of paths.

        One path gives the type of the node there: a field's term spec, or a
        group's RecordSpec. A list of paths gives the RecordSpec of the chosen
        nodes, each at its own path, in this spec's order.

        Raises:
            KeyError: A path is not a path of this spec.
            ValueError: The path reader refuses the argument (§ 12).
        """

    def __contains__(self, path: object, /) -> bool:
        """Whether path is a str and self[path] succeeds."""

    def keys(self) -> tuple[str, ...]:
        """The paths of the fields, in canonical order; groups are not listed."""

    def without(self, *paths: str) -> RecordSpec:
        """This type without the nodes at paths, each other node at its path.

        A group left with no fields goes too.

        Raises:
            KeyError: A path is not a path of this spec.
            ValueError: Nothing would be left, a path is given twice, or one
                path is inside another.
        """

    def rename(self, renames: Mapping[str, str] | None = None, /,
               **keyword_renames: str) -> RecordSpec:
        """Rename or move nodes by their paths, by the rules of II.6 (§ 4.5)."""

    def __and__(self, other_record_spec: RecordSpec, /) -> RecordSpec:
        """The children of both types, this type's first (§ 5.2).

        Raises:
            TypeError: The other operand is not a RecordSpec.
            ValueError: A top-level name is in both, a group included, or a
                dimension name is bound to two sizes.
        """


class Record(NamedTree[Any], TrackedTerm):
    """A tree of named values.

    A record is its own produced term, so its outputs are its children, and its
    output spec is OutputSpec(record.spec), derived and never stored. An empty
    record has no output spec.
    """

    def __init__(self, fields: Mapping[str, Any], /, *, label: str | None = None) -> None:
        """Build a record from a nested mapping, in which a mapping is a group.

        A field may hold any tracked term, such as a law, which keeps its
        identity. The paths of the record continue into that term (§ 2.4).
        """

    @property
    def spec(self) -> RecordSpec:
        """The record's type, with every dimension name bound."""

    def __getitem__(self, paths: str | Sequence[str], /) -> Any:
        """The node at a path, or the record of the nodes at a list of paths.

        One path gives the node itself: a field's value as a tracked term of
        its kind, a group as a Record, or the term that a field holds. A path
        continues into a term that a field holds, so record["G/mu"] is
        record["G"]["mu"]. A list of paths gives the record of the chosen
        nodes, each at its own path, in this record's order, labelled by its
        paths, as "(a, g/c)".

        Raises:
            KeyError: A path is not a path of this record; the message lists
                its paths.
            ValueError: The path reader refuses the argument (§ 12).
        """

    def __contains__(self, path: object, /) -> bool:
        """Whether path is a str and self[path] succeeds, so groups count."""

    def keys(self) -> tuple[str, ...]:
        """The paths of the fields, in canonical order; groups are not listed."""

    def at_path(self, *segments: str) -> Any:
        """self[path], for the path that joins segments with "/"; [] is the recommended form."""

    def output(self, output_name: str, /) -> Any:
        """The term at the output output_name, which is the child of that name (§ 4.6).

        Raises:
            KeyError: output_name is not a child of this record.
        """

    def without(self, *paths: str) -> Record:
        """This record without the nodes at paths, each other node at its path.

        A group left with no fields goes too, and so does a field holding a
        term all of whose leaf paths are removed.

        Raises:
            KeyError: A path is not a path of this record.
            ValueError: Nothing would be left, a path is given twice, or one
                path is inside another.
        """

    def rename(self, renames: Mapping[str, str] | None = None, /,
               **keyword_renames: str) -> Record:
        """Rename or move nodes by their paths, by the rules of II.6 (§ 4.5)."""

    def select(self, *paths: str, **remap: str) -> dict[str, Any]:
        """The nodes at paths, as a dict for passing as keyword arguments.

        Each path is keyed by its last name, and each keyword argument names
        its entry: select(x="g/c") is {"x": self["g/c"]}.

        Raises:
            ValueError: Two paths have the same last name.
        """

    def __and__(self, other_record: Record, /) -> Record:
        """The fields of both records, this record's first (§ 5.3).

        Raises:
            TypeError: The other operand is not a Record.
            ValueError: A top-level name is in both, a group included.
        """
```

### 14.2 Examples

Each example builds what it uses.

**A record and its parts.**
A record's paths name every node, groups included, while `keys()` lists only its fields.

```python
from probpipe import Record

draw = Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}}, label="draw")
draw.keys()          # ("a", "g/c", "g/e"): the fields
draw["a"]            # NumericArray 1.0, labelled "a"
draw["g"]            # Record {c: 2.0, e: 3.0}: a group is a node too
draw["g/c"]          # NumericArray 2.0
draw["g"]["c"]       # the same node as draw["g/c"]: paths chain
"g" in draw          # True, since draw["g"] succeeds
"g" in draw.keys()   # False, since g is a group, not a field
```

**Its outputs and its spec.**
The record offers its children, `a` and `g`, as the figure of § 3.5 draws.

```python
from probpipe import OutputSpec, Record

draw = Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}})
draw.spec                        # RecordSpec(a=NumericArraySpec(()), g=RecordSpec(c=..., e=...))
OutputSpec(draw.spec).outputs    # {a: NumericArraySpec(()), g: RecordSpec(c=..., e=...)}
draw.output("g")                 # Record {c: 2.0, e: 3.0}, the same as draw["g"]
```

![The record r with the outputs a and g on its left edge, and the fields c and e drawn as a tree below g.](../figures/record-parts.svg)

**Several paths, and removing.**
A list of paths keeps each chosen node at its own path, in the record's order, and removing is its complement.

```python
from probpipe import Record

draw = Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}})
draw["a", "g/c"]     # Record {a, g: {c}}, labelled "(a, g/c)"
draw["g/e", "a"]     # Record {a, g: {e}}: the record's order, not the order given
draw[["g"]]          # Record {g: {c, e}}: a list of one path gives a record that holds the node
draw.without("g/e")  # Record {a, g: {c}}, the complement of draw["a", "g/c"]
draw.without("g/c", "g/e")   # Record {a}: the emptied group goes
```

**A record type, indexed.**
A `RecordSpec` takes the same paths as the records it types, and gives their types.

```python
from probpipe import RecordSpec

spec = RecordSpec(a=(), g=RecordSpec(c=(), e=()))
spec["g"]                  # RecordSpec(c=NumericArraySpec(()), e=NumericArraySpec(()))
spec["g/c"]                # NumericArraySpec(())
spec[["a", "g/c"]]         # RecordSpec(a=NumericArraySpec(()), g=RecordSpec(c=NumericArraySpec(())))
"g/c" in spec              # True
spec.without("g/c", "g/e") # RecordSpec(a=NumericArraySpec(()))
```

**A field that holds a law.**
The paths of a record continue into a term that a field holds, while its fields are only its own.

```python
from probpipe import Normal, Record

post = Normal("mu", 0.0, 1.0) & Normal("tau", 1.0, 1.0)   # a law over {mu, tau}
holder = Record({"n": 3.0, "G": post})
holder.keys()          # ("n", "G"): the fields
holder["G"]            # the law post
holder["G/mu"]         # the law of mu, post["mu"]
holder["G"]["mu"]      # the same law
holder["n", "G/mu"]    # Record {n, G}, whose field G holds post[["mu"]]
"G/mu" in holder       # True
```

**Renaming and moving.**
A rename gives the node at a path a new path, which can move it to another group.

```python
from probpipe import Record

draw = Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}})
draw.rename({"g/c": "g/k"})   # Record {a, g: {k, e}}: renamed in place
draw.rename({"g/c": "c"})     # Record {a, g: {e}, c}: c moves to the top level, appended last
draw.rename(a="b")            # Record {b, g: {c, e}}
```

**Merging two records.**
Records merge by top-level name, so a group that both records have is refused.

```python
from probpipe import Record

r1 = Record({"a": 1.0})
r2 = Record({"b": 2.0, "g": {"c": 3.0}})
r1 & r2       # Record {a: 1.0, b: 2.0, g: {c: 3.0}}
r2 & r1       # Record {b, g: {c}, a}: the same fields, in the other order

Record({"g": {"c": 1.0}}) & Record({"g": {"e": 2.0}})
# ValueError: g is an output of both records; rename one, or merge the two groups first
```

![The merge of the records r1 = {a} and r2 = {b, g: {c}}: the result offers a, then b and g.](../figures/merge-records.svg)

**Passing fields to a function.**
`select` serves keyword arguments, so it keys each entry by the last name of its path.

```python
from probpipe import Record

draw = Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}})
draw.select("a", "g/c")    # {"a": ..., "c": ...}
draw.select(x="g/c")       # {"x": ...}

hier_draw = Record({"s1": {"theta": 0.1}, "s2": {"theta": 0.2}})
hier_draw.select("s1/theta", "s2/theta")
# ValueError: s1/theta and s2/theta both have the last name theta; name one with a keyword
```

<details>
<summary>Technical note: the errors of records</summary>

| Call, for `draw = Record({"a": 1.0, "g": {"c": 2.0, "e": 3.0}}, label="draw")` | Error | Reason |
| --- | --- | --- |
| `draw["x"]` | `KeyError`: "'x' is not a path of 'draw'; its paths: ['a', 'g', 'g/c', 'g/e']" | the path does not exist |
| `draw["g", "g/c"]` | `ValueError`: "'g/c' is inside 'g'; pick one" | a path inside another would hold one value twice |
| `draw["a", "a"]` | `ValueError` | the same path twice |
| `draw[[]]`, `draw[()]` | `ValueError` | no paths at all |
| `draw["a", 0]` | `TypeError`: "a path and a position cannot be mixed in one []" | a record has no positions |
| `draw.without(("a", "g/e"))` | `TypeError`: pass the paths as separate arguments | `without` takes one path per argument |
| `draw.without("a", "g")` | `ValueError` | nothing would be left |
| `draw.rename({"a": "g"})` | `ValueError` | the target collides with an existing node (II.6) |
| `draw & {"b": 2.0}` | `TypeError` | a plain mapping is not a record |
| `Record({"g": {"c": 1.0}}) & Record({"g": {"e": 2.0}})` | `ValueError` | condition 3 of § 5.2: `g` is an output of both |

</details>
