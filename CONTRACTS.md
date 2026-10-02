# Contract Discipline

**Audience:** every contributor, human or agent, whose PR adds or changes code in
`probpipe/`. Read this before you start a change, and follow it in every PR.

## Why this exists

Each abstraction and operation of ProbPipe has a *contract*: its API, shapes,
return types, error cases, invariants, and parameter names. Those contracts must
be **explicit, documented, and obeyed**, for correctness and consistency, and
because the docstrings that state them are the source of the API reference.
Write every docstring as reference text for a user.

## Standing directives (apply to every PR)

1. **Contract-first — clarify before you implement.** Before writing code, make
   the contract of every abstraction you touch clear in your own understanding.
   Its design section states the target contract, and its docstrings state the
   contract as implemented. If they leave **any** part ambiguous — shape
   conventions, single-vs-batched behavior, canonical orderings, error/raise
   cases, return types, invariants — **resolve the ambiguity before coding**: ask
   the maintainer, or pin it explicitly, and record the decision in the PR. Do
   not guess and do not let an unclear contract reach the code.

2. **Document the contract fully, where it lives.** When you implement an abstraction, document its
   contract *completely* in the **docstring** (NumPy style: Parameters / Returns / Raises). The
   docstring must state the *precise* contract, not a vague summary:
   - exact return type and **shape**, distinguishing single vs batched
     (e.g. `(vector_size,)` vs `(*batch_shape, vector_size)`);
   - canonical orderings (e.g. the leaf-traversal order for vectorization);
   - **every** error/raise condition;
   - invariants and round-trip guarantees (e.g. `from_vector(to_vector(v)) == v`).

   **Lead with the contract; defer the rationale.** The opening of every docstring — the one-line
   summary and the first paragraph(s) — must describe the **API**: what the abstraction *is*, what it
   accepts and returns, and how a caller uses it. Write it as precise, clear reference text for a
   *user*, not as a design log: prefer plain language, define or avoid jargon, and state the contract
   directly. Keep design reasoning, motivation, history, and implementation trade-offs *out* of the
   opening; when including them is justified, put them **later** — typically in a NumPy-style
   **Notes** section (or an explicitly labelled interim-detail note, per directive 5). A reader
   skimming the first lines should learn how to *use* the abstraction correctly, not why it was built
   that way.

   **Code documentation must stand on its own — no references to transient artifacts.** Do not cite
   PRs, issues, tracking numbers, or other out-of-band discussion in docstrings or code comments.
   The contract is whatever the docstring states; a reader should never need to open a PR or issue to
   understand it. (Such references belong in commit messages and PR descriptions, not the code.)

3. **Analyze clarity; make the code obey the contract.** As part of every PR:
   - explicitly assess whether the contracts you touched are unambiguous, and call out any that
     are not;
   - verify the **code obeys the documented contract** — there must be no drift between docstring
     and behavior — and add tests that *assert the contract* (shapes, orderings, and error cases,
     not just happy-path values);
   - use **consistent variable names** for the same concept across the codebase: the names of
     `design/glossary.md` § Canonical names. Renaming for consistency within the files you touch
     is in scope; flag larger inconsistencies you cannot fix within scope. A new contract that
     introduces a recurring parameter adds its name to that table.

4. **Stay in scope, but never ship an undocumented or contradicted contract.** Work to the scope of
   your PR. If the work reveals a contract in `design/` or in a docstring that is wrong, missing, or
   unclear, fix the documentation, or raise it with the maintainer, rather than coding around it.

5. **Document to the design's target, not the stale status quo.** The design reference in
   `design/` describes the target state, and most PRs move the code only part of the way toward
   it. When you implement or touch an abstraction, its docstrings and naming must describe the
   **design's target contract and terminology**, *not* the current behavior of code elsewhere in
   the repo that the design will change. Stale neighboring code is not the reference; the design is.
   - **Precedence.** Where `design/` disagrees with the code or with a contributor document, the
     design decides, and the PR brings the code toward the design without asking. Ask the
     maintainer when two parts of the design disagree, when the design leaves a real choice open,
     when the change names a public API, or when the change would amend the design.
   - Where your change must temporarily coexist with or delegate to not-yet-migrated code, you
     *may* note that as an explicit, clearly-labeled **interim implementation detail** — but never
     let it define the contract or blur a distinction the design draws.
   - In particular, keep the design's **vocabulary distinctions** intact even before the code that
     enforces them lands. *Example:* `to_vector` / `from_vector` are **value**
     operations — a spec describes structure and does not depend on the value
     type, so it carries neither. `to_vector` is `NumericRecord.to_vector` /
     `NumericRecordBatch.to_vector`; `from_vector(label, spec, vec)` is the
     classmethod pair `NumericRecord.from_vector` (single) /
     `NumericRecordBatch.from_vector` (batched), each taking the spec as an
     argument. These are the
     **numeric** 1-D (de)serialization — they ravel and concatenate numeric leaves (require
     `is_numeric`). The **general** (de)composition keeps each leaf whole (any type): export with
     `list(record.values())` and reconstruct with `Record.from_field_values`, visited at the
     **spec's** granularity in canonical `keys()` order. Both treat a container-valued opaque
     leaf (tuple/namedtuple; a dict is never a leaf) as **one** leaf; JAX's `jax.tree_util.tree_flatten` is the finer
     pytree view that descends into it (`Record` is a registered pytree, but `flatten`/`unflatten`
     are **not** `Record` methods — use `jax.tree_util` directly). Do not describe these as
     interchangeable.

## Where each contract is stated

The section of `design/` that owns an abstraction states its target contract, and
`design/package-structure.md` § Correspondence to the implementation maps each module of the
code to the package and section of its target. The docstrings of a module state its contracts
as implemented.

## The per-PR checklist

`.github/PULL_REQUEST_TEMPLATE.md` carries the checklist of these directives, and every PR body
copies it from the template.
