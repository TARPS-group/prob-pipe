---
name: criticize-with-docs
description: Extend a design criticizing session with repository-aware documentation discovery, terminology checks, decision status, and controlled documentation updates. Use when the user wants to stress-test a plan against existing code and documented intent, create a private working draft, or keep canonical design documentation aligned as decisions are accepted.
allowed-tools: Read Grep Glob Bash(gh *) Bash(git *) Agent
---

# Apply the core criticizing workflow

Use the `criticize-design` skill. Apply its decision tree, ledger, one-question-at-a-time rule, evidence gathering, option quality bar, scenario testing, and closure audit. This skill only adds documentation routing and recording rules.

# Choose the documentation mode

Infer the least expansive mode authorized by the user's request:

- **Discussion-only** — maintain the decision ledger in the conversation and make no file changes. Use this by default when the user asks only to discuss, review, or stress-test.
- **Working draft** — write to a user-approved draft location. Preserve `provisional` and `accepted` status explicitly. Use local-only or ignored storage when the user requests a private draft.
- **Canonical update** — update the repository's authoritative tracked documentation, but only when the user explicitly asks for documentation changes.

If the mode is unclear, continue read-only and ask only before the first write. Never promote a provisional decision into canonical documentation as accepted.

# Discover the repository's documentation system

Before recording anything:

1. Read repository instructions and documentation guidance such as `AGENTS.md`, `CONTRIBUTING.md`, and the README.
2. Discover existing design authorities and navigation: design references, RFCs, ADRs, architecture docs, API contracts, issue-based proposals, `CONTEXT.md`, or `CONTEXT-MAP.md`.
3. Identify which source describes the current implementation, which describes the target state, and which is merely a proposal.
4. Follow the repository's existing location, structure, terminology, and status conventions. Do not create `CONTEXT.md`, `CONTEXT-MAP.md`, or `docs/adr/` merely because they are absent.
5. If more than one document could be authoritative, surface the ambiguity and recommend one source of truth before writing.

# Classify conflicts before resolving them

When code, documentation, and the proposed design disagree, classify the mismatch before asking what to change:

- current-code defect;
- stale documentation;
- intentional target design ahead of implementation;
- new proposal beyond the current target;
- terminology mismatch without behavioral disagreement.

Do not call target-ahead-of-code documentation stale. State which layer each claim belongs to and preserve deliberate differences.

# Sharpen language and scenarios

- Challenge terminology against the repository's glossary or canonical design language when one exists.
- When language is vague or overloaded, propose a precise canonical term and identify aliases or meanings to avoid.
- Stress-test domain relationships and public contracts with concrete scenarios and edge cases.
- Cross-reference user claims with code and documentation. Surface contradictions with evidence rather than deference.
- Keep domain language separate from implementation names when the distinction matters.

# Record decisions without churn

- Record accepted decisions after a coherent decision cluster, not after every sentence.
- Keep provisional decisions visibly provisional in working drafts; omit them from accepted canonical contracts unless the repository has a proposed/open-points convention.
- Preserve rationale, rejected same-level alternatives, consequences, boundaries, and open points only when they help future readers understand the decision.
- Keep implementation facts in implementation-facing docs and domain/mathematical meaning in design-facing docs.
- Update navigation, cross-references, or supersession markers when the repository's conventions require them.
- Reconcile the updated document against the decision ledger before finishing.

# Route to an appropriate document profile

- **Repository-native design or API document — default.** Follow the existing format, such as Contract/Rationale/Open points, RFC sections, or API reference conventions.
- **Domain glossary/context document.** Use only when the repository already uses `CONTEXT.md`/`CONTEXT-MAP.md` or the user explicitly chooses that form. Then read and follow [CONTEXT-FORMAT.md](./CONTEXT-FORMAT.md).
- **ADR.** Offer or create one only when the decision is hard to reverse, surprising without context, and the result of a genuine trade-off. Then read and follow [ADR-FORMAT.md](./ADR-FORMAT.md) together with repository conventions.
- **Private or meeting draft.** Use a user-approved path, label status clearly, and keep it out of tracked documentation when requested.

# Close the documentation session

Report the documentation mode, files changed, decisions recorded, provisional/open decisions, conflicts found, and validation performed. Do not claim the canonical docs are aligned until the written result has been checked against the final decision ledger and repository navigation.
