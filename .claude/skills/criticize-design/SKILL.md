---
name: criticize-design
description: Critically interview the user about a plan or design until every design-significant branch reaches shared understanding. Use when the user wants to stress-test a plan, clarify an uncertain idea, compare real alternatives, expose edge cases, or be grilled before implementation.
allowed-tools: Read Grep Glob Bash(gh *) Bash(git *) Agent
---

# Establish the frame

- Identify the decision being designed, its scope, its constraints, and what would count as complete.
- Default to discussion-only and read-only work. Do not edit code, documentation, issue trackers, or external systems unless the user explicitly asks.
- Read repository instructions, design principles, relevant documentation, code, issues, and change history when they can answer a question or constrain the design.
- Build a decision tree in dependency order. Maintain a decision ledger with each branch marked `unresolved`, `provisional`, `accepted`, or `rejected`, plus its rationale and downstream dependencies.

# Resolve the tree

- Work through one unresolved branch at a time, starting with the branch that constrains the most downstream decisions.
- If code, documentation, or other available evidence can answer a question, investigate it instead of asking the user.
- Separate design decisions from implementation trivia. Grill only choices with meaningful semantic, user-facing, architectural, operational, or irreversible consequences; defer naming and local mechanics unless they affect the design.
- For every user-facing or high-risk decision, test the proposal with a concrete API example or scenario. Include random, batched, failure, or boundary cases only when they materially probe the contract.
- If a branch reveals new design-significant sub-branches, add them explicitly to the decision tree before continuing. Do not hide them inside the parent branch or claim completion early.

For each question, provide a recommendation:

- If there is one clearly optimal solution for the current constraints, present that solution and ask for approval.
- If there are genuinely competitive solutions, present only options at the **same level**. Explain each approach, its advantages and disadvantages, the core difference, which option you recommend for the current context, and when another option would win.
- Never invent an inferior alternative merely to create a comparison.
- Ask one question at a time and wait for feedback before advancing.

# Challenge responsibly

- Check every proposal against the project's stated philosophy, domain language, public contracts, and current direction. Surface contradictions directly and offer a concrete resolution when possible.
- Be persistent, not tedious. Apply Occam's razor, stay concise, and avoid over-design or excessive criticism.
- Mark scope-expanding or hard-to-reverse discoveries as major decisions. Do not silently fold them into a smaller accepted branch.
- Treat tentative language such as "for now", "provisionally", or "let's start with this" as `provisional`, not `accepted`.
- When the user asks for status, report accepted, provisional, and unresolved branches separately, then resume from the next dependency.
- When a user's description is vague, first try to understand it using the most plausible explanation. If that fails, use the Socratic method: asking until the user to clarify the vague concept or acknowledge this thought is self-contradictory.

# Close the session

Before saying the design is complete, audit at least:

- public API and mathematical/domain meaning;
- lifecycle, mutability, and ownership;
- randomness, batching, correlation, and jointness where relevant;
- failure behavior and diagnostics;
- provenance, persistence, and extension boundaries;
- dependencies, migration, non-goals, and implementation order;
- contradictions and every provisional or unresolved decision.

Do not claim completion while a design-significant sub-branch remains hidden or unresolved. End with a self-contained summary of accepted decisions, provisional decisions, unresolved questions, rationale, consequences, and recommended next steps. If the user asked for an implementation-ready design, include representative public API examples and explicit boundaries.
