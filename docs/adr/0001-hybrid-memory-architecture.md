# ADR 0001: Hybrid memory architecture — LLM-extracted style memory alongside the relational progression store

- **Status:** Accepted
- **Date:** 2026-09-07
- **Deciders:** Vignesh Radhakrishna

## Context

EduPAAL's canonical store is exact structured JSON with fail-loud writes:
`learners`, `preferences`, `nodes`, `prerequisites`, `plans`, `evidence`,
`mastery_records`, `overrides`. The `Mem0Provider` and `MemOSProvider`
(`edupaal/providers/`) deliberately persist every canonical record **verbatim**
— no LLM rewriting — with read-after-write verification, so third-party engines
give the same exactness guarantees as the default `SQLiteBackend`.

Mem0's native architecture stores LLM-extracted memories instead: the model
summarizes interactions into searchable memory entries. This is richer and more
flexible than a fixed schema, but it is lossy and non-deterministic. In the
native Mem0 evaluation (seed 7), the identical S2 scenario scored mastery
0.75, 0.25, and 0.50 across three runs — the extraction LLM gave three
different answers to the same inputs.

The question: should EduPAAL also store extracted memories like Mem0, and what
does that mean for MemOS?

## Decision

Adopt a **two-layer hybrid**:

1. **Canonical layer (unchanged).** Knowledge progression stays relational,
   exact, and fail-loud: curriculum hierarchy (`Space → Subject → Concept →
   Topic`), prerequisite graph with cycle checks, append-only evidence,
   versioned mastery records, overrides, and plans. This layer is the sole
   source of truth for progression decisions (what the learner knows, what
   comes next, what is gated).

2. **Extracted layer (new).** Learning style and observed learner memory follow
   Mem0's architecture: LLM-derived memory entries, tagged by learner (and by
   topic where relevant), explicitly marked as derived/non-authoritative, with
   provenance recording which interactions they were extracted from.

3. **Declared vs. observed split.** `LearnerPreferences` (cold-start manual
   input: `learning_style`, `pace`, `extra`) stays structured — the learner
   said it, so we store it exactly. Only *observed/inferred* style goes
   through LLM extraction.

### Guardrails

- **Advisory-only.** Progression and mastery logic reads the canonical layer
  exclusively. A "prefers fast pace" memory may change *how* a vertical
  presents material; it must never promote, demote, gate, or override
  prerequisites for a topic.
- **Provenance required.** Extracted memories carry their source interactions,
  consistent with EduPAAL's provenance principle, so a bad inference stays
  traceable and correctable.
- **Thin-evidence caution.** Inferred traits should carry evidence counts or
  confidence; one offhand comment must not crystallize into a permanent
  "learning style".

## Consequences

- **Richer personalization without weaker guarantees.** Semantic search over
  style memory, tutor-facing learner summaries, and natural-language queries
  become possible, while mastery claims remain verifiable.
- **Mem0 becomes a natural engine for the extracted layer** (managed or
  self-hosted), independent of its role as a canonical-record provider.
- **MemOS remains a candidate** for either layer, pending verification that it
  can faithfully implement the `StorageBackend` contract for canonical records
  (fail-loud duplicates, append-only evidence, cycle-checked prerequisites,
  insertion-order tie-breaking).
- **Implementation (not decided here):** a new `extracted_memories`
  table/collection in the backends, kept outside the canonical eight tables'
  semantics; extraction prompts, embedding choice, and refresh policy are
  follow-up decisions.

## Notes

- The governing test for which layer a fact belongs in: *where does wrongness
  hurt?* A wrong mastery claim misroutes the learner's progression
  (load-bearing → canonical). A wrong style guess only mistunes presentation
  (soft → extracted).
- This ADR does not change `StorageBackend` semantics, the mastery rubric, or
  the fail-loud contract. It adds a layer; it weakens nothing.
