# EduPAAL

**EduPAAL** (Personalization Architecture for Agentic Academic Learning) is a portable
**memory-management skill** for personalized structured learning — grades 1 through PhD,
plus self-directed study through online or other resources.

It is **not** a tutor, and it does **not** run learning progression. It is a thin skill
interface that feeds vertical learning agents (tutor, quest, assessment, visualization,
practice) the learner's preferences and pace, the current plan state, and the next
specific topic. The core contribution is underneath that interface:

- **Storage architecture** — entity schema, a four-level knowledge graph
  (Space → Subject → Concept → Topic), prerequisite edges, versioned records.
- **Mastery engine** — PAAL *owns* mastery transitions. Vertical agents report
  normalized evidence; transparent, deterministic, versioned heuristics turn that
  evidence into Beginner / Intermediate / Advanced mastery, with `Unknown` before
  any evidence exists.
- **Retrieval** — next topic honoring prerequisites and mastery gates; top/bottom-N
  strongest/weakest topics, concepts, and subjects via mastery rollup.

## Design principles

1. **Shared learning memory is primary.** One concept, many modalities, one memory.
   The quiz agent, the practice agent, and the visualization agent all write evidence
   about the same topic into the same store, and mastery advances on the combined
   picture. No vertical re-assesses from scratch.
2. **PAAL owns the mastery heuristics — but every knob is tunable.** The framework
   defines the *shape* of the promotion rules and ships transparent defaults; the
   deployment controls the parameters (evidence count, time window, performance bars,
   cross-modal requirements), globally or per concept/topic. Defaults are documented
   as *reasonable, not optimal* — no claim of optimality is made.
3. **Vertical-topology agnostic.** One universal agent or many fragmented specialists:
   the skill contract is identical. Provenance (`source_agent`, `activity_type`) on
   every evidence record is what makes multi-writer memory trustworthy.

## The mastery heuristics (heuristic-v1)

- First evidence for a topic: `Unknown → Beginner`.
- Promotion evaluates the up-to-**K** most recent evidence items inside a **W**-day
  window: need ≥K items, mean performance ≥ the bar for the next level
  (`t_intermediate` / `t_advanced`), and no item below `t_contradict` (it blocks).
- `Intermediate → Advanced` additionally requires evidence from **≥2 distinct
  activity types** when `cross_modal_advanced` is on (default) — acing quizzes alone
  is not the same as demonstrating it in practice.
- Defaults: `K=3, W=30d, t_intermediate=0.65, t_advanced=0.80, t_contradict=0.40`.
- Every transition writes a `MasteryRecord` with the rule version, the **exact
  parameters in effect**, and the evidence IDs that caused it. History is append-only.
- `assert_mastery()` is the privileged path for assessment agents/humans: it appends
  an assertion record with full provenance; it never rewrites history.
- Dynamic overrides are temporary unless explicitly promoted into a durable assertion.
  Precedence: a node-scoped override beats a global one; among active overrides of
  the same scope the **most recently created wins** (a newer correction supersedes an
  older one). Promoting an override retires the scope's whole temporary stack, so
  the new assertion stands on its own.

## Verticals and quorum aggregation

Evidence is reported by **verticals** — the agentic processes around the learner
(a tutor vertical, an evaluator vertical, a practice app). `Evidence.source_agent`
is a vertical's **stable identity**: one vertical must use one stable value, and
different values create independent mastery tracks. Quiz/practice/dialogue are
`activity_type` values *within* a vertical, not verticals themselves.

- Each vertical's evidence promotes its **own track** through heuristic-v1
  independently, using that vertical's parameters.
- The shared (learner, topic) level is a **quorum aggregation** over the tracks
  (`aggregate_vertical_mastery`, exported from `edupaal`):
  - all tracks `UNKNOWN` → shared `UNKNOWN`;
  - `ADVANCED` requires **`xvertical_quorum_advanced`** tracks at `ADVANCED`
    (default 2) — one vertical's `ADVANCED` is not enough to transfer;
  - a lone `ADVANCED` track caps the shared level at `INTERMEDIATE`;
  - otherwise the shared level is the highest attested track level.
- Per-vertical promotion parameters resolve in this precedence order: **topic plan
  override → nearest ancestor plan override → per-vertical parameters → engine
  defaults**. The shared quorum uses the plan override or engine defaults, never
  one particular vertical's parameters.
- **Assertions are learner-level, not vertical attestations**: `assert_mastery()`
  floors the shared level (it bypasses the quorum) and assertion records never
  double-count as vertical evidence. A later assertion replaces the floor
  (including lowering it); evidence-driven aggregation may rise above it.

## Knowledge graph

Four fixed upper levels: `Space → Subject → Concept → Topic`. Topics are
**recursively decomposable**: a topic may have sub-topics to arbitrary depth
(e.g. `Regularization → Dropout → Dropout Rate Tuning`). Two edge types:
`part_of` (parent links, including topic→topic) and `prerequisite`
(topic→topic at any depth; may cross concepts and subjects; cycles rejected).

Mastery rolls up the full chain — sub-topic → … → topic → concept → subject →
space. Any node *with children* reports the mean of its children's scores
(`Unknown` children excluded); leaves report their own mastery; an explicit
override on a node always wins. `effective_mastery()` applies the same rollup,
so asking for a concept's mastery never returns `Unknown` merely because no
direct record exists. Evidence and `assert_mastery()` **must** target leaf
TOPIC nodes — the engine rejects anything else outright; higher levels are
pure rollup.

This package ships an exemplar graph (`edupaal.seed`: Science → Maths/ML →
Algebra/Overfitting → topics including a two-level decomposition under
Regularization); deployments bring their own via `KnowledgeGraph`.

## Quickstart

```bash
pip install -e .
python examples/quickstart.py
```

```python
from edupaal import EduPAALSkill, Evidence, LearnerPreferences, SQLiteBackend, build_seed_graph
from datetime import datetime, timezone

skill = EduPAALSkill(SQLiteBackend("edupaal.db"), build_seed_graph())
skill.cold_start(
    learner_id="sam",
    node_selection=["linear-equations", "linearization"],
    preferences=LearnerPreferences(learning_style="visual", pace="steady"),
)
skill.record_evidence(Evidence(
    id="e1", learner_id="sam", node_id="linear-equations",
    source_agent="quiz-agent", activity_type="quiz",
    occurred_at=datetime.now(timezone.utc), performance=0.9,
))
print(skill.grounding_packet("linearization")["next_topic"])
print([n.name for n, _, _ in skill.weakest(2)])
```

## Tests

```bash
pip install pytest
pytest
```

## Storage providers

`SQLiteBackend` (stdlib only) is the default. `StorageBackend` is a protocol,
and two optional **memory providers** implement it — Mem0 and MemOS — so a
deployment can swap the storage engine without touching framework logic
(mastery math is untouched; it only depends on the protocol):

| | `SQLiteBackend` | `Mem0Provider` (`edupaal[mem0-provider]`) | `MemOSProvider` (`edupaal[memos-provider]`) |
|---|---|---|---|
| Install | stdlib only | `pip install "edupaal[mem0-provider]"` (`mem0ai`, `fastembed`) | `pip install "edupaal[memos-provider]"` (`MemoryOS`) |
| Record format | SQL rows | one Mem0 memory per entity, canonical JSON text, `infer=False` (verbatim, no LLM rewriting) | one MemOS `TextualMemoryItem` per entity, canonical JSON text via direct CRUD (never `extract()`) |
| Lookup | SQL | exact metadata filters (`edupaal_kind`/`edupaal_id`/`learner_id`/`node_id`) — vector search never used for authoritative reads | in-memory scan over metadata — same filters, no semantic search |
| Embeddings | n/a | required by the vector store but never consulted; local `fastembed` by default, no API keys | not used (`NaiveTextMemory` is the non-vector text memory) |
| Persistence | single file | embedded Qdrant dir (`EDUPAAL_MEM0_QDRANT_PATH`, default `~/.edupaal/mem0_qdrant`) | JSON file, write-through on every mutation (`EDUPAAL_MEMOS_DIR`, default `~/.edupaal/memos`) |
| Telemetry | n/a | forced off on import (`MEM0_TELEMETRY` defaults to `False`) — learner data must not leave the machine | n/a |

All three pass the same conformance suite (`tests/test_backend_conformance.py`):
exact round trips for every entity, plan versioning, append-only evidence/mastery
history (duplicate ids raise `ValueError`), override ordering/deletion, and
durability across reopen. Provider tests skip — loudly, never as fake passes —
when the extra isn't installed.

Honest limitations: the providers scan O(n) records per lookup (fine for a
personalization store, not for millions of rows); Mem0 generates its own memory
ids (the EduPAAL id lives in metadata and record text); MemOS persists via
`dump()`/`load()` with no transactional locking. None of the providers change
what the framework computes — they only change where the records live.

### Extracted layer (ADR 0001)

Alongside the canonical store, every backend persists **extracted memories**
(`ExtractedMemory`): LLM-derived learning-style / observed-preference notes
that are explicitly **advisory-only and non-authoritative**. They are stored
verbatim with provenance (source interaction ids) and an evidence count, and
saves are **upserts** — re-extraction replaces the previous inference, unlike
the append-only evidence and mastery history.

The boundary is enforced, not just documented: the mastery engine and
retrieval never read extracted memories (`tests/test_extracted_advisory.py`
asserts identical mastery with and without adversarial memories present, and
statically that `mastery.py`/`retrieval.py`/`graph.py`/`skill.py` never
reference the layer). Vertical agents may use them to tune *how* material is
presented, never *what* the learner's mastery is.

`edupaal/extraction.py` is the one place that *runs* extraction: it drives
Mem0's `Memory.add(..., infer=True)` over interaction messages. Extraction
namespaces are learner-scoped (`edupaal:{learner_id}`) by default, and each
extracted item gets a deterministic id derived from the extraction inputs, so
re-running over the same interactions replaces rather than duplicates.

```bash
pip install "edupaal[mem0-provider]"    # or "edupaal[memos-provider]", or "edupaal[all-providers]"
python -c "
from edupaal import Mem0Provider  # or MemOSProvider
from edupaal import EduPAALSkill, build_seed_graph
skill = EduPAALSkill(Mem0Provider.from_env(), build_seed_graph())
"
```

## Status

Draft reference implementation (v0.1.0). Clean, tested core — not feature-complete.
Out of scope for this draft: demotion/decay heuristics, a universal knowledge graph
(seed exemplars only), and any claim of empirical efficacy.
