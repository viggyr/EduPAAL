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

## Storage backends

`SQLiteBackend` (stdlib only) is the default. `StorageBackend` is a protocol —
a Mem0 adapter is the intended next backend (documented in `edupaal/store.py`,
not implemented): same protocol, semantic retrieval for evidence, mastery math
untouched.

## Status

Draft reference implementation (v0.1.0). Clean, tested core — not feature-complete.
Out of scope for this draft: demotion/decay heuristics, a universal knowledge graph
(seed exemplars only), the Mem0 adapter, and any claim of empirical efficacy.
