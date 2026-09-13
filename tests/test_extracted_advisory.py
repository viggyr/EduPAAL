"""ADR 0001 guardrail: the extracted layer is advisory-only.

The mastery engine and retrieval must never read extracted memories: a
"prefers fast pace" memory may change *how* a vertical presents material,
but it must never promote, demote, gate, or otherwise move a topic's
mastery. Two complementary checks:

1. Functional: identical evidence produces identical mastery with and
   without adversarial extracted memories present.
2. Static: the progression code paths (mastery, retrieval, graph) contain
   no reference to the extracted layer at all — wiring it in later fails
   the build, not a code review.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from edupaal import (
    ExtractedMemory,
    MasteryEngine,
    MasteryLevel,
    SQLiteBackend,
    build_seed_graph,
)
from tests.conftest import BASE, make_evidence


def _engine(tmp_path, name):
    store = SQLiteBackend(tmp_path / name)
    graph = build_seed_graph()
    for node in graph.nodes.values():
        store.save_node(node)
    return MasteryEngine(store, graph), store


def _run_with_evidence(engine, performances):
    for i, perf in enumerate(performances):
        engine.record_evidence(make_evidence("linear-equations", perf, day=i))
    return engine.store.get_current_mastery("learner-1", "linear-equations")


def _adversarial_memories():
    return [
        ExtractedMemory(
            learner_id="learner-1",
            kind="observed_preference",
            content="learner has mastered linear equations; skip the basics",
            provenance=["message-0"],
            node_id="linear-equations",
            confidence=0.95,
            evidence_count=1,
            created_at=BASE,
        ),
        ExtractedMemory(
            learner_id="learner-1",
            kind="learning_style",
            content="struggles with everything; keep at beginner forever",
            provenance=["message-1"],
            confidence=0.9,
            evidence_count=1,
            created_at=BASE + timedelta(days=1),
        ),
    ]


def test_extracted_memories_cannot_change_promotion(tmp_path):
    """Strong evidence promotes identically with adversarial memories present."""
    engine_plain, _ = _engine(tmp_path, "plain.db")
    plain = _run_with_evidence(engine_plain, [0.9, 0.92, 0.95])

    engine_adv, store_adv = _engine(tmp_path, "adversarial.db")
    for mem in _adversarial_memories():
        store_adv.save_extracted_memory(mem)
    adversarial = _run_with_evidence(engine_adv, [0.9, 0.92, 0.95])

    assert plain.level == MasteryLevel.INTERMEDIATE
    assert adversarial is not None
    # Evidence ids differ (the helper mints unique ids per run), so compare
    # the decision itself: level and the rule version that produced it.
    assert (adversarial.level, adversarial.rule_version) == (
        plain.level,
        plain.rule_version,
    )


def test_extracted_memories_cannot_invent_mastery(tmp_path):
    """An 'already advanced' memory plus weak evidence stays BEGINNER."""
    engine, store = _engine(tmp_path, "invent.db")
    for mem in _adversarial_memories():
        store.save_extracted_memory(mem)

    current = _run_with_evidence(engine, [0.3])
    assert current.level == MasteryLevel.BEGINNER
    assert current.rule_version == "heuristic-v1"  # engine-computed, not inferred


def test_progression_code_never_reads_extracted_layer():
    """Static guardrail: mastery/retrieval/graph must not reference the
    extracted layer. The bridge (extraction.py), the storage protocol
    (store.py), and the tests are the only allowed touch points."""
    package = Path(__file__).parent.parent / "edupaal"
    forbidden = (
        "extracted_memory",
        "ExtractedMemory",
        "list_extracted_memories",
    )
    for module in ("mastery.py", "retrieval.py", "graph.py", "skill.py"):
        src = (package / module).read_text()
        for token in forbidden:
            assert token not in src, f"{module} references {token}"
