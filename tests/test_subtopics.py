"""Recursive sub-topics: decomposition, full-chain rollup, depth-aware prereqs."""

from datetime import timedelta

import pytest

from edupaal import (
    EduPAALSkill,
    KnowledgeNode,
    MasteryLevel,
    NodeLevel,
)
from edupaal.entities import _utcnow
from tests.conftest import make_evidence, promote_to


def test_seed_decomposes_regularization_two_levels(graph):
    assert {n.id for n in graph.children("regularization")} == {"l1-l2", "dropout"}
    assert {n.id for n in graph.children("dropout")} == {
        "dropout-rate",
        "inverted-dropout",
    }
    # topics_under sees the whole subtree
    assert {n.id for n in graph.topics_under("regularization")} == {
        "regularization",
        "l1-l2",
        "dropout",
        "dropout-rate",
        "inverted-dropout",
    }


def test_topic_cannot_hang_under_subject(graph):
    with pytest.raises(ValueError):
        graph.add_node(
            KnowledgeNode(
                id="bad", level=NodeLevel.TOPIC, name="Bad", parent_id="maths"
            )
        )


def test_prereqs_exist_at_subtopic_depth(graph):
    assert "l1-l2" in graph.prerequisites["dropout"]
    assert "dropout-rate" in graph.prerequisites["inverted-dropout"]


def test_prereq_cycle_rejected_at_depth(graph):
    with pytest.raises(ValueError):
        graph.add_prerequisite("l1-l2", "dropout")  # closes dropout -> l1-l2


def test_rollup_walks_full_chain(skill):
    promote_to(skill, "dropout-rate", "advanced")
    r = skill._retriever_for()

    level, score = r.rollup("dropout")
    assert (level, score) == (MasteryLevel.ADVANCED, 2.0)
    level, score = r.rollup("regularization")
    assert (level, score) == (MasteryLevel.ADVANCED, 2.0)
    for nid in ("overfitting", "ml", "science"):
        level, _ = r.rollup(nid)
        assert level == MasteryLevel.ADVANCED, nid


def test_parent_topic_aggregates_children(skill):
    promote_to(skill, "l1-l2", "beginner")  # score 0
    promote_to(skill, "dropout-rate", "advanced")  # dropout -> 2.0
    level, score = skill._retriever_for().rollup("regularization")
    # mean(0, 2.0) = 1.0 -> INTERMEDIATE
    assert (level, score) == (MasteryLevel.INTERMEDIATE, 1.0)


def test_evidence_convention_finest_grained_node(skill):
    for day, perf in enumerate([0.90, 0.88, 0.92]):
        skill.record_evidence(make_evidence("dropout-rate", perf, day=day))
    assert skill.effective_mastery("dropout-rate") == MasteryLevel.INTERMEDIATE
    level, score = skill._retriever_for().rollup("dropout")
    assert (level, score) == (MasteryLevel.INTERMEDIATE, 1.0)


def test_override_on_parent_topic_wins_over_rollup(skill):
    promote_to(skill, "dropout-rate", "advanced")
    skill.set_override(
        MasteryLevel.BEGINNER,
        reason="recheck requested",
        scope_node_id="dropout",
        expires_at=_utcnow() + timedelta(days=1),
    )
    level, _ = skill._retriever_for().rollup("dropout")
    assert level == MasteryLevel.BEGINNER


def test_evidence_on_decomposed_topic_rejected(skill):
    with pytest.raises(ValueError, match="finest-grained"):
        skill.record_evidence(make_evidence("regularization", 0.9))
    with pytest.raises(ValueError, match="finest-grained"):
        skill.assert_mastery(
            "regularization", MasteryLevel.ADVANCED,
            asserted_by="tutor-agent", reason="exam",
        )


def test_next_topic_with_subtopics_in_plan(store, graph, prefs):
    skill = EduPAALSkill(store, graph)
    skill.cold_start(
        learner_id="sub-learner",
        node_selection=["dropout-rate", "inverted-dropout"],
        preferences=prefs,
    )
    assert skill.next_topic().id == "dropout-rate"
    promote_to(skill, "dropout-rate", "advanced", learner_id="sub-learner")
    # inverted-dropout's depth-level prereq now clears the gate
    assert skill.next_topic().id == "inverted-dropout"
    promote_to(skill, "inverted-dropout", "advanced", learner_id="sub-learner", day=10)
    assert skill.next_topic() is None
