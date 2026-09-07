"""Retrieval: next_topic gating, rollup, weakest/strongest."""

import pytest

from edupaal import MasteryLevel, NodeLevel
from tests.conftest import promote_to


def test_next_topic_honors_prerequisites(skill):
    # fresh learner: first topic with no prerequisites
    assert skill.next_topic().id == "linear-equations"
    # linearization is gated behind linear-equations
    promote_to(skill, "linear-equations", "beginner")
    assert skill.next_topic().id == "linear-equations"  # still needs work
    promote_to(skill, "linear-equations", "advanced")
    assert skill.next_topic().id == "linearization"


def test_next_topic_skips_advanced_and_respects_chain(skill):
    promote_to(skill, "linear-equations", "advanced")
    promote_to(skill, "linearization", "advanced")
    # bias-variance needs linearization (cross-concept prereq)
    assert skill.next_topic().id == "bias-variance"
    promote_to(skill, "bias-variance", "advanced")
    assert skill.next_topic().id == "regularization"
    # regularization is decomposed: advance it through its sub-topics
    promote_to(skill, "l1-l2", "advanced")
    promote_to(skill, "dropout-rate", "advanced")
    promote_to(skill, "inverted-dropout", "advanced")
    assert skill.next_topic() is None


def test_prereq_gate_defaults_to_intermediate(store, graph, prefs):
    from edupaal import EduPAALSkill
    from tests.conftest import promote_to as _pt

    # plan order puts the gated topic first: it stays locked until its
    # prerequisite clears the INTERMEDIATE gate
    skill = EduPAALSkill(store, graph)
    skill.cold_start(
        learner_id="gate-learner",
        node_selection=["linearization", "linear-equations"],
        preferences=prefs,
    )
    assert skill.next_topic().id == "linear-equations"  # linearization gated
    _pt(skill, "linear-equations", "beginner", learner_id="gate-learner")
    assert skill.next_topic().id == "linear-equations"  # still gated: gate is INTERMEDIATE
    _pt(skill, "linear-equations", "intermediate", learner_id="gate-learner", day=10)
    assert skill.next_topic().id == "linearization"  # gate cleared


def test_rollup_concept_subject(skill):
    promote_to(skill, "linear-equations", "advanced")  # score 2
    promote_to(skill, "linearization", "intermediate")  # score 1
    retriever = skill._retriever_for()

    level, score = retriever.rollup("algebra")
    assert score == 1.5
    assert level == MasteryLevel.INTERMEDIATE  # tie at 1.5 rounds down deterministically

    level, score = retriever.rollup("maths")
    assert score == 1.5  # only algebra has evaluated topics

    level, score = retriever.rollup("overfitting")
    assert level == MasteryLevel.UNKNOWN and score is None


def test_weakest_and_strongest(skill):
    promote_to(skill, "linear-equations", "advanced")
    promote_to(skill, "linearization", "intermediate")

    weakest = skill.weakest(2, NodeLevel.TOPIC)
    assert [n.id for n, _, _ in weakest] == ["bias-variance", "dropout"]
    assert all(lvl == MasteryLevel.UNKNOWN for _, lvl, _ in weakest)

    strongest = skill.strongest(2, NodeLevel.TOPIC)
    assert [n.id for n, _, _ in strongest] == ["linear-equations", "linearization"]

    # concept level works too
    weakest_concepts = skill.weakest(1, NodeLevel.CONCEPT)
    assert weakest_concepts[0][0].id == "overfitting"


def test_grounding_packet_shape(skill):
    promote_to(skill, "linear-equations", "advanced")
    packet = skill.grounding_packet("linearization")

    assert packet["preferences"]["learning_style"] == "visual"
    assert packet["preferences"]["pace"] == "steady"
    assert packet["plan"]["total_topics"] == 4
    assert packet["plan"]["advanced_topics"] == 1
    assert packet["next_topic"]["id"] == "linearization"
    mc = packet["mastery_context"]
    assert mc["node"]["id"] == "linearization"
    by_level = {a["level"]: a for a in mc["ancestors"]}
    assert by_level["concept"]["id"] == "algebra"
    assert by_level["subject"]["id"] == "maths"
    assert by_level["space"]["id"] == "science"
    assert mc["ancestors"][0]["id"] == "algebra"  # nearest first
    assert packet["active_override"] is None


def test_grounding_packet_subtopic_ancestors(skill):
    packet = skill.grounding_packet("dropout-rate")
    mc = packet["mastery_context"]
    assert mc["node"]["id"] == "dropout-rate"
    # several ancestors share level "topic": the list preserves the chain
    assert [a["id"] for a in mc["ancestors"]] == [
        "dropout", "regularization", "overfitting", "ml", "science",
    ]


def test_effective_mastery_rolls_up_non_leaf_nodes(skill):
    # no direct records can ever exist for a decomposed topic or concept —
    # effective_mastery must report the rollup, not UNKNOWN
    assert skill.effective_mastery("regularization") == MasteryLevel.UNKNOWN
    promote_to(skill, "dropout-rate", "advanced")
    assert skill.effective_mastery("dropout") == MasteryLevel.ADVANCED
    assert skill.effective_mastery("regularization") == MasteryLevel.ADVANCED
    assert skill.effective_mastery("overfitting") == MasteryLevel.ADVANCED


def test_grounding_packet_counts_decomposed_plan_topics(skill):
    # the quickstart-style plan lists "regularization" (decomposed); advancing
    # its sub-topics must move the plan's advanced_topics count
    assert skill.grounding_packet("linear-equations")["plan"]["advanced_topics"] == 0
    promote_to(skill, "l1-l2", "advanced")
    promote_to(skill, "dropout-rate", "advanced")
    promote_to(skill, "inverted-dropout", "advanced")
    packet = skill.grounding_packet("linear-equations")
    assert packet["plan"]["advanced_topics"] == 1
    assert packet["plan"]["total_topics"] == 4


def test_ranking_rejects_negative_n(skill):
    with pytest.raises(ValueError):
        skill.weakest(-1)
    with pytest.raises(ValueError):
        skill.strongest(-1, NodeLevel.CONCEPT)
