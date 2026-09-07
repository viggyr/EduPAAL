"""SQLite backend round-trips: versioned, append-only history."""

from edupaal import (
    Evidence,
    KnowledgeNode,
    LearnerPreferences,
    MasteryLevel,
    MasteryParams,
    NodeLevel,
)
from tests.conftest import BASE, make_evidence


def test_node_and_prerequisite_roundtrip(store, graph):
    for node in graph.nodes.values():
        store.save_node(node)
    store.save_prerequisite("linearization", "linear-equations")

    assert store.get_node("algebra").name == "Algebra"
    assert store.get_prerequisites("linearization") == ["linear-equations"]
    assert len(store.list_nodes(NodeLevel.TOPIC)) == 8


def test_plan_with_criteria_overrides_roundtrip(store):
    from edupaal import LearningPlan
    from datetime import timezone

    plan = LearningPlan(
        id="plan1",
        learner_id="l1",
        topic_ids=["t1", "t2"],
        criteria_overrides={"c1": MasteryParams(k_evidence=5)},
        created_at=BASE,
    )
    store.save_plan(plan)
    loaded = store.get_plan_for_learner("l1")
    assert loaded.topic_ids == ["t1", "t2"]
    assert loaded.criteria_overrides["c1"].k_evidence == 5


def test_evidence_and_mastery_history_are_append_only(skill, store):
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=1))

    assert len(store.list_evidence("learner-1", "linear-equations")) == 2
    history = store.get_mastery_history("learner-1", "linear-equations")
    assert len(history) == 1  # still BEGINNER: only the first-engagement record
    assert history[0].level == MasteryLevel.BEGINNER

    current = store.get_current_mastery("learner-1", "linear-equations")
    assert current.id == history[-1].id


def test_preferences_roundtrip(store):
    store.save_preferences(
        "l1", LearnerPreferences(learning_style="socratic", pace="fast",
                                 extra={"grade": 5})
    )
    prefs = store.get_preferences("l1")
    assert prefs.learning_style == "socratic"
    assert prefs.extra == {"grade": 5}
    assert store.get_preferences("nobody") is None


def test_override_crud(store):
    from edupaal import DynamicOverride
    from datetime import timedelta, timezone
    from datetime import datetime

    ovr = DynamicOverride(
        learner_id="l1",
        level=MasteryLevel.INTERMEDIATE,
        scope_node_id="t1",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        reason="r",
    )
    store.save_override(ovr)
    assert len(store.list_overrides("l1")) == 1
    store.delete_override(ovr.id)
    assert store.list_overrides("l1") == []
