"""Cold start: manual plan + declared preferences, nothing more."""

import pytest

from edupaal import EduPAALSkill, LearnerPreferences, MasteryLevel
from tests.conftest import make_evidence


def test_cold_start_creates_plan_and_preferences(store, graph, prefs):
    skill = EduPAALSkill(store, graph)
    plan = skill.cold_start(
        learner_id="ada",
        node_selection=["linear-equations", "linearization"],
        preferences=prefs,
    )
    assert plan.learner_id == "ada"
    assert plan.topic_ids == ["linear-equations", "linearization"]
    assert plan.version == 1

    saved = store.get_preferences("ada")
    assert saved.learning_style == "visual"
    assert saved.pace == "steady"

    # everything starts unevaluated
    assert skill.effective_mastery("linear-equations") == MasteryLevel.UNKNOWN
    assert skill.next_topic().id == "linear-equations"


def test_cold_start_honors_order(store, graph, prefs):
    skill = EduPAALSkill(store, graph)
    plan = skill.cold_start(
        learner_id="ada",
        node_selection=["linear-equations", "linearization"],
        preferences=prefs,
        order=["linearization", "linear-equations"],
    )
    assert plan.topic_ids == ["linearization", "linear-equations"]


def test_cold_start_rejects_bad_input(store, graph, prefs):
    skill = EduPAALSkill(store, graph)
    with pytest.raises(ValueError):
        skill.cold_start(learner_id="x", node_selection=[], preferences=prefs)
    with pytest.raises(ValueError):
        skill.cold_start(
            learner_id="x", node_selection=["nope"], preferences=prefs
        )
    with pytest.raises(ValueError):
        # concepts are not traversable topics
        skill.cold_start(
            learner_id="x", node_selection=["algebra"], preferences=prefs
        )
    with pytest.raises(ValueError):
        skill.cold_start(
            learner_id="x",
            node_selection=["linear-equations", "linearization"],
            preferences=prefs,
            order=["linear-equations"],  # not a permutation
        )


def test_methods_require_cold_start(store, graph):
    skill = EduPAALSkill(store, graph)
    with pytest.raises(ValueError):
        skill.next_topic()
