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


def test_recold_start_bumps_plan_version(store, graph, prefs):
    skill = EduPAALSkill(store, graph)
    plan1 = skill.cold_start(
        learner_id="ada",
        node_selection=["linear-equations"],
        preferences=prefs,
    )
    assert plan1.version == 1
    plan2 = skill.cold_start(
        learner_id="ada",
        node_selection=["linear-equations", "linearization"],
        preferences=prefs,
    )
    assert plan2.version == 2
    # latest version wins deterministically
    assert store.get_plan_for_learner("ada").id == plan2.id
    assert skill.next_topic().id == "linear-equations"


def test_cold_start_rejects_unknown_override_node(store, graph, prefs):
    from edupaal import MasteryParams

    skill = EduPAALSkill(store, graph)
    with pytest.raises(ValueError, match="unknown node"):
        skill.cold_start(
            learner_id="x",
            node_selection=["linear-equations"],
            preferences=prefs,
            criteria_overrides={"nope": MasteryParams()},
        )
