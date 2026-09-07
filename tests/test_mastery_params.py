"""The knobs are real: retuning parameters changes outcomes."""

from edupaal import EduPAALSkill, LearnerPreferences, MasteryLevel, MasteryParams
from tests.conftest import make_evidence, promote_to


def _fresh_skill(store, graph, prefs, overrides):
    skill = EduPAALSkill(store, graph)
    skill.cold_start(
        learner_id="learner-1",
        node_selection=["linear-equations"],
        preferences=prefs,
        criteria_overrides=overrides,
    )
    return skill


def test_lower_k_promotes_earlier(store, graph, prefs):
    skill = _fresh_skill(
        store, graph, prefs, {"linear-equations": MasteryParams(k_evidence=2)}
    )
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=0))
    # with defaults this would still be BEGINNER (2 < 3); with k=2 it promotes
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=1))
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE


def test_higher_threshold_blocks_promotion(store, graph, prefs):
    skill = _fresh_skill(
        store, graph, prefs, {"linear-equations": MasteryParams(t_intermediate=0.99)}
    )
    for day in range(3):
        skill.record_evidence(make_evidence("linear-equations", 0.9, day=day))
    assert skill.effective_mastery("linear-equations") == MasteryLevel.BEGINNER


def test_concept_level_override_applies_to_its_topics(store, graph, prefs):
    # overrides keyed by concept cascade to topics underneath
    skill = _fresh_skill(
        store, graph, prefs, {"algebra": MasteryParams(k_evidence=2)}
    )
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=1))
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE


def test_cross_modal_gate_blocks_advanced_by_default(skill):
    # three strong quiz-only evidences: INTERMEDIATE, but not ADVANCED
    for day, perf in enumerate([0.90, 0.88, 0.92]):
        skill.record_evidence(make_evidence("linear-equations", perf, day=day))
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE


def test_cross_modal_gate_can_be_relaxed(store, graph, prefs):
    skill = _fresh_skill(
        store, graph, prefs,
        {"linear-equations": MasteryParams(cross_modal_advanced=False)},
    )
    for day, perf in enumerate([0.90, 0.88, 0.92]):
        skill.record_evidence(make_evidence("linear-equations", perf, day=day))
    assert skill.effective_mastery("linear-equations") == MasteryLevel.ADVANCED


def test_params_are_recorded_in_effect(store, graph, prefs):
    skill = _fresh_skill(
        store, graph, prefs, {"linear-equations": MasteryParams(k_evidence=2)}
    )
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=1))
    history = skill.mastery_history("linear-equations")
    assert history[-1].params_in_effect["k_evidence"] == 2
    assert history[-1].rule_version == "heuristic-v1"
