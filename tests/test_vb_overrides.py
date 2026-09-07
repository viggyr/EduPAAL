"""OVERRIDE INTERPLAY: temporary corrections vs durable state.

The documented contract under test:
- Node-scoped override beats global.
- Among active same-scope overrides, the most recently created wins;
  the older one stays in history as an audit trail.
- Expired overrides stop shaping reads.
- Promoting an override retires the FULL temporary stack for that scope
  and writes a durable assertion record (rule_version "assertion-v1").
- Evidence arriving after an override/assertion still accumulates; the
  heuristic can promote past the asserted level but never demotes below it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from edupaal import MasteryLevel

from .vb_helpers import submit_all, vb_evidence, vb_skill

NODE = "linear-equations"
OTHER = "linearization"
BASE_NOW = datetime.now(timezone.utc)


def test_newest_same_scope_wins_and_history_kept(tmp_path):
    skill = vb_skill(tmp_path, learner_id="ovr-newest")
    first = skill.set_override(MasteryLevel.BEGINNER, reason="first guess", scope_node_id=NODE)
    second = skill.set_override(MasteryLevel.ADVANCED, reason="corrected", scope_node_id=NODE)
    assert skill.active_override(NODE).id == second.id
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
    # the superseded override is still in history (audit trail)
    assert {first.id, second.id} <= {o.id for o in skill.store.list_overrides("ovr-newest")}


def test_node_scoped_beats_global(tmp_path):
    skill = vb_skill(tmp_path, learner_id="ovr-scope")
    skill.set_override(MasteryLevel.BEGINNER, reason="global caution")
    skill.set_override(MasteryLevel.ADVANCED, reason="topic proven", scope_node_id=NODE)
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
    assert skill.effective_mastery(OTHER) == MasteryLevel.BEGINNER  # global applies
    # clearing the node scope reveals the global underneath
    skill.clear_override(NODE)
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER
    assert skill.active_override(NODE).scope_node_id is None


def test_expiry_releases_the_node(tmp_path):
    skill = vb_skill(tmp_path, learner_id="ovr-expire")
    skill.set_override(
        MasteryLevel.ADVANCED, reason="trial", scope_node_id=NODE,
        expires_at=BASE_NOW + timedelta(hours=1),
    )
    assert skill.active_override(NODE, now=BASE_NOW) is not None
    assert skill.active_override(NODE, now=BASE_NOW + timedelta(hours=2)) is None
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED  # not yet expired at BASE_NOW
    eff = skill.engine.effective_mastery("ovr-expire", NODE, now=BASE_NOW + timedelta(hours=2))
    assert eff == MasteryLevel.UNKNOWN  # expired: back to durable state


def test_promotion_retires_full_stack_and_asserts(tmp_path):
    skill = vb_skill(tmp_path, learner_id="ovr-promote")
    skill.set_override(MasteryLevel.BEGINNER, reason="stale", scope_node_id=NODE)
    skill.set_override(MasteryLevel.INTERMEDIATE, reason="updated", scope_node_id=NODE)
    skill.set_override(MasteryLevel.ADVANCED, reason="elsewhere", scope_node_id=OTHER)
    record = skill.promote_override(NODE)
    assert record.assertion is True
    assert record.rule_version == "assertion-v1"
    assert record.level == MasteryLevel.INTERMEDIATE  # the effective (newest) one
    assert record.evidence_ids == []  # assertion, not evidence-driven
    assert record.asserted_by == "override-promotion"
    assert "updated" in record.reason  # the promoted override's reason survives
    # the full temporary stack for the scope is retired: every same-scope
    # override is marked promoted, none shapes reads anymore
    leftovers = [o for o in skill.store.list_overrides("ovr-promote")
                 if o.scope_node_id == NODE]
    assert leftovers and all(o.promoted for o in leftovers)
    assert skill.active_override(NODE) is None
    assert skill.effective_mastery(NODE) == MasteryLevel.INTERMEDIATE
    # ...but other scopes are untouched
    assert skill.active_override(OTHER) is not None
    assert skill.active_override(OTHER).level == MasteryLevel.ADVANCED


def test_promote_with_no_active_override_fails_loudly(tmp_path):
    skill = vb_skill(tmp_path, learner_id="ovr-empty")
    with pytest.raises(ValueError, match="[Nn]o active override"):
        skill.promote_override(NODE)
    with pytest.raises(ValueError, match="[Nn]o active override"):
        skill.promote_override()  # nothing global either
    # promotion is a leaf-topic assertion: non-leaf scopes fail loudly
    skill.set_override(MasteryLevel.BEGINNER, reason="concept-level", scope_node_id="regularization")
    with pytest.raises(ValueError, match="cannot promote override"):
        skill.promote_override("regularization")
    # ...as do global overrides (scope them first)
    skill.set_override(MasteryLevel.BEGINNER, reason="global")
    with pytest.raises(ValueError, match="global overrides cannot be promoted"):
        skill.promote_override()


def test_evidence_after_override_still_accumulates(tmp_path):
    """An override shapes reads, but evidence keeps flowing into history.
    Once the override is cleared, the accumulated evidence speaks."""
    skill = vb_skill(tmp_path, learner_id="ovr-evidence")
    skill.set_override(MasteryLevel.BEGINNER, reason="holding", scope_node_id=NODE)
    submit_all(skill, [
        vb_evidence(NODE, 0.92, "quiz", "q", "ovr-evidence", day=0, ev_id="oe1"),
        vb_evidence(NODE, 0.93, "practice", "p", "ovr-evidence", day=1, ev_id="oe2"),
        vb_evidence(NODE, 0.94, "dialogue", "d", "ovr-evidence", day=2, ev_id="oe3"),
    ])
    # override still shapes the read...
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER
    # ...but the heuristic history advanced underneath
    assert skill.mastery_history(NODE)[-1].level == MasteryLevel.ADVANCED
    skill.clear_override(NODE)
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED


def test_evidence_after_assertion_can_promote_past_never_below(tmp_path):
    skill = vb_skill(tmp_path, learner_id="ovr-assert-ev")
    skill.assert_mastery(OTHER, MasteryLevel.INTERMEDIATE,
                         asserted_by="exam-board", reason="midterm")
    assert skill.effective_mastery(OTHER) == MasteryLevel.INTERMEDIATE
    # strong later evidence promotes past the assertion...
    submit_all(skill, [
        vb_evidence(OTHER, 0.92, "quiz", "q", "ovr-assert-ev", day=10, ev_id="oa1"),
        vb_evidence(OTHER, 0.93, "practice", "p", "ovr-assert-ev", day=11, ev_id="oa2"),
        vb_evidence(OTHER, 0.94, "dialogue", "d", "ovr-assert-ev", day=12, ev_id="oa3"),
    ])
    assert skill.effective_mastery(OTHER) == MasteryLevel.ADVANCED
    # ...while weak evidence never demotes below it (no demotion, by design)
    skill2 = vb_skill(tmp_path, learner_id="ovr-assert-ev2")
    skill2.assert_mastery(OTHER, MasteryLevel.INTERMEDIATE,
                          asserted_by="exam-board", reason="midterm")
    submit_all(skill2, [
        vb_evidence(OTHER, 0.10, "quiz", "q", "ovr-assert-ev2", day=10, ev_id="oa4"),
        vb_evidence(OTHER, 0.05, "quiz", "q", "ovr-assert-ev2", day=11, ev_id="oa5"),
    ])
    assert skill2.effective_mastery(OTHER) == MasteryLevel.INTERMEDIATE
    # the assertion record stays in history as the durable floor
    assert skill2.mastery_history(OTHER)[0].assertion is True


def test_override_does_not_leak_across_learners(tmp_path):
    from edupaal import EduPAALSkill, LearnerPreferences, SQLiteBackend, build_seed_graph

    graph = build_seed_graph()
    store = SQLiteBackend(tmp_path / "iso.db")
    prefs = LearnerPreferences(learning_style="visual", pace="steady")
    topics = ["linear-equations", "linearization"]
    a = EduPAALSkill(store, graph)
    a.cold_start(learner_id="ovr-learner-a", node_selection=topics, preferences=prefs)
    b = EduPAALSkill(store, graph)
    b.cold_start(learner_id="ovr-learner-b", node_selection=topics, preferences=prefs)
    a.set_override(MasteryLevel.ADVANCED, reason="a only", scope_node_id=NODE)
    assert b.effective_mastery(NODE) == MasteryLevel.UNKNOWN
    assert b.active_override(NODE) is None


def test_global_override_shapes_rollup_parents(tmp_path):
    """A global override is the coarsest correction: it shapes every node,
    including rollup parents, until cleared."""
    skill = vb_skill(tmp_path, learner_id="ovr-global-rollup")
    skill.set_override(MasteryLevel.INTERMEDIATE, reason="summer intensive")
    assert skill.effective_mastery("regularization") == MasteryLevel.INTERMEDIATE
    assert skill.effective_mastery("algebra") == MasteryLevel.INTERMEDIATE
    assert skill.effective_mastery(NODE) == MasteryLevel.INTERMEDIATE
    skill.clear_override()
    assert skill.effective_mastery(NODE) == MasteryLevel.UNKNOWN
