"""PROVENANCE: every transition must be replayable from its record alone.

The hypothesis demands explainability: for each mastery transition, the
record must carry the rule version, the exact parameters in effect, and the
evidence ids that caused it — and an independent re-implementation of the
documented rule, fed only with the record's contents, must reproduce the
recorded level.
"""

from edupaal import MasteryLevel, MasteryParams

from .vb_helpers import submit_all, vb_evidence, vb_skill

NODE = "linear-equations"
RULE_VERSIONS = {"heuristic-v1", "assertion-v1"}


def _drive(skill, learner_id):
    """A fixed evidence script ending at ADVANCED via two promotions."""
    evidences = [
        vb_evidence(NODE, 0.90, "quiz", "quiz-agent", learner_id, day=0, ev_id=f"{learner_id}-p1"),
        vb_evidence(NODE, 0.85, "quiz", "quiz-agent", learner_id, day=1, ev_id=f"{learner_id}-p2"),
        vb_evidence(NODE, 0.92, "quiz", "quiz-agent", learner_id, day=2, ev_id=f"{learner_id}-p3"),
        vb_evidence(NODE, 0.93, "practice", "practice-agent", learner_id, day=3, ev_id=f"{learner_id}-p4"),
        vb_evidence(NODE, 0.94, "dialogue", "tutor-agent", learner_id, day=4, ev_id=f"{learner_id}-p5"),
    ]
    submit_all(skill, evidences)
    return evidences


def test_every_record_carries_complete_provenance(tmp_path):
    skill = vb_skill(tmp_path, learner_id="prov-full")
    evidences = _drive(skill, "prov-full")
    # plus a privileged assertion on another leaf
    skill.assert_mastery("linearization", MasteryLevel.INTERMEDIATE,
                         asserted_by="exam-board", reason="final exam 87%")

    stored_ids = {e.id for e in evidences}
    for node in (NODE, "linearization"):
        history = skill.mastery_history(node)
        assert history, f"no history for {node}"
        for rec in history:
            assert rec.rule_version in RULE_VERSIONS, rec.rule_version
            # params round-trip through the stored dict
            params = MasteryParams.from_dict(rec.params_in_effect)
            assert params.as_dict() == rec.params_in_effect
            # every cited evidence id resolves to a stored evidence record
            for eid in rec.evidence_ids:
                stored = skill.store.get_evidence(eid)
                assert stored is not None, f"record {rec.id} cites missing evidence {eid}"
                assert stored.node_id == rec.node_id
                assert stored.learner_id == rec.learner_id
            if rec.assertion:
                assert rec.rule_version == "assertion-v1"
                assert rec.asserted_by, "assertion without asserted_by"
                assert rec.reason, "assertion without reason"
            else:
                assert rec.rule_version == "heuristic-v1"
                assert rec.evidence_ids, "heuristic transition with no evidence"
                assert set(rec.evidence_ids) <= stored_ids | {
                    e.id for e in skill.store.list_evidence("prov-full", "linearization")
                }


def test_transition_replayable_from_record_alone(tmp_path):
    """Replay: for each heuristic record, take (params, cited evidence,
    previous level) and recompute the documented rule by hand. The replay
    must yield exactly the recorded level."""
    skill = vb_skill(tmp_path, learner_id="prov-replay")
    evidences = _drive(skill, "prov-replay")
    by_id = {e.id: e for e in evidences}
    history = skill.mastery_history(NODE)
    assert [r.level for r in history] == [
        MasteryLevel.BEGINNER, MasteryLevel.INTERMEDIATE, MasteryLevel.ADVANCED,
    ]
    prev = MasteryLevel.UNKNOWN
    for rec in history:
        params = MasteryParams.from_dict(rec.params_in_effect)
        cited = [by_id[eid] for eid in rec.evidence_ids]
        # hand-rolled single-step check of heuristic-v1 for this transition
        if prev == MasteryLevel.UNKNOWN:
            expected = MasteryLevel.BEGINNER
            assert len(cited) == 1
        else:
            assert len(cited) == params.k_evidence
            mean = sum(e.performance for e in cited) / len(cited)
            bar = params.t_intermediate if prev == MasteryLevel.BEGINNER else params.t_advanced
            assert mean >= bar, f"record {rec.id}: mean {mean} below bar {bar}"
            assert all(e.performance >= params.t_contradict for e in cited)
            if rec.level == MasteryLevel.ADVANCED and params.cross_modal_advanced:
                assert len({e.activity_type for e in cited}) >= 2
            expected = (
                MasteryLevel.INTERMEDIATE if prev == MasteryLevel.BEGINNER
                else MasteryLevel.ADVANCED
            )
        assert rec.level == expected, f"record {rec.id} not replayable"
        prev = rec.level


def test_history_append_only_and_current_is_latest(tmp_path):
    skill = vb_skill(tmp_path, learner_id="prov-append")
    _drive(skill, "prov-append")
    history = skill.mastery_history(NODE)
    # updated_at is non-decreasing in history order (ties broken by rowid)
    stamps = [r.updated_at for r in history]
    assert stamps == sorted(stamps)
    current = skill.store.get_current_mastery("prov-append", NODE)
    assert current.id == history[-1].id
    assert current.level == MasteryLevel.ADVANCED
    # the full evidence trail is intact: every record's evidence still stored
    for rec in history:
        for eid in rec.evidence_ids:
            assert skill.store.get_evidence(eid) is not None


def test_params_in_effect_reflect_criteria_overrides(tmp_path):
    """Per-node criteria overrides must show up in the records they caused."""
    from edupaal import LearnerPreferences, MasteryParams as MP

    strict = MP(k_evidence=2, t_intermediate=0.9, t_advanced=0.95)
    skill = vb_skill(
        tmp_path,
        learner_id="prov-override",
        topics=["linear-equations", "linearization"],
        params=MasteryParams(),
    )
    # re-cold-start with a per-node override (bumps plan version)
    skill.cold_start(
        learner_id="prov-override",
        node_selection=["linear-equations", "linearization"],
        preferences=LearnerPreferences(learning_style="visual", pace="steady"),
        criteria_overrides={"linear-equations": strict},
    )
    submit_all(skill, [
        vb_evidence("linear-equations", 0.92, "quiz", "q", "prov-override", day=0, ev_id="po1"),
        vb_evidence("linear-equations", 0.93, "quiz", "q", "prov-override", day=1, ev_id="po2"),
    ])
    history = skill.mastery_history("linear-equations")
    assert history[-1].level == MasteryLevel.INTERMEDIATE  # 0.925 >= 0.9 with k=2
    assert history[-1].params_in_effect["k_evidence"] == 2
    assert history[-1].params_in_effect["t_intermediate"] == 0.9
    # the sibling node still uses defaults
    submit_all(skill, [
        vb_evidence("linearization", 0.70, "quiz", "q", "prov-override", day=0, ev_id="po3"),
        vb_evidence("linearization", 0.71, "quiz", "q", "prov-override", day=1, ev_id="po4"),
        vb_evidence("linearization", 0.72, "quiz", "q", "prov-override", day=2, ev_id="po5"),
    ])
    sib = skill.mastery_history("linearization")[-1]
    assert sib.params_in_effect["k_evidence"] == 3  # default
    assert sib.level == MasteryLevel.INTERMEDIATE


def test_provenance_survives_recold_start(tmp_path):
    """Re-planning must not disturb existing mastery history or provenance."""
    skill = vb_skill(tmp_path, learner_id="prov-replan")
    _drive(skill, "prov-replan")
    before = [(r.id, r.level, r.rule_version, tuple(r.evidence_ids)) for r in skill.mastery_history(NODE)]
    from edupaal import LearnerPreferences

    plan2 = skill.cold_start(
        learner_id="prov-replan",
        node_selection=["linear-equations", "dropout-rate"],
        preferences=LearnerPreferences(learning_style="auditory", pace="fast"),
    )
    assert plan2.version == 2
    after = [(r.id, r.level, r.rule_version, tuple(r.evidence_ids)) for r in skill.mastery_history(NODE)]
    assert before == after
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
