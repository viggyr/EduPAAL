"""Evidence-driven mastery transitions under heuristic-v1."""

import pytest

from edupaal import HEURISTIC_VERSION, MasteryLevel
from tests.conftest import make_evidence, promote_to


def test_first_evidence_moves_unknown_to_beginner(skill):
    records = skill.record_evidence(make_evidence("linear-equations", 0.9))
    assert len(records) == 1
    rec = records[0]
    assert rec.level == MasteryLevel.BEGINNER
    assert rec.rule_version == HEURISTIC_VERSION
    assert rec.evidence_ids is not None and len(rec.evidence_ids) == 1
    assert rec.params_in_effect["k_evidence"] == 3
    assert skill.effective_mastery("linear-equations") == MasteryLevel.BEGINNER


def test_strong_multimodal_evidence_chains_to_advanced(skill):
    skill.record_evidence(make_evidence("linear-equations", 0.90, "quiz", "quiz-agent", day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.85, "practice", "practice-agent", day=1))
    records = skill.record_evidence(
        make_evidence("linear-equations", 0.92, "visualization", "viz-agent", day=2)
    )
    # third evidence triggers BEGINNER -> INTERMEDIATE -> ADVANCED
    assert [r.level for r in records] == [
        MasteryLevel.INTERMEDIATE,
        MasteryLevel.ADVANCED,
    ]
    assert all(r.rule_version == HEURISTIC_VERSION for r in records)
    # the ADVANCED step saw all three modalities
    adv = records[-1]
    assert len(adv.evidence_ids) == 3

    # history is append-only: every step retained
    history = skill.mastery_history("linear-equations")
    assert [r.level for r in history] == [
        MasteryLevel.BEGINNER,
        MasteryLevel.INTERMEDIATE,
        MasteryLevel.ADVANCED,
    ]


def test_weak_evidence_does_not_promote(skill):
    skill.record_evidence(make_evidence("linear-equations", 0.50, day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.55, day=1))
    records = skill.record_evidence(make_evidence("linear-equations", 0.50, day=2))
    assert records == []
    assert skill.effective_mastery("linear-equations") == MasteryLevel.BEGINNER


def test_contradictory_evidence_blocks_promotion(skill):
    skill.record_evidence(make_evidence("linear-equations", 0.90, day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.85, day=1))
    records = skill.record_evidence(make_evidence("linear-equations", 0.20, day=2))
    assert records == []
    assert skill.effective_mastery("linear-equations") == MasteryLevel.BEGINNER


def test_stale_evidence_outside_window_does_not_count(skill):
    # K=3, W=30d: two strong evidences from 40+ days ago fall outside the
    # window anchored at the latest evidence, so promotion is blocked even
    # though 4 strong evidences exist in total
    skill.record_evidence(make_evidence("linear-equations", 0.95, day=0))
    skill.record_evidence(make_evidence("linear-equations", 0.95, day=1))
    skill.record_evidence(make_evidence("linear-equations", 0.95, day=40))
    records = skill.record_evidence(make_evidence("linear-equations", 0.95, day=41))
    assert records == []
    assert skill.effective_mastery("linear-equations") == MasteryLevel.BEGINNER


def test_evidence_requires_learner_and_node():
    from edupaal import Evidence
    from datetime import datetime, timezone

    for kwargs in (
        {"learner_id": "", "node_id": "t"},
        {"learner_id": "l", "node_id": ""},
    ):
        with pytest.raises(ValueError):
            Evidence(
                id="x", activity_type="quiz", source_agent="a",
                occurred_at=datetime.now(timezone.utc), performance=0.5,
                **kwargs,
            )


def test_evidence_must_target_topics(skill):
    with pytest.raises(ValueError):
        skill.record_evidence(make_evidence("algebra", 0.9))


def test_evidence_requires_source_and_activity():
    from edupaal import Evidence
    from datetime import datetime, timezone

    with pytest.raises(ValueError):
        Evidence(
            id="x", learner_id="l", node_id="t", source_agent="",
            activity_type="quiz", occurred_at=datetime.now(timezone.utc),
            performance=0.5,
        )
    with pytest.raises(ValueError):
        Evidence(
            id="x", learner_id="l", node_id="t", source_agent="a",
            activity_type="", occurred_at=datetime.now(timezone.utc),
            performance=0.5,
        )
    with pytest.raises(ValueError):
        Evidence(
            id="x", learner_id="l", node_id="t", source_agent="a",
            activity_type="quiz", occurred_at=datetime.now(timezone.utc),
            performance=1.5,
        )


def test_evidence_learner_mismatch_rejected(skill):
    ev = make_evidence("linear-equations", 0.9, learner_id="someone-else")
    with pytest.raises(ValueError):
        skill.record_evidence(ev)
