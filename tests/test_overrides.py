"""Overrides and assertions: provenance, expiry, promotion."""

from datetime import timedelta

import pytest

from edupaal import ASSERTION_VERSION, MasteryLevel
from edupaal.entities import _utcnow
from tests.conftest import make_evidence, promote_to


def test_override_shapes_effective_mastery_temporarily(skill):
    promote_to(skill, "linear-equations", "beginner")
    skill.set_override(
        MasteryLevel.ADVANCED,
        reason="teacher judgement",
        scope_node_id="linear-equations",
        expires_at=_utcnow() + timedelta(days=1),
    )
    assert skill.effective_mastery("linear-equations") == MasteryLevel.ADVANCED
    # the underlying record is untouched
    assert skill.mastery_history("linear-equations")[-1].level == MasteryLevel.BEGINNER


def test_expired_override_falls_back(skill):
    promote_to(skill, "linear-equations", "intermediate")
    skill.set_override(
        MasteryLevel.ADVANCED,
        reason="stale",
        scope_node_id="linear-equations",
        expires_at=_utcnow() - timedelta(seconds=1),
    )
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE


def test_global_override_applies_when_no_scoped_one(skill):
    promote_to(skill, "linear-equations", "beginner")
    skill.set_override(
        MasteryLevel.INTERMEDIATE,
        reason="placement test",
        expires_at=_utcnow() + timedelta(days=1),
    )
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE
    assert skill.effective_mastery("linearization") == MasteryLevel.INTERMEDIATE


def test_clear_override(skill):
    skill.set_override(
        MasteryLevel.ADVANCED, reason="x", scope_node_id="linear-equations"
    )
    assert skill.clear_override("linear-equations") == 1
    assert skill.effective_mastery("linear-equations") == MasteryLevel.UNKNOWN


def test_promote_override_writes_durable_assertion(skill):
    promote_to(skill, "linear-equations", "beginner")
    skill.set_override(
        MasteryLevel.INTERMEDIATE,
        reason="teacher confirms",
        scope_node_id="linear-equations",
    )
    record = skill.promote_override("linear-equations")
    assert record.assertion is True
    assert record.level == MasteryLevel.INTERMEDIATE
    assert record.asserted_by == "override-promotion"
    # override consumed; the assertion now stands on its own
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE
    with pytest.raises(ValueError):
        skill.promote_override("linear-equations")


def test_assert_mastery_appends_with_provenance(skill):
    promote_to(skill, "linear-equations", "beginner")
    record = skill.assert_mastery(
        "linear-equations",
        MasteryLevel.ADVANCED,
        asserted_by="assessment-agent",
        reason="validated exam, 95%",
    )
    assert record.assertion is True
    assert record.rule_version == ASSERTION_VERSION
    assert record.asserted_by == "assessment-agent"
    assert record.reason == "validated exam, 95%"
    assert record.evidence_ids == []
    # history is not rewritten: the beginner record is still there
    levels = [r.level for r in skill.mastery_history("linear-equations")]
    assert levels == [MasteryLevel.BEGINNER, MasteryLevel.ADVANCED]
    # later evidence continues from the asserted level
    skill.record_evidence(make_evidence("linear-equations", 0.9, day=10))
    assert skill.effective_mastery("linear-equations") == MasteryLevel.ADVANCED


def test_assert_unknown_rejected(skill):
    with pytest.raises(ValueError):
        skill.assert_mastery(
            "linear-equations", MasteryLevel.UNKNOWN,
            asserted_by="x", reason="y",
        )


def test_promote_override_on_decomposed_topic_fails_clearly(skill):
    skill.set_override(
        MasteryLevel.INTERMEDIATE,
        reason="teacher confirms",
        scope_node_id="dropout",  # decomposed: cannot carry a leaf assertion
    )
    with pytest.raises(ValueError, match="only leaf TOPIC nodes"):
        skill.promote_override("dropout")
