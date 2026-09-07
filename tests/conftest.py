"""Shared fixtures and helpers for the EduPAAL test suite."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pytest

from edupaal import (
    EduPAALSkill,
    Evidence,
    KnowledgeGraph,
    LearnerPreferences,
    MasteryParams,
    SQLiteBackend,
    build_seed_graph,
)

BASE = datetime(2026, 9, 1, tzinfo=timezone.utc)
_counter = itertools.count(1)


@pytest.fixture()
def graph() -> KnowledgeGraph:
    return build_seed_graph()


@pytest.fixture()
def store(tmp_path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "test.db")


@pytest.fixture()
def prefs() -> LearnerPreferences:
    return LearnerPreferences(learning_style="visual", pace="steady")


@pytest.fixture()
def skill(store, graph, prefs) -> EduPAALSkill:
    s = EduPAALSkill(store, graph)
    s.cold_start(
        learner_id="learner-1",
        node_selection=[
            "linear-equations",
            "linearization",
            "bias-variance",
            "regularization",
        ],
        preferences=prefs,
    )
    return s


def make_evidence(
    node_id: str,
    performance: float,
    activity_type: str = "quiz",
    source_agent: str = "quiz-agent",
    learner_id: str = "learner-1",
    day: int = 0,
) -> Evidence:
    n = next(_counter)
    return Evidence(
        id=f"ev{n:04d}",
        learner_id=learner_id,
        node_id=node_id,
        source_agent=source_agent,
        activity_type=activity_type,
        occurred_at=BASE + timedelta(days=day),
        performance=performance,
    )


def promote_to(skill: EduPAALSkill, node_id: str, level: str, day: int = 0,
               learner_id: str = "learner-1"):
    """Drive a topic to a target level with strong evidence.

    intermediate uses single-modality evidence so the default cross-modal
    gate holds it below ADVANCED; advanced uses three modalities.
    """
    from edupaal import MasteryLevel

    target = MasteryLevel(level)
    if target == MasteryLevel("beginner"):
        evidences = [("quiz", "quiz-agent", 0.90)]
    elif target == MasteryLevel("intermediate"):
        evidences = [
            ("quiz", "quiz-agent", 0.90),
            ("quiz", "quiz-agent", 0.88),
            ("quiz", "quiz-agent", 0.92),
        ]
    else:
        evidences = [
            ("quiz", "quiz-agent", 0.90),
            ("practice", "practice-agent", 0.85),
            ("visualization", "viz-agent", 0.92),
        ]
    for i, (activity, source, perf) in enumerate(evidences):
        skill.record_evidence(
            make_evidence(node_id, perf, activity, source,
                           learner_id=learner_id, day=day + i)
        )
    assert skill.effective_mastery(node_id) == target, (
        f"expected {target}, got {skill.effective_mastery(node_id)}"
    )
