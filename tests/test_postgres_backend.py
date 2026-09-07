"""Postgres backend contract tests: same semantics as the SQLite backend.

Needs a live Postgres: set EDUPAAL_TEST_DSN (e.g.
postgresql://user:pass@host:5432/db). Skipped when no DSN is configured or
the server is unreachable, so the default suite stays dependency-free.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from edupaal import (
    EduPAALSkill,
    Evidence,
    KnowledgeNode,
    LearnerPreferences,
    MasteryLevel,
    MasteryParams,
    MasteryRecord,
    NodeLevel,
    build_seed_graph,
)
from edupaal import postgres_backend
from tests.conftest import BASE, make_evidence

_TABLES = (
    "learners",
    "preferences",
    "nodes",
    "prerequisites",
    "plans",
    "evidence",
    "mastery_records",
    "overrides",
)


def _dsn() -> str | None:
    dsn = os.environ.get("EDUPAAL_TEST_DSN")
    if not dsn or postgres_backend.psycopg is None:
        return None
    try:
        conn = postgres_backend.psycopg.connect(dsn, connect_timeout=2)
        conn.close()
    except Exception:
        return None
    return dsn


pytestmark = pytest.mark.skipif(
    _dsn() is None, reason="EDUPAAL_TEST_DSN not set or Postgres unreachable"
)


@pytest.fixture()
def pg_store():
    from edupaal import PostgresBackend

    store = PostgresBackend(_dsn())
    with store._conn.cursor() as cur:
        cur.execute("TRUNCATE " + ", ".join(_TABLES))
    yield store
    store.close()


@pytest.fixture()
def pg_skill(pg_store):
    graph = build_seed_graph()
    prefs = LearnerPreferences(learning_style="visual", pace="steady")
    s = EduPAALSkill(pg_store, graph)
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


def test_node_and_prerequisite_roundtrip(pg_store):
    graph = build_seed_graph()
    for node in graph.nodes.values():
        pg_store.save_node(node)
    pg_store.save_prerequisite("linearization", "linear-equations")
    pg_store.save_prerequisite("linearization", "linear-equations")  # idempotent

    assert pg_store.get_node("algebra").name == "Algebra"
    assert pg_store.get_prerequisites("linearization") == ["linear-equations"]
    assert len(pg_store.list_nodes(NodeLevel.TOPIC)) == 8


def test_plan_with_criteria_overrides_roundtrip(pg_store):
    from edupaal import LearningPlan

    plan = LearningPlan(
        id="plan1",
        learner_id="l1",
        topic_ids=["t1", "t2"],
        criteria_overrides={"c1": MasteryParams(k_evidence=5)},
        created_at=BASE,
    )
    pg_store.save_plan(plan)
    loaded = pg_store.get_plan_for_learner("l1")
    assert loaded.topic_ids == ["t1", "t2"]
    assert loaded.criteria_overrides["c1"].k_evidence == 5


def test_evidence_and_mastery_history_are_append_only(pg_skill, pg_store):
    pg_skill.record_evidence(make_evidence("linear-equations", 0.9, day=0))
    pg_skill.record_evidence(make_evidence("linear-equations", 0.9, day=1))

    assert len(pg_store.list_evidence("learner-1", "linear-equations")) == 2
    history = pg_store.get_mastery_history("learner-1", "linear-equations")
    assert len(history) == 1  # still BEGINNER: only the first-engagement record
    assert history[0].level == MasteryLevel.BEGINNER

    current = pg_store.get_current_mastery("learner-1", "linear-equations")
    assert current.id == history[-1].id


def test_preferences_roundtrip(pg_store):
    pg_store.save_preferences(
        "l1",
        LearnerPreferences(
            learning_style="socratic", pace="fast", extra={"grade": 5}
        ),
    )
    prefs = pg_store.get_preferences("l1")
    assert prefs.learning_style == "socratic"
    assert prefs.extra == {"grade": 5}
    assert pg_store.get_preferences("nobody") is None


def test_override_crud(pg_store):
    from edupaal import DynamicOverride

    ovr = DynamicOverride(
        learner_id="l1",
        level=MasteryLevel.INTERMEDIATE,
        scope_node_id="t1",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        reason="r",
    )
    pg_store.save_override(ovr)
    assert len(pg_store.list_overrides("l1")) == 1
    pg_store.delete_override(ovr.id)
    assert pg_store.list_overrides("l1") == []


def test_duplicate_evidence_id_fails_loud(pg_store):
    ev = make_evidence("linear-equations", 0.9)
    pg_store.save_evidence(ev)
    with pytest.raises(ValueError, match="duplicate evidence id"):
        pg_store.save_evidence(ev)
    # the failed write left no partial state: still exactly one row
    assert len(pg_store.list_evidence(ev.learner_id, ev.node_id)) == 1


def test_duplicate_mastery_record_id_fails_loud(pg_store):
    rec = MasteryRecord(
        id="mr-dup",
        node_id="t1",
        learner_id="l1",
        level=MasteryLevel.BEGINNER,
        updated_at=BASE,
        rule_version="test",
        params_in_effect={},
        evidence_ids=[],
    )
    pg_store.save_mastery_record(rec)
    with pytest.raises(ValueError, match="duplicate mastery record id"):
        pg_store.save_mastery_record(rec)
    assert len(pg_store.get_mastery_history("l1", "t1")) == 1


def test_mastery_history_tiebreak_is_insertion_order(pg_store):
    # Same updated_at: SQLite orders by (updated_at, rowid); Postgres must
    # order by (updated_at, seq) so insertion order wins ties identically.
    for i, rid in enumerate(("mr-a", "mr-b", "mr-c")):
        pg_store.save_mastery_record(
            MasteryRecord(
                id=rid,
                node_id="t1",
                learner_id="l1",
                level=MasteryLevel.BEGINNER,
                updated_at=BASE,
                rule_version="test",
                params_in_effect={},
                evidence_ids=[],
            )
        )
    history = pg_store.get_mastery_history("l1", "t1")
    assert [r.id for r in history] == ["mr-a", "mr-b", "mr-c"]
    assert pg_store.get_current_mastery("l1", "t1").id == "mr-c"


def test_naive_datetimes_treated_as_utc(pg_store):
    naive = datetime(2026, 9, 1, 12, 0, 0)  # no tzinfo
    ev = make_evidence("linear-equations", 0.9)
    ev.occurred_at = naive
    pg_store.save_evidence(ev)
    loaded = pg_store.get_evidence(ev.id)
    assert loaded.occurred_at == naive.replace(tzinfo=timezone.utc)
    assert loaded.occurred_at.tzinfo is not None


def test_database_url_env_fallback(monkeypatch):
    from edupaal import PostgresBackend

    monkeypatch.setenv("DATABASE_URL", _dsn())
    store = PostgresBackend()  # no dsn arg: must honor DATABASE_URL
    assert store.get_learner("nobody") is None
    store.close()


def test_missing_dsn_raises():
    from edupaal import PostgresBackend

    old = os.environ.pop("DATABASE_URL", None)
    try:
        with pytest.raises(ValueError, match="DATABASE_URL"):
            PostgresBackend()
    finally:
        if old is not None:
            os.environ["DATABASE_URL"] = old
