"""Postgres storage backend for EduPAAL.

``PostgresBackend`` implements the same :class:`StorageBackend` protocol as
``SQLiteBackend`` on top of PostgreSQL (for example Neon). SQLite remains the
default backend; Postgres is opt-in for deployments that need a hosted,
multi-writer database (e.g. a dashboard reading one shared learner memory).

Requires ``psycopg`` v3 (``pip install "psycopg[binary]"``). Connects via an
explicit ``dsn`` or, when omitted, the ``DATABASE_URL`` environment variable.
Fail-loud: a missing DSN raises ``ValueError``; duplicate evidence/mastery
ids raise ``ValueError`` (never silently overwrite history).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .entities import (
    DynamicOverride,
    Evidence,
    KnowledgeNode,
    Learner,
    LearnerPreferences,
    LearningPlan,
    MasteryLevel,
    MasteryParams,
    MasteryRecord,
    NodeLevel,
)

try:
    import psycopg
    from psycopg import errors as _pg_errors
    from psycopg.rows import dict_row as _dict_row

    _PSYCOPG_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _e:  # pragma: no cover - import-time guard only
    psycopg = None  # type: ignore[assignment]
    _pg_errors = None  # type: ignore[assignment]
    _dict_row = None  # type: ignore[assignment]
    _PSYCOPG_IMPORT_ERROR = _e


def _ensure_aware(dt: datetime) -> datetime:
    """Naive datetimes are treated as UTC, matching SQLiteBackend's
    ``_dt_to_str`` normalization."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


_DDL = """
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS preferences (
    learner_id TEXT PRIMARY KEY,
    learning_style TEXT NOT NULL,
    pace TEXT NOT NULL,
    extra TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS prerequisites (
    topic_id TEXT NOT NULL,
    prereq_id TEXT NOT NULL,
    PRIMARY KEY (topic_id, prereq_id)
);
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    topic_ids TEXT NOT NULL,
    criteria_overrides TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    performance DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    attempts INTEGER,
    hints_used INTEGER,
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_evidence_learner_node
    ON evidence (learner_id, node_id);
CREATE TABLE IF NOT EXISTS mastery_records (
    -- seq is the insertion-order tiebreak, the Postgres analog of SQLite's
    -- rowid used in get_mastery_history's ORDER BY. Internal only: it is
    -- never exposed through the StorageBackend protocol.
    seq BIGSERIAL,
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    level TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    rule_version TEXT NOT NULL,
    params_in_effect TEXT NOT NULL,
    evidence_ids TEXT NOT NULL,
    assertion BOOLEAN NOT NULL DEFAULT FALSE,
    asserted_by TEXT,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_mastery_learner_node
    ON mastery_records (learner_id, node_id);
CREATE TABLE IF NOT EXISTS overrides (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    level TEXT NOT NULL,
    scope_node_id TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    promoted BOOLEAN NOT NULL DEFAULT FALSE
);
"""


class PostgresBackend:
    """StorageBackend on PostgreSQL. One connection, autocommit (each method
    commits on its own, mirroring SQLiteBackend's per-method commits)."""

    def __init__(self, dsn: Optional[str] = None) -> None:
        if psycopg is None:
            raise ImportError(
                'PostgresBackend requires psycopg v3: pip install "psycopg[binary]"'
            ) from _PSYCOPG_IMPORT_ERROR
        dsn = dsn or os.environ.get("DATABASE_URL")
        if not dsn:
            raise ValueError(
                "PostgresBackend needs a DSN: pass dsn=... or set DATABASE_URL"
            )
        self.dsn = dsn
        self._conn = psycopg.connect(dsn, row_factory=_dict_row, autocommit=True)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(_DDL)

    # -- learners & preferences --

    def save_learner(self, learner: Learner) -> None:
        self._conn.execute(
            "INSERT INTO learners (id, name) VALUES (%s, %s)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            (learner.id, learner.name),
        )

    def get_learner(self, learner_id: str) -> Optional[Learner]:
        row = self._conn.execute(
            "SELECT * FROM learners WHERE id = %s", (learner_id,)
        ).fetchone()
        return Learner(id=row["id"], name=row["name"]) if row else None

    def save_preferences(self, learner_id: str, prefs: LearnerPreferences) -> None:
        self._conn.execute(
            "INSERT INTO preferences (learner_id, learning_style, pace, extra)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (learner_id) DO UPDATE SET"
            " learning_style = EXCLUDED.learning_style,"
            " pace = EXCLUDED.pace,"
            " extra = EXCLUDED.extra",
            (learner_id, prefs.learning_style, prefs.pace, json.dumps(prefs.extra)),
        )

    def get_preferences(self, learner_id: str) -> Optional[LearnerPreferences]:
        row = self._conn.execute(
            "SELECT * FROM preferences WHERE learner_id = %s", (learner_id,)
        ).fetchone()
        if not row:
            return None
        return LearnerPreferences(
            learning_style=row["learning_style"],
            pace=row["pace"],
            extra=json.loads(row["extra"]),
        )

    # -- knowledge graph --

    def save_node(self, node: KnowledgeNode) -> None:
        self._conn.execute(
            "INSERT INTO nodes (id, level, name, parent_id, metadata)"
            " VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET"
            " level = EXCLUDED.level,"
            " name = EXCLUDED.name,"
            " parent_id = EXCLUDED.parent_id,"
            " metadata = EXCLUDED.metadata",
            (
                node.id,
                node.level.value,
                node.name,
                node.parent_id,
                json.dumps(node.metadata),
            ),
        )

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = %s", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self, level: Optional[NodeLevel] = None) -> List[KnowledgeNode]:
        if level is None:
            rows = self._conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE level = %s ORDER BY id", (level.value,)
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    @staticmethod
    def _row_to_node(row: Dict[str, Any]) -> KnowledgeNode:
        return KnowledgeNode(
            id=row["id"],
            level=NodeLevel(row["level"]),
            name=row["name"],
            parent_id=row["parent_id"],
            metadata=json.loads(row["metadata"]),
        )

    def save_prerequisite(self, topic_id: str, prereq_id: str) -> None:
        self._conn.execute(
            "INSERT INTO prerequisites (topic_id, prereq_id) VALUES (%s, %s)"
            " ON CONFLICT DO NOTHING",
            (topic_id, prereq_id),
        )

    def get_prerequisites(self, topic_id: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT prereq_id FROM prerequisites WHERE topic_id = %s"
            " ORDER BY prereq_id",
            (topic_id,),
        ).fetchall()
        return [r["prereq_id"] for r in rows]

    # -- plans --

    def save_plan(self, plan: LearningPlan) -> None:
        self._conn.execute(
            "INSERT INTO plans"
            " (id, learner_id, topic_ids, criteria_overrides, created_at, version)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET"
            " learner_id = EXCLUDED.learner_id,"
            " topic_ids = EXCLUDED.topic_ids,"
            " criteria_overrides = EXCLUDED.criteria_overrides,"
            " created_at = EXCLUDED.created_at,"
            " version = EXCLUDED.version",
            (
                plan.id,
                plan.learner_id,
                json.dumps(plan.topic_ids),
                json.dumps(
                    {k: v.as_dict() for k, v in plan.criteria_overrides.items()}
                ),
                _ensure_aware(plan.created_at),
                plan.version,
            ),
        )

    def get_plan(self, plan_id: str) -> Optional[LearningPlan]:
        row = self._conn.execute(
            "SELECT * FROM plans WHERE id = %s", (plan_id,)
        ).fetchone()
        return self._row_to_plan(row) if row else None

    def get_plan_for_learner(self, learner_id: str) -> Optional[LearningPlan]:
        row = self._conn.execute(
            "SELECT * FROM plans WHERE learner_id = %s ORDER BY version DESC LIMIT 1",
            (learner_id,),
        ).fetchone()
        return self._row_to_plan(row) if row else None

    @staticmethod
    def _row_to_plan(row: Dict[str, Any]) -> LearningPlan:
        return LearningPlan(
            id=row["id"],
            learner_id=row["learner_id"],
            topic_ids=json.loads(row["topic_ids"]),
            criteria_overrides={
                k: MasteryParams.from_dict(v)
                for k, v in json.loads(row["criteria_overrides"]).items()
            },
            created_at=_ensure_aware(row["created_at"]),
            version=row["version"],
        )

    # -- evidence --

    def save_evidence(self, evidence: Evidence) -> None:
        # Plain INSERT, not an upsert: evidence is append-only, so a duplicate
        # id is a caller bug and must fail loudly instead of silently
        # overwriting history. Surfaced as ValueError (not the backend's
        # native integrity error) so the contract is backend-agnostic.
        try:
            self._conn.execute(
                "INSERT INTO evidence"
                " (id, learner_id, node_id, source_agent, activity_type, occurred_at,"
                "  performance, confidence, attempts, hints_used, details)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    evidence.id,
                    evidence.learner_id,
                    evidence.node_id,
                    evidence.source_agent,
                    evidence.activity_type,
                    _ensure_aware(evidence.occurred_at),
                    evidence.performance,
                    evidence.confidence,
                    evidence.attempts,
                    evidence.hints_used,
                    json.dumps(evidence.details),
                ),
            )
        except _pg_errors.UniqueViolation:
            raise ValueError(f"duplicate evidence id: {evidence.id}") from None

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        row = self._conn.execute(
            "SELECT * FROM evidence WHERE id = %s", (evidence_id,)
        ).fetchone()
        return self._row_to_evidence(row) if row else None

    def list_evidence(
        self, learner_id: str, node_id: Optional[str] = None
    ) -> List[Evidence]:
        if node_id is None:
            rows = self._conn.execute(
                "SELECT * FROM evidence WHERE learner_id = %s ORDER BY occurred_at",
                (learner_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM evidence WHERE learner_id = %s AND node_id = %s"
                " ORDER BY occurred_at",
                (learner_id, node_id),
            ).fetchall()
        return [self._row_to_evidence(r) for r in rows]

    @staticmethod
    def _row_to_evidence(row: Dict[str, Any]) -> Evidence:
        return Evidence(
            id=row["id"],
            learner_id=row["learner_id"],
            node_id=row["node_id"],
            source_agent=row["source_agent"],
            activity_type=row["activity_type"],
            occurred_at=_ensure_aware(row["occurred_at"]),
            performance=row["performance"],
            confidence=row["confidence"],
            attempts=row["attempts"],
            hints_used=row["hints_used"],
            details=json.loads(row["details"]),
        )

    # -- mastery records --

    def save_mastery_record(self, record: MasteryRecord) -> None:
        # Plain INSERT, not an upsert: mastery history is append-only and must
        # never be rewritten; a duplicate id fails loudly. Backend-agnostic
        # ValueError, matching the StorageBackend contract.
        try:
            self._conn.execute(
                "INSERT INTO mastery_records"
                " (id, node_id, learner_id, level, updated_at, rule_version,"
                "  params_in_effect, evidence_ids, assertion, asserted_by, reason)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    record.id,
                    record.node_id,
                    record.learner_id,
                    record.level.value,
                    _ensure_aware(record.updated_at),
                    record.rule_version,
                    json.dumps(record.params_in_effect),
                    json.dumps(record.evidence_ids),
                    bool(record.assertion),
                    record.asserted_by,
                    record.reason,
                ),
            )
        except _pg_errors.UniqueViolation:
            raise ValueError(
                f"duplicate mastery record id: {record.id}"
            ) from None

    def get_mastery_history(
        self, learner_id: str, node_id: str
    ) -> List[MasteryRecord]:
        rows = self._conn.execute(
            "SELECT * FROM mastery_records WHERE learner_id = %s AND node_id = %s"
            " ORDER BY updated_at, seq",
            (learner_id, node_id),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_current_mastery(
        self, learner_id: str, node_id: str
    ) -> Optional[MasteryRecord]:
        history = self.get_mastery_history(learner_id, node_id)
        return history[-1] if history else None

    @staticmethod
    def _row_to_record(row: Dict[str, Any]) -> MasteryRecord:
        return MasteryRecord(
            id=row["id"],
            node_id=row["node_id"],
            learner_id=row["learner_id"],
            level=MasteryLevel(row["level"]),
            updated_at=_ensure_aware(row["updated_at"]),
            rule_version=row["rule_version"],
            params_in_effect=json.loads(row["params_in_effect"]),
            evidence_ids=json.loads(row["evidence_ids"]),
            assertion=bool(row["assertion"]),
            asserted_by=row["asserted_by"],
            reason=row["reason"],
        )

    # -- overrides --

    def save_override(self, override: DynamicOverride) -> None:
        self._conn.execute(
            "INSERT INTO overrides"
            " (id, learner_id, level, scope_node_id, expires_at, created_at,"
            " reason, promoted)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET"
            " learner_id = EXCLUDED.learner_id,"
            " level = EXCLUDED.level,"
            " scope_node_id = EXCLUDED.scope_node_id,"
            " expires_at = EXCLUDED.expires_at,"
            " created_at = EXCLUDED.created_at,"
            " reason = EXCLUDED.reason,"
            " promoted = EXCLUDED.promoted",
            (
                override.id,
                override.learner_id,
                override.level.value,
                override.scope_node_id,
                _ensure_aware(override.expires_at),
                _ensure_aware(override.created_at),
                override.reason,
                bool(override.promoted),
            ),
        )

    def list_overrides(self, learner_id: str) -> List[DynamicOverride]:
        rows = self._conn.execute(
            "SELECT * FROM overrides WHERE learner_id = %s ORDER BY expires_at",
            (learner_id,),
        ).fetchall()
        return [
            DynamicOverride(
                id=r["id"],
                learner_id=r["learner_id"],
                level=MasteryLevel(r["level"]),
                scope_node_id=r["scope_node_id"],
                expires_at=_ensure_aware(r["expires_at"]),
                created_at=_ensure_aware(r["created_at"]),
                reason=r["reason"],
                promoted=bool(r["promoted"]),
            )
            for r in rows
        ]

    def delete_override(self, override_id: str) -> None:
        self._conn.execute("DELETE FROM overrides WHERE id = %s", (override_id,))

    def close(self) -> None:
        self._conn.close()
