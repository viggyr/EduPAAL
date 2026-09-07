"""Pluggable storage for EduPAAL.

``StorageBackend`` is the persistence contract: nodes, prerequisite edges,
plans, evidence, mastery records (append-only history), preferences, and
overrides. ``SQLiteBackend`` is the default implementation, backed by the
standard library only.

A Mem0 adapter is the intended next backend: it would implement this same
protocol with Mem0's memory store underneath (memories keyed by
learner/node, semantic retrieval for evidence lookup). It is documented,
not implemented, in this draft.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

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
    _new_id,
)


def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class StorageBackend(Protocol):
    """Persistence contract. Implementations must keep mastery records and
    evidence append-only so history is never rewritten.

    Duplicate ids are caller bugs: ``save_evidence`` and
    ``save_mastery_record`` must raise ``ValueError`` on a duplicate id
    (never silently replace the existing row).
    """

    # -- learners & preferences --
    def save_learner(self, learner: Learner) -> None: ...
    def get_learner(self, learner_id: str) -> Optional[Learner]: ...
    def save_preferences(self, learner_id: str, prefs: LearnerPreferences) -> None: ...
    def get_preferences(self, learner_id: str) -> Optional[LearnerPreferences]: ...

    # -- knowledge graph --
    def save_node(self, node: KnowledgeNode) -> None: ...
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]: ...
    def list_nodes(self, level: Optional[NodeLevel] = None) -> List[KnowledgeNode]: ...
    def save_prerequisite(self, topic_id: str, prereq_id: str) -> None: ...
    def get_prerequisites(self, topic_id: str) -> List[str]: ...

    # -- plans --
    def save_plan(self, plan: LearningPlan) -> None: ...
    def get_plan(self, plan_id: str) -> Optional[LearningPlan]: ...
    def get_plan_for_learner(self, learner_id: str) -> Optional[LearningPlan]: ...

    # -- evidence (append-only) --
    def save_evidence(self, evidence: Evidence) -> None: ...
    def get_evidence(self, evidence_id: str) -> Optional[Evidence]: ...
    def list_evidence(
        self, learner_id: str, node_id: Optional[str] = None
    ) -> List[Evidence]: ...

    # -- mastery records (append-only history) --
    def save_mastery_record(self, record: MasteryRecord) -> None: ...
    def get_mastery_history(
        self, learner_id: str, node_id: str
    ) -> List[MasteryRecord]: ...
    def get_current_mastery(
        self, learner_id: str, node_id: str
    ) -> Optional[MasteryRecord]: ...

    # -- overrides --
    def save_override(self, override: DynamicOverride) -> None: ...
    def list_overrides(self, learner_id: str) -> List[DynamicOverride]: ...
    def delete_override(self, override_id: str) -> None: ...


class SQLiteBackend:
    """Default backend. One file, stdlib only, versioned records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
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
                created_at TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                learner_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                performance REAL NOT NULL,
                confidence REAL,
                attempts INTEGER,
                hints_used INTEGER,
                details TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_learner_node
                ON evidence (learner_id, node_id);
            CREATE TABLE IF NOT EXISTS mastery_records (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                learner_id TEXT NOT NULL,
                level TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                rule_version TEXT NOT NULL,
                params_in_effect TEXT NOT NULL,
                evidence_ids TEXT NOT NULL,
                assertion INTEGER NOT NULL DEFAULT 0,
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
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                promoted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # Lightweight migration for databases created before created_at
        # existed (draft-era only; no production data): pre-migration rows
        # get an empty created_at and sort as oldest.
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(overrides)").fetchall()
        }
        if "created_at" not in cols:
            self._conn.execute(
                "ALTER TABLE overrides ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
        self._conn.commit()

    # -- learners & preferences --

    def save_learner(self, learner: Learner) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO learners (id, name) VALUES (?, ?)",
            (learner.id, learner.name),
        )
        self._conn.commit()

    def get_learner(self, learner_id: str) -> Optional[Learner]:
        row = self._conn.execute(
            "SELECT * FROM learners WHERE id = ?", (learner_id,)
        ).fetchone()
        return Learner(id=row["id"], name=row["name"]) if row else None

    def save_preferences(self, learner_id: str, prefs: LearnerPreferences) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (learner_id, learning_style, pace, extra)"
            " VALUES (?, ?, ?, ?)",
            (learner_id, prefs.learning_style, prefs.pace, json.dumps(prefs.extra)),
        )
        self._conn.commit()

    def get_preferences(self, learner_id: str) -> Optional[LearnerPreferences]:
        row = self._conn.execute(
            "SELECT * FROM preferences WHERE learner_id = ?", (learner_id,)
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
            "INSERT OR REPLACE INTO nodes (id, level, name, parent_id, metadata)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                node.id,
                node.level.value,
                node.name,
                node.parent_id,
                json.dumps(node.metadata),
            ),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self, level: Optional[NodeLevel] = None) -> List[KnowledgeNode]:
        if level is None:
            rows = self._conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE level = ? ORDER BY id", (level.value,)
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> KnowledgeNode:
        return KnowledgeNode(
            id=row["id"],
            level=NodeLevel(row["level"]),
            name=row["name"],
            parent_id=row["parent_id"],
            metadata=json.loads(row["metadata"]),
        )

    def save_prerequisite(self, topic_id: str, prereq_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO prerequisites (topic_id, prereq_id) VALUES (?, ?)",
            (topic_id, prereq_id),
        )
        self._conn.commit()

    def get_prerequisites(self, topic_id: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT prereq_id FROM prerequisites WHERE topic_id = ? ORDER BY prereq_id",
            (topic_id,),
        ).fetchall()
        return [r["prereq_id"] for r in rows]

    # -- plans --

    def save_plan(self, plan: LearningPlan) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO plans"
            " (id, learner_id, topic_ids, criteria_overrides, created_at, version)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                plan.id,
                plan.learner_id,
                json.dumps(plan.topic_ids),
                json.dumps(
                    {k: v.as_dict() for k, v in plan.criteria_overrides.items()}
                ),
                _dt_to_str(plan.created_at),
                plan.version,
            ),
        )
        self._conn.commit()

    def get_plan(self, plan_id: str) -> Optional[LearningPlan]:
        row = self._conn.execute(
            "SELECT * FROM plans WHERE id = ?", (plan_id,)
        ).fetchone()
        return self._row_to_plan(row) if row else None

    def get_plan_for_learner(self, learner_id: str) -> Optional[LearningPlan]:
        row = self._conn.execute(
            "SELECT * FROM plans WHERE learner_id = ? ORDER BY version DESC LIMIT 1",
            (learner_id,),
        ).fetchone()
        return self._row_to_plan(row) if row else None

    @staticmethod
    def _row_to_plan(row: sqlite3.Row) -> LearningPlan:
        return LearningPlan(
            id=row["id"],
            learner_id=row["learner_id"],
            topic_ids=json.loads(row["topic_ids"]),
            criteria_overrides={
                k: MasteryParams.from_dict(v)
                for k, v in json.loads(row["criteria_overrides"]).items()
            },
            created_at=_str_to_dt(row["created_at"]),
            version=row["version"],
        )

    # -- evidence --

    def save_evidence(self, evidence: Evidence) -> None:
        # Plain INSERT, not OR REPLACE: evidence is append-only, so a duplicate
        # id is a caller bug and must fail loudly instead of silently
        # overwriting history. Surfaced as ValueError (not the backend's
        # native integrity error) so the contract is backend-agnostic.
        try:
            self._conn.execute(
                "INSERT INTO evidence"
                " (id, learner_id, node_id, source_agent, activity_type, occurred_at,"
                "  performance, confidence, attempts, hints_used, details)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evidence.id,
                    evidence.learner_id,
                    evidence.node_id,
                    evidence.source_agent,
                    evidence.activity_type,
                    _dt_to_str(evidence.occurred_at),
                    evidence.performance,
                    evidence.confidence,
                    evidence.attempts,
                    evidence.hints_used,
                    json.dumps(evidence.details),
                ),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"duplicate evidence id: {evidence.id}") from None
        self._conn.commit()

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        row = self._conn.execute(
            "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
        return self._row_to_evidence(row) if row else None

    def list_evidence(
        self, learner_id: str, node_id: Optional[str] = None
    ) -> List[Evidence]:
        if node_id is None:
            rows = self._conn.execute(
                "SELECT * FROM evidence WHERE learner_id = ? ORDER BY occurred_at",
                (learner_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM evidence WHERE learner_id = ? AND node_id = ?"
                " ORDER BY occurred_at",
                (learner_id, node_id),
            ).fetchall()
        return [self._row_to_evidence(r) for r in rows]

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> Evidence:
        return Evidence(
            id=row["id"],
            learner_id=row["learner_id"],
            node_id=row["node_id"],
            source_agent=row["source_agent"],
            activity_type=row["activity_type"],
            occurred_at=_str_to_dt(row["occurred_at"]),
            performance=row["performance"],
            confidence=row["confidence"],
            attempts=row["attempts"],
            hints_used=row["hints_used"],
            details=json.loads(row["details"]),
        )

    # -- mastery records --

    def save_mastery_record(self, record: MasteryRecord) -> None:
        # Plain INSERT, not OR REPLACE: mastery history is append-only and must
        # never be rewritten; a duplicate id fails loudly. Backend-agnostic
        # ValueError, matching the StorageBackend contract.
        try:
            self._conn.execute(
                "INSERT INTO mastery_records"
                " (id, node_id, learner_id, level, updated_at, rule_version,"
                "  params_in_effect, evidence_ids, assertion, asserted_by, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.node_id,
                    record.learner_id,
                    record.level.value,
                    _dt_to_str(record.updated_at),
                    record.rule_version,
                    json.dumps(record.params_in_effect),
                    json.dumps(record.evidence_ids),
                    int(record.assertion),
                    record.asserted_by,
                    record.reason,
                ),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"duplicate mastery record id: {record.id}") from None
        self._conn.commit()

    def get_mastery_history(
        self, learner_id: str, node_id: str
    ) -> List[MasteryRecord]:
        rows = self._conn.execute(
            "SELECT * FROM mastery_records WHERE learner_id = ? AND node_id = ?"
            " ORDER BY updated_at, rowid",
            (learner_id, node_id),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_current_mastery(
        self, learner_id: str, node_id: str
    ) -> Optional[MasteryRecord]:
        history = self.get_mastery_history(learner_id, node_id)
        return history[-1] if history else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MasteryRecord:
        return MasteryRecord(
            id=row["id"],
            node_id=row["node_id"],
            learner_id=row["learner_id"],
            level=MasteryLevel(row["level"]),
            updated_at=_str_to_dt(row["updated_at"]),
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
            "INSERT OR REPLACE INTO overrides"
            " (id, learner_id, level, scope_node_id, expires_at, created_at,"
            " reason, promoted)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                override.id,
                override.learner_id,
                override.level.value,
                override.scope_node_id,
                _dt_to_str(override.expires_at),
                _dt_to_str(override.created_at),
                override.reason,
                int(override.promoted),
            ),
        )
        self._conn.commit()

    def list_overrides(self, learner_id: str) -> List[DynamicOverride]:
        rows = self._conn.execute(
            "SELECT * FROM overrides WHERE learner_id = ? ORDER BY expires_at",
            (learner_id,),
        ).fetchall()
        return [
            DynamicOverride(
                id=r["id"],
                learner_id=r["learner_id"],
                level=MasteryLevel(r["level"]),
                scope_node_id=r["scope_node_id"],
                expires_at=_str_to_dt(r["expires_at"]),
                # Pre-migration rows have '': sort them as oldest.
                created_at=(
                    _str_to_dt(r["created_at"])
                    if r["created_at"]
                    else datetime.min.replace(tzinfo=timezone.utc)
                ),
                reason=r["reason"],
                promoted=bool(r["promoted"]),
            )
            for r in rows
        ]

    def delete_override(self, override_id: str) -> None:
        self._conn.execute("DELETE FROM overrides WHERE id = ?", (override_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Future backend: Mem0 adapter (documented, not implemented)
# ---------------------------------------------------------------------------
#
# A Mem0-backed implementation of StorageBackend would:
#   * persist each entity as a Mem0 memory with metadata
#     {kind: node|evidence|mastery_record|..., learner_id, node_id, ...},
#   * keep the append-only history by writing new memories per record
#     (never updating in place),
#   * use Mem0's semantic search for evidence lookup ("show me everything
#     about regularization from the quiz agent"),
#   * leave the mastery math untouched — the engine only depends on the
#     StorageBackend protocol, so swapping SQLite for Mem0 changes no
#     framework logic.
#
# The protocol above is deliberately narrow so the adapter stays small.
