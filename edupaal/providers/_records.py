"""Shared canonical record serialization for the optional memory providers.

Both :class:`Mem0Provider` and :class:`MemOSProvider` persist every EduPAAL
entity as one self-contained JSON record. The record text is stored
**verbatim** (no LLM rewriting, no summarization) and parsed back into the
exact same dataclass on read, so the providers give the same exactness
guarantees as ``SQLiteBackend``.

The encoding mirrors ``SQLiteBackend`` field-for-field: datetimes become UTC
ISO-8601 strings, enums become their ``value``, and ``MasteryParams``
becomes its ``as_dict()`` mapping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from ..entities import (
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

RECORD_FORMAT = "edupaal-record/1"

# Record kinds, one per persisted entity type.
KIND_LEARNER = "learner"
KIND_PREFERENCES = "preferences"
KIND_NODE = "node"
KIND_PREREQUISITE = "prerequisite"
KIND_PLAN = "plan"
KIND_EVIDENCE = "evidence"
KIND_MASTERY_RECORD = "mastery_record"
KIND_OVERRIDE = "override"


def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def to_record(entity: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Serialize an entity to ``(kind, record_id, record_dict)``.

    ``record_dict`` is the canonical JSON envelope; ``record_id`` is the
    entity's stable EduPAAL id (used for duplicate detection).
    """
    if isinstance(entity, Learner):
        data = {"id": entity.id, "name": entity.name}
        return KIND_LEARNER, entity.id, _envelope(KIND_LEARNER, entity.id, data)
    if isinstance(entity, LearnerPreferences):
        raise TypeError(
            "LearnerPreferences has no id; use preferences_to_record(learner_id, prefs)"
        )
    if isinstance(entity, KnowledgeNode):
        data = {
            "id": entity.id,
            "level": entity.level.value,
            "name": entity.name,
            "parent_id": entity.parent_id,
            "metadata": entity.metadata,
        }
        return KIND_NODE, entity.id, _envelope(KIND_NODE, entity.id, data)
    if isinstance(entity, LearningPlan):
        data = {
            "id": entity.id,
            "learner_id": entity.learner_id,
            "topic_ids": entity.topic_ids,
            "criteria_overrides": {
                k: v.as_dict() for k, v in entity.criteria_overrides.items()
            },
            "created_at": _dt_to_str(entity.created_at),
            "version": entity.version,
        }
        return KIND_PLAN, entity.id, _envelope(KIND_PLAN, entity.id, data)
    if isinstance(entity, Evidence):
        data = {
            "id": entity.id,
            "learner_id": entity.learner_id,
            "node_id": entity.node_id,
            "source_agent": entity.source_agent,
            "activity_type": entity.activity_type,
            "occurred_at": _dt_to_str(entity.occurred_at),
            "performance": entity.performance,
            "confidence": entity.confidence,
            "attempts": entity.attempts,
            "hints_used": entity.hints_used,
            "details": entity.details,
        }
        return KIND_EVIDENCE, entity.id, _envelope(KIND_EVIDENCE, entity.id, data)
    if isinstance(entity, MasteryRecord):
        data = {
            "id": entity.id,
            "node_id": entity.node_id,
            "learner_id": entity.learner_id,
            "level": entity.level.value,
            "updated_at": _dt_to_str(entity.updated_at),
            "rule_version": entity.rule_version,
            "params_in_effect": entity.params_in_effect,
            "evidence_ids": entity.evidence_ids,
            "assertion": entity.assertion,
            "asserted_by": entity.asserted_by,
            "reason": entity.reason,
        }
        return KIND_MASTERY_RECORD, entity.id, _envelope(
            KIND_MASTERY_RECORD, entity.id, data
        )
    if isinstance(entity, DynamicOverride):
        data = {
            "id": entity.id,
            "learner_id": entity.learner_id,
            "level": entity.level.value,
            "scope_node_id": entity.scope_node_id,
            "expires_at": _dt_to_str(entity.expires_at),
            "created_at": _dt_to_str(entity.created_at),
            "reason": entity.reason,
            "promoted": entity.promoted,
        }
        return KIND_OVERRIDE, entity.id, _envelope(KIND_OVERRIDE, entity.id, data)
    raise TypeError(f"unsupported entity type: {type(entity).__name__}")


def _envelope(kind: str, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": RECORD_FORMAT,
        "kind": kind,
        "id": record_id,
        "data": data,
    }


def preferences_to_record(
    learner_id: str, prefs: LearnerPreferences
) -> Tuple[str, str, Dict[str, Any]]:
    record_id = f"preferences:{learner_id}"
    data = {
        "learner_id": learner_id,
        "learning_style": prefs.learning_style,
        "pace": prefs.pace,
        "extra": prefs.extra,
    }
    return KIND_PREFERENCES, record_id, _envelope(KIND_PREFERENCES, record_id, data)


def prerequisite_to_record(
    topic_id: str, prereq_id: str
) -> Tuple[str, str, Dict[str, Any]]:
    record_id = f"prerequisite:{topic_id}:{prereq_id}"
    data = {"topic_id": topic_id, "prereq_id": prereq_id}
    return KIND_PREREQUISITE, record_id, _envelope(
        KIND_PREREQUISITE, record_id, data
    )


def from_record(record: Dict[str, Any]) -> Any:
    """Parse a canonical record envelope back into its entity."""
    if record.get("format") != RECORD_FORMAT:
        raise ValueError(f"not an EduPAAL record: {record.get('format')!r}")
    kind = record["kind"]
    data = record["data"]
    if kind == KIND_LEARNER:
        return Learner(id=data["id"], name=data["name"])
    if kind == KIND_PREFERENCES:
        return (
            data["learner_id"],
            LearnerPreferences(
                learning_style=data["learning_style"],
                pace=data["pace"],
                extra=data["extra"],
            ),
        )
    if kind == KIND_NODE:
        return KnowledgeNode(
            id=data["id"],
            level=NodeLevel(data["level"]),
            name=data["name"],
            parent_id=data["parent_id"],
            metadata=data["metadata"],
        )
    if kind == KIND_PREREQUISITE:
        return (data["topic_id"], data["prereq_id"])
    if kind == KIND_PLAN:
        return LearningPlan(
            id=data["id"],
            learner_id=data["learner_id"],
            topic_ids=data["topic_ids"],
            criteria_overrides={
                k: MasteryParams.from_dict(v)
                for k, v in data["criteria_overrides"].items()
            },
            created_at=_str_to_dt(data["created_at"]),
            version=data["version"],
        )
    if kind == KIND_EVIDENCE:
        return Evidence(
            id=data["id"],
            learner_id=data["learner_id"],
            node_id=data["node_id"],
            source_agent=data["source_agent"],
            activity_type=data["activity_type"],
            occurred_at=_str_to_dt(data["occurred_at"]),
            performance=data["performance"],
            confidence=data["confidence"],
            attempts=data["attempts"],
            hints_used=data["hints_used"],
            details=data["details"],
        )
    if kind == KIND_MASTERY_RECORD:
        return MasteryRecord(
            id=data["id"],
            node_id=data["node_id"],
            learner_id=data["learner_id"],
            level=MasteryLevel(data["level"]),
            updated_at=_str_to_dt(data["updated_at"]),
            rule_version=data["rule_version"],
            params_in_effect=data["params_in_effect"],
            evidence_ids=data["evidence_ids"],
            assertion=data["assertion"],
            asserted_by=data["asserted_by"],
            reason=data["reason"],
        )
    if kind == KIND_OVERRIDE:
        return DynamicOverride(
            id=data["id"],
            learner_id=data["learner_id"],
            level=MasteryLevel(data["level"]),
            scope_node_id=data["scope_node_id"],
            expires_at=_str_to_dt(data["expires_at"]),
            created_at=_str_to_dt(data["created_at"]),
            reason=data["reason"],
            promoted=data["promoted"],
        )
    raise ValueError(f"unknown record kind: {kind!r}")
