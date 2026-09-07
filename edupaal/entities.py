"""Core entity schemas for EduPAAL.

These dataclasses are the storage contract: every backend (SQLite, Mem0
adapter, ...) persists exactly these shapes. They carry no behavior beyond
validation — the mastery engine, retrieval, and skill layers interpret them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class NodeLevel(str, Enum):
    """The four fixed levels of the EduPAAL knowledge graph (framework layer).

    The four upper levels are fixed, but TOPIC is recursively decomposable:
    a topic may have child topics (sub-topics) to arbitrary depth, e.g.
    Regularization -> Dropout -> Dropout Rate Tuning.
    """

    SPACE = "space"  # e.g. Science, Social Science
    SUBJECT = "subject"  # e.g. Maths, Physics, ML, Economics
    CONCEPT = "concept"  # e.g. Algebra, Overfitting, Supply and Demand
    TOPIC = "topic"  # e.g. Linearization, Regularization; decomposable into sub-topics


class MasteryLevel(str, Enum):
    """Per-node mastery. UNKNOWN is the unevaluated state before any
    evidence exists — it is not a fourth mastery level."""

    UNKNOWN = "unknown"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# Numeric scale used for rollup and ranking. UNKNOWN is excluded from
# aggregates (it means "no information", not "zero knowledge").
MASTERY_SCORES: Dict[MasteryLevel, int] = {
    MasteryLevel.BEGINNER: 0,
    MasteryLevel.INTERMEDIATE: 1,
    MasteryLevel.ADVANCED: 2,
}


@dataclass
class KnowledgeNode:
    """One node in the knowledge graph: a space, subject, concept, or topic.

    Hierarchy is expressed through ``parent_id`` (the part_of edge).
    A TOPIC's parent may be a CONCEPT (a top-level topic) or another TOPIC
    (a sub-topic, recursively to any depth). Prerequisite edges are stored
    separately on the graph because they may cross concepts and subjects,
    and may link topics at any depth.
    """

    id: str
    level: NodeLevel
    name: str
    parent_id: Optional[str] = None  # None only for SPACE nodes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.level, str):
            self.level = NodeLevel(self.level)
        if self.level == NodeLevel.SPACE and self.parent_id is not None:
            raise ValueError("SPACE nodes must not have a parent_id")
        if self.level != NodeLevel.SPACE and not self.parent_id:
            raise ValueError(f"{self.level.value} nodes require a parent_id")


@dataclass
class LearnerPreferences:
    """Learner-declared preferences captured at cold start.

    ``learning_style`` and ``pace`` are free-form strings (e.g. "visual",
    "steady") — EduPAAL does not impose a taxonomy; vertical agents interpret
    them. ``extra`` carries anything else the deployment wants to remember.
    """

    learning_style: str
    pace: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Learner:
    id: str
    name: str = ""


@dataclass
class MasteryParams:
    """Tunable knobs for the mastery heuristics (heuristic-v1).

    The framework defines the *shape* of the promotion rules; the deployment
    controls these parameters — globally, or per concept/topic via the
    learning plan's ``criteria_overrides``.
    """

    k_evidence: int = 3  # evidence items (most recent, in window) per evaluation
    window_days: int = 30  # recency window for the evaluation set
    t_intermediate: float = 0.65  # mean performance bar: BEGINNER -> INTERMEDIATE
    t_advanced: float = 0.80  # mean performance bar: INTERMEDIATE -> ADVANCED
    t_contradict: float = 0.40  # any evaluation-set item below this blocks promotion
    cross_modal_advanced: bool = True  # ADVANCED needs >=2 distinct activity_types
    prereq_gate: MasteryLevel = MasteryLevel.INTERMEDIATE  # prereq satisfaction bar

    def __post_init__(self) -> None:
        if isinstance(self.prereq_gate, str):
            self.prereq_gate = MasteryLevel(self.prereq_gate)
        if self.k_evidence < 1:
            raise ValueError("k_evidence must be >= 1")
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")
        for name in ("t_intermediate", "t_advanced", "t_contradict"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "k_evidence": self.k_evidence,
            "window_days": self.window_days,
            "t_intermediate": self.t_intermediate,
            "t_advanced": self.t_advanced,
            "t_contradict": self.t_contradict,
            "cross_modal_advanced": self.cross_modal_advanced,
            "prereq_gate": self.prereq_gate.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MasteryParams":
        return cls(**d)


@dataclass
class LearningPlan:
    """The ordered traversal of topic nodes for one learner.

    The plan is a path through the knowledge graph: an ordered list of TOPIC
    node ids. ``criteria_overrides`` lets the deployment tune mastery knobs
    per node (usually per concept or topic).
    """

    id: str
    learner_id: str
    topic_ids: List[str]  # ordered traversal; every entry must be a TOPIC node
    criteria_overrides: Dict[str, MasteryParams] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    version: int = 1


@dataclass
class Evidence:
    """One observed learning signal about a TOPIC node.

    Convention: attach evidence at the finest-grained node available — if a
    topic is decomposed into sub-topics, report against the sub-topic, and
    let mastery roll up the chain. The engine accepts any TOPIC-depth node.

    The reporting vertical agent normalizes its own signal into
    ``performance`` in [0, 1]; EduPAAL never interprets raw scores — the
    normalized layer is what keeps the mastery rules agent-agnostic. ``details``
    preserves the raw payload for audit.
    """

    id: str
    learner_id: str
    node_id: str  # must reference a TOPIC node
    source_agent: str  # REQUIRED: which vertical agent reported this
    activity_type: str  # REQUIRED: quiz | practice | dialogue | visualization | quest | human_report | ...
    occurred_at: datetime
    performance: float  # REQUIRED, normalized by the reporter into [0, 1]
    confidence: Optional[float] = None  # reporter's self-assessed trust in [0, 1]
    attempts: Optional[int] = None
    hints_used: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.learner_id:
            raise ValueError("learner_id is required")
        if not self.node_id:
            raise ValueError("node_id is required")
        if not self.source_agent:
            raise ValueError("source_agent is required")
        if not self.activity_type:
            raise ValueError("activity_type is required")
        if not 0.0 <= self.performance <= 1.0:
            raise ValueError("performance must be within [0, 1]")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.occurred_at.tzinfo is None:
            self.occurred_at = self.occurred_at.replace(tzinfo=timezone.utc)


@dataclass
class MasteryRecord:
    """One durable mastery transition. History is append-only: the current
    mastery of a node is the latest record; every past record is retained so
    any memory view is reproducible."""

    node_id: str
    learner_id: str
    level: MasteryLevel
    updated_at: datetime
    rule_version: str  # e.g. "heuristic-v1", "assertion-v1"
    params_in_effect: Dict[str, Any]  # exact knobs that produced this transition
    evidence_ids: List[str]  # evidence that caused this transition
    assertion: bool = False  # True when written via assert_mastery / promoted override
    asserted_by: Optional[str] = None
    reason: Optional[str] = None
    id: str = field(default_factory=lambda: _new_id("mr"))

    def __post_init__(self) -> None:
        if isinstance(self.level, str):
            self.level = MasteryLevel(self.level)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)


@dataclass
class DynamicOverride:
    """A temporary mastery override. Overrides are scoped to one node or
    global, and they expire — unless explicitly promoted into a durable
    assertion via ``promote_override``."""

    learner_id: str
    level: MasteryLevel
    scope_node_id: Optional[str]  # None => global override
    expires_at: datetime
    reason: str
    promoted: bool = False
    id: str = field(default_factory=lambda: _new_id("ovr"))

    def __post_init__(self) -> None:
        if isinstance(self.level, str):
            self.level = MasteryLevel(self.level)
        if self.expires_at.tzinfo is None:
            self.expires_at = self.expires_at.replace(tzinfo=timezone.utc)

    def is_active(self, now: Optional[datetime] = None) -> bool:
        now = now or _utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return not self.promoted and now < self.expires_at
