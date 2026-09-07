"""The EduPAAL skill: the thin interface vertical agents call.

The skill is deliberately thin. It supplies vertical agents (tutor, quest,
assessment, visualization, practice) with the learner's preferences, the
current plan state, the next specific topic, and mastery context. It never
teaches and never runs progression — the mastery engine and retrieval do
the memory work underneath.

The skill is bound to one learner (set by ``cold_start``); share one
``StorageBackend`` across skill instances for many learners.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .entities import (
    DynamicOverride,
    Evidence,
    Learner,
    LearnerPreferences,
    LearningPlan,
    MasteryLevel,
    MasteryParams,
    MasteryRecord,
    NodeLevel,
    _new_id,
    _utcnow,
)
from .graph import KnowledgeGraph
from .mastery import MasteryEngine
from .retrieval import RankedNode, Retriever
from .store import StorageBackend


class EduPAALSkill:
    def __init__(
        self,
        store: StorageBackend,
        graph: KnowledgeGraph,
        default_params: Optional[MasteryParams] = None,
        learner_id: Optional[str] = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.engine = MasteryEngine(store, graph, default_params)
        self.learner_id = learner_id
        self._retriever: Optional[Retriever] = None

    # -------------------------------------------------------------- cold start

    def cold_start(
        self,
        learner_id: str,
        node_selection: List[str],
        preferences: LearnerPreferences,
        order: Optional[List[str]] = None,
        criteria_overrides: Optional[Dict[str, MasteryParams]] = None,
        learner_name: str = "",
    ) -> LearningPlan:
        """Manual cold start: a concept plan plus declared preferences.

        ``node_selection`` lists TOPIC ids at any depth (sub-topics included);
        ``order`` optionally reorders them (defaults to selection order).
        ``criteria_overrides`` tunes mastery knobs per node id (usually
        concept or topic).
        """
        if not node_selection:
            raise ValueError("node_selection must list at least one topic")
        self.graph.validate_plan_topics(node_selection)
        topic_ids = list(order) if order else list(node_selection)
        if set(topic_ids) != set(node_selection):
            raise ValueError("order must be a permutation of node_selection")
        self.graph.validate_plan_topics(topic_ids)

        overrides = criteria_overrides or {}
        for oid in overrides:
            try:
                self.graph.get(oid)  # unknown node -> clear ValueError
            except KeyError:
                raise ValueError(
                    f"criteria_overrides references unknown node: {oid}"
                ) from None

        self.store.save_learner(Learner(id=learner_id, name=learner_name))
        self.store.save_preferences(learner_id, preferences)
        existing = self.store.get_plan_for_learner(learner_id)
        plan = LearningPlan(
            id=_new_id("plan"),
            learner_id=learner_id,
            topic_ids=topic_ids,
            criteria_overrides=dict(overrides),
            created_at=_utcnow(),
            # re-planning bumps the version so get_plan_for_learner (latest
            # version wins) stays deterministic
            version=(existing.version + 1) if existing else 1,
        )
        self.store.save_plan(plan)
        self.learner_id = learner_id
        self._retriever = None
        return plan

    # ------------------------------------------------------------------ guards

    def _require_learner(self) -> str:
        if not self.learner_id:
            raise ValueError("no learner bound: call cold_start() first")
        return self.learner_id

    def _retriever_for(self) -> Retriever:
        learner_id = self._require_learner()
        if self._retriever is None:
            self._retriever = Retriever(
                self.store, self.graph, self.engine, learner_id
            )
        return self._retriever

    # ---------------------------------------------------------------- evidence

    def record_evidence(self, evidence: Evidence) -> List[MasteryRecord]:
        """Report one observed learning signal. The engine enforces the
        finest-grain rule: evidence targets leaf TOPIC nodes only."""
        learner_id = self._require_learner()
        if evidence.learner_id != learner_id:
            raise ValueError(
                f"evidence learner {evidence.learner_id} != bound learner {learner_id}"
            )
        return self.engine.record_evidence(evidence)

    def assert_mastery(
        self, node_id: str, level: MasteryLevel, asserted_by: str, reason: str
    ) -> MasteryRecord:
        """Privileged assertion path (assessment agents, humans). Appends to
        history; never rewrites it."""
        return self.engine.assert_mastery(
            self._require_learner(), node_id, level, asserted_by, reason
        )

    # ---------------------------------------------------------------- overrides

    def set_override(
        self,
        level: MasteryLevel,
        reason: str,
        scope_node_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> DynamicOverride:
        """Set a temporary override. Temporary unless explicitly promoted.

        Setting a new override for a scope that already has an active one
        supersedes it: among active same-scope overrides the most recently
        created wins (the older one stays in history as an audit trail).
        """
        if isinstance(level, str):
            level = MasteryLevel(level)
        if level == MasteryLevel.UNKNOWN:
            raise ValueError("cannot override to UNKNOWN")
        if scope_node_id is not None:
            self.graph.get(scope_node_id)
        override = DynamicOverride(
            learner_id=self._require_learner(),
            level=level,
            scope_node_id=scope_node_id,
            expires_at=expires_at or (_utcnow() + timedelta(days=7)),
            reason=reason,
        )
        self.store.save_override(override)
        return override

    def clear_override(self, scope_node_id: Optional[str] = None) -> int:
        """Remove overrides for a scope (None = global scope)."""
        learner_id = self._require_learner()
        doomed = [
            o
            for o in self.store.list_overrides(learner_id)
            if o.scope_node_id == scope_node_id
        ]
        for o in doomed:
            self.store.delete_override(o.id)
        return len(doomed)

    def promote_override(
        self, scope_node_id: Optional[str] = None
    ) -> MasteryRecord:
        """Promote a temporary override into a durable assertion."""
        return self.engine.promote_override(self._require_learner(), scope_node_id)

    def active_override(
        self, node_id: Optional[str] = None, now: Optional[datetime] = None
    ) -> Optional[DynamicOverride]:
        """The override currently shaping a node: node-scoped wins over
        global; among active same-scope overrides the newest wins."""
        return self.engine.active_override(self._require_learner(), node_id, now)

    # ----------------------------------------------------------------- reading

    def effective_mastery(self, node_id: str) -> MasteryLevel:
        return self.engine.effective_mastery(self._require_learner(), node_id)

    def mastery_history(self, node_id: str) -> List[MasteryRecord]:
        return self.store.get_mastery_history(self._require_learner(), node_id)

    def next_topic(self) -> Optional[KnowledgeNode]:
        return self._retriever_for().next_topic()

    def strongest(
        self, n: int, level: NodeLevel = NodeLevel.TOPIC
    ) -> List[RankedNode]:
        return self._retriever_for().strongest(n, level)

    def weakest(
        self, n: int, level: NodeLevel = NodeLevel.TOPIC
    ) -> List[RankedNode]:
        return self._retriever_for().weakest(n, level)

    # -------------------------------------------------------- grounding packet

    def grounding_packet(self, node_id: str) -> Dict[str, Any]:
        """The context vertical agents need: preferences/style/pace, plan
        state, the next specific topic, mastery context for the node, and the
        active override if one is shaping this node."""
        learner_id = self._require_learner()
        node = self.graph.get(node_id)
        prefs = self.store.get_preferences(learner_id)
        plan = self.store.get_plan_for_learner(learner_id)
        retriever = self._retriever_for()

        nxt = retriever.next_topic()
        advanced = sum(
            1
            for tid in (plan.topic_ids if plan else [])
            if self.engine.effective_mastery(learner_id, tid) == MasteryLevel.ADVANCED
        )
        try:
            position = plan.topic_ids.index(nxt.id) if (plan and nxt) else None
        except ValueError:  # pragma: no cover - defensive
            position = None

        level, score = retriever.rollup(node_id)
        context: Dict[str, Any] = {
            "node": {"id": node.id, "name": node.name, "level": node.level.value,
                     "mastery": level.value, "score": score},
            # nearest ancestor first; a list because sub-topics mean several
            # ancestors can share the same level
            "ancestors": [
                {
                    "id": a.id, "name": a.name, "level": a.level.value,
                    "mastery": rl.value, "score": sc,
                }
                for a in self.graph.ancestors(node_id)
                for rl, sc in [retriever.rollup(a.id)]
            ],
        }

        override = self.active_override(node_id)
        packet: Dict[str, Any] = {
            "learner_id": learner_id,
            "preferences": {
                "learning_style": prefs.learning_style if prefs else None,
                "pace": prefs.pace if prefs else None,
                "extra": prefs.extra if prefs else {},
            },
            "plan": {
                "id": plan.id if plan else None,
                "version": plan.version if plan else None,
                "total_topics": len(plan.topic_ids) if plan else 0,
                "advanced_topics": advanced,
                "next_position": position,
            },
            "next_topic": {"id": nxt.id, "name": nxt.name} if nxt else None,
            "mastery_context": context,
            "active_override": (
                {
                    "id": override.id,
                    "level": override.level.value,
                    "scope_node_id": override.scope_node_id,
                    "expires_at": override.expires_at.isoformat(),
                    "reason": override.reason,
                }
                if override
                else None
            ),
        }
        return packet
