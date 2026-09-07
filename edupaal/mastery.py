"""The EduPAAL mastery engine: versioned, explainable, deterministic heuristics.

PAAL *owns* mastery transitions — it is not dumb storage. Vertical agents
report normalized evidence; the engine applies the promotion rules. Every
rule parameter is a tunable knob (see ``MasteryParams``): the framework
defines the shape of the rules and ships transparent defaults, the
deployment controls the dials.

heuristic-v1 rules
------------------
* Evidence and assertions target *leaf* TOPIC nodes only (enforced — a
  decomposed topic must be reported through its sub-topics, the finest grain
  available). Concept/subject/space mastery, and the mastery of a decomposed
  topic, is pure rollup (see ``edupaal.retrieval``). Overrides may target any
  node and win over rollup for that node.
* The first ever evidence for a topic moves UNKNOWN -> BEGINNER: any
  engagement establishes the beginner state.
* From BEGINNER (or INTERMEDIATE), the engine evaluates the up-to-K most
  recent evidence items inside a W-day window anchored at the latest
  evidence timestamp:
    - need at least K items in the set,
    - mean performance must clear the bar for the next level
      (t_intermediate / t_advanced),
    - no item in the set may fall below t_contradict (it blocks promotion),
    - INTERMEDIATE -> ADVANCED additionally requires >= 2 distinct
      activity_types in the set when cross_modal_advanced is on.
* Promotions chain within one call (BEGINNER -> INTERMEDIATE -> ADVANCED),
  each step writing its own MasteryRecord. Demotion/decay is intentionally
  out of scope for v1.

Every transition writes a MasteryRecord carrying the rule version, the exact
parameters in effect, and the evidence IDs that caused it — so any past
decision can be replayed and explained.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .entities import (
    DynamicOverride,
    Evidence,
    MASTERY_SCORES,
    MasteryLevel,
    MasteryParams,
    MasteryRecord,
    NodeLevel,
    _new_id,
    _utcnow,
)
from .graph import KnowledgeGraph
from .store import StorageBackend

HEURISTIC_VERSION = "heuristic-v1"
ASSERTION_VERSION = "assertion-v1"


class MasteryEngine:
    def __init__(
        self,
        store: StorageBackend,
        graph: KnowledgeGraph,
        default_params: Optional[MasteryParams] = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.default_params = default_params or MasteryParams()

    # ------------------------------------------------------------ parameters

    def params_for(self, learner_id: str, node_id: str) -> MasteryParams:
        """Resolve the effective knobs: plan override for the node, else the
        nearest ancestor's override (concept, then subject...), else defaults."""
        plan = self.store.get_plan_for_learner(learner_id)
        if plan is not None:
            node = self.graph.get(node_id)
            candidates = [node_id] + [a.id for a in self.graph.ancestors(node_id)]
            for cid in candidates:
                if cid in plan.criteria_overrides:
                    return plan.criteria_overrides[cid]
        return self.default_params

    # ---------------------------------------------------------------- record

    def record_evidence(self, evidence: Evidence) -> List[MasteryRecord]:
        """Persist evidence and apply the promotion rules.

        Returns the MasteryRecords written by this call (possibly empty when
        the evidence does not move mastery). Raises on invalid evidence.
        """
        node = self.graph.get(evidence.node_id)
        if node.level != NodeLevel.TOPIC:
            raise ValueError(
                f"evidence must target a TOPIC node ({node.id} is {node.level.value}); "
                "higher levels roll up from topics"
            )
        if self.graph.children(node.id):
            raise ValueError(
                f"topic {node.id} is decomposed into sub-topics; record evidence "
                "on the finest-grained sub-topic instead"
            )
        self.store.save_evidence(evidence)

        written: List[MasteryRecord] = []
        if self._current_level(evidence.learner_id, evidence.node_id) == MasteryLevel.UNKNOWN:
            written.append(
                self._write_record(
                    evidence.learner_id,
                    evidence.node_id,
                    MasteryLevel.BEGINNER,
                    HEURISTIC_VERSION,
                    self.params_for(evidence.learner_id, evidence.node_id),
                    [evidence.id],
                    updated_at=evidence.occurred_at,
                )
            )

        # Fixed-point promotion: keep stepping while the evidence supports it.
        # Each step re-reads the stored level, so one strong evidence set can
        # chain BEGINNER -> INTERMEDIATE -> ADVANCED, writing one record per step.
        while True:
            level = self._current_level(evidence.learner_id, evidence.node_id)
            if level == MasteryLevel.ADVANCED:
                break
            nxt = self._try_promote(evidence.learner_id, evidence.node_id, level)
            if nxt is None:
                break
            written.append(nxt)

        return written

    def _try_promote(
        self, learner_id: str, node_id: str, current: MasteryLevel
    ) -> Optional[MasteryRecord]:
        params = self.params_for(learner_id, node_id)
        evidence = sorted(
            self.store.list_evidence(learner_id, node_id),
            key=lambda e: e.occurred_at,
        )
        if not evidence:
            return None
        anchor = max(e.occurred_at for e in evidence)
        cutoff = anchor - timedelta(days=params.window_days)
        windowed = [e for e in evidence if e.occurred_at >= cutoff]
        candidates = windowed[-params.k_evidence :]
        if len(candidates) < params.k_evidence:
            return None

        if current == MasteryLevel.BEGINNER:
            target, bar = MasteryLevel.INTERMEDIATE, params.t_intermediate
        elif current == MasteryLevel.INTERMEDIATE:
            target, bar = MasteryLevel.ADVANCED, params.t_advanced
        else:  # pragma: no cover - loop guard makes this unreachable
            return None

        mean_perf = sum(e.performance for e in candidates) / len(candidates)
        if mean_perf < bar:
            return None
        if any(e.performance < params.t_contradict for e in candidates):
            return None
        if (
            target == MasteryLevel.ADVANCED
            and params.cross_modal_advanced
            and len({e.activity_type for e in candidates}) < 2
        ):
            return None

        return self._write_record(
            learner_id,
            node_id,
            target,
            HEURISTIC_VERSION,
            params,
            [e.id for e in candidates],
            updated_at=anchor,
        )

    # --------------------------------------------------------------- assert

    def assert_mastery(
        self,
        learner_id: str,
        node_id: str,
        level: MasteryLevel,
        asserted_by: str,
        reason: str,
    ) -> MasteryRecord:
        """Privileged override path for assessment agents and humans.

        Writes a MasteryRecord marked as an assertion. It does not erase or
        rewrite history — it appends, like everything else. Subsequent
        evidence continues from the asserted level.
        """
        if isinstance(level, str):
            level = MasteryLevel(level)
        if level == MasteryLevel.UNKNOWN:
            raise ValueError("cannot assert UNKNOWN")
        node = self.graph.get(node_id)
        if node.level != NodeLevel.TOPIC:
            raise ValueError("assertions target TOPIC nodes; higher levels roll up")
        if self.graph.children(node_id):
            raise ValueError(
                f"topic {node_id} is decomposed into sub-topics; assert on the "
                "finest-grained sub-topic instead (or use an override)"
            )
        return self._write_record(
            learner_id,
            node_id,
            level,
            ASSERTION_VERSION,
            self.params_for(learner_id, node_id),
            evidence_ids=[],
            assertion=True,
            asserted_by=asserted_by,
            reason=reason,
            updated_at=_utcnow(),
        )

    # -------------------------------------------------------------- resolve

    def current_level(self, learner_id: str, node_id: str) -> MasteryLevel:
        """The node's own latest record: leaf-level, no rollup, no overrides.

        For aggregated views (concepts, subjects, decomposed topics) use
        ``effective_mastery`` / ``rolled_up_mastery`` instead."""

        return self._current_level(learner_id, node_id)

    def _current_level(self, learner_id: str, node_id: str) -> MasteryLevel:
        rec = self.store.get_current_mastery(learner_id, node_id)
        return rec.level if rec else MasteryLevel.UNKNOWN

    def active_override(
        self,
        learner_id: str,
        node_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional[DynamicOverride]:
        """The override currently shaping a node.

        Precedence: an active node-scoped override beats an active global
        one. Among active overrides of the *same* scope, the most recently
        created wins — a newer operator correction supersedes an older one
        (a long-lived stale override must not silently shadow a newer short
        correction). Shadowed overrides remain in history as an audit trail.
        ``node_id=None`` matches only global overrides.
        """
        now = now or _utcnow()
        scoped: List[DynamicOverride] = []
        glob: List[DynamicOverride] = []
        for override in self.store.list_overrides(learner_id):
            if not override.is_active(now):
                continue
            if node_id is not None and override.scope_node_id == node_id:
                scoped.append(override)
            elif override.scope_node_id is None:
                glob.append(override)
        pool = scoped or glob
        if not pool:
            return None
        return max(pool, key=lambda o: o.created_at)

    def effective_mastery(
        self, learner_id: str, node_id: str, now: Optional[datetime] = None
    ) -> MasteryLevel:
        """Mastery as vertical agents should treat it: an active override
        (node-scoped, else global) wins; otherwise mastery rolls up the full
        chain below the node. Leaf nodes report their own latest record, so
        for leaf TOPIC nodes this is exactly the record history; for
        concepts/subjects/spaces (and decomposed topics) it is the rollup —
        never UNKNOWN-just-because-no-direct-record-exists."""
        return self.rolled_up_mastery(learner_id, node_id, now)[0]

    def rolled_up_mastery(
        self, learner_id: str, node_id: str, now: Optional[datetime] = None
    ) -> Tuple[MasteryLevel, Optional[float]]:
        """(level, mean_score) for any node at any depth, with full-chain
        rollup: sub-topic -> ... -> topic -> concept -> subject -> space.

        An explicit active override on the node always wins. Nodes with
        children report the mean of their children's scores (UNKNOWN children
        excluded; a node with no evaluated children is UNKNOWN). Leaf nodes
        report their own latest record. Score is None when UNKNOWN.
        Deterministic: score ties round to the lower level.
        """
        override = self.active_override(learner_id, node_id, now)
        if override is not None:
            return override.level, float(MASTERY_SCORES[override.level])
        children = self.graph.children(node_id)
        if not children:
            level = self._current_level(learner_id, node_id)
            if level == MasteryLevel.UNKNOWN:
                return MasteryLevel.UNKNOWN, None
            return level, float(MASTERY_SCORES[level])
        scores: List[float] = []
        for child in children:
            _, s = self.rolled_up_mastery(learner_id, child.id, now)
            if s is not None:
                scores.append(s)
        if not scores:
            return MasteryLevel.UNKNOWN, None
        mean = sum(scores) / len(scores)
        nearest = min(
            MASTERY_SCORES.items(), key=lambda kv: (abs(kv[1] - mean), kv[1])
        )[0]
        return nearest, mean

    def promote_override(
        self, learner_id: str, scope_node_id: Optional[str] = None
    ) -> MasteryRecord:
        """Promote an active temporary override into a durable assertion.

        Promotes the *effective* override for the scope — the same one
        ``active_override`` reports (most recently created among active
        same-scope overrides). Promotion retires the scope's whole
        temporary stack: the effective override becomes the durable
        assertion and any shadowed same-scope overrides are retired with
        it, so the assertion stands on its own afterwards.
        """
        now = _utcnow()
        candidates = [
            o
            for o in self.store.list_overrides(learner_id)
            if o.is_active(now) and o.scope_node_id == scope_node_id
        ]
        if not candidates:
            raise ValueError("no active override for that scope to promote")
        target = max(candidates, key=lambda o: o.created_at)
        if target.scope_node_id is None:
            raise ValueError("global overrides cannot be promoted; scope them first")
        scope_node = self.graph.get(target.scope_node_id)
        if scope_node.level != NodeLevel.TOPIC or self.graph.children(scope_node.id):
            # assert_mastery would reject this deep inside with a message about
            # "use an override" — which is exactly what we are promoting. Fail
            # here instead, with the actual reason.
            raise ValueError(
                f"cannot promote override on {scope_node.id} "
                f"({scope_node.level.value}): promotion writes a leaf-topic "
                "assertion, and only leaf TOPIC nodes can carry one; keep the "
                "temporary override instead"
            )
        record = self.assert_mastery(
            learner_id,
            target.scope_node_id,
            target.level,
            asserted_by="override-promotion",
            reason=f"promoted override {target.id}: {target.reason}",
        )
        # Retire the scope's whole temporary stack so the assertion stands
        # on its own: no active temporary override may contradict it.
        for o in candidates:
            o.promoted = True
            self.store.save_override(o)
        return record

    # --------------------------------------------------------------- internals

    def _write_record(
        self,
        learner_id: str,
        node_id: str,
        level: MasteryLevel,
        rule_version: str,
        params: MasteryParams,
        evidence_ids: List[str],
        updated_at: datetime,
        assertion: bool = False,
        asserted_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> MasteryRecord:
        record = MasteryRecord(
            id=_new_id("mr"),
            node_id=node_id,
            learner_id=learner_id,
            level=level,
            updated_at=updated_at,
            rule_version=rule_version,
            params_in_effect=params.as_dict(),
            evidence_ids=list(evidence_ids),
            assertion=assertion,
            asserted_by=asserted_by,
            reason=reason,
        )
        self.store.save_mastery_record(record)
        return record
