"""Retrieval over the shared learning memory.

Mastery rolls up the full chain: sub-topic -> ... -> topic -> concept ->
subject -> space. The documented default: any node *with children* reports
the mean of its children's scores (UNKNOWN children excluded; a node with
no evaluated children is UNKNOWN). Leaf nodes report their own effective
mastery. An explicit active override on a node always wins over aggregation.
Prerequisite *gating* compares rolled-up mastery directly — weakest-link
semantics at the point of use.

Ranking rules (deterministic; ties broken by node id):
  * weakest(n): UNKNOWN sorts below BEGINNER — unevaluated nodes are the
    biggest gap and therefore "weakest".
  * strongest(n): UNKNOWN nodes are excluded — nothing unevaluated can be
    "strongest".
  * next_topic(): first topic in plan order that is not yet ADVANCED and
    whose prerequisites (at any topic depth) all clear the prereq gate.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .entities import MASTERY_SCORES, KnowledgeNode, MasteryLevel, NodeLevel
from .graph import KnowledgeGraph
from .mastery import MasteryEngine
from .store import StorageBackend

RankedNode = Tuple[KnowledgeNode, MasteryLevel, Optional[float]]


class Retriever:
    def __init__(
        self,
        store: StorageBackend,
        graph: KnowledgeGraph,
        engine: MasteryEngine,
        learner_id: str,
    ) -> None:
        self.store = store
        self.graph = graph
        self.engine = engine
        self.learner_id = learner_id

    # ---------------------------------------------------------------- rollup

    def rollup(self, node_id: str) -> Tuple[MasteryLevel, Optional[float]]:
        """(level, mean_score) for any node at any depth.

        Delegates to the engine so leaf reads, override precedence, and
        aggregation live in exactly one place. Score is None when UNKNOWN.
        Nodes with children aggregate their children; leaves report their own
        effective mastery; an explicit override on the node wins."""
        return self.engine.rolled_up_mastery(self.learner_id, node_id)

    # ---------------------------------------------------------------- ranking

    def _ranked(self, level: NodeLevel) -> List[RankedNode]:
        out: List[RankedNode] = []
        for node in self.graph.nodes_at(level):
            lvl, score = self.rollup(node.id)
            out.append((node, lvl, score))
        return out

    def weakest(self, n: int, level: NodeLevel = NodeLevel.TOPIC) -> List[RankedNode]:
        """Bottom-N nodes. UNKNOWN (-1) sorts below BEGINNER (0)."""
        if isinstance(level, str):
            level = NodeLevel(level)
        if n < 0:
            raise ValueError("n must be >= 0")

        def key(item: RankedNode) -> tuple:
            node, lvl, score = item
            return (-1.0 if score is None else score, node.id)

        return sorted(self._ranked(level), key=key)[:n]

    def strongest(self, n: int, level: NodeLevel = NodeLevel.TOPIC) -> List[RankedNode]:
        """Top-N nodes. UNKNOWN nodes are excluded."""
        if n < 0:
            raise ValueError("n must be >= 0")

        def key(item: RankedNode) -> tuple:
            node, _lvl, score = item
            return (-score, node.id)  # score is never None here

        ranked = [r for r in self._ranked(level) if r[2] is not None]
        return sorted(ranked, key=key)[:n]

    # -------------------------------------------------------------- next up

    def _meets_gate(self, topic_id: str) -> bool:
        params = self.engine.params_for(self.learner_id, topic_id)
        gate = MASTERY_SCORES[params.prereq_gate]
        for prereq in self.graph.prerequisites.get(topic_id, ()):
            level, _ = self.rollup(prereq)
            score = MASTERY_SCORES.get(level, -1)
            if score < gate:
                return False
        return True

    def next_topic(self) -> Optional[KnowledgeNode]:
        """First topic in plan order that still needs work (not ADVANCED by
        rollup) and whose prerequisites all clear the gate."""
        plan = self.store.get_plan_for_learner(self.learner_id)
        if plan is None:
            return None
        for tid in plan.topic_ids:
            level, _ = self.rollup(tid)
            if level == MasteryLevel.ADVANCED:
                continue
            if not self._meets_gate(tid):
                continue
            try:
                return self.graph.get(tid)
            except KeyError:
                raise ValueError(f"plan references unknown node: {tid}") from None
        return None
