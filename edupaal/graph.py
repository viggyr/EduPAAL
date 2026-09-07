"""The EduPAAL knowledge graph (framework layer).

Four fixed upper levels: Space -> Subject -> Concept -> Topic. Topics are
recursively decomposable: a TOPIC node may have child TOPIC nodes
(sub-topics) to arbitrary depth.

Two edge types:
  * part_of      — topic -> concept -> subject -> space, plus
                   sub-topic -> ... -> topic for decomposed topics
  * prerequisite — topic -> topic at any depth; may cross concepts/subjects.

The graph is content: deployments supply their own via ``add_node`` /
``add_prerequisite`` (or ``edupaal.seed`` for the exemplar). The framework
only enforces structural validity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .entities import KnowledgeNode, NodeLevel

_LEVEL_ORDER = [NodeLevel.SPACE, NodeLevel.SUBJECT, NodeLevel.CONCEPT, NodeLevel.TOPIC]


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.prerequisites: Dict[str, Set[str]] = {}  # topic_id -> {prereq topic_ids}

    # ------------------------------------------------------------------ build

    def add_node(self, node: KnowledgeNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        if node.parent_id is not None:
            parent = self.nodes.get(node.parent_id)
            if parent is None:
                raise ValueError(
                    f"node {node.id}: parent {node.parent_id} does not exist"
                )
            if node.level == NodeLevel.TOPIC:
                # topics hang under a concept, or under another topic (sub-topic)
                allowed = (NodeLevel.CONCEPT, NodeLevel.TOPIC)
            else:
                allowed = (_LEVEL_ORDER[_LEVEL_ORDER.index(node.level) - 1],)
            if parent.level not in allowed:
                names = "/".join(a.value for a in allowed)
                raise ValueError(
                    f"node {node.id}: {node.level.value} must hang under "
                    f"{names}, not {parent.level.value}"
                )
        self.nodes[node.id] = node
        self.prerequisites.setdefault(node.id, set())

    def add_prerequisite(self, topic_id: str, prereq_id: str) -> None:
        """Declare that ``topic_id`` requires ``prereq_id`` first.

        Both must be TOPIC nodes at any depth (sub-topics included); the edge
        may cross concepts and subjects. Cycles are rejected.
        """
        for nid in (topic_id, prereq_id):
            node = self.nodes.get(nid)
            if node is None:
                raise ValueError(f"prerequisite references unknown node: {nid}")
            if node.level != NodeLevel.TOPIC:
                raise ValueError(
                    f"prerequisites only link TOPIC nodes ({nid} is {node.level.value})"
                )
        if topic_id == prereq_id:
            raise ValueError("a topic cannot be its own prerequisite")
        self.prerequisites.setdefault(topic_id, set()).add(prereq_id)
        if self._has_cycle():
            self.prerequisites[topic_id].remove(prereq_id)
            raise ValueError(
                f"prerequisite {prereq_id} -> {topic_id} would create a cycle"
            )

    def _has_cycle(self) -> bool:
        visiting: Set[str] = set()
        done: Set[str] = set()

        def visit(n: str) -> bool:
            if n in done:
                return False
            if n in visiting:
                return True
            visiting.add(n)
            for p in self.prerequisites.get(n, ()):  # noqa: B023 - fine, sync loop
                if visit(p):
                    return True
            visiting.remove(n)
            done.add(n)
            return False

        return any(visit(n) for n in self.prerequisites)

    # -------------------------------------------------------------- traversal

    def get(self, node_id: str) -> KnowledgeNode:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown node: {node_id}") from None

    def children(self, node_id: str) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def ancestors(self, node_id: str) -> List[KnowledgeNode]:
        out: List[KnowledgeNode] = []
        node = self.get(node_id)
        while node.parent_id is not None:
            node = self.get(node.parent_id)
            out.append(node)
        return out

    def descendants(self, node_id: str) -> List[KnowledgeNode]:
        out: List[KnowledgeNode] = []
        stack = self.children(node_id)
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(self.children(node.id))
        return out

    def topics_under(self, node_id: str) -> List[KnowledgeNode]:
        """All TOPIC nodes in the subtree, including ``node_id`` itself when
        it is a topic (topics may be decomposed into sub-topics)."""
        node = self.get(node_id)
        result = [node] if node.level == NodeLevel.TOPIC else []
        result.extend(
            n for n in self.descendants(node_id) if n.level == NodeLevel.TOPIC
        )
        return result

    def nodes_at(self, level: NodeLevel) -> List[KnowledgeNode]:
        if isinstance(level, str):
            level = NodeLevel(level)
        return [n for n in self.nodes.values() if n.level == level]

    def validate_plan_topics(self, topic_ids: Iterable[str]) -> None:
        for tid in topic_ids:
            node = self.nodes.get(tid)
            if node is None:
                raise ValueError(f"plan references unknown node: {tid}")
            if node.level != NodeLevel.TOPIC:
                raise ValueError(
                    f"plan traversals list TOPIC nodes only ({tid} is {node.level.value})"
                )

    def __len__(self) -> int:
        return len(self.nodes)
