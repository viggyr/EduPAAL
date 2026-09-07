"""Shared helpers for the EduPAAL validation battery (tests/test_vb_*.py).

``independent_incremental_level`` and ``independent_rollup`` re-implement the
documented rules from scratch (not by calling engine internals) so the
battery compares implementation against specification.
"""

from __future__ import annotations

import itertools
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from edupaal import (
    EduPAALSkill,
    Evidence,
    KnowledgeGraph,
    KnowledgeNode,
    LearnerPreferences,
    MasteryLevel,
    MasteryParams,
    NodeLevel,
    SQLiteBackend,
    build_seed_graph,
)
from edupaal.entities import MASTERY_SCORES

BASE = datetime(2026, 9, 1, tzinfo=timezone.utc)

_counter = itertools.count()


def vb_evidence(
    node_id: str,
    performance: float,
    activity_type: str = "quiz",
    source_agent: str = "quiz-agent",
    learner_id: str = "vb-learner",
    day: int = 0,
    ev_id: Optional[str] = None,
) -> Evidence:
    return Evidence(
        id=ev_id or f"vb-{next(_counter)}",
        learner_id=learner_id,
        node_id=node_id,
        source_agent=source_agent,
        activity_type=activity_type,
        occurred_at=BASE + timedelta(days=day),
        performance=performance,
    )


def vb_skill(
    tmp_path,
    learner_id: str = "vb-learner",
    topics: Optional[List[str]] = None,
    params: Optional[MasteryParams] = None,
) -> EduPAALSkill:
    graph = build_seed_graph()
    store = SQLiteBackend(tmp_path / f"{learner_id}.db")
    skill = EduPAALSkill(store, graph, default_params=params)
    skill.cold_start(
        learner_id=learner_id,
        node_selection=topics or ["linear-equations", "linearization", "dropout-rate"],
        preferences=LearnerPreferences(learning_style="visual", pace="steady"),
    )
    return skill


def submit_all(skill: EduPAALSkill, evidences: List[Evidence]) -> None:
    for e in evidences:
        skill.record_evidence(e)


def independent_incremental_level(
    evidences_in_arrival_order: List[Evidence],
    params: Optional[MasteryParams] = None,
) -> MasteryLevel:
    """Independent re-implementation of heuristic-v1's *incremental* rule.

    The engine is a ratchet, not a batch fixed point: each evidence
    submission is evaluated against everything seen so far, and levels
    never step down. So the model replays the arrival sequence: first
    evidence -> BEGINNER, then after every arrival repeatedly evaluate the
    K most recent evidences inside the W-day window anchored at the latest
    *seen* timestamp (need >= K items, mean >= bar, no item below
    t_contradict, and for ADVANCED >= 2 distinct activity types when
    cross_modal_advanced is on).

    NOTE: a pure set fixed point is NOT a valid model of this engine: an
    early clean subset can promote and later diluting evidence cannot
    demote (demotion is deliberately out of scope). The model must replay
    the arrival order.
    """
    params = params or MasteryParams()
    level = MasteryLevel.UNKNOWN
    seen: List[Evidence] = []
    for e in evidences_in_arrival_order:
        seen.append(e)
        if level == MasteryLevel.UNKNOWN:
            level = MasteryLevel.BEGINNER
        while level != MasteryLevel.ADVANCED:
            ordered = sorted(seen, key=lambda x: (x.occurred_at, x.id))
            anchor = ordered[-1].occurred_at
            cutoff = anchor - timedelta(days=params.window_days)
            windowed = [x for x in ordered if x.occurred_at >= cutoff]
            cands = windowed[-params.k_evidence :]
            if len(cands) < params.k_evidence:
                break
            if level == MasteryLevel.BEGINNER:
                target, bar = MasteryLevel.INTERMEDIATE, params.t_intermediate
            else:
                target, bar = MasteryLevel.ADVANCED, params.t_advanced
            mean = sum(x.performance for x in cands) / len(cands)
            if mean < bar:
                break
            if any(x.performance < params.t_contradict for x in cands):
                break
            if (
                target == MasteryLevel.ADVANCED
                and params.cross_modal_advanced
                and len({x.activity_type for x in cands}) < 2
            ):
                break
            level = target
    return level


def independent_rollup(
    graph: KnowledgeGraph, leaf_levels: Dict[str, MasteryLevel]
) -> Dict[str, Tuple[MasteryLevel, Optional[float]]]:
    """Independent re-implementation of the documented rollup.

    Leaves report their own level (UNKNOWN excluded from aggregation);
    internal nodes report the level nearest the mean of evaluated
    children's scores, ties rounding down; no evaluated children ->
    UNKNOWN. The tie-down boundaries sit exactly halfway between scores
    (0.5, 1.5) — written here as thresholds rather than copying the
    engine's nearest-match expression, so the oracle is genuinely
    independent.
    """
    result: Dict[str, Tuple[MasteryLevel, Optional[float]]] = {}

    def score_of(level: MasteryLevel) -> Optional[float]:
        if level == MasteryLevel.UNKNOWN:
            return None
        return float(MASTERY_SCORES[level])

    def visit(node_id: str) -> Tuple[MasteryLevel, Optional[float]]:
        if node_id in result:
            return result[node_id]
        children = graph.children(node_id)
        if not children:
            lvl = leaf_levels.get(node_id, MasteryLevel.UNKNOWN)
            result[node_id] = (lvl, score_of(lvl))
            return result[node_id]
        scores = []
        for ch in children:
            _, s = visit(ch.id)
            if s is not None:
                scores.append(s)
        if not scores:
            result[node_id] = (MasteryLevel.UNKNOWN, None)
            return result[node_id]
        mean = sum(scores) / len(scores)
        # nearest level to the mean; ties (exactly 0.5 / 1.5) go down
        if mean <= 0.5:
            nearest = MasteryLevel.BEGINNER
        elif mean <= 1.5:
            nearest = MasteryLevel.INTERMEDIATE
        else:
            nearest = MasteryLevel.ADVANCED
        result[node_id] = (nearest, mean)
        return result[node_id]

    for node in graph.nodes.values():
        if node.parent_id is None:
            visit(node.id)
    return result


def random_topic_graph(
    rng: random.Random,
    n_leaf_topics: int,
    n_prereqs: int,
    max_depth: int = 2,
) -> KnowledgeGraph:
    """Build a random topic DAG: 1 space / 2 subjects / 2 concepts each,
    topics spread across concepts, some decomposed to ``max_depth``.
    Prerequisites are random topic-topic edges; cycles are rejected so the
    result is always a DAG.
    """
    g = KnowledgeGraph()
    g.add_node(KnowledgeNode(id="space", level=NodeLevel.SPACE, name="Space"))
    concepts = []
    ci = 0
    for si in range(2):
        sid = f"subject{si}"
        g.add_node(KnowledgeNode(id=sid, level=NodeLevel.SUBJECT, name=f"Subject {si}",
                                 parent_id="space"))
        for _ in range(2):
            cid = f"concept{ci}"
            g.add_node(KnowledgeNode(id=cid, level=NodeLevel.CONCEPT,
                                     name=f"Concept {ci}", parent_id=sid))
            concepts.append(cid)
            ci += 1

    topic_ids: List[str] = []
    leaves: List[str] = []
    t = 0

    def add_subtree(parent_id: str, depth: int) -> None:
        nonlocal t
        tid = f"t{t}"
        t += 1
        g.add_node(KnowledgeNode(id=tid, level=NodeLevel.TOPIC, name=f"Topic {tid}",
                                 parent_id=parent_id))
        topic_ids.append(tid)
        if depth < max_depth and rng.random() < 0.35:
            for _ in range(rng.randint(2, 3)):
                add_subtree(tid, depth + 1)
        else:
            leaves.append(tid)

    while len(leaves) < n_leaf_topics:
        add_subtree(rng.choice(concepts), 1)
        if t > n_leaf_topics * 6:  # safety valve; keeps the loop finite
            break

    added = 0
    attempts = 0
    while added < n_prereqs and attempts < n_prereqs * 20:
        attempts += 1
        a, b = rng.sample(topic_ids, 2)
        try:
            g.add_prerequisite(b, a)
            added += 1
        except ValueError:
            continue
    return g


def leaf_topics(graph: KnowledgeGraph) -> List[str]:
    return [
        n.id
        for n in graph.nodes.values()
        if n.level == NodeLevel.TOPIC and not graph.children(n.id)
    ]
