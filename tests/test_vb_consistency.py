"""CONSISTENCY: grounding fidelity and retrieval invariants.

The hypothesis requires that grounding packets always reflect the true
current state (no stale reads across verticals or skill instances) and that
next_topic never returns a mastered topic or one whose prerequisites are
unmet — on small graphs and on random ~200-node DAGs.
"""

import random

from edupaal import MasteryLevel, NodeLevel

from .vb_helpers import (
    leaf_topics,
    random_topic_graph,
    submit_all,
    vb_evidence,
    vb_skill,
)


def _packet_state(skill, node_id):
    p = skill.grounding_packet(node_id)
    return (
        p["mastery_context"]["node"]["mastery"],
        p["next_topic"]["id"] if p["next_topic"] else None,
        p["plan"]["advanced_topics"],
        p["plan"]["version"],
        p["active_override"]["level"] if p["active_override"] else None,
    )


def test_grounding_packet_fresh_after_every_write(tmp_path):
    """After every mutation, the packet must match direct engine reads."""
    skill = vb_skill(tmp_path, learner_id="cons-fresh")
    ops = [
        ("ev", ("linear-equations", 0.90, "quiz", "quiz-agent", 0)),
        ("ev", ("linear-equations", 0.85, "quiz", "quiz-agent", 1)),
        ("ev", ("linear-equations", 0.92, "practice", "practice-agent", 2)),
        ("ev", ("linearization", 0.70, "dialogue", "tutor-agent", 3)),
        ("override", (MasteryLevel.ADVANCED, "linearization")),
        ("ev", ("linearization", 0.20, "quiz", "quiz-agent", 4)),
        ("clear", ("linearization",)),
    ]
    n = 0
    for kind, args in ops:
        if kind == "ev":
            node, perf, act, agent, day = args
            n += 1
            skill.record_evidence(
                vb_evidence(node, perf, act, agent, "cons-fresh", day=day, ev_id=f"cf{n}")
            )
        elif kind == "override":
            level, node = args
            skill.set_override(level, reason="consistency check", scope_node_id=node)
        else:
            (node,) = args
            skill.clear_override(node)
        for check_node in ("linear-equations", "linearization", "dropout-rate"):
            mastery, nxt, adv, ver, ovr = _packet_state(skill, check_node)
            assert mastery == skill.effective_mastery(check_node).value
            direct_next = skill.next_topic()
            assert nxt == (direct_next.id if direct_next else None)
            direct_adv = sum(
                1
                for tid in skill.store.get_plan_for_learner("cons-fresh").topic_ids
                if skill.effective_mastery(tid) == MasteryLevel.ADVANCED
            )
            assert adv == direct_adv
            direct_ovr = skill.active_override(check_node)
            assert ovr == (direct_ovr.level.value if direct_ovr else None)


def test_two_skill_instances_share_one_memory(tmp_path):
    """Two skill instances over one store = two vertical processes: a write
    through one is immediately visible through the other."""
    from edupaal import EduPAALSkill, LearnerPreferences, SQLiteBackend, build_seed_graph

    graph = build_seed_graph()
    store = SQLiteBackend(tmp_path / "shared.db")
    prefs = LearnerPreferences(learning_style="visual", pace="steady")
    topics = ["linear-equations", "linearization"]
    guide = EduPAALSkill(store, graph)
    guide.cold_start(learner_id="cons-shared", node_selection=topics, preferences=prefs)
    evaluator = EduPAALSkill(store, graph)
    # second vertical binds the same learner (re-cold-start bumps plan version)
    evaluator.cold_start(learner_id="cons-shared", node_selection=topics, preferences=prefs)

    guide.record_evidence(vb_evidence("linear-equations", 0.9, "dialogue", "guide", "cons-shared", day=0, ev_id="sh1"))
    guide.record_evidence(vb_evidence("linear-equations", 0.9, "dialogue", "guide", "cons-shared", day=1, ev_id="sh2"))
    guide.record_evidence(vb_evidence("linear-equations", 0.9, "dialogue", "guide", "cons-shared", day=2, ev_id="sh3"))
    assert guide.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE

    # the evaluator's packet sees the guide's evidence immediately
    packet = evaluator.grounding_packet("linear-equations")
    assert packet["mastery_context"]["node"]["mastery"] == "intermediate"
    assert evaluator.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE

    evaluator.record_evidence(vb_evidence("linear-equations", 0.92, "quiz", "evaluator", "cons-shared", day=3, ev_id="sh4"))
    assert guide.effective_mastery("linear-equations") == MasteryLevel.ADVANCED
    assert guide.grounding_packet("linear-equations")["mastery_context"]["node"]["mastery"] == "advanced"


def test_next_topic_invariants_random_dags(tmp_path):
    """On random ~200-node DAGs with random evidence: next_topic is None or
    (a) not ADVANCED by rollup and (b) every prerequisite clears the gate.
    Also: identical builds give identical answers (deterministic)."""
    for seed in range(6):
        rng = random.Random(10_000 + seed)
        graph = random_topic_graph(rng, n_leaf_topics=200, n_prereqs=120)
        leaves = leaf_topics(graph)
        assert len(leaves) >= 150, f"seed {seed}: only {len(leaves)} leaves"

        def build(build_rng, learner_id, db_name):
            from edupaal import EduPAALSkill, LearnerPreferences, SQLiteBackend

            store = SQLiteBackend(tmp_path / db_name)
            skill = EduPAALSkill(store, graph)
            plan = build_rng.sample(leaves, 25)
            skill.cold_start(
                learner_id=learner_id,
                node_selection=plan,
                preferences=LearnerPreferences(learning_style="visual", pace="steady"),
            )
            return skill, plan

        skill, plan = build(rng, f"dag-{seed}", f"dag-{seed}.db")
        # random evidence on random leaves
        ev_rng = random.Random(77_000 + seed)
        for i in range(120):
            node = ev_rng.choice(leaves)
            skill.record_evidence(
                vb_evidence(
                    node,
                    round(ev_rng.uniform(0.2, 1.0), 2),
                    ev_rng.choice(["quiz", "practice", "dialogue"]),
                    f"agent-{ev_rng.randint(1, 4)}",
                    f"dag-{seed}",
                    day=ev_rng.randint(0, 25),
                    ev_id=f"dag-{seed}-{i}",
                )
            )
        nxt = skill.next_topic()
        if nxt is not None:
            level, _ = skill._retriever_for().rollup(nxt.id)
            assert level != MasteryLevel.ADVANCED, f"seed {seed}: next_topic mastered"
            params = skill.engine.params_for(f"dag-{seed}", nxt.id)
            from edupaal.entities import MASTERY_SCORES

            gate = MASTERY_SCORES[params.prereq_gate]
            for prereq in graph.prerequisites.get(nxt.id, ()):
                plvl, _ = skill._retriever_for().rollup(prereq)
                assert MASTERY_SCORES.get(plvl, -1) >= gate, (
                    f"seed {seed}: next_topic {nxt.id} has unmet prereq {prereq}"
                )
            assert nxt.id in plan

        # determinism: rebuild identically, expect the same answer
        rng2 = random.Random(10_000 + seed)
        graph2 = random_topic_graph(rng2, n_leaf_topics=200, n_prereqs=120)
        leaves2 = leaf_topics(graph2)
        assert leaves2 == leaves, f"seed {seed}: rebuild diverged"
        skill2, plan2 = build(rng2, f"dag2-{seed}", f"dag2-{seed}.db")
        assert plan2 == plan, f"seed {seed}: plan sample diverged"
        # replay the same evidence stream
        ev_rng2 = random.Random(77_000 + seed)
        for i in range(120):
            node = ev_rng2.choice(leaves2)
            skill2.record_evidence(
                vb_evidence(
                    node,
                    round(ev_rng2.uniform(0.2, 1.0), 2),
                    ev_rng2.choice(["quiz", "practice", "dialogue"]),
                    f"agent-{ev_rng2.randint(1, 4)}",
                    f"dag2-{seed}",
                    day=ev_rng2.randint(0, 25),
                    ev_id=f"dag2-{seed}-{i}",
                )
            )
        nxt2 = skill2.next_topic()
        assert (nxt.id if nxt else None) == (nxt2.id if nxt2 else None)


def test_ranking_invariants(tmp_path):
    """weakest/strongest obey their documented contracts."""
    skill = vb_skill(tmp_path, learner_id="cons-rank")
    submit_all(skill, [
        vb_evidence("linear-equations", 0.95, "quiz", "q", "cons-rank", day=0, ev_id="r1"),
        vb_evidence("linear-equations", 0.95, "practice", "p", "cons-rank", day=1, ev_id="r2"),
        vb_evidence("linear-equations", 0.95, "dialogue", "d", "cons-rank", day=2, ev_id="r3"),
        vb_evidence("linearization", 0.70, "quiz", "q", "cons-rank", day=0, ev_id="r4"),
        vb_evidence("linearization", 0.72, "quiz", "q", "cons-rank", day=1, ev_id="r5"),
        vb_evidence("linearization", 0.71, "quiz", "q", "cons-rank", day=2, ev_id="r6"),
    ])
    assert skill.effective_mastery("linear-equations") == MasteryLevel.ADVANCED
    assert skill.effective_mastery("linearization") == MasteryLevel.INTERMEDIATE

    weakest = skill.weakest(10)
    scores = [(-1.0 if s is None else s) for _, _, s in weakest]
    assert scores == sorted(scores), "weakest not in ascending order"
    assert all(lvl == MasteryLevel.UNKNOWN for _, lvl, s in weakest if s is None)

    strongest = skill.strongest(10)
    assert all(s is not None for _, _, s in strongest), "strongest must exclude UNKNOWN"
    s_scores = [s for _, _, s in strongest]
    assert s_scores == sorted(s_scores, reverse=True), "strongest not descending"
    assert strongest[0][0].id == "linear-equations"

    assert skill.weakest(0) == []
    assert skill.strongest(0) == []
    # n larger than population returns everything, deterministically ordered
    all_w = skill.weakest(10_000)
    assert len(all_w) == len(skill.graph.nodes_at(NodeLevel.TOPIC))


def test_next_topic_none_when_plan_complete(tmp_path):
    skill = vb_skill(tmp_path, learner_id="cons-done",
                     topics=["linear-equations", "dropout-rate"])
    for node in ("linear-equations", "dropout-rate"):
        submit_all(skill, [
            vb_evidence(node, 0.92, "quiz", "q", "cons-done", day=0, ev_id=f"done-{node}-1"),
            vb_evidence(node, 0.93, "practice", "p", "cons-done", day=1, ev_id=f"done-{node}-2"),
            vb_evidence(node, 0.94, "dialogue", "d", "cons-done", day=2, ev_id=f"done-{node}-3"),
        ])
        assert skill.effective_mastery(node) == MasteryLevel.ADVANCED
    assert skill.next_topic() is None
    packet = skill.grounding_packet("linear-equations")
    assert packet["next_topic"] is None
    assert packet["plan"]["advanced_topics"] == 2
