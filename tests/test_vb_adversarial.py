"""ADVERSARIAL: invalid inputs must fail loudly, never silently corrupt.

Every case here either raises a clear error or is asserted as deliberate,
documented behavior. Nothing may be silently swallowed or fabricate state.
"""

import pytest

from edupaal import Evidence, LearnerPreferences, MasteryLevel

from .vb_helpers import BASE, submit_all, vb_evidence, vb_skill

NODE = "linear-equations"


def test_unknown_node_ids_fail_loudly(tmp_path):
    skill = vb_skill(tmp_path, learner_id="adv-unknown")
    with pytest.raises(ValueError, match="unknown node"):
        skill.record_evidence(vb_evidence("no-such-node", 0.9, learner_id="adv-unknown", ev_id="au1"))
    with pytest.raises(ValueError, match="unknown node"):
        skill.assert_mastery("no-such-node", MasteryLevel.BEGINNER, asserted_by="x", reason="y")
    with pytest.raises(ValueError, match="unknown node"):
        skill.grounding_packet("no-such-node")
    with pytest.raises(ValueError, match="unknown node"):
        skill.effective_mastery("no-such-node")
    with pytest.raises(ValueError, match="unknown node"):
        skill.set_override(MasteryLevel.BEGINNER, reason="x", scope_node_id="no-such-node")
    # unknown node must NOT masquerade as "unevaluated"
    with pytest.raises(ValueError):
        skill.effective_mastery("typo-topci")


def test_evidence_validation(tmp_path):
    skill = vb_skill(tmp_path, learner_id="adv-ev")
    with pytest.raises(ValueError):
        vb_evidence(NODE, -0.1, learner_id="adv-ev", ev_id="ae1")
    with pytest.raises(ValueError):
        vb_evidence(NODE, 1.1, learner_id="adv-ev", ev_id="ae2")
    with pytest.raises(ValueError):
        Evidence(id="ae3", learner_id="", node_id=NODE, source_agent="a",
                 activity_type="quiz", occurred_at=BASE, performance=0.5)
    with pytest.raises(ValueError):
        Evidence(id="ae4", learner_id="adv-ev", node_id="", source_agent="a",
                 activity_type="quiz", occurred_at=BASE, performance=0.5)
    with pytest.raises(ValueError):
        Evidence(id="ae5", learner_id="adv-ev", node_id=NODE, source_agent="",
                 activity_type="quiz", occurred_at=BASE, performance=0.5)
    with pytest.raises(ValueError):
        Evidence(id="ae6", learner_id="adv-ev", node_id=NODE, source_agent="a",
                 activity_type="", occurred_at=BASE, performance=0.5)
    with pytest.raises(ValueError):
        Evidence(id="ae7", learner_id="adv-ev", node_id=NODE, source_agent="a",
                 activity_type="quiz", occurred_at=BASE, performance=0.5, confidence=2.0)
    # wrong-learner evidence is rejected at the skill boundary
    with pytest.raises(ValueError, match="!="):
        skill.record_evidence(vb_evidence(NODE, 0.5, learner_id="someone-else", ev_id="ae9"))


def test_duplicate_ids_fail_loudly_not_silently(tmp_path):
    """Append-only history: a duplicate id is a caller bug and must raise —
    never silently replace the existing row."""
    skill = vb_skill(tmp_path, learner_id="adv-dup")
    skill.record_evidence(vb_evidence(NODE, 0.9, learner_id="adv-dup", ev_id="dup1"))
    with pytest.raises(ValueError, match="duplicate evidence id"):
        skill.record_evidence(vb_evidence(NODE, 0.8, learner_id="adv-dup", ev_id="dup1"))
    # the original row is untouched
    assert skill.store.get_evidence("dup1").performance == 0.9

    from edupaal import MasteryRecord

    rec = skill.mastery_history(NODE)[0]
    with pytest.raises(ValueError, match="duplicate mastery record id"):
        skill.store.save_mastery_record(
            MasteryRecord(
                id=rec.id, node_id=rec.node_id, learner_id=rec.learner_id,
                level=rec.level, updated_at=rec.updated_at,
                rule_version=rec.rule_version,
                params_in_effect=rec.params_in_effect, evidence_ids=rec.evidence_ids,
            )
        )


def test_cyclic_prerequisites_rejected_and_graph_stays_usable(tmp_path):
    from edupaal import KnowledgeGraph, KnowledgeNode, NodeLevel

    g = KnowledgeGraph()
    g.add_node(KnowledgeNode(id="s", level=NodeLevel.SPACE, name="s"))
    g.add_node(KnowledgeNode(id="subj", level=NodeLevel.SUBJECT, name="subj", parent_id="s"))
    g.add_node(KnowledgeNode(id="c", level=NodeLevel.CONCEPT, name="c", parent_id="subj"))
    for t in ("a", "b", "c1", "d"):
        g.add_node(KnowledgeNode(id=t, level=NodeLevel.TOPIC, name=t, parent_id="c"))
    g.add_prerequisite("b", "a")
    g.add_prerequisite("c1", "b")
    with pytest.raises(ValueError, match="cycle"):
        g.add_prerequisite("a", "c1")  # would close a->b->c1->a
    with pytest.raises(ValueError, match="own prerequisite"):
        g.add_prerequisite("d", "d")
    # the rejected edge left no trace; the graph is still a valid DAG
    assert g.prerequisites["a"] == set()
    assert g.prerequisites["c1"] == {"b"}
    g.add_prerequisite("d", "c1")  # still usable
    assert g.prerequisites["d"] == {"c1"}


def test_ten_level_subtopic_chain(tmp_path):
    """Recursive decomposition to arbitrary depth: evidence on the deepest
    leaf rolls up the entire chain."""
    from edupaal import EduPAALSkill, KnowledgeGraph, KnowledgeNode, NodeLevel, SQLiteBackend

    g = KnowledgeGraph()
    g.add_node(KnowledgeNode(id="s", level=NodeLevel.SPACE, name="s"))
    g.add_node(KnowledgeNode(id="subj", level=NodeLevel.SUBJECT, name="subj", parent_id="s"))
    g.add_node(KnowledgeNode(id="c", level=NodeLevel.CONCEPT, name="c", parent_id="subj"))
    parent = "c"
    for i in range(10):
        tid = f"deep{i}"
        g.add_node(KnowledgeNode(id=tid, level=NodeLevel.TOPIC, name=tid, parent_id=parent))
        parent = tid
    leaf = "deep9"
    store = SQLiteBackend(tmp_path / "deep.db")
    skill = EduPAALSkill(store, g)
    skill.cold_start(
        learner_id="adv-deep",
        node_selection=[leaf],
        preferences=LearnerPreferences(learning_style="visual", pace="steady"),
    )
    submit_all(skill, [
        vb_evidence(leaf, 0.92, "quiz", "q", "adv-deep", day=0, ev_id="deep1"),
        vb_evidence(leaf, 0.93, "practice", "p", "adv-deep", day=1, ev_id="deep2"),
        vb_evidence(leaf, 0.94, "dialogue", "d", "adv-deep", day=2, ev_id="deep3"),
    ])
    assert skill.effective_mastery(leaf) == MasteryLevel.ADVANCED
    # every ancestor up the 10-deep chain reports ADVANCED via rollup
    for i in range(10):
        assert skill.effective_mastery(f"deep{i}") == MasteryLevel.ADVANCED, f"deep{i}"
    assert skill.effective_mastery("c") == MasteryLevel.ADVANCED
    assert skill.effective_mastery("subj") == MasteryLevel.ADVANCED
    assert skill.effective_mastery("s") == MasteryLevel.ADVANCED
    packet = skill.grounding_packet(leaf)
    assert len(packet["mastery_context"]["ancestors"]) == 12  # 9 parent topics + concept + subject + space


def test_future_timestamp_shifts_recency_anchor(tmp_path):
    """Characterized (deliberate): the W-day window anchors at the latest
    evidence timestamp, so a future-dated evidence shifts the anchor and can
    strand older evidence outside the window. Timestamps are the reporter's
    responsibility; the behavior is deterministic and self-healing — once K
    evidences land inside the new window, promotion proceeds."""
    skill = vb_skill(tmp_path, learner_id="adv-future")
    submit_all(skill, [
        vb_evidence(NODE, 0.95, "quiz", "q", "adv-future", day=0, ev_id="fu1"),
        vb_evidence(NODE, 0.94, "quiz", "q", "adv-future", day=1, ev_id="fu2"),
    ])
    # a buggy vertical reports with a clock 60 days in the future
    skill.record_evidence(vb_evidence(NODE, 0.96, "quiz", "q", "adv-future", day=61, ev_id="fu3"))
    # anchor is now day 61; window [31, 61] holds only fu3 (< K) -> no promotion
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER
    # self-healing: K evidences near the new anchor promote normally
    submit_all(skill, [
        vb_evidence(NODE, 0.95, "quiz", "q", "adv-future", day=62, ev_id="fu4"),
        vb_evidence(NODE, 0.96, "practice", "p", "adv-future", day=63, ev_id="fu5"),
    ])
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
    # history stayed sane and ordered throughout
    stamps = [r.updated_at for r in skill.mastery_history(NODE)]
    assert stamps == sorted(stamps)


def test_non_leaf_and_non_topic_targets_rejected(tmp_path):
    skill = vb_skill(tmp_path, learner_id="adv-leaf")
    with pytest.raises(ValueError, match="finest-grained"):
        skill.record_evidence(vb_evidence("regularization", 0.9, learner_id="adv-leaf", ev_id="nl1"))
    with pytest.raises(ValueError, match="TOPIC"):
        skill.record_evidence(vb_evidence("algebra", 0.9, learner_id="adv-leaf", ev_id="nl2"))
    with pytest.raises(ValueError, match="UNKNOWN"):
        skill.assert_mastery(NODE, MasteryLevel.UNKNOWN, asserted_by="x", reason="y")
    with pytest.raises(ValueError, match="TOPIC nodes"):
        skill.assert_mastery("algebra", MasteryLevel.BEGINNER, asserted_by="x", reason="y")
    with pytest.raises(ValueError, match="UNKNOWN"):
        skill.set_override(MasteryLevel.UNKNOWN, reason="x", scope_node_id=NODE)


def test_cold_start_validation(tmp_path):
    from edupaal import EduPAALSkill, SQLiteBackend, build_seed_graph

    graph = build_seed_graph()
    store = SQLiteBackend(tmp_path / "cs.db")
    skill = EduPAALSkill(store, graph)
    prefs = LearnerPreferences(learning_style="visual", pace="steady")
    with pytest.raises(ValueError, match="at least one topic"):
        skill.cold_start(learner_id="x", node_selection=[], preferences=prefs)
    with pytest.raises(ValueError, match="unknown node"):
        skill.cold_start(learner_id="x", node_selection=["nope"], preferences=prefs)
    with pytest.raises(ValueError, match="TOPIC nodes only"):
        skill.cold_start(learner_id="x", node_selection=["algebra"], preferences=prefs)
    with pytest.raises(ValueError, match="permutation"):
        skill.cold_start(learner_id="x", node_selection=["linear-equations", "linearization"],
                         preferences=prefs, order=["linear-equations"])
    # no learner bound: reads fail loudly
    with pytest.raises(ValueError, match="cold_start"):
        skill.next_topic()
    with pytest.raises(ValueError, match="cold_start"):
        skill.grounding_packet(NODE)


def test_ranking_rejects_negative_n(tmp_path):
    from edupaal import NodeLevel

    skill = vb_skill(tmp_path, learner_id="adv-n")
    with pytest.raises(ValueError):
        skill.weakest(-1)
    with pytest.raises(ValueError):
        skill.strongest(-5, NodeLevel.CONCEPT)
