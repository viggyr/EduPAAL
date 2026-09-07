"""ROLLUP PROPERTY: random trees vs the independent oracle.

The documented rollup contract: a parent's level is the mean of its
evaluated children's scores; UNKNOWN children are excluded; a node with no
evaluated children is UNKNOWN; the level nearest the mean wins, with exact
ties (0.5, 1.5) rounding DOWN. Overrides short-circuit the rollup for the
overridden node only.
"""

import random

from edupaal import MasteryLevel, NodeLevel

from .vb_helpers import (
    independent_incremental_level,
    independent_rollup,
    leaf_topics,
    random_topic_graph,
    submit_all,
    vb_evidence,
    vb_skill,
)


def test_rollup_matches_oracle_on_random_trees(tmp_path):
    """For every node in random trees: engine rollup == independent oracle."""
    for seed in range(8):
        rng = random.Random(50_000 + seed)
        graph = random_topic_graph(rng, n_leaf_topics=60, n_prereqs=0, max_depth=4)
        leaves = leaf_topics(graph)

        from edupaal import EduPAALSkill, LearnerPreferences, SQLiteBackend

        tag = f"t{seed}"
        store = SQLiteBackend(tmp_path / f"rollup-{tag}.db")
        skill = EduPAALSkill(store, graph)
        learner_id = f"rollup-{tag}"
        skill.cold_start(
            learner_id=learner_id,
            node_selection=leaves[:20],
            preferences=LearnerPreferences(learning_style="visual", pace="steady"),
        )
        ev_rng = random.Random(60_000 + seed)
        evidence_by_node = {}
        for i in range(80):
            node = ev_rng.choice(leaves)
            e = vb_evidence(
                node, round(ev_rng.uniform(0.0, 1.0), 2),
                ev_rng.choice(["quiz", "practice", "dialogue"]),
                f"agent-{ev_rng.randint(1, 3)}", learner_id,
                day=ev_rng.randint(0, 20), ev_id=f"{tag}-{i}",
            )
            evidence_by_node.setdefault(node, []).append(e)
            skill.record_evidence(e)

        leaf_levels = {
            # arrival order per leaf is the submission loop order: the model
            # replays exactly what the engine saw
            leaf: independent_incremental_level(evidence_by_node.get(leaf, []))
            for leaf in leaves
        }
        oracle = independent_rollup(graph, leaf_levels)
        for node_id, node in graph.nodes.items():
            if node.level == NodeLevel.SPACE:
                continue
            expected = oracle[node_id][0]
            got = skill.effective_mastery(node_id)
            assert got == expected, (
                f"seed {seed} node {node_id} ({node.level.value}): "
                f"engine={got.value} oracle={expected.value}"
            )


def test_unknown_leaves_excluded_not_zeroed(tmp_path):
    """UNKNOWN children must be *excluded* from the parent mean, not scored
    as zero: one ADVANCED leaf among unknown siblings rolls up ADVANCED."""
    skill = vb_skill(tmp_path, learner_id="rollup-excl")
    submit_all(skill, [
        vb_evidence("linear-equations", 0.95, "quiz", "q", "rollup-excl", day=0, ev_id="re1"),
        vb_evidence("linear-equations", 0.96, "practice", "p", "rollup-excl", day=1, ev_id="re2"),
        vb_evidence("linear-equations", 0.97, "dialogue", "d", "rollup-excl", day=2, ev_id="re3"),
    ])
    assert skill.effective_mastery("linear-equations") == MasteryLevel.ADVANCED
    # dropout-rate untouched (UNKNOWN) -> excluded, not averaged as 0
    assert skill.effective_mastery("dropout-rate") == MasteryLevel.UNKNOWN
    assert skill.effective_mastery("algebra") == MasteryLevel.ADVANCED
    # all-unknown children -> parent UNKNOWN (not BEGINNER, not zero):
    # the ml subtree is untouched
    assert skill.effective_mastery("overfitting") == MasteryLevel.UNKNOWN
    assert skill.effective_mastery("ml") == MasteryLevel.UNKNOWN


def test_ties_round_down(tmp_path):
    """A parent mean exactly halfway between two levels (1.5) rounds DOWN
    to the lower level — it must not sneak up to ADVANCED."""
    skill = vb_skill(tmp_path, learner_id="rollup-tie")
    submit_all(skill, [
        vb_evidence("linear-equations", 0.92, "quiz", "q", "rollup-tie", day=0, ev_id="rt1"),
        vb_evidence("linear-equations", 0.93, "practice", "p", "rollup-tie", day=1, ev_id="rt2"),
        vb_evidence("linear-equations", 0.94, "dialogue", "d", "rollup-tie", day=2, ev_id="rt3"),
        vb_evidence("linearization", 0.70, "quiz", "q", "rollup-tie", day=0, ev_id="rt4"),
        vb_evidence("linearization", 0.70, "quiz", "q", "rollup-tie", day=1, ev_id="rt5"),
        vb_evidence("linearization", 0.70, "quiz", "q", "rollup-tie", day=2, ev_id="rt6"),
    ])
    assert skill.effective_mastery("linear-equations") == MasteryLevel.ADVANCED   # score 2.0
    assert skill.effective_mastery("linearization") == MasteryLevel.INTERMEDIATE  # score 1.0
    # mean 1.5 -> tie -> INTERMEDIATE (down), never ADVANCED
    assert skill.effective_mastery("algebra") == MasteryLevel.INTERMEDIATE


def test_rollup_weighting_is_by_child_not_evidence_count(tmp_path):
    """Each child contributes one score (its own level) regardless of how
    much evidence it has: 10 evidences on one leaf don't outweigh 3 on
    another."""
    skill = vb_skill(tmp_path, learner_id="rollup-weight")
    for i in range(10):
        skill.record_evidence(
            vb_evidence("linear-equations", 0.95, "quiz", "q", "rollup-weight",
                        day=i, ev_id=f"rw-a{i}")
        )
    submit_all(skill, [
        vb_evidence("linearization", 0.70, "quiz", "q", "rollup-weight", day=0, ev_id="rw-b1"),
        vb_evidence("linearization", 0.70, "quiz", "q", "rollup-weight", day=1, ev_id="rw-b2"),
        vb_evidence("linearization", 0.70, "quiz", "q", "rollup-weight", day=2, ev_id="rw-b3"),
    ])
    # children: linear-equations ADVANCED (2.0), linearization INTERMEDIATE (1.0)
    # mean 1.5 -> tie rounds DOWN to INTERMEDIATE (not ADVANCED by volume)
    assert skill.effective_mastery("algebra") == MasteryLevel.INTERMEDIATE


def test_override_short_circuits_only_its_node(tmp_path):
    """An override on a parent wins for that parent, but the children's own
    rollups are unaffected."""
    skill = vb_skill(tmp_path, learner_id="rollup-ovr")
    submit_all(skill, [
        vb_evidence("linear-equations", 0.70, "quiz", "q", "rollup-ovr", day=0, ev_id="ro1"),
        vb_evidence("linear-equations", 0.70, "quiz", "q", "rollup-ovr", day=1, ev_id="ro2"),
        vb_evidence("linear-equations", 0.70, "quiz", "q", "rollup-ovr", day=2, ev_id="ro3"),
    ])
    skill.set_override(MasteryLevel.ADVANCED, reason="demo", scope_node_id="algebra")
    assert skill.effective_mastery("algebra") == MasteryLevel.ADVANCED
    assert skill.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE
    # the grandparent sees the parent's *effective* (overridden) level
    assert skill.effective_mastery("maths") == MasteryLevel.ADVANCED
    skill.clear_override("algebra")
    assert skill.effective_mastery("algebra") == MasteryLevel.INTERMEDIATE


def test_assertion_on_leaf_flows_through_rollup(tmp_path):
    skill = vb_skill(tmp_path, learner_id="rollup-assert")
    skill.assert_mastery("linear-equations", MasteryLevel.ADVANCED,
                         asserted_by="exam-board", reason="final")
    assert skill.effective_mastery("algebra") == MasteryLevel.ADVANCED
