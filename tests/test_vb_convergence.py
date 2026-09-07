"""CONVERGENCE: the shared-memory hypothesis, part 1.

The core claim: evidence from many verticals about one concept compounds
into ONE coherent mastery state. The engine is an *incremental ratchet*:
each submission is evaluated against everything seen so far, and levels
never step down (demotion is deliberately out of scope). The supported
contract, tested here:

- Each promotion *evaluation* is a pure function of the evidence set:
  identical timestamps break ties by (occurred_at, id), never by arrival
  order. (This was a real bug: the K-window used to be picked by arrival
  order, so the same subset evaluated at the same moment could promote
  differently.)
- When verticals submit in non-decreasing (occurred_at, id) order — the
  normal "report as it happens" pattern — the full trajectory is a pure
  function of the set: independent of agent ids and of interleaving with
  other nodes' evidence.
- Residual order dependence is explicit and deliberate: with tied
  timestamps, arrival order is the only temporal signal the engine has,
  and an early clean subset can promote before contradicting evidence
  arrives (no demotion). Both outcomes are fully explained by their
  records. A fully order-independent final state would require batch
  recomputation with demotion — out of scope by design.
"""

import random

from edupaal import MasteryLevel, MasteryParams

from .vb_helpers import independent_incremental_level, submit_all, vb_evidence, vb_skill

NODE = "linear-equations"


def _history_levels(skill, node_id):
    return [r.level for r in skill.mastery_history(node_id)]


def test_strong_evidence_set_converges_any_arrival_order(tmp_path):
    """When EVERY K-subset of the evidence set independently supports
    promotion (mean >= t_advanced, no veto, >= 2 modalities), arrival order
    cannot matter: whichever K items arrive first chain straight to
    ADVANCED, and later arrivals are no-ops. The final mastery is
    identical across all 120 permutations; the *histories* legitimately
    differ (different windows cited), so only the final level is asserted.
    This is the precise boundary of the order-independence claim: it holds
    for uniformly supportive evidence, not in general (see the dilution
    and tied-timestamp tests for the deliberate exceptions)."""
    perfs_acts = [
        (0.90, "quiz"), (0.85, "dialogue"), (0.92, "practice"),
        (0.88, "quiz"), (0.95, "visualization"),
    ]
    # every 3-subset: mean >= 0.876 >= 0.80, min >= 0.85 >= 0.40,
    # at most 2 quizzes so >= 2 distinct modalities always
    import itertools as _it
    for combo in _it.combinations(perfs_acts, 3):
        ps = [p for p, _ in combo]
        assert sum(ps) / 3 >= 0.80 and min(ps) >= 0.40
        assert len({a for _, a in combo}) >= 2

    results = set()
    for perm_idx, order in enumerate(_it.permutations(range(5))):
        learner_id = f"conv-perm-{perm_idx}"
        skill = vb_skill(tmp_path, learner_id=learner_id)
        for j in order:
            p, act = perfs_acts[j]
            skill.record_evidence(
                vb_evidence(NODE, p, act, f"agent-{j}", learner_id,
                            day=j, ev_id=f"{learner_id}-e{j + 1}")
            )
        results.add(skill.effective_mastery(NODE))
    assert results == {MasteryLevel.ADVANCED}, results


def test_backfill_dilution_is_path_dependent_by_design(tmp_path):
    """Documented order dependence #2: the same set {3x0.95, 3x0.50}
    converges differently depending on arrival order. Canonical
    (timestamp) order promotes on the early clean triple and the later
    diluting evidence cannot demote (ratchet). Reverse arrival evaluates
    the diluting triple first and never promotes. Both are correct
    executions of the documented incremental rule."""
    def make_set(learner_id):
        evs = [
            vb_evidence(NODE, 0.95, "quiz", "q", learner_id, day=0, ev_id=f"{learner_id}-h1"),
            vb_evidence(NODE, 0.95, "practice", "p", learner_id, day=1, ev_id=f"{learner_id}-h2"),
            vb_evidence(NODE, 0.95, "dialogue", "d", learner_id, day=2, ev_id=f"{learner_id}-h3"),
            vb_evidence(NODE, 0.50, "quiz", "q", learner_id, day=3, ev_id=f"{learner_id}-l1"),
            vb_evidence(NODE, 0.50, "quiz", "q", learner_id, day=4, ev_id=f"{learner_id}-l2"),
            vb_evidence(NODE, 0.50, "quiz", "q", learner_id, day=5, ev_id=f"{learner_id}-l3"),
        ]
        return evs

    skill_a = vb_skill(tmp_path, learner_id="conv-bf-a")
    submit_all(skill_a, make_set("conv-bf-a"))  # canonical order
    assert skill_a.effective_mastery(NODE) == MasteryLevel.ADVANCED

    skill_b = vb_skill(tmp_path, learner_id="conv-bf-b")
    submit_all(skill_b, list(reversed(make_set("conv-bf-b"))))  # diluting first
    assert skill_b.effective_mastery(NODE) == MasteryLevel.BEGINNER


def test_agent_identity_does_not_change_mastery(tmp_path):
    """Who reported the evidence must not matter — only what was reported.
    source_agent is provenance, not an input to the heuristics."""
    skill_a = vb_skill(tmp_path, learner_id="conv-agent-a")
    submit_all(skill_a, [
        vb_evidence(NODE, 0.90, "quiz", "quiz-agent", "conv-agent-a", day=0, ev_id="a1"),
        vb_evidence(NODE, 0.88, "quiz", "quiz-agent", "conv-agent-a", day=1, ev_id="a2"),
        vb_evidence(NODE, 0.92, "quiz", "quiz-agent", "conv-agent-a", day=2, ev_id="a3"),
    ])
    skill_b = vb_skill(tmp_path, learner_id="conv-agent-b")
    submit_all(skill_b, [
        vb_evidence(NODE, 0.90, "quiz", "totally-different-agent", "conv-agent-b", day=0, ev_id="b1"),
        vb_evidence(NODE, 0.88, "quiz", "another-agent", "conv-agent-b", day=1, ev_id="b2"),
        vb_evidence(NODE, 0.92, "quiz", "third-agent", "conv-agent-b", day=2, ev_id="b3"),
    ])
    assert skill_a.effective_mastery(NODE) == skill_b.effective_mastery(NODE)
    assert _history_levels(skill_a, NODE) == _history_levels(skill_b, NODE)


def test_identical_timestamps_evaluation_is_set_determined(tmp_path):
    """Regression test for the tie-break fix: every promotion record must
    cite its evidence in canonical (occurred_at, id) order, regardless of
    the order verticals submitted the items in. Before the fix, the K-window
    was picked by arrival order, so the same subset evaluated at the same
    moment could promote differently."""
    for trial in range(4):
        learner_id = f"conv-tie-{trial}"
        skill = vb_skill(tmp_path, learner_id=learner_id)
        items = [
            vb_evidence(NODE, 0.90, "quiz", "quiz-agent", learner_id, day=0, ev_id=f"{learner_id}-t{i}")
            for i in range(3)
        ]
        rng = random.Random(trial)
        rng.shuffle(items)
        submit_all(skill, items)
        history = skill.mastery_history(NODE)
        assert [r.level for r in history] == [MasteryLevel.BEGINNER, MasteryLevel.INTERMEDIATE]
        # the promotion record cites the canonical window, not arrival order
        assert history[1].evidence_ids == [f"{learner_id}-t{i}" for i in range(3)]


def test_tied_timestamps_path_dependence_is_deliberate(tmp_path):
    """Documented order dependence: with identical timestamps, arrival order
    is the only temporal signal the engine has. The engine evaluates
    incrementally and never demotes (deliberate scope exclusion), so a batch
    whose clean subset arrives first promotes, while the same batch with the
    contradicting item interleaved earlier does not. Both outcomes are fully
    explained by their records; neither is a malfunction."""
    perfs = [0.90, 0.90, 0.90, 0.10]

    def run(order):
        learner_id = f"conv-doc-{''.join(map(str, order))}"
        skill = vb_skill(tmp_path, learner_id=learner_id)
        items = [
            vb_evidence(NODE, p, "quiz", "quiz-agent", learner_id, day=0, ev_id=f"{learner_id}-t{i}")
            for i, p in enumerate(perfs)
        ]
        submit_all(skill, [items[i] for i in order])
        return skill

    # contradicting item arrives last: the clean triple promotes first
    skill_a = run([0, 1, 2, 3])
    assert skill_a.effective_mastery(NODE) == MasteryLevel.INTERMEDIATE
    assert [r.level for r in skill_a.mastery_history(NODE)] == [
        MasteryLevel.BEGINNER, MasteryLevel.INTERMEDIATE,
    ]
    # contradicting item arrives first: every K-window contains the veto
    skill_b = run([3, 0, 1, 2])
    assert skill_b.effective_mastery(NODE) == MasteryLevel.BEGINNER
    assert [r.level for r in skill_b.mastery_history(NODE)] == [MasteryLevel.BEGINNER]
    # both histories are complete and replayable from their records
    for skill in (skill_a, skill_b):
        for rec in skill.mastery_history(NODE):
            assert rec.rule_version == "heuristic-v1"
            assert rec.evidence_ids  # every transition cites its evidence


def test_canonical_order_submission_converges(tmp_path):
    """The supported contract: when verticals submit evidence in
    non-decreasing (occurred_at, id) order — the normal 'report as it
    happens' pattern — the full trajectory and final mastery are pure
    functions of the evidence set: independent of agent ids and of how
    other nodes' evidence interleaves."""
    def node_set(learner_id):
        return [
            vb_evidence(NODE, 0.90, "quiz", "quiz-agent", learner_id, day=0, ev_id=f"{learner_id}-n1"),
            vb_evidence(NODE, 0.85, "dialogue", "tutor-agent", learner_id, day=1, ev_id=f"{learner_id}-n2"),
            vb_evidence(NODE, 0.92, "practice", "practice-agent", learner_id, day=2, ev_id=f"{learner_id}-n3"),
            vb_evidence(NODE, 0.88, "quiz", "exam-agent", learner_id, day=3, ev_id=f"{learner_id}-n4"),
            vb_evidence(NODE, 0.95, "visualization", "viz-agent", learner_id, day=4, ev_id=f"{learner_id}-n5"),
        ]

    def other_node_set(learner_id, tag):
        return [
            vb_evidence("linearization", 0.70, "quiz", "quiz-agent", learner_id, day=d,
                        ev_id=f"{learner_id}-{tag}{d}")
            for d in range(5)
        ]

    reference = None
    # interleave the other node's evidence at different positions
    for trial, split in enumerate([0, 2, 5]):
        learner_id = f"conv-canon-{trial}"
        skill = vb_skill(tmp_path, learner_id=learner_id)
        main, other = node_set(learner_id), other_node_set(learner_id, f"o{trial}")
        stream = main[:split] + other + main[split:]
        # relabel agents to prove identity-independence
        for e in stream:
            e.source_agent = f"agent-{trial}-{e.activity_type}"
        submit_all(skill, stream)
        trajectory = (
            skill.effective_mastery(NODE),
            [r.level for r in skill.mastery_history(NODE)],
            skill.effective_mastery("linearization"),
        )
        if reference is None:
            reference = trajectory
        else:
            assert trajectory == reference, f"trial {trial} diverged: {trajectory} != {reference}"
    assert reference[0] == MasteryLevel.ADVANCED


def test_convergence_matches_independent_model(tmp_path):
    """The engine's incremental behavior matches an independent
    re-implementation of the documented rule replaying the same arrival
    sequence — a second implementation of the ratchet, not a copy of the
    engine's code."""
    params = MasteryParams()
    for trial in range(8):
        rng = random.Random(100 + trial)
        learner_id = f"conv-model-{trial}"
        skill = vb_skill(tmp_path, learner_id=learner_id, params=params)
        n = rng.randint(1, 7)
        evidences = [
            vb_evidence(
                NODE,
                round(rng.uniform(0.3, 1.0), 2),
                rng.choice(["quiz", "practice", "dialogue"]),
                f"agent-{rng.randint(1, 3)}",
                learner_id,
                day=rng.randint(0, 20),
                ev_id=f"{learner_id}-m{i}",
            )
            for i in range(n)
        ]
        order = list(range(n))
        rng.shuffle(order)
        arrival = [evidences[i] for i in order]
        submit_all(skill, arrival)
        assert skill.effective_mastery(NODE) == independent_incremental_level(arrival, params), \
            f"trial {trial}: engine diverged from the independent ratchet model"


def test_early_promotion_survives_later_dilution(tmp_path):
    """Deliberate ratchet, characterized: 3x0.95 promotes to ADVANCED; three
    0.50s arriving later dilute the K-window but cannot demote (demotion is
    out of scope). The independent model agrees — this is the documented
    no-demotion semantics, not a malfunction."""
    skill = vb_skill(tmp_path, learner_id="conv-dilute")
    arrival = [
        vb_evidence(NODE, 0.95, "quiz", "q", "conv-dilute", day=0, ev_id="dl1"),
        vb_evidence(NODE, 0.95, "practice", "p", "conv-dilute", day=1, ev_id="dl2"),
        vb_evidence(NODE, 0.95, "dialogue", "d", "conv-dilute", day=2, ev_id="dl3"),
        vb_evidence(NODE, 0.50, "quiz", "q", "conv-dilute", day=3, ev_id="dl4"),
        vb_evidence(NODE, 0.50, "quiz", "q", "conv-dilute", day=4, ev_id="dl5"),
        vb_evidence(NODE, 0.50, "quiz", "q", "conv-dilute", day=5, ev_id="dl6"),
    ]
    submit_all(skill, arrival)
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
    assert independent_incremental_level(arrival) == MasteryLevel.ADVANCED
    # the last-3 window is now all 0.50: a *batch* fixed point would say
    # BEGINNER, but the engine is an incremental ratchet, not a batch engine


def test_evidence_set_superset_monotone_or_stable(tmp_path):
    """Adding more evidence never silently *lowers* the heuristic level
    (demotion is out of scope for heuristic-v1): the level sequence is
    monotone non-decreasing as evidence accumulates."""
    skill = vb_skill(tmp_path, learner_id="conv-mono")
    rng = random.Random(3)
    levels = []
    order = [MasteryLevel.UNKNOWN, MasteryLevel.BEGINNER, MasteryLevel.INTERMEDIATE, MasteryLevel.ADVANCED]
    for i in range(10):
        e = vb_evidence(NODE, round(rng.uniform(0.0, 1.0), 2),
                        rng.choice(["quiz", "practice"]), f"agent-{i % 2}",
                        "conv-mono", day=i, ev_id=f"conv-mono-s{i}")
        skill.record_evidence(e)
        levels.append(skill.effective_mastery(NODE))
    idx = [order.index(l) for l in levels]
    assert idx == sorted(idx), f"heuristic level decreased: {levels}"
