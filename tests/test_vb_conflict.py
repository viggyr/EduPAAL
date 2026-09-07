"""CONFLICT: what the heuristics do with contradictory evidence.

Characterized (deliberate) behavior of heuristic-v1:
- A single evaluation-set item below t_contradict VETOES promotion, no
  matter which agent reported it. The engine does not weight reporters:
  source_agent is provenance-only. Trust in reporters is a deployment
  concern (which agents may write), not a heuristic input.
- The veto is recency-scoped: once the contradicting item ages out of the
  K-most-recent window, promotion proceeds on the remaining consensus.
- Sustained disagreement (every window contains a veto item) never
  promotes: no consensus, no promotion. Demotion is out of scope, so a
  previously promoted level is never lowered by later contradiction.
"""

from edupaal import MasteryLevel

from .vb_helpers import submit_all, vb_evidence, vb_skill

NODE = "linear-equations"


def test_single_contradiction_vetoes_promotion(tmp_path):
    skill = vb_skill(tmp_path, learner_id="conf-veto")
    submit_all(skill, [
        vb_evidence(NODE, 0.95, "quiz", "quiz-agent", "conf-veto", day=0, ev_id="cv1"),
        vb_evidence(NODE, 0.92, "quiz", "quiz-agent", "conf-veto", day=1, ev_id="cv2"),
        # mean would clear 0.65, but 0.15 < t_contradict blocks it
        vb_evidence(NODE, 0.15, "practice", "practice-agent", "conf-veto", day=2, ev_id="cv3"),
    ])
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER
    assert [r.level for r in skill.mastery_history(NODE)] == [MasteryLevel.BEGINNER]


def test_contradiction_ages_out_of_the_window(tmp_path):
    skill = vb_skill(tmp_path, learner_id="conf-age")
    submit_all(skill, [
        vb_evidence(NODE, 0.15, "practice", "practice-agent", "conf-age", day=0, ev_id="ca1"),
        vb_evidence(NODE, 0.95, "quiz", "quiz-agent", "conf-age", day=1, ev_id="ca2"),
        vb_evidence(NODE, 0.92, "quiz", "quiz-agent", "conf-age", day=2, ev_id="ca3"),
    ])
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER  # veto active
    submit_all(skill, [
        vb_evidence(NODE, 0.90, "quiz", "quiz-agent", "conf-age", day=3, ev_id="ca4"),
        vb_evidence(NODE, 0.93, "quiz", "quiz-agent", "conf-age", day=4, ev_id="ca5"),
        vb_evidence(NODE, 0.91, "quiz", "quiz-agent", "conf-age", day=5, ev_id="ca6"),
    ])
    # the 0.15 has left the K=3 window: [0.90, 0.93, 0.91] promotes
    assert skill.effective_mastery(NODE) == MasteryLevel.INTERMEDIATE


def test_veto_ignores_which_agent_reported(tmp_path):
    """Same activity, two agents, strong disagreement: the veto applies
    regardless of reporter. The engine never adjudicates trust between
    agents — deployments decide who may write."""
    skill = vb_skill(tmp_path, learner_id="conf-agent")
    submit_all(skill, [
        vb_evidence(NODE, 0.95, "quiz", "trusted-exam-agent", "conf-agent", day=0, ev_id="cg1"),
        vb_evidence(NODE, 0.94, "quiz", "trusted-exam-agent", "conf-agent", day=1, ev_id="cg2"),
        vb_evidence(NODE, 0.20, "quiz", "sketchy-agent", "conf-agent", day=2, ev_id="cg3"),
    ])
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER
    # and symmetrically: the "trusted" agent cannot outvote the veto either
    skill2 = vb_skill(tmp_path, learner_id="conf-agent2")
    submit_all(skill2, [
        vb_evidence(NODE, 0.20, "quiz", "sketchy-agent", "conf-agent2", day=0, ev_id="cg4"),
        vb_evidence(NODE, 0.95, "quiz", "trusted-exam-agent", "conf-agent2", day=1, ev_id="cg5"),
        vb_evidence(NODE, 0.94, "quiz", "trusted-exam-agent", "conf-agent2", day=2, ev_id="cg6"),
    ])
    assert skill2.effective_mastery(NODE) == MasteryLevel.BEGINNER


def test_sustained_disagreement_never_promotes(tmp_path):
    skill = vb_skill(tmp_path, learner_id="conf-sustain")
    for i in range(8):
        perf = 0.95 if i % 2 == 0 else 0.20
        skill.record_evidence(
            vb_evidence(NODE, perf, "quiz", f"agent-{i % 2}", "conf-sustain", day=i, ev_id=f"cs{i}")
        )
    # every K=3 window contains a 0.20 -> vetoed every time
    assert skill.effective_mastery(NODE) == MasteryLevel.BEGINNER
    assert [r.level for r in skill.mastery_history(NODE)] == [MasteryLevel.BEGINNER]


def test_contradiction_after_promotion_does_not_demote(tmp_path):
    """Demotion is out of scope for heuristic-v1: later contradiction is
    recorded as evidence but never lowers an achieved level."""
    skill = vb_skill(tmp_path, learner_id="conf-nodemote")
    submit_all(skill, [
        vb_evidence(NODE, 0.92, "quiz", "q", "conf-nodemote", day=0, ev_id="cn1"),
        vb_evidence(NODE, 0.93, "practice", "p", "conf-nodemote", day=1, ev_id="cn2"),
        vb_evidence(NODE, 0.94, "dialogue", "d", "conf-nodemote", day=2, ev_id="cn3"),
    ])
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
    # a run of terrible evidence afterwards: recorded, but no demotion
    submit_all(skill, [
        vb_evidence(NODE, 0.10, "quiz", "q", "conf-nodemote", day=3, ev_id="cn4"),
        vb_evidence(NODE, 0.05, "quiz", "q", "conf-nodemote", day=4, ev_id="cn5"),
        vb_evidence(NODE, 0.12, "quiz", "q", "conf-nodemote", day=5, ev_id="cn6"),
    ])
    assert skill.effective_mastery(NODE) == MasteryLevel.ADVANCED
    assert len(skill.store.list_evidence("conf-nodemote", NODE)) == 6


def test_contradiction_threshold_is_tunable(tmp_path):
    """Deployments that want majority-tolerant promotion can lower the
    veto bar; the knob must actually change behavior."""
    from edupaal import MasteryParams

    lax = MasteryParams(t_contradict=0.10)  # only extreme failure vetoes
    skill = vb_skill(tmp_path, learner_id="conf-tunable", params=lax)
    submit_all(skill, [
        vb_evidence(NODE, 0.95, "quiz", "q", "conf-tunable", day=0, ev_id="ct1"),
        vb_evidence(NODE, 0.92, "quiz", "q", "conf-tunable", day=1, ev_id="ct2"),
        vb_evidence(NODE, 0.15, "quiz", "q", "conf-tunable", day=2, ev_id="ct3"),
    ])
    # 0.15 >= 0.10: no veto; mean 0.673 >= 0.65 -> INTERMEDIATE
    assert skill.effective_mastery(NODE) == MasteryLevel.INTERMEDIATE
