"""SEMESTER SIMULATION: an end-to-end four-week narrative.

Three learners share one store with three verticals (guide/dialogue,
evaluator/quiz, practice app) writing normalized evidence:
- aria: steady progression across two topics over 4 weeks;
- dev: mid-plan reprioritization (re-cold-start) with history preserved;
- noah: sparse/conflicting evidence that never converges, then a final-exam
  assertion lands durable state.

The expected trajectories are hand-derived from the documented rules and
asserted exactly, not fitted after the fact.
"""

from datetime import datetime, timedelta, timezone

from edupaal import LearnerPreferences

from .vb_helpers import submit_all, vb_evidence, vb_skill

BASE = datetime(2026, 1, 5, tzinfo=timezone.utc)  # a Monday
PREFS = LearnerPreferences(learning_style="mixed", pace="steady")
PLAN = ["linear-equations", "linearization", "dropout-rate", "inverted-dropout"]

AGENTS = {"guide": "dialogue", "evaluator": "quiz", "practice": "practice"}


def ev(learner, vertical, node, perf, day, n):
    return vb_evidence(
        node, perf, AGENTS[vertical], vertical, learner,
        day=day, ev_id=f"{learner}-{n}",
    )


def test_four_week_semester(tmp_path):
    skills = {}
    for learner in ("aria", "dev", "noah"):
        skills[learner] = vb_skill(tmp_path, learner_id=learner, topics=PLAN)

    # ---- week 1: orientation dialogue + first quizzes (days 0-6) ----
    submit_all(skills["aria"], [
        ev("aria", "guide", "linear-equations", 0.80, 0, 1),
        ev("aria", "evaluator", "linear-equations", 0.72, 2, 2),
        ev("aria", "guide", "linear-equations", 0.78, 5, 3),  # [0.80,0.72,0.78] -> I
    ])
    submit_all(skills["dev"], [
        ev("dev", "guide", "linear-equations", 0.60, 1, 1),
        ev("dev", "evaluator", "linear-equations", 0.65, 3, 2),
        ev("dev", "practice", "linear-equations", 0.55, 5, 3),  # [0.60,0.65,0.55] -> B
    ])
    submit_all(skills["noah"], [
        ev("noah", "guide", "linear-equations", 0.90, 1, 1),
        ev("noah", "evaluator", "linear-equations", 0.30, 4, 2),  # disagreement
        ev("noah", "practice", "linear-equations", 0.85, 6, 3),  # 0.30 vetoes -> B
    ])

    from edupaal import MasteryLevel
    aria, dev, noah = skills["aria"], skills["dev"], skills["noah"]
    assert aria.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE
    assert dev.effective_mastery("linear-equations") == MasteryLevel.BEGINNER
    assert noah.effective_mastery("linear-equations") == MasteryLevel.BEGINNER

    # ---- week 2: practice intensifies (days 7-13) ----
    submit_all(aria, [
        ev("aria", "practice", "linear-equations", 0.88, 8, 4),
        ev("aria", "evaluator", "linear-equations", 0.91, 10, 5),
        ev("aria", "guide", "linear-equations", 0.86, 12, 6),  # [0.88,0.91,0.86] mean 0.883, 3 modalities -> A
        ev("aria", "evaluator", "linearization", 0.75, 11, 7),
        ev("aria", "guide", "linearization", 0.70, 13, 8),  # 2/3 -> B
    ])
    submit_all(dev, [
        ev("dev", "practice", "linear-equations", 0.70, 9, 4),
        ev("dev", "evaluator", "linear-equations", 0.75, 12, 5),  # window [0.55,0.70,0.75] mean 0.667, same modal -> I
    ])
    submit_all(noah, [
        ev("noah", "guide", "linearization", 0.85, 9, 4),
        ev("noah", "evaluator", "linearization", 0.25, 11, 5),  # veto again -> B after 3
        ev("noah", "practice", "linearization", 0.80, 13, 6),
    ])
    assert aria.effective_mastery("linear-equations") == MasteryLevel.ADVANCED
    assert aria.effective_mastery("linearization") == MasteryLevel.BEGINNER
    assert dev.effective_mastery("linear-equations") == MasteryLevel.INTERMEDIATE
    assert noah.effective_mastery("linearization") == MasteryLevel.BEGINNER
    # guide's grounding packet for aria reflects the durable state, not dialogue vibes
    packet = aria.grounding_packet("linear-equations")
    assert packet["mastery_context"]["node"]["mastery"] == "advanced"
    assert packet["next_topic"]["id"] != "linear-equations"

    # ---- week 3: dev's plan changes (day 15) ----
    from edupaal import MasteryLevel as ML

    plan2 = dev.cold_start(
        learner_id="dev",
        node_selection=["linear-equations", "inverted-dropout"],
        preferences=PREFS,
    )
    assert plan2.version == 2
    assert dev.effective_mastery("linear-equations") == ML.INTERMEDIATE  # preserved
    assert dev.effective_mastery("inverted-dropout") == ML.UNKNOWN
    submit_all(dev, [
        ev("dev", "evaluator", "inverted-dropout", 0.90, 16, 6),
        ev("dev", "practice", "inverted-dropout", 0.88, 18, 7),
        ev("dev", "guide", "inverted-dropout", 0.92, 20, 8),  # [0.90,0.88,0.92] -> A (mean 0.9, cross-modal)
    ])
    assert dev.effective_mastery("inverted-dropout") == ML.ADVANCED
    # dropout-rate was dropped from the plan but its (empty) history is intact
    assert dev.effective_mastery("dropout-rate") == ML.UNKNOWN

    # ---- week 4: finals (days 21-27) ----
    submit_all(aria, [
        ev("aria", "evaluator", "linearization", 0.78, 22, 9),  # [0.75,0.70,0.78] mean 0.743 -> I
    ])
    assert aria.effective_mastery("linearization") == ML.INTERMEDIATE
    # noah sits the final: the evaluator asserts linear-equations
    noah.assert_mastery("linear-equations", ML.INTERMEDIATE,
                        asserted_by="evaluator", reason="final exam 74%")
    assert noah.effective_mastery("linear-equations") == ML.INTERMEDIATE
    hist = noah.mastery_history("linear-equations")
    assert hist[-1].assertion is True and hist[-1].asserted_by == "evaluator"
    assert hist[-1].rule_version == "assertion-v1"

    # ---- final ledger ----
    assert aria.effective_mastery("linear-equations") == ML.ADVANCED
    assert aria.effective_mastery("linearization") == ML.INTERMEDIATE
    assert aria.effective_mastery("dropout-rate") == ML.UNKNOWN
    assert dev.effective_mastery("linear-equations") == ML.INTERMEDIATE
    assert dev.effective_mastery("inverted-dropout") == ML.ADVANCED
    assert noah.effective_mastery("linear-equations") == ML.INTERMEDIATE  # asserted
    assert noah.effective_mastery("linearization") == ML.BEGINNER        # never converged

    # every transition across all three learners is fully provenanced
    for learner, skill in skills.items():
        for node in PLAN + ["inverted-dropout"]:
            for rec in skill.mastery_history(node):
                assert rec.rule_version in ("heuristic-v1", "assertion-v1")
                for eid in rec.evidence_ids:
                    assert skill.store.get_evidence(eid) is not None
