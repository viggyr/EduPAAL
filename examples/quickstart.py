"""EduPAAL quickstart: one concept, many modalities, one shared memory.

Walks through:
  1. building the exemplar knowledge graph,
  2. cold start (plan + declared preferences),
  3. three vertical agents (quiz, practice, visualization) reporting evidence
     about the SAME topic into the SAME memory,
  4. mastery promotion driven by EduPAAL's heuristics,
  5. the grounding packet a vertical agent would consume,
  6. next_topic / weakest / strongest retrieval.

Run:  python examples/quickstart.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edupaal import (  # noqa: E402
    EduPAALSkill,
    Evidence,
    LearnerPreferences,
    MasteryLevel,
    SQLiteBackend,
    build_seed_graph,
)

BASE = datetime(2026, 9, 6, tzinfo=timezone.utc)


def report(day: int, activity: str, agent: str, perf: float, topic: str,
           learner: str, n: int) -> Evidence:
    return Evidence(
        id=f"qs-ev{n:02d}",
        learner_id=learner,
        node_id=topic,
        source_agent=agent,
        activity_type=activity,
        occurred_at=BASE + timedelta(days=day),
        performance=perf,
        details={"raw": f"{agent} session #{n}"},
    )


def main() -> None:
    graph = build_seed_graph()
    print(f"knowledge graph: {len(graph)} nodes "
          f"({', '.join(sorted({n.level.value for n in graph.nodes.values()}))})")

    db = tempfile.mkdtemp(prefix="edupaal-qs-") + "/edupaal.db"
    skill = EduPAALSkill(SQLiteBackend(db), graph)
    plan = skill.cold_start(
        learner_id="sam",
        node_selection=["linear-equations", "linearization",
                        "bias-variance", "regularization"],
        preferences=LearnerPreferences(learning_style="visual", pace="steady"),
        learner_name="Sam",
    )
    print(f"cold start: plan {plan.id} with {len(plan.topic_ids)} topics; "
          f"style=visual, pace=steady\n")

    topic = "linear-equations"
    sessions = [
        (0, "quiz", "quiz-agent", 0.90),
        (1, "practice", "practice-agent", 0.85),
        (2, "visualization", "viz-agent", 0.92),
    ]
    for i, (day, activity, agent, perf) in enumerate(sessions, start=1):
        records = skill.record_evidence(
            report(day, activity, agent, perf, topic, "sam", i))
        for r in records:
            print(f"day {day}: {agent} reports {activity}={perf:.2f} "
                  f"-> mastery {r.level.value} "
                  f"(rule {r.rule_version}, {len(r.evidence_ids)} evidence)")
    print(f"\n{topic}: effective mastery = "
          f"{skill.effective_mastery(topic).value}")
    assert skill.effective_mastery(topic) == MasteryLevel.ADVANCED

    print("\n--- grounding packet for 'linearization' ---")
    packet = skill.grounding_packet("linearization")
    print(f"style={packet['preferences']['learning_style']}, "
          f"pace={packet['preferences']['pace']}")
    print(f"plan: {packet['plan']['advanced_topics']}/"
          f"{packet['plan']['total_topics']} topics advanced")
    print(f"next topic: {packet['next_topic']['name']}")
    mc = packet["mastery_context"]
    by_level = {a["level"]: a for a in mc["ancestors"]}
    print(f"concept '{by_level['concept']['name']}': "
          f"{by_level['concept']['mastery']} (score {by_level['concept']['score']})")

    print("\n--- retrieval ---")
    nxt = skill.next_topic()
    print(f"next_topic(): {nxt.name}")
    print("weakest topics:",
          [n.name for n, _, _ in skill.weakest(2)])
    print("strongest topics:",
          [(n.name, lvl.value) for n, lvl, _ in skill.strongest(2)])

    print("\n--- sub-topics: evidence at the finest grain ---")
    for i, (day, activity, agent, perf) in enumerate(
        [(6, "quiz", "quiz-agent", 0.88),
         (7, "practice", "practice-agent", 0.91),
         (8, "dialogue", "tutor-agent", 0.87)], start=20):
        for r in skill.record_evidence(
                report(day, activity, agent, perf, "dropout-rate", "sam", i)):
            print(f"day {day}: {agent} reports {activity}={perf:.2f} on "
                  f"Dropout Rate Tuning -> {r.level.value}")
    retr = skill._retriever_for()
    for nid in ["dropout-rate", "dropout", "regularization", "overfitting"]:
        lvl, score = retr.rollup(nid)
        print(f"rollup {nid}: {lvl.value} (score {score})")

    print(f"\ndone. sqlite db at {db}")


if __name__ == "__main__":
    main()
