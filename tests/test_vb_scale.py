"""SCALE: 5,000 evidence records across 20 learners.

The framework's write and read paths must stay comfortably interactive at
this volume. Every measured mean is asserted well under a 1-second budget;
the test FAILS loudly if any operation exceeds it.

These are wall-clock timings on the validation machine — indicative of
order of magnitude, not a latency SLA.
"""

import random
import statistics
import time

from edupaal import MasteryLevel

from .vb_helpers import vb_evidence, vb_skill

NODE_A = "linear-equations"
NODE_B = "linearization"
ACTIVITIES = ["quiz", "practice", "dialogue"]

BUDGET_S = 1.0


def test_5k_evidence_20_learners_within_budget(tmp_path):
    write_times = []
    skills = {}
    for learner_i in range(20):
        learner_id = f"scale-{learner_i}"
        skills[learner_id] = vb_skill(tmp_path, learner_id=learner_id)

    rng = random.Random(20260907)
    evidences = []
    for i in range(5000):
        learner_id = f"scale-{i % 20}"
        node = NODE_A if (i // 20) % 2 == 0 else NODE_B
        evidences.append(
            vb_evidence(
                node, round(rng.uniform(0.4, 1.0), 2),
                rng.choice(ACTIVITIES), f"agent-{rng.randint(1, 3)}",
                learner_id, day=i // 20, ev_id=f"scale-{i}",
            )
        )

    for e in evidences:
        t0 = time.perf_counter()
        skills[e.learner_id].record_evidence(e)
        write_times.append(time.perf_counter() - t0)

    # every learner should have converged on evidence-driven state
    leveled = sum(
        1
        for skill in skills.values()
        for node in (NODE_A, NODE_B)
        if skill.effective_mastery(node) != MasteryLevel.UNKNOWN
    )
    assert leveled == 40, f"only {leveled}/40 node states leveled"

    # read-path timings across all learners
    def time_op(name, fn):
        ts = []
        for skill in skills.values():
            t0 = time.perf_counter()
            fn(skill)
            ts.append(time.perf_counter() - t0)
        mean_ms = statistics.fmean(ts) * 1000
        print(f"\n{name}: mean {mean_ms:.2f} ms over 20 learners "
              f"(budget {BUDGET_S * 1000:.0f} ms)")
        assert mean_ms / 1000 < BUDGET_S, f"{name} exceeded budget"
        return mean_ms

    write_mean_ms = statistics.fmean(write_times) * 1000
    print(f"\nrecord_evidence: mean {write_mean_ms:.3f} ms over 5,000 writes "
          f"(budget {BUDGET_S * 1000:.0f} ms)")
    assert write_mean_ms / 1000 < BUDGET_S

    time_op("next_topic", lambda s: s.next_topic())
    time_op("grounding_packet", lambda s: s.grounding_packet(NODE_A))
    time_op("weakest", lambda s: s.weakest(5))
    time_op("strongest", lambda s: s.strongest(5))
    time_op("effective_mastery", lambda s: s.effective_mastery(NODE_A))
