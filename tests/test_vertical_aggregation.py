"""QUORUM AGGREGATION: shared (learner, topic) mastery from per-vertical tracks.

Each vertical — one stable ``source_agent`` — promotes its own track from
its own evidence slice with its own thresholds. The shared (learner, topic)
level is derived by ``aggregate_vertical_mastery`` (pure, deterministic),
floored by the latest privileged assertion.

The characterized contract:

- UNKNOWN verticals are dropped; all UNKNOWN -> UNKNOWN.
- Overall ADVANCED needs ``xvertical_quorum_advanced`` (default 2)
  verticals attesting ADVANCED — the transfer bar: advancement must be
  confirmed across contexts.
- A lone ADVANCED caps at INTERMEDIATE overall.
- Anything less falls back to the highest attested level.
- Assertions bypass the quorum: the latest assertion floors the shared
  level, and assertion records never count as a vertical's attestation
  (no double-counting).
"""

from edupaal import (
    EduPAALSkill,
    LearnerPreferences,
    MasteryLevel,
    MasteryParams,
    SQLiteBackend,
    aggregate_vertical_mastery,
    build_seed_graph,
)

from .vb_helpers import submit_all, vb_evidence

NODE = "linear-equations"
B = MasteryLevel.BEGINNER
I = MasteryLevel.INTERMEDIATE
A = MasteryLevel.ADVANCED
U = MasteryLevel.UNKNOWN


def _levels(**kwargs):
    return kwargs


# ------------------------------------------------------------- pure aggregator


def test_all_unknown_gives_unknown():
    params = MasteryParams()
    assert aggregate_vertical_mastery({}, params) == U
    assert aggregate_vertical_mastery(_levels(a=U, b=U), params) == U


def test_single_beginner_reports_beginner():
    assert aggregate_vertical_mastery(_levels(quiz=B), MasteryParams()) == B


def test_single_intermediate_reports_intermediate():
    assert aggregate_vertical_mastery(_levels(quiz=I), MasteryParams()) == I


def test_lone_advanced_caps_at_intermediate():
    # The transfer bar: one vertical's ADVANCED is INTERMEDIATE overall —
    # advancement needs cross-context confirmation.
    assert aggregate_vertical_mastery(_levels(quiz=A), MasteryParams()) == I


def test_two_advanced_confirms_advanced():
    assert aggregate_vertical_mastery(_levels(quiz=A, practice=A), MasteryParams()) == A


def test_advanced_plus_beginner_caps_at_intermediate():
    assert aggregate_vertical_mastery(_levels(quiz=A, practice=B), MasteryParams()) == I


def test_advanced_plus_intermediate_caps_at_intermediate():
    # Cross-context confirmation failed even though another vertical attests
    # below: a lone ADVANCED never survives aggregation.
    assert aggregate_vertical_mastery(_levels(quiz=A, practice=I), MasteryParams()) == I


def test_unknown_verticals_are_dropped():
    assert aggregate_vertical_mastery(_levels(quiz=U, practice=I), MasteryParams()) == I
    assert aggregate_vertical_mastery(_levels(quiz=U, practice=A), MasteryParams()) == I


def test_three_verticals_two_advanced():
    assert (
        aggregate_vertical_mastery(_levels(a=A, b=A, c=B), MasteryParams()) == A
    )


def test_highest_attested_fallback():
    assert aggregate_vertical_mastery(_levels(a=B, b=B), MasteryParams()) == B
    assert aggregate_vertical_mastery(_levels(a=I, b=B), MasteryParams()) == I


def test_quorum_knob_is_real():
    strict = MasteryParams(xvertical_quorum_advanced=3)
    assert aggregate_vertical_mastery(_levels(a=A, b=A), strict) == I
    assert aggregate_vertical_mastery(_levels(a=A, b=A, c=A), strict) == A
    lax = MasteryParams(xvertical_quorum_advanced=1)
    assert aggregate_vertical_mastery(_levels(a=A), lax) == A


def test_quorum_knob_rejects_nonsense():
    import pytest

    with pytest.raises(ValueError):
        MasteryParams(xvertical_quorum_advanced=0)


# ------------------------------------------------------- engine integration


def _vskill(
    tmp_path,
    learner_id,
    vertical_params=None,
    default_params=None,
    criteria_overrides=None,
):
    """vb_skill plus vertical params / plan overrides."""
    graph = build_seed_graph()
    store = SQLiteBackend(tmp_path / f"{learner_id}.db")
    skill = EduPAALSkill(
        store, graph, default_params=default_params, vertical_params=vertical_params
    )
    skill.cold_start(
        learner_id=learner_id,
        node_selection=[NODE],
        preferences=LearnerPreferences(learning_style="visual", pace="steady"),
        criteria_overrides=criteria_overrides,
    )
    return skill


def _strong_multimodal(agent, learner_id, day=0, tag=""):
    """Three evidences, two activity types: enough for one track's ADVANCED."""
    return [
        vb_evidence(NODE, 0.90, "quiz", agent, learner_id, day=day, ev_id=f"{tag}s1"),
        vb_evidence(NODE, 0.85, "practice", agent, learner_id, day=day + 1, ev_id=f"{tag}s2"),
        vb_evidence(NODE, 0.92, "quiz", agent, learner_id, day=day + 2, ev_id=f"{tag}s3"),
    ]


def _strong_single_modal(agent, learner_id, day=0, tag=""):
    """Three strong quiz-only evidences: INTERMEDIATE, never ADVANCED."""
    return [
        vb_evidence(NODE, 0.90, "quiz", agent, learner_id, day=day, ev_id=f"{tag}q1"),
        vb_evidence(NODE, 0.88, "quiz", agent, learner_id, day=day + 1, ev_id=f"{tag}q2"),
        vb_evidence(NODE, 0.92, "quiz", agent, learner_id, day=day + 2, ev_id=f"{tag}q3"),
    ]


def test_vertical_tracks_are_isolated(tmp_path):
    skill = _vskill(tmp_path, "agg-isolated")
    submit_all(skill, _strong_single_modal("quiz-agent", "agg-isolated", tag="iso"))
    # quiz-agent's track promoted; practice-agent has no track at all
    assert skill.engine.vertical_level("agg-isolated", NODE, "quiz-agent") == I
    assert skill.engine.vertical_level("agg-isolated", NODE, "practice-agent") == U
    # lone INTERMEDIATE reports as-is
    assert skill.effective_mastery(NODE) == I
    # weak evidence on another vertical does not touch the first track
    submit_all(
        skill,
        [
            vb_evidence(NODE, 0.50, "quiz", "practice-agent", "agg-isolated", day=3, ev_id="iso-w1"),
            vb_evidence(NODE, 0.50, "quiz", "practice-agent", "agg-isolated", day=4, ev_id="iso-w2"),
            vb_evidence(NODE, 0.50, "quiz", "practice-agent", "agg-isolated", day=5, ev_id="iso-w3"),
        ],
    )
    assert skill.engine.vertical_level("agg-isolated", NODE, "quiz-agent") == I
    assert skill.engine.vertical_level("agg-isolated", NODE, "practice-agent") == B
    assert skill.engine.vertical_levels("agg-isolated", NODE) == {
        "quiz-agent": I,
        "practice-agent": B,
    }
    # highest attested below ADVANCED: INTERMEDIATE overall
    assert skill.effective_mastery(NODE) == I


def test_lone_advanced_track_caps_shared_at_intermediate(tmp_path):
    skill = _vskill(tmp_path, "agg-lone")
    submit_all(skill, _strong_multimodal("quiz-agent", "agg-lone", tag="lone"))
    assert skill.engine.vertical_level("agg-lone", NODE, "quiz-agent") == A
    assert skill.effective_mastery(NODE) == I  # transfer bar, not ADVANCED


def test_two_advanced_tracks_confirm_shared_advanced(tmp_path):
    skill = _vskill(tmp_path, "agg-confirm")
    submit_all(skill, _strong_multimodal("quiz-agent", "agg-confirm", tag="c1"))
    assert skill.effective_mastery(NODE) == I  # one ADVANCED: capped
    submit_all(skill, _strong_multimodal("practice-agent", "agg-confirm", tag="c2"))
    assert skill.engine.vertical_level("agg-confirm", NODE, "practice-agent") == A
    assert skill.effective_mastery(NODE) == A  # two ADVANCED: confirmed


def test_per_vertical_thresholds(tmp_path):
    # A strict vertical needs near-perfect performance for ADVANCED; the
    # default vertical does not. Same evidence, different tracks.
    skill = _vskill(
        tmp_path,
        "agg-thresholds",
        vertical_params={"strict-agent": MasteryParams(t_advanced=0.99)},
    )
    submit_all(skill, _strong_multimodal("quiz-agent", "agg-thresholds", tag="t1"))
    submit_all(skill, _strong_multimodal("strict-agent", "agg-thresholds", tag="t2"))
    assert skill.engine.vertical_level("agg-thresholds", NODE, "quiz-agent") == A
    # mean 0.89 < 0.99: stuck at INTERMEDIATE on the strict track
    assert skill.engine.vertical_level("agg-thresholds", NODE, "strict-agent") == I
    # lone ADVANCED among attested verticals: capped at INTERMEDIATE
    assert skill.effective_mastery(NODE) == I


def test_plan_topic_override_beats_vertical_params(tmp_path):
    # The plan's topic override governs promotion bars even when a vertical
    # has looser params: plan > vertical_params > engine defaults.
    skill = _vskill(
        tmp_path,
        "agg-plan",
        vertical_params={"quiz-agent": MasteryParams(t_intermediate=0.10)},
        criteria_overrides={NODE: MasteryParams(t_intermediate=0.99)},
    )
    submit_all(skill, _strong_single_modal("quiz-agent", "agg-plan", tag="plan"))
    # plan's 0.99 bar wins over the vertical's 0.10: still BEGINNER
    assert skill.engine.vertical_level("agg-plan", NODE, "quiz-agent") == B
    assert skill.effective_mastery(NODE) == B


def test_concept_override_cascades_and_topic_beats_concept(tmp_path):
    # "algebra" is the concept above linear-equations in the seed graph.
    skill = _vskill(
        tmp_path,
        "agg-cascade",
        criteria_overrides={
            "algebra": MasteryParams(k_evidence=2),
            NODE: MasteryParams(k_evidence=3),
        },
    )
    skill.record_evidence(vb_evidence(NODE, 0.9, "quiz", "quiz-agent", "agg-cascade", day=0, ev_id="cc1"))
    skill.record_evidence(vb_evidence(NODE, 0.9, "quiz", "quiz-agent", "agg-cascade", day=1, ev_id="cc2"))
    # topic's k=3 beats concept's k=2: two evidences are not enough
    assert skill.engine.vertical_level("agg-cascade", NODE, "quiz-agent") == B
    skill.record_evidence(vb_evidence(NODE, 0.9, "quiz", "quiz-agent", "agg-cascade", day=2, ev_id="cc3"))
    assert skill.engine.vertical_level("agg-cascade", NODE, "quiz-agent") == I

    # concept override alone cascades to its topics
    skill2 = _vskill(
        tmp_path,
        "agg-cascade2",
        criteria_overrides={"algebra": MasteryParams(k_evidence=2)},
    )
    skill2.record_evidence(vb_evidence(NODE, 0.9, "quiz", "quiz-agent", "agg-cascade2", day=0, ev_id="cc4"))
    skill2.record_evidence(vb_evidence(NODE, 0.9, "quiz", "quiz-agent", "agg-cascade2", day=1, ev_id="cc5"))
    assert skill2.engine.vertical_level("agg-cascade2", NODE, "quiz-agent") == I


def test_quorum_uses_shared_params(tmp_path):
    # The quorum is a shared-level concern: the plan's override (or engine
    # defaults) governs it, not per-vertical params.
    skill = _vskill(
        tmp_path,
        "agg-quorum",
        criteria_overrides={NODE: MasteryParams(xvertical_quorum_advanced=1)},
    )
    submit_all(skill, _strong_multimodal("quiz-agent", "agg-quorum", tag="q1"))
    assert skill.effective_mastery(NODE) == A  # quorum of 1: lone ADVANCED confirms


# ---------------------------------------------------------------- assertions


def test_assertion_floors_shared_level_with_no_evidence(tmp_path):
    skill = _vskill(tmp_path, "agg-assert")
    skill.assert_mastery(NODE, MasteryLevel.INTERMEDIATE, asserted_by="exam-board", reason="final")
    # the assertion is learner-level: no vertical track exists, yet the
    # shared level is floored
    assert skill.engine.vertical_levels("agg-assert", NODE) == {}
    assert skill.effective_mastery(NODE) == I


def test_assertion_does_not_count_as_vertical_attestation(tmp_path):
    # An exam-board ADVANCED assertion plus one vertical's heuristic
    # ADVANCED must NOT make a quorum of two: the assertion is excluded
    # from the tracks, so aggregation sees a lone ADVANCED -> INTERMEDIATE,
    # and the assertion floor then lifts the shared level to ADVANCED.
    skill = _vskill(tmp_path, "agg-assert2")
    skill.assert_mastery(NODE, MasteryLevel.ADVANCED, asserted_by="exam-board", reason="final")
    submit_all(skill, _strong_multimodal("quiz-agent", "agg-assert2", tag="a2"))
    levels = skill.engine.vertical_levels("agg-assert2", NODE)
    assert levels == {"quiz-agent": A}  # assertion not double-counted
    assert aggregate_vertical_mastery(levels, MasteryParams()) == I
    assert skill.effective_mastery(NODE) == A  # floor bypasses the quorum


def test_evidence_below_assertion_stays_floored(tmp_path):
    skill = _vskill(tmp_path, "agg-assert3")
    skill.assert_mastery(NODE, MasteryLevel.INTERMEDIATE, asserted_by="tutor", reason="placement")
    submit_all(
        skill,
        [
            vb_evidence(NODE, 0.50, "quiz", "quiz-agent", "agg-assert3", day=0, ev_id="a3w1"),
            vb_evidence(NODE, 0.50, "quiz", "quiz-agent", "agg-assert3", day=1, ev_id="a3w2"),
            vb_evidence(NODE, 0.50, "quiz", "quiz-agent", "agg-assert3", day=2, ev_id="a3w3"),
        ],
    )
    # the vertical's own track reflects its weak evidence...
    assert skill.engine.vertical_level("agg-assert3", NODE, "quiz-agent") == B
    # ...but the shared level never drops below the privileged assertion
    assert skill.effective_mastery(NODE) == I


def test_evidence_above_assertion_wins(tmp_path):
    skill = _vskill(tmp_path, "agg-assert4")
    skill.assert_mastery(NODE, MasteryLevel.INTERMEDIATE, asserted_by="tutor", reason="placement")
    submit_all(skill, _strong_multimodal("quiz-agent", "agg-assert4", tag="a4a"))
    submit_all(skill, _strong_multimodal("practice-agent", "agg-assert4", tag="a4b"))
    assert skill.effective_mastery(NODE) == A


def test_newer_assertion_replaces_floor(tmp_path):
    skill = _vskill(tmp_path, "agg-assert5")
    skill.assert_mastery(NODE, MasteryLevel.ADVANCED, asserted_by="exam-board", reason="final")
    assert skill.effective_mastery(NODE) == A
    skill.assert_mastery(NODE, MasteryLevel.INTERMEDIATE, asserted_by="tutor", reason="correction")
    assert skill.effective_mastery(NODE) == I


def test_promoted_override_floors_like_assertion(tmp_path):
    skill = _vskill(tmp_path, "agg-assert6")
    skill.set_override(MasteryLevel.INTERMEDIATE, reason="holding", scope_node_id=NODE)
    record = skill.promote_override(NODE)
    assert record.assertion is True
    assert skill.engine.vertical_levels("agg-assert6", NODE) == {}
    assert skill.effective_mastery(NODE) == I


# ------------------------------------------------------------------ backends


def test_history_vertical_filter(tmp_path):
    skill = _vskill(tmp_path, "agg-filter")
    submit_all(skill, _strong_single_modal("quiz-agent", "agg-filter", tag="f1"))
    submit_all(skill, _strong_single_modal("practice-agent", "agg-filter", tag="f2"))
    store = skill.store
    quiz_hist = store.get_mastery_history("agg-filter", NODE, vertical_id="quiz-agent")
    assert {r.vertical_id for r in quiz_hist} == {"quiz-agent"}
    assert [r.level for r in quiz_hist] == [B, I]
    practice_hist = store.get_mastery_history("agg-filter", NODE, vertical_id="practice-agent")
    assert {r.vertical_id for r in practice_hist} == {"practice-agent"}
    assert len(store.get_mastery_history("agg-filter", NODE)) == 4
    cur = store.get_current_mastery("agg-filter", NODE, vertical_id="quiz-agent")
    assert cur is not None and cur.level == I and cur.vertical_id == "quiz-agent"


def test_records_carry_reporting_vertical(tmp_path):
    skill = _vskill(tmp_path, "agg-carry")
    written = skill.record_evidence(
        vb_evidence(NODE, 0.9, "quiz", "quest-agent", "agg-carry", day=0, ev_id="carr1")
    )
    assert written and all(r.vertical_id == "quest-agent" for r in written)
    rec = skill.assert_mastery(NODE, MasteryLevel.BEGINNER, asserted_by="tutor", reason="x")
    assert rec.vertical_id == "default"


def test_legacy_records_without_vertical_id_read_as_default():
    # Provider record payloads written before vertical scoping have no
    # vertical_id key; they must read back as "default", not crash.
    from edupaal.providers._records import from_record, to_record

    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        graph = build_seed_graph()
        store = SQLiteBackend(os.path.join(d, "legacy.db"))
        s = EduPAALSkill(store, graph)
        s.cold_start(
            learner_id="legacy",
            node_selection=[NODE],
            preferences=LearnerPreferences(learning_style="visual", pace="steady"),
        )
        rec = s.assert_mastery(NODE, MasteryLevel.BEGINNER, asserted_by="t", reason="x")
        _, _, envelope = to_record(rec)
        del envelope["data"]["vertical_id"]  # simulate a pre-scoping payload
        restored = from_record(envelope)
        assert restored.vertical_id == "default"


def test_old_schema_migration_assigns_default_vertical(tmp_path):
    """Databases created before vertical scoping have a mastery_records
    table with no vertical_id column. Opening them must run the migration
    and assign every existing row to the \"default\" track."""
    import json
    import sqlite3

    db_path = tmp_path / "pre-scoping.db"
    conn = sqlite3.connect(db_path)
    # the pre-scoping schema: no vertical_id column
    conn.execute(
        """CREATE TABLE mastery_records (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            learner_id TEXT NOT NULL,
            level TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            rule_version TEXT NOT NULL,
            params_in_effect TEXT NOT NULL,
            evidence_ids TEXT NOT NULL,
            assertion INTEGER NOT NULL DEFAULT 0,
            asserted_by TEXT,
            reason TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO mastery_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "mr_legacy", NODE, "legacy-learner", "intermediate",
            "2026-01-01T00:00:00+00:00", "heuristic-v1", json.dumps({}),
            json.dumps(["ev1"]), 0, None, None,
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteBackend(db_path)  # runs the migration on open
    cols = {
        r[1]
        for r in sqlite3.connect(db_path).execute(
            "PRAGMA table_info(mastery_records)"
        ).fetchall()
    }
    assert "vertical_id" in cols
    history = store.get_mastery_history("legacy-learner", NODE)
    assert len(history) == 1
    assert history[0].vertical_id == "default"
    # the migrated record reads as the default vertical's current level
    current = store.get_current_mastery("legacy-learner", NODE, "default")
    assert current is not None and current.level == MasteryLevel.INTERMEDIATE
