"""Backend conformance: the same assertions against every StorageBackend.

SQLite always runs. The Mem0 and MemOS providers run when their extras are
installed *and* a usable backend can be constructed; otherwise they skip
with a clear reason — a skip is never reported as a pass.

Coverage: every entity type, plan versioning, append-only behavior
(duplicate ids fail loudly), override expiry/deletion, ordering, exact
equality after round trip, and durability across reopen.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from edupaal import (
    DynamicOverride,
    Evidence,
    KnowledgeNode,
    Learner,
    LearnerPreferences,
    LearningPlan,
    MasteryLevel,
    MasteryParams,
    MasteryRecord,
    NodeLevel,
    SQLiteBackend,
)

BASE = datetime(2026, 9, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# backend factories (may pytest.skip with a clear reason)
# ---------------------------------------------------------------------------

def _make_sqlite(tmp_path, monkeypatch):
    return SQLiteBackend(tmp_path / "conformance.db")


def _mem0_skip_reason():
    try:
        import mem0  # noqa: F401
    except ImportError:
        return "mem0ai not installed — `pip install \"edupaal[mem0-provider]\"` to run Mem0 conformance"
    return None


def _mem0_env(monkeypatch):
    # Must run before the first `import mem0`: mem0 reads MEM0_TELEMETRY at
    # import time, and with telemetry on every Memory opens a second
    # embedded Qdrant client at ~/.mem0/migrations_qdrant (which also
    # blocks reopening the store in the same process) and attempts PostHog
    # uploads. The provider module defaults it off on import too;
    # belt-and-braces here keeps the test hermetic.
    # The vendored httpx in this mem0/openai combination chokes on proxy
    # env vars (notably IPv6 entries in no_proxy); the provider itself
    # makes no network calls in this configuration.
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.setenv("OPENAI_API_KEY", "conformance-dummy-key")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("NO_PROXY", "localhost")
    for var in (
        "http_proxy", "https_proxy", "HTTP_PROXY",
        "HTTPS_PROXY", "ALL_PROXY", "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


def _memos_env(monkeypatch):
    # Imported here (not at module top) so this env hygiene applies first:
    # the ollama client created during memos import reads proxy env vars,
    # and this environment's no_proxy entry breaks its URL parsing.
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("NO_PROXY", "localhost")
    for var in (
        "http_proxy", "https_proxy", "HTTP_PROXY",
        "HTTPS_PROXY", "ALL_PROXY", "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_mem0(tmp_path, monkeypatch):
    """Mem0 backend with a keyless local config for tests.

    Qdrant runs embedded on disk; mem0's own MockEmbeddings stands in for a
    real embedding model (EduPAAL retrieval is exact metadata lookup, never
    vector search, so embedding quality is irrelevant here). The LLM entry
    is never invoked — all writes use infer=False.
    """
    from mem0 import Memory
    from mem0.embeddings.mock import MockEmbeddings

    from edupaal.providers.mem0_provider import Mem0Provider

    try:
        memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(tmp_path / "mem0_qdrant"),
                        "collection_name": "conformance",
                        "embedding_model_dims": 10,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small"},
                },
                "llm": {
                    "provider": "openai",
                    "config": {"api_key": "conformance-dummy-key"},
                },
                "history_db_path": str(tmp_path / "mem0_history.db"),
            }
        )
    except Exception as exc:
        pytest.skip(f"mem0 backend could not be constructed: {exc}")
    memory.embedding_model = MockEmbeddings()
    return Mem0Provider(memory)


def _memos_skip_reason():
    try:
        import memos  # noqa: F401
    except ImportError:
        return "MemoryOS not installed — `pip install \"edupaal[memos-provider]\"` to run MemOS conformance"
    return None


def _make_memos(tmp_path, monkeypatch):
    from memos.configs.memory import NaiveTextMemoryConfig
    from memos.memories.textual.naive import NaiveTextMemory

    from edupaal.providers.memos_provider import MemOSProvider

    try:
        config = NaiveTextMemoryConfig(
            memory_filename="edupaal_textual_memory.json",
            # Never invoked: the provider uses direct CRUD, not extract().
            extractor_llm={
                "backend": "ollama",
                "config": {"model_name_or_path": "llama3.1:latest"},
            },
        )
        memory = NaiveTextMemory(config)
    except Exception as exc:
        pytest.skip(f"memos backend could not be constructed: {exc}")
    return MemOSProvider(memory, dir=str(tmp_path / "memos"))


_BACKENDS = {
    # name: (env-setup hook or None, skip-reason hook or None, factory).
    # The env-setup hook runs before the first import of the backend's
    # library, so import-time env reads (telemetry flags, proxy vars) are
    # already sanitized.
    "sqlite": (None, None, _make_sqlite),
    "mem0": (_mem0_env, _mem0_skip_reason, _make_mem0),
    "memos": (_memos_env, _memos_skip_reason, _make_memos),
}


@pytest.fixture(params=list(_BACKENDS))
def backend_spec(request):
    return request.param


@pytest.fixture()
def make_backend(backend_spec):
    setup_env, skip_reason_fn, factory = _BACKENDS[backend_spec]

    def _make(tmp_path, monkeypatch):
        if setup_env is not None:
            setup_env(monkeypatch)
        if skip_reason_fn is not None:
            reason = skip_reason_fn()
            if reason:
                pytest.skip(reason)
        return factory(tmp_path, monkeypatch)

    _make.spec = backend_spec
    return _make


@pytest.fixture()
def backend(make_backend, tmp_path, monkeypatch):
    return make_backend(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node(node_id, level=NodeLevel.TOPIC, parent="concept-1", name=None):
    return KnowledgeNode(
        id=node_id,
        level=level,
        name=name or node_id.replace("-", " ").title(),
        parent_id=None if level == NodeLevel.SPACE else parent,
        metadata={"tag": "ünïcodé ✓"},
    )


def _evidence(ev_id, node_id="topic-1", day=0, **kw):
    params = dict(
        id=ev_id,
        learner_id="learner-1",
        node_id=node_id,
        source_agent="quiz-agent",
        activity_type="quiz",
        occurred_at=BASE + timedelta(days=day),
        performance=0.9,
        confidence=0.8,
        attempts=2,
        hints_used=1,
        details={"raw_score": "18/20", "note": "ünïcodé ✓"},
    )
    params.update(kw)
    return Evidence(**params)


def _mastery(rec_id, day=0, level=MasteryLevel.INTERMEDIATE, **kw):
    params = dict(
        id=rec_id,
        node_id="topic-1",
        learner_id="learner-1",
        level=level,
        updated_at=BASE + timedelta(days=day),
        rule_version="heuristic-v1",
        params_in_effect={"k_evidence": 3, "t_intermediate": 0.65},
        evidence_ids=["ev-1", "ev-2"],
        assertion=False,
        asserted_by=None,
        reason="promotion",
    )
    params.update(kw)
    return MasteryRecord(**params)


def _override(ovr_id, day=0, scope="topic-1", **kw):
    params = dict(
        id=ovr_id,
        learner_id="learner-1",
        level=MasteryLevel.ADVANCED,
        scope_node_id=scope,
        expires_at=BASE + timedelta(days=day + 7),
        created_at=BASE + timedelta(days=day),
        reason="operator correction ✓",
        promoted=False,
    )
    params.update(kw)
    return DynamicOverride(**params)


# ---------------------------------------------------------------------------
# conformance battery — identical assertions for every backend
# ---------------------------------------------------------------------------

def test_learner_roundtrip_exact(backend):
    assert backend.get_learner("ghost") is None
    backend.save_learner(Learner(id="l1", name="Sam ✓"))
    assert backend.get_learner("l1") == Learner(id="l1", name="Sam ✓")
    # replace semantics
    backend.save_learner(Learner(id="l1", name="Samantha"))
    assert backend.get_learner("l1") == Learner(id="l1", name="Samantha")


def test_preferences_roundtrip_exact(backend):
    assert backend.get_preferences("nobody") is None
    prefs = LearnerPreferences(
        learning_style="socratic", pace="fast", extra={"grade": 5, "note": "✓"}
    )
    backend.save_preferences("l1", prefs)
    assert backend.get_preferences("l1") == prefs
    # replace semantics
    prefs2 = LearnerPreferences(learning_style="visual", pace="steady")
    backend.save_preferences("l1", prefs2)
    assert backend.get_preferences("l1") == prefs2


def test_nodes_prerequisites_roundtrip_exact(backend):
    space = _node("space-1", NodeLevel.SPACE, parent=None)
    subject = _node("subject-1", NodeLevel.SUBJECT, parent="space-1")
    concept = _node("concept-1", NodeLevel.CONCEPT, parent="subject-1")
    topic_a = _node("topic-1", parent="concept-1")
    topic_b = _node("topic-2", parent="concept-1")
    for node in (space, subject, concept, topic_a, topic_b):
        backend.save_node(node)

    assert backend.get_node("concept-1") == concept
    assert backend.get_node("ghost") is None
    assert [n.id for n in backend.list_nodes()] == [
        "concept-1", "space-1", "subject-1", "topic-1", "topic-2",
    ]
    assert [n.id for n in backend.list_nodes(NodeLevel.TOPIC)] == [
        "topic-1", "topic-2",
    ]
    # replace semantics
    renamed = _node("topic-1", parent="concept-1", name="Renamed")
    backend.save_node(renamed)
    assert backend.get_node("topic-1") == renamed

    backend.save_prerequisite("topic-2", "topic-1")
    backend.save_prerequisite("topic-2", "topic-1")  # duplicate: no-op
    assert backend.get_prerequisites("topic-2") == ["topic-1"]
    assert backend.get_prerequisites("topic-1") == []


def test_plan_versioning_exact(backend):
    assert backend.get_plan_for_learner("l1") is None
    v1 = LearningPlan(
        id="plan-v1", learner_id="l1", topic_ids=["t1", "t2"],
        criteria_overrides={"c1": MasteryParams(k_evidence=5)},
        created_at=BASE, version=1,
    )
    v2 = LearningPlan(
        id="plan-v2", learner_id="l1", topic_ids=["t1", "t2", "t3"],
        criteria_overrides={"c1": MasteryParams(k_evidence=2)},
        created_at=BASE + timedelta(days=1), version=2,
    )
    backend.save_plan(v1)
    backend.save_plan(v2)
    assert backend.get_plan_for_learner("l1") == v2
    assert backend.get_plan("plan-v1") == v1
    assert backend.get_plan("ghost") is None
    # replace semantics for same plan id
    v2b = LearningPlan(
        id="plan-v2", learner_id="l1", topic_ids=["t9"],
        created_at=BASE + timedelta(days=2), version=2,
    )
    backend.save_plan(v2b)
    assert backend.get_plan("plan-v2") == v2b


def test_evidence_append_only_and_exact(backend):
    e1 = _evidence("ev-1", day=2)
    e2 = _evidence("ev-2", day=1, activity_type="practice",
                   source_agent="practice-agent")
    backend.save_evidence(e1)
    backend.save_evidence(e2)

    assert backend.get_evidence("ev-1") == e1
    assert backend.get_evidence("ghost") is None
    # ordered by occurred_at, like SQLiteBackend
    assert [e.id for e in backend.list_evidence("learner-1")] == ["ev-2", "ev-1"]
    assert [e.id for e in backend.list_evidence("learner-1", "topic-1")] == [
        "ev-2", "ev-1",
    ]
    assert backend.list_evidence("learner-1", "other-topic") == []
    assert backend.list_evidence("other-learner") == []

    # append-only: duplicates fail loudly, never silently replace
    with pytest.raises(ValueError):
        backend.save_evidence(_evidence("ev-1", day=5))
    assert backend.get_evidence("ev-1") == e1


def test_mastery_history_append_only_and_exact(backend):
    r1 = _mastery("mr-1", day=0, level=MasteryLevel.BEGINNER)
    r2 = _mastery("mr-2", day=3, level=MasteryLevel.INTERMEDIATE,
                  assertion=True, asserted_by="quiz-agent", reason="assert ✓")
    backend.save_mastery_record(r1)
    backend.save_mastery_record(r2)

    assert backend.get_mastery_history("learner-1", "topic-1") == [r1, r2]
    assert backend.get_mastery_history("learner-1", "other") == []
    assert backend.get_current_mastery("learner-1", "topic-1") == r2
    assert backend.get_current_mastery("learner-1", "other") is None

    with pytest.raises(ValueError):
        backend.save_mastery_record(_mastery("mr-1", day=9))
    assert backend.get_mastery_history("learner-1", "topic-1") == [r1, r2]


def test_override_crud_and_ordering(backend):
    o1 = _override("ovr-1", day=0, scope="topic-1")
    o2 = _override("ovr-2", day=1, scope=None, level=MasteryLevel.BEGINNER)
    backend.save_override(o1)
    backend.save_override(o2)

    listed = backend.list_overrides("learner-1")
    assert [o.id for o in listed] == ["ovr-1", "ovr-2"]  # by expires_at
    assert listed[0] == o1
    assert backend.list_overrides("other-learner") == []

    # replace semantics for same id
    o1b = _override("ovr-1", day=0, scope="topic-1",
                    level=MasteryLevel.BEGINNER)
    backend.save_override(o1b)
    assert [o.id for o in backend.list_overrides("learner-1")] == [
        "ovr-1", "ovr-2",
    ]

    backend.delete_override("ovr-1")
    assert [o.id for o in backend.list_overrides("learner-1")] == ["ovr-2"]
    # deleting a missing override is a no-op, not an error
    backend.delete_override("ovr-1")
    backend.delete_override("ghost")


def test_records_survive_reopen(make_backend, tmp_path, monkeypatch):
    """Durability: records written by one instance are visible to the next."""
    first = make_backend(tmp_path, monkeypatch)
    first.save_learner(Learner(id="l1", name="Sam"))
    first.save_preferences("l1", LearnerPreferences("visual", "steady"))
    first.save_node(_node("space-1", NodeLevel.SPACE, parent=None))
    first.save_evidence(_evidence("ev-1"))
    first.save_mastery_record(_mastery("mr-1"))
    first.save_override(_override("ovr-1"))
    first.save_plan(
        LearningPlan(id="p1", learner_id="l1", topic_ids=["t1"], created_at=BASE)
    )
    close = getattr(first, "close", None)
    if callable(close):
        close()

    second = make_backend(tmp_path, monkeypatch)
    assert second.get_learner("l1") == Learner(id="l1", name="Sam")
    assert second.get_preferences("l1") == LearnerPreferences("visual", "steady")
    assert second.get_node("space-1").name == "Space 1"
    assert second.get_evidence("ev-1") == _evidence("ev-1")
    assert second.get_current_mastery("learner-1", "topic-1") == _mastery("mr-1")
    assert [o.id for o in second.list_overrides("learner-1")] == ["ovr-1"]
    assert second.get_plan("p1").topic_ids == ["t1"]
