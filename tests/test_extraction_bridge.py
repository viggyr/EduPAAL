"""Tests for the Mem0 extraction bridge (edupaal/extraction.py).

The bridge is the one place that runs LLM extraction. These tests use a
fake ``mem0.Memory`` — no network, no LLM — and assert the bridge's
contract: ``infer=True``, fail-loud on empty input, learner-scoped
namespaces, and idempotent upsert ids.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from edupaal import ExtractedMemory, SQLiteBackend
from edupaal.extraction import extract_and_store, extract_memories


class FakeMemory:
    """Stand-in for mem0.Memory: records how add() was called."""

    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results
        self.calls: List[Dict[str, Any]] = []

    def add(self, messages, user_id=None, infer=False, **kwargs):
        self.calls.append(
            {"messages": messages, "user_id": user_id, "infer": infer}
        )
        return {"results": self._results}


def _messages():
    return [
        {"role": "user", "content": "can you show me with a worked example?"},
        {"role": "assistant", "content": "sure — here is a worked example ✓"},
    ]


def test_extraction_calls_mem0_with_infer_true():
    fake = FakeMemory([{"memory": "prefers worked examples"}])
    out = extract_memories(fake, "learner-1", _messages())
    assert len(fake.calls) == 1
    assert fake.calls[0]["infer"] is True
    assert fake.calls[0]["messages"] == _messages()
    assert len(out) == 1
    assert out[0].content == "prefers worked examples"
    assert out[0].learner_id == "learner-1"
    assert out[0].kind == "observed_preference"
    assert out[0].evidence_count == 2
    assert out[0].provenance == ["message-0", "message-1"]
    assert out[0].confidence is None  # never invented


def test_empty_messages_fail_loud():
    fake = FakeMemory([])
    with pytest.raises(ValueError, match="messages must not be empty"):
        extract_memories(fake, "learner-1", [])
    assert fake.calls == []  # the LLM is never invoked


def test_blank_results_are_skipped():
    fake = FakeMemory([{"memory": ""}, {"memory": "keeps this one"}])
    out = extract_memories(fake, "learner-1", _messages())
    assert [m.content for m in out] == ["keeps this one"]


def test_namespaces_are_learner_scoped_by_default():
    fake = FakeMemory([{"memory": "x"}])
    extract_memories(fake, "alice", _messages())
    extract_memories(fake, "bob", _messages())
    assert [c["user_id"] for c in fake.calls] == [
        "edupaal:alice",
        "edupaal:bob",
    ]


def test_explicit_user_id_is_respected():
    fake = FakeMemory([{"memory": "x"}])
    extract_memories(fake, "alice", _messages(), user_id="shared-cohort-7")
    assert fake.calls[0]["user_id"] == "shared-cohort-7"


def test_rerun_over_same_messages_is_idempotent(tmp_path):
    """Same inputs -> same ids -> backend upsert replaces, not duplicates."""
    backend = SQLiteBackend(tmp_path / "x.db")
    first = FakeMemory([{"memory": "prefers worked examples"}])
    second = FakeMemory([{"memory": "prefers worked examples, v2"}])
    extract_and_store(backend, first, "learner-1", _messages())
    extract_and_store(backend, second, "learner-1", _messages())
    stored = backend.list_extracted_memories("learner-1")
    assert len(stored) == 1
    assert stored[0].content == "prefers worked examples, v2"


def test_different_learners_get_different_ids():
    fake = FakeMemory([{"memory": "x"}])
    a = extract_memories(fake, "alice", _messages())
    b = extract_memories(fake, "bob", _messages())
    assert a[0].id != b[0].id


def test_explicit_provenance_is_kept():
    fake = FakeMemory([{"memory": "x"}])
    out = extract_memories(
        fake, "learner-1", _messages(), provenance=["chat-42", "chat-43"]
    )
    assert out[0].provenance == ["chat-42", "chat-43"]


def test_extracted_memory_entity_validation():
    with pytest.raises(ValueError):
        ExtractedMemory(learner_id="l", kind="k", content="")  # empty content
