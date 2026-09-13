"""LLM extraction bridge for the extracted layer (ADR 0001).

The extracted layer stores *already-extracted* memories
(:class:`ExtractedMemory`); every storage backend persists them verbatim.
This module is the one place that *runs* extraction: it drives Mem0's LLM
extraction (``Memory.add(..., infer=True)``) over interaction messages and
converts the results into :class:`ExtractedMemory` entities.

Advisory-only, always: extracted memories tune *how* a vertical presents
material, never *what* the learner's mastery is. Nothing in
``edupaal/mastery.py`` or ``edupaal/retrieval.py`` reads them
(see ``tests/test_extracted_advisory.py``).

``mem0ai`` is required only when these functions are called — importing this
module never imports mem0. Unlike :class:`Mem0Provider` (which writes with
``infer=False`` and never calls the LLM), extraction genuinely needs one:
pass a ``mem0.Memory`` configured with a real LLM.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .entities import ExtractedMemory
from .store import StorageBackend


def _default_user_id(learner_id: str) -> str:
    """Learner-scoped Mem0 namespace: learners must never share one."""
    return f"edupaal:{learner_id}"


def _extraction_id(
    learner_id: str,
    kind: str,
    node_id: Optional[str],
    user_id: str,
    messages: List[Dict[str, str]],
    index: int,
) -> str:
    """Deterministic id for one extracted item.

    Derived from the extraction *inputs* (not the LLM's output text), so
    re-running extraction over the same interactions yields the same ids and
    the backend upsert replaces the previous inference instead of
    duplicating it. ``index`` disambiguates the several items one run can
    produce. If the extractor returns a different number of items on a
    re-run, stale items from the earlier run linger — delete by
    ``(learner_id, kind, node_id)`` for a full refresh in that case.
    """
    h = hashlib.sha1()
    h.update(learner_id.encode("utf-8"))
    h.update(b"\x00" + kind.encode("utf-8"))
    h.update(b"\x00" + (node_id or "").encode("utf-8"))
    h.update(b"\x00" + user_id.encode("utf-8"))
    for message in messages:
        h.update(b"\x00" + str(message.get("role", "")).encode("utf-8"))
        h.update(b"\x01" + str(message.get("content", "")).encode("utf-8"))
    return f"xm_{h.hexdigest()[:16]}_{index}"


def extract_memories(
    memory: Any,
    learner_id: str,
    messages: List[Dict[str, str]],
    *,
    kind: str = "observed_preference",
    node_id: Optional[str] = None,
    user_id: Optional[str] = None,
    provenance: Optional[List[str]] = None,
) -> List[ExtractedMemory]:
    """Run Mem0 extraction over ``messages`` and return the new memories.

    ``messages`` are ``{"role": ..., "content": ...}`` dicts (one
    interaction transcript). ``kind`` labels what the caller asked Mem0 to
    extract — ``"learning_style"``, ``"observed_preference"``, or any
    free-form label the vertical interprets. ``provenance`` defaults to one
    label per message (``"message-0"``, ...) so the inference stays
    traceable; pass explicit interaction ids when you have them.
    ``evidence_count`` is the number of messages the inference was drawn
    from — thin-evidence caution (ADR 0001) starts here.

    ``user_id`` is the Mem0 namespace the extraction runs in. It defaults
    to ``edupaal:{learner_id}`` so learners never share extraction context;
    pass an explicit id only when you deliberately want a shared namespace.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    user_id = user_id or _default_user_id(learner_id)
    result = memory.add(messages, user_id=user_id, infer=True)
    provenance = (
        list(provenance)
        if provenance is not None
        else [f"message-{i}" for i in range(len(messages))]
    )
    extracted: List[ExtractedMemory] = []
    for index, item in enumerate(result.get("results", [])):
        text = item.get("memory", "")
        if not text:
            continue
        extracted.append(
            ExtractedMemory(
                id=_extraction_id(
                    learner_id, kind, node_id, user_id, messages, index
                ),
                learner_id=learner_id,
                kind=kind,
                content=text,
                provenance=provenance,
                node_id=node_id,
                evidence_count=len(messages),
            )
        )
    return extracted


def extract_and_store(
    backend: StorageBackend,
    memory: Any,
    learner_id: str,
    messages: List[Dict[str, str]],
    *,
    kind: str = "observed_preference",
    node_id: Optional[str] = None,
    user_id: Optional[str] = None,
    provenance: Optional[List[str]] = None,
) -> List[ExtractedMemory]:
    """Extract memories from ``messages`` and persist them on ``backend``.

    Saves are upserts with deterministic ids: re-running extraction over
    the same interactions replaces the previous inference rather than
    duplicating it.
    """
    extracted = extract_memories(
        memory,
        learner_id,
        messages,
        kind=kind,
        node_id=node_id,
        user_id=user_id,
        provenance=provenance,
    )
    for mem in extracted:
        backend.save_extracted_memory(mem)
    return extracted
