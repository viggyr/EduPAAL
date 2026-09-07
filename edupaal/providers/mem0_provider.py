"""Mem0 as an EduPAAL memory provider.

``Mem0Provider`` implements the :class:`StorageBackend` protocol on top of
Mem0's open-source ``Memory``. Each EduPAAL entity is stored as **one Mem0
memory** whose text is the canonical JSON record (see
``edupaal.providers._records``), written with ``infer=False`` so Mem0 stores
it **verbatim** — no LLM extraction, no rewriting, no summarization.
Deterministic metadata (``edupaal_kind`` / ``edupaal_id`` / ``learner_id`` /
``node_id``) supports exact lookup; Mem0's vector search is never used for
authoritative reads.

Install: ``pip install "edupaal[mem0-provider]"`` (``mem0ai`` + ``fastembed``).

Configuration (environment):
    ``EDUPAAL_MEM0_QDRANT_PATH`` — local Qdrant directory
        (default ``~/.edupaal/mem0_qdrant``).
    ``EDUPAAL_MEM0_COLLECTION`` — Qdrant collection name (default ``edupaal``).
    ``EDUPAAL_MEM0_EMBEDDER`` — mem0 embedder provider (default ``fastembed``;
        any mem0 embedder works, e.g. ``openai`` with ``OPENAI_API_KEY`` set,
        or ``ollama`` with a local server).
    ``EDUPAAL_MEM0_EMBEDDER_MODEL`` — embedding model for fastembed
        (default ``BAAI/bge-small-en-v1.5``; downloaded once on first use).

Design notes and limitations (see README for the full comparison):

* The provider never calls the LLM: every write uses ``infer=False`` and
  every read uses ``get`` / ``get_all`` with metadata filters. Embeddings
  exist only because the vector store requires a vector per point; they are
  not consulted for correctness.
* Mem0 generates its own memory ids (UUIDs); the EduPAAL entity id lives in
  metadata and in the record text. Lookups scan the collection filtered by
  kind (and learner/node where applicable) — O(n) in the number of stored
  records, fine for a personalization store, not for millions of rows.
* ``MEM0_TELEMETRY`` defaults to off when this module is imported: EduPAAL
  records are learner data and must not leave the machine. (mem0 reads the
  variable at import time, so set it in the environment before importing
  mem0 directly if you bypass this module. An explicit
  ``MEM0_TELEMETRY=True`` is still honored.) With telemetry on, every
  ``Memory`` also opens a second embedded Qdrant client, which would block
  reopening the store in the same process.
* Append-only semantics are enforced by the provider: ``save_evidence`` and
  ``save_mastery_record`` raise ``ValueError`` on a duplicate id instead of
  writing a second copy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Must precede `import mem0`: mem0 reads MEM0_TELEMETRY at import time, and
# with telemetry enabled every Memory opens a second embedded Qdrant client
# and attempts PostHog uploads. setdefault keeps an explicit user choice.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from ..entities import (
    DynamicOverride,
    Evidence,
    KnowledgeNode,
    Learner,
    LearnerPreferences,
    LearningPlan,
    MasteryRecord,
    NodeLevel,
)
from . import _records as R

try:
    from mem0 import Memory as _Mem0Memory
except Exception:  # pragma: no cover - import error is surfaced on use
    _Mem0Memory = None

# All EduPAAL records live under one Mem0 user id; the learner is carried in
# metadata. This keeps global lookups (get_evidence / get_node by id) to a
# single filtered scan instead of one scan per learner.
_MEM0_USER_ID = "edupaal"

_META_KIND = "edupaal_kind"
_META_ID = "edupaal_id"
_META_LEARNER = "learner_id"
_META_NODE = "node_id"

_DEFAULT_COLLECTION = "edupaal"
_DEFAULT_EMBEDDER = "fastembed"
_DEFAULT_EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
# fastembed model -> vector dims, so the Qdrant collection matches.
_EMBEDDER_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
}

# Upper bound per metadata-filtered scan. Mem0's get_all takes top_k with no
# upper bound; scans are O(n) in the collection, which is the documented
# scaling limit of this provider.
_SCAN_LIMIT = 100_000


def _default_qdrant_path() -> str:
    return os.environ.get(
        "EDUPAAL_MEM0_QDRANT_PATH",
        str(Path.home() / ".edupaal" / "mem0_qdrant"),
    )


def build_default_memory(
    *,
    qdrant_path: Optional[str] = None,
    collection: Optional[str] = None,
    embedder: Optional[str] = None,
    embedder_model: Optional[str] = None,
    dims: Optional[int] = None,
) -> Any:
    """Build a local-first ``mem0.Memory`` for the provider.

    No API keys, no servers: Qdrant runs embedded on disk and ``fastembed``
    runs the embedding model locally (downloaded once on first use). The LLM
    entry is a never-called placeholder — the provider always writes with
    ``infer=False``.

    ``dims`` overrides the vector width for the Qdrant collection; it
    defaults from the known fastembed model map (384 for the default
    ``BAAI/bge-small-en-v1.5``). Pass it explicitly for any other model —
    a wrong width fails collection creation loudly rather than silently.
    """
    if _Mem0Memory is None:
        raise ImportError(
            "mem0ai is not installed. Install the provider extra with "
            '`pip install "edupaal[mem0-provider]"`.'
        )
    # NOTE: MEM0_TELEMETRY already defaults to False via the setdefault at
    # this module's top (it must be set before mem0 is imported); nothing
    # more to do here.

    embedder = embedder or os.environ.get("EDUPAAL_MEM0_EMBEDDER", _DEFAULT_EMBEDDER)
    collection = collection or os.environ.get(
        "EDUPAAL_MEM0_COLLECTION", _DEFAULT_COLLECTION
    )
    path = qdrant_path or _default_qdrant_path()

    embedder_config: Dict[str, Any] = {}
    width = dims  # explicit override wins
    if embedder == "fastembed":
        model = (
            embedder_model
            or os.environ.get("EDUPAAL_MEM0_EMBEDDER_MODEL", _DEFAULT_EMBEDDER_MODEL)
        )
        embedder_config["model"] = model
        if width is None:
            try:
                width = _EMBEDDER_DIMS[model]
            except KeyError:
                raise ValueError(
                    f"unknown vector width for fastembed model {model!r}; "
                    "pass dims= explicitly to build_default_memory"
                ) from None
    elif embedder == "openai":
        if width is None:
            width = 1536
    elif width is None:
        raise ValueError(
            f"unknown vector width for embedder {embedder!r}; "
            "pass dims= explicitly to build_default_memory"
        )

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": path,
                "collection_name": collection,
                "embedding_model_dims": width,
            },
        },
        "embedder": {"provider": embedder, "config": embedder_config},
        # Never invoked (all writes use infer=False); the placeholder key
        # only satisfies client construction without requiring OPENAI_API_KEY.
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": os.environ.get("OPENAI_API_KEY", "edupaal-unused"),
            },
        },
    }
    return _Mem0Memory.from_config(config)


class Mem0Provider:
    """``StorageBackend`` implemented on Mem0 (``mem0ai``).

    Pass a ready-made ``mem0.Memory`` to take full control of its
    configuration, or use :meth:`from_env` for the local-first default
    (embedded Qdrant + local embeddings, no keys, no servers).
    """

    def __init__(self, memory: Any = None, **kwargs: Any) -> None:
        if memory is None:
            memory = build_default_memory(**kwargs)
        self._memory = memory

    @classmethod
    def from_env(cls, **kwargs: Any) -> "Mem0Provider":
        """Build a provider from environment configuration (see module docs)."""
        return cls(build_default_memory(**kwargs))

    # -- internal helpers --

    def _metadata(
        self, kind: str, record_id: str, entity: Any = None
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {_META_KIND: kind, _META_ID: record_id}
        learner_id = getattr(entity, "learner_id", None)
        if learner_id:
            meta[_META_LEARNER] = learner_id
        node_id = getattr(entity, "node_id", None)
        if node_id:
            meta[_META_NODE] = node_id
        return meta

    def _scan(
        self,
        kind: str,
        learner_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """All raw memory dicts for a kind (+ optional learner/node filter)."""
        filters: Dict[str, Any] = {"user_id": _MEM0_USER_ID, _META_KIND: kind}
        if learner_id is not None:
            filters[_META_LEARNER] = learner_id
        if node_id is not None:
            filters[_META_NODE] = node_id
        result = self._memory.get_all(filters=filters, top_k=_SCAN_LIMIT)
        return result.get("results", [])

    def _find(self, kind: str, record_id: str) -> Optional[Dict[str, Any]]:
        for item in self._scan(kind):
            if (item.get("metadata") or {}).get(_META_ID) == record_id:
                return item
        return None

    def _read_item(self, item: Dict[str, Any]) -> Any:
        record = json.loads(item["memory"])
        return R.from_record(record)

    def _write_new(
        self, kind: str, record_id: str, record: Dict[str, Any], entity: Any
    ) -> None:
        """Write one record verbatim, then verify the round trip."""
        text = json.dumps(record, sort_keys=True)
        res = self._memory.add(
            [{"role": "user", "content": text}],
            user_id=_MEM0_USER_ID,
            metadata=self._metadata(kind, record_id, entity),
            infer=False,
        )
        mem_id = res["results"][0]["id"]
        # Read-after-write verification: the authoritative record must come
        # back byte-identical; fail loudly otherwise.
        stored = self._memory.get(mem_id)
        if stored is None or stored.get("memory") != text:
            raise RuntimeError(
                f"mem0 write verification failed for {kind}:{record_id}"
            )

    def _replace(self, kind: str, record_id: str, record: Dict[str, Any], entity: Any) -> None:
        existing = self._find(kind, record_id)
        if existing is not None:
            self._memory.delete(existing["id"])
        self._write_new(kind, record_id, record, entity)

    def _write_append_only(
        self, kind: str, record_id: str, record: Dict[str, Any], entity: Any
    ) -> None:
        if self._find(kind, record_id) is not None:
            label = "evidence" if kind == R.KIND_EVIDENCE else "mastery record"
            raise ValueError(f"duplicate {label} id: {record_id}")
        self._write_new(kind, record_id, record, entity)

    # -- learners & preferences --

    def save_learner(self, learner: Learner) -> None:
        kind, record_id, record = R.to_record(learner)
        self._replace(kind, record_id, record, learner)

    def get_learner(self, learner_id: str) -> Optional[Learner]:
        item = self._find(R.KIND_LEARNER, learner_id)
        return self._read_item(item) if item else None

    def save_preferences(self, learner_id: str, prefs: LearnerPreferences) -> None:
        kind, record_id, record = R.preferences_to_record(learner_id, prefs)
        self._replace(kind, record_id, record, prefs)

    def get_preferences(self, learner_id: str) -> Optional[LearnerPreferences]:
        item = self._find(R.KIND_PREFERENCES, f"preferences:{learner_id}")
        if not item:
            return None
        _, prefs = self._read_item(item)
        return prefs

    # -- knowledge graph --

    def save_node(self, node: KnowledgeNode) -> None:
        kind, record_id, record = R.to_record(node)
        self._replace(kind, record_id, record, node)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        item = self._find(R.KIND_NODE, node_id)
        return self._read_item(item) if item else None

    def list_nodes(self, level: Optional[NodeLevel] = None) -> List[KnowledgeNode]:
        nodes = [self._read_item(item) for item in self._scan(R.KIND_NODE)]
        if level is not None:
            nodes = [n for n in nodes if n.level == level]
        return sorted(nodes, key=lambda n: n.id)

    def save_prerequisite(self, topic_id: str, prereq_id: str) -> None:
        kind, record_id, record = R.prerequisite_to_record(topic_id, prereq_id)
        if self._find(kind, record_id) is None:  # INSERT OR IGNORE semantics
            self._write_new(kind, record_id, record, None)

    def get_prerequisites(self, topic_id: str) -> List[str]:
        prereqs = []
        for item in self._scan(R.KIND_PREREQUISITE):
            t, p = self._read_item(item)
            if t == topic_id:
                prereqs.append(p)
        return sorted(prereqs)

    # -- plans --

    def save_plan(self, plan: LearningPlan) -> None:
        kind, record_id, record = R.to_record(plan)
        self._replace(kind, record_id, record, plan)

    def get_plan(self, plan_id: str) -> Optional[LearningPlan]:
        item = self._find(R.KIND_PLAN, plan_id)
        return self._read_item(item) if item else None

    def get_plan_for_learner(self, learner_id: str) -> Optional[LearningPlan]:
        plans = [
            self._read_item(item)
            for item in self._scan(R.KIND_PLAN, learner_id=learner_id)
        ]
        if not plans:
            return None
        return max(plans, key=lambda p: p.version)

    # -- evidence (append-only) --

    def save_evidence(self, evidence: Evidence) -> None:
        kind, record_id, record = R.to_record(evidence)
        self._write_append_only(kind, record_id, record, evidence)

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        item = self._find(R.KIND_EVIDENCE, evidence_id)
        return self._read_item(item) if item else None

    def list_evidence(
        self, learner_id: str, node_id: Optional[str] = None
    ) -> List[Evidence]:
        items = self._scan(R.KIND_EVIDENCE, learner_id=learner_id, node_id=node_id)
        evidence = [self._read_item(item) for item in items]
        return sorted(evidence, key=lambda e: (e.occurred_at, e.id))

    # -- mastery records (append-only history) --

    def save_mastery_record(self, record: MasteryRecord) -> None:
        kind, record_id, rec = R.to_record(record)
        self._write_append_only(kind, record_id, rec, record)

    def get_mastery_history(
        self, learner_id: str, node_id: str
    ) -> List[MasteryRecord]:
        items = self._scan(
            R.KIND_MASTERY_RECORD, learner_id=learner_id, node_id=node_id
        )
        records = [self._read_item(item) for item in items]
        # updated_at, then id as a deterministic tiebreak (SQLite uses rowid).
        return sorted(records, key=lambda r: (r.updated_at, r.id))

    def get_current_mastery(
        self, learner_id: str, node_id: str
    ) -> Optional[MasteryRecord]:
        history = self.get_mastery_history(learner_id, node_id)
        return history[-1] if history else None

    # -- overrides --

    def save_override(self, override: DynamicOverride) -> None:
        kind, record_id, record = R.to_record(override)
        self._replace(kind, record_id, record, override)

    def list_overrides(self, learner_id: str) -> List[DynamicOverride]:
        items = self._scan(R.KIND_OVERRIDE, learner_id=learner_id)
        overrides = [self._read_item(item) for item in items]
        return sorted(overrides, key=lambda o: (o.expires_at, o.id))

    def delete_override(self, override_id: str) -> None:
        item = self._find(R.KIND_OVERRIDE, override_id)
        if item is not None:
            self._memory.delete(item["id"])

    def close(self) -> None:
        client = getattr(getattr(self._memory, "vector_store", None), "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
