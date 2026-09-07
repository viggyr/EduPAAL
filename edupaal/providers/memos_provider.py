"""MemOS as an EduPAAL memory provider.

``MemOSProvider`` implements the :class:`StorageBackend` protocol on top of
MemOS's textual memory. Each EduPAAL entity is stored as **one
``TextualMemoryItem``** whose ``memory`` text is the canonical JSON record
(see ``edupaal.providers._records``), written **directly** through the
memory's ``add`` / ``get`` / ``get_all`` / ``delete`` CRUD — the provider
never calls MemOS's LLM extraction (``extract()``) or its semantic
``search()`` for authoritative reads, so records round-trip byte-identical.

Install: ``pip install "edupaal[memos-provider]"`` (``MemoryOS``).

Configuration (environment):
    ``EDUPAAL_MEMOS_DIR`` — directory for the memory file
        (default ``~/.edupaal/memos``).

Design notes and limitations (see README for the full comparison):

* The provider uses MemOS's ``NaiveTextMemory`` — the framework's direct
  CRUD text memory — rather than the high-level ``MOS.add()`` pipeline,
  because ``MOS.add()`` runs LLM extraction/dedup, which is lossy and
  unsuitable for authoritative structured records. Deployments that want
  MemOS's semantic search or scheduler can layer them on top; the
  authoritative store stays exact.
* ``NaiveTextMemory`` persists via ``dump()``/``load()`` to a JSON file.
  The provider loads on construction and dumps **write-through after every
  mutation**, so records survive restarts. This is MemOS's own persistence
  mechanism for this memory type.
* ``TextualMemoryItem.id`` must be a UUID, so the provider generates one
  per record; the EduPAAL entity id lives in metadata (``edupaal_id``) and
  in the record text. Lookups scan the in-memory list — O(n), fine for a
  personalization store.
* Append-only semantics are enforced by the provider: ``save_evidence``
  and ``save_mastery_record`` raise ``ValueError`` on a duplicate id.
* The ``extractor_llm`` config is required by ``NaiveTextMemory`` but never
  invoked by this provider; it defaults to a local Ollama placeholder.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    from memos.memories.textual.naive import NaiveTextMemory
    from memos.configs.memory import NaiveTextMemoryConfig
except Exception:  # pragma: no cover - import error is surfaced on use
    NaiveTextMemory = None
    NaiveTextMemoryConfig = None

_META_KIND = "edupaal_kind"
_META_ID = "edupaal_id"
_META_LEARNER = "learner_id"
_META_NODE = "node_id"

_DEFAULT_FILENAME = "edupaal_textual_memory.json"


def _default_dir() -> str:
    return os.environ.get(
        "EDUPAAL_MEMOS_DIR", str(Path.home() / ".edupaal" / "memos")
    )


def build_default_memory(dir: Optional[str] = None) -> Any:
    """Build a ``NaiveTextMemory`` for the provider, loading prior state.

    The extractor LLM is a never-called placeholder: this provider only
    uses the memory's direct CRUD, never ``extract()``.
    """
    if NaiveTextMemory is None:
        raise ImportError(
            "MemoryOS is not installed. Install the provider extra with "
            '`pip install "edupaal[memos-provider]"`.'
        )
    directory = dir or _default_dir()
    config = NaiveTextMemoryConfig(
        memory_filename=_DEFAULT_FILENAME,
        # Never invoked: the provider uses the memory's direct CRUD, never
        # extract(). The placeholder only satisfies config validation.
        extractor_llm={
            "backend": "ollama",
            "config": {"model_name_or_path": "llama3.1:latest"},
        },
    )
    memory = NaiveTextMemory(config)
    # Load prior state if this directory was used before; a fresh directory
    # simply starts empty.
    if Path(directory, _DEFAULT_FILENAME).exists():
        memory.load(directory)
    return memory


class MemOSProvider:
    """``StorageBackend`` implemented on MemOS (``MemoryOS``).

    Pass a ready-made ``NaiveTextMemory`` (or any ``BaseTextMemory`` with
    the same CRUD surface) to take full control, or use :meth:`from_env`
    for the file-backed default.
    """

    def __init__(self, memory: Any = None, *, dir: Optional[str] = None) -> None:
        if memory is None:
            self._dir = dir or _default_dir()
            memory = build_default_memory(self._dir)
        else:
            # A caller-supplied memory manages its own persistence; without
            # an explicit dir there is nothing to dump to.
            self._dir = dir
            if self._dir is not None:
                # Load prior state from the provider's persistence dir, but
                # only into a fresh memory: MemOS's load() appends, so
                # loading into a non-empty memory would duplicate records.
                filename = getattr(
                    getattr(memory, "config", None),
                    "memory_filename",
                    _DEFAULT_FILENAME,
                )
                if (
                    Path(self._dir, filename).exists()
                    and len(memory.get_all()) == 0
                ):
                    memory.load(self._dir)
        self._memory = memory

    @classmethod
    def from_env(cls, **kwargs: Any) -> "MemOSProvider":
        """Build a provider from environment configuration (see module docs)."""
        return cls(**kwargs)

    # -- internal helpers --

    def _persist(self) -> None:
        if self._dir is not None:
            self._memory.dump(self._dir)

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
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """``(memory_uuid, record_dict)`` pairs for a kind (+ filters)."""
        out = []
        for item in self._memory.get_all():
            meta = item.metadata.model_dump() if hasattr(item.metadata, "model_dump") else dict(item.metadata or {})
            if meta.get(_META_KIND) != kind:
                continue
            if learner_id is not None and meta.get(_META_LEARNER) != learner_id:
                continue
            if node_id is not None and meta.get(_META_NODE) != node_id:
                continue
            record = json.loads(item.memory)
            out.append((item.id, record))
        return out

    def _find(self, kind: str, record_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        for mem_id, record in self._scan(kind):
            if record.get("id") == record_id:
                return mem_id, record
        return None

    def _write_new(
        self, kind: str, record_id: str, record: Dict[str, Any], entity: Any
    ) -> None:
        text = json.dumps(record, sort_keys=True)
        self._memory.add(
            [
                {
                    "id": str(uuid.uuid4()),
                    "memory": text,
                    "metadata": self._metadata(kind, record_id, entity),
                }
            ]
        )
        self._persist()
        # Read-after-write verification: the authoritative record must come
        # back byte-identical; fail loudly otherwise.
        found = self._find(kind, record_id)
        if found is None or json.dumps(found[1], sort_keys=True) != text:
            raise RuntimeError(
                f"memos write verification failed for {kind}:{record_id}"
            )

    def _replace(
        self, kind: str, record_id: str, record: Dict[str, Any], entity: Any
    ) -> None:
        existing = self._find(kind, record_id)
        if existing is not None:
            self._memory.delete([existing[0]])
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
        found = self._find(R.KIND_LEARNER, learner_id)
        return R.from_record(found[1]) if found else None

    def save_preferences(self, learner_id: str, prefs: LearnerPreferences) -> None:
        kind, record_id, record = R.preferences_to_record(learner_id, prefs)
        self._replace(kind, record_id, record, prefs)

    def get_preferences(self, learner_id: str) -> Optional[LearnerPreferences]:
        found = self._find(R.KIND_PREFERENCES, f"preferences:{learner_id}")
        if not found:
            return None
        _, prefs = R.from_record(found[1])
        return prefs

    # -- knowledge graph --

    def save_node(self, node: KnowledgeNode) -> None:
        kind, record_id, record = R.to_record(node)
        self._replace(kind, record_id, record, node)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        found = self._find(R.KIND_NODE, node_id)
        return R.from_record(found[1]) if found else None

    def list_nodes(self, level: Optional[NodeLevel] = None) -> List[KnowledgeNode]:
        nodes = [R.from_record(record) for _, record in self._scan(R.KIND_NODE)]
        if level is not None:
            nodes = [n for n in nodes if n.level == level]
        return sorted(nodes, key=lambda n: n.id)

    def save_prerequisite(self, topic_id: str, prereq_id: str) -> None:
        kind, record_id, record = R.prerequisite_to_record(topic_id, prereq_id)
        if self._find(kind, record_id) is None:  # INSERT OR IGNORE semantics
            self._write_new(kind, record_id, record, None)

    def get_prerequisites(self, topic_id: str) -> List[str]:
        prereqs = []
        for _, record in self._scan(R.KIND_PREREQUISITE):
            t, p = R.from_record(record)
            if t == topic_id:
                prereqs.append(p)
        return sorted(prereqs)

    # -- plans --

    def save_plan(self, plan: LearningPlan) -> None:
        kind, record_id, record = R.to_record(plan)
        self._replace(kind, record_id, record, plan)

    def get_plan(self, plan_id: str) -> Optional[LearningPlan]:
        found = self._find(R.KIND_PLAN, plan_id)
        return R.from_record(found[1]) if found else None

    def get_plan_for_learner(self, learner_id: str) -> Optional[LearningPlan]:
        plans = [
            R.from_record(record)
            for _, record in self._scan(R.KIND_PLAN, learner_id=learner_id)
        ]
        if not plans:
            return None
        return max(plans, key=lambda p: p.version)

    # -- evidence (append-only) --

    def save_evidence(self, evidence: Evidence) -> None:
        kind, record_id, record = R.to_record(evidence)
        self._write_append_only(kind, record_id, record, evidence)

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        found = self._find(R.KIND_EVIDENCE, evidence_id)
        return R.from_record(found[1]) if found else None

    def list_evidence(
        self, learner_id: str, node_id: Optional[str] = None
    ) -> List[Evidence]:
        evidence = [
            R.from_record(record)
            for _, record in self._scan(
                R.KIND_EVIDENCE, learner_id=learner_id, node_id=node_id
            )
        ]
        return sorted(evidence, key=lambda e: (e.occurred_at, e.id))

    # -- mastery records (append-only history) --

    def save_mastery_record(self, record: MasteryRecord) -> None:
        kind, record_id, rec = R.to_record(record)
        self._write_append_only(kind, record_id, rec, record)

    def get_mastery_history(
        self, learner_id: str, node_id: str
    ) -> List[MasteryRecord]:
        records = [
            R.from_record(record)
            for _, record in self._scan(
                R.KIND_MASTERY_RECORD, learner_id=learner_id, node_id=node_id
            )
        ]
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
        overrides = [
            R.from_record(record)
            for _, record in self._scan(R.KIND_OVERRIDE, learner_id=learner_id)
        ]
        return sorted(overrides, key=lambda o: (o.expires_at, o.id))

    def delete_override(self, override_id: str) -> None:
        found = self._find(R.KIND_OVERRIDE, override_id)
        if found is not None:
            self._memory.delete([found[0]])
            self._persist()

    def close(self) -> None:
        self._persist()
