"""EduPAAL: a portable memory-management skill for personalized learning."""

from .entities import (
    MASTERY_SCORES,
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
)
from .graph import KnowledgeGraph
from .mastery import ASSERTION_VERSION, HEURISTIC_VERSION, MasteryEngine
from .retrieval import Retriever
from .seed import build_seed_graph
from .skill import EduPAALSkill
from .store import SQLiteBackend, StorageBackend

__all__ = [
    "MASTERY_SCORES",
    "ASSERTION_VERSION",
    "HEURISTIC_VERSION",
    "DynamicOverride",
    "EduPAALSkill",
    "Evidence",
    "KnowledgeGraph",
    "Learner",
    "LearnerPreferences",
    "LearningPlan",
    "MasteryEngine",
    "MasteryLevel",
    "MasteryParams",
    "MasteryRecord",
    "NodeLevel",
    "KnowledgeNode",
    "Retriever",
    "SQLiteBackend",
    "StorageBackend",
    "build_seed_graph",
    # Optional memory providers (edupaal/providers): Mem0Provider, MemOSProvider.
    # Imported lazily so the core package stays dependency-free; importing
    # edupaal itself never requires mem0ai or MemoryOS.
    "Mem0Provider",
    "MemOSProvider",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy optional-provider exports: `from edupaal import Mem0Provider`
    # works when the corresponding extra is installed, without making
    # `import edupaal` depend on it.
    if name in ("Mem0Provider", "MemOSProvider"):
        from . import providers

        return getattr(providers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
