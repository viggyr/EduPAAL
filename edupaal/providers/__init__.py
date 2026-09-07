"""Optional memory providers for EduPAAL.

Each provider implements the :class:`edupaal.store.StorageBackend` protocol
on top of a third-party memory framework, so deployments can swap the
storage engine without touching framework logic:

* :class:`Mem0Provider` — Mem0 (``pip install "edupaal[mem0-provider]"``).
* :class:`MemOSProvider` — MemOS (``pip install "edupaal[memos-provider]"``).

Both persist every entity as a canonical JSON record stored **verbatim**
(no LLM rewriting), with read-after-write verification, so they give the
same exactness guarantees as the default ``SQLiteBackend``. See each
provider's module docstring and the README's "Storage providers" section
for configuration, trade-offs, and limitations.

Imports are lazy: importing this package never requires the optional
dependencies — they are only needed when a provider is instantiated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .mem0_provider import Mem0Provider
    from .memos_provider import MemOSProvider

__all__ = ["Mem0Provider", "MemOSProvider"]


def __getattr__(name: str) -> Any:
    if name == "Mem0Provider":
        from .mem0_provider import Mem0Provider

        return Mem0Provider
    if name == "MemOSProvider":
        from .memos_provider import MemOSProvider

        return MemOSProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
