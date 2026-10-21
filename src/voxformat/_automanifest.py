"""Auto-manifest sync for ``add()`` / ``remove()`` (mirrors ``VoxFile+AutoManifest.swift``)."""

from __future__ import annotations

from typing import Any

from .entry import VoxEntry
from .manifest import EMBEDDING_FIELD_ORDER, REFERENCE_FIELD_ORDER, derive_embedding_key, ordered


def _str(meta: dict[str, Any], key: str) -> str | None:
    v = meta.get(key)
    return v if isinstance(v, str) else None


class AutoManifestMixin:
    manifest: dict[str, Any]

    def _add_manifest_entry(self, entry: VoxEntry) -> None:
        if entry.path.startswith("reference/"):
            self._add_reference_audio_entry(entry)
        elif entry.path.startswith("embeddings/"):
            self._add_embedding_entry(entry)

    def _remove_manifest_entry(self, path: str) -> None:
        if path.startswith("reference/"):
            ref = self.manifest.get("reference_audio")
            if isinstance(ref, list):
                ref[:] = [r for r in ref if not (isinstance(r, dict) and r.get("file") == path)]
                if not ref:
                    del self.manifest["reference_audio"]
        elif path.startswith("embeddings/"):
            emb = self.manifest.get("embeddings")
            if isinstance(emb, dict):
                for key in [k for k, v in emb.items() if isinstance(v, dict) and v.get("file") == path]:
                    del emb[key]
                if not emb:
                    del self.manifest["embeddings"]

    def _add_reference_audio_entry(self, entry: VoxEntry) -> None:
        meta = entry.metadata
        duration = meta.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            duration = None
        ref = ordered(
            {
                "file": entry.path,
                "transcript": _str(meta, "transcript") or "",
                "language": _str(meta, "language"),
                "duration_seconds": duration,
                "context": _str(meta, "context"),
                "model": _str(meta, "model"),
                "engine": _str(meta, "engine"),
            },
            REFERENCE_FIELD_ORDER,
        )
        lst = self.manifest.get("reference_audio")
        if isinstance(lst, list):
            lst.append(ref)
        else:
            self.manifest["reference_audio"] = [ref]

    def _add_embedding_entry(self, entry: VoxEntry) -> None:
        meta = entry.metadata
        key = _str(meta, "key") or derive_embedding_key(entry.path)
        emb = ordered(
            {
                "model": _str(meta, "model") or "",
                "engine": _str(meta, "engine"),
                "file": entry.path,
                "format": _str(meta, "format"),
                "description": _str(meta, "description"),
                "language": _str(meta, "language"),
            },
            EMBEDDING_FIELD_ORDER,
        )
        if not isinstance(self.manifest.get("embeddings"), dict):
            self.manifest["embeddings"] = {}
        self.manifest["embeddings"][key] = emb
