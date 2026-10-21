"""Silent manifest upgrade on read (mirrors Swift ``VoxMigrator``).

Behaviour (identical to Swift):

* If ``embeddings`` is present and non-empty, only ``vox_version`` is bumped.
* If there are no files under ``embeddings/`` in the archive, only the version is bumped.
* Otherwise ``embeddings`` entries are inferred from the binaries, using hints from
  ``extensions`` (string values that start with ``embeddings/`` or live under the keys
  ``clone_prompt`` / ``embedding_file``) where available.

Swift iterates ``Set``/``Dictionary`` values in undefined order; Python iterates the
binaries in sorted path order and extension hints in manifest order, so results are
deterministic. When two binaries derive the same key (e.g. two files in one directory)
the later one wins, exactly as in Swift.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

from .entry import path_extension, split_extension
from .manifest import CURRENT_VERSION


def _derive_key(relative_path: str) -> str:
    components = [c for c in relative_path.split("/") if c]
    dirs = components[:-1]
    if not dirs:
        return split_extension(relative_path)[0]
    return "-".join(dirs)


def _derive_engine(relative_path: str) -> str | None:
    components = [c for c in relative_path.split("/") if c]
    return components[0] if len(components) > 1 else None


def _derive_format(relative_path: str) -> str | None:
    ext = path_extension(relative_path).lower()
    return ext or None


def _extension_hints(extensions: Any) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if not isinstance(extensions, dict):
        return hints
    for namespace, value in extensions.items():
        if not isinstance(value, dict):
            continue
        model = value.get("model") if isinstance(value.get("model"), str) else None
        for inner_key, inner_value in value.items():
            if not isinstance(inner_value, str):
                continue
            if inner_value.startswith("embeddings/") or inner_key in ("clone_prompt", "embedding_file"):
                hints.append(
                    {"key": f"{namespace}-{inner_key}", "engine": namespace, "model": model, "file": inner_value}
                )
    return hints


def migrate_manifest(manifest: dict[str, Any], embedding_paths: Iterable[str]) -> dict[str, Any]:
    """Return an upgraded *copy* of ``manifest``.

    :param embedding_paths: archive paths relative to ``embeddings/`` that hold data.
    """
    result = copy.deepcopy(manifest)
    existing = result.get("embeddings")
    if isinstance(existing, dict) and existing:
        result["vox_version"] = CURRENT_VERSION
        return result

    rel_paths = sorted(p for p in embedding_paths if p)
    if not rel_paths:
        result["vox_version"] = CURRENT_VERSION
        return result

    hints = _extension_hints(result.get("extensions"))
    entries: dict[str, dict[str, Any]] = {}
    for rel in rel_paths:
        full = f"embeddings/{rel}"
        hint = next((h for h in hints if h["file"] in (full, rel)), None)
        fmt = _derive_format(rel)
        if hint is not None:
            entry = {
                "model": hint["model"] or _derive_key(rel),
                "engine": hint["engine"],
                "file": full,
                "format": fmt,
                "description": "Migrated from v0.1.0 extensions",
            }
            key = hint["key"]
        else:
            entry = {
                "model": _derive_key(rel),
                "engine": _derive_engine(rel),
                "file": full,
                "format": fmt,
                "description": "Inferred from archive binary (v0.1.0 migration)",
            }
            key = _derive_key(rel)
        entries[key] = {k: v for k, v in entry.items() if v is not None}

    if entries:
        result["embeddings"] = entries
    else:
        result.pop("embeddings", None)
    result["vox_version"] = CURRENT_VERSION
    return result
