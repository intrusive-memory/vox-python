"""Manifest helpers.

The manifest is kept as a plain ``dict`` (the parsed ``manifest.json``) so that unknown
fields, unknown ``extensions`` and the author's key order survive a read/write round-trip.
The Swift reference implementation decodes into a typed struct and therefore *drops*
unknown fields; the Python implementation intentionally does not (see README).

This module provides the typed *views* and helpers that mirror the Swift types.
"""

from __future__ import annotations

import datetime as _dt
import posixpath
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from .entry import split_extension

#: Current VOX format version (Swift ``VoxFormat.currentVersion``).
CURRENT_VERSION = "0.4.0"

#: Reserved provider identifiers for ``manifest.extensions`` (spec "Extension Namespaces").
RESERVED_PROVIDERS: tuple[str, ...] = (
    "qwen3-tts",
    "apple",
    "elevenlabs",
    "openai",
    "coqui",
    "parler",
    "google",
    "azure",
    "mlx-audio",
)

VALID_GENDERS = frozenset({"male", "female", "nonbinary", "neutral"})
KNOWN_METHODS = frozenset({"designed", "synthesized", "cloned", "preset", "hybrid"})
CLONE_CONSENT = frozenset({"self", "granted"})

UUID_V4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

# Swift decodes `created` with JSONDecoder.DateDecodingStrategy.iso8601, i.e.
# ISO8601DateFormatter(.withInternetDateTime): no fractional seconds, Z or +hh:mm offset.
_ISO_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})$"
)

EMBEDDING_FIELD_ORDER = ("model", "engine", "file", "format", "description", "language")
REFERENCE_FIELD_ORDER = (
    "file",
    "transcript",
    "language",
    "duration_seconds",
    "context",
    "model",
    "engine",
)


def now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ`` (schema-conformant, no fractions)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso8601(value: Any) -> _dt.datetime | None:
    """Parse a timestamp the way Swift's ``.iso8601`` strategy does; ``None`` if invalid."""
    if not isinstance(value, str):
        return None
    m = _ISO_RE.match(value)
    if not m:
        return None
    tz = m.group(7)
    if tz == "Z":
        tzinfo = _dt.timezone.utc
    else:
        sign = -1 if tz[0] == "-" else 1
        digits = tz[1:].replace(":", "")
        tzinfo = _dt.timezone(sign * _dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
    try:
        return _dt.datetime(*(int(m.group(i)) for i in range(1, 7)), tzinfo=tzinfo)
    except ValueError:
        return None


def is_valid_uuid_v4(value: Any) -> bool:
    """Swift ``isValidUUIDv4``: lowercases first, then matches the v4 pattern."""
    return isinstance(value, str) and bool(UUID_V4_RE.match(value.lower()))


def new_manifest(name: str, description: str, **voice_fields: Any) -> dict[str, Any]:
    """A fresh manifest dict (Swift ``VoxFile(name:description:)``)."""
    voice: dict[str, Any] = {"name": name, "description": description}
    voice.update({k: v for k, v in voice_fields.items() if v is not None})
    return {
        "vox_version": CURRENT_VERSION,
        "id": str(uuid.uuid4()),
        "created": now_iso(),
        "voice": voice,
    }


def derive_embedding_key(path: str) -> str:
    """Swift ``deriveEmbeddingKey``.

    ``embeddings/qwen3-tts/0.6b/clone-prompt.bin`` -> ``qwen3-tts-0.6b-clone-prompt``;
    ``embeddings/qwen3-tts/0.6b/es/sample-audio.wav`` -> ``qwen3-tts-0.6b-es-sample-audio``.
    """
    stripped = path[len("embeddings/") :] if path.startswith("embeddings/") else path
    # Swift's split(separator:) omits empty subsequences.
    components = [c for c in stripped.split("/") if c]
    if not components:
        return ""
    dirs = components[:-1]
    stem = split_extension(components[-1])[0]
    if not dirs:
        return stem
    return "-".join(dirs) + "-" + stem


def ordered(d: Mapping[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    """Return ``d`` with keys in ``order`` first (dropping ``None``), then the rest."""
    out = {k: d[k] for k in order if k in d and d[k] is not None}
    out.update({k: v for k, v in d.items() if k not in out and v is not None})
    return out


@dataclass(frozen=True)
class EmbeddingEntry:
    """Read-only view of one ``manifest.embeddings`` entry (Swift ``EmbeddingEntry``)."""

    key: str
    model: str
    file: str
    engine: str | None = None
    format: str | None = None
    description: str | None = None
    language: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, key: str, d: Mapping[str, Any]) -> "EmbeddingEntry":
        def s(name: str) -> str | None:
            v = d.get(name)
            return v if isinstance(v, str) else None

        return cls(
            key=key,
            model=d.get("model") if isinstance(d.get("model"), str) else "",
            file=d.get("file") if isinstance(d.get("file"), str) else "",
            engine=s("engine"),
            format=s("format"),
            description=s("description"),
            language=s("language"),
            extra={k: v for k, v in d.items() if k not in EMBEDDING_FIELD_ORDER},
        )

    def to_dict(self) -> dict[str, Any]:
        base = {
            "model": self.model,
            "engine": self.engine,
            "file": self.file,
            "format": self.format,
            "description": self.description,
            "language": self.language,
        }
        base.update(self.extra)
        return ordered(base, EMBEDDING_FIELD_ORDER)


def embedding_items(manifest: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(key, raw dict)`` pairs of ``manifest.embeddings`` in manifest order (dicts only)."""
    emb = manifest.get("embeddings")
    if not isinstance(emb, dict):
        return []
    return [(k, v) for k, v in emb.items() if isinstance(v, dict)]


def embedding_entries(manifest: Mapping[str, Any]) -> list[EmbeddingEntry]:
    return [EmbeddingEntry.from_dict(k, v) for k, v in embedding_items(manifest)]


def is_safe_archive_path(path: str) -> bool:
    """Reject absolute paths, ``..`` traversal, backslashes and NULs (zip-slip safety)."""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    parts = posixpath.normpath(path).split("/")
    return ".." not in path.split("/") and parts[0] not in ("..", "")
