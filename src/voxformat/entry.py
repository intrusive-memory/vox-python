"""Archive entries and MIME detection (mirrors Swift ``VoxEntry`` / ``VoxMIME``)."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Any

_MIME = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "bin": "application/octet-stream",
    "safetensors": "application/x-safetensors",
    "onnx": "application/x-onnx",
    "json": "application/json",
}


def mime_type_for_extension(ext: str) -> str:
    """Return the MIME type for a file extension (without the dot)."""
    return _MIME.get(ext.lower().lstrip("."), "application/octet-stream")


def mime_type_for_path(path: str) -> str:
    """Return the MIME type for an archive path."""
    return mime_type_for_extension(path_extension(path))


def split_extension(name: str) -> tuple[str, str]:
    """Split ``name`` into (stem, ext) with ``NSString`` semantics.

    Matches Foundation's ``deletingPathExtension`` / ``pathExtension``:
    ``".bin"`` -> (".bin", ""), ``"x."`` -> ("x.", ""), ``"a.b.c"`` -> ("a.b", "c").
    """
    idx = name.rfind(".")
    if idx <= 0 or idx == len(name) - 1:
        return name, ""
    return name[:idx], name[idx + 1 :]


def path_extension(path: str) -> str:
    """Extension of the last path component, like ``NSString.pathExtension``."""
    return split_extension(posixpath.basename(path))[1]


@dataclass
class VoxEntry:
    """A single file inside a ``.vox`` archive.

    ``metadata`` is only used for manifest auto-management on :meth:`VoxFile.add`;
    it is not persisted inside the archive (same as Swift).
    """

    path: str
    data: bytes
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mime_type:
            self.mime_type = mime_type_for_path(self.path)
        if self.metadata is None:
            self.metadata = {}

    @property
    def size(self) -> int:
        return len(self.data)
