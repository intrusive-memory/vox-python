"""The container-first ``VoxFile`` API (mirrors Swift ``VoxFile``)."""

from __future__ import annotations

import copy
import io
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Union

from ._automanifest import AutoManifestMixin
from ._query import ModelQueryMixin
from .entry import VoxEntry
from .errors import (
    InvalidJSONError,
    InvalidPathError,
    InvalidZipError,
    ManifestNotFoundError,
    MissingRequiredMetadataError,
    WriteFailedError,
)
from .manifest import (
    CURRENT_VERSION,
    EmbeddingEntry,
    embedding_entries,
    is_safe_archive_path,
    new_manifest,
    now_iso,
)
from .migrator import migrate_manifest

PathLike = Union[str, "os.PathLike[str]"]
MANIFEST = "manifest.json"
_ZIP_MAGIC = b"PK\x03\x04"


@dataclass(frozen=True)
class Readiness:
    """Swift ``VoxFile.Readiness``: ``state`` is ``ready`` / ``needs_regeneration`` / ``invalid``."""

    state: str
    missing: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class VoxFile(AutoManifestMixin, ModelQueryMixin):
    """A mutable, in-memory ``.vox`` container.

    ``manifest`` is the parsed ``manifest.json`` as a plain dict: unknown fields and
    unknown ``extensions`` are preserved verbatim. Entries are kept in archive order;
    untouched entries are written back with their original ZIP metadata.
    """

    def __init__(self, name: str | None = None, description: str | None = None, *,
                 manifest: Mapping[str, Any] | None = None, **voice_fields: Any):
        if manifest is not None:
            self.manifest: dict[str, Any] = copy.deepcopy(dict(manifest))
        else:
            if name is None or description is None:
                raise TypeError("VoxFile() requires name and description, or manifest=")
            self.manifest = new_manifest(name, description, **voice_fields)
        self._storage: dict[str, VoxEntry] = {}
        self._infos: dict[str, zipfile.ZipInfo] = {}  # original ZIP metadata of untouched entries
        self._order: list[str] = []  # original archive order (incl. manifest + directories)
        self._dirs: dict[str, zipfile.ZipInfo] = {}
        self.path: Path | None = None
        #: The manifest exactly as read from disk (before migration); ``None`` for new files.
        self.raw_manifest: dict[str, Any] | None = None
        #: ``vox_version`` declared on disk before migration.
        self.source_version: str | None = None
        self._dirty = manifest is None

    # ------------------------------------------------------------------ I/O --

    @classmethod
    def open(cls, path: PathLike, *, migrate: bool = True) -> "VoxFile":
        """Read a ``.vox`` archive into memory (Swift ``init(contentsOf:)``).

        Like Swift, the manifest is silently migrated to the current version. Unlike
        Swift, decoding is lenient: structurally bad manifests still open so that
        :func:`voxformat.validate_file` can report every problem. Only a non-ZIP file,
        a missing ``manifest.json`` or a manifest that is not a JSON object raise.
        """
        p = Path(path)
        try:
            zf = zipfile.ZipFile(p)
        except (zipfile.BadZipFile, OSError) as e:
            raise InvalidZipError(f"Not a valid ZIP archive: {p.name} ({e})") from e
        with zf:
            vox = cls.__new__(cls)
            vox._storage, vox._infos, vox._order, vox._dirs = {}, {}, [], {}
            manifest_bytes: bytes | None = None
            for info in zf.infolist():
                name = info.filename
                if name not in vox._order:
                    vox._order.append(name)
                if info.is_dir():
                    vox._dirs[name] = info
                    continue
                data = zf.read(info)
                if name == MANIFEST:
                    manifest_bytes = data
                    continue
                vox._storage[name] = VoxEntry(path=name, data=data)
                vox._infos[name] = info
        if manifest_bytes is None:
            raise ManifestNotFoundError(f"manifest.json not found in archive: {p.name}")
        try:
            parsed = json.loads(manifest_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise InvalidJSONError(f"Invalid JSON in manifest.json from {p.name}: {e}") from e
        if not isinstance(parsed, dict):
            raise InvalidJSONError(f"manifest.json in {p.name} is not a JSON object")
        vox.raw_manifest = copy.deepcopy(parsed)
        vox.source_version = parsed.get("vox_version") if isinstance(parsed.get("vox_version"), str) else None
        if migrate:
            rel = [k[len("embeddings/"):] for k in vox._storage if k.startswith("embeddings/")]
            vox.manifest = migrate_manifest(parsed, rel)
        else:
            vox.manifest = parsed
        vox.path = p
        vox._dirty = False
        return vox

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "VoxFile":
        """Swift ``VoxFile(manifest:)`` - no entries, manifest used as-is (no migration)."""
        return cls(manifest=manifest)

    def manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest, indent=2, ensure_ascii=False).encode("utf-8")

    def write(self, path: PathLike | None = None, *, touch: bool | None = None) -> Path:
        """Write atomically (temp file in the same directory, then ``os.replace``).

        Like Swift, ``vox_version`` is stamped to the current version. If the container
        was modified through this API (or ``touch=True``), ``modified`` is set to now.
        Untouched entries keep their original order, timestamps and compression.
        """
        dest = Path(path) if path is not None else self.path
        if dest is None:
            raise WriteFailedError("No destination path given and VoxFile was not opened from disk")
        if touch or (touch is None and self._dirty):
            self.manifest["modified"] = now_iso()
        self.manifest["vox_version"] = CURRENT_VERSION
        manifest_data = self.manifest_bytes()

        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
                    self._write_entries(zf, manifest_data)
                fh.flush()
                os.fsync(fh.fileno())
            with open(tmp, "rb") as fh:
                if fh.read(4) != _ZIP_MAGIC:
                    raise WriteFailedError(f"Output file does not have ZIP magic bytes: {dest}")
            try:
                mode = stat.S_IMODE(os.stat(dest).st_mode)
            except FileNotFoundError:
                umask = os.umask(0)
                os.umask(umask)
                mode = 0o644 & ~umask
            os.chmod(tmp, mode)
            os.replace(tmp, dest)
        except WriteFailedError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise WriteFailedError(f"Failed to write VOX file: {dest.name} ({e})") from e

        self.path = dest
        self._dirty = False
        return dest

    def _write_entries(self, zf: zipfile.ZipFile, manifest_data: bytes) -> None:
        written: set[str] = set()
        order = list(self._order)
        if MANIFEST not in order:
            order.insert(0, MANIFEST)
        for name in order:
            if name in written:
                continue
            if name == MANIFEST:
                zf.writestr(_new_info(MANIFEST), manifest_data, compress_type=zipfile.ZIP_DEFLATED)
            elif name in self._dirs:
                zf.writestr(_clone_info(self._dirs[name]), b"")
            elif name in self._storage:
                self._write_one(zf, self._storage[name])
            else:
                continue  # removed
            written.add(name)
        for name, entry in self._storage.items():
            if name not in written:
                self._write_one(zf, entry)
                written.add(name)

    def _write_one(self, zf: zipfile.ZipFile, entry: VoxEntry) -> None:
        info = self._infos.get(entry.path)
        if info is not None:
            zf.writestr(_clone_info(info), entry.data)
        else:
            zf.writestr(_new_info(entry.path), entry.data, compress_type=zipfile.ZIP_DEFLATED)

    def to_bytes(self) -> bytes:
        """Serialize to an in-memory ``.vox`` (does not stamp ``modified``)."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            self._write_entries(zf, self.manifest_bytes())
        return buf.getvalue()

    # ----------------------------------------------------- entry management --

    def add(self, data: bytes, path: str, *, mime_type: str | None = None,
            metadata: Mapping[str, Any] | None = None) -> VoxEntry:
        """Add (or replace) an entry, auto-updating the manifest (Swift ``add(_:at:...)``).

        * ``reference/...`` creates a ``reference_audio`` item from metadata
          (``transcript``, ``language``, ``duration_seconds``, ``context``, ``model``, ``engine``).
        * ``embeddings/...`` creates an ``embeddings`` entry; ``model`` is **required**;
          optional ``engine``, ``format``, ``description``, ``key``, ``language``.
          Key defaults to :func:`voxformat.manifest.derive_embedding_key`.
        * other paths are stored without manifest effect.
        """
        trimmed = path.strip(" \t")
        if not trimmed:
            raise InvalidPathError(f"Invalid path '{path}': path must not be empty")
        if trimmed == MANIFEST:
            raise InvalidPathError(f"Invalid path '{path}': manifest.json is reserved")
        if not is_safe_archive_path(trimmed) or trimmed.endswith("/"):
            raise InvalidPathError(f"Invalid path '{path}': must be a relative file path without '..'")
        meta = dict(metadata or {})
        if trimmed.startswith("embeddings/"):
            model = meta.get("model")
            if not isinstance(model, str) or not model.strip():
                raise MissingRequiredMetadataError(trimmed, "model")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes")
        entry = VoxEntry(path=trimmed, data=bytes(data), mime_type=mime_type or "", metadata=meta)
        if trimmed in self._storage:
            self._remove_manifest_entry(trimmed)
        self._storage[trimmed] = entry
        self._infos.pop(trimmed, None)
        self._add_manifest_entry(entry)
        self._dirty = True
        return entry

    def add_file(self, source: PathLike, path: str, **kwargs: Any) -> VoxEntry:
        return self.add(Path(source).read_bytes(), path, **kwargs)

    def remove(self, path: str) -> VoxEntry | None:
        """Remove an entry and its manifest item (Swift ``remove(at:)``)."""
        entry = self._storage.pop(path, None)
        if entry is None:
            return None
        self._infos.pop(path, None)
        self._remove_manifest_entry(path)
        self._dirty = True
        return entry

    def __getitem__(self, path: str) -> VoxEntry | None:
        return self._storage.get(path)

    def get(self, path: str) -> VoxEntry | None:
        return self._storage.get(path)

    def __contains__(self, path: object) -> bool:
        return path in self._storage

    def contains(self, path: str) -> bool:
        return path in self._storage

    def __iter__(self) -> Iterator[VoxEntry]:
        return iter(list(self._storage.values()))

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def entries(self) -> list[VoxEntry]:
        return list(self._storage.values())

    @property
    def entry_count(self) -> int:
        return len(self._storage)

    def entries_under(self, prefix: str) -> list[VoxEntry]:
        return [e for p, e in self._storage.items() if p.startswith(prefix)]

    def _entry_data(self, path: str) -> Optional[bytes]:
        e = self._storage.get(path)
        return e.data if e else None

    # ----------------------------------------------------- manifest editing --

    @property
    def name(self) -> str:
        return str(self.manifest.get("voice", {}).get("name", ""))

    @property
    def description(self) -> str:
        return str(self.manifest.get("voice", {}).get("description", ""))

    @property
    def extensions(self) -> dict[str, Any]:
        ext = self.manifest.get("extensions")
        return ext if isinstance(ext, dict) else {}

    @property
    def embedding_entries(self) -> list[EmbeddingEntry]:
        return embedding_entries(self.manifest)

    def update_voice(self, **fields: Any) -> None:
        """Set ``voice.*`` fields; a value of ``None`` deletes the field."""
        voice = self.manifest.setdefault("voice", {})
        for k, v in fields.items():
            if v is None:
                voice.pop(k, None)
            else:
                voice[k] = v
        self._dirty = True

    def set_field(self, key: str, value: Any) -> None:
        """Set (or with ``None`` delete) a top-level manifest field."""
        if key in ("embeddings", "reference_audio"):
            raise InvalidPathError(f"'{key}' is managed by add()/remove()")
        if value is None:
            self.manifest.pop(key, None)
        else:
            self.manifest[key] = value
        self._dirty = True

    def set_extension(self, provider: str, data: Mapping[str, Any], *, merge: bool = False) -> dict[str, Any]:
        """Set ``manifest.extensions[provider]``. ``merge=True`` shallow-merges into existing."""
        if not provider or not isinstance(provider, str):
            raise ValueError("provider must be a non-empty string")
        ext = self.manifest.get("extensions")
        if not isinstance(ext, dict):
            ext = self.manifest["extensions"] = {}
        current = ext.get(provider)
        if merge and isinstance(current, dict):
            current.update(copy.deepcopy(dict(data)))
            value = current
        else:
            value = copy.deepcopy(dict(data))
            ext[provider] = value
        self._dirty = True
        return value

    def remove_extension(self, provider: str) -> Any:
        ext = self.manifest.get("extensions")
        if not isinstance(ext, dict) or provider not in ext:
            return None
        value = ext.pop(provider)
        if not ext:
            del self.manifest["extensions"]
        self._dirty = True
        return value

    def touch(self) -> None:
        """Mark modified so the next :meth:`write` bumps ``modified``."""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------ validity --

    def validate(self, *, check_bundle: bool = True) -> list:
        """Swift ``validate()`` rules. Returns a list of :class:`voxformat.Issue`."""
        from .validation import semantic_issues

        return semantic_issues(self.manifest, self._storage.keys() if check_bundle else None)

    @property
    def is_valid(self) -> bool:
        from .validation import Severity

        return not any(i.severity is Severity.ERROR for i in self.validate())

    @property
    def readiness(self) -> Readiness:
        voice = self.manifest.get("voice") if isinstance(self.manifest.get("voice"), dict) else {}
        reasons = []
        desc = str(voice.get("description", "")).strip(" \t")
        if len(desc) < 10:
            reasons.append(f"Voice description is too short ({len(desc)} chars, need >= 10)")
        if not str(voice.get("name", "")).strip(" \t"):
            reasons.append("Voice name is empty")
        if reasons:
            return Readiness("invalid", reasons=tuple(reasons))
        missing: list[str] = []
        for e in self.embedding_entries:
            if e.file not in self._storage:
                alt = e.file if e.file.startswith("embeddings/") else f"embeddings/{e.file}"
                if alt not in self._storage:
                    missing.append(e.key)
        for clip in self.manifest.get("reference_audio") or []:
            f = clip.get("file", "") if isinstance(clip, dict) else ""
            if f not in self._storage:
                alt = f if f.startswith("reference/") else f"reference/{f}"
                if alt not in self._storage:
                    missing.append(f"reference:{f}")
        if missing:
            return Readiness("needs_regeneration", missing=tuple(missing))
        return Readiness("ready")

    @property
    def is_ready(self) -> bool:
        return self.readiness.state == "ready"

    def __repr__(self) -> str:
        return f"<VoxFile name={self.name!r} entries={len(self._storage)} path={str(self.path) if self.path else None!r}>"


def _clone_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Fresh ZipInfo carrying the original name, timestamp, compression and attributes.

    The ``extra`` field is deliberately not copied: it may hold stale ZIP64 size records.
    """
    clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    clone.compress_type = info.compress_type
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.comment = info.comment
    return clone


def _new_info(name: str) -> zipfile.ZipInfo:
    import time

    info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info
