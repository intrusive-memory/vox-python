"""Validation: Swift ``validate()`` rules + versioned JSON schema + archive checks.

Layers (in report order):

1. **semantic** - a line-for-line port of ``VoxFile+Validation.swift`` (required fields,
   UUID v4, timestamp, description length, age range, gender, embedding entries, bundle
   completeness, provenance ethics, orphaned reference-audio model tags).
2. **schema** - the manifest *as stored on disk* against the schema for its declared
   ``vox_version`` (falls back to the latest bundled schema with an info note).
   Schema violations on a field that already has a semantic error are not repeated.
   A schema violation is an *error* only where Swift would also reject the file
   (decode failure: missing required key / wrong type / bad ``created``); stricter
   schema-only constraints are *warnings* (see :func:`schema_severity`).
3. **archive / compat** - Python-only extras: missing ``provenance.source`` files,
   extension values pointing at missing ``embeddings/`` files, undeclared embedding
   binaries (info), and JSON ``null`` inside ``extensions`` (which the Swift
   ``AnyCodable`` decoder cannot read - warning).
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import VoxError
from .manifest import (
    CLONE_CONSENT,
    KNOWN_METHODS,
    VALID_GENDERS,
    embedding_items,
    is_valid_uuid_v4,
    parse_iso8601,
)
from ._query import _contains
from .schema import jsonschema_available, select_schema_version, validate_against_schema


class Severity(enum.IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Issue:
    """Swift ``VoxIssue`` plus the layer that produced it."""

    severity: Severity
    message: str
    field: str | None = None
    source: str = "semantic"

    def __str__(self) -> str:
        if self.field:
            return f"[{self.severity}] {self.field}: {self.message}"
        return f"[{self.severity}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {"severity": str(self.severity).lower(), "message": self.message, "field": self.field,
                "source": self.source}


@dataclass
class ValidationReport:
    target: str
    issues: list[Issue] = field(default_factory=list)
    schema_version: str | None = None
    declared_version: str | None = None
    schema_engine: str | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "valid": self.is_valid,
            "declared_version": self.declared_version,
            "schema_version": self.schema_version,
            "schema_engine": self.schema_engine,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


# --------------------------------------------------------------------------- semantic


def _blank(v: Any) -> bool:
    return not isinstance(v, str) or not v.strip(" \t")


def semantic_issues(manifest: Mapping[str, Any], archive_paths: Iterable[str] | None = None) -> list[Issue]:
    """Port of Swift ``VoxFile.validate()``.

    ``archive_paths=None`` skips bundle-completeness checks (manifest-only validation).
    Where the Swift decoder would *throw* on a wrong type, this reports an error instead.
    """
    issues: list[Issue] = []
    add = lambda sev, msg, fld=None: issues.append(Issue(sev, msg, fld, "semantic"))  # noqa: E731
    E, W = Severity.ERROR, Severity.WARNING

    # Required fields
    if _blank(manifest.get("vox_version")):
        add(E, "Required field is empty or missing", "vox_version")
    mid = manifest.get("id")
    if not is_valid_uuid_v4(mid):
        add(E, f"Invalid UUID v4 format: '{mid if mid is not None else ''}'", "id")
    if parse_iso8601(manifest.get("created")) is None:
        add(E, "Invalid timestamp", "created")
    voice = manifest.get("voice")
    if not isinstance(voice, dict):
        add(E, "Required field is empty or missing", "voice")
        voice = {}
    if _blank(voice.get("name")):
        add(E, "Required field is empty or missing", "voice.name")
    desc = voice.get("description")
    desc_len = len(desc.strip(" \t")) if isinstance(desc, str) else 0
    if desc_len == 0:
        add(E, "Required field is empty or missing", "voice.description")
    elif desc_len < 10:
        add(E, f"Too short ({desc_len} characters, minimum 10)", "voice.description")

    # Optional fields
    age = voice.get("age_range")
    if isinstance(age, list) and len(age) == 2 and all(isinstance(a, int) and not isinstance(a, bool) for a in age):
        if age[0] >= age[1]:
            add(E, f"Minimum ({age[0]}) must be less than maximum ({age[1]})", "voice.age_range")
    gender = voice.get("gender")
    if gender is not None and gender not in VALID_GENDERS:
        add(E, f"Invalid gender value '{gender}'. Must be one of: male, female, nonbinary, neutral", "voice.gender")
    ref = manifest.get("reference_audio")
    clips = [c for c in ref if isinstance(c, dict)] if isinstance(ref, list) else []
    if isinstance(ref, list):
        for idx, clip in enumerate(ref):
            if not isinstance(clip, dict) or _blank(clip.get("file")):
                add(E, f"Reference audio entry at index {idx} has an empty file path", f"reference_audio[{idx}].file")

    # Embedding entries
    items = embedding_items(manifest)
    for key, entry in items:
        if _blank(entry.get("model")):
            add(E, "model must not be empty", f"embeddings.{key}")
        f = entry.get("file")
        f = f.strip(" \t") if isinstance(f, str) else ""
        if not f:
            add(E, "file must not be empty", f"embeddings.{key}")
        elif not f.startswith("embeddings/"):
            add(E, f"file must start with 'embeddings/' (got '{f}')", f"embeddings.{key}")

    # Bundle completeness (warnings, as in Swift)
    if archive_paths is not None:
        paths = set(archive_paths)
        for key, entry in items:
            if entry.get("file") not in paths:
                add(W, f"Declared file '{entry.get('file')}' is missing from the archive", f"embeddings.{key}")
        for clip in clips:
            if clip.get("file") not in paths:
                add(W, f"Declared file '{clip.get('file')}' is missing from the archive", "reference_audio")

    # Provenance (v0.3.0 ethics)
    prov = manifest.get("provenance")
    if isinstance(prov, dict):
        method = prov.get("method")
        if isinstance(method, str):
            if method not in KNOWN_METHODS:
                add(W, f"Unknown provenance method '{method}'. Known values: designed, synthesized, cloned, preset, hybrid",
                    "provenance.method")
            if method == "cloned":
                source = prov.get("source")
                if not isinstance(source, list) or not source:
                    add(E, "Cloned voices must have provenance.source with at least one entry for traceability",
                        "provenance.source")
                consent = prov.get("consent")
                if consent is None:
                    add(E, "Cloned voices require consent of 'self' or 'granted', but consent is nil",
                        "provenance.consent")
                elif consent not in CLONE_CONSENT:
                    add(E, f"Cloned voices require consent of 'self' or 'granted', got '{consent}'",
                        "provenance.consent")

    # Reference-audio model tags (warnings)
    models = {str(e.get("model", "")).lower() for _, e in items}
    for idx, clip in enumerate(ref if isinstance(ref, list) else []):
        if not isinstance(clip, dict) or not isinstance(clip.get("model"), str):
            continue
        cm = clip["model"].lower()
        if not any(_contains(m, cm) or _contains(cm, m) for m in models):
            add(W, f"Model-tagged reference audio '{clip['model']}' has no matching embeddings entry",
                f"reference_audio[{idx}].model")
    return issues


# --------------------------------------------------------------------------- extras


def _null_paths(value: Any, path: str) -> list[str]:
    out: list[str] = []
    if value is None:
        out.append(path)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.extend(_null_paths(v, f"{path}.{k}"))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.extend(_null_paths(v, f"{path}[{i}]"))
    return out


def compat_issues(manifest: Mapping[str, Any]) -> list[Issue]:
    issues = []
    ext = manifest.get("extensions")
    if isinstance(ext, dict):
        for p in _null_paths(ext, "extensions"):
            issues.append(Issue(Severity.WARNING,
                                "JSON null inside extensions cannot be decoded by the Swift reference "
                                "implementation (AnyCodable); omit the key instead", p, "compat"))
    return issues


def archive_issues(manifest: Mapping[str, Any], archive_paths: Iterable[str]) -> list[Issue]:
    paths = set(archive_paths)
    issues: list[Issue] = []
    prov = manifest.get("provenance")
    if isinstance(prov, dict) and isinstance(prov.get("source"), list):
        for i, src in enumerate(prov["source"]):
            if isinstance(src, str) and src not in paths:
                issues.append(Issue(Severity.WARNING, f"Provenance source '{src}' is missing from the archive",
                                    f"provenance.source[{i}]", "archive"))
    ext = manifest.get("extensions")
    if isinstance(ext, dict):
        for ns, val in ext.items():
            if not isinstance(val, dict):
                continue
            for k, v in val.items():
                if isinstance(v, str) and v.startswith("embeddings/") and v not in paths:
                    issues.append(Issue(Severity.WARNING, f"Extension references '{v}', which is missing from the archive",
                                        f"extensions.{ns}.{k}", "archive"))
    declared = {e.get("file") for _, e in embedding_items(manifest)}
    for p in sorted(paths):
        if p.startswith("embeddings/") and p not in declared:
            issues.append(Issue(Severity.INFO, f"Archive file '{p}' has no manifest embeddings entry", None, "archive"))
    return issues


def _covered(fld: str, error_fields: set[str]) -> bool:
    return any(fld == f or fld.startswith(f + ".") or fld.startswith(f + "[") for f in error_fields)


def _schema_layer(raw: Mapping[str, Any], engine: str, existing: list[Issue], report: ValidationReport) -> None:
    version, note = select_schema_version(raw.get("vox_version"))
    report.schema_version = version
    if engine == "auto":
        engine = "jsonschema" if jsonschema_available() else "builtin"
    report.schema_engine = engine
    if note:
        report.issues.append(Issue(Severity.INFO, note, "vox_version", "schema"))
    error_fields = {i.field for i in existing if i.severity is Severity.ERROR and i.field}
    for err in validate_against_schema(raw, version, engine=engine):
        if _covered(err.field, error_fields):
            continue
        report.issues.append(Issue(schema_severity(err), f"{err.message} (schema v{version})", err.field, "schema"))


#: Schema keywords whose violation would make the Swift ``VoxManifest`` decoder throw.
_DECODE_KEYWORDS = frozenset({"type", "required"})


def schema_severity(err: Any) -> Severity:
    """ERROR when Swift would reject the file; WARNING for schema-only strictness.

    Swift is the behavioural source of truth. It rejects a manifest when decoding fails
    (missing required key, wrong JSON type, unparseable ``created``) or when
    ``validate()`` reports an error (handled by the semantic layer). Other schema
    constraints - ``additionalProperties: false`` (Swift ignores unknown keys), the
    lowercase-only UUID pattern (Swift lowercases first), enum values Swift tolerates,
    ``minLength`` on transcripts (Swift's own ``add()`` writes ``""``) - are reported as
    warnings so ``--strict`` can still enforce full schema conformance.
    """
    if err.keyword in _DECODE_KEYWORDS:
        return Severity.ERROR
    if err.field == "created" and err.keyword == "pattern":
        return Severity.ERROR
    return Severity.WARNING


def validate_manifest(manifest: Mapping[str, Any], *, target: str = "<manifest>", engine: str = "auto") -> ValidationReport:
    """Validate a bare manifest dict (no archive; file-existence checks are skipped)."""
    report = ValidationReport(target=target)
    if not isinstance(manifest, dict):
        report.issues.append(Issue(Severity.ERROR, "manifest is not a JSON object", None, "decode"))
        return report
    report.declared_version = manifest.get("vox_version") if isinstance(manifest.get("vox_version"), str) else None
    sem = semantic_issues(manifest, None)
    report.issues.extend(sem)
    _schema_layer(manifest, engine, sem, report)
    report.issues.extend(compat_issues(manifest))
    return report


def validate_vox(vox: "Any", *, target: str | None = None, engine: str = "auto") -> ValidationReport:
    """Validate an opened :class:`VoxFile` (raw on-disk manifest for the schema layer)."""
    report = ValidationReport(target=target or (str(vox.path) if vox.path else "<memory>"))
    raw = vox.raw_manifest if vox.raw_manifest is not None else vox.manifest
    report.declared_version = raw.get("vox_version") if isinstance(raw.get("vox_version"), str) else None
    paths = [e.path for e in vox.entries]
    sem = semantic_issues(vox.manifest, paths)
    report.issues.extend(sem)
    _schema_layer(raw, engine, sem, report)
    report.issues.extend(compat_issues(raw))
    report.issues.extend(archive_issues(vox.manifest, paths))
    return report


def validate_file(path: str | Path, *, engine: str = "auto") -> ValidationReport:
    """Validate a ``.vox`` archive or a bare ``manifest.json``. Never raises for bad input."""
    from .voxfile import VoxFile

    p = Path(path)
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            r = ValidationReport(target=str(p))
            r.issues.append(Issue(Severity.ERROR, f"Cannot read manifest: {e}", None, "decode"))
            return r
        return validate_manifest(data, target=str(p), engine=engine)
    try:
        vox = VoxFile.open(p)
    except (VoxError, OSError) as e:
        r = ValidationReport(target=str(p))
        r.issues.append(Issue(Severity.ERROR, str(e), None, "decode"))
        return r
    return validate_vox(vox, target=str(p), engine=engine)
