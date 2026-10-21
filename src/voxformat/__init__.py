"""voxformat - Python reference implementation of the VOX voice-identity format.

This package tracks the Swift reference implementation
(``implementations/swift/Sources/VoxFormat``), which is the behavioural source of truth.
"""

from .entry import VoxEntry, mime_type_for_extension, mime_type_for_path
from .engines import (
    add_engine_embedding,
    add_engine_sample,
    derive_model_slug,
    embedding_path,
    set_provider_extension,
)
from .errors import (
    InvalidJSONError,
    InvalidPathError,
    InvalidZipError,
    ManifestNotFoundError,
    MissingRequiredMetadataError,
    PlayerNotFoundError,
    SampleNotFoundError,
    VoxError,
    WriteFailedError,
)
from .manifest import CURRENT_VERSION, RESERVED_PROVIDERS, EmbeddingEntry, derive_embedding_key
from .migrator import migrate_manifest
from .validation import Issue, Severity, ValidationReport, validate_file, validate_manifest, validate_vox
from .voxfile import Readiness, VoxFile

__version__ = "0.4.0"

__all__ = [
    "CURRENT_VERSION",
    "RESERVED_PROVIDERS",
    "EmbeddingEntry",
    "InvalidJSONError",
    "InvalidPathError",
    "InvalidZipError",
    "Issue",
    "ManifestNotFoundError",
    "MissingRequiredMetadataError",
    "PlayerNotFoundError",
    "Readiness",
    "SampleNotFoundError",
    "Severity",
    "ValidationReport",
    "VoxEntry",
    "VoxError",
    "VoxFile",
    "WriteFailedError",
    "add_engine_embedding",
    "add_engine_sample",
    "derive_embedding_key",
    "derive_model_slug",
    "embedding_path",
    "migrate_manifest",
    "mime_type_for_extension",
    "mime_type_for_path",
    "set_provider_extension",
    "validate_file",
    "validate_manifest",
    "validate_vox",
]
