"""Typed error hierarchy, mirroring Swift ``VoxError``."""

from __future__ import annotations


class VoxError(Exception):
    """Base class for every error raised by :mod:`voxformat`."""


class InvalidZipError(VoxError):
    """The file is not a valid ZIP archive (Swift ``invalidZipFile``)."""


class ManifestNotFoundError(VoxError):
    """``manifest.json`` is missing from the archive (Swift ``manifestNotFound``)."""


class InvalidJSONError(VoxError):
    """``manifest.json`` could not be decoded (Swift ``invalidJSON``)."""


class InvalidPathError(VoxError):
    """An archive path is empty, reserved, or unsafe (Swift ``invalidPath``)."""


class MissingRequiredMetadataError(VoxError):
    """A required metadata key is missing for an entry (Swift ``missingRequiredMetadata``)."""

    def __init__(self, path: str, key: str):
        self.path = path
        self.key = key
        super().__init__(f"Missing required metadata '{key}' for entry at '{path}'")


class WriteFailedError(VoxError):
    """Writing the archive failed (Swift ``writeFailed``)."""


class SampleNotFoundError(VoxError):
    """No sample audio resolves for the requested model/language."""


class PlayerNotFoundError(VoxError):
    """No supported audio player binary is available on this system."""
