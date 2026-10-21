"""Attach TTS engine data from any source, spec-compliantly.

Three kinds of engine data, three homes:

=====================================  =====================================================
What you have                           Where it goes
=====================================  =====================================================
Local binary embedding (clone prompt,   ``embeddings/<engine>/<model-slug>/[<lang>/]<name>``
safetensors, onnx, ...)                 + a ``manifest.embeddings`` entry
                                        (:func:`add_engine_embedding`)
Engine-generated sample audio           ``embeddings/<engine>/<model-slug>/[<lang>/]sample-audio.wav``
                                        + ``embeddings`` entry, language-neutral by default
                                        (:func:`add_engine_sample`)
Server-side / cloud voice handle        ``manifest.extensions.<provider>`` (reserved keys:
(ElevenLabs voice_id, Gemini voice id,  ``google``, ``elevenlabs``, ``openai``, ``apple``, ...)
OpenAI preset, Apple voice id)          (:func:`set_provider_extension`). Never stored as a
                                        fake embedding file.
=====================================  =====================================================

Embedding keys are derived exactly like Swift ``deriveEmbeddingKey`` (path minus
``embeddings/``, directories joined by ``-``, plus the filename stem), e.g.
``embeddings/google/gemini-3.8-flash-tts/sample-audio.wav`` ->
``google-gemini-3.8-flash-tts-sample-audio``.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Mapping

from .entry import path_extension
from .manifest import RESERVED_PROVIDERS, EmbeddingEntry, derive_embedding_key

if TYPE_CHECKING:
    from .voxfile import VoxFile

SAMPLE_AUDIO_NAME = "sample-audio.wav"
CLONE_PROMPT_NAME = "clone-prompt.bin"

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_LANG_RE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")


class UnknownProviderWarning(UserWarning):
    """Raised (as a warning) when an extension namespace is not a reserved provider id."""


def derive_model_slug(model: str) -> str:
    """Default path slug for a model id: last ``/`` component, lowercased, sanitized.

    ``"gemini-3.8-flash-tts"`` -> ``"gemini-3.8-flash-tts"``;
    ``"Qwen/Qwen3-TTS-12Hz-0.6B"`` -> ``"qwen3-tts-12hz-0.6b"``. The established Qwen
    layout uses the short variant (``0.6b``/``1.7b``) - pass ``model_slug=`` explicitly.
    """
    last = model.rstrip("/").split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9._+-]+", "-", last).strip("-.")
    if not slug:
        raise ValueError(f"cannot derive a path slug from model '{model}'")
    return slug


def _check_segment(value: str, what: str) -> str:
    if not _SEGMENT_RE.match(value) or value in (".", ".."):
        raise ValueError(f"invalid {what} path segment '{value}'")
    return value


def check_language(language: str | None) -> str | None:
    if language is None:
        return None
    if language.lower() == "default":
        raise ValueError("'default' is reserved; omit language for the language-neutral entry")
    if not _LANG_RE.match(language):
        raise ValueError(f"'{language}' is not a BCP 47 language tag")
    return language


def embedding_path(engine: str, slug: str, name: str, language: str | None = None) -> str:
    """``embeddings/<engine>/<slug>/[<language>/]<name>``."""
    parts = ["embeddings", _check_segment(engine, "engine"), _check_segment(slug, "model slug")]
    if language:
        parts.append(_check_segment(check_language(language) or "", "language"))
    parts.append(_check_segment(name, "file name"))
    return "/".join(parts)


def add_engine_embedding(
    vox: "VoxFile",
    data: bytes,
    *,
    engine: str,
    model: str,
    name: str = CLONE_PROMPT_NAME,
    model_slug: str | None = None,
    language: str | None = None,
    format: str | None = None,
    description: str | None = None,
    key: str | None = None,
) -> EmbeddingEntry:
    """Store a local engine binary and its ``embeddings`` manifest entry.

    Replaces any entry already at the same path. Returns the manifest entry view.
    """
    slug = model_slug or derive_model_slug(model)
    path = embedding_path(engine, slug, name, language)
    meta: dict[str, Any] = {
        "model": model,
        "engine": engine,
        "format": format or (path_extension(name).lower() or None),
        "description": description,
        "language": language,
    }
    if key:
        meta["key"] = key
    vox.add(data, path, metadata={k: v for k, v in meta.items() if v is not None})
    k = key or derive_embedding_key(path)
    return EmbeddingEntry.from_dict(k, vox.manifest["embeddings"][k])


def add_engine_sample(
    vox: "VoxFile",
    wav: bytes,
    *,
    engine: str,
    model: str,
    model_slug: str | None = None,
    language: str | None = None,
    description: str | None = None,
) -> EmbeddingEntry:
    """Store engine-generated sample audio (language-neutral unless ``language`` given)."""
    slug = model_slug or derive_model_slug(model)
    if description is None:
        description = f"Engine-generated voice sample ({slug})"
        if language:
            description = f"Engine-generated voice sample ({slug}, {language})"
    return add_engine_embedding(
        vox, wav, engine=engine, model=model, name=SAMPLE_AUDIO_NAME, model_slug=slug,
        language=language, format="wav", description=description,
    )


def strip_none(value: Any) -> Any:
    """Recursively drop ``None`` values (Swift's AnyCodable cannot decode JSON null)."""
    if isinstance(value, dict):
        return {k: strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_none(v) for v in value if v is not None]
    return value


def set_provider_extension(
    vox: "VoxFile",
    provider: str,
    data: Mapping[str, Any],
    *,
    merge: bool = False,
    drop_none: bool = True,
) -> dict[str, Any]:
    """Store a server-side voice handle / engine settings under ``extensions.<provider>``.

    Non-reserved provider ids are allowed (readers MUST ignore unknown extensions) but
    emit :class:`UnknownProviderWarning`. ``None`` values are dropped by default because
    the Swift reference reader cannot decode JSON ``null`` inside ``extensions``.
    """
    if provider not in RESERVED_PROVIDERS:
        warnings.warn(
            f"'{provider}' is not a reserved VOX provider id ({', '.join(RESERVED_PROVIDERS)})",
            UnknownProviderWarning,
            stacklevel=2,
        )
    payload = strip_none(dict(data)) if drop_none else dict(data)
    return vox.set_extension(provider, payload, merge=merge)
