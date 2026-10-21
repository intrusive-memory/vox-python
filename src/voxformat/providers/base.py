"""Provider-adapter interface: plug any TTS source into a ``.vox``.

An adapter turns a provider's native response (an API JSON payload, a preset name, a
local model's output) into a :class:`ProviderVoice` - a neutral description of what to
store - and :func:`attach` writes it into the container using the spec-compliant homes
from :mod:`voxformat.engines`:

* ``ProviderVoice.extension``  -> ``manifest.extensions[<provider>]`` (opaque handles)
* ``ProviderVoice.samples``    -> ``embeddings/<engine>/<slug>/[<lang>/]sample-audio.wav``
* ``ProviderVoice.embeddings`` -> ``embeddings/<engine>/<slug>/[<lang>/]<name>``

Writing a new adapter::

    class MyAdapter(ProviderAdapter):
        provider = "azure"          # reserved extensions key
        engine = "azure"            # embeddings/<engine>/ namespace

        def voice_from_response(self, response, **context):
            return ProviderVoice(provider=self.provider,
                                 extension={"voice_id": response["ShortName"]})

    register_adapter(MyAdapter)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from ..engines import (
    add_engine_embedding,
    add_engine_sample,
    derive_model_slug,
    embedding_path,
    SAMPLE_AUDIO_NAME,
    set_provider_extension,
)

if TYPE_CHECKING:
    from ..voxfile import VoxFile


@dataclass
class ProviderSample:
    """Engine-generated preview audio (WAV)."""

    model: str
    data: bytes
    model_slug: str | None = None
    language: str | None = None
    description: str | None = None


@dataclass
class ProviderEmbedding:
    """A local binary the engine needs to reproduce the voice (clone prompt etc.)."""

    model: str
    data: bytes
    name: str
    model_slug: str | None = None
    language: str | None = None
    format: str | None = None
    description: str | None = None
    key: str | None = None


@dataclass
class ProviderVoice:
    """Neutral, spec-shaped description of what a provider contributes to a voice."""

    provider: str
    extension: dict[str, Any] | None = None
    engine: str | None = None  # defaults to provider
    samples: list[ProviderSample] = field(default_factory=list)
    embeddings: list[ProviderEmbedding] = field(default_factory=list)


def attach(vox: "VoxFile", voice: ProviderVoice, *, merge_extension: bool = False) -> None:
    """Write a :class:`ProviderVoice` into ``vox`` (replacing same-path entries)."""
    engine = voice.engine or voice.provider
    if voice.extension is not None:
        set_provider_extension(vox, voice.provider, voice.extension, merge=merge_extension)
    for s in voice.samples:
        add_engine_sample(vox, s.data, engine=engine, model=s.model, model_slug=s.model_slug,
                          language=s.language, description=s.description)
    for e in voice.embeddings:
        add_engine_embedding(vox, e.data, engine=engine, model=e.model, name=e.name, model_slug=e.model_slug,
                             language=e.language, format=e.format, description=e.description, key=e.key)


def detach(vox: "VoxFile", provider: str, *, engine: str | None = None, remove_files: bool = True) -> list[str]:
    """Remove ``extensions[provider]`` and (optionally) every ``embeddings/<engine>/`` entry."""
    removed: list[str] = []
    if vox.remove_extension(provider) is not None:
        removed.append(f"extensions.{provider}")
    if remove_files:
        prefix = f"embeddings/{engine or provider}/"
        for e in vox.entries_under(prefix):
            vox.remove(e.path)
            removed.append(e.path)
    return removed


class ProviderAdapter(ABC):
    """Base class for provider adapters."""

    #: ``manifest.extensions`` key (use a reserved provider id where one exists).
    provider: ClassVar[str]
    #: ``embeddings/<engine>/`` namespace; defaults to :attr:`provider`.
    engine: ClassVar[str | None] = None

    @abstractmethod
    def voice_from_response(self, response: Mapping[str, Any], **context: Any) -> ProviderVoice:
        """Map a provider-native payload to a :class:`ProviderVoice`."""

    def attach(self, vox: "VoxFile", response: Mapping[str, Any], **context: Any) -> ProviderVoice:
        voice = self.voice_from_response(response, **context)
        attach(vox, voice)
        return voice

    def extension(self, vox: "VoxFile") -> dict[str, Any] | None:
        ext = vox.extensions.get(self.provider)
        return ext if isinstance(ext, dict) else None

    def detach(self, vox: "VoxFile", *, remove_files: bool = True) -> list[str]:
        return detach(vox, self.provider, engine=self.engine or self.provider, remove_files=remove_files)

    def sample_path(self, model: str, *, model_slug: str | None = None, language: str | None = None) -> str:
        return embedding_path(self.engine or self.provider, model_slug or derive_model_slug(model),
                              SAMPLE_AUDIO_NAME, language)


_REGISTRY: dict[str, type[ProviderAdapter]] = {}


def register_adapter(cls: type[ProviderAdapter], name: str | None = None) -> type[ProviderAdapter]:
    """Register an adapter class (usable as a decorator)."""
    _REGISTRY[name or cls.provider] = cls
    return cls


def get_adapter(name: str, **kwargs: Any) -> ProviderAdapter:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"no adapter registered for '{name}' (have: {', '.join(sorted(_REGISTRY))})") from None


def registered_adapters() -> list[str]:
    return sorted(_REGISTRY)
