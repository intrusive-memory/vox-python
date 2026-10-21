"""Adapters for server-side voices that are just opaque handles (no local binaries).

These mirror ``examples/multi-engine`` and the spec's ``extensions`` example:

* ``elevenlabs`` -> ``{"voice_id": ..., "model_id": ...}``
* ``openai``     -> ``{"voice_id": <preset name>, "model_id": ...}``
* ``apple``      -> ``{"voice_id": "en-US/Aaron", "fallback": true}``
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import ProviderAdapter, ProviderVoice, register_adapter


def _require(response: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        v = response.get(k)
        if isinstance(v, str) and v:
            return v
    raise ValueError(f"response is missing {' / '.join(keys)}")


@register_adapter
class ElevenLabsAdapter(ProviderAdapter):
    """ElevenLabs Voices API object (``GET /v1/voices/{id}``) -> ``extensions.elevenlabs``."""

    provider = "elevenlabs"

    def voice_from_response(self, response: Mapping[str, Any], *, model_id: str | None = None,
                            **context: Any) -> ProviderVoice:
        ext: dict[str, Any] = {"voice_id": _require(response, "voice_id", "id")}
        model = model_id or response.get("model_id")
        if model:
            ext["model_id"] = model
        if response.get("name"):
            ext["name"] = response["name"]
        ext.update(context)
        return ProviderVoice(provider=self.provider, extension=ext)


@register_adapter
class OpenAIAdapter(ProviderAdapter):
    """OpenAI TTS preset voice (e.g. ``alloy``) -> ``extensions.openai``."""

    provider = "openai"

    def voice_from_response(self, response: Mapping[str, Any], *, model_id: str | None = None,
                            **context: Any) -> ProviderVoice:
        ext: dict[str, Any] = {"voice_id": _require(response, "voice_id", "voice", "id")}
        model = model_id or response.get("model_id") or response.get("model")
        if model:
            ext["model_id"] = model
        if response.get("instructions"):
            ext["instructions"] = response["instructions"]
        ext.update(context)
        return ProviderVoice(provider=self.provider, extension=ext)


@register_adapter
class AppleAdapter(ProviderAdapter):
    """Apple AVSpeechSynthesis voice identifier -> ``extensions.apple``."""

    provider = "apple"

    def voice_from_response(self, response: Mapping[str, Any], *, fallback: bool | None = None,
                            **context: Any) -> ProviderVoice:
        ext: dict[str, Any] = {"voice_id": _require(response, "voice_id", "identifier", "id")}
        if fallback is not None:
            ext["fallback"] = fallback
        elif isinstance(response.get("fallback"), bool):
            ext["fallback"] = response["fallback"]
        ext.update(context)
        return ProviderVoice(provider=self.provider, extension=ext)
