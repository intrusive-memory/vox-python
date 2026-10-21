"""Google Gemini 3.8 Flash TTS designed ("prompted") voices.

Layout written into the ``.vox`` (identical to the proof-of-concept)::

    manifest.extensions.google = {
        "voice_id", "model_id": "gemini-3.8-flash-tts", "type", "display_name",
        "gender", "language_code", "prompted_input", "created", "expire_time", "refreshed"
    }
    embeddings/google/gemini-3.8-flash-tts/sample-audio.wav
    manifest.embeddings["google-gemini-3.8-flash-tts-sample-audio"] = {
        "model": "gemini-3.8-flash-tts", "engine": "google",
        "file": "embeddings/google/gemini-3.8-flash-tts/sample-audio.wav", "format": "wav",
        "description": "Engine-generated voice sample (gemini-3.8-flash-tts)"
    }                                     # no "language": the sample is language-neutral

The voice handle lives in ``extensions`` because it is an opaque server-side id, not an
engine binary. The Voices API has no PATCH, so "update" = create a new voice, swap it
into the ``.vox``, write, then delete the old id.

The API key is read only from ``GEMINI_API_KEY`` (or passed explicitly). It is sent
only in the ``x-goog-api-key`` header and never logged, printed, or written to disk.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ..errors import VoxError
from ..manifest import now_iso
from .base import ProviderAdapter, ProviderSample, ProviderVoice, attach, register_adapter

if TYPE_CHECKING:
    from ..voxfile import VoxFile

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-3.8-flash-tts"
ENGINE = "google"
PREFIX = f"embeddings/{ENGINE}/{MODEL}"
SAMPLE_PATH = f"{PREFIX}/sample-audio.wav"
SAMPLE_KEY = f"{ENGINE}-{MODEL}-sample-audio"
SAMPLE_DESCRIPTION = f"Engine-generated voice sample ({MODEL})"
LEGACY_VOICE_JSON = f"{PREFIX}/voice.json"  # pre-fix PoC layout (handle stored as fake embedding)
LEGACY_VOICE_REF_KEY = f"{ENGINE}-{MODEL}-voice-ref"
EXTENSION_ORDER = ("voice_id", "model_id", "type", "display_name", "gender", "language_code",
                   "prompted_input", "created", "expire_time", "refreshed")
API_KEY_ENV = "GEMINI_API_KEY"

#: ``transport(method, url, headers, body, timeout) -> (status, response_bytes)``
Transport = Callable[[str, str, Mapping[str, str], "bytes | None", float], "tuple[int, bytes]"]


class GeminiError(VoxError):
    """Base error for the Gemini adapter."""


class GeminiAuthError(GeminiError):
    """No API key available."""


class GeminiAPIError(GeminiError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"Gemini API error {status}: {message}")


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                     timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, data=body, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https base
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""


class GeminiVoicesClient:
    """Minimal Gemini Voices API client (create / get / list / delete / synthesize)."""

    def __init__(self, api_key: str | None = None, *, base_url: str = API_BASE,
                 transport: Transport | None = None, timeout: float = 180.0):
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise GeminiAuthError(f"{API_KEY_ENV} is not set")
        self.__key = key
        self.base_url = base_url.rstrip("/")
        self.transport: Transport = transport or urllib_transport
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"<GeminiVoicesClient base_url={self.base_url!r}>"

    def _redact(self, text: str) -> str:
        return text.replace(self.__key, "***") if self.__key else text

    def _call(self, method: str, path: str, body: Any = None, query: Mapping[str, str] | None = None) -> dict:
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"x-goog-api-key": self.__key, "Content-Type": "application/json"}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        try:
            status, raw = self.transport(method, url, headers, data, self.timeout)
        except (urllib.error.URLError, OSError) as e:
            raise GeminiError(self._redact(f"request failed: {e}")) from None
        try:
            payload = json.loads(raw) if raw else {}
        except (ValueError, UnicodeDecodeError):
            payload = {"raw": raw[:200].decode("utf-8", "replace")}
        if not 200 <= status < 300:
            msg = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else None
            raise GeminiAPIError(status, self._redact(msg or json.dumps(payload)[:300]))
        return payload if isinstance(payload, dict) else {"data": payload}

    @staticmethod
    def _voice_path(voice_id: str) -> str:
        vid = voice_id[len("voices/"):] if voice_id.startswith("voices/") else voice_id
        if not vid:
            raise ValueError("voice_id must not be empty")
        return "/voices/" + urllib.parse.quote(vid, safe="")

    def create_voice(self, prompt: str, display_name: str, *, gender: str | None = None,
                     language_code: str = "en-US", model: str = MODEL, store: bool = True) -> dict:
        voice: dict[str, Any] = {"model": model, "type": "prompted", "display_name": display_name,
                                 "language_code": language_code, "prompted": {"input": prompt}}
        if gender:
            voice["gender"] = gender
        return self._call("POST", "/voices", {"store": store, "voice": voice})

    def get_voice(self, voice_id: str) -> dict:
        return self._call("GET", self._voice_path(voice_id))

    def list_voices(self, voice_type: str | None = "prompted", page_token: str | None = None) -> dict:
        q: dict[str, str] = {}
        if voice_type:
            q["type"] = voice_type
        if page_token:
            q["pageToken"] = page_token
        return self._call("GET", "/voices", query=q or None)

    def delete_voice(self, voice_id: str) -> dict:
        return self._call("DELETE", self._voice_path(voice_id))

    def synthesize(self, voice_id: str, text: str, *, style: str | None = None, model: str = MODEL) -> bytes:
        """Render ``text`` with a stored voice via ``/interactions``; returns the audio bytes (WAV)."""
        content: dict[str, Any] = {"type": "text", "text": text}
        if style:
            content["annotations"] = [{"type": "speech_metadata", "style": style}]
        resp = self._call("POST", "/interactions", {
            "model": model,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": voice_id}]},
        })
        audio = [c for step in resp.get("steps", []) if step.get("type") == "model_output"
                 for c in step.get("content", []) if c.get("type") == "audio" and c.get("data")]
        if not audio:
            raise GeminiError("synthesis returned no audio")
        return base64.b64decode(audio[-1]["data"])


# ----------------------------------------------------------------------------- adapter


def response_voice_id(resp: Mapping[str, Any]) -> str | None:
    vid = resp.get("id") or resp.get("voice_id")
    if not vid and isinstance(resp.get("name"), str):
        vid = resp["name"].split("/")[-1]
    return vid if isinstance(vid, str) and vid else None


def response_sample(resp: Mapping[str, Any]) -> bytes | None:
    sa = resp.get("sample_audio") or resp.get("sampleAudio") or {}
    data = sa.get("data") if isinstance(sa, dict) else None
    return base64.b64decode(data) if data else None


def response_expire_time(resp: Mapping[str, Any]) -> str | None:
    v = resp.get("expire_time") or resp.get("expireTime")
    return v if isinstance(v, str) else None


def redact_response(resp: Mapping[str, Any]) -> dict:
    """Copy of an API response with base64 audio replaced by a size marker (for display)."""
    out = dict(resp)
    for k in ("sample_audio", "sampleAudio"):
        if isinstance(out.get(k), dict):
            out[k] = {**out[k], "data": f"<{len(out[k].get('data', ''))} b64 chars>"}
    return out


def _ordered_ext(ext: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: ext[k] for k in EXTENSION_ORDER if ext.get(k) is not None}
    out.update({k: v for k, v in ext.items() if k not in out and v is not None})
    return out


@register_adapter
class GeminiAdapter(ProviderAdapter):
    """Maps Gemini Voices API responses into ``extensions.google`` + a sample embedding."""

    provider = ENGINE
    engine = ENGINE
    model = MODEL

    def voice_from_response(self, response: Mapping[str, Any], *, display_name: str | None = None,
                            prompt: str | None = None, gender: str | None = None,
                            language_code: str | None = None, voice_type: str = "prompted",
                            created: str | None = None, refreshed: str | None = None,
                            base: Mapping[str, Any] | None = None, **_: Any) -> ProviderVoice:
        """``base`` is an existing ``extensions.google`` dict to update (refresh)."""
        ext: dict[str, Any] = dict(base or {})
        vid = response_voice_id(response) or ext.get("voice_id")
        if not vid:
            raise GeminiError("API response has no voice id")
        ext.update({
            "voice_id": vid,
            "model_id": self.model,
            "type": response.get("type") or ext.get("type") or voice_type,
            "display_name": display_name or response.get("display_name") or response.get("displayName")
            or ext.get("display_name"),
            "gender": gender or response.get("gender") or ext.get("gender"),
            "language_code": language_code or response.get("language_code") or response.get("languageCode")
            or ext.get("language_code"),
            "prompted_input": prompt or ((response.get("prompted") or {}).get("input")
                                         if isinstance(response.get("prompted"), dict) else None)
            or ext.get("prompted_input"),
            "created": created or ext.get("created") or now_iso(),
            "expire_time": response_expire_time(response) or ext.get("expire_time"),
            "refreshed": refreshed or ext.get("refreshed"),
        })
        samples = []
        wav = response_sample(response)
        if wav:
            samples.append(ProviderSample(model=self.model, data=wav, model_slug=self.model,
                                          description=SAMPLE_DESCRIPTION))
        return ProviderVoice(provider=self.provider, engine=self.engine, extension=_ordered_ext(ext),
                             samples=samples)

    def read_extension(self, vox: "VoxFile", *, migrate_legacy: bool = True) -> dict[str, Any] | None:
        """``extensions.google``; migrates the pre-fix ``voice.json`` PoC layout if found."""
        ext = self.extension(vox)
        if ext is not None:
            return ext
        legacy = vox[LEGACY_VOICE_JSON]
        if legacy is None or not migrate_legacy:
            return None
        old = json.loads(legacy.data)
        ext = _ordered_ext({
            "voice_id": old.get("id"), "model_id": self.model, "type": old.get("type"),
            "display_name": old.get("display_name"), "gender": old.get("gender"),
            "language_code": old.get("language_code"), "prompted_input": old.get("prompt"),
            "created": old.get("created"), "expire_time": old.get("expire_time"),
        })
        vox.remove(LEGACY_VOICE_JSON)
        emb = vox.manifest.get("embeddings")
        if isinstance(emb, dict):
            emb.pop(LEGACY_VOICE_REF_KEY, None)
            if not emb:
                del vox.manifest["embeddings"]
        vox.set_extension(self.provider, ext)
        return ext


# ------------------------------------------------------------------------- operations


@dataclass
class UpdateResult:
    old_voice_id: str
    new_voice_id: str
    old_deleted: bool
    delete_error: str | None = None


def _defaults(vox: "VoxFile", prompt, display_name, gender, language_code, prev: Mapping[str, Any] | None):
    voice = vox.manifest.get("voice") if isinstance(vox.manifest.get("voice"), dict) else {}
    prev = prev or {}
    prompt = prompt or prev.get("prompted_input") or voice.get("description")
    display_name = display_name or prev.get("display_name") or voice.get("name")
    g = voice.get("gender")
    gender = gender or prev.get("gender") or (g if g in ("male", "female") else None)
    language_code = language_code or prev.get("language_code") or voice.get("language") or "en-US"
    if not prompt or not display_name:
        raise GeminiError("a prompt and display name are required (voice.description / voice.name are empty)")
    return prompt, display_name, gender, language_code


def create_voice_for_vox(vox: "VoxFile", client: GeminiVoicesClient, *, prompt: str | None = None,
                         display_name: str | None = None, gender: str | None = None,
                         language_code: str | None = None, force: bool = False,
                         adapter: GeminiAdapter | None = None) -> ProviderVoice:
    """POST a new designed voice and attach it. Defaults come from ``manifest.voice``."""
    adapter = adapter or GeminiAdapter()
    if adapter.read_extension(vox) is not None and not force:
        raise GeminiError("this .vox already has extensions.google; use 'update' (or force=True) "
                          "so the existing server voice is not orphaned")
    prompt, display_name, gender, language_code = _defaults(vox, prompt, display_name, gender, language_code, None)
    resp = client.create_voice(prompt, display_name, gender=gender, language_code=language_code,
                               model=adapter.model)
    voice = adapter.voice_from_response(resp, display_name=display_name, prompt=prompt, gender=gender,
                                        language_code=language_code, created=now_iso())
    attach(vox, voice)
    return voice


def refresh_vox(vox: "VoxFile", client: GeminiVoicesClient, *, adapter: GeminiAdapter | None = None) -> ProviderVoice:
    """GET the stored voice; refresh the sample, ``expire_time`` and ``refreshed``."""
    adapter = adapter or GeminiAdapter()
    ext = adapter.read_extension(vox)
    if not ext or not ext.get("voice_id"):
        raise GeminiError("this .vox has no extensions.google.voice_id")
    resp = client.get_voice(ext["voice_id"])
    voice = adapter.voice_from_response(resp, base=ext, refreshed=now_iso())
    attach(vox, voice)
    return voice


def update_vox(vox: "VoxFile", client: GeminiVoicesClient, *, prompt: str | None = None,
               display_name: str | None = None, gender: str | None = None, language_code: str | None = None,
               write: bool = True, adapter: GeminiAdapter | None = None) -> UpdateResult:
    """Recreate the voice (the API has no PATCH), write the ``.vox``, then delete the old id.

    The old voice is deleted only after the new one is safely written, so a failure at
    any point leaves a working voice referenced by the file.
    """
    adapter = adapter or GeminiAdapter()
    old = adapter.read_extension(vox)
    if not old or not old.get("voice_id"):
        raise GeminiError("this .vox has no extensions.google.voice_id; use 'create'")
    old_id = old["voice_id"]
    prompt, display_name, gender, language_code = _defaults(vox, prompt, display_name, gender, language_code, old)
    resp = client.create_voice(prompt, display_name, gender=gender, language_code=language_code,
                               model=adapter.model)
    voice = adapter.voice_from_response(resp, display_name=display_name, prompt=prompt, gender=gender,
                                        language_code=language_code, created=now_iso())
    vox.remove_extension(adapter.provider)  # full replace: drop stale fields such as "refreshed"
    attach(vox, voice)
    new_id = voice.extension["voice_id"] if voice.extension else ""
    if write:
        vox.write()
    if new_id == old_id:
        return UpdateResult(old_id, new_id, False, "new voice id equals old id; old voice kept")
    try:
        client.delete_voice(old_id)
    except GeminiError as e:
        return UpdateResult(old_id, new_id, False, str(e))
    return UpdateResult(old_id, new_id, True)


def delete_voice(client: GeminiVoicesClient, voice_id: str, vox: "VoxFile | None" = None,
                 *, adapter: GeminiAdapter | None = None) -> list[str]:
    """DELETE a server voice; if ``vox`` is given and references it, detach it too."""
    adapter = adapter or GeminiAdapter()
    client.delete_voice(voice_id)
    if vox is None:
        return []
    ext = adapter.read_extension(vox)
    if ext and ext.get("voice_id") == voice_id:
        return adapter.detach(vox)
    return []
