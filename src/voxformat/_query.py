"""Model / language queries (mirrors ``VoxFile+ModelQuery.swift``).

``language`` is a BCP 47 *storage-slot selector* (region subtags preserved), not an
engine generation language. Resolution for ``(model, language)``:

1. exact language (case-insensitive)
2. base language (``fr-FR`` -> stored ``fr``)
3. default / language-neutral entry (no ``language`` field)
4. legacy path ``embeddings/qwen3-tts/{clone-prompt.bin|sample-audio.wav}``

``language=None`` or ``"default"`` resolves only step 3 (then 4). An entry tagged with a
``language`` is therefore never returned by a default lookup.

Within one candidate step the first matching entry in *manifest order* wins (Swift
iterates a ``Dictionary``, whose order is undefined). As in Swift, once an entry matches,
its data is returned even if the file is absent from the archive (``None``) - the chain
does not continue to later candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .manifest import EmbeddingEntry, embedding_items

LEGACY_CLONE_PROMPT = "embeddings/qwen3-tts/clone-prompt.bin"
LEGACY_SAMPLE_AUDIO = "embeddings/qwen3-tts/sample-audio.wav"

ResolvedVia = Literal["exact", "base", "default", "legacy"]


@dataclass(frozen=True)
class Resolved:
    """Result of a language-aware lookup."""

    path: str
    data: Optional[bytes]
    via: ResolvedVia
    entry: Optional[EmbeddingEntry] = None


def language_candidates(language: str | None) -> list[str | None]:
    """Swift ``languageCandidates(for:)``."""
    if language is None or language.lower() == "default":
        return [None]
    lang = language.lower()
    candidates: list[str | None] = [lang]
    dash = lang.find("-")
    if dash != -1:
        base = lang[:dash]
        if base and base != lang:
            candidates.append(base)
    candidates.append(None)
    return candidates


def _contains(haystack: str, needle: str) -> bool:
    """Foundation ``String.contains`` semantics: an empty needle never matches."""
    return bool(needle) and needle in haystack


def _lang_matches(stored: Any, candidate: str | None) -> bool:
    stored = stored if isinstance(stored, str) else None
    if stored is None and candidate is None:
        return True
    if stored is not None and candidate is not None:
        return stored.lower() == candidate
    return False


class ModelQueryMixin:
    manifest: dict[str, Any]

    def _entry_data(self, path: str) -> Optional[bytes]:  # provided by VoxFile
        raise NotImplementedError

    # -- basic model matching -------------------------------------------------

    def embedding_entry(self, query: str) -> EmbeddingEntry | None:
        """Swift ``embeddingEntry(for:)``: exact key, ci key, ci model-contains, ci key-contains."""
        items = embedding_items(self.manifest)
        if not items:
            return None
        q = query.lower()
        for key, d in items:
            if key == query:
                return EmbeddingEntry.from_dict(key, d)
        for key, d in items:
            if key.lower() == q:
                return EmbeddingEntry.from_dict(key, d)
        for key, d in items:
            if _contains(str(d.get("model", "")).lower(), q):
                return EmbeddingEntry.from_dict(key, d)
        for key, d in items:
            if _contains(key.lower(), q):
                return EmbeddingEntry.from_dict(key, d)
        return None

    def supports_model(self, query: str) -> bool:
        return self.embedding_entry(query) is not None

    def embedding_data(self, query: str) -> Optional[bytes]:
        entry = self.embedding_entry(query)
        return self._entry_data(entry.file) if entry else None

    @property
    def supported_models(self) -> list[str]:
        """All ``model`` values (manifest order, duplicates kept - same as Swift)."""
        return [str(d.get("model", "")) for _, d in embedding_items(self.manifest)]

    def reference_audio_for(self, model: str) -> list[dict[str, Any]]:
        """Swift ``referenceAudio(for:)``: model-matched clips, else universal clips."""
        clips = [c for c in (self.manifest.get("reference_audio") or []) if isinstance(c, dict)]
        q = model.lower()
        matched = [c for c in clips if isinstance(c.get("model"), str) and _contains(c["model"].lower(), q)]
        if matched:
            return matched
        return [c for c in clips if c.get("model") is None]

    # -- language-aware resolution -------------------------------------------

    def _resolve(self, substring: str, query: str, language: str | None, legacy: str) -> Resolved | None:
        items = embedding_items(self.manifest)
        if not items:
            data = self._entry_data(legacy)
            return Resolved(legacy, data, "legacy") if data is not None else None
        q = query.lower()
        candidates = language_candidates(language)
        for i, candidate in enumerate(candidates):
            via: ResolvedVia = "default" if candidate is None else ("exact" if i == 0 else "base")
            for key, d in items:
                file = d.get("file")
                if not isinstance(file, str) or substring not in file:
                    continue
                if not (_contains(key.lower(), q) or _contains(str(d.get("model", "")).lower(), q)):
                    continue
                if _lang_matches(d.get("language"), candidate):
                    return Resolved(file, self._entry_data(file), via, EmbeddingEntry.from_dict(key, d))
        data = self._entry_data(legacy)
        return Resolved(legacy, data, "legacy") if data is not None else None

    def resolve_sample_audio(self, query: str, language: str | None = None) -> Resolved | None:
        return self._resolve("sample-audio", query, language, LEGACY_SAMPLE_AUDIO)

    def resolve_clone_prompt(self, query: str, language: str | None = None) -> Resolved | None:
        return self._resolve("clone-prompt", query, language, LEGACY_CLONE_PROMPT)

    def sample_audio_data(self, query: str, language: str | None = None) -> Optional[bytes]:
        r = self.resolve_sample_audio(query, language)
        return r.data if r else None

    def clone_prompt_data(self, query: str, language: str | None = None) -> Optional[bytes]:
        r = self.resolve_clone_prompt(query, language)
        return r.data if r else None

    def _languages(self, substring: str, query: str) -> list[str]:
        q = query.lower()
        seen: set[str] = set()
        result: list[str] = []
        for key, d in embedding_items(self.manifest):
            file = d.get("file")
            if not isinstance(file, str) or substring not in file:
                continue
            if not (_contains(key.lower(), q) or _contains(str(d.get("model", "")).lower(), q)):
                continue
            lang = d.get("language")
            if not isinstance(lang, str):
                continue
            if lang.lower() not in seen:
                seen.add(lang.lower())
                result.append(lang)
        return sorted(result)

    def sample_audio_languages(self, query: str) -> list[str]:
        return self._languages("sample-audio", query)

    def clone_prompt_languages(self, query: str) -> list[str]:
        return self._languages("clone-prompt", query)
