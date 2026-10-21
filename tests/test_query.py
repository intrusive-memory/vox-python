"""Model queries + v0.4.0 language fallback chain (ports EmbeddingLanguageTests / EmbeddingEntryTests)."""

from __future__ import annotations

import pytest

from voxformat import VoxFile
from voxformat._query import language_candidates

from conftest import (
    DEFAULT_PROMPT,
    DEFAULT_SAMPLE,
    ES_PROMPT,
    ES_SAMPLE,
    FR_SAMPLE,
)


# --- language candidates -----------------------------------------------------

@pytest.mark.parametrize("lang,expected", [
    (None, [None]),
    ("default", [None]),
    ("DEFAULT", [None]),
    ("es", ["es", None]),
    ("fr-FR", ["fr-fr", "fr", None]),
    ("zh-Hant-TW", ["zh-hant-tw", "zh", None]),
])
def test_language_candidates(lang, expected):
    assert language_candidates(lang) == expected


# --- exact / base / default ---------------------------------------------------

def test_exact_language(multi_language_vox):
    assert multi_language_vox.sample_audio_data("0.6b", "es") == ES_SAMPLE
    assert multi_language_vox.clone_prompt_data("0.6b", "es") == ES_PROMPT


def test_exact_language_case_insensitive(multi_language_vox):
    assert multi_language_vox.sample_audio_data("0.6b", "ES") == ES_SAMPLE


def test_falls_back_to_default_when_language_absent(multi_language_vox):
    assert multi_language_vox.sample_audio_data("0.6b", "de") == DEFAULT_SAMPLE
    assert multi_language_vox.clone_prompt_data("0.6b", "de") == DEFAULT_PROMPT


def test_base_language_fallback(multi_language_vox):
    r = multi_language_vox.resolve_sample_audio("0.6b", "es-MX")
    assert r.data == ES_SAMPLE and r.via == "base"


def test_exact_beats_base(multi_language_vox):
    r = multi_language_vox.resolve_sample_audio("0.6b", "fr-FR")
    assert r.data == FR_SAMPLE and r.via == "exact"


def test_region_query_falls_back_to_default(multi_language_vox):
    r = multi_language_vox.resolve_sample_audio("0.6b", "de-DE")
    assert r.data == DEFAULT_SAMPLE and r.via == "default"


def test_nil_language_returns_default_and_never_language_tagged(multi_language_vox):
    assert multi_language_vox.sample_audio_data("0.6b") == DEFAULT_SAMPLE
    assert multi_language_vox.sample_audio_data("0.6b", None) == DEFAULT_SAMPLE
    assert multi_language_vox.sample_audio_data("0.6b", "default") == DEFAULT_SAMPLE
    assert multi_language_vox.clone_prompt_data("0.6b") == DEFAULT_PROMPT


def test_language_tagged_only_is_not_returned_by_default_lookup():
    vox = VoxFile("OnlyEs", "A voice that only ships a Spanish sample, nothing else.")
    vox.add(ES_SAMPLE, "embeddings/qwen3-tts/0.6b/es/sample-audio.wav",
            metadata={"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "language": "es"})
    assert vox.sample_audio_data("0.6b") is None
    assert vox.sample_audio_data("0.6b", "es") == ES_SAMPLE
    assert vox.sample_audio_data("0.6b", "de") is None


def test_unknown_model_and_no_samples(multi_language_vox):
    assert multi_language_vox.sample_audio_data("3.0b", "es") is None
    empty = VoxFile("Empty", "A voice with no embeddings whatsoever, just metadata.")
    assert empty.sample_audio_data("0.6b", "es") is None
    assert empty.sample_audio_data("0.6b") is None


def test_discovery_helpers(multi_language_vox):
    assert multi_language_vox.sample_audio_languages("0.6b") == ["es", "fr-FR"]
    assert multi_language_vox.clone_prompt_languages("0.6b") == ["es", "fr-FR"]
    default_only = VoxFile("DefaultOnly", "A voice with only a default sample, no languages.")
    default_only.add(b"RIFF", "embeddings/qwen3-tts/0.6b/sample-audio.wav",
                     metadata={"model": "Qwen/Qwen3-TTS-12Hz-0.6B"})
    assert default_only.sample_audio_languages("0.6b") == []


def test_old_style_vox_language_query_resolves_default():
    vox = VoxFile("Legacy", "A pre-0.4.0 voice with only language-neutral samples.")
    m = {"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts"}
    vox.add(b"\xc0\x00", "embeddings/qwen3-tts/0.6b/clone-prompt.bin", metadata=m)
    vox.add(DEFAULT_SAMPLE, "embeddings/qwen3-tts/0.6b/sample-audio.wav", metadata=m)
    for lang in ("es", "fr-FR", None):
        assert vox.sample_audio_data("0.6b", lang) == DEFAULT_SAMPLE
    assert vox.clone_prompt_data("0.6b", "es") == b"\xc0\x00"


def test_legacy_path_fallback():
    vox = VoxFile.from_manifest({"vox_version": "0.1.0", "id": "12345678-1234-4234-8234-123456789abc",
                                 "created": "2026-01-01T00:00:00Z",
                                 "voice": {"name": "L", "description": "A legacy voice, no embedding entries."}})
    assert vox.clone_prompt_data("1.7b") is None
    # Inject a legacy binary without a manifest entry (as an old archive would have).
    from voxformat.entry import VoxEntry
    vox._storage["embeddings/qwen3-tts/clone-prompt.bin"] = VoxEntry("embeddings/qwen3-tts/clone-prompt.bin", b"\xaa")
    vox._storage["embeddings/qwen3-tts/sample-audio.wav"] = VoxEntry("embeddings/qwen3-tts/sample-audio.wav", b"RIFFL")
    assert vox.clone_prompt_data("anything") == b"\xaa"
    r = vox.resolve_sample_audio("anything", "es")
    assert r.via == "legacy" and r.data == b"RIFFL"


def test_matched_entry_with_missing_file_returns_none_like_swift(multi_language_vox):
    # Swift returns self[entry.file]?.data for the first match without continuing the chain.
    multi_language_vox._storage.pop("embeddings/qwen3-tts/0.6b/es/sample-audio.wav")
    r = multi_language_vox.resolve_sample_audio("0.6b", "es")
    assert r is not None and r.data is None and r.path.endswith("es/sample-audio.wav")
    assert multi_language_vox.sample_audio_data("0.6b", "es") is None


# --- embedding_entry / supports_model ----------------------------------------

def test_clone_prompt_vs_sample_disambiguation(mm_vox):
    for _ in range(20):
        assert mm_vox.clone_prompt_data("1.7b") == b"\xc1\x71"
    assert mm_vox.clone_prompt_data("0.6b") == b"\xc0\x01"
    assert mm_vox.sample_audio_data("0.6b") == b"RIFF06"
    assert mm_vox.sample_audio_data("1.7b") == b"RIFF17"
    assert mm_vox.clone_prompt_data("3.0b") is None


@pytest.mark.parametrize("query,expected", [
    ("1.7b", b"RIFF17"),
    ("qwen3-tts-1.7b", b"RIFF17"),
    ("0.6B", b"RIFF06"),
    ("gemini", b"RIFFGEM"),
    ("gemini-3.8-flash-tts", b"RIFFGEM"),
    ("google", b"RIFFGEM"),
])
def test_player_style_queries(mm_vox, query, expected):
    assert mm_vox.sample_audio_data(query) == expected


def test_supports_model_priority():
    vox = VoxFile("T", "A test voice with a long enough description.")
    vox.add(b"\x00\x01\x02\x03", "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
            metadata={"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts", "key": "qwen3-tts-0.6b"})
    vox.add(b"\x10\x11\x12\x13\x14", "embeddings/qwen3-tts/1.7b/clone-prompt.bin",
            metadata={"model": "Qwen/Qwen3-TTS-12Hz-1.7B", "engine": "qwen3-tts", "key": "qwen3-tts-1.7b"})
    for q in ("qwen3-tts-0.6b", "QWEN3-TTS-0.6B", "Qwen3-TTS-1.7b", "0.6B", "1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B"):
        assert vox.supports_model(q), q
    for q in ("3.0b", "whisper"):
        assert not vox.supports_model(q)
    assert vox.embedding_data("0.6b") == b"\x00\x01\x02\x03"
    assert vox.embedding_data("1.7b") == b"\x10\x11\x12\x13\x14"
    assert sorted(vox.supported_models) == ["Qwen/Qwen3-TTS-12Hz-0.6B", "Qwen/Qwen3-TTS-12Hz-1.7B"]
    e = vox.embedding_entry("0.6b")
    assert e.model == "Qwen/Qwen3-TTS-12Hz-0.6B" and e.engine == "qwen3-tts"


def test_empty_query_matches_nothing_like_foundation(mm_vox):
    # Swift test: XCTAssertFalse(vox.supportsModel("")) - Foundation's contains("") is false.
    assert not mm_vox.supports_model("")
    assert mm_vox.sample_audio_data("") is None
    assert mm_vox.sample_audio_languages("") == []
    assert VoxFile("T", "A test voice with a long enough description.").supported_models == []


def test_reference_audio_for_model():
    vox = VoxFile("T", "A test voice with a long enough description.")
    vox.add(b"RIFF", "reference/universal.wav", metadata={"transcript": "Hello there."})
    vox.add(b"RIFF", "reference/q.wav", metadata={"transcript": "Hi.", "model": "Qwen/Qwen3-TTS-12Hz-0.6B"})
    assert [c["file"] for c in vox.reference_audio_for("0.6b")] == ["reference/q.wav"]
    assert [c["file"] for c in vox.reference_audio_for("1.7b")] == ["reference/universal.wav"]
