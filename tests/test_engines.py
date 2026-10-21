"""Engine/provider attachment API and the handle adapters."""

from __future__ import annotations

import warnings

import pytest

from voxformat import VoxFile, add_engine_embedding, add_engine_sample, derive_model_slug, set_provider_extension
from voxformat.engines import UnknownProviderWarning, embedding_path
from voxformat.providers import (
    ProviderAdapter,
    ProviderEmbedding,
    ProviderSample,
    ProviderVoice,
    attach,
    detach,
    get_adapter,
    register_adapter,
    registered_adapters,
)


def new():
    return VoxFile("T", "A test voice with a long enough description.")


def test_model_slug():
    assert derive_model_slug("gemini-3.8-flash-tts") == "gemini-3.8-flash-tts"
    assert derive_model_slug("Qwen/Qwen3-TTS-12Hz-0.6B") == "qwen3-tts-12hz-0.6b"
    with pytest.raises(ValueError):
        derive_model_slug("///")


def test_embedding_path_and_validation():
    assert embedding_path("qwen3-tts", "0.6b", "clone-prompt.bin") == "embeddings/qwen3-tts/0.6b/clone-prompt.bin"
    assert embedding_path("qwen3-tts", "0.6b", "clone-prompt.bin", "fr-FR") == \
        "embeddings/qwen3-tts/0.6b/fr-FR/clone-prompt.bin"
    for bad in [("..", "x", "n"), ("e", "a/b", "n"), ("e", "s", "n", "default"), ("e", "s", "n", "not a tag")]:
        with pytest.raises(ValueError):
            embedding_path(*bad)


def test_add_engine_embedding_qwen_layout():
    vox = new()
    e = add_engine_embedding(vox, b"\xc0", engine="qwen3-tts", model="Qwen/Qwen3-TTS-12Hz-0.6B", model_slug="0.6b")
    assert e.key == "qwen3-tts-0.6b-clone-prompt"
    assert vox.manifest["embeddings"]["qwen3-tts-0.6b-clone-prompt"] == {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts",
        "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin", "format": "bin"}
    es = add_engine_embedding(vox, b"\xe5", engine="qwen3-tts", model="Qwen/Qwen3-TTS-12Hz-0.6B",
                              model_slug="0.6b", language="es")
    assert es.key == "qwen3-tts-0.6b-es-clone-prompt" and es.language == "es"
    st = add_engine_embedding(vox, b"st", engine="coqui", model="xtts-v2", name="speaker.safetensors",
                              description="XTTS speaker latents")
    assert st.file == "embeddings/coqui/xtts-v2/speaker.safetensors" and st.format == "safetensors"
    assert vox.clone_prompt_data("0.6b", "es-MX") == b"\xe5"
    assert vox.validate() == []


def test_add_engine_sample_is_language_neutral_by_default():
    vox = new()
    e = add_engine_sample(vox, b"RIFF", engine="google", model="gemini-3.8-flash-tts")
    assert e.key == "google-gemini-3.8-flash-tts-sample-audio"
    assert vox.manifest["embeddings"][e.key] == {
        "model": "gemini-3.8-flash-tts", "engine": "google",
        "file": "embeddings/google/gemini-3.8-flash-tts/sample-audio.wav", "format": "wav",
        "description": "Engine-generated voice sample (gemini-3.8-flash-tts)"}
    assert "language" not in vox.manifest["embeddings"][e.key]
    assert vox.sample_audio_data("gemini") == b"RIFF"
    lang = add_engine_sample(vox, b"RIFFes", engine="google", model="gemini-3.8-flash-tts", language="es")
    assert lang.file.endswith("/es/sample-audio.wav") and lang.language == "es"
    assert vox.sample_audio_data("gemini") == b"RIFF"
    assert vox.sample_audio_data("gemini", "es") == b"RIFFes"


def test_set_provider_extension_reserved_and_unknown():
    vox = new()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_provider_extension(vox, "elevenlabs", {"voice_id": "vid-example-abc123", "model_id": None})
    assert vox.extensions["elevenlabs"] == {"voice_id": "vid-example-abc123"}  # None dropped
    with pytest.warns(UnknownProviderWarning):
        set_provider_extension(vox, "acme-tts", {"voice": "x"})
    assert "acme-tts" in vox.extensions
    assert not any(p.startswith("embeddings/") for p in (e.path for e in vox.entries))  # handles are not files


def test_handle_adapters():
    assert {"apple", "elevenlabs", "google", "openai"} <= set(registered_adapters())
    vox = new()
    get_adapter("elevenlabs").attach(vox, {"voice_id": "vid-example-abc123", "name": "Narrator"},
                                     model_id="eleven_multilingual_v2")
    get_adapter("openai").attach(vox, {"voice": "alloy"}, model_id="gpt-4o-mini-tts")
    get_adapter("apple").attach(vox, {"identifier": "en-US/Aaron"}, fallback=True)
    assert vox.extensions == {
        "elevenlabs": {"voice_id": "vid-example-abc123", "model_id": "eleven_multilingual_v2", "name": "Narrator"},
        "openai": {"voice_id": "alloy", "model_id": "gpt-4o-mini-tts"},
        "apple": {"voice_id": "en-US/Aaron", "fallback": True},
    }
    assert vox.entry_count == 0
    with pytest.raises(ValueError):
        get_adapter("elevenlabs").voice_from_response({})
    with pytest.raises(KeyError):
        get_adapter("nope")


def test_custom_adapter_plugs_in():
    @register_adapter
    class AzureAdapter(ProviderAdapter):
        provider = "azure"

        def voice_from_response(self, response, **context):
            return ProviderVoice(
                provider=self.provider,
                extension={"voice_id": response["ShortName"]},
                samples=[ProviderSample(model="azure-neural", data=b"RIFFaz")],
                embeddings=[ProviderEmbedding(model="azure-neural", data=b"\x01", name="profile.bin",
                                              language="en-US")],
            )

    vox = new()
    get_adapter("azure").attach(vox, {"ShortName": "en-US-AvaNeural"})
    assert vox.extensions["azure"] == {"voice_id": "en-US-AvaNeural"}
    assert vox["embeddings/azure/azure-neural/sample-audio.wav"].data == b"RIFFaz"
    assert vox["embeddings/azure/azure-neural/en-US/profile.bin"].data == b"\x01"
    assert set(vox.manifest["embeddings"]) == {"azure-azure-neural-sample-audio", "azure-azure-neural-en-US-profile"}
    removed = get_adapter("azure").detach(vox)
    assert "extensions.azure" in removed and vox.entry_count == 0 and "embeddings" not in vox.manifest


def test_attach_and_detach_functions():
    vox = new()
    attach(vox, ProviderVoice(provider="qwen3-tts", extension={"design_instruct": "warm"},
                              embeddings=[ProviderEmbedding(model="Qwen/Qwen3-TTS-12Hz-1.7B", data=b"\x17",
                                                            name="clone-prompt.bin", model_slug="1.7b")]))
    assert vox.clone_prompt_data("1.7b") == b"\x17"
    detach(vox, "qwen3-tts", remove_files=False)
    assert "extensions" not in vox.manifest and vox.entry_count == 1
