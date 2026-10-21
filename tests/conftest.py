from __future__ import annotations

import socket
from pathlib import Path

import pytest

from voxformat import VoxFile

# Inside vox-format this is the repo root; in the standalone package the spec's
# examples/ and schemas/ are vendored under tests/fixtures/vox-format/.
_IN_SPEC_REPO = Path(__file__).resolve().parents[3]
REPO_ROOT = (_IN_SPEC_REPO if (_IN_SPEC_REPO / "schemas" / "manifest-v0.4.0.json").is_file()
             else Path(__file__).resolve().parent / "fixtures" / "vox-format")
EXAMPLES = REPO_ROOT / "examples"
SCHEMAS = REPO_ROOT / "schemas"

QWEN_06B = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_SAMPLE = b"RIFF\x00"
ES_SAMPLE = b"RIFF\xe5"
FR_SAMPLE = b"RIFF\xff"
DEFAULT_PROMPT = b"\xc0\x00"
ES_PROMPT = b"\xc0\xe5"
FR_PROMPT = b"\xc0\xff\x01"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Tests must never touch the network."""

    def guard(*args, **kwargs):
        raise AssertionError("network access attempted in tests")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def multi_language_vox() -> VoxFile:
    """Port of Swift ``EmbeddingLanguageTests.makeMultiLanguageVox()``."""
    vox = VoxFile("TestVoice", "A test voice with sufficient description length.")
    m = {"model": QWEN_06B, "engine": "qwen3-tts"}
    vox.add(DEFAULT_PROMPT, "embeddings/qwen3-tts/0.6b/clone-prompt.bin", metadata={**m, "format": "bin"})
    vox.add(DEFAULT_SAMPLE, "embeddings/qwen3-tts/0.6b/sample-audio.wav", metadata={**m, "format": "wav"})
    vox.add(ES_PROMPT, "embeddings/qwen3-tts/0.6b/es/clone-prompt.bin", metadata={**m, "format": "bin", "language": "es"})
    vox.add(ES_SAMPLE, "embeddings/qwen3-tts/0.6b/es/sample-audio.wav", metadata={**m, "format": "wav", "language": "es"})
    vox.add(FR_PROMPT, "embeddings/qwen3-tts/0.6b/fr-FR/clone-prompt.bin",
            metadata={**m, "format": "bin", "language": "fr-FR"})
    vox.add(FR_SAMPLE, "embeddings/qwen3-tts/0.6b/fr-FR/sample-audio.wav",
            metadata={**m, "format": "wav", "language": "fr-FR"})
    return vox


def multi_model_vox() -> VoxFile:
    """Qwen 0.6b + 1.7b clone prompts and samples plus a Gemini sample."""
    vox = VoxFile("CLARK", "Male, 30s. A quiet, low tenor voice used only for tests.")
    for slug, model, cp, wav in (
        ("0.6b", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", b"\xc0\x01", b"RIFF06"),
        ("1.7b", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", b"\xc1\x71", b"RIFF17"),
    ):
        vox.add(cp, f"embeddings/qwen3-tts/{slug}/clone-prompt.bin",
                metadata={"model": model, "engine": "qwen3-tts", "format": "bin"})
        vox.add(wav, f"embeddings/qwen3-tts/{slug}/sample-audio.wav",
                metadata={"model": model, "engine": "qwen3-tts", "format": "wav"})
    vox.add(b"RIFFGEM", "embeddings/google/gemini-3.8-flash-tts/sample-audio.wav",
            metadata={"model": "gemini-3.8-flash-tts", "engine": "google", "format": "wav"})
    return vox


@pytest.fixture
def mm_vox() -> VoxFile:
    return multi_model_vox()
