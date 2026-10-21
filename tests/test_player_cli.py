"""Player resolution (subprocess mocked) and CLI commands."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from voxformat import PlayerNotFoundError, SampleNotFoundError, VoxFile, player
from voxformat.cli import main

from conftest import EXAMPLES, multi_model_vox


@pytest.fixture
def mm_path(tmp_path) -> Path:
    return multi_model_vox().write(tmp_path / "CLARK.vox")


# --- player --------------------------------------------------------------------

def test_playable_samples(mm_vox):
    s = {x.key: x for x in player.playable_samples(mm_vox)}
    assert set(s) == {"qwen3-tts-0.6b-sample-audio", "qwen3-tts-1.7b-sample-audio",
                      "google-gemini-3.8-flash-tts-sample-audio"}
    assert all(x.present and x.language is None for x in s.values())


def test_resolve_errors(mm_vox):
    with pytest.raises(SampleNotFoundError, match="Available samples"):
        player.resolve(mm_vox, "3.0b")
    mm_vox._storage.pop("embeddings/qwen3-tts/1.7b/sample-audio.wav")
    with pytest.raises(SampleNotFoundError, match="missing from the archive"):
        player.resolve(mm_vox, "1.7b")


def test_find_player_order_and_fallback():
    assert player.find_player(which=lambda n: f"/usr/bin/{n}")[0] == "/usr/bin/afplay"
    only_ff = lambda n: "/opt/bin/ffplay" if n == "ffplay" else None  # noqa: E731
    assert player.find_player(which=only_ff) == ["/opt/bin/ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    only_aplay = lambda n: "/usr/bin/aplay" if n == "aplay" else None  # noqa: E731
    assert player.find_player(which=only_aplay) == ["/usr/bin/aplay", "-q"]
    with pytest.raises(PlayerNotFoundError):
        player.find_player(which=lambda n: None)
    with pytest.raises(PlayerNotFoundError):
        player.find_player("mpv", which=lambda n: None)


@pytest.mark.parametrize("query,lang,expected", [
    ("1.7b", None, b"RIFF17"),
    ("0.6b", "es", b"RIFF06"),  # no Spanish sample -> default
    ("gemini", None, b"RIFFGEM"),
])
def test_play_invokes_player_with_resolved_wav(mm_vox, query, lang, expected):
    seen = {}

    def runner(argv, check):
        seen["argv"] = argv
        seen["data"] = Path(argv[-1]).read_bytes()

    r = player.play(mm_vox, query, lang, runner=runner, which=lambda n: f"/usr/bin/{n}")
    assert seen["argv"][0] == "/usr/bin/afplay" and seen["data"] == expected
    assert not Path(seen["argv"][-1]).exists()  # temp file cleaned up
    assert r.data == expected


def test_extract(mm_vox, tmp_path):
    r, out = player.extract(mm_vox, "gemini", tmp_path / "g.wav")
    assert out.read_bytes() == b"RIFFGEM" and r.path.endswith("gemini-3.8-flash-tts/sample-audio.wav")
    r, out = player.extract(mm_vox, "1.7b", tmp_path)
    assert out == tmp_path / "sample-audio.wav"


# --- CLI -------------------------------------------------------------------------

def test_cli_play_extract_and_models(mm_path, tmp_path, capsys, monkeypatch):
    assert main(["play", str(mm_path), "qwen3-tts-1.7b", "--extract", str(tmp_path / "a.wav")]) == 0
    assert (tmp_path / "a.wav").read_bytes() == b"RIFF17"
    assert main(["play", str(mm_path), "gemini-3.8-flash-tts", "--extract", str(tmp_path / "b.wav")]) == 0
    assert (tmp_path / "b.wav").read_bytes() == b"RIFFGEM"
    assert main(["play", str(mm_path), "nope", "--extract", str(tmp_path / "c.wav")]) == 1
    assert main(["models", str(mm_path)]) == 0
    out = capsys.readouterr().out
    assert "google-gemini-3.8-flash-tts-sample-audio" in out and "language=default" in out
    assert main(["models", str(mm_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "gemini-3.8-flash-tts" in data["supported_models"]
    assert main(["play", str(mm_path), "--list"]) == 0

    calls = []
    monkeypatch.setattr("voxformat.player.shutil.which", lambda n: f"/usr/bin/{n}")
    monkeypatch.setattr("voxformat.player.subprocess.run", lambda argv, check: calls.append(argv))
    assert main(["play", str(mm_path), "0.6b"]) == 0
    assert calls and calls[0][0] == "/usr/bin/afplay"


def test_cli_extract_variants(mm_path, tmp_path):
    assert main(["extract", str(mm_path), "0.6b", "-o", str(tmp_path / "s.wav")]) == 0
    assert (tmp_path / "s.wav").read_bytes() == b"RIFF06"
    assert main(["extract", str(mm_path), "1.7b", "--clone-prompt", "-o", str(tmp_path / "cp.bin")]) == 0
    assert (tmp_path / "cp.bin").read_bytes() == b"\xc1\x71"
    assert main(["extract", str(mm_path), "--entry", "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
                 "-o", str(tmp_path / "e.bin")]) == 0
    assert (tmp_path / "e.bin").read_bytes() == b"\xc0\x01"


def test_cli_validate(capsys, tmp_path):
    assert main(["validate", str(EXAMPLES / "minimal/narrator.vox"), str(EXAMPLES / "multi-model/manifest.json")]) == 0
    assert main(["validate", str(EXAMPLES.parent / "schemas/test/invalid-gender.json")]) == 1
    capsys.readouterr()
    assert main(["validate", "--json", str(EXAMPLES / "minimal/narrator.vox")]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    # strict mode fails on warnings
    p = tmp_path / "w.vox"
    vox = VoxFile("W", "A voice with an unknown voice field for strict mode.")
    vox.manifest["voice"]["accent"] = "rp"
    vox.write(p)
    assert main(["validate", str(p)]) == 0
    assert main(["validate", "--strict", str(p)]) == 1


def test_cli_create_inspect_and_mutations(tmp_path, capsys):
    p = tmp_path / "n.vox"
    assert main(["create", str(p), "--name", "NARRATOR", "--description", "A warm narrator for the CLI test.",
                 "--language", "en-GB", "--gender", "male", "--age-range", "40", "50", "--tag", "narrator",
                 "--method", "designed", "--license", "CC0-1.0"]) == 0
    assert main(["create", str(p), "--name", "X", "--description", "Duplicate should be refused."]) == 1
    src = tmp_path / "cp.bin"
    src.write_bytes(b"\x01\x02")
    wav = tmp_path / "s.wav"
    wav.write_bytes(b"RIFFsample")
    assert main(["add-embedding", str(p), str(src), "--engine", "qwen3-tts", "--model",
                 "Qwen/Qwen3-TTS-12Hz-0.6B", "--slug", "0.6b", "--name", "clone-prompt.bin"]) == 0
    assert main(["add-sample", str(p), str(wav), "--engine", "qwen3-tts", "--model", "Qwen/Qwen3-TTS-12Hz-0.6B",
                 "--slug", "0.6b", "--language", "es"]) == 0
    assert main(["set-extension", str(p), "elevenlabs", "--set", "voice_id=vid-example-abc123",
                 "--set", "model_id=eleven_multilingual_v2"]) == 0
    assert main(["set-extension", str(p), "elevenlabs", "--merge", "--json", '{"stability": 0.5}']) == 0
    assert main(["set-extension", str(p), "acme", "--set", "x=1"]) == 0
    err = capsys.readouterr().err
    assert "not a reserved VOX provider id" in err
    vox = VoxFile.open(p)
    assert vox.extensions["elevenlabs"] == {"voice_id": "vid-example-abc123", "model_id": "eleven_multilingual_v2",
                                            "stability": 0.5}
    assert vox.extensions["acme"] == {"x": 1}
    assert vox.clone_prompt_data("0.6b") == b"\x01\x02"
    assert vox.sample_audio_data("0.6b", "es") == b"RIFFsample"
    assert vox.sample_audio_data("0.6b") is None  # Spanish-only sample is not the default
    assert vox.validate() == []
    out = tmp_path / "copy.vox"
    assert main(["set-extension", str(p), "acme", "--remove", "-o", str(out)]) == 0
    assert "acme" in VoxFile.open(p).extensions and "acme" not in VoxFile.open(out).extensions
    assert main(["remove", str(p), "embeddings/qwen3-tts/0.6b/es/sample-audio.wav"]) == 0
    assert main(["remove", str(p), "does/not/exist"]) == 1
    vox = VoxFile.open(p)
    assert list(vox.manifest["embeddings"]) == ["qwen3-tts-0.6b-clone-prompt"]
    capsys.readouterr()
    assert main(["inspect", str(p)]) == 0
    text = capsys.readouterr().out
    assert "NARRATOR" in text and "qwen3-tts-0.6b-clone-prompt" in text
    assert main(["inspect", "--json", str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["manifest"]["voice"]["gender"] == "male"


def test_cli_preserves_untouched_entries(tmp_path):
    src = EXAMPLES / "multi-engine/cross-platform.vox"
    p = tmp_path / "x.vox"
    p.write_bytes(src.read_bytes())
    assert main(["set-extension", str(p), "openai", "--set", "voice_id=alloy"]) == 0
    m = json.loads(zipfile.ZipFile(p).read("manifest.json"))
    orig = json.loads(zipfile.ZipFile(src).read("manifest.json"))
    assert m["extensions"]["elevenlabs"] == orig["extensions"]["elevenlabs"]
    assert m["extensions"]["openai"] == {"voice_id": "alloy"}
    assert m["provenance"] == orig["provenance"]


def test_cli_errors_are_reported_not_raised(tmp_path, capsys):
    bad = tmp_path / "bad.vox"
    bad.write_text("x")
    assert main(["inspect", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err
