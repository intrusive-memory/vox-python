"""VoxFile CRUD, round-trip, auto-manifest and migration (ports Roundtrip/Writer/Reader/Migrator tests)."""

from __future__ import annotations

import json
import os
import stat
import zipfile

import pytest

from voxformat import (
    CURRENT_VERSION,
    InvalidJSONError,
    InvalidPathError,
    InvalidZipError,
    ManifestNotFoundError,
    MissingRequiredMetadataError,
    VoxFile,
    derive_embedding_key,
    migrate_manifest,
)

from conftest import EXAMPLES


def _zip(path, files: dict[str, bytes], *, compress=zipfile.ZIP_DEFLATED, dirs=()):
    with zipfile.ZipFile(path, "w", compress) as zf:
        for d in dirs:
            zf.writestr(zipfile.ZipInfo(d), b"")
        for name, data in files.items():
            zf.writestr(name, data)


V1 = {
    "vox_version": "0.1.0",
    "id": "12345678-1234-4234-8234-123456789abc",
    "created": "2024-02-13T12:00:00Z",
    "voice": {"name": "TestVoice", "description": "A test voice with a sufficiently long description for validation."},
}


# --- key derivation parity -----------------------------------------------------

@pytest.mark.parametrize("path,key", [
    ("embeddings/qwen3-tts/0.6b/clone-prompt.bin", "qwen3-tts-0.6b-clone-prompt"),
    ("embeddings/qwen3-tts/0.6b/sample-audio.wav", "qwen3-tts-0.6b-sample-audio"),
    ("embeddings/qwen3-tts/0.6b/es/sample-audio.wav", "qwen3-tts-0.6b-es-sample-audio"),
    ("embeddings/qwen3-tts/0.6b/fr-FR/clone-prompt.bin", "qwen3-tts-0.6b-fr-FR-clone-prompt"),
    ("embeddings/google/gemini-3.8-flash-tts/sample-audio.wav", "google-gemini-3.8-flash-tts-sample-audio"),
    ("embeddings/qwen3-tts/clone-prompt.bin", "qwen3-tts-clone-prompt"),
    ("embeddings/voice.bin", "voice"),
    ("embeddings/a//b/model.tar.gz", "a-b-model.tar"),
    ("embeddings/x/.hidden", "x-.hidden"),
    ("embeddings/x/name.", "x-name."),
])
def test_derive_embedding_key_matches_swift(path, key):
    assert derive_embedding_key(path) == key


# --- create / write / read -----------------------------------------------------

def test_new_voxfile_has_required_fields():
    vox = VoxFile("Narrator", "A warm, clear narrator voice.", language="en-US")
    m = vox.manifest
    assert m["vox_version"] == CURRENT_VERSION
    assert m["voice"] == {"name": "Narrator", "description": "A warm, clear narrator voice.", "language": "en-US"}
    assert vox.is_valid
    assert vox.readiness.state == "ready"


def test_roundtrip_minimal(tmp_path):
    vox = VoxFile("Narrator", "A warm, clear narrator voice for audiobooks.")
    out = vox.write(tmp_path / "n.vox")
    assert out.read_bytes()[:4] == b"PK\x03\x04"
    back = VoxFile.open(out)
    assert back.manifest["id"] == vox.manifest["id"]
    assert back.manifest["voice"]["name"] == "Narrator"
    assert "modified" in back.manifest  # new file is dirty -> modified stamped
    assert back.entry_count == 0
    with zipfile.ZipFile(out) as zf:  # system zip readable
        assert zf.testzip() is None


def test_roundtrip_with_reference_audio_and_embeddings(tmp_path):
    vox = VoxFile("Narrator", "A warm, clear narrator voice for audiobooks.")
    vox.add(b"RIFFaudio", "reference/sample.wav",
            metadata={"transcript": "Hello, welcome.", "language": "en-US", "duration_seconds": 3.5})
    vox.add(b"\x01\x02", "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
            metadata={"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts"})
    vox.add(b"PNG", "assets/headshot.png")
    p = vox.write(tmp_path / "v.vox")
    back = VoxFile.open(p)
    assert back.manifest["reference_audio"] == [
        {"file": "reference/sample.wav", "transcript": "Hello, welcome.", "language": "en-US", "duration_seconds": 3.5}
    ]
    assert back.manifest["embeddings"] == {"qwen3-tts-0.6b-clone-prompt": {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts",
        "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin"}}
    assert back["reference/sample.wav"].data == b"RIFFaudio"
    assert back["reference/sample.wav"].mime_type == "audio/wav"
    assert back["assets/headshot.png"].data == b"PNG"
    assert "assets/headshot.png" in back
    assert {e.path for e in back.entries_under("embeddings/")} == {"embeddings/qwen3-tts/0.6b/clone-prompt.bin"}
    assert back.validate() == []


def test_add_replace_keeps_single_manifest_entry():
    vox = VoxFile("T", "A test voice with a long enough description.")
    vox.add(b"a", "reference/s.wav", metadata={"transcript": "one"})
    vox.add(b"b", "reference/s.wav", metadata={"transcript": "two"})
    assert vox.manifest["reference_audio"] == [{"file": "reference/s.wav", "transcript": "two"}]
    vox.add(b"x", "embeddings/e/m/clone-prompt.bin", metadata={"model": "M1", "key": "custom"})
    vox.add(b"y", "embeddings/e/m/clone-prompt.bin", metadata={"model": "M2"})
    assert list(vox.manifest["embeddings"]) == ["e-m-clone-prompt"]
    assert vox["embeddings/e/m/clone-prompt.bin"].data == b"y"


def test_remove_keeps_manifest_in_sync():
    vox = VoxFile("T", "A test voice with a long enough description.")
    vox.add(b"a", "reference/s.wav", metadata={"transcript": "one"})
    vox.add(b"x", "embeddings/e/m/clone-prompt.bin", metadata={"model": "M1"})
    assert vox.remove("reference/s.wav").data == b"a"
    assert "reference_audio" not in vox.manifest
    vox.remove("embeddings/e/m/clone-prompt.bin")
    assert "embeddings" not in vox.manifest
    assert vox.remove("nope") is None


@pytest.mark.parametrize("bad", ["", "   ", "manifest.json", "/abs/x", "../x", "a/../../x", "dir/", "a\\b"])
def test_invalid_paths(bad):
    vox = VoxFile("T", "A test voice with a long enough description.")
    with pytest.raises(InvalidPathError):
        vox.add(b"x", bad)


def test_embedding_requires_model():
    vox = VoxFile("T", "A test voice with a long enough description.")
    with pytest.raises(MissingRequiredMetadataError):
        vox.add(b"x", "embeddings/qwen3-tts/0.6b/clone-prompt.bin")
    with pytest.raises(MissingRequiredMetadataError):
        vox.add(b"x", "embeddings/qwen3-tts/0.6b/clone-prompt.bin", metadata={"model": "  "})


def test_path_is_trimmed():
    vox = VoxFile("T", "A test voice with a long enough description.")
    e = vox.add(b"x", "  assets/a.png\t")
    assert e.path == "assets/a.png" and "assets/a.png" in vox


# --- preservation ------------------------------------------------------------

def test_unknown_fields_extensions_and_order_preserved(tmp_path):
    manifest = {
        "vox_version": "0.3.0",
        "x_top": {"nested": [1, 2, {"k": "v"}]},
        "id": "e1a2b3c4-d5e6-4f7a-8b9c-0d1e2f3a4b5c",
        "created": "2026-02-23T00:00:00+05:30",
        "voice": {"name": "N", "description": "A narrator voice for preservation tests.", "x_accent": "rp"},
        "extensions": {"some-future-engine": {"handle": "abc", "nested": {"a": [True, 1.5]}},
                       "elevenlabs": {"voice_id": "vid-example-abc123"}},
        "embeddings": {"k": {"model": "M", "file": "embeddings/e/m/x.bin", "x_extra": 7}},
    }
    src = tmp_path / "src.vox"
    _zip(src, {"assets/z.txt": b"z", "embeddings/e/m/x.bin": b"bin", "manifest.json": json.dumps(manifest).encode(),
               "reference/r.wav": b"RIFF"}, dirs=["assets/"])
    vox = VoxFile.open(src)
    out = vox.write(tmp_path / "out.vox")
    back = json.loads(zipfile.ZipFile(out).read("manifest.json"))
    expected = dict(manifest, vox_version=CURRENT_VERSION)
    assert back == expected
    assert list(back) == list(expected)  # key order kept
    assert back["created"] == "2026-02-23T00:00:00+05:30"  # not normalized (Swift would re-encode)
    names = [i.filename for i in zipfile.ZipFile(out).infolist()]
    assert names == ["assets/", "assets/z.txt", "embeddings/e/m/x.bin", "manifest.json", "reference/r.wav"]
    assert "modified" not in back  # nothing changed through the API


def test_untouched_entries_keep_compression_and_timestamps(tmp_path):
    src = tmp_path / "src.vox"
    with zipfile.ZipFile(src, "w") as zf:
        info = zipfile.ZipInfo("embeddings/q/0.6b/sample-audio.wav", date_time=(2020, 1, 2, 3, 4, 6))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, b"RIFF" * 100)
        zf.writestr("manifest.json", json.dumps(V1))
    vox = VoxFile.open(src)
    vox.add(b"new", "assets/new.bin")
    out = vox.write(tmp_path / "o.vox")
    infos = {i.filename: i for i in zipfile.ZipFile(out).infolist()}
    kept = infos["embeddings/q/0.6b/sample-audio.wav"]
    assert kept.compress_type == zipfile.ZIP_STORED and kept.date_time == (2020, 1, 2, 3, 4, 6)
    assert infos["assets/new.bin"].compress_type == zipfile.ZIP_DEFLATED
    assert list(infos)[-1] == "assets/new.bin"


def test_write_stamps_version_and_modified_only_when_dirty(tmp_path):
    src = tmp_path / "v1.vox"
    _zip(src, {"manifest.json": json.dumps(V1).encode()})
    vox = VoxFile.open(src)
    assert vox.source_version == "0.1.0" and vox.manifest["vox_version"] == CURRENT_VERSION
    vox.write()
    m = json.loads(zipfile.ZipFile(src).read("manifest.json"))
    assert m["vox_version"] == CURRENT_VERSION and "modified" not in m
    vox.update_voice(gender="male")
    vox.write()
    m = json.loads(zipfile.ZipFile(src).read("manifest.json"))
    assert m["voice"]["gender"] == "male" and "modified" in m


def test_atomic_write_preserves_mode_and_leaves_no_temp(tmp_path):
    p = tmp_path / "v.vox"
    VoxFile("T", "A test voice with a long enough description.").write(p)
    os.chmod(p, 0o640)
    vox = VoxFile.open(p)
    vox.touch()
    vox.write()
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o640
    assert [x.name for x in tmp_path.iterdir()] == ["v.vox"]


def test_failed_write_keeps_original(tmp_path, monkeypatch):
    p = tmp_path / "v.vox"
    VoxFile("T", "A test voice with a long enough description.").write(p)
    before = p.read_bytes()
    vox = VoxFile.open(p)
    vox.add(b"x", "assets/x.bin")

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(vox, "_write_entries", boom)
    from voxformat import WriteFailedError
    with pytest.raises(WriteFailedError):
        vox.write()
    assert p.read_bytes() == before
    assert [x.name for x in tmp_path.iterdir()] == ["v.vox"]


def test_set_and_remove_extension():
    vox = VoxFile("T", "A test voice with a long enough description.")
    vox.set_extension("elevenlabs", {"voice_id": "vid-example-abc123"})
    vox.set_extension("elevenlabs", {"model_id": "eleven_multilingual_v2"}, merge=True)
    assert vox.extensions["elevenlabs"] == {"voice_id": "vid-example-abc123", "model_id": "eleven_multilingual_v2"}
    vox.set_extension("elevenlabs", {"voice_id": "vid-2"})
    assert vox.extensions["elevenlabs"] == {"voice_id": "vid-2"}
    assert vox.remove_extension("elevenlabs") == {"voice_id": "vid-2"}
    assert "extensions" not in vox.manifest


def test_set_field_rejects_managed_sections():
    vox = VoxFile("T", "A test voice with a long enough description.")
    with pytest.raises(InvalidPathError):
        vox.set_field("embeddings", {})
    vox.set_field("provenance", {"method": "designed"})
    assert vox.manifest["provenance"] == {"method": "designed"}


# --- error cases -------------------------------------------------------------

def test_invalid_zip(tmp_path):
    p = tmp_path / "x.vox"
    p.write_text("This is not a ZIP file")
    with pytest.raises(InvalidZipError):
        VoxFile.open(p)


def test_missing_manifest(tmp_path):
    p = tmp_path / "x.vox"
    _zip(p, {"dummy.txt": b"Some content"})
    with pytest.raises(ManifestNotFoundError):
        VoxFile.open(p)


@pytest.mark.parametrize("payload", [b"{not json", b"[1,2]", b"\xff\xfe"])
def test_invalid_manifest_json(tmp_path, payload):
    p = tmp_path / "x.vox"
    _zip(p, {"manifest.json": payload})
    with pytest.raises(InvalidJSONError):
        VoxFile.open(p)


# --- examples ----------------------------------------------------------------

@pytest.mark.parametrize("rel", ["minimal/narrator.vox", "character/narrator-with-context.vox",
                                 "multi-engine/cross-platform.vox"])
def test_read_existing_example_and_rewrite(tmp_path, rel):
    vox = VoxFile.open(EXAMPLES / rel)
    raw = vox.raw_manifest
    out = vox.write(tmp_path / "copy.vox")
    back = VoxFile.open(out)
    assert back.manifest == dict(raw, vox_version=CURRENT_VERSION)
    assert back.validate() == []


# --- readiness -----------------------------------------------------------------

def test_readiness_states():
    vox = VoxFile.from_manifest(dict(V1, embeddings={"qwen3-tts-0.6b": {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B", "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin"}}))
    assert vox.readiness.state == "needs_regeneration"
    assert vox.readiness.missing == ("qwen3-tts-0.6b",)
    vox.add(b"x", "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
            metadata={"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "key": "qwen3-tts-0.6b"})
    assert vox.is_ready
    short = VoxFile.from_manifest(dict(V1, voice={"name": "X", "description": "Short"}))
    assert short.readiness.state == "invalid"


# --- migrator ------------------------------------------------------------------

def test_migrate_no_embeddings_bumps_version():
    m = migrate_manifest(V1, [])
    assert m["vox_version"] == CURRENT_VERSION and "embeddings" not in m
    assert V1["vox_version"] == "0.1.0"  # input untouched


def test_migrate_infers_entries_from_binaries():
    m = migrate_manifest(V1, ["qwen3-tts/0.6b/clone-prompt.bin"])
    assert m["embeddings"] == {"qwen3-tts-0.6b": {
        "model": "qwen3-tts-0.6b", "engine": "qwen3-tts", "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
        "format": "bin", "description": "Inferred from archive binary (v0.1.0 migration)"}}


def test_migrate_uses_extension_hints():
    src = dict(V1, extensions={"qwen3-tts": {"clone_prompt": "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
                                             "model": "Qwen/Qwen3-TTS-12Hz-0.6B"}})
    m = migrate_manifest(src, ["qwen3-tts/0.6b/clone-prompt.bin"])
    assert m["embeddings"] == {"qwen3-tts-clone_prompt": {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts",
        "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin", "format": "bin",
        "description": "Migrated from v0.1.0 extensions"}}


def test_migrate_preserves_existing_entries_and_multiple_binaries():
    existing = {"qwen3-tts-0.6b": {"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts",
                                   "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin"}}
    m = migrate_manifest(dict(V1, embeddings=existing), ["qwen3-tts/0.6b/clone-prompt.bin", "x/y.bin"])
    assert m["embeddings"] == existing
    m2 = migrate_manifest(V1, ["qwen3-tts/0.6b/clone-prompt.bin", "qwen3-tts/1.7b/clone-prompt.bin"])
    assert len(m2["embeddings"]) == 2


def test_reader_auto_migrates(tmp_path):
    p = tmp_path / "old.vox"
    _zip(p, {"manifest.json": json.dumps(V1).encode(), "embeddings/qwen3-tts/0.6b/clone-prompt.bin": b"\x01"})
    vox = VoxFile.open(p)
    assert vox.manifest["vox_version"] == CURRENT_VERSION
    assert vox.supports_model("0.6b")
    assert vox.raw_manifest == V1
    assert VoxFile.open(p, migrate=False).manifest == V1
