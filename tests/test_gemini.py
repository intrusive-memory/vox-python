"""Gemini adapter + Voices API client, with the HTTP layer mocked (no network)."""

from __future__ import annotations

import base64
import json
import zipfile

import pytest

from voxformat import VoxFile, validate_file
from voxformat.cli import main
from voxformat.providers import gemini as g

KEY = "test-key-NOT-REAL-0123456789"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


class FakeAPI:
    """In-memory stand-in for the Gemini Voices API, used as the client transport."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict, dict | None]] = []
        self.voices: dict[str, dict] = {}
        self.n = 0
        self.fail: dict[str, tuple[int, dict]] = {}

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, dict(headers), payload))
        if method in self.fail:
            status, err = self.fail.pop(method)
            return status, json.dumps(err).encode()
        path = url.split("/v1beta", 1)[1]
        if method == "POST" and path == "/voices":
            self.n += 1
            vid = f"voice_example_{self.n:03d}"
            v = {"id": vid, "type": "prompted", "expire_time": f"2027-10-20T00:00:0{self.n}.5Z",
                 "sample_audio": {"data": base64.b64encode(WAV + bytes([self.n])).decode()}}
            self.voices[vid] = v
            return 200, json.dumps(v).encode()
        if method == "GET" and path.startswith("/voices?"):
            return 200, json.dumps({"voices": [{"id": k} for k in self.voices]}).encode()
        if path.startswith("/voices/"):
            vid = path[len("/voices/"):]
            if vid not in self.voices:
                return 404, json.dumps({"error": {"code": 404, "message": f"voice {vid} not found"}}).encode()
            if method == "GET":
                v = dict(self.voices[vid], expire_time="2028-01-01T00:00:00Z",
                         sample_audio={"data": base64.b64encode(WAV + b"refreshed").decode()})
                return 200, json.dumps(v).encode()
            if method == "DELETE":
                del self.voices[vid]
                return 200, b"{}"
        if method == "POST" and path == "/interactions":
            vid = payload["generation_config"]["speech_config"][0]["voice"]
            if vid not in self.voices:
                return 404, json.dumps({"error": {"code": 404, "message": f"voice {vid} not found"}}).encode()
            audio = base64.b64encode(WAV + payload["input"][0]["content"][0]["text"].encode()).decode()
            return 200, json.dumps({"steps": [{"type": "model_output",
                                               "content": [{"type": "audio", "mime_type": "audio/wav", "data": audio}]}]}).encode()
        return 400, b'{"error": {"message": "bad request"}}'


@pytest.fixture
def api():
    return FakeAPI()


@pytest.fixture
def client(api):
    return g.GeminiVoicesClient(KEY, transport=api)


@pytest.fixture
def vox(tmp_path):
    v = VoxFile("CLARK", "Male, 30s. A quiet, low tenor voice used only for tests.", language="en-US", gender="male")
    v.write(tmp_path / "CLARK.vox")
    return VoxFile.open(tmp_path / "CLARK.vox")


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(g.GeminiAuthError):
        g.GeminiVoicesClient()
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    c = g.GeminiVoicesClient(transport=lambda *a: (200, b"{}"))
    assert KEY not in repr(c) and KEY not in str(vars(c).get("base_url"))


def test_client_requests(client, api):
    created = client.create_voice("A warm narrator.", "example-NARRATOR", gender="male", language_code="en-US")
    method, url, headers, body = api.calls[-1]
    assert (method, url) == ("POST", "https://generativelanguage.googleapis.com/v1beta/voices")
    assert headers["x-goog-api-key"] == KEY and KEY not in url
    assert body == {"store": True, "voice": {"model": "gemini-3.8-flash-tts", "type": "prompted",
                                             "display_name": "example-NARRATOR", "language_code": "en-US",
                                             "prompted": {"input": "A warm narrator."}, "gender": "male"}}
    vid = created["id"]
    client.get_voice(vid)
    assert api.calls[-1][:2] == ("GET", f"https://generativelanguage.googleapis.com/v1beta/voices/{vid}")
    client.list_voices()
    assert api.calls[-1][1].endswith("/voices?type=prompted")
    client.delete_voice(f"voices/{vid}")
    assert api.calls[-1][:2] == ("DELETE", f"https://generativelanguage.googleapis.com/v1beta/voices/{vid}")
    with pytest.raises(ValueError):
        client.get_voice("")
    # path injection is escaped (404 from the fake API, never a different endpoint)
    with pytest.raises(g.GeminiAPIError):
        client.get_voice("a/../../models")
    assert api.calls[-1][1].endswith("/voices/a%2F..%2F..%2Fmodels")


def test_api_errors_are_redacted(api):
    api.fail["GET"] = (403, {"error": {"message": f"API key {KEY} is invalid"}})
    client = g.GeminiVoicesClient(KEY, transport=api)
    with pytest.raises(g.GeminiAPIError) as ei:
        client.get_voice("voice_example_001")
    assert ei.value.status == 403 and KEY not in str(ei.value) and "***" in str(ei.value)


def test_create_writes_exact_layout(vox, client, tmp_path):
    voice = g.create_voice_for_vox(vox, client)
    vox.write()
    z = zipfile.ZipFile(vox.path)
    m = json.loads(z.read("manifest.json"))
    ext = m["extensions"]["google"]
    assert list(ext) == ["voice_id", "model_id", "type", "display_name", "gender", "language_code",
                         "prompted_input", "created", "expire_time"]
    assert ext["voice_id"] == "voice_example_001"
    assert ext["model_id"] == "gemini-3.8-flash-tts"
    assert ext["type"] == "prompted"
    assert ext["display_name"] == "CLARK"
    assert ext["gender"] == "male" and ext["language_code"] == "en-US"
    assert ext["prompted_input"] == vox.manifest["voice"]["description"]
    assert ext["expire_time"] == "2027-10-20T00:00:01.5Z"
    assert m["embeddings"] == {"google-gemini-3.8-flash-tts-sample-audio": {
        "model": "gemini-3.8-flash-tts", "engine": "google",
        "file": "embeddings/google/gemini-3.8-flash-tts/sample-audio.wav", "format": "wav",
        "description": "Engine-generated voice sample (gemini-3.8-flash-tts)"}}
    assert z.read("embeddings/google/gemini-3.8-flash-tts/sample-audio.wav") == WAV + b"\x01"
    assert "modified" in m
    assert not any(n.endswith("voice.json") for n in z.namelist())  # handle is NOT a fake embedding
    r = validate_file(vox.path)
    assert r.is_valid and r.warnings == []
    assert VoxFile.open(vox.path).sample_audio_data("gemini") == WAV + b"\x01"
    assert voice.extension["voice_id"] == "voice_example_001"
    # A second create would orphan the server voice -> refused without force.
    with pytest.raises(g.GeminiError):
        g.create_voice_for_vox(vox, client)


def test_refresh_updates_sample_expiry_and_refreshed(vox, client):
    g.create_voice_for_vox(vox, client)
    vox.write()
    g.refresh_vox(vox, client)
    ext = vox.extensions["google"]
    assert ext["expire_time"] == "2028-01-01T00:00:00Z" and "refreshed" in ext
    assert list(ext)[-1] == "refreshed"
    assert vox.sample_audio_data("gemini-3.8-flash-tts") == WAV + b"refreshed"
    assert len(vox.manifest["embeddings"]) == 1


def test_update_recreates_writes_then_deletes_old(vox, client, api, monkeypatch):
    g.create_voice_for_vox(vox, client)
    vox.write()
    order = []
    real_write = vox.write
    monkeypatch.setattr(vox, "write", lambda *a, **k: (order.append(api.calls[-1][0]), real_write(*a, **k))[1])

    res = g.update_vox(vox, client, prompt="A new, brighter prompt for the same character.")
    assert res.old_deleted and res.old_voice_id == "voice_example_001" and res.new_voice_id == "voice_example_002"
    methods = [c[0] for c in api.calls]
    assert methods[-2:] == ["POST", "DELETE"]
    assert order == ["POST"]  # written after the create, before the delete
    on_disk = json.loads(zipfile.ZipFile(vox.path).read("manifest.json"))["extensions"]["google"]
    assert on_disk["voice_id"] == "voice_example_002"
    assert on_disk["prompted_input"] == "A new, brighter prompt for the same character."
    assert "voice_example_001" not in api.voices


def test_update_reports_failed_delete_without_losing_new_voice(vox, client, api):
    g.create_voice_for_vox(vox, client)
    vox.write()
    api.fail["DELETE"] = (500, {"error": {"message": "backend unavailable"}})
    res = g.update_vox(vox, client)
    assert not res.old_deleted and "backend unavailable" in res.delete_error
    assert VoxFile.open(vox.path).extensions["google"]["voice_id"] == res.new_voice_id


def test_update_requires_existing_voice(vox, client):
    with pytest.raises(g.GeminiError):
        g.update_vox(vox, client)


def test_delete_detaches_matching_vox(vox, client):
    g.create_voice_for_vox(vox, client)
    removed = g.delete_voice(client, "voice_example_001", vox)
    assert "extensions.google" in removed
    assert "embeddings/google/gemini-3.8-flash-tts/sample-audio.wav" in removed
    assert vox.entry_count == 0 and "extensions" not in vox.manifest


def test_legacy_voice_json_layout_is_migrated(tmp_path, client):
    manifest = VoxFile("OLD", "A voice written by the pre-fix proof of concept.").manifest
    manifest["embeddings"] = {"google-gemini-3.8-flash-tts-voice-ref": {
        "model": "gemini-3.8-flash-tts", "engine": "google",
        "file": "embeddings/google/gemini-3.8-flash-tts/voice.json"}}
    legacy = {"id": "voice_example_001", "type": "prompted", "display_name": "example-OLD", "gender": "female",
              "language_code": "en-US", "prompt": "Old prompt.", "created": "2026-10-01T00:00:00Z"}
    p = tmp_path / "OLD.vox"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("embeddings/google/gemini-3.8-flash-tts/voice.json", json.dumps(legacy))
    vox = VoxFile.open(p)
    client.transport.voices["voice_example_001"] = {"id": "voice_example_001", "type": "prompted"}
    g.refresh_vox(vox, client)
    assert "embeddings/google/gemini-3.8-flash-tts/voice.json" not in vox
    assert list(vox.manifest["embeddings"]) == ["google-gemini-3.8-flash-tts-sample-audio"]
    assert vox.extensions["google"]["prompted_input"] == "Old prompt."
    assert vox.extensions["google"]["voice_id"] == "voice_example_001"


def test_adapter_omits_nulls_for_swift_compat():
    voice = g.GeminiAdapter().voice_from_response({"id": "voice_example_abc123"}, display_name="X", prompt="p")
    assert None not in voice.extension.values()
    assert "gender" not in voice.extension and "expire_time" not in voice.extension
    assert voice.samples == []


# --- CLI ------------------------------------------------------------------------

def test_cli_gemini_roundtrip(vox, api, monkeypatch, capsys):
    monkeypatch.setattr("voxformat.cli._gemini_client", lambda: g.GeminiVoicesClient(KEY, transport=api))
    p = str(vox.path)
    assert main(["gemini", "create", p, "--prompt", "A calm prompt."]) == 0
    assert main(["gemini", "create", p]) == 1  # refuses to orphan
    assert main(["gemini", "refresh", p]) == 0
    assert main(["gemini", "get", "voice_example_001"]) == 0
    out = capsys.readouterr().out
    assert "b64 chars" in out  # audio redacted in display
    assert main(["gemini", "list"]) == 0
    assert main(["gemini", "update", p]) == 0
    assert main(["gemini", "delete", "voice_example_002", "--vox", p]) == 2  # needs --yes
    assert main(["gemini", "delete", "voice_example_002", "--vox", p, "--yes"]) == 0
    out = capsys.readouterr()
    assert KEY not in out.out and KEY not in out.err
    assert "google" not in VoxFile.open(p).extensions


def test_cli_gemini_without_key(monkeypatch, capsys, vox):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["gemini", "list"]) == 1
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_synthesize_uses_stored_voice_id(api, client, vox):
    g.create_voice_for_vox(vox, client, prompt="A warm narrator.", display_name="example-NARRATOR")
    vid = g.GeminiAdapter().read_extension(vox)["voice_id"]
    audio = client.synthesize(vid, "Hello there.", style="dry")
    assert audio == WAV + b"Hello there."
    body = api.calls[-1][3]
    assert body["generation_config"]["speech_config"] == [{"voice": vid}]
    assert body["input"][0]["content"][0]["annotations"][0]["style"] == "dry"


def test_synthesize_unknown_voice_raises(client):
    with pytest.raises(g.GeminiAPIError) as e:
        client.synthesize("voice_missing", "Hi")
    assert e.value.status == 404


def test_cli_synthesize(api, client, vox, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("voxformat.cli._gemini_client", lambda: client)
    g.create_voice_for_vox(vox, client, prompt="A warm narrator.", display_name="example-NARRATOR")
    vox.write()
    out = tmp_path / "line.wav"
    assert main(["gemini", "synthesize", str(vox.path), "Pick a lane.", "-o", str(out)]) == 0
    assert out.read_bytes() == WAV + b"Pick a lane."
    assert KEY not in capsys.readouterr().out
