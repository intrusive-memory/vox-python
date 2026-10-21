"""Validation: Swift validate() rules, versioned schemas, examples, negative fixtures."""

from __future__ import annotations

import copy
import json
import zipfile

import pytest

from voxformat import Severity, VoxFile, validate_file, validate_manifest
from voxformat.schema import available_versions, jsonschema_available, load_schema, select_schema_version
from voxformat.validation import validate_vox

from conftest import EXAMPLES, SCHEMAS

ENGINES = ["builtin", pytest.param("jsonschema", marks=pytest.mark.skipif(
    not jsonschema_available(), reason="optional jsonschema extra not installed"))]

BASE = {
    "vox_version": "0.4.0",
    "id": "ad7aa7d7-570d-4f9e-99da-1bd14b99cc78",
    "created": "2026-02-13T12:00:00Z",
    "voice": {"name": "Test", "description": "A valid description with enough characters."},
}


def m(**over):
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        if k.startswith("voice_"):
            d["voice"][k[6:]] = v
        else:
            d[k] = v
    return d


def errs(manifest, check_bundle=False):
    return [i for i in VoxFile.from_manifest(manifest).validate(check_bundle=check_bundle)
            if i.severity is Severity.ERROR]


def fields(issues):
    return {i.field for i in issues}


# --- bundled schemas mirror the repo ----------------------------------------------

def test_bundled_schemas_identical_to_repo():
    for v in available_versions():
        bundled = load_schema(v)
        repo = json.loads((SCHEMAS / f"manifest-v{v}.json").read_text())
        assert bundled == repo, f"src/voxformat/schemas/manifest-v{v}.json drifted from schemas/"
    repo_versions = sorted(p.name for p in SCHEMAS.glob("manifest-v*.json"))
    assert [f"manifest-v{v}.json" for v in available_versions()] == repo_versions


def test_schema_version_selection():
    assert select_schema_version("0.1.0") == ("0.1.0", None)
    assert select_schema_version("0.3.0")[0] == "0.3.0"
    v, note = select_schema_version("9.9.9")
    assert v == "0.4.0" and "9.9.9" in note
    assert select_schema_version(None)[1]


# --- Swift ValidatorTests ports ------------------------------------------------------

def test_minimal_and_full_manifest_pass():
    assert errs(BASE) == []
    full = m(voice_language="en-GB", voice_gender="male", voice_age_range=[45, 55], voice_tags=["a"],
             prosody={"pitch_base": "low"}, character={"role": "Narrator"},
             provenance={"method": "designed", "consent": None, "license": "CC0-1.0"})
    assert errs(full) == []


@pytest.mark.parametrize("over,field", [
    ({"vox_version": ""}, "vox_version"),
    ({"vox_version": "   "}, "vox_version"),
    ({"id": "not-a-valid-uuid"}, "id"),
    ({"id": "ad7aa7d7-570d-1f9e-99da-1bd14b99cc78"}, "id"),
    ({"created": "February 13, 2026"}, "created"),
    ({"created": "2026-02-13T12:00:00.123Z"}, "created"),
    ({"voice_name": ""}, "voice.name"),
    ({"voice_description": ""}, "voice.description"),
    ({"voice_description": "Short"}, "voice.description"),
    ({"voice_age_range": [50, 30]}, "voice.age_range"),
    ({"voice_age_range": [30, 30]}, "voice.age_range"),
    ({"voice_gender": "robot"}, "voice.gender"),
    ({"reference_audio": [{"file": "", "transcript": "x"}]}, "reference_audio[0].file"),
])
def test_rejections(over, field):
    assert field in fields(errs(m(**over)))


def test_uppercase_uuid_accepted_like_swift():
    vox_errs = errs(m(id="AD7AA7D7-570D-4F9E-99DA-1BD14B99CC78"))
    assert "id" not in fields(vox_errs)
    # ...but the schema (lowercase-only pattern) still flags it as a warning.
    r = validate_manifest(m(id="AD7AA7D7-570D-4F9E-99DA-1BD14B99CC78"), engine="builtin")
    assert r.is_valid and any(i.field == "id" and i.severity is Severity.WARNING for i in r.issues)


@pytest.mark.parametrize("g", ["male", "female", "nonbinary", "neutral"])
def test_valid_genders(g):
    assert errs(m(voice_gender=g)) == []


def test_collects_multiple_errors():
    bad = m(vox_version="", id="not-a-uuid", voice_name="", voice_description="Short",
            voice_gender="invalid", voice_age_range=[50, 30])
    assert len(errs(bad)) >= 4


def test_embedding_entry_rules():
    e = {"model": "", "file": "clone.bin"}
    found = errs(m(embeddings={"k": e}))
    assert [i.message for i in found if i.field == "embeddings.k"] == [
        "model must not be empty", "file must start with 'embeddings/' (got 'clone.bin')"]
    assert "embeddings.k" in fields(errs(m(embeddings={"k": {"model": "M", "file": ""}})))
    assert errs(m(embeddings={"k": {"model": "M", "file": "embeddings/x.bin"}})) == []


# --- provenance ethics -------------------------------------------------------------

@pytest.mark.parametrize("prov,field", [
    ({"method": "cloned", "consent": "self"}, "provenance.source"),
    ({"method": "cloned", "consent": "self", "source": []}, "provenance.source"),
    ({"method": "cloned", "consent": "unknown", "source": ["reference/a.wav"]}, "provenance.consent"),
    ({"method": "cloned", "consent": None, "source": ["reference/a.wav"]}, "provenance.consent"),
    ({"method": "cloned", "source": ["reference/a.wav"]}, "provenance.consent"),
])
def test_cloned_voice_rules(prov, field):
    assert field in fields(errs(m(provenance=prov)))


@pytest.mark.parametrize("consent", ["self", "granted"])
def test_cloned_with_consent_and_source_passes(consent):
    assert errs(m(provenance={"method": "cloned", "consent": consent, "source": ["reference/a.wav"]})) == []


def test_synthesized_passes_and_unknown_method_warns():
    assert errs(m(provenance={"method": "synthesized"})) == []
    issues = VoxFile.from_manifest(m(provenance={"method": "magic"})).validate()
    assert [i.severity for i in issues if i.field == "provenance.method"] == [Severity.WARNING]


def test_model_tagged_reference_audio_warnings():
    clip = {"file": "reference/q.wav", "transcript": "Hi.", "model": "Qwen/Qwen3-TTS-12Hz-0.6B"}
    issues = VoxFile.from_manifest(m(reference_audio=[clip])).validate(check_bundle=False)
    assert [(i.severity, i.field) for i in issues] == [(Severity.WARNING, "reference_audio[0].model")]
    ok = m(reference_audio=[clip], embeddings={"k": {"model": "Qwen/Qwen3-TTS-12Hz-0.6B",
                                                     "file": "embeddings/qwen3-tts/0.6b/clone-prompt.bin"}})
    assert VoxFile.from_manifest(ok).validate(check_bundle=False) == []


def test_bundle_completeness_warnings():
    vox = VoxFile.from_manifest(m(embeddings={"k": {"model": "M", "file": "embeddings/a.bin"}},
                                  reference_audio=[{"file": "reference/r.wav", "transcript": "x"}]))
    issues = vox.validate()
    assert {(i.severity, i.field) for i in issues} == {(Severity.WARNING, "embeddings.k"),
                                                       (Severity.WARNING, "reference_audio")}
    assert vox.is_valid


def test_issue_str_is_descriptive():
    i = errs(m(voice_gender="robot"))[0]
    assert str(i).startswith("[ERROR] voice.gender: Invalid gender value 'robot'")


# --- examples & negative fixtures (SchemaExampleValidationTests parity) ---------------

EXAMPLE_MANIFESTS = sorted(str(p.relative_to(EXAMPLES)) for p in EXAMPLES.glob("*/manifest.json")) + [
    "character/narrator-with-context-manifest.json"]
EXAMPLE_ARCHIVES = sorted(str(p.relative_to(EXAMPLES)) for p in EXAMPLES.rglob("*.vox"))
NEGATIVE = sorted(p.name for p in (SCHEMAS / "test").glob("*.json"))


def test_example_inventory_is_complete():
    # Every Swift-listed example is covered here.
    for rel in ["minimal/manifest.json", "character/narrator-with-context-manifest.json",
                "multi-engine/manifest.json", "multi-model/manifest.json", "multi-language/manifest.json"]:
        assert rel in EXAMPLE_MANIFESTS
    for rel in ["minimal/narrator.vox", "character/narrator-with-context.vox", "multi-engine/cross-platform.vox"]:
        assert rel in EXAMPLE_ARCHIVES
    assert len(NEGATIVE) == 5


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("rel", EXAMPLE_MANIFESTS)
def test_example_manifests_validate_cleanly(rel, engine):
    r = validate_file(EXAMPLES / rel, engine=engine)
    assert r.is_valid, [str(i) for i in r.errors]
    assert r.warnings == [], [str(i) for i in r.warnings]  # fully schema-conformant too
    assert r.schema_version == r.declared_version


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("rel", EXAMPLE_ARCHIVES)
def test_example_archives_validate_cleanly(rel, engine):
    r = validate_file(EXAMPLES / rel, engine=engine)
    assert r.is_valid, [str(i) for i in r.errors]
    assert r.warnings == [], [str(i) for i in r.warnings]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NEGATIVE)
def test_negative_fixtures_rejected(name, engine):
    r = validate_file(SCHEMAS / "test" / name, engine=engine)
    assert not r.is_valid


def test_negative_fixture_reasons():
    expect = {
        "invalid-age-range.json": "voice.age_range[1]",
        "invalid-gender.json": "voice.gender",
        "invalid-missing-vox-version.json": "vox_version",
        "invalid-timestamp.json": "created",
        "invalid-uuid.json": "id",
    }
    for name, fld in expect.items():
        r = validate_file(SCHEMAS / "test" / name, engine="builtin")
        assert fld in {i.field for i in r.errors}, (name, [str(i) for i in r.issues])


def test_cloned_example_needs_its_source_files_in_an_archive(tmp_path):
    manifest = json.loads((EXAMPLES / "cloned-voice/manifest.json").read_text())
    p = tmp_path / "c.vox"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    r = validate_file(p)
    assert r.is_valid  # missing files are warnings, as in Swift
    assert {i.field for i in r.warnings} >= {"reference_audio", "provenance.source[0]", "provenance.source[1]"}


# --- schema-only strictness is a warning, Swift decode failures are errors -----------

@pytest.mark.parametrize("engine", ENGINES)
def test_schema_only_violations_are_warnings(engine):
    r = validate_manifest(m(voice_accent="rp", provenance={"method": "designed", "consent": "maybe"}), engine=engine)
    assert r.is_valid
    assert {"voice.accent", "provenance.consent"} <= {i.field for i in r.warnings}


@pytest.mark.parametrize("engine", ENGINES)
def test_type_errors_are_errors(engine):
    r = validate_manifest(m(voice_tags="not-a-list"), engine=engine)
    assert "voice.tags" in {i.field for i in r.errors}
    r = validate_manifest(m(embeddings={"k": {"file": "embeddings/a.bin"}}), engine=engine)
    assert not r.is_valid


def test_version_specific_schema_is_used():
    # `language` on an embedding is new in v0.4.0; the v0.3.0 schema forbids it (warning).
    emb = {"k": {"model": "M", "file": "embeddings/a.bin", "language": "es"}}
    r3 = validate_manifest(m(vox_version="0.3.0", embeddings=emb), engine="builtin")
    r4 = validate_manifest(m(vox_version="0.4.0", embeddings=emb), engine="builtin")
    assert any(i.field == "embeddings.k.language" for i in r3.warnings)
    assert not any(i.field == "embeddings.k.language" for i in r4.issues)


def test_null_in_extensions_warns_swift_incompat():
    r = validate_manifest(m(extensions={"google": {"voice_id": "voice_example_abc123", "gender": None}}))
    assert r.is_valid
    assert any(i.field == "extensions.google.gender" and i.source == "compat" for i in r.warnings)


def test_archive_extras(tmp_path):
    manifest = m(extensions={"qwen3-tts": {"clone_prompt": "embeddings/qwen3-tts/clone-prompt.bin"}})
    p = tmp_path / "a.vox"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", json.dumps(dict(manifest, embeddings={
            "k": {"model": "M", "file": "embeddings/x/declared.bin"}})))
        zf.writestr("embeddings/x/declared.bin", b"1")
        zf.writestr("embeddings/x/undeclared.bin", b"2")
    r = validate_file(p)
    assert r.is_valid
    assert any(i.field == "extensions.qwen3-tts.clone_prompt" for i in r.warnings)
    assert any("undeclared.bin" in i.message for i in r.infos)


def test_unreadable_inputs_report_instead_of_raising(tmp_path):
    bad = tmp_path / "bad.vox"
    bad.write_text("nope")
    r = validate_file(bad)
    assert not r.is_valid and r.errors[0].source == "decode"
    j = tmp_path / "bad.json"
    j.write_text("{")
    assert not validate_file(j).is_valid


def test_validate_vox_uses_raw_manifest_for_schema(tmp_path):
    p = tmp_path / "v.vox"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", json.dumps(m(vox_version="0.1.0")))
    vox = VoxFile.open(p)
    r = validate_vox(vox)
    assert r.declared_version == "0.1.0" and r.schema_version == "0.1.0"
    d = r.to_dict()
    assert d["valid"] is True and d["errors"] == 0
