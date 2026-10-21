---
type: doc
---

# voxformat (Python)

A Python 3.11+ library and CLI (`voxpy`) for the VOX voice-identity format (`.vox`, spec v0.4.0).

> **This is a reference implementation that tracks the Swift one.**
> The spec and the Swift `VoxFormat` library live in
> [intrusive-memory/vox-format](https://github.com/intrusive-memory/vox-format)
> (`docs/VOX-FORMAT.md`, `implementations/swift/Sources/VoxFormat`); Swift is the behavioural source of truth. When the
> Swift `VoxFile` API or its semantics change, update this package to match. The places
> where Python intentionally differs are listed under
> [Differences from Swift](#differences-from-swift).

The core uses only the standard library (`zipfile`, `json`, `urllib`). Installing the
optional `jsonschema` extra lets the validator use `jsonschema` instead of its built-in
schema engine. Both engines read the same bundled schema files.

## Install

```bash
pip install "git+https://github.com/intrusive-memory/vox-python.git"   # library + voxpy CLI

# or, to develop:
git clone https://github.com/intrusive-memory/vox-python.git && cd vox-python
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'      # or '.' for stdlib-only, '.[jsonschema]' for the extra
.venv/bin/python -m pytest             # run the tests
```

Agents: see [AGENTS.md](AGENTS.md) for how to store samples and voices for each TTS model.

## Library

### Create, read, update, delete

```python
from voxformat import VoxFile

vox = VoxFile("NARRATOR", "A warm, clear narrator voice for audiobooks.", language="en-GB")
vox.add(wav_bytes, "reference/sample.wav", metadata={"transcript": "Hello.", "language": "en-GB"})
vox.add(prompt_bytes, "embeddings/qwen3-tts/0.6b/clone-prompt.bin",
        metadata={"model": "Qwen/Qwen3-TTS-12Hz-0.6B", "engine": "qwen3-tts"})   # model is required
vox.write("narrator.vox")              # atomic: temp file + os.replace; keeps file mode

vox = VoxFile.open("narrator.vox")     # older versions are migrated in memory (Swift VoxMigrator)
vox["reference/sample.wav"].data       # subscript -> VoxEntry | None
vox.remove("reference/sample.wav")     # manifest kept in sync
vox.update_voice(gender="male")        # None deletes a field
vox.set_field("provenance", {"method": "synthesized", "license": "CC0-1.0"})
vox.write()                            # in place; stamps vox_version and (if changed) `modified`
```

Auto-manifest follows Swift `VoxFile+AutoManifest.swift`:

* `add()` under `reference/` appends a `reference_audio` item built from `transcript`,
  `language`, `duration_seconds`, `context`, `model`, `engine`.
* `add()` under `embeddings/` creates an `embeddings` entry from `model` (required),
  `engine`, `format`, `description`, `language`, and an optional `key`. If no key is given,
  it is derived the same way as Swift `deriveEmbeddingKey`:
  `embeddings/qwen3-tts/0.6b/clone-prompt.bin` -> `qwen3-tts-0.6b-clone-prompt`.
* Calling `add()` on an existing path replaces the entry. `remove()` deletes the matching
  manifest item.

The manifest is a plain `dict`. **Unknown fields and unknown `extensions` survive a
round-trip unchanged, in their original key order.** Archive entries you do not touch
keep their order, timestamps and compression (for example, STORED stays STORED).

### Model and language queries

```python
vox.supports_model("0.6b"); vox.supported_models; vox.embedding_data("1.7b")
vox.clone_prompt_data("0.6b", language="fr-FR")    # exact -> base (fr) -> default -> legacy path
vox.sample_audio_data("gemini")                    # default (language-neutral) sample only
vox.sample_audio_languages("0.6b")                 # ["es", "fr-FR"]; language-specific only
vox.resolve_sample_audio("0.6b", "es-MX")          # Resolved(path, data, via="base", entry)
```

A lookup with the default language (`language=None`) **never** returns an entry tagged
with a `language`. If the only sample is Spanish, `sample_audio_data("0.6b")` returns
`None`.

### Attaching TTS engine data from any source

| What you have | Where it goes | API |
|---|---|---|
| Local binary (clone prompt, `.safetensors`, `.onnx`) | `embeddings/<engine>/<model-slug>/[<lang>/]<name>` plus an `embeddings` entry | `add_engine_embedding()` |
| Engine-generated sample audio | `embeddings/<engine>/<model-slug>/[<lang>/]sample-audio.wav`, language-neutral by default | `add_engine_sample()` |
| Server-side voice handle (ElevenLabs voice ID, Gemini voice ID, OpenAI preset, Apple voice ID) | `manifest.extensions.<provider>`. Reserved IDs: `google`, `elevenlabs`, `openai`, `apple`, `qwen3-tts`, ... | `set_provider_extension()` |

Never store a JSON handle as a fake embedding file.

```python
from voxformat import add_engine_embedding, add_engine_sample, set_provider_extension

add_engine_embedding(vox, data, engine="qwen3-tts", model="Qwen/Qwen3-TTS-12Hz-1.7B", model_slug="1.7b")
add_engine_sample(vox, wav, engine="google", model="gemini-3.8-flash-tts")
set_provider_extension(vox, "elevenlabs", {"voice_id": "vid-example-abc123", "model_id": "eleven_multilingual_v2"})
```

`set_provider_extension` warns when the provider ID is not reserved. It drops `None`
values by default because the Swift `AnyCodable` decoder cannot read JSON `null` inside
`extensions`.

#### Provider adapters

`voxformat.providers.ProviderAdapter` maps a provider's native response to a neutral
`ProviderVoice`, which holds an extension dict plus any samples and embeddings.
`attach()` then writes it to the spec-compliant locations listed above. Built-in
adapters: `google` (Gemini), `elevenlabs`, `openai`, `apple`.

```python
from voxformat.providers import ProviderAdapter, ProviderVoice, register_adapter, get_adapter

@register_adapter
class AzureAdapter(ProviderAdapter):
    provider = "azure"
    def voice_from_response(self, response, **ctx):
        return ProviderVoice(provider="azure", extension={"voice_id": response["ShortName"]})

get_adapter("azure").attach(vox, {"ShortName": "en-US-AvaNeural"})
```

#### Gemini 3.8 Flash TTS designed voices

`voxformat.providers.gemini` contains a stdlib `GeminiVoicesClient` (create, get, list,
delete) and `GeminiAdapter`. The adapter writes this layout:

* `extensions.google = {voice_id, model_id: "gemini-3.8-flash-tts", type, display_name, gender, language_code, prompted_input, created, expire_time, refreshed}`
* `embeddings/google/gemini-3.8-flash-tts/sample-audio.wav`, with manifest key
  `google-gemini-3.8-flash-tts-sample-audio`, engine `google`, model
  `gemini-3.8-flash-tts`, format `wav`, and no `language`.

The Voices API has no PATCH, so `update_vox()` works in three steps: create a new voice,
write the `.vox`, then delete the old ID. If any step fails, the file still points to a
working voice. If the delete fails, the result reports it and the old voice is left on
the server.

The API key comes only from `GEMINI_API_KEY`, or from an explicit argument. It is sent
only in the `x-goog-api-key` header and is redacted from error messages. It is never
printed, logged, or written to disk.

### Validation

```python
from voxformat import validate_file
r = validate_file("voice.vox")          # also accepts a bare manifest .json
r.is_valid, r.errors, r.warnings, r.schema_version
```

Validation runs in three layers:

1. **Semantic.** A port of Swift `validate()`: required fields, UUID v4, `created`,
   description length of at least 10, age range, gender, embedding entries, bundle
   completeness (warnings), and provenance ethics. Cloned voices **must** have a
   non-empty `provenance.source` and `consent` of `"self"` or `"granted"`. Model tags on
   reference audio with no matching embedding produce a warning.
2. **Schema.** Checks the on-disk manifest against the schema for its declared
   `vox_version` (`manifest-v0.1.0` to `v0.4.0`). If there is no schema for that
   version, the latest is used and an info note is added.
3. **Archive and compatibility.** Reports missing `provenance.source` files, extension
   paths that point to missing `embeddings/` files, undeclared embedding binaries (info),
   and `null` values inside `extensions` (warning, because Swift cannot decode them).

Severity follows Swift. A problem is an **error** when Swift would reject the file:
`validate()` returns an error, or decoding fails because of a missing required key, a
wrong JSON type, or an unparseable `created`. Constraints that only the schema enforces
are **warnings**. Examples are unknown keys under `additionalProperties: false`, an
uppercase UUID (Swift lowercases it first), and enum values Swift tolerates. Run
`voxpy validate --strict` to fail on warnings too.

Every example in `examples/` passes with zero warnings. Every fixture in `schemas/test/`
fails.

### Player

```python
from voxformat import VoxFile, player
player.play(VoxFile.open("CLARK.vox"), "1.7b")                     # afplay, then ffplay, then aplay
player.extract(VoxFile.open("CLARK.vox"), "gemini", "clark-gemini.wav")
player.playable_samples(vox)
```

## CLI

```text
voxpy create OUT.vox --name NARRATOR --description "..." [--language en-GB --gender male --method designed]
voxpy inspect FILE [--json]
voxpy validate FILE... [--json] [--strict] [-v] [--engine auto|builtin|jsonschema]
voxpy models FILE [--json]                       # models, playable samples (+languages), providers
voxpy play FILE MODEL [-l LANG] [--player BIN] [--extract PATH] [--list]
voxpy extract FILE MODEL -o PATH [-l LANG] [--clone-prompt] | extract FILE --entry PATH -o OUT
voxpy add-embedding FILE SRC --engine E --model M [--slug S] [--name N] [-l LANG] [--format F] [--key K] [-o OUT]
voxpy add-sample FILE SRC.wav --engine E --model M [--slug S] [-l LANG] [-o OUT]
voxpy set-extension FILE PROVIDER (--json '{...}' | --json-file F | --set k=v ...) [--merge] [--remove] [-o OUT]
voxpy remove FILE PATH... [-o OUT]
voxpy gemini create FILE [--prompt TEXT | --prompt-file F] [--display-name N] [--gender male|female] [-l en-US] [--force]
voxpy gemini get VOICE_ID | list [--type prompted|all] | refresh FILE... | update FILE [...] | delete VOICE_ID [--vox FILE] --yes
```

Examples:

```bash
voxpy play CLARK.vox 1.7b                         # Qwen3-TTS 1.7B sample
voxpy play CLARK.vox gemini --extract /tmp/c.wav  # write the Gemini sample instead of playing it
voxpy play POLYGLOT.vox 0.6b -l es-MX             # base-language fallback to the "es" sample
voxpy add-embedding v.vox cp.bin --engine qwen3-tts --model Qwen/Qwen3-TTS-12Hz-0.6B --slug 0.6b --name clone-prompt.bin
voxpy set-extension v.vox elevenlabs --set voice_id=vid-example-abc123 --set model_id=eleven_multilingual_v2
GEMINI_API_KEY=... voxpy gemini create v.vox      # prompt defaults to voice.description
```

Commands that change a file update it in place, atomically, unless `-o` is given.

## Differences from Swift

| Area | Swift | Python | Why |
|---|---|---|---|
| Unknown manifest fields | Dropped on decode | Preserved, in their original order | The spec says readers must ignore unknown data. A Python round-trip loses nothing. |
| Archive entry order and metadata | Rewritten from a `Dictionary` (no fixed order) | Original order, timestamps and compression kept | Never reorder or recompress entries the caller did not touch |
| `created` re-encoding | Re-encoded as UTC `Z` | Original string kept | Lossless |
| Opening a structurally bad manifest | Throws a decode error | Opens; `validate_file` reports the problems | Lets the validator report every problem at once |
| `add()` under `embeddings/` without `model` | Stores `model: ""` (the docs say it throws) | Raises `MissingRequiredMetadataError` | Follows the documented contract |
| Unsafe paths (`..`, absolute, `\`) | Accepted | `InvalidPathError` | Prevents zip-slip |
| `modified` | Not modeled | Set on write when the container changed | The spec lists it as a field |
| Ties between candidates in a lookup | `Dictionary` order (undefined) | Manifest order | Deterministic |
