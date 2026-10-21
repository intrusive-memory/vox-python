---
type: project
---

# vox-python — Agent Instructions

`voxformat` is the Python library and `voxpy` CLI for **`.vox`** voice-identity files
(spec v0.4.0). Use it to create, read, update and delete `.vox` files, attach TTS data
from any engine or cloud provider, validate files, play or extract a model's sample, and
render audio from a stored cloud voice on any machine.

- Spec + Swift reference implementation: [intrusive-memory/vox-format](https://github.com/intrusive-memory/vox-format)
  (`docs/VOX-FORMAT.md`). **Swift is the behavioural source of truth**; when it changes, this
  package must be updated to match. Known intentional differences are in `README.md`.
- API and CLI reference: `README.md`. This file covers *how to store things correctly*.

---

## 1. The one rule: one voice = one `.vox`

A character/voice has exactly **one** `.vox`. Every engine, model size, and language is a
keyed entry *inside* that file. Never write `alice-0.6b.vox`, `alice-gemini.vox`, `alice-es.vox`.

```
CLARK.vox (zip)
├── manifest.json
├── embeddings/qwen3-tts/0.6b/clone-prompt.bin          ← local engine binary
├── embeddings/qwen3-tts/0.6b/sample-audio.wav          ← sample from that model
├── embeddings/qwen3-tts/1.7b/clone-prompt.bin
├── embeddings/qwen3-tts/1.7b/sample-audio.wav
├── embeddings/qwen3-tts/1.7b/es/clone-prompt.bin       ← optional language variant
├── embeddings/google/gemini-3.8-flash-tts/sample-audio.wav
└── (manifest.extensions.google = server-side voice handle)
```

## 2. Where each kind of data goes

| What you have | Where it goes | How |
|---|---|---|
| A **local engine binary** (clone prompt, speaker embedding, safetensors) | `embeddings/<engine>/<model-slug>/[<lang>/]<name>` + a manifest `embeddings` entry | `add_engine_embedding()` / `voxpy add-embedding` |
| **Sample audio an engine generated** for this voice | `embeddings/<engine>/<model-slug>/[<lang>/]sample-audio.wav` | `add_engine_sample()` / `voxpy add-sample` |
| A **cloud/server-side voice handle** (Gemini voice id, ElevenLabs voice id, OpenAI preset, Apple voice id) | `manifest.extensions.<provider>` — **never** a fake file under `embeddings/` | `set_provider_extension()` / `voxpy set-extension`, or a provider adapter |
| **Human reference recordings** (the source for cloning) | `reference/<file>.wav` + `referenceAudio` entry (with transcript) | `vox.add(data, "reference/…")` |

Rules that make lookups work (they mirror the Swift reader):

- **Model slug** is the path segment after the engine (`0.6b`, `1.7b`, `gemini-3.8-flash-tts`).
  `model` in the manifest entry is the fully qualified id (`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`).
- **File names matter**: lookups find clone prompts by `clone-prompt` in the path and samples by
  `sample-audio`. Keep those names.
- **Manifest keys** are derived from the path — `embeddings/qwen3-tts/1.7b/sample-audio.wav` →
  `qwen3-tts-1.7b-sample-audio`. Let the library derive them (`add()` updates the manifest for you).
- **Language**: leave the default entry **untagged**. A `language`-tagged entry is *not* returned by
  a default lookup; it is only for variants. Lookup order: exact tag → base language (`fr-FR`→`fr`)
  → untagged default → legacy path.
- **No JSON `null` inside `extensions`** — the Swift reader cannot decode it. The library drops
  `None` values for you.
- Reserved provider ids: `qwen3-tts`, `apple`, `elevenlabs`, `openai`, `coqui`, `parler`,
  `google`, `azure`, `mlx-audio`. Others are allowed but warn.

## 3. Recipes per model

### Qwen3-TTS (local, fully portable)

```bash
voxpy add-embedding CLARK.vox cp-1.7b.bin --engine qwen3-tts \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --slug 1.7b --name clone-prompt.bin
voxpy add-sample CLARK.vox sample-1.7b.wav --engine qwen3-tts \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --slug 1.7b
# language variant:
voxpy add-embedding CLARK.vox cp-es.bin --engine qwen3-tts \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --slug 1.7b --name clone-prompt.bin -l es
```

```python
from voxformat import VoxFile, add_engine_embedding, add_engine_sample
vox = VoxFile.open("CLARK.vox")
add_engine_embedding(vox, prompt_bytes, engine="qwen3-tts",
                     model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", model_slug="1.7b")
add_engine_sample(vox, wav_bytes, engine="qwen3-tts",
                  model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", model_slug="1.7b")
vox.write()
data = vox.clone_prompt_data("1.7b")              # or ("1.7b", language="es")
```

### Google Gemini 3.8 Flash TTS (designed voices — server-side)

Key from env `GEMINI_API_KEY` only. Never print it, log it, or write it into a file.

```bash
voxpy create NARRATOR.vox --name NARRATOR --description "A warm, unhurried narrator in his 60s." --method designed
voxpy gemini create NARRATOR.vox --gender male -l en-US     # prompt defaults to voice.description
voxpy gemini synthesize NARRATOR.vox "The town had one stoplight." -o line.wav [--style "wry"] [--play]
voxpy gemini refresh NARRATOR.vox                             # re-fetch sample + expire_time
voxpy gemini update NARRATOR.vox --prompt "…new prompt…"      # no PATCH in the API: recreate, write, delete old id
voxpy gemini list
voxpy gemini delete voice_… --vox NARRATOR.vox --yes          # irreversible
```

What lands in the file:

```json
"extensions": { "google": {
  "voice_id": "voice_example_abc123", "model_id": "gemini-3.8-flash-tts", "type": "prompted",
  "display_name": "NARRATOR", "gender": "male", "language_code": "en-US",
  "prompted_input": "A warm, unhurried narrator in his 60s.",
  "created": "…", "expire_time": "…", "refreshed": "…" } }
```
plus `embeddings/google/gemini-3.8-flash-tts/sample-audio.wav` (untagged, key
`google-gemini-3.8-flash-tts-sample-audio`).

### ElevenLabs / OpenAI / Apple (handles only)

```bash
voxpy set-extension V.vox elevenlabs --set voice_id=vid-example-abc123 --set model_id=eleven_multilingual_v2
voxpy set-extension V.vox openai --set voice=alloy --set model=gpt-4o-mini-tts
voxpy set-extension V.vox apple --set voice_id=en-US/Aaron
```
If you have engine-generated sample audio for these, add it with `add-sample --engine <provider>`.

## 4. Cross-machine use — what is actually portable

| Engine | What the `.vox` holds | Works on another machine? |
|---|---|---|
| Qwen3-TTS | the voice itself (`clone-prompt.bin`) | **Yes, offline**, any machine, forever |
| Gemini | a handle (`voice_id`) + recipe (`prompted_input`) + preview sample | **Yes, if** the machine's `GEMINI_API_KEY` belongs to the **same Google project** that created the voice, and before `expire_time` (1 year). Otherwise the id won't resolve |
| ElevenLabs / OpenAI / Apple | a handle | Same account / same OS voice installed |

Gemini specifics (verified against the live API, 2026-10):
- Designed voices **cannot be exported** as data. `store: false` is rejected for prompted voices
  ("Prompted voice creation requires store=true"); you get only the id, the prompt and `sample_audio`.
- `voices.get` returns the original prompt and an exact `expire_time`. There is no PATCH.
- Recreating from `prompted_input` (`voxpy gemini update`) gives a **new id and a different-sounding
  voice**. Treat it as a last resort after expiry or a project change.
- The stored Gemini sample can't recreate the voice inside Gemini: replication requires a consent
  recording by the real human speaker, which a designed voice doesn't have. Don't fake one.
  For an offline fallback, clone the sample into a local engine (e.g. Qwen3-TTS Base from the sample
  + its transcript) and store that as a normal `qwen3-tts` clone prompt in the same `.vox`. Set
  `provenance.source` to the Gemini sample. Check Google's terms on reusing generated audio first.

## 5. Validate and play

```bash
voxpy validate V.vox [--strict] [--json]   # schema + Swift validate() rules + every referenced file exists
voxpy models V.vox                          # models, playable samples, providers
voxpy play V.vox 1.7b                       # or 0.6b, gemini, qwen3-tts-1.7b; -l es for a variant
voxpy play V.vox gemini --extract out.wav
voxpy play V.vox --list
```
Always run `voxpy validate` after writing. Errors are what the Swift reader would reject;
`--strict` also fails on warnings.

## 6. Ethics and hygiene (required)

- Every `.vox` records **provenance** (`method`, `engine`, notes). A **cloned** voice MUST have a
  non-empty `provenance.source` and `consent` of `"self"` or `"granted"`; the validator enforces this.
- Never create voices of real people (public figures included) without their verifiable consent.
  Designed voices (from a text description) are the safe default.
- Never commit API keys or real voice ids in examples or tests. Use placeholders like
  `voice_example_abc123` / `vid-example-abc123`.

## 7. Working on this repo

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # no network: HTTP is faked, sockets are blocked
```
- Tests use the spec's `examples/` and `schemas/`, copied into `tests/fixtures/vox-format/`.
  When vox-format changes them, re-copy them; `test_validation.py` fails if the bundled
  `src/voxformat/schemas/` drift from the copies.
- This package is also kept in `vox-format/implementations/python`. Keep the two in sync or
  retire one; don't let them diverge.
