# VOX Format Examples

This directory contains example `.vox` files demonstrating the VOX format specification.

## Table of Contents

- [Introduction](#introduction)
- [Minimal Example](#minimal-example)
- [Character with Reference Audio](#character-with-reference-audio)
- [Character with Context](#character-with-context)
- [Multi-Engine Example](#multi-engine-example)
- [Multi-Language Example](#multi-language-example)
- [Validation](#validation)

## Introduction

VOX is a vendor-neutral file format for voice identities used in text-to-speech synthesis. These examples demonstrate different levels of detail and use cases, from the simplest minimal voice to rich character voices with screenplay context and multi-engine support. Each `.vox` file is a ZIP archive containing a JSON manifest and optional assets like reference audio.

## Minimal Example

### Use Case

The minimal example demonstrates the simplest valid `.vox` file containing only the required fields. This is suitable for basic voice identification when no reference audio or engine-specific data is available. The text description alone allows any voice-design-capable TTS engine to generate an approximation of the desired voice.

### Required Fields

All `.vox` files must contain these fields:

- `vox_version` - Specification version (currently "0.4.0")
- `id` - Unique UUID v4 identifier
- `created` - ISO 8601 timestamp
- `voice.name` - Display name for the voice
- `voice.description` - Natural language description of the voice characteristics

### Inspection

List the contents of the archive:

```bash
unzip -l minimal/narrator.vox
```

Extract and view the manifest:

```bash
unzip minimal/narrator.vox && cat manifest.json
```

View the [narrator.vox](minimal/narrator.vox) file directly.

## Character with Reference Audio

### Use Case

This example demonstrates a character voice with metadata about reference audio files. When actual reference audio is provided (deferred to Phase 2), the manifest includes file paths, transcripts, and context. This allows TTS engines with voice cloning capabilities to generate high-fidelity reproductions of the target voice.

### File Structure

```
character/
├── manifest.json
└── reference/
    └── sample-01.wav
```

### Reference Audio Fields

The `reference_audio` array contains objects with:

```json
{
  "file": "reference/sample-01.wav",
  "transcript": "Exact text spoken in the audio clip",
  "language": "en",
  "duration_seconds": 4.5,
  "context": "Emotional or situational context"
}
```

### Inspection

```bash
unzip -l character/manifest.json
cat character/manifest.json | jq .reference_audio
```

## Character with Context

### Use Case

The character context example shows how `.vox` files integrate with screenplay production workflows. It includes character role descriptions, emotional range, relationships to other characters, and provenance information. This is particularly useful for narrative audio production where voice identity is tied to story context.

### Character Fields

The manifest includes:

- `character.role` - Character's narrative function and background
- `character.emotional_range` - Array of emotions the character expresses
- `character.relationships` - Map of relationships to other characters
- `character.source` - Reference to the screenplay or narrative source

### File Structure

```
character/narrator-with-context.vox
└── manifest.json (includes character, prosody, provenance)
```

### Inspection

```bash
unzip -p character/narrator-with-context.vox manifest.json | jq .character
```

View the [narrator-with-context.vox](character/narrator-with-context.vox) file.

## Multi-Engine Example

### Use Case

This example demonstrates how a single `.vox` file can contain engine-specific extensions for multiple TTS providers while remaining vendor-neutral. The core voice description is universal, while optional extensions provide optimized parameters for Apple TTS, ElevenLabs, and Qwen3-TTS.

### Extension Namespaces

The `extensions` object contains provider-specific data:

- `extensions.apple` - Apple AVSpeechSynthesis voice ID
- `extensions.elevenlabs` - ElevenLabs voice ID and model
- `extensions.qwen3-tts` - Qwen3-TTS design instruction

Conforming readers must ignore unknown extensions, ensuring forward compatibility.

### File Structure

```
multi-engine/cross-platform.vox
└── manifest.json (includes extensions for multiple providers)
```

### Inspection

```bash
unzip -p multi-engine/cross-platform.vox manifest.json | jq .extensions
```

View the [cross-platform.vox](multi-engine/cross-platform.vox) file.

## Multi-Language Example

### Use Case

This example (v0.4.0) demonstrates *optional per-language samples and clone prompts* for a single model. The same 0.6B model carries a default (language-neutral) clone-prompt + sample-audio pair, plus Spanish (`es`) and French-France (`fr-FR`) variants. A reader requesting a specific `(model, language)` resolves the matching language; if absent, it falls back to the base language, then to the default.

### Language Resolution

For a `(model, language)` lookup, readers resolve in this order:

1. **Exact** language match (e.g. request `es` → the `es` embedding).
2. **Base-language** fallback (e.g. request `es-MX` → the `es` embedding).
3. **Default / language-neutral** embedding (the one with no `language` field).
4. `nil` if none of the above exist.

Requesting `language: null` or `"default"` resolves only the default/language-neutral embedding, preserving pre-0.4.0 behavior.

### Embedding Fields

Each embedding entry may carry an optional `language` field (BCP 47). Its absence marks the default/language-neutral embedding:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B",
  "engine": "qwen3-tts",
  "file": "embeddings/qwen3-tts/0.6b/es/sample-audio.wav",
  "format": "wav",
  "language": "es"
}
```

The language is also reflected as a path segment (`.../0.6b/es/...`) by convention, but the `language` field is the source of truth for resolution.

### File Structure

```
multi-language/manifest.json
└── embeddings/qwen3-tts/0.6b/
    ├── clone-prompt.bin        (default)
    ├── sample-audio.wav        (default)
    ├── es/clone-prompt.bin     (Spanish)
    ├── es/sample-audio.wav     (Spanish)
    ├── fr-FR/clone-prompt.bin  (French, France)
    └── fr-FR/sample-audio.wav  (French, France)
```

### Inspection

```bash
cat multi-language/manifest.json | jq '.embeddings | map_values(.language)'
```

## Validation

All example `.vox` files are valid ZIP archives containing well-formed JSON manifests. You can validate them using:

### Manual Validation

```bash
# Test ZIP integrity
unzip -t examples/minimal/narrator.vox

# Validate JSON syntax
unzip -p examples/minimal/narrator.vox manifest.json | jq .
```

### Schema Validation

The canonical validator is the Swift implementation. Example and negative-fixture conformance runs in the test suite:

```bash
cd implementations/swift
swift test --filter SchemaExampleValidationTests
```

To validate a single `.vox` file from the command line:

```bash
swift run vox validate examples/minimal/narrator.vox   # from tools/vox-cli
```

The JSON Schema document in `schemas/` is also usable with any external JSON Schema (Draft 2020-12) validator (e.g. `ajv`) for editor integrations — see [`schemas/README.md`](../schemas/README.md).
