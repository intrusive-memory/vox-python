# Voice Creation Workflow

## Overview

This document describes the end-to-end process for creating, validating, and publishing a voice to the VOX Voice Library. Follow these steps to produce a library-quality `.vox` file and make it available on the CDN for public consumption.

## Prerequisites

The following tools are required:

- **vox** -- VOX command-line tool for creating, validating, and inspecting `.vox` files. Build from `tools/vox-cli/` (`swift build`); validate with `swift run vox validate <file>.vox`.
- **rclone** or **aws-cli** -- File transfer to Cloudflare R2 CDN. Install: [rclone.org/install](https://rclone.org/install/) or `brew install awscli`

## Steps

### Step 1: Choose a Voice Target

Select a voice from `VOICE_TARGETS.md`. Note the voice ID, category, and all metadata fields.

```
Target: NARR-001 (Audiobook Narrator, female, 30-45, en)
```

### Step 2: Create the Base VOX File

Use `vox-cli create` to generate the initial `.vox` file with required fields:

```bash
vox create --name "Audiobook Narrator" \
  --description "Warm, resonant alto with excellent breath control and subtle emotional modulation. Maintains listener engagement through gentle pitch variation and measured pacing." \
  --language en --gender female \
  --output NARR-001.vox
```

### Step 3: Edit the Manifest

Extract and enhance the manifest with prosody, tags, and provenance:

```bash
vox extract NARR-001.vox --output-dir ./tmp-narr-001
```

Edit `./tmp-narr-001/manifest.json` to add:

```json
{
  "prosody": {
    "pitch_base": "medium-low",
    "pitch_range": "moderate",
    "rate": "measured",
    "energy": "medium",
    "emotion_default": "warm engagement"
  },
  "provenance": {
    "method": "designed",
    "engine": null,
    "consent": null,
    "license": "CC0-1.0",
    "notes": "Voice designed from text description for the VOX library."
  }
}
```

Add tags to the voice object: `["narrator", "audiobook", "warm", "alto", "long-form"]`.

### Step 4: Repackage as .vox

Create the final `.vox` archive from the edited directory:

```bash
cd ./tmp-narr-001 && zip -r ../NARR-001.vox manifest.json && cd ..
```

### Step 5: Validate in Strict Mode

Run validation to confirm the file meets all library requirements:

```bash
vox validate --strict NARR-001.vox
```

Also validate the manifest against the JSON Schema:

```bash
swift run vox validate ./narr-001.vox   # from tools/vox-cli
```

### Step 6: Upload to CDN

Transfer the validated `.vox` file to the appropriate CDN category directory:

```bash
rclone copy NARR-001.vox r2:vox-library/v1/narrators/
```

Or with aws-cli:

```bash
aws s3 cp NARR-001.vox s3://vox-library/v1/narrators/ --endpoint-url https://your-r2-endpoint.r2.cloudflarestorage.com
```

### Step 7: Update the Library Index

Add a new entry to `index.json` for the published voice:

```json
{
  "file": "narrators/NARR-001.vox",
  "name": "Audiobook Narrator",
  "description": "Warm, resonant alto with excellent breath control and subtle emotional modulation.",
  "tags": ["narrator", "audiobook", "warm", "alto", "long-form"],
  "language": "en",
  "gender": "female",
  "age_range": [30, 45],
  "category": "narrator"
}
```

Upload the updated index:

```bash
rclone copy index.json r2:vox-library/v1/
```

### Step 8: Verify the Published Voice

Confirm the voice is accessible at its public URL:

```bash
curl -o /tmp/test.vox https://cdn.intrusive-memory.productions/vox-library/v1/narrators/NARR-001.vox
vox validate --strict /tmp/test.vox
```

## Quality Checklist

Before merging a new library voice, verify all of the following:

- [ ] Voice validates with `vox validate --strict` (exit code 0)
- [ ] Manifest validates against `schemas/manifest-v0.1.0.json`
- [ ] `voice.description` is detailed and specific (minimum 100 characters)
- [ ] `voice.tags` contains at least 3 searchable tags
- [ ] `provenance.method` is `"designed"` (never `"cloned"`)
- [ ] `provenance.license` is `"CC0-1.0"`
- [ ] `voice.language` is a valid BCP 47 code
- [ ] `voice.gender` is one of: male, female, nonbinary, neutral
- [ ] Entry added to `index.json` with correct file path and metadata
- [ ] CDN URL returns HTTP 200 and the file downloads correctly

## Troubleshooting

### Validation Fails

**Symptom:** `vox validate --strict` reports errors.

**Common causes:** Missing required fields, description too short, invalid UUID format, or `age_range` where min exceeds max. Fix the manifest, repackage, and revalidate.

### Upload Fails

**Symptom:** `rclone copy` or `aws s3 cp` returns an error.

**Common causes:** Incorrect R2 credentials, wrong bucket name, or network issues. Verify credentials with `rclone lsd r2:` to list buckets. Ensure the path includes `v1/` and the correct category subdirectory.

### Index Malformed

**Symptom:** `index.json` fails to parse or search returns unexpected results.

**Common causes:** Trailing comma after last entry, mismatched brackets, or duplicate voice IDs. Validate with `jq . index.json` to find syntax errors. Ensure each entry has all required fields.
