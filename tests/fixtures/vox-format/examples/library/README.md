# VOX Voice Library

This directory contains curated `.vox` voice files organized by category for easy discovery and reuse.

## Categories

### Narrators (`narrators/`)

Professional narrator voices optimized for long-form content:
- **Audiobook narrators:** Warm, engaging, suitable for 8+ hour narration
- **Documentary narrators:** Authoritative, clear, educational tone
- **Storytelling narrators:** Expressive, dynamic, theatrical

### Characters (`characters/`)

Character voices for fiction, drama, and narrative production:
- Age-appropriate voices (young, middle-aged, elderly)
- Diverse emotional ranges and personalities
- Character context fields included (role, relationships)

### Accents (`accents/`)

Regional and international accent variations:
- Geographic distinctions (e.g., British RP, Southern US, Australian)
- Same base language, different pronunciation patterns
- Cultural authenticity when designed from description

## Naming Conventions

Voice files use descriptive kebab-case names:
- `narrators/audiobook.vox` - Self-explanatory purpose
- `characters/young-protagonist.vox` - Age + role
- `accents/british-rp.vox` - Region + style

Index entries use the same naming for consistency.

## Provenance Requirements

**CRITICAL:** All library voices MUST meet these requirements:

### Method: "designed"

All voices in this library are created from text descriptions only. No real person voices are included.

**Why?** This avoids consent and licensing complications. Description-designed voices are:
- Ethically unambiguous (no cloned identity)
- Legally safe (no person's likeness rights)
- Universally compatible (any TTS engine can interpret the description)

### License: CC0-1.0

All library voices are dedicated to the public domain under Creative Commons CC0 1.0 Universal.

**Why?** Maximum freedom for users:
- Use commercially without attribution
- Modify and redistribute freely
- No legal barriers to adoption

### Checklist for Adding Voices

Before adding a voice to the library, verify:

- [ ] `provenance.method` is `"designed"` (never `"cloned"`)
- [ ] `provenance.license` is `"CC0-1.0"`
- [ ] `voice.description` is detailed (100+ words)
- [ ] Voice validates successfully (`vox-cli validate --strict`)
- [ ] Unique use case (not redundant with existing library voices)
- [ ] Entry added to `library/index.json`
- [ ] Tags are descriptive and searchable

## Why No Real Person Voices?

Real person voices require:
1. **Explicit consent** from the voice owner
2. **License agreements** for commercial use
3. **Provenance verification** (who cloned when?)
4. **Liability risk** (deepfake misuse, identity theft)

By restricting the library to description-designed voices, we eliminate these concerns entirely.

## CDN Hosting

The VOX Voice Library is distributed via CDN for programmatic access. Voices can be downloaded directly by URL without cloning the repository, enabling integrations where applications fetch voice definitions on demand.

### URL Structure

All library voices are served under a versioned path:

```
https://cdn.intrusive-memory.productions/vox-library/v1/{category}/{voice-id}.vox
```

Example URLs:

- `https://cdn.intrusive-memory.productions/vox-library/v1/narrators/NARR-001.vox`
- `https://cdn.intrusive-memory.productions/vox-library/v1/characters/CHAR-003.vox`
- `https://cdn.intrusive-memory.productions/vox-library/v1/accents/ACNT-002.vox`
- `https://cdn.intrusive-memory.productions/vox-library/v1/ages/AGE-001.vox`
- `https://cdn.intrusive-memory.productions/vox-library/v1/genres/GENR-001.vox`

### File Naming

CDN files use the voice target ID as the filename: `{CATEGORY-ABBREV}-{NUMBER}.vox`. The category abbreviations are:

- `NARR` - Narrators
- `CHAR` - Characters
- `ACNT` - Accents
- `AGE` - Ages (note: no trailing S)
- `GENR` - Genres

### Library Index

The full library index is available at the CDN root:

```
https://cdn.intrusive-memory.productions/vox-library/v1/index.json
```

This JSON file contains metadata for every voice in the library, enabling search and discovery without downloading individual files.

### Directory Structure

```
vox-library/
└── v1/
    ├── index.json
    ├── narrators/
    │   ├── NARR-001.vox
    │   ├── NARR-002.vox
    │   └── ...
    ├── characters/
    │   ├── CHAR-001.vox
    │   └── ...
    ├── accents/
    │   ├── ACNT-001.vox
    │   └── ...
    ├── ages/
    │   ├── AGE-001.vox
    │   └── ...
    └── genres/
        ├── GENR-001.vox
        └── ...
```

### Upload Process

To publish a new voice to the CDN:

1. **Create** the `.vox` file locally using `vox-cli create`
2. **Validate** with `vox-cli validate --strict` to ensure compliance
3. **Upload** via rclone or aws-cli: `rclone copy NARR-001.vox r2:vox-library/v1/narrators/`
4. **Update** `index.json` with the new entry and upload the revised index
5. **Verify** the download URL returns the correct file: `curl -I https://cdn.intrusive-memory.productions/vox-library/v1/narrators/NARR-001.vox`

### CORS Configuration

The CDN is configured for broad client access:

- **Allowed methods:** GET, HEAD
- **Allowed origins:** `*` (all origins permitted)
- **Cache duration:** 24 hours (`Cache-Control: public, max-age=86400`)
- **Compression:** gzip enabled for `index.json` and manifest data
- **Content-Type:** `application/zip` for `.vox` files, `application/json` for index

### Versioning

The `v1/` path prefix represents the current library layout. If the index schema, directory structure, or naming conventions change in a breaking way, a new version directory (`v2/`) will be created. The previous version remains available for backward compatibility. Non-breaking additions (new voices, updated descriptions) are published in-place under the existing version.

## Contributing

To contribute a voice:
1. Read the provenance requirements above
2. Create the `.vox` file using `vox-cli create`
3. Validate with `vox-cli validate --strict`
4. Add entry to `library/index.json`
5. Submit PR with rationale for inclusion

See root `CONTRIBUTING.md` for full guidelines.

---

**Last Updated:** 2026-02-13
