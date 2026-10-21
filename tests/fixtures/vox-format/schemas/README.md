# VOX Manifest JSON Schema

## Overview

This directory contains the JSON Schema definition for the VOX manifest format. The schema provides machine-readable validation rules for `manifest.json` files found inside `.vox` archives. It enforces structural constraints such as required fields, data types, string patterns (UUID v4, ISO 8601 timestamps, semver), and enumerated values. Use the schema during development to catch malformed manifests early, in CI pipelines to gate releases, or in editor integrations for live feedback while authoring voice identity files.

The current schema version is **v0.4.0**, targeting JSON Schema Draft 2020-12.

## Validation

The **canonical validator is the Swift reference implementation** (`implementations/swift/`). It enforces the same rules as this schema (required fields, UUID v4, ISO 8601 timestamps, gender enum, age-range constraints, embedding paths) in code, and is exercised in CI.

### Canonical (Swift) — no external dependencies

Example and negative-fixture conformance is validated by the test suite, which decodes each manifest with `VoxManifest` and runs `VoxFile.validate()`:

```bash
cd implementations/swift
swift test --filter SchemaExampleValidationTests
```

This covers every example manifest, every example `.vox` archive, and every negative fixture in `schemas/test/`. To validate an arbitrary `.vox` file from the command line, use the Swift CLI:

```bash
swift run vox validate path/to/voice.vox   # from tools/vox-cli
```

### Optional: validating the JSON Schema document with an external engine

The `manifest-v{version}.json` document is provided for editor integrations and language-agnostic tooling. Any JSON Schema (Draft 2020-12) validator works; for example with [`ajv`](https://ajv.js.org/):

```bash
ajv validate -s schemas/manifest-v0.4.0.json -d examples/minimal/manifest.json --spec=draft2020
```

[`jv`](https://github.com/santhosh-tekuri/jsonschema) (Go) is another option. These are optional conveniences — the Swift suite is the source of truth for conformance.

## Troubleshooting

**Error: `'vox_version' is a required property`**
The manifest is missing the `vox_version` field. Every VOX manifest must include `"vox_version": "0.4.0"` at the top level. Ensure the field name is spelled exactly as `vox_version` (with underscore, not camelCase).

**Error: `'<value>' does not match '<pattern>'`**
A string field does not match its required format. Common causes: the `id` field is not a valid UUID v4 (must be lowercase hex with the version-4 nibble), or the `created` field is not in ISO 8601 format (must be `YYYY-MM-DDTHH:MM:SSZ` or with a timezone offset like `+05:30`). Double-check the exact format requirements in the schema.

## References

- [JSON Schema Specification (Draft 2020-12)](https://json-schema.org/draft/2020-12/json-schema-core)
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/)
- [VOX Format Specification](../docs/VOX-FORMAT.md)
- [VOX Example Files](../examples/)
