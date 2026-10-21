"""Versioned JSON-schema loading plus a small, dependency-free validator.

The built-in engine implements exactly the JSON Schema (draft 2020-12) keywords used by
``schemas/manifest-v*.json``: ``type``, ``required``, ``properties``,
``additionalProperties``, ``items``, ``enum``, ``const``, ``pattern``, ``minLength``,
``maxLength``, ``minItems``, ``maxItems``, ``minimum``, ``maximum``, ``oneOf``,
``anyOf``, ``allOf``, ``if``/``then``/``else``. Annotation keywords are ignored.

If the optional ``jsonschema`` package is installed it can be used instead
(``engine="jsonschema"``); both engines interpret the same bundled schema files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Iterator

from .manifest import CURRENT_VERSION

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class SchemaError:
    field: str
    message: str
    keyword: str = ""


def available_versions() -> list[str]:
    names = [p.name for p in resources.files("voxformat").joinpath("schemas").iterdir()]
    versions = []
    for n in names:
        m = re.match(r"^manifest-v(\d+\.\d+\.\d+)\.json$", n)
        if m:
            versions.append(m.group(1))
    return sorted(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


@lru_cache(maxsize=None)
def load_schema(version: str) -> dict[str, Any]:
    path = resources.files("voxformat").joinpath("schemas", f"manifest-v{version}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def select_schema_version(declared: Any) -> tuple[str, str | None]:
    """Pick the schema for a declared ``vox_version``.

    Returns ``(version, note)``; ``note`` explains a fallback to the latest schema.
    """
    versions = available_versions()
    latest = versions[-1] if versions else CURRENT_VERSION
    if isinstance(declared, str) and declared in versions:
        return declared, None
    if isinstance(declared, str) and _VERSION_RE.match(declared):
        return latest, f"No bundled schema for vox_version '{declared}'; validated against v{latest}"
    return latest, f"vox_version missing or malformed; validated against v{latest}"


def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if t == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else key


def _short(v: Any) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= 60 else s[:57] + "..."


def iter_errors(schema: dict[str, Any] | bool, value: Any, path: str = "") -> Iterator[SchemaError]:
    """Yield every schema violation of ``value``."""
    if schema is True or schema == {}:
        return
    if schema is False:
        yield SchemaError(path or "$", "is not allowed", "false")
        return
    assert isinstance(schema, dict)
    where = path or "$"

    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            yield SchemaError(where, f"{_short(value)} is not of type {' or '.join(types)}", "type")
            return

    if "const" in schema and value != schema["const"]:
        yield SchemaError(where, f"{_short(value)} must equal {_short(schema['const'])}", "const")
    if "enum" in schema and value not in schema["enum"]:
        yield SchemaError(where, f"{_short(value)} is not one of {schema['enum']}", "enum")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            yield SchemaError(where, f"is shorter than {schema['minLength']} characters", "minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            yield SchemaError(where, f"is longer than {schema['maxLength']} characters", "maxLength")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            yield SchemaError(where, f"{_short(value)} does not match '{schema['pattern']}'", "pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            yield SchemaError(where, f"{value} is less than the minimum of {schema['minimum']}", "minimum")
        if "maximum" in schema and value > schema["maximum"]:
            yield SchemaError(where, f"{value} is greater than the maximum of {schema['maximum']}", "maximum")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            yield SchemaError(where, f"has {len(value)} items, expected at least {schema['minItems']}", "minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            yield SchemaError(where, f"has {len(value)} items, expected at most {schema['maxItems']}", "maxItems")
        if "items" in schema:
            for i, item in enumerate(value):
                yield from iter_errors(schema["items"], item, _join(path, i))

    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                yield SchemaError(_join(path, req), "is a required property", "required")
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in value:
                yield from iter_errors(sub, value[k], _join(path, k))
        addl = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                continue
            if addl is False:
                yield SchemaError(_join(path, k), "additional property is not allowed", "additionalProperties")
            elif isinstance(addl, dict):
                yield from iter_errors(addl, v, _join(path, k))

    if "allOf" in schema:
        for sub in schema["allOf"]:
            yield from iter_errors(sub, value, path)
    if "anyOf" in schema:
        if not any(not list(iter_errors(sub, value, path)) for sub in schema["anyOf"]):
            yield SchemaError(where, f"{_short(value)} does not match any allowed schema", "anyOf")
    if "oneOf" in schema:
        branch_errors = [list(iter_errors(sub, value, path)) for sub in schema["oneOf"]]
        matches = sum(1 for errs in branch_errors if not errs)
        if matches != 1:
            # If every branch rejected the value's *type*, report it as a type error.
            all_type = matches == 0 and all(errs and errs[0].keyword == "type" and errs[0].field == where
                                            for errs in branch_errors)
            yield SchemaError(where, f"{_short(value)} must match exactly one allowed schema (matched {matches})",
                              "type" if all_type else "oneOf")
    if "if" in schema:
        if not list(iter_errors(schema["if"], value, path)):
            if "then" in schema:
                yield from iter_errors(schema["then"], value, path)
        elif "else" in schema:
            yield from iter_errors(schema["else"], value, path)


def jsonschema_available() -> bool:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def iter_errors_jsonschema(schema: dict[str, Any], value: Any) -> Iterator[SchemaError]:
    import jsonschema

    cls = jsonschema.validators.validator_for(schema)
    validator = cls(schema)
    for err in sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path)):
        path = ""
        for part in err.absolute_path:
            path = _join(path, part)
        if err.validator == "additionalProperties" and err.validator_value is False and isinstance(err.instance, dict):
            # One issue per unexpected property, located at the property (like the built-in engine).
            known = set(err.schema.get("properties", {}))
            for k in err.instance:
                if k not in known:
                    yield SchemaError(_join(path, k), "additional property is not allowed", "additionalProperties")
            continue
        if err.validator == "required":
            # Point at the missing property like the built-in engine does.
            m = re.match(r"^'(.+)' is a required property$", err.message)
            if m:
                yield SchemaError(_join(path, m.group(1)), "is a required property", "required")
                continue
        yield SchemaError(path or "$", err.message, str(err.validator))


def validate_against_schema(manifest: Any, version: str, *, engine: str = "builtin") -> list[SchemaError]:
    schema = load_schema(version)
    if engine == "jsonschema":
        return list(iter_errors_jsonschema(schema, manifest))
    if engine != "builtin":
        raise ValueError(f"unknown schema engine '{engine}'")
    return list(iter_errors(schema, manifest))
