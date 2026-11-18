"""A small JSON Schema validator for the keywords the committed schemas use.

jsonschema is not installed in this environment, and adding it for one check is not worth the
dependency. Supported keywords: type, const, enum, required, properties, additionalProperties
(false), items, minItems, maxItems, minLength, minimum, pattern. Annotation keywords ($schema,
$id, title, description) are ignored. A schema that uses any other validating keyword is
rejected, so a silent pass cannot happen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_SUPPORTED = {
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "const",
    "enum",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "minimum",
    "pattern",
}


def _type_ok(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported type name {name!r}")


def validate(instance: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Return a list of error messages; an empty list means the instance is valid."""
    unknown = set(schema) - _SUPPORTED
    if unknown:
        raise ValueError(f"schema uses unsupported keywords at {path}: {sorted(unknown)}")
    errors: list[str] = []
    if "type" in schema:
        names: Sequence[str] = (
            [schema["type"]] if isinstance(schema["type"], str) else list(schema["type"])
        )
        if not any(_type_ok(instance, n) for n in names):
            errors.append(f"{path}: expected type {'/'.join(names)}")
            return errors
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: not one of {schema['enum']!r}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        props: Mapping[str, Any] = schema.get("properties", {})
        for key, value in instance.items():
            if key in props:
                errors += validate(value, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unexpected key {key!r}")
    if isinstance(instance, list):
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, value in enumerate(instance):
                errors += validate(value, schema["items"], f"{path}[{i}]")
    return errors
