"""Strict OpenRouter JSON Schemas for executable Zendo particle populations."""

from __future__ import annotations

from typing import Any

from scripts.zendo_path_dependent_belief_gate import (
    COLORS,
    ORIENTATIONS,
    PARTICLE_COUNT,
    SIZES,
)


def _closed_object(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties),
        "additionalProperties": False,
    }


def _const(value: str) -> dict[str, Any]:
    return {"type": "string", "const": value}


def _attribute_predicate(attribute: str, values: list[Any]) -> dict[str, Any]:
    value_schema: dict[str, Any]
    if all(isinstance(value, bool) for value in values):
        value_schema = {"type": "boolean"}
    else:
        value_schema = {"type": "string", "enum": values}
    return _closed_object(
        {
            "op": _const("attribute"),
            "attribute": _const(attribute),
            "value": value_schema,
        }
    )


def zendo_particle_response_format() -> dict[str, Any]:
    predicate_ref = {"$ref": "#/$defs/predicate"}
    rule_ref = {"$ref": "#/$defs/rule"}
    predicate = {
        "oneOf": [
            _closed_object({"op": _const("any")}),
            _attribute_predicate("color", list(COLORS)),
            _attribute_predicate("size", list(SIZES)),
            _attribute_predicate("orientation", list(ORIENTATIONS)),
            _attribute_predicate("grounded", [True, False]),
            _closed_object(
                {
                    "op": {"type": "string", "enum": ["and", "or"]},
                    "args": {
                        "type": "array",
                        "items": predicate_ref,
                        "minItems": 2,
                        "maxItems": 4,
                    },
                }
            ),
            _closed_object(
                {"op": _const("not"), "arg": predicate_ref}
            ),
        ]
    }
    quantified = [
        _closed_object(
            {"op": _const(op), "predicate": predicate_ref}
        )
        for op in ("exists", "forall")
    ]
    rule = {
        "oneOf": [
            *quantified,
            _closed_object(
                {
                    "op": _const("count"),
                    "predicate": predicate_ref,
                    "comparison": {
                        "type": "string",
                        "enum": ["eq", "ge", "le"],
                    },
                    "value": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 6,
                    },
                }
            ),
            _closed_object(
                {
                    "op": _const("all_same"),
                    "attribute": {
                        "type": "string",
                        "enum": [
                            "color",
                            "size",
                            "orientation",
                            "grounded",
                        ],
                    },
                }
            ),
            _closed_object(
                {
                    "op": _const("touching"),
                    "left": predicate_ref,
                    "right": predicate_ref,
                }
            ),
            _closed_object(
                {
                    "op": _const("stacking"),
                    "upper": predicate_ref,
                    "lower": predicate_ref,
                }
            ),
            _closed_object(
                {"op": _const("largest_all"), "predicate": predicate_ref}
            ),
            _closed_object(
                {
                    "op": {"type": "string", "enum": ["and", "or"]},
                    "rules": {
                        "type": "array",
                        "items": rule_ref,
                        "minItems": 2,
                        "maxItems": 4,
                    },
                }
            ),
            _closed_object({"op": _const("not"), "rule": rule_ref}),
        ]
    }
    hypothesis = _closed_object(
        {
            "id": {"type": "string", "pattern": "^H(0[1-9]|1[0-2])$"},
            "rule_text": {"type": "string", "minLength": 1},
            "rule": rule_ref,
        }
    )
    schema = _closed_object(
        {
            "hypotheses": {
                "type": "array",
                "items": hypothesis,
                "minItems": PARTICLE_COUNT,
                "maxItems": PARTICLE_COUNT,
            }
        }
    )
    schema["$defs"] = {"predicate": predicate, "rule": rule}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "zendo_particle_population",
            "strict": True,
            "schema": schema,
        },
    }


def zendo_scorer_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "zendo_root_scores",
            "strict": True,
            "schema": _closed_object(
                {
                    "root_scores": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "minItems": 4,
                        "maxItems": 4,
                    }
                }
            ),
        },
    }
