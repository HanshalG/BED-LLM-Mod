from __future__ import annotations

from scripts.zendo_structured_outputs import (
    zendo_particle_response_format,
    zendo_scorer_response_format,
)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_particle_schema_is_strict_recursive_and_cross_field_typed() -> None:
    response_format = zendo_particle_response_format()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"]
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["hypotheses"]["minItems"] == 12
    assert schema["properties"]["hypotheses"]["maxItems"] == 12
    assert "$defs" in schema
    assert all(
        node.get("additionalProperties") is False
        for node in _walk(schema)
        if node.get("type") == "object"
    )
    predicate_variants = schema["$defs"]["predicate"]["oneOf"]
    color_variant = next(
        variant
        for variant in predicate_variants
        if variant.get("properties", {})
        .get("attribute", {})
        .get("const")
        == "color"
    )
    assert color_variant["properties"]["value"]["enum"] == [
        "blue",
        "red",
        "green",
    ]
    assert "large" not in color_variant["properties"]["value"]["enum"]


def test_scorer_schema_requires_four_bounded_integers() -> None:
    schema = zendo_scorer_response_format()["json_schema"]["schema"]
    scores = schema["properties"]["root_scores"]
    assert scores["minItems"] == scores["maxItems"] == 4
    assert scores["items"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
    }
