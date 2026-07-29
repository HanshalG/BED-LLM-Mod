import json

from scripts import number_game_diversity_grid as grid


def _valid_response() -> str:
    return json.dumps(
        {
            "families": {
                "periodic_digit": [
                    {"name": "even", "expression": "divisible(n, 2)"},
                    {"name": "mod3", "expression": "n % 3 == 0"},
                    {"name": "mod5", "expression": "n % 5 == 1"},
                    {"name": "ends7", "expression": "ends_with(n, 7)"},
                    {"name": "ds_even", "expression": "digit_sum(n) % 2 == 0"},
                    {"name": "ds_low", "expression": "digit_sum(n) < 8"},
                ],
                "ordered": [
                    {"name": "under10", "expression": "n < 10"},
                    {"name": "under20", "expression": "n < 20"},
                    {"name": "over20", "expression": "n > 20"},
                    {"name": "over50", "expression": "n >= 50"},
                    {"name": "middle", "expression": "20 < n and n < 80"},
                    {"name": "low_band", "expression": "5 <= n and n <= 30"},
                ],
                "number_theoretic": [
                    {"name": "prime", "expression": "is_prime(n)"},
                    {"name": "square", "expression": "is_square(n)"},
                    {"name": "power2", "expression": "is_power_of_two(n)"},
                    {"name": "shift_prime", "expression": "is_prime(n + 1)"},
                    {"name": "scaled_square", "expression": "is_square(2 * n + 1)"},
                    {"name": "shift_power", "expression": "is_power_of_two(n + 2)"},
                ],
                "compositional": [
                    {
                        "name": "even_low",
                        "expression": "divisible(n, 2) and n < 40",
                    },
                    {
                        "name": "prime_high",
                        "expression": "is_prime(n) and n > 30",
                    },
                    {
                        "name": "square_or_end7",
                        "expression": "is_square(n) or ends_with(n, 7)",
                    },
                    {
                        "name": "power_or_band",
                        "expression": "is_power_of_two(n) or (40 < n and n < 60)",
                    },
                    {
                        "name": "digit_high",
                        "expression": "digit_sum(n) < 7 and n > 20",
                    },
                    {
                        "name": "prime_or_even_band",
                        "expression": "is_prime(n) or (divisible(n, 2) and n < 20)",
                    },
                ],
            }
        }
    )


def test_feasibility_audit_fails_closed_on_hardest_periodic_cell():
    result = grid.run_feasibility_audit()

    assert result["status"] == "gated_null"
    assert result["gates"]["all_pass"] is False
    assert len(result["cases"]) == 10
    hardest = result["cases"][8]
    assert hardest["observations"] == [[42, True], [75, True]]
    assert hardest["feasible_unique_by_family"]["periodic_digit"] == 14
    assert min(case["feasible_unique_union"] for case in result["cases"]) >= 4_000


def test_parse_response_accepts_four_behavioral_families():
    hypotheses, diagnostics = grid.parse_response(_valid_response())

    assert len(hypotheses) == 24
    assert diagnostics["valid_unique_by_family"] == {
        "periodic_digit": 6,
        "ordered": 6,
        "number_theoretic": 6,
        "compositional": 6,
    }
    assert not any(diagnostics["rejected"].values())


def test_parse_response_rejects_observation_memorization():
    payload = json.loads(_valid_response())
    payload["families"]["ordered"][0] = {
        "name": "memorized",
        "expression": "n == 10",
    }

    hypotheses, diagnostics = grid.parse_response(
        json.dumps(payload),
        observations=((10, True),),
    )

    assert len(hypotheses) < 24
    assert diagnostics["rejected"]["observation_memorization"] == 1


def test_parse_response_rejects_wrong_family_and_duplicate_behavior():
    payload = json.loads(_valid_response())
    payload["families"]["ordered"][0] = {
        "name": "misfiled",
        "expression": "divisible(n, 11)",
    }
    payload["families"]["periodic_digit"][1] = {
        "name": "duplicate_even",
        "expression": "n % 2 == 0",
    }

    _, diagnostics = grid.parse_response(json.dumps(payload))

    assert diagnostics["rejected"]["wrong_family"] == 1
    assert diagnostics["rejected"]["duplicate_extension"] == 1


def test_response_schema_requires_exact_family_grid():
    schema = grid.response_format()["json_schema"]["schema"]

    assert schema["required"] == ["families"]
    family_schema = schema["properties"]["families"]
    assert family_schema["required"] == list(grid.FAMILIES)
    assert all(
        family_schema["properties"][family]["minItems"] == 6
        and family_schema["properties"][family]["maxItems"] == 6
        for family in grid.FAMILIES
    )
