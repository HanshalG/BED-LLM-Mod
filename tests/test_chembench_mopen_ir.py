from __future__ import annotations

import pytest

from environments.chembench_mopen.ir import INPUT_NAMES, RateLaw, RateLawError


def _payload(expression: str = "kcat * Enz * C_A / (Km + C_A)") -> dict:
    return {
        "name": "michaelis_menten",
        "expr": expression,
        "params": [
            {"name": "kcat", "low": 0.1, "high": 20.0},
            {"name": "Km", "low": 0.01, "high": 100.0},
        ],
        "rationale": "Saturating substrate response.",
    }


def _inputs() -> dict[str, float]:
    return dict(zip(INPUT_NAMES, (1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0), strict=True))


def test_rate_law_compiles_and_canonicalizes_commutative_terms() -> None:
    law = RateLaw.from_payload(_payload())
    evaluate = law.compile()
    assert evaluate(_inputs(), {"kcat": 5.0, "Km": 1.0}) == pytest.approx(2.5)
    commuted = RateLaw.from_payload(_payload("Enz * kcat * C_A / (Km + C_A)"))
    assert law.canonical_key == commuted.canonical_key
    law.stress_test([_inputs()])


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "C_A.__class__",
        "params[0]",
        "abs(C_A)",
        "unknown * C_A",
        "(lambda x: x)(C_A)",
    ],
)
def test_rate_law_rejects_unsafe_or_undeclared_syntax(expression: str) -> None:
    with pytest.raises(RateLawError):
        RateLaw.from_payload(_payload(expression))


def test_rate_law_rejects_bad_parameters_and_nonfinite_evaluation() -> None:
    payload = _payload()
    payload["params"][0]["low"] = 0.0
    with pytest.raises(RateLawError, match="positive lower bound"):
        RateLaw.from_payload(payload)

    law = RateLaw.from_payload(_payload("exp(kcat * T)"))
    with pytest.raises(RateLawError, match="evaluation failed"):
        law.compile()(_inputs(), {"kcat": 20.0, "Km": 1.0})


def test_stress_grid_rejects_negative_rates() -> None:
    law = RateLaw.from_payload(_payload("-kcat * C_A"))
    with pytest.raises(RateLawError, match="negative rate"):
        law.stress_test([_inputs()])
