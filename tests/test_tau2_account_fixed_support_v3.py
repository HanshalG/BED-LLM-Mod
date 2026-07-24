from __future__ import annotations

from pathlib import Path

from helpers import load_config
from scripts.tau2_account_fixed_support_v3 import (
    ZERO_INFORMATION_ROOTS,
    add_observational_equivalence_gates,
)


ROOT = Path(__file__).resolve().parents[1]


def _payload(value: float, network: float) -> dict:
    actions = {
        action: {"predicted_d1_information": value}
        for action in ZERO_INFORMATION_ROOTS
    }
    actions["network_status"] = {
        "predicted_d1_information": network
    }
    return {
        "schema_version": 2,
        "status": "passed",
        "protocol": {},
        "summary": {"gates": {"base": True, "all_pass": True}},
        "records": [{"actions_by_id": actions}],
    }


def test_equivalence_gates_accept_physical_observation_model() -> None:
    payload = add_observational_equivalence_gates(_payload(0.0, 0.5))
    assert payload["status"] == "passed"
    assert payload["summary"]["gates"][
        "all_zero_information_roots_preserve_equivalence"
    ]
    assert payload["summary"]["gates"]["network_status_predicted_informative"]


def test_equivalence_gates_reject_latent_diagnosis_leak() -> None:
    payload = add_observational_equivalence_gates(_payload(0.3, 0.5))
    assert payload["status"] == "gate_failed"
    assert not payload["summary"]["gates"][
        "all_zero_information_roots_preserve_equivalence"
    ]


def test_gpt54_openrouter_config_disables_reasoning_without_thinking_flag() -> None:
    config = load_config(
        str(ROOT / "configs" / "config_tau2_account_fixed_v3_openrouter.yaml")
    )
    spec = config.model_pairs[0].questioner
    assert spec.model == "openai/gpt-5.4"
    assert spec.backend == "openrouter"
    assert spec.thinking is None
    assert spec.reasoning_effort is None
    assert spec.reasoning_max_tokens is None
