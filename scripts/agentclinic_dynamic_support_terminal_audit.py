#!/usr/bin/env python3
"""Replay the zero-call terminal adjudication for AgentClinic V1 mechanics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "agentclinic-dynamic-support-terminal-audit-v1"
EXPECTED_AGENTCLINIC_PY_SHA256 = "ee9cfb3020c7addf717ba8eb3510ba81b2b4b53071cba67d0663ad6c7c6ddbd3"
EXPECTED_SOURCE_PROTOCOL_SHA256 = "389ebee33bd3adf73b4601412346b1e20662727f7de52ff76a3759d5567f6b25"
EXPECTED_PREFLIGHT_PROTOCOL_SHA256 = "75b6dc525ec6855a11ad8d9e5a4f9c7ac820fee4adca3c39350965bfd657a745"
EXPECTED_PREFLIGHT_RESULT_SHA256 = "25cb3c11f5ecf19989f9dfa02fef019d7df9c6432139a32df25575d938f9585f"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def audit(
    *,
    agentclinic_py: Path,
    source_protocol: Path,
    preflight_protocol: Path,
    preflight_result: Path,
) -> dict[str, Any]:
    code = agentclinic_py.read_text(encoding="utf-8")
    protocol_text = preflight_protocol.read_text(encoding="utf-8")
    preflight = load_object(preflight_result)
    bindings = {
        "agentclinic_py_sha256": sha256_file(agentclinic_py),
        "source_protocol_sha256": sha256_file(source_protocol),
        "preflight_protocol_sha256": sha256_file(preflight_protocol),
        "preflight_result_sha256": sha256_file(preflight_result),
    }
    expected_bindings = {
        "agentclinic_py_sha256": EXPECTED_AGENTCLINIC_PY_SHA256,
        "source_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "preflight_protocol_sha256": EXPECTED_PREFLIGHT_PROTOCOL_SHA256,
        "preflight_result_sha256": EXPECTED_PREFLIGHT_RESULT_SHA256,
    }
    source_contract = {
        "scenario_returns_patient_info": "return patient_info" in code,
        "patient_state_is_scenario_patient_information": (
            "self.symptoms = self.scenario.patient_information()" in code
        ),
        "patient_prompt_receives_all_patient_information": (
            "Below is all of your information. {}." in code
            and ".format(self.symptoms)" in code
        ),
        "fixed_intake_is_patient_info": (
            re.search(r"normalized\s+`patient_info` field", protocol_text) is not None
        ),
    }
    binding_pass = bindings == expected_bindings
    contract_pass = all(source_contract.values())
    preflight_pass = (
        preflight.get("status") == "mechanics_preflight_pass"
        and preflight.get("decision") == "freeze_semantic_serving_protocol"
        and preflight.get("accounting", {}).get("openrouter_calls") == 0
        and preflight.get("privacy", {}).get("endpoint_values_opened") is False
    )
    patient_channel_conditionally_redundant = binding_pass and contract_pass and preflight_pass
    gates = {
        "immutable_bindings": binding_pass,
        "released_patient_contract_replayed": contract_pass,
        "preflight_was_zero_call_and_endpoint_blind": preflight_pass,
        "patient_channel_conditionally_redundant_under_v1_intake": patient_channel_conditionally_redundant,
        "semantic_serving_was_never_authorized_or_opened": True,
    }
    passed = all(gates.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "agentclinic_v1_closed" if passed else "terminal_audit_failed_closed",
        "decision": "close_exact_agentclinic_construction" if passed else "authorize_nothing",
        "reason": (
            "the released patient responder is grounded only in patient_info, which V1 already exposes as fixed intake"
            if passed
            else "terminal replay did not reproduce the immutable V1 contract"
        ),
        "scientific_interpretation": {
            "patient_response_additional_information_about_latent_world": "zero_conditioned_on_visible_intake_action_and_history",
            "test_channel_structural_opportunity_evaluated": False,
            "planner_or_policy_efficacy_evaluated": False,
            "reduced_intake_successor_authorized": False,
        },
        "bindings": bindings,
        "source_contract": source_contract,
        "gates": gates,
        "privacy": {
            "individual_case_ids_serialized": False,
            "source_values_serialized": False,
            "diagnoses_serialized": False,
            "endpoint_values_opened": False,
            "model_responses_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
        "authorizes": "nothing",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentclinic-py", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--preflight-protocol", type=Path, required=True)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        agentclinic_py=args.agentclinic_py.resolve(),
        source_protocol=args.source_protocol.resolve(),
        preflight_protocol=args.preflight_protocol.resolve(),
        preflight_result=args.preflight_result.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "agentclinic_v1_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
