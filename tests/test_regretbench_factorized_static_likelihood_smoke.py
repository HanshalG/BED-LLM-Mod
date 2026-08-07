from __future__ import annotations

import json

from scripts import regretbench_factorized_static_likelihood_smoke as smoke
from tests.test_regretbench_deepseek_smc_dynamic_depth2_experiment import (
    _Adapter,
    _install_primary_stage,
)


class _FactorizedAdapter(_Adapter):
    def _static(self, payload: dict) -> str:
        task = self.info[payload["task_id"]]
        return json.dumps(
            {
                "particles": [
                    {
                        "particle_index": row["particle_index"],
                        "predicted_replies": [
                            task["second_reply"]
                            if row["particle_index"] == 0
                            else f"static {row['particle_index']} {question}"
                            for question in range(4)
                        ],
                    }
                    for row in payload["child_particles"]
                ]
            }
        )

    def chat_complete_seeded_messages_batched_structured(
        self,
        messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens,
    ):
        name = response_format["json_schema"]["name"]
        self.requests += len(messages)
        self.seeds.extend(seeds)
        self.schema_names.extend([name] * len(messages))
        responses = []
        for request in messages:
            payload = json.loads(request[1]["content"])
            if name == "regretbench_smc_parent_annotation":
                responses.append(self._annotation(payload))
            elif name == "regretbench_smc_enriched_transition":
                responses.append(self._transition(payload))
            else:
                assert name == "regretbench_static_child_likelihood"
                responses.append(self._static(payload))
        return responses


def test_exact_ten_factorized_smoke(tmp_path, monkeypatch) -> None:
    primary_dir = tmp_path / "primary-smoke"
    info = _install_primary_stage(primary_dir, "smoke")
    adapter = _FactorizedAdapter(info)
    monkeypatch.setattr(
        smoke.transport,
        "validate_support_predecessors",
        lambda **kwargs: {
            "support_smoke_sha256": "smoke",
            "support_development_sha256": "development",
        },
    )
    result = smoke.run_smoke(
        output_dir=tmp_path / "factorized-smoke",
        adapter=adapter,
        primary_smoke_dir=primary_dir,
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        daily_budget_status={"authorized": False},
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["authorizes"] == "separate_factorized_policy_preregistration_only"
    assert result["protocol"]["efficacy_accessed"] is False
    assert result["protocol"]["paid_execution_authorized_by_implementation"] is False
    assert adapter.requests == 10
    assert adapter.schema_names.count("regretbench_smc_parent_annotation") == 2
    assert adapter.schema_names.count("regretbench_smc_enriched_transition") == 4
    assert adapter.schema_names.count("regretbench_static_child_likelihood") == 4
    assert adapter.seeds == [
        smoke.INITIAL_SEED_START,
        smoke.INITIAL_SEED_START + 1,
        smoke.TRANSITION_SEED_START,
        smoke.TRANSITION_SEED_START,
        smoke.TRANSITION_SEED_START + 1,
        smoke.TRANSITION_SEED_START + 1,
        smoke.STATIC_SEED_START,
        smoke.STATIC_SEED_START,
        smoke.STATIC_SEED_START + 1,
        smoke.STATIC_SEED_START + 1,
    ]
    raw = json.loads(
        (tmp_path / "factorized-smoke/private/RAW_RESPONSES.json").read_text()
    )
    assert len(raw["initial"]) == 2
    assert len(raw["transitions"]) == 4
    assert len(raw["static"]) == 4
    privacy = json.loads(
        (tmp_path / "factorized-smoke/private/PRIVACY.json").read_text()
    )["audits"]
    assert len(privacy) == 10
    assert all(row["passed"] for row in privacy)


def test_protocol_rejects_hash_change(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "PROTOCOL_SHA256", "wrong")
    try:
        smoke.validate_protocol()
    except ValueError as error:
        assert "protocol changed" in str(error)
    else:
        raise AssertionError("protocol hash mismatch was accepted")
