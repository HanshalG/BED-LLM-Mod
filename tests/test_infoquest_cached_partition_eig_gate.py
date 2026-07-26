from __future__ import annotations

import json

import pytest

from scripts import infoquest_cached_partition_eig_gate as gate
from scripts import infoquest_partition_eig_causal_gate as partition
from scripts import infoquest_support_causal_link_gate as base


def _fixtures() -> list[base.WorldFixture]:
    return [
        base.WorldFixture(
            fixture_id=f"I{record_id}W{world}",
            record_id=record_id,
            world=world,
            seed_message=f"Ambiguous request {record_id}",
            simulator_system="Answer one specific question.",
            truth_packet={},
            checklist=tuple(f"Checklist {index}" for index in range(5)),
        )
        for record_id in base.MECHANICS_IDS
        for world in (1, 2)
    ]


def _initial(record_id: int) -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(
            (
                f"Record {record_id} hidden context {index} has goal and "
                "constraint."
            )
            for index in range(1, 9)
        ),
        roots=tuple(
            f"What is hidden detail {index}?" for index in range(1, 6)
        ),
    )


def _models(checklist_model=None) -> gate.ModelBundle:
    return gate.ModelBundle(
        generator=gate.CachedDeterministicGenerator("generator"),
        simulator=base.DeterministicFixtureModel("simulator"),
        checklist_judge=(
            checklist_model
            or base.DeterministicFixtureModel("checklist_judge")
        ),
    )


def _initial_response(initial: base.InitialPolicy) -> str:
    return json.dumps(
        {
            **{
                f"h{index}": initial.hypotheses[index - 1]
                for index in range(1, 9)
            },
            **{
                f"q{index}": initial.roots[index - 1]
                for index in range(1, 6)
            },
        }
    )


def test_cached_loader_verifies_hashes_and_shape(tmp_path):
    fixtures = _fixtures()
    raw_path = tmp_path / "raw.json"
    public_path = tmp_path / "public.json"
    raw = {
        "interface_version": gate.discrete_interface_version(),
        "private_fixtures": [
            {"fixture_id": fixture.fixture_id} for fixture in fixtures
        ],
        "initial": [
            _initial_response(_initial(record_id))
            for record_id in base.MECHANICS_IDS
        ],
        "root_answers": ["Answer."] * 30,
    }
    raw_path.write_text(json.dumps(raw, sort_keys=True))
    raw_sha = gate._sha256_path(raw_path)
    public = {
        "protocol": {
            "interface_version": gate.discrete_interface_version(),
            "private_raw_sha256": raw_sha,
        }
    }
    public_path.write_text(json.dumps(public, sort_keys=True))
    public_sha = gate._sha256_path(public_path)

    initials, answers = gate.load_cached_histories(
        fixtures,
        raw_path=raw_path,
        public_path=public_path,
        expected_raw_sha256=raw_sha,
        expected_public_sha256=public_sha,
    )
    assert len(initials) == 3
    assert len(answers) == 6

    with pytest.raises(ValueError, match="private raw"):
        gate.load_cached_histories(
            fixtures,
            raw_path=raw_path,
            public_path=public_path,
            expected_raw_sha256="0" * 64,
            expected_public_sha256=public_sha,
        )


def test_dry_serving_is_exactly_five_calls(tmp_path):
    result = gate.run_serving_gate(
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 5
    assert result["usage"]["http_attempts"] == 5
    assert result["gates"]["all_pass"] is True


def test_dry_mechanics_is_exactly_126_calls_and_passes(tmp_path):
    fixtures = _fixtures()
    initials = {
        record_id: _initial(record_id) for record_id in base.MECHANICS_IDS
    }
    result = gate.run_mechanics_gate(
        fixtures,
        initials,
        [["Answer."] * 5 for _ in fixtures],
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["usage"]["physical_requests"] == 126
    assert result["usage"]["http_attempts"] == 126
    assert result["gates"]["all_pass"] is True


class BadChecklistModel(base.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        self.requests += len(batch_messages)
        return ["malformed"] * len(batch_messages)


def test_serving_checkpoints_malformed_response_before_parse(tmp_path):
    raw_path = tmp_path / "raw.json"
    with pytest.raises(gate.GateExecutionError):
        gate.run_serving_gate(
            _models(BadChecklistModel("checklist_judge")),
            raw_path=raw_path,
        )
    raw = json.loads(raw_path.read_text())
    assert raw["checklist_judgment"] == ["malformed"]
