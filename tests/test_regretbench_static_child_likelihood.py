from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import regretbench_static_child_likelihood as static


def _support() -> dict:
    hypotheses = []
    for index in range(8):
        hypotheses.append(
            {
                "parent_index": index,
                "revision_type": "retained" if index < 4 else "revised",
                "interpretation": f"meaning {index}",
                "final_answer": f"answer {index}",
                "prior_weight": index + 1.0,
                "probability": (index + 1) / 36.0,
                "predicted_replies": [
                    f"OLD-SECRET-{index}-{question}" for question in range(4)
                ],
            }
        )
    return {
        "hypotheses": hypotheses,
        "questions": [f"Facet {index}?" for index in range(4)],
        "diagnostic": {
            "retained_count": 4,
            "parent_index_permutation_exact": True,
        },
    }


def _raw() -> str:
    return json.dumps(
        {
            "particles": [
                {
                    "particle_index": index,
                    "predicted_replies": [
                        f"STATIC-{index}-{question}" for question in range(4)
                    ],
                }
                for index in range(8)
            ]
        }
    )


def test_request_excludes_history_weights_lineage_and_old_replies(monkeypatch) -> None:
    monkeypatch.setattr(
        static.primary,
        "public_payload",
        lambda cig, history: {
            "task_id": cig.cig_id,
            "ambiguous_prompt": "public prompt",
            "visible_dialogue": list(history),
        },
    )
    monkeypatch.setattr(
        static.primary,
        "privacy_audit",
        lambda cig, payload: {"passed": payload["visible_dialogue"] == []},
    )
    support = _support()
    messages, audit = static.messages_for(SimpleNamespace(cig_id="task"), support)
    payload = json.loads(messages[1]["content"])
    serialized = messages[1]["content"]
    assert payload["visible_dialogue"] == []
    assert audit["passed"] is True
    assert audit["dialogue_excluded"] is True
    assert audit["particle_probabilities_excluded"] is True
    assert audit["lineage_metadata_excluded"] is True
    assert audit["updated_predicted_replies_excluded"] is True
    assert all("probability" not in row and "prior_weight" not in row for row in payload["child_particles"])
    assert all("revision_type" not in row for row in payload["child_particles"])
    assert all(reply not in serialized for row in support["hypotheses"] for reply in row["predicted_replies"])


def test_common_reply_text_in_public_prompt_does_not_fail_structural_exclusion(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        static.primary,
        "public_payload",
        lambda cig, history: {
            "task_id": cig.cig_id,
            "ambiguous_prompt": "The user may answer yes.",
            "visible_dialogue": list(history),
        },
    )
    monkeypatch.setattr(
        static.primary, "privacy_audit", lambda cig, payload: {"passed": True}
    )
    support = _support()
    support["hypotheses"][0]["predicted_replies"][0] = "yes"
    _, audit = static.messages_for(SimpleNamespace(cig_id="task"), support)
    assert audit["passed"] is True
    assert audit["updated_predicted_replies_excluded"] is True


def test_parser_replaces_only_reply_matrix() -> None:
    source = _support()
    expected = copy.deepcopy(source)
    output = static.parse_annotation(_raw(), source)
    for index, (before, after) in enumerate(
        zip(expected["hypotheses"], output["hypotheses"], strict=True)
    ):
        assert after["predicted_replies"] == [
            f"STATIC-{index}-{question}" for question in range(4)
        ]
        assert {key: value for key, value in after.items() if key != "predicted_replies"} == {
            key: value for key, value in before.items() if key != "predicted_replies"
        }
    assert output["questions"] == expected["questions"]
    assert output["diagnostic"]["retained_count"] == 4
    assert output["diagnostic"]["history_conditioned_child_particles_preserved"] is True
    assert output["diagnostic"]["updated_likelihood_replies_discarded"] is True
    assert output["support_sha256"] != output["diagnostic"]["source_child_support_sha256"]


def test_parser_requires_exact_particle_permutation() -> None:
    value = json.loads(_raw())
    value["particles"][7]["particle_index"] = 6
    with pytest.raises(ValueError, match="exact permutation"):
        static.parse_annotation(json.dumps(value), _support())


def test_structural_hash_ignores_old_replies_weights_and_lineage() -> None:
    first = _support()
    second = copy.deepcopy(first)
    for row in second["hypotheses"]:
        row["predicted_replies"] = ["changed"] * 4
        row["probability"] = 0.125
        row["prior_weight"] = 99.0
        row["revision_type"] = "revised"
    first_particles = static._public_child_particles(first)
    second_particles = static._public_child_particles(second)
    assert static.structural_child_sha256(first_particles, first["questions"]) == static.structural_child_sha256(second_particles, second["questions"])


def test_response_schema_is_strict_and_exact() -> None:
    schema = static.response_format()["json_schema"]
    assert schema["strict"] is True
    particle = schema["schema"]["properties"]["particles"]
    assert particle["minItems"] == particle["maxItems"] == 8
    replies = particle["items"]["properties"]["predicted_replies"]
    assert replies["minItems"] == replies["maxItems"] == 4


def test_zero_call_core_binding_matches_sources() -> None:
    root = Path(static.__file__).resolve().parents[1]
    binding = json.loads(
        (
            root
            / "results/nonmyopic/regretbench_factorized_static_likelihood/CORE_BINDING.json"
        ).read_text()
    )
    assert binding["status"] == (
        "zero_call_core_frozen_before_any_regretbench_policy_response"
    )
    assert binding["paid_calls_authorized"] == 0
    for row in ("opportunity_audit", "core"):
        path = root / binding[row]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding[row]["sha256"]
