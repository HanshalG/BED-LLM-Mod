from __future__ import annotations

import json
from pathlib import Path
import random

from scripts import number_game_atomic_particle_mechanics as mechanics
from scripts import number_game_atomic_particle_verify as verifier
from scripts.number_game_generator_aware_bed import RuleHypothesis, compile_expression


def synthetic_rules() -> list[RuleHypothesis]:
    rows: list[RuleHypothesis] = []
    seen: set[tuple[bool, ...]] = set()
    for modulus in range(5, 32):
        for multiplier in range(1, modulus):
            for offset in range(modulus):
                for threshold in range(1, modulus):
                    expression = f"((n * {multiplier} + {offset}) % {modulus}) < {threshold}"
                    try:
                        extension = compile_expression(expression)
                    except ValueError:
                        continue
                    if extension not in seen:
                        seen.add(extension)
                        rows.append(RuleHypothesis(f"synthetic-{len(rows)}", expression, extension))
                if len(rows) >= 12000:
                    return rows
    return rows


class SyntheticAdapter:
    def __init__(self) -> None:
        self.rules = synthetic_rules()
        self.requests = 0
        self.rows: list[dict] = []

    def complete(self, messages, seeds, *, response_format, max_tokens):
        assert response_format == mechanics.response_format()
        assert max_tokens == mechanics.MAX_TOKENS
        output = []
        for prompt, seed in zip(messages, seeds, strict=True):
            payload = json.loads(prompt[1]["content"])
            history = tuple(
                (row["number"], row["answer"] == "YES")
                for row in payload["observations"]
            )
            compatible = [
                rule
                for rule in self.rules
                if all(rule.extension[number] is answer for number, answer in history)
            ]
            rule = compatible[random.Random(int(seed)).randrange(len(compatible))]
            output.append(json.dumps({"name": rule.name, "expression": rule.expression}))
            prompt_hash = mechanics.hashlib.sha256(
                json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            self.rows.append({
                "seed": int(seed),
                "model_requested": mechanics.MODEL_ID,
                "model_returned": mechanics.MODEL_ID,
                "finish_reasons": ["stop"],
                "prompt_sha256": prompt_hash,
            })
            self.requests += 1
        return output

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "adapter_cost_usd": 0.25,
        }

    def records(self):
        return list(self.rows)


def test_full_synthetic_transaction_preserves_schedule_and_half_banks(tmp_path: Path):
    adapter = SyntheticAdapter()
    result = mechanics.produce_bank(output_dir=tmp_path, adapter=adapter)
    assert len(result["raw"]["responses"]) == len(mechanics.MODEL_SEEDS) == 6400
    assert result["topology"]["request_count"] == 6400
    assert all(result["topology"]["transport"]["gates"].values())
    assert all(
        len(bank["initial_slots"]) == mechanics.INITIAL_SLOTS
        and all(len(values) == mechanics.GENERATED_SLOTS for values in bank["first_conditioned_slots"].values() for values in values.values())
        and all(len(values) == mechanics.GENERATED_SLOTS for values in bank["second_conditioned_slots"].values() for values in values.values())
        for bank in result["banks"]
    )
    assert all(len(bank["half_bank_stability"]["halves"]) == 2 for bank in result["banks"])
    assert result["topology"]["canonical_targets_opened"] is False
    assert result["topology"]["classical_grammar_opened"] is False
    verified = verifier.verify(tmp_path, output=tmp_path / "VERIFICATION.json")
    assert verified["status"] == "verification_pass"


def test_transport_rejects_incomplete_adapter():
    adapter = SyntheticAdapter()
    adapter.requests = 6399
    summary = mechanics.transport_summary(adapter)
    assert summary["gates"]["exact_6400_accepted_and_http"] is False
    assert summary["gates"]["all_clean_stops"] is False


def test_independent_replay_rejects_prompt_identity_tampering(tmp_path: Path):
    adapter = SyntheticAdapter()
    mechanics.produce_bank(output_dir=tmp_path, adapter=adapter)
    raw_path = tmp_path / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    raw["responses"][0]["prompt_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(raw))
    try:
        verifier.verify(tmp_path)
    except RuntimeError as error:
        assert "identity changed" in str(error)
    else:
        raise AssertionError("tampered prompt identity passed replay")


def test_canonical_collapse_receives_maximum_brier():
    even = RuleHypothesis("even", "divisible(n, 2)", compile_expression("divisible(n, 2)"))
    odd = RuleHypothesis("odd", "n % 2 == 1", compile_expression("n % 2 == 1"))
    score, covered, collapsed = mechanics._trajectory_score(None, even, (2,))
    assert (score, covered, collapsed) == (1.0, False, True)
    score, covered, collapsed = mechanics._trajectory_score([odd], even, (2,))
    assert score >= 0.0
    assert covered is False
    assert collapsed is False


def test_scientific_scorer_opens_bound_targets_only_after_bank(monkeypatch, tmp_path: Path):
    produced = mechanics.produce_bank(output_dir=tmp_path, adapter=SyntheticAdapter())
    targets = synthetic_rules()[:33]
    monkeypatch.setattr(mechanics, "canonical_targets", lambda: targets)
    monkeypatch.setattr(
        mechanics,
        "build_classical_grammar_bank",
        lambda: (set(), {"sha256": mechanics.GRAMMAR_SHA256, "unique_nonconstant_extension_count": 0}),
    )
    result = mechanics.score_complete_bank({"banks": produced["banks"]})
    assert result["status"] in {"atomic_particle_mechanics_pass", "atomic_particle_mechanics_null"}
    assert result["targets_opened_after_complete_bank"] is True
    assert result["classical_grammar_opened_after_complete_bank"] is True
    assert "both_half_estimators_agree_on_at_least_3_trees" in result["gates"]
    assert all("particle_collapse_rate" in row for row in result["trees"])
