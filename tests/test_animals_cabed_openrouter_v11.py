from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.animals.prompts import answer_likelihood_messages
from helpers import load_config
from scripts.animals_cabed_openrouter_v11 import (
    BatchedSemanticOpenRouterModel,
    batch_classification_messages,
    parse_batch_classification,
    parse_likelihood_conversation,
    run_openrouter_stage,
)
from scripts.animals_cabed_aligned_v12 import (
    AlignedSemanticAnimalsEnvironment,
    run_aligned_stage,
)
from scripts.animals_cabed_aligned_v13 import (
    OversampledAlignedAnimalsEnvironment,
    RecordingBatchedSemanticModel,
    run_v13_stage,
)
from scripts.animals_cabed_shared_tree_v10 import DeterministicMechanicsModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config_animals_cabed_openrouter_v11.yaml"


class _Config:
    openrouter_max_output_tokens = 4096


class _Delegate:
    def __init__(self) -> None:
        self.calls = 0

    def chat_complete_messages_batched(
        self,
        messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        self.calls += 1
        outputs = []
        for conversation in messages:
            prompt = conversation[-1]["content"]
            animals = json.loads(
                prompt.split("Animals: ", 1)[1].split("\n", 1)[0]
            )
            outputs.append(
                json.dumps(
                    {
                        "answers": [
                            {
                                "animal": animal,
                                "answer": (
                                    "Yes"
                                    if animal in {"Bat", "Eagle"}
                                    else "No"
                                ),
                            }
                            for animal in animals
                        ]
                    }
                )
            )
        return outputs


class _EndToEndDelegate(DeterministicMechanicsModel):
    def chat_complete_messages_batched(
        self,
        messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        outputs = []
        for conversation in messages:
            prompt = conversation[-1]["content"]
            animals = json.loads(
                prompt.split("Animals: ", 1)[1].split("\n", 1)[0]
            )
            outputs.append(
                json.dumps(
                    {
                        "answers": [
                            {
                                "animal": animal,
                                "answer": (
                                    "Yes"
                                    if hash((animal, prompt)) % 2
                                    else "No"
                                ),
                            }
                            for animal in animals
                        ]
                    }
                )
            )
        return outputs

    def usage_snapshot(self):
        return {
            "adapter_requests": 1,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_parse_likelihood_conversation_extracts_animal_and_question() -> None:
    messages = answer_likelihood_messages(
        "African elephant",
        "Is it larger than a person?",
        ["Yes", "No"],
    )
    assert parse_likelihood_conversation(messages) == (
        "African elephant",
        "Is it larger than a person?",
    )


def test_batch_classification_parser_requires_exact_order() -> None:
    animals = ["Bat", "Whale"]
    payload = {
        "answers": [
            {"animal": "Bat", "answer": "Yes"},
            {"animal": "Whale", "answer": "No"},
        ]
    }
    assert parse_batch_classification(
        json.dumps(payload),
        animals,
    ) == ("Yes", "No")
    payload["answers"].reverse()
    with pytest.raises(ValueError, match="order or name"):
        parse_batch_classification(json.dumps(payload), animals)


def test_batch_prompt_contains_every_animal_once() -> None:
    messages = batch_classification_messages(
        "Can it fly?",
        ["Bat", "Eagle", "Whale"],
    )
    assert messages[-1]["content"].count('"Bat"') == 2
    assert messages[-1]["content"].count('"Eagle"') == 2
    assert messages[-1]["content"].count('"Whale"') == 2


def test_wrapper_batches_rows_by_question_and_restores_input_order() -> None:
    delegate = _Delegate()
    model = BatchedSemanticOpenRouterModel(delegate, _Config())
    messages = [
        answer_likelihood_messages("Bat", "Can it fly?", ["Yes", "No"]),
        answer_likelihood_messages("Whale", "Can it swim?", ["Yes", "No"]),
        answer_likelihood_messages("Whale", "Can it fly?", ["Yes", "No"]),
        answer_likelihood_messages("Eagle", "Can it swim?", ["Yes", "No"]),
    ]
    rows = model.chat_probabilities_messages_batched(
        messages,
        ["Yes", "No"],
        temperature=0.0,
        block_size=128,
    )
    assert delegate.calls == 1
    assert rows == [
        {"Yes": 1.0, "No": 0.0},
        {"Yes": 0.0, "No": 1.0},
        {"Yes": 0.0, "No": 1.0},
        {"Yes": 1.0, "No": 0.0},
    ]
    assert len(model.classification_records) == 2


def test_zero_cost_v11_smoke_composes_with_shared_tree() -> None:
    config = load_config(str(CONFIG_PATH))
    model = BatchedSemanticOpenRouterModel(
        _EndToEndDelegate(),
        config,
    )
    payload = run_openrouter_stage(
        config,
        stage="serving_smoke",
        model=model,
    )
    assert payload["status"] == "passed"
    assert payload["summary"]["num_states"] == 2
    assert payload["summary"]["gates"]["zero_reasoning_tokens"] is True
    assert payload["protocol"]["semantic_batch_count"] > 0


def test_aligned_environment_requires_precomputed_table() -> None:
    config = load_config(str(CONFIG_PATH))
    model = BatchedSemanticOpenRouterModel(
        _EndToEndDelegate(),
        config,
    )
    env = AlignedSemanticAnimalsEnvironment(
        config=config,
        answerer=model,
        target_animals=["Wombat"],
    )
    with pytest.raises(ValueError, match="precomputed target-blind"):
        env.observe("Does it have fur?", "Wombat", None)
    env._likelihood_cache[
        ("wombat", "does it have fur?")
    ] = (0.85, 0.15)
    assert env.observe("Does it have fur?", "Wombat", None) == "Yes"


def test_zero_cost_v12_smoke_uses_aligned_semantic_observations() -> None:
    config = load_config(str(CONFIG_PATH))
    model = BatchedSemanticOpenRouterModel(
        _EndToEndDelegate(),
        config,
    )
    payload = run_aligned_stage(
        config,
        stage="serving_smoke",
        model=model,
    )
    assert payload["status"] == "passed"
    assert payload["summary"]["num_states"] == 2
    assert payload["protocol"]["independent_answer_llm_calls"] is False
    assert payload["summary"]["gates"]["zero_reasoning_tokens"] is True


def test_zero_cost_v13_smoke_oversamples_before_shared_filter() -> None:
    config = load_config(str(CONFIG_PATH))
    model = RecordingBatchedSemanticModel(
        _EndToEndDelegate(),
        config,
    )
    payload = run_v13_stage(
        config,
        stage="serving_smoke",
        model=model,
    )
    assert payload["status"] == "passed"
    assert payload["protocol"]["candidate_oversample"] == 2
    assert payload["protocol"]["requested_followup_candidates"] == 5
    assert payload["generation_records"]
    assert all(
        "Generate up to 6 " in record["messages"][-1]["content"]
        or "Generate up to 5 " in record["messages"][-1]["content"]
        for record in payload["generation_records"]
    )


def test_v13_environment_restores_requested_width() -> None:
    config = load_config(str(CONFIG_PATH))
    model = RecordingBatchedSemanticModel(
        _EndToEndDelegate(),
        config,
    )
    env = OversampledAlignedAnimalsEnvironment(
        config=config,
        answerer=model,
        target_animals=["Wombat"],
    )
    original = config.target_num_questions
    belief = env.initial_belief_state(model, config)
    env.generate_candidate_actions(belief, [], model, config)
    assert config.target_num_questions == original
