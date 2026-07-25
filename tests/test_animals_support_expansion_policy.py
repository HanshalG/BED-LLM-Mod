from __future__ import annotations

import pytest

from core import BeliefState
from environments.animals import questions
from helpers import Config
from scripts import animals_support_expansion_policy as policy
from scripts.animals_cabed_aligned_v13 import RecordingBatchedSemanticModel


def _candidate(
    *,
    p_yes: float,
    yes_size: int,
    no_size: int,
    eig: float,
    retention: float,
    yes_covered: bool,
    no_covered: bool,
    yes_support: list[str] | None = None,
    no_support: list[str] | None = None,
) -> dict[str, object]:
    return {
        "p_yes": p_yes,
        "p_no": 1.0 - p_yes,
        "support_size_if_yes": yes_size,
        "support_size_if_no": no_size,
        "immediate_eig": eig,
        "expected_current_support_retention": retention,
        "truth_covered_if_yes": yes_covered,
        "truth_covered_if_no": no_covered,
        "support_if_yes": yes_support or [f"y{i}" for i in range(yes_size)],
        "support_if_no": no_support or [f"n{i}" for i in range(no_size)],
        "expected_truth_coverage": (
            p_yes * float(yes_covered)
            + (1.0 - p_yes) * float(no_covered)
        ),
    }


def test_expected_support_size_uses_branch_probability():
    candidate = _candidate(
        p_yes=0.25,
        yes_size=20,
        no_size=4,
        eig=0.1,
        retention=0.5,
        yes_covered=True,
        no_covered=False,
    )
    assert policy.expected_support_size(candidate) == pytest.approx(8.0)


def test_confirmation_targets_remain_sealed():
    with pytest.raises(ValueError, match="separate holdout preregistration"):
        policy.stage_targets(
            {
                "development_targets": ["lynx"],
                "holdout_targets": ["otter"],
            },
            "confirmation",
        )


def test_selectors_share_candidates_but_choose_distinct_objectives():
    candidates = [
        _candidate(
            p_yes=0.5,
            yes_size=20,
            no_size=20,
            eig=0.1,
            retention=0.2,
            yes_covered=True,
            no_covered=True,
        ),
        _candidate(
            p_yes=0.5,
            yes_size=4,
            no_size=4,
            eig=0.9,
            retention=0.3,
            yes_covered=False,
            no_covered=False,
        ),
        _candidate(
            p_yes=0.5,
            yes_size=8,
            no_size=8,
            eig=0.2,
            retention=0.8,
            yes_covered=False,
            no_covered=True,
        ),
    ]
    selected = policy.selector_indices(candidates, random_index=1)
    assert selected["support_expansion"] == 0
    assert selected["immediate_eig"] == 1
    assert selected["support_retention"] == 2
    assert selected["random"] == 1


def test_realized_endpoint_uses_only_realized_answer_branch():
    candidate = _candidate(
        p_yes=0.5,
        yes_size=10,
        no_size=5,
        eig=0.2,
        retention=0.2,
        yes_covered=True,
        no_covered=False,
    )
    yes = policy.realized_candidate_endpoint(candidate, "Yes")
    no = policy.realized_candidate_endpoint(candidate, "No")
    assert yes == {
        "truth_covered": 1,
        "support_size": 10,
        "uniform_truth_probability": pytest.approx(0.1),
    }
    assert no == {
        "truth_covered": 0,
        "support_size": 5,
        "uniform_truth_probability": 0.0,
    }


def test_summary_detects_support_expansion_gain(monkeypatch):
    monkeypatch.setitem(
        policy.STAGE_COSTS,
        "serving_smoke",
        (0.1, 1.0),
    )
    candidates = [
        {
            **_candidate(
                p_yes=0.5,
                yes_size=20,
                no_size=20,
                eig=0.1,
                retention=0.3,
                yes_covered=True,
                no_covered=True,
            ),
            "expected_support_size": 20.0,
            "branch_union_size": 40,
            "realized_answer": "Yes",
            "realized_endpoint": {
                "truth_covered": 1,
                "support_size": 20,
                "uniform_truth_probability": 0.05,
            },
        },
        {
            **_candidate(
                p_yes=0.5,
                yes_size=5,
                no_size=5,
                eig=0.9,
                retention=0.2,
                yes_covered=False,
                no_covered=False,
            ),
            "expected_support_size": 5.0,
            "branch_union_size": 10,
            "realized_answer": "No",
            "realized_endpoint": {
                "truth_covered": 0,
                "support_size": 5,
                "uniform_truth_probability": 0.0,
            },
        },
        {
            **_candidate(
                p_yes=0.5,
                yes_size=8,
                no_size=8,
                eig=0.2,
                retention=0.8,
                yes_covered=False,
                no_covered=False,
            ),
            "expected_support_size": 8.0,
            "branch_union_size": 16,
            "realized_answer": "Yes",
            "realized_endpoint": {
                "truth_covered": 0,
                "support_size": 8,
                "uniform_truth_probability": 0.0,
            },
        },
    ]
    record = {
        "truth_covered_before_counterfactuals": False,
        "candidate_dynamics": candidates,
        "selector_indices": {
            "support_expansion": 0,
            "immediate_eig": 1,
            "support_retention": 2,
            "branch_union": 0,
            "random": 1,
        },
    }
    result = policy.summarize(
        [record, record],
        stage="serving_smoke",
        usage={
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.1,
        },
    )
    assert result["support_expansion_vs_eig"][
        "mean_realized_coverage_gain"
    ] == 1.0
    assert result["support_expansion_vs_eig"]["wins_ties_losses"] == [
        2,
        0,
        0,
    ]
    assert result["gates"]["all_pass"] is True


def test_exact_current_support_keeps_production_branch_updates(monkeypatch):
    seen: dict[str, bool] = {}
    belief = BeliefState(["lynx", "otter"], [0.5, 0.5])

    def fake_draw(beliefs, deterministic, num_mc_samples):
        del beliefs, num_mc_samples
        seen["score_deterministically"] = deterministic
        return ["lynx", "otter"], [0.5, 0.5]

    def fake_rows(*args, **kwargs):
        del args, kwargs
        return [
            {"Yes": 0.8, "No": 0.2},
            {"Yes": 0.2, "No": 0.8},
        ]

    def fake_score(*args, **kwargs):
        del args, kwargs
        return [0.2], [0.5], [0.5]

    def fake_future(
        beliefs,
        history,
        question_answers,
        questioner,
        deterministic,
        config,
    ):
        del beliefs, history, question_answers, questioner, config
        seen["branch_deterministic"] = deterministic
        return [belief, belief]

    monkeypatch.setattr(questions, "_draw_belief_samples", fake_draw)
    monkeypatch.setattr(questions, "_answer_probability_rows", fake_rows)
    monkeypatch.setattr(
        questions,
        "_score_questions_from_probability_rows",
        fake_score,
    )
    monkeypatch.setattr(
        questions,
        "_future_beliefs_for_answers_batched",
        fake_future,
    )

    output = questions.evaluate_candidate_coverage_dynamics(
        belief,
        [],
        ["Does it swim?"],
        "otter",
        deterministic=False,
        questioner=object(),
        config=Config(),
        exact_current_support=True,
    )

    assert len(output) == 1
    assert seen == {
        "score_deterministically": True,
        "branch_deterministic": False,
    }


def test_recording_model_captures_batched_generation():
    class Delegate:
        def chat_complete_messages_batched(
            self,
            batch_messages,
            temperature,
            block_size,
            max_new_tokens=None,
        ):
            del temperature, block_size, max_new_tokens
            return ["lynx"] * len(batch_messages)

    model = RecordingBatchedSemanticModel(Delegate(), Config())
    output = model.chat_complete_messages_batched(
        [[{"role": "user", "content": "Generate."}]],
        temperature=0.7,
        block_size=16,
        max_new_tokens=32,
    )

    assert output == ["lynx"]
    assert model.generation_records == [
        {
            "batched": True,
            "messages": [[{"role": "user", "content": "Generate."}]],
            "temperature": 0.7,
            "max_new_tokens": 32,
            "outputs": ["lynx"],
        }
    ]
