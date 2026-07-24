import math

import pytest

from scripts.detective_cabed_depth_ranking import (
    Belief,
    bayes_update,
    immediate_eig,
    parse_answer,
    parse_likelihoods,
    parse_questions,
    likelihood_prompt,
    question_prompt,
    answer_prompt,
    spearman,
)


HYPOTHESES = (
    "Professor Ada Stone",
    "Dr. Ben Lake",
    "Clara North",
    "Henry West",
)


def test_eig_and_bayes_update_match_manual_binary_calculation():
    belief = Belief.uniform(HYPOTHESES)
    likelihoods = (0.8, 0.6, 0.2, 0.1)

    posterior, marginal = bayes_update(belief, likelihoods, "Yes")

    assert marginal == pytest.approx(0.425)
    assert posterior.probabilities == pytest.approx(
        tuple(value / 1.7 for value in likelihoods)
    )
    manual = (
        -marginal * math.log(marginal)
        - (1.0 - marginal) * math.log(1.0 - marginal)
        - sum(
            0.25
            * (
                -value * math.log(value)
                - (1.0 - value) * math.log(1.0 - value)
            )
            for value in likelihoods
        )
    )
    assert immediate_eig(belief, likelihoods) == pytest.approx(manual)


def test_parse_questions_requires_exact_targets_and_unique_history():
    response = "\n".join(
        [
            "Reasoning.",
            "##Question##: [Target: Professor Ada Stone] Did you enter the library?",
            "##Question##: [Target: Dr. Ben Lake] Did you see the weapon?",
            "##Question##: [Target: Clara North] Did you hear an argument?",
        ]
    )
    parsed = parse_questions(
        response,
        width=3,
        hypotheses=HYPOTHESES,
        history=[],
    )
    assert len(parsed) == 3

    with pytest.raises(ValueError, match="required 3"):
        parse_questions(
            response,
            width=3,
            hypotheses=HYPOTHESES,
            history=[(parsed[0], "Yes")],
        )


def test_parse_likelihoods_aligns_rows_and_applies_cabed_smoothing():
    response = "\n".join(
        [
            "##Clara North##: 0.25",
            "##Professor Ada Stone##: 1",
            "##Henry West##: 0",
            "##Dr. Ben Lake##: 0.5",
        ]
    )
    parsed = parse_likelihoods(response, HYPOTHESES, confidence=0.7)
    assert parsed == pytest.approx((0.85, 0.5, 0.325, 0.15))

    with pytest.raises(ValueError, match="missing rows"):
        parse_likelihoods("##Professor Ada Stone##: 0.5", HYPOTHESES)

    duplicate = response + "\n##Clara North##: 0.4"
    with pytest.raises(ValueError, match="duplicates"):
        parse_likelihoods(duplicate, HYPOTHESES)


def test_answer_and_spearman_parsers_are_strict():
    assert parse_answer("##Answer##: 'Yes'") == "Yes"
    assert parse_answer("##Answer##: No") == "No"
    with pytest.raises(ValueError, match="exactly one"):
        parse_answer("Yes")

    assert spearman((1.0, 2.0, 3.0), (10.0, 20.0, 30.0)) == pytest.approx(1.0)
    assert spearman((1.0, 2.0, 3.0), (30.0, 20.0, 10.0)) == pytest.approx(-1.0)
    assert spearman((1.0, 1.0, 1.0), (1.0, 2.0, 3.0)) is None


def test_concise_prompts_forbid_explanatory_output():
    case = {
        "time": "Evening",
        "location": "Library",
        "victim": {
            "name": "Victim",
            "introduction": "A historian.",
            "cause_of_death": "Blunt force trauma",
            "murder_weapon": "Statue",
        },
        "suspects": [
            {
                "name": hypothesis,
                "introduction": "A suspect",
                "reason_at_scene": "Invited",
                "testimony": "Saw nothing",
            }
            for hypothesis in HYPOTHESES
        ],
    }
    belief = Belief.uniform(HYPOTHESES)
    questions = question_prompt(
        case,
        [],
        belief,
        3,
        prompt_style="concise",
    )
    likelihoods = likelihood_prompt(
        case,
        "[Target: Clara North] Did you enter the library?",
        HYPOTHESES,
        prompt_style="concise",
    )
    assert "Return no explanation" in questions
    assert "Return exactly 3 rows" in questions
    assert "Return no explanation" in likelihoods
    assert "Return exactly 4 rows" in likelihoods


def test_counterfactual_answer_prompt_matches_true_role_and_can_flip_role():
    case = {
        "time": "Evening",
        "location": "Library",
        "victim": {
            "name": "Victim",
            "introduction": "A historian.",
            "cause_of_death": "Blunt force trauma",
            "murder_weapon": "Statue",
        },
        "suspects": [
            {
                "name": hypothesis,
                "introduction": "A suspect",
                "reason_at_scene": "Invited",
                "testimony": "Saw nothing",
                "story": "Private story",
                "is_murderer": hypothesis == HYPOTHESES[0],
            }
            for hypothesis in HYPOTHESES
        ],
    }
    question = "[Target: Professor Ada Stone] Did you enter the library?"
    actual = answer_prompt(case, question)
    aligned = answer_prompt(
        case,
        question,
        assumed_murderer="Professor Ada Stone",
    )
    counterfactual_innocent = answer_prompt(
        case,
        question,
        assumed_murderer="Dr. Ben Lake",
    )
    assert actual == aligned
    assert "You are the MURDERER" in aligned
    assert "You are INNOCENT" in counterfactual_innocent
