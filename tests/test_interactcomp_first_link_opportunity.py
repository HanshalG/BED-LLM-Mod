import base64
import hashlib

import pytest

from scripts.interactcomp_first_link_opportunity import (
    Hypothesis,
    _decrypt_field,
    entropy_from_labels,
    normalize_entity,
    parse_classification,
    parse_hypothesis,
    parse_question,
    parse_responder_answer,
    score_task,
)


def _encrypt(value: str, password: str) -> str:
    plain = value.encode()
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(plain) // len(digest)) + digest[: len(plain) % len(digest)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(plain, key))).decode()


def test_decryption_round_trip():
    encrypted = _encrypt("hidden context", "InteractComp")
    assert _decrypt_field(encrypted, "InteractComp") == "hidden context"


def test_strict_text_parsers():
    hypothesis = parse_hypothesis(
        "ENTITY: Example Entity\nPROFILE: Has a distinctive property."
    )
    assert hypothesis.entity == "Example Entity"
    assert parse_question("Was it established before 1990?").endswith("?")
    assert parse_classification("YNUU") == "YNUU"
    assert parse_responder_answer("i don't know") == "U"
    with pytest.raises(ValueError):
        parse_classification("Y N U U")


def test_normalization_and_entropy():
    assert normalize_entity("The Example-Entity!") == "theexampleentity"
    assert entropy_from_labels(["Y", "Y", "N", "N"]) == pytest.approx(
        0.6931471805599453
    )


def test_score_task_keeps_eig_and_external_truth_mass_separate():
    target = "Target Entity"
    initial = [
        Hypothesis("Distractor A", "profile a"),
        Hypothesis("Distractor B", "profile b"),
        Hypothesis("Target Entity", "profile target"),
        Hypothesis("Distractor C", "profile c"),
    ]
    classifications = ["YNNN", "YNNN", "NYYY", "NYYY"]
    questions = ["Q0?", "Q1?", "Q2?", "Q3?"]
    refreshed = [
        [
            Hypothesis("Target Entity", "p"),
            Hypothesis("Target Entity", "p"),
            Hypothesis("Other", "p"),
            Hypothesis("Other", "p"),
        ],
        [Hypothesis("Other", "p")] * 4,
        [Hypothesis("Target Entity", "p")] * 4,
        [Hypothesis("Other", "p")] * 4,
    ]
    record = score_task(
        task_id=1,
        initial=initial,
        classifications=classifications,
        questions=questions,
        true_responses=["Y", "N", "Y", "U"],
        refreshed=refreshed,
        target=target,
    )
    assert record["initial_truth_mass"] == pytest.approx(0.25)
    assert record["selected_root_index"] == 0
    assert record["selected_endpoint"] == pytest.approx(0.5)
    assert record["oracle_root_index"] == 2
    assert record["oracle_endpoint"] == pytest.approx(1.0)
