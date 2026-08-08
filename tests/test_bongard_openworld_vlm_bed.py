from __future__ import annotations

from io import BytesIO
import json
import math

from PIL import Image

from scripts import bongard_openworld_vlm_bed as bed


def _response(*, history: tuple[tuple[str, bool], ...] = ()) -> str:
    labels = dict(history)
    rows = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        probabilities = []
        for image_index in range(bed.NUM_IMAGES):
            image_id = f"image-{image_index:02d}"
            value = 15 + ((image_index * 7 + hypothesis_index * 9) % 70)
            if image_id in labels:
                value = 85 - hypothesis_index if labels[image_id] else 15 + hypothesis_index
            probabilities.append(value)
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": f"distinct visual rule number {hypothesis_index + 1}",
                "history_weight": 10 + hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": rows})


def _belief(history: tuple[tuple[str, bool], ...] = ()) -> bed.SemanticBelief:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    return bed.parse_belief_response(
        _response(history=history), image_ids=image_ids, history=history
    )


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (40, 30), (100, 120, 140)).save(output, format="PNG")
    return output.getvalue()


def _task() -> bed.VisualTask:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    return bed.VisualTask(
        task_id="task-opaque123456",
        image_ids=image_ids,
        initial_history=(("image-00", True), ("image-01", False)),
        candidate_ids=tuple(f"image-{index:02d}" for index in range(2, 10)),
        endpoint_ids=("image-12", "image-13"),
        image_bytes={image_id: _image_bytes() for image_id in image_ids},
        actual_labels={image_id: index % 2 == 0 for index, image_id in enumerate(image_ids)},
        hidden_values=("secret concept", "images/0123/pos__0__source.jpg"),
    )


def test_parser_uses_history_conditioned_weights_without_double_counting() -> None:
    belief = _belief((("image-00", True), ("image-01", False)))
    assert len(belief.hypotheses) == 10
    expected = bed.normalize_weights(tuple(range(10, 20)))
    assert belief.history_weights == expected
    assert all(weight > 0 for weight in belief.history_weights)
    assert bed.observed_history_fit_log_loss(belief) < math.log(2)


def test_parser_rejects_duplicate_rules_and_probability_shape() -> None:
    value = json.loads(_response())
    value["hypotheses"][1]["rule"] = value["hypotheses"][0]["rule"]
    try:
        bed.parse_belief_response(
            json.dumps(value),
            image_ids=tuple(f"image-{index:02d}" for index in range(14)),
            history=(),
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate rules should fail")

    value = json.loads(_response())
    value["hypotheses"][0]["positive_probabilities"].pop()
    try:
        bed.parse_belief_response(
            json.dumps(value),
            image_ids=tuple(f"image-{index:02d}" for index in range(14)),
            history=(),
        )
    except ValueError as exc:
        assert "fourteen" in str(exc)
    else:
        raise AssertionError("short probability vector should fail")


def test_eig_and_fixed_depth_two_are_finite_and_nonconstant() -> None:
    belief = _belief()
    candidates = tuple(f"image-{index:02d}" for index in range(2, 10))
    myopic = bed.candidate_eigs(belief, candidates)
    fixed = bed.fixed_support_depth_two_scores(belief, candidates)
    assert all(value >= 0 and math.isfinite(value) for value in myopic.values())
    assert all(fixed[key] >= myopic[key] for key in candidates)
    assert len({round(value, 8) for value in myopic.values()}) > 1
    assert bed.select_best(myopic) in candidates


def test_eig_matches_binary_mutual_information_identity() -> None:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    hypotheses = tuple(
        bed.SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            rule=f"rule {index} separates the image",
            history_weight=1.0,
            positive_probabilities=(
                (0.9 if index < 5 else 0.1),
                *([0.5] * 13),
            ),
        )
        for index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS)
    )
    uniform = tuple([0.1] * 10)
    belief = bed.SemanticBelief(
        image_ids=image_ids,
        history=(),
        hypotheses=hypotheses,
        history_weights=uniform,
    )
    binary_entropy_09 = -0.9 * math.log(0.9) - 0.1 * math.log(0.1)
    expected = math.log(2.0) - binary_entropy_09
    assert math.isclose(
        bed.expected_information_gain(belief, "image-00"),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_hypothesis_eig_can_disagree_with_endpoint_predictive_value() -> None:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    query_a = (0.01, 0.99, 0.01, 0.99)
    query_b = (0.25, 0.25, 0.75, 0.75)
    endpoint = (0.10, 0.10, 0.90, 0.90)
    hypotheses = tuple(
        bed.SemanticHypothesis(
            hypothesis_id=f"H{index + 1:02d}",
            rule=f"counterexample rule {index + 1}",
            history_weight=1.0,
            positive_probabilities=(
                query_a[index],
                query_b[index],
                endpoint[index],
                *([0.5] * 11),
            ),
        )
        for index in range(4)
    )
    belief = bed.SemanticBelief(
        image_ids=image_ids,
        history=(),
        hypotheses=hypotheses,
        history_weights=(0.25, 0.25, 0.25, 0.25),
    )

    hypothesis_scores = bed.candidate_eigs(
        belief, ("image-00", "image-01")
    )
    endpoint_scores = bed.candidate_endpoint_eigs(
        belief, ("image-00", "image-01"), ("image-02",)
    )

    assert hypothesis_scores["image-00"] > hypothesis_scores["image-01"]
    assert math.isclose(endpoint_scores["image-00"], 0.0, abs_tol=1e-12)
    assert endpoint_scores["image-01"] > 0.08
    assert bed.select_best(endpoint_scores) == "image-01"


def test_endpoint_depth_two_matches_manual_terminal_entropy() -> None:
    belief = _belief()
    candidates = ("image-02", "image-03", "image-04")
    endpoints = ("image-12", "image-13")
    scores = bed.fixed_support_endpoint_depth_two_scores(
        belief, candidates, endpoints
    )
    root_entropy = bed.endpoint_predictive_entropy(belief, endpoints)
    first = candidates[0]
    probability = bed.predictive_probability(belief, first)
    expected_terminal = 0.0
    for label, outcome_probability in (
        (True, probability),
        (False, 1.0 - probability),
    ):
        weights = bed.updated_weights_for_label(belief, first, label)
        best_terminal = math.inf
        for second in candidates[1:]:
            second_probability = bed.predictive_probability(
                belief, second, weights=weights
            )
            terminal = 0.0
            for second_label, second_outcome_probability in (
                (True, second_probability),
                (False, 1.0 - second_probability),
            ):
                terminal_weights = bed.updated_weights_for_label(
                    belief, second, second_label, weights=weights
                )
                terminal += second_outcome_probability * bed.endpoint_predictive_entropy(
                    belief, endpoints, weights=terminal_weights
                )
            best_terminal = min(best_terminal, terminal)
        expected_terminal += outcome_probability * best_terminal
    assert math.isclose(
        scores[first], root_entropy - expected_terminal, abs_tol=1e-12
    )


def test_dynamic_depth_two_uses_branch_specific_support() -> None:
    root = _belief()
    candidates = tuple(f"image-{index:02d}" for index in range(2, 10))
    branches = {
        (candidate, label): _belief(((candidate, label),))
        for candidate in candidates
        for label in (False, True)
    }
    dynamic = bed.dynamic_support_depth_two_scores(root, candidates, branches)
    assert set(dynamic) == set(candidates)
    assert all(math.isfinite(value) and value >= 0 for value in dynamic.values())

    incomplete = dict(branches)
    incomplete.pop(next(iter(incomplete)))
    try:
        bed.dynamic_support_depth_two_scores(root, candidates, incomplete)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete branch support should fail")

    endpoint_dynamic = bed.dynamic_support_endpoint_depth_two_scores(
        root, candidates, ("image-12", "image-13"), branches
    )
    assert set(endpoint_dynamic) == set(candidates)
    assert all(math.isfinite(value) for value in endpoint_dynamic.values())


def test_history_blind_depth_two_analytically_updates_fresh_root_support() -> None:
    root = _belief()
    candidates = tuple(f"image-{index:02d}" for index in range(2, 10))
    blind = {
        (candidate, label): root
        for candidate in candidates
        for label in (False, True)
    }
    scores = bed.history_blind_depth_two_scores(root, candidates, blind)
    fixed = bed.fixed_support_depth_two_scores(root, candidates)
    assert scores == fixed
    endpoint_scores = bed.history_blind_endpoint_depth_two_scores(
        root, candidates, ("image-12", "image-13"), blind
    )
    endpoint_fixed = bed.fixed_support_endpoint_depth_two_scores(
        root, candidates, ("image-12", "image-13")
    )
    assert endpoint_scores == endpoint_fixed

    invalid = dict(blind)
    invalid[(candidates[0], True)] = _belief(((candidates[0], True),))
    try:
        bed.history_blind_depth_two_scores(root, candidates, invalid)
    except ValueError as exc:
        assert "root history" in str(exc)
    else:
        raise AssertionError("answer-conditioned support cannot enter blind scoring")


def test_multimodal_prompt_contains_only_opaque_interface() -> None:
    task = _task()
    messages = bed.build_belief_messages(task, task.initial_history)
    assert bed.prompt_hidden_state_errors(task, task.initial_history, messages) == []
    request = bed.request_payload(messages)
    assert any(
        "present in positive examples and absent from negative examples" in item
        for item in request["requirements"]
    )
    text = bed.request_text(messages)
    assert "secret concept" not in text
    assert "pos__" not in text
    assert "selectable_image_ids" not in text
    assert "endpoint_image_ids" not in text
    content = messages[0]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 14
    assert all(
        item["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for item in images
    )


def test_endpoint_metrics_use_external_labels() -> None:
    belief = _belief()
    metrics = bed.endpoint_metrics(
        belief, {"image-12": True, "image-13": False}
    )
    assert 0 <= metrics["mean_brier"] <= 1
    assert metrics["mean_log_loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1
    assert len(metrics["rows"]) == 2
