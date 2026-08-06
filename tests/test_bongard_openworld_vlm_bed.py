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
                "prior_weight": 10 + hypothesis_index,
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


def test_parser_builds_finite_exact_posterior() -> None:
    belief = _belief((("image-00", True), ("image-01", False)))
    assert len(belief.hypotheses) == 10
    assert math.isclose(sum(belief.prior_weights), 1.0)
    assert math.isclose(sum(belief.posterior_weights), 1.0)
    assert all(weight > 0 for weight in belief.posterior_weights)
    assert bed.prior_history_log_loss(belief) < math.log(2)


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
            prior_weight=1.0,
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
        prior_weights=uniform,
        posterior_weights=uniform,
    )
    binary_entropy_09 = -0.9 * math.log(0.9) - 0.1 * math.log(0.1)
    expected = math.log(2.0) - binary_entropy_09
    assert math.isclose(
        bed.expected_information_gain(belief, "image-00"),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
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


def test_multimodal_prompt_contains_only_opaque_interface() -> None:
    task = _task()
    messages = bed.build_belief_messages(task, task.initial_history)
    assert bed.prompt_hidden_state_errors(task, task.initial_history, messages) == []
    text = bed.request_text(messages)
    assert "secret concept" not in text
    assert "pos__" not in text
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
