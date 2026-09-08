import json

import pytest
import yaml

from scripts.scilaws_public_metadata_audit import (
    metadata_path,
    numeric_range,
    project,
    read_metadata,
)


def fixture():
    row = dict(
        task_id="fixture",
        group_structure="single",
        n_inputs="1",
        discipline="fixture",
        license="fixture",
    )
    meta = dict(
        task_id="fixture",
        target={"name": "y", "range": {"test": [9001, 9002]}},
        inputs=[{"name": "x", "range": {"train": [0, 1], "test": [9003, 9004]}}],
        priors=["PRIVATE_MARKER"],
        formula_source="PRIVATE_MARKER",
        n_test=9005,
    )
    return row, meta


def test_projection_excludes_test_summaries_and_unapproved_fields():
    row, meta = fixture()
    result = project(row, yaml.safe_dump(meta).encode())
    text = json.dumps(result)
    for token in (
        "9001",
        "9002",
        "9003",
        "9004",
        "9005",
        "PRIVATE_MARKER",
        "formula_source",
    ):
        assert token not in text
    assert result["feasibility_candidate"]
    assert not result["measurement_support_verified"]


@pytest.mark.parametrize(
    "value",
    [[0, float("nan")], [0, float("inf")], [True, 2], [1, 1], [2, 1], ["0", "1"], None],
)
def test_range_validation(value):
    assert not numeric_range(value)


def test_multigroup_requires_group_support():
    row, meta = fixture()
    row["group_structure"] = "multi"
    result = project(row, yaml.safe_dump(meta).encode())
    assert not result["feasibility_candidate"]


def test_aggregate_range_is_recorded_as_unverified_not_reinterpreted():
    row, meta = fixture()
    meta["inputs"][0]["range"] = [0, 1]
    result = project(row, yaml.safe_dump(meta).encode())
    assert not result["feasibility_candidate"]
    assert result["inputs"][0]["public_train_range"] is None


@pytest.mark.parametrize(
    "path",
    [
        "tasks/typeI/a/data/test.csv",
        "tasks/typeI/a/simulator/state.joblib",
        "tasks/typeI/../metadata.yaml",
    ],
)
def test_nonmetadata_paths_rejected_before_network(path):
    with pytest.raises(ValueError, match="only public"):
        read_metadata(path)


def test_identity_and_path_fail_closed():
    row, meta = fixture()
    row["task_id"] = "../fixture"
    with pytest.raises(ValueError):
        metadata_path(row)
    with pytest.raises(ValueError):
        project(row, yaml.safe_dump(meta).encode())
