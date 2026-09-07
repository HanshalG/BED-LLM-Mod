import json

import pytest

from scripts.grnbench_steady_state_source_audit import PROTOCOL, audit, implementations


SOURCE = """
def _build_params(family, difficulty, version):
    return {"reporter_scale": 100.0}
def _simulate_chain(p, q):
    return {"A": 1.0, "B": 1.0, "C": 0.5, "R": 0.0}
def _simulate_negative_feedback(p, q):
    c = 0.0
    for _ in range(18):
        c = 1.0 - c
    return {"A": 1.0, "B": 1.0, "C": c, "R": 0.0}
def _simulate_toggle(p, q):
    c = 0.5
    for _ in range(28):
        c = 0.5
    return {"A": 1.0, "B": 1.0, "C": c, "R": 0.0}
_SIMULATORS = {
    "g0_activation_chain": _simulate_chain,
    "g1_coherent_feedforward": _simulate_chain,
    "g2_incoherent_feedforward": _simulate_chain,
    "g3_negative_feedback": _simulate_negative_feedback,
    "g4_toggle_switch": _simulate_toggle,
}
def hidden_world_constructor():
    raise AssertionError("must not execute")
class Oracle:
    raise AssertionError("must not construct")
import package_that_does_not_exist
"""


def test_exact_iteration_changes_and_no_oracle_or_import_execution():
    variants = implementations(
        SOURCE, {"_simulate_negative_feedback": 18, "_simulate_toggle": 28}
    )
    for label, value in [
        ("published", 0.0),
        ("next", 1.0),
        ("long", 0.0),
        ("long_next", 1.0),
    ]:
        namespace = variants[label]
        assert namespace["_simulate_negative_feedback"]({}, {})["C"] == value
        assert "hidden_world_constructor" not in namespace
        assert "Oracle" not in namespace
    assert "range(18)" in SOURCE


def test_complete_fixed_grid_detects_oscillation_without_selecting_cases():
    config = json.loads(PROTOCOL.read_text())
    config.update(difficulties=["easy"], versions=["v0"], expected_cases=165)
    result = audit(SOURCE, config)
    assert result["status"] == "source_steady_state_interpretation_failed"
    assert len(result["rows"]) == 165
    assert all(s["cases"] == 33 for s in result["summary"].values())
    for family, summary in result["summary"].items():
        expected = 33 if family == "g3_negative_feedback" else 0
        assert summary["published_nonfixed"] == expected
        assert summary["long_nonfixed"] == expected
    assert result["policy_efficacy_tested"] is False
    assert result["model_calls"] == 0


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("range(18)", "range(19)"),
        SOURCE.replace("_simulate_negative_feedback", "_changed"),
    ],
)
def test_changed_source_recurrence_rejected(source):
    with pytest.raises(ValueError):
        implementations(
            source, {"_simulate_negative_feedback": 18, "_simulate_toggle": 28}
        )


def test_time_and_coverage_caps():
    config = json.loads(PROTOCOL.read_text())
    config["max_seconds"] = -1
    with pytest.raises(TimeoutError):
        audit(SOURCE, config)
    config.update(
        max_seconds=60, difficulties=["easy"], versions=["v0"], expected_cases=1
    )
    with pytest.raises(ValueError, match="coverage"):
        audit(SOURCE, config)
