from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clarifybench_source_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "clarifybench_source_audit",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_records_detects_absent_branching_support():
    summary = MODULE.summarize_records(
        [
            (
                "ambiguous",
                "a",
                {
                    "user_intention": "If asked, the user says blue.",
                    "potential_follow_ups": ["Then do the next task."],
                    "ground_truth_tool_calls": [
                        {"tool_name": "paint", "parameters": {"color": "blue"}}
                    ],
                },
            ),
            (
                "explicit",
                "b",
                {
                    "user_intention": "Use blue.",
                    "potential_follow_ups": [],
                    "ground_truth_tool_calls": [
                        {
                            "tool_name": "paint",
                            "parameters": {"color": "blue"},
                            "turn": 1,
                        }
                    ],
                },
            ),
        ]
    )

    assert summary["records"] == 2
    assert summary["class_counts"] == {"ambiguous": 1, "explicit": 1}
    assert summary["records_with_fixed_follow_ups"] == 1
    assert summary["records_with_native_turn_annotations"] == 1
    assert summary["native_turn_annotated_tool_calls"] == 1
    assert summary["records_with_alternative_world_support"] == 0
    assert (
        summary["records_with_question_conditioned_answer_branches"] == 0
    )
    assert summary["records_whose_intention_mentions_if_asked"] == 1


def test_summarize_records_detects_explicit_branching_keys():
    summary = MODULE.summarize_records(
        [
            (
                "ambiguous",
                "a",
                {
                    "hypotheses": ["blue", "green"],
                    "answer_map": {"What color?": {"blue": "Blue"}},
                    "potential_follow_ups": [],
                    "ground_truth_tool_calls": [],
                },
            )
        ]
    )

    assert summary["records_with_alternative_world_support"] == 1
    assert (
        summary["records_with_question_conditioned_answer_branches"] == 1
    )


def test_function_control_flow_helpers():
    source = """
def run_simulation(simulator):
    simulator.get_response_to_question("Which?")
    simulator.current_turn += 1
"""
    assert MODULE.function_method_calls(source, "run_simulation") == {
        "get_response_to_question"
    }
    assert MODULE.function_assigns_attribute(
        source,
        "run_simulation",
        "current_turn",
    )


def test_bed_implementation_pattern_rejects_metric_only_fields():
    implementation_pattern = (
        r"\bclass\s+\w*(?:sage|pomdp)\w*\b"
        r"|\bdef\s+(?:calculate|compute|estimate)_evpi\b"
    )
    source = 'metrics = {"evpi": 0.0}\ndef retrievePrevious():\n    pass\n'
    assert not MODULE.re.search(implementation_pattern, source.casefold())
    assert MODULE.re.search(
        implementation_pattern,
        "def compute_evpi():\n    pass\n",
    )
