from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/hiddenbench_adaptive_elicitation_source_audit.py"
SPEC = importlib.util.spec_from_file_location("hiddenbench_source", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_bindings_and_split_cover_population() -> None:
    assert MODULE.VERSION == "hiddenbench-adaptive-elicitation-source-v1"
    assert MODULE.SALT == "hiddenbench-adaptive-elicitation-v1|"
    assert sum(MODULE.SPLIT_SIZES.values()) == 65
    assert MODULE.SPLIT_SIZES == {
        "mechanics": 4,
        "opportunity": 12,
        "development": 16,
        "confirmation": 24,
        "reserve": 9,
    }
    assert len(MODULE.BOUND_FILES) == 7
    assert all(len(expected) == 64 for _, expected in MODULE.BOUND_FILES.values())


def test_split_is_deterministic_complete_and_disjoint() -> None:
    tasks = [{"id": index} for index in range(65)]
    first = MODULE.split_tasks(tasks)
    second = MODULE.split_tasks(list(reversed(tasks)))
    assert first == second
    ids = [task["id"] for values in first.values() for task in values]
    assert len(ids) == len(set(ids)) == 65


def test_string_contract_rejects_duplicates_and_blanks() -> None:
    assert MODULE.unique_nonempty_strings(["a", "b", "c"])
    assert not MODULE.unique_nonempty_strings(["a", "a", "c"])
    assert not MODULE.unique_nonempty_strings(["a", " ", "c"])
    assert not MODULE.unique_nonempty_strings("abc")


def test_native_channel_contract_requires_sequential_private_information() -> None:
    valid = "\n".join((
        "num_agents = len(hidden_info)",
        "facts = list(task.shared_information)",
        "facts.append(hidden_info[index])",
        "assigned_hidden = [hidden_info[index]]",
        "previous_messages.append",
        "other.history[-1]['content']",
        "response = agent.chat(prompt)",
    ))
    assert MODULE.simulator_contract(valid)
    assert not MODULE.simulator_contract(valid.replace("facts.append(hidden_info[index])", ""))


def test_public_contract_never_serializes_task_values() -> None:
    source = SCRIPT.read_text()
    for marker in (
        '"task_ids_serialized": False',
        '"descriptions_serialized": False',
        '"facts_serialized": False',
        '"answers_serialized": False',
        '"rationales_serialized": False',
        '"endpoints_opened": False',
    ):
        assert marker in source
    assert '"split_ordered_id_sha256"' in source


def test_real_manifest_contains_no_semantic_source_values() -> None:
    source_root = Path("/tmp/bed-source-audits/hiddenbench")
    manifest_path = ROOT / "results/nonmyopic/hiddenbench_adaptive_elicitation_source/MANIFEST.json"
    if not source_root.exists() or not manifest_path.exists():
        return
    import json

    tasks = json.loads((source_root / "data/benchmark.json").read_text())
    manifest_text = manifest_path.read_text()
    semantic_values = []
    for task in tasks:
        semantic_values.extend((task["name"], task["description"], task["correct_answer"]))
        semantic_values.extend(task["shared_information"])
        semantic_values.extend(task["hidden_information"])
        semantic_values.extend(task["possible_answers"])
        if task.get("rationale"):
            semantic_values.append(task["rationale"])
    assert all(value not in manifest_text for value in semantic_values)
