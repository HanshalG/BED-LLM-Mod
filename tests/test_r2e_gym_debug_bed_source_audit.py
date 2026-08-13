from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r2e_gym_debug_bed_source_audit.py"
SPEC = importlib.util.spec_from_file_location("r2e_source", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_projection_is_metadata_only() -> None:
    forbidden = {
        "problem_statement", "prompt", "parsed_commit_content", "modified_files",
        "modified_entity_summaries", "expected_output_json",
        "execution_result_content", "relevant_files",
    }
    assert not (forbidden & set(MODULE.ALLOWED_COLUMNS))
    tree = ast.parse(SCRIPT.read_text())
    read_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "parquet"
    ]
    assert len(read_calls) == 1
    columns = next(keyword for keyword in read_calls[0].keywords if keyword.arg == "columns")
    assert isinstance(columns.value, ast.Call)
    assert isinstance(columns.value.func, ast.Name)
    assert columns.value.func.id == "list"


def test_split_constants_match_frozen_protocol() -> None:
    assert MODULE.VERSION == "r2e-gym-debug-bed-source-v1"
    assert MODULE.SALT == "r2e-gym-debug-bed-v1|"
    assert len(MODULE.SHARDS) == 8
    assert len({sha for _, sha, _ in MODULE.SHARDS}) == 8
    assert sum(size for _, _, size in MODULE.SHARDS) == 944255230


def test_public_contract_uses_hashes_not_identifiers() -> None:
    source = SCRIPT.read_text()
    assert '"split_ordered_id_sha256"' in source
    assert '"commit_hashes_serialized": False' in source
    assert '"docker_images_serialized": False' in source
    assert '"task_payload_columns_materialized": False' in source


def test_runtime_bindings_are_content_addressed() -> None:
    assert set(MODULE.RUNTIME_FILES) == {
        "r2e_docker_runtime", "r2e_environment", "r2e_log_parser"
    }
    assert set(MODULE.DEBUG_FILES) == {"debuggym_r2e_adapter", "debuggym_pdb"}
    for _, expected in (*MODULE.RUNTIME_FILES.values(), *MODULE.DEBUG_FILES.values()):
        assert len(expected) == 64


def test_pdb_contract_uses_released_class_name() -> None:
    source = SCRIPT.read_text()
    assert '"class PDBTool" in pdb' in source
    assert '"class Pdb" in pdb' not in source
    assert '"def start_pdb(" in pdb' in source
    assert '"def restart_pdb(" in pdb' in source
    assert '"def interact_with_pdb(" in pdb' in source
