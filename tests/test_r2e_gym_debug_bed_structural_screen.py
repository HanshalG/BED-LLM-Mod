from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r2e_gym_debug_bed_structural_screen.py"
SPEC = importlib.util.spec_from_file_location("r2e_structural", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_payload_scan_is_predicate_pushed() -> None:
    tree = ast.parse(SCRIPT.read_text())
    scanner_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "scanner"
    ]
    assert len(scanner_calls) == 1
    assert {keyword.arg for keyword in scanner_calls[0].keywords} == {"columns", "filter"}
    assert "isin(selected_ids)" in ast.unparse(scanner_calls[0])


def test_payload_projection_excludes_outcomes_and_text() -> None:
    assert not (MODULE.FORBIDDEN_COLUMNS & set(MODULE.PAYLOAD_COLUMNS))
    assert "commit_hash" in MODULE.PAYLOAD_COLUMNS
    assert "modified_entity_summaries" in MODULE.PAYLOAD_COLUMNS


def test_structural_qualification() -> None:
    row = {
        "parsed_commit_content": '{"file_diffs": [{"path": "src/a.py"}]}',
        "modified_files": ["src/a.py"],
        "modified_entity_summaries": [
            {"type": "function", "file_name": "src/a.py", "name": "alpha", "start_lineno": 2, "end_lineno": 5},
            {"ast_type_str": "FunctionDef", "file_name": "src/a.py", "name": "beta", "start_lineno": 8, "end_lineno": 11},
        ],
    }
    assert MODULE._qualify(row, 2) == (True, 2, 1)
    assert MODULE._qualify(row, 3) == (False, 2, 1)


def test_test_entities_are_excluded() -> None:
    row = {
        "parsed_commit_content": '{"file_diffs": [{"path": "tests/test_a.py"}]}',
        "modified_files": ["tests/test_a.py"],
        "modified_entity_summaries": [
            {"type": "function", "file_name": "tests/test_a.py", "name": "test_alpha", "start_lineno": 2, "end_lineno": 5},
            {"type": "function", "file_name": "tests/test_a.py", "name": "test_beta", "start_lineno": 8, "end_lineno": 11},
        ],
    }
    assert MODULE._qualify(row, 2) == (False, 0, 1)


def test_public_result_contains_no_private_identity_fields() -> None:
    source = SCRIPT.read_text()
    assert '"identifiers_serialized_publicly": False' in source
    assert '"mechanics_ordered_id_sha256"' in source
    assert '"mechanics_ordered_ids"' in source
    assert "private =" in source

