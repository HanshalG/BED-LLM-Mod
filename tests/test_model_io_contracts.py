from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_model_io_contracts_module", ROOT / "model.py")
model = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def test_qwen_thinking_output_uses_standard_think_block():
    adapter = model.QwenVLLMAdapter.__new__(model.QwenVLLMAdapter)
    adapter.model_name = "Qwen/Qwen3.5-0.8B"
    adapter.thinking = True

    output = type("Output", (), {"text": "<think>\nreasoning\n</think>\n\nYes"})()

    assert adapter._normalize_completion_output(output) == "Yes"


def test_gemma_uses_parse_response_content_only():
    adapter = model.GemmaVLLMAdapter.__new__(model.GemmaVLLMAdapter)
    adapter.model_name = "google/gemma-4-e2b-it"
    adapter.tokenizer = type(
        "Tokenizer",
        (),
        {"parse_response": staticmethod(lambda payload: {"content": "Yes", "thinking": "reasoning"})},
    )()

    output = type("Output", (), {"text": "raw", "token_ids": [1, 2, 3]})()

    assert adapter._normalize_completion_output(output) == "Yes"


def test_harmony_can_be_smoke_tested_with_synthetic_completion_tokens():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "test_harmony_parse.py"
    if not script_path.exists():
        pytest.skip(f"harmony smoke-test script not present at {script_path}")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert '"channel": "final"' in completed.stdout
    assert '"text": "{\\"Yes\\": 1}"' in completed.stdout
