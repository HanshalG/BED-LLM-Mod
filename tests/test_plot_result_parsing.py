from pathlib import Path

import numpy as np
import pytest

# These tests cover plotting helpers that may not be present in every checkout.
# Skip the whole module cleanly when the supporting modules are unavailable
# rather than failing collection.
parse_gemma_result_file = pytest.importorskip(
    "plots.plot_gemma_e4b_results",
    reason="plots.plot_gemma_e4b_results is not present in this checkout",
).parse_result_file
plot_results_module = pytest.importorskip(
    "plots.plot_results",
    reason="plots.plot_results is not importable in this checkout",
)
discover_results = plot_results_module.discover_results
parse_result_file = plot_results_module.parse_result_file


def test_parse_result_file_accepts_new_belief_state_mode_suffix():
    parsed = parse_result_file(
        Path("results/run123_EIG_Q:Qwen_Qwen3.5-4B,A:google_gemma-4-E4B-it_categorical_0_animals.npy")
    )

    assert parsed is not None
    assert parsed.belief_state_mode == "categorical"
    assert parsed.version == "0"


def test_parse_result_file_remains_backward_compatible_without_belief_state_mode_suffix():
    parsed = parse_result_file(
        Path("results/run123_EIG_Q:Qwen_Qwen3.5-4B,A:google_gemma-4-E4B-it_0_animals.npy")
    )

    assert parsed is not None
    assert parsed.belief_state_mode is None
    assert parsed.version == "0"


def test_parse_gemma_result_file_accepts_new_belief_state_mode_suffix():
    parsed = parse_gemma_result_file(
        Path("results/run123_EIG_Q:google_gemma-4-E4B-it,A:google_gemma-4-E4B-it_uniform_3_animals.npy")
    )

    assert parsed is not None
    assert parsed.belief_state_mode == "uniform"
    assert parsed.version == "3"


def test_discover_results_reads_new_run_item_metadata(tmp_path):
    run_dir = tmp_path / "runs" / "run123_config"
    item_dir = run_dir / "items" / "000_EIG"
    item_dir.mkdir(parents=True)
    np.save(item_dir / "accuracy.npy", np.array([1.0, 0.0]))
    (item_dir / "metadata.json").write_text(
        """
{
  "artifacts": {
    "accuracy": "items/000_EIG/accuracy.npy",
    "correct_belief_mass": "items/000_EIG/correct_belief_mass.npy"
  },
  "answerer": {
    "model": "google/gemma-4-E4B-it",
    "thinking": true
  },
  "belief_state_mode": "categorical",
  "item_id": "000_EIG",
  "method": "EIG",
  "questioner": {
    "model": "Qwen/Qwen3.5-4B",
    "thinking": false
  },
  "search_depth": 1,
  "task": "animals",
  "version": 0
}
""".strip(),
        encoding="utf-8",
    )

    grouped = discover_results(run_dir)

    pair_key = ("Qwen_Qwen3.5-4B__thinking-off", "google_gemma-4-E4B-it__thinking-on")
    assert grouped[pair_key]["EIG"]["0"] == item_dir / "accuracy.npy"


def test_discover_results_ignores_invalid_run_items(tmp_path):
    run_dir = tmp_path / "runs" / "run123_config"
    invalid_item_dir = run_dir / "items" / "000_EIG"
    valid_item_dir = run_dir / "items" / "001_naive"
    invalid_item_dir.mkdir(parents=True)
    valid_item_dir.mkdir(parents=True)
    (invalid_item_dir / "metadata.json").write_text("{not-json", encoding="utf-8")
    np.save(valid_item_dir / "accuracy.npy", np.array([0.0, 1.0]))
    (valid_item_dir / "metadata.json").write_text(
        """
{
  "artifacts": {"accuracy": "items/001_naive/accuracy.npy"},
  "answerer": {"model": "meta/llama-3"},
  "method": "naive",
  "questioner": {"model": "meta/llama-3"},
  "task": "animals",
  "version": 1
}
""".strip(),
        encoding="utf-8",
    )

    grouped = discover_results(run_dir)

    assert list(grouped.keys()) == [("meta_llama-3", "meta_llama-3")]
    assert grouped[("meta_llama-3", "meta_llama-3")]["naive"]["1"] == valid_item_dir / "accuracy.npy"
