from pathlib import Path

from helpers import load_config
from scripts.tau_knowledge_qwen14b_compact_scorer import (
    INTERFACE_VERSION,
    MODEL_ID,
    THINKING_FINAL_MAX_NEW_TOKENS,
    THINKING_MAX_NEW_TOKENS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qwen14b_compact_config_is_frozen() -> None:
    config = load_config(
        str(
            ROOT
            / "configs/"
            "config_tau_knowledge_qwen14b_compact_scorer_openrouter.yaml"
        )
    )
    spec = config.model_pairs[0].questioner
    assert MODEL_ID == "qwen/qwen3-14b"
    assert INTERFACE_VERSION == "qwen3-14b-thinking-compact-1"
    assert spec.model == MODEL_ID
    assert spec.thinking is True
    assert spec.thinking_max_new_tokens == THINKING_MAX_NEW_TOKENS == 7680
    assert (
        spec.thinking_final_max_new_tokens
        == THINKING_FINAL_MAX_NEW_TOKENS
        == 512
    )
    assert config.openrouter_max_output_tokens == 8192
    assert config.openrouter_concurrency == 256
    assert config.mediq_seed == 24359
