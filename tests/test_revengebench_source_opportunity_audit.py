from __future__ import annotations

from scripts import revengebench_source_opportunity_audit as audit


def test_python_summary_detects_state_dependent_action_logic() -> None:
    summary = audit.summarize_python(
        """
def move(game_state):
    if game_state.health < 20:
        return 'left'
    if game_state.food:
        return 'right'
    return 'up'
"""
    )

    assert summary.parseable
    assert summary.branch_count == 2
    assert summary.semantically_nontrivial


def test_c_like_lexer_excludes_comments_and_strings() -> None:
    summary = audit.summarize_c_like(
        """
// if (enemy) return move;
const char *message = "if attack state";
int move(struct GameState *state) {
  if (state->energy < 10) return 1;
  if (state->enemy > 0) return 2;
  return 0;
}
""",
        "c",
    )

    assert summary.branch_count == 2
    assert summary.semantically_nontrivial


def test_source_summary_never_contains_source_text() -> None:
    secret = "PRIVATE_SOURCE_LITERAL"
    summary = audit.summarize_python(f"def move(game_state):\n    if game_state.health: return '{secret}'\n    if game_state.food: return 'left'\n")

    assert secret not in audit.canonical_json(summary.__dict__)
