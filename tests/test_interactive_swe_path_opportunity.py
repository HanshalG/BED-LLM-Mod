from scripts.interactive_swe_path_opportunity import (
    DEVELOPMENT_SIZE,
    EXCLUDED_IDS,
    HOLDOUT_SIZE,
    OPPORTUNITY_SIZE,
    ROOT_PROBES,
    SOURCE_ROWS,
    build_target_blind_tree,
    hidden_gold_tokens,
    score_tree,
    split_ids,
)


def test_split_ids_reproduces_and_excludes_preview() -> None:
    values = list(EXCLUDED_IDS) + [
        f"repo__task-{index:03d}"
        for index in range(SOURCE_ROWS - len(EXCLUDED_IDS))
    ]
    # Frozen hashes intentionally reject a different ID universe.
    try:
        split_ids(values)
    except ValueError as exc:
        assert "hashes changed" in str(exc)
    else:
        raise AssertionError("different ID universe should fail closed")


def test_tree_is_target_blind_and_deduplicates_root_answers() -> None:
    original = (
        "Expected behavior is a stable scalar result.\n\n"
        "Actual failure raises ValueError in parse_value.\n\n"
        "Reproduce with parse_value('x') on version 3.\n\n"
        "The implementation is in parser/core.py."
    )
    chunks, roots = build_target_blind_tree("Parsing gives the wrong result.", original)

    assert chunks
    assert roots
    assert len({root.answer_index for root in roots}) == len(roots)
    assert all(root.probe in ROOT_PROBES for root in roots)
    assert all(root.answer_hash for root in roots)


def test_single_chunk_tree_has_no_continuation() -> None:
    chunks, roots = build_target_blind_tree(
        "A short underspecified issue.",
        "The only hidden detail is that parse_value should preserve empty input.",
    )

    assert len(chunks) == 1
    assert len(roots) == 1
    assert roots[0].continuation_indices == ()
    assert roots[0].continuation_query_hashes == ()


def test_hidden_gold_tokens_excludes_visible_vocabulary() -> None:
    target = hidden_gold_tokens(
        "Parser returns the wrong value.",
        "Use parse_value and raise ValueError from parser/core.py.",
        "+def parse_value(value):\n+    raise ValueError(value)",
        "",
        '["parser/core.py"]',
    )

    assert "parse_value" in target
    assert "valueerror" in target
    assert "core" in target
    assert "parser" not in target
    assert "value" not in target


def test_score_tree_can_prefer_delayed_root() -> None:
    from scripts.interactive_swe_path_opportunity import RootBranch

    chunks = [
        "alpha beta beta",
        "gamma",
        "delta epsilon zeta",
        "theta",
    ]
    roots = [
        RootBranch(0, "root zero", 0, "a", (1,), ("q0",)),
        RootBranch(1, "root one", 1, "b", (2,), ("q1",)),
    ]
    scored = score_tree(
        chunks,
        roots,
        {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"},
    )

    assert scored["greedy"]["root_index"] == 0
    assert scored["depth_two"]["root_index"] == 1
    assert (
        scored["depth_two"]["best_final_coverage"]
        > scored["greedy"]["best_final_coverage"]
    )


def test_frozen_split_sizes_sum_to_source() -> None:
    assert (
        len(EXCLUDED_IDS)
        + OPPORTUNITY_SIZE
        + DEVELOPMENT_SIZE
        + HOLDOUT_SIZE
        == SOURCE_ROWS
    )
