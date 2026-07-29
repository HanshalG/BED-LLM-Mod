from __future__ import annotations

import itertools

from scripts import semantic_norms_object_game_opportunity as audit


def _objects() -> tuple[list[dict[str, str]], list[str]]:
    rows = []
    vocab = []
    for category_index in range(9):
        for item_index in range(5):
            finnish = f"fi_{category_index}_{item_index}"
            english = f"object_{category_index}_{item_index}"
            vocab.append(finnish)
            rows.append(
                {
                    "id": str(len(rows)),
                    "fin_name": finnish,
                    "eng_name": english,
                    "category": f"category_{category_index}",
                    "word_class": "noun",
                    "aaltoprod": finnish,
                    "homonym_fin": "",
                    "homonym_eng": "",
                }
            )
    rows.append(
        {
            "id": "abstract",
            "fin_name": "abstract",
            "eng_name": "abstract",
            "category": "abstract",
            "word_class": "noun",
            "aaltoprod": "abstract",
            "homonym_fin": "",
            "homonym_eng": "",
        }
    )
    vocab.append("abstract")
    return rows, vocab


def test_selection_is_deterministic_stratified_and_excludes_abstract() -> None:
    rows, vocab = _objects()
    eligible = audit.eligible_objects(rows, vocab)
    first = audit.select_object_universe(eligible)
    second = audit.select_object_universe(eligible)

    assert first == second
    assert len(first) == 32
    assert len({item["category"] for item in first}) == 8
    assert all(
        sum(item["category"] == category for item in first) == 4
        for category in {item["category"] for item in first}
    )
    assert all(item["category"] != "abstract" for item in first)


def test_feature_extensions_filter_and_dedupe_exact_memberships() -> None:
    selected = [{"vector_row": index} for index in range(4)]
    columns = (
        (1, 1, 1, 0),
        (1, 1, 1, 0),
        (1, 1, 0, 1),
        (1, 0, 0, 0),
    )
    vectors = {
        row: tuple(float(column[row]) for column in columns)
        for row in range(4)
    }

    extensions, stats = audit.feature_extensions(
        selected,
        vectors,
        min_members=2,
        max_members=3,
    )

    assert len(extensions) == 2
    assert stats == {
        "eligible_feature_columns": 3,
        "duplicate_feature_columns": 1,
    }


def test_exact_planner_never_underperforms_greedy_entropy() -> None:
    extensions = tuple(
        mask
        for mask in range(1, 1 << 5)
        if mask.bit_count() in {2, 3}
    )
    planning = audit.evaluate_planning(extensions, num_queries=5)

    for result in planning.values():
        assert result["entropy_gain_nats"] >= -1e-12
        assert 0.0 <= result["exact_terminal_brier"] <= 1.0
        assert 0.0 <= result["greedy_terminal_brier"] <= 1.0


def test_exact_depth_three_matches_exhaustive_leaf_entropy() -> None:
    extensions = (0b001, 0b010, 0b011, 0b100, 0b111)
    planning = audit.evaluate_planning(extensions, num_queries=3, depths=(3,))
    exact = planning["depth3"]

    best = float("inf")
    for order in itertools.permutations(range(3)):
        signatures = [
            tuple((extension >> query) & 1 for query in order)
            for extension in extensions
        ]
        counts = {
            signature: signatures.count(signature) for signature in set(signatures)
        }
        value = sum(
            count / len(extensions) * audit._terminal_entropy(range(count))
            for count in counts.values()
        )
        best = min(best, value)
    assert abs(exact["exact_terminal_entropy_nats"] - best) < 1e-12


def test_summary_fails_closed_when_opportunity_checks_miss() -> None:
    selected = [
        {
            "source_id": str(index),
            "english_name": f"item_{index}",
            "category": f"category_{index // 4}",
            "vector_row": index,
        }
        for index in range(32)
    ]
    planning = {
        "depth3": {
            "root_differs": False,
            "entropy_gain_nats": 0.0,
            "brier_gain": 0.0,
        }
    }
    result = audit.summarize(
        eligible=selected,
        selected=selected,
        extensions=(0b111,),
        extension_stats={
            "eligible_feature_columns": 1,
            "duplicate_feature_columns": 0,
        },
        planning=planning,
        source_hashes={},
    )

    assert result["status"] == "opportunity_failed"
    assert not result["checks"]["depth3_root_differs"]
    assert not result["checks"]["unique_target_extensions"]
