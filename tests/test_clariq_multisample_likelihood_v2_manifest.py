from scripts.clariq_multisample_likelihood_v2_manifest import _structural_roots


def test_structural_roots_check_keys_without_utility_values() -> None:
    synthetic = {}
    row_id = 0

    def add(facet, history, context_id, question, answer):
        nonlocal row_id
        synthetic[row_id] = {
            "topic_id": 1,
            "facet_id": facet,
            "conversation_context": history,
            "context_id": context_id,
            "question": question,
            "answer": answer,
        }
        row_id += 1

    for facet, context_id in (("A", 10), ("B", 20)):
        add(facet, [], context_id, "root one", f"one-{facet}")
        add(facet, [], context_id, "root two", f"two-{facet}")
        add(
            facet,
            [{"question": "root one", "answer": f"one-{facet}"}],
            context_id + 1,
            "root two",
            "unused",
        )
        add(
            facet,
            [{"question": "root two", "answer": f"two-{facet}"}],
            context_id + 2,
            "root one",
            "unused",
        )
    evaluation = {
        10: {"Q1": {}, "Q2": {}},
        20: {"Q1": {}, "Q2": {}},
        11: {"Q2": {}},
        21: {"Q2": {}},
        12: {"Q1": {}},
        22: {"Q1": {}},
    }
    roots = _structural_roots(
        synthetic,
        evaluation,
        {(1, "root one"): "Q1", (1, "root two"): "Q2"},
        1,
        ["A", "B"],
    )
    assert roots == ["Q1", "Q2"]
