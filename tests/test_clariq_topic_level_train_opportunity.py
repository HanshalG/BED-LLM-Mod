from scripts.clariq_topic_level_train_opportunity import analyze_topics


def test_topic_level_policy_uses_one_root_across_facets() -> None:
    synthetic = {}
    row_id = 0

    def add(
        facet: str,
        history: list[dict[str, str]],
        context_id: int,
        question: str,
        answer: str,
    ) -> None:
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
        10: {"Q1": {"with_answer": 0.7}, "Q2": {"with_answer": 0.6}},
        20: {"Q1": {"with_answer": 0.7}, "Q2": {"with_answer": 0.6}},
        11: {"Q2": {"with_answer": 0.5}},
        21: {"Q2": {"with_answer": 0.5}},
        12: {"Q1": {"with_answer": 0.8}},
        22: {"Q1": {"with_answer": 0.8}},
    }
    result = analyze_topics(
        synthetic,
        evaluation,
        {(1, "root one"): "Q1", (1, "root two"): "Q2"},
        ["1"],
    )
    record = result["records"][0]
    assert record["greedy_question_id"] == "Q1"
    assert record["depth_two_question_id"] == "Q2"
    assert abs(record["terminal_gain"] - 0.3) < 1e-12
