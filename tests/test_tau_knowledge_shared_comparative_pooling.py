from scripts.tau_knowledge_shared_comparative_pooling import (
    ROOT_COUNT,
    aggregate_orders,
    compact_rank_input,
    merge_record_pools,
    order_scores,
    pairwise_order_agreement,
    parse_rank_order,
    presentation_order,
)


def _result(document_id: str) -> dict:
    return {
        "id": document_id,
        "title": f"Title {document_id}",
        "content": f"Content for {document_id}",
        "bm25_score": 1.0,
    }


def _record(task_id: str, prefix: str) -> dict:
    return {
        "task_id": task_id,
        "opening": "I need help with a banking policy.",
        "required_documents": ["needed"],
        "initial_information_need_hypotheses": [
            f"{prefix} need {index}" for index in range(8)
        ],
        "first_branches": [
            {
                "query": f"{prefix} root {root}",
                "first_results": [_result(f"{prefix}-first-{root}")],
                "refreshed_information_need_hypotheses": [
                    f"{prefix} refreshed {root}-{index}" for index in range(8)
                ],
                "followups": [
                    {
                        "query": f"{prefix} follow {root}-{follow}",
                        "results": [_result(f"{prefix}-{root}-{follow}")],
                    }
                    for follow in range(4)
                ],
            }
            for root in range(5)
        ],
    }


def test_merge_record_pools_builds_ten_roots_and_dedupes_hypotheses() -> None:
    primary = _record("task_1", "a")
    secondary = _record("task_1", "b")
    secondary["initial_information_need_hypotheses"][0] = (
        primary["initial_information_need_hypotheses"][0]
    )
    merged = merge_record_pools([primary], [secondary])
    assert len(merged) == 1
    assert len(merged[0]["first_branches"]) == ROOT_COUNT
    assert len(merged[0]["initial_information_need_hypotheses"]) == 15


def test_compact_rank_input_hides_futures_from_myopic_view() -> None:
    record = merge_record_pools(
        [_record("task_1", "a")],
        [_record("task_1", "b")],
    )[0]
    order = list(reversed(range(ROOT_COUNT)))
    myopic = compact_rank_input(
        record,
        include_futures=False,
        presentation_order=order,
    )
    full = compact_rank_input(
        record,
        include_futures=True,
        presentation_order=order,
    )
    assert myopic["roots"][0]["root_id"] == "R10"
    assert "followups" not in myopic["roots"][0]
    assert len(full["roots"][0]["followups"]) == 4


def test_parse_rank_order_requires_one_complete_permutation() -> None:
    text = "ORDER|" + "|".join(f"R{index}" for index in range(10, 0, -1))
    assert parse_rank_order(text) == list(reversed(range(ROOT_COUNT)))
    try:
        parse_rank_order(text.replace("R1", "R2"))
    except ValueError as exc:
        assert "permutation" in str(exc)
    else:
        raise AssertionError("duplicate root should fail")


def test_borda_aggregation_and_pairwise_agreement() -> None:
    ascending = list(range(ROOT_COUNT))
    almost = [0, 2, 1, *range(3, ROOT_COUNT)]
    aggregate = aggregate_orders([ascending, ascending, almost])
    assert aggregate[:3] == [0, 1, 2]
    assert order_scores(ascending)[0] == ROOT_COUNT
    assert order_scores(ascending)[-1] == 1
    assert pairwise_order_agreement(ascending, ascending) == 1.0
    assert pairwise_order_agreement(ascending, list(reversed(ascending))) == 0.0


def test_presentation_orders_are_valid_and_mode_specific() -> None:
    myopic = presentation_order(
        task_index=3,
        replicate_index=1,
        include_futures=False,
    )
    full = presentation_order(
        task_index=3,
        replicate_index=1,
        include_futures=True,
    )
    assert sorted(myopic) == list(range(ROOT_COUNT))
    assert sorted(full) == list(range(ROOT_COUNT))
    assert myopic != full
