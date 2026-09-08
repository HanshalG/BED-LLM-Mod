from copy import deepcopy

import pytest

from scripts.scilaws_development_split import family, split


def rows():
    return [
        dict(task_id=f"source_{d}_{f}__target", discipline=f"domain{d}")
        for d in range(6)
        for f in range(3)
    ]


def test_complete_disjoint_reorder_invariant_split():
    original = rows()
    result = split(original)
    assert result == split(list(reversed(original)))
    assert len(result["development"]) == 8
    assert len({r["discipline"] for r in result["development"]}) == 6
    assert sorted(r["task_id"] for group in result.values() for r in group) == sorted(
        r["task_id"] for r in original
    )


def test_shared_source_siblings_cannot_enter_holdout():
    original = rows()
    selected = split(original)["development"][0]
    original.append(
        dict(
            task_id=family(selected["task_id"]) + "__other",
            discipline=selected["discipline"],
        )
    )
    result = split(original)
    assert len(result["guarded_siblings"]) == 1
    assert family(result["guarded_siblings"][0]["task_id"]) not in {
        family(r["task_id"]) for r in result["holdout_candidates"]
    }


def test_selection_ignores_results_and_noise_fields():
    original = rows()
    changed = deepcopy(original)
    for i, r in enumerate(changed):
        r.update(published_score=i, noise_scale=i + 1)
    for key in split(original):
        assert [r["task_id"] for r in split(original)[key]] == [
            r["task_id"] for r in split(changed)[key]
        ]


def test_duplicates_and_missing_domains_fail():
    with pytest.raises(ValueError):
        split(rows() + rows()[:1])
    with pytest.raises(ValueError):
        split(rows()[:9])
