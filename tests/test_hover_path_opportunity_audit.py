from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "hover_path_opportunity_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hover_path_opportunity_audit",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_title_aliases_strip_one_trailing_parenthetical():
    assert MODULE.title_aliases("Providence Hospital (Washington, D.C.)") == (
        "providence hospital washington d c",
        "providence hospital",
    )
    assert MODULE.title_aliases("It") == ()


def test_best_path_counts_distinct_support_documents():
    adjacency = {
        0: (1,),
        1: (2,),
        2: (),
    }

    assert MODULE.best_path(
        root_index=0,
        adjacency=adjacency,
        support_indexes={1, 2},
        max_documents=2,
    ) == (1, (0, 1))
    assert MODULE.best_path(
        root_index=0,
        adjacency=adjacency,
        support_indexes={1, 2},
        max_documents=3,
    ) == (2, (0, 1, 2))


def test_analyze_task_detects_robust_immediate_sacrifice():
    titles = [
        "Immediate Support",
        "Setup Document",
        "Second Support",
        "Third Support",
    ]
    texts = {
        "Immediate Support": "This route ends here.",
        "Setup Document": "The next evidence is Second Support.",
        "Second Support": "Continue to Third Support.",
        "Third Support": "This route ends here.",
    }

    result = MODULE.analyze_task(
        task_id="task",
        num_hops=3,
        supporting_titles=[
            "Immediate Support",
            "Second Support",
            "Third Support",
        ],
        candidate_titles=titles,
        text_by_title=texts,
        root_count=2,
    )

    assert result["rank_myopic_root_index"] == 0
    assert result["rank_myopic_value3"] == 1
    assert result["robust_myopic_root_index"] == 0
    assert result["oracle_d2_value"] == 1
    assert result["oracle_d3_root_index"] == 1
    assert result["oracle_d3_value"] == 2
    assert result["rank_root_gain"] == 1
    assert result["robust_root_gain"] == 1
    assert result["rank_sensitive_opportunity"]
    assert result["robust_sacrifice_opportunity"]
    assert result["depth3_exceeds_depth2"]


def test_summarize_requires_robust_not_only_rank_sensitive_opportunity():
    row = {
        "num_hops": 3,
        "nonempty_graph": True,
        "depth3_support_reachability": 1,
        "depth3_root_value_range": 1,
        "rank_sensitive_opportunity": True,
        "robust_sacrifice_opportunity": False,
        "depth3_exceeds_depth2": False,
        "rank_root_gain": 1,
        "robust_root_gain": 0,
    }
    summary = MODULE.summarize(
        [row] * 400,
        resolved_candidate_rows=400,
    )

    assert summary["rank_sensitive_opportunity_count"] == 400
    assert summary["robust_sacrifice_opportunity_count"] == 0
    assert not summary["gates"]["robust_sacrifice_opportunities_at_least_8"]
    assert not summary["gates"]["all_pass"]
