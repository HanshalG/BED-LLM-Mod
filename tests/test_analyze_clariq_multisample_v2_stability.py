from scripts.analyze_clariq_multisample_v2_stability import _selected


def test_selected_uses_requested_sample_indices() -> None:
    maps = {
        "Q1": ["YN", "YN", "YY"],
        "Q2": ["NY", "NY", "YY"],
        "Q3": ["UU", "UU", "UU"],
    }
    assert _selected(maps, [0, 1]) in maps
    assert _selected(maps, [2]) in maps
