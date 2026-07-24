from scripts.animals_stratified_prior_development import (
    recovered_after_initial_omission,
)


def test_recovered_after_initial_omission_excludes_initially_covered_truths():
    records = [
        {
            "truth_covered_before_counterfactuals": False,
            "candidate_dynamics": [
                {"truth_covered_if_yes": True, "truth_covered_if_no": False}
            ],
        },
        {
            "truth_covered_before_counterfactuals": True,
            "candidate_dynamics": [
                {"truth_covered_if_yes": True, "truth_covered_if_no": True}
            ],
        },
    ]
    assert recovered_after_initial_omission(records) == 1
