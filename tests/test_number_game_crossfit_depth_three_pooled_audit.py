import pytest

from scripts.number_game_crossfit_depth_three_pooled_audit import (
    FIXED_POLICY_RESULT,
    FIXED_POLICY_RESULT_SHA256,
    FRESH_REPLICATION_RESULT,
    FRESH_REPLICATION_RESULT_SHA256,
    exact_one_sided_sign_pvalue,
    sha256_file,
    stratified_bootstrap_interval,
)


def test_pooled_sources_are_hash_bound() -> None:
    assert sha256_file(FIXED_POLICY_RESULT) == FIXED_POLICY_RESULT_SHA256
    assert (
        sha256_file(FRESH_REPLICATION_RESULT)
        == FRESH_REPLICATION_RESULT_SHA256
    )


def test_stratified_bootstrap_preserves_constant_study_values() -> None:
    assert stratified_bootstrap_interval(
        [[-1.0, -1.0], [-3.0, -3.0]],
        seed=1,
        samples=100,
    ) == [-2.0, -2.0]


def test_exact_sign_pvalue_excludes_ties() -> None:
    assert exact_one_sided_sign_pvalue(3, 0) == pytest.approx(0.125)
    assert exact_one_sided_sign_pvalue(0, 0) == pytest.approx(1.0)
