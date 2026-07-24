from scripts.animals_stratified_prior_targets import STRATA, generation_messages


def test_each_stratum_prompt_is_explicit_and_target_free():
    assert len(STRATA) == 8
    for stratum in STRATA:
        messages = generation_messages(stratum)
        assert stratum in messages[0]["content"]
        assert "exactly 16" in messages[0]["content"]
        assert "target" not in messages[0]["content"].casefold()
