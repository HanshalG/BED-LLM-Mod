from scripts.animals_implicit_prior_targets import dedupe_names


def test_dedupe_names_is_case_insensitive_and_preserves_first_order():
    assert dedupe_names(
        ["Cat", " dog ", "CAT", "", "Dog", "Axolotl"]
    ) == ["Cat", "dog", "Axolotl"]
