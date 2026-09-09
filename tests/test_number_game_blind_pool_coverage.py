from fractions import Fraction
from itertools import combinations
from scripts.number_game_blind_pool_coverage import inclusion_probability


def test_matches_exhaustive_pool_selection():
    for n in range(1, 7):
        for occurrences in range(n+1):
            for width in range(1, n+1):
                pools = list(combinations(range(n), width))
                covered = sum(any(i < occurrences for i in pool) for pool in pools)
                assert inclusion_probability(n, occurrences, width) == Fraction(covered, len(pools))
