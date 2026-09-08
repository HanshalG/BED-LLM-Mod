import pytest

from scripts.scilaws_particle_continuation_audit import assess, BRANCHES, SCENARIOS


def cases():
    return [dict(task_index=t, seed=s, scenario=c, branch_index=b, reason=None,
                 roots=[dict(value=1.) for _ in range(8)],
                 reference=[dict(value=1., error_estimate=0., tail_bound=0., mass_error=0.) for _ in range(8)])
            for t in range(8) for s in (1304, 1305) for c in SCENARIOS for b in BRANCHES]


def test_coverage_and_failed_closed():
    rows = cases()
    assert assess(rows)['screen_passed']
    assert not assess(rows)['full_tree_qualified']
    rows[-1]['roots'][0]['value'] = 1.01
    assert assess(rows)['passed_cases'] == 143
    with pytest.raises(ValueError):
        assess(rows[:-1])
    with pytest.raises(ValueError):
        assess(rows[::-1])


def test_invalid_reference_and_prefix():
    rows = cases()
    rows[0]['reference'][0]['value'] = float('nan')
    rows[1]['reason'] = 'budget exceeded'
    rows[1]['roots'] = []
    assert assess(rows)['passed_cases'] == 142
