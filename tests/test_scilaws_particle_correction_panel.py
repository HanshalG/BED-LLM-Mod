from copy import deepcopy

import pytest

from scripts.scilaws_particle_correction_panel import assess, read_bound, PREFIX, PREFIX_SHA, COUNTS
from scripts.scilaws_particle_integration_panel import SCENARIOS


def fixture():
    cases = [dict(task_index=t, seed=s, scenario=c,
                  candidates=[dict(branch_count=q, status='complete', action=0,
                                   roots=[dict(value=1.) for _ in range(8)], seconds=.1, states=32)
                              for q in COUNTS])
             for t in range(8) for s in (1304, 1305) for c in SCENARIOS]
    refs = [dict(c, reference_reason=None,
                 reference=[dict(value=1., error_estimate=0., tail_bound=0., mass_error=0.)
                            for _ in range(8)]) for c in cases]
    return cases, refs


def test_full_pass_and_single_failure():
    cases, refs = fixture()
    assert assess(cases, refs)[4]['qualified']
    cases[-1]['candidates'][0]['roots'] = [dict(value=1.01) for _ in range(8)]
    assert assess(cases, refs)[4]['passed_cases'] == 47
    assert assess(cases, refs)[8]['qualified']
    with pytest.raises(ValueError):
        assess(cases[:-1], refs)
    with pytest.raises(ValueError):
        assess(cases, refs[::-1])


def test_invalid_reference_and_action_rejected():
    cases, refs = fixture()
    bad = deepcopy(refs)
    bad[0]['reference'][0]['value'] = float('nan')
    with pytest.raises(ValueError):
        assess(cases, bad)
    cases[0]['candidates'][0]['action'] = 2
    with pytest.raises(ValueError):
        assess(cases, refs)


def test_exact_prefix(tmp_path):
    assert len(read_bound(PREFIX, PREFIX_SHA)['cases']) == 3
    path = tmp_path / 'tampered.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        read_bound(path, PREFIX_SHA)
