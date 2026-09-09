from unittest.mock import patch
import subprocess

import pytest

from scripts.black_reference_disagreement import inputs, run_container, summarize


def test_fixed_disjoint_domain():
    rows = inputs()
    assert len(rows) == len({r['source'] for r in rows}) == 96
    assert sum(r['role'] == 'query' for r in rows) == 32
    assert rows == inputs()


def test_saturation_fails_and_incomplete_rejected():
    rows = inputs()
    assert not summarize(rows, [0]*96, [0]*96)['source_eligibility']
    with pytest.raises(ValueError):
        summarize(rows, [], [])


def test_timeout_cleans_only_owned_container():
    with patch('scripts.black_reference_disagreement.checked',
               side_effect=subprocess.TimeoutExpired('docker', 60)) as launch:
        with patch('scripts.black_reference_disagreement.subprocess.run') as cleanup:
            with pytest.raises(subprocess.TimeoutExpired):
                run_container('image')
    args = launch.call_args.args[0]
    name = args[args.index('--name')+1]
    assert name.startswith('bed-black-audit-')
    assert cleanup.call_args.args[0] == ['docker', 'rm', '-f', name]
