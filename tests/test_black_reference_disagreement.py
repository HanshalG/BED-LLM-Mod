from unittest.mock import patch
import io
import json
import subprocess
import types

import pytest

from scripts.black_reference_disagreement import RUNNER, inputs, parent_binding, run_container, summarize


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


def test_wrong_parent_tag_rejected():
    with patch('scripts.black_reference_disagreement.checked', return_value='[{"Id":"wrong"}]'):
        with pytest.raises(ValueError, match='parent image changed'):
            parent_binding()


@pytest.mark.parametrize('recognized', [True, False])
def test_only_documented_safe_mode_error_is_categorical(recognized, capsys):
    class NothingChanged(Exception):
        pass

    class InvalidInput(Exception):
        pass

    def formatter(*args, **kwargs):
        assert kwargs['fast'] is False
        message = ('cannot use --safe with this file; failed to parse source file'
                   if recognized else 'unexpected equivalence failure')
        raise AssertionError(message)

    fake = types.SimpleNamespace(format_file_contents=formatter, FileMode=lambda: None,
                                 NothingChanged=NothingChanged, InvalidInput=InvalidInput)
    with patch.dict('sys.modules', {'black': fake}):
        with patch('builtins.open', return_value=io.StringIO('[{"source":"print x"}]')):
            if recognized:
                exec(RUNNER, {})
                assert json.loads(capsys.readouterr().out) == [{'kind': 'SourceAstUnsupported'}]
            else:
                with pytest.raises(AssertionError, match='unexpected'):
                    exec(RUNNER, {})
