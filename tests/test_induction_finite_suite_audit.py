import copy
import json

import pytest

from scripts import induction_finite_suite_audit as audit


def row(identity='private-id', formula='(P x)'):
    return dict(schemaVersion='induction_benchmark_record_v1', task='FullObs',
                instanceId=identity, problemDescription=dict(hiddenTarget=dict(formula=formula)),
                problem=dict(worlds=[dict(domain=['secret-object'],
                                         targetExtension=dict(T_true=['secret-object']))]))


def test_aggregate_privacy_determinism_and_grouping():
    rows = [row(), row('other-id', '( P   x )')]
    result = audit.inspect_records(rows, 2)
    assert result == audit.inspect_records(list(reversed(rows)), 2)
    assert result['normalized_text_groups'] == 1
    assert sum(result['tentative_split_records'].values()) == 2
    assert len(result['tentative_split_records']) == 1
    output = json.dumps(result)
    for private in ['private-id', 'other-id', 'secret-object', '(P x)', 'T_true']:
        assert private not in output


def test_labels_do_not_affect_output():
    first = [row()]
    second = copy.deepcopy(first)
    second[0]['problem']['worlds'][0]['targetExtension'] = {'T_true': []}
    second[0]['problemDescription']['description'] = 'private narrative'
    assert audit.inspect_records(first, 1) == audit.inspect_records(second, 1)


@pytest.mark.parametrize('rows,count', [([], 1), ([row(), row()], 2),
                                      ([row(formula='')], 1), ([{}], 1)])
def test_reject_invalid(rows, count):
    with pytest.raises(ValueError):
        audit.inspect_records(rows, count)


def test_byte_binding():
    with pytest.raises(ValueError, match='binding'):
        audit.inspect_bytes(b'not the authorized artifact')


def test_overwrite_before_network(tmp_path, monkeypatch):
    path = tmp_path / 'existing.json'
    path.touch()
    def forbidden(*args, **kwargs):
        pytest.fail('network opened for an existing result')
    monkeypatch.setattr(audit.urllib.request, 'urlopen', forbidden)
    with pytest.raises(FileExistsError):
        audit.run(path)
