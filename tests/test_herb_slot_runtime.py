import hashlib
import json
import pytest
from scripts.herb_slot_runtime import decode, MANIFEST, GRAMMAR, search


def metadata():
    data = json.loads((GRAMMAR/'source.json').read_text())
    data['rules'] = [r.strip() for r in (GRAMMAR/'grammar.jl').read_text().splitlines()
                     if r.strip().startswith('Value = ')]
    return data


def raw(expressions=None, scores=None, expansions=50, status='complete'):
    return (f'status = "{status}"\nfailure = ""\nno_api_key = true\n'
        f'manifest_sha256 = "{hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}"\n'
        f'expansions = {expansions}\nexpressions = '+json.dumps(expressions or ['I']*56)+
        '\nlog_weights = '+json.dumps(scores or [-2.]*56)+'\n').encode()


def test_invalid_and_duplicate_slots_preserved():
    expressions = ['I', '__bed_call1(I, I)', 'vmirror(I)', 'unknown(I)']+['I']*52
    result = decode(raw(expressions), metadata(), 56, 50000)
    assert len(result['slots']) == 56
    assert result['replacement_candidates'] == 0
    assert [r['slot'] for r in result['slots']] == list(range(56))
    assert [r['expression'] for r in result['slots']] == expressions
    assert result['slots'][1]['graph'] is None
    assert result['slots'][3]['graph'] is None
    assert result['slots'][0]['graph'] == result['slots'][4]['graph']
    assert result['slots'][2]['conversion_error'] is None


@pytest.mark.parametrize('value', [float('nan'), float('inf'), .1])
def test_nonfinite_or_positive_scores_rejected(value):
    contents = raw().replace(b'-2.0', str(value).lower().encode(), 1)
    with pytest.raises(ValueError):
        decode(contents, metadata(), 56, 50000)


def test_order_short_prefix_cap_and_failure_rejected():
    for contents in (raw(scores=[-3., -2.]+[-4.]*54), raw(expressions=['I']),
                     raw(expansions=50001), raw(status='failed')):
        with pytest.raises(ValueError):
            decode(contents, metadata(), 56, 50000)


def test_budget_rejected_before_files_or_worker(tmp_path):
    root = tmp_path/'bank'
    with pytest.raises(ValueError):
        search([], 57, 50000, root)
    assert not root.exists()


def test_existing_bank_rejected_before_worker(tmp_path):
    with pytest.raises(FileExistsError):
        search([], 56, 50000, tmp_path)


def test_actual_worker_bank_replay_without_process_and_tamper_rejection(tmp_path, monkeypatch):
    import shutil
    from scripts.herb_slot_runtime import replay
    source = GRAMMAR.parent/'herb_slot_runtime_smoke_20260909'
    bank = tmp_path/'bank'
    shutil.copytree(source, bank)
    def bomb(*args, **kwargs):
        raise AssertionError('replay dispatched process')
    monkeypatch.setattr('subprocess.run', bomb)
    assert replay(bank) == {'status': 'exact_replay', 'slots': 56, 'new_calls': 0}
    path = bank/'result.json'
    result = json.loads(path.read_text())
    result['slots'].pop()
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='candidate replay'):
        replay(bank)
