from scripts.rearc_source_smoke import selected_functions, SEEDS
import json
from pathlib import Path


def test_extracts_only_frozen_functions_without_execution():
    text = 'raise RuntimeError("not executed")\ndef a(): return 1\ndef secret(): return 99'
    assert selected_functions(text, {'a'}) == 'def a(): return 1'
    assert SEEDS == (31000,31001,31002,31003)


def test_bank_has_exact_coverage_and_verifier_agreement():
    scope = json.loads(Path('results/nonmyopic/REARC_SOURCE_SCOPE_20260909.json').read_text())
    result = json.loads(Path('results/nonmyopic/REARC_SOURCE_SMOKE_20260909.json').read_text())
    assert result['status'] == 'passed'
    assert [(r['task'],r['seed']) for r in result['rows']] == [(t,s) for t in scope['selected_ids'] for s in SEEDS]
    assert all(r['status']=='ok' and r['returncode']==0 and r['output_sha256']==r['verifier_sha256'] for r in result['rows'])
    assert result['model_calls'] == result['cost_usd'] == 0
    assert result['examples_emitted'] is False
