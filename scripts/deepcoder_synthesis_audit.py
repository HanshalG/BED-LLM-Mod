"""Exploratory CrossBeam comparison on banked rejection histories."""
import argparse
import hashlib
import json
from pathlib import Path

from environments.program_induction.synthesis import synthesize, evaluate_expression, PIN
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input, outcome
from scripts.deepcoder_support_audit import digest

REJECTION = 'results/nonmyopic/DEEPCODER_REJECTION_AUDIT_20260908.json'
REJECTION_SHA = 'a4d8b2e575762b22b1593c71a6ea95a3a5c1406fb2e2209c7b95a3de1ef1c9c9'


def raw(value):
    return None if value == 'ERROR' else json.loads(value)['value']


def run(output):
    data = Path(REJECTION).read_bytes()
    if hashlib.sha256(data).hexdigest() != REJECTION_SHA:
        raise ValueError('rejection artifact identity failed')
    old = json.loads(data)
    path = Path(output)
    report = dict(status='incomplete', rows=[], rejection_sha256=REJECTION_SHA,
                  crossbeam_pin=PIN, model_calls=0, model_cost_usd=0,
                  exploratory=True, posterior_samples=False, paid_calls_authorized=False)
    with path.open('x') as stream:
        json.dump(report, stream)
    try:
        dsl = load_dsl()
        for entry in old['rows']:
            panel, slot, length = entry['panel'], entry['slot'], entry['history_length']
            inputs = [sample_input(4100000+panel*1000+i) for i in range(40)]
            truth = sample_program(dsl, 6100000+panel*1000+slot)
            encoded = tuple((x, outcome(truth, x)) for x in inputs[:length])
            if digest(encoded) != entry['history_sha256']:
                raise ValueError('observed history identity failed')
            result = synthesize(dsl, [(x, raw(y)) for x, y in encoded],
                                max_attempts=2048, max_weight=9, seconds=5)
            expression = result.pop('expression')
            result.update(panel=panel, slot=slot, history_length=length,
                          history_sha256=entry['history_sha256'],
                          rejection_found_any=entry['accepted'] > 0)
            if expression is not None:
                predictions = [evaluate_expression(expression, x) for x in inputs[8:]]
                result['prediction_sha256'] = digest(predictions)
                targets = [raw(outcome(truth, x)) for x in inputs[8:]]
                result['found_only_point_brier'] = sum(x != y for x, y in zip(predictions, targets))/32
                result['expression'] = expression.expression()
            report['rows'].append(result)
            path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        report['status'] = 'exploratory_comparison_complete'
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if report['status'] == 'failed_closed':
        raise RuntimeError('audit failure banked; no retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
