"""Prospective fixed-history rejection feasibility, not a depth comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from time import monotonic

from environments.program_induction.rejection import conditioned_draws
from scripts.deepcoder_opportunity import load_dsl, sample_input, sample_program, outcome
from scripts.deepcoder_support_audit import BANK, BANK_SHA, digest, moments


def run(output):
    raw = Path(BANK).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA:
        raise ValueError('bank identity failed')
    old = json.loads(raw)
    path = Path(output)
    report = dict(status='incomplete', rows=[], bank_sha256=BANK_SHA,
                  model_calls=0, model_cost_usd=0, paid_calls_authorized=False,
                  policy_comparisons_run=0)
    with path.open('x') as stream:
        json.dump(report, stream)
    try:
        dsl = load_dsl()
        for panel in range(4):
            inputs = [sample_input(4100000+panel*1000+i) for i in range(40)]
            if digest(inputs) != old['panels'][panel]['input_sha256']:
                raise ValueError('input identity failed')
            for slot in range(4):
                truth = sample_program(dsl, 6100000+panel*1000+slot)
                observations = [outcome(truth, x) for x in inputs[:4]]
                for length in range(1, 5):
                    start = monotonic()
                    rng = random.Random(7100000+panel*1000+slot*10+length)
                    history = tuple(zip(inputs[:length], observations[:length]))
                    result = conditioned_draws(
                        lambda: sample_program(dsl, rng.getrandbits(128)), outcome, history,
                        num_particles=16, max_draws=2048, deadline=start+10)
                    row = dict(panel=panel, slot=slot, history_length=length,
                               complete=result.complete, accepted=len(result.particles),
                               draws=result.draws, history_evaluations=result.evaluations,
                               history_sha256=digest(history),
                               accepted_programs_sha256=digest([str(p) for p in result.particles]))
                    # Prediction is fixed by accepted draws before target outcomes are read.
                    if result.complete:
                        predictions = [[outcome(p, x) for x in inputs[8:]] for p in result.particles]
                        row['prediction_sha256'] = digest(predictions)
                        targets = [outcome(truth, x) for x in inputs[8:]]
                        loss, risk, zeros = moments(predictions, targets, 0)
                        row.update(complete_only_brier=loss, complete_only_internal_risk=risk,
                                   complete_only_zero_target_mass_count=zeros, target_count=32)
                    row['elapsed_seconds'] = monotonic()-start
                    if row['elapsed_seconds'] > 10:
                        raise TimeoutError('complete context cap')
                    report['rows'].append(row)
                    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        report['status'] = ('all_contexts_filled_not_calibration_certificate'
                            if all(r['complete'] for r in report['rows']) else 'rejection_budget_inadequate')
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if report['status'] == 'failed_closed':
        raise RuntimeError('banked failure, no retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
