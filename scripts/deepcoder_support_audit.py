"""Independent-program support diagnostic, with no planning or policy reruns."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import monotonic

from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input, outcome


BANK = 'results/nonmyopic/DEEPCODER_ACTIVE_PILOT_20260908.json'
BANK_SHA = '84bdcb025e70d3bbbb3640c5e40f86ae271f005098468f3056637665ba9aa476'


def moments(rows, truth, start):
    """Score the empirical predictor without smoothing unseen output categories."""
    if not rows or not truth or not 0 <= start < len(truth):
        raise ValueError('nonempty predictor and targets required')
    if any(len(row) != len(truth) for row in rows):
        raise ValueError('inconsistent target shapes')
    n = len(rows)
    squared, risk, zeros = 0., 0., 0
    for t in range(start, len(truth)):
        counts = Counter(row[t] for row in rows)
        sum_squared = sum(c*c for c in counts.values())/(n*n)
        p_true = counts[truth[t]]/n
        squared += .5*(1+sum_squared-2*p_true)
        risk += .5*(1-sum_squared)
        zeros += not counts[truth[t]]
    return squared/(len(truth)-start), risk/(len(truth)-start), zeros


def summarize(reference, independent, queries):
    if not reference or not independent or not 0 < queries < len(reference[0]):
        raise ValueError('nonempty reference, independent worlds and target columns required')
    size = len(reference[0])
    if any(len(row) != size for row in reference+independent):
        raise ValueError('ragged matrix')
    prior_losses, prior_risks, losses, risks, supports = [], [], [], [], []
    unsupported = zeros = prior_zeros = 0
    by_query = [0]*queries
    for truth in independent:
        loss, risk, zero = moments(reference, truth, queries)
        prior_losses.append(loss)
        prior_risks.append(risk)
        prior_zeros += zero
        for query in range(queries):
            matching = [row for row in reference if row[query] == truth[query]]
            if not matching:
                unsupported += 1
                by_query[query] += 1
                continue
            loss, risk, zero = moments(matching, truth, queries)
            losses.append(loss)
            risks.append(risk)
            supports.append(len(matching))
            zeros += zero
    def mean(values):
        return sum(values)/len(values) if values else None
    return dict(independent_programs=len(independent), query_conditions=len(independent)*queries,
                unsupported_query_conditions=unsupported, unsupported_by_query=by_query,
                supported_conditions=len(losses),
                prior_brier=mean(prior_losses), prior_internal_risk=mean(prior_risks),
                prior_zero_target_mass_count=prior_zeros,
                prior_target_count=len(independent)*(size-queries),
                supported_only_brier=mean(losses), supported_only_internal_risk=mean(risks),
                supported_only_mean_particles=mean(supports),
                supported_only_zero_target_mass_count=zeros,
                supported_only_target_count=len(losses)*(size-queries))


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def run(output):
    source = Path(BANK).read_bytes()
    if hashlib.sha256(source).hexdigest() != BANK_SHA:
        raise ValueError('banked pilot identity failed')
    bank = json.loads(source)
    path = Path(output)
    result = dict(status='incomplete', bank_sha256=BANK_SHA, panels=[], model_calls=0,
                  model_cost_usd=0, paid_calls_authorized=False, policy_comparisons_run=0)
    with path.open('x') as stream:
        json.dump(result, stream)
    try:
        dsl = load_dsl()
        for index, old in enumerate(bank['panels']):
            start = monotonic()
            inputs = [sample_input(4100000+index*1000+i) for i in range(40)]
            programs = [sample_program(dsl, 3100000+index*1000+i) for i in range(128)]
            reference = [[outcome(p, inp) for inp in inputs] for p in programs]
            if (old['index'] != index or digest(inputs) != old['input_sha256']
                    or digest([str(p) for p in programs]) != old['program_sha256']
                    or digest(reference) != old['matrix_sha256']):
                raise ValueError('reconstructed reference differs from bank')
            fresh = [sample_program(dsl, 5100000+index*1000+i) for i in range(128)]
            independent = []
            for program in fresh:
                if monotonic()-start > 30:
                    raise TimeoutError('panel construction cap')
                independent.append([outcome(program, inp) for inp in inputs])
            report = summarize(reference, independent, 8)
            if monotonic()-start > 30:
                raise TimeoutError('panel analysis cap')
            report.update(index=index, reference_matrix_sha256=digest(reference),
                          independent_program_sha256=digest([str(p) for p in fresh]),
                          independent_matrix_sha256=digest(independent),
                          elapsed_seconds=monotonic()-start)
            result['panels'].append(report)
            path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        result.update(status='support_failure_observed' if any(
            p['unsupported_query_conditions'] for p in result['panels'])
                      else 'no_support_failure_in_sample_not_certification')
    except Exception as exc:
        result.update(status='failed_closed', error_type=type(exc).__name__)
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    if result['status'] == 'failed_closed':
        raise RuntimeError('audit failed and banked, no automatic retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
