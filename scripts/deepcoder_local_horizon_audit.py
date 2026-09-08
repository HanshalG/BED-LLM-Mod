"""Model-internal horizon opportunity; never execute a source truth program."""
import hashlib
import json
from pathlib import Path

from environments.program_induction import constrained, prediction as pred
from environments.program_induction.local_support import expand, evaluate
from environments.program_induction.reference import ProgramReference
from scripts import deepcoder_luna_medium_probe as probe
from scripts.deepcoder_opportunity import sample_input


def measure(rows, query_count=8):
    ref = ProgramReference(rows, query_count, seconds=120)
    try:
        state, menu = ref.initial_state, tuple(range(query_count))
        values = {f'h{h}': ref.deployed(state, menu, 4, h) for h in (1, 2, 3)}
        values['openloop_h3'] = ref.deployed(state, menu, 4, 3, 'open_loop')
        values['random'] = ref.random(state, menu, 4)
        values['optimal_b4'] = ref.planner.plan(state, 4, available=menu).root.expected_risk
        return dict(initial_risk=ref.risk(state), terminal_risk=values,
                    root_actions={f'h{h}': ref.root(state, menu, h, 'adaptive')[0] for h in (1, 2, 3)})
    finally:
        ref.clear()


def main():
    source = Path('results/nonmyopic/DEEPCODER_LOCAL_SUPPORT_AUDIT_20260909.json')
    bank = json.loads(source.read_text())
    parent = (probe.PARENT/'forecasts.json').read_bytes()
    if hashlib.sha256(parent).hexdigest() != probe.PARENT_SHA:
        raise ValueError('parent identity mismatch')
    cases = json.loads(parent)['cases']
    dsl, rows = probe.load_dsl(), {}
    for key in bank['cases']:
        raw = json.loads((probe.ROOT/(key+'_history_aware.response.json')).read_text())
        pool, _ = expand(dsl, constrained.decode(dsl, probe.validate_response(raw)), cases[key]['history'])
        if [str(p) for p in pool] != bank['cases'][key]['history_aware']['programs']:
            raise ValueError('support replay mismatch')
        queries = [sample_input(15100000+100*int(key)+i) for i in range(8)]
        if any(q in cases[key]['targets'] for q in queries):
            raise ValueError('query/target overlap')
        matrix = [[pred.category(evaluate(p, x)) for x in queries+cases[key]['targets']] for p in pool]
        try:
            result = dict(status='complete', **measure(matrix))
        except (TimeoutError, RuntimeError) as exc:
            result = dict(status='incomplete', error_type=type(exc).__name__)
        rows[key] = dict(result=result, query_inputs=queries, matrix_sha256=pred.digest(matrix), support_size=len(pool))
    report = dict(cases=rows, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  interpretation='uniform_restricted_pool_internal_risk_not_true_generalization',
                  calls=0, cost_usd=0, hidden_answers_opened=False, efficacy_claim=False)
    probe.save(Path('results/nonmyopic/DEEPCODER_LOCAL_HORIZON_AUDIT_20260909.json'), report, exclusive=True)
    print(pred.canonical(rows))


if __name__ == '__main__':
    main()
