"""One-shot independent Luna joint simulator screen, with account-wide caps."""
import argparse
from datetime import datetime
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from environments.program_induction import constrained, constrained_request as cr
from environments.program_induction import prediction as pred, joint_forecast as joint
from environments.program_induction import independent_joint_screen as scoring
from environments.program_induction.local_support import expand, evaluate
from environments.program_induction.structural_support import prepare
from environments.program_induction.transition_screen import program_forecast
from scripts import deepcoder_luna_medium_probe as luna
from scripts.deepcoder_luna_transition import execute
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input
from scripts.deepcoder_proposal_gate import save, money
from scripts.openrouter_daily_budget import read_live_credits, require_budget, budget_status

ROOT = Path('results/nonmyopic/deepcoder_independent_joint_20260909')
PROTOCOL = Path('results/nonmyopic/DEEPCODER_INDEPENDENT_JOINT_PROTOCOL_20260909.md')
PROTOCOL_SHA = '513b6d0efc84c942835cbf5b62a463e6f6e1fa8a07cb4ee46fa1e70d028ae9cf'
CAP, RESERVE = Decimal('1.28'), Decimal('.04')
route = luna.route


def cases(dsl):
    result = {}
    for i in range(8):
        p = sample_program(dsl, 32100000+i)
        xs = [sample_input(33100000+100*i+j) for j in range(35)]
        result[str(i)] = dict(history=[dict(inputs=x, output=evaluate(p,x)) for x in xs[:2]],
                              query=xs[2], targets=xs[3:])
    return result


def observe(dsl, key, x):
    return dict(inputs=x, output=evaluate(sample_program(dsl,32100000+int(key)),x))


def outcomes(dsl, public):
    return {i:dict(target_inputs=c['targets'], outputs=[evaluate(
        sample_program(dsl,32100000+int(i)),x) for x in c['targets']]) for i,c in public.items()}


def run(ledger_path):
    if ROOT.exists():
        raise RuntimeError('already opened; no retry')
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError('protocol mismatch')
    ledger = json.loads(ledger_path.read_text())
    live = read_live_credits()
    status = require_budget(ledger, projected_cost_usd=float(CAP), total_usage_usd=live['total_usage_usd'])
    if money(live['balance_usd']) < CAP:
        raise RuntimeError('insufficient balance')
    carry = dict(ledger.get('pending_reservations', {}))
    base = money(status['spent_today_usd'])
    if sum((money(v) for v in carry.values()), Decimal(0)) > base:
        raise RuntimeError('unaccounted prior reservations')
    ledger['recorded_actual_spend_usd'] = float(base)
    save(ledger_path, ledger)
    dsl = load_dsl()
    initial_route = route()
    ROOT.mkdir()
    save(ROOT/'route.json', initial_route, exclusive=True)
    accepted, uncertain = Decimal(0), Decimal(0)
    pending = dispatched = False
    report = dict(status='incomplete', calls=0, endpoints_opened=False,
                  depth_authorized=False, protocol_sha256=PROTOCOL_SHA,
                  implementation_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in (__file__, 'environments/program_induction/joint_forecast.py',
                              'environments/program_induction/independent_joint_screen.py',
                              'environments/program_induction/rollout_risk.py',
                              'environments/program_induction/local_support.py',
                              'environments/program_induction/structural_support.py',
                              'environments/program_induction/prior.py',
                              'environments/program_induction/constrained.py',
                              'environments/program_induction/constrained_request.py',
                              'environments/program_induction/transition_screen.py',
                              'scripts/deepcoder_luna_medium_probe.py',
                              'scripts/deepcoder_luna_transition.py',
                              'scripts/deepcoder_opportunity.py',
                              'scripts/deepcoder_proposal_gate.py',
                              'scripts/openrouter_daily_budget.py')})

    def account():
        nonlocal ledger
        latest = json.loads(ledger_path.read_text())
        if latest.get('pending_reservations', {}) != ledger.get('pending_reservations', {}):
            raise RuntimeError('concurrent reservation change')
        ledger = latest
        live = read_live_credits()
        s = budget_status(ledger, total_usage_usd=live['total_usage_usd'])
        ledger['recorded_actual_spend_usd'] = float(max(money(s['spent_today_usd']), base+accepted))
        save(ledger_path, ledger)
        require_budget(ledger, projected_cost_usd=float(RESERVE), total_usage_usd=live['total_usage_usd'])
        if money(live['balance_usd']) < RESERVE:
            raise RuntimeError('insufficient dispatch balance')

    def call(key, arm, history, seed):
        nonlocal accepted, pending, dispatched
        tag = key+'_'+arm
        report.update(current=tag, phase='prepare_request')
        save(ROOT/'result.json', report)
        body = luna.bumped(cr.request(dsl, history, seed, history_blind=False))
        save(ROOT/(tag+'.route.json'), route(), exclusive=True)
        save(ROOT/(tag+'.request.json'), body, exclusive=True)
        if accepted+RESERVE > CAP:
            raise RuntimeError('block cap exceeded')
        account()
        ledger['pending_reservations'] = dict(carry, **{'independent_joint:'+tag:float(RESERVE)})
        save(ledger_path, ledger)
        pending, dispatched = True, False
        account()
        report.update(calls=report['calls']+1, phase='http_attempt')
        save(ROOT/'result.json', report)
        dispatched = True
        raw = execute(body)
        save(ROOT/(tag+'.response.json'), raw, exclusive=True)
        accepted += money(raw['usage']['cost'])
        ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']),base+accepted))
        ledger['pending_reservations'] = dict(carry)
        save(ledger_path,ledger)
        pending = False
        report.update(phase='validate_response', accepted_cost_usd=float(accepted))
        programs = constrained.decode(dsl,luna.validate_response(raw))
        pool, work = expand(dsl, programs, history)
        save(ROOT/(tag+'.support.json'),dict(programs=[str(p) for p in pool],work=work),exclusive=True)
        return pool

    try:
        public = cases(dsl)
        save(ROOT/'public.json',public,exclusive=True)
        panel = {}
        for i, case in public.items():
            seed = 34100000+10*int(i)
            a = call(i,'a',case['history'],seed)
            b = call(i,'b',case['history'],seed+1)
            insertion, _ = prepare(dsl,a,case['history'])
            teachers = {name:joint.forecast(dsl,pool,case['history'],[case['query']],case['targets'])
                        for name,pool in [('a',a),('insertion',insertion),('ab',a+b)]}
            seal = dict(case=case,teachers=teachers)
            data = pred.canonical(seal)
            sha = hashlib.sha256((data+'\n').encode()).hexdigest()
            path = ROOT/(i+'.preanswer.json')
            save(path,seal,exclusive=True)
            save(ROOT/(i+'.preanswer.seal.json'),dict(sha256=sha),exclusive=True)
            loaded = json.loads(path.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha or loaded != seal:
                raise ValueError('pre-answer seal failed')
            for law in teachers.values():
                if law is not None and joint.validate(law) != (1,32):
                    raise ValueError('teacher shape')
            obs = observe(dsl,i,case['query'])
            save(ROOT/(i+'.observation.json'),obs,exclusive=True)
            branch = dict(case,history=case['history']+[obs])
            forecasts = {name:program_forecast(dsl,pool,branch)
                         for name,pool in [('filter',a),('insertion',insertion)]}
            for arm in (['regenerated','repeat'] if int(i)%2 == 0 else ['repeat','regenerated']):
                pool = call(i,arm,branch['history'] if arm == 'regenerated' else case['history'],seed+2)
                forecasts[arm] = program_forecast(dsl,a+pool,branch)
            panel[i] = dict(case=case,observation=obs,teachers=teachers,forecasts=forecasts)
        scoring.validate(panel)
        path = ROOT/'forecasts.json'
        save(path,panel,exclusive=True)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        report.update(phase='sealed_evaluation',forecast_sha256=sha)
        save(ROOT/'result.json',report)

        def load_outcomes():
            report['endpoints_opened'] = True
            save(ROOT/'result.json',report)
            truth = outcomes(dsl,public)
            save(ROOT/'outcomes.json',truth,exclusive=True)
            return truth

        report.update(scoring.score_sealed(path,sha,load_outcomes))
    except Exception as exc:
        report.update(status='failed_closed',error_type=type(exc).__name__)
    finally:
        if pending and dispatched:
            uncertain = RESERVE
        ledger['pending_reservations'] = dict(carry, **({'independent_joint:uncertain':float(uncertain)} if uncertain else {}))
        ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']),base+accepted+uncertain))
        save(ledger_path,ledger)
        report.update(accepted_cost_usd=float(accepted),uncertain_exposure_usd=float(uncertain),
                      completed_at=datetime.now(ZoneInfo('Europe/London')).isoformat())
        save(ROOT/'result.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    args = parser.parse_args()
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(pred.canonical(run(args.ledger)))
