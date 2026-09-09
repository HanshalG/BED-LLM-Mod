"""One-shot Luna-medium prefix synthesis versus matched whole-program generation."""
import argparse
import fcntl
import hashlib
from pathlib import Path

from environments.program_induction import constrained, decomposed_request as dr
from environments.program_induction import decomposed_screen as screen, prediction as pred
from environments.program_induction.execution_steps import completed_program
from environments.program_induction.local_support import evaluate
from environments.program_induction.short_support import enumerate_two_statement
from environments.program_induction.transition_screen import program_forecast
from scripts import deepcoder_luna_medium_probe as luna
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT = Path('results/nonmyopic/deepcoder_decomposed_20260909')
PROTOCOL = Path('results/nonmyopic/DEEPCODER_DECOMPOSED_PROTOCOL_20260909.md')
PROTOCOL_SHA = '180e43d5a95e71990ae20c4abafaaf4213f53c10de89462304687899168eca8f'
BINDINGS = [__file__, 'scripts/paid_program_probe.py', 'scripts/openrouter_daily_budget.py',
    'scripts/deepcoder_luna_medium_probe.py', 'scripts/deepcoder_luna_transition.py',
    'scripts/deepcoder_proposal_gate.py', 'scripts/deepcoder_opportunity.py'] + [
    'environments/program_induction/'+p+'.py' for p in (
        'decomposed_request', 'decomposed_screen', 'execution_steps', 'short_support',
        'constrained', 'constrained_request', 'prior', 'proposals', 'local_support',
        'prediction', 'transition_screen', 'independent_joint_screen', 'rollout_risk')]


def cases(dsl):
    result = {}
    for i in range(8):
        p = sample_program(dsl, 39100000+i)
        xs = [sample_input(40100000+100*i+j) for j in range(35)]
        result[str(i)] = dict(history=[dict(inputs=x, output=evaluate(p, x)) for x in xs[:3]],
                              targets=xs[3:])
    return result


def outcomes(dsl, public):
    return {i: dict(target_inputs=c['targets'], source_length=len(sample_program(dsl,39100000+int(i)).statements),
                    outputs=[evaluate(sample_program(dsl,39100000+int(i)),x) for x in c['targets']])
            for i,c in public.items()}


def construct(dsl, case, case_id, rpc, record):
    short, work = enumerate_two_statement(dsl, case['history'])
    pools = {a: list(short) for a in screen.ARMS}
    paths = {a: [[] for _ in range(8)] for a in ('subgoal', 'execution')}
    audits = []
    for step in range(4):
        order = ['subgoal', 'execution', 'whole']
        shift = (case_id+step)%3
        for arm in order[shift:]+order[:shift]:
            seed = 41100000+100*case_id+step
            body = (dr.whole_request(dsl,case['history'],seed) if arm=='whole' else
                    dr.request(dsl,case['history'],paths[arm],seed,subgoals=arm=='subgoal'))
            tag = f'{case_id}_{step}_{arm}'
            raw = rpc(tag,luna.bumped(body))
            text = luna.validate_response(raw)
            audit = dict(tag=tag)
            if arm=='whole':
                ps = constrained.decode(dsl,text)
            else:
                paths[arm], detail = dr.decode(dsl,text,case['history'],paths[arm],subgoals=arm=='subgoal')
                audit.update(detail)
                ps = [completed_program(dsl,p) for p in paths[arm]] if step>=1 else []
            fits = [p for p in ps if all(evaluate(p,h['inputs'])==h['output'] for h in case['history'])]
            pools[arm].extend(fits)
            audit.update(proposed=len(ps), fits=len(fits))
            audits.append(audit)
            record(tag+'.construction.json',audit)
    unique = {a: list({str(p):p for p in ps}.values()) for a,ps in pools.items()}
    return dict(case=case, forecasts={a:program_forecast(dsl,ps,case) for a,ps in unique.items()},
                supports={a:[str(p) for p in ps] for a,ps in unique.items()},
                short_work=work, construction=audits)


def run(ledger):
    with PaidProbe(ROOT,ledger,3.84,PROTOCOL,PROTOCOL_SHA,BINDINGS) as block:
        dsl = load_dsl()
        public = cases(dsl)
        save(ROOT/'public.json',public,exclusive=True)
        panel = {k:construct(dsl,c,int(k),block.request,
                            lambda name,obj:save(ROOT/name,obj,exclusive=True)) for k,c in public.items()}
        screen.validate(panel)
        path = ROOT/'forecasts.json'
        save(path,panel,exclusive=True)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        block.report.update(phase='sealed_evaluation',forecast_sha256=sha)
        save(ROOT/'result.json',block.report)
        def load_outcomes():
            block.report['endpoints_opened'] = True
            save(ROOT/'result.json',block.report)
            truth = outcomes(dsl,public)
            save(ROOT/'outcomes.json',truth,exclusive=True)
            return truth
        block.report.update(screen.score_sealed(path,sha,load_outcomes))
    return block.report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    args = parser.parse_args()
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(pred.canonical(run(args.ledger)))
