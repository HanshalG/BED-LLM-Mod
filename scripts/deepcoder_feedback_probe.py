"""Frozen one-shot public execution feedback versus matched revision control."""
import argparse
import fcntl
import hashlib
from pathlib import Path

from environments.program_induction import constrained, constrained_request as cr
from environments.program_induction import prediction as pred, feedback_screen as screen
from environments.program_induction.execution_feedback import checks, revision_request
from environments.program_induction.local_support import expand, evaluate
from environments.program_induction.transition_screen import program_forecast
from scripts import deepcoder_luna_medium_probe as luna
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT=Path('results/nonmyopic/deepcoder_feedback_20260909')
PROTOCOL=Path('results/nonmyopic/DEEPCODER_EXECUTION_FEEDBACK_PROTOCOL_20260909.md')
PROTOCOL_SHA='145dd8038b21d9167ce99f7fe8486f49d4edc25d8c79c978a938d718ccd49cc1'
BINDINGS=[__file__,'scripts/paid_program_probe.py','scripts/openrouter_daily_budget.py',
    'scripts/deepcoder_proposal_gate.py','scripts/deepcoder_luna_transition.py',
    'scripts/deepcoder_luna_medium_probe.py','scripts/deepcoder_opportunity.py',
    'environments/program_induction/execution_feedback.py',
    'environments/program_induction/constrained.py','environments/program_induction/constrained_request.py',
    'environments/program_induction/proposals.py','environments/program_induction/prediction.py',
    'environments/program_induction/local_support.py','environments/program_induction/prior.py',
    'environments/program_induction/transition_screen.py','environments/program_induction/feedback_screen.py',
    'environments/program_induction/independent_joint_screen.py','environments/program_induction/rollout_risk.py']


def cases(dsl):
    result={}
    for i in range(8):
        p=sample_program(dsl,35100000+i)
        xs=[sample_input(36100000+100*i+j) for j in range(35)]
        result[str(i)]=dict(history=[dict(inputs=x,output=evaluate(p,x)) for x in xs[:3]],targets=xs[3:])
    return result


def outcomes(dsl,public):
    return {i:dict(target_inputs=c['targets'],outputs=[evaluate(sample_program(dsl,35100000+int(i)),x)
        for x in c['targets']]) for i,c in public.items()}


def run(ledger):
    with PaidProbe(ROOT,ledger,.96,PROTOCOL,PROTOCOL_SHA,BINDINGS) as block:
        dsl=load_dsl()
        public=cases(dsl)
        save(ROOT/'public.json',public,exclusive=True)
        panel={}
        for key,case in public.items():
            seed=37100000+10*int(key)
            initial=None
            forecasts,quality={},{}
            order=['initial']+(['feedback','control'] if int(key)%2==0 else ['control','feedback'])
            for arm in order:
                body=(cr.request(dsl,case['history'],seed,history_blind=False) if arm=='initial' else
                      revision_request(dsl,initial,case['history'],seed+1,with_feedback=arm=='feedback'))
                raw=block.request(key+'_'+arm,luna.bumped(body))
                ps=constrained.decode(dsl,luna.validate_response(raw))
                pool,work=expand(dsl,ps,case['history'])
                quality[arm]=dict(raw_count=len(ps),raw_unique=len({str(p) for p in ps}),
                    raw_compatible=sum(r['fits_observed_history'] for r in checks(dsl,ps,case['history'])),
                    expanded_support=len(pool),work=work)
                save(ROOT/(key+'_'+arm+'.support.json'),dict(programs=[str(p) for p in pool],quality=quality[arm]),exclusive=True)
                if arm=='initial':
                    initial,initial_pool=ps,pool
                forecasts[arm]=program_forecast(dsl,pool if arm=='initial' else initial_pool+pool,case)
            panel[key]=dict(case=case,forecasts=forecasts,quality=quality)
        screen.validate(panel)
        path=ROOT/'forecasts.json'
        save(path,panel,exclusive=True)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        block.report.update(phase='sealed_evaluation',forecast_sha256=sha)
        save(ROOT/'result.json',block.report)
        def load_outcomes():
            block.report['endpoints_opened']=True
            save(ROOT/'result.json',block.report)
            truth=outcomes(dsl,public)
            save(ROOT/'outcomes.json',truth,exclusive=True)
            return truth
        block.report.update(screen.score_sealed(path,sha,load_outcomes))
    return block.report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    args=parser.parse_args()
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(pred.canonical(run(args.ledger)))
