"""Zero-call prompt preflight using the banked, identical two-observation panel."""
import hashlib
import json
from pathlib import Path
from scripts.rearc_scene_source import ROOT as SOURCE_ROOT, SceneExamples, scene_cases
from scripts.rearc_scene_update import scene_update
from scripts.rearc_named_plan_contract import body
from scripts.rearc_public_source_journal import save

ROOT = Path('results/nonmyopic/rearc_scene_preflight_v2_20260909')


def main():
    ROOT.mkdir(exist_ok=False)
    report = {'status': 'running', 'model_calls': 0, 'paid_authorized': False,
              'future_endpoints_opened': False, 'request_bytes': []}
    try:
        cases = scene_cases(SOURCE_ROOT/'pool', [json.loads(
            (SOURCE_ROOT/f'second_{i}.json').read_text()) for i in range(4)])
        if cases != json.loads((SOURCE_ROOT/'public.json').read_text()):
            raise ValueError('public source binding')
        with SceneExamples() as source:
            if source.bindings != json.loads((SOURCE_ROOT/'bindings.json').read_text()):
                raise ValueError('source binding')
            for i, c in enumerate(cases):
                def request(stage, prompt, fmt):
                    value = body(prompt, 51400+3*i+{'plan':0,'compile':1,'repair':2}[stage.split('_')[1]], fmt)
                    report['request_bytes'].append({'task_index':i,'stage':stage,
                        'bytes':len(json.dumps(value).encode())})
                    return json.dumps({f'p{j}':'x'*512 for j in range(4)} if stage.endswith('plan')
                        else {'hypotheses':['def transform(g): return g']*8})
                scene_update(inputs=c['inputs']+c['query_inputs']+c['target_inputs'],
                    observations=[{'index':j,'output':y} for j,y in enumerate(c['outputs'])],
                    dsl_source=source.dsl,request=request,
                    diagnose=lambda code,x:{'status':'ok','output':x},bank_update=lambda *a:None)
        save(ROOT/'public.json',cases)
        report['public_sha256']=hashlib.sha256((ROOT/'public.json').read_bytes()).hexdigest()
        report['status']='initial_public_prompt_preflight_pass'
        report['later_request_checks']='Actual generated-code repair prompts checked before every dispatch; synthetic repair is not a worst-case code-size bound.'
    except Exception as error:
        report.update(status='failed_closed',error_type=type(error).__name__)
    finally:
        save(ROOT/'result.json',report)
    print(json.dumps(report))


if __name__=='__main__': main()
