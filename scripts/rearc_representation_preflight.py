"""Replay cached source channels and check actual public prompt sizes, no API."""
import hashlib
import json
from pathlib import Path
from scripts.rearc_representation_source import RepresentationExamples, public_cases, ROOT as SOURCE_ROOT
from scripts.rearc_representation_update import compile_messages
from scripts.rearc_named_plan_interface import messages, schema
from scripts.rearc_named_plan_contract import body
from scripts.rearc_public_source_journal import save

ROOT = Path('results/nonmyopic/rearc_representation_preflight_20260909')


def main():
    ROOT.mkdir(exist_ok=False)
    result = {'status':'running','calls':0,'withheld_labels_opened':False,'paid_authorized':False}
    try:
        with RepresentationExamples() as source:
            if source.bindings != json.loads((SOURCE_ROOT/'bindings.json').read_text()):
                raise ValueError('source binding changed')
            cases = public_cases(SOURCE_ROOT, source.cohort)
            save(ROOT/'public.json', cases)
            result['public_sha256'] = hashlib.sha256((ROOT/'public.json').read_bytes()).hexdigest()
            result['request_bytes'] = []
            for i,case in enumerate(cases):
                common = dict(inputs=case['inputs']+case['query_inputs']+case['target_inputs'],
                    observations=[{'index':0,'output':case['outputs'][0]}],dsl_source=source.dsl)
                prompts = {'plan':messages('plan','contrasting',**common)}
                for arm in ('python','dsl'):
                    prompts[arm] = compile_messages(arm,**common,plan={f'p{j}':'x'*512 for j in range(4)})
                for stage,prompt in prompts.items():
                    request = body(prompt,46400+3*i+(stage!='plan'),schema('plan' if stage=='plan' else 'compile'))
                    result['request_bytes'].append({'task_index':i,'stage':stage,
                        'bytes':len(json.dumps(request).encode())})
            result['status'] = 'public_prompt_preflight_pass'
            result['repair_size_check'] = 'performed on actual response before each repair dispatch'
    except Exception as error:
        result.update(status='failed_closed',error_type=type(error).__name__)
    finally:
        save(ROOT/'result.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
