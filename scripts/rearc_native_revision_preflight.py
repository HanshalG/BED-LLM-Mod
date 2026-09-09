"""Initial public prompt preflight; second observations remain sealed."""
import hashlib
import json
from pathlib import Path
from scripts.rearc_native_revision_source import RevisionExamples,revision_cases,ROOT as SOURCE_ROOT
from scripts.rearc_named_plan_interface import messages,schema
from scripts.rearc_representation_update import compile_messages
from scripts.rearc_named_plan_contract import body
from scripts.rearc_public_source_journal import save

ROOT=Path('results/nonmyopic/rearc_native_revision_preflight_20260909')


def main():
    ROOT.mkdir(exist_ok=False)
    report={'status':'running','model_calls':0,'paid_authorized':False,'second_answers_opened':False,
            'future_endpoints_opened':False}
    try:
        with RevisionExamples() as source:
            if source.bindings!=json.loads((SOURCE_ROOT/'bindings.json').read_text()):
                raise ValueError('source binding changed')
            cases=revision_cases(SOURCE_ROOT/'pool')
            if len(cases)!=4:
                raise ValueError('four-task coverage')
            save(ROOT/'public.json',cases)
            report['public_sha256']=hashlib.sha256((ROOT/'public.json').read_bytes()).hexdigest()
            report['request_bytes']=[]
            for i,c in enumerate(cases):
                common=dict(inputs=c['inputs']+[c['reveal_input']]+c['query_inputs']+c['target_inputs'],
                    observations=[{'index':0,'output':c['outputs'][0]}],dsl_source=source.dsl)
                prompts={'plan':messages('plan','contrasting',**common),
                    'compile':compile_messages('python',**common,plan={f'p{j}':'x'*512 for j in range(4)})}
                for stage,prompt in prompts.items():
                    request=body(prompt,50400+6*i+(stage=='compile'),schema(stage))
                    report['request_bytes'].append({'task_index':i,'stage':stage,
                        'bytes':len(json.dumps(request).encode())})
            report['status']='initial_public_prompt_preflight_pass'
            report['later_request_checks']='required on actual prior programs and revealed example before each dispatch'
    except Exception as error:
        report.update(status='failed_closed',error_type=type(error).__name__)
    finally:
        save(ROOT/'result.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
