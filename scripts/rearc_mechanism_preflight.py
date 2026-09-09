"""One-shot public collection and worst-plan prompt check, no model calls."""
import hashlib
import json
from pathlib import Path
from scripts.rearc_mechanism_examples import MechanismExamples
from scripts.rearc_mechanism_interface import messages,schema
from scripts.rearc_mechanism_contract import body

ROOT = Path('results/nonmyopic/rearc_mechanism_preflight_20260909')


def main():
    ROOT.mkdir(exist_ok=False)
    result = {'status':'running','calls':0,'withheld_labels_opened':False,'paid_authorized':False}
    try:
        with MechanismExamples() as source:
            cases = source.public()
            raw = json.dumps(cases,sort_keys=True).encode()
            (ROOT/'public.json').write_bytes(raw)
            result['public_sha256']=hashlib.sha256(raw).hexdigest()
            result['request_bytes']=[]
            plan = {'plans':[{'id':f'p{i}','description':'x'*512} for i in range(4)]}
            for i,case in enumerate(cases):
                for mode in ('contrasting','ordinary'):
                    for stage in ('plan','compile'):
                        prompt = messages(stage,mode,inputs=case['inputs']+case['query_inputs']+case['target_inputs'],
                            observations=[{'index':0,'output':case['outputs'][0]}],dsl_source=source.dsl,
                            plan=plan if stage=='compile' else None)
                        request = body(prompt,38400+3*i+(stage=='compile'),schema(stage))
                        result['request_bytes'].append({'task_index':i,'mode':mode,'stage':stage,
                                                       'bytes':len(json.dumps(request).encode())})
            result['status']='public_prompt_preflight_pass'
    except Exception as error:
        result.update(status='failed_closed',error_type=type(error).__name__,error=str(error)[:200])
    finally:
        (ROOT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
