"""Bank public inputs and check all initial prompt sizes without model calls."""
import hashlib
import json
from pathlib import Path
from scripts.rearc_expression_examples import ExpressionExamples
from scripts.rearc_expression_interface import build_messages
from scripts.rearc_expression_panel import body

ROOT = Path('results/nonmyopic/rearc_expression_preflight_20260909')


def main():
    ROOT.mkdir(exist_ok=False)
    report = {'status':'running','calls':0,'target_labels_opened':False,'paid_authorized':False}
    try:
        with ExpressionExamples() as source:
            cases = source.public()
            raw = json.dumps(cases,sort_keys=True).encode()
            (ROOT/'public.json').write_bytes(raw)
            report['public_sha256'] = hashlib.sha256(raw).hexdigest()
            report['request_bytes'] = []
            for i,case in enumerate(cases):
                for arm in ('initial','aware','blind'):
                    visible = 3 if arm=='aware' else 1
                    messages = build_messages(inputs=case['inputs']+case['target_inputs'],
                        observations=[{'index':j,'output':case['outputs'][j]} for j in range(visible)],
                        dsl_source=source.dsl)
                    request = body(messages,(34300 if arm=='initial' else 34400)+2*i)
                    report['request_bytes'].append({'task_index':i,'arm':arm,'bytes':len(json.dumps(request).encode())})
            report['status'] = 'public_prompt_preflight_pass'
    except Exception as error:
        report.update(status='failed_closed',error_type=type(error).__name__,error=str(error)[:200])
    finally:
        (ROOT/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
