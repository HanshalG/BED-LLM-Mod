"""Hand-built runtime checks only; no benchmark tasks or outputs."""
import json
from pathlib import Path
from scripts.rearc_graph_runtime import execute, IMAGE, DSL_SHA


def main():
    output = Path('results/nonmyopic/REARC_GRAPH_RUNTIME_SMOKE_20260909.json')
    if output.exists():
        raise FileExistsError(output)
    cases = {
        'higher_order_identity': {'steps': [
            {'id': 'x0', 'op': 'compose', 'args': ['vmirror','vmirror']},
            {'id': 'x1', 'op': 'x0', 'args': ['I']}], 'output': 'x1'},
        'invalid_callable': {'steps': [{'id':'x0','op':'eval','args':['I']}], 'output':'x0'},
        'bounded_excessive_work': {'steps': [
            {'id':'x0','op':'power','args':['vmirror','TEN']}]+
            [{'id':f'x{i}','op':'power','args':[f'x{i-1}','TEN']} for i in range(1,9)]+
            [{'id':'x9','op':'x8','args':['I']}], 'output':'x9'},
    }
    rows = {name: execute(graph, [[1,2],[3,4]]) for name,graph in cases.items()}
    passed = (rows['higher_order_identity'].get('output') == [[1,2],[3,4]]
              and rows['invalid_callable'].get('returncode') == 1
              and rows['bounded_excessive_work'].get('returncode') in (137, 152))
    result = {'status':'passed' if passed else 'failed', 'image':IMAGE, 'dsl_sha256':DSL_SHA,
              'rows':rows,'model_calls':0,'cost_usd':0,'benchmark_tasks_executed':0}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError('runtime smoke failed')


if __name__ == '__main__':
    main()
