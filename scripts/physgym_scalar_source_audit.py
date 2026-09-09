"""Source formula compatibility on diagnostic points, not prediction evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from environments.program_induction.scalar_expression import ScalarExpression
from scripts.physgym_development_source import REV


def main():
    path=Path('results/nonmyopic/PHYSGYM_SCALAR_SOURCE_AUDIT_20260909.json')
    if path.exists():
        raise ValueError('already banked')
    manifest=json.loads(Path('results/nonmyopic/physgym_development_source_20260909/manifest.json').read_text())
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/principia-ai/PhysGym/{REV}/physgym/samples/full_samples.json',timeout=30) as response:
        raw=response.read(5000001)
    if hashlib.sha256(raw).hexdigest()!=manifest['source_sha256']:
        raise ValueError('source mismatch')
    rows=json.loads(raw)
    results={}
    for i,task in enumerate(manifest['selected_ids']):
        row=next(r for r in rows if str(r['id'])==task)
        tree=ast.parse(row['python_code'])
        returns=[n.value for n in ast.walk(tree) if isinstance(n,ast.Return)]
        if len(returns)!=1:
            raise ValueError('single-return source required')
        names=sorted(row['input_variables'])
        a=ScalarExpression(row['equation'],names)
        b=ScalarExpression(ast.unparse(returns[0]),names)
        rng=random.Random(52100000+i)
        pairs=[]
        for _ in range(32):
            point={n:rng.randint(3,12) if n=='N' else rng.uniform(.5,2) for n in names}
            pairs.append((a(point),b(point)))
        results[task]=dict(inputs=names,diagnostic_points=32,
                           maximum_formula_return_difference=max(abs(x-y) for x,y in pairs),
                           minimum_output=min(x for x,y in pairs),maximum_output=max(x for x,y in pairs),
                           scientifically_valid_input_distribution=False)
    result=dict(tasks=results,model_calls=0,cost_usd=0,paid_authority=False,
                source_code_executed=False,source_sha256=manifest['source_sha256'])
    with path.open('x') as output:
        json.dump(result,output,indent=2,sort_keys=True)
        output.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
