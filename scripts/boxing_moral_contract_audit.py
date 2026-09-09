"""Inspect pinned source and execute only two isolated pure feature methods."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

REV = 'b43e38cb03d09c13efa9cf4d9bae740d51157bfd'
SHA = '1934c2ff397764925a3d290e61e6ae1cabcfc95c81357442ba7571d5b77d610f'
URL = f'https://raw.githubusercontent.com/kanishkg/boxing-gym/{REV}/src/boxing_gym/envs/moral_machines.py'


def audit(raw):
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('source mismatch')
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='MoralMachine')
    methods = {n.name:n for n in cls.body if isinstance(n, ast.FunctionDef)}
    safe = ast.Module(body=[methods['calculate_attr_diff'], methods['_get_char_attr']], type_ignores=[])
    namespace = {}
    # Neither method imports, calls a service, mutates state, or samples outcomes.
    exec(compile(ast.fix_missing_locations(safe), URL, 'exec'), namespace)
    feature, attrs = namespace['calculate_attr_diff'], namespace['_get_char_attr']
    cases = [('count', ['boy'], []), ('age', ['boy'], ['old_man']),
             ('gender', ['large_woman'], ['large_man']),
             ('social_status', ['male_doctor'], ['homeless'])]
    results = []
    for attribute, a, b in cases:
        if attribute == 'count':
            # This case contains only humans: step's human count is group size.
            generated = len(a)-len(b)
        else:
            generated = sum(attrs(None,c)[attribute] for c in a)-sum(attrs(None,c)[attribute] for c in b)
        exported = feature(None,a,b,attribute)
        results.append(dict(attribute=attribute,group1=a,group2=b,
                            simulator_feature=generated,exported_feature=exported,
                            mismatch=generated!=exported))
    step = ast.unparse(methods['step'])
    return dict(source_url=URL, source_sha256=SHA, cases=results,
                step_calls_chat_completion='self.llm.chat.completions.create' in step,
                rationale_receives_hidden_weights='beta_age=self.beta_age' in step,
                model_calls=0, cost_usd=0, environment_instantiated=False,
                paid_authority=False)


def main():
    path = Path('results/nonmyopic/BOXING_MORAL_CONTRACT_AUDIT_20260909.json')
    if path.exists():
        raise ValueError('already banked')
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read(1000001)
    result = audit(raw)
    with path.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
