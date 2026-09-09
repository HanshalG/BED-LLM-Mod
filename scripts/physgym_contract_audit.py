"""Static source contract audit; never import candidate or benchmark code."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

REV = 'fe68079c0921029dde679ed44bc3192dd3b270ab'
HASHES = {
    'physgym/phyenv.py': 'ed943a6dd4c29ee7c0017031e1ebf4624f75021a59ed4535a83cef78ad7547d0',
    'physgym/interface.py': 'f369b1dcd1b70212266956d7d01c41e6257edb056f0016b95b7c76afb58f0e44',
    'physgym/utils/metrics.py': '729a47f9ee205c101ee970bb43e387d5012f530f0783f8423c54e5748745228f',
    'physgym/utils/sandbox.py': '31823f33eb02115956aeed6cbd8c11b236814525725db89e0260253de444a72a',
}


def calls(raw):
    return [dict(line=n.lineno, callee=ast.unparse(n.func),
                 keywords={k.arg:ast.unparse(k.value) for k in n.keywords})
            for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Call)]


def main():
    path = Path('results/nonmyopic/PHYSGYM_CONTRACT_AUDIT_20260909.json')
    if path.exists():
        raise ValueError('already banked')
    files = {}
    for name, sha in HASHES.items():
        url = f'https://raw.githubusercontent.com/principia-ai/PhysGym/{REV}/{name}'
        with urllib.request.urlopen(url,timeout=30) as response:
            raw = response.read(1000001)
        if hashlib.sha256(raw).hexdigest()!=sha:
            raise ValueError('source mismatch')
        sites = calls(raw)
        selected = [s for s in sites if s['callee'] in {
            'exec', 'evaluate_hypothesis', 'create_function_from_string',
            'check_function_equivalence_llm', 'try_symbolic_equivalence'}]
        files[name] = dict(sha256=sha, sites=selected)
    result = dict(revision=REV, files=files, model_calls=0, cost_usd=0,
                  source_executed=False, task_data_opened=False, paid_authority=False)
    with path.open('x') as output:
        json.dump(result,output,indent=2,sort_keys=True)
        output.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
