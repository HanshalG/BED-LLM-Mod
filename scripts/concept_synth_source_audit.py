"""Read one pinned generator as AST, never import it or load benchmark outcomes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = 'c1f71f98623ab6e3513f820a8e52235d5b498694'
SOURCE_SHA = 'a8777d18e80fcfa095385676df9e5fa9d3e6d71e9a28865612d1fdb5d362a1c8'
URL = f'https://raw.githubusercontent.com/SerafimBatzoglou/concept-synth/{COMMIT}/src/concept_synth/induction/generator.py'


def inspect_source(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'formula_library')
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.List):
        raise ValueError('formula library no longer a fixed literal list')
    templates = []
    for node in returns[0].value.elts:
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                or node.func.id != 'FormulaTemplate' or len(node.args) != 4):
            raise ValueError('unexpected formula template')
        family, key = (ast.literal_eval(x) for x in node.args[:2])
        templates.append(dict(family=family, key=key))
    metadata = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_base_problem_description')
    metadata_keys = sorted({n.value for node in ast.walk(metadata) if isinstance(node, ast.Dict)
                            for n in node.keys if isinstance(n, ast.Constant) and isinstance(n.value, str)})
    return dict(template_count=len(templates), templates=templates, metadata_keys=metadata_keys,
                source_executed=False, dataset_rows_loaded=0, prediction_rows_loaded=0,
                endpoint_labels_loaded=0)


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('generator source hash mismatch')
    result = inspect_source(raw.decode('utf-8'))
    result.update(commit=COMMIT, source_sha256=SOURCE_SHA, url=URL,
                  source_only=True, native_generator_headline_authorized=False,
                  model_calls=0, paid_cost_usd=0)
    with path.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
