"""Private, bounded logical duplicate audit; never evaluate benchmark labels."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import yaml
import z3
from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.fol import formulas as fo

from scripts.induction_finite_suite_audit import (
    COMMIT, MAX_DECOMPRESSED, MAX_RAW, SHA, URL, inspect_records,
)


def translate(formula, domain, predicates, bindings=None):
    bindings = {} if bindings is None else bindings

    def term(value):
        if isinstance(value, fo.Var):
            return bindings.get(value.name, z3.Const('free:' + value.name, domain))
        if isinstance(value, fo.Constant):
            return z3.Const('constant:' + value.name, domain)
        raise ValueError('unsupported term')

    if isinstance(formula, fo.Pred):
        key = (formula.name, len(formula.args))
        if key not in predicates:
            predicates[key] = z3.Function(formula.name, *([domain] * key[1]), z3.BoolSort())
        return predicates[key](*[term(x) for x in formula.args])
    if isinstance(formula, fo.Eq):
        return term(formula.left) == term(formula.right)
    if isinstance(formula, fo.FONot):
        return z3.Not(translate(formula.child, domain, predicates, bindings))
    for cls, operator in [(fo.FOAnd, z3.And), (fo.FOOr, z3.Or),
                          (fo.FOImplies, z3.Implies), (fo.FOBiconditional, lambda a, b: a == b)]:
        if isinstance(formula, cls):
            return operator(translate(formula.left, domain, predicates, bindings),
                            translate(formula.right, domain, predicates, bindings))
    if isinstance(formula, (fo.Forall, fo.Exists)):
        variable = z3.FreshConst(domain, 'bound')
        body = translate(formula.body, domain, predicates, {**bindings, formula.var.name: variable})
        return (z3.ForAll if isinstance(formula, fo.Forall) else z3.Exists)([variable], body)
    raise ValueError('unsupported formula')


def pair_status(left, right, timeout_ms=20):
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=0)
    solver.add(z3.Xor(left, right))
    result = solver.check()
    return 'equivalent' if result == z3.unsat else 'distinguishable' if result == z3.sat else 'unknown'


def old_split(digest):
    bucket = int(hashlib.sha256(('induction-finite-suite-v1:' + digest).encode()).hexdigest(), 16) % 10
    return 'development' if bucket < 6 else 'validation' if bucket < 8 else 'confirmation'


def audit(records, max_seconds=90):
    inspect_records(records)
    texts = {}
    for record in records:
        value = record['problemDescription']['hiddenTarget']['formula']
        value = ' '.join(value.replace('(', ' ( ').replace(')', ' ) ').split())
        texts[hashlib.sha256(value.encode()).hexdigest()] = value
    keys = sorted(texts)
    domain, predicates = z3.DeclareSort('Object'), {}
    expressions = []
    for key in keys:
        formula = parse_sexpr_formula(texts[key])
        if formula.free_vars() != {'x'}:
            raise ValueError('unexpected free variables')
        expressions.append(translate(formula, domain, predicates))
    parent = list(range(len(keys)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    started = time.monotonic()
    counts, cross = Counter(), 0
    total = len(keys) * (len(keys) - 1) // 2
    for i in range(len(keys)):
        for j in range(i):
            if time.monotonic() - started >= max_seconds:
                break
            status = pair_status(expressions[i], expressions[j])
            counts[status] += 1
            if status == 'equivalent':
                parent[root(i)] = root(j)
                cross += old_split(keys[i]) != old_split(keys[j])
        else:
            continue
        break
    return dict(source_commit=COMMIT, source_sha256=SHA, z3_version=z3.get_version_string(),
                unique_texts=len(keys), total_pairs=total, checked_pairs=sum(counts.values()),
                pair_status_counts=dict(sorted(counts.items())),
                unvisited_pairs=total-sum(counts.values()),
                proven_equivalence_components=len({root(i) for i in range(len(keys))}),
                proven_cross_tentative_split_equivalent_pairs=cross,
                elapsed_seconds=time.monotonic()-started, pair_timeout_ms=20,
                wall_budget_seconds=max_seconds, labels_consulted=False,
                private_rows_materialized=True, model_calls=0, paid_cost_usd=0,
                finite_domain_equivalence_certified=False, final_split_authorized=False,
                downstream_experiment_authorized=False)


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError('output already exists')
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read(MAX_RAW + 1)
    if len(raw) > MAX_RAW or hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('source binding mismatch')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        content = stream.read(MAX_DECOMPRESSED + 1)
    if len(content) > MAX_DECOMPRESSED:
        raise ValueError('decompression limit')
    result = audit(yaml.safe_load(content))
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        run(args.output)
    except Exception:
        raise SystemExit('Equivalence audit failed closed; private context suppressed.') from None
