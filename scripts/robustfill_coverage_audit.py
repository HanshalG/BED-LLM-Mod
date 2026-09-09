"""Retrospective representation witnesses, never predictive or BED evaluation."""
import hashlib
import json
from pathlib import Path
import types
import urllib.request

REV = 'ef046ce2cc3fcd024e32f5dfe00e69700dac82ed'
SHA = '58fc2606ba386f8b8b40acf0e874d2813cd3f3d8d07d803aa0e6711addd9a4ce'
URL = f'https://raw.githubusercontent.com/google-deepmind/exedec/{REV}/tasks/robust_fill/dsl.py'


def load():
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read(100001)
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('source mismatch')
    module = types.ModuleType('pinned_robustfill')
    # Only the fully inspected, immutable upstream DSL executes, never model code.
    exec(compile(raw, URL, 'exec'), module.__dict__)
    return module


def atoms(dsl):
    for t in dsl.Type:
        yield dsl.GetAll(t)
        for i in dsl.INDEX:
            yield dsl.GetToken(t, i)
    yield dsl.Trim()
    for c in dsl.CHARACTER:
        yield dsl.ConstStr(c)
    for a in range(-100, 101):
        for b in range(-100, 101):
            yield dsl.SubStr(a, b)


def search(vectors, targets, max_parts=6, max_states=10000):
    """Exact bounded concatenation search over behavioral atom equivalence classes."""
    start = (0,) * len(targets)
    end = tuple(map(len, targets))
    frontier = {start: ()}
    seen = {start}
    for _ in range(max_parts):
        following = {}
        for positions, path in frontier.items():
            if positions == end:
                return {'status': 'witness', 'path': path, 'states': len(seen)}
            for outputs, name in vectors.items():
                if not any(outputs):
                    continue
                if not all(y.startswith(o, p) for y, o, p in zip(targets, outputs, positions)):
                    continue
                new = tuple(p + len(o) for p, o in zip(positions, outputs))
                if new == end:
                    return {'status': 'witness', 'path': path + (name,), 'states': len(seen)}
                if new not in seen:
                    seen.add(new)
                    if len(seen) > max_states:
                        return {'status': 'incomplete_state_cap', 'states': len(seen)}
                    following[new] = path + (name,)
        frontier = following
    return {'status': 'no_witness_in_declared_subset', 'states': len(seen)}


def audit(dsl, pairs):
    inputs, targets = zip(*pairs)
    vectors, objects = {}, {}
    attempted = 0
    for atom in atoms(dsl):
        variants = [atom]
        if not isinstance(atom, dsl.ConstStr):
            variants += [dsl.Compose(dsl.ToCase(c), atom) for c in dsl.Case]
        for expression in variants:
            attempted += 1
            outputs = tuple(expression(x) for x in inputs)
            # A concatenated piece must occur within every complete target.
            if any(outputs) and all(o in y for o, y in zip(outputs, targets)):
                name = expression.to_string()
                vectors.setdefault(outputs, name)
                objects[name] = expression
    result = search(vectors, targets)
    if result['status'] == 'witness':
        program = dsl.Concat(*(objects[n] for n in result['path']))
        actual = [program(x) for x in inputs]
        if actual != list(targets):
            raise ValueError('witness execution mismatch')
        result.update(verified_examples=len(pairs), outputs=actual)
    return dict(result, attempted_atoms=attempted, compatible_piece_behaviors=len(vectors))


def main():
    from scripts.author_strings_opportunity import source
    destination = Path('results/nonmyopic/ROBUSTFILL_DEV_COVERAGE_20260909.json')
    if destination.exists():
        raise ValueError('already banked')
    dsl = load()
    report = dict(source_url=URL, source_sha256=SHA, model_calls=0, cost_usd=0,
                  predictive_evaluation=False, paid_authority=False,
                  subset='GetAll/GetToken/Trim/ConstStr/SubStr; optional ToCase; Concat<=6',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tasks={t: audit(dsl, source(t)) for t in ('1', '10')})
    with destination.open('x') as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
