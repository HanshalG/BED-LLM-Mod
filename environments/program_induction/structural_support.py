"""One-stage insertion after an existing statement, without changing the DSL."""
from .local_support import evaluate
from .prior import program_probability, statement_options
from .proposals import _history


def insertions(dsl, program):
    program_probability(dsl, program)
    n = len(program.statements)
    if n == 4:
        return
    for k in range(1, n+1):
        prefix = list(program.statements[:k])
        variables = [('x0', list), ('x1', list)]+[(s.variable, s.operation.output_type) for s in prefix]
        previous = variables[-1][0]
        for op, args in statement_options(dsl, variables, True):
            if k < n and op.output_type != variables[-1][1]:
                continue
            statements = prefix+[dsl.Statement(f'x{k+2}', op, args)]
            for j in range(k, n):
                old = program.statements[j]
                renamed = []
                for arg in old.args:
                    if j == k and arg == previous:
                        arg = f'x{k+2}'
                    elif isinstance(arg, str) and int(arg[1:]) >= k+2:
                        arg = f'x{int(arg[1:])+1}'
                    renamed.append(arg)
                statements.append(dsl.Statement(f'x{j+3}', old.operation, renamed))
            candidate = dsl.Program(['x0', 'x1'], statements)
            program_probability(dsl, candidate)
            yield candidate


def prepare(dsl, previous_pool, previous_history):
    history = _history(previous_history)
    if len(history) >= 4:
        raise ValueError('no room for another observation')
    roots, candidates = {}, {}
    for root in previous_pool:
        program_probability(dsl, root)
        if all(evaluate(root, h['inputs']) == h['output'] for h in history):
            roots.setdefault(str(root), root)
    attempts = 0
    for root in roots.values():
        candidates.setdefault(str(root), root)
        for p in insertions(dsl, root):
            attempts += 1
            candidates.setdefault(str(p), p)
    retained = []
    evaluations = 0
    for p in candidates.values():
        fits = True
        for h in history:
            evaluations += 1
            if evaluate(p, h['inputs']) != h['output']:
                fits = False
                break
        if fits:
            retained.append(p)
    return retained, dict(roots=len(roots), insertion_attempts=attempts,
                          unique_candidates=len(candidates), retained=len(retained),
                          history_evaluations=evaluations, full_posterior=False)
