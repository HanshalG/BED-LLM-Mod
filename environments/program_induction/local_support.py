"""Exhaustive one-statement, type-preserving neighborhoods of supplied programs."""
from .prior import program_probability, statement_options
from .proposals import _history


def evaluate(program, inputs):
    state = program.run(inputs)
    return None if state is None else state.get_output()


def expand(dsl, programs, history):
    history = _history(history)
    roots = {}
    for program in programs:
        program_probability(dsl, program)
        if all(evaluate(program, h['inputs']) == h['output'] for h in history):
            roots.setdefault(str(program), program)
    candidates = dict(roots)
    attempts = 0
    for root in roots.values():
        variables = [('x0', list), ('x1', list)]
        for i, old in enumerate(root.statements):
            for op, args in statement_options(dsl, variables, i > 0):
                if op.output_type != old.operation.output_type:
                    continue
                attempts += 1
                statements = list(root.statements)
                statements[i] = dsl.Statement(old.variable, op, args)
                candidate = dsl.Program(['x0', 'x1'], statements)
                program_probability(dsl, candidate)
                candidates.setdefault(str(candidate), candidate)
            variables.append((old.variable, old.operation.output_type))
    retained = []
    evaluations = 0
    for candidate in candidates.values():
        fits = True
        for row in history:
            evaluations += 1
            if evaluate(candidate, row['inputs']) != row['output']:
                fits = False
                break
        if fits:
            retained.append(candidate)
    return retained, dict(compatible_roots=len(roots), substitution_attempts=attempts,
                          unique_candidates=len(candidates), history_evaluations=evaluations,
                          compatible_expanded=len(retained), full_posterior=False)
