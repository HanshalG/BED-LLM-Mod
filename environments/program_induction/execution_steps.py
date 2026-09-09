"""Public-history, source-typed execution steps for a future decomposed proposer.

This is not a posterior or a search policy. Menu indices encode syntax choices,
never likelihood weights; proposed subgoals cannot overwrite interpreter results.
"""
from .local_support import evaluate
from .prior import statement_options
from .proposals import _history


def _compile(dsl, choices):
    if type(choices) is not list or len(choices) > 4:
        raise ValueError('zero to four source choice indices required')
    variables = [('x0', list), ('x1', list)]
    statements = []
    for i, choice in enumerate(choices):
        options = statement_options(dsl, variables, i > 0)
        if type(choice) is not int or not 0 <= choice < len(options):
            raise ValueError('invalid source choice index')
        op, args = options[choice]
        statements.append(dsl.Statement(f'x{i+2}', op, args))
        variables.append((f'x{i+2}', op.output_type))
    return variables, statements


def public_state(dsl, choices, history):
    """Replay a prefix on observed inputs only; ERROR is absorbing, not pruned."""
    history = _history(history)
    variables, statements = _compile(dsl, choices)
    rows = []
    for row in history:
        values = dict(zip(['x0', 'x1'], row['inputs']))
        failed = False
        for i, statement in enumerate(statements):
            value = evaluate(dsl.Program(['x0', 'x1'], statements[:i+1]), row['inputs'])
            values[statement.variable] = value
            failed = failed or value is None
        output = values[statements[-1].variable] if statements else None
        rows.append(dict(values=values, execution_error=failed, expected=row['output'],
                         matches=bool(statements) and output == row['output']))
    # Empty history is not evidence that a prefix solves an observed task.
    return dict(choices=list(choices), types={k: v.__name__ for k, v in variables},
                rows=rows, can_stop=len(statements) >= 2,
                fits_observations=bool(rows) and all(r['matches'] for r in rows))


def next_menu(dsl, choices):
    """Enumerate type-valid next syntax, with no execution or target access."""
    variables, statements = _compile(dsl, choices)
    if len(statements) == 4:
        return []
    return [dict(choice=i, statement=str(dsl.Statement(f'x{len(statements)+2}', op, args)))
            for i, (op, args) in enumerate(statement_options(dsl, variables, bool(statements)))]


def completed_program(dsl, choices):
    """Reject early stopping before returning a source-valid executable program."""
    _, statements = _compile(dsl, choices)
    if len(statements) < 2:
        raise ValueError('source programs need at least two steps')
    return dsl.Program(['x0', 'x1'], statements)
