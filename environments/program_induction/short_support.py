"""Exact two-statement component, not the full 2/3/4-statement source prior."""
from .local_support import evaluate
from .prior import statement_options
from .proposals import _history


def enumerate_two_statement(dsl, history):
    """Enumerate every source-valid two-statement syntax before history filtering.

    First-output types may change; the second statement is constructed against
    the new type rather than patched in place. No LLM roots or target labels enter.
    """
    history = _history(history)
    retained = []
    candidates = evaluations = 0
    initial = [('x0',list),('x1',list)]
    suffixes = {}
    for op,args in statement_options(dsl,initial,False):
        if op.output_type not in suffixes:
            suffixes[op.output_type] = statement_options(dsl,initial+[('x2',op.output_type)],True)
        for next_op,next_args in suffixes[op.output_type]:
            p = dsl.Program(['x0','x1'],[dsl.Statement('x2',op,args),dsl.Statement('x3',next_op,next_args)])
            candidates += 1
            fits = True
            for h in history:
                evaluations += 1
                if evaluate(p,h['inputs']) != h['output']:
                    fits = False
                    break
            if fits:
                retained.append(p)
    return retained,dict(candidates=candidates,history_executions=evaluations,
        compatible=len(retained),complete_two_statement_component=True,full_source_posterior=False)
