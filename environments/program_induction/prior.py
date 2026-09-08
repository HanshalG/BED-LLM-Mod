"""Exact syntactic probabilities for the frozen active-DeepCoder source prior."""
from fractions import Fraction
from itertools import product


def statement_options(dsl, variables, require_previous):
    """The same finite type-valid choice set used by the source sampler.

    No executions, histories or target values enter the choice count. Returned
    tuples keep distinct variable references even if their runtime values agree.
    """
    options = []
    for op in dsl.OPERATIONS:
        argument_options = []
        for typ in op.inputs_type:
            if isinstance(typ, tuple):
                argument_options.append([lam for lam in dsl.LAMBDAS
                                         if (lam.inputs_type, lam.output_type) == typ])
            else:
                argument_options.append([name for name, t in variables if t == typ])
        for args in product(*argument_options):
            if not require_previous or variables[-1][0] in args:
                options.append((op, args))
    return options


def program_probability(dsl, program):
    """Return P(program syntax), not mass of its semantic equivalence class."""
    if program.input_variables != ['x0', 'x1'] or not 2 <= len(program.statements) <= 4:
        raise ValueError('program outside frozen input/length prior')
    variables = [('x0', list), ('x1', list)]
    probability = Fraction(1, 3)
    for i, statement in enumerate(program.statements):
        if statement.variable != f'x{i+2}':
            raise ValueError('noncanonical variable naming')
        options = statement_options(dsl, variables, require_previous=i > 0)
        if (statement.operation, tuple(statement.args)) not in options:
            raise ValueError('statement outside frozen source prior')
        probability /= len(options)
        variables.append((statement.variable, statement.operation.output_type))
    return probability


def restricted_weights(dsl, programs):
    """Normalize source prior on distinct supplied syntax, WITHOUT a coverage claim.

    If supplied programs already fit all deterministic observations, these are
    posterior weights conditional on restricting the model to that set. The
    caller must verify consistency. Selection by data/search is not corrected,
    and this is not full-grammar posterior inference or evidence estimation.
    """
    probabilities = {}
    for program in programs:
        probability = program_probability(dsl, program)
        probabilities.setdefault(str(program), probability)
    if not probabilities:
        raise ValueError('nonempty candidate set required')
    total = sum(probabilities.values(), Fraction())
    return {key: value/total for key, value in probabilities.items()}
