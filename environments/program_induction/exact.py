"""Full source-prior prediction by exact execution-prefix mass aggregation."""
from collections import defaultdict
from fractions import Fraction
from time import monotonic

from .prediction import _targets, category
from .prior import statement_options
from .proposals import _history


class EnumerationLimit(RuntimeError):
    def __init__(self, stats):
        super().__init__('full grammar enumeration incomplete; no prediction')
        self.stats = dict(stats)


def predict(dsl, history, targets, *, max_states=100000, max_transitions=1000000,
            seconds=5):
    """Return exact joint target probabilities, or raise without a partial forecast.

    Merge only identical typed execution environments across ALL history/target
    inputs. Distinct syntax multiplicities retain their original probability.
    Targets are inputs only; no target outcomes or hidden program are accepted.
    """
    if (type(max_states) is not int or max_states < 1
            or type(max_transitions) is not int or max_transitions < 1
            or not 0 < seconds < float('inf')):
        raise ValueError('finite positive enumeration limits required')
    history, targets = _history(history), _targets(targets)
    inputs = [r['inputs'] for r in history]+targets
    expected = tuple(category(r['output']) for r in history)
    initial = tuple(tuple(tuple(inp[i]) for inp in inputs) for i in range(2))
    current = {((list, list), initial): Fraction(1)}
    joint, choices = defaultdict(Fraction), {}
    stats = dict(transitions=0, peak_states=1, completed_depth=0)
    start = monotonic()

    def check():
        if monotonic()-start >= seconds:
            raise EnumerationLimit(stats)

    for depth in range(1, 5):
        following = defaultdict(Fraction)
        for (types, values), mass in current.items():
            check()
            if types not in choices:
                variables = [(f'x{i}', typ) for i, typ in enumerate(types)]
                choices[types] = statement_options(dsl, variables, depth > 1)
            options = choices[types]
            if not options:
                raise ValueError('source grammar has no continuation')
            weight = mass/len(options)
            for op, args in options:
                check()
                if stats['transitions'] >= max_transitions:
                    raise EnumerationLimit(stats)
                stats['transitions'] += 1
                outputs = []
                for row in range(len(inputs)):
                    actual = []
                    for arg in args:
                        value = values[int(arg[1:])][row] if isinstance(arg, str) else arg.func
                        actual.append(list(value) if isinstance(value, tuple) else value)
                    output = None if any(v is None for v in actual) else op.run(actual)
                    category(output)
                    outputs.append(tuple(output) if isinstance(output, list) else output)
                outputs = tuple(outputs)
                if depth >= 2:
                    labels = tuple(category(list(x) if isinstance(x, tuple) else x) for x in outputs)
                    if labels[:len(history)] == expected:
                        target = labels[len(history):]
                        if target not in joint and len(current)+len(following)+len(joint) >= max_states:
                            raise EnumerationLimit(stats)
                        joint[target] += weight/3
                if depth < 4:
                    key = (types+(op.output_type,), values+(outputs,))
                    if key not in following and len(current)+len(following)+len(joint) >= max_states:
                        raise EnumerationLimit(stats)
                    following[key] += weight
                stats['peak_states'] = max(stats['peak_states'], len(current)+len(following)+len(joint))
        check()
        stats['completed_depth'] = depth
        current = following
    evidence = sum(joint.values(), Fraction())
    if not evidence:
        raise ValueError('history has zero probability under full source prior')
    return dict(evidence=evidence, probabilities={k: v/evidence for k, v in joint.items()},
                stats=stats, elapsed_seconds=monotonic()-start)
