"""Bounded CrossBeam search over the pinned DeepCoder interpreter.

This returns one history-compatible expression, not a posterior distribution.
Only observed examples enter search; new inputs can be evaluated afterwards.
"""
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import io
import random
import sys
from time import monotonic
import types
import urllib.request


PIN = 'c43cb523fa9887513fb18bf088eb588f3fec5835'
FILES = {
    'crossbeam.dsl.value': ('crossbeam/dsl/value.py',
                          '064bf03692832ea814bf6a5301553f92143f730b218a2a733359ee37e555a1ee'),
    'crossbeam.algorithm.baseline_enumeration': ('crossbeam/algorithm/baseline_enumeration.py',
                          '9f81d63a56ba98f7599152663b06ee16533eb9019d03ba36c94944c4380b5a61'),
}


@lru_cache(maxsize=1)
def load_search():
    names = ('crossbeam', 'crossbeam.dsl', *FILES)
    if any(name in sys.modules for name in names):
        raise RuntimeError('refuse to replace an existing CrossBeam import')
    modules = {}
    try:
        for name in names:
            module = types.ModuleType(name)
            modules[name] = sys.modules[name] = module
            if name in FILES:
                path, expected = FILES[name]
                url = f'https://raw.githubusercontent.com/google-research/crossbeam/{PIN}/{path}'
                with urllib.request.urlopen(url, timeout=30) as response:
                    source = response.read(100000)
                if hashlib.sha256(source).hexdigest() != expected:
                    raise ValueError('CrossBeam source identity mismatch')
                exec(compile(source, url, 'exec'), module.__dict__)
            else:
                module.__path__ = []
            if name == 'crossbeam.dsl.value':
                modules['crossbeam.dsl'].value = module
        return modules['crossbeam.dsl.value'], modules['crossbeam.algorithm.baseline_enumeration']
    finally:
        for name, module in modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


class SynthesisLimit(RuntimeError):
    pass


def evaluate_expression(expression, inputs):
    """Evaluate the found tree directly; never evaluate a generated code string."""
    if hasattr(expression, 'name'):
        return inputs[int(expression.name[1:])]
    args = [evaluate_expression(child, inputs) for child in expression.arg_values]
    return expression.operation.single(args)


def synthesize(dsl, history, *, max_attempts=2048, max_weight=9, seconds=5, order_seed=None):
    """Search one compatible expression, explicitly exposing bounded failures.

    `history` contains (two-list input, raw int/list/None output) pairs. None is
    the DSL's observed ERROR, not a missing example. The vocabulary is public;
    no truth program, held-out inputs, or held-out labels are arguments here.
    """
    if not history:
        raise ValueError('observed examples required')
    for value in (max_attempts, max_weight):
        if type(value) is not int or value < 1:
            raise ValueError('positive integer search limits required')
    if not isinstance(seconds, (int, float)) or not 0 < seconds < float('inf'):
        raise ValueError('positive finite time limit required')
    if order_seed is not None and type(order_seed) is not int:
        raise ValueError('operation ordering seed must be an integer')
    for inputs, observed in history:
        if (not isinstance(inputs, list) or len(inputs) != 2
                or any(type(xs) is not list or len(xs) > 5 or not dsl.validate_result(xs) for xs in inputs)
                or (observed is not None and not dsl.validate_result(observed))):
            raise ValueError('invalid DeepCoder history')
    values, search = load_search()
    start, attempts, evaluations = monotonic(), 0, 0

    class BoundOperation:
        weight = 1

        def __init__(self, op, lam=None):
            self.op, self.lam = op, lam
            self.types = tuple(op.inputs_type[1:] if lam else op.inputs_type)
            self.arity = len(self.types)
            self.name = op.token + (lam.token if lam else '')

        def arg_types(self):
            return self.types

        def single(self, args):
            if any(arg is None for arg in args):
                return None
            return self.op.run(([self.lam.func] if self.lam else []) + args)

        def apply(self, args):
            nonlocal attempts, evaluations
            if attempts >= max_attempts or monotonic()-start > seconds:
                raise SynthesisLimit('operation or time budget exhausted')
            attempts += 1
            outputs = []
            for example in range(len(history)):
                evaluations += 1
                outputs.append(self.single([arg[example] for arg in args]))
            if monotonic()-start > seconds:
                raise SynthesisLimit('time budget exhausted')
            result = values.OperationValue(outputs, self, args)
            # First observed output may be ERROR even for a well-typed operation.
            result.type = self.op.output_type
            return result

        def tokenized_expression(self, args):
            result = [self.name, '(']
            for i, arg in enumerate(args):
                if i:
                    result.append(',')
                result.extend(arg.tokenized_expression())
            return result+[')']

    operations = []
    for op in dsl.OPERATIONS:
        if isinstance(op, dsl.HigherOrderOperation):
            for lam in dsl.LAMBDAS:
                if (lam.inputs_type, lam.output_type) == op.inputs_type[0]:
                    operations.append(BoundOperation(op, lam))
        else:
            operations.append(BoundOperation(op))
    if order_seed is not None:
        random.Random(order_seed).shuffle(operations)
    task = types.SimpleNamespace(num_examples=len(history),
                                 inputs_dict={f'x{i}': [inp[i] for inp, _ in history] for i in range(2)},
                                 outputs=[out for _, out in history])
    domain = types.SimpleNamespace(constants=[], constants_extractor=None,
                                   operations=operations, small_value_filter=None)
    found, status = None, 'search_exhausted'
    try:
        # Suppress upstream progress, which is not part of an experiment artifact.
        with redirect_stdout(io.StringIO()):
            found, _, _, _ = search.synthesize_baseline(
                task, domain, max_weight=max_weight, timeout=seconds,
                max_values_explored=max_attempts)
        if monotonic()-start > seconds:
            raise SynthesisLimit('time budget exhausted')
        if found is not None:
            if any(evaluate_expression(found, inp) != out for inp, out in history):
                raise ValueError('found expression fails independent history replay')
            if monotonic()-start > seconds:
                raise SynthesisLimit('time budget exhausted during verification')
            status = 'compatible_expression_found'
        elif attempts >= max_attempts:
            status = 'budget_exhausted'
    except SynthesisLimit:
        found, status = None, 'budget_exhausted'
    finally:
        search.generate_partitions.cache_clear()
    return dict(status=status, expression=found, operation_attempts=attempts,
                example_evaluations=evaluations, elapsed_seconds=monotonic()-start,
                posterior_samples=False, paid_calls_authorized=False)
