"""Observed-example execution feedback; no unobserved outcomes are accepted."""
import json

from . import constrained, constrained_request
from .local_support import evaluate
from .prediction import canonical, category
from .prior import program_probability
from .proposals import _history


def checks(dsl, programs, history):
    history = _history(history)
    if not 1 <= len(programs) <= 8:
        raise ValueError('one to eight source programs required')
    rows = []
    for program in programs:
        program_probability(dsl, program)
        observed = []
        for i, example in enumerate(history):
            actual = evaluate(program, example['inputs'])
            # Category comparison preserves ERROR, integer and list distinctions.
            passed = category(actual) == category(example['output'])
            observed.append(dict(example_index=i, actual_output=actual,
                                 expected_output=example['output'], passed=passed))
        rows.append(dict(checks=observed, fits_observed_history=all(r['passed'] for r in observed)))
    return rows


def revision_request(dsl, programs, history, seed, *, with_feedback):
    if type(with_feedback) is not bool:
        raise ValueError('explicit feedback mode required')
    evidence = checks(dsl, programs, history)
    body = constrained_request.request(dsl, history, seed, history_blind=False)
    body['messages'][0]['content'] += (
        ' Previous candidate programs are provided as starting points for revision. '
        'When present, execution_feedback is produced by the trusted interpreter '
        'on the displayed examples only. Correct inconsistencies and retain plausible '
        'behaviorally different alternatives. Matching these examples does not prove '
        'correctness on unseen inputs. Return a complete program pool in the same schema.'
    )
    public = json.loads(body['messages'][1]['content'])
    public['previous_programs'] = constrained.encode(dsl, programs)['programs']
    if with_feedback:
        public['execution_feedback'] = evidence
    body['messages'][1]['content'] = canonical(public)
    if len(canonical(body).encode()) > 32768:
        raise ValueError('complete revision request exceeds byte cap')
    return body
