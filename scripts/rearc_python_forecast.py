"""Use the existing fixed-slot predictive scorer for native Python programs."""
import ast
from scripts.rearc_python_contract import validate
from scripts.rearc_slot_forecast import forecast_slots


def forecast_python_slots(sources, case, evaluate_code):
    if not sources:
        raise ValueError('nonempty attempted slots required')
    representatives={}
    slots=[]
    for i,code in enumerate(sources):
        if code is None:
            key=None
        else:
            # Formatting/comments cannot multiply the prior mass of identical ASTs.
            key=ast.dump(validate(code),include_attributes=False)
            representatives.setdefault(key,code)
        slots.append({'slot':i,'graph':None if key is None else {'language':'python','ast':key}})
    def evaluate(identity,inputs):
        return evaluate_code(representatives[identity['ast']],inputs)
    return forecast_slots(slots,case,evaluate)
