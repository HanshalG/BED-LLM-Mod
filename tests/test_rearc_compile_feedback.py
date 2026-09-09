import json
from scripts.rearc_compile_feedback import compiler_feedback

DSL = 'def identity(x: Any) -> Any:\n return x\n'


def test_repair_receives_location_and_explanation_not_salvaged_programs():
    value = {'hypotheses':['identity(I)']*7+['__bed_call1(I)']}
    result = compiler_feedback(json.dumps(value), DSL)
    assert result['accepted_slots'] == 0
    assert result['slot_errors'][0]['slot'] == 7
    assert result['slot_errors'][0]['error'] == 'computed call arity'
    assert '__bed_call1(identity,I)' in result['slot_errors'][0]['hint']


def test_feedback_never_echoes_untrusted_text_and_is_bounded():
    secret = 'DO_NOT_ECHO_THIS'
    result = compiler_feedback(json.dumps({'hypotheses':[secret+'(']*8}), DSL)
    assert secret not in json.dumps(result)
    assert len(json.dumps(result).encode()) < 4096
    assert len(result['slot_errors']) == 8


def test_valid_structure_is_not_semantic_success():
    result = compiler_feedback(json.dumps({'hypotheses':['identity(I)']*8}), DSL)
    assert result == {'status':'structurally_valid','semantic_validity':'not_tested'}
