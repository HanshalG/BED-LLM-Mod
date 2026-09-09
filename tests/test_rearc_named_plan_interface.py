import itertools
import json
import pytest
from jsonschema import Draft202012Validator
from scripts.rearc_named_plan_interface import parse_plan,parse_programs,plan_diagnostics,messages,schema,FIELDS


def plan(text='rule'):
    return dict.fromkeys(FIELDS,text)


@pytest.mark.parametrize('token',['a','\u2014','\u00e9','\U0001f642','e\u0301'])
@pytest.mark.parametrize('size',[0,1,511,512,513])
def test_schema_parser_character_boundary_parity(token,size):
    text=(token*513)[:size]
    value=plan(text)
    expected=Draft202012Validator(schema('plan')['json_schema']['schema']).is_valid(value)
    for ascii_encoding in (True,False):
        encoded=json.dumps(value,ensure_ascii=ascii_encoding)
        if expected:
            assert parse_plan(encoded)==value
        else:
            with pytest.raises(ValueError,match='schema invalid'):
                parse_plan(encoded)


def test_all_field_permutations_canonicalize_without_text_changes():
    value={key:'explanation '+key for key in FIELDS}
    for order in itertools.permutations(FIELDS):
        result=parse_plan(json.dumps({key:value[key] for key in order}))
        assert result==value and list(result)==list(FIELDS)


@pytest.mark.parametrize('value',[{}, {'p0':'a'}, {**plan(),'extra':'x'},plan(2),plan(None),plan([])])
def test_invalid_structures_agree_with_schema(value):
    assert not Draft202012Validator(schema('plan')['json_schema']['schema']).is_valid(value)
    with pytest.raises(ValueError,match='schema invalid'):
        parse_plan(json.dumps(value))


def test_blank_is_schema_valid_but_not_semantic_evidence():
    assert parse_plan(json.dumps(plan(' ')))==plan(' ')
    assert plan_diagnostics(plan(' '))['blank_fields']==list(FIELDS)


def test_duplicate_json_keys_rejected_and_schema_copy_is_isolated():
    with pytest.raises(ValueError,match='duplicate'):
        parse_plan('{"p0":"x","p0":"y","p1":"x","p2":"x","p3":"x"}')
    changed=schema('plan')
    changed['json_schema']['schema']['properties']['p0']['maxLength']=1
    assert parse_plan(json.dumps(plan('long')))==plan('long')


def test_unicode_compiler_context_and_executable_failure_are_separate():
    dsl='def identity(x: Any) -> Any:\n return x\n'
    value=plan('\U0001f642'*512)
    prompt=messages('compile','contrasting',inputs=[[[0]]],observations=[{'index':0,'output':[[0]]}],
                    dsl_source=dsl,plan=value)
    assert json.loads(prompt[1]['content'])['plan']==value
    assert len(json.dumps(prompt).encode())<65536
    text=json.dumps({'hypotheses':['unknown(I)']*8})
    assert Draft202012Validator(schema('compile')['json_schema']['schema']).is_valid(json.loads(text))
    with pytest.raises(ValueError,match='unknown function'):
        parse_programs(text,dsl)


@pytest.mark.parametrize('mode',['contrasting','ordinary'])
def test_named_plan_full_update_uses_same_schema_and_keeps_slots(mode):
    from scripts.rearc_named_plan_update import update
    from scripts.rearc_named_plan_contract import body
    calls=[]
    def request(stage,prompt,format):
        value=body(prompt,40400+len(calls),format)
        calls.append((stage,value))
        return json.dumps(plan('\u2014'*512) if stage=='plan' else {'hypotheses':['identity(I)']*8})
    result=update(mode=mode,inputs=[[[0]],[[1]]],observations=[{'index':0,'output':[[0]]}],
        dsl_source='def identity(x: Any) -> Any:\n return x\n',request=request,
        diagnose=lambda g,x:{'status':'ok','output':x})
    assert [stage for stage,_ in calls]==['plan','compile','repair']
    assert len(result['slots'])==16
    assert result['plan']==plan('\u2014'*512)
    assert result['plan_diagnostics']['blank_fields']==[]
    assert result['plan_diagnostics']['semantic_correctness']=='not_established_by_schema'
