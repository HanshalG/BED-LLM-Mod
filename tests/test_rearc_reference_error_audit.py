from scripts.rearc_reference_error_audit import errors


def test_nested_expression_is_recorded_not_compiled():
    graph={'steps':[{'id':'x0','op':'identity','args':['identity(I)']}],'output':'x0'}
    assert errors(graph,{'identity'},set())==[{'kind':'nested_expression','step':0,'value':'identity(I)'}]


def test_forward_and_output_references_remain_failures():
    graph={'steps':[{'id':'x0','op':'identity','args':['x1']}],'output':'x2'}
    assert [r['kind'] for r in errors(graph,{'identity'},set())]==['unknown_reference','output_reference']


def test_rows_are_not_banned():
    graph={'steps':[{'id':'x0','op':'first','args':['I']}],'output':'x0'}
    assert errors(graph,{'first'},set())==[]
