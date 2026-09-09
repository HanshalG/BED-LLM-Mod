import pytest
from scripts.mqtt_source_metadata import metadata

DOT=b'digraph { __start0 [label=""]; s0; s1; __start0 -> s0; s0 -> s1 [label="arm / ack"]; s1 -> s1 [label="arm / ack"]; }'


def test_graphviz_metadata_without_response_trace():
    r=metadata(DOT)
    assert r=={'states':2,'reachable_states':2,'transitions':2,'input_alphabet':['arm'],
        'output_symbols':1,'total_deterministic':True,'explicit_reset_state':True}


def test_duplicate_and_incomplete_rejected():
    with pytest.raises(ValueError,match='deterministic'):
        metadata(DOT.replace(b'}',b's0 -> s0 [label="arm / other"]; }'))
    with pytest.raises(ValueError,match='incomplete'):
        metadata(DOT.replace(b'}',b's0 -> s0 [label="probe / yes"]; }'))


def test_missing_start_rejected():
    with pytest.raises(ValueError,match='exactly one start'):
        metadata(DOT.replace(b'__start0 -> s0;',b''))
