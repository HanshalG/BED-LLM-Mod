import pytest
from scripts.author_strings_audit import inspect, parse


def test_character_facts_without_execution():
    r=inspect("in(e1,1,'a'). width(e1,1).", "pos(out(e1,1,'A')). neg(out(e1,1,'a')).")
    assert r['examples']==1 and r['positive_facts']==1
    assert not r['complete_function_outside_examples']
    assert parse("char('''').")==[('char',("'",))]
    with pytest.raises(Exception):
        parse(':- shell(touch_file).')


@pytest.mark.parametrize('bk,exs',[
    ("in(e1,1,'a'). width(e1,1).", "pos(out(e1,1,'a')). neg(out(e1,1,'a'))."),
    ("in(e1,2,'a'). width(e1,1).", "pos(out(e1,1,'a'))."),
    ("in(e1,1,'a'). width(e1,2).", "pos(out(e1,1,'a'))."),
    ("in(e1,1,'a'). width(e1,1).", ""),
])
def test_conflicts_gaps_and_missing_output_rejected(bk,exs):
    with pytest.raises(ValueError):
        inspect(bk,exs)
