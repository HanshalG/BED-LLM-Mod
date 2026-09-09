from scripts.rearc_source_smoke import selected_functions, SEEDS


def test_extracts_only_frozen_functions_without_execution():
    text = 'raise RuntimeError("not executed")\ndef a(): return 1\ndef secret(): return 99'
    assert selected_functions(text, {'a'}) == 'def a(): return 1'
    assert SEEDS == (31000,31001,31002,31003)
