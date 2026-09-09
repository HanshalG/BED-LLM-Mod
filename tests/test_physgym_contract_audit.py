from scripts.physgym_contract_audit import calls


def test_static_calls_never_execute_and_preserve_flags():
    result = calls("raise RuntimeError('must not run')\nf(x, sandbox=True, fast_local=True)")
    assert result[1] == dict(line=2,callee='f',keywords={'sandbox':'True','fast_local':'True'})
