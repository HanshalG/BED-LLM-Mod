from scripts.herb_support_collector import collect


def test_multiple_survivors_duplicates_and_exact_cap():
    calls = []
    def execute(graph, value):
        calls.append(graph)
        return {'status': 'ok', 'output': value}
    def candidates():
        yield from ['identity(I)', 'vmirror(I)', 'identity(I)', 'unknown(I)']
        raise AssertionError('overran candidate cap')
    result = collect(candidates(), [{'input': [[0, 0]], 'output': [[0, 0]]}],
                     {'identity', 'vmirror'}, set(), execute, attempts=4)
    assert result['posterior']['weights'] == [.5, .5]
    assert len(calls) == 2
    assert result['attempts'] == 4
    assert [r['status'] for r in result['records']] == ['evaluated', 'evaluated', 'duplicate', 'invalid_expression']


def test_empty_support_is_failure_not_prior_reset():
    result = collect(['bad(I)'], [{'input': [[0]], 'output': [[0]]}],
                     {'identity'}, set(), lambda *args: None, attempts=1)
    assert result['posterior']['failed']
    assert result['posterior']['weights'] == []
