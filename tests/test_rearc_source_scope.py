from scripts.rearc_source_scope import select_ids


def test_scope_uses_only_names_never_executes_source():
    source = 'raise RuntimeError("must not execute")\n'
    source += '\n'.join(f'def verify_{i:08x}(grid):\n    raise RuntimeError("private")' for i in range(8))
    ids, selected = select_ids(source)
    changed = source.replace('private', 'different solution')
    assert select_ids(changed) == (ids, selected)
    assert len(ids) == 8 and len(selected) == 4
