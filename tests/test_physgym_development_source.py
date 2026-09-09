import pytest
from scripts.physgym_development_source import select


def test_selection_uses_ids_not_content():
    rows = [{'id':i,'solution':'not read'} for i in range(10)]
    assert select(rows)==select([{'id':r['id']} for r in rows[::-1]])
    assert len(select(rows))==4
    with pytest.raises(ValueError):
        select(rows+rows)
