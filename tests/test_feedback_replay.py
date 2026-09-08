import json
import pytest
from scripts import deepcoder_feedback_replay as r


@pytest.mark.parametrize('status',['incomplete','failed_closed'])
def test_terminal_guard_before_source_access(tmp_path,monkeypatch,status):
    (tmp_path/'result.json').write_text(json.dumps({'status':status}))
    def bomb():
        raise AssertionError('source opened')
    monkeypatch.setattr(r.g,'load_dsl',bomb)
    with pytest.raises(ValueError,match='complete terminal'):
        r.verify(tmp_path)
