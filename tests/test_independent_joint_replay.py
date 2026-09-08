import json
import pytest

from scripts import deepcoder_independent_joint_replay as replay


@pytest.mark.parametrize('status',['incomplete','failed_closed'])
def test_no_endpoint_or_source_access_before_complete(tmp_path,monkeypatch,status):
    (tmp_path/'result.json').write_text(json.dumps({'status':status}))
    def bomb():
        raise AssertionError('source/endpoint opened')
    monkeypatch.setattr(replay.g,'load_dsl',bomb)
    with pytest.raises(ValueError,match='complete terminal'):
        replay.verify(tmp_path)
