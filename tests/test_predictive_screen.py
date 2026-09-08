import hashlib
import json
import pytest
from environments.program_induction import predictive_screen as s


def test_manual_brier_and_seal_precedes_labels(tmp_path):
    f = dict(distributions=[{'1':.25,'2':.75}]*32, support_size=2)
    c = dict(targets=[[[1],[2]]]*32)
    panel = dict(cases={str(i):c for i in range(4)},forecasts={str(i):{a:f for a in s.ARMS} for i in range(4)})
    path = tmp_path/'panel.json'
    path.write_text(json.dumps(panel))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    truth = {str(i):dict(target_inputs=c['targets'],outputs=[1]*32) for i in range(4)}
    r = s.score_sealed(path,sha,lambda:truth)
    assert r['mean_brier']['history_aware'] == pytest.approx(.5625)
    def bomb():
        raise AssertionError('labels opened')
    with pytest.raises(ValueError,match='identity'):
        s.score_sealed(path,'wrong',bomb)
    panel['forecasts']['0'].pop('history_blind')
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError,match='five-arm'):
        s.score_sealed(path,hashlib.sha256(path.read_bytes()).hexdigest(),bomb)
