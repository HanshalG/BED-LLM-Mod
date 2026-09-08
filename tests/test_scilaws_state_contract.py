import pickle

import joblib
import numpy as np
import pytest

from scripts.scilaws_state_contract import ResidualState, inspect_state


def fixture():
    bootstrap = ResidualState()
    bootstrap._res = np.array([-0.1, 0.1])
    return dict(
        used_inputs=["x"],
        support={"x": {"min": 0.0, "max": 1.0}},
        noise_scale=1.0,
        fetch_budget_rows=10,
        residual_space="linear",
        bootstrap=bootstrap,
        formula_source="SECRET_FORMULA",
        law_constants={"a": 93847},
        real_rows=[{"y": 93847}],
    )


@pytest.mark.parametrize("compress", [0, 3])
def test_safe_projection_excludes_hidden_values(tmp_path, compress):
    path = tmp_path / "state.joblib"
    joblib.dump(fixture(), path, compress=compress)
    result = inspect_state(path)
    assert result["status"] == "source_contract_valid"
    assert result["bounds"] == {"x": [0.0, 1.0]}
    assert "SECRET_FORMULA" not in str(result)
    assert "93847" not in str(result)
    assert result["measurements_generated"] == 0


def test_arbitrary_pickle_global_is_rejected(tmp_path):
    class Bad:
        def __reduce__(self):
            return (eval, ("1+1",))

    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(Bad()))
    with pytest.raises(ValueError, match="unapproved"):
        inspect_state(path)


def test_object_array_is_rejected(tmp_path):
    state = fixture()
    state["bootstrap"]._res = np.array([object()], dtype=object)
    path = tmp_path / "state.joblib"
    joblib.dump(state, path)
    with pytest.raises(ValueError, match="object"):
        inspect_state(path)
