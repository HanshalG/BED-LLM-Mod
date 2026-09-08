"""Retrospective self-risk versus realized proper loss on banked Luna forecasts."""
import hashlib
import json
from pathlib import Path

from environments.program_induction import prediction as pred
from environments.program_induction import transition_screen as screen
from environments.program_induction.rollout_risk import expected_brier

ROOT = Path('results/nonmyopic/deepcoder_luna_transition_20260909')
TERMINAL_SHA = 'a3aa0e8cb9cc92012b4419817c0108befdd9983ed1494e2b8e3701221cc53942'


def main():
    terminal_bytes = (ROOT/'result.json').read_bytes()
    if hashlib.sha256(terminal_bytes).hexdigest() != TERMINAL_SHA:
        raise ValueError('banked terminal changed')
    terminal = json.loads(terminal_bytes)
    replay = screen.score_sealed(ROOT/'forecasts.json', terminal['forecast_sha256'],
                                 lambda: json.loads((ROOT/'outcomes.json').read_text()))
    if replay['scores'] != terminal['scores']:
        raise ValueError('banked score replay failed')
    panel = json.loads((ROOT/'forecasts.json').read_text())
    outcomes = json.loads((ROOT/'outcomes.json').read_text())
    cases = {}
    for key, arms in panel['forecasts'].items():
        cases[key] = {}
        for arm, forecast in arms.items():
            labels = [pred.category(y) for y in outcomes[key]['outputs']]
            forecasts = forecast['distributions'] if forecast else [None]*len(labels)
            rows = [expected_brier({y: 1.}, q) for y, q in zip(labels, forecasts)]
            loss = sum(r['expected_loss'] for r in rows)/len(rows)
            if abs(loss-terminal['scores'][key][arm]['brier']) > 1e-12:
                raise ValueError('proper loss mismatch')
            cases[key][arm] = dict(realized_brier=loss, abstained=forecast is None,
                forecast_self_risk=(sum(r['forecast_self_risk'] for r in rows)/len(rows)
                                    if forecast else None))
    report = dict(cases=cases, parent_terminal_sha256=TERMINAL_SHA,
                  forecast_sha256=terminal['forecast_sha256'],
                  outcomes_sha256=hashlib.sha256((ROOT/'outcomes.json').read_bytes()).hexdigest(),
                  model_calls=0, cost_usd=0, new_hidden_answers_opened=False,
                  interpretation='retrospective_scoring_diagnostic_not_calibration_or_policy_test',
                  depth_authorized=False)
    path = Path('results/nonmyopic/DEEPCODER_ROLLOUT_RISK_AUDIT_20260909.json')
    with path.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    main()
