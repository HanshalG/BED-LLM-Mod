"""One-case bounded allocation trace; no replacement reference or source calls."""

import argparse
import json
from pathlib import Path

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from scripts.scilaws_mixed_refinement_audit import fixture


class TracedReference(AdaptiveReference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = 0
        self.outer_callbacks = 0
        self.current_y = None
        self.records = []

    def integrate(self, fn, **kwargs):
        outer = self.level == 0
        self.level += 1

        def tracked(y):
            if outer:
                self.outer_callbacks += 1
                self.current_y = y
            return fn(y)

        try:
            return super().integrate(tracked, **kwargs)
        finally:
            self.level -= 1

    def terminal(self, state, action):
        before = self.evaluations
        row = dict(outer_callback=self.outer_callbacks, observation=self.current_y,
                   action=action, status='incomplete')
        try:
            value, error = super().terminal(state, action)
            row.update(status='completed', value=value, error=error)
            return value, error
        finally:
            row['evaluations'] = self.evaluations - before
            self.records.append(row)


def summarize(ref):
    complete = [r for r in ref.records if r['status'] == 'completed']
    groups = {}
    for r in complete:
        groups.setdefault(r['outer_callback'], {})[r['action']] = r
    decisions = []
    for group in groups.values():
        if set(group) == {0, 1}:
            a, b = group[0], group[1]
            decisions.append(dict(observation=a['observation'],
                                  gap=a['value']-b['value'],
                                  chosen_action=0 if a['value'] <= b['value'] else 1))
    decisions.sort(key=lambda r: r['observation'])
    switches = [(a['observation'], b['observation'])
                for a, b in zip(decisions, decisions[1:])
                if a['chosen_action'] != b['chosen_action']]
    counts = sorted(r['evaluations'] for r in complete)
    return dict(outer_callbacks=ref.outer_callbacks, total_evaluations=ref.evaluations,
                terminal_calls=len(ref.records), completed_terminal_calls=len(complete),
                terminal_evaluations=sum(r['evaluations'] for r in ref.records),
                inner_median_evaluations=counts[len(counts)//2] if counts else None,
                inner_max_evaluations=max(counts) if counts else None,
                sampled_switch_intervals=switches, decisions=decisions)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model, state = fixture(8, ())
    ref = TracedReference(model, predictive_coordinates=True)
    result = dict(case=0, history=[], root_action=0, depth=2,
                  status='incomplete', source_measurements=0, model_calls=0,
                  paid_cost_usd=0, reference_qualified=False,
                  interpretation='one_root_work_allocation_only_not_a_complete_plan')
    try:
        result['root_value'], result['outer_error'] = ref.action(state, 0, 2)
        result['status'] = 'root_complete'
    except (SearchLimitExceeded, ValueError) as exc:
        result['reason'] = str(exc)
    result.update(summary=summarize(ref), records=ref.records)
    with output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
