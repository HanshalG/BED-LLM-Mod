"""Exact public fixture: demonstration equivalence is not BED equivalence."""
import json
from pathlib import Path
from scripts.rearc_program_weights import condition_programs
from scripts.rearc_predictive_score import mixture_scores


def diagnostic():
    graphs=[{'steps':[{'id':'x0','op':op,'args':['I']}],'output':'x0'} for op in ('identity','vmirror')]
    demonstration=[[0,0]]
    possible_answers=[[[1,2]],[[2,1]]]
    belief=condition_programs(graphs,[[demonstration],[demonstration]],[demonstration])
    prior_risk=sum(mixture_scores(possible_answers,y,belief['weights'])['whole_grid_brier']/2 for y in possible_answers)
    after_risk=0.0
    for answer in possible_answers:
        posterior=condition_programs(graphs,[[demonstration,y] for y in possible_answers],[demonstration,answer])
        after_risk+=mixture_scores(possible_answers,answer,posterior['weights'])['whole_grid_brier']/2
    collapsed_predicted=mixture_scores([possible_answers[0]],possible_answers[0],[1.])['whole_grid_brier']
    collapsed_actual=sum(mixture_scores([possible_answers[0]],y,[1.])['whole_grid_brier']/2 for y in possible_answers)
    return {'retained_programs':belief['unique_programs'],'retained_weights':belief['weights'],
            'correct_prior_expected_brier':prior_risk,'correct_after_query_expected_brier':after_risk,
            'query_value':prior_risk-after_risk,'merged_predicted_brier':collapsed_predicted,
            'merged_actual_expected_brier':collapsed_actual,
            'model_calls':0,'cost_usd':0,'benchmark_examples':0,
            'scope':'exact regression for destructive demonstration-only equivalence; not a depth result or Herb runtime test'}


if __name__=='__main__':
    result=diagnostic()
    with Path('results/nonmyopic/BED_SYNTHESIS_EQUIVALENCE_AUDIT_20260909.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))
