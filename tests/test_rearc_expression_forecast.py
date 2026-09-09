from scripts.rearc_expression_forecast import forecast
from scripts.rearc_qualification_panel import forecast as exhaustive
from scripts.rearc_predictive_score import mixture_scores


def test_short_circuit_preserves_scores_and_failure_mass():
    graphs = [{'name':name} for name in ('bad','good','uncertain','good')]
    case = {'inputs':[[[0]],[[1]],[[2]]], 'outputs':[[[0]],[[1]],[[2]]],
            'target_inputs':[[[3+i%7]] for i in range(8)]}
    def evaluate(graph, inputs):
        return [None if graph['name']=='bad' or (graph['name']=='uncertain' and x[0][0]>=3) else x for x in inputs]
    expected = exhaustive(graphs,case,evaluate)
    calls = []
    def counted(graph,inputs):
        calls.append((graph['name'],len(inputs)))
        return evaluate(graph,inputs)
    actual = forecast(graphs,case,counted)
    assert actual['weights'] == expected['weights']
    assert actual['demonstrations_checked'] == [1,3,3,3]
    assert sum(n for name,n in calls if name=='bad') == 1
    for a,b,truth in zip(actual['outputs'],expected['outputs'],case['target_inputs']):
        assert mixture_scores(a,truth,actual['weights']) == mixture_scores(b,truth,expected['weights'])
