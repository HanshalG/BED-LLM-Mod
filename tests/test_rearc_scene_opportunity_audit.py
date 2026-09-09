from scripts.rearc_scene_opportunity_audit import summarize


def test_unsupported_is_not_false_certainty():
    result=summarize({'conditioning':{'failed':True}})
    assert result['status']=='unsupported' and result['internal_risk'] is None


def test_syntactic_diversity_does_not_imply_predictive_diversity():
    f={'conditioning':{'failed':False,'consistent_programs':2},
       'weights':[.5,.5],'outputs':[[[[1]],[[1]]]]*9}
    r=summarize(f)
    assert r['unanimous_all_public_inputs'] and r['actual_correctness']=='unknown_sealed'
    assert all(d['internal_half_brier_risk']==0 for d in r['distributions'])


def test_predictive_disagreement_and_failure_are_distinct():
    f={'conditioning':{'failed':False,'consistent_programs':2},
       'weights':[.5,.5],'outputs':[[[[1]],[[2]]],[None,[[2]]]]}
    r=summarize(f)
    assert not r['unanimous_all_public_inputs']
    assert r['distributions'][0]['internal_half_brier_risk']==.25
    assert r['distributions'][1]['failure_probability']==.5
