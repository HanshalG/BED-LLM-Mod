from scripts.mqtt_opportunity_run import menus,aggregate


def test_target_menu_is_label_free_and_complete():
    g=[{'scenario':'a','input_alphabet':['x','y']},{'scenario':'b','input_alphabet':['z']}]
    a=menus(g)
    assert a==menus(g) and len(a)==2
    assert all(len(r['words'])==512 for r in a)
    assert all(len(w)==6 for r in a for w in r['words'])
    assert a[1]['words']==[[0]*6]*512


def test_incomplete_and_plateau_never_pass():
    assert not aggregate([])['opportunity_passed']
    rows=[{'status':'complete','initial_risk':'1/2','risk':
           {k:'1/4' for k in ('h1','h2','h3','adaptive_full','committed','random')}}]*8
    assert not aggregate(rows)['opportunity_passed']


def test_all_gates_are_conjoined():
    r={'status':'complete','initial_risk':'1/2','risk':{'h1':'2/5','h2':'3/10','h3':'1/5',
        'adaptive_full':'1/10','committed':'1/5','random':'9/20'}}
    assert aggregate([r]*8)['opportunity_passed']
    r={**r,'risk':{**r['risk'],'committed':'1/10'}}
    assert not aggregate([r]*8)['opportunity_passed']
