from scripts.rearc_exact_repair_source import schedule


def test_exact_channel_schedule():
    cohort={'selected_ids':[f'{i:08x}' for i in range(6)],'demo_seeds':[44100],
        'query_seeds':[44200,44201],'target_seeds':list(range(44300,44308))}
    rows=schedule(cohort)
    assert len(rows)==66
    assert sum(r['mode']=='demonstration' for r in rows)==6
    assert sum(r['mode']=='input' for r in rows)==60
    assert rows[:3]==[{'task':'00000000','seed':s,'mode':m} for s,m in
        [(44100,'demonstration'),(44200,'input'),(44201,'input')]]
    assert len({(r['task'],r['seed']) for r in rows})==66
