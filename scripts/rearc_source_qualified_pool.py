"""Bounded source-only selection; never score a model or expose hidden outputs."""
import json
from pathlib import Path
from scripts.rearc_public_source_journal import collect, replay, save, validate_schedule


def validate(schedules, needed):
    if type(needed) is not int or not 1 <= needed <= len(schedules) <= 24:
        raise ValueError('bounded pool size')
    tasks = []
    for schedule in schedules:
        validate_schedule(schedule)
        task = schedule[0]['task']
        if len(schedule)!=11 or any(r['task']!=task for r in schedule):
            raise ValueError('one task and eleven channels')
        if [r['mode'] for r in schedule]!=['demonstration']+['input']*10:
            raise ValueError('exact public channel order')
        tasks.append(task)
    if len(set(tasks))!=len(tasks):
        raise ValueError('distinct candidate tasks')


def summarize(root, schedules, needed, processed):
    accepted, rejected = [], []
    terminal = None
    for i in range(processed):
        if terminal is not None:
            raise ValueError('work after terminal boundary')
        journal = root/f'{i:02d}'
        replay(journal)
        if json.loads((journal/'schedule.json').read_text())!=schedules[i]:
            raise ValueError('candidate schedule identity')
        result = json.loads((journal/'result.json').read_text())
        task = schedules[i][0]['task']
        if result['status']=='public_schedule_complete':
            accepted.append({'task':task,'candidate_index':i})
            if len(accepted)==needed:
                terminal='source_pool_qualified'
        else:
            failure=json.loads((journal/f"{result['failed_index']:03d}.failure.json").read_text())
            # Skip only the prospectively specified reference-verification failure.
            # Timeout, malformed records and all other failures abort the whole pool.
            skippable = failure.get('worker_failure') == {
                'status':'source_failed','phase':'verify','error_type':'ValueError'}
            rejected.append({'task':task,'candidate_index':i,'failed_index':result['failed_index'],
                             'reason':'reference_verification' if skippable else 'infrastructure_or_other'})
            if not skippable:
                terminal='failed_closed'
    if terminal is None and processed==len(schedules):
        terminal='insufficient_source_valid_tasks'
    return {'status':terminal or 'collecting','accepted':accepted,'rejected':rejected,
            'processed_candidates':processed,'needed':needed,'model_calls':0,'cost_usd':0,
            'paid_authorized':False,'population':'fixed-schedule reference-valid candidates'}


def qualify(root, schedules, dispatch, needed=6):
    validate(schedules,needed)
    root=Path(root)
    root.mkdir(exist_ok=False)
    save(root/'manifest.json',{'schedules':schedules,'needed':needed})
    for i,schedule in enumerate(schedules):
        collect(root/f'{i:02d}',schedule,dispatch)
        result=summarize(root,schedules,needed,i+1)
        if result['status']!='collecting':
            save(root/'result.json',result)
            return result
    raise AssertionError('missing terminal state')


def replay_pool(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    schedules,needed=manifest['schedules'],manifest['needed']
    validate(schedules,needed)
    saved=json.loads((root/'result.json').read_text())
    count=saved['processed_candidates']
    if type(count) is not int or not 1<=count<=len(schedules):
        raise ValueError('processed coverage')
    result=summarize(root,schedules,needed,count)
    if result!=saved or result['status']=='collecting':
        raise ValueError('terminal result identity')
    if {p.name for p in root.iterdir()}!={'manifest.json','result.json'}|{f'{i:02d}' for i in range(count)}:
        raise ValueError('unexpected pool artifacts')
    if any(p.is_symlink() for p in root.iterdir()):
        raise ValueError('symlink in pool')
    return result


def public_cases(root):
    root=Path(root)
    result=replay_pool(root)
    if result['status']!='source_pool_qualified':
        raise ValueError('source pool closed')
    cases=[]
    for item in result['accepted']:
        journal=root/f"{item['candidate_index']:02d}"
        demo,*rest=[json.loads((journal/f'{i:03d}.public.json').read_text()) for i in range(11)]
        cases.append({'inputs':[demo['input']],'outputs':[demo['output']],
            'query_inputs':[r['input'] for r in rest[:2]],'target_inputs':[r['input'] for r in rest[2:]],
            'target_hashes':[r['output_sha256'] for r in rest]})
    return cases
