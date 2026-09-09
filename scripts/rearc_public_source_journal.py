"""One-shot public schedule collection with a preserved, replayable prefix."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from scripts.rearc_graph_worker import grid


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def validate_schedule(schedule):
    if not isinstance(schedule, list) or not schedule:
        raise ValueError('empty schedule')
    seen = set()
    for row in schedule:
        if (set(row)!={'task','seed','mode'} or not isinstance(row['task'],str)
                or re.fullmatch('[0-9a-f]{8}',row['task']) is None
                or type(row['seed']) is not int or not 0<=row['seed']<2**32
                or row['mode'] not in ('demonstration','input')):
            raise ValueError('public schedule violation')
        key = (row['task'],row['seed'])
        if key in seen:
            raise ValueError('duplicate source example')
        seen.add(key)


def decode_public(raw, mode):
    if len(raw)>16384:
        raise ValueError('response_size')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate_key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    expected = {'input','output_sha256'} | ({'output'} if mode=='demonstration' else set())
    if not isinstance(value,dict) or set(value)!=expected:
        raise ValueError('source_channel')
    if not isinstance(value['output_sha256'],str) or re.fullmatch('[0-9a-f]{64}',value['output_sha256']) is None:
        raise ValueError('output_hash')
    grid(value['input'])
    if mode=='demonstration':
        y=grid(value['output'])
        if hashlib.sha256(json.dumps(y,separators=(',',':')).encode()).hexdigest()!=value['output_sha256']:
            raise ValueError('demonstration_hash')
    return value


def collect(root, schedule, dispatch):
    """dispatch returns CompletedProcess bytes; no retries or hidden output mode."""
    validate_schedule(schedule)
    root = Path(root)
    root.mkdir(exist_ok=False)
    save(root/'schedule.json',schedule)
    prefix = []
    for i,request in enumerate(schedule):
        save(root/f'{i:03d}.request.json',request)
        evidence = {'index':i,'request':request,'phase':'transport'}
        try:
            response = dispatch(dict(request))
            evidence.update(returncode=response.returncode, stdout_bytes=len(response.stdout),
                stdout_sha256=hashlib.sha256(response.stdout).hexdigest(),
                stderr_bytes=len(response.stderr),stderr_sha256=hashlib.sha256(response.stderr).hexdigest())
            if response.returncode:
                raise RuntimeError('worker_exit')
            evidence['phase']='public_channel_validation'
            value = decode_public(response.stdout,request['mode'])
        except Exception as error:
            evidence.update(status='failed_closed',error_type=type(error).__name__)
            # No arbitrary exception text, stdout or stderr that might contain labels.
            if isinstance(error,subprocess.TimeoutExpired):
                evidence['timeout_seconds']=error.timeout
            save(root/f'{i:03d}.failure.json',evidence)
            result={'status':'failed_closed','completed':len(prefix),'failed_index':i,
                    'calls':0,'paid_authorized':False}
            save(root/'result.json',result)
            return result
        save(root/f'{i:03d}.public.json',value)
        evidence['public_sha256']=hashlib.sha256((root/f'{i:03d}.public.json').read_bytes()).hexdigest()
        save(root/f'{i:03d}.receipt.json',{**evidence,'status':'ok'})
        prefix.append(value)
    result={'status':'public_schedule_complete','completed':len(prefix),'calls':0,'paid_authorized':False}
    save(root/'result.json',result)
    return result


def replay(root):
    """Validate a banked success/failure prefix without redispatching source work."""
    root = Path(root)
    schedule=json.loads((root/'schedule.json').read_text())
    validate_schedule(schedule)
    result=json.loads((root/'result.json').read_text())
    n=result['completed']
    if type(n) is not int or not 0<=n<=len(schedule):
        raise ValueError('prefix coverage')
    expected={'schedule.json','result.json'}
    for i in range(n):
        names=[f'{i:03d}.{suffix}.json' for suffix in ('request','public','receipt')]
        expected.update(names)
        if json.loads((root/names[0]).read_text())!=schedule[i]:
            raise ValueError('request identity')
        decode_public((root/names[1]).read_bytes(),schedule[i]['mode'])
        receipt=json.loads((root/names[2]).read_text())
        if receipt['public_sha256']!=hashlib.sha256((root/names[1]).read_bytes()).hexdigest():
            raise ValueError('public identity')
        if receipt['status']!='ok' or receipt['returncode']!=0 or receipt['request']!=schedule[i]:
            raise ValueError('receipt identity')
    if result['status']=='failed_closed':
        if n==len(schedule) or result['failed_index']!=n:
            raise ValueError('failure coverage')
        names=[f'{n:03d}.{suffix}.json' for suffix in ('request','failure')]
        expected.update(names)
        if json.loads((root/names[0]).read_text())!=schedule[n] or json.loads((root/names[1]).read_text())['request']!=schedule[n]:
            raise ValueError('failure identity')
    elif result['status']!='public_schedule_complete' or n!=len(schedule):
        raise ValueError('terminal coverage')
    if {p.name for p in root.iterdir()}!=expected or any(p.is_symlink() for p in root.iterdir()):
        raise ValueError('unexpected artifacts')
    return {'status':'prefix_replay','completed':n,'new_source_calls':0}
