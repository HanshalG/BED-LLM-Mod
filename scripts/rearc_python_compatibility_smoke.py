"""New synthetic compatibility checks for native executor revision two."""
import hashlib
from pathlib import Path
from scripts.rearc_python_runtime import execute
from scripts.rearc_python_contract import validate
from scripts.rearc_public_source_journal import save


def main():
    root=Path('results/nonmyopic/rearc_python_compatibility_20260909')
    root.mkdir(exist_ok=False)
    names=['rearc_python_contract.py','rearc_python_worker.py','rearc_python_runtime.py','rearc_python_forecast.py']
    save(root/'bindings.json',{n:hashlib.sha256(Path('scripts',n).read_bytes()).hexdigest() for n in names})
    cases=[('annotations','def transform(g: list[list[int]]) -> list[list[int]]:\n return g',[[1]]),
        ('libraries','''import math
import heapq
from collections import deque
from itertools import chain
from functools import reduce
def transform(g):
    q=deque([1,2])
    h=[]
    for x in chain(q,[3]): heapq.heappush(h,x)
    return [[reduce(lambda a,b:a+b,h)+math.isqrt(4)]]
''',[[8]]),
        ('iterator','def transform(g): return [[next(iter([3]))]]',[[3]])]
    rows=[]
    for name,code,expected in cases:
        result=execute(code,[[1]])
        save(root/(name+'.json'),{'code':code,'input':[[1]],'expected':expected,'result':result})
        assert result=={'status':'ok','output':expected}
        rows.append({'case':name,'passed':True})
    code="def transform(g): return [[ord(list(set(['a','b','c']))[0])%10]]"
    results=[execute(code,[[1]]) for _ in range(2)]
    save(root/'repeat.json',{'code':code,'results':results})
    assert results[0]==results[1] and results[0]['status']=='ok'
    rows.append({'case':'fresh_container_hash_order_repeat','passed':True})
    for name in ('random','time','socket','os'):
        try:validate('import '+name+'\ndef transform(g): return g')
        except ValueError:rows.append({'case':name+'_unavailable','passed':True})
        else:raise AssertionError('unavailable module accepted')
    save(root/'result.json',{'status':'passed','rows':rows,'actual_executions':5,
        'benchmark_examples':0,'model_calls':0,'cost_usd':0})
    print('compatibility passed: five executions, four unavailable-module checks')


if __name__=='__main__':main()
