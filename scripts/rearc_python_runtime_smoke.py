"""Handcrafted native-Python runtime tests, never benchmark examples or LLM calls."""
import hashlib
import json
from pathlib import Path
from scripts.rearc_python_runtime import execute
from scripts.rearc_graph_runtime import IMAGE
from scripts.rearc_public_source_journal import save

FLOOD='''from collections import deque
def transform(g):
    h,w=len(g),len(g[0])
    out=[row[:] for row in g]
    seen=set()
    for r in range(h):
        for c in range(w):
            if g[r][c]!=0 or (r,c) in seen: continue
            q=deque([(r,c)])
            seen.add((r,c))
            region=[]
            border=False
            while q:
                a,b=q.popleft()
                region.append((a,b))
                border=border or a==0 or b==0 or a==h-1 or b==w-1
                for x,y in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if 0<=x<h and 0<=y<w and g[x][y]==0 and (x,y) not in seen:
                        seen.add((x,y))
                        q.append((x,y))
            for a,b in region: out[a][b]=3 if border else 2
    return out
'''


def main():
    root=Path('results/nonmyopic/rearc_python_runtime_smoke_20260909')
    root.mkdir(exist_ok=False)
    files=['rearc_python_runtime.py','rearc_python_worker.py','rearc_python_contract.py','rearc_graph_worker.py']
    save(root/'bindings.json',{'image':IMAGE,'sha256':{n:hashlib.sha256(Path('scripts',n).read_bytes()).hexdigest() for n in files}})
    cases=[('transpose','def transform(g): return [list(r) for r in zip(*g)]',[[1,2,3],[4,5,6]],[[1,4],[2,5],[3,6]]),
        ('flood_closed',FLOOD,[[1,1,1],[1,0,1],[1,1,1]],[[1,1,1],[1,2,1],[1,1,1]]),
        ('flood_open',FLOOD,[[1,0,1],[1,0,1],[1,1,1]],[[1,3,1],[1,3,1],[1,1,1]]),
        ('state_a','n=0\ndef transform(g):\n global n\n n+=1\n return [[n]]',[[1]],[[1]]),
        ('state_b','n=0\ndef transform(g):\n global n\n n+=1\n return [[n]]',[[2]],[[1]]),
        ('output','def transform(g): return [[99]]',[[1]],'failed'),
        ('filesystem','def transform(g): return open("/tmp/not-authorized","w")',[[1]],'failed'),
        ('loop','def transform(g):\n while True: pass',[[1]],'runtime_failed'),
        ('memory','def transform(g):\n x=[0]*100000000\n return g',[[1]],'runtime_failed')]
    rows=[]
    for name,code,x,expected in cases:
        save(root/(name+'.request.json'),{'code':code,'input':x,'expected':expected})
        value=execute(code,x)
        save(root/(name+'.response.json'),value)
        passed=value.get('output')==expected if isinstance(expected,list) else value['status']==expected
        rows.append({'case':name,'passed':passed})
        print(name,value,flush=True)
        if not passed:break
    result={'status':'passed' if len(rows)==len(cases) and all(r['passed'] for r in rows) else 'failed',
        'rows':rows,'model_calls':0,'benchmark_examples':0,'cost_usd':0}
    save(root/'result.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
