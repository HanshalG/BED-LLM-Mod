"""Selected trusted source: expose only the requested example channel."""
import hashlib
import json
import random
import resource
import sys
from rearc_graph_worker import grid


def main():
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    request=json.loads(sys.stdin.buffer.read(4096))
    if set(request)!={'task','seed','mode'} or request['mode'] not in {'demonstration','input','output'}:
        raise ValueError('source request')
    import selected_source
    random.seed(request['seed'])
    value=getattr(selected_source,'generate_'+request['task'])(0,1)
    x,y=grid(value['input']),grid(value['output'])
    if grid(getattr(selected_source,'verify_'+request['task'])(x))!=y:
        raise ValueError('source mismatch')
    result={'output_sha256':hashlib.sha256(json.dumps(y,separators=(',',':')).encode()).hexdigest()}
    if request['mode']!='output': result['input']=x
    if request['mode']!='input': result['output']=y
    print(json.dumps(result))


if __name__=='__main__':
    main()
