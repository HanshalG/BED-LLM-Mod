"""Container-only public source channels with sanitized failure phases."""
import contextlib
import hashlib
import json
import os
import random
import re
import resource
import sys
from rearc_graph_worker import grid


def main():
    phase='request'
    try:
        resource.setrlimit(resource.RLIMIT_CPU,(2,2))
        raw=sys.stdin.buffer.read(4097)
        if len(raw)>4096:
            raise ValueError('request size')
        request=json.loads(raw)
        if (set(request)!={'task','seed','mode'} or request['mode'] not in ('demonstration','input')
                or not isinstance(request['task'],str) or re.fullmatch('[0-9a-f]{8}',request['task']) is None
                or type(request['seed']) is not int or not 0<=request['seed']<2**32):
            raise ValueError('public request')
        with open(os.devnull,'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            phase='source_import'
            import selected_source
            random.seed(request['seed'])
            phase='generate'
            value=getattr(selected_source,'generate_'+request['task'])(0,1)
            phase='grid_validation'
            x,y=grid(value['input']),grid(value['output'])
            phase='verify'
            if grid(getattr(selected_source,'verify_'+request['task'])(x))!=y:
                raise ValueError('source mismatch')
        result={'input':x,'output_sha256':hashlib.sha256(json.dumps(y,separators=(',',':')).encode()).hexdigest()}
        if request['mode']=='demonstration':
            result['output']=y
        print(json.dumps(result))
        return 0
    except Exception as error:
        kind=type(error).__name__
        if kind not in ('ValueError','TypeError','KeyError','IndexError','NameError','AttributeError','ImportError','ModuleNotFoundError','MemoryError','RuntimeError'):
            kind='Exception'
        print(json.dumps({'status':'source_failed','phase':phase,'error_type':kind}))
        return 1


if __name__=='__main__':
    sys.exit(main())
