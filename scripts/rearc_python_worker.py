"""Container-only candidate execution. Never mount benchmark source or hidden data."""
import builtins
import __future__
import contextlib
import json
import os
import resource
import sys
import traceback
from rearc_graph_worker import grid
from rearc_python_contract import validate,MODULES

NAMES=('abs all any bool dict enumerate filter float frozenset int isinstance iter len list map max min next pow range reversed round set slice sorted str sum tuple zip divmod chr ord print Exception ValueError TypeError IndexError KeyError StopIteration ZeroDivisionError').split()


def restricted_import(name,globals=None,locals=None,fromlist=(),level=0):
    if level or name not in MODULES:
        raise ImportError('module unavailable')
    return builtins.__import__(name,globals,locals,fromlist,level)


def main():
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    phase='request'
    try:
        raw=sys.stdin.buffer.read(65537)
        if len(raw)>65536:raise ValueError('request size')
        request=json.loads(raw)
        if set(request)!={'code','input'}:raise ValueError('request fields')
        x=[list(row) for row in grid(request['input'])]
        phase='compile'
        tree=validate(request['code'])
        program=compile(tree,'candidate.py','exec',flags=__future__.annotations.compiler_flag,dont_inherit=True)
        safe={name:getattr(builtins,name) for name in NAMES}
        safe['__import__']=restricted_import
        namespace={'__builtins__':safe}
        phase='execute'
        with open(os.devnull,'w') as sink, contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            exec(program,namespace,namespace)
            output=namespace['transform'](x)
        phase='output'
        value=grid(output)
        result={'status':'ok','output':value}
    except Exception as error:
        lines=[f.lineno for f in traceback.extract_tb(error.__traceback__) if f.filename=='candidate.py']
        result={'status':'failed','phase':phase,'error_type':type(error).__name__,
                'candidate_line':lines[-1] if lines else None}
    print(json.dumps(result))


if __name__=='__main__':main()
