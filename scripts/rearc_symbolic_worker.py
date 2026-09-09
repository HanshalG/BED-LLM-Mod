"""Isolated first-order symbolic search on public demonstrations only."""
import json
import os
from pathlib import Path
import resource
import sys
from scripts.rearc_symbolic_beam import search


def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20))
    raw=sys.stdin.buffer.read(65537)
    if len(raw)>65536:
        raise ValueError('request size')
    request=json.loads(raw)
    if set(request)!={'inputs','outputs'}:
        raise ValueError('public demonstration fields only')
    import dsl
    result=search(Path('/app/dsl.py').read_text(),vars(dsl),request['inputs'],request['outputs'])
    result.update(status='ok',uid=os.getuid(),no_api_key='OPENROUTER_API_KEY' not in os.environ)
    print(json.dumps(result))


if __name__=='__main__':
    main()
