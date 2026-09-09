"""Container-only graph executor for a pinned trusted DSL."""
import json
import os
from pathlib import Path
import resource
import socket
import sys


def grid(value):
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 30:
        raise ValueError('grid height')
    if not isinstance(value[0], (tuple, list)) or not 1 <= len(value[0]) <= 30:
        raise ValueError('grid width')
    width = len(value[0])
    if any(not isinstance(row, (tuple, list)) or len(row) != width or
           any(type(v) is not int or not 0 <= v <= 9 for v in row) for row in value):
        raise ValueError('grid cells')
    return tuple(tuple(row) for row in value)


def main():
    from rearc_program_graph import exports, validate_graph
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError('request size')
    request = json.loads(raw)
    if set(request) != {'graph', 'input'}:
        raise ValueError('request fields')
    functions, constants = exports(Path('/app/dsl.py').read_text())
    validate_graph(request['graph'], functions, constants)
    import dsl
    values = {name: getattr(dsl, name) for name in functions | constants}
    values['I'] = grid(request['input'])
    for step in request['graph']['steps']:
        op = values[step['op']]
        if not callable(op):
            raise ValueError('noncallable intermediate')
        values[step['id']] = op(*(values[arg] for arg in step['args']))
    output = grid(values[request['graph']['output']])
    try:
        Path('/write-probe').write_text('x')
        read_only = False
    except OSError:
        read_only = True
    with socket.socket() as sock:
        sock.settimeout(.1)
        try:
            sock.connect(('1.1.1.1', 443))
            network_denied = False
        except OSError:
            network_denied = True
    print(json.dumps({'status': 'ok', 'output': output, 'uid': os.getuid(),
                     'read_only': read_only, 'network_denied': network_denied,
                     'no_api_key': 'OPENROUTER_API_KEY' not in os.environ}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'invalid', 'error_type': type(error).__name__}))
        sys.exit(1)
