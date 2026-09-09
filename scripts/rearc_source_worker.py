"""Container-only selected-source check; emits hashes/shapes, not examples."""
import hashlib
import json
import random
import resource
import sys
from rearc_graph_worker import grid


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    request = json.loads(sys.stdin.buffer.read(4096))
    import selected_source
    random.seed(request['seed'])
    example = getattr(selected_source, 'generate_'+request['task'])(0, 1)
    input_grid, output = grid(example['input']), grid(example['output'])
    verified = grid(getattr(selected_source, 'verify_'+request['task'])(input_grid))
    digest = lambda value: hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()
    print(json.dumps({'status':'ok' if output == verified else 'mismatch',
        'input_shape':[len(input_grid),len(input_grid[0])],
        'output_shape':[len(output),len(output[0])], 'identity':input_grid == output,
        'input_sha256':digest(input_grid), 'output_sha256':digest(output),
        'verifier_sha256':digest(verified)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'failed', 'error_type':type(error).__name__}))
        sys.exit(1)
