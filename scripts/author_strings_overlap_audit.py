"""Input-only provenance audit; no output inspection or predictive scoring."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

from scripts.author_strings_audit import COMMIT, parse
from scripts.syntra_source_audit import COMMIT as SYNTRA_COMMIT, FILES


def choose(ids):
    return sorted(set(ids)-{'1', '10'}, key=lambda t: hashlib.sha256(
        ('author-strings-overlap-v1:' + t).encode()).hexdigest())[:12]


def compare(inputs, derivative):
    source = set(inputs)
    matches = {name: len(source & xs) for name, xs in derivative.items()}
    return dict(distinct_inputs=len(source),
                exact_derivative_subsets=sorted(name for name, xs in derivative.items()
                                               if len(xs) >= 4 and xs <= source),
                maximum_shared_inputs=max(matches.values(), default=0))


def get(url, limit):
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.headers.get('Link'):
            raise ValueError('unexpected pagination')
        raw = response.read(limit+1)
    if len(raw) > limit:
        raise ValueError('oversized source')
    return raw


def main():
    root = Path('results/nonmyopic/author_strings_overlap_20260909')
    if root.exists():
        raise ValueError('already opened')
    raw = get(f'https://raw.githubusercontent.com/klee972/SYNTRA/{SYNTRA_COMMIT}/SYNTRA/data/playgol_v2.jsonl', 200000)
    if hashlib.sha256(raw).hexdigest() != FILES['playgol_v2.jsonl']:
        raise ValueError('derivative mismatch')
    derivative = {}
    for line in raw.splitlines():
        row = json.loads(line)
        xs = [pair['input'] for split in ('train', 'test') for pair in row[split]]
        if not all(type(x) is str for x in xs) or row['name'] in derivative:
            raise ValueError('invalid input schema')
        derivative[row['name']] = set(xs)
    listing = get(f'https://huggingface.co/api/datasets/andrewcropper/ilp-datasets/tree/{COMMIT}/strings?limit=1000', 200000)
    ids = [r['path'].split('/')[-1] for r in json.loads(listing) if r['type']=='directory']
    if len(ids) != 329 or not all(t.isdigit() for t in ids):
        raise ValueError('source inventory changed')
    selected = choose(ids)
    root.mkdir()
    def save(name, data):
        with (root/name).open('x') as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write('\n')
    save('selection.json', dict(ids=selected, source_commit=COMMIT,
                               rule='first12 SHA256(author-strings-overlap-v1:<id>), exclude1/10',
                               listing_sha256=hashlib.sha256(listing).hexdigest(),
                               outcome_authority=False))
    records = {}
    for task in selected:
        raw = get(f'https://huggingface.co/datasets/andrewcropper/ilp-datasets/resolve/{COMMIT}/strings/{task}/train/bk.pl', 1000000)
        examples = defaultdict(dict)
        for name, args in parse(raw.decode()):
            if name == 'in':
                example, position, char = args
                if type(position) is not int or type(char) is not str or len(char)!=1 or position in examples[example]:
                    raise ValueError('invalid input position')
                examples[example][position] = char
        inputs = []
        for positions in examples.values():
            if sorted(positions) != list(range(1,len(positions)+1)):
                raise ValueError('noncontiguous input')
            inputs.append(''.join(positions[i] for i in sorted(positions)))
        records[task] = dict(compare(inputs, derivative), bk_sha256=hashlib.sha256(raw).hexdigest())
    save('result.json', dict(records=records, model_calls=0, cost_usd=0,
                             outputs_inspected=False, paid_authority=False,
                             implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(records,sort_keys=True))


if __name__ == '__main__':
    main()
