"""Metadata-selected four-task source inspection; no benchmark execution."""
import hashlib
import json
from pathlib import Path
import urllib.request

REV = 'fe68079c0921029dde679ed44bc3192dd3b270ab'
ROOT = Path('results/nonmyopic/physgym_development_source_20260909')


def select(rows):
    ids = [str(row['id']) for row in rows]
    if len(set(ids)) != len(ids) or len(ids)<4:
        raise ValueError('invalid task inventory')
    return sorted(ids,key=lambda i:hashlib.sha256(('physgym-dev-v1:'+i).encode()).hexdigest())[:4]


def main():
    if ROOT.exists():
        raise ValueError('already opened')
    url = f'https://raw.githubusercontent.com/principia-ai/PhysGym/{REV}/physgym/samples/full_samples.json'
    with urllib.request.urlopen(url,timeout=30) as response:
        raw = response.read(5000001)
    if len(raw)>5000000:
        raise ValueError('source cap')
    rows = json.loads(raw)
    ids = select(rows)
    ROOT.mkdir()
    manifest = dict(revision=REV,source_sha256=hashlib.sha256(raw).hexdigest(),
                    all_ids=sorted(str(r['id']) for r in rows), selected_ids=ids,
                    model_calls=0,cost_usd=0,paid_authority=False)
    with (ROOT/'manifest.json').open('x') as output:
        json.dump(manifest,output,indent=2,sort_keys=True)
        output.write('\n')
    print(json.dumps(manifest,sort_keys=True))
    for task in ids:
        row = next(r for r in rows if str(r['id'])==task)
        # Do not copy the full dataset or unselected solutions into artifacts.
        print(json.dumps({k:v for k,v in row.items() if k not in ('solution','answer')},sort_keys=True))


if __name__ == '__main__':
    main()
