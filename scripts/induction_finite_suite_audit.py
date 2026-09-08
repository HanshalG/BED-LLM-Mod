"""Aggregate a pinned private suite without executing concepts or inspecting labels."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import yaml

COMMIT = 'c1f71f98623ab6e3513f820a8e52235d5b498694'
SHA = '55e21dee281be6ab779119dcce609d7d969e5528863c82a38fa4782fca0100ed'
URL = f'https://raw.githubusercontent.com/SerafimBatzoglou/concept-synth/{COMMIT}/benchmarks/induction/data/induction_fullobs_v1.yaml.gz'
MAX_RAW = 1024 * 1024
MAX_DECOMPRESSED = 64 * 1024 * 1024


def inspect_records(records, expected_count=375):
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError('unexpected record count')
    ids, groups = set(), Counter()
    world_counts, domain_sizes, token_counts = Counter(), Counter(), Counter()
    for record in records:
        if (not isinstance(record, dict)
                or record.get('schemaVersion') != 'induction_benchmark_record_v1'
                or record.get('task') != 'FullObs'):
            raise ValueError('invalid record identity')
        identity = record.get('instanceId')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('missing or duplicate identity')
        ids.add(identity)
        try:
            formula = record['problemDescription']['hiddenTarget']['formula']
            worlds = record['problem']['worlds']
        except (KeyError, TypeError):
            raise ValueError('missing structural fields') from None
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError('invalid reference text')
        # Text grouping is deliberately weaker than semantic equivalence.
        tokens = formula.replace('(', ' ( ').replace(')', ' ) ').split()
        normalized = ' '.join(tokens)
        groups[hashlib.sha256(normalized.encode()).hexdigest()] += 1
        token_counts[len(tokens)] += 1
        if not isinstance(worlds, list) or not worlds:
            raise ValueError('invalid world list')
        world_counts[len(worlds)] += 1
        for world in worlds:
            domain = world.get('domain') if isinstance(world, dict) else None
            if (not isinstance(domain, list) or not domain
                    or any(not isinstance(x, str) for x in domain)
                    or len(set(domain)) != len(domain)):
                raise ValueError('invalid domain')
            domain_sizes[len(domain)] += 1
    split_groups, split_records = Counter(), Counter()
    for digest, count in groups.items():
        bucket = int(hashlib.sha256(('induction-finite-suite-v1:' + digest).encode()).hexdigest(), 16) % 10
        split = 'development' if bucket < 6 else 'validation' if bucket < 8 else 'confirmation'
        split_groups[split] += 1
        split_records[split] += count
    return dict(
        record_count=len(records), normalized_text_groups=len(groups),
        group_size_histogram=dict(sorted(Counter(groups.values()).items())),
        lexical_token_count_histogram=dict(sorted(token_counts.items())),
        worlds_per_record_histogram=dict(sorted(world_counts.items())),
        domain_size_histogram=dict(sorted(domain_sizes.items())),
        tentative_split_groups=dict(sorted(split_groups.items())),
        tentative_split_records=dict(sorted(split_records.items())),
        semantic_equivalence_checked=False, endpoint_values_evaluated=0,
        private_rows_materialized=True, labels_consulted=False,
        model_calls=0, paid_cost_usd=0, downstream_experiment_authorized=False,
    )


def inspect_bytes(raw):
    if len(raw) > MAX_RAW or hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('artifact binding mismatch')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        content = stream.read(MAX_DECOMPRESSED + 1)
    if len(content) > MAX_DECOMPRESSED:
        raise ValueError('decompressed size exceeded')
    try:
        records = yaml.safe_load(content)
    except yaml.YAMLError:
        raise ValueError('source YAML invalid; private context suppressed') from None
    result = inspect_records(records)
    result.update(source_commit=COMMIT, source_sha256=SHA, source_url=URL)
    return result


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError('output already exists')
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read(MAX_RAW + 1)
    result = inspect_bytes(raw)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        run(args.output)
    except Exception:
        raise SystemExit('Audit failed closed; no private context emitted.') from None
