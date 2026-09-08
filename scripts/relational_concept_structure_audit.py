"""Aggregate the frozen new grammar without evaluating any world labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from environments.relational_concepts.generative import MAX_HEIGHT, derivation_count, sample_concept


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError('result already exists')
    retries, lengths, hashes = Counter(), Counter(), set()
    failures = 0
    for seed in range(1024):
        try:
            concept = sample_concept(seed)
        except ValueError:
            failures += 1
            continue
        retries[concept.structural_draws] += 1
        lengths[len(concept.formula.replace('(', ' ( ').replace(')', ' ) ').split())] += 1
        hashes.add(hashlib.sha256(concept.formula.encode()).hexdigest())
    result = dict(attempted_seeds=1024, successful_seeds=1024-failures, failed_seeds=failures,
                  distinct_strings=len(hashes), structural_draws_histogram=dict(retries),
                  lexical_length_histogram=dict(lengths),
                  unconditioned_ordered_derivation_count=derivation_count(MAX_HEIGHT, 1),
                  world_labels_evaluated=0, model_calls=0, paid_cost_usd=0,
                  semantic_diversity_verified=False, downstream_experiment_authorized=False)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
