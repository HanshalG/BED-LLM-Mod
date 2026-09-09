"""Freeze a disjoint cohort using only the previously banked ID inventory."""
import hashlib
import json
from pathlib import Path

PARENT = Path('results/nonmyopic/REARC_SOURCE_SCOPE_20260909.json')
OUTPUT = Path('results/nonmyopic/REARC_FEEDBACK_COHORT_20260909.json')


def select(inventory, excluded):
    if len(inventory)!=400 or len(set(inventory))!=400 or not set(excluded)<=set(inventory):
        raise ValueError('source inventory mismatch')
    return sorted(set(inventory)-set(excluded),
                  key=lambda x:hashlib.sha256(('bed-rearc-feedback-v1:'+x).encode()).digest())[:8]


def main():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='bd379c5233ca24d94e8b51ce45a94c033e5fcf7cd3806518ec3e96c0a7c9b32c':
        raise ValueError('inventory binding')
    parent = json.loads(raw)
    value = {'selected_ids':select(parent['all_ids'],parent['selected_ids']),
             'excluded_ids':parent['selected_ids'],'source_commit':parent['source_commit'],
             'selection':'first8 SHA256(bed-rearc-feedback-v1:+id), excluding closed four; no replacement',
             'source_bodies_inspected':False,'model_calls':0,'cost_usd':0,
             'paid_authorized':False,'scope':'source-contract audit only'}
    with OUTPUT.open('x') as stream: json.dump(value,stream,indent=2)
    print(json.dumps(value))


if __name__=='__main__': main()
