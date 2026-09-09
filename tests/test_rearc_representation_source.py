import hashlib
import json
from scripts.rearc_representation_source import COHORT, BASE, COMMIT, schedule
from scripts.rearc_public_source_journal import validate_schedule


def test_frozen_metadata_selection_and_exact_schedule():
    cohort = json.loads(COHORT.read_text())
    previous = json.loads((BASE/'REARC_EXACT_REPAIR_COHORT_20260909.json').read_text())
    inventory = json.loads((BASE/'REARC_SOURCE_SCOPE_20260909.json').read_text())['all_ids']
    excluded = set(previous['excluded_ids']+previous['selected_ids'])
    selected = sorted(set(inventory)-excluded,
        key=lambda k:hashlib.sha256(('bed-rearc-representation-v1:'+k).encode()).digest())[:6]
    assert len(excluded)==44 and cohort['excluded_ids']==sorted(excluded)
    assert cohort['selected_ids']==selected and len(set(selected))==6
    assert cohort['source_commit']==COMMIT
    expected = [{'task':task,'seed':seed,'mode':mode} for task in selected
        for mode,seeds in [('demonstration',[46100]),('input',[46200,46201]+list(range(46300,46308)))]
        for seed in seeds]
    assert schedule(cohort)==expected and len(expected)==66
    validate_schedule(expected)
