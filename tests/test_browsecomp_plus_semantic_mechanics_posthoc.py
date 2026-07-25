from __future__ import annotations

from scripts import browsecomp_plus_semantic_mechanics_posthoc as posthoc


def test_frozen_hashes_are_distinct_and_complete():
    assert len(posthoc.RAW_SHA256) == 64
    assert len(posthoc.FAILURE_SHA256) == 64
    assert posthoc.RAW_SHA256 != posthoc.FAILURE_SHA256
