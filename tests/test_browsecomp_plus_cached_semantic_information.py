from __future__ import annotations

import math

import pytest

from scripts import browsecomp_plus_cached_semantic_information as audit
from scripts import browsecomp_plus_semantic_mechanics as mechanics


def test_alias_equivalence_is_target_independent_and_handles_short_aliases():
    assert audit.alias_equivalent("37 kg", "37kg")
    assert audit.alias_equivalent(
        "Emmanuel Kwesi Danso Arthur Junior",
        "Kwesi Arthur",
    )
    assert not audit.alias_equivalent("Mode 9", "Babatunde Olusegun Adewale")


def test_semantic_clusters_use_connected_components():
    values = [
        "Emmanuel Kwesi Danso Arthur Junior",
        "Emmanuel Kwesi Arthur",
        "Kwesi Arthur",
        "Muthoni Ndonga",
    ]
    assert audit.semantic_cluster_ids(values) == [0, 0, 0, 1]


def test_entropy_and_js_divergence_are_stable():
    assert audit.entropy([0.5, 0.5]) == pytest.approx(math.log(2.0))
    assert audit.entropy([1.0, 0.0]) == pytest.approx(0.0)
    assert audit.js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(
        0.0
    )
    assert audit.js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(
        math.log(2.0)
    )


def test_belief_distribution_merges_alias_weights():
    belief = mechanics.Belief(
        hypotheses=("Kwesi Arthur", "Emmanuel Kwesi Arthur"),
        weights=(40, 60),
    )
    distribution = audit.belief_distribution(
        belief,
        value_to_cluster={
            "Kwesi Arthur": 0,
            "Emmanuel Kwesi Arthur": 0,
        },
        cluster_count=1,
    )
    assert distribution == pytest.approx([1.0])
