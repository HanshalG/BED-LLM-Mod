from __future__ import annotations

from scripts import revengebench_execution_opportunity_manifest as manifest


def test_seed_derivation_is_reproducible_and_domain_separated() -> None:
    first = [manifest.seed(manifest.SEED_PREFIX, "battlesnake", index) for index in range(3)]
    second = [manifest.seed(manifest.SEED_PREFIX, "battlesnake", index) for index in range(3)]
    endpoint = [manifest.seed(manifest.ENDPOINT_PREFIX, "battlesnake", index) for index in range(3)]

    assert first == second
    assert len(set(first)) == 3
    assert set(first).isdisjoint(endpoint)
    assert all(value > 0 for value in first + endpoint)


def test_probe_selection_is_order_independent() -> None:
    ids = [f"target-{index}" for index in range(10)]
    key = lambda target_id: manifest.sha256_text(manifest.PROBE_PREFIX + "halite:" + target_id)

    assert sorted(ids, key=key)[:3] == sorted(reversed(ids), key=key)[:3]
