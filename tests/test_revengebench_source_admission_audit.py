from __future__ import annotations

from pathlib import Path

from scripts import revengebench_source_admission_audit as audit


def test_split_is_reproducible_disjoint_and_complete() -> None:
    target_ids = [f"target-{index:02d}" for index in range(40)]

    first = audit.split_target_ids("battlesnake", target_ids)
    second = audit.split_target_ids("battlesnake", list(reversed(target_ids)))

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "mechanics": 1,
        "opportunity": 3,
        "development": 4,
        "confirmation": 4,
        "reserve": 28,
    }
    flattened = [target_id for ids in first.values() for target_id in ids]
    assert len(flattened) == len(set(flattened)) == len(target_ids)


def test_target_inventory_never_reads_policy_contents(tmp_path: Path) -> None:
    secret = "PRIVATE_TARGET_POLICY_SHOULD_NOT_BE_SERIALIZED"
    for arena, entrypoint in audit.ARENA_ENTRYPOINTS.items():
        for index in range(12):
            target = tmp_path / "data" / "targets" / arena / f"target-{index:02d}"
            target.mkdir(parents=True)
            (target / entrypoint).write_text(secret, encoding="utf-8")

    result = audit.inventory_targets(tmp_path)

    assert result["arena_count"] == 5
    assert result["target_count"] == 60
    assert result["split_counts"] == {
        "mechanics": 5,
        "opportunity": 15,
        "development": 20,
        "confirmation": 20,
        "reserve": 0,
    }
    assert secret not in audit.canonical_json(result)


def test_condition_normalization_allows_only_frozen_probe_differences() -> None:
    active = """# active\nname: model_full_pool\ntournament:\n  seed: 42\n  max_probes_per_round: 5\ngame:\n  sims_per_round: 20\nprompts:\n  <<: !include prompts/game/battlesnake.yaml\n"""
    no_probe = """# control\nname: model_no_probe\ntournament:\n  seed: 42\n  max_probes_per_round: 0\ngame:\n  sims_per_round: 20\nprompts:\n  <<: !include prompts/game/no_probe/battlesnake.yaml\n"""

    assert audit.normalized_condition_config(active) == audit.normalized_condition_config(no_probe)
    assert audit.normalized_condition_config(active) != audit.normalized_condition_config(
        no_probe.replace("sims_per_round: 20", "sims_per_round: 19")
    )


def test_probe_contract_accepts_wrapped_public_readme(tmp_path: Path) -> None:
    paths = {
        "README.md": "*probe opponents*: runnable policies\nwith no privileged access to target source or\ninternal state.\n",
        "LICENSE": "MIT License\nPermission is hereby granted\n",
        "src/revenge_bench/tournaments/inverse_strategy.py": (
            "Learner does NOT play in simulations\noffline evaluation mean action distance\n"
            "self.target_agent self.learner_agent actions_distance(learner_action, target_action)\n"
        ),
        "src/revenge_bench/tournaments/inverse_strategy_interventionist.py": "def _execute_probe(): pass\n",
        "src/revenge_bench/arenas/arena.py": "random.shuffle(agents)\n",
        "src/revenge_bench/arenas/battlesnake/battlesnake.py": "random.shuffle(players)\n",
    }
    for relative, content in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for index in range(5):
        path = tmp_path / "configs" / "baselines" / "bpi" / "model" / f"arena-{index}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tournament: {}\n", encoding="utf-8")

    contracts = audit.source_contracts(tmp_path)

    assert contracts["probe_is_normal_gameplay_opponent"]
    assert not contracts["battlesnake_command_has_explicit_seed"]
