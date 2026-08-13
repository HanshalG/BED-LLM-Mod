from __future__ import annotations

from scripts import causalab_llm_native_source_admission as source


def _features() -> dict[str, bool]:
    return {
        "hidden_scm_declared": True,
        "manipulator_and_transfer_objects": True,
        "finite_intervention_budget": True,
        "machine_parseable_hypothesis": True,
        "graph_binding": True,
        "environment_seed_binding": True,
        "policy_can_omit_hidden_graph": True,
        "complete_candidate_loader": True,
        "exact_transition_simulator": True,
        "deterministic_consistency_filter": True,
        "candidate_bank_can_enter_prompt": True,
    }


def _manifest() -> dict[str, object]:
    return {
        "included_file_count": 19,
        "included_record_count": 950,
        "files": [{"records": 50} for _ in range(19)],
    }


def test_complete_executable_candidate_filter_fails_llm_ownership() -> None:
    result = source.evaluate(
        commit=source.EXPECTED_COMMIT,
        tree=source.EXPECTED_TREE,
        protocol_sha256=source.EXPECTED_PROTOCOL_SHA256,
        license_text="Apache License\nVersion 2.0",
        manifest=_manifest(),
        features=_features(),
        source_blob_sha256={"causal_tool": "0" * 64},
    )

    assert result["status"] == "source_failed_closed"
    assert result["failure_gate"] == "open_support_ownership"
    assert result["gates"]["bindings_and_release"] is True
    assert result["gates"]["native_sequential_experiment"] is True
    assert result["gates"]["replay_and_sealing"] is True
    assert result["gates"]["open_support_ownership"] is False
    assert result["gates"]["irreducible_llm_role"] is None
    assert result["gates"]["prospective_classical_control"] is None
    assert result["access_accounting"] == {
        "graph_records_read": 0,
        "episode_trajectories_read": 0,
        "environment_executions": 0,
        "intervention_outcomes_opened": 0,
        "transfer_endpoints_opened": 0,
        "model_calls": 0,
        "openrouter_cost_usd": 0.0,
        "cluster_jobs": 0,
    }


def test_source_feature_detection_requires_all_filter_components() -> None:
    texts = {
        "readme": (
            "Each episode hides a freshly sampled structural causal model. "
            "Use the Property Manipulator and transfer to a held-out reactor. "
            "Repeat until budget exhausted."
        ),
        "scenario": (
            'graph.budget = config_dict.get("budget", None)\n'
            "CAUSAL_GRAPH_CONFIG CAUSAL_GRAPH_ID ENV_SEED"
        ),
        "causal_tool": (
            "def _load_candidates():\n for line in f:\n  "
            'x = {"config": config}\n'
            "load_causal_graph_from_config(config, random.Random(0))\n"
            "def _simulate_transition():\n graph.compute_values("
            "override_base_values=base_values)\n"
            "def add_transition():\n self._transition_matches_graph\n"
            "self.active_candidates\n"
            'x = {"candidate_graphs": [], "edges": config.get("edges", [])}'
        ),
        "prompt": (
            'Previous State {"hypothesis":{"edges":[],"freq_equation":null,'
            '"coefficients":{}}}'
        ),
        "metrics": "ast.parse(text)",
    }
    features = source.source_features(texts)
    assert all(features.values())

    texts["causal_tool"] = texts["causal_tool"].replace(
        "load_causal_graph_from_config(config, random.Random(0))", ""
    )
    assert source.source_features(texts)["complete_candidate_loader"] is False
