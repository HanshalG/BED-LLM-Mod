#!/usr/bin/env python3
"""Privacy-preserving source opportunity audit for frozen RevengeBench targets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "revengebench-source-opportunity-v1"
ELIGIBLE_ARENAS = ("battlesnake", "halite", "huskybench", "robocode")
ARENA_ENTRYPOINTS = {
    "battlesnake": "main.py",
    "halite": "main.c",
    "huskybench": "player.py",
    "robocode": "MyTank.java",
}
TARGET_SALT = "revengebench-opportunity-20260813:"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SyntaxSummary:
    parser: str
    branch_count: int
    comparison_count: int
    action_site_count: int
    state_reference_count: int
    function_count: int
    parseable: bool

    @property
    def semantically_nontrivial(self) -> bool:
        return self.parseable and self.branch_count >= 2 and self.action_site_count >= 1 and self.state_reference_count >= 2


ACTION_TERMS = {
    "action",
    "attack",
    "bet",
    "call",
    "check",
    "direction",
    "fire",
    "fold",
    "move",
    "raise",
    "shoot",
    "speed",
    "turn",
}
STATE_TERMS = {
    "board",
    "enemy",
    "energy",
    "food",
    "game",
    "hand",
    "health",
    "map",
    "opponent",
    "player",
    "position",
    "radar",
    "round",
    "snake",
    "state",
    "strength",
    "unit",
}


def _term_matches(name: str, terms: set[str]) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in terms)


def summarize_python(source: str) -> SyntaxSummary:
    tree = ast.parse(source)
    branches = 0
    comparisons = 0
    actions = 0
    state_names: set[str] = set()
    functions = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.Match)):
            branches += 1
        elif isinstance(node, ast.Compare):
            comparisons += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
            if _term_matches(node.name, ACTION_TERMS):
                actions += 1
        elif isinstance(node, ast.Return) and node.value is not None:
            actions += 1
        elif isinstance(node, ast.Name):
            if _term_matches(node.id, STATE_TERMS):
                state_names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            if _term_matches(node.attr, STATE_TERMS):
                state_names.add(node.attr.lower())
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if _term_matches(name, ACTION_TERMS):
                actions += 1
    return SyntaxSummary(
        parser="python_ast",
        branch_count=branches,
        comparison_count=comparisons,
        action_site_count=actions,
        state_reference_count=len(state_names),
        function_count=functions,
        parseable=True,
    )


TOKEN_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|"
    r"[A-Za-z_$][A-Za-z0-9_$]*|==|!=|<=|>=|&&|\|\||[{}()<>?:]",
    re.DOTALL,
)


def language_tokens(source: str) -> list[str]:
    tokens = []
    for match in TOKEN_RE.finditer(source):
        token = match.group(0)
        if token.startswith(("//", "/*", '"', "'")):
            continue
        tokens.append(token)
    return tokens


def summarize_c_like(source: str, language: str) -> SyntaxSummary:
    tokens = language_tokens(source)
    parseable = tokens.count("{") == tokens.count("}") and tokens.count("(") == tokens.count(")")
    lowered = [token.lower() for token in tokens]
    branches = sum(token in {"if", "switch", "case", "?"} for token in lowered)
    comparisons = sum(token in {"==", "!=", "<", ">", "<=", ">=", "&&", "||"} for token in lowered)
    actions = sum(_term_matches(token, ACTION_TERMS) for token in lowered if token[0].isalpha() or token[0] in "_$" )
    state_names = {token for token in lowered if _term_matches(token, STATE_TERMS)}
    functions = sum(
        1
        for index, token in enumerate(tokens[:-1])
        if tokens[index + 1] == "(" and token not in {"if", "for", "while", "switch", "catch"}
    )
    return SyntaxSummary(
        parser=f"{language}_comment_string_aware_lexer",
        branch_count=branches,
        comparison_count=comparisons,
        action_site_count=actions,
        state_reference_count=len(state_names),
        function_count=functions,
        parseable=parseable,
    )


def summarize_source(path: Path) -> SyntaxSummary:
    source = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".py":
        return summarize_python(source)
    if suffix == ".c":
        return summarize_c_like(source, "c")
    if suffix == ".java":
        return summarize_c_like(source, "java")
    raise ValueError(f"unsupported opportunity source language: {suffix}")


def audit(source_root: Path, admission_audit: Path, replay_audit: Path) -> dict[str, Any]:
    admission = json.loads(admission_audit.read_text(encoding="utf-8"))
    replay = json.loads(replay_audit.read_text(encoding="utf-8"))
    if replay.get("status") != "pass" or replay.get("summary", {}).get("pass_count", 0) < 4:
        raise ValueError("deterministic replay predecessor did not pass")

    records = []
    parser_counts: dict[str, int] = {}
    for arena in ELIGIBLE_ARENAS:
        targets = admission["targets"]["arenas"][arena]["splits"]["opportunity"]
        if len(targets) != 3:
            raise ValueError(f"{arena}: expected three frozen opportunity targets")
        for target_id in targets:
            path = source_root / "data" / "targets" / arena / target_id / ARENA_ENTRYPOINTS[arena]
            if not path.is_file():
                raise ValueError(f"missing frozen target entrypoint: {arena}/{target_id}")
            summary = summarize_source(path)
            parser_counts[summary.parser] = parser_counts.get(summary.parser, 0) + 1
            records.append(
                {
                    "arena": arena,
                    "salted_target_hash": sha256_text(TARGET_SALT + arena + ":" + target_id),
                    "parser": summary.parser,
                    "parseable": summary.parseable,
                    "semantically_nontrivial": summary.semantically_nontrivial,
                    "branch_count": summary.branch_count,
                    "comparison_count": summary.comparison_count,
                    "action_site_count": summary.action_site_count,
                    "state_reference_count": summary.state_reference_count,
                    "function_count": summary.function_count,
                }
            )

    parseable_count = sum(record["parseable"] for record in records)
    nontrivial_count = sum(record["semantically_nontrivial"] for record in records)
    nontrivial_arenas = sorted({record["arena"] for record in records if record["semantically_nontrivial"]})
    source_prerequisite = parseable_count >= 8 and nontrivial_count >= 8 and len(nontrivial_arenas) >= 3

    # Arbitrary learner-written probe programs alter full game trajectories.
    # Static target syntax does not define their response partitions or the
    # answer-conditioned best continuation, so contracts 2--7 require paired
    # execution and cannot be inferred from branch counts.
    status = "execution_opportunity_required" if source_prerequisite else "fail"
    decision = "freeze_zero_call_execution_opportunity_audit" if source_prerequisite else "close_revengebench_route"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "decision": decision,
        "cohort": {
            "eligible_arenas": list(ELIGIBLE_ARENAS),
            "target_count": len(records),
            "ordered_salted_target_hash": sha256_text(canonical_json([record["salted_target_hash"] for record in records])),
        },
        "source_summary": {
            "parseable_count": parseable_count,
            "semantically_nontrivial_count": nontrivial_count,
            "semantically_nontrivial_arena_count": len(nontrivial_arenas),
            "parser_counts": parser_counts,
            "source_prerequisite_pass": source_prerequisite,
            "targets": records,
        },
        "structural_contracts": {
            "semantic_latent_behavior_source_supported": source_prerequisite,
            "intervention_sensitive_observation_static_proven": False,
            "adaptive_continuation_static_proven": False,
            "irreversible_first_action_source_supported": True,
            "nonadditive_horizon_static_proven": False,
            "compute_matched_receding_myopic_witness_static_proven": False,
            "endpoint_separation_static_proven": False,
            "llm_native_opening_source_supported": True,
        },
        "reason": (
            "The frozen targets are parseable and semantically nontrivial, but arbitrary executable probe policies "
            "induce dynamic game trajectories. Static source cannot establish answer-conditioned response partitions, "
            "a depth-two first-probe witness, or held-out endpoint separation without paired execution."
        ),
        "privacy": {
            "source_text_serialized": False,
            "target_provenance_opened": False,
            "released_trajectories_opened": False,
            "released_outcomes_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args.source_root, args.admission_audit, args.replay_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
