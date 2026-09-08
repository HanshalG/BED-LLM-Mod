"""Complete finite-grammar reference panel; no LLM or hidden scene selection."""

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
from time import monotonic

from environments.semantic_scene.rules import (
    ATTRIBUTES,
    _evaluate,
    compile_rule,
    parse_scene,
)
from environments.semantic_scene.symbolic import candidates, canonical_json
from scripts.number_game_initial_horizon_audit import ExactMembershipHorizon, digest
from scripts.number_game_initial_openloop_audit import OpenLoopMembership


PROTOCOL = Path("results/nonmyopic/SEMANTIC_SCENE_OPPORTUNITY_PROTOCOL_20260908.json")


def scenes():
    objects = list(product(*ATTRIBUTES.values()))
    return [
        {"objects": [dict(zip(ATTRIBUTES, obj)) for obj in group]}
        for n in (1, 2)
        for group in combinations_with_replacement(objects, n)
    ]


def menu_for(universe, index, size=12):
    def key(i):
        text = canonical_json(universe[i])
        return hashlib.sha256(
            f"semantic-scene-v1:{index}:{text}".encode()
        ).hexdigest(), text

    return tuple(sorted(sorted(range(len(universe)), key=key)[:size]))


class MenuHorizon(ExactMembershipHorizon):
    def __init__(self, extensions, menu, **kwargs):
        super().__init__(extensions, **kwargs)
        self.menu = tuple(menu)

    def partitions(self, mask):
        seen = set()
        for query in self.menu:
            left = mask & self.columns[query]
            right = mask ^ left
            pair = tuple(sorted((left, right)))
            if left and right and pair not in seen:
                seen.add(pair)
                yield query, left, right

    @lru_cache(maxsize=None)
    def deployed(self, mask, remaining, horizon):
        self.check()
        query = self.plan(mask, min(remaining, horizon))[1]
        if not remaining or query is None:
            return self.risk(mask)
        positive = mask & self.columns[query]
        return sum(
            (
                Fraction(child.bit_count(), mask.bit_count())
                * self.deployed(child, remaining - 1, horizon)
                for child in (positive, mask ^ positive)
                if child
            ),
            Fraction(0),
        )

    @lru_cache(maxsize=None)
    def random_policy(self, mask, remaining_queries, budget):
        self.check()
        if not budget:
            return self.risk(mask)
        total = Fraction(0)
        for query in remaining_queries:
            positive = mask & self.columns[query]
            rest = tuple(q for q in remaining_queries if q != query)
            total += sum(
                (
                    Fraction(child.bit_count(), mask.bit_count())
                    * self.random_policy(child, rest, budget - 1)
                    for child in (positive, mask ^ positive)
                    if child
                ),
                Fraction(0),
            )
        return total / len(remaining_queries)


def evaluate(extensions, menu, config):
    solver = MenuHorizon(
        extensions,
        menu,
        max_seconds=config["max_menu_seconds"],
        max_states=config["max_states_per_menu"],
    )
    values = {f"h{h}": solver.deployed(solver.full, 3, h) for h in (1, 2, 3)}
    control = OpenLoopMembership(solver, max_sets=config["max_openloop_sets_per_menu"])
    values["committed_three"] = control.plan(solver.full, 3)[0]
    values["receding_openloop_h3"] = control.receding(solver.full, 3)
    values["random_without_replacement"] = solver.random_policy(solver.full, menu, 3)
    if not values["h3"] <= values["receding_openloop_h3"] <= values["committed_three"]:
        raise ArithmeticError("exact policy dominance failed")
    if values["h3"] != solver.plan(solver.full, 3)[0]:
        raise ArithmeticError("deployed h3 differs from full three-step tree")
    row = {
        "menu": menu,
        "values": {k: float(v) for k, v in values.items()},
        "exact_values": {k: str(v) for k, v in values.items()},
        "first_queries": {f"h{h}": solver.plan(solver.full, h)[1] for h in (1, 2, 3)},
        "initial_risk": str(solver.risk(solver.full)),
        "openloop_sets": control.sets,
        "elapsed_seconds": monotonic() - solver.started,
    }
    solver.deployed.cache_clear()
    solver.random_policy.cache_clear()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    for path, expected in config["bindings"].items():
        if digest(path) != expected:
            raise ValueError(f"source binding changed: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    report = {
        "status": "incomplete",
        "protocol_sha256": digest(PROTOCOL),
        "runner_sha256": digest(__file__),
        "rows": [],
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
    }
    try:
        universe = scenes()
        assert len(universe) == 189
        normalized = [parse_scene(s) for s in universe]
        menus = [menu_for(universe, i) for i in range(config["query_menus"])]
        # Freeze public action/target identities to disk before evaluating any rule.
        (args.output_dir / "PUBLIC.json").write_text(
            json.dumps({"scenes": universe, "menus": menus}, indent=2) + "\n"
        )
        extensions, seen, count = [], set(), 0
        for _, candidate in candidates():
            if monotonic() - started > config["max_seconds"]:
                raise TimeoutError("whole-panel time cap during grammar evaluation")
            compiled = compile_rule(canonical_json(candidate))
            extension = tuple(_evaluate(compiled.key, s) for s in normalized)
            count += 1
            if extension not in seen:
                seen.add(extension)
                extensions.append(extension)
        if count != config["grammar_candidates"]:
            raise ValueError("grammar coverage changed")
        report.update(
            grammar_candidates=count,
            distinct_extensions=len(extensions),
            extension_sha256=hashlib.sha256(
                b"".join(bytes(e) for e in extensions)
            ).hexdigest(),
        )
        for menu in menus:
            remaining = config["max_seconds"] - (monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("whole-panel time cap")
            local = dict(
                config, max_menu_seconds=min(remaining, config["max_menu_seconds"])
            )
            row = evaluate(extensions, menu, local)
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
        means = {
            k: sum(
                (Fraction(r["exact_values"][k]) for r in report["rows"]), Fraction(0)
            )
            / len(menus)
            for k in report["rows"][0]["exact_values"]
        }
        passed = (
            means["h1"] > 0
            and means["h2"] > 0
            and means["h2"] <= Fraction(95, 100) * means["h1"]
            and means["h3"] <= Fraction(95, 100) * means["h2"]
        )
        report.update(
            status="reference_opportunity_pass"
            if passed
            else "reference_opportunity_null",
            means={k: float(v) for k, v in means.items()},
            exact_means={k: str(v) for k, v in means.items()},
        )
    except (
        TimeoutError,
        ValueError,
        RuntimeError,
        ArithmeticError,
        AssertionError,
    ) as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = monotonic() - started
    (args.output_dir / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report["status"] == "failed_closed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
