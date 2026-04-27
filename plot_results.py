from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFERRED_METHOD_ORDER = ("EIG", "Entropy", "naive")
METHOD_COLORS = {
    "EIG": "C0",
    "Entropy": "C1",
    "naive": "C2",
}

RESULT_PATTERN = re.compile(
    r"^(?P<run_id>[^_]+)_(?P<method>[^_]+)_Q:(?P<questioner>.+),A:(?P<answerer_and_suffix>.+)$"
)

KNOWN_BELIEF_STATE_MODES = (
    "categorical_depth-2",
    "uniform_depth-1",
    "categorical",
    "uniform",
)


@dataclass(frozen=True)
class ResultFile:
    run_id: str
    method: str
    questioner: str
    answerer: str
    belief_state_mode: str | None
    version: str
    path: Path

    @property
    def pair_key(self) -> tuple[str, str]:
        return self.questioner, self.answerer


def parse_result_file(path: Path) -> ResultFile | None:
    match = RESULT_PATTERN.match(path.stem)
    if match is None:
        return None

    answerer_and_suffix = match.group("answerer_and_suffix")
    suffix_match = re.match(r"^(?P<body>.+)_(?P<version>\d+)_animals$", answerer_and_suffix)
    if suffix_match is None:
        return None

    body = suffix_match.group("body")
    version = suffix_match.group("version")
    belief_state_mode = None
    answerer = body
    for candidate_mode in KNOWN_BELIEF_STATE_MODES:
        marker = f"_{candidate_mode}"
        if body.endswith(marker):
            answerer = body[: -len(marker)]
            belief_state_mode = candidate_mode
            break

    return ResultFile(
        run_id=match.group("run_id"),
        method=match.group("method"),
        questioner=match.group("questioner"),
        answerer=answerer,
        belief_state_mode=belief_state_mode,
        version=version,
        path=path,
    )


def discover_results(results_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, Path]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(results_dir.glob("*.npy")):
        parsed = parse_result_file(path)
        if parsed is None:
            continue
        grouped[parsed.pair_key][parsed.method][parsed.version] = parsed.path
    return grouped


def choose_pair(
    grouped: dict[tuple[str, str], dict[str, dict[str, Path]]],
    pair: str | None,
) -> tuple[tuple[str, str], dict[str, dict[str, Path]]]:
    if not grouped:
        raise ValueError("No result files found in the requested folder.")

    if pair is None:
        if len(grouped) > 1:
            options = ", ".join(
                f"Q:{questioner},A:{answerer}"
                for questioner, answerer in sorted(grouped)
            )
            raise ValueError(
                "Multiple model pairs found. Re-run with --pair and choose one of: "
                f"{options}"
            )
        selected_pair = next(iter(grouped))
        return selected_pair, grouped[selected_pair]

    if not pair.startswith("Q:") or ",A:" not in pair:
        raise ValueError("--pair must look like 'Q:<questioner>,A:<answerer>'")

    questioner, answerer = pair[2:].split(",A:", 1)
    selected_pair = (questioner, answerer)
    if selected_pair not in grouped:
        options = ", ".join(
            f"Q:{candidate_questioner},A:{candidate_answerer}"
            for candidate_questioner, candidate_answerer in sorted(grouped)
        )
        raise ValueError(f"Pair {pair!r} not found in {options}")

    return selected_pair, grouped[selected_pair]


def choose_versions(methods: dict[str, dict[str, Path]], requested_versions: list[str] | None) -> list[str]:
    available_versions = sorted(
        {
            version
            for method_paths in methods.values()
            for version in method_paths
        },
        key=int,
    )
    if not available_versions:
        raise ValueError("No versioned result files found for the selected model pair.")

    if requested_versions is None:
        return available_versions

    requested = [str(version) for version in requested_versions]
    matching = [version for version in requested if version in available_versions]
    if not matching:
        raise ValueError(
            "None of the requested versions are present. Available versions: "
            + ", ".join(available_versions)
        )
    return matching


def collect_method_curves(
    methods: dict[str, dict[str, Path]],
    versions: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    method_curves: dict[str, np.ndarray] = {}
    used_versions: dict[str, list[str]] = {}

    for method, version_paths in sorted(methods.items()):
        existing_versions = [version for version in versions if version in version_paths]
        if not existing_versions:
            continue

        curves = [np.load(version_paths[version]) for version in existing_versions]
        lengths = {len(curve) for curve in curves}
        if len(lengths) != 1:
            raise ValueError(f"{method} curves do not all have the same length.")

        method_curves[method] = np.mean(curves, axis=0)
        used_versions[method] = existing_versions

    if not method_curves:
        raise ValueError("No matching methods found for the selected versions.")

    return method_curves, used_versions


def ordered_methods(method_curves: dict[str, np.ndarray]) -> list[str]:
    preferred = [method for method in PREFERRED_METHOD_ORDER if method in method_curves]
    extras = sorted(method for method in method_curves if method not in PREFERRED_METHOD_ORDER)
    return preferred + extras


def prettify_model_name(model_spec: str) -> str:
    name = model_spec
    name = name.replace("__thinking-on", " (thinking)")
    name = name.replace("__thinking-off", "")
    name = name.replace("__reasoning-low", " (reasoning low)")
    name = name.replace("__reasoning-medium", " (reasoning medium)")
    name = name.replace("__reasoning-high", " (reasoning high)")
    name = name.replace("_", "/")
    return name


def default_title(pair_key: tuple[str, str], results_dir: Path) -> str:
    questioner, answerer = pair_key
    if questioner == answerer:
        return prettify_model_name(questioner)
    return results_dir.name or f"{prettify_model_name(questioner)} vs {prettify_model_name(answerer)}"


def default_output_path(results_dir: Path, pair_key: tuple[str, str]) -> Path:
    questioner, answerer = pair_key
    if questioner == answerer:
        slug = questioner
    else:
        slug = f"{questioner}__{answerer}"
    return Path("plots") / f"results_{slug}.png"


def plot_curves(method_curves: dict[str, np.ndarray], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    method_names = ordered_methods(method_curves)
    for index, method in enumerate(method_names):
        values = method_curves[method] * 100.0
        x = np.arange(1, len(values) + 1)
        ax.plot(
            x,
            values,
            linewidth=1.7,
            label=method,
            color=METHOD_COLORS.get(method, f"C{index}"),
        )

    question_count = len(next(iter(method_curves.values())))
    ax.set_title(title)
    ax.set_xlabel("# Questions")
    ax.set_ylabel("% correct guesses")
    ax.set_xticks([tick for tick in (5, 10, 15, 20) if tick <= question_count])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-5, 102)
    ax.legend(loc="upper left")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregated accuracy curves from a results folder."
    )
    parser.add_argument("results_dir", type=Path, help="Folder containing .npy result files.")
    parser.add_argument("--pair", help="Model pair to plot, formatted as 'Q:<questioner>,A:<answerer>'.")
    parser.add_argument("--versions", nargs="+", help="Specific version suffixes to combine.")
    parser.add_argument("--title", help="Custom plot title.")
    parser.add_argument("--output", type=Path, help="Where to save the figure.")
    args = parser.parse_args()

    grouped = discover_results(args.results_dir)
    pair_key, methods = choose_pair(grouped, args.pair)
    versions = choose_versions(methods, args.versions)
    method_curves, used_versions = collect_method_curves(methods, versions)

    output_path = args.output or default_output_path(args.results_dir, pair_key)
    title = args.title or default_title(pair_key, args.results_dir)
    plot_curves(method_curves, title, output_path)

    print(f"Plotted aggregate curves to {output_path}")
    print(f"  pair: Q:{pair_key[0]},A:{pair_key[1]}")
    print(f"  versions requested: {', '.join(versions)}")
    for method in ordered_methods(method_curves):
        print(f"  {method} versions: {', '.join(used_versions[method])}")


if __name__ == "__main__":
    main()
