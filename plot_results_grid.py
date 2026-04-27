from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot_results import (
    METHOD_COLORS,
    choose_pair,
    choose_versions,
    collect_method_curves,
    default_title,
    discover_results,
    ordered_methods,
)


def load_curves(results_dir: Path, pair: str | None, versions: list[str] | None):
    grouped = discover_results(results_dir)
    pair_key, methods = choose_pair(grouped, pair)
    selected_versions = choose_versions(methods, versions)
    method_curves, used_versions = collect_method_curves(methods, selected_versions)
    return pair_key, selected_versions, method_curves, used_versions


def plot_panel(ax, method_curves, title: str, show_ylabel: bool) -> None:
    method_names = ordered_methods(method_curves)
    for index, method in enumerate(method_names):
        values = method_curves[method] * 100.0
        x = range(1, len(values) + 1)
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
    if show_ylabel:
        ax.set_ylabel("% correct guesses")
    ax.set_xticks([tick for tick in (5, 10, 15, 20) if tick <= question_count])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-5, 102)
    ax.legend(loc="upper left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregated accuracy curves from multiple results folders in a grid."
    )
    parser.add_argument("results_dirs", nargs="+", type=Path, help="Results folders to plot.")
    parser.add_argument("--titles", nargs="*", help="Panel titles in the same order as the results folders.")
    parser.add_argument("--pair", help="Model pair to plot, formatted as 'Q:<questioner>,A:<answerer>'.")
    parser.add_argument("--versions", nargs="+", help="Specific version suffixes to combine.")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns in the grid.")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the combined figure.")
    args = parser.parse_args()

    if args.titles and len(args.titles) != len(args.results_dirs):
        raise ValueError("--titles must match the number of results folders when provided.")

    panel_data = [load_curves(results_dir, args.pair, args.versions) for results_dir in args.results_dirs]

    cols = max(1, args.cols)
    rows = math.ceil(len(args.results_dirs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.0 * rows), sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index, (results_dir, panel) in enumerate(zip(args.results_dirs, panel_data)):
        pair_key, _selected_versions, method_curves, _used_versions = panel
        title = args.titles[index] if args.titles else default_title(pair_key, results_dir)
        plot_panel(axes_list[index], method_curves, title, show_ylabel=(index % cols == 0))

    for ax in axes_list[len(args.results_dirs):]:
        ax.axis("off")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Plotted grid aggregate curves to {args.output}")
    for results_dir, panel in zip(args.results_dirs, panel_data):
        pair_key, selected_versions, method_curves, used_versions = panel
        print(f"  folder: {results_dir}")
        print(f"  pair: Q:{pair_key[0]},A:{pair_key[1]}")
        print(f"  versions requested: {', '.join(selected_versions)}")
        for method in ordered_methods(method_curves):
            print(f"  {method} versions: {', '.join(used_versions[method])}")


if __name__ == "__main__":
    main()
