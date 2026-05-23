"""CLI entry point delegating to :mod:`core.experiment`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cli_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--method-name",
        default=None,
        help="Override method selection (defaults to config.method_names[0])",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory under which per-invocation run directories are created",
    )
    args = parser.parse_args(argv)

    from helpers import load_config
    from model import build_model_adapter
    from core.experiment import run_from_config

    config = load_config(args.config)
    if not config.model_pairs:
        raise SystemExit("config.model_pairs is empty; cannot construct LLM adapters")
    if args.method_name is not None:
        config.method_names = [args.method_name]

    model_specs = []
    seen = set()
    for pair in config.model_pairs:
        for spec in (pair.questioner, pair.answerer):
            if config.task == "location_finding" and spec == pair.answerer:
                continue
            if spec in seen:
                continue
            seen.add(spec)
            model_specs.append(spec)
    models = {spec: build_model_adapter(spec, config=config) for spec in model_specs}

    results = []
    for pair in config.model_pairs:
        questioner = models[pair.questioner]
        answerer = None if config.task == "location_finding" else models[pair.answerer]
        for method_name in config.method_names:
            _run, summary = run_from_config(
                config,
                questioner,
                answerer,
                method_name=method_name,
                output_dir=args.output_root,
            )
            results.append((pair, method_name, summary))

    print(f"Finished {len(results)} run(s).")
    for pair, method_name, summary in results:
        print(f"  {pair.questioner.model} / {method_name}:")
        for name, values in summary.metrics.items():
            print(f"    {name}: {values}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main(sys.argv[1:]))
