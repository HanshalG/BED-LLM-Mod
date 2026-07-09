# Path A Phase 4 Launch Handoff

This is the final local handoff for the Path A minimum publishable package.
It is intentionally command-oriented and avoids extra experiment branches.

## Current State

- Ranking-fidelity gate: present and passing.
  - `results/ranking_fidelity/REPORT.md`
- Constrained oracle evidence: present.
  - `results/constrained_oracle/REPORT.md`
- Pre-registered Phase 4 endpoint: present.
  - `LOCATION_DEPTH_PATH_A_RUNBOOK.md`, "Pre-registered Analysis"
- Final sweep artifacts: not present yet.
  - `results/location_depth_sweeps/*_REPORT.md`
  - `plots/location_depth_sweeps/*_headline_rmse.png`
  - `results/cost_vs_depth/*_cost_vs_depth.md`
- Live cluster state as of 2026-07-09 01:55 London:
  - Old GH200 originals `101778`/`101779`: canceled because they were alive but
    effectively too slow and occupying GH200 nodes.
  - `101993` constrained full50 optimized GH200: running on `gh200` / `oat21`,
    no metrics yet; inside first StrategyEIG depth-1 belief-refresh block.
  - `101994` unconstrained full50 optimized GH200: pending on `gh200`.
  - `101998` constrained full50 optimized MSC: running on `msc` / `oat15`,
    no metrics yet; inside first StrategyEIG depth-1 belief-refresh block.
  - `101996` unconstrained full50 optimized MSC: running on `msc` / `oat14`,
    no metrics yet; inside first StrategyEIG depth-1 belief-refresh block.
  - `102018` constrained MPP30 fallback: running on `msc` / `oat16`, no metrics
    yet; inside first StrategyEIG depth-1 belief-refresh block.
  - `102019` unconstrained MPP30 fallback: pending on `msc`; run directory not
    created yet.
- Active/pending job count for the Path A sweep is 6. Do not launch more until
  something finishes or the user explicitly asks to cancel/relaunch.

## Local Preflight

Run these from the local repo root before touching the cluster:

```bash
python scripts/path_a_sync_commands.py --list
python scripts/path_a_preflight.py --root .
python scripts/path_a_remote_readiness.py
python scripts/path_a_launch_commands.py
pytest tests/ -q
```

Expected local status before final sweeps:

- `path_a_preflight.py`: launch readiness is OK.
- `validate_path_a_package.py`: package validation is incomplete until final sweep
  reports/plots/cost table exist.

## Sync To Cluster

Review the file list, then sync current changed code/configs/scripts/tests:

```bash
python scripts/path_a_sync_commands.py --list
python scripts/path_a_sync_commands.py
python scripts/path_a_remote_readiness.py
```

The generated `rsync -avR ...` command targets:

```text
oat0:/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z/
```

By default it excludes generated `results/`, `plots/`, and `runs/`.

## Launch Commands

The original final GH200 Singularity launch commands are preserved below for
reproducibility. They have already been submitted as jobs `101778` and
`101779`; do not submit them again unless intentionally creating a new run name.

```bash
cd /users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z

BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}' \
BED_LLM_LOG_REASONING_TRACES=1 \
sbatch --partition=gh200 --job-name=loc_branch_constr26_f50 \
  scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh \
  configs/config_location_branch_decoy_local_final50_26b_a4b.yaml \
  --run-name loc_branch_decoy_local_constrained_final50_26b_a4b \
  --max-depth 5 \
  --include-myopic-controls

BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}' \
BED_LLM_LOG_REASONING_TRACES=1 \
sbatch --partition=gh200 --job-name=loc_branch_uncon26_f50 \
  scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh \
  configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml \
  --run-name loc_branch_decoy_local_unconstrained_final50_26b_a4b \
  --max-depth 5 \
  --include-myopic-controls
```

These use thinking-enabled `google/gemma-4-26B-A4B-it`, 50 paired trials, 6
rounds, 16 rollouts, and include matched-compute `StrategyEIG-myopic-dN`
controls.

The current MPP fallback commands, already submitted as jobs `102018` and
`102019`, use the same configs with subset flags to reduce the package to the
depths needed for the paper:

```bash
BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}' \
BED_LLM_LOG_REASONING_TRACES=1 \
sbatch --partition=msc --exclude=oat10,oat12 --job-name=loc_branch_constr26_mpp30 \
  scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh \
  configs/config_location_branch_decoy_local_final50_26b_a4b.yaml \
  --run-name loc_branch_decoy_local_constrained_mpp30_26b_a4b_msc \
  --max-depth 5 --num-trials 30 --include-myopic-controls \
  --strategy-depths 1,3,5 --eval-depths 1,3,5 --myopic-control-depths 3,5

BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}' \
BED_LLM_LOG_REASONING_TRACES=1 \
sbatch --partition=msc --exclude=oat10,oat12 --job-name=loc_branch_uncon26_mpp30 \
  scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh \
  configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml \
  --run-name loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_msc \
  --max-depth 5 --num-trials 30 --include-myopic-controls \
  --strategy-depths 1,3,5 --eval-depths 1,3,5 --myopic-control-depths 3,5
```

## Monitor

```bash
squeue -u hanyal -o "%.18i %.40j %.20P %.2t %.12M %.60R %.50N"

tail -n 80 runs/loc_branch_decoy_local_constrained_final50_26b_a4b/run.log
tail -n 80 runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b/run.log

grep -E "Traceback|RuntimeError|ValueError|could not produce a valid location|OOM|Killed|CANCELLED|TIMEOUT" \
  runs/loc_branch_decoy_local_constrained_final50_26b_a4b/run.log \
  runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b/run.log
```

## Build The Package

After both sweep metrics files exist:

```bash
python scripts/build_path_a_package.py \
  --constrained runs/loc_branch_decoy_local_constrained_final50_26b_a4b/fixed_root_depth_sweep_metrics.json \
  --unconstrained runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b/fixed_root_depth_sweep_metrics.json \
  --output-dir results/location_depth_sweeps \
  --cost-dir results/cost_vs_depth \
  --plot-dir plots/location_depth_sweeps \
  --run-name location_branch_decoy_depth_contrast_26b_a4b
```

Then audit:

```bash
python scripts/validate_path_a_package.py --root .
```

Completion requires this validator to pass.

If a run exits before the metrics file is written, use
`scripts/recover_depth_sweep_metrics.py` first. The recovery command is recorded
in `LOCATION_DEPTH_PATH_A_RUNBOOK.md`; it writes
`fixed_root_depth_sweep_metrics_recovered.json` by default so recovered metrics
can be inspected before promotion.
