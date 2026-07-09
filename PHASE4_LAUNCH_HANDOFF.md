# Path A Phase 4 Launch Handoff

This is the final local handoff for the Path A minimum publishable package.
It is intentionally command-oriented and avoids extra experiment branches.

## Current State

- Ranking-fidelity gate: present and passing.
  - `results/ranking_fidelity/REPORT.md`
  - `results/ranking_fidelity/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_aggregate_plot.png`
- Constrained oracle evidence: present.
  - `results/constrained_oracle/REPORT.md`
- Robustness evidence: present.
  - `plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png`
- Pre-registered Phase 4 endpoint: present.
  - `LOCATION_DEPTH_PATH_A_RUNBOOK.md`, "Pre-registered Analysis"
- Final sweep artifacts: not present yet.
  - `results/location_depth_sweeps/*_REPORT.md`
  - `plots/location_depth_sweeps/*_depth_contrast.png`
  - `plots/location_depth_sweeps/*_headline_rmse.png`
  - `plots/location_depth_sweeps/*_paired_trial_rmse_deltas.png`
  - `plots/location_depth_sweeps/*_paired_trial_truth_log_probability_deltas.png`
  - `results/location_qualitative/*_constrained_qualitative_examples.md`
  - `results/location_qualitative/*_constrained_qualitative_example_*.png`
- Cost-vs-depth artifacts: present for the pre-registered settings.
  - `results/cost_vs_depth/path_a_preregistered_cost_vs_depth.md`
  - `results/cost_vs_depth/path_a_preregistered_cost_vs_depth.png`
- Live cluster state as of 2026-07-09 14:45 London:
  - Old GH200 originals `101778`/`101779`: canceled because they were alive but
    effectively too slow and occupying GH200 nodes.
  - Optimized GH200 relaunches `101993`/`101994`: canceled to free GH200 capacity
    after they showed the same low-yield throughput pattern.
  - `101998`/`101996` optimized full50 MSC jobs: no longer active.
  - `102018` constrained MPP30 fallback and `102019` unconstrained MPP30 fallback:
    canceled after the user requested freeing the effectively-too-slow long-running
    jobs. They had no final or recovered metrics.
- Active job count for the Path A sweep is 0.
- The recommended relaunch path is now split MPP30: three 10-trial blocks per side
  using `--trial-offset` 0, 10, and 20 plus `--total-trials 30`, then combine block
  metrics before packaging.

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
  reports, headline plot, and qualitative strategy examples exist.

## Sync To Cluster

Review the file list, then sync current changed code/configs/scripts/tests. The sync
manifest includes the package-builder dependency chain (`compare_location_depth_sweeps.py`,
`cost_vs_depth_table.py`, `extract_location_qualitative_examples.py`, and
`llm_token_usage.py`) plus the banked ranking-fidelity, constrained-oracle,
robustness-heatmap, and preregistered cost artifacts so remote packaging does
not accidentally use stale helpers or miss already-banked evidence:

```bash
python scripts/path_a_sync_commands.py --list
python scripts/path_a_sync_commands.py
python scripts/path_a_remote_readiness.py
```

The generated `rsync -avR ...` command targets:

```text
oat0:/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z/
```

By default it excludes unlisted generated `results/`, `plots/`, and `runs/`,
while still syncing the exact banked evidence artifacts required by
`validate_path_a_package.py`.

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

The canceled MPP fallback commands used one 30-trial job per side. Do not reuse
that monolithic shape unless there is ample idle capacity. Prefer the split MPP30
commands below so partial blocks finish and can be combined.

Each block uses the same configs with subset flags to reduce the package to the
depths needed for the paper. The `--trial-offset` option replays skipped trial RNG
draws before running the block, and `--total-trials 30` draws the same observation-noise
array as the intended single 30-trial run.

```bash
python scripts/path_a_launch_commands.py --split-mpp30
```

Submit the printed `sbatch` commands after syncing current code to the cluster checkout.
The equivalent explicit shape for each constrained/unconstrained pair is:

```bash
BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}' \
BED_LLM_LOG_REASONING_TRACES=1 \
sbatch --partition=gh200 --job-name=loc_branch_constr26_f50_b00 \
  scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh \
  configs/config_location_branch_decoy_local_final50_26b_a4b.yaml \
  --run-name loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b00_10 \
  --max-depth 5 --include-myopic-controls \
  --strategy-depths 1,3,5 --eval-depths 1,3,5 --myopic-control-depths 3,5 \
  --num-trials 10 --trial-offset 0 --total-trials 30

BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}' \
BED_LLM_LOG_REASONING_TRACES=1 \
sbatch --partition=gh200 --job-name=loc_branch_uncon26_f50_b00 \
  scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh \
  configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml \
  --run-name loc_branch_decoy_local_unconstrained_final50_26b_a4b_mpp30_b00_10 \
  --max-depth 5 --include-myopic-controls \
  --strategy-depths 1,3,5 --eval-depths 1,3,5 --myopic-control-depths 3,5 \
  --num-trials 10 --trial-offset 0 --total-trials 30
```

Use offsets 10 and 20 with matching `_b10_10` and `_b20_10` run names for the
remaining blocks.

## Monitor

```bash
squeue -u hanyal -o "%.18i %.40j %.20P %.2t %.12M %.60R %.50N"

for side in constrained unconstrained; do
  for off in 0 10 20; do
    run="runs/loc_branch_decoy_local_${side}_final50_26b_a4b_mpp30_b$(printf "%02d" "$off")_10"
    echo "--- ${run} ---"
    test -s "${run}/fixed_root_depth_sweep_metrics.json" && echo "metrics: present" || echo "metrics: missing"
    wc -l "${run}/fixed_root_depth_sweep_decisions.jsonl" 2>/dev/null || true
    tail -n 30 "${run}/run.log" 2>/dev/null || true
    grep -E "Traceback|RuntimeError|ValueError|could not produce a valid location|OOM|Killed|CANCELLED|TIMEOUT" \
      "${run}/run.log" 2>/dev/null || true
  done
done
```

## Build The Package

Preferred packaging path: after all three constrained blocks and all three
unconstrained blocks have metrics, combine each side, then build the minimum
publishable package from the combined 30-trial paired constrained and
unconstrained summaries:

```bash
python scripts/combine_location_fixed_root_depth_sweeps.py \
  runs/loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b00_10 \
  runs/loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b10_10 \
  runs/loc_branch_decoy_local_constrained_final50_26b_a4b_mpp30_b20_10 \
  --output runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_split/fixed_root_depth_sweep_metrics.json \
  --report runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_split/REPORT.md \
  --plot runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_split/fixed_root_depth_sweep.png \
  --paired-delta-plot runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_split/paired_trial_rmse_deltas.png

python scripts/combine_location_fixed_root_depth_sweeps.py \
  runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b_mpp30_b00_10 \
  runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b_mpp30_b10_10 \
  runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b_mpp30_b20_10 \
  --output runs/loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_split/fixed_root_depth_sweep_metrics.json \
  --report runs/loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_split/REPORT.md \
  --plot runs/loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_split/fixed_root_depth_sweep.png \
  --paired-delta-plot runs/loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_split/paired_trial_rmse_deltas.png

python scripts/build_path_a_package.py \
  --constrained runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_split/fixed_root_depth_sweep_metrics.json \
  --unconstrained runs/loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_split/fixed_root_depth_sweep_metrics.json \
  --output-dir results/location_depth_sweeps \
  --cost-dir results/cost_vs_depth \
  --plot-dir plots/location_depth_sweeps \
  --run-name location_branch_decoy_depth_contrast_26b_a4b_mpp30_split
```

If the full50 optimized MSC pair finishes first, package that larger run
instead:

```bash
python scripts/build_path_a_package.py \
  --constrained runs/loc_branch_decoy_local_constrained_final50_26b_a4b_sharedopt_msc2/fixed_root_depth_sweep_metrics.json \
  --unconstrained runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b_sharedopt_msc/fixed_root_depth_sweep_metrics.json \
  --output-dir results/location_depth_sweeps \
  --cost-dir results/cost_vs_depth \
  --plot-dir plots/location_depth_sweeps \
  --run-name location_branch_decoy_depth_contrast_26b_a4b_full50_msc
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
