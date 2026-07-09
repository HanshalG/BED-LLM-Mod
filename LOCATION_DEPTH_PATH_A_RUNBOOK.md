# Location-Finding Path A Runbook

This runbook records the reproducible command path for the Path A depth-effect
experiments. Do not launch the final Phase 4 sweeps until the Phase 1
ranking-fidelity gate has been aggregated.

## Cluster Setup

Use the cluster checkout:

```bash
cd /users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z
```

For repeated launches on A100/conda launchers where the environment is already prepared:

```bash
export BED_LLM_SKIP_ENV_SETUP=1
export BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}'
export BED_LLM_LOG_REASONING_TRACES=1
```

Add `--exclude=oat12` to `sbatch` commands when that node should be avoided.

## Reproducibility Snapshot

Current code state to protect before interpreting the final sweeps:

- Local branch: `codex/location-finding-llmstrategy`.
- Cluster checkout:
  `/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z`.
- Final reproducibility tag: `path-a-final-sweep`.
- Synced cluster code includes the shared-state optimization and MPP30
  fallback depth subset support in:
  `scripts/location_fixed_root_depth_sweep.py`,
  `scripts/path_a_launch_commands.py`,
  `tests/test_location_fixed_root_depth_sweep.py`, and
  `tests/test_path_a_launch_commands.py`.

Live Phase 4 cluster state as of 2026-07-09:

| Job ID | Job name | Partition | State | Node | Run name | Notes |
|---:|---|---|---|---|---|---|
| 101778 | `loc_branch_constr26_f50` | `gh200` | canceled | `oat21` | `loc_branch_decoy_local_constrained_final50_26b_a4b` | Original full50 constrained job; canceled after 13h+ because it remained at 150 decisions with no metrics and was blocking GH200 capacity. |
| 101779 | `loc_branch_uncon26_f50` | `gh200` | canceled | `oat22` | `loc_branch_decoy_local_unconstrained_final50_26b_a4b` | Original full50 unconstrained job; canceled after 13h+ because it remained at 150 decisions with no metrics and was blocking GH200 capacity. |
| 101993 | `loc_branch_constr26_f50_opt` | `gh200` | canceled | `oat21` | `loc_branch_decoy_local_constrained_final50_26b_a4b_sharedopt` | Optimized GH200 relaunch; canceled to free GH200 capacity after showing low-yield throughput. |
| 101994 | `loc_branch_uncon26_f50_opt` | `gh200` | canceled | - | `loc_branch_decoy_local_unconstrained_final50_26b_a4b_sharedopt` | Optimized GH200 relaunch; canceled before running. |
| 101998 | `loc_branch_constr26_f50_optm2` | `msc` | running | `oat15` | `loc_branch_decoy_local_constrained_final50_26b_a4b_sharedopt_msc2` | Optimized full50 constrained MSC run; no metrics yet, active in vLLM generation. |
| 101996 | `loc_branch_uncon26_f50_optm` | `msc` | running | `oat14` | `loc_branch_decoy_local_unconstrained_final50_26b_a4b_sharedopt_msc` | Optimized full50 unconstrained MSC run; no metrics yet, active in vLLM generation. |
| 102018 | `loc_branch_constr26_mpp30` | `msc` | running | `oat16` | `loc_branch_decoy_local_constrained_mpp30_26b_a4b_msc` | MPP fallback: 30 trials, depths 1/3/5, myopic controls 3/5; no metrics yet, active in vLLM generation. |
| 102019 | `loc_branch_uncon26_mpp30` | `msc` | running | `oat21` | `loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_msc` | MPP fallback: 30 trials, depths 1/3/5, myopic controls 3/5; model loading/starting at latest check. |

Wall-clock check as of 2026-07-09:

- Slurm time limits are not the immediate risk: running GH200 jobs report
  `TimeLimit=UNLIMITED`, running MSC jobs report `TimeLimit=365-00:00:00`,
  and pending GH200/MSC jobs report `TimeLimit=UNLIMITED`.
- The immediate risk is throughput. No active Phase 4 run has written a final
  metrics file yet. The old GH200 jobs showed the worst throughput and were
  canceled; the active MSC jobs are making progress through vLLM generation
  batches, but completion time remains uncertain.
- The MPP30 fallback jobs are now running on MSC. Treat the first completed
  constrained/unconstrained pair as the canonical packaging input, preferring
  MPP30 for workshop scope unless the full50 MSC pair finishes first.
- On 2026-07-08, the old original GH200 jobs `101778`/`101779` were canceled
  after the user noted other people were waiting on those nodes. This does not
  remove them from the ledger; it makes them ineligible as canonical completed
  runs under the rule below.

## Phase 1 Gate

Completed 26B A4B result: `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`.
The 60-record gate passes on entropy and truth-log-probability ranking, with
depth 3 best on entropy Spearman and depth 5 best on top-1 entropy regret. This
unblocks Phase 4, but does not by itself prove a monotonic depth effect.

To reproduce the completed 26B A4B gate, run the configured scorer over five
four-trial chunks. Each chunk uses the same committed config and only changes
the trial offset.

Prefer the GH200 Singularity launcher for these ranking jobs. Do not run the
standard A100/conda ranking launcher on GH200; it will hit the ARM/aarch64
environment mismatch. The GH200 path runs inside the vLLM container.

```bash
for off in 0 4 8 12 16; do
  sbatch --partition=gh200 --job-name="rankfid26b_a4b_v2ghs_o${off}" \
    scripts/run_strategy_ranking_fidelity_gh200_singularity.sh \
    configs/config_strategy_ranking_fidelity_26b_a4b.yaml \
    --run-name "rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o${off}" \
    --num-trials 4 --trial-offset "$off" \
    --state-rounds 0,3,6 --depths 2,3,5 \
    --num-candidates 8 --deployments 8 --score-variants configured
done
```

Model-scaling note: Gemma 4B/E4B with thinking may be too weak for the spatial
reasoning needed here. Treat E4B as a cheap sanity variant only. Use
thinking-enabled 26B A4B for serious evidence, with thinking-enabled 15B as a
possible cheaper follow-up. Do not spend on non-thinking variants for this
diagnostic.

Monitor:

```bash
squeue -u hanyal -o "%.18i %.40j %.20P %.2t %.12M %.60R %.50N"

for d in runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o0 \
         runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o4 \
         runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o8 \
         runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o12 \
         runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o16; do
  echo "---$d---"
  wc -l "$d/strategy_ranking_fidelity_records.jsonl" 2>/dev/null || true
  grep -nE "completed probe|Summary:|Traceback|RuntimeError|ValueError|OOM|Killed|CANCELLED|TIMEOUT" \
    "$d/run.log" 2>/dev/null | tail -n 12 || true
done
```

Aggregate only after all five summaries exist:

```bash
python scripts/aggregate_strategy_ranking_fidelity.py \
  runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o0/strategy_ranking_fidelity_summary.json \
  runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o4/strategy_ranking_fidelity_summary.json \
  runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o8/strategy_ranking_fidelity_summary.json \
  runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o12/strategy_ranking_fidelity_summary.json \
  runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o16/strategy_ranking_fidelity_summary.json \
  --output-dir results/ranking_fidelity \
  --run-name rankfid26b_a4b_gate_v2ghs_configured_t20_m8
```

Proceed to Phase 4 only if the aggregate report's configured target-depth
Spearman entropy gate assessment is `proceed` at the 0.4 threshold. If it is
`stop`, use the ranking-fidelity report as the fallback SNR analysis.

## Phase 3 Oracle Evidence

The selected constrained environment is the branch-decoy/local-bump variant.
Regenerate the oracle report with:

```bash
python scripts/constrained_oracle_check.py \
  --num-trials 120 --num-rounds 6 --num-particles 64 --grid-size 11 \
  --arena 2.2 --max-step-radius 0.5 --noise-sd 0.15 \
  --planner-depth 2 --planning-support-size 8 \
  --source-prior branch_decoy --source-radius 2.2 \
  --signal-model local_bump --signal-lengthscale 0.5 --signal-amplitude 8.0 \
  --run-name constrained_oracle_branch_decoy_local_r22_l05_t120_d2_g11 \
  --output-dir results/constrained_oracle
```

The headline oracle evidence should remain in
`results/constrained_oracle/REPORT.md`.

## Environment Robustness Heatmap

To address the hand-tuned-environment critique, run the CPU-only robustness
sweep over signal lengthscale, movement radius, and observation noise. The
script writes per-cell oracle summaries plus a heatmap where negative values
mean the depth planner beats greedy EIG on final RMSE:

```bash
python scripts/constrained_oracle_robustness_sweep.py \
  --signal-lengthscales 0.35,0.5,0.75 \
  --max-step-radii 0.4,0.5,0.7 \
  --noise-sds 0.1,0.15,0.25 \
  --num-trials 100 \
  --num-rounds 6 \
  --num-particles 64 \
  --grid-size 13 \
  --arena 2.2 \
  --planner-depth 2 \
  --planning-support-size 8 \
  --source-prior branch_decoy \
  --source-radius 2.2 \
  --signal-amplitude 8.0 \
  --workers 8 \
  --resume \
  --run-name branch_decoy_local_robustness \
  --output-dir results/constrained_oracle_robustness \
  --plot-dir plots/constrained_oracle_robustness
```

Smoke-tested locally with a 2x2x1, two-trial sweep:
`results/constrained_oracle_robustness_smoke/smoke_robustness_REPORT.md` and
`plots/constrained_oracle_robustness_smoke/smoke_robustness_heatmap.png`. This
smoke is a mechanics check only, not paper evidence.

## Phase 4 Final Sweeps

After the Phase 1 gate passes, launch the constrained and unconstrained paired
fixed-root sweeps from committed configs. Use the thinking-enabled 26B A4B
configs for headline evidence; the E4B final50 configs are retained only as
cheap sanity variants because E4B may not be strong enough for this spatial
reasoning task. If compute pressure requires a middle point, prefer a
thinking-enabled 15B config over non-thinking variants.

Print the exact non-mutating command set before submitting:

```bash
python scripts/path_a_sync_commands.py --list
python scripts/path_a_sync_commands.py
python scripts/path_a_preflight.py --root .
python scripts/path_a_remote_readiness.py
python scripts/path_a_launch_commands.py
```

```bash
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

When both finish, compare them:

```bash
python scripts/build_path_a_package.py \
  --constrained runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_msc/fixed_root_depth_sweep_metrics.json \
  --unconstrained runs/loc_branch_decoy_local_unconstrained_mpp30_26b_a4b_msc/fixed_root_depth_sweep_metrics.json \
  --output-dir results/location_depth_sweeps \
  --cost-dir results/cost_vs_depth \
  --plot-dir plots/location_depth_sweeps \
  --run-name location_branch_decoy_depth_contrast_26b_a4b_mpp30
```

If the full50 optimized MSC pair finishes first, use:

```bash
python scripts/build_path_a_package.py \
  --constrained runs/loc_branch_decoy_local_constrained_final50_26b_a4b_sharedopt_msc2/fixed_root_depth_sweep_metrics.json \
  --unconstrained runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b_sharedopt_msc/fixed_root_depth_sweep_metrics.json \
  --output-dir results/location_depth_sweeps \
  --cost-dir results/cost_vs_depth \
  --plot-dir plots/location_depth_sweeps \
  --run-name location_branch_decoy_depth_contrast_26b_a4b_full50_msc
```

## Pre-registered Analysis

This section was written before any Phase 4 fixed-root sweep metrics were
available. It is the analysis endpoint for the final constrained/unconstrained
depth sweeps and supersedes older endpoint wording elsewhere.

Primary endpoint:

- Paired final truth log posterior probability, comparing StrategyEIG depths
  against the greedy EIG reference and against depth 1 when testing a depth
  effect.
- Report the mean paired difference and a bootstrap confidence interval over
  trials. The paired unit is the shared hidden state, initial belief, and
  observation-noise stream used by the fixed-root sweep.

Secondary endpoints:

- Paired final point-RMSE difference.
- Paired final expected posterior RMSE, defined as the posterior expectation of
  distance to the true source configuration. This is smoother than the
  point-estimate RMSE and should better match the belief-quality objective.
- Paired final posterior entropy difference.
- Per-trial win rate and paired-difference distribution plots for each main
  contrast.
- Matched-compute myopic controls, reported as attribution checks rather than
  the primary endpoint.

Supporting statistics:

- Wilcoxon signed-rank tests are supporting only. The effect estimate and
  bootstrap interval are the primary readout.
- Report constrained and unconstrained arms separately before any pooled
  statement. The constrained arm is the headline environment; the unconstrained
  arm is the contrast arm.

Rationale:

- The ranking-fidelity gate showed positive correlation for entropy and truth
  log posterior probability, but RMSE Spearman correlation was approximately
  zero. Making point-RMSE the primary endpoint would risk falsely rejecting a
  method that improves the calibrated posterior while point estimates remain
  noisy.
- The oracle check showed a large mean greedy-to-planner gap but only a 0.525
  planner win rate, so rank tests alone can miss heavy-tailed planning gains.
  Mean paired effects with bootstrap intervals are therefore the correct
  primary summary.

Canonical run-selection rule:

- Select the canonical constrained/unconstrained pair before looking at any
  Phase 4 metric values. A pair is eligible only if both arms complete the full
  requested run and write a fixed-root metrics summary.
- First choose the eligible pair with the largest completed paired-trial count.
  Therefore any complete 50-trial pair beats any complete 30-trial fallback
  pair.
- If multiple eligible pairs have the same completed paired-trial count, break
  ties by launch priority, not by result values:
  1. original full50 GH200 pair: jobs `101778` / `101779`;
  2. optimized full50 GH200 pair: jobs `101993` / `101994`;
  3. optimized full50 MSC pair: jobs `101998` / `101996`;
  4. MPP30 MSC fallback pair: jobs `102018` / `102019`.
- Before a non-original variant can become canonical, verify scientific
  config-equivalence against the original intended run: same environment,
  source prior, signal model, seed policy, trial count, round count, strategy
  depths, rollout count, scoring mode, posterior mode, and matched-compute
  controls. Throughput-only changes such as branch grouping, partition, node,
  or launch wrapper are allowed only if they do not change the scientific
  policy definition. Any result-affecting difference makes the variant an
  appendix run rather than the canonical run.
- Every other completed pair must be reported as replication or sensitivity
  evidence in the appendix. Redundant completions are never silently dropped.

## Rollout Ablation

The committed rollout-count configs are:

- `configs/config_location_branch_decoy_local_final50_rollouts8.yaml`
- `configs/config_location_branch_decoy_local_final50.yaml`
- `configs/config_location_branch_decoy_local_final50_rollouts32.yaml`

Launch each with `scripts/run_location_fixed_root_depth_sweep.sh`, then compare:

```bash
python scripts/compare_location_rollout_ablation.py \
  runs/loc_branch_decoy_local_constrained_rollouts8/fixed_root_depth_sweep_metrics.json \
  runs/loc_branch_decoy_local_constrained_final50/fixed_root_depth_sweep_metrics.json \
  runs/loc_branch_decoy_local_constrained_rollouts32/fixed_root_depth_sweep_metrics.json \
  --output-dir results/location_rollout_ablation \
  --plot-dir plots/location_rollout_ablation \
  --run-name location_branch_decoy_rollout_ablation
```

## Required Evidence Checklist

- `results/ranking_fidelity/REPORT.md`: Spearman, SNR, and top-1 regret by
  depth, including the aggregate gate assessment.
- `results/constrained_oracle/REPORT.md`: oracle greedy-vs-planner gap and
  linked RMSE figure for the selected constrained environment.
- `results/location_depth_sweeps/*_REPORT.md`: constrained/unconstrained
  paired RMSE and entropy deltas versus EIG with CIs and test statistics,
  including the `StrategyEIG-myopic-dN` matched-compute controls.
- `plots/location_depth_sweeps/*_headline_rmse.png`: paper-facing constrained
  RMSE trace for greedy EIG and StrategyEIG depths 1, 3, and 5.
- `results/location_rollout_ablation/*_REPORT.md`: rollout-count ablation.
- `results/cost_vs_depth/*_cost_vs_depth.md`: token-cost table generated from
  completed run summaries/logs.
- Each generated report should include the config path, model, host, SLURM job,
  and token usage summary when LLM calls were made.

The package builder above also generates the cost table. To regenerate only the
cost table after the fixed-root sweeps finish:

```bash
python scripts/cost_vs_depth_table.py \
  runs/loc_branch_decoy_local_constrained_final50_26b_a4b \
  runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b \
  --output-dir results/cost_vs_depth \
  --run-name location_branch_decoy_26b_a4b
```

Audit the minimum publishable package before writing up claims:

```bash
python scripts/validate_path_a_package.py --root .
```

## Recovery From Incremental Decisions

If a fixed-root sweep exits before `fixed_root_depth_sweep_metrics.json` is
written, recover the partial or complete summary from
`fixed_root_depth_sweep_decisions.jsonl`:

```bash
python scripts/recover_depth_sweep_metrics.py \
  --config configs/config_location_branch_decoy_local_final50_26b_a4b.yaml \
  --run-dir runs/loc_branch_decoy_local_constrained_final50_26b_a4b \
  --max-depth 5 \
  --include-myopic-controls
```

For the MPP30 fallback runs, pass the same subset flags used at launch:

```bash
python scripts/recover_depth_sweep_metrics.py \
  --config configs/config_location_branch_decoy_local_final50_26b_a4b.yaml \
  --run-dir runs/loc_branch_decoy_local_constrained_mpp30_26b_a4b_msc \
  --max-depth 5 \
  --include-myopic-controls \
  --strategy-depths 1,3,5 \
  --eval-depths 1,3,5 \
  --myopic-control-depths 3,5
```

Recovery reuses the selected actions and observations already in the decision
JSONL, so it does not regenerate strategies or rollout scores. Because the
current decision JSONL does not store refreshed hypothesis supports, exact
posterior metrics still replay the normal belief-generation/update path from
the config. Treat `*_recovered.json` as an auditable recovery artifact before
promoting it to the canonical metrics filename.
