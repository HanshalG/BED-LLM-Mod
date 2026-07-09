# STATE — living project state

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug. If this file conflicts with GOAL.md's design
history, this file wins.

## CURRENT STATE (updated 2026-07-09)

Phases 1–3 are DONE. Phase 4 has one completed constrained support-grid MPP30 sweep and
one unconstrained support-grid MPP30 contrast split currently live. The user has asked to
use only `msc` and `llm` for cluster launches for now; keep `--exclude=oat12` on new
Slurm jobs.

Minimum Publishable Package status:

1. Ranking-fidelity gate: PASSED. Evidence is in
   `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md` and
   `results/ranking_fidelity/REPORT.md`. The 26B A4B gate had entropy Spearman rho
   around 0.38–0.44, truth-log-prob rho positive around 0.37, top-1 regret improving with
   depth, and RMSE rho near zero.
2. Constrained oracle: DONE. Evidence is in `results/constrained_oracle/REPORT.md` and
   the robustness heatmap artifacts. This remains the non-LLM yardstick showing the
   locality-constrained branch-decoy task can have a planning gap.
3. Headline constrained LLM depth sweep: COMPLETED for the support-grid constrained MPP30
   variant. Jobs `102226`, `102227`, and `102228` ran on `msc` / `oat11` with
   `BED_LLM_SKIP_ENV_SETUP=1` and `--exclude=oat12`. They were combined locally into
   `runs/loc_branch_decoy_local_constrained_supportgrid_mpp30_26b_a4b_split/`.
   A tracked summary is in
   `results/location_depth_sweeps/constrained_supportgrid_mpp30_26b_a4b_summary.md`.
4. Unconstrained contrast arm: LAUNCHED for the support-grid path as three 10-trial split
   jobs on `msc,llm` with `--exclude=oat12`:
   - `102238` / `loc_branch_uncon26_sg_b00`, trials 0--9, allocated on `msc` / `oat11`.
   - `102239` / `loc_branch_uncon26_sg_b10`, trials 10--19, allocated on `msc` / `oat11`.
   - `102240` / `loc_branch_uncon26_sg_b20`, trials 20--29, allocated on `msc` / `oat14`.
   These mirror the constrained support-grid MPP30 sweep but use
   `configs/config_location_branch_decoy_local_unconstrained_supportgrid_mpp30_26b_a4b.yaml`.
5. Paper package: INCOMPLETE. The paper skeleton exists and had compiled before this
   support-grid result, but result framing now needs to reflect the actual MPP30 outcome.

Completed constrained support-grid MPP30 result:

- Config/run shape: seed 1304, 30 paired trials, 6 rounds, branch-decoy/local-bump source,
  max step radius 0.5, 26B A4B thinking, analytical posterior, fixed-support deployed
  beliefs, analytic rollout future queries, support-grid candidate generation, depths
  1/3/5 plus matched-compute myopic controls.
- Cost: 1008 LLM calls and 3,869,520 total tokens across the three split jobs.
- Hidden-path sanity: combined block counters had zero LLM candidate-generation calls,
  zero strategy-location rollout calls, support-grid active, and belief refresh disabled.
- Final RMSE means: naive 0.166, EIG 0.412, StrategyEIG-d5 0.542, StrategyEIG-d1 0.929,
  StrategyEIG-d3 1.127, naive+belief 1.137. Standard deviations are large; see the
  tracked summary for the full table.
- Final truth-log-prob means: naive -1.484, EIG -1.672, d5 -1.982, d1 -2.532, d3 -2.590,
  naive+belief -2.578.
- Interpretation: this is NOT evidence that StrategyEIG beats greedy EIG or naive. It
  does show a depth/objective effect within StrategyEIG: d5 is materially better than
  d1/d3 and the matched-compute myopic controls on RMSE, entropy, truth log-prob, selected
  EIG, and realized entropy drop. The honest framing is "non-myopic scoring improves over
  myopic/short-horizon StrategyEIG under constraints, but the LLM strategy scaffold still
  trails the analytic greedy/naive baselines in this run."

Latest cluster state:

- Live jobs: `102238`, `102239`, and `102240`, all on `msc` nodes and none on `oat12`.
  As of the latest health check, `102238` and `102239` are on `msc` / `oat11`, and
  `102240` is on `msc` / `oat14`. The run directories exist and each has a `run.log`.
  At about 4 minutes elapsed, all three had loaded the model, completed vLLM warmup/graph
  capture, and were still in initial hypothesis generation. No decision files or metrics
  existed yet, which is expected this early. The first initial-belief calls had succeeded
  on the split jobs, with parsed 12-valid-source batches appearing in logs. Early greps
  found zero traceback/runtime/OOM/killed/location-parse errors. Note for future checks:
  token usage events are logged as lowercase `llm_token_usage`, not uppercase `LLM_USAGE`.
- For any additional launch, use `--partition=msc,llm --exclude=oat12` unless the user
  changes this again. Do not use GH200 unless explicitly requested again.

## NEXT ACTIONS (in order)

1. Monitor `102238`, `102239`, and `102240` until they finish, then rsync their run
   directories locally and combine them with `scripts/combine_location_fixed_root_depth_sweeps.py`.
2. If the unconstrained arm finishes, run the current package builder with the constrained
   and unconstrained support-grid summaries; if the result framing still needs a
   constrained-only fallback, implement that explicitly rather than silently bypassing
   validation.
3. Update the paper/result framing away from "StrategyEIG beats baselines" and toward the
   outcome-playbook row where planning depth/objective improves information metrics or
   StrategyEIG internals, but RMSE/baseline wins remain partial.
4. If packaging constrained-only evidence, either extend the package scripts explicitly or
   create a separate constrained-only appendix/report path. Do not silently pass off a
   constrained-only package as the full constrained/unconstrained MPP.

## OPERATIONAL KNOWLEDGE

- Remote checkout: `/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z`.
- Use `ssh oat0` for cluster access. Use `PYTHONNOUSERSITE=1` for remote login-node Python
  when importing scientific packages; add `PYTHONPATH=.` for repo imports.
- `runs/` is gitignored. Copy paper-facing summaries into `results/` or `plots/` if they
  must be tracked.
- For `msc`/`llm` launches of the fixed-root sweep, use
  `BED_LLM_SKIP_ENV_SETUP=1 sbatch --partition=msc,llm --exclude=oat12 ...`.
- Keep at most 8 active jobs. Count running and pending jobs before new submissions.
- `location_candidate_generation_mode: support_grid` removes LLM candidate generation for
  EIG/root candidate sets. It does not remove strategy generation calls.
- Total EIG bounds for LLM policies must compute from primary histories unless the user
  explicitly requests held-out rollouts; bounds evaluation should not trigger extra LLM
  calls.
