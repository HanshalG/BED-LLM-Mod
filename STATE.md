# STATE — living project state

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug. If this file conflicts with GOAL.md's design
history, this file wins.

## CURRENT STATE (updated 2026-07-10)

Status: **Path B negative-result package complete/validated** — the task-loss scorer did not reach the preregistered
ranking-fidelity threshold on the standard task, so Gate 1 cluster spend is closed. The
canonical common-support depth-1 replay reaches macro Spearman rho 0.231 versus realized
posterior-risk reduction (threshold approximately 0.3); 32/256/1024-rollout sensitivity
runs all fail. Evidence is in `results/ranking_fidelity/PATH_B_GATE0_TASK_LOSS.md` and
`results/ranking_fidelity/PATH_B_GATE0_DIAGNOSIS.md`. No LLM calls or cluster jobs were
used. `results/path_b/GATE1_NOT_RUN.md` records the preregistered stop. The five-page
negative-result draft and rendered PDF pass the Path B package and paper validators;
the package is traced to tag `path-b-gate0-negative-20260710`. The Path A package below
remains banked motivation material.

Phases 1–4 are DONE for the Path A workshop package. The ranking-fidelity gate, constrained
oracle, constrained support-grid MPP30 sweep, unconstrained contrast arm, paper-facing
package, 6-page paper draft, ledger, commit, push, and tag are complete. The user has
asked to use only `msc` and `llm` for any future cluster launches for now; keep
`--exclude=oat12` on new Slurm jobs.

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
4. Unconstrained contrast arm: COMPLETED for the support-grid path. Jobs `102238`,
   `102239`, and `102240` ran on `msc` nodes with `BED_LLM_SKIP_ENV_SETUP=1`,
   `--partition=msc,llm`, and `--exclude=oat12`. They were rsynced locally and combined
   into `runs/loc_branch_decoy_local_unconstrained_supportgrid_mpp30_26b_a4b_split/`.
5. Paper package: VALIDATED and SUBMITTED/AWAITING. `scripts/build_path_a_package.py` produced the final
   constrained/unconstrained comparison artifacts under `results/location_depth_sweeps/`,
   `plots/location_depth_sweeps/`, `results/cost_vs_depth/`, and
   `results/location_qualitative/`. `scripts/validate_path_a_package.py --root .` and
   `scripts/validate_experiments_ledger.py` pass. `python scripts/validate_paper_draft.py`
   passes with a 6-page draft and no TODO markers. The current branch is pushed and the
   latest submitted/awaiting state is tagged `path-a-package-20260709`.

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

Completed unconstrained support-grid MPP30 contrast:

- Config/run shape mirrors the constrained sweep, except `location_max_step_radius` is
  unset. Seed 1304, 30 paired trials, 6 rounds, branch-decoy/local-bump source, 26B A4B
  thinking, analytical posterior, fixed-support deployed beliefs, analytic rollout future
  queries, support-grid candidate generation, depths 1/3/5 plus matched-compute myopic
  controls.
- Cost: 1023 LLM calls and 3,757,774 total tokens across the three split jobs.
- Final RMSE means: naive 0.0889, naive+belief 0.0951, EIG 0.1014, StrategyEIG-d5 0.1028,
  StrategyEIG-d1 0.1029, StrategyEIG-d3 0.1077.
- Interpretation: the unconstrained arm is flat across StrategyEIG depths and all methods
  are tightly clustered; naive slightly beats EIG on final RMSE in this run. This supports
  the intended contrast that removing the movement constraint removes the measurable depth
  effect, while also reinforcing that StrategyEIG does not beat the simple baselines here.

Latest cluster state:

- Live jobs: none as of the latest `squeue -u hanyal` check. Jobs `102238`, `102239`, and
  `102240` all completed with metrics/report/plot artifacts, 480 decision rows each, and
  zero traceback/runtime/OOM/killed/location-parse errors. Token usage events are logged
  as lowercase `llm_token_usage`, not uppercase `LLM_USAGE`.
- For any additional launch, use `--partition=msc,llm --exclude=oat12` unless the user
  changes this again. Do not use GH200 unless explicitly requested again.

## NEXT ACTIONS (in order)

Path B reset (2026-07-10): GOAL.md now targets goal-oriented arbitration that must beat
naive AND greedy EIG on the STANDARD task. Path A artifacts are banked as motivation.

1. Await user review/submission feedback on the Path B negative-result package. No
   cluster jobs are needed or authorized by the completed experiment chain.
2. If revisiting the method after review, treat later-round-only abstaining arbitration
   as a new hypothesis requiring a new pre-registration; do not retroactively call the
   current Gate 0 a pass.

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
