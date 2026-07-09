# STATE — living project state

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug. If this file conflicts with GOAL.md's design
history, this file wins.

## CURRENT STATE (updated 2026-07-09)

Phases 1–3 are DONE. Phase 4 has one completed constrained support-grid MPP30 sweep and
no active cluster jobs as of the latest `squeue` check. The user has asked to use only
`msc` and `llm` for cluster launches for now; keep `--exclude=oat12` on new Slurm jobs.

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
4. Unconstrained contrast arm: NOT RUN for the support-grid path. The existing
   `scripts/build_path_a_package.py` still expects both constrained and unconstrained
   summaries. Do not assume the package validator can pass from constrained-only evidence.
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

- `squeue -u hanyal` returned no jobs. Nothing is active or pending for this user.
- For any new launch, use `--partition=msc,llm --exclude=oat12` unless the user changes
  this again. Do not use GH200 unless explicitly requested again.

## NEXT ACTIONS (in order)

1. Keep state and ledger truthful: validate `EXPERIMENTS.md` after every edit and commit
   the state/ledger/result summary update.
2. Decide with evidence whether to spend on the unconstrained support-grid contrast arm.
   It is needed by the current Path A package builder, but the constrained result already
   failed the optimistic "StrategyEIG beats EIG/naive" headline. If launched, use split
   jobs on `msc,llm`, `--exclude=oat12`, and keep the active-job cap at or below 8.
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
