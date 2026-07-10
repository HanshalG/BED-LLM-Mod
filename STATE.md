# STATE — living project state

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug. If this file conflicts with GOAL.md's design
history, this file wins.

## CURRENT STATE (updated 2026-07-10)

Status: **Path E active** — non-myopic LLM experimental design on EXTERNAL interactive
benchmarks (GOAL.md): primary env = Paprika customer-service troubleshooting tasks
(arXiv:2502.17543, released), second env = MediQ (arXiv:2406.00922; AgentClinic is fallback). Method =
BED-LLM-style beliefs + selective 2-step lookahead (tie/gating-triggered). Must beat
naive AND 1-step EIG on paired external-benchmark endpoints. 20Q/Wordle/Mastermind are
harness/unit-test only (greedy near-optimal there); location finding is closed. All
Path A/B/C/D location & 20Q material is banked history below.

Path E Step 0 implementation status (2026-07-10): implementation through commit `35bc9dd` is pushed. The
Paprika customer-service adapter now loads the hash-pinned official release, preserves
the released public `agent` scenario and private `env` solution verbatim, uses the
native semantic success rule, generates 3--5 outcome answer spaces, scores categorical
one-step EIG, judge-maps free-text replies, logs mapping coverage, and writes per-turn
artifacts. Full depth-two categorical EIG now branches over each root outcome,
recomputes the posterior, generates branch-conditioned follow-ups, and optimizes the
expected second-step gain. Paired method runs share prompt-scoped generated hypotheses, root candidates, and
simulator replies whenever their states are identical. Categorical likelihoods are
batched by action. After each mapped observation, the adapter generates history-conditioned
refinements, explicitly filters them for consistency, recomputes full-history posterior
weights, and prunes to the configured support cap. Focused adapter/config/registry/runner
tests pass. A deterministic five-task, two-round mechanics smoke against the official
pinned file has 100% mapping coverage;
it is explicitly labeled NOT LLM EVIDENCE in
`results/path_e/step0a_adapter_smoke/REPORT.json`. The exact-posterior Mastermind harness
for depth 1/2 is implemented and tested. The required real 26B A4B five-task coverage
smoke has not launched because `ssh oat0` currently fails with `No route to host`.

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

Path E reset (2026-07-10): external benchmarks with structural sequential gaps. See
GOAL.md for the six environment requirements (R1-R6) and the full validation chain.

1. **Finish Step 0a's real-model gate:** when `oat0` is reachable, sync pushed commit
   `35bc9dd`, run `scripts/fetch_paprika.py`, and launch
   `configs/config_paprika_step0a_smoke_26b_a4b.yaml` on
   `--partition=msc,llm --exclude=oat12`. Inspect all five transcripts, answer-set
   coverage (must be >= ~85%), parse failures, semantic success behavior, and forced
   thinking exits. Revise prompts and repeat rather than scaling if coverage fails.
2. **Step 0b mechanics are complete locally:** exact Mastermind posterior, one-step EIG,
   and optimal two-step lookahead tests pass. Before claims runs, retain this as a
   harness-only invariant and do not promote Mastermind to an evaluation environment.
3. **Step 1 gap pilot (cheap, gates everything):** 10 paired customer-service tasks,
   naive vs 1-step EIG vs full 2-step, 26B A4B thinking (per Hanshal: 26B A4B for ALL
   runs incl. pilots; log forced-thinking-exit rate, raise budget to 8k+ if >10-15%). Claim-1 check: 1-step > naive.
   Claim-2 check: 2-step > 1-step (>=6/10 or clear edge). Claim-2 fail -> descope to the
   claim-1 transfer study and continue. Claim-1 fail -> STOP and discuss with Hanshal.
4. **Step 2:** implement selective lookahead (tie test epsilon = scoring-noise SE +
   availability-gating trigger; CRN across tied set; depth cap 2; trigger/token
   logging); 10-task four-arm pilot; calibrate then FREEZE epsilon; check selective
   lookahead tokens <= ~40% of full 2-step.
5. **Step 3:** pre-register endpoints/analysis/epsilon/canonical-run rule in the runbook,
   then 50-100 paired customer-service tasks (powered from Step 2), four arms; then
   AgentClinic 50+ cases if on schedule. `--partition=msc,llm --exclude=oat12`.
6. **Step 4:** paper per GOAL.md playbook; location-finding saga = one honest paragraph.

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
