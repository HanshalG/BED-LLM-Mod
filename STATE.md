# STATE — living project state (the agent MUST keep this file current)

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug — fix it before doing anything else. If this file
conflicts with GOAL.md's design history, this file wins.

## CURRENT STATE (updated 2026-07-09)

Phases 1–3 are DONE; Phase 4 is RUNNING; paper skeleton is started in `paper/` and
compiles. The Path A infrastructure is committed locally and tagged `path-a-final-sweep`.
GOAL.md holds the design specs, outcome playbook, and definition of done — consult it for
detail; execute from here.

Status of the Minimum Publishable Package:

1. Ranking-fidelity gate: **PASSED** — `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`
   (26B A4B, 60 records: entropy ρ 0.38–0.44, truth-log-prob ρ ≈ 0.37 positive at all
   depths, top-1 regret improves with depth 0.33→0.21, RMSE ρ ≈ 0).
2. Headline depth sweep: **RUNNING** — checked 2026-07-09 06:58 London after the user
   requested canceling the slow/low-yield GH200 path to free nodes for other users. GH200
   jobs `101993` and `101994` were already absent from the live queue; direct `gh200`
   partition checks through 06:58 London showed no user-owned GH200 jobs, so no `scancel`
   was needed.
   Remaining Path A jobs are optimized MSC jobs `101998`/`101996` running on
   `oat15`/`oat14`, constrained MPP30 job `102018` running on `oat16`, and unconstrained
   MPP30 job `102019` running on `oat21`. At 06:58 London, those four MSC jobs were the
   only live jobs for this user. No final or recovered metrics files existed at the last
   check. Decision logs were still at the cheap-control counts: MPP30 constrained `90`,
   MPP30 unconstrained `90`, full50 constrained `150`, and full50 unconstrained `150`. Slurm
   stderr is non-erroring for all four jobs. Recent progress snapshots were `101996`
   reached 98/256 in its current prompt block, `101998` reached 120/256, `102018`
   finished the 131-prompt block and started a new 55-prompt block, and `102019` reached
   167/256.
   None of the four decision logs advanced, and no metrics or recovered metrics files
   exist yet. All four jobs remain live in `squeue`; all four Slurm stderr files advanced
   during this poll.
3. Matched-compute myopic controls: **RUNNING** — included in the same jobs
   (`--include-myopic-controls`).
4. Cost-vs-depth table: script ready (`scripts/cost_vs_depth_table.py`), runs at packaging.

Oracle evidence: `results/constrained_oracle/REPORT.md` (branch-decoy/local-bump env,
greedy 0.56 vs planner 0.15 final RMSE, win rate 0.525 — heavy-tailed wins).
Environment robustness heatmap: **DONE** —
`results/constrained_oracle_robustness/branch_decoy_local_robustness_REPORT.md` and
`plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png`
(27 cells, 100 trials/cell). The Phase 4 operating point is one of the strongest
mapped cells: planner − greedy final RMSE = -0.175 at lengthscale 0.5, radius 0.5,
noise 0.15; many nearby cells are weak/near-zero, so frame this as a mapped
planning-sensitive regime rather than a universal property of the location family.
Positioning: `results/POSITIONING.md` (COPEx + IPP covered). Operational commands:
`LOCATION_DEPTH_PATH_A_RUNBOOK.md` and `PHASE4_LAUNCH_HANDOFF.md`; both now point the
packaging command at the active MPP30 MSC run names first, with the optimized full50 MSC
pair as the fallback if it finishes first. The Phase 4 endpoint is pre-registered in
`LOCATION_DEPTH_PATH_A_RUNBOOK.md` before metrics landed. `validate_path_a_package.py`
currently passes the ranking-fidelity and oracle checks but fails the expected missing
depth-sweep report, headline plot, and cost-vs-depth table until Phase 4 metrics exist.
Paper env framing: **DONE for the current skeleton** — `paper/main.tex` now presents the
task as a mobile-sensor, movement-cost, finite-range-sensing, branch-decoy environment;
states that geometry selection was method-blind with respect to StrategyEIG; includes the
robustness heatmap figure; and scopes the claim to a mapped planning-sensitive regime.
Paper ranking-fidelity section: **DONE for the current skeleton** — `paper/main.tex`
now reports the 26B A4B Phase 1 gate numbers, includes the aggregate ranking-fidelity
plot, states the positive entropy/truth-log-prob first-link result, and preserves the
near-zero RMSE-rank caveat.
Paper method/protocol section: **DONE for the current skeleton** — `paper/main.tex`
now describes strategy/root generation, analytical rollout scoring, fixed-common paired
scoring controls, the ranking-fidelity deployment diagnostic, and the pre-registered
paired depth-sweep endpoints/myopic controls.
Paper related-work citations: **DONE for the current skeleton** — `paper/main.tex` now
cites DAD, COPEx/constrained BED, and BED-LLM using `paper/references.bib`. The paper
compiled to 5 pages with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`; generated PDF and
auxiliary files were removed from the worktree.
Paper limitations section: **DONE for the current skeleton** — `paper/main.tex` now
covers workshop-scale trial counts, one constrained environment family, constructed
method-blind geometry with robustness heatmap scope evidence, forced-thinking-exit/token
reporting, RMSE as a noisy secondary endpoint, and the missing MPC/action-sequence
ablation as future work. The paper compiled to 5 pages with `pdflatex` twice after this
edit; generated PDF/auxiliary files were removed from the worktree.

RMSE repair analysis: **DONE for current records** —
`results/ranking_fidelity/RMSE_REPAIR.md` and
`results/ranking_fidelity/rmse_repair_analysis.json`. Realized entropy/truth-log-prob
gains are only weakly rank-aligned with realized point-RMSE gains; expected posterior RMSE
cannot be recovered exactly from the current aggregate records because final posterior
supports/probabilities were not logged.
Posterior-state logging for future ranking-fidelity runs: **DONE in code** —
`scripts/strategy_ranking_fidelity.py` now records start expected posterior RMSE,
candidate-level expected posterior RMSE means/drops, and per-deployment final posterior
hypothesis supports/probabilities. `scripts/ranking_fidelity_rmse_repair.py` detects these
future fields and reports expected-posterior-RMSE-drop alignment when available.
Config archive cleanup: **DONE locally** — numbered pilot/smoke configs were moved from
top-level `configs/` into `configs/archive/numbered/`; live top-level configs are now the
descriptively named Path A/ranking/oracle configs plus `configs/cluster_smoke/`.

## NEXT ACTIONS (in order, all local-only, none touch the running jobs)

1. When jobs finish: recovery-or-normal packaging via `PHASE4_LAUNCH_HANDOFF.md`, then
   analysis strictly per the pre-registered section, then results into the skeleton
   following the OUTCOME PLAYBOOK row in GOAL.md that applies.

## OPERATIONAL KNOWLEDGE (repo memory — keep updated here, not in chat)

- Cluster: `ssh oat0`, checkout
  `/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z`.
- GH200 nodes MUST use the Singularity/container launchers
  (`scripts/run_*_gh200_singularity.sh`, container `docker://vllm/vllm-openai:gemma4`);
  the A100/conda path hits an ARM/aarch64 wheel mismatch.
- Partition preference: `gh200` > `msc` > `llm`. Keep ≤8 active jobs. Exclude `oat12`
  (`--exclude=oat12`). Watch oat19 scratch SSD usage (was nearly full once; caches and
  old model families were purged).
- Standard launch env: `BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}'`,
  `BED_LLM_LOG_REASONING_TRACES=1` (add `BED_LLM_REASONING_TRACE_CHARS=full` for full
  traces); `BED_LLM_SKIP_ENV_SETUP=1` on prepared A100/conda launchers.
- Sync: `python scripts/path_a_sync_commands.py` (explicitly includes the final50
  configs because `.gitignore` ignores `configs/*` — keep that in mind for new configs).
- Models: thinking-enabled `google/gemma-4-26B-A4B-it` for headline evidence; 4B/E4B are
  too weak for spatial strategies (sanity runs only); non-thinking variants are not worth
  spending on. Thinking budget 4096 → ~30% forced-exit rate (12k/41k calls in the final
  sweeps) — first suspect if results are marginal; the one reserved appendix follow-up is
  an 8k-budget replicate of depths {1, 5}.
- Tests: `pytest tests/ -q` must stay green (last known: 404 passed, 1 skipped).
