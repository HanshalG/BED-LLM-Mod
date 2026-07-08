# STATE — living project state (the agent MUST keep this file current)

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug — fix it before doing anything else. If this file
conflicts with GOAL.md's design history, this file wins.

## CURRENT STATE (updated 2026-07-08)

Phases 1–3 are DONE; Phase 4 is RUNNING; paper skeleton is started in `paper/` and
compiles. The Path A infrastructure is committed locally and tagged `path-a-final-sweep`.
GOAL.md holds the design specs, outcome playbook, and definition of done — consult it for
detail; execute from here.

Status of the Minimum Publishable Package:

1. Ranking-fidelity gate: **PASSED** — `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`
   (26B A4B, 60 records: entropy ρ 0.38–0.44, truth-log-prob ρ ≈ 0.37 positive at all
   depths, top-1 regret improves with depth 0.33→0.21, RMSE ρ ≈ 0).
2. Headline depth sweep: **RUNNING/PENDING** — six active Path A jobs remain after
   canceling the old too-slow GH200 originals `101778`/`101779`: optimized GH200 job
   `101993` is running on `oat21`, optimized GH200 job `101994` is pending, optimized
   MSC jobs `101998`/`101996` are running on `oat15`/`oat14`, constrained MPP30 job
   `102018` is running on `oat16`, and unconstrained MPP30 job `102019` is pending.
3. Matched-compute myopic controls: **RUNNING** — included in the same jobs
   (`--include-myopic-controls`).
4. Cost-vs-depth table: script ready (`scripts/cost_vs_depth_table.py`), runs at packaging.

Oracle evidence: `results/constrained_oracle/REPORT.md` (branch-decoy/local-bump env,
greedy 0.56 vs planner 0.15 final RMSE, win rate 0.525 — heavy-tailed wins).
Positioning: `results/POSITIONING.md` (COPEx + IPP covered). Operational commands:
`LOCATION_DEPTH_PATH_A_RUNBOOK.md` and `PHASE4_LAUNCH_HANDOFF.md`. The Phase 4 endpoint
is pre-registered in `LOCATION_DEPTH_PATH_A_RUNBOOK.md` before metrics landed.

## NEXT ACTIONS (in order, all local-only, none touch the running jobs)

1. **Confirm MPP30 startup and keep active jobs <=8.** `102018` started after canceling
   `101778`/`101779`, but its run directory had not appeared at the first post-start
   check. Recheck startup logs/run directory before assuming progress.
2. **RMSE repair analyses (free, from existing gate JSONL — no new runs).**
   (a) Exculpation check: correlation between REALIZED entropy drop and REALIZED ΔRMSE
   across candidates within each probe, plus RMSE between/within-strategy SNR. If
   realized-realized ≈ 0, no scorer could rank point-RMSE at probe horizons — the
   endpoint is unrankable and the estimator is exonerated; that sentence goes in the
   paper. (b) Recompute realized RMSE as EXPECTED posterior RMSE from the stored final
   posteriors + truth, and redo the Spearman ρ table with it. Expected: it correlates
   near truth-log-prob levels, completing the story (scorer predicts posterior quality;
   point-RMSE is a noisy discretization). Append both to
   `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`.
3. **Environment robustness heatmap (CPU-only, addresses the "hand-tuned env" critique).**
   Extend `scripts/constrained_oracle_check.py` into a parameter sweep: oracle gap
   (planner − greedy final RMSE, non-LLM, analytic) over a grid of signal lengthscale ×
   max step radius × noise_sd (~3×3×3, 100+ trials per cell, embarrassingly parallel on
   CPU). Deliverable: a heatmap of where non-myopia pays, with the Phase 4 operating
   point marked. This converts "we tuned until it worked" into "we mapped the region
   where planning matters and evaluated inside it" — the strongest available answer to
   the contrived-environment review. Appendix figure + 2 sentences in main text.
4. **Env framing in the paper (free, write into the skeleton).** Present the env with
   its physical semantics — mobile agent, movement cost (locality constraint),
   short-range sensor (local-bump finite-range signal), junction structure (branch-decoy
   prior) — not as an abstract tuned geometry. State explicitly that geometry selection
   was METHOD-BLIND (tuned against non-LLM oracle policies only; StrategyEIG never
   entered the tuning loop). Report the failed geometries transparently as a finding:
   most geometries are greedy-friendly, myopic traps are rare in this family — which
   explains the original null results and motivates the constructed instance.
5. **Archive dead configs.** Move numbered `configs/config*.yaml` not referenced by any
   Path A artifact into `configs/archive/`; live configs must be findable at a glance.
6. When jobs finish: recovery-or-normal packaging via `PHASE4_LAUNCH_HANDOFF.md`, then
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
