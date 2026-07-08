# GOAL: Make non-myopic StrategyEIG show a real depth effect (Path A), gated on a ranking-fidelity diagnostic

## HOW TO USE THIS DOCUMENT

This file drives all work toward a NeurIPS-workshop submission. Rules:

- **Maintenance contract**: at the end of every working session, update CURRENT STATE and
  NEXT ACTIONS to reflect reality, and append a row to `EXPERIMENTS.md` for anything
  launched. A stale CURRENT STATE is a bug — fix it before doing anything else.
- **Precedence**: if CURRENT STATE / NEXT ACTIONS conflicts with the design history below
  the divider, the top sections win.
- **Cluster restriction**: keep at most 8 active jobs on the cluster. No other standing
  restrictions.

## OUTCOME PLAYBOOK (decide framing on results day by table lookup, not improvisation)

| Phase 4 outcome | Headline framing | Results-section emphasis |
|---|---|---|
| Depth effect on truth-log-prob AND RMSE, constrained only | "NL-strategy planning recovers non-myopic gains under constraints" | Main contrast figure; gap-closure vs oracle; controls confirm objective not compute |
| Effect on truth-log-prob/entropy but not RMSE | "LLMs can rank their own experimental plans; localization gains are partial" | Ranking fidelity + info-metrics primary; honest RMSE discussion tied to gate's RMSE ρ≈0 |
| Depth effect exists but matched-compute myopic control MATCHES n-step scoring | "Rollout compute, not the non-myopic objective, drives gains" | Honest attribution study; still novel (nobody has budget-matched this); diagnostics explain why |
| No paired effect anywhere | "Can LLMs evaluate their own experimental plans? A diagnostic study" | Gate + SNR analysis + oracle gap as the open problem; sweep as evidence of the estimation-deployment gap |

All rows are the same paper skeleton with different emphasis — the framing was chosen to
make this true. No row requires new experiments before submission.

## DEFINITION OF DONE

1. Pre-registered analysis executed exactly as written; results packaged
   (`validate_path_a_package.py` passes).
2. Four figures exist: main contrast, ranking-fidelity diagnostics, cost-vs-depth,
   qualitative strategies/trajectories.
3. Complete 4–6 page draft in `paper/` following the OUTCOME PLAYBOOK row that applies,
   with limitations section covering: forced-thinking-exit rate, single environment
   family, no MPC ablation (future work), workshop-scale trial counts.
4. `EXPERIMENTS.md` accounts for every result in the paper, each traceable to a commit/tag.
5. Repo state committed and tagged; CURRENT STATE updated to "submitted/awaiting".

## WRITE-NOW LIST (paper content writable before the sweeps finish)

Method section (from the design specs below), ranking-fidelity results (banked),
constrained-env design + oracle evidence (banked), related work (from
`results/POSITIONING.md`), intro/motivation (from Context + Paper story sections),
limitations skeleton. Only the Phase 4 results subsection and abstract numbers wait
on the jobs.

## CURRENT STATE (updated 2026-07-08) — read this first

Phases 1–3 are DONE; Phase 4 is RUNNING; writing has not started. The Path A
infrastructure is committed locally and tagged `path-a-final-sweep`. Later sections of
this document are the design history and detailed specs — consult them when a step below
needs detail, but execute from here.

Status of the Minimum Publishable Package:

1. Ranking-fidelity gate: **PASSED** — `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`
   (26B A4B, 60 records: entropy ρ 0.38–0.44, truth-log-prob ρ ≈ 0.37 positive at all
   depths, top-1 regret improves with depth 0.33→0.21, RMSE ρ ≈ 0).
2. Headline depth sweep: **RUNNING/PENDING** — eight active Path A jobs are tracked in
   `LOCATION_DEPTH_PATH_A_RUNBOOK.md` and `EXPERIMENTS.md`: original full50 jobs
   `101778`/`101779`, optimized GH200 jobs `101993`/`101994`, optimized MSC jobs
   `101998`/`101996`, and MPP30 fallback jobs `102018`/`102019`.
3. Matched-compute myopic controls: **RUNNING** — included in the same jobs
   (`--include-myopic-controls`).
4. Cost-vs-depth table: script ready (`scripts/cost_vs_depth_table.py`), runs at packaging.

Oracle evidence: `results/constrained_oracle/REPORT.md` (branch-decoy/local-bump env,
greedy 0.56 vs planner 0.15 final RMSE, win rate 0.525 — heavy-tailed wins).
Positioning: `results/POSITIONING.md` (COPEx + IPP covered). Operational commands:
`LOCATION_DEPTH_PATH_A_RUNBOOK.md` and `PHASE4_LAUNCH_HANDOFF.md`. The Phase 4 endpoint
is pre-registered in `LOCATION_DEPTH_PATH_A_RUNBOOK.md` before metrics landed.

## NEXT ACTIONS (in order, all local-only, none touch the running jobs)

1. **Check the gh200/msc SLURM wall-clock limits against projected runtime.** Recovery
   script exists (`scripts/recover_depth_sweep_metrics.py`) and is smoke-tested, but the
   live jobs may still hit wall-clock before a complete round of StrategyEIG metrics.
2. **Create the paper skeleton.** `paper/` with a 4–6 page
   workshop layout: abstract stub using the question-driven framing ("Can LLMs evaluate
   their own experimental plans?"), section stubs, and placeholder slots for the four
   figures: (i) main paired depth/contrast figure, (ii) ranking-fidelity diagnostics,
   (iii) cost-vs-depth table, (iv) qualitative strategies+trajectories figure. The
   WRITE-NOW LIST covers what can be drafted from already-banked results.
3. **Add qualitative-figure extraction to packaging.** Extend
   `scripts/build_path_a_package.py` (or a sibling script) to pull 2–3 verbatim evolved
   NL strategies with their query trajectories from the sweep runs — ideally a trial
   where greedy stalls at a decoy branch and StrategyEIG routes past it. Reasoning
   traces are already logged.
4. **RMSE repair analyses (free, from existing gate JSONL — no new runs).**
   (a) Exculpation check: correlation between REALIZED entropy drop and REALIZED ΔRMSE
   across candidates within each probe, plus RMSE between/within-strategy SNR. If
   realized-realized ≈ 0, no scorer could rank point-RMSE at probe horizons — the
   endpoint is unrankable and the estimator is exonerated; that sentence goes in the
   paper. (b) Recompute realized RMSE as EXPECTED posterior RMSE from the stored final
   posteriors + truth, and redo the Spearman ρ table with it. Expected: it correlates
   near truth-log-prob levels, completing the story (scorer predicts posterior quality;
   point-RMSE is a noisy discretization). Append both to
   `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`.
5. **Archive dead configs.** Move numbered `configs/config*.yaml` not referenced by any
   Path A artifact into `configs/archive/`; live configs must be findable at a glance.
6. When jobs finish: recovery-or-normal packaging via `PHASE4_LAUNCH_HANDOFF.md`, then
   analysis strictly per the pre-registered section, then results into the skeleton
   following the OUTCOME PLAYBOOK row that applies.

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

---

Everything below is the detailed design and its history. It remains authoritative for
specs (gate definitions, ablation designs, scope discipline, positioning, fallback), but
day-to-day execution runs off CURRENT STATE / NEXT ACTIONS above. Note: some specifics
below predate later decisions (e.g. headline model is now thinking-enabled 26B A4B, the
launched sweeps run depths 1–5 at 50/50 trials, and the pre-registered analysis section
in the runbook owns the endpoint definitions) — where they differ, CURRENT STATE wins.

## Context

This repo implements LLM-driven sequential Bayesian experimental design (see `README.md`).
The claim we want to publish: **strategy-based n-step EIG planning (StrategyEIG) beats myopic
EIG and naive prompting, and improves with planning depth.** Current results are null on the
`location_finding` env (see `plots/location_rmse_trace_{4b,26b}_tse.png` — D1–D5 traces are
statistically indistinguishable, 10 unpaired trials).

Suspected causes, in order:

1. **Strategy ranking is noise**: the MC rollout estimator of n-step EIG may not rank
   candidate strategies better than chance, especially at depth ≥ 3. If true, nothing
   downstream can work. This is the first thing to test.
2. **No statistical power**: 10 unpaired trials with cross-trial RMSE SD ~0.4 cannot detect
   plausible depth effects.
3. **Environment is greedy-friendly**: static Gaussian source localisation has approximately
   submodular info gain, so greedy EIG is near-optimal and depth has little to buy.
4. **Task dynamic range too small**: 3 sources = 6 params, noise_sd 0.5, 10 rounds → RMSE
   only moves ~1.1 → 0.75.

The plan: Phase 1 measures (1). Phase 2 fixes estimator variance. Phase 3 fixes (3)+(4) with
a locality-constrained variant of the env. Phase 4 runs the properly powered depth sweep.
**Do not skip the Phase 1 gate.**

## SCOPE DISCIPLINE — this is a WORKSHOP paper (read first)

Target is a NeurIPS workshop (4–6 pages, lenient review), not a conference paper. The
dominant risk is not reviewer rigor — it is not finishing, or diluting the story across
half-run experiments. The **Minimum Publishable Package (MPP)** is exactly four results:

1. Ranking-fidelity result (Phase 1 — nearly complete; independently interesting).
2. Headline figure: constrained env, StrategyEIG depths {1, 3, 5} vs greedy EIG, paired,
   30–50 trials, with CIs.
3. Matched-compute myopic-scoring control (defends the thesis: non-myopic objective, not
   extra compute).
4. Cost-vs-depth table: StrategyEIG vs brute-force n-step EIG, from existing token logs
   (nearly free).

Everything else in this document is appendix material or cut, in this order when time
pressures bite: Phase 5 (cut first — it is a second paper), 26B replication, MPC ablation,
unconstrained-env contrast arm (shrink to 25 trials before cutting), extra baselines
(keep only greedy EIG + naive+belief in the main figure). Do not start any non-MPP item
while an MPP item is incomplete or unlaunched. A positive-but-modest result presented
honestly ("depth ≥ 3 closes X% of the greedy→oracle gap under locality constraints;
ranking fidelity ρ ≈ 0.5 explains why gains are partial") clears the workshop bar. So
does the fallback SNR paper. A sprawling half-finished grid does not.

## Key code locations

- Strategy generation + rollout scoring: `environments/location_finding/strategy.py`
  (`evaluate_location_strategies_by_rollout_many` ~L1077, `_rollout_entropy_reduction_score`
  ~L740, scoring-support modes ~L653–714, rollout controls with shared
  `(truth, start_probability, scoring_seed, noise_zs)` ~L595–604).
- Method dispatch: `methods/strategy.py` → `environment.choose_strategy_action(s)_many`.
- Env: `environments/location_finding/env.py`, physics in `physics.py`, greedy EIG in
  `eig.py` / `depth2_eig.py`.
- Config dataclass + strategy knobs: `helpers.py` L116–130
  (`location_strategy_num_rollouts`, `location_strategy_planning_depth`,
  `location_strategy_rollout_scoring_support_mode`, `location_strategy_rollout_score_mode`, …).
- Depth-sweep runner (paired-branch structure already exists): `scripts/location_fixed_root_depth_sweep.py`.
- Example config: `config_location_finding.yaml`. Runner: `python main.py -c <config>`.

Run `pytest tests/` before and after changes. Keep every new experiment as a script under
`scripts/` plus a config under `configs/`, writing JSON results + a plot under `results/` and
`plots/`.

## Phase 1 — Ranking-fidelity diagnostic (THE GATE)

**Question: does the estimated n-step EIG of a strategy predict its realized information
gain / RMSE reduction when actually deployed?**

Build `scripts/strategy_ranking_fidelity.py`:

1. For T trials (T ≥ 20) × several belief states (e.g. at rounds 0, 3, 6 of a run):
   a. Generate K candidate strategies (K ≥ 8) via the existing generation path.
   b. Score each with the rollout estimator at depths d ∈ {2, 3, 5} → `estimated_score[k]`.
   c. **Ground truth**: deploy each strategy for d real simulated rounds against the true
      hidden state (M ≥ 8 independent deployments per strategy, fresh noise), run the real
      belief-update pipeline, record realized entropy reduction on a fixed common support
      AND realized ΔRMSE → `realized_score[k]` (mean over M).
2. Report, per depth: Spearman rank correlation between estimated and realized scores
   (per belief-state, then averaged with a bootstrap CI); top-1 regret (realized score of
   the estimator's argmax vs the true best strategy); and the estimator's within-strategy
   rollout variance vs between-strategy variance (SNR = var_between / var_within).
   **Top-1 regret is the primary deployment-relevant number** — only the argmax strategy
   is ever deployed, so report it most prominently.
3. **Truth-anchored realized metric (required)**: realized entropy drop and estimated EIG
   are both posterior-concentration measures, so they can correlate even when the posterior
   concentrates on the WRONG hypothesis. Add realized log posterior probability of the
   truth (nearest reservoir hypothesis to the true source config) at deployment end, and
   compute Spearman ρ against it as well. RMSE alone is too noisy to serve this role.
4. Also compute the same correlations after Phase 2 variance fixes, to show the improvement.
5. **Strategy execution fidelity** (undetected failure mode otherwise): from the realized
   deployments already collected in step 1c, test whether different strategies produce
   systematically different query trajectories — between-strategy vs within-strategy
   (across deployment seeds) distances between query sets. If between ≈ within, the
   query generator ignores the strategy text and no amount of ranking fidelity matters;
   fix prompts/temperature before proceeding.

**Gate:**
- Spearman ρ ≥ ~0.4 on realized entropy AND clearly positive ρ on truth-log-prob at the
  target depth (after Phase 2 fixes) → proceed to Phases 3–4.
- Entropy ρ high but truth-log-prob ρ ≈ 0 → estimator selects for confident-but-wrong
  posteriors. Diagnose (reservoir doesn't contain truth? support mismatch?) before any
  Phase 4 spend; this outcome is itself a publishable calibration finding.
- ρ ≈ 0 that cannot be rescued by Phase 2 → STOP Path A; the deliverable becomes the
  diagnostic itself + SNR-vs-depth analysis (this is the fallback workshop paper).
  Write up findings in `results/ranking_fidelity/REPORT.md` either way.

## Phase 2 — Variance reduction in strategy scoring

1. **Verify common random numbers (CRN)**: rollout controls are already built once and
   shared across strategies (`strategy.py` ~L595). Confirm that every stochastic component
   downstream — LLM query generation temperature/seed, observation noise (`noise_zs`),
   hypothesis sampling, final belief refresh — is identical across strategies within a
   rollout index. Fix any leaks so strategy comparisons are fully paired.
2. **Remove the noisiest scoring component**: make the end-of-rollout "deployment-realistic"
   LLM belief-refresh optional and OFF for scoring (config flag). Score entropy reduction by
   closed-form reweighting on a fixed common support built once per scoring call
   (union of current reservoir + truth), i.e. extend
   `location_strategy_rollout_scoring_support_mode` with a `fixed_common` mode where the
   SAME support is used for all strategies and all rollouts in a comparison.
3. **Paired scoring statistic**: rank strategies by mean of per-rollout-index paired
   differences vs a reference strategy (or just rely on CRN + common support).
4. Re-run the Phase 1 diagnostic after each change; keep a small table of ρ and SNR per
   configuration in `results/ranking_fidelity/`.

## Phase 3 — Myopic-trap environment: locality-constrained location finding

Greedy must have a provable gap for depth to matter.

1. Add config option `environment.max_step_radius: float` (e.g. 0.5 on a [-2,2]² arena).
   Constraint: each query must lie within `max_step_radius` of the previous query (first
   query unconstrained or fixed at origin). Enforce in the env: clip/reject invalid actions
   (project onto the feasible disk); tell the LLM the constraint in every prompt
   (`prompts.py`), including strategy generation and rollout query generation.
2. Make the task better-conditioned: default to `num_sources: 1` (or 2), `noise_sd: 0.25`,
   `num_rounds: 15–20`, so RMSE has dynamic range.
3. Sanity check the trap with a cheap non-LLM oracle experiment
   (`scripts/constrained_oracle_check.py`): four arms on a coarse action grid with
   analytic likelihood, 200+ trials — (a) grid greedy-EIG, (b) depth-d receding-horizon
   planner, (c) **non-adaptive lawnmower/systematic-coverage policy**, (d) random walk.
   **Requirements: planner > greedy AND planner > lawnmower, both by clear margins.**
   The lawnmower arm is critical: under locality constraints, non-adaptive coverage is
   the classic killer baseline — if lawnmower ≈ planner, the env rewards coverage, not
   adaptive planning, and the contrast is trivial. Tune the env until adaptivity pays:
   round budget short enough that full coverage is impossible, sources concentrated so
   signal-dependent routing matters. This defines the Phase 4 env and is a paper figure.
   **This check needs no LLM compute — run it in parallel with Phase 1, not after it.**
4. **Power calculation before Phase 4 launch**: use the measured greedy→planner gap and
   cross-trial SD from the oracle check to compute the trial count needed to detect
   StrategyEIG closing ~50% of the gap (paired test, 80% power). If it exceeds the
   compute budget, adjust env difficulty (noise, rounds) to widen the gap first.

## Phase 4 — Powered, paired depth sweep

1. Methods (workshop-scoped): `naive+belief`, greedy `EIG` (quadrature), `StrategyEIG`
   at depths {1, 3, 5}. Analytical posterior mode (LLM posterior is out of scope).
   Before launching, use the token-usage logs from Phase 1 to project total cost/wall-clock
   and run the power calc from Phase 3.4. 30–50 paired trials.
2. **Pairing**: identical hidden states, initial beliefs, and per-round observation noise
   across ALL methods/depths within a trial (extend the branch structure in
   `scripts/location_fixed_root_depth_sweep.py`). 50 trials minimum; 100 if compute allows.
3. Metrics per round: RMSE (best-permutation matching for multi-source), posterior entropy,
   log-prob of truth under posterior, expected posterior RMSE. Endpoints and tests:
   exactly as written in the "Pre-registered analysis" section of
   `LOCATION_DEPTH_PATH_A_RUNBOOK.md` (see NEXT ACTIONS #2) — that section, once written,
   supersedes any endpoint statement elsewhere in this document. Report per-trial paired
   difference plots, not just mean traces.
4. UNconstrained contrast arm at reduced size (25 trials, depths {1, 5} only): expectation
   is depth helps under the constraint and not without it. Shrink or cut this before
   touching the constrained arm — the constrained result alone can carry the paper.
5. Include the non-LLM receding-horizon planner from Phase 3.3 as a reference line in the
   location results, framed as the oracle YARDSTICK: the headline statistic is the
   **fraction of the greedy-EIG → oracle gap that StrategyEIG closes** at each depth.
   This turns "a classical planner solves this env" from a weakness into the measurement.
6. **REQUIRED (MPP #3) — myopic scoring at matched compute**: identical d5 rollouts and
   token budget, but strategies scored by first-step-only entropy drop (myopic objective).
   This de-confounds "non-myopic objective" from "more compute": depth uses ~d× the
   tokens of d1, and a reviewer will attribute gains to budget unless this control exists.
   If n-step scoring beats myopic scoring at identical compute, that is the paper's
   cleanest evidence. This is the ONE required ablation — protect it.
7. **Contingency — if StrategyEIG lands below analytic greedy** (LLM execution noise in
   query coordinates can swamp planning gains): switch headline method to the hybrid
   already sketched in the repo (`StrategyEIG+root` direction) — LLM strategy proposes
   the region/plan, analytic EIG scores and executes the query within the constraint.
   Frame as "LLM plans, calibrated model executes". Decide from the first 10 paired
   trials, not at the end.
8. OPTIONAL (appendix, only after MPP complete, in this priority order):
   (a) MPC-style direct action-sequence rollouts at matched budget — defends "why NL
   strategies"; if skipped, state it as a limitation/future work honestly;
   (b) 26B replication of depths {1, 5}, 25 trials — scale contrast;
   (c) strategy evolution on/off.

## Phase 4b — Tractability baseline: brute-force n-step EIG (existing animals data)

IMPORTANT correction: the existing animals depth-1/2/3 results
(`results/cluster-gemma4b-nstep-comparison/`, `plots/results_gemma4_4b_four_line_comparison.png`)
are BRUTE-FORCE n-step EIG tree expansion (BED-LLM style), NOT StrategyEIG. And vanilla
20Q is near-myopic-optimal (greedy question-splitting ≈ optimal), so it cannot demonstrate
a non-myopia benefit. Do not present it as one.

Its correct role: **brute-force tree expansion is the competing way to be non-myopic with
an LLM, and it scales exponentially in depth — that is the tractability argument for NL
strategies.** Tasks:

1. Extract per-depth token/compute cost from the brute-force animals runs (d1/d2/d3) and
   from StrategyEIG location runs (d1–d5). Produce a cost-vs-depth figure: brute force
   exponential, StrategyEIG ~linear. Pair with performance where comparable.
2. If cheap, run brute-force depth-2 EIG on the CONSTRAINED location env (coarse action
   grid) at matched token budget to StrategyEIG d2 — a budget-matched comparison on the
   headline env is stronger than a cross-env one.
3. Paired stats on the existing animals traces (bootstrap CIs) — reported as an LLM-EIG
   vs naive result consistent with BED-LLM, not as a StrategyEIG or non-myopia result.

## Phase 5 (CUT BY DEFAULT — this is a second paper) — Discrete NL env with a verifiable trap

Do NOT start this for the workshop submission. It is the natural follow-up paper (and the
right headline for a future conference version). Recorded here so the idea isn't lost.
Original sketch — a 20Q variant with
prerequisite structure: a finite question pool where some high-EIG questions are unlocked
only by asking low-EIG precursor questions (e.g. category-establishing questions unlock
attribute questions). Because the hypothesis set and question pool are finite, the
greedy-vs-depth-2 gap is EXACTLY computable — a verifiable trap with no oracle needed.
This env has both the myopic trap AND an NL hypothesis space no classical planner covers;
if it lands, it becomes the headline and location finding becomes the controlled study.
Design + exact-gap verification first (no LLM compute); commit to runs only if the
verified gap is large.

## Paper story & positioning (read before Phase 4)

**Recommended framing — question-driven, robust to either Phase 4 outcome**: lead with
"Can LLMs evaluate their own experimental plans?" The ranking-fidelity measurement is the
core contribution (a quotable ρ, interesting whether high or low); the constrained-env
depth results then show where estimation fidelity does and does not translate into
deployment gains. This makes the "main" and "fallback" papers the SAME paper with
different emphasis in the results section — no late-stage fork in the writing.

Central claim (resolves the core tension — envs that admit classical planners don't need
LLMs; envs that need LLMs tend to be myopic-optimal): **a general-purpose NL-strategy
scaffold recovers a substantial fraction of the greedy→oracle planning gap without
task-specific planning code, at tractable (≈linear-in-depth) cost where brute-force
n-step EIG expansion is exponential.**

**Venue targeting**: the NeurIPS 2026 workshop list is not yet announced (proposals under
review as of early July). When it drops, pick 2–3 targets and match the abstract: a
BED/adaptive-experimentation workshop → lead with EIG-estimation rigor and diagnostics;
an LLM-agents/reasoning workshop → lead with the planning scaffold and depth results.
Same results, different first paragraph.

Contributions as a reviewer would list them: (1) NL-strategy-as-policy with evolutionary
refinement and rollout n-step EIG scoring — non-myopic + LLM, vs BED-LLM (myopic + LLM,
arXiv:2508.21184, whose brute-force n-step extension is our tractability baseline) and
DAD/RL-BED (non-myopic, no LLM, fixed parametric spaces, require training); (2) the
ranking-fidelity diagnostic — a direct measurement of whether LLM rollout EIG
estimates rank strategies by realized gain (scope the claim: EIG-estimator-quality
literature exists; the novelty is doing this for NL strategies with LLM simulators);
(3) oracle-gap-closure and cost-vs-depth results on the constrained env, plus the MPC
and matched-compute-myopic ablations isolating what the NL strategy layer and the
non-myopic objective each contribute.

**Also position against informative path planning (IPP)**: the locality-constrained
location task is structurally IPP / adaptive sampling from robotics (GP-based planners,
source seeking). A robotics-adjacent reviewer WILL raise this. The novelty claim must be
the general-purpose LLM/NL-strategy scaffold that transfers across envs without a
task-specific planner — not the environment. Cover IPP in POSITIONING.md and cite 2–3
representative works.

**Required**: read arXiv:2605.26990 ("Constrained BED via Online Planning", May 2026) and
write 3–5 sentences in `results/POSITIONING.md` on overlap and differentiation BEFORE
launching Phase 4. If it already demonstrates the constrained location-finding contrast
with a classical planner, lean the framing harder on (1) and (2) and on 20Q.

Claims discipline: the main comparison is StrategyEIG vs greedy EIG — naive is a floor,
not the story. Never claim "monotonic in depth" unless the paired CIs support it;
"depth ≥ 2 beats depth 1 under constraints" is a sufficient and safer headline.

## Success criteria / deliverables

- `results/ranking_fidelity/REPORT.md` with ρ, SNR, top-1 regret per depth, before/after
  Phase 2.
- Oracle-check figure showing greedy vs planner gap in the constrained env.
- Main figure: paired ΔRMSE (and entropy) vs depth, constrained vs unconstrained, with CIs.
- **Qualitative figure (required, nearly free from logs)**: 2–3 example evolved NL
  strategies shown verbatim next to their query trajectories in the constrained arena —
  ideally one trial where greedy EIG stalls at a local signal bump while StrategyEIG
  routes around it. This is the memorable figure; workshop papers live on these.
- Keep main-text scope compressible to 4–6 pages: one main quantitative figure, one
  diagnostics figure, one cost table, one qualitative figure; everything else to appendix.
- All runs reproducible from committed configs; note model + hardware used for each.
- If the gate fails: the fallback report ("when does lookahead help LLM-driven BED —
  an SNR analysis of MC n-step EIG estimation") using the same artifacts.

## Constraints

- Don't refactor `core/` interfaces; add hooks/flags rather than rewriting.
- Keep all existing tests passing; add tests for the constraint projection and the
  `fixed_common` scoring mode.
- Prefer the small questioner model for iteration; only run the big model for final sweeps.
- Log per-call LLM token counts so compute cost can be reported.
