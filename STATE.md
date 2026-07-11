# STATE — living project state

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug. If this file conflicts with GOAL.md's design
history, this file wins.

## CURRENT STATE (updated 2026-07-11)

Status: **Path E active** — non-myopic LLM experimental design on EXTERNAL interactive
benchmarks (GOAL.md): primary env = Paprika customer-service troubleshooting tasks
(arXiv:2502.17543, released), second env = MediQ (arXiv:2406.00922; AgentClinic is fallback). Method =
BED-LLM-style beliefs + selective 2-step lookahead (tie/gating-triggered). Must beat
naive AND 1-step EIG on paired external-benchmark endpoints. 20Q/Wordle/Mastermind are
harness/unit-test only (greedy near-optimal there); location finding is closed. All
Path A/B/C/D location & 20Q material is banked history below.

Path E Step 0 implementation status (2026-07-11): **Step 0a and Step 0b are passed**.
The canonical real-model Paprika smoke is
`20260711T000402_paprika-step0a-openrouter-26b-nonthinking-v6` at commit `8b84edf`.
Across five official eval tasks and nine realized turns it achieved 8/9 = 88.89%
manually verified answer-set coverage, zero terminal structured failures, zero runtime
failures, and one genuine exact-remedy resolution. It used OpenRouter
`google/gemma-4-26b-a4b-it` without reasoning: 642 requests, 198,897 tokens, no forced
exits, and $0.04004851. Evidence is tracked in
`results/path_e/STEP0A_OPENROUTER_SMOKE.md`. Cumulative OpenRouter development spend is
$0.19704997 of $20. The exact-posterior Mastermind depth-1/depth-2 harness remains green.

The Paprika implementation is pushed. The
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
smoke originally waited on the unavailable cluster and was completed through the
authorized OpenRouter bridge instead.
The naive arm is now truly belief-free while retaining native early stopping and endpoint
metrics. Full two-step execution batches the complete root/branch/follow-up tree rather
than issuing serial calls; its integration test requires four logical batches including
the deployed posterior refresh.
Strict JSON/schema parsing now has two bounded repair attempts by default. Failed items
inside batched likelihood/candidate calls are retried together, and cumulative retry and
terminal-failure counts are emitted as metrics. Tests cover recovery for malformed
single and batched responses.
`scripts/analyze_paprika_smoke.py` implements the real-smoke automated gate: five tasks,
nonempty turns, coverage >= 0.85, zero terminal structured failures, and no runtime-error
signatures. It also reports forced-exit rate and surfaces all queries/replies, while
requiring a separate manual semantic transcript review before Step 0a can pass.

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

**DECISION BOUNDARY ADDENDUM (pre-registered 2026-07-11, BEFORE rescue results are
combined — Hanshal-reviewed):**

Context: canonical Step 1 failed both claims (naive 0.40 > EIG 0.30 > full2 0.20;
full2 vs EIG 0/1/9). Diagnosis: scaffold-generated candidates/hypotheses are
off-target; EIG selection and answer mapping are functioning. Third environment with
the same scaffold<naive inversion. Consequences regardless of rescue outcome:
- Full 2-step is DROPPED from all further Path E runs (failed its gate; $5.16/10 tasks
  is unaffordable at $20 scale). The lookahead claim is closed for this cycle.
- Reminder: all 10-task reads are +/-1 task from flipping; treat as directional.

**If the combined rescue PASSES matched Claim 1** (thinking-generation EIG >= naive
under the frozen rules): continue the plan with 1-step EIG as the method arm —
Step 2 selective lookahead is NOT revived (claim 2 is closed); instead proceed to a
larger Paprika task set to firm up claim 1, then MediQ transfer.

**If the combined rescue FAILS or TIES**: exactly ONE further pre-authorized variant,
then stop-and-discuss regardless of anything else:
- **Naive-primary arbitration**: each round, the thinking-naive policy proposes k=3
  candidate actions (its natural next action plus two alternatives, from one prompt);
  the belief state scores ONLY these by categorical 1-step EIG; select naive's top
  choice unless another proposal beats it by the margin rule (score gap > 1 SE).
  Everything else (mapping, likelihoods, refresh) as in the canonical EIG arm.
  Same 10 canonical tasks, same seed, frozen censoring/tie rules, projected <= $2.
  Rationale (pre-registered): candidates from the 0.40-resolution native policy remove
  the demonstrated candidate-quality bottleneck; beliefs do selection only; regret vs
  naive bounded by the default rule.
- READ: arbitration > naive on the frozen rules -> this becomes the Path E method claim
  ("belief-guided selection over native LLM proposals") and the paper pivots
  accordingly; scale it on more tasks before MediQ.
- Arbitration ties/loses -> STOP. No further variants. Discussion covers the honest
  remaining options (including the cross-environment characterization: scaffolding
  helps only when the hypothesis space exceeds native reasoning capacity — 20Q yes,
  Paprika no, location no).
Budget note: ~$11 remains; reserve >= $6 for whichever endgame is chosen.


**PRE-LAUNCH AMENDMENTS TO STEP 1 (Hanshal-reviewed, 2026-07-11 — apply BEFORE launching
the Step 1 configs):**

A. **Extend the horizon: >= 4 rounds (prefer 5), not 2.** With a 2-turn budget the
   second action has no future, so 2-step lookahead can influence exactly one decision
   per task, prerequisite chains cannot manifest, and most tasks will be censored —
   claim 2 could read null purely from horizon truncation (false negative at the key
   gate). Re-project cost from the micro-pilot (~$0.084/task/round for full2 =>
   ~$8-10 scaffolded at 5 rounds); the full $20 is authorized and a correct Step 1
   read outranks the savings. Update both YAML configs before committing them.

B. **Add a non-thinking naive arm** (same tasks/seed; ~$0.50). The thinking-naive arm is
   the adversarial headline comparator, but if the non-thinking scaffold loses to
   thinking naive alone, "scaffolding fails" is confounded with "thinking wins".
   Claim-1 then has two pre-registered readings: matched (EIG vs naive-nonthinking —
   the clean comparison; this is the gate) and adversarial (EIG vs naive-thinking —
   the headline if it holds; report honestly either way).

C. **Pre-planned rescue variant (decided now, not post-hoc):** if the matched claim-1
   read fails, ONE authorized variant may run before the STOP-and-discuss: enable
   thinking on the scaffold's generation calls only (hypothesis generation + candidate
   generation — a handful of calls per round), keeping likelihood/mapper/judge calls
   non-thinking. Anything beyond that single variant is a stop-and-discuss.

D. Analyzer must be finalized (censoring rule, tie handling in the 6/10 count — ties
   count for neither side, and mostly-tie outcomes are 'insufficient signal' not
   'fail') BEFORE any Step 1 results are viewed. Watch forced-exit rate on the
   thinking-naive arm (8k budget).


Path E reset (2026-07-10): external benchmarks with structural sequential gaps. See
GOAL.md for the six environment requirements (R1-R6) and the full validation chain.

1. **Step 1 gap pilot (active gate):** 10 paired customer-service tasks, five rounds,
   with non-thinking naive, thinking naive, one-step EIG, and full two-step. Belief-
   scaffolded EIG/full2 and matched naive run without reasoning; 8k-thinking naive is
   the adversarial comparator. Use OpenRouter for the whole paired set, seed 1304, and
   the same non-thinking answerer. EIG/full2 share root candidates and prompt cache;
   provider seed 1304 is sent on every API request.
   The one-task/one-round full2 cost micro-pilot completed with 1,235 requests, 419,382
   tokens, $0.08430807, zero terminal failures, and 236.7 seconds at concurrency 24.
   The two premature two-round launches were canceled with no metrics after spending
   $0.02221529 total. The five-round scaffolded run reserves a conservative $12
   projection to cover support/prompt growth; concurrency is 128.
   Matched Claim-1 gate: EIG > naive non-thinking. Adversarial read: EIG vs naive
   thinking, reported but not substituted for the matched gate. A majority-tie result
   is insufficient signal, not failure. If the matched gate truly fails, the single
   pre-planned rescue is thinking only for hypothesis/candidate generation; otherwise
   stop and discuss.
   The first concurrent five-round launch set exposed a spend-ledger interprocess race
   and produced no metrics. All processes were stopped; provider usage was reconciled
   exactly to $0.31118538. Relaunch only after the `flock`/atomic-write stress test and
   focused suite pass.
   The repaired scaffolded run was launched. Serial naive throughput was diagnosed before
   evidence landed; both naive configs now use trial batch size 10 and Paprika batches
   policy generation across public scenarios without exposing private solutions.
   Matched non-thinking naive completed in 155 seconds for $0.00534679. The scaffolded
   relaunch stopped with no metrics after a repaired likelihood response still omitted
   one outcome key; omitted outcomes now receive zero mass (as explicit null already
   did), while all-zero rows remain invalid. Relaunch scaffolded from the parser fix.
   Manual review then rejected the first completed matched-naive artifact: coverage was
   31/37 = 83.78%, and one failed connectivity attempt was falsely resolved. The whole
   companion set was stopped. Final semantics now guarantee an uncertainty outcome,
   treat prospective "I'll try/check" replies as uncertainty, reject failed corrective
   attempts before success judging, recognize embedded "Goal reached", and require
   atomic candidate actions. All arms must rerun from this shared behavior.
   The next matched run reached 39/46 = 84.78% coverage. Audit showed five of seven
   misses had a direct listed outcome, but a mapper `null` did not trigger the bounded
   remap. Explicit non-uncertain replies now get the same non-forcing repair whether the
   first mapper chose uncertainty or returned null. Relaunch all arms from that fix.
   Final matched naive now passes coverage at 35/40 = 87.5%. The first final thinking
   attempt lost its tenth long response to `http.client.IncompleteRead`; bounded
   transport retries now include `HTTPException`/connection errors. Relaunch only the
   thinking arm from that transport fix; scaffolded remains valid and active.
   Audit of the retry revealed mapper/judge calls also inherited questioner thinking.
   Thinking is now isolated to naive policy generation: likelihood, mapping, and success
   evaluation route through the common non-thinking answerer adapter. The scaffolded
   run is unaffected (both adapters are non-thinking); relaunch thinking naive only.
   Both naive controls are now complete and pass coverage: matched 87.5% / 4 resolved;
   policy-only thinking 85.0% / 4 resolved, with one forced exit. Scaffolded EIG later
   hit one empty response after exhausting two repairs at request 2,602; Step 1 configs
   now use five bounded structured repairs. The repair-five EIG arm completed with
   42/45 = 93.3% answer coverage and 2/10 resolutions, but Full2 later exhausted all
   five repairs on a likelihood response whose listed outcomes had zero total mass.
   The failed invocation spent $1.59969375 over 22,333 requests; no paired gate result
   is used. Because every Paprika candidate now has a guaranteed uncertainty outcome,
   an all-zero row deterministically assigns its residual mass to that outcome (rows
   without a guaranteed uncertainty outcome remain invalid). The canonical rerun is
   isolated into ten one-task paired EIG+Full2 shards so each shard preserves exact
   shared root candidates and any future terminal failure loses at most one task. Run
   at most five shards concurrently with per-shard OpenRouter concurrency 24, then
   combine only ten completed canonical task IDs with
   `scripts/combine_paprika_step1_splits.py` before the frozen analyzer.
   The first offsets-0-4 launch wave was canceled after 42 combined requests /
   $0.00380851 because second-resolution run IDs collided in the spend ledger. No
   result is used. Concurrent shard launches must now be staggered by at least two
   seconds so every run retains independently auditable usage.
   Split wave A completed for offsets 0-4 with distinct run IDs `20260711T033611`,
   `20260711T033613`, `20260711T033615`, `20260711T033617`, and
   `20260711T033619`. All five contain paired EIG+Full2 artifacts for canonical task
   IDs 0000-0004 with zero terminal/parse failures. EIG coverage is 20/23 and Full2
   coverage is 21/23; wave cost $2.49816920 over 33,641 requests. Launch offsets 5-9
   under the same staggered protocol, then combine all ten completed shards.
   Split wave B completed for offsets 5-9 with run IDs `20260711T042601`,
   `20260711T042603`, `20260711T042605`, `20260711T042607`, and
   `20260711T042609`. All five paired artifacts cover canonical tasks 0005-0009;
   EIG coverage is 18/22 and Full2 coverage 23/25. There were zero terminal/parse
   failures; two offset-6 non-thinking responses reached the output-length limit but
   repaired successfully. Wave cost $2.91180721 over 39,728 requests.
   The frozen analyzer has now run once on the deterministic ten-shard combination.
   Matched Claim 1 FAILS: EIG resolved 3/10 versus non-thinking naive 4/10,
   wins/losses/ties 3/3/4, mean censored-turn delta +0.6. Adversarial EIG versus
   thinking naive is 1/4/5, also +0.6. Claim 2 also fails: Full2 resolved 2/10 versus
   EIG 3/10, 0/1/9, delta +0.4. EIG coverage is 38/45 = 84.4% (just below the pilot
   threshold); Full2 is 44/48 = 91.7%. Per the predeclared playbook, launch exactly
   ONE rescue: EIG with thinking on hypothesis/refinement/candidate generation only;
   likelihood/filter/mapper/judge/customer remain non-thinking. If that matched read
   does not pass, STOP and discuss with Hanshal; do not implement Step 2 or pivot.
   The first sole-rescue invocation `20260711T052319` failed without metrics when an
   OpenRouter HTTP body was truncated inside JSON; it spent $0.16261849 over 1,298
   requests, with 235,506 reasoning tokens and 6/41 generation forced exits. This is
   an operational failure, not a second scientific read. JSON decode failures are now
   included in bounded transport retries. Relaunch the exact same rescue policy as ten
   isolated one-task shards (five at a time, concurrency 24 each, staggered run IDs)
   using `config_paprika_step1_eig_generation_thinking_rescue_split_openrouter.yaml`.
   Combine only ten completed EIG artifacts, then compare once against the canonical
   matched naive. If that read does not pass, STOP and discuss with Hanshal.
   Rescue split wave A completed for offsets 0-4 as run IDs `20260711T071533`,
   `20260711T071535`, `20260711T071537`, `20260711T071539`, and
   `20260711T071541`. All five canonical tasks completed with zero parse/terminal
   failures, 19/20 clean turns, 2 resolutions, and 2/55 generation forced exits.
   Wave cost $0.22756252 over 2,031 requests and 260,153 reasoning tokens. Launch
   rescue offsets 5-9 under the identical protocol, then combine and analyze once.
   Rescue split wave B completed as run IDs `20260711T075742`, `20260711T075744`,
   `20260711T075746`, `20260711T075748`, and `20260711T075750`: zero terminal/parse
   failures, 18/21 clean turns, 1 resolution, 11/67 generation forced exits,
   2,190 requests, 367,005 reasoning tokens, and $0.27236181.
   The frozen rescue analyzer has run once. Result: **INSUFFICIENT SIGNAL; STOP AND
   DISCUSS**. Rescue EIG resolved 3/10 versus matched naive 4/10, but paired outcomes
   are 1 win / 1 loss / 8 ties, so the majority-tie rule prevents calling this a
   directional failure. Mean censored-turn delta is +0.2; coverage improved to 37/41
   = 90.2%. Against thinking naive it is 2/2/6, also +0.2. Total rescue usage is
   4,221 requests, 627,158 reasoning tokens, 13/122 generation forced exits (10.7%),
   and $0.49992433. Authoritative artifacts are
   `results/path_e/step1/PAPRIKA_STEP1_RESCUE.{json,md}`.
   **NEXT ACTION REQUIRES HANSHAL:** choose whether to (a) stop Path E and write the
   honest negative/insufficient external-benchmark result, or (b) explicitly authorize
   a new pre-registered path. Do not implement selective Step 2, scale Paprika, add
   another rescue, or pivot to MediQ autonomously.
   Claim-2 check: 2-step > 1-step (>=6/10 or clear edge). Claim-2 fail -> descope to the
   claim-1 transfer study and continue. Claim-1 fail -> STOP and discuss with Hanshal.
2. **Step 2:** implement selective lookahead (tie test epsilon = scoring-noise SE +
   availability-gating trigger; CRN across tied set; depth cap 2; trigger/token
   logging); 10-task four-arm pilot; calibrate then FREEZE epsilon; check selective
   lookahead tokens <= ~40% of full 2-step.
3. **Step 3:** pre-register endpoints/analysis/epsilon/canonical-run rule in the runbook,
   then 50-100 paired customer-service tasks (powered from Step 2), four arms; then
   AgentClinic 50+ cases if on schedule. `--partition=msc,llm --exclude=oat12`.
4. **Step 4:** paper per GOAL.md playbook; location-finding saga = one honest paragraph.

## OPERATIONAL KNOWLEDGE

- **OpenRouter (while cluster is down)**: base_url https://openrouter.ai/api/v1,
  key in `OPENROUTER_API_KEY` (never commit). $20 total budget, FULLY authorized;
  refuse runs projected past remaining budget; flag Hanshal at ~$18 cumulative; track
  spend per run in `EXPERIMENTS.md`. Cost-project every run from
  smoke tokens first (Path A reference: a 30-trial 3-arm location sweep used ~3.9M
  tokens; Paprika turns are longer — measure, don't assume). Cluster ops notes below
  still apply once `ssh oat0` recovers.

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
