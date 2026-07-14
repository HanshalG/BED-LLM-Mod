# STATE — living project state

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug. If this file conflicts with GOAL.md's design
history, this file wins.

## CURRENT STATE (updated 2026-07-13)

Status: **Track 1 is complete and collapses as a credible non-myopic paper spine;
Track 2 (MediQ) is now the main experimental line and Track 3 (paper) proceeds in
parallel.** The free banked animals audit made no model calls and is tracked in
`results/path_e/ANIMALS_REANALYSIS.md`, with reproducible trial-level data in
`results/path_e/animals_reanalysis/ANIMALS_REANALYSIS.json`. One-step EIG remains a
strong paired result against both naive baselines, but the old apparent two-step gain
does not survive the comparability audit: the full depth-2 versus depth-1 AUC delta is
+0.027 with an unpaired 95% bootstrap CI [-0.076, 0.132], while the only exactly paired
seed block reverses to -0.040 [-0.105, 0.025] (3/1/6 wins/ties/losses). Depth 3 is
materially worse than depth 2 on 40 paired trials, -0.126 [-0.224, -0.035]. Q@80 is 9,
8, and 12 for depths 1, 2, and 3 respectively. The EIG configurations are scientifically
matched, but depth 1 shares only ten exact target/prior conditions with depths 2/3;
the logs also lack Git-commit provenance and the depth-3 recursive implementation
postdates the depth-1/2 runs. Animals can support one-step BED transfer, not the
non-myopic claim. MediQ must carry that claim.

MediQ Step 0 environment status (2026-07-14): **PASS, automated and manual. A newly
registered likelihood-calibration gate must pass before Claim 1 is preregistered.**
The adapter uses the exact released multiple-choice labels as the
finite BED target, a temperature-zero judged initial distribution, option-conditioned
categorical response likelihoods, recursive one-likelihood-at-a-time Bayes updates,
grounded Fact-Select patient replies assembled verbatim from released atomic facts,
and exact categorical depth-two branch expansion. The naive arm is belief-free and
directly decodes from the complete observed conversation. Per-query artifacts now log
all root likelihood tables plus predicted EIG, realized entropy drop, realized truth
log-probability gain, predictive outcome probability, and true-label likelihood. The
frozen analyzer independently reconstructs every EIG table and verifies data provenance,
verbatim grounding, relevance, mapping coverage, parse failures, and runtime failures;
manual semantic review remains mandatory.

The hash-pinned official iMEDQA file has 1,272 rows. Three source rows (224, 298, 779)
have both empty context and empty atomic facts and are static knowledge questions, so
they violate MediQ's interactive patient-task definition. They are explicitly excluded,
leaving 1,269 usable rows; every run writes a manifest with the raw hash, commit,
excluded IDs, and selected source IDs. A zero-cost routing-model dry run over the exact
five-case/two-round smoke shape initially passed every automated analyzer gate at 235
logical requests. Cost projection is in
`results/path_e/mediq_step0/COST_PROJECTION.md`: expected $0.02-$0.05 after repair,
conservatively reserved at $0.12. The initial paid run
`20260714T033819_mediq-step0-eig-nonthinking-26b-seed1304` completed at commit
`7825898` in 208.93 seconds: 235 requests, 70,115 prompt tokens, 18,128 completion
tokens, no reasoning or forced exits, and $0.01416921. It passed every frozen automated
check (90% mapping coverage, 100% verbatim grounding and automatic relevance, zero
terminal/runtime failures, all 50 EIG tables valid), but the mandatory manual review
failed. Case 0 inferred new sexual partners from only "sexually active" after a compound
query; case 2 inferred an unsupported symptom frequency; case 4 exposed a numeric
outcome gap at glucose 450 mg/dL; and some candidates duplicated unavailable categories.
This localizes the remaining validity issue to atomic query/outcome partition generation
and explicit-entailment relevance mapping, not the finite-target EIG arithmetic. The run
is diagnostic only; see `results/path_e/mediq_step0/INITIAL_SMOKE_MANUAL_REVIEW.md`.
The repair now uses canonical atomic yes/no predicates with exactly one unavailable
outcome, rejects compound/open-ended queries, retains validated candidates while
regenerating only the deficit through a temperature-zero structural critic, checks
selected facts for explicit entailment without exposing response categories, and only
then maps relevant facts. It also forbids a relevant fact from being mapped to the
unavailable bucket. The analyzer requires logged successful candidate validation and
independently requires the canonical yes/no/unavailable support. The exact
official-data zero-cost dry run passes at 305 requests (50 individual candidate audits,
10 set-level dedup audits, and 10 separate relevance audits); 21 focused tests pass and
the full suite is 595 passed, 1 skipped,
with only the same unrelated stale Path A wording assertion failing.
The first paid repeat attempt `20260714T035736` failed closed before likelihood or
patient calls because only 3/5 compound-filtered candidates survived two opaque count-only
repairs. It produced no result and cost $0.00117655 over 13 requests. Rejection feedback
now names every failed query and reason, with a regression test covering the repair.
The second attempt `20260714T040005` reached likelihood scoring but failed closed when
the critic rejected a non-exhaustive synovial-fluid category set and a hallucinated
`serum protein A` variable after bounded whole-set regeneration. It produced no complete
item and cost $0.01065559 over 192 requests. This directly motivated the canonical
binary observation support and deficit-only replenishment. The resulting complete run
`20260714T041055` passed its initial automated gate at 10/10 clean mappings, 10/10
grounding/relevance, and 50/50 finite-target EIG tables, with 288 requests, zero
reasoning, and $0.01408867. Its mandatory manual audit still failed: case 0 used
drug-class queries as disguised decodes of the medication target; case 2 repeated
excessive worry under a paraphrase after unavailable; and unselected candidates included
derived stability and management-status predicates. The hardened analyzer now catches
all of these retrospectively. Generation and parsing now require pre-decision patient
evidence, reject diagnosis/management/test-status queries and derived clinical summaries,
and use content-token semantic deduplication against history and the accepted pool. The
next complete run `20260714T042154` passed the hardened analyzer and every realized turn
passed manual review, but its case-3 round-1 candidate pool contained both `renal calculi`
and `nephrolithiasis`. This is a narrow proposal-diversity failure: both actions are valid
alone, but per-candidate checks cannot see medical synonyms across a set. A new
temperature-zero set-level auditor now retains one representative per duplicate group,
replenishes only the deficit, logs its decision, and is required by the analyzer. The
305-request exact-shape dry run passes this final set contract.
The first paid set-dedup attempt `20260714T043240` failed closed during round-2
replacement after correctly rejecting `feeling excessive worry` as a paraphrase of the
deployed `feelings of excessive worry`; deterministic token matching missed the
singular/plural form and one-at-a-time replacement then exhausted bounded retries. It
produced no complete item and cost $0.01076385 over 212 requests. Content tokens now use
light singular normalization, and deficit-one replenishment requests two alternatives
so a repeated concept need not consume the only slot. Focused tests cover both paths.
The exact replay `20260714T043855` then passed: 10/10 clean mappings, grounding, and
relevance; 50/50 individually valid candidates; 10/10 semantically distinct candidate
sets; zero terminal/runtime failures; and every selected interaction and final candidate
passed manual review. It used 300 requests, 103,785 tokens, no reasoning, and $0.01402286.
Its 3/5 endpoint is ignored as smoke-only. Canonical evidence is in
`results/path_e/mediq_step0/FINAL_REPORT.json` and `FINAL_MANUAL_REVIEW.md`.
Post-gate analysis found a separate model-specification failure in the legacy
`joint_option` scorer. Across the ten selected turns, predicted EIG averaged 0.140 nats
but realized true-label log-probability gain averaged -0.290 nats; the realized outcome
was less likely under the true label than under the prior mixture on 6/10 turns. The
failure is structural: an MCQ answer such as “treat hypoperfusion first” is a decision,
not a mutually exclusive patient state, yet the scorer treated high glucose and acidosis
as evidence against that true answer. It also let record unavailability vary by label,
so four missing-record replies spuriously changed diagnosis beliefs. The registered
repair, `factored_record`, predicts record answerability once without a label and then
predicts Yes/No conditional on each option, explicitly allowing coexisting findings and
priority decisions. Missingness is therefore exactly posterior-neutral. The frozen
ten-turn replay and no-tuning decision rule are registered in
`results/path_e/mediq_likelihood_calibration/PREREGISTRATION.md`; code/tests are complete,
and the paid replay is the next action. OpenRouter now also receives `mediq_seed`, fixing
an API reproducibility omission.
OpenRouter ledger: $16.78909024 spent of the user-authorized $40 cap, leaving
$23.21090976.

Path E remains stopped at the Paprika invalid-endpoint/method-claim gate. Its research
target was non-myopic LLM experimental design on external interactive benchmarks
(GOAL.md): Paprika customer-service troubleshooting exposed the boundary failure;
MediQ (arXiv:2406.00922) is the aligned primary environment now. The attempted Paprika
method was BED-LLM-style beliefs plus lookahead/arbitration, and it failed to beat naive
and one-step EIG on paired external-benchmark endpoints. Wordle/Mastermind are
harness/unit-test only; location finding is closed. The animals/20Q result is banked
corroboration under the target-decode contract, not the non-myopic headline. All Path
A/B/C/D location and 20Q material is banked history below.

The explicitly authorized post-stop method investigation is complete. The tracked
report is
`results/path_e/arbitration_headline/NON_MYOPIC_BED_FAILURE_ANALYSIS.md`, with
reproducible descriptive output in `METHOD_FAILURE_DIAGNOSTICS.json` generated by
`scripts/analyze_paprika_method_failure.py`. It used accepted artifacts only, made no
model calls, spent no API budget, and did not rerun the frozen headline analyzer. The
diagnosis is that the current method is non-myopic information acquisition over an
unstable free-text support, not non-myopic task planning: its EIG target omits terminal
resolution/turn cost, its simulated branches do not match deployed belief refresh or
terminal behavior, its JSON likelihoods and one-SE rule are uncalibrated, and Paprika
does not enforce the assumed prerequisite/gating gap. The recommended new method is a
goal-oriented Bayes-adaptive planner over typed diagnostic and remedy actions, using
task return, a calibrated deployment-matched world model, and an oracle-verified
planning gap. This is analysis, not authorization to implement or launch that new path;
the stop-and-discuss boundary below remains active.

Path E Step 0 implementation status (2026-07-11): **terminal-repaired Step 0a and Step
0b are passed**. The canonical real-model Paprika smoke is
`20260711T130833_paprika-step0a-terminal-faithfulness-repaired-tasks0-4-seed1304` from
implementation commit `a5da29a`. Across five official eval tasks and nine realized
turns it achieved 9/9 = 100% answer-set coverage, zero terminal structured failures,
zero runtime failures, zero raw/final simulator inconsistencies, and one exact-remedy
resolution. The specialized terminal gate logged one claim, one check, and zero
rejections. Manual review passed all five transcripts, including dishwasher alternatives
that correctly remained non-terminal. It used OpenRouter
`google/gemma-4-26b-a4b-it` without reasoning: 669 requests, 210,536 tokens, no forced
exits, and $0.04443824. Evidence is tracked in
`results/path_e/step0a_terminal_repaired/`. All earlier endpoint results remain invalid. The
exact-posterior Mastermind depth-1/depth-2 harness remains green.

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

**EXECUTION PLAN (2026-07-12, per Hanshal — three tracks, this order):**

Track 1 (COMPLETE, FREE): the banked animals depth re-analysis. Result: **COLLAPSES**.
The analyzer reconstructed all 200 trial-level 20-round traces, verified their aggregate
curves exactly against the banked metrics, audited scientific configs and target/prior
pairing, and computed deterministic 20,000-replicate bootstrap intervals. The old
depth-2 advantage was an unpaired point estimate driven by easier restarted seed blocks;
the only paired block reverses, and paired depth 3 is worse than depth 2. Deliverables:
`scripts/analyze_animals_depth_reanalysis.py`,
`results/path_e/ANIMALS_REANALYSIS.md`, and
`results/path_e/animals_reanalysis/ANIMALS_REANALYSIS.json`. Consequence: retain
animals for the one-step BED-transfer claim; the non-myopic claim rides entirely on
the MediQ claim-2 pilot.

Track 2 (main line): MediQ per the registered design requirements. Step 0 now passes its
final environment gate; all earlier smokes remain diagnostic-only. Commit and push the
registered `factored_record` likelihood replay, then run it once on the frozen ten turns.
If it passes, run one held-out calibration smoke before freezing Claim 1; if it fails,
stop the efficacy path and implement only the preregistered richer-world fallback. Note the endpoint is
exact-match on the MC label — no success judge, no remedy adjudication; the remaining
validity gate is likelihood calibration. Calibration -> pre-register ->
claim-1 study (naive asking, native Expert baseline(s), 1-step EIG; ~$2-4) ->
claim-2 pilot (2-step vs 1-step, 10 cases, gated) only if claim 1 holds.

Track 3 (parallel, starts NOW): the paper. Stable arc independent of pending results:
(i) the target-decode contract as problem setting; (ii) in-contract evidence (banked
20Q + MediQ claim 1); (iii) the non-myopic question (Track 1 + MediQ claim-2 pilot);
(iv) boundary of applicability (the Paprika autopsy, quantified mechanisms). Salvage
motivation material from the Path B draft. Target: full draft minus MediQ numbers
before the MediQ headline runs.

NON-MOVES (fences): no Paprika reruns (guess-rule variant = future work), no
Bayes-adaptive build, no new environments beyond MediQ, no further endpoint-repair
campaigns on Paprika. Timeline anchor: NeurIPS 2026 workshop author notifications are
mandated by Sept 29; expect paper deadlines late Aug - early Sep; watch for the
accepted-workshop list and pick 2-3 targets when it drops.


**DIRECTION (2026-07-12, per Hanshal — supersedes the task-value probe and the
Bayes-adaptive proposal): the project's identity is non-myopic sequential BED with
LLMs; EIG is the acquisition objective.** Refined contract (per Hanshal): EIG need not
be the literal benchmark reward — the task must expose a latent target with ground
truth, and success must be a monotone readout of posterior quality via an explicit
DECODE rule (BED-LLM's guess protocol: acquisition maximizes EIG about the target; a
separate belief->action rule — argmax guess at confidence threshold or at budget —
produces the scored output; winning stats proxy information gain).

**MediQ is the primary environment** (in-contract by construction): target = the
answer variable, decode = argmax option at budget (or MediQ's native answer/abstain
decision), acquisition = EIG over patient questions.

**Paprika architectural note (from the autopsy, via the refined contract):** the
deployed agent conflated experiments and decodes — remedies competed inside EIG
scoring, where a cure is correctly a poor experiment. In-contract design: remedies are
DECODES (execute argmax remedy when top-hypothesis mass > tau, tau pre-registered),
never EIG candidates; acquisition ranks diagnostic questions only. This is recorded as
an optional boundary experiment ONLY IF the (third) endpoint invalidity is resolved
and time permits after MediQ — it is not the primary path. The autopsy remains the
paper's boundary-of-applicability evidence either way.

MediQ design requirements (the autopsy's fixes, applied as BED-LLM prescribes):
1. EIG TARGETS THE FINITE ANSWER VARIABLE (the MedQA option set), never free-form
   hypothesis prose. Intermediate finding-hypotheses, if used, support likelihoods only.
2. Calibrated likelihoods over the small categorical target: temperature-0 judged
   distributions (or logprobs if available); no single-shot JSON probability guesses
   over open text. Log a calibration check (reliability of P(answer|findings) against
   outcomes) as a pre-registered diagnostic.
3. No history double-use in updates; semantic dedup of any generated finding
   hypotheses; acquisition and update must use the same model of the answer.
4. Patient simulator faithfulness gate + answer-mapping coverage gate as on Paprika;
   manual transcript review before any claims run (the endpoint discipline carries
   over unchanged).
5. Arms and ladder (pre-register before launch): naive asking, MediQ native Expert
   baseline(s), 1-step EIG (claim 1: EIG > naive AND >= native baselines — MediQ's own
   naive-asking-hurts result implies headroom), then 2-step vs 1-step pilot (claim 2 —
   the non-myopic bet, gated: 10-case pilot, >=6/10 or clear accuracy@budget edge
   before scaling; full 2-step cost cap learned from Paprika applies).
6. Endpoint: accuracy @ question budget, paired per case, frozen censoring rules;
   analyzer finalized before results are viewed; outcome blindness until complete.
7. Budget: $23.21090976 remains of $40. Integration smoke complete, claim-1 study ~$2-4, claim-2
   pilot ~$1-2. Project before each launch as usual.
8. Paper identity: "Non-myopic sequential BED with LLMs: where EIG works, and where it
   cannot" — MediQ as the aligned demonstration (claims 1, and 2 if it holds), Paprika
   autopsy as the misalignment boundary, 20Q/animals banked data as corroboration of
   the aligned regime. The Bayes-adaptive goal-oriented planner is parked as the
   documented conference follow-up for goal-directed tasks.


**STOP-AND-DISCUSS BOUNDARY REACHED (2026-07-12): do not execute the historical
mid-headline reminders below.** The frozen 50-task analyzer ran exactly once and did
not confirm Claim B. Arbitration resolved 16/50 (32%) with mean censored turns 5.12,
versus thinking naive 20/50 (40%) and 4.68 turns, and candidate 0 20/50 (40%) and 4.96
turns. Arbitration-vs-thinking-naive was 6/16/28 wins/losses/ties with mean paired
delta +0.44 turns and 95% bootstrap CI [0.0, 0.9]; arbitration-vs-candidate0 was
6/12/32 with delta +0.16 and CI [-0.2405, 0.56]. Positive deltas mean arbitration was
slower, so neither preregistered comparison supports Claim B. Best-N EIG also resolved
16/50 (32%), with 4.96 mean censored turns; versus thinking naive it was 11/12/27,
delta +0.28, CI [-0.14, 0.74].

The mandatory manual audit then found a fatal endpoint contradiction on task 13:
the private remedy says the receipt printer is out of paper and replacing/refilling the
roll restores printing, but best-N instructed replacement of the current paper roll and
the simulator replied that changing it did not help. This is exactly the frozen
"correct performed remedy claimed to fail" invalidation case. Tasks 10 and 12 passed;
review stopped at the first fatal contradiction as required. The complete five-arm
headline is endpoint-invalid and cannot be used as policy evidence. Do not drop the
task/arm, rerun the analyzer, launch MediQ, or launch another rescue autonomously.
Discuss with Hanshal whether to end Path E as an endpoint-validity/negative diagnostic
paper, redesign the simulator/evaluation under a new preregistration, or stop the paper.
The current `paper/` draft predates this read and is not submission-ready.
A cost/scope decision memo is tracked at
`results/path_e/arbitration_headline/STOP_DECISION_MEMO.md`. It recommends closing Path
E as a method-claims project because the frozen effect points against the method even
before endpoint invalidation. The two alternatives requiring explicit authorization
are a no-new-LLM endpoint-validity paper (complete the remaining manual audit) or a
fresh preregistered simulator/evaluation study. No option has been selected yet.

Post-stop diagnostic work requested by Hanshal is complete. Before any implementation
or launch, review
`results/path_e/arbitration_headline/NON_MYOPIC_BED_FAILURE_ANALYSIS.md` and choose
whether to authorize its proposed fresh preregistered path. The minimum sequence for
that path is endpoint/action-credit repair, typed task-aligned utility, calibrated
world-model and belief-update gates, one-step task-value ranking fidelity, and an
oracle-verified depth-two gap. Increasing current EIG depth/rollouts or tuning the
one-SE margin is explicitly not a next action.

**MID-HEADLINE REMINDERS (2026-07-11, per Hanshal):**
1. The pre-registered BEST-N ELICITATION PROBE (10 canonical tasks 0-9, disjoint from
   the held-out 10-59, ~$0.6) has not run — launch it IN PARALLEL with the remaining
   waves. It settles the paper's method framing (generate-and-select vs calibrated
   arbitration) and whether a design-(ii) arm joins the held-out comparison later.
   The hypothesis-elicitation micro-probe stays queued strictly after it.
2. OUTCOME BLINDNESS: no headline resolution/turn metrics are viewed until all 50 tasks
   and all arms are complete; analyzer runs ONCE. Operational monitoring (coverage,
   gate failures, pairing, cost) remains allowed and required.
3. PAPER REWRITE STARTS NOW (parallel to waves): motivation (faithful transfer fails),
   method, environment + endpoint-validation section (the audit chain is methods
   content), related work, limitations — everything except results. The Path B draft's
   salvageable parts fold into motivation.


**PRE-REGISTERED PROBE (2026-07-11, per Hanshal, run BEFORE or alongside the scale-up):
best-n elicitation for plain EIG.** Hanshal identified an uncontrolled variable: the
plain-EIG candidate prompt ("Propose concise customer-service diagnostic questions or
corrective solution attempts...") never asks for GOOD actions — it elicits
schema-shaped candidates with no goal anchoring, while the arbitration prompt is
goal-anchored. The generation-thinking rescue changed reasoning effort but never the
instruction content, so this is untested. Probe: ONE run, 10 canonical tasks, frozen
design/endpoint, plain argmax 1-step EIG with the candidate prompt changed to
best-n elicitation (e.g. "Propose your N best next actions to resolve this customer's
issue as quickly as possible", N matching the current candidate count; discrete answer
spaces unchanged; no natural-action anchor — plain EIG has no default slot). Thinking
per the rescue config. Projected <= ~$0.6. Pre-registered reads:
- Best-n EIG still loses to thinking naive -> Claim A is robust to elicitation; the
  arbitration structure (default + margin) is demonstrated load-bearing; say so in the
  paper.
- Best-n EIG matches/beats arbitration -> the honest story shifts to "goal-anchored
  elicitation + EIG selection"; arbitration's default/margin is reported as a
  robustness variant; scale-up arms are reconsidered WITH Hanshal before launch.
The ARBITRATION prompt is NOT changed (its natural-action anchor is load-bearing for
the margin rule and the candidate-0 control). Any elicitation change to arbitration is
a post-scale-up ablation only.

**FRAMING NOTE (per Hanshal, 2026-07-11):** the preferred method identity is "generate
n good action candidates, use EIG to select among them" — not "naive with a fallback
override". The best-n probe above IS that method (design (ii) in the ladder: (i) bland
elicitation + argmax = failed BED-LLM transfer; (ii) best-n elicitation + argmax = the
probe; (iii) anchored elicitation + default/margin = pilot-validated arbitration).
Resolution rule: if (ii) ~ (iii) on the probe read, the paper adopts the clean
generate-and-select framing with the margin rule reported as an optional safety knob;
if (ii) < (iii), the margin rule is load-bearing and is framed as CALIBRATED selection
(act only on score differences exceeding scoring noise — a statistical decision rule,
not a hedge). If (ii) is competitive, the scale-up carries BOTH (ii) and (iii) as arms
(same candidate costs; the selection-rule contrast is the ablation reviewers will ask
for anyway).

**SECOND MICRO-PROBE (queued AFTER the candidate probe, never simultaneously — one
change at a time for attribution):** goal-anchored hypothesis elicitation. Current
hypothesis prompts ask for plausible issues; probe variant asks for "the n most likely
root causes given this conversation so far, ranked". 10 canonical tasks, one arm,
<= ~$0.6, pre-registered read before launch. Prompt sensitivity is acknowledged as a
finding-in-itself: log every prompt variant in the ledger and report the elicitation
sensitivity honestly in the paper.


**AUTHORIZATION AT THE STOP-AND-DISCUSS BOUNDARY (2026-07-11, per Hanshal): the
arbitration scale-up is approved.** The revised Path E claim structure supersedes the
original GOAL.md claims (STATE precedence):

- Claim A (honest negative, kept): faithful 1-step BED/EIG scaffolding does not beat
  naive on Paprika; full 2-step failed and is closed.
- Claim B (the method claim): belief-guided EIG arbitration over native thinking-LLM
  proposals improves interactive troubleshooting (pilot: 6/0/4 vs thinking naive,
  +1.2 censored turns, CI [0.4, 2.2], resolution 40%->60%).
- Claim C (generality, pending): the same arbitration transfers to MediQ.

**Scale-up design (pre-register in the runbook BEFORE launch, then do not deviate):**

1. Tasks: the next N unseen Paprika customer-service eval tasks in released order (no
   selection). N from a power calc on the pilot effect (assume the win margin shrinks;
   target ~80% power for a halved effect) subject to the eval pool size and budget —
   expect N in the 30-50 range. Seed 1304, 5 rounds, frozen censoring/tie rules.
2. Arms (paired per task): (i) thinking naive; (ii) **candidate-0 control — MANDATORY:
   identical 3-proposal generation, always execute candidate 0, no EIG scoring** (this
   isolates the override's causal effect from the effect of eliciting 3 proposals);
   (iii) arbitration with the FROZEN 1-SE margin rule (no threshold retuning);
   (iv) naive non-thinking (cheap context arm). Generation-thinking EIG is NOT rerun at
   scale (its pilot read stands as Claim A evidence).
3. Primary endpoint: paired censored turns-to-resolution, arbitration vs thinking
   naive; co-primary: arbitration vs candidate-0 control (the causal read).
   Secondary: resolution@5, win/tie/loss, cost per resolution. Bootstrap CIs; Wilcoxon
   supporting.
4. Pre-registered mechanism analyses (from logs, no extra spend): override rate,
   per-override outcome, score-margin distribution for good vs bad overrides
   (calibration of the 1-SE rule — descriptive only, no post-hoc threshold change).
5. Manual endpoint review: ALL transcripts of any task where arms disagree on success;
   spot-check 10 random others. Same INVALID-ENDPOINT discipline if anything surfaces.
6. Budget check before launch (~$10 remains; projected all-arms cost at N=40 is ~$3-4,
   verify from pilot per-task costs). MediQ port (arbitration arm, native baselines) is
   authorized AFTER the scale-up read, conditional on Claim B holding: if it holds,
   MediQ is the generality experiment; if it collapses at scale, stop-and-discuss.
7. Paper reframe per playbook: the paper is now Claims A+B(+C), i.e. "beliefs select,
   they don't generate: EIG arbitration over native LLM proposals" with the transfer
   negative honestly reported as motivation. Lookahead remains closed this cycle.

Scale-up pre-launch status: **READY AND FROZEN, NOT YET LAUNCHED.** The prompt-matched
belief-free `NaivePrimaryCandidate0` control is implemented and registered only for
Paprika. Tests prove it always executes candidate 0, writes full selection artifacts,
and shares the exact ordered proposal set with arbitration whenever public histories
match. `PATH_E_ARBITRATION_RUNBOOK.md` freezes N=50, eval offsets 10--59, method order,
pairing checks, endpoints, claim reads, mechanism analysis, manual audit, canonical
recovery, budget, and concurrency. The pilot power calculation is tracked at
`results/path_e/arbitration_headline/POWER.json`; N=50 gives approximately 78.2% normal
power at half the pilot effect (53 would give 80%). The generic combiner now supports
both 50 one-task thinking shards and five 10-task non-thinking blocks. The frozen
headline analyzer and configs are implemented. Pre-launch verification: 143 focused
tests pass; ledger and existing paper validators pass. Commit and push this complete
design before launching, then record every launch in `EXPERIMENTS.md`.
Headline launch status: wave 1 is running from frozen commit `7bc2263`, timestamp
`20260711T163517`, offsets 10--19, as ten isolated thinking-triplet shards at aggregate
configured concurrency 250. Do not overlap the non-thinking wave. Monitor only health,
spend, and completion; apply the frozen whole-triplet recovery rule on failures.
Hanshal added $10 during wave 1, so the authorized OpenRouter total is now $30 and the
remaining authorization from the pre-wave spend is $19.61803. Wave 1 retains its
already-instantiated $20 tracker cap; future frozen configs change only the operational
budget cap to $30. No scientific or concurrency parameter changes.
Wave-1 original offset 12 failed after its arbitration item because the candidate-0
customer simulator contradicted the private solution after bounded repairs. The whole
invocation is noncanonical; do not reuse its arbitration artifact. Exact full-triplet
recovery 1 launched at `20260711T164811`, using the unchanged scientific design and the
documented $30 administrative cap. With nine original shards still active, aggregate
triplet concurrency remains 250.
Offset-12 recovery 1 failed closed with the same bounded simulator-faithfulness error
during arbitration and banked no item. Exact full-triplet recovery 2 launched at
`20260711T170138`; no task artifact from either failed attempt is canonical.
Wave 1 is complete and canonical for offsets 10--19. Recovery 2 is the accepted offset
12 artifact. The combined health validation found exact task coverage, three complete
methods per task, and 21/21 matching proposal sets at every identical-history
arbitration/candidate-0 turn. Accepted cost was $0.578927 over 3,576 requests;
operational spend including failed attempts was $0.619374. Cumulative spend is
$11.001346/$30. Canonical provenance is tracked in
`results/path_e/arbitration_headline/CANONICAL_RUNS.json`. Wave 2 (offsets 20--29) is
next under the same frozen scientific design and aggregate concurrency 250.
Wave 2 launched at `20260711T180556`, offsets 20--29, as ten isolated full-triplet
shards at aggregate configured concurrency 250. Same canonical and health-only
monitoring rules apply.
Wave-2 original offset 25 failed after arbitration because candidate 0 hit the strict
bounded simulator-faithfulness failure. The complete invocation is noncanonical. Exact
full-triplet recovery 1 launched at `20260711T185350`; no artifact from the failed
attempt may enter the headline.
Wave 2 is complete and canonical for offsets 20--29, with recovery 1 as the accepted
offset-25 source. Proposal pairing passed 15/15 eligible identical-history turns.
Accepted cost was $0.577599 over 3,475 requests; operational spend including the failed
attempt was $0.630444. Cumulative spend is $11.631791/$30. Wave 3 (offsets 30--39) is
next under the unchanged design.
Wave 3 launched at `20260711T195210`, offsets 30--39, as ten isolated triplets at
aggregate configured concurrency 250. Health-only monitoring remains in force.
Wave-3 original offset 30 failed closed during arbitration on the strict bounded
simulator-faithfulness check and banked no item. Exact full-triplet recovery 1 launched
at `20260711T195708`; the failed invocation is noncanonical.
Wave-3 original offset 35 also failed closed during arbitration on the strict bounded
simulator-faithfulness check and banked no item. Exact full-triplet recovery 1 launched
at `20260711T200209`; both wave-3 recoveries preserve aggregate concurrency 250.
The preregistered best-N EIG candidate-elicitation probe is implemented at commit
`4708cbe`. The `standard` prompt mode preserves the faithful-EIG prompt; `best_n` adds
only the frozen goal anchor requesting the five best next actions for resolving the
issue quickly. The config uses canonical tasks 0--9, seed 1304, five rounds, thinking
26B A4B, EIG argmax, 25 OpenRouter concurrency, and a $0.60 projection. Its initial
`20260711T201520` process was terminated by the local launch wrapper before any API
request and is noncanonical. Exact recovery 1 launched in a managed session at
`20260711T201603`. With nine active wave-3 processes, aggregate configured concurrency
is 250. The probe remains outcome-blind until completion and does not overlap the
held-out headline task set.
Wave-3 original offset 32 failed closed during candidate 0 after arbitration had
completed. The complete invocation is noncanonical and its first item is discarded.
Exact full-triplet recovery 1 launched unchanged at `20260711T202059`. During exception
triage, a context-bearing grep command inadvertently printed the failed invocation's
arbitration metric line. This was a noncanonical artifact already quarantined before
inspection; no canonical task outcome was viewed, no comparison was made, and no design,
recovery, or analysis decision changed. Subsequent health checks must extract only the
final exception line without surrounding log context.
The Path E paper's outcome-independent rewrite is now a compiling five-page draft. It
covers the external benchmark, native-primary EIG arbitration, candidate-0 causal
control, answer-space and endpoint audit, frozen analysis, outcome blindness, and
limitations. Its Path E validator passes. The results section explicitly remains sealed
until the canonical 50-task analyzer and manual audit.
Wave-3 original offset 38 failed closed during candidate 0 after arbitration completed.
The whole invocation is noncanonical. Exact full-triplet recovery 1 launched unchanged
at `20260711T202641`; exception-only triage confirmed the same bounded simulator
faithfulness failure without exposing another metric line.
Wave 3 is complete and canonical for offsets 30--39. Recoveries 1 are the accepted
sources for offsets 30, 32, 35, and 38; all other offsets use their original invocation.
The health-only combination has exact task/method coverage and 19/19 matching ordered
proposal sets at eligible identical-history turns. Accepted canonical cost was
$0.483515 over 3,368 requests; operational wave cost including four failed attempts was
$0.585700 over 4,238 requests. Actual cumulative spend, including the concurrently
running best-N probe, was $12.304248/$30 at banking time. No canonical policy outcome
was inspected. Wave 4 offsets 40--49 is next; while the 25-concurrency probe remains
active, launch at most nine 25-concurrency shards and fill the tenth slot only after one
process completes, keeping aggregate configured concurrency at or below 250.
While closing completed managed tool sessions after Wave 3 had already been fixed, the
session flush printed terminal metric lines for the naive arm of canonical offsets 32
and 38. This was an additional outcome-blindness protocol deviation: two single-arm
canonical task outcomes were inadvertently visible, but no arbitration/candidate-0
outcome or cross-arm comparison was inspected, and the sample, design, analyzer,
recovery sources, and launch decisions were already frozen and remain unchanged. Do not
flush completed run sessions again; use metadata-only polling.
Wave 4 began at `20260711T205544` with offsets 40--48 as nine isolated unchanged
thinking triplets. Together with the still-running 25-concurrency best-N probe, aggregate
configured concurrency is 250. Offset 49 remains intentionally unlaunched and will fill
the first released 25-concurrency slot. Health-only and whole-triplet recovery rules are
unchanged.
Wave-4A original offset 44 failed closed during thinking naive after arbitration and
candidate 0 completed. The entire invocation is noncanonical. Exact full-triplet
recovery 1 launched unchanged at `20260711T210824` into the released slot; offset 49
remains queued and aggregate configured concurrency remains 250.
Original offsets 40 and 46 completed, releasing two slots. The queued original offset
49 launched unchanged at `20260711T211424`. With six other original shards, recovery
44, and the best-N probe still active, aggregate configured concurrency is 225.
Offset-44 recovery 1 failed closed during arbitration and banked no item. Exact
full-triplet recovery 2 launched unchanged at `20260711T211909`; both earlier attempts
remain noncanonical. Aggregate configured concurrency is 225.
Offset-44 recovery 2 also failed closed during arbitration and banked no item. Exact
full-triplet recovery 3 launched unchanged at `20260711T212925`; all three earlier
attempts remain noncanonical.
Offset-44 recovery 3 also failed closed during arbitration and banked no item. Exact
full-triplet recovery 4 launched unchanged at `20260711T213901`; all four earlier
attempts remain noncanonical.
Offset-44 recovery 4 cleared arbitration but failed closed during candidate 0. Exact
full-triplet recovery 5 launched unchanged at `20260711T214607`; all five earlier
attempts remain noncanonical.
Wave 4 is complete and canonical for offsets 40--49. Recovery 5 is the accepted source
for offset 44; all other offsets use their first invocation. The health-only combination
has exact task/method coverage and 16/16 matching ordered proposal sets at eligible
identical-history turns. Accepted cost was $0.521274 over 3,359 requests; operational
wave cost including five failed offset-44 attempts was $0.614086 over 4,155 requests.
Actual cumulative spend, including the still-running best-N probe, was $13.090964/$30 at
banking time. No additional canonical policy outcome was inspected. Wave 5 offsets
50--59 is next under the same frozen design and concurrency rule.
Wave 5 began at `20260711T220534` with offsets 50--58 as nine isolated unchanged
thinking triplets. Together with the still-running 25-concurrency best-N probe,
aggregate configured concurrency is 250. Offset 59 remains intentionally queued and
will fill the first released slot. This is the final thinking-headline wave.
Wave-5A original offset 51 failed closed during candidate 0 after arbitration completed.
The whole invocation is noncanonical. Exact full-triplet recovery 1 launched unchanged
at `20260711T221626` into the released slot; offset 59 remains queued and aggregate
configured concurrency remains 250.
Original offset 58 completed and released a slot. The queued original offset 59 launched
unchanged at `20260711T222015`; aggregate configured concurrency returned to 250.
Wave 5 is complete and canonical for offsets 50--59. Recovery 1 is the accepted source
for offset 51; all other offsets use their first invocation. The health-only combination
has exact task/method coverage and 17/17 matching ordered proposal sets at eligible
identical-history turns. Accepted cost was $0.566552 over 3,817 requests; operational
wave cost including the failed offset-51 attempt was $0.589193 over 3,994 requests.
Actual cumulative spend, including the still-running best-N probe, was $13.849445/$30 at
banking time. The complete 50-task thinking headline is now fixed. Next: combine all 50
thinking triplets, finish and bank the best-N probe, then launch the five frozen
non-thinking blocks without overlapping another large wave.
The 50 canonical thinking triplets are combined at
`runs/paprika-headline-triplet-combined-seed1304`. Structural validation found exactly
50 records per arm over tasks 10--59 and 88/88 matching proposal sets at all eligible
identical-history turns. Canonical thinking evaluation totals are $2.727867, 17,595
requests, 5,188,570 prompt tokens, 5,712,603 completion tokens, 4,525,450 reasoning
tokens, and 64 forced exits. No policy endpoint comparison has been computed.
The non-thinking headline wave began at `20260711T230647` with 10-task blocks starting
at offsets 10, 20, 30, and 40. Four 51-concurrency blocks plus the still-running
25-concurrency best-N probe give aggregate configured concurrency 229. The final block
at offset 50 remains queued and will fill the first released 51-concurrency slot. No
thinking-headline process overlaps this wave.
The best-N EIG probe completed at `20260711T201603`: one 10-task EIG item, zero structured
or simulator-faithfulness failures, $0.435815, 3,698 requests, 596,195 reasoning tokens,
and 10 forced exits. Its development-task outcome read is now allowed and remains
separate from the sealed headline.
Non-thinking original block 40 failed closed before banking an item. Exact block-40
recovery 1 and the queued original block 50 launched at `20260711T230931`. Together with
original blocks 10, 20, and 30, aggregate configured non-thinking concurrency is 255.
The preregistered best-N outcome read is complete. Best-N EIG resolved 5/10 tasks with
mean censored turns 4.0, versus thinking naive 4/10 and 4.9 turns (4/1/5 paired
wins/losses/ties, delta -0.9, bootstrap CI [-2.2, 0.4]) and arbitration 6/10 and 3.7
turns (2/4/4, delta +0.3, CI [-1.2, 1.6]). It is directionally better than thinking
naive and statistically indistinguishable from arbitration. Per the decision rule
written before launch, this is competitive: the paper adopts goal-anchored
generate-and-select as the clean method identity, arbitration remains the calibrated
robustness variant, and a best-N EIG arm must run on held-out tasks 10--59 before the
headline analyzer. This arm addition is triggered solely by the preregistered
development probe; no held-out comparison has been computed. Evidence is in
`results/path_e/best_n_probe/`.
Non-thinking blocks 10, 20, 30, and 50 completed. Block-40 recovery 1 also failed closed
before banking an item. Exact block-40 recovery 2 launched at `20260711T231618` with
best-N held-out offsets 10--17 as eight isolated one-task shards. The recovery uses 51
concurrency and best-N uses 8 x 25 = 200, for aggregate configured concurrency 251.
Remaining best-N offsets 18--59 fill released slots in canonical order.
Best-N offsets 10 and 11 completed. Their two released slots were filled by offsets 18
and 19 at `20260711T232014`; aggregate configured concurrency remains 251.
Non-thinking block-40 recovery 2 completed and is canonical, completing all five
non-thinking blocks over tasks 10--59. Its released capacity was filled by best-N
offsets 20 and 21 at `20260711T232220`; eight best-N shards remain live at aggregate
configured concurrency 200, plus the two new shards at 50, for 250 total.
The five canonical non-thinking blocks were structurally combined as
`runs/paprika-headline-nonthinking-combined-seed1304`: 50 records, exact task IDs
10--59, one `naive` item per task. Outcomes remain sealed until the frozen joint
analyzer runs after the best-N arm completes.
Best-N offset 12 completed and is canonical. Its released slot was filled by offset 22
at `20260711T233549`; ten best-N shards remain live at aggregate configured concurrency
250. Canonical best-N offsets 10--12 are recorded in the headline manifest.
Best-N offset 18 completed and is canonical. Its released slot was filled by offset 23
at `20260711T233747`; aggregate configured concurrency remains 250.
Best-N offset 15 completed and is canonical. Its released slot was filled by offset 24
at `20260711T234011`; aggregate configured concurrency remains 250.
Best-N offsets 17 and 19 completed and are canonical. Their released slots were filled
by offsets 25 and 26 at `20260711T234258`; aggregate configured concurrency remains 250.
Best-N offset 14 completed and is canonical. Its released slot was filled by offset 27
at `20260711T234542`; aggregate configured concurrency remains 250.
Best-N offset 13 completed and is canonical. Its released slot was filled by offset 28
at `20260711T234628`; aggregate configured concurrency remains 250.
Best-N offset 21 completed and is canonical. Its released slot was filled by offset 29
at `20260711T234839`; aggregate configured concurrency remains 250.
Best-N offset 20 completed and is canonical. Its released slot was filled by offset 30
at `20260711T235510`; aggregate configured concurrency remains 250.
Best-N offset 22 completed and is canonical. Its released slot was filled by offset 31
at `20260711T235622`; aggregate configured concurrency remains 250.
Best-N offset 16 completed and is canonical, completing the contiguous canonical block
10--22. Its released slot was filled by offset 32 at `20260711T235943`; aggregate
configured concurrency remains 250.
Best-N offsets 28 and 30 completed and are canonical. Their released slots were filled
by offsets 33 and 34 at `20260712T000300`; aggregate configured concurrency remains 250.
Best-N offset 23 completed and is canonical, extending the contiguous canonical block
to 10--23. Its released slot was filled by offset 35 at `20260712T000713`; aggregate
configured concurrency remains 250.
Best-N offset 24 completed and is canonical, extending the contiguous canonical block
to 10--24. Its released slot was filled by offset 36 at `20260712T000928`; aggregate
configured concurrency remains 250.
Best-N offset 26 completed and is canonical. Its released slot was filled by offset 37
at `20260712T001304`; aggregate configured concurrency remains 250. Offset 25 remains
live, so the contiguous canonical prefix remains 10--24.
Best-N offsets 34 and 36 completed and are canonical. Their released slots were filled
by offsets 38 and 39 at `20260712T001354`; aggregate configured concurrency remains 250.
Best-N offset 25 failed after exhausting structured parsing repairs with
`Model response did not contain a JSON object`; the whole shard is noncanonical. Exact
unchanged recovery 1 launched at `20260712T001510` in the released slot. Aggregate
configured concurrency remains 250; offset 40 remains queued until another slot clears.
Best-N offset 29 completed and is canonical while offset-25 recovery 1 remains live.
Its independent released slot was filled by offset 40 at `20260712T001627`; aggregate
configured concurrency remains 250.
Best-N offset 27 completed and is canonical while offset-25 recovery 1 remains live.
Its independent released slot was filled by offset 41 at `20260712T001723`; aggregate
configured concurrency remains 250.
Best-N offset 33 completed and is canonical while offset-25 recovery 1 remains live.
Its independent released slot was filled by offset 42 at `20260712T001947`; aggregate
configured concurrency remains 250.
Best-N offset 31 failed after transport retries on a transient DNS resolution error;
the whole shard is noncanonical. Exact unchanged recovery 1 launched in its released
slot at `20260712T022747`. Offset-25 recovery 1 remains live, and aggregate configured
concurrency remains 250.
Best-N offset 38 completed and is canonical while both recoveries remain live. Its
independent released slot was filled by offset 43 at `20260712T030151`; aggregate
configured concurrency remains 250.
Local DNS resolution recovered and provider spend resumed. Best-N offset 32 completed
and is canonical while both recoveries remain live. Its released slot was filled by
offset 44 at `20260712T030346`; aggregate configured concurrency remains 250.
Best-N offset 37 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 45 at `20260712T030648`; aggregate configured
concurrency remains 250.
Best-N offset 39 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 46 at `20260712T030923`; aggregate configured
concurrency remains 250.
Best-N offsets 35 and 41 completed and are canonical while both recoveries remain live.
Their released slots were filled by offsets 47 and 48 at `20260712T031446`; aggregate
configured concurrency remains 250.
Best-N offset 44 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 49 at `20260712T031549`; aggregate configured
concurrency remains 250.
Best-N offset 46 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 50 at `20260712T031713`, beginning the final ten
held-out offsets; aggregate configured concurrency remains 250.
Best-N offset 49 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 51 at `20260712T031915`; aggregate configured
concurrency remains 250.
Best-N offset 43 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 52 at `20260712T032935`; aggregate configured
concurrency remains 250.
Best-N offset 40 completed and is canonical while both recoveries remain live. Its
released slot was filled by offset 53 at `20260712T033111`; aggregate configured
concurrency remains 250.
Hanshal added another $10 of OpenRouter authorization during the held-out best-N run.
The authorized total is now $40; at the amendment point tracked spend was $16.325627,
leaving $23.674373. This changes only the administrative budget cap. Active shards keep
their instantiated cap, while future launches use $40; no scientific or concurrency
parameter changes.
Best-N offset-31 recovery 1 and original offsets 47--48 completed and passed the sealed
structural check (one metrics item each). They are canonical, bringing the held-out arm
to 37/50 tasks. Their three slots were filled by offsets 54--56 at
`20260712T034447`, `20260712T034449`, and `20260712T034451`. Ten one-task shards are
again live at aggregate configured concurrency 250. The new shards use the $40 cap;
the seven older live shards retain their instantiated $30 cap.
Best-N offset 50 completed and passed the sealed one-item structural check, becoming
canonical task 38/50. Its slot was filled by offset 57 at `20260712T034827`; ten shards
remain live at aggregate configured concurrency 250.
Best-N offset-25 recovery 1 completed and passed the sealed one-item structural check,
becoming canonical task 39/50. Its slot was filled by offset 58 at
`20260712T035044`; ten shards remain live at aggregate configured concurrency 250.
Best-N offsets 42, 45, 51, and 58 completed and passed sealed one-item structural
checks, bringing the held-out arm to 43/50 canonical tasks. Offset 59, the final
regular shard, launched at `20260712T035514`. Seven shards remain live at aggregate
configured concurrency 175; no additional tasks are queued.
Best-N offset 59 completed and passed the sealed one-item structural check, becoming
canonical task 44/50. Six shards remain live at aggregate configured concurrency 150;
all intended task IDs have been launched and no additional work is queued.
Best-N offset 55 completed and passed the sealed one-item structural check, becoming
canonical task 45/50. Five shards remain live at aggregate configured concurrency 125.
Best-N offsets 52, 53, and 57 completed and passed sealed one-item structural checks,
bringing the arm to 48/50 canonical tasks. Only offsets 54 and 56 remain live, at
aggregate configured concurrency 50.
Best-N offset 56 completed and passed the sealed one-item structural check, bringing
the arm to 49/50 canonical tasks. Offset 54 is the sole remaining live shard at
configured concurrency 25.
Best-N offset 54 completed and passed the sealed one-item structural check. The held-out
best-N arm is now 50/50 canonical across exact offsets 10--59, with no live OpenRouter
processes. Operational cumulative spend is $16.710100/$40. Next: combine the 50
canonical shards and structurally validate the combined arm before running the frozen
headline analyzer exactly once.
The 50 canonical best-N shards are combined at
`runs/paprika-headline-best-n-combined-seed1304`. Structural validation found one EIG
item, 50 unique sources, exact task IDs 10--59, and 50 records. Canonical best-N usage
is $2.672365, 23,570 requests, 3,488,175 reasoning tokens, and 55 forced exits. No
endpoint outcome was inspected. All five arms are now fixed; next run the frozen joint
headline analyzer exactly once, then perform the preregistered manual endpoint audit.
The frozen analyzer then ran exactly once and returned
`claim_b_not_confirmed_requires_manual_review`. Automated endpoint and candidate
pairing checks passed, but both preregistered policy comparisons failed to favor
arbitration. The manual audit found the fatal task-13 paper-roll false failure described
in NEXT ACTIONS and marked the complete headline INVALID. Authoritative artifacts are
`results/path_e/arbitration_headline/PAPRIKA_HEADLINE.{json,md}` and
`results/path_e/arbitration_headline/MANUAL_REVIEW.md`. No MediQ run was launched.


**AUTHORIZATION (2026-07-11, per Hanshal): Option 2 — the repaired-endpoint path is
authorized.** Do NOT stop Path E on the invalid measurement. Conditions, all mandatory:

1. **Quarantine first.** All prior Step 1 + rescue results (canonical run, rescue waves)
   are endpoint-invalid: mark their artifacts INVALID-ENDPOINT, keep them as diagnostics
   in `results/path_e/step1_invalid/`, and never cite them as policy evidence. The
   PAPRIKA_ENDPOINT_AUDIT.md stays as the record of why.
2. **Endpoint repair mirrors Paprika's native complete-conversation success protocol
   exactly** — no house variant. Every discrepancy example in the audit (trailer
   connector, kiosk cleaning inconsistency, kiosk remedy contradiction) becomes a
   committed regression test that must pass.
3. **Simulator-faithfulness gate (new, required).** The audit shows the simulator can
   contradict the hidden ground truth — that corrupts the interaction itself, not just
   scoring, and it biased AGAINST arms that proposed correct remedies. Add a
   faithfulness check to the revalidation smoke: rate of simulator replies inconsistent
   with the private solution must be ~0; contradictions are adapter/prompt bugs to fix,
   not noise to average over.
4. **Revalidate Step 0a from scratch**: fresh 5-task real-model smoke under the repaired
   protocol — coverage >= 85%, zero terminal failures, faithfulness ~0, manual
   transcript review, tracked report. Step 0a's previous pass is void until this runs.
5. **Fresh paired Step 1** under the same frozen design (10 canonical tasks, 5 rounds,
   seed 1304, frozen censoring/tie rules): arms = naive non-thinking, naive thinking,
   EIG with thinking-generation (the authorized rescue config — candidate quality was
   the diagnosed bottleneck, so it is the primary scaffold arm). NO full 2-step (stays
   dropped). Projected cost <= ~$3; check against remaining budget before launch.
6. **The decision-boundary addendum carries over unchanged** to the fresh Step 1:
   claim-1 pass -> scale claim 1 (more tasks, then MediQ); claim-1 fail under VALID
   endpoints -> the one pre-registered naive-primary arbitration variant -> then
   stop-and-discuss regardless.
7. No other scope changes; no new environments; no new method variants beyond the
   above.

Repaired-endpoint implementation status: prior Step 1 evidence is quarantined under
`results/path_e/step1_invalid/` with explicit INVALID-ENDPOINT labels and a source-run
manifest. The adapter now uses the released customer role instructions, checks every
simulator reply against the private solution, regenerates contradictions within the
bounded repair budget, fails closed if a contradiction survives, and applies the
authorized complete-conversation success judge every turn with Paprika's
`Goal reached OR judge` rule. It logs raw contradiction and final inconsistency rates.
All audited trailer/kiosk discrepancies are committed regressions. Focused verification:
117 tests pass; the project-wide suite is 536 passed / 1 skipped with four unrelated
stale Path A validator assertions. Fresh Step 0a config is
`configs/config_paprika_step0a_repaired_endpoint_openrouter.yaml` (eval tasks 5-9,
including both audited task families). Do not launch until this implementation commit
is pushed.
Fresh repaired-endpoint Step 0a run `20260711T095626` initially passed on eval tasks
5-9, but is now superseded by a terminal-faithfulness discrepancy found during the
required Step 1 transcript review. The first fresh Step 1 attempt is INVALID-ENDPOINT
and quarantined in `results/path_e/step1_invalid/PAPRIKA_STEP1_REPAIRED_ATTEMPT.md`;
its automated policy comparison must not be cited. The exact dishwasher mismatch is a
committed regression. Literal `Goal reached` replies now receive a second strict check
that requires the latest action to directly match the private cause/remedy; merely
plausible alternatives are rejected and regenerated. Fresh five-task real-model
revalidation on tasks 0-4 is the active gate. No arbitration or policy scaling is
allowed before that gate and a fully fresh Step 1 pass manual review.
The terminal-gate smoke `20260711T130833` passed from implementation commit `a5da29a`
with 100% coverage, zero failures/inconsistencies, terminal metrics present, and manual
review passed. Fresh Step 1 is authorized again, but every arm must rerun from scratch
under the terminal-repaired endpoint; no artifact from the invalid attempt can be reused.
The fully fresh three-arm Step 1 launched as timestamp `20260711T133046` from commit
`2c290dc`: one batched naive non-thinking process, ten isolated naive-thinking tasks,
and ten isolated generation-thinking EIG tasks. Effective aggregate concurrency is near
250. All 21 processes completed with zero failures, and manual review passed all 30
transcripts. This is the first valid Step 1 result. Resolution@5 was 0.30 naive
non-thinking, 0.40 naive thinking, and 0.30 generation-thinking EIG. EIG versus matched
naive non-thinking was 2 wins / 3 losses / 5 ties with +0.5 mean censored-turn delta,
so Claim 1 failed. Evidence is tracked under `results/path_e/step1_terminal/`.

Per the unchanged pre-registration, exactly one final variant is now authorized:
`NaivePrimaryArbitration`. One thinking-native prompt proposes exactly three ordered
actions; candidate 0 is the native default. Beliefs score only those three with
categorical one-step EIG, and an alternative overrides candidate 0 only when its score
gap exceeds one combined standard error. The SE is deterministic from weighted
per-hypothesis expected information contributions and adds no LLM calls. The frozen
analyzer compares arbitration primarily against thinking naive and requires endpoint
validity. After this run, stop-and-discuss regardless of outcome.
The frozen arbitration run launched at `20260711T143605` from commit `542dfc3` as ten
one-task shards (offsets 0--9), with aggregate configured OpenRouter concurrency 230.
Do not change its settings in flight. The user permits concurrency up to 256 for future
work when healthy and useful, but this run remains frozen for comparability.
Nine original arbitration shards completed. Original offset 8 exhausted its bounded
structured repairs during the round-4 analytical belief refresh and produced only an
empty error-metrics artifact. The identical offset-8-only recovery `20260711T153606`
completed and is the only accepted offset-8 artifact.

**ARBITRATION RESULT: PASS, THEN STOP-AND-DISCUSS.** Manual endpoint review passed all
10 transcripts. Resolution@5 was 0.60 arbitration versus 0.40 thinking naive and 0.30
non-thinking naive. Against thinking naive, arbitration had 6 wins / 0 losses / 4 ties,
mean censored-turn delta -1.2, bootstrap CI [-2.2, -0.4]. EIG overrode candidate 0 on
12/33 turns; four overrides immediately selected the exact remedy, but several were
unhelpful. This is a strong 10-task pilot signal, not a definitive effect estimate:
candidate generation is stochastic and the accepted arm cost 2.52x thinking naive.
Per the frozen rule, DO NOT launch scaling, MediQ, or another rescue automatically.
Discuss the paper direction with Hanshal first. Evidence is under
`results/path_e/arbitration_terminal/`.

**STOP-AND-DISCUSS NEXT DECISION:** a costed decision memo is now tracked at
`results/path_e/arbitration_terminal/DECISION_MEMO.md`. The recommended path is a
pre-registered 50-task held-out Paprika headline on eval offsets 10--59. A
prompt-matched candidate-0 arm is scientifically required at scale because the pilot's
thinking-naive baseline used a different one-action prompt; without that control, the
gain cannot be attributed specifically to EIG overrides rather than three-candidate
generation. Proposed arms are arbitration, prompt-matched candidate 0, thinking naive,
and non-thinking naive. Nominal OpenRouter projection is about $2.67, with a conservative
2x envelope of $5.34; current cumulative spend is $10.38197/$20. This launch is NOT
authorized yet. Await Hanshal's explicit choice; do not start scaling or MediQ.
The initial batched thinking-naive process was terminated after 39 minutes because one
provider response held the entire ten-task batch after nine first-round completions.
It produced no artifact and is not used. The exact same arm is being recovered as ten
one-task shards; model, thinking budget, prompts, seed, task set, rounds, and endpoint
are unchanged. Recovery timestamp is `20260711T105624`, launched from commit `fbddab9`.
Original EIG offset 5 also exhausted its bounded structured-repair budget on malformed
hypothesis-refresh JSON and produced no artifact. The identical isolated offset-5 shard
was relaunched as timestamp `20260711T110229` from commit `882c93f`; only the successful
recovery artifact will enter the paired result.
Original EIG offset 8 likewise exhausted the bounded structured-repair budget because
refresh responses lacked six unique hypotheses. Both EIG recovery shards and all ten
thinking-naive recovery shards completed, but the assembled result is quarantined due
to the manual endpoint failure above.


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
   `results/path_e/step1_invalid/PAPRIKA_STEP1_RESCUE.{json,md}` and is quarantined
   as INVALID-ENDPOINT diagnostic evidence only.
   A post-gate endpoint audit identified a deeper validity problem: the adapter
   does not implement Paprika's released complete-conversation success protocol.
   It can label a prospective `I'll try that` reply as solved, miss explicit
   `that fixed it` replies, and accept simulator replies that contradict the
   released private remedy. See
   `results/path_e/step1_invalid/PAPRIKA_ENDPOINT_AUDIT.md`.
   Therefore these Step 1 numbers are endpoint-integration diagnostics plus an
   insufficient policy comparison, not valid evidence that EIG loses on Paprika.
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
  key in `OPENROUTER_API_KEY` (never commit). $40 total budget, FULLY authorized;
  refuse runs projected past remaining budget; flag Hanshal before the remaining
  authorization becomes tight; track
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
