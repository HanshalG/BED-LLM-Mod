# GOAL (Path B): Goal-oriented LLM experimental design that beats naive AND greedy EIG on standard location finding

## HOW TO USE THIS DOCUMENT

This file drives all work toward a NeurIPS-workshop submission. It is immutable during
the run; all LIVING state lives in `STATE.md`, which you can edit. Rules:

- **First action of every session**: read `STATE.md`. Execute from it; use this file for
  the target result, method spec, gates, playbook, and definition of done.
- **Maintenance contract**: at the end of every working session, update `STATE.md` and
  append a row to `EXPERIMENTS.md` for anything launched. A stale `STATE.md` is a bug.
- **Precedence**: `STATE.md` > this file.
- **Cluster restriction**: at most 8 active jobs; `--partition=msc,llm --exclude=oat12`
  unless the user changes this.

## THE RESULT WE ARE AFTER (all four required)

A method that, on **standard multi-source location finding** (2–3 sources, 10 rounds,
NO movement constraint — the DAD-style benchmark setting, not a tuned trap):

1. **Beats naive LLM prompting** on paired final RMSE (primary) with bootstrap CI
   excluding zero.
2. **Beats greedy LLM EIG** on the same endpoint.
3. Uses **no environment engineering** — if it only works in a hand-tuned geometry, it
   does not satisfy this goal.
4. Has a scorer whose estimates **rank realized ΔRMSE with clearly positive ρ** — the
   Path A entropy scorer's ρ≈0 against RMSE is disqualifying and must not recur.

## WHY THIS SHOULD WORK (evidence from Path A — treat as banked motivation)

- Naive thinking-LLM ≈ non-LLM oracle planner (0.166 vs 0.153 final RMSE, constrained
  arm); greedy-EIG scaffolding HURT vs naive in BOTH arms. Lesson: do not replace the
  LLM's native policy — arbitrate over it.
- Entropy scoring decorrelates from RMSE (gate ρ≈0) because entropy rewards posterior
  concentration on ANY hypothesis, including wrong branches. Naive had the best
  truth-log-prob and RMSE while being middling on entropy drop. Lesson: EIG is the wrong
  utility for a localization task; use task-loss-aligned utility.
- Early run `90338`/`90339`: on the standard harder task, StrategyEIG-d3 beat naive by
  ~15% (3 trials). Lesson: the standard task has headroom; the easy 1-source/6-round
  variant is at ceiling (all methods 0.089–0.108) and cannot show anything.

## METHOD SPEC

**Utility (fixes requirement 4).** Score candidate queries by rollout-estimated
**expected posterior loss reduction**: expected posterior RMSE = E_{θ∼posterior}
||θ − θ̂(posterior)|| (best-permutation matched for multi-source), computable at decision
time with no access to truth. Rollouts reuse the existing analytic machinery (sample
θ from posterior, simulate observation(s), closed-form reweight, measure expected
posterior RMSE drop). Non-myopic depth n is a knob, not the headline; start at n ∈ {1, 3}.

**Candidate set (requirement 2).** Every round, candidates MUST include: (a) the naive
LLM's proposed query — the PLAIN raw-history naive policy, NOT naive+belief (Path A:
naive 0.166 vs naive+belief 1.137; belief-summary conditioning poisons the policy),
(b) the greedy-EIG argmax query, (c) LLM strategic proposals (reuse strategy generation,
small K). Report the
selection-frequency table (how often each proposal type wins) — it is the paper's
attribution analysis.

**Conservative override (this, not argmax, delivers the ≈naive floor).** Plain argmax
over noisy scores overrides naive exactly when scorer noise is largest (winner's curse)
and can end up WORSE than naive. Rule: default to the naive proposal; depart only when
another candidate's scored advantage exceeds δ = c · SE, where SE is the per-decision
standard error of that candidate's rollout score (computable at deployment from the R
rollouts) and c is fixed a priori (c = 1; pre-registered before Phase 2, not tuned). This is
what makes "no worse than naive, better when the scorer has signal" approximately true.

**Name the method** something honest like Task-Aligned Design Arbitration (TADA) or
goal-oriented StrategyEIG; final name is a writing decision.

## VALIDATION CHAIN (gates in order — do not skip)

**Gate 0 (local, no cluster, do first).** Re-score the existing ranking-fidelity records
(`runs/rankfid26b_a4b_gate_v2ghs_*`) with expected posterior RMSE as the estimated
utility, and compute Spearman ρ against realized ΔRMSE and realized
expected-posterior-RMSE drop. If the stored records lack rollout-level posterior
snapshots for the estimated side, fall back to re-scoring the stored candidates with the
analytic machinery locally — the scorer needs no LLM, so this remains cluster-free.
The GATING metric is ρ against realized expected-posterior-RMSE drop (the smooth,
rankable target): PASS ≥ ~0.3 at some depth. Also report ρ against realized point-ΔRMSE
alongside its rankability ceiling — the realized-vs-realized correlation between
expected-posterior-RMSE drop and point-ΔRMSE — so a low point-RMSE ρ is attributed to
endpoint noise, not the objective (Path A showed point-ΔRMSE may be unrankable at probe
horizons by ANY scorer). FAIL: the objective doesn't rank the smooth target → stop,
diagnose supports/horizons before any cluster spend. Either way append to
`results/ranking_fidelity/`.

**Gate 1 (cheap pilot, cluster).** Standard task (2–3 sources, 10 rounds, noise as in
`config_location_finding.yaml`), 5 paired trials, arms: naive, greedy EIG, arbitration
n=1. Checks: (a) naive final RMSE ≥ ~0.4 (headroom exists — if naive is at ceiling,
increase sources/reduce rounds and re-pilot); (b) **utility-divergence check**: on the
same candidate sets, fraction of decisions where EIG-argmax ≠ task-loss-argmax — if the
two utilities rarely disagree (≲15%), there is no room to beat greedy EIG on this task
regardless of scorer quality; verify posteriors stay multimodal (multi-source ambiguity
should do this naturally) or the claim needs re-scoping; (c) **override rate + gain**:
how often arbitration departs from naive and the mean scored/realized gain per override —
the achievable effect vs naive is bounded by their product; use it to POWER the Phase 2
trial count instead of defaulting to 30–50; (d) selection-frequency table is
non-degenerate.

**Phase 2 (headline sweep).** 30–50 paired trials. REQUIRED arms: naive, greedy EIG,
arbitration n=1, arbitration n=3, matched-compute entropy-utility control (same rollout
budget — isolates the utility change), and **the no-LLM control: analytic A-optimal
selection over the same support-grid candidates with NO LLM proposals**. The no-LLM arm
is not optional — it decides what the paper is: if it alone beats naive, the honest
contribution is "task-aligned utility" with the LLM as proposer only where the selection
table shows LLM proposals winning; if LLM proposals (naive's query, strategies) are
frequently selected and the full method beats the no-LLM arm, the LLM proposal
distribution is demonstrably load-bearing. Optional arms (cut first): naive+belief,
no-naive-candidate ablation. Pre-register endpoints in the runbook BEFORE launch:
primary = paired final RMSE vs naive AND vs greedy EIG; secondary = expected posterior
RMSE, truth-log-prob, entropy, arbitration-vs-no-LLM delta; bootstrap CIs primary,
Wilcoxon supporting. Pre-declare the canonical run if variants are launched.

**Phase 3 (packaging + paper).** Reuse Path A packaging/validators. Paper arc:
(i) measured failure: entropy-scored LLM design decorrelates from task loss (Path A gate
+ sweep as motivation, honestly reported); (ii) explanation: multimodal posteriors,
confident-but-wrong concentration; (iii) fix: goal-oriented utility + proposal-inclusive
arbitration; (iv) result: beats naive and greedy EIG on a standard task; (v) attribution:
matched-compute entropy control + selection-frequency table + no-naive-candidate
ablation. Positioning — CLAIMS DISCIPLINE IS CRITICAL HERE: expected-posterior-loss
utility is classical decision-theoretic design (Bernardo 1979; A-optimality vs
D-optimality/EIG; loss-calibrated inference). NEVER claim to propose goal-oriented
design. The claim is: (a) a measured decorrelation between EIG and task loss for
LLM-scaffolded design agents, and (b) bringing task-aligned utilities + conservative
proposal arbitration to LLM experimental design. REQUIRED reading before Phase 2: arXiv
2605.26093 ("Goal-driven BOED for Robust Decision-Making", May 2026) — directly adjacent,
non-LLM; write the differentiation into `results/POSITIONING.md`. BED-LLM, DAD, and
Path A are all EIG-based; the LLM × task-aligned-utility corner is the open one.
Transferability scoping (a reviewer WILL ask): this testbed has an analytic likelihood,
so the task-loss scorer is exact; in open-ended LLM-BED settings (20Q-style) the
posterior is LLM-estimated and the scorer inherits that noise. State explicitly that
this work isolates the utility question under exact scoring, and that task-aligned
utilities under LLM-estimated posteriors are the follow-up — do not imply the fix
transfers for free.

## OUTCOME PLAYBOOK

| Phase 2 outcome | Framing |
|---|---|
| Beats naive AND greedy, and beats the no-LLM control | Full claim: task-aligned utility + LLM proposal distribution both load-bearing; entropy control isolates the utility, no-LLM control isolates the proposals |
| Beats naive AND greedy, but no-LLM control matches | Honest reframe: "task-aligned utility fixes LLM-BED scaffolds" — the utility is the contribution, LLM proposals are optional on this task; selection table + 20Q-style tasks as future work for where proposals must matter |
| Beats greedy, ties naive | "Scaffolds stop hurting: conservative task-aligned arbitration recovers native LLM competence and dominates EIG scaffolds"; selection table shows when overriding naive pays |
| Ties both (selection ≈ always picks naive) | Diagnostic paper: even task-aligned scorers cannot out-select a strong native policy; scorer-fidelity + selection analysis is the contribution |
| Gate 0 fails | The decorrelation is deeper than the objective: report entropy AND task-loss scorers both failing to rank realized RMSE — a serious negative result about LLM/rollout plan evaluation, written up with the Path A evidence |

Every row is publishable if executed honestly; only the first two are strong. No row
requires environment engineering.

## DEFINITION OF DONE

1. Gate 0 and Gate 1 artifacts in `results/`; Phase 2 pre-registered before launch and
   analyzed exactly as registered.
2. Headline figure: paired final RMSE deltas vs naive and vs greedy EIG with CIs, plus
   selection-frequency table and matched-compute entropy control.
3. 4–6 page draft in `paper/` following the applicable playbook row, reusing Path A
   material as motivation; validators pass; every result traceable via `EXPERIMENTS.md`
  to a commit/tag.
4. `STATE.md` updated to reflect completion.

## SALVAGE FROM PATH A (do not rebuild)

Paired fixed-root sweep machinery, support-grid + analytic rollout scoring (swap the
utility function), ranking-fidelity scripts (swap scored metric), packaging/validators,
ledger, oracle/robustness artifacts (background material), the Path A paper draft
(becomes the motivation section), all operational knowledge in `STATE.md`.

## SCOPE DISCIPLINE

This is a workshop paper. The MPP is Gate 0 + Gate 1 + Phase 2 primary arms + one
attribution control + the paper. Cut order under pressure: no-naive-candidate ablation,
n=3 arm, naive+belief arm. Do not add environments, constraints, or new scoring modes
beyond the spec. Do not start any non-MPP item while an MPP item is incomplete.
