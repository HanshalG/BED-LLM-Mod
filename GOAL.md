# GOAL: Non-myopic Bayesian experimental design for interactive LLM agents, evaluated on external benchmarks (Paprika customer service + MediQ)

## HOW TO USE THIS DOCUMENT

This file is immutable during the run. All living state lives in `STATE.md`, which you
can and must edit.

- **First action of every session**: read `STATE.md`. Execute from its NEXT ACTIONS; use
  this file for the claims, method spec, gates, playbook, and definition of done.
- **Maintenance contract**: at the end of every working session update `STATE.md`; append
  a row to `EXPERIMENTS.md` for every launch (run name, job ID, config, commit/tag,
  status, key metric, artifacts). A stale `STATE.md` is a bug — fix it first.
- **Precedence**: `STATE.md` > this file.
- **Cluster**: at most 8 active jobs; `--partition=msc,llm --exclude=oat12`; no GH200
  unless Hanshal asks.
- **API fallback (added 2026-07-10, cluster down)**: OpenRouter with a $20 budget is
  authorized for Gemma 4 26B A4B inference (details/spend tracking in `STATE.md`).
  Rules: never commit the API key (env var only); log cost per run in `EXPERIMENTS.md`;
  the FULL $20 is authorized (per Hanshal, 2026-07-10) — still cost-project each run
  before launch and prioritize spend in validation-chain order (smoke → Step 1 → Step 2
  → headline), and tell Hanshal when cumulative spend reaches ~$18 so a top-up can be
  arranged before anything stalls mid-run; never mix backends (API vs cluster vLLM)
  WITHIN one paired comparison — a paired run set completes on the backend it started
  on, and the ledger records the backend per run.

## MOTIVATION (one paragraph of history — details in banked Path A–D artifacts)

Prior phases established, on location finding: (i) belief scaffolding fails where a naive
thinking-LLM is already near-oracle (low-dim continuous tasks); (ii) plan-score ranking
against realized per-instance gains is bounded by posterior concentration (aleatoric
ceiling), so ranking-fidelity gates must target the smooth expected quantity and method
value must be measured as average paired policy performance; (iii) lookahead estimator
noise grows with depth (cap at 2). This phase applies those lessons where the mechanism
has room to win: external interactive benchmarks with large semantic hypothesis spaces
and genuine sequential structure.

## ENVIRONMENT DOCTRINE (six requirements — all enforced)

- R1 Naive must not be near-oracle: hypothesis space too large to track in-context.
- R2 Greedy must have a STRUCTURAL gap given by the task (prerequisite chains, gated
  actions, budgets) — never engineered by us. Vanilla 20Q/Wordle/Mastermind fail R2;
  location finding fails R1. Neither may return as a claims environment.
- R3 Semantic: LLM priors/generation must be load-bearing.
- R4 Simulable with ground truth (rollouts + eval).
- R5 EXTERNAL benchmark — someone else's task definition.
- R6 Runnable with ≤26B models on msc/llm.

## ENVIRONMENTS

**Primary: Paprika customer-service troubleshooting** ("Training a Generally Curious
Agent", arXiv:2502.17543; code + environments at github.com/tajwarfahim/paprika).
Belief = candidate issues (LLM-generated + filtered, BED-LLM style — the animals-env
machinery generalizes); queries = questions/diagnostic actions; user simulator =
answerer model. The myopic gap is structural: troubleshooting has prerequisite chains
(cheap establishing questions unlock and inform later checks) which 1-step EIG cannot
see. Endpoint: turns-to-resolution / resolution@turn-budget, paired per task.
Positioning bonus: Paprika obtains information-seeking behavior via RL training; we test
whether an inference-time Bayesian scaffold achieves it without training (and can later
serve as the SFT/RL teacher — the project's original vision).

**Second env: MediQ** (arXiv:2406.00922, NeurIPS 2024) — MedQA converted to an
interactive question-asking benchmark with a patient simulator. Purpose-built for this
question, and its own headline finding is our motivation: naively prompting models to
ask questions DROPS accuracy ~11.3% vs not asking (naive interactive info-seeking is
certified-bad by the benchmark authors — R1 by external evidence). Belief = differential
diagnosis; queries = history questions. Endpoint: diagnosis accuracy @ question budget,
paired per case; MediQ's Expert-system variants are native baselines. Fallback if MediQ
integration stalls: AgentClinic (arXiv:2405.07960). Conference-version upgrade (not this
workshop): SDBench (arXiv:2506.22405, native dollar-cost axis).

**Harness-only: Mastermind (hardcoded filtering, exact posterior)** — unit-tests the
belief + EIG + lookahead pipeline free of LLM noise. Never a claims environment.

## METHOD

Belief state and 1-step EIG exactly as BED-LLM (arXiv:2508.21184): LLM-proposed
hypotheses (generation/refinement/filtering), LLM likelihoods, categorical posterior.
The repo's animals machinery implements this; new envs adapt via `core.Environment`.

**Answer-space handling (required — these envs answer in free text, unlike 20Q):** per
candidate query, the LLM proposes a small discrete answer set (3–5 mutually exclusive
outcomes); likelihoods P(outcome | hypothesis, query) are scored over that set; the
simulator's actual free-text reply is mapped to the nearest outcome (LLM judge call).
Log answer-set coverage (fraction of replies that map cleanly); if coverage < ~85% in
pilots, revise the outcome-proposal prompt before scaling — do not proceed on a leaky
answer space.

**Stopping/commit rule (required):** use the benchmark's native criterion where defined
(MediQ's Expert abstention decision; Paprika's task success check). Where a commit
decision is ours, commit when the posterior top-hypothesis mass exceeds a threshold or
the budget ends (declare the posterior argmax); the threshold is fixed in Step 2 and
pre-registered — never tuned per run.

**Shared candidates (required):** 1-step, full 2-step, and selective arms score the SAME
K candidate queries per round (and shared rollout seeds where applicable), so arms
differ only in scoring depth — otherwise lookahead value is confounded with proposal
quality.

**Model defaults (per Hanshal, 2026-07-10): 26B A4B thinking is the questioner/belief
model for ALL runs, pilots included** — no 4B pilot tier. User/patient simulator and
answer-mapping judge = 26B class. Same simulator model + per-task seed across arms for
pairing. Operational carry-over from Path A: 26B A4B at a 4096 thinking budget produced
~30% forced-thinking exits — log the forced-exit rate from the first smoke run onward,
and raise the budget (8k+) if it exceeds ~10–15%; degraded truncated generations were a
prime suspect in earlier marginal results.

**Selective 2-step lookahead** on top. Each round, score K candidate queries by 1-step
EIG. Expand a candidate one extra step ONLY when:
(a) **tie trigger** — the top candidates are within ε of each other, ε = scoring-noise SE
    (bootstrap/CRN variance of the EIG estimates), calibrated once in Step 2 then FROZEN
    and pre-registered. Rationale (the aleatoric lesson): when scores are within noise
    the myopic argmax is arbitrary — exactly and only then can lookahead change the
    decision; or
(b) **gating trigger** — the candidate is an availability-gated action whose EIG depends
    on an unestablished precondition.
Expansion: simulate answer branches under the current posterior, generate K′ < K
follow-ups per branch, score 2-step EIG with common random numbers across the tied set
(paired comparison, not absolute estimation). Depth cap 2 (banked evidence: depth 3 adds
noise, not value). Log trigger rate and tokens per round for the cost frontier.

**Arms everywhere**: naive agent; benchmark-native baselines (MediQ Expert variants;
Paprika-reported numbers where comparable); BED-LLM 1-step EIG (faithful — baseline and
base); full 2-step (upper anchor + cost); selective (ours).

## THE THREE NESTED CLAIMS (each independently publishable — this is the scope design)

1. **Transfer**: BED-LLM-style belief scaffolding + 1-step EIG beats naive and native
   baselines on Paprika customer service and MediQ. Nobody has run BED scaffolding on
   either benchmark; MediQ's naive-asking-hurts result makes headroom near-certain.
   This claim alone is a positive workshop paper.
2. **Lookahead**: full 2-step beats 1-step where sequential structure exists (the
   genuine bet; gated cheaply in Step 1).
3. **Efficiency**: selective triggering captures most of 2-step's gain at ≤ ~40% of its
   lookahead tokens (the mechanism).

Write the paper so claim 1 carries it if claim 2 is weak; claims 2+3 elevate it if they
land. All endpoints paired per task instance with bootstrap CIs; Wilcoxon supporting.

## VALIDATION CHAIN (in order; gates are numeric and pre-registered)

**Step 0a — environment standing.** Clone Paprika (ledger the commit/URL), adapt the
customer-service tasks into `core.Environment` (reuse animals-env belief machinery;
user simulator = answerer model). ACCEPTANCE CRITERION: each adapted task must expose a
well-defined hidden ground truth (the true issue) and a success check usable for
likelihood evaluation and endpoint scoring. If the released tasks don't support this
cleanly, STOP and report to Hanshal with specifics — do not silently invent task
structure (that is how Path A died). Deliverable: adapter + tests + 5-task smoke log
including answer-set coverage numbers.

**Step 0b — Mastermind harness test.** Full belief + 1-step EIG + 2-step lookahead
pipeline with hardcoded filtering (exact posterior, no LLM noise). Unit tests green
before any claims run.

**Step 1 — gap pilot (cheap; gates the claims).** 10 paired customer-service tasks,
arms: naive, 1-step EIG, full 2-step (26B A4B, per model defaults). Two independent
reads:
- Claim-1 check: 1-step > naive directionally. Expected to pass.
- Claim-2 check: 2-step > 1-step (≥6/10 or clear turns-to-resolution edge).
Claim-2 fail → descope to the claim-1 transfer study and CONTINUE (do not stop).
Claim-1 fail → STOP and discuss with Hanshal (this would contradict BED-LLM's core
result in a new setting — important, but no autonomous pivot).

**Step 2 — selective implementation + pilot.** Add tie/gating triggers; 10 paired
tasks, all arms. Checks: trigger rate non-degenerate (neither ~0% nor ~100%); selective
lookahead tokens ≤ ~40% of full 2-step; selective ≥ 1-step directionally. Calibrate ε
here, then freeze it. Use pilot effect sizes to power Step 3.

**Step 3 — pre-register, then headline runs.** Endpoints, analysis plan, frozen ε, and
canonical-run rule written into the runbook BEFORE launch. Customer service: 50–100
paired tasks (powered from Step 2), all arms. Then MediQ: 50+ paired cases, accuracy @
question budget, native Expert baselines included. Analyze exactly as registered.

**Step 4 — paper.** Arc: interactive agents must gather information under real task
structure → BED-LLM's greedy EIG is the right foundation but myopic where actions gate
actions → selective lookahead triggered exactly where the myopic decision is
statistically arbitrary → results per the playbook row, on two external benchmarks, at
bounded extra cost. Location-finding history = one honest paragraph ("when scaffolds
don't help: tasks where the LLM's native policy is near-oracle").

## OUTCOME PLAYBOOK (choose by table lookup on results day)

| Outcome | Paper |
|---|---|
| All three claims land on both envs | Full method paper: non-myopic BED scaffolding for interactive LLM agents on external benchmarks |
| Claims 1+2 land; selective captures little | "Lookahead helps interactive LLM agents" + efficiency-frontier analysis; trigger refinement as future work |
| Claim 1 only (2-step ≈ 1-step) | Transfer study: first BED-LLM evaluation on Paprika + MediQ, beating naive and native baselines; lookahead honestly reported as unnecessary on these tasks |
| Effects on customer service only (MediQ flat/infeasible) | Single-env version of the applicable row + Mastermind harness validation; MediQ honestly reported or dropped |
| Claim 1 fails | STOP. Talk to Hanshal. No autonomous pivot. |

Four of five rows are positive papers. No row requires environment engineering.

## POSITIONING / CLAIMS DISCIPLINE

- BED-LLM (arXiv:2508.21184): 1-step; our baseline and foundation — implement faithfully,
  compare respectfully.
- Paprika (arXiv:2502.17543): RL-trained curiosity; we are inference-time, training-free;
  natural teacher-policy story for future SFT/RL.
- MediQ (arXiv:2406.00922): benchmark + the naive-asking-hurts finding we build on.
- MeDxAgent (arXiv:2606.03416): agentic consultation with maintained hypotheses —
  hypothesis-tracking in medicine is NOT novel by itself; our medical claim rests on
  principled EIG + lookahead. Compare if their code is runnable.
- Uncertainty-Aware Clarification via IG (arXiv:2606.03135): greedy IG for clarification
  — the field stops at greedy; we test beyond it.
- "Shoot First, Ask Questions Later" (arXiv:2510.20886): independent evidence that
  1-step lookahead mitigates myopic question front-loading — cite as motivation.
- Learning to Ask (EMNLP 2024 Findings): EIG + preference optimization.
- DAD/RL-BED: trained, parametric spaces; we are inference-time, open NL spaces.
- Selective/adaptive search (B*-style): cite, never claim to invent selective expansion.
- Claims are sample-efficiency / accuracy-at-budget claims, paired, with CIs. No
  final-accuracy claims the curves don't support. No "first to do X" claims without a
  check against the positioning list above.

## DEFINITION OF DONE

1. Step 0 tests green; Step 1/2 gate artifacts in `results/path_e/`; Step 3
   pre-registration in the runbook before launch; every run in `EXPERIMENTS.md`
   traceable to a commit/tag.
2. Figures: paired endpoint curves (all arms, both envs); performance-vs-lookahead-cost
   frontier; trigger-rate-vs-round; one qualitative prerequisite-chain example where
   lookahead flips the decision and pays off.
3. 4–6 page draft in `paper/` following the applicable playbook row; paper/package/ledger
   validators pass; `STATE.md` updated to reflect completion.

## SCOPE DISCIPLINE

MPP = Steps 0–2 + customer-service headline + paper. MediQ is the first extension and
the only one. Depth cap 2. Do not touch location finding, 20Q/Wordle/Mastermind claims,
science-workflow envs, dollar costs, or new utilities/scoring modes. If any gate fails,
the response is the playbook — never an autonomous new path. If anything surprising
happens outside the playbook's rows, stop and ask Hanshal.
