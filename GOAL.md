# GOAL (Endgame): Non-myopic sequential Bayesian experimental design with LLMs — the target-decode contract, the evidence, and the paper

## HOW TO USE THIS DOCUMENT

Immutable during the run; all living state in `STATE.md` (read first every session,
update every session, `EXPERIMENTS.md` row for every launch/spend). Precedence:
`STATE.md` > this file. Backend: OpenRouter, `google/gemma-4-26b-a4b-it` for all
questioner/policy roles; budget $30 total authorized, project every run before launch;
cluster (`msc,llm --exclude=oat12`, ≤8 jobs) if it returns. Never commit the API key.

## PROJECT IDENTITY (fixed, per Hanshal)

The subject is **non-myopic sequential BED with LLMs, with EIG as the acquisition
objective**. The refined contract that governs environment choice:

> A task is in-contract iff it exposes a latent target with ground truth and its score
> is a monotone readout of posterior quality via an explicit DECODE rule (BED-LLM's
> guess protocol). Acquisition maximizes EIG about the target; a separate belief→action
> rule (argmax guess at confidence τ or at budget) produces the scored output. Actions
> that resolve the task are decodes, never EIG candidates.

Out-of-contract tasks (score rewards episode termination or action success, e.g.
Paprika resolution) are BOUNDARY evidence, not method environments.

## EVIDENCE INVENTORY (banked — the paper is assembled from these)

1. **In-contract, non-myopic (pending re-verification — Track 1):** banked 20Q/animals
   runs: 1-step EIG ≫ naive; 2-step > 1-step in rounds 2–11; 3-step degrades.
   `results/cluster-gemma4b-nstep-comparison/`, `results/gemma4-4b-categorical-1step-2step/`,
   `results/cluster-gemma4b-3step-eig-*`.
2. **Boundary (complete):** the Paprika program — validated adapter, three endpoint
   audits, faithful 1-step EIG 30% vs naive 40%, full 2-step 20.8× cost for worse
   outcomes, arbitration pilot 6/0/4 reversing at N=50 (32% vs candidate-0 40%), and
   the mechanistic autopsy (EIG ranks the better policy's action first 61/222; prefers
   diagnostics over cures; schema discarded decisive replies; no prerequisite
   structure; endpoint independently invalid).
   `results/path_e/arbitration_headline/NON_MYOPIC_BED_FAILURE_ANALYSIS.md`.
3. **Infrastructure boundary (complete unless the retry passes):** iMEDQA audit +
   iCRAFT profile-gate failure — 26B cannot author calibrated patient profiles, so the
   simulator, not the policy, is the binding constraint on medical LLM-BED.
   `results/path_e/TRACK2_COMPLETION_AUDIT_AND_DECISION.md`,
   `results/path_e/icraft_profile_gates/FINAL_REPORT.md`.
4. **Method-design principles (established, citable):** experiment/decode separation;
   target choice over finite variables; calibrated categorical likelihoods; the
   aleatoric ceiling on per-instance plan ranking (Path B/C artifacts); winner's-curse
   pilot reversal as a case study in pre-registration.

## REMAINING WORK (three tracks — order enforced)

**Track 1 (BLOCKING, free): banked animals re-analysis.** Comparability check across
the banked runs, then Q@80% and accuracy-AUC with bootstrap CIs (paired where
possible). Deliverable `results/path_e/ANIMALS_REANALYSIS.md`. HOLDS → it is the
paper's non-myopic spine. COLLAPSES (configs incomparable or effect vanishes) → the
non-myopic claim is presented as open, with the gate methodology as the contribution.
No other experimental work until this exists.

**Track 2 (bounded, conditional): iCRAFT apparatus retry + gate chain.** One retry of
the profile gate with a stronger OpenRouter model FOR PROFILE GENERATION ONLY
(questioner stays 26B; simulator fidelity and agent capability are separate roles).
Cap $2, one attempt, model + rationale ledgered. Pass → continue the pre-registered
chain unchanged: calibration → likelihood validity → ranking fidelity → oracle
greedy-vs-depth-2 gap → ONLY THEN policy runs (claim 1: 1-step EIG vs naive asking vs
native baseline; claim 2: depth, gated on the oracle gap; caps pre-registered).
Fail → CLOSE the external claim permanently; the finding joins evidence item 3.
Fences: no profile/prompt tuning toward lookahead structure; no third attempt; no new
environments; no Paprika reruns; no Bayes-adaptive build (parked as the conference
follow-up).

**Track 3 (parallel, always on): the paper.** 4–6 pages:

- Problem setting: the target-decode contract (identity section above, formalized).
- In-contract evidence: animals re-analysis (+ iCRAFT results if Track 2 unlocks).
- The non-myopic question: where lookahead paid (animals rounds 2–11), where it
  provably cannot (oracle gap, if measured), the cost frontier (20.8× full-2-step).
- Boundary of applicability: the Paprika autopsy, quantified — the paper's likely
  most-cited section. Plus the infrastructure boundary (simulator calibration).
- Methods appendix: endpoint-validation discipline (three audits, regression
  batteries, quarantine protocol) — reusable methodology, presented as such.
- Limitations: single model family, workshop-scale n, banked-data caveats as found
  by Track 1.

## OUTCOME PLAYBOOK

| State | Paper |
|---|---|
| Track 1 holds + Track 2 unlocks and claims land | Full result: non-myopic EIG with LLMs on two in-contract families + boundary study |
| Track 1 holds + Track 2 closed | Non-myopic EIG demonstrated on 20Q-family; external in-contract envs shown infrastructure-bound; boundary study |
| Track 1 collapses + Track 2 unlocks | iCRAFT carries in-contract evidence; animals reported honestly as unverifiable |
| Track 1 collapses + Track 2 closed | The contract + boundary + gate-methodology paper; non-myopia framed as open, with the measurement framework as the contribution |

All four rows are submitted papers. No row authorizes new environments or methods.

## POSITIONING (unchanged obligations)

BED-LLM (foundation + the guess protocol; our claims 1 are its evaluation on new
ground); DAD/RL-BED (trained, parametric); Paprika (RL-trained curiosity; boundary
env); MediQ (benchmark + naive-asking-hurts); MeDxAgent; Adaptive Elicitation
(arXiv:2504.04204 — base-LLM simulators harm planning; corroborates our simulator
findings); goal-driven BOED (arXiv:2605.26093); A-/D-optimality classicism; selective
search cited not claimed. Claims discipline: paired stats with CIs; no "first to X"
without checking this list; the Paprika negative stated as a contract violation, not a
method defect.

## DEFINITION OF DONE

1. `ANIMALS_REANALYSIS.md` exists with its verdict; Track 2 resolved (unlocked-and-run
   or permanently closed) with all gates ledgered.
2. Paper draft complete per the applicable playbook row; paper/package/ledger
   validators pass; every cited number traceable to a commit/tag via `EXPERIMENTS.md`.
3. Venue: when the NeurIPS 2026 workshop list drops (author notifications mandated
   Sept 29; expect submissions late Aug–early Sep), pick 2–3 targets and tune the
   abstract per venue (BED/probabilistic-methods venue → contract + gates emphasis;
   agents venue → boundary/autopsy emphasis).
4. `STATE.md` updated to submitted/awaiting.

## SCOPE DISCIPLINE

The MPP is Track 1 + Track 3, with Track 2 as the only experimental extension. Depth
cap 2 everywhere. No new environments, methods, utilities, or endpoint-repair
campaigns. Every gate outcome routes through the playbook; anything outside the
playbook's rows is a stop-and-ask-Hanshal. The project ends with a submission, not
with a sixth path.
