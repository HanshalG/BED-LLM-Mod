# Venue Submission Plan

Last checked: 2026-07-14. This is a submission-framing document only. It does
not alter the frozen evidence, add experiments, or broaden the paper's claims.

## Ranked targets

| Rank | Venue | Why it fits this paper | Format and deadline | Submission framing |
| --- | --- | --- | --- | --- |
| 1 | Verification in the Age of AI Scientists (NeurIPS 2026) | Direct match: the workshop asks when an imperfect simulator can justify action and how scarce verification should be allocated. The paper supplies a concrete audit of exactly this problem for LLM information-gathering agents. | 4-8 pages excluding references; abstract tab by Aug. 22 and paper by Aug. 25, AoE. | Submit the current validation-first paper with the verification-centered title and abstract below. |
| 2 | Scaling Environments for Agents (SEA, NeurIPS 2026) | Direct systems/benchmark fit: its CFP names environment design, benchmark construction, evaluation robustness, learned environment fidelity, and transfer. The paper turns those concerns into a validation contract for simulated LLM rollouts. | Short papers up to 4 pages; long papers up to 9 pages; Aug. 29, AoE. | Submit as a long paper, preserving the present 5-page technical evidence and emphasizing environment-model/deployment equivalence. |
| 3 | Science for Artificial Intelligence (Sci-fAI, NeurIPS 2026) | A good position/methodology fit: the CFP explicitly welcomes evaluation standards, epistemology, and position pieces about valid measurements of AI systems. | Extended abstracts up to 4 pages excluding references; Aug. 29, AoE. | Condense to a four-page position-plus-evidence version, foregrounding falsifiable validation gates rather than the full environment chronology. |

Reserve only: AutoResearch 2026 is thematically adjacent through evaluation of AI
scientists, but its stated center is end-to-end autonomous scientific research
and robot scientists. It is a weaker fit than the three targets above.

The CFPs are non-archival. Sci-fAI explicitly permits concurrent submissions;
the other venue policies must be rechecked in their OpenReview forms before
submitting the same manuscript to more than one venue.

## Venue-specific titles and abstracts

### 1. Verification in the Age of AI Scientists

**Title:** When Can an AI Scientist Trust Its Simulator? A Validation Ladder for
Non-Myopic LLM Information Gathering

**Abstract:**

AI agents increasingly use learned or language-model-derived simulators to
choose which experiment, question, or tool call to make next. A deeper rollout
is useful only when that simulator is a credible verifier of the information it
predicts. We audit non-myopic Bayesian experimental design (BED) for LLM agents
across an exact location task, semantic 20 Questions, and two external
interactive benchmarks. Depth-two planning succeeds in the exact positive
control, but the incremental depth effect does not survive paired evaluation in
the semantic task. In Paprika, the planned objective and endpoint diverge. In
MediQ, a strict simulator audit passes while held-out likelihood calibration
fails because the nominal latent state is insufficient for the observed
responses. A bounded repair also fails to construct adequate latent support.
These cases show that planning depth is downstream of verification: before
allocating rollout budget, an AI scientist needs a deployment-matched latent
state, calibrated response likelihoods, branch/deployment equivalence, a
verified greedy gap, and ranking fidelity. We present this validation ladder as
an actionable contract for deciding when an imperfect simulator is sufficient
to guide sequential scientific information gathering.

### 2. Scaling Environments for Agents

**Title:** Before Deeper Rollouts: Validating Environment Models for
Non-Myopic LLM Agents

**Abstract:**

Long-horizon agent evaluation often treats a simulator as an interchangeable
source of rollout outcomes. For information-gathering agents, that assumption
is especially consequential: non-myopic Bayesian experimental design selects a
query by optimizing beliefs produced by the simulator itself. We audit this
setup across an exact constrained-location task, semantic 20 Questions, and
two external interactive benchmarks. Exact depth-two planning improves final
RMSE when the delayed-information structure and response model are controlled.
In less controlled settings, however, larger horizons can optimize simulator
errors rather than deployment-relevant information. We identify two distinct
failures: a mismatch between the planned objective and the evaluation endpoint,
and a response model whose target state is insufficient to generate deployed
observations. We provide an auditable validation contract spanning target and
latent-state sufficiency, likelihood calibration, branch/deployment equivalence,
a measurable greedy gap, and rank fidelity. The result is a practical standard
for environment design and benchmark evaluation: validate the rollout model
before attributing performance changes to agent planning horizon.

### 3. Science for Artificial Intelligence

**Title:** What Does a Planning Depth Measure? A Validation Ladder for
Non-Myopic LLM Agents

**Abstract:**

Claims about agentic planning depth are empirical claims about an AI system and
therefore require a valid measurement model. We study non-myopic Bayesian
experimental design for language-model agents, where a policy selects questions
by simulating future observations under an LLM-derived probabilistic model.
Across an exact positive control, semantic 20 Questions, and two external
interactive benchmarks, the same recursion produces three qualitatively
different outcomes: a real depth-two gain under an exact delayed-information
trap, a non-replicating incremental effect under semantic hypotheses, and
failures caused by endpoint mismatch or an insufficient latent state. The
pattern is not evidence that lookahead is intrinsically ineffective. It shows
that a measured depth effect is meaningful only after the model used to measure
it has passed falsifiable checks. We formulate a validation ladder: sufficient
target and latent state, calibrated likelihoods, matched simulated and deployed
updates, a verified non-greedy structure, and fidelity of the final policy
ranking. This turns a vague demand for better agents into concrete measurement
obligations for a science of AI.

## Preparation checklist

1. Recheck each OpenReview link, anonymity rule, and dual-submission rule on the
   day of submission; the public pages describe several dates as tentative.
2. Produce one anonymized NeurIPS-format source tree per target. Do not alter
   the shared evidence table or results claims between variants.
3. For the Verification and SEA versions, preserve the current validation chain,
   exact positive control, paired animals re-analysis, and the two external
   failure modes.
4. For Sci-fAI, remove operational chronology and retain one compact table or
   figure plus the validation ladder so the paper reaches four pages excluding
   references.
5. Treat all workshop submissions as non-archival presentations and retain the
   frozen repository package as the provenance record.

## Sources

- Verification in the Age of AI Scientists: https://ai4sciencecommunity.github.io/neurips26.html
- Scaling Environments for Agents: https://sea-workshop.github.io/
- Science for Artificial Intelligence: https://sci-fai-workshop.github.io/
- AutoResearch 2026: https://autoresearch2026.github.io/
