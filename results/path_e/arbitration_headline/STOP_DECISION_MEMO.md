# Path E Stop Decision Memo

Date: 2026-07-12

## Decision Trigger

The frozen 50-task analysis did not confirm Claim B even before manual endpoint
adjudication:

| Comparison | Wins / losses / ties | Mean paired turn delta | 95% bootstrap CI |
|---|---:|---:|---:|
| arbitration vs thinking naive | 6 / 16 / 28 | +0.44 | [0.0, 0.9] |
| arbitration vs candidate 0 | 6 / 12 / 32 | +0.16 | [-0.2405, 0.56] |
| best-N EIG vs thinking naive | 11 / 12 / 27 | +0.28 | [-0.14, 0.74] |

Positive deltas mean the belief-guided method took more censored turns. Resolution at
five turns was 32% for arbitration and best-N EIG, 40% for thinking naive and candidate
0, and 38% for non-thinking naive.

The mandatory manual audit then found a fatal endpoint contradiction on task 13. The
private remedy says the receipt printer is out of paper and replacing/refilling the
roll restores printing. Best-N instructed replacement of the current paper roll, but
the customer simulator replied that changing it did not help. The preregistered rule
classifies this exact-remedy false failure as invalidating the complete headline.

## What The Evidence Supports

- The endpoint-repaired adapter passed its small smoke but did not guarantee semantic
  faithfulness at headline scale.
- Goal-anchored proposal elicitation helped in a ten-task development probe, but that
  signal did not replicate on the held-out tasks.
- Neither calibrated arbitration nor ordinary best-N EIG improved the held-out policy
  endpoint relative to thinking naive.
- The five-arm headline cannot support even a clean null policy comparison because its
  simulator endpoint failed the frozen semantic audit.
- MediQ is not authorized by the frozen gate.

## Options

### 1. Close Path E As A Method-Claims Project (Recommended)

Archive the Paprika work as an invalid endpoint diagnostic and retain the already
validated Path A package as the completed workshop result. Do not spend the remaining
OpenRouter authorization on another Paprika or MediQ run.

Consequences:

- no new model cost;
- strongest claims discipline;
- the current Path E paper is retired rather than submitted;
- Paprika findings can appear only as a short future-work/evaluation-caution note in a
  broader project report, not as policy evidence.

### 2. Authorize A Negative Endpoint-Validity Paper

Reframe the paper around a different question: whether strict automated private-remedy
checks are sufficient to validate LLM-simulated interactive-agent benchmarks. Complete
the remaining 33-task manual audit, quantify false-success and false-failure modes, and
report the frozen policy statistics only as invalidated diagnostics.

Consequences:

- no new LLM calls are required for the first complete draft;
- this is a new paper claim and therefore needs explicit authorization;
- the title, abstract, contributions, results, discussion, validator, and figures must
  be rewritten;
- it does not establish that BED, arbitration, or lookahead improves troubleshooting.

### 3. Preregister A New Simulator/Evaluation Study

Replace free-generating remedy responses with a more constrained benchmark-faithful
execution protocol, rerun validation from smoke through held-out evaluation, and treat
all current headline artifacts as development-only.

Consequences:

- this is a new experimental path, not a recovery;
- it requires a new preregistration and fresh held-out task plan;
- the current automated effect points against the method, so a cleaner rerun is more
  likely to establish a valid null than a positive Claim B;
- no launch should occur without an explicit scope and budget decision.

## Paper Salvage Map

The current `paper/main.tex` is a pre-result method draft and is not submission-ready.

Retainable material:

- related work;
- the external Paprika task definition;
- belief, answer-space, and arbitration mechanics;
- the preregistered pairing, recovery, and audit protocol;
- the development-to-held-out chronology, clearly labeled diagnostic.

Material that must change or be removed under any publishable path:

- the sealed-outcome abstract and Results Status section;
- positive method-facing contribution language;
- the conclusion framing the held-out result as still open;
- the validator's `sealed_headline` requirement;
- any figure or table that presents the invalid headline as policy evidence.

## Recommendation

Choose option 1 unless a paper specifically about interactive-benchmark endpoint
validity is strategically valuable. Do not choose option 3 to rescue the method claim:
the frozen effect already points in the wrong direction, independently of the fatal
simulator contradiction.
