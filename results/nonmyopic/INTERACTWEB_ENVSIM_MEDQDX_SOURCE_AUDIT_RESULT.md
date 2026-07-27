# InteractWeb-Bench, EnvSimBench, and MedQDx Source Audit

Audited: 2026-07-27

This was a zero-call source audit. It used released repositories and benchmark
artifacts only. It did not use OpenRouter or OatML resources and authorizes no
paid experiment.

## Frozen Sources

| Candidate | Public source | Pinned state |
| --- | --- | --- |
| InteractWeb-Bench | `https://github.com/AIforIP/InteractWeb-Bench` | commit `56040f368e9b41e770fc814d17a7c207c6be7bd9` |
| EnvSimBench | `https://github.com/cookieApril/EnvSimBench` | commit `938c5ce23793edb00db4e315d4e2fe432a2c71d0` |
| MedQDx | `https://github.com/MaiWert/MedQDx` | commit `caec1794c4bdbb6fee46e506cc820a01263dc2d8` |

## InteractWeb-Bench

The release contains 404 rows: 101 base website tasks crossed with four user
personas. Its user simulator is history-conditioned and can reveal hidden
requirements in response to semantically matching clarification questions.
Each base task, however, has the same ground-truth website requirements across
all four persona variants. The release does not define a prior over mutually
exclusive requirement worlds.

The interaction protocol also does not prohibit a broad request for all missing
requirements. Its 15, 20, and 25 turn limits combine clarification, coding,
verification, and submission rather than imposing an information-acquisition
budget. The final website score uses a WebVoyager-style VLM/LLM evaluator, so it
is not an exact programmatic endpoint.

Creating a BED benchmark would require inventing alternative latent requirement
worlds, a prior over them, an atomic query budget, and a new endpoint. Those are
the central scientific objects, not a thin wrapper around the released task.

**Decision:** close direct benchmark use. A derived construction is not
authorized without a separate zero-call manifest proving source-grounded
alternative worlds, nontrivial query constraints, and an exact target-blind
endpoint.

## EnvSimBench

The released benchmark has 400 transition-prediction samples over 167
environments with exact programmatic reference outputs. Each item gives the
environment code, current state, and action, and asks the model to predict the
next observation and state.

This is a one-step, fully observed environment-simulation benchmark. It has no
hidden world sampled from a prior, no adaptive information-gathering action,
and no delayed value of information. Chaining rows would invent rather than
recover a sequential BED process.

**Decision:** close direct use. Retain it only as related evidence that LLMs can
simulate heterogeneous environment code.

## MedQDx

The repository describes an upstream symptom-disease table with 132 binary
symptoms and 41 disease labels. Its released `Patient Cases.csv` contains 100
generated cases spanning 29 observed diseases, and its completed
`MedQDx_Benchmark.csv` contains 99 rows with three fixed question, answer, and
diagnosis rounds.

The notebook does include an online LLM patient. It answers arbitrary questions
from the row's `100% Case`, while the doctor initially sees the `50% Case`.
However, the full case was itself generated directly from the row's published
binary symptom list. The prompt instructs the patient to answer negatively when
a queried detail is absent. There is therefore no hidden semantic patient state
beyond the explicit symptom vector and disease label.

A classical policy can enumerate the released disease-symptom support, compute
the response relation, and own the belief update. The LLM patient contributes
natural-language paraphrase and possible response noise, but not irreducible
hypothesis support or likelihood dynamics. The released benchmark also stores
only one previously generated three-turn trajectory per case.

**Decision:** retain as a possible diagnostic-dialogue supporting benchmark,
but close it for the headline LLM-native result. Do not spend on it unless a
future release adds source-grounded latent clinical state not recoverable from
the symptom table.

## Budget Effect

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML jobs: `0`
- Paid gate unlocked: no
