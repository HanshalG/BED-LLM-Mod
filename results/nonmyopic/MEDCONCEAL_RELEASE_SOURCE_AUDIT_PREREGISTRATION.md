# MedConceal Release Source Audit Preregistration

Date: 2026-07-28

## Why This Is A New Source State

The MedConceal paper describes an unusually close match to the target problem:
free-form clinical dialogue with simulator-internal hidden concerns, stateful
revelation and addressing criteria, and history-dependent patient behavior. Earlier
searches found only the paper and human-study interface. An official public repository
appeared on 2026-07-25:

- repository: `https://github.com/FAIRHealth/MedConceal`;
- observed `main`/`HEAD`: `f98d02c1eb9819325f091c0afd0dc4d63d90a21b`;
- commit message: `Initialize MedConceal repo`.

Repository metadata has been observed, but no tracked file, case value, hidden concern,
prompt, response, or simulator implementation has yet been opened.

## Scientific Admission Question

MedConceal is admissible as a headline LLM-native non-myopic BED route only if the
official release supplies enough material to reproduce its interactive hidden-state
process without inventing benchmark semantics. All of the following must hold:

1. the release identifies its code and data licensing boundaries;
2. released cases can be enumerated and deterministically partitioned before their
   visible context, hidden concerns, or target plans are inspected;
3. the released runtime can instantiate a patient interaction from a case and exposes
   an external hidden-state or success endpoint for evaluation;
4. the clinician never receives the hidden concerns or target endpoint;
5. free-form clinician language is interpreted semantically rather than selected from
   a complete finite action table;
6. patient observations or hidden-state transitions depend on dialogue history, not
   only on a fixed current action/hidden-state lookup;
7. the LLM is load-bearing in at least one of semantic patient response generation,
   concern revelation/addressing likelihoods, or path-dependent belief/support
   generation;
8. paired myopic and non-myopic policies can be evaluated against the same patient
   randomness and endpoint without exposing hidden labels to either policy.

The paper alone is not an executable specification. Missing cases, prompts, state
transition code, evaluator code, or model/version settings must not be reconstructed
from prose.

## Access Boundary

The audit will:

1. clone only commit `f98d02c1eb9819325f091c0afd0dc4d63d90a21b`;
2. record the complete tracked-file manifest and repository hash;
3. inspect licenses, README/data cards, schemas, dependency files, and runtime code;
4. inspect case identifiers and metadata fields only, without emitting case text,
   hidden concerns, target plans, or recorded conversations;
5. freeze a deterministic case partition before any task values are opened.

If at least `300` unique case IDs are available, order them by
`SHA256("medconceal-24421:" + case_id)` and assign:

- first `20`: mechanics;
- next `80`: development;
- next `100`: confirmation;
- all remaining cases: retained.

If the release has fewer than `300` unique cases, duplicate IDs, or no separable
metadata/value boundary, the source gate fails. No substitution from paper examples,
recorded clinician interactions, scraped health discussions, or third-party mirrors is
allowed.

## Zero-Call Structural Gate

After the manifest is frozen, only mechanics values may be opened. No OpenRouter call
is authorized by this document. The structural audit must report:

- the exact released case, simulator, prompt, state, and evaluator artifacts;
- whether all paper-reported model settings are specified;
- what is stochastic and whether common random numbers are implementable;
- whether revelation and addressing are evaluated by executable code or only by
  unreleased judgments;
- whether dialogue history can change the distribution of the next response or state
  for the same current utterance;
- whether the terminal endpoint is independent of the planning score;
- whether a complete finite classical action/observation table could replace the
  semantic model without changing the released task.

The route passes only if all eight admission conditions hold on official artifacts.
Any missing load-bearing component closes the exact release before model use.

## Conditional Paid Work

A full source pass authorizes only a separate response-blind experiment
preregistration. That successor should first run a cheap paired mechanics test of:

- a non-reasoning LLM-native myopic policy;
- a compute-matched non-myopic policy planning over complete dialogue continuations;
- a random or fixed-question control;
- naive thinking as a distinct baseline, not the engine of the BED policies.

Primary outcomes should be hidden-concern confirmation/intervention success,
truth-anchored concern coverage, and turn-normalized success rather than self-reported
planner scores. Development and confirmation remain sealed until their preceding
gates pass.

There is no protected OpenRouter reserve. The full available balance may be allocated
by expected scientific value after validity gates. OpenRouter is the only execution
backend; OatML, Slurm, and SSH are prohibited.
