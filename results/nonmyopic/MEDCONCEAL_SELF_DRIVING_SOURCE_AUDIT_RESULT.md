# MedConceal and Self-Driving Negotiator Source Audit

Audited: 2026-07-27

This was a zero-call source audit. It used public repositories and environment
artifacts only. It did not use OpenRouter or OatML resources and authorizes no
paid experiment.

## Frozen Sources

| Candidate | Public source | Pinned state |
| --- | --- | --- |
| MedConceal | `https://github.com/FAIRHealth/MedConceal` | commit `f98d02c1eb9819325f091c0afd0dc4d63d90a21b` |
| Self-Driving Negotiator | `https://app.primeintellect.ai/dashboard/environments/ashu1069/self-driving-negotiator` | version `0.4.0`, version id `ezi36zqgegr39xk51e6ej5ta`, content hash `ff1cd70115fedcfe4d795ce0269b579766a617566b0ed74efd9ed2b433ace496`, source SHA-256 `23fc9433a8199fcdaea083a7541586937c182bcda7ee8bffb0a32c9ee80c7eca` |

## MedConceal

The public release includes 300 hidden-concern cases, human and model dialogue
traces, per-case metrics, and evaluation scripts. It does not include the
patient simulator. The repository README explicitly says the simulator will be
released after paper publication, and the published traces remove latent policy
weights and hidden-state transitions.

The logged conversations therefore cannot answer counterfactual questions under
the same hidden patient state. Reconstructing a simulator from the traces would
invent the action-conditioned transition and likelihood model, which is exactly
the object a sequential BED result must evaluate rather than assume.

**Decision:** close direct use until the official patient simulator is released.
Keep it in the release-retry queue because hidden concerns, natural dialogue,
and exact reveal states are an unusually good LLM-native fit once the transition
model becomes available.

## Self-Driving Negotiator

The Prime Intellect release is complete and reproducible. It contains seeded
simulators, source, tests, three scenarios, hard information-structure tiers,
privileged hidden dispositions, exact terminal outcomes, and non-LLM baselines.
In `yield_standoff`, later tiers delay the intent cue and optionally bluff before
reverting to the true disposition. Ego speed therefore controls when evidence
arrives and how much reaction time remains.

This is a real non-myopic POMDP opportunity, but it does not satisfy the headline
LLM-native criterion:

- the hidden disposition and bluff state form a tiny source-defined latent space;
- public observations are deterministic functions of numeric kinematics plus
  optional Gaussian noise;
- the action grammar is a small maneuver set plus continuous acceleration;
- the bundled scripted expert already solves the relevant creep, observe, and
  commit logic.

An exact classical planner can enumerate the latent states and simulate every
action without semantic hypothesis generation, semantic likelihoods, or an
LLM-owned belief transition. An LLM policy would be a language interface to a
small control problem, not the irreducible belief machinery.

**Decision:** retain as a possible supporting exact-control benchmark, but do
not spend on it and do not present it as the primary non-myopic LLM-BED result.

## Budget Effect

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML jobs: `0`
- Paid gate unlocked: no
