# Detective Cases Open-Ended Source Audit Result

## Verdict

The proposed open-ended/categorical revisit does **not** pass a zero-cost
hidden-environment gate. Richer answer text cannot create coherent alternative
murderer worlds that the released CA-BED data does not specify. No OpenRouter
call is authorized.

## Released Structure

- Source:
  `external/ca-bed/src/ca_bed/tasks/detective_cases/DetectiveCases.json`
- SHA-256:
  `049ea3003753b15e3319483d15993591b5d50ac7eedb3187dca6ef3951cd2a57`
- Cases: 100
- Suspects per case: exactly 4
- Canonical murderers per case: exactly 1
- Story records per suspect: exactly 1
- Counterfactual stories or response maps per alternate murderer: 0

Each suspect's sole private `story` is written for the released canonical role.
The murderer story contains the canonical crime and concealment; innocent
stories contain their canonical movements and observations. The public case
context already exposes each suspect's `testimony`.

## Why Open-Ended Answers Do Not Repair The Task

To score BED over four murderer hypotheses, the environment must define how
each suspect responds under every possible murderer world. The release instead
defines one realized story per suspect. Asking an LLM to reuse that story while
changing only a role instruction produces contradictory counterfactuals: the
canonical murderer's story still describes the crime when instructed to be
innocent, while another suspect has no alternate story describing how they
committed it.

This is the same source-level mismatch isolated by the prior aligned-likelihood
diagnostic:

- direct textual likelihood invented rich but wrong counterfactual behavior;
- the deployment-matched role-only response function restored truth ranking;
- only 23/241 questions (9.5%) then distinguished murderer from innocent.

Open-ended answers would increase output dimensionality but still depend on the
same missing counterfactual worlds. Treating free-form model generations as new
world records would invent the benchmark environment and make planner and
endpoint share a self-generated simulator.

## Decision

Close the Detective Cases revisit, including categorical/open-ended
interrogation, unless CA-BED releases counterfactual suspect stories or a
world-conditioned response model. The 24 cases reserved by the earlier
diagnostic remain unused.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
