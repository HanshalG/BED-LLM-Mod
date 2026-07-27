# CuriosiTree and YIELD Source Audit Result

## Decision

Neither release supports an honest non-myopic LLM-native BED experiment without
inventing the missing transition model. No paid run is authorized on either
source.

## CuriosiTree

- Official repository:
  `https://github.com/cooper-mj-research/CuriosiTree`
- Audited commit:
  `7b31474f2d353527b4262c56b3c5d35185d1cf7e`

The clinical task has a useful surface form: an LLM proposes diagnoses,
questions, laboratory tests, retrievals, and semantic compatibility judgments.
The released implementation does not provide the counterfactual environment
needed for depth attribution:

- each node scores one action by one-step expected support reduction;
- the medical respondent is an unseeded Llama prompt conditioned only on a
  diagnosis name;
- no patient record, exact response table, simulator snapshot, or paired trace
  replay is released; and
- the endpoint compares the model's diagnosis string with the same supplied
  diagnosis label.

Extending this to depth two would compound two newly sampled LLMs rather than
evaluate a deployment-matched patient process.

## YIELD

- Official repository: `https://github.com/infosenselab/yield`
- Audited commit:
  `4700a279fabfe677945bf46ef22e67a19c5c156e`

YIELD releases 2,281 real human-to-human interviews, training code, and metrics
for factual novelty, progression, conformity, and turn length. Each record is
one completed transcript. The release does not include:

- a latent respondent fact state;
- alternative responses to unasked questions;
- a counterfactual respondent or replay model; or
- an exact terminal fact-recovery endpoint.

The POMDP formulation is therefore conceptual for the static corpus. Branching
from a transcript prefix would require a new generative respondent whose
counterfactuals are not validated against the original person.

## Implication

Both sources are relevant literature and possible training resources.
CuriosiTree demonstrates LLM-generated semantic actions and likelihood checks;
YIELD demonstrates realistic elicitation language. Neither can currently
support paired myopic/non-myopic policy evaluation. The next route must preserve
their semantic richness while using a released response table or deterministic
simulator.

## Cost

OpenRouter calls: `0`; cost: `$0`; OatML/cluster use: none.
