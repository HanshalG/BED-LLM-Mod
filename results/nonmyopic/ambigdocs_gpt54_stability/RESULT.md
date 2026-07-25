# AmbigDocs GPT-5.4 Stability Result

## Outcome

The final target-free AmbigDocs serving gate failed the semantic opportunity
conjunction after clean, stable serving.

- Run: `ambigdocs-gpt54-stability-20260725T142236Z`
- Requests / HTTP attempts: `10 / 10`
- Reasoning tokens / retries / forced exits: `0 / 0 / 0`
- Cost: `$0.028235`
- Four unique questions and all six-character maps parsed.
- All four base partitions were informative.
- Three identical classification requests returned the exact same map: `NNYNNN`.
- Only two unique base partitions were produced (required at least three).
- EIG range was `0` nats (required at least `.10`).
- No hidden target, responder, policy score, or endpoint was accessed.

## Mechanism

All four questions ask variants of the same one-versus-rest distinction: whether
“Minsk” means the city. Three map only the city document to yes (`NNYNNN`); the fourth
maps it unknown (`NNUNNN`). GPT-5.4 therefore repairs Mini's likelihood instability
but not the semantic action collapse.

This means the exact action bank has no useful score range for comparing myopic and
non-myopic roots. Proceeding to a target endpoint would test a cherry-picked or
manually diversified bank rather than the preregistered LLM-native generator.

## Decision

AmbigDocs closes with no efficacy run and the official test split sealed. The result
localizes the bottleneck:

- external support preservation: works;
- exact serving: works with GPT-5.4;
- repeated likelihood stability: works with GPT-5.4;
- autonomous semantic root diversity: fails.

No prompt diversification, manual action injection, threshold change, target sampling,
or stronger-model rerun is allowed. OatML was not used.
