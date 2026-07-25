# tau-Knowledge GPT-5.4 Mini Scorer Replication Result

## Decision

The serving interface passed, but the frozen efficacy smoke failed. The
140-call confirmation was not run. This is negative evidence for cross-size
scorer transfer on the two frozen smoke trees, not a second held-out policy
result.

## Serving Results

- Physical requests: exact 14/14.
- Reasoning tokens: 0.
- Forced exits: 0.
- Parsed myopic roots: 2/2.
- Parsed full-tree roots: 2/2.
- Parsed focused roots: 10/10.
- Focused roots with nonconstant scores: at least 8/10, passed.
- Focused pairwise accuracy: `.2647` versus required `.55`, failed.
- Oracle-optimal focused selections: 6/10 versus required 7/10, failed.
- Focused mean regret: `.40`.
- Cost: `$0.04375125`.

The Mini scorer often assigned count bands inconsistently with the exact
required-document values. For example, on one root it gave its largest score to
a zero-value continuation while assigning zero to a one-document continuation.
This is an efficacy failure, not a parser, budget, or serving failure.

## Interpretation

GPT-5.4 Mini can consume the long root and focused inputs under explicit
non-reasoning and return every required object, but it does not meet the frozen
semantic continuation-ranking threshold. No prompt simplification, threshold
change, response repair, smoke replacement, or confirmation run followed.

The result narrows the supported claim: GPT-5.4 semantic scoring is
load-bearing relative to deterministic retrieval heuristics, but transfer to a
smaller same-family scorer is not established. Claude Sonnet 5 and Gemini 3.1
Pro were blocked at serving, so cross-family efficacy also remains unmeasured.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_GPT54MINI_SCORER_REPLICATION_PREREGISTRATION.md`
- Smoke:
  `results/nonmyopic/tau_knowledge_gpt54mini_scorer_smoke/tau-knowledge-gpt54mini-scorer-smoke-20260725T021848Z/SERVING_SMOKE.json`
- Private raw responses: stored outside git.
