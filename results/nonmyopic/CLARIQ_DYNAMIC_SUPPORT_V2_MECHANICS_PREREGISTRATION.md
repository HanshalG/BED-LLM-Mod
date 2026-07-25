# ClariQ Dynamic-Support V2 Full Mechanics Preregistration

## Authorization and Boundary

The fresh V2 one-call serving gate passed every frozen condition. Its response
is discarded. This run generates a new complete tree on already-disclosed
mechanics topic `60`; topic `60` cannot support a generalization claim.

Development topic `148` and holdouts `102/11/103/141` remain untouched. No
development run is authorized unless every mechanics gate below passes.

Manifest SHA:
`fa5a34e55ab455359a4a64bd2aba00ea5789f27fb2f2d932ea9e2330cf90ca03`.

## Frozen Tree

- Model: `openai/gpt-5.4`, OpenRouter, non-reasoning, temperature `.7`.
- One fresh initial eight-hypothesis support.
- One independent regenerated support for each of 90 root-answer branches.
- Exactly 91 model requests and HTTP attempts.
- Exact single-space response-code grammar from the passed V2 serving gate.
- Branch batch concurrency `90`.
- Raw initial and branch responses checkpoint before strict parsing.
- No repair, normalization of malformed output, response reissue, retry,
  substitution, or reuse of the serving response.

## Policies and Endpoint

All policies use the same fresh tree:

- myopic initial-support EIG;
- fixed-support depth two;
- branch-regenerated dynamic-support depth two;
- within-root answer-link-shuffled dynamic support; and
- seeded random.

Every support, score, continuation, and selected root freezes before loading the
already-open exact ClariQ oracle-tail NDCG@20 values. The first-link endpoint is
the exact terminal NDCG of each selected root. Initial and branch exact hidden
response-profile coverage are diagnostic only.

## Conjunctive Gates

Serving and dynamics:

- exactly 91 physical requests and HTTP attempts;
- all 91 supports parse;
- zero retries, reasoning tokens, and forced exits;
- at least 80/90 branch intent supports differ from the initial intent support;
- at least 75/90 branches contain at least two positive-mass response profiles;
- at least 75/90 branches have positive best continuation EIG;
- myopic score range and dynamic future-value range are each at least `.05`
  nats;
- at least one dynamic/fixed root score differs by `.02` nats; and
- cost at most `$1.25`.

First-link fidelity:

- dynamic depth two changes the root from myopic;
- dynamic all-root score-to-terminal-NDCG Spearman is at least `.20`;
- dynamic Spearman exceeds both myopic and fixed-support Spearman by at least
  `.02`;
- the selected dynamic root strictly beats both myopic and fixed-support roots
  on terminal NDCG; and
- dynamic is nonworse than answer-link-shuffled dynamic.

All gates must pass to authorize a separately frozen one-topic development run
on `148`. Failure closes this exact V2 method without rerun, threshold repair,
model swap, or subset analysis.

## Budget

Projected cost is `$1.00`, with a hard `$1.25` cap. The project ledger has
`$13.082379290776686` remaining before this run, and the authenticated account
retains more than the protected `$25` Monday reserve. OpenRouter only; no
OatML/cluster use.
