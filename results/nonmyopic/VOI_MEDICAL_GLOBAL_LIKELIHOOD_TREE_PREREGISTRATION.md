# VoI Medical Global-Likelihood Tree V2 Preregistration

## Scope

V1 failed because independently classifying a repeated semantic question in
different root-local batches produced inconsistent diagnosis labels. V2 changes
only the likelihood-serving interface. It remains a source-qualified mechanics
gate, not a patient-level or efficacy experiment.

## Frozen Correction

- Source, 15-diagnosis empirical prior, GPT-5.4 nonreasoning model, four roots,
  three root outcomes, two branch follow-ups, exact Bayes updates, policies, and
  scientific score gates are unchanged from V1.
- Every root is classified in its own one-question request.
- After all branch follow-ups are generated, questions are normalized by
  whitespace and case in deterministic root/outcome/order traversal.
- A follow-up matching any root or earlier follow-up reuses that question's
  already frozen map.
- Every previously unseen follow-up is classified exactly once in an isolated
  one-question request.
- Each semantic action therefore has exactly one global deterministic
  diagnosis-to-`Yes/No/Maybe` map.

The first root must not be repeated within its own branch. Repeated questions
elsewhere are allowed and reuse the global map; there is no reclassification,
vote, repair, or consistency retry.

## Requests And Budget

The exact dynamic request count is:

```text
1 root proposal + 4 isolated root maps + 12 branch proposals
+ U isolated maps for unique new follow-up questions
```

where `U` is determined once the strict branch outputs parse. Thus exact
requests and HTTP attempts must equal `17 + U`.

- Model: `openai/gpt-5.4` through OpenRouter
- Temperature: `0`
- Reasoning, retries, repairs, parser fallbacks, and forced finalization: zero
- Concurrency: at most `12`
- Projected cost: `$0.10`
- Hard cap: `$0.25`
- Frozen allowance before V2: `$1.20891155`
- Protected balance: `$25` through Monday
- OatML/cluster execution: prohibited

## Exact Policies And Gate

Myopic chooses the root with maximum immediate entropy reduction, then the best
available follow-up in the realized branch. Depth two chooses the root with
maximum expected two-step entropy reduction with branch-adaptive follow-ups.
Both policies use the same generated tree, global likelihood maps, and
second-step compute.

All V1 scientific gates remain conjunctive:

- every root has at least two positive-probability outcomes;
- every root has at least two distinct follow-up sets across realizable answers;
- immediate and depth-two score ranges are each at least `0.05` nats;
- depth two selects a different root;
- its selected root improves two-step EIG by at least `0.03` nats over the
  myopic-selected root.

Serving also requires exact `17 + U` requests/attempts, complete strict maps,
one map per normalized unique question, zero retries/reasoning/forced
finalization, and cost at most `$0.25`.

Failure closes V2 with no alternate seed, repair, threshold change, partial
score, or patient endpoint. Passing authorizes only a separately preregistered
small patient-grounded first-link validation.
