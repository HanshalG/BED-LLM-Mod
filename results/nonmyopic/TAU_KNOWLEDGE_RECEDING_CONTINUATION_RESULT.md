# tau-Knowledge Focused Receding Continuation Result

## Verdict

The public serving smoke passed, but the preregistered old-tree development gate
failed. The 20-task fresh confirmation split remains sealed and no confirmation
calls were made.

## Results

| Stage | Calls | Cost | Pair accuracy | Optimal | Regret | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Public smoke | 10 | `$0.093890` | 0.5882 | 7/10 | 3 | Pass |
| Old-tree development | 100 | `$0.945605` | 0.7079 | 74/100 | 32 | Fail |

Both stages used GPT-5.4 with zero reasoning tokens, zero retries, and complete
canonical responses.

Development passed the minimum comparable-pair count, pairwise accuracy,
optimal-selection rate, and all end-to-end controls versus myopic:

- non-myopic receding versus myopic: 4 wins, 1 loss, 15 ties, total `+4`;
- non-myopic receding versus seeded random: 12 wins, 1 loss, 7 ties,
  total `+23`.

It failed three frozen gates:

- all-root regret was `32`, above the maximum `28`;
- loss below the selected non-myopic roots was `6`, above the maximum `5`;
- gain over the original joint continuation selector was `+4`, below the
  required `+5`.

No threshold was changed and fresh confirmation was not released.

## Diagnosis

The focused scorer usually ranked continuations well, but a systematic failure
remained: it sometimes scored the *intended meaning of the followup query*
rather than the evidence in the returned documents. For example, a query about
international ATM reimbursement received a very high score even when its three
retrieved documents covered no required endpoint document. Rationales described
the policy the query sought rather than the policy the results actually
contained.

Some endpoint disagreements are not clear semantic failures. tau-Knowledge
required-document sets include policies needed by facts revealed later in the
script, while this policy is intentionally target blind and sees only the
generated opening and inferred information needs. In several cases the scorer
preferred a directly useful document over unrelated returned documents that
happened to count toward the hidden future endpoint. This limits how literally
exact-document regret should be interpreted.

The one justified next interface is therefore evidence-only continuation
scoring: omit candidate query strings and require every score to be supported by
the returned document contents. That repair must pass a newly preregistered
smoke and old-tree gate before the untouched confirmation split can be used.

## Artifacts

- Smoke:
  `results/nonmyopic/tau_knowledge_receding_continuation_smoke/tau-knowledge-receding-smoke-20260725T014800Z/SERVING_SMOKE.json`
- Development:
  `results/nonmyopic/tau_knowledge_receding_continuation_development/tau-knowledge-receding-development-20260725T015100Z/DEVELOPMENT.json`
- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_RECEDING_CONTINUATION_PREREGISTRATION.md`
