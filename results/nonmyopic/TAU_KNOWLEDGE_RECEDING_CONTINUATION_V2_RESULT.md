# tau-Knowledge Evidence-Only Receding Continuation V2 Result

## Verdict

The public smoke passed strongly. The old-tree development run failed closed
because 2 of 100 otherwise successful responses omitted trailing schema keys.
No response was repaired or reissued, and no fresh confirmation calls were
made.

## Serving

| Stage | Calls | Cost | Reasoning | Result |
|---|---:|---:|---:|---|
| Public smoke | 10 | `$0.108510` | 0 | Pass |
| Old-tree development | 100 | `$1.091120` | 0 | Fail closed |

Smoke improved over V1 to 0.7059 pairwise accuracy, 9/10 oracle-optimal
followups, and total regret 1.

Development returned 98 complete canonical responses. Two responses ended after
2 and 3 of the required 4 followup entries. The frozen parser correctly rejected
the batch. There were no retries, forced exits, reasoning tokens, or HTTP
failures.

## Descriptive Bound

The failed batch cannot establish a V2 result. For diagnosis only, the 98
complete roots achieved:

- pairwise accuracy `0.7944`, versus V1's `0.7038` on the same roots;
- 79/98 oracle-optimal selections, versus V1's 72/98;
- regret 25, versus V1's 32.

One missing root had identical endpoint value for all four continuations. The
other could add at most one regret. Thus even the worst completion would have
met the all-root accuracy, optimal-rate, and regret gates.

All policy-selected roots were among the 98 valid responses, so their
development diagnostics are exact: non-myopic receding versus myopic was 3
wins, 1 loss, total `+4`, but selected-root loss remained `6` and improvement
over joint remained `+4`. Those values still fail the frozen `<=5` and `>=5`
policy gates. A parser-only rerun is therefore not justified.

## Interpretation

Hiding query wording successfully reduced query-intent hallucination and
improved general branch ranking. The remaining selected-root errors came from
free-form importance weighting and path anchoring. The scorer sometimes
preferred one urgent policy document over two distinct useful documents, or
treated a refreshed product-specific hypothesis as settled even though the
original objective required comparing products.

A final compact development interface is justified: use the LLM as a
target-blind semantic classifier of distinct new useful documents under the
opening plus initial and refreshed hypotheses. Document count must dominate any
quality tie-break, and generated refreshed needs remain explicitly fallible.

## Artifacts

- Smoke:
  `results/nonmyopic/tau_knowledge_receding_continuation_v2_smoke/tau-knowledge-receding-v2-smoke-20260725T020000Z/SERVING_SMOKE.json`
- Development failure:
  `results/nonmyopic/tau_knowledge_receding_continuation_v2_development/tau-knowledge-receding-v2-development-20260725T020300Z/DEVELOPMENT_FAILURE.json`
- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_RECEDING_CONTINUATION_V2_PREREGISTRATION.md`
