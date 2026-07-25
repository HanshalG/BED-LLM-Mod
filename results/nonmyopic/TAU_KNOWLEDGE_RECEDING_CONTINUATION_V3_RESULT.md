# tau-Knowledge Count-Dominant Receding Continuation V3 Result

## Verdict

The public smoke passed. Development failed closed on the frozen canonical
string parser, so the untouched 20-task confirmation split remains sealed.
There will be no V4 on these development trees.

## Serving

| Stage | Calls | Cost | Reasoning | Result |
|---|---:|---:|---:|---|
| Public smoke | 10 | `$0.076350` | 0 | Pass |
| Old-tree development | 100 | `$0.763025` | 0 | Fail closed |

Smoke achieved 0.6471 pairwise accuracy, 7/10 oracle-optimal selections, and
regret 3, passing every frozen gate.

All 100 development responses contained the four requested scores. However, 29
responses encoded at least one single-digit score with a leading zero, such as
`"02"` or `"00"`. The frozen parser required canonical strings (`"2"` and
`"0"`), so the run was rejected. There were zero reasoning tokens, retries,
forced exits, or HTTP failures. No response was repaired or reissued.

## Diagnostic Only

The following is explicitly post hoc and cannot release confirmation. Parsing
the zero-padded strings as their unambiguous integer values would have yielded:

| Metric | Post hoc V3 | Frozen development gate |
|---|---:|---:|
| Followup pairwise accuracy | 0.8144 | >=0.60 |
| Oracle-optimal followups | 82/100 | >=72/100 |
| All-root regret | 21 | <=28 |
| Selected-root continuation loss | 2 | <=5 |
| Non-myopic vs myopic | 4W / 0L / `+6` | >=3W / <=2L / >=`+3` |
| Non-myopic vs joint | 8W / 1L / `+8` | >=`+5` |
| Non-myopic vs seeded random | 12W / 0L / `+27` | descriptive |

Thus the count-dominant semantic document utility appears to fix the substantive
development bottleneck. It is not a confirmation result: the normalization was
not preregistered, the development trees informed the method, and the registered
serving gate failed.

## Implication

The strongest next LLM-native design is now concrete:

1. let the LLM maintain path-dependent initial and refreshed semantic needs;
2. treat refreshed needs as fallible so observations do not erase the original
   objective;
3. use the LLM to classify which newly retrieved documents support those needs;
4. make distinct useful-document count dominate free-form importance scores.

This should be transferred with a frozen permissive numeric parser to a new
independent environment or task source. The current paper should retain the
qualified tau first-link result and describe continuation as unconfirmed.

## Artifacts

- Smoke:
  `results/nonmyopic/tau_knowledge_receding_continuation_v3_smoke/tau-knowledge-receding-v3-smoke-20260725T023000Z/SERVING_SMOKE.json`
- Development failure:
  `results/nonmyopic/tau_knowledge_receding_continuation_v3_development/tau-knowledge-receding-v3-development-20260725T023200Z/DEVELOPMENT_FAILURE.json`
- Post hoc diagnostic:
  `results/nonmyopic/tau_knowledge_receding_continuation_v3_development/tau-knowledge-receding-v3-development-20260725T023200Z/DEVELOPMENT_DIAGNOSTIC.json`
- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_RECEDING_CONTINUATION_V3_PREREGISTRATION.md`
