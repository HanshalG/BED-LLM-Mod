# GSM-Agent Non-Myopic Retrieval Opportunity Result

Date: 2026-07-25

## Outcome

**Gate failed. No LLM smoke is authorized.**

The frozen zero-call audit found 25 strict non-myopic opportunities among 500
official test tasks (`5.0%`), below the preregistered minimum of 40. The exact
lexical construction is therefore too sparse for the planned powered
LLM-native comparison. There was no threshold, query-generator, seed, split,
or task repair after observing the result.

## Frozen Results

| Metric | Result | Gate |
|---|---:|---:|
| Opportunity tasks | 500 | exactly 500 |
| Official test-split tasks | 500/500 | all |
| Strict non-myopic opportunities | 25 | at least 40 |
| Strict opportunities with at least four oracle documents | 25 | at least 25 |
| Mean two-step coverage gain over all tasks | +0.236 | at least +0.08 |
| Strict rows with immediate sacrifice and final gain | 25/25 | all |

The sole failed gate was strict-opportunity prevalence.

## Diagnostic Decomposition

- The non-myopic and greedy roots differed on 96/500 tasks.
- All 96 differing roots had higher best two-step required-document coverage.
- On 71/96, however, the roots tied on immediate coverage. These establish
  useful lookahead tie-breaking but do not force a myopic policy to choose the
  worse root.
- Only 25/96 required a strict immediate sacrifice and therefore qualified.
- All 25 strict rows gained exactly one required document after two queries.
- Immediate sacrifices were one document on 19 tasks, two on five tasks, and
  three on one task.
- A continuation improved over its own root page on 233/500 tasks, confirming
  that the environment supports meaningful query refinement even though the
  strict first-action reversal is sparse.

Required-document counts in the opportunity split ranged from 1 to 12; 348/500
tasks had at least four. The four-document power gate passed exactly at 25.

## Interpretation

GSM-Agent is a genuine external retrieval environment: observations come from
a fixed 32,315-document database, queries are open vocabulary, and retrieved
IDs/content/metadata can enable later searches. It is therefore a better
LLM-native substrate than static 20 Questions or an LLM-simulated mystery.

The frozen lexical action construction nevertheless does not produce enough
strict exploration-exploitation reversals. Most lookahead gains either leave
the greedy root unchanged or break an immediate-coverage tie. Scaling an LLM
policy on this split would risk claiming non-myopia from arbitrary myopic tie
breaking rather than from a forced delayed-information tradeoff.

This closes the exact question-derived BM25 root and visible-history
continuation route. A scientifically distinct future GSM-Agent route would
need a prospectively defined external constraint that makes early retrieval
consume or unlock access, not a relaxed 25-task threshold or a tuned lexical
query bank.

## Integrity And Cost

- Official source commit:
  `a596464ea79ae0b8b84830d1c78a7d065177b0e8`.
- Official full-database SHA-256:
  `948e1ad488ef5e7bc1ad9d605684441e77cf796ef923639d0c5414ae4fe3778c`.
- Opportunity/development/holdout split hashes reproduced.
- Development 100 and holdout 473 endpoints remain sealed.
- OpenRouter calls: 0.
- OpenRouter cost: `$0`.
- Cluster jobs: 0; OatML remained paused.
- Audit artifact:
  `results/nonmyopic/gsm_agent_nonmyopic_opportunity/AUDIT.json`.
- Artifact SHA-256:
  `f28e71c1eddbf69281e10f6bcf02ed247181d7efc18a31fb866b6e0bbf0a74fa`.
