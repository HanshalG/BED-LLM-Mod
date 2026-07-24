# Paprika Structured-Hypothesis Unlock Result

Date: 2026-07-24

Status: **development gate failed; this Paprika line stops before a ranker or policy
comparison.**

## Frozen Test

The test used the 12 preregistered Paprika train tasks selected by seed `24287`.
Non-thinking Gemma 4 26B generated eight target-hidden cause-and-remedy hypotheses,
four diagnostic-only questions, faithful private-solution-conditioned replies, and
eight refreshed hypotheses per realized branch. GPT-5.4 Mini saw the private solution
only after generation and scored semantic coverage. Coverage required the same cause
and a compatible remedy at score at least `0.80`.

Paprika's prior invalid resolution endpoint was not used.

## Result

| Gate | Required | Observed | Pass |
|---|---:|---:|---:|
| Complete tasks | 12 | 12 | yes |
| Complete candidate branches | 48 | 48 | yes |
| Initial omissions | at least 6 | 3 | **no** |
| Omitted truths recovered | at least 3 | 1 | **no** |
| Mean oracle best-match gain | at least 0.10 | 0.105 | yes |
| Tasks with candidate spread at least 0.15 | at least 4 | 5 | yes |

Initial support covered 9/12 private solutions. Among the three initial omissions, some
diagnostic branch recovered only one, for recovery `1/3`. Mean within-task candidate
spread was `0.3508`.

The only binary recovery was a blood-pressure-monitor task, where the best semantic
match rose from `0.12` to `1.00`. A bus-stop networking task improved continuously from
`0.40` to `0.78` but stayed below the coverage threshold. Several initially covered
tasks had large branch spread because a poor diagnostic reply caused the generator to
lose a correct hypothesis: ice maker `0.90` spread, blender `0.95`, and gaming
controller `0.88`.

## Integrity And Cost

- 48/48 private-solution-conditioned replies passed faithfulness on the first attempt.
- Zero simulator contradictions, repairs, terminal claims, or final inconsistencies.
- All 60 semantic-coverage rows had valid support IDs and best-hypothesis indices.
- Successful gate run: 228 requests, 82,443 prompt tokens, 26,102 completion tokens,
  zero reasoning, `$0.03919172`.
- Complete line including failed-closed smokes, one format diagnostic, and three
  apparatus attempts: `$0.10475953`.

The strict schemas required three banked pre-result repairs. They changed only
serialization handling; the task sample, prompts, models, support sizes, coverage
threshold, and pass gates stayed frozen. See
`PAPRIKA_STRUCTURED_UNLOCK_DEVELOPMENT_AMENDMENT.md`.

## Interpretation

Paprika supplies structured semantic hypotheses and dense, faithful observations, but
its released scenarios are simple enough that eight initial hypotheses usually contain
the private cause and remedy. The generator is not incapable of path-dependent change:
candidate branches have substantial spread and can both recover and destroy a correct
hypothesis. The missing ingredient is headroom. Broad enough initial support saturates
truth coverage; narrowing it after inspecting this result would manufacture omissions.

This reproduces the Animals boundary in a genuinely structured space:

- narrow or tail-mismatched support creates omissions but poor recovery;
- broad, semantically competent support contains the truth before interaction and
  removes the open-world non-myopic opportunity.

Per the preregistered stop rule, there is no target-blind ranker, held-out Paprika
rerun, or policy comparison. The next environment must begin with deliberately partial
evidence and reveal guaranteed structured evidence, rather than relying on a naturally
underspecified but already easy troubleshooting sentence.
