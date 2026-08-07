# RegretBench Independent Result Replay Amendment

Date: 2026-08-07
Status: frozen before support-recovery or policy responses

## Problem

The support and policy producers previously summarized their own raw outputs.
Their deterministic tests checked implementation behavior, but a shared bug in
parsing, root selection, terminal conditioning, or bootstrap analysis could
survive because no separate program reconstructed the claimed result.

## Amendment

`scripts/regretbench_deepseek_result_verify.py` is a zero-call verifier that
does not import either experiment producer. From the public result and private
raw artifacts it independently reconstructs:

- official source order, seeded truths, environment mappings, and conservative
  answer-alias matching;
- support-recovery root, conditioned, and history-blind coverage;
- strict enriched beliefs, information scores, dynamic/history-blind/fixed/
  myopic/random root selection, and frozen pre-truth choices;
- enriched-smoke and formal distinct semantic-action checks using the official
  mapper;
- exact generated-likelihood conditioning on the realized second reply;
- paired comparisons, correlations, bootstraps, scientific gates, and status;
- request schedules, common-random-number groups, schema counts, privacy
  payload schedules, action/matchability floors, and cost gates.

The verifier reports exact mismatch paths, makes zero provider calls, and
cannot rescue a producer failure. The daily support executor requires verified
smoke and development artifacts. The daily policy executor requires those
hash-bound support verifications, a verified enriched-policy smoke, and a
verified policy development artifact. Any replay disagreement fails closed
after normal spend reconciliation.

## Claim Boundary

Independent replay establishes artifact consistency and catches producer-only
analysis errors. It does not create scientific evidence, improve a gated-null
result, or authorize confirmation beyond the original preregistered gates.
