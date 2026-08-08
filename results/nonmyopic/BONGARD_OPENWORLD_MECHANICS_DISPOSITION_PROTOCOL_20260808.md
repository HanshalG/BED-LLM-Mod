# Bongard Open-World Mechanics Disposition Protocol

Date frozen: 2026-08-08, before any August 10 serving or mechanics response.

## Purpose

This protocol fixes how the banked Bongard serving/mechanics outcome determines
the next scientific action. The classifier is deterministic and makes zero model
calls. It cannot authorize paid calls, reruns, or development. A clean mechanics
pass may only carry the authorization already recorded by the frozen August 10
wrapper into the separately dated development preflight.

The implementation is
`scripts/bongard_openworld_mechanics_disposition.py`. It accepts a serving
`RESULT.json`, mechanics `RESULT.json`, component `FAILURE.json`, or the final
August 10 wrapper `RESULT.json`. Wrapper inputs must retain exact component and
raw-response hashes. Every input must use the current schema, interface, status,
and complete Boolean gate set; unknown, missing, extra, non-Boolean, or
inconsistent gates produce `invalid_record` and no authorization.

## Ordered Dispositions

When several gates fail, the first matching category below is primary. The
record also reports every failed gate and every matching category.

1. `transport_or_schema_inconclusive`: request accounting, bounded retries,
   reasoning/forced-exit constraints, parsing/support schema, or component cost.
   Bank as infrastructure/schema inconclusive; do not infer a semantic null and
   do not rerun the same run.
2. `integrity_or_leakage_failure`: replay, pairing/common-random-number,
   objective/control exactness, support mapping, finite metrics, or prompt
   privacy. Bank as scientifically uninterpretable; repair only in a separately
   preregistered future instrument.
3. `predictive_belief_invalid`: simulated/terminal label obedience or root
   candidate Brier fails. A future experiment must change the predictive belief
   interface, not tune the planner or add samples.
4. `path_dependence_absent`: too few simulated label branches materially change
   unobserved beliefs. Do not open development under this mechanism.
5. `nonmyopic_opportunity_absent`: no robust dynamic-vs-myopic/history-blind
   first-action change, no matched-updater second-action change, or insufficient
   control-path diversity. More tasks under the frozen mechanism cannot create
   the required first-link opportunity.
6. `endpoint_saturated`: the myopic endpoint is already too easy. Any successor
   must prospectively define a harder endpoint/task split.

A fully passing serving result is `serving_pass_mechanics_unobserved`; it may
continue only through the existing frozen wrapper. A fully passing mechanics
result is `mechanics_pass`; it may proceed only through the existing dated
development preflight and authorization chain. A component `failed_closed`
artifact is infrastructure/schema inconclusive because no complete scientific
gate vector exists.

## Non-Authorization Invariants

Every valid or invalid disposition sets:

- `this_record_authorizes_paid_calls: false`
- `this_record_authorizes_rerun: false`
- `this_record_authorizes_development: false`

Only a hash-bound wrapper mechanics pass can set
`existing_wrapper_authorizes_development: true`. This describes an existing
authorization; it does not create one. No outcome from this instrument changes
the frozen August 10 requests, seeds, prompts, actions, budgets, or paper claims.
