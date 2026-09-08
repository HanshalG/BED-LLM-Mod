# Structured-scene executable mechanics

Status: implementation and unit-test evidence only. No scientific cohort or
opportunity outcome opened; no model call or paid authorization. This is an
independent abstract structured-scene variant, not a ZendoWorld reproduction or
ported upstream implementation.

## Implemented

`environments/semantic_scene/rules.py` specifies an order-invariant multiset of
1-7 objects with color (red/blue/yellow), shape (block/wedge/pyramid), and size
(small/large). Objects with identical attributes remain distinct objects.
There is no geometry, gravity, touching relation, renderer, or visual input.

Strict JSON rules support typed object attribute predicates, Boolean composition,
scene-level counts (equal/at least/at most), and existential pairs of distinct
objects sharing or differing in a named attribute. The same interpreter supplies
all labels; neither scene schema nor rule schema accepts hidden-label fields.
No Python execution, language judge, teacher counterexample or model call occurs
inside compilation/evaluation. Invalid inputs raise rather than returning False.

Bounds: 16 KiB rule JSON, depth 8, 63 pre-deduplication nodes, at most four Boolean
operands per node. Duplicate JSON keys/nonfinite constants and wrong types fail.
Canonical keys merge operand order and duplicate Boolean children, plus symmetric
pair roles. This is NOT complete logical-equivalence reduction; a later symbolic
prior must account for its chosen syntax/semantic multiplicities explicitly.

`belief.py` conditions an explicit caller-supplied finite prior using deterministic
membership likelihoods. Exact rational weights prevent float tie artifacts.
Full history is replayed from the original prior, repeated identical observations
are idempotent, contradictions and unsupported observations fail explicitly, and
duplicate canonical proposals are rejected rather than silently multiplying mass.
Prediction and fixed-target Bayes Brier risk are available. The caller supplies
target weights; targets are not deleted when queried.

These are conditional finite-pool calculations, not a prior over LLM discovery
or a correction for history-dependent proposal selection. Unsupported history
must eventually be reported/handled by a prospectively defined discovery updater;
this module does not reset to a confident arbitrary distribution. Prediction on
a misspecified pool can still be confidently wrong.

## Verification

36 focused tests pass (0.52s), including the preceding source-contract tests.
Rule checks exhaust all color strings of length 1-4 against independent count
formulas, test logical composition/permutation invariance, preserve duplicate
objects, exercise same/different pairs, and reject invalid schemas/types/limits.
Belief checks use an independent literal truth matrix to verify exact posterior
mass, evidence and Brier risk, plus repeat-history/contradiction/missing-support
checks. This verifies the tested mechanics, not a whole-scene semantic benchmark,
global logical equivalence, likelihood calibration, or planning opportunity.

## Next required evidence

Freeze a bounded task/scene generation protocol and actual symbolic prior/search
control before evaluating its outcomes. Reuse the existing genuine receding-horizon
solver where its model interface permits; do not build another planning engine.
Complete the whole source-only panel with fixed targets, equal real query budgets,
receding open-loop and random controls. Do not select winning rule families after
looking at that panel. Only a dependency-valid opportunity result may precede the
held-out LLM proposal/feedback-control gate. No monotonic-depth result is claimed.

The overarching plan remains incomplete: this layer enables the same executable
semantics for LLM and symbolic proposals but does not demonstrate either LLM
usefulness or anticipation of future hypothesis discovery. Prior nulls stay closed,
automation remains paused, and no fresh account balance is asserted.
