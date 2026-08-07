# Bongard Luna Semantic-Validity Amendment

Date frozen: 2026-08-06, before any Bongard model request.

This amendment supersedes three details in the earlier exact-10 and full-tree
mechanics preregistrations. The changes were made after zero-call code review
identified history double-conditioning and two weak mechanics checks. No model
output or scientific endpoint existed when this amendment was frozen.

## History-Conditioned Belief Weights

Every support is generated after the model sees that branch's labelled
history. The ten emitted weights are therefore explicitly named
`history_weight` and interpreted as the model's posterior plausibility over its
newly generated rules at that history.

The evaluator normalizes these weights and uses them directly. It does not
multiply them by the likelihood of the same observed history again. Analytical
Bayes updates are applied only to a hypothetical or newly released future
label while the support remains fixed. At the next regeneration, the new
history-conditioned model weights replace that temporary update.

This replaces the earlier `prior_weight` wording and the parser path that
conditioned those weights on the already-seen labels a second time.

## Evaluator-Private Image Roles

The model receives opaque image order and observed labels only. The prompt no
longer identifies which unlabelled images are selectable candidates or held-out
endpoints. Acquisition eligibility and endpoint membership remain private to
the analytical evaluator.

The model still emits one likelihood per opaque image. The evaluator may use
candidate likelihoods for EIG and endpoint likelihoods for scoring after the
appropriate label-release boundary, but role membership cannot shape the
model response explicitly.

## Nontrivial Branch Sensitivity

Branch sensitivity excludes every observed image, including the simulated
first-query image whose label differs across the pair. Prediction change is
the mean absolute difference over still-unobserved images only. A branch pair
is material when this unobserved-image change is at least 0.05 or the canonical
rule-set Jaccard is at most 0.8.

The former query-image shift diagnostic and its gate are removed because
copying the simulated label can satisfy them without changing any future
belief. The full mechanics threshold remains 24 of 32 branch pairs, now under
the nontrivial definition.

## Serving Gate Interpretation

Observed-history fit remains a descriptive self-consistency statistic because
the model has seen those labels. It is not a predictive calibration result and
is no longer a pass gate. The exact-10 serving gate instead requires strict
parsing, finite positive history-conditioned weights, nontrivial unobserved
branch changes, nondegenerate candidate EIG, hidden-state hygiene, clean
transport, and bounded cost.

The later simulated-branch obedience amendment does not reinstate aggregate
observed-history fit as a predictive gate. It checks only the newly supplied
counterfactual branch label and requires positive and negative branch calls
separately to beat constant-half Brier, establishing that each regenerated
state obeys the observation it is meant to condition on.

## All-First-Action Mechanics

The full four-task mechanics tree now generates the realized branch-greedy
continuation for every one of the eight possible first actions. Any distinct
fixed-depth-two or random-policy final history is added. Reciprocal action
orders may share one six-label history, so this produces 4--10 final supports
per task, 16--40 final requests, and at most 108 total requests.

After every first action and final history is frozen, the evaluator opens the
two mechanics endpoint labels and reports within-task Spearman correlation
between each root score and negative realized endpoint Brier. It also reports
root candidate-label Brier. Mechanics requires all eight paths, finite ranking
diagnostics, and candidate Brier below 0.25; the sign of four-task ranking
correlation remains descriptive because four tasks are not a powered test.

The mechanics cap is revised from `$1.50` to `$1.75`, still under the same
account-wide `$5.00` daily ledger.

Interface versions are bumped to serving `-2`, mechanics `-2`, and development
`-2`; any artifact from an earlier interface is rejected.
